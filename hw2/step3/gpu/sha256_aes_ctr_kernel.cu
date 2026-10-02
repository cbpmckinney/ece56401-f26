// CUDA kernel for Step 3's "stuffing" attack -- straight port of
// sha256_aes_ctr_kernel.c (the PyOpenCL version, verified against your
// Mac's real GPU) to CUDA C, same mechanical changes already applied to
// step2/gpu/sha256crypt_kernel.c -> sha256crypt_kernel.cu:
//   * __kernel / __global / __constant (OpenCL)  ->  extern "C" __global__ /
//     (plain pointer) / __constant__ (CUDA)
//   * plain helper functions (OpenCL auto-inlines them for kernel use) ->
//     explicit __device__ on every helper called from the kernel
//   * get_global_id(0) (OpenCL)                   ->  blockIdx.x * blockDim.x
//                                                      + threadIdx.x (CUDA)
//   * kernel dispatch, buffers, compilation        ->  handled host-side in
//     gpu_crack_cuda.py via pycuda instead of pyopencl
//
// `extern "C"` on the kernel keeps its name unmangled so PyCUDA's
// mod.get_function("crack_sha256_aes_ctr") can find it -- nvcc otherwise
// compiles .cu files as C++ and mangles function names (same reason
// sha256crypt_kernel.cu uses it).
//
// Algorithm itself is unchanged from sha256_aes_ctr_kernel.c -- see that
// file's header for how it was verified (pure-Python mirror checked
// against this repo's real encrypt.py across 5 test vectors, then the
// literal kernel file compiled as plain C and run against those same
// vectors). This .cu file is a mechanical translation of that
// already-verified source, and was re-checked the same way: CUDA syntax
// stubbed out, compiled as plain C++, and run against 10 known-answer
// test vectors (including one planted at the real secret.html.crypt's
// exact 1673-byte plaintext length) -- all 10 passed, since a hand port
// can introduce its own typos even when the logic doesn't change. Still
// no actual CUDA toolchain/GPU in the environment this was written in --
// please run --sanity-check on the real dual-4090 machine before
// trusting a "no match" result from a long unattended run.

typedef unsigned int WORD;
typedef unsigned char BYTE;

#define ROTR(len,val) (((val) >> (len)) | ((val) << (32 - (len))))
#define SHR(len,val) ((val) >> (len))
#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(2,x) ^ ROTR(13,x) ^ ROTR(22,x))
#define EP1(x) (ROTR(6,x) ^ ROTR(11,x) ^ ROTR(25,x))
#define SIG0(x) (ROTR(7,x) ^ ROTR(18,x) ^ SHR(3,x))
#define SIG1(x) (ROTR(17,x) ^ ROTR(19,x) ^ SHR(10,x))

__constant__ WORD K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__device__ void sha256_process_block(WORD state[8], const BYTE block[64]) {
    WORD w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = (block[i*4] << 24) | (block[i*4+1] << 16) | (block[i*4+2] << 8) | (block[i*4+3]);
    }
    for (int i = 16; i < 64; i++) {
        w[i] = SIG1(w[i-2]) + w[i-7] + SIG0(w[i-15]) + w[i-16];
    }

    WORD a = state[0], b = state[1], c = state[2], d = state[3];
    WORD e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; i++) {
        WORD t1 = h + EP1(e) + CH(e,f,g) + K[i] + w[i];
        WORD t2 = EP0(a) + MAJ(a,b,c);
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

typedef struct {
    WORD state[8];
    BYTE buffer[64];
    int buffer_len;
    unsigned long long total_len;
} SHA256_CTX;

__device__ void sha256_ctx_init(SHA256_CTX *ctx) {
    ctx->state[0] = 0x6a09e667; ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372; ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f; ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab; ctx->state[7] = 0x5be0cd19;
    ctx->buffer_len = 0;
    ctx->total_len = 0;
}

__device__ void sha256_ctx_update(SHA256_CTX *ctx, const BYTE *data, int len) {
    ctx->total_len += len;
    int i = 0;
    if (ctx->buffer_len > 0) {
        while (i < len && ctx->buffer_len < 64) {
            ctx->buffer[ctx->buffer_len++] = data[i++];
        }
        if (ctx->buffer_len == 64) {
            sha256_process_block(ctx->state, ctx->buffer);
            ctx->buffer_len = 0;
        }
    }
    while (len - i >= 64) {
        sha256_process_block(ctx->state, data + i);
        i += 64;
    }
    while (i < len) {
        ctx->buffer[ctx->buffer_len++] = data[i++];
    }
}

__device__ void sha256_ctx_final(SHA256_CTX *ctx, BYTE out_digest[32]) {
    unsigned long long bits = ctx->total_len * 8ULL;

    BYTE pad = 0x80;
    ctx->buffer[ctx->buffer_len++] = pad;
    if (ctx->buffer_len == 64) {
        sha256_process_block(ctx->state, ctx->buffer);
        ctx->buffer_len = 0;
    }
    while (ctx->buffer_len != 56) {
        ctx->buffer[ctx->buffer_len++] = 0x00;
        if (ctx->buffer_len == 64) {
            sha256_process_block(ctx->state, ctx->buffer);
            ctx->buffer_len = 0;
        }
    }
    for (int i = 0; i < 8; i++) {
        ctx->buffer[ctx->buffer_len++] = (BYTE)((bits >> (56 - i * 8)) & 0xFF);
    }
    sha256_process_block(ctx->state, ctx->buffer);
    ctx->buffer_len = 0;

    for (int i = 0; i < 8; i++) {
        out_digest[i*4]   = (ctx->state[i] >> 24) & 0xFF;
        out_digest[i*4+1] = (ctx->state[i] >> 16) & 0xFF;
        out_digest[i*4+2] = (ctx->state[i] >> 8) & 0xFF;
        out_digest[i*4+3] = ctx->state[i] & 0xFF;
    }
}

// ---------------------------------------------------------------------
// AES-256, forward encrypt direction only (CTR mode never needs AES
// *decrypt* -- see sha256_aes_ctr_kernel.c's header comment). Unchanged
// from the verified OpenCL source.
// ---------------------------------------------------------------------

__constant__ BYTE AES_SBOX[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

// 1-indexed; AES-256's key schedule only ever needs RCON[1..7] (i/Nk tops
// out at 7 for Nk=8, Nr=14). Index 0 is an unused placeholder.
__constant__ BYTE AES_RCON[8] = {0x00,0x01,0x02,0x04,0x08,0x10,0x20,0x40};

__device__ BYTE aes_xtime(BYTE x) {
    return (BYTE)(((x << 1) ^ ((x & 0x80) ? 0x1B : 0x00)) & 0xFF);
}

// round_keys: 15 round keys x 16 bytes = 240 bytes (AES-256, Nr=14).
__device__ void aes256_key_expansion(const BYTE key[32], BYTE round_keys[15][16]) {
    BYTE w[60][4];
    for (int i = 0; i < 8; i++) {
        w[i][0] = key[4*i]; w[i][1] = key[4*i+1];
        w[i][2] = key[4*i+2]; w[i][3] = key[4*i+3];
    }
    for (int i = 8; i < 60; i++) {
        BYTE temp[4] = { w[i-1][0], w[i-1][1], w[i-1][2], w[i-1][3] };
        if (i % 8 == 0) {
            BYTE t0 = temp[0];
            temp[0] = AES_SBOX[temp[1]]; temp[1] = AES_SBOX[temp[2]];
            temp[2] = AES_SBOX[temp[3]]; temp[3] = AES_SBOX[t0];
            temp[0] ^= AES_RCON[i / 8];
        } else if (i % 8 == 4) {
            temp[0] = AES_SBOX[temp[0]]; temp[1] = AES_SBOX[temp[1]];
            temp[2] = AES_SBOX[temp[2]]; temp[3] = AES_SBOX[temp[3]];
        }
        for (int k = 0; k < 4; k++) w[i][k] = w[i-8][k] ^ temp[k];
    }
    for (int r = 0; r < 15; r++) {
        for (int k = 0; k < 4; k++) {
            round_keys[r][4*k]   = w[4*r+k][0];
            round_keys[r][4*k+1] = w[4*r+k][1];
            round_keys[r][4*k+2] = w[4*r+k][2];
            round_keys[r][4*k+3] = w[4*r+k][3];
        }
    }
}

__device__ void aes_shift_rows(BYTE s[16]) {
    BYTE t;
    t = s[1]; s[1] = s[5]; s[5] = s[9]; s[9] = s[13]; s[13] = t;
    t = s[2]; s[2] = s[10]; s[10] = t;
    t = s[6]; s[6] = s[14]; s[14] = t;
    t = s[3]; s[3] = s[15]; s[15] = s[11]; s[11] = s[7]; s[7] = t;
}

__device__ void aes_mix_columns(BYTE s[16]) {
    for (int c = 0; c < 4; c++) {
        int b = c * 4;
        BYTE s0 = s[b], s1 = s[b+1], s2 = s[b+2], s3 = s[b+3];
        s[b]   = aes_xtime(s0) ^ (aes_xtime(s1) ^ s1) ^ s2 ^ s3;
        s[b+1] = s0 ^ aes_xtime(s1) ^ (aes_xtime(s2) ^ s2) ^ s3;
        s[b+2] = s0 ^ s1 ^ aes_xtime(s2) ^ (aes_xtime(s3) ^ s3);
        s[b+3] = (aes_xtime(s0) ^ s0) ^ s1 ^ s2 ^ aes_xtime(s3);
    }
}

__device__ void aes256_encrypt_block(const BYTE round_keys[15][16], const BYTE in[16], BYTE out[16]) {
    BYTE state[16];
    for (int i = 0; i < 16; i++) state[i] = in[i] ^ round_keys[0][i];
    for (int rnd = 1; rnd < 14; rnd++) {
        for (int i = 0; i < 16; i++) state[i] = AES_SBOX[state[i]];
        aes_shift_rows(state);
        aes_mix_columns(state);
        for (int i = 0; i < 16; i++) state[i] ^= round_keys[rnd][i];
    }
    for (int i = 0; i < 16; i++) state[i] = AES_SBOX[state[i]];
    aes_shift_rows(state);
    for (int i = 0; i < 16; i++) state[i] ^= round_keys[14][i];
    for (int i = 0; i < 16; i++) out[i] = state[i];
}

// ---------------------------------------------------------------------
// Main kernel. Mirrors decrypt_target()/decrypt() in stuffingtest2.py /
// encrypt.py exactly: SHA256(password) -> AES-256-CTR decrypt -> compare
// SHA256(plaintext) to the real trailing-32-byte tag. One candidate per
// GPU thread; out_match[idx] is 1 on a match, 0 otherwise -- the host
// does a single vectorized compare across the whole batch.
// ---------------------------------------------------------------------

#define MAX_PWD_LEN 64
#define MAX_CT_LEN 2048

extern "C" __global__ void crack_sha256_aes_ctr(
    const BYTE *passwords, int pass_stride,
    const BYTE *ciphertext, int ct_len,
    const BYTE *target_tag,
    BYTE *out_match,
    int num_items)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;

    const BYTE *p_cand = passwords + (idx * pass_stride);
    int p_actual_len = 0;
    while (p_actual_len < pass_stride && p_cand[p_actual_len] != 0) {
        p_actual_len++;
    }

    BYTE pwd[MAX_PWD_LEN];
    for (int i = 0; i < p_actual_len; i++) pwd[i] = p_cand[i];

    // key = SHA256(password)
    SHA256_CTX ctx;
    BYTE key[32];
    sha256_ctx_init(&ctx);
    sha256_ctx_update(&ctx, pwd, p_actual_len);
    sha256_ctx_final(&ctx, key);

    BYTE round_keys[15][16];
    aes256_key_expansion(key, round_keys);

    // AES-256-CTR decrypt, streamed DIRECTLY into a running SHA-256 of
    // the plaintext (the tag) one 16-byte block at a time -- deliberately
    // NOT materialized into one big per-thread plaintext[] buffer first
    // (that used to be `BYTE plaintext[MAX_CT_LEN]`, i.e. 2048 bytes of
    // per-thread storage). No GPU has anywhere near 2KB of register file
    // per thread, so a fixed array that size forces the compiler to
    // spill it to slow per-thread "local memory" (uncached, global-
    // memory-backed) -- and just HAVING that much per-thread state alive
    // craters how many threads/warps can be resident on an SM at once,
    // independent of how little of it any single thread actually
    // touches. This is almost certainly why a 4090 (thousands of cores)
    // benchmarked SLOWER than an M3 Pro's integrated GPU here: NVIDIA's
    // register-constrained architecture punishes this pattern far more
    // than Apple's seems to. Streaming the tag update like this gives
    // the IDENTICAL digest (SHA-256 over N bytes is the same whether
    // it's fed in one call or in chunks) -- same trick already used for
    // the sha256crypt kernel's digest P/C computation in step2, see that
    // kernel's comments. `ctx` is reused from the key computation above
    // rather than declaring a second SHA256_CTX -- sha256_ctx_init()
    // resets it fully.
    sha256_ctx_init(&ctx);

    // Counter starts at integer 1 represented as a 16-byte big-endian
    // value and increments by 1 per 16-byte block -- matches encrypt.py's
    // CTR_NONCE = (1).to_bytes(16, "big") exactly.
    BYTE counter_block[16];
    for (int i = 0; i < 15; i++) counter_block[i] = 0;
    counter_block[15] = 1;

    int off = 0;
    while (off < ct_len) {
        BYTE keystream[16];
        aes256_encrypt_block(round_keys, counter_block, keystream);
        int blocklen = (ct_len - off < 16) ? (ct_len - off) : 16;
        BYTE block[16];
        for (int i = 0; i < blocklen; i++) {
            block[i] = ciphertext[off + i] ^ keystream[i];
        }
        sha256_ctx_update(&ctx, block, blocklen);
        off += 16;
        for (int i = 15; i >= 0; i--) {
            if (++counter_block[i] != 0) break;
        }
    }

    // candidate_tag = SHA256(plaintext), compare to the real tag
    BYTE computed_tag[32];
    sha256_ctx_final(&ctx, computed_tag);

    int match = 1;
    for (int i = 0; i < 32; i++) {
        if (computed_tag[i] != target_tag[i]) { match = 0; break; }
    }
    out_match[idx] = (BYTE)match;
}
