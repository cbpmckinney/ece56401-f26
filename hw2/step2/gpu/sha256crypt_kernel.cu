// sha256crypt ($5$) CUDA kernel.
//
// This is a straight port of sha256crypt_kernel.c (the PyOpenCL version
// verified against your Mac's GPU) to CUDA C. The algorithm is byte-for-
// byte identical -- only the platform-specific bits changed:
//   * __kernel / __global / __constant (OpenCL)  ->  extern "C" __global__ /
//     (plain pointer) / __constant__ (CUDA)
//   * get_global_id(0) (OpenCL)                   ->  blockIdx.x * blockDim.x
//                                                      + threadIdx.x (CUDA)
//   * kernel dispatch, buffers, compilation        ->  handled host-side in
//     gpu_nvidia_cuda.py via pycuda instead of pyopencl
//
// `extern "C"` on the kernel keeps its name unmangled so PyCUDA's
// mod.get_function("sha256crypt_cuda") can find it -- nvcc otherwise
// compiles .cu files as C++ and mangles function names.

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

// ---------------------------------------------------------------------
// Streaming SHA-256 (see sha256crypt_kernel.c for the full rationale --
// digest P alone needs to hash up to pwd_len*pwd_len bytes, which would
// overflow a fixed single-shot buffer, so this processes 64-byte blocks
// as they fill instead of materializing whole messages up front).
// ---------------------------------------------------------------------

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
    unsigned long long total_len;   // message bytes seen so far (pre-padding)
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

    ctx->buffer[ctx->buffer_len++] = 0x80;
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
// sha256crypt ($5$) -- digest B, A, P, S, then `rounds` rounds combining
// A/P/S into digest C. See sha256crypt_kernel.c for the per-step spec
// references; the logic here is identical, only the platform glue differs.
// ---------------------------------------------------------------------

extern "C" __global__ void sha256crypt_cuda(
    const BYTE *passwords, int pass_len,
    const BYTE *salt, int salt_len,
    BYTE *out_digests, int num_items)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;

    const BYTE *p_cand = passwords + (idx * pass_len);
    int p_actual_len = 0;
    while (p_actual_len < pass_len && p_cand[p_actual_len] != 0) {
        p_actual_len++;
    }

    BYTE pwd[64];
    for (int i = 0; i < p_actual_len; i++) pwd[i] = p_cand[i];
    BYTE slt[16];
    for (int i = 0; i < salt_len; i++) slt[i] = salt[i];

    SHA256_CTX ctx;
    BYTE digest_b[32], digest_a[32];

    // digest B = SHA256(pwd + salt + pwd)
    sha256_ctx_init(&ctx);
    sha256_ctx_update(&ctx, pwd, p_actual_len);
    sha256_ctx_update(&ctx, slt, salt_len);
    sha256_ctx_update(&ctx, pwd, p_actual_len);
    sha256_ctx_final(&ctx, digest_b);

    // digest A = SHA256( pwd + salt
    //                    + repeat_string(digest_b, pwd_len)
    //                    + [for each bit of pwd_len, LSB first: digest_b if 1 else pwd] )
    sha256_ctx_init(&ctx);
    sha256_ctx_update(&ctx, pwd, p_actual_len);
    sha256_ctx_update(&ctx, slt, salt_len);
    {
        BYTE rep_db[64];
        for (int i = 0; i < p_actual_len; i++) rep_db[i] = digest_b[i % 32];
        sha256_ctx_update(&ctx, rep_db, p_actual_len);
    }
    {
        int i = p_actual_len;
        while (i > 0) {
            if (i & 1) {
                sha256_ctx_update(&ctx, digest_b, 32);
            } else {
                sha256_ctx_update(&ctx, pwd, p_actual_len);
            }
            i >>= 1;
        }
    }
    sha256_ctx_final(&ctx, digest_a);

    // digest P ("dp") = repeat_string( SHA256(pwd repeated pwd_len times), pwd_len )
    BYTE dp[64];
    {
        BYTE dp_seed[32];
        sha256_ctx_init(&ctx);
        for (int r = 0; r < p_actual_len; r++) {
            sha256_ctx_update(&ctx, pwd, p_actual_len);
        }
        sha256_ctx_final(&ctx, dp_seed);
        for (int i = 0; i < p_actual_len; i++) dp[i] = dp_seed[i % 32];
    }

    // digest S ("ds") = repeat_string( SHA256(salt repeated (16 + digest_a[0]) times), salt_len )
    BYTE ds[16];
    {
        BYTE ds_seed[32];
        int ds_repeat_count = 16 + digest_a[0];
        sha256_ctx_init(&ctx);
        for (int r = 0; r < ds_repeat_count; r++) {
            sha256_ctx_update(&ctx, slt, salt_len);
        }
        sha256_ctx_final(&ctx, ds_seed);
        for (int i = 0; i < salt_len; i++) ds[i] = ds_seed[i % 32];
    }

    // digest C: `rounds` rounds (fixed at 1000 for this target). The round
    // loop runs 1000x per candidate -- by far the hottest part of the
    // kernel -- so each round's message is built into one small flat
    // buffer with plain array writes and hashed with a single call,
    // rather than several separate streamed update() calls (same lesson
    // learned/applied on the OpenCL side -- see sha256crypt_kernel.c).
    BYTE current_digest[32];
    for (int i = 0; i < 32; i++) current_digest[i] = digest_a[i];

    BYTE round_buf[224];   // 64 (dp/dc) + 16 (ds) + 64 (dp) + 64 (dc/dp), rounded up
    for (int r = 0; r < 1000; r++) {
        int rlen = 0;
        if (r & 1) {
            for (int j = 0; j < p_actual_len; j++) round_buf[rlen++] = dp[j];
        } else {
            for (int j = 0; j < 32; j++) round_buf[rlen++] = current_digest[j];
        }
        if (r % 3 != 0) {
            for (int j = 0; j < salt_len; j++) round_buf[rlen++] = ds[j];
        }
        if (r % 7 != 0) {
            for (int j = 0; j < p_actual_len; j++) round_buf[rlen++] = dp[j];
        }
        if (r & 1) {
            for (int j = 0; j < 32; j++) round_buf[rlen++] = current_digest[j];
        } else {
            for (int j = 0; j < p_actual_len; j++) round_buf[rlen++] = dp[j];
        }

        sha256_ctx_init(&ctx);
        sha256_ctx_update(&ctx, round_buf, rlen);
        sha256_ctx_final(&ctx, current_digest);
    }

    for (int i = 0; i < 32; i++) {
        out_digests[(idx * 32) + i] = current_digest[i];
    }
}
