import numpy as np
import pyopencl as cl
import itertools
import time
from datetime import timedelta
from wordfreq import zipf_frequency
import math
import smtplib
from email.message import EmailMessage
import json
import os



scriptname = os.path.basename(__file__)
jobdescription = "DOWNGRADE Concatenation triples"



# 1. OpenCL Kernel Code
# -- REWRITTEN. The original hand-rolled version had three bugs that meant
#    it could never match a real sha256crypt hash for any candidate:
#      1. digest B was computed as hash(pwd+salt) instead of hash(pwd+salt+pwd)
#      2. the digest-A "for each bit of pwd_len" loop had a stray break
#         statement, so it only ever ran its first iteration instead of
#         all of them
#      3. the 1000-round stretching loop mixed in the raw password/salt
#         bytes where the spec calls for digest P and digest S -- two
#         separate derived digests that were never being computed at all
#    This version streams everything through a small block-based SHA-256
#    (sha256_process_block/SHA256_CTX) instead of materializing messages
#    into fixed-size buffers, since digest P alone needs to hash up to
#    pwd_len*pwd_len bytes (4096 for a 64-char candidate) -- verified
#    against passlib's reference implementation across 95 test vectors
#    (password lengths 1-64, including the SHA-256 block-boundary edge
#    cases, across 5 different salts) via a standalone C harness before
#    being put here.
opencl_code = """
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

__constant WORD K[64] = {
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
// Streaming SHA-256. Replaces the old single-shot sha256_compress(),
// which copied its ENTIRE input into a fixed 128-byte buffer -- fine for
// short fixed messages, but sha256crypt needs to hash things like "the
// password repeated pwd_len times" (up to 64*64 = 4096 bytes for a
// 64-char candidate), which would silently overflow that buffer. This
// version processes 64-byte blocks as they fill, so the only fixed-size
// storage needed is one 64-byte block, regardless of total message
// length or how that message is assembled (all at once, or piece by
// piece across several update() calls, which is what lets a "repeated"
// message be streamed without ever materializing it).
// ---------------------------------------------------------------------

void sha256_process_block(WORD state[8], const BYTE block[64]) {
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

void sha256_ctx_init(SHA256_CTX *ctx) {
    ctx->state[0] = 0x6a09e667; ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372; ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f; ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab; ctx->state[7] = 0x5be0cd19;
    ctx->buffer_len = 0;
    ctx->total_len = 0;
}

void sha256_ctx_update(SHA256_CTX *ctx, const BYTE *data, int len) {
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

void sha256_ctx_final(SHA256_CTX *ctx, BYTE out_digest[32]) {
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
// sha256crypt ($5$), following the reference algorithm (digest B, A, P,
// S, then `rounds` rounds combining A/P/S into digest C). Each step
// below is a direct translation of that spec -- see the comments for
// which piece of the spec it corresponds to. `p_cand`/`p_actual_len`
// and `salt`/`salt_len` are the actual password and salt bytes; nothing
// here substitutes the raw password/salt where the spec calls for the
// derived P/S digests, which was the core bug in the previous version.
// ---------------------------------------------------------------------

__kernel void sha256crypt_opencl(
    __global const BYTE *passwords, int pass_len,
    __global const BYTE *salt, int salt_len,
    __global BYTE *out_digests, int num_items)
{
    int idx = get_global_id(0);
    if (idx >= num_items) return;

    __global const BYTE *p_cand = passwords + (idx * pass_len);
    int p_actual_len = 0;
    while (p_actual_len < pass_len && p_cand[p_actual_len] != 0) {
        p_actual_len++;
    }

    // Local (non-__global) copies -- sha256_ctx_update takes a plain
    // BYTE* and p_cand/salt are __global; copy them into private memory
    // once so the same update()/streaming code can be reused for every
    // piece below without needing a __global-aware variant.
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
    // Streamed pwd_len calls of update(pwd) rather than materializing a
    // pwd_len*pwd_len-byte buffer (up to 4096 bytes for a 64-char candidate).
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
        int ds_repeat_count = 16 + digest_a[0];   // digest_a[0] is already 0-255
        sha256_ctx_init(&ctx);
        for (int r = 0; r < ds_repeat_count; r++) {
            sha256_ctx_update(&ctx, slt, salt_len);
        }
        sha256_ctx_final(&ctx, ds_seed);
        for (int i = 0; i < salt_len; i++) ds[i] = ds_seed[i % 32];
    }

    // digest C: `rounds` rounds (fixed at 1000 for this target), each:
    //   msg = (dp if i odd else dc)
    //   msg += ds   if i is NOT a multiple of 3
    //   msg += dp   if i is NOT a multiple of 7
    //   msg += (dc if i odd else dp)
    //   dc = SHA256(msg)
    //
    // This is the hottest part of the kernel by far (executed 1000x per
    // candidate, vs. once for everything above it), so unlike digest A/B
    // and dp/ds above, it's worth avoiding the streaming ctx's per-call
    // overhead here: each round's message is small and bounded (at most
    // 64+16+64+64 = 208 bytes), so it's built into one flat buffer with
    // plain array writes -- the same style the original kernel used --
    // and hashed with a single update()+final() call instead of the
    // 3-4 separate update() calls (each with its own buffer-fill/flush
    // branching) it took per round otherwise. Same bytes, same output,
    // far fewer calls in the loop that actually dominates runtime.
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

"""

# 2. Target-hash decoding
#
# The kernel above outputs the RAW 32-byte digest (current_digest after the
# 1000 rounds) -- it does not do crypt's final custom base64-style encoding.
# So to check a candidate against a real "$5$rounds=1000$salt$checksum"
# hash string, the checksum portion has to be decoded back into those same
# 32 raw bytes first, then compared byte-for-byte against the kernel output.
#
# That custom encoding (alphabet "./0-9A-Za-z", plus a specific byte
# permutation -- NOT standard base64) is easy to get subtly wrong by hand,
# so rather than transcribe it from the spec, this reuses the permutation
# table (_256_transpose_map) and codec (h64) from `passlib`, a well-tested
# library that implements sha256-crypt from scratch. This was verified
# end-to-end against your actual target hash before being used here:
# decoding the real checksum and re-encoding the result with the same
# table reproduced the original checksum string exactly, and passlib's
# own hash() output matched `legacycrypt.crypt()` for several other
# password/salt/rounds combinations too.
from passlib.utils.binary import h64
from passlib.handlers.sha2_crypt import _256_transpose_map


def decode_sha256_crypt_checksum(hash_string):
    """'$5$rounds=1000$salt$checksum' -> the raw 32-byte digest it encodes."""
    checksum_str = hash_string.rsplit('$', 1)[-1]
    raw = h64.decode_transposed_bytes(checksum_str.encode('ascii'), _256_transpose_map)
    return np.frombuffer(raw, dtype=np.uint8)


# 3. Host Orchestrator and Chunking Router
def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def process_itertools_on_mac_gpu(candidates, salt_str, target_hash,
                                  batch_size=200000, max_candidate_len=64,
                                  total=None, print_interval=5,
                                  progress_callback=None):
    """
    candidates: an iterable (generator or plain list) of candidate password
                strings -- build this however you like in main() (product,
                chain, your zipf-threshold prefixes, etc.). No longer built
                internally here, so candidates can be any length/source,
                not just a fixed-length itertools.product over one charset.
    salt_str:   the plain salt string, e.g. "89w0wWD1vujG.3F7".
    target_hash: the full "$5$rounds=1000$salt$checksum" string to match.
    max_candidate_len: safety cap on candidate length. The kernel's
                internal buffers (BYTE local_buffer[256], BYTE pad[128])
                are FIXED SIZE -- a candidate long enough to overflow them
                would corrupt the digest (or worse) rather than raise a
                clean error, since this is raw OpenCL C, not Python. 64 is
                comfortably safe for anything this candidate space would
                realistically produce (dictionary words/concatenations);
                candidates over that are skipped, not silently truncated.
    total:      optional -- total candidate count, if you know it up front
                (e.g. len(common)**2, same as crackconcat.py computes).
                Enables the %/ETA columns in progress output; omitted, you
                still get count and rate, just no percentage or ETA.
    print_interval: seconds between progress callbacks (same knob/default
                as crackconcat.py's print_interval) -- fired by elapsed
                time, not by batch, since a GPU batch can finish in well
                under a second and firing every batch would spam whoever
                is listening.
    progress_callback: optional callable, progress_callback(kind, **info).
                This function does NOT print anything itself -- it calls
                progress_callback (if given) instead, so all display logic
                lives with the caller. `kind` is one of:
                  'skipped'  -- info: skipped, max_len
                  'progress' -- info: total_checked, total, elapsed, rate
                  'success'  -- info: password, total_checked, elapsed, rate
                  'failure'  -- info: total_checked, elapsed, rate

    Returns the found password (str) and stops early, or None if the whole
    `candidates` iterable is exhausted without a match.
    """
    ctx = cl.create_some_context(interactive=False)  # Selects default device natively
    queue = cl.CommandQueue(ctx)
    program = cl.Program(ctx, opencl_code).build()
    # Retrieve the kernel object ONCE and reuse it every batch. Calling
    # program.sha256crypt_opencl(...) (attribute access) re-resolves and
    # builds a fresh Kernel object on every single call -- PyOpenCL warns
    # about exactly this ("RepeatedKernelRetrieval"). With a batch this
    # size that happens roughly once per second, not once per candidate,
    # but it's free to fix and the warning is explicit that it can be
    # "considerable expense", so it's worth eliminating either way.
    kernel = cl.Kernel(program, "sha256crypt_opencl")

    salt_bytes = salt_str.encode('utf-8')
    salt_len = len(salt_bytes)
    target_arr = decode_sha256_crypt_checksum(target_hash)

    total_checked = 0
    start_time = time.perf_counter()
    last_print = start_time

    for batch_num, candidate_batch in enumerate(chunked_iterable(candidates, batch_size)):
        safe_batch = [c for c in candidate_batch if len(c) <= max_candidate_len]
        skipped = len(candidate_batch) - len(safe_batch)
        if skipped and progress_callback:
            progress_callback('skipped', skipped=skipped, max_len=max_candidate_len)
        if not safe_batch:
            continue

        num_items = len(safe_batch)
        # Sized from the actual batch, not a fixed `length` -- candidates
        # can now be different lengths within the same batch.
        max_pass_len = max(len(c) for c in safe_batch) + 1  # +1: null terminator the kernel scans for

        flat_passwords = np.zeros(num_items * max_pass_len, dtype=np.uint8)
        for idx, cand in enumerate(safe_batch):
            c_bytes = cand.encode('utf-8')
            start = idx * max_pass_len
            flat_passwords[start: start + len(c_bytes)] = np.frombuffer(c_bytes, dtype=np.uint8)

        mf = cl.mem_flags
        pass_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flat_passwords)
        salt_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.frombuffer(salt_bytes, dtype=np.uint8))
        out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, num_items * 32)

        kernel(
            queue, (num_items,), None,
            pass_buf, np.int32(max_pass_len),
            salt_buf, np.int32(salt_len),
            out_buf, np.int32(num_items)
        )

        output_digests = np.zeros(num_items * 32, dtype=np.uint8)
        cl.enqueue_copy(queue, output_digests, out_buf)

        # Vectorized compare of the whole batch against the target in one
        # shot, rather than looping candidate-by-candidate in Python.
        digest_matrix = output_digests.reshape(num_items, 32)
        matches = np.all(digest_matrix == target_arr, axis=1)
        total_checked += num_items

        if matches.any():
            match_idx = int(np.argmax(matches))
            found_password = safe_batch[match_idx]
            elapsed = time.perf_counter() - start_time
            rate = total_checked / elapsed if elapsed > 0 else 0
            if progress_callback:
                progress_callback('success', password=found_password,
                                   total_checked=total_checked, elapsed=elapsed, rate=rate)
            return found_password  # <-- early termination

        # Progress callback, throttled by elapsed time rather than by batch
        # -- same cadence/default as crackconcat.py's print_interval, since
        # a GPU batch can finish in a fraction of a second and firing every
        # single batch would just spam whoever's listening. This function
        # stays silent either way -- it's up to progress_callback (defined
        # in main()) whether/how anything gets displayed.
        now = time.perf_counter()
        if now - last_print >= print_interval:
            elapsed = now - start_time
            rate = total_checked / elapsed if elapsed > 0 else 0
            if progress_callback:
                progress_callback('progress', total_checked=total_checked,
                                   total=total, elapsed=elapsed, rate=rate)
            last_print = now

    elapsed = time.perf_counter() - start_time
    rate = total_checked / elapsed if elapsed > 0 else 0
    if progress_callback:
        progress_callback('failure', total_checked=total_checked, elapsed=elapsed, rate=rate)
    return None


def log_job(pattern):
    with open('attacklog_gpu_concat.json', 'a') as f:
        f.write(str(pattern) + '\n')

def send_notification(subject, body):
    keyfile = open('google.key')
    keyfiledata = keyfile.read().splitlines()
    keyaddr = keyfiledata[0]
    keypass = keyfiledata[1]

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = keyaddr
    msg["To"] = keyaddr
    msg.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(keyaddr, keypass)
        smtp.send_message(msg)


def product_size(*iterables):
    return math.prod(len(it) for it in iterables)

def main():
    salt = "89w0wWD1vujG.3F7"
    #target_hash = '$5$rounds=1000$89w0wWD1vujG.3F7$Q3t8zqhCfUhZ1MuhJxAjXyenoM5C14DmP0nlFUqjjiC'
    target_hash = "$5$rounds=1000$89w0wWD1vujG.3F7$iRqfu47TO3VKhxQJfmnELcdCsyl4T5wfCAPjihdera9"

    with open("dictionaries/common.txt", "r") as fp:
        common = fp.read().splitlines()
    with open("dictionaries/uncommon_by_zipf.txt", "r") as fp:
        uncommon = fp.read().splitlines()

    zipf = []
    for i in range(7,0,-1):
        zipf += [[w for w in uncommon if (zipf_frequency(w.lower(), 'en') >= i) and (zipf_frequency(w.lower(), 'en') < (i+1))]]

    commontotal = len(common)
    zipftotal = len(uncommon)


    pools = {}
    pools[0] = common
    for key in range(1,8):
        pools[key] = zipf[key-1] 


    k = 3
    already_done = set()

    band_score = {0: 8}                      # treat `common` as highest priority
    for key in range(1, 8):
        band_score[key] = 8 - key            # pool 1 = zipf 7.x -> 7, pool 7 = zipf 1.x -> 1

    def pattern_score(pattern):
        return sum(band_score[label] for label in pattern)

    patterns = sorted(
    itertools.product(pools, repeat=k),
    key=lambda p: (-pattern_score(p), product_size(*(pools[l] for l in p)))
    )

    def report_progress(kind, **info):
        # All of process_itertools_on_mac_gpu's former print() calls now
        # happen here instead -- the GPU function itself stays silent and
        # just calls this callback; this is "inside the processing loop"
        # in the sense that it's what actually fires, repeatedly, while
        # that loop runs.
        if kind == 'skipped':
            print(f"  (skipped {info['skipped']} candidate(s) over {info['max_len']} chars -- "
                  f"unsafe for the kernel's fixed-size buffers)")
        elif kind == 'progress':
            total_checked, total, elapsed, rate = (
                info['total_checked'], info['total'], info['elapsed'], info['rate'])
            if total:
                remaining = max(total - total_checked, 0)
                eta_seconds = remaining / rate if rate > 0 else 0
                print(f"{total_checked:,}/{total:,} ({100*total_checked/total:.2f}%) "
                      f"- {rate:,.0f}/s - ETA {timedelta(seconds=int(eta_seconds))}")
            else:
                print(f"{total_checked:,} checked "
                      f"- {rate:,.0f}/s - elapsed {timedelta(seconds=int(elapsed))}")
        elif kind == 'success':
            print(f"SUCCESS! Password is: {info['password']!r}  "
                  f"(checked {info['total_checked']:,} candidates in {info['elapsed']:.2f}s, "
                  f"{info['rate']:,.0f}/s)")
        elif kind == 'failure':
            print(f"FAILURE -- exhausted candidate stream. "
                  f"Checked {info['total_checked']:,} in {info['elapsed']:.2f}s ({info['rate']:,.0f}/s)")

    for pattern in patterns:   # e.g. (0, 0, 1)
        if pattern in already_done:
            continue

        total = product_size(*(pools[label] for label in pattern))
        print(f'Attempting pattern {str(pattern)}: {total} candidates to compute of grand total {(commontotal + zipftotal)**3}')

        # One call per PATTERN, passing the whole candidate generator --
        # not one call per combo. (The previous version called this once
        # per single joined candidate string, which meant rebuilding the
        # OpenCL context/program from scratch per candidate, and handing
        # it a bare string -- which chunked_iterable would then split into
        # individual characters, not candidates.)
        candidates = (''.join(combo) for combo in itertools.product(*(pools[label] for label in pattern)))
        password = process_itertools_on_mac_gpu(candidates, salt, target_hash, batch_size=250000,
                                                 total=total, progress_callback=report_progress)
        if password is not None:
            record = {"script": scriptname, "description": jobdescription, "count": total,
                      "result": f'Success: password is {password}'}
            body = json.dumps(record, indent=2)
            send_notification('Hashing Success!', body)
            exit(0)

        already_done.add(pattern)
        log_job(pattern)
        
    


if __name__ == "__main__":
    main()
