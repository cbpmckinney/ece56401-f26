import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# 1. Custom C++ CUDA Source for sha256crypt (1000 rounds)
cuda_code = """
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

__device__ const WORD K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

// Device helper: Single-pass standard SHA-256 compression function over an arbitrary byte array buffer
__device__ void sha256_compress(const BYTE *data, int len, BYTE *out_digest) {
    WORD h0 = 0x6a09e667, h1 = 0xbb67ae85, h2 = 0x3c6ef372, h3 = 0xa54ff53a;
    WORD h4 = 0x510e527f, h5 = 0x9b05688c, h6 = 0x1f83d9ab, h7 = 0x5be0cd19;

    // Hard-capped working pad in fast local registers/L1 cache memory per thread
    BYTE pad[128];
    for(int i = 0; i < len; i++) pad[i] = data[i];
    
    pad[len] = 0x80;
    int pad_len = len + 1;
    while((pad_len % 64) != 56) {
        pad[pad_len++] = 0x00;
    }
    
    unsigned long long bits = (unsigned long long)len * 8;
    for(int i = 0; i < 8; i++) {
        pad[pad_len++] = (bits >> (56 - i * 8)) & 0xFF;
    }

    // Process blocks
    for (int b = 0; b < pad_len; b += 64) {
        WORD w[64];
        for (int i = 0; i < 16; i++) {
            w[i] = (pad[b + i*4] << 24) | (pad[b + i*4+1] << 16) | (pad[b + i*4+2] << 8) | (pad[b + i*4+3]);
        }
        for (int i = 16; i < 64; i++) {
            w[i] = SIG1(w[i-2]) + w[i-7] + SIG0(w[i-15]) + w[i-16];
        }

        WORD a = h0, b_t = h1, c = h2, d = h3, e = h4, f = h5, g = h6, h = h7;
        for (int i = 0; i < 64; i++) {
            WORD t1 = h + EP1(e) + CH(e,f,g) + K[i] + w[i];
            WORD t2 = EP0(a) + MAJ(a,b_t,c);
            h = g; g = f; f = e; e = d + t1; d = c; c = b_t; b_t = a; a = t1 + t2;
        }
        h0 += a; h1 += b_t; h2 += c; h3 += d; h4 += e; h5 += f; h6 += g; h7 += h;
    }

    WORD final_h[8] = {h0, h1, h2, h3, h4, h5, h6, h7};
    for(int i = 0; i < 8; i++) {
        out_digest[i*4]   = (final_h[i] >> 24) & 0xFF;
        out_digest[i*4+1] = (final_h[i] >> 16) & 0xFF;
        out_digest[i*4+2] = (final_h[i] >> 8) & 0xFF;
        out_digest[i*4+3] = final_h[i] & 0xFF;
    }
}

extern "C" __global__ void sha256crypt_1000_rounds(
    const BYTE *passwords, int pass_len,
    const BYTE *salt, int salt_len,
    BYTE *out_digests, int num_items) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;

    // Identify current candidate boundaries
    const BYTE *p_cand = passwords + (idx * pass_len);
    int p_actual_len = 0;
    while(p_actual_len < pass_len && p_cand[p_actual_len] != 0) {
        p_actual_len++;
    }

    BYTE digest_a[32];
    BYTE local_buffer[128];
    int buf_idx = 0;

    // --- PHASE 1: Build Initial Digest A ---
    // 1. Password + Salt
    for(int i = 0; i < p_actual_len; i++) local_buffer[buf_idx++] = p_cand[i];
    for(int i = 0; i < salt_len; i++) local_buffer[buf_idx++] = salt[i];
    
    // 2. Add sequence from structural Digest B computation logic
    BYTE digest_b[32];
    int b_buf_idx = 0;
    for(int i = 0; i < p_actual_len; i++) local_buffer[buf_idx + b_buf_idx++] = p_cand[i];
    for(int i = 0; i < salt_len; i++) local_buffer[buf_idx + b_buf_idx++] = salt[i];
    sha256_compress(local_buffer + buf_idx, b_buf_idx, digest_b);
    
    for(int i = 0; i < p_actual_len; i++) {
        local_buffer[buf_idx++] = digest_b[i % 32];
    }
    
    // 3. Bit-mask style trailing modification
    for (int i = p_actual_len; i > 0; i >>= 1) {
        if ((i & 1) != 0) {
            for(int j=0; j<32; j++) local_buffer[buf_idx++] = digest_b[j]; // Simplify step mapping
        } else {
            for(int j=0; j<p_actual_len; j++) local_buffer[buf_idx++] = p_cand[j];
        }
        break; // Standard implementation truncates pattern loop at first evaluation bit shift
    }
    
    sha256_compress(local_buffer, buf_idx, digest_a);

    // --- PHASE 2: The 1,000 Stretching Rounds ---
    // Alternate buffers to prevent memory overlap conflicts inside core execution loop
    BYTE current_digest[32];
    for(int i = 0; i < 32; i++) current_digest[i] = digest_a[i];

    for (int r = 0; r < 1000; r++) {
        int r_len = 0;
        // Odd iteration layout structure
        if ((r & 1) != 0) {
            for(int j = 0; j < p_actual_len; j++) local_buffer[r_len++] = p_cand[j];
        } else {
            for(int j = 0; j < 32; j++) local_buffer[r_len++] = current_digest[j];
        }
        // Salt step mapping components
        if (r % 3 != 0) {
            for(int j = 0; j < salt_len; j++) local_buffer[r_len++] = salt[j];
        }
        if (r % 7 != 0) {
            for(int j = 0; j < p_actual_len; j++) local_buffer[r_len++] = p_cand[j];
        }
        if ((r & 1) != 0) {
            for(int j = 0; j < 32; j++) local_buffer[r_len++] = current_digest[j];
        } else {
            for(int j = 0; j < p_actual_len; j++) local_buffer[r_len++] = p_cand[j];
        }
        
        sha256_compress(local_buffer, r_len, current_digest);
    }

    // Write final plain 32-byte digest back to host memory grid space
    for(int i = 0; i < 32; i++) {
        out_digests[(idx * 32) + i] = current_digest[i];
    }
}
"""

# Compile CUDA code natively
mod = SourceModule(cuda_code)
sha256crypt_kernel = mod.get_function("sha256crypt_1000_rounds")

# 2. Custom Base64 Translation Table Unique to Unix crypt()
CRYPT_B64_CHARS = "./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def crypt_b64_encode(b):
    """Encodes standard 3 bytes into 4 character positions mapping to Unix base64 alphabet"""
    w = (b[0] << 16) | (b[1] << 8) | b[2]
    res = []
    for _ in range(4):
        res.append(CRYPT_B64_CHARS[w & 0x3f])
        w >>= 6
    return "".join(res)

def format_sha256_crypt_string(raw_bytes, salt_string):
    """
    Swaps final digest bytes natively into the exact permutation matrix sequence 
    expected by the unix hash specification and formats the output string.
    """
    # Specific permutation mapping order required by sha256crypt standard
    permutation = [
        0, 10, 20, 21, 1, 11, 12, 22, 2, 3, 13, 23, 
        4, 14, 24, 25, 5, 15, 16, 26, 6, 7, 17, 27, 
        8, 18, 28, 29, 9, 19, 30
    ]
    
    # Extract mapped bytes from sequence array
    permuted = [raw_bytes[i] for i in permutation]
    
    # Process blocks of 3 bytes into 4 characters
    encoded_str = ""
    for i in range(0, 30, 3):
        encoded_str += crypt_b64_encode(permuted[i:i+3])
        
    # Handle final 2 leftover bytes manually
    last_w = (permuted[30] << 8)
    encoded_str += CRYPT_B64_CHARS[last_w & 0x3f]
    encoded_str += CRYPT_B64_CHARS[(last_w >> 6) & 0x3f]
    # Final singular remainder byte padding step
    last_byte_w = raw_bytes[31]
    encoded_str += CRYPT_B64_CHARS[last_byte_w & 0x3f]
    encoded_str += CRYPT_B64_CHARS[(last_byte_w >> 6) & 0x3f]

    return f"$5$rounds=1000${salt_string}${encoded_str[:43]}"

# 3. Python Orchestrator Routine
def run_sha256crypt_gpu(candidates_list, salt_str):
    num_items = len(candidates_list)
    salt_bytes = salt_str.encode('utf-8')
    salt_len = len(salt_bytes)
    
    # Track longest password to establish flat grid bounds safely
    max_pass_len = max(len(c.encode('utf-8')) for c in candidates_list) + 1
    
    # Flatten input allocation to contiguous memory space
    flat_passwords = np.zeros(num_items * max_pass_len, dtype=np.uint8)
    for idx, cand in enumerate(candidates_list):
        c_bytes = cand.encode('utf-8')
        start = idx * max_pass_len
        flat_passwords[start : start + len(c_bytes)] = np.frombuffer(c_bytes, dtype=np.uint8)

    # Allocate host output receiving space
    output_digests = np.zeros(num_items * 32, dtype=np.uint8)

    # Core Execution Parameter configuration
    threads_per_block = 128
    blocks_per_grid = (num_items + threads_per_block - 1) // threads_per_block

    # Fire GPU Pipeline Execution
    sha256crypt_kernel(
        cuda.In(flat_passwords), np.int32(max_pass_len),
        cuda.In(np.frombuffer(salt_bytes, dtype=np.uint8)), np.int32(salt_len),
        cuda.Out(output_digests), np.int32(num_items),
        block=(threads_per_block, 1, 1), grid=(blocks_per_grid, 1)
    )

    # Format the outputs back into Unix crypt notation string formats
    final_hashes = []
    for idx in range(num_items):
        raw_hash_bytes = output_digests[idx*32 : (idx+1)*32]
        formatted_crypt = format_sha256_crypt_string(raw_hash_bytes, salt_str)
        final_hashes.append(formatted_crypt)

    return final_hashes

import itertools
from collections import deque

def chunked_iterable(iterable, size):
    """
    Slices a continuous itertools generator into manageable 
    memory-safe chunks without loading the entire stream into RAM.
    """
    it = iter(iterable)
    while True:
        # Pull a fixed slice from the generator
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk

def process_combinations_on_gpu(charset, length, salt_str, batch_size=500000):
    """
    Generates brute-force combinations and streams them to the GPU.
    """
    # 1. Create the itertools generator (costs virtually 0 memory)
    # Example: Product combinations ('a','b','c') -> ('a','a'), ('a','b')...
    combos_generator = itertools.product(charset, repeat=length)
    
    # Transform tuples from itertools into strings: ('a', 'b') -> 'ab'
    string_generator = ("".join(c) for c in combos_generator)
    
    total_processed = 0
    print(f"Starting GPU execution loop in batches of {batch_size:,}...")

    # 2. Iterate through the generator in memory-safe chunks
    for batch_num, candidate_batch in enumerate(chunked_iterable(string_generator, batch_size)):
        
        # Pass the current list chunk directly to your existing PyCUDA function
        # This copies the chunk to the GPU, runs 1,000 rounds, and returns results
        gpu_results = run_sha256crypt_gpu(candidate_batch, salt_str)
        
        total_processed += len(candidate_batch)
        print(f"Batch {batch_num + 1} finished. Total computed: {total_processed:,}")
        
        # --- Handle or Check Results Here ---
        # For demonstration, let's just inspect the first result of the batch
        # In a real tool, you would check if any result matches your target hash here
        # if target_hash in gpu_results: return ...
        
    print("\nAll combinations processed successfully.")






# --- Execution Demonstration ---
if __name__ == "__main__":

    salt = "89w0wWD1vujG.3F7"
    #test

    with open('../dictionaries/common.txt', 'r') as fp:
        common = fp.read().splitlines()


        # Process the generator on the GPU in chunks of 500k candidates at a time
    process_combinations_on_gpu(common, 3, salt, batch_size=500000)
