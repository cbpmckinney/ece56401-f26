import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# 1. Write the CUDA Kernel (Highly unrolled, optimized for parallel execution)
# This processes independent 64-byte chunks (standard SHA-256 block size)
cuda_code = """
typedef unsigned int WORD;

#define ROTR(len,val) (((val) >> (len)) | ((val) << (32 - (len))))
#define SHR(len,val) ((val) >> (len))
#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(2,x) ^ ROTR(13,x) ^ ROTR(22,x))
#define EP1(x) (ROTR(6,x) ^ ROTR(11,x) ^ ROTR(25,x))
#define SIG0(x) (ROTR(7,x) ^ ROTR(18,x) ^ SHR(3,x))
#define SIG1(x) (ROTR(17,x) ^ ROTR(19,x) ^ SHR(10,x))

__device__ const WORD k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

extern "C" __global__ void sha256_batch(const BYTE *in_data, WORD *out_hashes, int num_items) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;

    // Fixed 64-byte chunks per item
    const BYTE *msg = in_data + (idx * 64);
    
    WORD a = 0x6a09e667, b = 0xbb67ae85, c = 0x3c6ef372, d = 0xa54ff53a;
    WORD e = 0x510e527f, f = 0x9b05688c, g = 0x1f83d9ab, h = 0x5be0cd19;
    
    WORD w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = (msg[i*4] << 24) | (msg[i*4+1] << 16) | (msg[i*4+2] << 8) | (msg[i*4+3]);
    }
    for (int i = 16; i < 64; i++) {
        w[i] = SIG1(w[i-2]) + w[i-7] + SIG0(m[i-15]) + w[i-16];
    }

    for (int i = 0; i < 64; i++) {
        WORD t1 = h + EP1(e) + CH(e,f,g) + k[i] + w[i];
        WORD t2 = EP0(a) + MAJ(a,b,c);
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }

    // Write 32-byte (8 WORDs) output back to global host array
    int out_offset = idx * 8;
    out_hashes[out_offset]   = 0x6a09e667 + a;
    out_hashes[out_offset+1] = 0xbb67ae85 + b;
    out_hashes[out_offset+2] = 0x3c6ef372 + c;
    out_hashes[out_offset+3] = 0xa54ff53a + d;
    out_hashes[out_offset+4] = 0x510e527f + e;
    out_hashes[out_offset+5] = 0x9b05688c + f;
    out_hashes[out_offset+6] = 0x1f83d9ab + g;
    out_hashes[out_offset+7] = 0x5be0cd19 + h;
}
"""

# Compile kernel code via CUDA compiler JIT
mod = SourceModule(cuda_code.replace('BYTE', 'unsigned char'))
sha256_kernel = mod.get_function("sha256_batch")

# 2. Python Host Helper Function
def hash_strings_on_gpu(string_list):
    num_items = len(string_list)
    
    # Pack input items into fixed 64-byte blocks for parallel thread indexing
    # Ensure strings are shorter than 55 bytes to leave space for standard SHA padding if needed, 
    # or handle padding manually in your Python generation routine.
    packed_inputs = np.zeros((num_items, 64), dtype=np.uint8)
    
    for i, s in enumerate(string_list):
        encoded = s.encode('utf-8')
        length = len(encoded)
        packed_inputs[i, :length] = np.frombuffer(encoded, dtype=np.uint8)
        
        # Simple standard SHA-256 message padding rule (0x80 marker + length in bits)
        packed_inputs[i, length] = 0x80
        bits_length = length * 8
        packed_inputs[i, 60] = (bits_length >> 24) & 0xFF
        packed_inputs[i, 61] = (bits_length >> 16) & 0xFF
        packed_inputs[i, 62] = (bits_length >> 8) & 0xFF
        packed_inputs[i, 63] = bits_length & 0xFF

    # Allocate pinned output space for the resulting 32-byte hashes (8 uint32 elements)
    output_hashes = np.zeros(num_items * 8, dtype=np.uint32)

    # Grid mapping parameters
    threads_per_block = 256
    blocks_per_grid = (num_items + threads_per_block - 1) // threads_per_block

    # Run execution directly on GPU hardware 
    sha256_kernel(
        cuda.In(packed_inputs), 
        cuda.Out(output_hashes), 
        np.int32(num_items),
        block=(threads_per_block, 1, 1), 
        grid=(blocks_per_grid, 1)
    )

    # Convert results into human-readable hex digests
    hex_results = []
    for i in range(num_items):
        words = output_hashes[i*8 : (i+1)*8]
        # Reconstruct full 64-character hexadecimal notation string
        hex_str = "".join(f"{word:08x}" for word in words)
        hex_results.append(hex_str)
        
    return hex_results

# --- Example Evaluation ---
if __name__ == "__main__":
    # Generate items to process in Python
    candidates = [f"password_candidate_{i}" for i in range(100000)]
    
    print(f"Processing {len(candidates)} items on the GPU...")
    hashes = hash_strings_on_gpu(candidates)
    
    print(f"First result: {candidates[0]} -> {hashes[0]}")
