import numpy as np
import pycuda.driver as cuda
import pycuda.compiler as compiler
import threading

# We drop the string logic entirely. 
# The GPU kernel now accepts an integer 'start_index' and computes its own letters.
cuda_kernel_src = """
typedef unsigned char BYTE;
typedef unsigned int WORD;

// [Include all SHA-256 compression functions and K tables here as before]

extern "C" __global__ void sha256crypt_gpu_generator(
    unsigned long long start_index,
    const BYTE *charset, int charset_len,
    int combo_length, const BYTE *salt, int salt_len,
    BYTE *out_digests, int num_items) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;

    // Calculate this thread's global odometer value
    unsigned long long global_idx = start_index + idx;
    
    // 1. ONSITE KEY GENERATION (Replaces itertools.product)
    BYTE p_cand[16]; // Hard cap candidate length 
    unsigned long long temp = global_idx;
    
    // Unroll the combinatorial state backward into characters
    for(int i = combo_length - 1; i >= 0; i--) {
        p_cand[i] = charset[temp % charset_len];
        temp /= charset_len;
    }
    int p_actual_len = combo_length;

    // --- [Proceed with Phase 1 and Phase 2 1000-round hashing exactly as before] ---
    // The key 'p_cand' is now built natively in the thread's ultra-fast registers!
}
"""

def run_on_gpu_device(device_id, start_idx, num_items, charset, combo_len, salt):
    """Worker thread function to execute an exact batch slice on a specific target GPU device"""
    cuda.init()
    dev = cuda.Device(device_id)
    ctx = dev.make_context()
    
    try:
        # Compile kernel locally within this device context
        mod = compiler.SourceModule(cuda_kernel_src)
        kernel = mod.get_function("sha256crypt_gpu_generator")
        
        charset_bytes = charset.encode('utf-8')
        salt_bytes = salt.encode('utf-8')
        output_digests = np.zeros(num_items * 32, dtype=np.uint8)
        
        threads_per_block = 256
        blocks_per_grid = (num_items + threads_per_block - 1) // threads_per_block
        
        kernel(
            np.uint64(start_idx),
            cuda.In(np.frombuffer(charset_bytes, dtype=np.uint8)), np.int32(len(charset_bytes)),
            np.int32(combo_len),
            cuda.In(np.frombuffer(salt_bytes, dtype=np.uint8)), np.int32(len(salt_bytes)),
            cuda.Out(output_digests), np.int32(num_items),
            block=(threads_per_block, 1, 1), grid=(blocks_per_grid, 1)
        )
        
        # Process output_digests here or check for matches...
        
    finally:
        ctx.pop()

def parallel_brute_force(charset, length, salt, total_combinations, step_size=4194304):
    """Orchestrates total keyspace and splits steps evenly across both RTX 4090s"""
    current_offset = 0
    
    while current_offset < total_combinations:
        # Give half the chunk to GPU 0, half to GPU 1
        half_step = min(step_size // 2, (total_combinations - current_offset) // 2)
        if half_step == 0: 
            half_step = total_combinations - current_offset
            
        t0 = threading.Thread(target=run_on_gpu_device, args=(0, current_offset, half_step, charset, length, salt))
        threads = [t0]
        
        if current_offset + half_step < total_combinations:
            t1 = threading.Thread(target=run_on_gpu_device, args=(1, current_offset + half_step, half_step, charset, length, salt))
            threads.append(t1)
            
        for t in threads: t.start()
        for t in threads: t.join()
        
        current_offset += (half_step * len(threads))
        print(f"Progress: {current_offset:,} / {total_combinations:,} processed.")

if __name__ == "__main__":
    charset = "abcdefghijklmnopqrstuvwxyz0123456789"
    length = 6 
    salt = "89w0wWD1vujG.3F7"
    
    total_combos = len(charset) ** length # 36^6 = 2,176,782,336
    
    # Process in large 4-Million execution grids to saturate both massive GPUs
    parallel_brute_force(charset, length, salt, total_combos, step_size=4194304)
