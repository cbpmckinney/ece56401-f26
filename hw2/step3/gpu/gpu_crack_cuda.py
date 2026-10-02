"""
CUDA port of gpu_crack_mac.py (the PyOpenCL version), for the dual-RTX-4090
machine, following the same host-side conventions already proven in
../../step2/gpu/gpu_nvidia_cuda.py's CUDA port of the Step 2 cracker:

  * pyopencl                            -> pycuda
  * cl.Program(...).build()             -> pycuda.compiler.SourceModule,
    compiling sha256_aes_ctr_kernel.cu (a straight CUDA port of
    sha256_aes_ctr_kernel.c, see that file's header)
  * cl.Buffer / cl.enqueue_copy         -> cuda.mem_alloc / cuda.memcpy_htod
                                            / cuda.memcpy_dtoh
  * queue.enqueue_nd_range (num_items,) -> explicit block=(threads,1,1),
                                            grid=(blocks,1) launch config

MAIN() STRUCTURE IS DELIBERATELY KEPT LIKE gpu_crack_mac.py's: it still
just imports MUTATION_PATTERNS/pattern_count/ciphertext straight from
stuffingtest2.py and chains them into one candidate stream -- NOT step2's
elaborate per-pattern pool-splitting/resume-log/email-notification
design, which doesn't fit here (MUTATION_PATTERNS is several
differently-shaped generators chained together, not one big clean
product pool to slice evenly up front).

MULTI-GPU DESIGN (two RTX 4090s)
---------------------------------
Generalizes gpu_crack_mac.py's single-GPU threaded producer/consumer
overlap (added after mactop showed ~50% utilization there -- see that
file) to multiple GPUs: ONE producer -- running in this process, never
touching CUDA -- builds GPU-ready batches and feeds them onto a single
shared multiprocessing.Queue; one worker PROCESS per GPU in GPU_DEVICES
pulls from that same queue and dispatches to its own device. Whichever
GPU finishes its current batch first just grabs the next one off the
queue -- natural load balancing, no static up-front split of the
candidate space the way step2's per-pattern-slice design does (that
doesn't generalize cleanly to several differently-shaped generators).

Each GPU is driven by its own spawned OS PROCESS, not a thread -- this is
a hard CUDA requirement, not a style choice (see gpu_nvidia_cuda.py's
docstring): CUDA contexts must NOT be inherited across fork(), so this
uses multiprocessing.get_context('spawn'), never imports pycuda.autoinit
at module level, and every function that touches the GPU creates its own
context explicitly with cuda.init() / cuda.Device(N).make_context() /
ctx.pop(). stuffingtest2.py is intentionally NOT imported at module
level (only main() needs it) -- this module gets re-imported from
scratch in every spawned worker process, and the workers never touch
st2's ~300K-line dictionaries, so importing it up here would mean every
worker process redundantly reloads them for nothing.

Batches cross the process boundary (producer -> worker) as raw `bytes`
rather than numpy arrays, since bytes pickle/unpickle in CPython close
to a straight buffer copy, cheaper than numpy's pickling overhead. At
very high combined throughput this inter-process copy could itself
become the next bottleneck (watch for workers spending a lot of time
blocked on the queue in that case) -- not worth the complexity of
building batches independently inside each worker before confirming
that's actually necessary.

VERIFICATION NOTE: no CUDA toolchain/GPU is available in the environment
this was written in. Same two-pronged verification as the OpenCL kernel:
(1) the algorithm itself was already checked in pure Python against this
repo's real encrypt.py (`cryptography`-library-backed) across several
test vectors; (2) this .cu file's EXACT source was then re-verified by
stubbing out CUDA-specific syntax (extern "C" __global__ / __device__ /
__constant__ / blockIdx.x*blockDim.x+threadIdx.x) and compiling+running
it as plain C++ against 10 known-answer test vectors -- including one at
the real secret.html.crypt's exact 1673-byte plaintext length -- all 10
passed. The multiprocessing producer/consumer/sentinel/early-stop control
flow was likewise verified against a pure-Python mock GPU stand-in
before being wired to real PyCUDA calls (same approach used for the
OpenCL version's threaded pipeline). None of that replaces an actual
nvcc compile and real GPU run on your machine -- run --sanity-check
first and confirm it reports PASSED before trusting a "no match" result
from a long unattended run.
"""

import os
import sys
import time
import itertools
import threading
import multiprocessing as mp
import queue as queue_mod
from datetime import timedelta

import numpy as np

try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except ImportError:
    sys.exit(
        "pycuda is required for this script (pip install pycuda), and needs a "
        "working NVIDIA CUDA toolkit/driver on this machine to build against."
    )

STEP3_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# NOTE: stuffingtest2 is intentionally NOT imported here at module level
# -- see the module docstring. Only main() imports it, after this
# process is confirmed to be the real entry point (not a spawned CUDA
# worker, which never needs it). encrypt.py is different: it's just the
# crypto primitives (hashlib + cryptography), no big dictionary files to
# load, so importing it here -- including redundantly in every spawned
# worker -- costs nothing. Needed so a match can be decrypted and shown
# immediately, not just reported as a password string.
sys.path.insert(0, STEP3_DIR)
from encrypt import decrypt

KERNEL_SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sha256_aes_ctr_kernel.cu")

MAX_PWD_LEN = 64      # must match MAX_PWD_LEN in the kernel
MAX_CT_LEN = 2048      # must match MAX_CT_LEN in the kernel

# nvidia-smi -L lists indices in this order (0-based). Use a single-
# element list (e.g. [0]) to pin this to one card instead of splitting.
GPU_DEVICES = [0, 1]

# Same flag/meaning as gpu_crack_mac.py.
INCLUDE_ALREADY_TRIED = False

# Explicit spawn context -- required so worker processes get a fresh
# interpreter (no inherited CUDA state) instead of a fork() copy.
MP_CTX = mp.get_context('spawn')


def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def _build_cuda_context(device_id):
    """Every GPU-touching function (worker process or the single-GPU
    sanity-check path) creates its own context explicitly rather than
    relying on pycuda.autoinit -- see module docstring for why."""
    cuda.init()
    dev = cuda.Device(device_id)
    ctx = dev.make_context()
    return ctx


def _build_kernel():
    with open(KERNEL_SOURCE_PATH, "r") as f:
        kernel_source = f.read()
    module = SourceModule(kernel_source)
    return module.get_function("crack_sha256_aes_ctr")


def _gpu_worker(device_id, worker_id, work_queue, result_queue, stop_event,
                 ciphertext_bytes, target_tag_bytes, threads_per_block=256):
    """One process per GPU. Pulls prebuilt, GPU-ready batches off the
    SHARED work_queue (fed by the producer running in the main process
    -- see run_batches_multi_gpu) and dispatches them to its own CUDA
    device. Natural load balancing: whichever GPU finishes its current
    batch first just pulls the next one off the queue, rather than each
    GPU being statically assigned a fixed half of the candidate space."""
    try:
        ctx = _build_cuda_context(device_id)
    except Exception as exc:
        result_queue.put({'worker_id': worker_id, 'kind': 'worker_error',
                           'info': {'error': f"failed to init CUDA device {device_id}: {exc}"}})
        return

    try:
        kernel = _build_kernel()

        ct_arr = np.frombuffer(ciphertext_bytes, dtype=np.uint8)
        tag_arr = np.frombuffer(target_tag_bytes, dtype=np.uint8)
        ct_buf = cuda.mem_alloc(ct_arr.nbytes)
        cuda.memcpy_htod(ct_buf, ct_arr)
        tag_buf = cuda.mem_alloc(tag_arr.nbytes)
        cuda.memcpy_htod(tag_buf, tag_arr)

        total_checked = 0
        last_report = time.perf_counter()

        while not stop_event.is_set():
            try:
                item = work_queue.get(timeout=1)
            except queue_mod.Empty:
                continue

            if item is None:
                # Sentinel: candidates are exhausted. Put it back so the
                # NEXT worker sharing this queue sees it too, then stop.
                work_queue.put(None)
                break

            safe_batch, packed_bytes, num_items = item
            flat_passwords = np.frombuffer(packed_bytes, dtype=np.uint8)

            pass_buf = cuda.mem_alloc(flat_passwords.nbytes)
            cuda.memcpy_htod(pass_buf, flat_passwords)
            out_buf = cuda.mem_alloc(num_items)

            blocks = (num_items + threads_per_block - 1) // threads_per_block
            kernel(pass_buf, np.int32(MAX_PWD_LEN),
                   ct_buf, np.int32(len(ciphertext_bytes)),
                   tag_buf,
                   out_buf, np.int32(num_items),
                   block=(threads_per_block, 1, 1), grid=(blocks, 1))

            out_match = np.empty(num_items, dtype=np.uint8)
            cuda.memcpy_dtoh(out_match, out_buf)
            pass_buf.free()
            out_buf.free()

            total_checked += num_items
            matches = out_match == 1
            if matches.any():
                match_idx = int(np.argmax(matches))
                found_password = safe_batch[match_idx]
                stop_event.set()
                result_queue.put({'worker_id': worker_id, 'kind': 'found',
                                   'info': {'password': found_password, 'total_checked': total_checked}})
                return

            now = time.perf_counter()
            if now - last_report >= 1.0:
                result_queue.put({'worker_id': worker_id, 'kind': 'progress',
                                   'info': {'total_checked': total_checked}})
                last_report = now

        result_queue.put({'worker_id': worker_id, 'kind': 'worker_done',
                           'info': {'total_checked': total_checked}})
    finally:
        ctx.pop()


def run_batches_multi_gpu(candidates, ciphertext_bytes, target_tag_bytes,
                           gpu_devices=None, batch_size=500000,
                           total=None, print_interval=5, lookahead_per_gpu=3):
    """candidates: an iterable of password-guess strings. Returns the
    matching password (str) and stops early, or None once candidates is
    exhausted with no match across every GPU.

    See module docstring for the overall design: one producer thread
    (right here, CPU-only) feeding a shared queue that N per-GPU worker
    processes consume from."""
    if gpu_devices is None:
        gpu_devices = GPU_DEVICES

    work_queue = MP_CTX.Queue(maxsize=lookahead_per_gpu * len(gpu_devices))
    result_queue = MP_CTX.Queue()
    stop_event = MP_CTX.Event()

    workers = []
    for worker_id, device_id in enumerate(gpu_devices):
        p = MP_CTX.Process(target=_gpu_worker, args=(
            device_id, worker_id, work_queue, result_queue, stop_event,
            ciphertext_bytes, target_tag_bytes))
        workers.append(p)
        p.start()

    def producer_feed():
        """Builds batches and feeds work_queue; stops feeding as soon as
        stop_event is set (a worker found the match) instead of racing
        to finish generating the whole remaining candidate space."""
        for candidate_batch in chunked_iterable(candidates, batch_size):
            if stop_event.is_set():
                return
            safe_batch = [c for c in candidate_batch if len(c) <= MAX_PWD_LEN]
            if not safe_batch:
                continue
            packed = b''.join(
                cand.encode("ascii").ljust(MAX_PWD_LEN, b'\x00')
                for cand in safe_batch
            )
            while not stop_event.is_set():
                try:
                    work_queue.put((safe_batch, packed, len(safe_batch)), timeout=1)
                    break
                except queue_mod.Full:
                    continue
        if not stop_event.is_set():
            work_queue.put(None)

    producer_thread = threading.Thread(target=producer_feed, daemon=True)
    producer_thread.start()

    checked_per_worker = {i: 0 for i in range(len(gpu_devices))}
    found_password = None
    worker_errors = []
    done_workers = 0
    start_time = time.perf_counter()
    last_print = start_time

    while done_workers < len(workers):
        try:
            msg = result_queue.get(timeout=1)
        except queue_mod.Empty:
            for p in workers:
                if not p.is_alive() and p.exitcode not in (0, None):
                    stop_event.set()
                    raise RuntimeError(
                        f"a GPU worker process exited unexpectedly "
                        f"(exit code {p.exitcode}) -- check console output for its traceback"
                    )
            continue

        kind = msg['kind']
        worker_id = msg['worker_id']
        if kind == 'progress':
            checked_per_worker[worker_id] = msg['info']['total_checked']
        elif kind == 'found':
            checked_per_worker[worker_id] = msg['info']['total_checked']
            found_password = msg['info']['password']
            done_workers += 1
        elif kind == 'worker_done':
            checked_per_worker[worker_id] = msg['info']['total_checked']
            done_workers += 1
        elif kind == 'worker_error':
            worker_errors.append(msg['info']['error'])
            done_workers += 1

        now = time.perf_counter()
        if now - last_print >= print_interval:
            combined = sum(checked_per_worker.values())
            elapsed = now - start_time
            rate = combined / elapsed if elapsed > 0 else 0
            per_gpu = ", ".join(f"gpu{gid}={checked_per_worker[i]:,}"
                                for i, gid in enumerate(gpu_devices))
            if total:
                pct = 100 * combined / total
                remaining = total - combined
                eta = timedelta(seconds=int(remaining / rate)) if rate > 0 else "?"
                print(f"{combined:,}/{total:,} ({pct:.2f}%) - {rate:,.0f}/s - "
                      f"ETA {eta} - {per_gpu}")
            else:
                print(f"{combined:,} checked - {rate:,.0f}/s - {per_gpu}")
            last_print = now

    # In case something ended abnormally (e.g. worker_error) without a
    # 'found' message, make sure the producer isn't left blocked forever
    # on a full queue.
    stop_event.set()
    producer_thread.join(timeout=5)

    for p in workers:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
            p.join(timeout=10)

    if worker_errors:
        raise RuntimeError("GPU worker(s) failed: " + "; ".join(worker_errors))

    combined = sum(checked_per_worker.values())
    elapsed = time.perf_counter() - start_time
    rate = combined / elapsed if elapsed > 0 else 0

    if found_password is not None:
        print(f"SUCCESS!  Password is: {found_password!r}  "
              f"({combined:,} checked, {elapsed:.1f}s, {rate:,.0f}/s)")
        try:
            plaintext = decrypt(found_password, ciphertext_bytes + target_tag_bytes)
            print(f"Decrypted content: {plaintext}")
        except Exception as exc:
            print(f"  (warning: a GPU worker reported a match, but re-decrypting on CPU "
                  f"raised {exc!r} -- this shouldn't happen if the kernel is correct; "
                  f"investigate before trusting this result)")
        return found_password

    print(f"FAILURE! {combined:,} checked in {elapsed:.1f}s ({rate:,.0f}/s)")
    return None


def _run_single_gpu_once(device_id, candidate_password, ciphertext_bytes, target_tag_bytes):
    """Minimal, synchronous one-candidate dispatch on a single GPU --
    used only by test_known_password(). The real run goes through
    run_batches_multi_gpu()/the _gpu_worker processes instead."""
    ctx = _build_cuda_context(device_id)
    try:
        kernel = _build_kernel()

        packed = candidate_password.encode("ascii").ljust(MAX_PWD_LEN, b"\x00")
        flat_passwords = np.frombuffer(packed, dtype=np.uint8)
        ct_arr = np.frombuffer(ciphertext_bytes, dtype=np.uint8)
        tag_arr = np.frombuffer(target_tag_bytes, dtype=np.uint8)

        pass_buf = cuda.mem_alloc(flat_passwords.nbytes)
        cuda.memcpy_htod(pass_buf, flat_passwords)
        ct_buf = cuda.mem_alloc(ct_arr.nbytes)
        cuda.memcpy_htod(ct_buf, ct_arr)
        tag_buf = cuda.mem_alloc(tag_arr.nbytes)
        cuda.memcpy_htod(tag_buf, tag_arr)
        out_buf = cuda.mem_alloc(1)

        kernel(pass_buf, np.int32(MAX_PWD_LEN),
               ct_buf, np.int32(len(ciphertext_bytes)),
               tag_buf,
               out_buf, np.int32(1),
               block=(1, 1, 1), grid=(1, 1))

        out_match = np.empty(1, dtype=np.uint8)
        cuda.memcpy_dtoh(out_match, out_buf)
        return bool(out_match[0] == 1)
    finally:
        ctx.pop()


def test_known_password():
    """Run before trusting a real crack attempt -- same two known-answer
    files as gpu_crack_mac.py's sanity check (deliberately NOT the real
    secret.html.crypt: a kernel bug could coincidentally match a wrong
    tag by luck once, so requiring two independent known answers is a
    much stronger check). Only exercises GPU_DEVICES[0] -- this proves
    the KERNEL is correct, which is device-independent; it doesn't need
    to prove both cards work identically (a real problem on the second
    4090 would show up as that worker reporting 'worker_error' during
    the actual multi-GPU run instead)."""
    device_id = GPU_DEVICES[0]
    known_cases = [
        (os.path.join(STEP3_DIR, "server", "templates", "test.crypt"), "password"),
        (os.path.join(STEP3_DIR, "server", "templates", "test2.crypt"), "password1234"),
    ]

    all_passed = True
    for path, known_password in known_cases:
        with open(path, "rb") as fp:
            data = fp.read()
        ciphertext_bytes = data[:-32]
        target_tag_bytes = data[-32:]
        print(f"Running known-password sanity check on GPU {device_id} against "
              f"{os.path.basename(path)} (expected password {known_password!r})...")
        matched = _run_single_gpu_once(device_id, known_password, ciphertext_bytes, target_tag_bytes)
        if matched:
            print(f"  PASSED -- kernel correctly matched {os.path.basename(path)}.")
            try:
                plaintext = decrypt(known_password, data)
                print(f"  Decrypted content: {plaintext}")
            except Exception as exc:
                print(f"  (warning: kernel reported a match, but re-decrypting on CPU "
                      f"raised {exc!r} -- investigate before trusting the kernel)")
                all_passed = False
        else:
            print(f"  FAILED -- do not trust a real run until this passes "
                  f"({os.path.basename(path)}, expected {known_password!r}).")
            all_passed = False

    if all_passed:
        print("Sanity check PASSED on both known files -- safe to trust a real run against "
              "secret.html.crypt now.")
    else:
        print("Sanity check FAILED -- fix the kernel/host script before running against "
              "secret.html.crypt for real.")
    return all_passed


def main():
    sys.path.insert(0, STEP3_DIR)
    os.chdir(STEP3_DIR)
    import stuffingtest2 as st2

    patterns = list(st2.MUTATION_PATTERNS)
    if INCLUDE_ALREADY_TRIED:
        patterns = list(st2.ALREADY_TRIED) + patterns

    print(f"Running {len(patterns)} pattern pool(s) on {len(GPU_DEVICES)} GPU(s) "
          f"({'including' if INCLUDE_ALREADY_TRIED else 'excluding'} ALREADY_TRIED)...")
    total = sum(st2.pattern_count(fn) for fn in patterns)
    print(f"Total candidates: {total}")

    candidates = itertools.chain.from_iterable(fn() for fn in patterns)
    ciphertext_bytes = st2.ciphertext[:-32]
    target_tag_bytes = st2.ciphertext[-32:]

    result = run_batches_multi_gpu(candidates, ciphertext_bytes, target_tag_bytes,
                                    gpu_devices=GPU_DEVICES, total=total)

    if result is not None:
        plaintext = st2.decrypt_target(result)
        print(f"Secret is: {plaintext}")


if __name__ == "__main__":
    if "--sanity-check" in sys.argv:
        test_known_password()
    else:
        main()
