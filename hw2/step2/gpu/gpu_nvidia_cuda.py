"""
NVIDIA/CUDA port of gpu_mac_test1.py, with multi-GPU support.

Same search strategy, same pools/pattern-sorting logic, same host-side
batching/progress-callback design as the Mac (PyOpenCL) version -- only
the GPU dispatch mechanics changed, since that's the only part that's
actually platform-specific:

  * pyopencl                              -> pycuda
  * cl.Program(...).build()               -> pycuda.compiler.SourceModule
    (compiling sha256crypt_kernel.cu, read from disk -- kept as its own
    file rather than embedded as a string here, so there's one source of
    truth for the kernel instead of two copies that could drift apart)
  * cl.Buffer / cl.enqueue_copy           -> cuda.mem_alloc / cuda.memcpy_htod
                                              / cuda.memcpy_dtoh
  * queue.enqueue_nd_range (num_items,)   -> explicit block=(threads,1,1),
                                              grid=(blocks,1) launch config

Requires the `pycuda` package and a working CUDA toolkit/driver -- neither
is available in the environment this was written and tested in, so unlike
sha256crypt_kernel.c (verified against your Mac's real GPU), the kernel
itself has only been verified algorithmically: compiled as plain C++
(CUDA-specific syntax stubbed out) and checked against the same 95
known-answer test vectors used for the OpenCL version -- all 95 matched.
Real nvcc compilation and actual GPU execution were confirmed working by
you on the real dual-GPU machine (single-GPU path).

MULTI-GPU DESIGN (this revision)
---------------------------------
Uses the "synchronized split-per-pattern" strategy we settled on, not a
round-robin "dispatch whole patterns to whichever GPU is free" queue:
for each pattern, the pattern's FIRST pool is split into contiguous
slices, one per GPU, and every GPU works on the SAME pattern at the same
time. The search only moves on to the next pattern (in the existing
priority-sorted order) once every GPU has either found the password or
fully exhausted its slice. This was chosen over the "first available"
model specifically so that a pattern being ruled out is a definite,
synchronized fact -- both halves get checked before moving on -- rather
than an emergent property of a shared queue.

Each GPU is driven by its own spawned OS process (see MP_CTX below).
This is a hard requirement, not a style choice: CUDA contexts must NOT be
inherited across fork() -- a forked child sharing the parent's CUDA
context corrupts state on both sides. multiprocessing.get_context('spawn')
starts each worker as a fresh interpreter instead of forking, and
`pycuda.autoinit` is intentionally NOT imported anywhere in this file,
because importing it at module level would create a CUDA context in the
*parent* process before any workers are spawned. Instead, every function
that touches the GPU (the single-GPU standalone path too) creates its own
context explicitly with cuda.init() / cuda.Device(N).make_context().

The multiprocessing coordination logic itself (queue draining, stop_event
early-termination, join, crash detection) was verified offline against a
mock GPU stand-in before being wired up here, since no GPU is available
in the environment this was written in -- see mp_orchestration_test.py.
Still, as with the CUDA port itself, please do a short real run on your
actual dual-GPU machine before trusting this for an unattended multi-hour
job.
"""

import os
import sys
import time
import json
import math
import smtplib
import itertools
import multiprocessing as mp
from datetime import timedelta
from email.message import EmailMessage

import numpy as np
from wordfreq import zipf_frequency

try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except ImportError:
    sys.exit(
        "pycuda is required for this script (pip install pycuda), and needs a "
        "working NVIDIA CUDA toolkit/driver on this machine to build against."
    )

from passlib.utils.binary import h64
from passlib.handlers.sha2_crypt import _256_transpose_map


scriptname = os.path.basename(__file__)
jobdescription = "DOWNGRADE Concatenation triples (NVIDIA/CUDA, multi-GPU)"

KERNEL_SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sha256crypt_kernel.cu")

# Explicit spawn context -- required so worker processes get a fresh
# interpreter (no inherited CUDA state) instead of a fork() copy.
MP_CTX = mp.get_context('spawn')

# Which CUDA device indices to use, and how many GPUs to split each
# pattern across. Check yours with `nvidia-smi -L` (indices are 0-based in
# the order nvidia-smi lists them). Use a single-element list (e.g. [0])
# to pin this to one GPU instead of splitting.
GPU_DEVICES = [0, 1]


def decode_sha256_crypt_checksum(hash_string):
    """'$5$rounds=1000$salt$checksum' -> the raw 32-byte digest it encodes."""
    checksum_str = hash_string.rsplit('$', 1)[-1]
    raw = h64.decode_transposed_bytes(checksum_str.encode('ascii'), _256_transpose_map)
    return np.frombuffer(raw, dtype=np.uint8)


def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def _run_candidate_batches(kernel, candidates, salt_buf, salt_len, target_arr,
                            batch_size, max_candidate_len, total, print_interval,
                            progress_callback, stop_event=None, threads_per_block=256):
    """
    Shared batching/dispatch/matching loop used by BOTH the single-GPU
    standalone path (process_candidates_on_nvidia_gpu) and each multi-GPU
    worker process (_gpu_pattern_worker). Owns none of the device context
    or kernel compilation -- the caller sets those up, since each worker
    process needs its own independent CUDA context.

    If `stop_event` is given, it's checked between batches so a sibling
    worker can be told to give up early once another worker (on a
    different GPU, working the other half of the same pattern) has
    already found the match.

    Returns (password_or_None, total_checked, elapsed_seconds). Calls
    progress_callback for every kind ('skipped', 'progress', 'success',
    'failure') exactly like the single-GPU version always has -- when this
    is used from a multi-GPU worker, the worker wraps this in a filtering
    'relay' so only 'progress'/'skipped' make it out to the parent process;
    the parent synthesizes its own combined 'success'/'failure' once all
    workers for a pattern have finished.
    """
    total_checked = 0
    start_time = time.perf_counter()
    last_print = start_time

    for candidate_batch in chunked_iterable(candidates, batch_size):
        if stop_event is not None and stop_event.is_set():
            break

        safe_batch = [c for c in candidate_batch if len(c) <= max_candidate_len]
        skipped = len(candidate_batch) - len(safe_batch)
        if skipped and progress_callback:
            progress_callback('skipped', skipped=skipped, max_len=max_candidate_len)
        if not safe_batch:
            continue

        num_items = len(safe_batch)
        max_pass_len = max(len(c) for c in safe_batch) + 1  # +1: null terminator the kernel scans for

        flat_passwords = np.zeros(num_items * max_pass_len, dtype=np.uint8)
        for idx, cand in enumerate(safe_batch):
            c_bytes = cand.encode('utf-8')
            start = idx * max_pass_len
            flat_passwords[start: start + len(c_bytes)] = np.frombuffer(c_bytes, dtype=np.uint8)

        pass_buf = cuda.mem_alloc(flat_passwords.nbytes)
        cuda.memcpy_htod(pass_buf, flat_passwords)
        out_buf = cuda.mem_alloc(num_items * 32)

        blocks = (num_items + threads_per_block - 1) // threads_per_block
        kernel(pass_buf, np.int32(max_pass_len),
               salt_buf, np.int32(salt_len),
               out_buf, np.int32(num_items),
               block=(threads_per_block, 1, 1), grid=(blocks, 1))

        output_digests = np.empty(num_items * 32, dtype=np.uint8)
        cuda.memcpy_dtoh(output_digests, out_buf)

        # Explicitly free this batch's device buffers rather than relying
        # on Python GC to get to them eventually -- this loop can allocate
        # a lot of short-lived device memory over a long run.
        pass_buf.free()
        out_buf.free()

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
            return found_password, total_checked, elapsed

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
    return None, total_checked, elapsed


def process_candidates_on_nvidia_gpu(candidates, salt_str, target_hash,
                                      batch_size=200000, max_candidate_len=64,
                                      total=None, print_interval=5,
                                      progress_callback=None, threads_per_block=256,
                                      device_id=0):
    """
    Single-GPU CUDA path -- kept for standalone use/testing on one card.
    For a real multi-hour run across both GPUs, main() now calls
    process_pattern_multi_gpu() instead; this function is what that one
    calls internally, once per GPU, via _gpu_pattern_worker.

    Creates its own CUDA context explicitly on `device_id` (no
    pycuda.autoinit -- see the module docstring for why) and pops it again
    before returning.
    """
    cuda.init()
    dev = cuda.Device(device_id)
    ctx = dev.make_context()
    try:
        with open(KERNEL_SOURCE_PATH, "r") as f:
            kernel_source = f.read()
        module = SourceModule(kernel_source)
        kernel = module.get_function("sha256crypt_cuda")

        salt_bytes = salt_str.encode('utf-8')
        salt_len = len(salt_bytes)
        salt_arr = np.frombuffer(salt_bytes, dtype=np.uint8)
        target_arr = decode_sha256_crypt_checksum(target_hash)

        salt_buf = cuda.mem_alloc(salt_arr.nbytes)
        cuda.memcpy_htod(salt_buf, salt_arr)

        password, _total_checked, _elapsed = _run_candidate_batches(
            kernel, candidates, salt_buf, salt_len, target_arr,
            batch_size, max_candidate_len, total, print_interval,
            progress_callback, stop_event=None, threads_per_block=threads_per_block)
        return password
    finally:
        ctx.pop()


def _gpu_pattern_worker(device_id, worker_id, sub_pool, rest_pools, salt_str, target_hash,
                         batch_size, max_candidate_len, print_interval, stop_event, result_queue):
    """
    Entry point for one spawned worker process, one per GPU. Builds its
    own candidate stream from its slice of the pattern's first pool
    (sub_pool) crossed with the rest of the pattern's pools (rest_pools),
    creates its own CUDA context on `device_id`, and reports back to the
    parent via result_queue:
      - 'progress' / 'skipped' messages as it goes (relayed from
        _run_candidate_batches, filtered to just those two kinds)
      - exactly one final 'worker_done' message with the password it
        found (or None) and how many candidates it checked
      - 'worker_error' instead, if CUDA init itself failed on this device

    Sets stop_event as soon as it finds the password, so its sibling
    worker(s) on other GPUs stop between batches instead of grinding
    through their full remaining slice.
    """
    try:
        cuda.init()
        dev = cuda.Device(device_id)
        ctx = dev.make_context()
    except Exception as exc:
        result_queue.put({'worker_id': worker_id, 'kind': 'worker_error',
                           'info': {'error': f"failed to init CUDA device {device_id}: {exc}"}})
        return

    try:
        with open(KERNEL_SOURCE_PATH, "r") as f:
            kernel_source = f.read()
        module = SourceModule(kernel_source)
        kernel = module.get_function("sha256crypt_cuda")

        salt_bytes = salt_str.encode('utf-8')
        salt_len = len(salt_bytes)
        salt_arr = np.frombuffer(salt_bytes, dtype=np.uint8)
        target_arr = decode_sha256_crypt_checksum(target_hash)

        salt_buf = cuda.mem_alloc(salt_arr.nbytes)
        cuda.memcpy_htod(salt_buf, salt_arr)

        def relay(kind, **info):
            if kind in ('progress', 'skipped'):
                result_queue.put({'worker_id': worker_id, 'kind': kind, 'info': info})

        candidates = (''.join(combo) for combo in itertools.product(sub_pool, *rest_pools))

        password, total_checked, _elapsed = _run_candidate_batches(
            kernel, candidates, salt_buf, salt_len, target_arr,
            batch_size, max_candidate_len, total=None, print_interval=print_interval,
            progress_callback=relay, stop_event=stop_event)

        if password is not None:
            stop_event.set()

        result_queue.put({'worker_id': worker_id, 'kind': 'worker_done',
                           'info': {'password': password, 'total_checked': total_checked}})
    finally:
        ctx.pop()


def process_pattern_multi_gpu(pool_tuple, salt_str, target_hash, gpu_devices,
                               batch_size=250000, max_candidate_len=64,
                               total=None, print_interval=5, progress_callback=None):
    """
    Runs one whole pattern (e.g. common x common x common) across all of
    gpu_devices at once, synchronized split-per-pattern style: pool_tuple[0]
    is split into len(gpu_devices) contiguous slices, one worker process
    per GPU, all working this SAME pattern together. Returns the password
    once every worker has reported in (found or exhausted its slice) --
    never returns partway through a pattern, so a pattern that comes back
    empty is a definite, fully-checked "no" across both GPUs.
    """
    first_pool = pool_tuple[0]
    rest_pools = pool_tuple[1:]
    n = len(first_pool)
    n_workers = min(len(gpu_devices), max(n, 1))
    bounds = [round(i * n / n_workers) for i in range(n_workers + 1)]

    stop_event = MP_CTX.Event()
    result_queue = MP_CTX.Queue()
    checked_per_worker = {}

    procs = []
    for worker_id in range(n_workers):
        lo, hi = bounds[worker_id], bounds[worker_id + 1]
        sub_pool = first_pool[lo:hi]
        if not sub_pool:
            continue
        device_id = gpu_devices[worker_id]
        checked_per_worker[worker_id] = 0
        p = MP_CTX.Process(target=_gpu_pattern_worker, args=(
            device_id, worker_id, sub_pool, rest_pools, salt_str, target_hash,
            batch_size, max_candidate_len, print_interval, stop_event, result_queue))
        procs.append(p)
        p.start()

    found_password = None
    done_workers = 0
    worker_errors = []
    start_time = time.perf_counter()
    last_combined_print = start_time

    while done_workers < len(procs):
        try:
            msg = result_queue.get(timeout=1)
        except Exception:
            # No message in the last second -- normal while a batch is
            # still running on the GPU. Just make sure nothing silently
            # died without telling us.
            for p in procs:
                if not p.is_alive() and p.exitcode not in (0, None):
                    raise RuntimeError(
                        f"a GPU worker process exited unexpectedly "
                        f"(exit code {p.exitcode}) -- check the console for its traceback"
                    )
            continue

        worker_id = msg['worker_id']
        if msg['kind'] == 'worker_done':
            checked_per_worker[worker_id] = msg['info']['total_checked']
            if msg['info']['password'] is not None:
                found_password = msg['info']['password']
            done_workers += 1
        elif msg['kind'] == 'worker_error':
            worker_errors.append(msg['info']['error'])
            done_workers += 1
        elif msg['kind'] == 'progress':
            checked_per_worker[worker_id] = msg['info']['total_checked']
        elif msg['kind'] == 'skipped':
            if progress_callback:
                progress_callback('skipped', **msg['info'])

        now = time.perf_counter()
        if now - last_combined_print >= print_interval:
            combined_checked = sum(checked_per_worker.values())
            elapsed = now - start_time
            rate = combined_checked / elapsed if elapsed > 0 else 0
            if progress_callback:
                progress_callback('progress', total_checked=combined_checked,
                                   total=total, elapsed=elapsed, rate=rate)
            last_combined_print = now

    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
            p.join(timeout=10)

    if worker_errors:
        raise RuntimeError("GPU worker(s) failed: " + "; ".join(worker_errors))

    combined_checked = sum(checked_per_worker.values())
    elapsed = time.perf_counter() - start_time
    rate = combined_checked / elapsed if elapsed > 0 else 0

    if found_password is not None:
        if progress_callback:
            progress_callback('success', password=found_password,
                               total_checked=combined_checked, elapsed=elapsed, rate=rate)
        return found_password

    if progress_callback:
        progress_callback('failure', total_checked=combined_checked, elapsed=elapsed, rate=rate)
    return None


def log_job(pattern):
    with open('attacklog_gpu_concat_nvidia.json', 'a') as f:
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
    target_hash = '$5$rounds=1000$89w0wWD1vujG.3F7$iRqfu47TO3VKhxQJfmnELcdCsyl4T5wfCAPjihdera9'

    with open("dictionaries/common.txt", "r") as fp:
        common = fp.read().splitlines()
    with open("dictionaries/uncommon_by_zipf.txt", "r") as fp:
        uncommon = fp.read().splitlines()

    zipf = []
    for i in range(7, 0, -1):
        zipf += [[w for w in uncommon if (zipf_frequency(w.lower(), 'en') >= i) and (zipf_frequency(w.lower(), 'en') < (i + 1))]]

    commontotal = len(common)
    zipftotal = len(uncommon)

    pools = {}
    pools[0] = common
    for key in range(1, 8):
        pools[key] = zipf[key - 1]

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

        pool_tuple = tuple(pools[label] for label in pattern)
        total = product_size(*pool_tuple)
        print(f'Attempting pattern {str(pattern)}: {total} candidates to compute of grand total {(commontotal + zipftotal)**3}')

        password = process_pattern_multi_gpu(pool_tuple, salt, target_hash, GPU_DEVICES,
                                              batch_size=250000, total=total,
                                              progress_callback=report_progress)
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
