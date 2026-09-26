"""
NVIDIA/CUDA multi-GPU sha256crypt cracker -- PERSISTENT WORKER POOL variant.

This is a new file, not a rewrite of gpu_nvidia_cuda.py: same kernel, same
pools/pattern-sorting/resume-log/found-password logic, same synchronized
split-per-pattern multi-GPU strategy -- only HOW the GPU workers are
managed has changed.

WHY THIS EXISTS
---------------
In gpu_nvidia_cuda.py, process_pattern_multi_gpu() spawns a brand-new OS
process per GPU for EVERY pattern, and each fresh process re-imports
everything (numpy, pycuda, wordfreq's frequency tables, passlib),
re-initializes a CUDA context, recompiles/reloads the kernel module, and
re-copies the salt to the device -- all before it does any actual
hashing. None of that touches the GPU at real load, so nvtop/nvidia-smi
shows the card idling, spooling up, then spooling down every time a
pattern boundary is crossed. For a pattern with billions of candidates
that startup cost is negligible; for smaller patterns it's a much bigger
fraction of the pattern's wall-clock time, which is exactly the "rate
drifting down over time" you saw.

The fix: each GPU's worker process is spawned exactly ONCE, at the start
of the whole run, and does its CUDA init / kernel compile / salt upload
exactly once too -- since the salt and target hash never change between
patterns anyway. From then on it just sits in a loop pulling pattern
assignments off a queue. No more idle-spool-idle sawtooth at pattern
boundaries.

WHAT'S UNCHANGED FROM gpu_nvidia_cuda.py
-----------------------------------------
_run_candidate_batches() (the actual batch-dispatch/matching loop) is
copied verbatim -- it's already been tested and used on real hardware,
and none of that logic needed to change. Same for the kernel file
(sha256crypt_kernel.cu, read from disk, not duplicated), the resume log
(load_already_done/log_job), the found-password file
(save_found_password), the pools/zipf/pattern_sort_key logic, and the
email notifications. Only the process-management layer
(_gpu_pattern_worker / process_pattern_multi_gpu) is replaced, by
_persistent_gpu_worker / GpuWorkerPool below.

VERIFICATION
------------
As with the original multi-GPU wiring, this was verified end to end with
a fake CUDA backend that computes real passlib sha256crypt hashes (no
GPU is available in the environment this was written in) -- see
persistent_pool_integration_test.py. That test specifically exercises the
NEW risk this design introduces that the old spawn-per-pattern design
didn't have: state carrying over between patterns on the SAME worker
process (stop_event correctly cleared between patterns, the reused salt
buffer/kernel handle still producing correct results on the second and
third pattern dispatched to a worker, no cross-talk between patterns).
Still, please do a short real run on your dual-GPU machine before
trusting this for an unattended multi-hour job -- same caveat as before.
"""

import os
import sys
import time
import json
import math
import ast
import smtplib
import itertools
import multiprocessing as mp
from datetime import timedelta, datetime
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
jobdescription = "DOWNGRADE Concatenation triples (NVIDIA/CUDA, persistent multi-GPU pool)"

KERNEL_SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sha256crypt_kernel.cu")

# Explicit spawn context -- required so worker processes get a fresh
# interpreter (no inherited CUDA state) instead of a fork() copy.
MP_CTX = mp.get_context('spawn')

# Which CUDA device indices to use, and how many GPUs to split each
# pattern across. Check yours with `nvidia-smi -L` (indices are 0-based in
# the order nvidia-smi lists them). Use a single-element list (e.g. [0])
# to pin this to one GPU instead of splitting.
GPU_DEVICES = [0, 1]

# Same log paths as gpu_nvidia_cuda.py, on purpose -- so swapping to this
# file mid-run picks up exactly where the old one left off (already_done)
# and writes to the same found-password file.
ALREADY_DONE_LOG_PATH = 'attacklog_gpu_concat_nvidia.json'
FOUND_PASSWORD_LOG_PATH = 'found_password.txt'

# How long to wait for all GPU workers to finish starting up (CUDA init +
# kernel compile) before giving up and raising -- generous, since a cold
# nvcc compile can take a few seconds.
POOL_STARTUP_TIMEOUT = 120


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
    Unchanged from gpu_nvidia_cuda.py -- see that file for the full
    docstring. Shared batching/dispatch/matching loop, now called by
    _persistent_gpu_worker once per pattern task instead of once per
    spawned process.
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
                                      batch_size=2000000, max_candidate_len=64,
                                      total=None, print_interval=5,
                                      progress_callback=None, threads_per_block=256,
                                      device_id=0):
    """
    Single-GPU CUDA path, unchanged from gpu_nvidia_cuda.py -- kept here
    too for standalone use/testing on one card. main() below uses the
    persistent GpuWorkerPool instead.
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


def _persistent_gpu_worker(device_id, worker_id, salt_str, target_hash, kernel_source_path,
                            task_queue, result_queue, stop_event, threads_per_block=256):
    """
    Entry point for a LONG-LIVED worker process, one per GPU, spawned
    exactly once for the whole run. Does all the one-time setup (CUDA
    context, kernel compile, salt upload) ONCE, reports back
    'worker_ready' (or 'worker_error' if setup failed), then loops
    forever pulling pattern tasks off task_queue:

      task = (pattern_id, sub_pool, rest_pools, batch_size,
              max_candidate_len, print_interval)

    until it receives the sentinel None, at which point it pops its CUDA
    context and exits. Per-task results go on result_queue tagged with
    both worker_id and pattern_id:
      - 'progress' / 'skipped' as it goes (relayed, filtered to just
        those two kinds, same as the old per-pattern worker)
      - exactly one 'worker_done' per task, with the password found (or
        None) and how many candidates that task checked
      - 'task_error' instead, if something raised partway through a
        task -- reported without killing the worker process, so one bad
        pattern doesn't take down the whole pool for the rest of the run

    Sets stop_event as soon as it finds a password, so its sibling
    worker(s) on other GPUs stop between batches instead of grinding
    through their remaining slice of the SAME pattern. The parent clears
    stop_event before dispatching each new pattern's tasks -- by
    construction that only happens after every worker has already
    reported 'worker_done' for the previous pattern, so there's no race
    between a worker still checking stop_event for pattern N and the
    parent clearing it for pattern N+1.
    """
    try:
        cuda.init()
        dev = cuda.Device(device_id)
        ctx = dev.make_context()
        with open(kernel_source_path, "r") as f:
            kernel_source = f.read()
        module = SourceModule(kernel_source)
        kernel = module.get_function("sha256crypt_cuda")

        salt_bytes = salt_str.encode('utf-8')
        salt_len = len(salt_bytes)
        salt_arr = np.frombuffer(salt_bytes, dtype=np.uint8)
        target_arr = decode_sha256_crypt_checksum(target_hash)

        salt_buf = cuda.mem_alloc(salt_arr.nbytes)
        cuda.memcpy_htod(salt_buf, salt_arr)
    except Exception as exc:
        result_queue.put({'worker_id': worker_id, 'pattern_id': None, 'kind': 'worker_error',
                           'info': {'error': f"failed to initialize GPU worker on device {device_id}: {exc}"}})
        return

    result_queue.put({'worker_id': worker_id, 'pattern_id': None, 'kind': 'worker_ready', 'info': {}})

    current_pattern_id = {'value': None}

    def relay(kind, **info):
        if kind in ('progress', 'skipped'):
            result_queue.put({'worker_id': worker_id, 'pattern_id': current_pattern_id['value'],
                               'kind': kind, 'info': info})

    try:
        while True:
            task = task_queue.get()
            if task is None:  # shutdown sentinel
                break

            pattern_id, sub_pool, rest_pools, batch_size, max_candidate_len, print_interval = task
            current_pattern_id['value'] = pattern_id

            try:
                candidates = (''.join(combo) for combo in itertools.product(sub_pool, *rest_pools))
                password, total_checked, _elapsed = _run_candidate_batches(
                    kernel, candidates, salt_buf, salt_len, target_arr,
                    batch_size, max_candidate_len, total=None, print_interval=print_interval,
                    progress_callback=relay, stop_event=stop_event, threads_per_block=threads_per_block)

                if password is not None:
                    stop_event.set()

                result_queue.put({'worker_id': worker_id, 'pattern_id': pattern_id, 'kind': 'worker_done',
                                   'info': {'password': password, 'total_checked': total_checked}})
            except Exception as exc:
                result_queue.put({'worker_id': worker_id, 'pattern_id': pattern_id, 'kind': 'task_error',
                                   'info': {'error': f"worker {worker_id} (device {device_id}) failed on "
                                                      f"pattern {pattern_id}: {exc}"}})
    finally:
        ctx.pop()


class GpuWorkerPool:
    """
    Manages one long-lived worker process per GPU in gpu_devices. Spawn
    once with start(), then call run_pattern() once per pattern (as many
    times as you like -- that's the whole point), and shutdown() once at
    the very end.
    """

    def __init__(self, gpu_devices, salt_str, target_hash,
                 kernel_source_path=KERNEL_SOURCE_PATH, threads_per_block=256):
        self.gpu_devices = gpu_devices
        self.salt_str = salt_str
        self.target_hash = target_hash
        self.kernel_source_path = kernel_source_path
        self.threads_per_block = threads_per_block

        self.task_queues = []
        self.procs = []
        self.result_queue = MP_CTX.Queue()
        self.stop_event = MP_CTX.Event()
        self._next_pattern_id = 0
        self._started = False

    def start(self):
        """Spawn all worker processes and block until every one of them
        has finished CUDA init + kernel compile (or raise if any failed)."""
        if self._started:
            return
        for worker_id, device_id in enumerate(self.gpu_devices):
            task_queue = MP_CTX.Queue()
            self.task_queues.append(task_queue)
            p = MP_CTX.Process(target=_persistent_gpu_worker, args=(
                device_id, worker_id, self.salt_str, self.target_hash, self.kernel_source_path,
                task_queue, self.result_queue, self.stop_event, self.threads_per_block))
            p.start()
            self.procs.append(p)

        ready = 0
        errors = []
        deadline = time.time() + POOL_STARTUP_TIMEOUT
        while ready < len(self.procs):
            if time.time() > deadline:
                raise RuntimeError(
                    f"timed out after {POOL_STARTUP_TIMEOUT}s waiting for GPU workers to start "
                    f"(CUDA init/kernel compile) -- check the console for a worker traceback"
                )
            try:
                msg = self.result_queue.get(timeout=1)
            except Exception:
                for p in self.procs:
                    if not p.is_alive() and p.exitcode not in (0, None):
                        raise RuntimeError(
                            f"a GPU worker process died during startup (exit code {p.exitcode}) "
                            f"-- check the console for its traceback"
                        )
                continue
            if msg['kind'] == 'worker_ready':
                ready += 1
            elif msg['kind'] == 'worker_error':
                errors.append(msg['info']['error'])
                ready += 1

        if errors:
            self.shutdown()
            raise RuntimeError("GPU worker(s) failed to start: " + "; ".join(errors))

        self._started = True

    def run_pattern(self, pool_tuple, batch_size=2000000, max_candidate_len=64,
                     print_interval=5, total=None, progress_callback=None):
        """
        Runs one whole pattern across every worker, synchronized
        split-per-pattern style, exactly like process_pattern_multi_gpu()
        in gpu_nvidia_cuda.py did -- except this reuses the already-running
        worker processes instead of spawning new ones. Returns the
        password once every dispatched worker has reported in (found or
        exhausted its slice); never returns partway through a pattern.
        """
        if not self._started:
            raise RuntimeError("GpuWorkerPool.start() must be called before run_pattern()")

        pattern_id = self._next_pattern_id
        self._next_pattern_id += 1

        self.stop_event.clear()

        first_pool = pool_tuple[0]
        rest_pools = pool_tuple[1:]
        n = len(first_pool)
        n_workers = min(len(self.gpu_devices), max(n, 1))
        bounds = [round(i * n / n_workers) for i in range(n_workers + 1)]

        checked_per_worker = {}
        dispatched = []
        for worker_id in range(n_workers):
            lo, hi = bounds[worker_id], bounds[worker_id + 1]
            sub_pool = first_pool[lo:hi]
            if not sub_pool:
                continue
            checked_per_worker[worker_id] = 0
            dispatched.append(worker_id)
            self.task_queues[worker_id].put(
                (pattern_id, sub_pool, rest_pools, batch_size, max_candidate_len, print_interval))

        found_password = None
        done_workers = 0
        task_errors = []
        start_time = time.perf_counter()
        last_combined_print = start_time

        while done_workers < len(dispatched):
            try:
                msg = self.result_queue.get(timeout=1)
            except Exception:
                for worker_id in dispatched:
                    p = self.procs[worker_id]
                    if not p.is_alive() and p.exitcode not in (0, None):
                        raise RuntimeError(
                            f"GPU worker {worker_id} (device {self.gpu_devices[worker_id]}) died "
                            f"unexpectedly mid-pattern (exit code {p.exitcode}) -- check the console"
                        )
                continue

            # A message tagged with an older pattern_id shouldn't normally
            # arrive here (each worker fully finishes one task before
            # taking the next), but ignore it defensively rather than let
            # it corrupt this pattern's bookkeeping.
            if msg.get('pattern_id') not in (pattern_id, None):
                continue

            worker_id = msg['worker_id']
            if msg['kind'] == 'worker_done':
                checked_per_worker[worker_id] = msg['info']['total_checked']
                if msg['info']['password'] is not None:
                    found_password = msg['info']['password']
                done_workers += 1
            elif msg['kind'] == 'task_error':
                task_errors.append(msg['info']['error'])
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

        if task_errors:
            raise RuntimeError("GPU worker task(s) failed: " + "; ".join(task_errors))

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

    def shutdown(self):
        """Tell every worker to stop (sentinel) and wait for them to exit.
        Safe to call more than once."""
        for q in self.task_queues:
            try:
                q.put(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)
        self._started = False


def save_found_password(password, pattern, total, log_path=FOUND_PASSWORD_LOG_PATH):
    """
    Append the found password (plus context: which pattern it came from,
    how many candidates that pattern had, and a timestamp) to a plain
    text file, in addition to printing it and emailing it -- a third,
    independent way to make sure it isn't lost.
    """
    record = (
        f"FOUND at {datetime.now().isoformat(timespec='seconds')}\n"
        f"  script:      {scriptname}\n"
        f"  description: {jobdescription}\n"
        f"  pattern:     {pattern}\n"
        f"  candidates in pattern: {total:,}\n"
        f"  password:    {password}\n\n"
    )
    with open(log_path, 'a') as f:
        f.write(record)


def load_already_done(log_path=ALREADY_DONE_LOG_PATH):
    """
    Reload the set of already-completed (fully-exhausted) patterns from
    the attack log, so a restarted run resumes where a previous one left
    off instead of re-checking patterns that can take hours each. Starts
    fresh (empty set) if the log doesn't exist yet. Tolerates a stray
    unparseable line (e.g. if the file got truncated mid-write) by
    skipping it with a warning rather than crashing the whole run.
    """
    done = set()
    if not os.path.exists(log_path):
        return done
    with open(log_path, 'r') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                done.add(ast.literal_eval(line))
            except (ValueError, SyntaxError):
                print(f"  (warning: skipping unparseable line {line_no} in {log_path}: {line!r})")
    return done


def log_job(pattern, log_path=ALREADY_DONE_LOG_PATH):
    with open(log_path, 'a') as f:
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
    already_done = load_already_done()
    if already_done:
        print(f"Resuming: {len(already_done)} pattern(s) already completed per "
              f"{ALREADY_DONE_LOG_PATH}, skipping those.")

    def pattern_sort_key(pattern):
        # Most-probable-first: a pattern's RAREST component dominates how
        # unlikely a candidate from it is, so sort primarily by the
        # highest-numbered (rarest) pool label present in the pattern, then
        # by the next-rarest, and so on. Pool 0 (common) is most probable;
        # pool 7 (zipf 1.x) is least. This guarantees, e.g., that every
        # pattern built only from common words and zipf bands 6-7 is fully
        # exhausted before any pattern containing so much as one zipf-5
        # word is attempted -- and the same all the way down to zipf-1.
        return tuple(sorted(pattern, reverse=True))

    patterns = sorted(itertools.product(pools, repeat=k), key=pattern_sort_key)

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

    pool = GpuWorkerPool(GPU_DEVICES, salt, target_hash)
    print(f"Starting {len(GPU_DEVICES)} persistent GPU worker(s) on device(s) {GPU_DEVICES}...")
    pool.start()
    print("GPU worker(s) ready.")

    try:
        for pattern in patterns:   # e.g. (0, 0, 1)
            if pattern in already_done:
                #body = f'Skipping pattern {pattern}'
                #subject = body
                #send_notification(subject, body)
                continue

            pool_tuple = tuple(pools[label] for label in pattern)
            total = product_size(*pool_tuple)
            print(f'Attempting pattern {str(pattern)}: {total} candidates to compute of grand total {(commontotal + zipftotal)**3}')
            send_notification(f'Starting pattern {pattern}', f'Starting pattern {pattern}')

            password = pool.run_pattern(pool_tuple, batch_size=2000000, total=total,
                                         progress_callback=report_progress)
            if password is not None:
                save_found_password(password, pattern, total)
                record = {"script": scriptname, "description": jobdescription, "count": total,
                          "result": f'Success: password is {password}'}
                body = json.dumps(record, indent=2)
                send_notification('Hashing Success!', body)
                return

            already_done.add(pattern)
            log_job(pattern)
            send_notification(f'Completed pattern {pattern}', f'Completed pattern {pattern}')
    finally:
        pool.shutdown()


if __name__ == "__main__":
    main()
