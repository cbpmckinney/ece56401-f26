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
file) to multiple GPUs: N producer PROCESSES (one per candidate-space
SHARD -- see NUM_PRODUCER_SHARDS and stuffingtest2.py's
two_word_substitution_shards()) build GPU-ready batches and feed them
onto a single shared multiprocessing.Queue; one worker PROCESS per GPU
in GPU_DEVICES pulls from that same queue and dispatches to its own
device. Whichever GPU finishes its current batch first just grabs the
next one off the queue -- natural load balancing, no static up-front
split of the candidate space the way step2's per-pattern-slice design
does (that doesn't generalize cleanly to several differently-shaped
generators); the sharding happens one level up, on the GENERATION side.

Earlier version of this file used a SINGLE producer thread instead of N
producer processes. That was fine as long as the GPU was the
bottleneck, but once the kernel occupancy fix made per-candidate GPU
work cheap, one GIL-bound Python thread could no longer generate+pack
candidates fast enough to keep two fast GPUs fed -- visible in nvtop as
both GPUs "working" but mostly idle. True parallelism for pure-Python
candidate generation requires separate OS processes, not threads (the
GIL serializes CPU-bound bytecode across threads in one process), so
generation itself now also runs as N processes, each over a disjoint
slice of the candidate space (see two_word_substitution_shards()'s
docstring for how a slice is carved out without redundant overlap).

SHARED-MEMORY RING BUFFER (why 2x4090 still wasn't "wiping the floor")
------------------------------------------------------------------------
Switching to N producer PROCESSES fixed candidate generation, but
dual-4090 throughput was still nowhere close to what two 4090s should
do (barely edging out a single Mac integrated GPU). Root cause: every
batch -- the full 32MB of packed candidate bytes -- was going through a
single shared multiprocessing.Queue to reach the GPU workers. Measured
directly (see the verification note below): ONE multiprocessing.Queue
tops out around ~370MB/s moving bytes through its one underlying OS
pipe, REGARDLESS of how many producer/consumer processes are attached
to it (more producers contending for the same pipe measured WORSE, not
better). At MAX_PWD_LEN=64 bytes/candidate that's an absolute ceiling
of ~5-6M candidates/s combined, no matter how fast the kernel or the
GPUs are -- which lines up exactly with the two 4090s getting ~7.3M/s
combined.

Fix: producers now write each packed batch directly into a slot of a
preallocated shared-memory pool (num_slots slots, each
batch_size*MAX_PWD_LEN bytes, created once in run_batches_multi_gpu and
attached by name in every producer/worker process). Only a (slot
index, item count) PAIR OF SMALL INTEGERS crosses any
multiprocessing.Queue now (free_slots / ready_queue) -- the 32MB of
actual candidate data never touches a pipe at all. A GPU worker reads
its batch as a zero-copy numpy view straight onto the shared-memory
slot (np.frombuffer(shm.buf, ...)), which is exactly the kind of buffer
cuda.memcpy_htod already expects. A matched password is recovered by
slicing that same slot directly, rather than shipping a parallel list
of candidate strings alongside the bytes (an earlier, smaller version
of this same mistake -- pickling 500,000 individual Python string
objects per batch was ALSO a measurable tax before the shared-memory
redesign made the whole question moot).

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
OpenCL version's threaded pipeline). The shared-memory ring buffer
(free_slots/ready_queue slot handoff, multi-producer + multi-consumer
contention, early-stop mid-generation, password recovery straight from
a shared-memory slot) was verified the same way, under real
multiprocessing.get_context('spawn'), before being wired in here --
including the specific claim that motivated it (a multiprocessing.Queue
carrying full packed batches has a hard, measured bandwidth ceiling
through its one underlying pipe no matter how many producers/consumers
attach to it). What this environment CANNOT verify is actual GPU
throughput -- there's no CUDA toolchain or GPU available here, so the
"2x4090 should wipe the floor" expectation can only be confirmed on
your real machine. None of this replaces an actual nvcc compile and
real GPU run -- run --sanity-check first and confirm it reports PASSED
before trusting a "no match" result from a long unattended run, and
watch nvtop/htop together on the real run to see whether this closed
the gap.
"""

import os
import sys
import time
import itertools
import multiprocessing as mp
from multiprocessing import shared_memory
import queue as queue_mod
import smtplib
from email.message import EmailMessage
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

# Same google.key convention as ../../step2/crackconcat.py -- reused
# directly from step2 (two lines: Gmail address, then an app password
# for it) rather than keeping a second copy of the secret here. Covered
# either way by the repo's "**/google.key" gitignore rule.
GOOGLE_KEY_PATH = os.path.join(os.path.dirname(STEP3_DIR), "step2", "google.key")

KERNEL_SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sha256_aes_ctr_kernel.cu")

MAX_PWD_LEN = 64      # must match MAX_PWD_LEN in the kernel
MAX_CT_LEN = 2048      # must match MAX_CT_LEN in the kernel

# nvidia-smi -L lists indices in this order (0-based). Use a single-
# element list (e.g. [0]) to pin this to one card instead of splitting.
GPU_DEVICES = [0, 1]

# Same flag/meaning as gpu_crack_mac.py.
INCLUDE_ALREADY_TRIED = False

# How many independent producer PROCESSES to split EACH shardable
# pattern into (see stuffingtest2.py's two_word_substitution_shards()).
# A single producer thread is GIL-bound pure-Python string formatting --
# fine when the GPU itself is the bottleneck, but once the kernel is
# fast enough (two RTX 4090s after the occupancy fix), one producer
# can't generate+pack candidates fast enough to keep both fed (this is
# what "both GPUs sitting mostly idle" in nvtop means). Pick a number
# that leaves headroom for the main process (draining results/
# printing) and the len(GPU_DEVICES) GPU worker processes -- e.g. on a
# machine with 16+ logical cores, 8-12 is a reasonable starting point;
# tune against htop/nvtop together (CPU cores pegged but GPUs still
# idle => raise this; CPU maxed out and GPUs now busy => you've found
# the balance).
NUM_PRODUCER_SHARDS = 8

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


def send_notification(subject, body):
    """Emails a short notification via Gmail SMTP -- same approach as
    ../../step2/crackconcat.py's send_notification(), and identical to
    gpu_crack_mac.py's copy of this function. Best-effort: the caller
    wraps this in try/except so a real crack result is never lost just
    because the email couldn't be sent (missing/stale google.key, no
    network, etc.). Called only from the main process (never from a
    spawned GPU worker or producer process), so there's no multiprocessing
    concern here despite the rest of this file being multi-process."""
    with open(GOOGLE_KEY_PATH) as keyfile:
        keyaddr, keypass = keyfile.read().splitlines()[:2]

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = keyaddr
    msg["To"] = keyaddr
    msg.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(keyaddr, keypass)
        smtp.send_message(msg)


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


def _gpu_worker(device_id, worker_id, shm_names, free_slots, ready_queue, result_queue, stop_event,
                 ciphertext_bytes, target_tag_bytes, batch_size, threads_per_block=256):
    """One process per GPU. Pulls (slot_idx, num_items) pairs off the
    SHARED ready_queue (fed by producer processes -- see
    run_batches_multi_gpu) and dispatches the batch sitting in that
    shared-memory slot to its own CUDA device. Natural load balancing:
    whichever GPU finishes its current batch first just pulls the next
    one off the queue, rather than each GPU being statically assigned a
    fixed half of the candidate space.

    IMPORTANT: only (slot_idx, num_items) -- two small integers -- cross
    the process boundary through ready_queue, never the actual 32MB of
    packed candidate bytes. The real candidate data lives in a shared-
    memory slot that this process attaches to directly and reads with a
    zero-copy numpy view; see module docstring for why this replaced an
    earlier design that shipped the packed bytes themselves through a
    multiprocessing.Queue (measured hard ceiling: ~370MB/s through that
    queue's one underlying pipe, REGARDLESS of producer/consumer count
    -- i.e. ~5-6M candidates/s combined no matter how fast the GPUs
    are, which is exactly the kind of ceiling that would make two 4090s
    barely edge out one Mac integrated GPU).

    pass_buf/out_buf are allocated ONCE (sized to `batch_size`) and
    reused for every batch rather than cudaMalloc'd/freed each time --
    cudaMalloc/cudaFree are synchronous driver calls, and doing two of
    them per batch (many times a second) adds real, avoidable overhead.
    Safe because _producer_process() always packs candidates to the
    same fixed MAX_PWD_LEN stride, so every batch's byte layout is the
    same size."""
    try:
        ctx = _build_cuda_context(device_id)
    except Exception as exc:
        result_queue.put({'worker_id': worker_id, 'kind': 'worker_error',
                           'info': {'error': f"failed to init CUDA device {device_id}: {exc}"}})
        return

    shms = []
    try:
        shms = [shared_memory.SharedMemory(name=n) for n in shm_names]
        kernel = _build_kernel()

        ct_arr = np.frombuffer(ciphertext_bytes, dtype=np.uint8)
        tag_arr = np.frombuffer(target_tag_bytes, dtype=np.uint8)
        ct_buf = cuda.mem_alloc(ct_arr.nbytes)
        cuda.memcpy_htod(ct_buf, ct_arr)
        tag_buf = cuda.mem_alloc(tag_arr.nbytes)
        cuda.memcpy_htod(tag_buf, tag_arr)

        # Preallocated ONCE and reused every batch -- see function
        # docstring note above.
        pass_buf = cuda.mem_alloc(batch_size * MAX_PWD_LEN)
        out_buf = cuda.mem_alloc(batch_size)

        total_checked = 0
        last_report = time.perf_counter()

        while not stop_event.is_set():
            try:
                item = ready_queue.get(timeout=1)
            except queue_mod.Empty:
                continue

            if item is None:
                # Sentinel: candidates are exhausted. Put it back so the
                # NEXT worker sharing this queue sees it too, then stop.
                ready_queue.put(None)
                break

            slot_idx, num_items = item
            # Zero-copy view directly onto the shared-memory slot -- no
            # pickling, no pipe, this IS the host buffer memcpy_htod
            # reads from.
            flat_passwords = np.frombuffer(shms[slot_idx].buf, dtype=np.uint8,
                                            count=num_items * MAX_PWD_LEN)

            cuda.memcpy_htod(pass_buf, flat_passwords)

            blocks = (num_items + threads_per_block - 1) // threads_per_block
            kernel(pass_buf, np.int32(MAX_PWD_LEN),
                   ct_buf, np.int32(len(ciphertext_bytes)),
                   tag_buf,
                   out_buf, np.int32(num_items),
                   block=(threads_per_block, 1, 1), grid=(blocks, 1))

            out_match = np.empty(num_items, dtype=np.uint8)
            cuda.memcpy_dtoh(out_match, out_buf)

            total_checked += num_items
            matches = out_match == 1
            if matches.any():
                match_idx = int(np.argmax(matches))
                # Recover the matched password straight from the slot
                # (still valid -- we haven't released it yet) instead of
                # from a parallel list of strings that would otherwise
                # have had to travel through ready_queue too.
                start = match_idx * MAX_PWD_LEN
                raw = bytes(shms[slot_idx].buf[start:start + MAX_PWD_LEN])
                found_password = raw.rstrip(b'\x00').decode('ascii')
                free_slots.put(slot_idx)
                stop_event.set()
                result_queue.put({'worker_id': worker_id, 'kind': 'found',
                                   'info': {'password': found_password, 'total_checked': total_checked}})
                return

            # No match in this batch -- release the slot for reuse now
            # that memcpy_htod has already copied it onto the device.
            free_slots.put(slot_idx)

            now = time.perf_counter()
            if now - last_report >= 1.0:
                result_queue.put({'worker_id': worker_id, 'kind': 'progress',
                                   'info': {'total_checked': total_checked}})
                last_report = now

        result_queue.put({'worker_id': worker_id, 'kind': 'worker_done',
                           'info': {'total_checked': total_checked}})
    finally:
        # flat_passwords (if the loop ever ran) is a numpy view straight
        # onto shared memory via the buffer protocol -- shm.close() (an
        # mmap.close() under the hood) refuses with "cannot close
        # exported pointers exist" while any such view is still alive.
        # Dropping the name here is enough to release it before we try
        # to close; always safe to assign even if the loop never ran.
        flat_passwords = None
        for shm in shms:
            shm.close()
        ctx.pop()


def _producer_process(shard_fn, batch_size, shm_names, free_slots, ready_queue,
                       stop_event, producers_remaining, producers_lock):
    """One process per candidate-space shard (see NUM_PRODUCER_SHARDS).
    Generates+packs batches from its own disjoint shard and writes each
    one DIRECTLY into a free shared-memory slot -- see module docstring
    -- then hands only (slot_idx, num_items) across to the GPU workers
    via ready_queue. This is still purely about parallelizing candidate
    GENERATION across CPU cores (previously one GIL-bound thread,
    before that a single-threaded multiprocessing.Queue producer),
    independent of which GPU ends up dispatching a given batch -- the
    shared-memory step on top of that is what keeps N producers from
    all serializing through one pipe's bandwidth ceiling.

    Sentinel handling: N independent producer processes finish at
    different times, but exactly ONE None must reach ready_queue once
    ALL of them are done -- pushing one per producer would interleave
    sentinels with real batches and make a GPU worker stop early while
    candidates from a slower producer remain unsent. producers_remaining
    (a shared Value, decremented under producers_lock) tracks how many
    producers are still running; only the one that brings it to 0 pushes
    the sentinel, and only if the search wasn't already stopped early by
    a match (stop_event)."""
    shms = [shared_memory.SharedMemory(name=n) for n in shm_names]
    try:
        for candidate_batch in chunked_iterable(shard_fn(), batch_size):
            if stop_event.is_set():
                break
            safe_batch = [c for c in candidate_batch if len(c) <= MAX_PWD_LEN]
            if not safe_batch:
                continue
            packed = b''.join(
                cand.encode("ascii").ljust(MAX_PWD_LEN, b'\x00')
                for cand in safe_batch
            )

            slot_idx = None
            while not stop_event.is_set():
                try:
                    slot_idx = free_slots.get(timeout=1)
                    break
                except queue_mod.Empty:
                    continue
            if slot_idx is None:
                break  # stop_event fired while waiting for a free slot

            shms[slot_idx].buf[:len(packed)] = packed

            put_ok = False
            while not stop_event.is_set():
                try:
                    ready_queue.put((slot_idx, len(safe_batch)), timeout=1)
                    put_ok = True
                    break
                except queue_mod.Full:
                    continue
            if not put_ok:
                free_slots.put(slot_idx)  # never handed off -- give it back
                break
    finally:
        for shm in shms:
            shm.close()

    with producers_lock:
        producers_remaining.value -= 1
        is_last_producer = producers_remaining.value == 0

    if is_last_producer and not stop_event.is_set():
        ready_queue.put(None)


def run_batches_multi_gpu(candidate_shards, ciphertext_bytes, target_tag_bytes,
                           gpu_devices=None, batch_size=500000,
                           total=None, print_interval=5, lookahead_per_gpu=3):
    """candidate_shards: a list of zero-arg callables, each returning an
    independent iterator over a DISJOINT slice of the password-guess
    space (see stuffingtest2.py's *_shards() helpers, and main() below
    for how a non-shardable pattern just becomes a list of one).
    Returns the matching password (str) and stops early, or None once
    every shard is exhausted with no match across every GPU.

    See module docstring for the overall design: ONE PRODUCER PROCESS
    PER SHARD (CPU-only, no CUDA) writes batches into a shared-memory
    ring buffer (see num_slots/shm_pool below) that N per-GPU worker
    processes consume from -- only a (slot index, item count) pair of
    small integers crosses any multiprocessing.Queue; the actual 32MB
    of candidate bytes never does. This replaced an earlier design that
    shipped the full packed batch through a multiprocessing.Queue,
    which measured out at a hard ~370MB/s ceiling through that queue's
    single underlying pipe NO MATTER how many producers or consumers
    were attached to it -- ~5-6M candidates/s combined at
    MAX_PWD_LEN=64 bytes/candidate, which is exactly the kind of wall
    that would make two RTX 4090s barely edge out one Mac integrated
    GPU. (Before that, it was a single GIL-bound producer THREAD, which
    couldn't even saturate that pipe by itself.)"""
    if gpu_devices is None:
        gpu_devices = GPU_DEVICES

    # Shared-memory ring buffer. num_slots gives producers and the
    # per-GPU lookahead some room to overlap (same role lookahead_per_gpu
    # played for the old work_queue's maxsize); bumped up to also cover
    # every shard having a slot of its own if there happen to be more
    # shards than that, so no producer sits totally starved for a slot
    # right out of the gate.
    num_slots = max(lookahead_per_gpu * len(gpu_devices), len(candidate_shards))
    slot_size = batch_size * MAX_PWD_LEN
    shm_pool = [shared_memory.SharedMemory(create=True, size=slot_size) for _ in range(num_slots)]
    shm_names = [shm.name for shm in shm_pool]

    try:
        free_slots = MP_CTX.Queue()
        for i in range(num_slots):
            free_slots.put(i)
        ready_queue = MP_CTX.Queue(maxsize=num_slots)
        result_queue = MP_CTX.Queue()
        stop_event = MP_CTX.Event()

        workers = []
        for worker_id, device_id in enumerate(gpu_devices):
            p = MP_CTX.Process(target=_gpu_worker, args=(
                device_id, worker_id, shm_names, free_slots, ready_queue, result_queue, stop_event,
                ciphertext_bytes, target_tag_bytes, batch_size))
            workers.append(p)
            p.start()

        producers_remaining = MP_CTX.Value('i', len(candidate_shards))
        producers_lock = MP_CTX.Lock()
        producers = []
        for shard_fn in candidate_shards:
            p = MP_CTX.Process(target=_producer_process, args=(
                shard_fn, batch_size, shm_names, free_slots, ready_queue, stop_event,
                producers_remaining, producers_lock))
            producers.append(p)
            p.start()

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
                # Printed the moment it happens -- a GPU that fails to init
                # (wrong device index, driver issue, already in use, etc.)
                # should be visible right away, not discovered only after a
                # multi-hour run finally ends on whatever GPU(s) survived.
                print(f"  WARNING: GPU worker {worker_id} (device {gpu_devices[worker_id]}) "
                      f"failed: {msg['info']['error']} -- continuing with the remaining "
                      f"GPU(s), if any. Check `nvidia-smi -L` and this GPU's availability.")

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
        # 'found' message, make sure no producer is left blocked forever on
        # a full queue.
        stop_event.set()

        for p in producers:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)

        for p in workers:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)

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
            if worker_errors:
                print(f"  (note: found despite {len(worker_errors)} GPU worker failure(s) "
                      f"during this run: {'; '.join(worker_errors)})")
            return found_password

        if worker_errors:
            raise RuntimeError("GPU worker(s) failed: " + "; ".join(worker_errors))

        print(f"FAILURE! {combined:,} checked in {elapsed:.1f}s ({rate:,.0f}/s)")
        return None
    finally:
        # Every producer/worker process is expected to have already
        # called .close() on its own handle to each of these (see their
        # finally blocks) by the time we get here on the normal path;
        # unlink() actually removes the OS-level shared memory object.
        # Safe to do even if some child is still winding down after an
        # abnormal exit above -- unlinking only removes the *name*, it
        # doesn't invalidate a mapping a process already attached.
        for shm in shm_pool:
            shm.close()
            shm.unlink()


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

    # Shard every pattern that opts in (has .make_shards -- currently
    # just two_word_substitution_variants) into NUM_PRODUCER_SHARDS
    # independent producer-process shards; a pattern without
    # .make_shards just becomes a single-shard list of itself, so it
    # still gets its own producer process (no change in behavior for
    # small/cheap patterns, just no parallel generation benefit).
    candidate_shards = []
    for fn in patterns:
        if hasattr(fn, "make_shards"):
            candidate_shards.extend(fn.make_shards(NUM_PRODUCER_SHARDS))
        else:
            candidate_shards.append(fn)
    print(f"Split into {len(candidate_shards)} producer-process shard(s) "
          f"(NUM_PRODUCER_SHARDS={NUM_PRODUCER_SHARDS}) across {len(patterns)} pattern pool(s)")

    ciphertext_bytes = st2.ciphertext[:-32]
    target_tag_bytes = st2.ciphertext[-32:]

    result = run_batches_multi_gpu(candidate_shards, ciphertext_bytes, target_tag_bytes,
                                    gpu_devices=GPU_DEVICES, total=total)

    if result is not None:
        plaintext = st2.decrypt_target(result)
        print(f"Secret is: {plaintext}")
        try:
            send_notification(
                "Step 3 cracker (CUDA): SUCCESS",
                f"Password found: {result!r}\n\nDecrypted content:\n{plaintext}"
            )
            print("Notification email sent.")
        except Exception as exc:
            print(f"  (warning: password was found, but sending the notification email "
                  f"failed: {exc!r} -- check hw2/step2/google.key)")


if __name__ == "__main__":
    if "--sanity-check" in sys.argv:
        test_known_password()
    else:
        main()
