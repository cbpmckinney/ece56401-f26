"""
GPU-accelerated Step 3 ("stuffing") cracker for Bob's secret, using
PyOpenCL -- same tool used for the Step 2 sha256crypt attack on this Mac
(see ../../step2/gpu/gpu_mac_test3.py), just pointed at a completely
different, much cheaper-per-guess scheme: SHA256(password) -> AES-256-CTR
decrypt -> SHA256(plaintext) tag check (see encrypt.py / stuffingtest2.py
in the parent directory).

This file does NOT reimplement candidate generation -- it imports
MUTATION_PATTERNS (and, optionally, ALREADY_TRIED) directly from
stuffingtest2.py in the parent directory, so there is exactly one place
the mutation axes are defined. stuffingtest2.py itself is untouched; this
is purely a second, GPU-powered way to consume the same generators,
dispatching candidates to the kernel in large batches instead of a CPU
multiprocessing.Pool.

KERNEL CORRECTNESS NOTE: no GPU (or OpenCL runtime at all) was available
in the environment this was written in, so sha256_aes_ctr_kernel.c could
not be compiled/run as an actual OpenCL kernel before being handed to
you. What WAS done: the exact algorithm (AES-256 S-box, key schedule,
MixColumns/ShiftRows, CTR construction) was mirrored in pure Python and
checked against this repo's own encrypt.py -- i.e. against the real
`cryptography` library -- across several test vectors (empty password,
1-byte plaintext/password, 500-byte plaintext, and a plaintext the exact
length of the real secret.html.crypt, 1673 bytes) and all matched
exactly. The kernel .c file itself was then ALSO compiled as plain C
(OpenCL address-space qualifiers stripped, get_global_id swapped for a
passed-in index) and run against the same test vectors directly -- also
all passed, including a case that caught a real indexing bug in the test
harness itself (not the kernel) along the way. That's strong evidence the
algorithm and the C logic are both correct, but it is NOT the same as a
real OpenCL compile -- please run test_known_password() below first and
confirm it reports a match before trusting a "no match" result from a
long unattended run.
"""

import os
import sys
import time
import itertools
import threading
import queue as queue_mod   # aliased: `queue` elsewhere in this file means
                              # the OpenCL command queue, not this module
from datetime import timedelta

import numpy as np
import pyopencl as cl

# --- locate and import stuffingtest2.py from the parent directory,
# without duplicating any of its candidate-generation logic. chdir there
# first since stuffingtest2.py opens its ciphertext file with a path
# relative to the CURRENT DIRECTORY, not relative to itself. ---
STEP3_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, STEP3_DIR)
os.chdir(STEP3_DIR)
import stuffingtest2 as st2

KERNEL_SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sha256_aes_ctr_kernel.c")
with open(KERNEL_SOURCE_PATH) as f:
    KERNEL_SOURCE = f.read()

MAX_PWD_LEN = 64     # must match MAX_PWD_LEN in the kernel
MAX_CT_LEN = 2048     # must match MAX_CT_LEN in the kernel

# Which pattern lists to run. ALREADY_TRIED was already exhaustively
# tested on CPU with zero hits -- it's cheap to re-run on GPU as a sanity
# double-check, but it's not included by default so a normal run doesn't
# waste time re-proving something already known.
INCLUDE_ALREADY_TRIED = False


def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def build_gpu_context():
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    device = ctx.devices[0]
    device_type = cl.device_type.to_string(device.type)
    print(f"OpenCL device selected: {device.name!r} "
          f"(type={device_type}, compute_units={device.max_compute_units}) -- "
          f"if type isn't GPU, create_some_context() picked the wrong platform.")
    program = cl.Program(ctx, KERNEL_SOURCE).build()
    kernel = cl.Kernel(program, "crack_sha256_aes_ctr")
    return ctx, queue, kernel


def run_batches(ctx, queue, kernel, candidates, ciphertext_bytes, target_tag_bytes,
                 batch_size=500000, total=None, print_interval=5, lookahead=3):
    """candidates: an iterable of password-guess strings. Returns the
    matching password (str) and stops early, or None if candidates is
    exhausted with no match.

    Candidate generation/packing (pure Python, single-threaded, CPU-bound)
    and GPU dispatch used to run strictly serially: build batch N, THEN
    dispatch it, THEN build batch N+1, etc. -- mactop showing ~50% GPU
    utilization at ~3.3M/s is exactly what that predicts: the GPU sits
    idle while Python is building the next batch, and Python sits idle
    while blocked on cl.enqueue_copy() waiting for the GPU to finish.
    Those two halves don't have to be serialized -- they're independent
    work, one CPU-bound and one hardware-bound. This runs candidate
    generation/packing on a background thread that keeps a small queue
    of prebuilt, GPU-ready batches topped up, while this function's own
    loop (the "consumer") just dispatches whatever's next in the queue.
    Overlap actually works here (not just cooperative switching) because
    pyopencl's blocking calls release the GIL while genuinely waiting on
    hardware, so the producer thread keeps running real Python bytecode
    during that wait instead of being frozen by it. `lookahead` bounds
    how many prebuilt batches can queue up (backpressure / memory cap,
    not a correctness knob -- verified against a pure-Python stand-in of
    this exact producer/consumer/sentinel/early-stop logic before being
    wired up to real PyOpenCL calls, since this environment has no GPU
    to test the real thing against)."""
    assert len(ciphertext_bytes) <= MAX_CT_LEN, (
        f"ciphertext is {len(ciphertext_bytes)} bytes, over the kernel's "
        f"MAX_CT_LEN={MAX_CT_LEN} -- bump MAX_CT_LEN in both this file and "
        f"the kernel .c file (they must match) before running."
    )

    ct_np = np.frombuffer(ciphertext_bytes, dtype=np.uint8)
    tag_np = np.frombuffer(target_tag_bytes, dtype=np.uint8)

    mf = cl.mem_flags
    ct_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ct_np)
    tag_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tag_np)

    # Still one reused pair, not reallocated per batch (see the previous
    # round of fixes) -- the dispatch loop below is still synchronous
    # per-call (it blocks on the copy-back before moving to the next
    # item), so there's no risk of the producer thread's NEXT batch
    # overwriting these while the GPU is still using them.
    pass_buf = cl.Buffer(ctx, mf.READ_WRITE, batch_size * MAX_PWD_LEN)
    out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, batch_size)

    total_checked = 0
    skipped_total = 0
    gen_time_total = 0.0
    build_time_total = 0.0
    gpu_time_total = 0.0
    # Time the DISPATCH side spends blocked on work_queue.get() waiting
    # for the next prebuilt batch -- i.e. GPU-idle time, the thing this
    # whole change is meant to shrink. High/growing = still host-bound
    # (the producer can't keep up); near-zero = genuinely GPU-bound now.
    wait_time_total = 0.0
    start_time = time.perf_counter()
    last_print = start_time

    work_queue = queue_mod.Queue(maxsize=lookahead)
    stop_event = threading.Event()

    def producer():
        nonlocal gen_time_total, build_time_total, skipped_total
        candidate_iter = chunked_iterable(candidates, batch_size)
        while not stop_event.is_set():
            t0 = time.perf_counter()
            try:
                candidate_batch = next(candidate_iter)
            except StopIteration:
                work_queue.put(None)
                return
            safe_batch = [c for c in candidate_batch if len(c) <= MAX_PWD_LEN]
            t1 = time.perf_counter()
            skipped = len(candidate_batch) - len(safe_batch)
            if skipped:
                skipped_total += skipped
            if not safe_batch:
                gen_time_total += (t1 - t0)
                continue
            # Fixed MAX_PWD_LEN stride, same reasoning as before: keeps
            # every batch's packed size identical so pass_buf can be
            # reused instead of reallocated.
            packed = b''.join(
                cand.encode("ascii").ljust(MAX_PWD_LEN, b'\x00')
                for cand in safe_batch
            )
            flat_passwords = np.frombuffer(packed, dtype=np.uint8)
            t2 = time.perf_counter()
            gen_time_total += (t1 - t0)
            build_time_total += (t2 - t1)
            work_queue.put((safe_batch, flat_passwords))
        work_queue.put(None)

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()

    while True:
        t_wait0 = time.perf_counter()
        item = work_queue.get()
        t_wait1 = time.perf_counter()
        wait_time_total += (t_wait1 - t_wait0)
        if item is None:
            break
        safe_batch, flat_passwords = item
        num_items = len(safe_batch)

        t_gpu0 = time.perf_counter()
        cl.enqueue_copy(queue, pass_buf, flat_passwords)

        kernel(
            queue, (num_items,), None,
            pass_buf, np.int32(MAX_PWD_LEN),
            ct_buf, np.int32(len(ciphertext_bytes)),
            tag_buf,
            out_buf, np.int32(num_items)
        )

        out_match = np.zeros(num_items, dtype=np.uint8)
        cl.enqueue_copy(queue, out_match, out_buf)
        t_gpu1 = time.perf_counter()
        gpu_time_total += (t_gpu1 - t_gpu0)

        total_checked += num_items
        matches = out_match == 1
        if matches.any():
            match_idx = int(np.argmax(matches))
            found_password = safe_batch[match_idx]
            elapsed = time.perf_counter() - start_time
            rate = total_checked / elapsed if elapsed > 0 else 0
            print(f"SUCCESS!  Password is: {found_password!r}  "
                  f"({total_checked} checked, {elapsed:.1f}s, {rate:.0f}/s)")
            try:
                plaintext = st2.decrypt(found_password, ciphertext_bytes + target_tag_bytes)
                print(f"Decrypted content: {plaintext}")
            except Exception as exc:
                print(f"  (warning: GPU reported a match, but re-decrypting on CPU raised "
                      f"{exc!r} -- this shouldn't happen if the kernel is correct; investigate "
                      f"before trusting this result)")
            stop_event.set()
            return found_password

        now = time.perf_counter()
        if now - last_print >= print_interval:
            elapsed = now - start_time
            rate = total_checked / elapsed if elapsed > 0 else 0
            timing = (f"gen={gen_time_total:.1f}s host_build={build_time_total:.1f}s "
                      f"gpu_dispatch={gpu_time_total:.1f}s gpu_idle_wait={wait_time_total:.1f}s")
            if total:
                pct = 100 * total_checked / total
                remaining = total - total_checked
                eta = timedelta(seconds=int(remaining / rate)) if rate > 0 else "?"
                print(f"{total_checked}/{total} ({pct:.2f}%) - {rate:.0f}/s - "
                      f"ETA {eta} - skipped so far: {skipped_total} - {timing}")
            else:
                print(f"{total_checked} checked - {rate:.0f}/s - skipped so far: {skipped_total} - {timing}")
            last_print = now

    stop_event.set()
    elapsed = time.perf_counter() - start_time
    rate = total_checked / elapsed if elapsed > 0 else 0
    print(f"FAILURE! {total_checked} checked in {elapsed:.1f}s ({rate:.0f}/s), "
          f"{skipped_total} skipped (over MAX_PWD_LEN={MAX_PWD_LEN}) -- "
          f"gen={gen_time_total:.1f}s host_build={build_time_total:.1f}s "
          f"gpu_dispatch={gpu_time_total:.1f}s gpu_idle_wait={wait_time_total:.1f}s")
    return None


def test_known_password():
    """Run before trusting a real crack attempt: confirms the kernel
    actually finds a password it's GIVEN directly as a one-item batch,
    end to end through real PyOpenCL/GPU dispatch (not just the plain-C
    harness this was checked against before handing it to you).

    Deliberately does NOT touch the real target (secret.html.crypt) --
    it decrypts two throwaway files instead, each with a password we
    already know for certain:
      server/templates/test.crypt      password "password"
      server/templates/test2.crypt     password "password1234"
    A kernel bug could coincidentally "match" against a wrong tag by
    luck once; requiring both small known-answer cases to pass is a much
    stronger check than running against the real target would be (where
    we don't yet know the right answer, so a false negative and a true
    negative look identical)."""
    ctx, queue, kernel = build_gpu_context()

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
        print(f"Running known-password sanity check against {os.path.basename(path)} "
              f"(expected password {known_password!r})...")
        result = run_batches(ctx, queue, kernel, [known_password], ciphertext_bytes, target_tag_bytes,
                              batch_size=1, total=1, print_interval=9999)
        if result == known_password:
            print(f"  PASSED -- kernel correctly matched {os.path.basename(path)}.")
        else:
            print(f"  FAILED -- do not trust a real run until this passes "
                  f"({os.path.basename(path)}, expected {known_password!r}, got {result!r}).")
            all_passed = False

    if all_passed:
        print("Sanity check PASSED on both known files -- safe to trust a real run against "
              "secret.html.crypt now.")
    else:
        print("Sanity check FAILED -- fix the kernel/host script before running against "
              "secret.html.crypt for real.")
    return all_passed


def main():
    ctx, queue, kernel = build_gpu_context()
    ciphertext_bytes = st2.ciphertext[:-32]
    target_tag_bytes = st2.ciphertext[-32:]

    patterns = list(st2.MUTATION_PATTERNS)
    if INCLUDE_ALREADY_TRIED:
        patterns = list(st2.ALREADY_TRIED) + patterns

    print(f"Running {len(patterns)} pattern pool(s) on GPU "
          f"({'including' if INCLUDE_ALREADY_TRIED else 'excluding'} ALREADY_TRIED)...")
    total = sum(st2.pattern_count(fn) for fn in patterns)
    print(f"Total candidates: {total}")

    candidates = itertools.chain.from_iterable(fn() for fn in patterns)
    # run_batches() itself prints both the password and the decrypted
    # content on a match (see its SUCCESS branch) -- nothing more to do
    # here with the result.
    run_batches(ctx, queue, kernel, candidates, ciphertext_bytes, target_tag_bytes,
                batch_size=250000, total=total, print_interval=5)


if __name__ == "__main__":
    if "--sanity-check" in sys.argv:
        test_known_password()
    else:
        main()
