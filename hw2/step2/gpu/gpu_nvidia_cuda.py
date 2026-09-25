"""
NVIDIA/CUDA port of gpu_mac_test1.py.

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
sha256crypt_kernel.c (verified against your Mac's real GPU), this has only
been verified algorithmically: the kernel logic was compiled as plain C++
(CUDA-specific syntax stubbed out) and checked against the same 95
known-answer test vectors used for the OpenCL version -- all 95 matched.
Real nvcc compilation and actual GPU execution have NOT been tested. First
things to check on your actual machine: that `pip install pycuda` succeeds
against your CUDA toolkit, and that this correctly finds/uses the NVIDIA
GPU rather than any other OpenCL/compute device on the system (pycuda.autoinit
below picks CUDA device 0 by default -- see its comment if you need to pick
a specific GPU on a multi-GPU box).
"""

import os
import sys
import time
import json
import math
import smtplib
import itertools
from datetime import timedelta
from email.message import EmailMessage

import numpy as np
from wordfreq import zipf_frequency

try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401 -- side effect: creates a CUDA context on
                             # the default device (device 0) as soon as this is
                             # imported. For a multi-GPU machine where you want a
                             # specific card, replace this with explicit
                             # `cuda.init(); dev = cuda.Device(N); ctx = dev.make_context()`
                             # instead, picking N for the NVIDIA GPU you want.
    from pycuda.compiler import SourceModule
except ImportError:
    sys.exit(
        "pycuda is required for this script (pip install pycuda), and needs a "
        "working NVIDIA CUDA toolkit/driver on this machine to build against."
    )

from passlib.utils.binary import h64
from passlib.handlers.sha2_crypt import _256_transpose_map


scriptname = os.path.basename(__file__)
jobdescription = "DOWNGRADE Concatenation triples (NVIDIA/CUDA)"

KERNEL_SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sha256crypt_kernel.cu")


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


def process_candidates_on_nvidia_gpu(candidates, salt_str, target_hash,
                                      batch_size=200000, max_candidate_len=64,
                                      total=None, print_interval=5,
                                      progress_callback=None, threads_per_block=256):
    """
    CUDA counterpart of process_itertools_on_mac_gpu() -- same interface,
    same batching/matching logic, same silent-unless-progress_callback
    behavior (see gpu_mac_test1.py for the full parameter docs; they're
    unchanged here). Only the GPU dispatch mechanics differ.

    threads_per_block: CUDA block size for the kernel launch. 256 is a
    reasonable default for a kernel like this one (no shared memory use,
    modest register pressure); tune if you profile this on your actual
    hardware and find something else works better.
    """
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

    total_checked = 0
    start_time = time.perf_counter()
    last_print = start_time

    for candidate_batch in chunked_iterable(candidates, batch_size):
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
            return found_password

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

        total = product_size(*(pools[label] for label in pattern))
        print(f'Attempting pattern {str(pattern)}: {total} candidates to compute of grand total {(commontotal + zipftotal)**3}')

        candidates = (''.join(combo) for combo in itertools.product(*(pools[label] for label in pattern)))
        password = process_candidates_on_nvidia_gpu(candidates, salt, target_hash, batch_size=250000,
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
