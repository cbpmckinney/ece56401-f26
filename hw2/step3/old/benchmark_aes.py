#!/usr/bin/env python
"""
benchmark_aes.py - measures per-guess cost for three different "is this
password right?" checks, to compare against yescrypt:

  1. pyaes            - pure-Python AES-CTR (what encrypt.py uses)
  2. cryptography     - hardware-accelerated AES-CTR (OpenSSL/AES-NI)
  3. yescrypt          - via legacycrypt, same call your crack scripts use

For the AES cases, "checking a guess" means the full pipeline a real
attacker would run: derive a key from the password, decrypt, recompute
a tag, and compare -- not just a bare AES call. That's the fair
comparison against yescrypt, since yescrypt.crypt() also bundles key
derivation + comparison into one call.

This does NOT touch encrypt.py or wsgi.py; it stands alone.

Usage: python benchmark_aes.py [iterations_fast] [iterations_yescrypt]
Defaults: 20000 fast-path iterations, 200 yescrypt iterations
  (yescrypt is deliberately slow, so it gets far fewer reps)
"""
import sys
import time
import hashlib
import secrets

import pyaes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import legacycrypt as crypt

# Same target hash your crack scripts (crackcaesar.py etc.) use -- reused
# here purely as a realistic yescrypt salt/cost-parameter source, not to
# actually crack anything.
YESCRYPT_TARGET = '$y$j9T$F/vLDJRdzzspQonYxyqKl1$Q/nOKF5ECoPwQIJAZSlNcRt21Y3b1eV42Usj5SkfBX9'

# Build a realistic-size payload once (setup cost, not timed) -- similar in
# size to secret.html.crypt (1705 bytes total).
PLAINTEXT = secrets.token_bytes(1673)
SETUP_PASSWORD = "correct horse battery staple"
SETUP_KEY = hashlib.sha256(SETUP_PASSWORD.encode("ascii")).digest()
NONCE = b'\x00' * 16

_cipher = Cipher(algorithms.AES(SETUP_KEY), modes.CTR(NONCE))
_enc = _cipher.encryptor()
CIPHERTEXT = _enc.update(PLAINTEXT) + _enc.finalize()
TAG = hashlib.sha256(PLAINTEXT).digest()


def guess_pyaes(password):
    """Full guess pipeline using pyaes (pure Python)."""
    key = hashlib.sha256(password.encode("ascii")).digest()
    aes = pyaes.AESModeOfOperationCTR(key)
    decrypted = aes.decrypt(CIPHERTEXT)
    newtag = hashlib.sha256(decrypted).digest()
    return newtag == TAG


def guess_cryptography(password):
    """Full guess pipeline using cryptography (hardware AES)."""
    key = hashlib.sha256(password.encode("ascii")).digest()
    cipher = Cipher(algorithms.AES(key), modes.CTR(NONCE))
    decryptor = cipher.decryptor()
    decrypted = decryptor.update(CIPHERTEXT) + decryptor.finalize()
    newtag = hashlib.sha256(decrypted).digest()
    return newtag == TAG


def guess_yescrypt(password):
    """Same call your crack scripts make: crypt.crypt(candidate, targethash)."""
    return crypt.crypt(password, YESCRYPT_TARGET) == YESCRYPT_TARGET


def time_guesses(fn, n, label):
    # Use a different candidate each call so nothing can special-case a
    # repeated input; correctness of the guess doesn't matter for timing.
    candidates = [f"guess-{i}" for i in range(n)]
    start = time.perf_counter()
    for c in candidates:
        fn(c)
    elapsed = time.perf_counter() - start
    per_guess = elapsed / n
    print(f"{label:<14} {n:>7} guesses in {elapsed:8.4f}s "
          f"-> {per_guess*1e6:10.2f} us/guess "
          f"-> {1/per_guess:12,.0f} guesses/sec")
    return per_guess


def main():
    n_fast = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    n_slow = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    print(f"payload size: {len(PLAINTEXT)} bytes plaintext, "
          f"{len(CIPHERTEXT) + len(TAG)} bytes ciphertext+tag\n")

    t_pyaes = time_guesses(guess_pyaes, n_fast, "pyaes")
    t_crypto = time_guesses(guess_cryptography, n_fast, "cryptography")
    t_yescrypt = time_guesses(guess_yescrypt, n_slow, "yescrypt")

    print()
    print(f"yescrypt is {t_yescrypt / t_pyaes:,.0f}x slower per guess than pyaes")
    print(f"yescrypt is {t_yescrypt / t_crypto:,.0f}x slower per guess than cryptography (hardware AES)")
    print(f"cryptography is {t_pyaes / t_crypto:,.1f}x faster per guess than pyaes")


if __name__ == "__main__":
    main()
