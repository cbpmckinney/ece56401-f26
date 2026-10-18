#!/usr/bin/env python
import hashlib

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

# pyaes.AESModeOfOperationCTR's default Counter starts at integer 1,
# represented as a 16-byte big-endian value, and increments by 1 per
# 16-byte block. cryptography's modes.CTR() takes that same 16-byte value
# directly as its initial counter block (verified byte-for-byte identical
# output against pyaes for a multi-block message before swapping this in)
# -- so every .crypt file already produced by the old pyaes-based code
# still decrypts correctly with no format change.
CTR_NONCE = (1).to_bytes(16, "big")

def derive_key(password):
    # password may arrive as str (a typed-in password, or plaintext read
    # from a text-mode file) or as bytes (e.g. compute_tag() is called on
    # the *output* of decryption). Only str needs encoding -- bytes is
    # already what hashlib wants.
    if isinstance(password, str):
        password = password.encode("ascii")
    return hashlib.sha256(password).digest()

def compute_tag(plaintext):

    return derive_key(plaintext)

def _aes_ctr_crypt(key, data):
    # CTR mode is symmetric -- the same keystream XOR operation both
    # encrypts and decrypts, so one helper covers both directions, same
    # as pyaes.AESModeOfOperationCTR's .encrypt()/.decrypt() did.
    if isinstance(data, str):
        data = data.encode("utf-8")
    cipher = Cipher(algorithms.AES(key), modes.CTR(CTR_NONCE))
    cryptor = cipher.encryptor()
    return cryptor.update(data) + cryptor.finalize()

def encrypt(password, plaintext):
    key = derive_key(password)
    ciphertext = _aes_ctr_crypt(key, plaintext)
    tag = compute_tag(plaintext)

    return ciphertext + tag

def decrypt(password, ciphertext):

    key = derive_key(password)
    tag = ciphertext[-32:]
    ciphertext = ciphertext[:-32]

    decrypted = _aes_ctr_crypt(key, ciphertext)
    try:
        # Both tag and newtag are raw 32-byte SHA-256 digests -- compare
        # them as bytes directly. (The old code called .decode("utf-8") on
        # a digest, which is essentially random bytes and almost never
        # valid UTF-8; that's what was raising the "wrong password or
        # malformed payload!" exception even for the RIGHT password.)
        newtag = compute_tag(decrypted)
    except Exception:
        raise Exception("wrong password or malformed payload!")
    if newtag != tag:
        raise Exception("tag doesn't match! {}, {}".format(tag, newtag))


    # decrypted is bytes (that's what aes.decrypt() returns); decode back
    # to str so it matches the type of the plaintext that was originally
    # encrypted (encrypt() accepts str and pyaes encodes it internally).
    return decrypted.decode("utf-8")

if __name__ == "__main__":
    import sys

    password = sys.argv[1]
    with open(sys.argv[2]) as fp:
        plaintext = fp.read()
    target = sys.argv[3] 

    ciphertext = encrypt(password, plaintext)
    print(repr(ciphertext))

    decrypted = decrypt(password, ciphertext)
    print(repr(decrypted))
    if decrypted != plaintext:
        print("didn't match: {} != {}".format(decrypted, plaintext))

    with open(target, 'wb') as fp:
        fp.write(ciphertext)
