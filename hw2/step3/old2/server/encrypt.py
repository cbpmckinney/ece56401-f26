#!/usr/bin/env python
import pyaes
import hashlib

def derive_key(password):
    # password may arrive as str (a typed-in password, or plaintext read
    # from a text-mode file) or as bytes (e.g. compute_tag() is called on
    # the *output* of aes.decrypt(), which pyaes always returns as bytes).
    # Only str needs encoding -- bytes is already what hashlib wants.
    if isinstance(password, str):
        password = password.encode("ascii")
    return hashlib.sha256(password).digest()

def compute_tag(plaintext):

    return derive_key(plaintext)

def encrypt(password, plaintext):
    key = derive_key(password)
    aes = pyaes.AESModeOfOperationCTR(key)
    ciphertext = aes.encrypt(plaintext)
    tag = compute_tag(plaintext)

    return ciphertext + tag

def decrypt(password, ciphertext):

    key = derive_key(password)
    tag = ciphertext[-32:]
    ciphertext = ciphertext[:-32]


    # The counter mode of operation maintains state, so decryption requires
    # a new instance be created
    aes = pyaes.AESModeOfOperationCTR(key)
    decrypted = aes.decrypt(ciphertext)
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
