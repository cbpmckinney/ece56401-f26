#!/usr/bin/env python
import pyaes
import hashlib

def derive_key(password):
    return hashlib.sha256(password.encode("ascii")).digest()

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
        newtag = compute_tag(decrypted).decode("utf-8")
    except:
        raise Exception("wrong password or malformed payload!")
    if newtag != tag:
        raise Exception("tag doesn't match! {}, {}".format(tag, newtag))


    return decrypted

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
