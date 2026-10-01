"""
stuffingAES.py - same structure/flow as stuffing.py, but does its own
decrypt using the `cryptography` library (hardware-accelerated AES-CTR)
instead of importing the broken decrypt() from encrypt.py.

Why a separate decrypt implementation instead of `from encrypt import decrypt`:
encrypt.py's decrypt() unconditionally throws (see step3test.py /
step3testdecode.py) because compute_tag() calls .encode() on bytes.
This reimplements the *intended* logic -- same key derivation
(sha256(password)), same AES-CTR construction, same tag-compare idea --
just with the str/bytes bug fixed and using cryptography's AES instead
of pyaes for speed. It is a standalone script; encrypt.py is untouched.

Important compatibility note: cryptography's CTR mode nonce is the full
16-byte initial counter block. pyaes's default Counter starts at integer
value 1, represented as 15 zero bytes followed by 0x01. Using that same
16-byte nonce here was verified byte-for-byte against pyaes's own output
before using this against secret.html.crypt, so decrypting ciphertext
that encrypt.py/pyaes produced works correctly.
"""
import hashlib
import time
import os
import json
from datetime import timedelta
from multiprocessing import Pool
import string
import itertools

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

scriptname = os.path.basename(__file__)
jobdescription = "Stuffing Test (AES via cryptography)"

# Matches pyaes.Counter's default initial_value=1, as a 16-byte big-endian
# counter block -- verified to produce identical ciphertext to pyaes.
PYAES_DEFAULT_COUNTER_NONCE = b'\x00' * 15 + b'\x01'

with open("secret.html.crypt", "rb") as fp:
    ciphertext = fp.read()


def decrypt(password: str, ciphertext: bytes) -> bytes:
    """Same intent as encrypt.py's decrypt(), fixed and using hardware AES."""
    key = hashlib.sha256(password.encode("ascii")).digest()
    tag = ciphertext[-32:]
    body = ciphertext[:-32]

    cipher = Cipher(algorithms.AES(key), modes.CTR(PYAES_DEFAULT_COUNTER_NONCE))
    decryptor = cipher.decryptor()
    decrypted = decryptor.update(body) + decryptor.finalize()

    newtag = hashlib.sha256(decrypted).digest()
    if newtag != tag:
        raise Exception("tag doesn't match!")
    return decrypted


def send_notification(subject, body):
    keyfile = open('../step2/google.key')
    keyfiledata = keyfile.read().splitlines()
    keyaddr = keyfiledata[0]
    keypass = keyfiledata[1]

    import smtplib
    from email.message import EmailMessage
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = keyaddr
    msg["To"] = keyaddr
    msg.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(keyaddr, keypass)
        smtp.send_message(msg)


def check_candidate(candidate: str):
    try:
        result = decrypt(candidate, ciphertext)
    except Exception:
        return None
    return result


def log_job(record):
    with open('stuffingAESlog.json', 'a') as f:
        f.write(json.dumps(record) + '\n')


def main():
    ip = open('../step2/dictionaries/all.txt', 'r')
    all_words = ip.read().splitlines()
    ip.close()

    alphabet = string.ascii_letters + string.digits + string.punctuation

    len1 = (''.join(t) for t in itertools.product(alphabet, repeat=1))
    len2 = (''.join(t) for t in itertools.product(alphabet, repeat=2))
    len3 = (''.join(t) for t in itertools.product(alphabet, repeat=3))
    len4 = (''.join(t) for t in itertools.product(alphabet, repeat=4))
    len5 = (''.join(t) for t in itertools.product(alphabet, repeat=5))
    len6 = (''.join(t) for t in itertools.product(alphabet, repeat=6))
        
    all_candidates = itertools.chain(len1, len2, len3, len4, len5)
        
        
        

    total = sum(len(alphabet)**i for i in range(1, 6))
    count = 0

    start = time.perf_counter()
    last_print = start
    print_interval = 5  # seconds between progress prints

    with Pool(processes=12) as pool:
        for result in pool.imap_unordered(check_candidate, all_candidates, chunksize=128):
            count += 1
            now = time.perf_counter()
            if now - last_print >= print_interval:
                elapsed = now - start
                rate = count / elapsed
                remaining = total - count
                eta_seconds = remaining / rate if rate > 0 else 0
                print(f"{count}/{total} ({100*count/total:.2f}%) "
                      f"- {rate:.1f}/s - ETA {timedelta(seconds=int(eta_seconds))}")
                last_print = now

            if result is not None:
                print(f"SUCCESS!  Password is: {result}")
                pool.terminate()

                record = {"script": scriptname, "description": jobdescription,
                          "count": total, "result": f'Success: password is {result}'}
                body = json.dumps(record, indent=2)
                #send_notification('Stuffing Success!', body)
                log_job(record)
                exit(0)

    print('FAILURE!')
    record = {"script": scriptname, "description": jobdescription,
              "count": total, "result": 'Failure'}
    log_job(record)
    body = json.dumps(record, indent=2)
    #send_notification('Stuffing Failure!', body)


if __name__ == "__main__":
    main()
