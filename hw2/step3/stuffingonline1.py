import requests
import json
import string
import itertools
import time 
from datetime import timedelta


url = "http://127.0.0.1:5000/decrypt"


OLDPASSWORD = "aw2246french44pledgeor"
WORD1 = "aw"
WORD2 = "french"
WORD3 = "pledgor"
WORDS = (WORD1, WORD2, WORD3)
NUMBERS = ("2246", "44")

alphabet = string.ascii_letters
digits = string.digits
punct = string.punctuation

symbols = alphabet + digits + punct


def suffixes(base: str, chars, maxlen: int):
    for length in range(1, maxlen + 1):
        for combo in itertools.product(chars, repeat=length):
            yield base + ''.join(combo)

def prefixes(base: str, chars, maxlen: int):
    for length in range(1, maxlen + 1):
        for combo in itertools.product(chars, repeat=length):
            yield ''.join(combo) + base


start = time.perf_counter()
last_print = start
print_interval = 5  # seconds between progress prints

total = (len(symbols) ** 3)*2
count = 0
len = 3



for guess in prefixes(OLDPASSWORD, symbols, len):
    resp = requests.post(url, data={"password": guess})
    print(f"Trying password: {guess} - Status code: {resp.status_code}")
    count += 1
    now = time.perf_counter()
    if now - last_print >= print_interval:
        elapsed = now - start
        rate = count / elapsed                      # candidates per second
        remaining = total - count
        eta_seconds = remaining / rate
        print(f"{count}/{total} ({100*count/total:.2f}%) "f"- {rate:.1f}/s - ETA {timedelta(seconds=int(eta_seconds))}")
        last_print = now
    if resp.status_code == 200:
        print(f"Found password: {guess}")
        break

