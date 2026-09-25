import itertools
import string
import legacycrypt as crypt
from multiprocessing import Pool
import math
import time 
from datetime import timedelta
from hw2.step2.old2.lengthbuckets import iter_length_buckets, count_length_buckets
import smtplib
from email.message import EmailMessage
import json
import os

targethash = '$y$j9T$F/vLDJRdzzspQonYxyqKl1$Q/nOKF5ECoPwQIJAZSlNcRt21Y3b1eV42Usj5SkfBX9'

scriptname = os.path.basename(__file__)
jobdescription = "Concatenation of 1-5 with common, both sides"





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

def check_candidate(candidate: str):

    result = crypt.crypt(candidate, targethash)
    if result == targethash:
        return candidate
    else:
        return None

# password hash method is yescrypt

def product_size(*iterables):
    return math.prod(len(it) for it in iterables)


def log_job(record):
    with open('attacklog.json', 'a') as f:
        f.write(json.dumps(record) + '\n')

def rotations(word):
    n = len(word)
    return (word[i:] + word[:i] for i in range(1, n))

def main():

    candidates1 = iter_length_buckets(min_len=1, max_len=5)
    candidates2 = iter_length_buckets(min_len=1, max_len=5)

    with open('dictionaries/common.txt', 'r') as f:
        common = f.read().splitlines()

    cross1a = (''.join(t) for t in itertools.product(common, candidates1))
    cross1b = (''.join(t) for t in itertools.product(candidates2, common))

    
    all_candidates = itertools.chain(cross1b, cross1a)
    total = 2 * len(common) * count_length_buckets(min_len=1, max_len=5)

    count = 0


    start = time.perf_counter()
    last_print = start
    print_interval = 5  # seconds between progress prints

    with Pool(processes=12) as pool:
        for result in pool.imap_unordered(check_candidate, all_candidates, chunksize=128):
            #print(result)
            count += 1
            now = time.perf_counter()
            if now - last_print >= print_interval:
                elapsed = now - start
                rate = count / elapsed                      # candidates per second
                remaining = total - count
                eta_seconds = remaining / rate
                print(f"{count}/{total} ({100*count/total:.2f}%) "f"- {rate:.1f}/s - ETA {timedelta(seconds=int(eta_seconds))}")
                last_print = now

            if result is not None:
                print(f"SUCCESS!  Password is: {result}")
                #fp.write(f"SUCCESS!  Password is: {result}")
                #fp.close()
                pool.terminate()
                
                record = {"script": scriptname, "description": jobdescription, "count": total, "result": f'Success: password is {result}'}
                body = json.dumps(record, indent=2)
                send_notification('Hashing Success!', body)
                log_job(record)
                exit(0)

    print('FAILURE!')
    record = {"script": scriptname, "description": jobdescription, "count": total, "result": f'Failure'}
    log_job(record)
    body = json.dumps(record, indent=2)
    send_notification('Hashing Failure!', body)



if __name__ == "__main__":
    main()

