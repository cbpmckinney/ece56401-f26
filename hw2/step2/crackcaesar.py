import itertools
import string
import legacycrypt as crypt
from multiprocessing import Pool
import math
import time 
from datetime import timedelta
from lengthbuckets import iter_length_buckets, count_length_buckets
import smtplib
from email.message import EmailMessage
import json
import os

targethash = '$y$j9T$F/vLDJRdzzspQonYxyqKl1$Q/nOKF5ECoPwQIJAZSlNcRt21Y3b1eV42Usj5SkfBX9'
#targethash = '$y$j9T$F/vLDJRdzzspQonYxyqKl1$1iAvzyCgCA7PxGXOIHnaKKuL1HcaFs.IGrFpDdsLWbD'

scriptname = os.path.basename(__file__)
jobdescription = "Caesar cipher for uncommon words"

def caesar(password: str, shamt: int) -> str:
    ans = ''
    for i in range(len(password)):
        curchar = password[i]
        if curchar.isalpha():
            if curchar.islower:
                ans += chr((ord(curchar) - 0x61 + shamt) % 26 + 0x61)
            else:
                ans += chr((ord(curchar) - 0x41 + shamt) % 26 + 0x41)

        else:
            ans += curchar

    return ans





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


def main():

    ip = open('dictionaries/uncommon.txt', 'r')
    uncommon = ip.read().splitlines()
    ip.close()

    all_candidates = (caesar(p, i) for p in uncommon for i in range(1,26))

    count = 0
    total = 25*len(uncommon)


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

