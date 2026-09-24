from encrypt import decrypt

import itertools
import string
import legacycrypt as crypt
from multiprocessing import Pool
import math
import time 
from datetime import timedelta
import smtplib
from email.message import EmailMessage
import json
import os

scriptname = os.path.basename(__file__)
jobdescription = "Stuffing Test"


with open("secret.html.crypt", "rb") as fp:
    ciphertext = fp.read()

def send_notification(subject, body):
    keyfile = open('../step2/google.key')
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

    try:
        result = decrypt(candidate, ciphertext=ciphertext)
    except:
        return None
    return result

def log_job(record):
    with open('stuffinglog.json', 'a') as f:
        f.write(json.dumps(record) + '\n')

def main():

    ip = open('../step2/dictionaries/all.txt', 'r')
    all = ip.read().splitlines()
    ip.close()

    all_candidates = (''.join(t) for t in all)
    total = len(all)
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
                send_notification('Stuffing Success!', body)
                log_job(record)
                exit(0)

    print('FAILURE!')
    record = {"script": scriptname, "description": jobdescription, "count": total, "result": f'Failure'}
    log_job(record)
    body = json.dumps(record, indent=2)
    send_notification('Stuffing Failure!', body)



if __name__ == "__main__":
    main()

