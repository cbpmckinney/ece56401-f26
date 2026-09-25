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
from wordfreq import zipf_frequency




with open("target.txt", 'r') as fp:
    targethash = fp.read()

scriptname = os.path.basename(__file__)
jobdescription = "DOWNGRADE Concatenation common * zipf4"


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
    

def product_size(*iterables):
    return math.prod(len(it) for it in iterables)


def log_job(record):
    with open('attacklogdowngrade.json', 'a') as f:
        f.write(json.dumps(record) + '\n')




def main():

    with open('dictionaries/common.txt', 'r') as f:
        common = f.read().splitlines()

    with open('dictionaries/uncommon_by_zipf.txt', 'r') as f:
        uncommon = f.read().splitlines()



    zipf_min7 = [w for w in uncommon if zipf_frequency(w.lower(), 'en') >= 7]
    zipf_min6 = [w for w in uncommon if zipf_frequency(w.lower(), 'en') >= 5]

    zipf_min = []
    zipf = []
    for i in range(7,0,-1):
        zipf_min += [[w for w in uncommon if zipf_frequency(w.lower(), 'en') >= i]]
        zipf += [[w for w in uncommon if (zipf_frequency(w.lower(), 'en') >= i) and (zipf_frequency(w.lower(), 'en') < (i+1))]]

    #print(zipf_min[1])
    
    #cross1a = (''.join(t) for t in itertools.product(zipf_min[1], repeat =2))
    cross1b = (''.join(t) for t in itertools.product(common, repeat =3))
    #cross1c = (''.join(t) for t in itertools.product(zipf_min[1], repeat =4))
    #cross1d = (''.join(t) for t in itertools.product(zipf_min[1], repeat =5))

    #cross2a = (''.join(t) for t in itertools.product(zipf_min[1], repeat =2))
    #cross2b = (''.join(t) for t in itertools.product(zipf_min[1], repeat =3))
    #cross2c = (''.join(t) for t in itertools.product(zipf_min[1], repeat =4))


    all_candidates = itertools.chain(cross1b)

    total = len(common)**3

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

