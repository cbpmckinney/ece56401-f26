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

targethash = '$y$j9T$F/vLDJRdzzspQonYxyqKl1$Q/nOKF5ECoPwQIJAZSlNcRt21Y3b1eV42Usj5SkfBX9'


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

    ip = open('dictionaries/common.txt', 'r')
    common = ip.read().splitlines()
    ip.close()

    ip = open('dictionaries/all.txt', 'r')
    all = ip.read().splitlines()
    ip.close()


    digits = '0123456789'
    #punct = string.punctuation
    year = '2026'

    digits1 = list(''.join(t) for t in itertools.product(digits, repeat = 1))
    digits2 = list(''.join(t) for t in itertools.product(digits, repeat = 2))
    digits3 = list(''.join(t) for t in itertools.product(digits, repeat = 3))
    digits4 = list(''.join(t) for t in itertools.product(digits, repeat = 4))

    cross1a = (''.join(t) for t in itertools.product(digits1, common))
    cross1b = (''.join(t) for t in itertools.product(common, digits1))
    cross2a = (''.join(t) for t in itertools.product(digits2, common))
    cross2b = (''.join(t) for t in itertools.product(common, digits2))
    cross3a = (''.join(t) for t in itertools.product(digits3, common))
    cross3b = (''.join(t) for t in itertools.product(common, digits3))
    cross4a = (''.join(t) for t in itertools.product(digits4, common))
    cross4b = (''.join(t) for t in itertools.product(common, digits4))

    yeara = (''.join(t) for t in itertools.product(all, year))
    yearb = (''.join(t) for t in itertools.product(year, all))

    all_candidates = itertools.chain(yeara, yearb, cross1a, cross1b, cross2a, cross2b, cross3a, cross3b)

    count = 0
    total = 2*len(common)*(1110) + 2*len(all)


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
                
                record = {"script": "crackappendprepend.py", "description": "Prepend and append 1-4 digits to common", "count": total, "result": f'Success: password is {result}'}
                body = json.dumps(record, indent=2)
                send_notification('Hashing Success!', body)
                log_job(record)
                exit(0)

    print('FAILURE!')
    record = {"script": "crackappendprepend.py", "description": "Prepend and append 1-3 digits to common + 2026 to all", "count": total, "result": f'Failure'}
    log_job(record)
    body = json.dumps(record, indent=2)
    send_notification('Hashing Failure!', body)



if __name__ == "__main__":
    main()

