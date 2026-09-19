import itertools
import string
import legacycrypt as crypt
from multiprocessing import Pool
import math
import time 
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import smtplib
from email.message import EmailMessage


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

    targethash = '$y$j9T$F/vLDJRdzzspQonYxyqKl1$Q/nOKF5ECoPwQIJAZSlNcRt21Y3b1eV42Usj5SkfBX9'

    #computer1 test: computer is in the dictionary, adding 1
    #hash for computer1 using the given salt is
    # $y$j9T$F/vLDJRdzzspQonYxyqKl1$vSd5IS8bJ9suGIKokVZA2c1gWG.GEJhe.m/CQSwkAq9
    #targethash = '$y$j9T$F/vLDJRdzzspQonYxyqKl1$vSd5IS8bJ9suGIKokVZA2c1gWG.GEJhe.m/CQSwkAq9'


    result = crypt.crypt(candidate, targethash)
    if result == targethash:
        return candidate
    else:
        return None

# password hash method is yescrypt

def product_size(*iterables):
    return math.prod(len(it) for it in iterables)



def main():

    ip = open('common.txt', 'r')
    fp = open('results-concatenation-common.txt', 'w')

    passwords = ip.read().splitlines()
    ip.close()

    all_candidates = (''.join(t) for t in itertools.product(passwords, repeat=2))

    count = 0
    total = product_size(passwords, passwords)

    start = time.perf_counter()

    max_workers = 12

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_candidate, c) for c in itertools.islice(all_candidates, max_workers)}

        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)

            for future in done:
                result = future.result()
                count += 1
                if (count % 10000 == 0):
                    elapsed = time.perf_counter() - start
                    rate = count / elapsed                      # candidates per second
                    remaining = total - count
                    eta_seconds = remaining / rate
                    print(f"{count}/{total} ({100*count/total:.2f}%) "f"- {rate:.1f}/s - ETA {timedelta(seconds=int(eta_seconds))}")

                if result is not None:
                    print(f"SUCCESS!  Password is: {result}")
                    fp.write(f"SUCCESS!  Password is: {result}")
                    fp.close()
                    send_notification('Hashing Success (concatenation common)!', body=f'Found password: {result}')
                    executor.shutdown(wait=False, cancel_futures=True)
                    exit(0)

            for c in itertools.islice(all_candidates, len(done)):
                futures.add(executor.submit(check_candidate, c))

    print('FAILURE!')
    fp.write('FAILURE!')
    fp.close()
    send_notification('Hashing failure (concatenation common) :(', body=f'Hashing failed')



if __name__ == "__main__":
    main()

