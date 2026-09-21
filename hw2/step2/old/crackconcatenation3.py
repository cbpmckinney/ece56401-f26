import itertools
import string
import legacycrypt as crypt
from multiprocessing import Pool
import math
import time 
from datetime import timedelta
from hw2.step2.old.lengthbuckets import iter_length_buckets, count_length_buckets



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

    ip = open('dictionaries/common.txt', 'r')
    common = ip.read().splitlines()
    ip.close()

    digits = '0123456789'
    punct = string.punctuation

    digits1 = (''.join(t) for t in itertools.product(digits, repeat=1))
    digits2 = (''.join(t) for t in itertools.product(digits, repeat=2))
    digits3 = (''.join(t) for t in itertools.product(digits, repeat=3))
    digits4 = (''.join(t) for t in itertools.product(digits, repeat=4))


    cross1a = (''.join(t) for t in itertools.product(common, digits1))
    cross1b = (''.join(t) for t in itertools.product(digits1, common))
    cross2a = (''.join(t) for t in itertools.product(common, digits2))
    cross2b = (''.join(t) for t in itertools.product(digits2, common))
    cross3a = (''.join(t) for t in itertools.product(common, digits))
    cross3b = (''.join(t) for t in itertools.product(digits3, common))

    cross4a = (''.join(t) for t in itertools.product(common, digits4))
    cross4b = (''.join(t) for t in itertools.product(digits4, common))
        
    all_candidates = itertools.chain(cross3a, cross3b)

    count = 0
    total = 2*len(common)*(1000)

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
                send_notification('Hashing Success (concatenation mixed)!', body=f'Found password: {result}')

                exit(0)

    print('FAILURE!')
    #fp.write('FAILURE!')
    #fp.close()
    send_notification('Hashing failure (concatenation mixed) :(', body=f'Hashing failed')



if __name__ == "__main__":
    main()

