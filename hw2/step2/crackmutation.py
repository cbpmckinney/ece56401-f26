import itertools
import string
import legacycrypt as crypt
from multiprocessing import Pool
import math
import time 
from datetime import timedelta

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
    #computer12 hash
    #targethash = '$y$j9T$F/vLDJRdzzspQonYxyqKl1$w3qEw/I.yfXHwGeD9fdfSG21XPdY13B714Nu0YiCRu6'


    result = crypt.crypt(candidate, targethash)
    if result == targethash:
        return candidate
    else:
        return None

# password hash method is yescrypt

def product_size(*iterables):
    return math.prod(len(it) for it in iterables)



def main():
    
    #alphabet = string.ascii_letters + string.digits + string.punctuation
    #alphabet = string.digits + string.punctuation
    alphabet = string.digits + '!@#$%^&*()-_+='

    ip = open('dictionaries/common.txt', 'r')
    fp = open('results-mutations1to4.txt', 'w')

    passwords = ip.read().splitlines()
    #print(passwords)
    ip.close()


    #mutations = [''.join(t) for n in (1, 2) for t in itertools.product(alphabet, repeat=n)]
    # Change to only n=1 for tractability.  Otherwise it'll take 100 days!
    mutations1 = [''.join(t) for t in itertools.product(alphabet, repeat=1)]
    mutations2 = [''.join(t) for t in itertools.product(alphabet, repeat=2)]
    mutations3 = [''.join(t) for t in itertools.product(alphabet, repeat=3)]
    #mutations4 = [''.join(t) for t in itertools.product(alphabet, repeat=4)]
    
    #prepended = (s + password for password, s in itertools.product(passwords, mutations))
    appended1 = (password + s for password, s in itertools.product(passwords, mutations1))
    appended2 = (password + s for password, s in itertools.product(passwords, mutations2))
    appended3 = (password + s for password, s in itertools.product(passwords, mutations3))
    #appended4 = (password + s for password, s in itertools.product(passwords, mutations4))


    #all_candidates =  itertools.chain(appended, prepended)
    all_candidates = itertools.chain(appended1, appended2, appended3)
    #for word in all_candidates:
    #    print(word, end='')

    
    count = 0
    total = product_size(passwords, mutations1) + product_size(passwords, mutations2) + product_size(passwords, mutations3)

    start = time.perf_counter()
    last_print = start
    print_interval = 5  # seconds between progress prints

    with Pool(processes=32) as pool:
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
                fp.write(f"SUCCESS!  Password is: {result}")
                fp.close()
                pool.terminate()
                send_notification('Hashing Success (mutation)!', body=f'Found password: {result}')
                print(f'Started at: {start}')
                print(f'Ended at: {time.perf_counter()}')
                print(f'Time used: {time.perf_counter()-start}')


                exit(0)

    print('FAILURE!')
    fp.write('FAILURE!')
    fp.close()
    send_notification('Hashing failure (mutation) :(', body=f'Hashing failed')



if __name__ == "__main__":
    main()

