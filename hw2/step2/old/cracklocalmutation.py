# Local mutation: attempts capitalization, leetspeak substitution, reversal, vanilla
# Does not append anything.  crackmutation.py does the appending/prepending case.

import itertools
import string
import legacycrypt as crypt
from multiprocessing import Pool
import math
import time 
from datetime import timedelta

import smtplib
from email.message import EmailMessage

leet_map = {
    'a': '4',
    'e': '3',
    'i': '1',
    'o': '0',
    's': '5',
    't': '7',
    'b': '8',
    'g': '9',
    'l': '1',
}



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

    result = crypt.crypt(candidate, targethash)
    if result == targethash:
        return candidate
    else:
        return None



# password hash method is yescrypt

def product_size(*iterables):
    return math.prod(len(it) for it in iterables)


def case_variants(word):
    choices = [tuple({c.lower(), c.upper()}) for c in word]
    return (''.join(t) for t in itertools.product(*choices))


def case_variant_count(word):
    return 2 ** sum(1 for c in word if c.lower() != c.upper())


def leet_variants(word):
    choices = [(c, leet_map[c]) if c in leet_map else (c,) for c in word]
    return (''.join(t) for t in itertools.product(*choices))


def leet_variant_count(word):
    return 2 ** sum(1 for c in word if c in leet_map)



def main():
    ip = open('dictionaries/all.txt', 'r')
    fp = open('results-localmutation.txt', 'w')

    passwords = ip.read().splitlines()
    #print(passwords)
    ip.close()

    firstcap = (password.capitalize() for password in passwords)
    reversed = (password[::-1] for password in passwords )
    allcap = (password.upper() for password in passwords)
    #leetcase = (v for password in passwords for v in leet_variants(password))
    #fullcase = (v for password in passwords for v in case_variants(password))

    
    all_candidates = itertools.chain(passwords, reversed, firstcap, allcap)#, leetcase, fullcase)
    
    count = 0
    fullcasecount = sum(case_variant_count(password) for password in passwords)
    leetcount = sum(leet_variant_count(password) for password in passwords)
    
    total = 4*len(passwords)# + leetcount + fullcasecount


    start = time.perf_counter()
    last_print = start
    print_interval = 5  # seconds between progress prints

    with Pool(processes=12) as pool:
        for result in pool.imap_unordered(check_candidate, all_candidates, chunksize=32):
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
                send_notification('Hashing Success (local mutation)!', body=f'Found password: {result}')
                print(f'Started at: {start}')
                print(f'Ended at: {time.perf_counter()}')
                print(f'Time used: {time.perf_counter()-start}')


                exit(0)

    print('FAILURE!')
    fp.write('FAILURE!')
    fp.close()
    send_notification('Hashing failure (local mutation) :(', body=f'Hashing failed')



if __name__ == "__main__":
    main()

