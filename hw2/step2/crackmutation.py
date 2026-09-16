import itertools
import string
import legacycrypt as crypt
from multiprocessing import Pool
import math
import time 
from datetime import timedelta

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



def main():

    
    alphabet = string.ascii_letters + string.digits + string.punctuation

    ip = open('all.txt', 'r')
    fp = open('results-mutation1.txt', 'w')

    passwords = ip.readlines()
    ip.close()

    #mutations = [''.join(t) for n in (1, 2) for t in itertools.product(alphabet, repeat=1)]
    mutations = [''.join(t) for t in itertools.product(alphabet, repeat=1)]
    prepended = (s + password for password, s in itertools.product(passwords, mutations))
    appended = (password + s for password, s in itertools.product(passwords, mutations))

    all_candidates =  itertools.chain(prepended, appended)
    
    count = 0
    total = 2*product_size(passwords, mutations)

    start = time.perf_counter()

    with Pool(processes=12) as pool:
        for result in pool.imap_unordered(check_candidate, all_candidates, chunksize=1):
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
                pool.terminate()
                exit(0)

    print('FAILURE!')
    fp.write('FAILURE!')
    fp.close()




if __name__ == "__main__":
    main()




import itertools
import string
import legacycrypt as crypt
from multiprocessing import Pool
import math

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



def main():



    ip = open('all.txt', 'r')
    fp = open('results-crackall.txt', 'w')

    passwords = ip.readlines()
    ip.close()

    count = 0
    total = product_size(passwords)

    with Pool(processes=12) as pool:
        for result in pool.imap_unordered(check_candidate, passwords, chunksize=1):
            count += 1
            if (count % 1000 == 0):
                print(f"{count} / {total} checked ({100*count/total:.2f}%)")

            if result is not None:
                print(f"SUCCESS!  Password is: {result}")
                fp.write(f"SUCCESS!  Password is: {result}")
                fp.close()
                pool.terminate()
                exit(0)

    print('FAILURE!')
    fp.write('FAILURE!')
    fp.close()




if __name__ == "__main__":
    main()




