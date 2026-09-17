import itertools
import string
import legacycrypt as crypt
from multiprocessing import Pool
import math
import time 
from datetime import timedelta

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
    
    alphabet = string.ascii_letters + string.digits + string.punctuation

    ip = open('all.txt', 'r')
    fp = open('results-mutation1.txt', 'w')

    passwords = ip.read().splitlines()
    #print(passwords)
    ip.close()


    #mutations = [''.join(t) for n in (1, 2) for t in itertools.product(alphabet, repeat=n)]
    # Change to only n=1 for tractability.  Otherwise it'll take 100 days!
    mutations = [''.join(t) for t in itertools.product(alphabet, repeat=1)]
    prepended = (s + password for password, s in itertools.product(passwords, mutations))
    appended = (password + s for password, s in itertools.product(passwords, mutations))

    all_candidates =  itertools.chain(appended, prepended)
    #all_candidates = appended
    #for word in all_candidates:
    #    print(word, end='')

    
    count = 0
    total = 2*product_size(passwords, mutations)

    start = time.perf_counter()

    with Pool(processes=12) as pool:
        for result in pool.imap_unordered(check_candidate, all_candidates, chunksize=1):
            #print(result)
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

