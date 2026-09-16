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




