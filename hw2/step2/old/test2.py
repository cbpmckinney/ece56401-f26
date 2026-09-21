import itertools
import string
import legacycrypt as crypt
from multiprocessing import Pool
import math
import time 
from datetime import timedelta

def check_candidate(candidate: str):
    targethash = '$y$j9T$F/vLDJRdzzspQonYxyqKl1$vr0YsQd0asHQTZqf9v6nsGBLgmgWLzJdA/hPOZHQ9h5'
    #targethash = '$y$j9T$F/vLDJRdzzspQonYxyqKl1$Q/nOKF5ECoPwQIJAZSlNcRt21Y3b1eV42Usj5SkfBX9'
    targetsalt = '$y$j9T$F/vLDJRdzzspQonYxyqKl1'
    result = crypt.crypt(candidate, targetsalt)
    if result == targethash:
        return candidate
    else:
        return None


def main():

    word = 'apple'
    hash = crypt.crypt(word, '$y$j9T$F/vLDJRdzzspQonYxyqKl1')
    print(hash)
    result = check_candidate('apple1')
    print(result)



if __name__ == '__main__':
    main()


