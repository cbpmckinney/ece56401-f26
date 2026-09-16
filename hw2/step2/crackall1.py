import itertools
import string
import legacycrypt as crypt



# password hash method is yescrypt

bobhash = '$y$j9T$F/vLDJRdzzspQonYxyqKl1$Q/nOKF5ECoPwQIJAZSlNcRt21Y3b1eV42Usj5SkfBX9'


ip = open('all.txt', 'r')
fp = open('results-crackall.txt', 'w')

passwords = ip.readlines()
ip.close()

counter = 1
for candidate in passwords:
    result = crypt.crypt(candidate, bobhash)
    if (counter % 1000 == 0):
        print(f'Completed {counter} hashes')

    counter += 1
    if result == bobhash:
        print('SUCCESS!')
        fp.write('SUCCESS!\n')
        fp.write(f'Password is: {candidate}')
        fp.close()
        exit(0)

print('FAILURE!\n')
fp.write('FAILURE!\n')
fp.close()

