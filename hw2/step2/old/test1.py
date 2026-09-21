import itertools
import string


length = 1
fp = open(filename, 'w')
ip = open('all.txt', 'r')
pp = open('chars' + str(length) + '.txt')

passwords = ip.readlines()
ip.close()

prefixes = pp.readlines()
pp.close()



for chars in itertools.product(passwords, prefixes):
    candidate = ''.join(chars)
    fp.write(candidate + '\n')

fp.close()

