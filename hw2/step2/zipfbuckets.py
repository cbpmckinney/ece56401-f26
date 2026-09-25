import os
import string
import wordfreq

with open('dictionaries/common.txt', 'r') as fp:
    common = fp.read().splitlines()

with open('dictionaries/dictionary.txt', 'r') as fp:
    uncommon = fp.read().splitlines()


    