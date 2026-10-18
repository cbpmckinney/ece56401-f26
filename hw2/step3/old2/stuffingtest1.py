from encrypt import decrypt
import itertools
from multiprocessing import Pool
import time 
from datetime import timedelta
import smtplib
from email.message import EmailMessage
import string
import re


# GLOBAL DATA
with open("server/templates/secret.html.crypt", "rb") as fp:
    ciphertext = fp.read()

OLDPASSWORD = "aw2246french44pledgeor"
WORD1 = "aw"
WORD2 = "french"
WORD3 = "pledgor"
WORDS = (WORD1, WORD2, WORD3)
NUMBERS = ("2246", "44")
_CASE_PATTERNS = (str.upper, str.lower, str.capitalize, lambda w: w)


def decrypt_target(password):
    try:
        plaintext = decrypt(password, ciphertext)

    except Exception as e:
        return None

    print(password)
    return plaintext




def word_order_variants(words=WORDS):
    for permutation in itertools.permutations(words):
        yield ''.join(permutation)

def digit_runs(s):
    for m in re.finditer(r'\d+', s):
        yield m.start(), m.end(), m.group()

def number_mutations(s):
    yield s
    for start, end, digits in digit_runs(s):
        for permutation in set(itertools.permutations(digits)):
            yield s[:start] + ''.join(permutation) + s[end:]

def capitalization_variants(s):
    yield s
    yield s.upper()
    yield s.capitalize()
    yield s[0].upper() + s[1:]
    yield s[:-1] + s[-1].upper()


def rearranged_and_number_mutated():
    return itertools.chain.from_iterable(
        number_mutations(base) for base in word_order_variants()
    )


def rearranged_number_and_case_mutated():
    return itertools.chain.from_iterable(
        capitalization_variants(c) for c in rearranged_and_number_mutated()
    )

def per_word_capitalization_variants(words):
    """4 patterns ** len(words) combinations, e.g. 64 for 3 words."""
    for combo in itertools.product(_CASE_PATTERNS, repeat=len(words)):
        yield ''.join(fn(w) for fn, w in zip(combo, words))

def per_word_order_and_case_variants(words=WORDS):
    """Order permutation x per-word case, joined last: 6 * 64 = 384 candidates."""
    for perm in itertools.permutations(words):
        yield from per_word_capitalization_variants(perm)

def per_word_full_mutation():
    """... then number_mutations() on the joined string -- digit-run
    location doesn't depend on case, so applying it after the join is fine."""
    return itertools.chain.from_iterable(
        number_mutations(c) for c in per_word_order_and_case_variants()
    )


MUTATION_PATTERNS = [
        word_order_variants,
        rearranged_and_number_mutated,
        rearranged_number_and_case_mutated,
        per_word_capitalization_variants,
        per_word_order_and_case_variants,
        per_word_full_mutation
    ]

def all_candidates():
    return itertools.chain.from_iterable(fn() for fn in MUTATION_PATTERNS)

def main():

    

    start = time.perf_counter()
    last_print = start
    print_interval = 5  # seconds between progress prints
    count = 0
    with Pool(processes=12) as pool:
            for result in pool.imap_unordered(decrypt_target, all_candidates(), chunksize=128):
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
                    print(f"SUCCESS!  Secret is: {result}")
                    #fp.write(f"SUCCESS!  Password is: {result}")
                    #fp.close()
                    pool.terminate()
                    
                    #log_job(record)
                    exit(0)
    
    print('FAILURE!')
       #record = {"script": scriptname, "description": jobdescription, "count": total, "result": f'Failure'}
        #log_job(record)
        #body = json.dumps(record, indent=2)
        #send_notification('Hashing Failure!', body)

    


    







if __name__ == '__main__':
    main()
