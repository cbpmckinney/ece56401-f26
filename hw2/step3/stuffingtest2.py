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
WORD1 = "aw2246"
WORD2 = "french44"
WORD3 = "pledgor"
WORDS = (WORD1, WORD2, WORD3)
NUMBERS = ("2246", "44")
_CASE_PATTERNS = (str.upper, str.lower, str.capitalize, lambda w: w)

# word_order_variants() below only permutes the 3 pre-fused word+number
# blocks (WORD1="aw2246", etc) -- a number always stays glued to whichever
# word it started on. To let a number land ANYWHERE independently (e.g.
# "frenchpledgor44aw2246"), treat each bare word and each number as its
# own movable token instead.
BARE_WORDS = ("aw", "french", "pledgor")
NUMBER_PARTS = ("2246", "44")
ALL_PARTS = BARE_WORDS + NUMBER_PARTS

# A common "policy made me add a special character" habit: keep the same
# words, just glue them together with a separator instead of concatenating
# directly. '' isn't included -- straight concatenation is already covered
# by word_order_variants() and friends, no need to duplicate those.
SEPARATORS = ("-", "_", ".", "!", "$")


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


def exhaustive_case_variants(s):
    """Every independent upper/lower choice for EACH alphabetic character in
    s -- true 2**k brute force (k = number of letters in s), unlike
    capitalization_variants()'s curated handful of ~5 realistic patterns.
    Digits/punctuation have no case, so they don't multiply the search --
    only the 15 letters in "aw2246french44pledgor" do, e.g."""
    letter_positions = [i for i, ch in enumerate(s) if ch.isalpha()]
    base = list(s.lower())
    for choice in itertools.product((str.lower, str.upper), repeat=len(letter_positions)):
        chars = base[:]
        for pos, fn in zip(letter_positions, choice):
            chars[pos] = fn(chars[pos])
        yield ''.join(chars)


def rearranged_and_number_mutated():
    return itertools.chain.from_iterable(
        number_mutations(base) for base in word_order_variants()
    )


def rearranged_number_and_case_mutated():
    return itertools.chain.from_iterable(
        capitalization_variants(c) for c in rearranged_and_number_mutated()
    )


def rearranged_number_and_exhaustive_case():
    """Same order+digit-permuted base strings as rearranged_number_and_case_mutated(),
    but crossed with EVERY case variation instead of the curated handful --
    84 base shapes * 2**15 case choices = 2,752,512 candidates."""
    return itertools.chain.from_iterable(
        exhaustive_case_variants(c) for c in rearranged_and_number_mutated()
    )


def part_order_variants(parts=ALL_PARTS):
    """Every ordering of each word AND number as its OWN movable token --
    unlike word_order_variants() (6 orderings of 3 pre-fused blocks), this
    treats all 5 pieces independently: 5! = 120 orderings, including things
    like "frenchpledgor44aw2246" that word_order_variants() can't reach."""
    for perm in itertools.permutations(parts):
        yield ''.join(perm)


def part_order_and_number_mutated():
    return itertools.chain.from_iterable(
        number_mutations(base) for base in part_order_variants()
    )


def part_order_number_and_exhaustive_case():
    """The full-generality version: every word/number placement x every
    digit-run permutation x every per-letter case choice."""
    return itertools.chain.from_iterable(
        exhaustive_case_variants(c) for c in part_order_and_number_mutated()
    )


def part_order_variants_with_separators(parts=ALL_PARTS, separators=SEPARATORS):
    """Same 5-independent-token placement as part_order_variants(), but
    joined with a separator instead of straight concatenation --
    120 orderings * 5 separators = 600 base shapes."""
    for perm in itertools.permutations(parts):
        for sep in separators:
            yield sep.join(perm)


def part_separated_and_number_mutated():
    return itertools.chain.from_iterable(
        number_mutations(base) for base in part_order_variants_with_separators()
    )


def part_separated_number_and_exhaustive_case():
    """The largest pool yet: every word/number placement x every separator
    x every digit-run permutation x every per-letter case choice."""
    return itertools.chain.from_iterable(
        exhaustive_case_variants(c) for c in part_separated_and_number_mutated()
    )


def word_order_variants_with_separators(words=WORDS, separators=SEPARATORS):
    """Same word-order permutations as word_order_variants(), but joined
    with a separator character instead of straight concatenation --
    6 orderings * 5 separators = 30 base shapes."""
    for permutation in itertools.permutations(words):
        for sep in separators:
            yield sep.join(permutation)


def separated_and_number_mutated():
    return itertools.chain.from_iterable(
        number_mutations(base) for base in word_order_variants_with_separators()
    )


def separated_number_and_exhaustive_case():
    """Separator-joined order x digit-run permutation x every per-letter
    case choice. The separator chars themselves aren't letters, so they
    don't add to the 2**15 case-choice count -- only multiply the base
    shape count (30 vs. word_order_variants()'s 6)."""
    return itertools.chain.from_iterable(
        exhaustive_case_variants(c) for c in separated_and_number_mutated()
    )

def per_word_capitalization_variants(words=WORDS):
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
        per_word_full_mutation,
        rearranged_number_and_exhaustive_case,
        word_order_variants_with_separators,
        separated_and_number_mutated,
        separated_number_and_exhaustive_case,
        part_order_variants,
        part_order_and_number_mutated,
        part_order_number_and_exhaustive_case,
        part_order_variants_with_separators,
        part_separated_and_number_mutated,
        part_separated_number_and_exhaustive_case,  # most expensive -- last
    ]

def all_candidates():
    return itertools.chain.from_iterable(fn() for fn in MUTATION_PATTERNS)



def main():

    start = time.perf_counter()
    last_print = start
    print_interval = 5  # seconds between progress prints
    count = 0
    total = sum(sum(1 for _ in fn()) for fn in MUTATION_PATTERNS)
    print(f'Total candidates: {total}')
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
