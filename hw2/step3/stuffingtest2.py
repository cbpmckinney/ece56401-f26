from encrypt import decrypt
import functools
import itertools
import os
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


def pattern_count(fn):
    """O(1) candidate count for fn if it has a closed-form .count
    attribute attached (set right after a function's own definition --
    see single_word_substitution_variants/two_word_substitution_variants
    below for examples); otherwise falls back to actually enumerating
    fn() once. New functions you add work automatically via the
    fallback -- only worth attaching a .count once a pool is big enough
    that enumerating it just to COUNT it becomes its own bottleneck."""
    if hasattr(fn, "count"):
        return fn.count()
    return sum(1 for _ in fn())


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


def whole_string_reversed_variants(words=WORDS):
    """Every word-order permutation, each reversed end-to-end as a single
    string -- "aw2246french44pledgor" -> "rogdelp44hcnerf6422wa", etc.
    Different from per_word_reversed_variants() below, which reverses each
    word in place but keeps their left-to-right order."""
    for base in word_order_variants(words):
        yield base[::-1]


def per_word_reversed_variants(words=WORDS):
    """Each word reversed individually (letters AND digits within that
    word, since a word like "aw2246" is one token here), words themselves
    still laid out in every order permutation -- NOT the same as reversing
    the whole joined string above."""
    for perm in itertools.permutations(words):
        yield ''.join(word[::-1] for word in perm)


def reversed_and_number_mutated():
    """Both reversal styles, crossed with digit-run permutation (reversing
    a word already reverses its digit run too, but number_mutations() on
    top still explores rearranging that reversed run further)."""
    return itertools.chain.from_iterable(
        number_mutations(base)
        for base in itertools.chain(whole_string_reversed_variants(), per_word_reversed_variants())
    )


def caesar_shift(s, shift):
    """Shift every ALPHABETIC character by `shift` positions, wrapping
    a->z / A->Z; digits and punctuation pass through unchanged (same
    "case/digits don't interact" rule as exhaustive_case_variants())."""
    out = []
    for ch in s:
        if ch.isalpha():
            base = ord('a') if ch.islower() else ord('A')
            out.append(chr((ord(ch) - base + shift) % 26 + base))
        else:
            out.append(ch)
    return ''.join(out)


def caesar_variants(words=WORDS):
    """Every word-order permutation, run through every nontrivial Caesar
    shift (1-25 -- shift 0 is just the plain string, already covered
    elsewhere, no point re-testing it): 6 orders * 25 shifts = 150."""
    for base in word_order_variants(words):
        for shift in range(1, 26):
            yield caesar_shift(base, shift)


def caesar_and_number_mutated():
    return itertools.chain.from_iterable(
        number_mutations(c) for c in caesar_variants()
    )


# Length-bucketed leaked-password dictionaries from Step 2, matched to the
# length of each of the three known words ("aw"=2, "french"=6,
# "pledgor"=7). These let us test "same pattern (word+2246+word+44+word),
# different actual words" without having to cross the full ~300K-word
# dictionary against itself.
_DICT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "step2", "dictionaries")

def _load_wordlist(filename):
    path = os.path.join(_DICT_DIR, filename)
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [line.rstrip("\n") for line in f if line.strip()]

WORDLIST_LEN2 = _load_wordlist("all_len2.txt")
WORDLIST_LEN6 = _load_wordlist("all_len6.txt")
WORDLIST_LEN7 = _load_wordlist("all_len7.txt")
WORDLIST_ALL = _load_wordlist("all.txt")
WORDLIST_COMMON = _load_wordlist("common.txt")

def change_last_word():
    for word in WORDLIST_ALL:
        yield WORD1+WORD2+word


def single_word_substitution_variants():
    """Keep the "word+2246+word+44+word" PATTERN and the known digits, but
    swap exactly ONE of the three words for a different same-length
    candidate from the dictionaries -- the other two stay exactly as
    cracked. Tests "Bob kept two of three words, changed one": the
    cheapest, most behaviorally plausible version of "similar pattern,
    not identical." 300 + 24,429 + 35,037 = 59,766 candidates."""
    w1_known, w2_known, w3_known = BARE_WORDS
    n1, n2 = NUMBER_PARTS
    for w1 in WORDLIST_ALL:
        yield f"{w1}{n1}{w2_known}{n2}{w3_known}"
    for w2 in WORDLIST_ALL:
        yield f"{w1_known}{n1}{w2}{n2}{w3_known}"
    for w3 in WORDLIST_ALL:
        yield f"{w1_known}{n1}{w2_known}{n2}{w3}"

single_word_substitution_variants.count = lambda: 3 * len(WORDLIST_ALL)


def two_word_substitution_variants():
    """Keep exactly ONE word fixed at its known value, vary the OTHER two
    across every combination from the length-matched dictionaries -- 3
    ways to pick which word stays fixed:
      fix "aw"      -> len6 x len7 =  24,429 * 35,037 ~ 855,968,673
      fix "french"  -> len2 x len7 =     300 * 35,037 ~  10,511,100
      fix "pledgor" -> len2 x len6 =     300 * 24,429 ~   7,328,700
    ~874M total -- a real job (~tens of minutes at current throughput),
    not free like single_word_substitution_variants() above."""
    w1_known, w2_known, w3_known = BARE_WORDS
    n1, n2 = NUMBER_PARTS
    for w2 in WORDLIST_ALL:
        for w3 in WORDLIST_ALL:
            yield f"{w1_known}{n1}{w2}{n2}{w3}"
    for w1 in WORDLIST_ALL:
        for w3 in WORDLIST_ALL:
            yield f"{w1}{n1}{w2_known}{n2}{w3}"
    for w1 in WORDLIST_ALL:
        for w2 in WORDLIST_ALL:
            yield f"{w1}{n1}{w2}{n2}{w3_known}"

two_word_substitution_variants.count = lambda: 3 * (len(WORDLIST_ALL) ** 2)


def two_word_substitution_shard(which_fixed, lo, hi):
    """One contiguous slice of two_word_substitution_variants(): same 3
    fix-one-word/vary-the-other-two branches, but the OUTER loop variable
    of branch `which_fixed` (0="aw" fixed, 1="french" fixed, 2="pledgor"
    fixed) is restricted to WORDLIST_ALL[lo:hi] instead of the whole
    list. Every other branch is skipped entirely -- this is what makes a
    set of these, one per (branch, slice) pair, a true disjoint partition
    of the full candidate space rather than a redundant re-scan of it.
    Built with functools.partial by two_word_substitution_shards() below;
    not meant to be called directly."""
    w1_known, w2_known, w3_known = BARE_WORDS
    n1, n2 = NUMBER_PARTS
    if which_fixed == 0:
        for w2 in WORDLIST_ALL[lo:hi]:
            for w3 in WORDLIST_ALL:
                yield f"{w1_known}{n1}{w2}{n2}{w3}"
    elif which_fixed == 1:
        for w1 in WORDLIST_ALL[lo:hi]:
            for w3 in WORDLIST_ALL:
                yield f"{w1}{n1}{w2_known}{n2}{w3}"
    else:
        for w1 in WORDLIST_ALL[lo:hi]:
            for w2 in WORDLIST_ALL:
                yield f"{w1}{n1}{w2}{n2}{w3_known}"


def two_word_substitution_shards(n_total_shards):
    """Splits two_word_substitution_variants() into n_total_shards (or
    fewer, if that doesn't divide evenly into >=1-sized pieces per
    branch) independent, disjoint zero-arg generator callables --
    together they yield EXACTLY the same multiset of strings as calling
    two_word_substitution_variants() once, just spread across however
    many OS processes the caller wants to run in parallel (one process
    generating candidates in pure Python is GIL-bound, so a single
    producer can't keep two fast GPUs fed; this is what lets candidate
    GENERATION itself be parallelized, not just GPU dispatch). Each of
    the 3 branches gets an equal share of the shards, and each branch's
    own word list is cut into that many contiguous (not strided --
    strided wouldn't reduce wall-clock generation time, only reduce
    packing/transfer work) index-range slices. Verified against a toy
    wordlist in /tmp/test_sharding.py: for every tested shard count, the
    combined, sorted output of all shards exactly reproduces
    sorted(list(two_word_substitution_variants())) with no gaps or
    duplicates."""
    n = len(WORDLIST_ALL)
    per_branch = max(1, n_total_shards // 3)
    shard_fns = []
    for branch in range(3):
        bounds = [round(i * n / per_branch) for i in range(per_branch + 1)]
        for i in range(per_branch):
            lo, hi = bounds[i], bounds[i + 1]
            if lo == hi:
                continue
            shard_fns.append(functools.partial(two_word_substitution_shard, branch, lo, hi))
    return shard_fns


two_word_substitution_variants.make_shards = two_word_substitution_shards


# Already run to completion against the real ciphertext -- no hits. Kept
# as a record so we don't accidentally re-test these (they're expensive)
# rather than deleting the evidence we tried them.
ALREADY_TRIED = [
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
        part_separated_number_and_exhaustive_case,
        whole_string_reversed_variants,
        per_word_reversed_variants,
        reversed_and_number_mutated,
        caesar_variants,
        caesar_and_number_mutated,
        single_word_substitution_variants,
        two_word_substitution_variants,  # only tested for common words
    ]

MUTATION_PATTERNS = [
        two_word_substitution_variants,  # only tested for common words
    ]

def all_candidates():
    return itertools.chain.from_iterable(fn() for fn in MUTATION_PATTERNS)



def main():

    
    
    print_interval = 5  # seconds between progress prints
    count = 0
    total = sum(pattern_count(fn) for fn in MUTATION_PATTERNS)
    start = time.perf_counter()
    last_print = start
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
