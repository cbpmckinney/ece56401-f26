import functools
import itertools
import os
import sys
import time 
from datetime import timedelta
import smtplib
from email.message import EmailMessage
import string
import re
import requests


# GLOBAL DATA
url = "http://127.0.0.1:8080/decrypt"

# Offline crypto helpers. This module's own main() cracks via the Flask
# server (check_password_online() below); the GPU offline crackers in
# gpu/ import THIS module for its candidate generators AND for these three
# names (ciphertext / decrypt / decrypt_target). gpu_crack_*.py chdir to
# step3 before importing, so these relative paths resolve the same way they
# do when this script is run directly from step3.
from encrypt import decrypt
with open("server/templates/secret.html.crypt", "rb") as _fp:
    ciphertext = _fp.read()


def decrypt_target(password):
    """Offline single-candidate check (used by gpu/gpu_crack_cuda.py):
    returns the decrypted plaintext on success, None on failure."""
    try:
        return decrypt(password, ciphertext)
    except Exception:
        return None


OLDPASSWORD = "aw2246french44pledgeor"
WORD1 = "aw2246"
WORD2 = "french44"
WORD3 = "pledgeor"
WORDS = (WORD1, WORD2, WORD3)
NUMBERS = ("2246", "44")
_CASE_PATTERNS = (str.upper, str.lower, str.capitalize, lambda w: w)

# word_order_variants() below only permutes the 3 pre-fused word+number
# blocks (WORD1="aw2246", etc) -- a number always stays glued to whichever
# word it started on. To let a number land ANYWHERE independently (e.g.
# "frenchpledgor44aw2246"), treat each bare word and each number as its
# own movable token instead.
BARE_WORDS = ("aw", "french", "pledgeor")
NUMBER_PARTS = ("2246", "44")
ALL_PARTS = BARE_WORDS + NUMBER_PARTS

# A common "policy made me add a special character" habit: keep the same
# words, just glue them together with a separator instead of concatenating
# directly. '' isn't included -- straight concatenation is already covered
# by word_order_variants() and friends, no need to duplicate those.
SEPARATORS = ("-", "_", ".", "!", "$")

# Full letters+digits+punctuation alphabet, for tacking a short run of
# characters onto the FRONT or BACK of the whole known password --
# stuffingonline1.py defined a suffixes() helper for exactly this, but its
# main loop only ever called prefixes(), so the suffix side was never
# actually tested against the server. This is arguably the single most
# "trivial" mutation a person makes (add a digit or symbol at the end),
# so it's worth covering both directions here, short lengths first.
SYMBOLS = string.ascii_letters + string.digits + string.punctuation


def _fixed_length_suffixes(base, chars, length):
    for combo in itertools.product(chars, repeat=length):
        yield base + ''.join(combo)


def _fixed_length_prefixes(base, chars, length):
    for combo in itertools.product(chars, repeat=length):
        yield ''.join(combo) + base


def suffix_variants_short():
    """1-2 character suffix appended to OLDPASSWORD, from the full
    letters+digits+punctuation alphabet. 94 + 94**2 = 8,930 candidates."""
    return itertools.chain(
        _fixed_length_suffixes(OLDPASSWORD, SYMBOLS, 1),
        _fixed_length_suffixes(OLDPASSWORD, SYMBOLS, 2),
    )


def prefix_variants_short():
    """Mirror of suffix_variants_short(), prepended instead -- also
    re-covers the short end of stuffingonline1.py's length-1..3 prefix
    sweep in case that run didn't finish. 8,930 candidates."""
    return itertools.chain(
        _fixed_length_prefixes(OLDPASSWORD, SYMBOLS, 1),
        _fixed_length_prefixes(OLDPASSWORD, SYMBOLS, 2),
    )


def suffix_variants_long():
    """3-character suffix -- the untested counterpart to
    stuffingonline1.py's 3-character prefix sweep. 94**3 = 830,584
    candidates (~35 min @ ~400 req/s)."""
    return _fixed_length_suffixes(OLDPASSWORD, SYMBOLS, 3)


def prefix_variants_long():
    """3-character prefix, re-run here in case stuffingonline1.py's
    original run didn't finish. 830,584 candidates (~35 min @ ~400
    req/s)."""
    return _fixed_length_prefixes(OLDPASSWORD, SYMBOLS, 3)


def check_password_online(password):
    try:
        resp = requests.post(url, data={"password": password})
    except requests.RequestException:
        return None

    if resp.status_code != 403:
        print(password)
        return resp.text

    return None


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


# ---------------------------------------------------------------------
# "Single trivial edit" pools -- added after the first full sweep came up
# empty. The pools above mostly STACK transformations (order x case x
# digit-permute x ...), which overshoots "trivial" while missing the
# simplest one-change edits a person makes on a forced reset. These cover
# those single edits on the known old password. All are tiny (hundreds
# total), so they go first in MUTATION_PATTERNS.
# ---------------------------------------------------------------------

# Tokens people glue on during a forced change. 1-3 char cases are already
# covered exhaustively by suffix/prefix_variants, so this leans on 4+ char
# and "year" tokens those can't reach.
_COMMON_TOKENS = (
    [str(y) for y in range(2010, 2028)] +
    ["1", "12", "123", "1234", "12345", "0", "00", "000", "01", "007",
     "69", "420", "101", "69420", "666", "777", "1337"] +
    ["!", "!!", "!!!", "@", "#", "$", "%", "?", ".", "*", "_", "-"] +
    ["1!", "!1", "12!", "123!", "!123", "@123", "#1", "2024!", "2025!",
     "2026!", "00!", "007!"]
)

def append_prepend_common_tokens():
    """OLDPASSWORD -- and its capitalize-first form, since 'cap the first
    letter AND add a suffix' is a classic combo the pure-append pools
    never reach -- with each common token glued on the end, then the
    front."""
    bases = (OLDPASSWORD, OLDPASSWORD[:1].upper() + OLDPASSWORD[1:])
    seen = set()
    for base in bases:
        for tok in _COMMON_TOKENS:
            for cand in (base + tok, tok + base):
                if cand not in seen:
                    seen.add(cand)
                    yield cand


def number_value_variants():
    """Change the VALUE of an embedded number run (the pools above only
    PERMUTE its digits). For each run (2246, 44) one at a time: +/-1..20
    (natural and original-width-zero-padded), every year 2010-2027, and a
    few common number strings -- rest of the password left exactly as-is.
    The classic 'I just bumped the number' reset."""
    common = ["0000", "1111", "1234", "2468", "1212", "9999", "0123", "4321"]
    seen = set()
    for start, end, digits in digit_runs(OLDPASSWORD):
        n = int(digits); width = len(digits)
        reps = set()
        for d in range(-20, 21):
            if d == 0:
                continue
            v = n + d
            if v < 0:
                continue
            reps.add(str(v)); reps.add(str(v).zfill(width))
        reps.update(str(y) for y in range(2010, 2028))
        reps.update(common)
        for rep in reps:
            cand = OLDPASSWORD[:start] + rep + OLDPASSWORD[end:]
            if cand != OLDPASSWORD and cand not in seen:
                seen.add(cand)
                yield cand


# Each letter -> its common leet symbol options (first = canonical,
# used by the type-global leet_variants). Enriched after the thin
# a/e/o/l/g map missed c and h (both in "french") and all alternates.
_LEET_OPTIONS = {
    "a": ["@", "4"], "b": ["8"], "c": ["(", "<"], "e": ["3"],
    "g": ["9", "6"], "h": ["#"], "i": ["1", "!"], "l": ["1", "!"],
    "o": ["0"], "s": ["$", "5"], "t": ["7", "+"], "z": ["2"],
}
_LEET_MAP = {k: v[0] for k, v in _LEET_OPTIONS.items()}

def leet_variants():
    """Every subset of leet substitutions on OLDPASSWORD: each
    substitutable letter TYPE present is either replaced everywhere it
    occurs or left alone -> 2**(#distinct substitutable letters present),
    ~2**5 here. Yields the plain string once (empty subset); harmless."""
    present = [ch for ch in dict.fromkeys(OLDPASSWORD.lower()) if ch in _LEET_MAP]
    seen = set()
    for r in range(len(present) + 1):
        for combo in itertools.combinations(present, r):
            cand = "".join(_LEET_MAP[ch] if ch in combo else ch
                           for ch in OLDPASSWORD.lower())
            if cand not in seen:
                seen.add(cand)
                yield cand


def number_bruteforce_single():
    """Replace ONE embedded number run with EVERY same-width value
    (0000-9999 for the 4-digit run, 00-99 for the 2-digit run); the other
    run and all letters stay put. 10,000 + 100 = 10,100. The exhaustive
    version of the curated number_value_variants above."""
    for start, end, digits in digit_runs(OLDPASSWORD):
        width = len(digits)
        for v in range(10 ** width):
            yield OLDPASSWORD[:start] + str(v).zfill(width) + OLDPASSWORD[end:]


def number_bruteforce_both():
    """Replace BOTH embedded number runs at once with every same-width
    combination: 10,000 (4-digit) x 100 (2-digit) = 1,000,000. Covers
    'both numbers changed', which number_bruteforce_single can't reach."""
    runs = list(digit_runs(OLDPASSWORD))
    (s1, e1, d1), (s2, e2, d2) = runs[0], runs[1]
    w1, w2 = len(d1), len(d2)
    pre, mid, post = OLDPASSWORD[:s1], OLDPASSWORD[e1:s2], OLDPASSWORD[e2:]
    for v1 in range(10 ** w1):
        r1 = str(v1).zfill(w1)
        for v2 in range(10 ** w2):
            yield pre + r1 + mid + str(v2).zfill(w2) + post


# ---------------------------------------------------------------------
# "Two edits of the SAME KIND at two DIFFERENT locations" (professor's
# refined hint). Everything else applies an edit once / at the ends /
# globally-by-type; these apply one positional single-char edit at every
# PAIR of applicable positions. Two-letter CASE changes are already
# covered by the exhaustive-case pools, and both-number changes by
# number_bruteforce_both, so those kinds are not repeated here.
# ---------------------------------------------------------------------

_INSERT_CHARS = "0123456789!@#$%._-"

def _leetable_positions(s):
    return [i for i, ch in enumerate(s) if ch.lower() in _LEET_MAP]

def leet_two_positions():
    """Leet exactly TWO characters of OLDPASSWORD -- every pair of leetable
    positions, every symbol option for each (a->@/4, g->9/6, c->(/<, h->#,
    ...). Unlike leet_variants, which replaces a letter TYPE everywhere.
    Enriched map: c and h are leetable now, plus alternate symbols."""
    s = OLDPASSWORD
    seen = set()
    for i, j in itertools.combinations(_leetable_positions(s), 2):
        for si in _LEET_OPTIONS[s[i].lower()]:
            for sj in _LEET_OPTIONS[s[j].lower()]:
                chars = list(s)
                chars[i] = si
                chars[j] = sj
                cand = "".join(chars)
                if cand not in seen:
                    seen.add(cand); yield cand

def transpose_two_positions():
    """Swap each of TWO different adjacent character pairs. C(len-1, 2)."""
    s = OLDPASSWORD
    seen = set()
    for a, b in itertools.combinations(range(len(s) - 1), 2):
        chars = list(s)
        chars[a], chars[a+1] = chars[a+1], chars[a]
        chars[b], chars[b+1] = chars[b+1], chars[b]
        cand = "".join(chars)
        if cand != s and cand not in seen:
            seen.add(cand); yield cand

def delete_two_positions():
    """Delete TWO characters, every pair of positions. C(len, 2)."""
    s = OLDPASSWORD
    seen = set()
    for i, j in itertools.combinations(range(len(s)), 2):
        cand = "".join(ch for k, ch in enumerate(s) if k != i and k != j)
        if cand not in seen:
            seen.add(cand); yield cand

def duplicate_two_positions():
    """Double TWO characters, every pair of positions. C(len, 2)."""
    s = OLDPASSWORD
    seen = set()
    for i, j in itertools.combinations(range(len(s)), 2):
        chars = list(s)
        for k in sorted((i, j), reverse=True):
            chars.insert(k + 1, chars[k])
        cand = "".join(chars)
        if cand not in seen:
            seen.add(cand); yield cand

def insert_two_positions():
    """Insert the SAME character at two different gaps, for each char in a
    small insert set. C(len+1, 2) gap-pairs x len(_INSERT_CHARS)."""
    s = OLDPASSWORD
    n = len(s)
    seen = set()
    for ch in _INSERT_CHARS:
        for i, j in itertools.combinations(range(n + 1), 2):
            chars = list(s)
            chars.insert(j, ch)   # higher gap first so i stays valid
            chars.insert(i, ch)
            cand = "".join(chars)
            if cand not in seen:
                seen.add(cand); yield cand


def reverse_two_words():
    """Reverse exactly TWO of the three bare words in place (letters only;
    the numbers stay put). per_word_reversed_variants reversed ALL words
    as fused word+number blocks, never 'exactly two bare words'. C(3,2)=3."""
    spans = [(mt.start(), mt.end()) for mt in re.finditer(r'[A-Za-z]+', OLDPASSWORD)]
    seen = set()
    for a, b in itertools.combinations(range(len(spans)), 2):
        chars = list(OLDPASSWORD)
        for st, en in (spans[a], spans[b]):
            chars[st:en] = list(OLDPASSWORD[st:en][::-1])
        cand = "".join(chars)
        if cand != OLDPASSWORD and cand not in seen:
            seen.add(cand); yield cand


def grow_both_numbers():
    """Grow BOTH number runs by one digit (same kind -- 'add a digit to a
    number' -- at two locations). number_bruteforce_both held the widths
    fixed, so these length-changing edits are new. append-to-both (10x10)
    + prepend-to-both (10x10) = 200."""
    runs = list(digit_runs(OLDPASSWORD))
    (s1, e1, _), (s2, e2, _) = runs[0], runs[1]
    seen = set()
    for d1 in "0123456789":
        for d2 in "0123456789":
            app = OLDPASSWORD[:e1] + d1 + OLDPASSWORD[e1:e2] + d2 + OLDPASSWORD[e2:]
            pre = OLDPASSWORD[:s1] + d1 + OLDPASSWORD[s1:s2] + d2 + OLDPASSWORD[s2:]
            for cand in (app, pre):
                if cand not in seen:
                    seen.add(cand); yield cand


# Closed-form candidate counts for the pools that would otherwise be
# ENUMERATED in pure Python just to compute main()'s startup `total`
# (the exhaustive-case pools alone are hundreds of millions of candidates
# -- counting them by enumeration would stall the run's start for 15+
# minutes). exhaustive_case_variants(s) yields exactly 2**(letters in s),
# and every base string these pools expand carries the same letters as
# OLDPASSWORD (reordering, separators and digit permutation never change
# the letter multiset), so each pool is just (#base strings) *
# 2**(#letters in OLDPASSWORD). Validated against full enumeration of the
# smallest pool. _N_LETTERS tracks OLDPASSWORD (16 now, after the
# pledgor->pledgeor fix -- was 15 before), so these stay correct if the
# known words ever change.
_N_LETTERS = sum(ch.isalpha() for ch in OLDPASSWORD)
rearranged_number_and_exhaustive_case.count = lambda: sum(1 for _ in rearranged_and_number_mutated()) * (2 ** _N_LETTERS)
separated_number_and_exhaustive_case.count = lambda: sum(1 for _ in separated_and_number_mutated()) * (2 ** _N_LETTERS)
part_order_number_and_exhaustive_case.count = lambda: sum(1 for _ in part_order_and_number_mutated()) * (2 ** _N_LETTERS)
part_separated_number_and_exhaustive_case.count = lambda: sum(1 for _ in part_separated_and_number_mutated()) * (2 ** _N_LETTERS)
suffix_variants_short.count = lambda: len(SYMBOLS) + len(SYMBOLS) ** 2
prefix_variants_short.count = lambda: len(SYMBOLS) + len(SYMBOLS) ** 2
suffix_variants_long.count = lambda: len(SYMBOLS) ** 3
prefix_variants_long.count = lambda: len(SYMBOLS) ** 3
_NUM_RUNS = list(digit_runs(OLDPASSWORD))
number_bruteforce_single.count = lambda: sum(10 ** len(d) for _, _, d in _NUM_RUNS)
number_bruteforce_both.count = lambda: 10 ** len(_NUM_RUNS[0][2]) * 10 ** len(_NUM_RUNS[1][2])


# Fully exhausted against the GPU with the CORRECTED "pledgeor" stem and
# NO hit. The whole ~844M trivial-mutation space (everything here) finishes
# in ~10 min at observed GPU rates, and the post-fix sweep ran well past
# that, so these are genuinely done -- kept here so a normal run doesn't
# re-test them. (two_word_substitution_variants is deliberately NOT here:
# at ~270B it never finished, so it stays open in MUTATION_PATTERNS.)
ALREADY_TRIED = [
        # tiny structural / case

        # ~1M
        word_order_variants,
        whole_string_reversed_variants,
        per_word_reversed_variants,
        word_order_variants_with_separators,
        per_word_capitalization_variants,
        rearranged_and_number_mutated,
        part_order_variants,
        caesar_variants,
        reversed_and_number_mutated,
        per_word_order_and_case_variants,
        rearranged_number_and_case_mutated,
        separated_and_number_mutated,
        part_order_variants_with_separators,
        # small
        caesar_and_number_mutated,
        part_order_and_number_mutated,
        per_word_full_mutation,
        part_separated_and_number_mutated,
        suffix_variants_short,
        prefix_variants_short,
        suffix_variants_long,
        prefix_variants_long,
        single_word_substitution_variants,
        # millions -> hundreds of millions (exhaustive per-letter case)
        rearranged_number_and_exhaustive_case,
        separated_number_and_exhaustive_case,
        part_order_number_and_exhaustive_case,
        part_separated_number_and_exhaustive_case,
        # single trivial edits + exhaustive number replacement
        # (2nd full sweep after the pledgeor fix -- no hit)
        leet_variants,
        number_value_variants,
        append_prepend_common_tokens,
        number_bruteforce_single,
        number_bruteforce_both,
        # two-location char edits (fully exhausted online, no hit)
        transpose_two_positions,
        delete_two_positions,
        duplicate_two_positions,
        insert_two_positions,
        leet_two_positions,        # ~150  (leet 2 chars, ENRICHED map: +c +h +alt symbols)
        reverse_two_words,         # 3     (reverse 2 of the 3 bare words in place)
        grow_both_numbers,         # 200   (add a digit to BOTH numbers, append/prepend)

        rearranged_number_and_exhaustive_case,
                    separated_number_and_exhaustive_case,
                    part_order_number_and_exhaustive_case,
                    part_separated_number_and_exhaustive_case,
    ]

# EVERYTHING, re-ordered cheapest-first so trivial hits surface quickly
# whether this runs online (~400/s -- realistically only the first few
# tiers are feasible here) or on the GPU crackers in gpu/, which import
# this list and can chew through the lower tiers in seconds and the
# hundred-million-candidate ones in minutes. Counts are candidates per
# pool (at OLDPASSWORD's 16 letters):
# UNTRIED against the corrected stem -- cheapest-first. Everything that
# has already been fully swept now lives in ALREADY_TRIED above. The
# single-edit + number pools are tiny (seconds, even online);
# two_word_substitution_variants is the lone monster and never finished,
# so it stays last.
# TWO edits of the SAME KIND at two DIFFERENT locations (professor's
# refined hint). The char-level kinds we fully exhausted online
# (transpose/delete/duplicate/insert) and the ones covered elsewhere
# (2-letter CASE via exhaustive-case, both-number via number_bruteforce_both)
# are all in ALREADY_TRIED now. What's left here: (1) positional leet
# RE-RUN with an enriched symbol map -- the first pass used a thin
# a/e/o/l/g map, one symbol each, and missed c, h and all alternates
# (a4 g6 h# c( ...); (2) token-level same-kind edits earlier pools never
# isolated. ~350 candidates, ~1s online.
MUTATION_PATTERNS = [
            # millions -> hundreds of millions (exhaustive per-letter case)
            
            # single trivial edits + exhaustive number replacement
            # (2nd full sweep after the pledgeor fix -- no hit)
            leet_variants,
            number_value_variants,
            append_prepend_common_tokens,
            number_bruteforce_single,
            number_bruteforce_both,
            # two-location char edits (fully exhausted online, no hit)
            transpose_two_positions,
            delete_two_positions,
            duplicate_two_positions,
            insert_two_positions,
            leet_two_positions,        # ~150  (leet 2 chars, ENRICHED map: +c +h +alt symbols)
            reverse_two_words,         # 3     (reverse 2 of the 3 bare words in place)
            grow_both_numbers,         # 200   (add a dig

    ]

# PARKED (deliberately NOT in MUTATION_PATTERNS): two whole-word dictionary
# swaps is a valid "same-kind x 2-location" edit, but at ~270B it's a
# GPU-only overnight grind, out of scope for the online 400/s approach the
# professor assumes. The generator is still defined above -- re-add
# two_word_substitution_variants to MUTATION_PATTERNS to grind it on GPU.

def all_candidates():
    return itertools.chain.from_iterable(fn() for fn in MUTATION_PATTERNS)


def write_exemplars(path="exemplars.txt", n_per_pattern=10, patterns=None):
    """Writes a small, human-readable sample of candidates from EACH
    pattern category to `path` -- one password per line, grouped under a
    '## function_name' header per category -- so you can eyeball that
    every pattern is actually producing the shape of candidate you
    expect before spending real GPU time running it.

    Defaults to ALREADY_TRIED rather than MUTATION_PATTERNS: MUTATION_
    PATTERNS is just whichever pattern(s) are currently ACTIVE (one,
    right now -- two_word_substitution_variants), while ALREADY_TRIED
    is the full catalog of every named mutation category this file
    defines (and already includes two_word_substitution_variants too,
    so nothing is duplicated or missed by using it instead). Pass an
    explicit `patterns` list to sample something else instead -- e.g.
    patterns=MUTATION_PATTERNS for just the active one(s), or
    patterns=ALREADY_TRIED + MUTATION_PATTERNS for literally everything
    (redundant today since MUTATION_PATTERNS is already a subset of
    ALREADY_TRIED, but future-proof if that ever stops being true).

    Each generator is only pulled from lazily via itertools.islice, so
    even a multi-hundred-million-candidate pattern like
    two_word_substitution_variants costs nothing extra here -- it's
    never fully enumerated, just asked for its first n_per_pattern
    items."""
    if patterns is None:
        patterns = ALREADY_TRIED

    with open(path, "w", encoding="utf-8") as f:
        for fn in patterns:
            f.write(f"## {fn.__name__}\n")
            n_written = 0
            for cand in itertools.islice(fn(), n_per_pattern):
                f.write(cand + "\n")
                n_written += 1
            if n_written == 0:
                f.write("(no candidates produced)\n")
            f.write("\n")

    print(f"Wrote exemplars for {len(patterns)} pattern categor"
          f"{'y' if len(patterns) == 1 else 'ies'} (up to {n_per_pattern} each) to {path!r}.")


def main():

    print_interval = 5  # seconds between progress prints
    count = 0
    total = sum(pattern_count(fn) for fn in MUTATION_PATTERNS)
    start = time.perf_counter()
    last_print = start
    print(f'Total candidates: {total}')

    for password in all_candidates():
        result = check_password_online(password)
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
            #log_job(record)
            exit(0)

    print('FAILURE!')
       #record = {"script": scriptname, "description": jobdescription, "count": total, "result": f'Failure'}
        #log_job(record)
        #body = json.dumps(record, indent=2)
        #send_notification('Hashing Failure!', body)

    


if __name__ == '__main__':
    if "--exemplars" in sys.argv:
        write_exemplars()
    else:
        main()
