
import itertools
import string



OLDPASSWORD = "aw2246french44pledgeor"
OLDMUTATED = "4w2246french44pledge0r"
OLDSPANISH = "aw2246spanish44pledgeor"
OLDGREEK = "aww2246greek44pledgeor"
SHORT = "aw2246french44"

WORD1 = "aw"
NUMBER1 = "2246"
WORD2 = "french"
NUMBER2 = "44"
WORD3 = "pledge"
WORD4 = "or"

MUTABLEWORDS = [WORD1, WORD2, WORD3, WORD4]
MUTABLEWORDS2 = ["@w", "french", "pledge", "0r"]
MUTABLEWORDS3 = [WORD1, "greek", WORD3, WORD4]


LEETMAP = {'a': ['4', '@'], 'e': ['3'], 'i': ['1', '!'], 'o': ['0'], 's': ['5', '$'], 't': ['7']}

def leet_variants(s):
    yield s  # unchanged baseline

    # a small number of consistent full-substitution "themes"
    for theme_idx in range(2):  # 0 = prefer first option, 1 = prefer second if it exists
        yield "".join(
            LEETMAP[c][theme_idx] if c in LEETMAP and theme_idx < len(LEETMAP[c]) else LEETMAP.get(c, [c])[0]
            for c in s
        )

    # one letter-type fully substituted at a time, trying each of its options
    for letter, subs in LEETMAP.items():
        for sub in subs:
            yield s.replace(letter, sub)


def case_variants(word):
    yield word.lower()
    yield word.upper()
    yield word.capitalize()
    yield word[:-1] + word[-1].upper()




WORDVARIANTS = [case_variants(word) for word in MUTABLEWORDS]
LEETVARIANTS = [case_variants(word) for word in MUTABLEWORDS2]
SPANISHVARIANTS = [case_variants(word) for word in ["aw", "spanish", "pledge", "or"]]    

SHORTERVARS = [case_variants(word) for word in ["aw", "french"]]


candidates = []

for aw, fr, pl, or_ in itertools.product(*WORDVARIANTS):
    candidate = aw + NUMBER1 + fr + NUMBER2 + pl + or_
    candidates.append(candidate)

for aw, fr, pl, or_ in itertools.product(*LEETVARIANTS):
    candidate = aw + NUMBER1 + fr + NUMBER2 + pl + or_
    candidates.append(candidate)

for aw, sp, pl, or_ in itertools.product(*SPANISHVARIANTS):
    candidate = aw + NUMBER1 + sp + NUMBER2 + pl + or_
    candidates.append(candidate)

for aw, fr in itertools.product(*SHORTERVARS):
    candidate = aw + NUMBER1 + fr + NUMBER2
    candidates.append(candidate)



for punct in string.punctuation:
    candidates.append(OLDPASSWORD + punct)
    candidates.append(punct + OLDPASSWORD)
    candidates.append(OLDMUTATED + punct)
    candidates.append(punct + OLDMUTATED)
    candidates.append(OLDSPANISH + punct)
    candidates.append(punct + OLDSPANISH)
    candidates.append(SHORT + punct)
    candidates.append(punct + SHORT)


for digit in string.digits:
    candidates.append(OLDPASSWORD + digit)
    candidates.append(digit + OLDPASSWORD)
    candidates.append(OLDMUTATED + digit)
    candidates.append(digit + OLDMUTATED)
    candidates.append(OLDSPANISH + digit)
    candidates.append(digit + OLDSPANISH)
    candidates.append(SHORT + digit)
    candidates.append(digit + SHORT)

# Add in leet variants of the old password, mutated password, and Spanish password
candidates.extend(leet_variants(OLDPASSWORD))
candidates.extend(leet_variants(OLDMUTATED))
candidates.extend(leet_variants(OLDSPANISH))

# Reversal tests
candidates.append(WORD1[::-1] + NUMBER1 + WORD2       + NUMBER2 + WORD3       + WORD4)
candidates.append(WORD1       + NUMBER1 + WORD2[::-1] + NUMBER2 + WORD3       + WORD4)
candidates.append(WORD1       + NUMBER1 + WORD2       + NUMBER2 + WORD3[::-1] + WORD4)
candidates.append(WORD1       + NUMBER1 + WORD2       + NUMBER2 + WORD3       + WORD4[::-1])
candidates.append(OLDPASSWORD[::-1])
candidates.append(OLDMUTATED[::-1])
candidates.append(OLDSPANISH[::-1])
candidates.append(SHORT[::-1])
candidates.append(WORD1[::-1] + NUMBER1 + WORD2[::-1] + NUMBER2 + WORD3[::-1] + WORD4[::-1])
candidates.append(WORD1[::-1] + NUMBER1 + WORD2[::-1] + NUMBER2 + "pledgeor")
candidates.append(WORD1[::-1] + NUMBER1 + "french44" + WORD3[::-1] + "pledgeor"[::-1])



print(f"Writing {len(candidates)} candidates to candidates.txt...")
with open("candidates.txt", "w") as f:
    for candidate in candidates:
        f.write(candidate + "\n")


