import os
import string
import time
import wordfreq

with open('dictionaries/common.txt', 'r') as fp:
    common = fp.read().splitlines()

with open('dictionaries/uncommon.txt', 'r') as fp:
    uncommon = fp.read().splitlines()


# Sort the uncommon list by how common each word actually is in English,
# most-likely-to-be-a-password first, and DROP words wordfreq doesn't
# recognize at all (zipf score 0) -- these are overwhelmingly archaic,
# technical, or taxonomic dictionary entries (see the sample check we ran:
# "zoocytium", "sarcosporidial", "annexionist", etc.) rather than anything
# a real person would plausibly choose as a password.
#
# word_frequency() and zipf_frequency() give IDENTICAL ordering --
# zipf_frequency(w) == log10(word_frequency(w) * 1e9) is a strictly
# monotonic transform -- zipf_frequency() is used here just because its
# ~0-8 scale is more readable and gives a natural threshold (score > 0)
# to filter on.

start = time.perf_counter()
scored = [(wordfreq.zipf_frequency(w.lower(), 'en'), w) for w in uncommon]
known = [(score, w) for score, w in scored if score > 0]
zero = [w for score, w in scored if score == 0]
uncommon_by_zipf = [w for score, w in sorted(known, key=lambda t: t[0], reverse=True)]
elapsed = time.perf_counter() - start

dropped = len(zero)
print(f"scored {len(uncommon)} words in {elapsed:.2f}s")
print(f"  kept    {len(uncommon_by_zipf)} ({100*len(uncommon_by_zipf)/len(uncommon):.1f}%) words with a nonzero zipf score")
print(f"  set aside {dropped} ({100*dropped/len(uncommon):.1f}%) words unknown to wordfreq")
max_score = max(score for score, w in known)
print(f"  most common word: {uncommon_by_zipf[0]!r} (zipf={max_score:.2f})")
print(f"  total candidate pool: {len(common)} common + {len(uncommon_by_zipf)} scored = {len(common) + len(uncommon_by_zipf)}")

with open('dictionaries/uncommon_by_zipf.txt', 'w') as fp:
    fp.write('\n'.join(uncommon_by_zipf) + '\n')

with open('dictionaries/uncommon_by_zipf0.txt', 'w') as fp:
    fp.write('\n'.join(zero) + '\n')

print("wrote dictionaries/uncommon_by_zipf.txt   (nonzero-score words, sorted most-to-least common)")
print("wrote dictionaries/uncommon_by_zipf0.txt  (zero-score / unknown-to-wordfreq words, original order)")
print("(dictionaries/common.txt is untouched -- keep loading it separately, ahead of this list, same as before)")

# Histogram: how many words fall in each integer zipf band (7.x, 6.x, ...).
# int(score) truncates toward zero, so e.g. 7.73 -> band 7, 3.38 -> band 3.
from collections import Counter
band_counts = Counter(int(score) for score, w in known)

print()
print("word count by zipf band (nonzero scores only):")
running_total = 0
for band in sorted(band_counts, reverse=True):
    running_total += band_counts[band]
    print(f"  {band}.x : {band_counts[band]:>7}   (cumulative >= {band}.0: {running_total:>7}, "
          f"+ common = {running_total + len(common):>7})")
