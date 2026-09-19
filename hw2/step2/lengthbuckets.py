# Helpers for loading the per-length wordlist files produced by
# splitbylength.py (all_len1.txt, all_len2.txt, ...) without having to
# name each file by hand.

import glob
import os
import re

_PATTERN_TEMPLATE = r'^{prefix}(\d+)\.txt$'


def _matching_files(directory, prefix, min_len, max_len):
    pattern = re.compile(_PATTERN_TEMPLATE.format(prefix=re.escape(prefix)))
    matches = []
    for path in glob.glob(os.path.join(directory, f'{prefix}*.txt')):
        m = pattern.match(os.path.basename(path))
        if not m:
            continue
        length = int(m.group(1))
        if min_len is not None and length < min_len:
            continue
        if max_len is not None and length > max_len:
            continue
        matches.append((length, path))
    matches.sort()  # numeric by length, not alphabetical by filename
    return matches


def iter_length_buckets(directory='dictionaries', prefix='all_len', min_len=None, max_len=None):
    """Lazily yields every word from all matching all_lenN.txt files,
    in increasing order of N. Missing lengths (no words of that length)
    are skipped automatically."""
    for length, path in _matching_files(directory, prefix, min_len, max_len):
        with open(path, 'r') as f:
            yield from f.read().splitlines()


def count_length_buckets(directory='dictionaries', prefix='all_len', min_len=None, max_len=None):
    """Total word count across all matching files, for use in a
    progress/ETA total. Cheap: just counts lines, doesn't keep them."""
    total = 0
    for _, path in _matching_files(directory, prefix, min_len, max_len):
        with open(path, 'r') as f:
            total += sum(1 for _ in f)
    return total
