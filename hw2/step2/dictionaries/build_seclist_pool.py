#!/usr/bin/env python3
"""
build_seclist_pool.py

Builds a new, small, HIGH-priority candidate pool for the concatenation
search, ranked by real-world password popularity instead of English-word
frequency (Zipf) or thematic relevance (the bob.txt/bob2.txt approach).

Background: the professor's hint ("word triplets... think of 'top' not
necessarily as top in the wordlist, as it's not sorted by popularity")
turns out to describe all.txt itself pretty literally. Its first 2,812
lines (== common.txt) are a classic common-password list, but that block
is NOT actually sorted by real popularity -- e.g. "passwd" sits at
position 4, ahead of "123456" at position 5, even though every published
frequency study puts "123456" enormously ahead of "passwd". So even the
highest-priority pool in the existing search isn't really being walked in
popularity order.

seclist.txt fixes that: it's a real popularity-RANKED password list
(rank = line number, most common first), sourced from SecLists'
10k-most-common.txt (see seclist_source.txt for details and its
provenance/limitations -- notably it's currently only ~130 entries long,
not the full 10,000, due to a retrieval limitation; longer cuts can be
dropped in later and reprocessed with no script changes).

METHODOLOGY
-----------
1. Load seclist.txt IN ORDER -- unlike the corpus_*.txt files used for
   bob.txt (which are unordered bags of words scored by mention count),
   order here IS the signal: line 1 is the single most commonly observed
   real password, and so on.
2. Load all.txt (the actual candidate universe).
3. Compute the set of words ALREADY EXHAUSTED by the existing search:
   common.txt (pool 0) plus every word in the zipf>=4 bands (pools 1-4)
   -- i.e. every word whose all-same-word triple (and every triple drawn
   only from pools 0-4) has already been checked. This is recomputed
   here from the same source files/logic as gpu_nvidia_cuda.py's main(),
   rather than hardcoded, so it stays correct if those files change.
4. Filter seclist.txt down to: (a) present in all.txt, (b) NOT already
   exhausted. This is the "dedup by construction" approach discussed --
   rather than a runtime check against a candidate log, any pool built
   from words already excluded here cannot regenerate a candidate string
   the zipf>=4 search has already tried, no matter how positions combine.
5. Write the survivors, in their original rank order, to
   seclist_pool.txt -- ready to be dropped in as a new highest-priority
   pool (e.g. pool label 8) ahead of everything except pool 0 itself.

Usage:
    python3 build_seclist_pool.py [--seclist seclist.txt] [--all-txt all.txt]
                                   [--out seclist_pool.txt] [--limit 5000]

--limit caps the size of the final pool (default 5000), applied AFTER the
all.txt-membership and already-exhausted filters, keeping the top-ranked
(most popular) survivors and discarding the rest. Pass --limit 0 to keep
the whole filtered pool uncapped.
"""
import argparse
import os

from wordfreq import zipf_frequency

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_already_exhausted(script_dir):
    """
    Recomputes the exact word set covered by pools 0-4 in
    gpu_nvidia_cuda.py's main(): common.txt (pool 0) plus every uncommon
    word with zipf frequency >= 4 (pools 1-4, zipf bands 7.x down to 4.x).
    """
    with open(os.path.join(script_dir, "common.txt")) as f:
        common = {w.strip().lower() for w in f if w.strip()}

    with open(os.path.join(script_dir, "uncommon_by_zipf.txt")) as f:
        uncommon = [w.strip() for w in f if w.strip()]

    zipf4plus = {w.lower() for w in uncommon if zipf_frequency(w.lower(), "en") >= 4}

    return common | zipf4plus


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seclist", default=os.path.join(SCRIPT_DIR, "seclist.txt"))
    parser.add_argument("--all-txt", default=os.path.join(SCRIPT_DIR, "all.txt"))
    parser.add_argument("--out", default=os.path.join(SCRIPT_DIR, "seclist_pool.txt"))
    parser.add_argument("--limit", type=int, default=5000,
                         help="Cap on final pool size, applied after filtering, keeping the "
                              "highest-ranked (most popular) survivors. 0 = no cap. Default 5000.")
    args = parser.parse_args()

    with open(args.seclist) as f:
        ranked = [w.strip().lower() for w in f if w.strip()]
    print(f"Loaded {len(ranked)} ranked entries from {args.seclist} (rank order preserved)")

    with open(args.all_txt) as f:
        all_words = {w.strip().lower() for w in f if w.strip()}
    print(f"Loaded {len(all_words):,} candidate words from {args.all_txt}")

    exhausted = load_already_exhausted(SCRIPT_DIR)
    print(f"{len(exhausted):,} words already exhausted by the zipf>=4 + common search "
          f"(pool 0 + pools 1-4)")

    seen = set()
    pool = []
    not_in_all_txt = []
    already_exhausted_hits = []
    for w in ranked:
        if w in seen:
            continue
        seen.add(w)
        if w not in all_words:
            not_in_all_txt.append(w)
            continue
        if w in exhausted:
            already_exhausted_hits.append(w)
            continue
        pool.append(w)

    kept_before_cap = len(pool)
    if args.limit and len(pool) > args.limit:
        pool = pool[:args.limit]

    with open(args.out, "w") as f:
        for w in pool:
            f.write(w + "\n")

    print(f"\n{kept_before_cap} words survived the all.txt/already-exhausted filters")
    if args.limit and kept_before_cap > args.limit:
        print(f"  capped to --limit={args.limit}: kept the top {len(pool)} by popularity rank, "
              f"dropped the bottom {kept_before_cap - args.limit}")
    print(f"{len(pool)} words written -> {args.out}")
    print(f"  {len(already_exhausted_hits)} were dropped: already covered by the existing "
          f"zipf>=4/common search -- {already_exhausted_hits[:50]}"
          f"{' ...' if len(already_exhausted_hits) > 50 else ''}")
    print(f"  {len(not_in_all_txt)} were dropped: not present in all.txt at all -- {not_in_all_txt[:50]}"
          f"{' ...' if len(not_in_all_txt) > 50 else ''}")

    preview_n = min(len(pool), 100)
    print(f"\nFirst {preview_n} of {len(pool)} kept words, in real-world popularity rank order:")
    for i, w in enumerate(pool[:preview_n], 1):
        print(f"  {i:3d}. {w}")
    if len(pool) > preview_n:
        print(f"  ... ({len(pool) - preview_n} more written to {args.out})")


if __name__ == "__main__":
    main()
