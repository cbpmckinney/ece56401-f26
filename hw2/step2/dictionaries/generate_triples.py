#!/usr/bin/env python3
"""
generate_triples.py

Prints every three-word concatenation candidate drawn from
seclist_pool.txt (all three positions drawn from that same list) to
stdout, one candidate per line -- meant to be piped straight into John
the Ripper's stdin-candidate mode, e.g.:

    python3 generate_triples.py | john --format=crypt --stdin=candidates target.txt

WHY seclist_pool.txt SPECIFICALLY
----------------------------------
seclist_pool.txt (built by build_seclist_pool.py) is already:
  - filtered down to words that are actually present in all.txt (the
    professor's candidate universe), and
  - filtered to EXCLUDE every word already exhausted by the
    zipf>=4 + common GPU search (pool 0 + pools 1-4 in
    gpu_nvidia_cuda*.py). "Dedup by construction": since every word
    this script draws from already excludes exhausted words, no
    candidate it emits can be one the zipf>=4/common search has
    already tried, regardless of which of the three slots it lands in.
  - capped to a manageable size (--limit in build_seclist_pool.py,
    default 5000) and ordered by real-world password popularity rank
    (most popular first).

Because seclist_pool.txt is rank-ordered, the plain nested-loop order
below (a outermost, b middle, c innermost) front-loads the single
highest-ranked word into the outer position first, then works through
progressively lower-ranked combinations -- the same convention already
used elsewhere in this codebase (e.g. crackconcat.py's
itertools.product(common, repeat=3)), just applied to this pool
instead. It is NOT a full joint re-sort by combined triple-likelihood;
see the --pool-a/-b/-c options below if you ever want asymmetric pools
per position instead of the same list in all three slots.

Everything here is written to stdout ONLY as bare candidate strings
(one per line, no numbering/prefix/blank lines) so it's safe to pipe
directly into `john --stdin`. All progress/diagnostic output goes to
stderr instead, so it can never corrupt the candidate stream.

CAUTION ON SCALE: with N words in the pool, this generates N^3
candidates. seclist_pool.txt is capped at 5000 by default, but 5000^3
is far too many to ever finish -- in practice check len(pool) first
(the script prints it to stderr immediately) and, if it's more than a
few hundred entries, use --max-len/--min-len to prune, or point --pool
at a smaller cut, rather than assuming the full cross product is
tractable on CPU/John.

Usage:
    python3 generate_triples.py [--pool seclist_pool.txt]
                                 [--pool-a A.txt --pool-b B.txt --pool-c C.txt]
                                 [--min-len N] [--max-len N]
                                 [--progress-every 10000000]
                                 [--dry-run]

    # straight into John:
    python3 generate_triples.py | john --format=crypt --stdin=candidates target.txt

    # or capture to a file first, if you'd rather inspect/wc -l it:
    python3 generate_triples.py > triples_seclist.txt
"""
import argparse
import itertools
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_POOL = os.path.join(SCRIPT_DIR, "seclist_pool.txt")


def load_pool(path):
    with open(path) as f:
        return [w.strip() for w in f if w.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", default=DEFAULT_POOL,
                         help="Wordlist to draw all three positions from when --pool-a/-b/-c "
                              "are not given (default: seclist_pool.txt)")
    parser.add_argument("--pool-a", default=None, help="Wordlist for position 1 (overrides --pool)")
    parser.add_argument("--pool-b", default=None, help="Wordlist for position 2 (overrides --pool)")
    parser.add_argument("--pool-c", default=None, help="Wordlist for position 3 (overrides --pool)")
    parser.add_argument("--min-len", type=int, default=None,
                         help="Skip candidates shorter than this many characters")
    parser.add_argument("--max-len", type=int, default=None,
                         help="Skip candidates longer than this many characters")
    parser.add_argument("--progress-every", type=int, default=10_000_000,
                         help="Print a progress line to stderr every N candidates considered "
                              "(default 10,000,000; 0 disables progress reporting)")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the pool size(s) and total candidate count to stderr, "
                              "then exit without generating anything")
    args = parser.parse_args()

    pool_a = load_pool(args.pool_a or args.pool)
    pool_b = load_pool(args.pool_b or args.pool)
    pool_c = load_pool(args.pool_c or args.pool)

    total = len(pool_a) * len(pool_b) * len(pool_c)
    print(f"[generate_triples] pools: {len(pool_a)} x {len(pool_b)} x {len(pool_c)} "
          f"-> {total:,} candidate triples", file=sys.stderr)

    if args.dry_run:
        print("[generate_triples] --dry-run given, exiting without generating", file=sys.stderr)
        return

    if total == 0:
        print("[generate_triples] a pool is empty, nothing to emit", file=sys.stderr)
        return

    out = sys.stdout
    count = 0
    emitted = 0
    for a, b, c in itertools.product(pool_a, pool_b, pool_c):
        count += 1
        candidate = a + b + c

        if args.min_len is not None and len(candidate) < args.min_len:
            pass
        elif args.max_len is not None and len(candidate) > args.max_len:
            pass
        else:
            out.write(candidate)
            out.write("\n")
            emitted += 1

        if args.progress_every and count % args.progress_every == 0:
            print(f"[generate_triples] {count:,}/{total:,} considered "
                  f"({emitted:,} emitted after length filtering)", file=sys.stderr)

    out.flush()
    print(f"[generate_triples] done: {count:,}/{total:,} considered, "
          f"{emitted:,} written to stdout", file=sys.stderr)


if __name__ == "__main__":
    main()
