#!/usr/bin/env python3
"""
build_bob_dictionary.py

Builds a small, prioritized candidate word list ("bob.txt") for the
concatenation-attack search, targeting the specific persona described in
the assignment: "Bob," a fictional identity thief with a fake bank
terminal, going after three "dragon-related" targets (Arya Stark, Jon
Snow, Tyrion Lannister). The professor's hint is that the password is
three words. The existing Zipf-based sort already covers "words a random
English speaker would plausibly use" -- this instead targets "words THIS
persona would plausibly use," across several intersecting themes drawn
from what's known about Bob:

  1. Game of Thrones lore -- the houses/dragons connected to his targets.
  2. Heist / bank-robbery / confidence-trick vocabulary -- his own MO.
  3. Pirate / buried-treasure / shark vocabulary -- Bob had a file linking
     to "Sharkman Frank," a kids' pirate-treasure-hunt novelty song.
  4. Radio / audio-hardware vocabulary -- Bob's file also referenced an
     Intel audio card, NYC public radio, and "StreamGuys" (a real
     internet-radio streaming provider), suggesting an internet-radio /
     audio-hardware angle.

METHODOLOGY
-----------
Vocabulary for each theme was gathered by extracting the thematic,
non-proper-noun vocabulary from a handful of reference articles (full
source list in sources.txt) -- lore pages from A Wiki of Ice and Fire for
the GoT theme, true-crime/heist Wikipedia articles for the heist theme,
and piracy/treasure/shark and broadcasting/sound-card Wikipedia articles
for the pirate and radio themes, respectively. We deliberately did NOT
pull the actual lyrics of the "Sharkman Frank" song itself -- that's
copyrighted creative content, not something to scrape -- and instead used
the genre it belongs to (pirate/treasure-hunt novelty music) as a stand-in
theme, the same way the heist theme was built from a general "heist film"
article rather than any specific film's script.

Those raw extracted terms live in the corpus_*.txt files alongside this
script, with duplicates preserved deliberately: a word that showed up in
several different source articles is more likely to be load-bearing
vocabulary for its theme than a one-off mention.

This script:
  1. Loads every corpus_*.txt file present alongside this script, plus a
     small hand-curated set of obvious terms the source articles didn't
     happen to use verbatim (e.g. "terminal", "atm", "pin", "intel",
     "streamguys" itself isn't a real word so its parts -- "stream" and
     "guys" -- are curated in instead) -- kept separate and clearly
     labeled below, so the methodology stays honest about what was
     sourced vs. assumed.
  2. Tokenizes everything (splits on non-letters, so multi-word phrases
     and hyphenated compounds like "getaway car" or "safe-cracking"
     contribute their component words too), lowercases, and counts how
     many times each word occurs, and in how many distinct themes.
  3. Filters to ONLY words that are actually present in all.txt, since
     that's the dictionary the assignment says to concentrate on -- a
     thematically perfect word is useless here if it isn't a crackable
     candidate.
  4. Ranks the survivors: words that appear across MORE distinct themes
     rank first (a word tying together e.g. the GoT and heist themes is
     a stronger signal than one that only showed up once in one
     article), then by total mention count, then by Zipf frequency as a
     final tiebreaker (ordinary/memorable words over obscure ones).
  5. Writes the top --limit (default 1000) words to bob.txt, one per
     line, in that priority order -- ready to be dropped in as a new
     pool in the same priority-sorted search the rest of the assignment
     already uses.

Usage:
    python3 build_bob_dictionary.py [--limit 1000] [--all-txt all.txt]
                                     [--out bob.txt]
"""
import argparse
import glob
import os
import re
from collections import Counter, defaultdict

from wordfreq import zipf_frequency

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Hand-curated core terms not reliably present in the source articles,
# but clearly relevant given specific details known about Bob. Kept
# separate from the "sourced" vocabulary above rather than silently
# folded in, and tagged with which theme prompted each one.
HAND_CURATED_TERMS = {
    # Bob ran a fake BANK TERMINAL to steal money.
    "heist": [
        "terminal", "atm", "pin", "login", "username", "hacker", "hack",
        "hacking", "breach", "exploit", "backdoor", "transfer", "withdraw",
        "withdrawal", "deposit", "wire", "fund", "funds", "ledger",
        "account", "balance", "statement", "digit", "code", "cipher",
        "encrypt", "encrypted", "decrypt", "spoof", "spoofing", "identity",
        "impersonate", "forge", "forged", "counterfeit", "launder",
        "laundering",
    ],
    # "Sharkman Frank" the artist name -- "shark" is already covered by
    # the pirate corpus, but "frank" (also an ordinary English word) and
    # "man" are worth including directly.
    "pirate": ["frank", "man"],
    # A file of Bob's referenced an Intel audio card, NYC public radio,
    # and "StreamGuys" (a real internet-radio streaming company).
    # "streamguys" itself isn't a dictionary word, so its components are
    # curated in individually.
    "radio": ["intel", "stream", "streaming", "guys", "podcast", "nyc"],
}

WORD_RE = re.compile(r"[a-zA-Z]+")


def load_term_counts(path):
    """Tokenize a raw extracted-vocabulary file into a word -> count Counter."""
    if not os.path.exists(path):
        return Counter()
    with open(path, "r") as f:
        text = f.read()
    return Counter(w.lower() for w in WORD_RE.findall(text))


def discover_themes(script_dir):
    """
    corpus_<theme>.txt -> {theme: Counter}, for every corpus file present.
    A file named corpus_<theme>_<anything>.txt (e.g. corpus_got_extra.txt,
    used to widen a theme's vocabulary later without touching the
    original corpus_got.txt / any list already generated from it) is
    folded into the SAME <theme> bucket -- the theme name is just the
    first underscore-delimited segment after "corpus_".
    """
    themes = defaultdict(Counter)
    for path in sorted(glob.glob(os.path.join(script_dir, "corpus_*.txt"))):
        name = os.path.basename(path)[len("corpus_"):-len(".txt")]
        theme = name.split("_")[0]
        themes[theme] += load_term_counts(path)
    return dict(themes)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=1000,
                         help="max words to write to bob.txt (default 1000)")
    parser.add_argument("--all-txt", default=os.path.join(SCRIPT_DIR, "all.txt"))
    parser.add_argument("--out", default=os.path.join(SCRIPT_DIR, "bob.txt"))
    args = parser.parse_args()

    with open(args.all_txt, "r") as f:
        all_words = {w.strip().lower() for w in f if w.strip()}
    print(f"Loaded {len(all_words):,} candidate words from {args.all_txt}")

    themes = discover_themes(SCRIPT_DIR)
    for theme, extra_terms in HAND_CURATED_TERMS.items():
        themes.setdefault(theme, Counter())
        for term in extra_terms:
            themes[theme][term.lower()] += 1

    for theme, counts in themes.items():
        print(f"  theme '{theme}': {len(counts)} unique words, "
              f"{sum(counts.values())} total mentions")

    # word -> total mentions across all themes; word -> set of themes it appears in
    combined_counts = Counter()
    theme_membership = defaultdict(set)
    for theme, counts in themes.items():
        for w, c in counts.items():
            combined_counts[w] += c
            theme_membership[w].add(theme)

    candidates = set(combined_counts) & all_words
    dropped = len(combined_counts) - len(candidates)
    print(f"\n{len(candidates)} thematic words are present in all.txt "
          f"({dropped} thematic words were NOT found in all.txt and were dropped)")

    def sort_key(w):
        return (-len(theme_membership[w]), -combined_counts[w], -zipf_frequency(w, "en"))

    ranked = sorted(candidates, key=sort_key)
    final = ranked[: args.limit]

    with open(args.out, "w") as f:
        for w in final:
            f.write(w + "\n")

    multi_theme = sum(1 for w in final if len(theme_membership[w]) > 1)
    print(f"\nWrote {len(final)} words to {args.out}")
    print(f"  ({multi_theme} of them showed up under MORE THAN ONE theme)")
    if len(candidates) > args.limit:
        print(f"  ({len(candidates) - args.limit} additional thematic matches were "
              f"cut off by --limit {args.limit})")

    print("\nTop 30 (by priority):")
    for w in final[:30]:
        th = ",".join(sorted(theme_membership[w]))
        print(f"  {w:15s} themes={th:20s} mentions={combined_counts[w]:2d}  zipf={zipf_frequency(w, 'en'):.2f}")


if __name__ == "__main__":
    main()
