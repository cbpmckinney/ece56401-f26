import itertools
import sys

def main():
    # Load your candidate list safely
    try:
        with open("seclist_pool.txt", "r", encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Error: candidates.txt not found.", file=sys.stderr)
        return

    # Generate all triplets (allowing repetitions)
    # repeat=3 creates the n^3 search space
    for triplet in itertools.product(words, repeat=3):
        # Join the tuple of strings into a single string and output it
        sys.stdout.write("".join(triplet) + "\n")

if __name__ == "__main__":
    main()
