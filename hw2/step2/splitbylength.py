# Splits a wordlist into separate files by word length, e.g. all.txt ->
# all_len1.txt, all_len2.txt, all_len3.txt, ...
#
# This doesn't reduce the total number of words, but it lets you run a
# concatenation attack against just the short (plausible) suffixes first,
# without paying the cost of the whole dictionary, and lets you split
# different length buckets across different machines.

from collections import defaultdict

def main():
    input_file = 'dictionaries/all.txt'
    output_prefix = 'dictionaries/all_len'

    with open(input_file, 'r') as ip:
        words = ip.read().splitlines()

    buckets = defaultdict(list)
    for word in words:
        buckets[len(word)].append(word)

    print(f'{len(words)} words read from {input_file}')
    print(f'{"length":>6} {"count":>8}  output file')

    for length in sorted(buckets):
        outname = f'{output_prefix}{length}.txt'
        with open(outname, 'w') as op:
            op.write('\n'.join(buckets[length]) + '\n')
        print(f'{length:6d} {len(buckets[length]):8d}  {outname}')

if __name__ == '__main__':
    main()
