with open('dictionaries/all.txt', 'r') as f:
    words = f.read().splitlines()

print(words)

sorted = sorted(words, key=len)

print(sorted)

with open('dictionaries/sorted.txt', 'w') as f:
    for word in sorted:
        f.write(word + '\n')

