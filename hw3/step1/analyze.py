from zxcvbn import zxcvbn

OLDPASSWORD = "aw2246french44pledgeor"
OLDMUTATED = "@w2246french44pledge0r"

results1 = zxcvbn(OLDPASSWORD)
results2 = zxcvbn(OLDMUTATED)

with open("results1.json", "w") as f:
    f.write(str(results1))

with open("results2.json", "w") as f:
    f.write(str(results2))

