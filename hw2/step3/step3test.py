#!/usr/bin/env python
"""
step3test.py - exercises the crypto in encrypt.py without modifying it.
Usage: python step3test.py <password> <plaintext_file> [output_file]
"""
import sys
import pyaes
sys.path.insert(0, '.')   # adjust if encrypt.py lives elsewhere
import encrypt            # the provided file, completely unmodified

password = sys.argv[1]
with open(sys.argv[2]) as fp:
    plaintext = fp.read()
target = sys.argv[3] if len(sys.argv) > 3 else None

# 1. encrypt() is not buggy -- call it directly, as-is. This is the same
#    function encrypt.py's own __main__ block calls to build a .crypt file.
ciphertext = encrypt.encrypt(password, plaintext)
print("encrypt() succeeded, ciphertext length:", len(ciphertext))

# 2. Verify the AES round-trip itself is sound, bypassing only the buggy
#    tag-check inside decrypt(). This reuses derive_key() and pyaes directly
#    -- both are correct as given, so nothing here requires editing the file.
key = encrypt.derive_key(password)
raw_ciphertext = ciphertext[:-32]          # strip the 32-byte tag, same as decrypt() does
aes = pyaes.AESModeOfOperationCTR(key)     # fresh instance, same as decrypt() does
decrypted = aes.decrypt(raw_ciphertext)

if decrypted.decode("ascii") == plaintext:
    print("AES round-trip OK: decrypted output matches original plaintext")
else:
    print("MISMATCH:", repr(decrypted), "!=", repr(plaintext))

# 3. Confirm the provided decrypt() throws regardless of password correctness
#    -- documents the bug without editing anything.
for label, pw in [("correct password", password), ("wrong password", password + "_wrong")]:
    try:
        encrypt.decrypt(pw, ciphertext)
        print(f"{label}: decrypt() succeeded (unexpected)")
    except Exception as e:
        print(f"{label}: decrypt() raised -> {e}")

# 4. Optionally write the ciphertext out, the same way encrypt.py's own
#    __main__ block does at the end:
#        with open(target, 'wb') as fp: fp.write(ciphertext)
#    This gives you a real .crypt file, built with the *unmodified* encrypt()
#    function, that you can compare against secret.html.crypt, feed into
#    wsgi.py, or run more experiments against.
if target:
    with open(target, 'wb') as fp:
        fp.write(ciphertext)
    print(f"wrote {len(ciphertext)} bytes to {target}")
else:
    print("(no output file given -- pass a 3rd argument to write one, e.g.:")
    print("   python step3test.py <password> <plaintext_file> <output_file>)")
