#!/usr/bin/env python
"""
step3testdecode.py - decoding counterpart to step3test.py.

Mirrors exactly what wsgi.py's /decrypt route does: read a ciphertext file
and call the UNMODIFIED encrypt.decrypt(password, ciphertext), reporting
success or "access denied" the same way the server would (HTTP 403).

This deliberately does NOT patch or work around anything in encrypt.py --
the point is to test whether encrypt+decrypt works using the professor's
code exactly as given, the same way wsgi.py does it:

    try:
        plaintext = decrypt(password, ciphertext)
    except Exception as e:
        abort(403)
    return plaintext

Usage: python step3testdecode.py <password> <ciphertext_file>
"""
import sys
sys.path.insert(0, '.')   # adjust if encrypt.py lives elsewhere
import encrypt            # the provided file, completely unmodified

if len(sys.argv) != 3:
    print("Usage: python step3testdecode.py <password> <ciphertext_file>")
    sys.exit(2)

password = sys.argv[1]
with open(sys.argv[2], 'rb') as fp:
    ciphertext = fp.read()

print(f"decrypting {sys.argv[2]!r} ({len(ciphertext)} bytes) with password {password!r} ...")

# This block is a line-for-line mirror of wsgi.py's decrypt_view.
try:
    plaintext = encrypt.decrypt(password, ciphertext)
except Exception as e:
    print(f"DENIED (like HTTP 403) -- decrypt() raised: {e!r}")
    sys.exit(1)

print("SUCCESS -- decrypted plaintext:")
print(plaintext)
