import functools
import itertools
import os
import sys
import time 
from datetime import timedelta
import smtplib
from email.message import EmailMessage
import string
import re
import requests


# GLOBAL DATA
url = "http://127.0.0.1:8080/decrypt"



def check_password_online(password):
    try:
        resp = requests.post(url, data={"password": password})
    except requests.RequestException:
        print("Error: Could not connect to the server. Please ensure the server is running and accessible.")
        return None

    if resp.status_code != 403:
        print(password)
        return resp.text

    return None


with open("candidates.txt", "r") as f:
    candidates = [line.strip() for line in f.readlines()]

with open("tried.txt", "r") as f:
    tried = set(line.strip() for line in f.readlines())

testcandidates = [c for c in candidates if c not in tried]




print(f"Testing {len(testcandidates)} candidates against the online password checker...")
num = 0
for candidate in testcandidates:
    time.sleep(0.1)
    result = check_password_online(candidate)
    num += 1
    with open("tried.txt", "a") as f:
        f.write(candidate + "\n")

    if result is not None:
        print(f"Password '{candidate}' is correct! Server response: {result}")
        exit(0)


print("No candidates matched the password.")
print(f"Writing {num} tried candidates to tried.txt.")
exit(0)