from flask import Flask, render_template, request, abort
from encrypt import decrypt

app = Flask(__name__)

@app.route("/")
def hello_world():
    response = render_template("index.html")
    return response

@app.route("/decrypt", methods=["POST"])
def decrypt_view():
    with open("templates/test2.crypt", "rb") as fp:
        ciphertext = fp.read()

    password = request.form['password']
    
    try:
        plaintext = decrypt(password, ciphertext)
    except Exception as e:
        print(f"Password given was {password}")
        abort(403)
    return plaintext


