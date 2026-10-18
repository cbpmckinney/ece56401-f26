from flask import Flask, render_template, request, abort
from encrypt import decrypt

app = Flask(__name__)

@app.route("/")
def hello_world():
    response = render_template("index.html")
    return response

@app.route("/decrypt", methods=["POST"])
def decrypt_view():
    with open("templates/secret.html.crypt", "rb") as fp:
        ciphertext = fp.read()

    password = request.form['password']
    
    try:
        plaintext = decrypt(password, ciphertext)
    except Exception as e:
        abort(403)
    return plaintext


