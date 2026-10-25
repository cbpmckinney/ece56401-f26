from flask import Flask, render_template, request, abort, make_response, redirect
from encrypt import decrypt
import uuid
import sqlite3
import os, base64

app = Flask(__name__)
db_handle = sqlite3.connect("file::memory:?cache=shared", uri=True)

def makepw():
    pw = os.urandom(8)
    pw = base64.b64encode(pw)
    return pw.decode("ascii")

def make_login():
    return str(uuid.uuid4())

def initdb():
    cur = db_handle.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS users(uid INTEGER PRIMARY KEY, user, password)")
    cur.execute("CREATE TABLE IF NOT EXISTS logins(uuid, uid, FOREIGN KEY(uid) REFERENCES user(uid))")

    pw = makepw()
    cur.execute("INSERT INTO users VALUES (NULL, 'admin', '{}')".format(pw))
    row = cur.execute("SELECT * FROM users WHERE user='admin'")
    login = make_login()
    uid = row.fetchone()[0]
    cur.execute("INSERT INTO logins VALUES ('{}', {})".format(login, uid))
    import pdb; pdb.set_trace()
    cur.close()

    db_handle.commit()

initdb()


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

@app.route("/admin")
def admin():

    new = False
    login_id = request.cookies.get("x-user-id", None)
    if not login_id:
        new = True
        login_id = str(uuid.uuid4())

    params = {
            "login_id": login_id,
            "new": new
    }

    response = make_response(render_template("admin.html", **params))
    response.set_cookie("x-user-id", login_id)
    return response

@app.route("/admin/login", methods=["POST"])
def login():
    db_handle = sqlite3.connect("file::memory:?cache=shared", uri=True)

    password = request.form['password']
    username = request.form['username']
    print("attempt = {}, {}".format(username, password))

    cur = db_handle.cursor()
    cur.execute("SELECT * FROM users WHERE user=? AND password=?", (username, password))
    retval = cur.fetchone()

    login_id = request.cookies.get("x-user-id", None)

    if retval:
        # we are logged in add an entry for the admin in the logins
        cur.execute("INSERT INTO logins VALUES (?, ?)", (login_id, retval[0]))

        response = make_response(redirect('/admin/panel'))
    else:
        response = make_response(redirect("/admin?passfail=True"))
    cur.close()
    db_handle.commit()

    return response

@app.route("/admin/reset", methods=["POST"])
def reset():
    response = make_response(redirect("/admin"))
    response.delete_cookie("x-user-id")
    return response


@app.route("/admin/panel")
def admin_panel():
    db_handle = sqlite3.connect("file::memory:?cache=shared", uri=True)
    login_id = request.cookies.get("x-user-id", None)

    cur = db_handle.cursor()
    # get uid of user that's logged in 
    print("SELECT * FROM logins WHERE uuid='{}'".format(login_id))
    cur.execute("SELECT * FROM logins WHERE uuid='{}'".format(login_id))
    identity = cur.fetchone()
    cur.close()

    if identity and identity[1] == 1:
        response = make_response(render_template("admin_panel.html"))
    else:
        response = make_response(redirect("/admin?passfail=True"))
    return response
