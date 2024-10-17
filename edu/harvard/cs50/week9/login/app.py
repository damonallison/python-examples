from flask import Flask, redirect, render_template, request, session
from flask_session import Session

import json

app = Flask(__name__)

# don't store session in the cookie itself, store it serverside
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


@app.route("/")
def index():
    return render_template("index.html", name=session.get("name"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        print(f"setting name {request.form.get('name')} for {session}")
        session["name"] = request.form.get("name")
        return redirect("/")

    return render_template("login.html")


@app.route("/logout", methods=["GET"])
def logout():
    session.clear()
    return redirect("/")


# normally you'd pull this from the database
SHOWS = ["Seinfeld", "The office", "Officer and a gentleman"]


@app.route("/search")
def search():
    q = (request.args.get("q") or "").lower()
    if not q:
        return ""
    elts = []
    for show in SHOWS:
        if q in show.lower():
            elts.append({"name": show})

    return json.dumps(elts)
