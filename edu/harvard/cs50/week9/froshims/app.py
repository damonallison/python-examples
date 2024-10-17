from flask import Flask, render_template, request
from cs50 import SQL

app = Flask(__name__)

# todo: pull from db
SPORTS = ["basketball", "soccer", "ultimate frisbee"]

db = SQL("sqlite:///sports.db")


@app.route("/")
def index():
    return render_template("index.html", sports=SPORTS)


@app.route("/register", methods=["POST"])
def register():

    # __important__ - always validate user inputs server side, never trust user input
    name = request.form.get("name")
    if not name:
        return render_template("failure.html", message="name is missing")

    user_sports = request.form.getlist("sport")
    if not user_sports:
        return render_template("failure.html", message="sport is missing")

    for sport in user_sports:
        if sport not in SPORTS:
            return render_template(
                "failure.html", message=f"{sport} is not a valid sport"
            )

    # register user
    db.execute("DELETE FROM registrants where name = ?", name)
    for sport in user_sports:
        db.execute("INSERT INTO registrants (name, sport) VALUES (?, ?)", name, sport)

    print(db.execute("SELECT * FROM registrants"))

    return render_template("success.html")
