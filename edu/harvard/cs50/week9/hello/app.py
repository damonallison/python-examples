from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/greet", methods=["GET", "POST"])
def greet():
    # jinja templating
    if request.method == "GET":
        name = request.args.get("name") or "world"
    else:
        name = request.form.get("name") or "world"
    return render_template("greet.html", name=name)
