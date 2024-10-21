from collections import defaultdict
from typing import Any, Literal

import json
import logging
import os

from cs50 import SQL
from flask import Flask, flash, redirect, render_template, request, session
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash

from helpers import apology, login_required, lookup, usd

logger = logging.getLogger(__name__)

# Configure application
app = Flask(__name__)

# Custom filter
app.jinja_env.filters["usd"] = usd

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Configure CS50 Library to use SQLite database
db = SQL("sqlite:///finance.db")


@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


def user_exists(username: str) -> bool:
    db_result = db.execute(
        "SELECT COUNT(*) as count from users where username = ?", username
    )
    count = db_result[0]["count"]
    assert isinstance(count, int)
    return count > 0


def get_user(id: int) -> dict[str | Any] | None:
    db_result = db.execute("SELECT * FROM users where id = ?", id)
    if not db_result or not len(db_result):
        return None
    return db_result[0]


def insert_tx(
    user_id: int,
    tx_type: Literal["buy", "sell"],
    symbol: str,
    shares: int,
    price: float,
) -> bool:
    user = get_user(session["user_id"])
    assert user is not None
    balance = float(user["cash"])
    total = shares * price

    if balance < total:
        return False

    logger.debug("buying %s of %s at %s", shares, symbol, price)
    logger.debug("transaction amount: %s - user balance: %s", total, balance)

    db.execute(
        "INSERT INTO transactions (user_id, tx_type, symbol, shares, price) values (?, ?, ?, ?, ?)",
        user_id,
        tx_type,
        symbol,
        shares,
        price,
    )

    tx_amt = shares * price
    if tx_type == "buy":
        tx_amt = -tx_amt

    db.execute("update users set cash = cash + ?", tx_amt)
    return True


def stocks(id: int) -> list[dict[str, Any]]:
    stocks: dict[str, Any] = defaultdict(int)

    db_result = db.execute(
        """SELECT
            SUM(shares) as shares,
            upper(symbol) as symbol,
            tx_type,
            symbol
           FROM transactions where user_id = ? group by upper(symbol), tx_type order by upper(symbol)""",
        id,
    )
    for row in db_result:
        tx_type = row["tx_type"].lower()
        symbol = row["symbol"].upper()
        shares = row["shares"]

        if tx_type == "buy":
            stocks[symbol] += shares
        else:
            stocks[symbol] -= shares

    positions: list[dict[str, Any]] = []
    for symbol, shares in stocks.items():
        positions.append(
            {
                "symbol": symbol,
                "shares": shares,
                "price": lookup(symbol)["price"],
            }
        )
    logger.debug("positions: %s", positions)
    return positions


@app.route("/add", methods=["POST"])
def add():
    amount = request.form.get("amount")
    if amount is None:
        flash("invalid amount")
        return redirect("/")
    try:
        f_amount = float(amount)
    except ValueError:
        flash("invalid amount")
        return redirect("/")

    if f_amount < 0:
        flash("invalid amount")
        return redirect("/")

    db.execute(
        "update users set cash = cash + ? where id = ?",
        f_amount,
        session["user_id"],
    )
    return redirect("/")


@app.route("/")
@login_required
def index():
    """Show portfolio of stocks"""

    # retrieve all stocks bought
    # retrieve all sells
    # for each bought (add)
    # for each sell (subtract)

    # for each stock - get quote / total value
    # get cash balance
    positions = stocks(session["user_id"])

    user = get_user(session["user_id"])
    cash = float(user["cash"])
    stock_total = 0
    for position in positions:
        stock_total += position["shares"] * position["price"]

    portfolio = {
        "cash": cash,
        "stocks": stock_total,
    }

    return render_template(
        "index.html",
        stocks=positions,
        portfolio=portfolio,
    )


@app.route("/buy", methods=["GET", "POST"])
@login_required
def buy():
    """Buy shares of stock"""
    if request.method == "POST":
        symbol = request.form.get("symbol")

        if not symbol:
            flash("symbol required")
            return render_template("buy.html")

        quote = lookup(symbol)
        if not quote:
            flash("invalid symbol")
            return render_template("buy.html")

        shares = request.form.get("shares")
        if not shares or not shares.isdigit():
            flash("invalid shares")
            return render_template("buy.html")

        # validate purchase
        shares = int(shares)
        price = float(quote["price"])

        if not insert_tx(
            user_id=int(session["user_id"]),
            tx_type="buy",
            symbol=symbol.upper(),
            shares=shares,
            price=price,
        ):
            flash("not enough funds")
            return render_template("buy.html")
        return redirect("/")
    return render_template("buy.html")


@app.route("/history")
@login_required
def history():
    """Show history of transactions"""
    db_results = db.execute(
        "SELECT tx_type, UPPER(symbol) as symbol, shares, price, created_at FROM transactions WHERE user_id = ? order by created_at desc",
        session["user_id"],
    )
    logger.debug("results: %s", db_results)
    return render_template("history.html", txs=db_results)


@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":
        # Ensure username was submitted
        if not request.form.get("username"):
            return apology("must provide username", 403)

        # Ensure password was submitted
        elif not request.form.get("password"):
            return apology("must provide password", 403)

        # Query database for username
        rows = db.execute(
            "SELECT * FROM users WHERE username = ?", request.form.get("username")
        )

        # Ensure username exists and password is correct
        if len(rows) != 1 or not check_password_hash(
            rows[0]["hash"], request.form.get("password")
        ):
            return apology("invalid username and/or password", 403)

        # Remember which user has logged in
        session["user_id"] = rows[0]["id"]

        # Redirect user to home page
        return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")


@app.route("/logout")
def logout():
    """Log user out"""

    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/")


@app.route("/quote", methods=["GET", "POST"])
@login_required
def quote():
    """Get stock quote"""
    if request.method == "POST":
        symbol = request.form.get("symbol")
        if not symbol:
            flash("quote symbol required")
            render_template("quote.html")
        quote = lookup(symbol)
        if quote is None:
            flash("invalid symbol")
            return render_template("quote.html")

        return render_template("quoted.html", quote=quote)

    return render_template("quote.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""

    if request.method == "POST":
        # validation
        username = request.form.get("username")
        if not username:
            return apology("username required", 403)

        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        if not password:
            return apology("password required", 403)
        if password != confirm_password:
            return apology("passwords must match", 403)

        if user_exists(username):
            return apology("username already exists", 403)

        # create user
        hashed_password = generate_password_hash(password=password)
        logger.debug(
            db.execute(
                "INSERT INTO users (username, hash) VALUES(?, ?)",
                username,
                hashed_password,
            )
        )
        db_result = db.execute("SELECT id from users where username = ?", username)
        session["user_id"] = db_result[0]["id"]

        logger.debug("created user: %s: %s", session["user_id"], username)
        return redirect("/")
    else:
        return render_template("register.html")


@app.route("/sell", methods=["GET", "POST"])
@login_required
def sell():
    portfolio = stocks(session["user_id"])
    if request.method == "POST":
        symbol = request.form.get("symbol")
        shares = request.form.get("shares")
        if not shares.isdigit():
            flash("invalid shares")
            return render_template("sell.html", stocks=portfolio)

        shares = int(shares)
        if shares <= 0:
            flash("invalid shares")
            return render_template("sell.html", stocks=portfolio)

        if not symbol:
            flash("invalid symbol")
            return render_template("sell.html", stocks=portfolio)

        symbol = symbol.upper()

        stock_list = list(filter(lambda x: x["symbol"] == symbol, portfolio))
        if len(stock_list) == 0:
            flash("invalid symbol")
            return render_template("sell.html", stocks=portfolio)

        stock = stock_list[0]

        if stock["shares"] < shares:
            flash(f"you don't own {shares} shares")
            return render_template("sell.html", stocks=portfolio)

        quote = lookup(symbol)
        # add a transaction and increase cash
        logger.debug("selling stock: %s shares: %s", stock, shares)
        insert_tx(
            user_id=session["user_id"],
            tx_type="sell",
            symbol=symbol,
            shares=shares,
            price=quote["price"],
        )
        return redirect("/")
    return render_template("sell.html", stocks=portfolio)
