"""
Week 4: Libraries (Modules / Packages)

Symbol resolution order (the types of namespaces)
* keywords (if, then, else, raise, for)
* built-ins (functions compiled into the interpreter)
* modules (built into python) - standard library
* packages (third party) - installable modules

Importing modules
-----------------
* "from" allows you to import particular functions. Typically import entire
  modules to retain the scope from which the module is imported from.
* "as" allows you to rename the imported symbols


Python environment
------------------
* sys.argv holds the program (sys.argv[0] and any other arguments the user adds
  (sys.argv[1:])
* sys.exit() terminates the program

Packaging
---------
* https://pypi.org

APIs
----
* requests (http)

"""

from typing import Any, cast

import emoji
import pyfiglet
import json
import random
import sys
import cowsay
import requests


def coin_flip() -> bool:
    return random.choice([True, False])

    # another implementation
    # random.randint(0, 9) < 5


def args() -> list[str]:
    # argv[0] is always the name of the module / file being ran.
    # return *only* the user entered args.
    return sys.argv[1:]


def cow_say_hello(name: str) -> None:
    """An example of using a 3rd party package.

    Requires the `cowsay` library installed via pip:

    $ pip install cowsay
    """
    cowsay.cow(f"Hello, {name}")


def call_itunes_api(artist: str) -> list[str]:
    base_url = "https://itunes.apple.com/search"
    params = {
        "term": artist,
        "media": "music",
        "limit": 5,
    }
    response = requests.get(base_url, params=params)
    response.raise_for_status()

    json_response = response.json()

    # "pretty printing" the JSON value for debugging
    # print(json.dumps(json_response, indent=2))

    songs: list[str] = []
    for song in json_response["results"]:
        song = cast(dict[str, Any], song)

        track_name = "unknown"
        if song.get("trackName") is not None:
            track_name = song.get("trackName")

        album_name = "unknown"
        if song.get("collectionName") is not None:
            album_name = song.get("collectionName")

        songs.append(f"{album_name}: {track_name}")
    return songs


#
# Exercises
#


def emojize(input: str) -> str:
    """Requires the `emoji` library

    $ pip install emoji
    """
    return emoji.emojize(input)


def fig_letter(input: str) -> str:
    """Requires the `pyfiglet` library

    $ pip install pyfiglet

    Expects zero or two command-line arguments:
    0: random font
    2: <--font | -f> <font-name>

    from pyriglet import Figlet
    figlet = Figlet()
    figlet.getFonts()  # return a list of fonts
    figlet.setFont(f)  # where f is a name of the font (str)
    """

    figlet = pyfiglet.Figlet()
    if len(sys.argv) == 1:
        figlet.renderText(input)
    if (
        len(sys.argv) == 3
        and sys.argv[1] in ["--font", "-f"]
        and sys.argv[2] in figlet.getFonts()
    ):
        figlet.setFont(font=sys.argv[2])
        print(figlet.renderText(input))
        return
    raise ValueError("invalid arguments")


def adieu(names: str = []) -> str:
    """Implement a program that prompts the user for names, one per line, until
    the user inputs control-d. Bid "adieu" to those names, separating two names
    with one and n names with n-1 commas and one and.

    Example:
    damon, allison
    Adieu, adieu, to damon and allison

    damon, ryan, allison
    Adieu, adieu, to damon, ryan and allison
    """

    if len(names) == 0:
        try:
            while True:
                names.append(input("Enter a name: "))
        except KeyboardInterrupt:
            pass

    name_str = ""
    if len(names) > 1:
        name_str = ", ".join(names[0:-1])
        name_str += " and "
    name_str += names[-1]

    return "Adieu, adieu, to " + name_str


def guessing_game(guess: int = 0) -> str:
    """Randomly generate an integer between 1 and 10. Prompt the user to guess.
    If the guess is smaller, print "too small". If the guess is larger, print
    "too large". If the guess is correct, print "you win" and exit the
    program. If the user types "exit", exit the program. If the user types
    anything else, print "invalid input" and prompt the user to guess again.
    """

    number = random.randint(1, 10)

    # Uncomment if you want to loop until you find a winner
    # while True:
    try:
        # guess = int(input("Enter a number between 1 and 10: "))
        if guess == number:
            return "you win"
        if guess < number:
            # print("too small")
            return "too small"
        if guess > number:
            # print("too large")
            return "too large"
    except ValueError:
        print("invalid input")
    except KeyboardInterrupt:
        pass
        # break


def little_professor() -> None:
    """
    * Prompt the user for a level. If not 1, 2, 3, prompt again.
    * Randomly generate 10 math problems formatted as X + Y = where X and Y are
      random integer with n digits.
    * Prompt the user to solve each problem. If the answer is not correct, the
      program should output EEE and prompt the user again, allowing up to 3
      tries for each problem. If the user hasn't answered correctly after 3
      tries, the program should output the correct answer.
    * The program should ultimately output the user's score: the number correct out of 10

    """

    def level_selection() -> int:
        while True:
            level = input("enter a level from 1-3: ")
            try:
                ilevel = int(level)
                if ilevel in [1, 2, 3]:
                    return ilevel
            except ValueError:
                continue

    def generate_integer(level: int) -> int:
        return random.randint(1 * 10 ** (level - 1), 9 * 10 ** (level - 1))

    def generate_problems(level: int) -> list[tuple[int, int, int]]:
        problems: list[tuple[int, int, int]] = []
        for _ in range(10):
            x = generate_integer(level)
            y = generate_integer(level)
            problems.append((x, y, x + y))
        return problems

    def solve_problem(x: int, y: int, answer: int) -> bool:
        for _ in range(3):
            guess = input(f"{x} + {y} = ")
            try:
                if int(guess) == answer:
                    return True
            except ValueError:
                pass
            print("EEE")
        return False

    level = level_selection()
    print(generate_problems(level))

    correct = 0
    for prob in generate_problems(level):
        if solve_problem(*prob):
            correct += 1

    print("You got", correct, "out of 10")


def bitcoin_price_index(count: float) -> str:
    """Returns the current USD value for the given amount of bitcoin.

    Example coindesk response
    -------------------------
    {
        "time": {
            "updated": "Apr 17, 2024 14:10:47 UTC",
            "updatedISO": "2024-04-17T14:10:47+00:00",
            "updateduk": "Apr 17, 2024 at 15:10 BST"
        },
        "disclaimer": "This data was produced from the CoinDesk Bitcoin Price Index (USD). Non-USD currency data converted using hourly conversion rate from openexchangerates.org",
        "chartName": "Bitcoin",
        "bpi": {
            "USD": {
                "code": "USD",
                "symbol": "&#36;",
                "rate": "62,280.927",
                "description": "United States Dollar",
                "rate_float": 62280.9267
            },
            "GBP": {
                "code": "GBP",
                "symbol": "&pound;",
                "rate": "50,023.418",
                "description": "British Pound Sterling",
                "rate_float": 50023.4175
            },
            "EUR": {
                "code": "EUR",
                "symbol": "&euro;",
                "rate": "58,563.254",
                "description": "Euro",
                "rate_float": 58563.2536
            }
        }
    }

    """

    def current_price() -> float:
        try:
            return requests.get(
                "https://api.coindesk.com/v1/bpi/currentprice.json"
            ).json()["bpi"]["USD"]["rate_float"]

        except requests.exceptions.RequestException as e:
            print("error: ", e)

    # format as currency with two decimal places
    return f"${(current_price() * count):,.2f}"
