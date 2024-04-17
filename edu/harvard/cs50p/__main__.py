"""
Week 0: Introduction to programming in python

Main topics: data types, variables, functions

Environment
-----------
* purpose of programming: solving problems by moving data
* engineering mindset
* command line
* running vs. debugging
    * python file.py
    * debugger python file.py
* syntax vs. runtime bugs

Data types, variables, and functions
------------------------------------
* data types
    * https://docs.python.org/3/library/stdtypes.html
* strict vs dynamic typing
* strings 's' vs "s" vs \"""\"""
* variables (value vs. reference)
* function declaration
* arguments (calling) vs. parameters (definition)
* functions (declared function) vs. methods (object)
* passing variables (immutable types)

Program structure
-----------------
* objects
* object oriented vs. procedural vs. functional
* scoping
* modules
* built-ins vs packages

Logistics / environment
------------------------
* python --version
* github / git
* command line
* homebrew
* pip / poetry


Types:


* int
* float
* bool
* complex
* str

* sequence types
    * list
    * tuple (immutable)
    * range

* mapping types
    * set
    * dict


"""

from edu.harvard.cs50p import week0, week1, week2, week3, week4


def run_week0() -> None:
    # print(f"indoor voice: {week0.indoor_voice()}")
    # print(f"playback speed: {week0.playback_speed()}")
    # print(f"making faces: {week0.making_faces()}")
    # print(f"eninsten: {week0.einstein()}")
    print(f"tip calculator: {week0.tip_calculator()}")


def run_week1() -> None:
    # week1.compare()
    # week1.grade()
    # week1.parity()

    # week1.deep_thought()
    # week1.bank()
    # print(week1.file_extensions())
    # week1.math_interpreter()
    week1.meal_time()


def run_week2() -> None:
    # week2.test_while(5)
    # week2.test_for(5)
    # week2.test_operator(5)
    # week2.test_infinite()
    week2.test_lists()
    # week2.test_dictionaries()
    # week2.print_square(3)
    # week2.print_square_nested(4)
    # week2.camel_case("aTestName")
    # week2.coke_machine()
    week2.twttr("sam raiche")

    assert week2.validate_vanity_plate("AA123")
    assert week2.validate_vanity_plate("AA")

    assert not week2.validate_vanity_plate("1")  # too short
    assert not week2.validate_vanity_plate("AA34567")  # too long
    assert not week2.validate_vanity_plate("A123456")  # must start with 2 latters
    assert not week2.validate_vanity_plate("AA1A")  # numbers must come at end
    assert not week2.validate_vanity_plate("AA0123")  # first number cannot be zero
    assert not week2.validate_vanity_plate("AA 123")  # cannot have spaces
    assert not week2.validate_vanity_plate("AA.123")  # cannot have periods
    assert not week2.validate_vanity_plate("AA,123")  # cannot have punctuation

    assert week2.nutrition_facts("apple") == 130
    assert week2.nutrition_facts("banana") == 110
    assert week2.nutrition_facts("orange") is None


def run_week3() -> None:
    week3.validate_int(1) == 1
    week3.validate_int(-1) == -1
    week3.validate_int(0) == 0
    week3.validate_int("damon") == 0
    week3.validate_int("") == 0

    week3.fuel_gauge("1/1") == "F"
    week3.fuel_gauge("1/2") == "50%"
    week3.fuel_gauge("1/100") == "E"

    week3.felipes_taqueria(["Baja Taco", "Burrito"]) == 11.75
    # print(week3.live_input())

    week3.sort_groceries(["banana", "apple", "banana"]) == ["1 APPLE", "2 BANANA"]

    week3.outdated("1/2/2020") == "2020-01-02"
    week3.outdated("January 2, 2020") == "2020-01-02"


def run_week4() -> None:

    week4.coin_flip() in [True, False]

    # set args on the command line or vscode launch.json
    print(f"User entered args: {week4.args()[1:]}")

    week4.cow_say_hello("Damon")
    for song in week4.call_itunes_api("billy joel"):
        print(song)

    # exercises
    assert week4.emojize("hello, :thumbs_up:") == "hello, 👍"

    week4.fig_letter("damon")

    assert week4.adieu(["damon"]) == "Adieu, adieu, to damon"
    assert week4.adieu(["damon", "allison"]) == "Adieu, adieu, to damon and allison"
    assert (
        week4.adieu(["damon", "ryan", "allison"])
        == "Adieu, adieu, to damon, ryan and allison"
    )

    assert week4.guessing_game(5) in ["you win", "too small", "too large"]

    # Uncomment if you want to play the game. Requires user input.
    # week4.little_professor()

    print(f"current bitcoin price: {week4.bitcoin_price_index(1.0)}")


if __name__ == "__main__":
    """__name__ will be __main__ when the file is the "main" module or script being executed.

    This allows the script to distinguish between being run directly or being imported as a module into another script.
    """
    run_week4()
