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

import datetime
import os
from pathlib import Path
import tempfile


from edu.harvard.cs50p import (
    week0,
    week1,
    week2,
    week3,
    week4,
    week5,
    week6,
    week7,
    week8,
)


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

    assert week3.outdated("1/2/2020") == "2020-01-02"
    assert week3.outdated("January 2, 2020") == "2020-01-02"
    assert week3.outdated("January 2, 05") == "0005-01-02"


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


def run_week5() -> None:
    week5.test_this()


def run_week6() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        week6.test_csv(tmp_path)

        tmp_file = tmp_path / "test.txt"
        with open(tmp_file, "w") as f:
            f.writelines([f"damon{os.linesep}", f"sam{os.linesep}"])
            f.writelines([f"# a comment{os.linesep}", os.linesep])

        assert week6.lines_of_code(tmp_file) == 2

    week6.pizza_py()
    assert week6.scourgify() == [
        {"first": "damon", "last": "allison", "house": "maple grove"},
        {"first": "sam", "last": "raiche", "house": "plymouth"},
    ]
    week6.cs50_tshirt()


def run_week7() -> None:
    assert week7.is_ipv4("0.0.0.0")
    assert not week7.is_ipv4("1")
    assert not week7.is_ipv4("1.2")
    assert not week7.is_ipv4("1.2.3")
    assert not week7.is_ipv4("0256.0.0.0")

    assert week7.Person.hi("damon") == "hi damon"
    p = week7.Person("damon", 47)
    assert p.greet() == "hello, damon. you are 47"

    youtube_urls = week7.youtube(
        """<iframe width="560" height="315" src="https://www.youtube.com/embed/1" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
            <iframe width="560" height="315" src="https://www.youtube.com/embed/2" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
            <iframe width="560" height="315" src="https://www.youtube.com/embed/3" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>"""
    )
    expected = [
        "https://www.youtube.com/embed/1",
        "https://www.youtube.com/embed/2",
        "https://www.youtube.com/embed/3",
    ]
    assert youtube_urls == expected

    assert week7.nine_to_five("9:00 AM to 5:00 PM") == "09:00 to 17:00"
    assert week7.nine_to_five("12:00 AM to 12:00 PM") == "00:00 to 12:00"

    assert week7.count_ums("this is, um, an, um, long ultimatum") == 2

    assert week7.validate_email("damon@damonallison.com")
    assert not week7.validate_email("damon@damonallison dot com")


def run_week8() -> None:
    print(week8.how_old_in_minutes(datetime.date(1976, 8, 22)))
    print(week8.how_old_in_minutes(datetime.date(2004, 3, 17)))
    print(week8.how_old_in_minutes(datetime.date(2006, 10, 21)))

    week8.shirtificate("damon allison")


if __name__ == "__main__":
    """__name__ will be __main__ when the file is the "main" module or script being executed.

    This allows the script to distinguish between being run directly or being imported as a module into another script.
    """
    # run_week3()
    # run_week6()
    # run_week7()
    run_week8()
