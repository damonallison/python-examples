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

from edu.harvard.cs50p import week0, week1, week2


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
    # week2.test_lists()
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


if __name__ == "__main__":

    # week2.test_dictionaries()
    try:
        week2.test_exc(11)
    except ValueError as e:
        print(e)
        raise e
    exit(0)

    # create a list
    students = [
        "damon",  # 0
        "sam",  # 1
        "grace",  # 2
    ]

    # read things in or about the list
    assert len(students) == 3

    assert students[0] == "damon"
    assert students[0:2] == ["damon", "sam"]
    assert students[-1] == "grace"
    assert students[-2] == "sam"
    assert students[:-1] == ["damon", "sam"]
    assert students[0 : len(students)] == students

    # manipulate the list
    students.append("lily")
    students.extend(["cris", "joe"])
    students.remove("damon")

    # check if items exist
    assert "damon" not in students
    assert "cole" not in students

    print(students)

    # run_week2()
