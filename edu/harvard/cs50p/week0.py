from typing import Any


def concat(p1: str, p2: str, p3: str) -> None:
    return p1 + p2 + p3


def print_string(s: str) -> None:
    """An example of a function.

    Functions take arguments and return a value. None is a special value that
    can be returned which means "nothing". This function implicitly returns None
    by omitting a return statement.

    This comment is called a "docstring" and it describes how the function
    works. It's used by programs like visual studio code to provide contextual
    help when you are examining a function definition. Documentation generators
    use docstrings to automatically generate documentation for each function in
    a code base. Documentation is stored in the "__doc__" attribute of a
    function. You can retrieve the documentation in your code with:

    Example:
        print(print_string.__doc__)

    Args:
        s: the string you want to print
    """

    print(s)


def is_int(val: Any) -> bool:
    try:
        _ = int(val)
        return True
    except TypeError:
        return False


def indoor_voice() -> str:
    """Lowercase input."""
    val = input("Enter something: ")
    return val.lower()


def playback_speed() -> str:
    """Replace spaces with ..."""
    val = input("Enter something :")
    return val.replace(" ", "...")


def making_faces() -> None:
    """Converts one substring to another"""
    val = input("Enter something with ':)' or ':(': ")
    return val.replace(":)", "🙂").replace(":(", "🙁")


def einstein() -> None:
    m = input("m: ")
    return int(m) * (300000000**2)


def tip_calculator() -> None:

    def dollars_to_float(d: str) -> float:
        return float(d.removeprefix("$"))

    def percent_to_float(p: str) -> float:
        return float(p.removesuffix("%")) / 100.0

    dollars = dollars_to_float(input("How much was the meal? "))
    percent = percent_to_float(input("What percentage would you like to tip? "))
    tip = dollars * percent
    print(f"Leave ${tip:.2f}")
