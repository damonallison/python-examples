"""
Week 5: unit testing

poetry run pytest edu/harvard/cs50p/week5.py

* Test corner / edge cases / negative cases (exceptions)
* Decompose large functions into smaller, testable functions
* Write small, focused tests around a common pattern
* Avoid side effects (print / file i/o) in functions
    * Functional programming improves testability and understanding


Packages
--------

Packages are folders with an __init__.py file which tells python the folder is a
package.
"""

import pytest

from edu.harvard.cs50p import week1, week2, week3


def square(value: int) -> int:
    return value**2


def test_this() -> None:
    assert square(-2) == 4
    assert square(-1) == 1
    assert square(1) == 1
    assert square(0) == 0
    assert square(2) == 4
    assert square(3) == 9


@pytest.mark.parametrize(
    ["value", "expected"],
    [(-3, 9), (-2, 4), (-1, 1), (0, 0), (1, 1), (2, 4), (3, 9)],
)
def test_parameterized(value: int, expected: int) -> None:
    assert square(value) == expected


def test_raises() -> None:
    with pytest.raises(TypeError):
        square("no")


def test_twttr() -> None:
    assert week2.twttr("world") == "wrld"
    assert week2.twttr("twitter") == "twttr"


def test_bank() -> None:
    assert week1.bank("hello, world") == 0
    assert week1.bank("can i help you") == 100


def test_vanity_plate() -> None:
    assert week2.validate_vanity_plate("AA")

    assert not week2.validate_vanity_plate("A")
    assert not week2.validate_vanity_plate("AA1A1")


def test_fuel() -> None:
    assert week3.fuel_gauge("1/10") == "10%"
    assert week3.fuel_gauge("1/1") == "F"
    assert week3.fuel_gauge("1/1000") == "E"
