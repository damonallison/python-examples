"""An enum is a set of symbollic names (members) bound to unique, constant values.

Enums:
https://docs.python.org/3/library/enum.html
"""
import sys

import pytest

from enum import Enum, unique

# By default, Python enums allow you to define multiple enum members with the
# same value. That should *not* be allowed, but because it's "Python", of course
# it is. In order to ensure all member values are unique, use the @unique
# decorator.
@unique
class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"

    def __str__(self):
        return self.value


def test_enum() -> None:
    c = Color.RED

    assert type(c) is Color

    # It's preferred to compare enums by identity, not "=="
    assert c is Color.RED
    assert c == Color.RED
    assert c.name == "RED"
    assert c.value == "red"

    assert str(c) == c.value == "red"

    # Creating an instance by value.
    assert Color("red") == c

    # By defining __str__, we avoid having to call `.value` when using the enum
    # as a string.
    assert f"c == {c}" == "c == red"


def test_creation_by_value() -> None:
    """An example showing creating an enum by value.

    Create an enum instance by value using the enum constructor. An invalid enum
    value will raise a ValueError.
    """
    assert Color("red") == Color.RED

    with pytest.raises(ValueError):
        Color(10)
