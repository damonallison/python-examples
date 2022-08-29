"""Tuples are fixed length, immutable sequences.

`namedtuple` is a tuple subclass which provides named fields
"""

from collections import namedtuple
import pandas as pd


def test_namedtuple() -> None:
    Person = namedtuple("Person", ["first", "last"])
    """namedtuple(s) are tuples with named atribute access."""

    t1 = Person("damon", last="allison")
    assert t1.first == "damon" and t1.last == "allison"

    d = {"first": "cole", "last": "allison"}
    t2 = Person(**d)
    assert t2.first == "cole" and t2.last == "allison"


def test_tuple_create() -> None:
    t = (1, 2)

    assert 2 == len(t)
    assert 1 == t[0]

    # Parens can be omitted when defining a tuple var,
    # but really shouldn't be.
    t2 = t[1], 3, 4

    assert t2 == (2, 3, 4)


def test_tuple_unpacking() -> None:
    """Tuples can be used to assign multiple variables at once."""

    coords = (44.23432, -123.234322)
    lat, lon = coords  # unpacking

    assert 44.23432 == lat
    assert -123.234322 == lon


def test_single_element_tuples() -> None:
    """Single element tuples must be defined with a trailing comma.

    t = 1, # *don't* use this syntax
    t = (1,)
    """

    # Typical tuple assignment
    t = (1, 2, 3)
    assert (1, 2, 3) == t

    # You need the comment after the element for a single element tuple.
    t1 = (1,)
    assert (1,) == t1


def test_tuple_list_conversion() -> None:
    """Tuples and lists are easily castable to eachother."""
    tup = 1, 2
    lst = [3, 4]

    assert type(tup) == tuple
    assert type(lst) == list

    # Convert a list to a tuple
    tup2 = tuple(lst)

    assert type(tup2) == tuple
    assert tup2 == (3, 4)

    # Convert a tuple to a list
    lst2 = list(tup)

    assert type(lst2) == list
    assert lst2 == [1, 2]
