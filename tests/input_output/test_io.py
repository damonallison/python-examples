"""Tests that show various forms of I/O.

* String formatting to console.
* File I/O.
"""

import pathlib

import csv
import json
import os
import tempfile
import unittest


def test_basic_printing() -> None:
    """Use str() and repr() for basic writing.

    * str() == human readable output
    * repr() == interpreter readable output.

    Some types (int, float) have the same string representation for both
    "readable" and "non-readable" output. Strings, however, do not.
    """
    assert "damon" == str("damon")
    assert"'damon'" == repr("damon")

    # Numbers have the same representation across both str() and repr()
    x = 100.
    assert "100.0" == str(x) == repr(x)


def test_fixture(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "test.txt"

    with p.open(mode='w') as f:
        assert f.writable()
        f.write("hello, world")

    with p.open(mode="r") as f:
        assert "hello, world" == f.read()


def test_string_formatting() -> None:
    """Shows various string formatting operations.

    Formatted string literals (f-strings) are preferred, but only supported
    in Python 3.6 and later.

    str.format() allows you to put variables into your format strings,
    similar to f-strings. The difference between this and f-strings is that
    variables must be specified as additional arguments to the format()
    function.

    Both forms will call the __format__ function on a class, allowing you to
    customize a class's string representation.

    def __format__(self, format_spec):
        return f"MyClass {some_var}"

    """

    class Person:
        def __init__(self, first_name, last_name):
            self.first_name = first_name
            self.last_name = last_name

        def __format__(self, format_spec):
            return f"{self.first_name} {self.last_name}"

    person = Person("damon", "allison")
    assert "damon allison" == format(person)

    amt = 100.

    #
    # str.format() uses positional arguments
    #
    assert ("damon allison $100.00" == "{0} ${1:.2f}".format(person, amt))

    #
    # Format using "f-strings" (format strings)
    #
    # You can put any expression within {}
    #
    assert "$100.00" == f"${amt:.2f}"
    assert "damon allison 4" == f"{person} {2 + 2}"

    #
    # format using keyword arguments
    #
    assert "damon allison" == "{first} {last}".format(first=person.first_name,
                                                      last=person.last_name)

    assert "$100 is $100.00" == "${var} is ${var:.2f}".format(var=100)

    # format using an unpacked dictionary as an argument
    d = {"first_name": "damon", "last_name": "allison"}

    assert "damon allison" == f"{d['first_name']} {d['last_name']}"
    assert "damon allison" == "{first_name} {last_name}".format(**d)


def test_reading_writing_files(tmp_path: pathlib.Path) -> None:
    """`with` automatically closes it's resource (file).

    Generally, always use with() to ensure your file handle is closed
    properly. It's more efficient / safe than manually closing / dealing
    with exceptions.

    Mode
    ----
    * r  = read only (default)
    * w  = write only (an existing file w/ the same name will be erased)
    * a  = append r+ = read / write
    """

    p = tmp_path / "test.txt"

    # Adding the newlines is extremely lame.
    lines = ["Hello\n", "World\n"]
    with open(p, mode="w") as f:
        f.writelines(lines)

    with open(p, mode="r") as f:
        readLines = f.readlines()

    assert lines == [x for x in readLines]

    # The default iterator for a file object will read lines.
    # Note: The default "mode" is "r" (read only)
    with open(p) as f:
        assert lines == [x for x in f]


def test_json_read_write(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "test.txt"

    d = {"first_name": "damon",
         "last_name": "allison",
         "age": 42}

    with open(p, mode="w") as f:
        json.dump(d, f)

    with open(p, mode="r") as f:
        d2 = json.load(f)

    assert "damon" == d2["first_name"]
    assert d == d2


def test_csv_read_write(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "test.txt"

    d = {'AAPL': 119.12, "GOOGL": 1546.23}

    with open(p, mode="w") as f:
        w = csv.writer(f, delimiter=",")
        for stock, price in d.items():
            w.writerow([stock, price])

    d2 = dict()
    with open(p, mode="r", newline='') as f:
        r = csv.reader(f)
        for row in r:
            d2[row[0]] = float(row[1])

    assert d2["AAPL"] == 119.12
    assert d2["GOOGL"] == 1546.23
