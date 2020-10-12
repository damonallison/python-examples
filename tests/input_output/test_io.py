"""Tests that show various forms of I/O.

* String formatting to console.
* File I/O.
"""

import unittest
import os
import json


class TestIO(unittest.TestCase):
    """Tests for Python I/O."""

    _testFileName = "test.txt"

    def clean(self):
        """Ensure any resources created by this test suite are deleted."""
        try:
            os.remove(self._testFileName)
        except FileNotFoundError:
            pass

    def setUp(self):
        self.clean()

    def tearDown(self):
        self.clean()

    def test_basic_printing(self) -> None:
        """Use str() and repr() for basic writing.

        * str() == human readable output
        * repr() == interpreter readable output.

        Some types (int, float) have the same string representation for both
        "readable" and "non-readable" output. Strings, however, do not.
        """
        self.assertEqual("damon", str("damon"))
        self.assertEqual("'damon'", repr("damon"))

        # Numbers have the same representation across both str() and repr()
        x = 100.
        self.assertEqual("100.0", str(x))
        self.assertEqual("100.0", repr(x))

    def test_string_formatting(self) -> None:
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

        amt = 100.
        self.assertEqual(type(amt), float)

        #
        # str.format() uses positional arguments
        #
        self.assertEqual("damon allison $100.00",
                         "{0} ${1:.2f}".format(person, amt))

        #
        # Format using "f-strings" (format strings)
        #
        # You can put any expression within {}
        #
        self.assertEqual("$100.00", f"${amt:.2f}")
        self.assertEqual("damon allison 4", f"{person} {2 + 2}")

        #
        # format using keyword arguments
        #
        self.assertEqual("damon allison",
                         "{first} {last}".format(first=person.first_name,
                                                 last=person.last_name))

        self.assertEqual("$100 is $100.00",
                         "${var} is ${var:.2f}".format(var=100))

    def test_format_using_dictionary(self) -> None:
        """Shows using a dictionary as a parameter to str.format().

        Use ** to unpack the dictionary, similar to how * unpacks a tuple.
        """

        d = {"first_name": "damon",
             "last_name": "allison"}

        self.assertEqual(
            "damon allison", f"{d['first_name']} {d['last_name']}")

        self.assertEqual("damon allison",
                         "{first_name} {last_name}".format(**d))

    def test_reading_writing_files(self) -> None:
        """With will automatically close the file.

        Generally, always use with() to ensure your file handle is closed
        properly. It's more efficient than writing a try/finally block (the
        other option).

        Mode
        ----
        r  = read only (default)
        w  = write only (an existing file w/ the same name will be erased)
        a  = append
        r+ = read / write
        """

        # Adding the newlines is extremely lame.
        lines = ["Hello\n", "World\n"]

        with open(self._testFileName, mode="w") as f:
            f.writelines(lines)

        with open(self._testFileName, mode="r") as f:
            readLines = f.readlines()

        self.assertEqual(lines, [x for x in readLines])

        # The default iterator for a file object will read lines.
        # Note: The default "mode" is "r" (read only)
        with open(self._testFileName) as f:
            self.assertEqual(lines, [x for x in f])

    def test_json_read_write(self) -> None:
        d = {"first_name": "damon",
             "last_name": "allison",
             "age": 42}

        with open(self._testFileName, mode="w") as f:
            json.dump(d, f)

        d2 = None
        with open(self._testFileName, mode="r") as f:
            d2 = json.load(f)

        self.assertEqual("damon", d2["first_name"])
        self.assertDictEqual(d, d2)


if __name__ == '__main__':
    unittest.main()
