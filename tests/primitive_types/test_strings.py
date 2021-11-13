import re
from string import Template


class TestStrings:
    def test_string_creation(self) -> None:
        """Python has multiple ways of creating strings.

        tl;dr: use f strings if you are using Python 3.6 or later

        FYI: Python strings are immutable.
        """

        # Single and double quoted strings are identical - one form does not have
        # special features the other doesn't.
        x = 'This "is" a test'
        y = 'This "is" a test'
        assert isinstance(x, str) and isinstance(y, str)

        # == uses value equality (same value)
        assert x == "This " + '"is" a test'

        # is uses identity equality (same instance)
        assert x is not "This " + '"is" a test'

        # Multi-line string literal
        # \ at the end of line will prevent a newline from being added.
        """\
Usage: test [OPTIONS]
    -h        help
    -H        hostname
"""

        # Raw strings do not escape special special characters
        assert r"damon\nallison" == "damon\\nallison"

        # f strings (formatted string literals) provide string interpolation.
        #
        # f strings are available in python 3.6 and later
        name = "damon"
        assert f"my name is {name}" == "my name is damon"

    def test_string_formatting_concatenation(self) -> None:
        # strings can be concatenated with +
        f_name = "damon"
        l_name = "allison"
        assert f_name + l_name == "damonallison"

        # strings are indexable. note - python does *not* have a character type.
        # characters are 1 length strings
        assert "d" == f_name[0]
        assert "on" == f_name[-2:] and f_name.endswith("on")

        # str is immutable
        fn = f_name.replace("a", "ae")
        assert fn == "daemon"
        assert f_name == "damon"

    def test_string_format(self) -> None:
        """Python 3 introduced string.format()"""

        assert "Hello {}".format("damon") == "Hello damon"

        num = 100
        assert (
            "Hello {name}, {num:d}".format(name="damon", num=num) == "Hello damon, 100"
        )
        pi = 3.14159
        assert "Hello 3.14" == "Hello {num:.2f}".format(num=pi)

    def test_string_formatting_f_strings(self):
        """Python 3.6+ added formatted string literals, or 'f-strings'.

        f-strings allow you to embed arbitrary expressions into strings and are
        the preferred method for string formatting.
        """
        name = "damon"
        num = 100
        assert f"hello {name}, {num:#d}" == "hello damon, 100"
        assert f"hello {name}, {num:0>10d}" == "hello damon, 0000000100"
        assert f"hello {name}, {num:.2f}" == "hello damon, 100.00"

    def test_templates(self):
        """Templates allow simple name / value substitution.

        Templates are less flexible than str.format() or f-strings. All
        formatting must be done outside the template."""

        t = Template("Hello $name, $num")
        assert t.substitute(name="damon", num=100) == "Hello damon, 100"

    def test_join(self) -> None:
        names = ["damon", "ryan", "allison"]
        assert "damon ryan allison" == " ".join(names)

    def test_regex(self) -> None:
        # To determine if a string contains a substring, use "in" and "not in"
        assert "test" in "this is a test" and "damon" not in "cole allison"

        assert re.search(r"damon", "damon allison", re.IGNORECASE) is not None
        assert re.search(r"^[dD]amon$", "damon") is not None
