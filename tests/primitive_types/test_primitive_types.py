import re

"""Examples of python's primitive types.

Python has 4 primitive types:
    * int
    * float
    * string
    * bool
"""


def test_bool():
    x = True
    assert type(x) == bool

    # bool is also equal to 1 and 0.
    assert x == 1 and x <= 10

def test_type_conversion() -> None:
    # By default, Python uses type inference
    i = 100   # int
    f = 100.  # float (python will add a .0 to an integer)

    assert i == f and f == i

    # type
    assert type(i) == int
    assert type(f) == float

    # Type casting
    k = int(f)
    assert type(k) == int
    assert str(f) == "100.0"

    # int will only take the integer portion, dropping the decimal
    # (no rounding)
    f = 100.9
    assert 100 == int(f)

    # Use type conversion functions (float(), int(), bool()) to convert
    # types or create variables of a known type.

    # Anything other than 0 will be True
    assert bool(i) and bool(f)

    # Any object can be converted to a string with str()
    message = str(13) + " is my lucky number"
    assert message == "13 is my lucky number"
    assert "True" == str(True)
    assert "42" == str(42)
    assert "42.0" == str(42.)


def test_string_creation() -> None:
    """Python has multiple ways of creating strings. tl;dr: use f strings"""

    # Single and double quoted strings are identical - one form does not have
    # special features the other doesn't.
    x = "This isn't a test"
    y = 'This "is" a test'

    # Escape the string delimiter character with a \ in both cases.
    assert 'This isn\'t a test' == x
    assert "This \"is\" a test" == y

    assert type(x) == str

    # Multi-line string literal
    # \ at the end of line will prevent a newline from being added.
    z = """\
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


def test_string_manipulation() -> None:
    # strings can be concatenated with +
    f_name = "damon"
    l_name = "allison"
    assert f"{f_name}{l_name}" == f_name + l_name

    # strings are indexable. note - python does *not* have a character type.
    # characters are 1 length strings
    assert "d" == f_name[0]
    assert "on" == f_name[-2:]

    # str is immutable
    f_name = f_name.replace("a", "ae")
    assert "daemon" == f_name

    assert "daemon allison" == " ".join([f_name, "allison"])


def test_join() -> None:
    names = ["damon", "ryan", "allison"]
    assert "damon ryan allison" == " ".join(names)


def test_regex() -> None:
    assert re.search(r"damon", "damon allison", re.IGNORECASE) is not None
    assert re.search(r"^[dD]amon$", "damon") is not None

