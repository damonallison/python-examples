import unittest


class TestPrimitiveTypes(unittest.TestCase):
    """Examples of python's primitive types

    Python has 4 primitive types:

        * int
        * float
        * string
        * bool

        """

    def test_type_conversion(self):
        """Shows type conversion

        Use `type(var)` for determining a variable's type.

        Python has built-ins (int, float, str, bool) for type casting.
        """

        # By default, Python uses type inference
        x = 10
        self.assertTrue(int, type(x))

        # Use type conversion functions (float(), int(), bool()) to convert
        # types or create variables of a known type.

        y = float(x)
        self.assertTrue(float, type(y))

        # Anything other than 0 will be True
        self.assertTrue(bool(100))

        # When converting to int, the decimal portion is cut off entirely
        # (not rounded)
        y = 100.9
        self.assertEqual(100, int(y))

        # str()

        message = str(13) + " is my lucky number"
        self.assertEqual("13 is my lucky number", message)

    def test_int_float_conversion(self):

        i = 100   # int
        j = 100.  # float (python will add a .0 to an integer)

        self.assertEqual(i, j)

        # type
        self.assertEqual(int, type(i))
        self.assertEqual(float, type(j))
        self.assertEqual(TestPrimitiveTypes, type(self))

        # Type casting
        k = int(j)
        self.assertEqual(int, type(k))

        # int will only take the integer portion, dropping the decimal
        # (no rounding)
        j = 100.9
        self.assertEqual(100, int(j))

    def test_string_definition(self):
        """String definition"""

        # Strings can be defined with single or double quotes.
        # They are equal representations.
        x = "This isn't a test"
        y = 'This "is" a test'

        # Escape the string delimiter character with a \
        self.assertEqual('This isn\'t a test', x)
        self.assertEqual("This \"is\" a test", y)

        self.assertEqual(str, type(x))

        # Raw strings r"str" will *not* interpret \ as a special character
        self.assertEqual(r"A \test", "A \\test")

        # Multi-line string literal
        # \ at the end of line will prevent a newline from being added.
        print("""\
Usage: test [OPTIONS]
    -h        help
    -H        hostname
""")

    def test_string_conversion(self):
        """Use str() to get a string representation from any variable"""

        self.assertEqual("True", str(True))
        self.assertEqual("42", str(42))

    def test_string_concatenation(self):
        first_name = "Damon"
        last_name = "Allison"

        self.assertEqual("Damon Allison", first_name + " " + last_name)

    def test_string_slicing(self):
        """Strings are sequences and can be indexed.

        Strings are also immutable, so the following will error
        str[0] = "D"
        """

        s = "Damon"
        self.assertEqual(len(s), 5)
        self.assertEqual(s[0:3].lower() + s[-1].lower(), "damn")
