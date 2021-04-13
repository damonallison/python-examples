"""Examples of python's primitive types.

Python has 4 primitive types:
    * int
    * float
    * string
    * bool
"""


class TestPrimitiveTypes:
    def test_bool(self):
        x = True
        assert type(x) == bool

        # bool is also equal to 1 and 0.
        assert x == 1 and x <= 10

    def test_type_conversion(self) -> None:
        # By default, Python uses type inference
        i = 100  # int
        f = 100.0  # float (python will add a .0 to an integer)

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
        assert bool(i) and "test"

        # Any object can be converted to a string with str()
        message = str(13) + " is my lucky number"
        assert message == "13 is my lucky number"
        assert "True" == str(True)
        assert "42" == str(42)
        assert "42.0" == str(42.0)
