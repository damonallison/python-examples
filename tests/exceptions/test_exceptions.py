import unittest

from .custom_derived_error import CustomDerivedError
from .custom_error import CustomError


class TestExceptions(unittest.TestCase):
    """Tests exception handling"""

    testFileName = "test.txt"

    @classmethod
    def setUpClass(cls) -> None:
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_exception_handling(self) -> None:
        """Handling exceptions is simple. Wrap code into a try / except block.

        You can handle multiple types
        """
        try:
            print(str(10 / 0))
            self.fail("Should have thrown a ZeroDivisionError")
        except (ZeroDivisionError, TypeError, NameError) as err:
            self.assertTrue(type(err) is ZeroDivisionError)

    def test_exception_object_hierarchy(self) -> None:
        """The except clause is compatible with an exception if it is the same class or a base class.

        Catch the most derived exceptions first, followed by the most generic.

        The last except clause may omit the exception name(s) to serve as a wildcard.
        **Be careful** with this approach, you may mask real errors.
        """

        try:
            raise CustomDerivedError(state="test")
        except CustomDerivedError as cex:
            self.assertTrue(type(cex) is CustomDerivedError)
            self.assertEqual("test", cex.state)
        except CustomError as cex:
            self.fail("CustomDerivedError should have caught the exception." + str(cex))
        except:
            self.fail("A generic wildcard handler will catch all exception types")

    def test_else_clause(self) -> None:
        """The `else` clause (optional) will be executed when no exception is raised."""

        result = ["start"]
        try:
            result.append("middle")
        except:
            self.fail("No exception should be raised")
        else:
            result.append("end")
        finally:
            # Finally will *always* be executed (even if try was exited with `break`)
            # If the exception was not caught (or it was re-raised),
            # it will be re-raised after the finally clause.
            result.append("finally")

        self.assertEqual(["start", "middle", "end", "finally"], result)

if __name__ == '__main__':
    unittest.main()
