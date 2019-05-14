import unittest

from .custom_derived_error import CustomDerivedError
from .custom_error import CustomError


class TestExceptions(unittest.TestCase):
    """Tests exception handling.

    There are two types of errors:

    SyntaxError : the code couldn't be parsed.
    Exception   : runtime error.
    """

    testFileName = "test.txt"

    def test_exception_handling_all_clauses(self):
        """try/catch is a little different than traditional OOP languages like Java or C#.

        Python's `try` statement can have an `else` clause, which is only
        executed when no exception is raised.

        `finally` is *always* executed before the try block returns,
        regardless of how it completes. If it completes with a break, continue,
        or return statement, an error is raised and caught, or an error is
        raised and *not* caught.

        `else` is executed only when an exception is *not* raised in the try block.

        If an error is raised and *not* caught, it will be re-raised after the
        finally executes.
        """
        result = ["start"]
        try:
            result.append("middle")
        except CustomError:
            pass
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

    def test_exception_handling(self) -> None:
        """Handling exceptions is simple. Wrap code into a try / except block.

        You can catch multiple types by providing a tuple to an except clause.
        """
        try:
            print(str(10 / 0))
            self.fail("Should have thrown a ZeroDivisionError")
        except (ZeroDivisionError, TypeError, NameError) as err:
            self.assertTrue(type(err) is ZeroDivisionError)

    def test_base_exception(self):
        """Exception is the base class for all built in exceptions"""
        try:
            1 / 0
            self.fail("Exception should have been thrown")
        except Exception as e:
            self.assertTrue(isinstance(e, ZeroDivisionError))

    def test_exception_object_hierarchy(self):
        """The except clause is compatible with an exception when the else clause
        contains the exception being raised of is a base class of the exception
        being raised.

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

if __name__ == '__main__':
    unittest.main()
