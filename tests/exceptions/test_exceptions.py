"""Tests exception handling.

    There are two classes of errors in Python:

    * SyntaxError : the code couldn't be parsed
    * Exception   : runtime error
"""

import sys
import unittest

from .custom_derived_error import CustomDerivedError
from .custom_error import CustomError


def test_exception_handling_all_clauses() -> None:
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
        assert 0
    else:
        result.append("end")
    finally:
        # Finally will *always* be executed (even if try was exited with `break`)
        # If the exception was not caught (or it was re-raised),
        # it will be re-raised after the finally clause.
        result.append("finally")

    assert ["start", "middle", "end", "finally"] == result


def test_exception_handling() -> None:
    """Handling exceptions is simple. Wrap code into a try / except block.

    You can catch multiple types by providing a tuple to an except clause.
    """
    try:
        1 / 0
        assert 0, "should have thrown a ZeroDivisionError"
    except (ZeroDivisionError, TypeError, NameError) as err:
        assert type(err) is ZeroDivisionError


def test_base_exception() -> None:
    """Exception is the base class for all built in exceptions"""
    try:
        1 / 0
    except Exception as e:
        assert isinstance(e, ZeroDivisionError)


def test_exception_object_hierarchy() -> None:
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
        assert type(cex) is CustomDerivedError
        assert "test" == cex.state
    except CustomError as cex:
        assert 0, "CustomDerivedError should have caught the exception."
    except:
        assert 0, f"Unhandled exception: {sys.exc_info()[0]}"
        raise


def test_exception_chaining() -> None:
    """Exception chaining allows you to keep a back stack of exceptions.

    Whenever you are catching and re-raising an exception, chain it.
    """

    def f():
        raise IOError

    try:
        try:
            f()
        except IOError as ioe:
            raise RuntimeError("some i/o operation failed") from ioe
    except RuntimeError as r:
        pass
