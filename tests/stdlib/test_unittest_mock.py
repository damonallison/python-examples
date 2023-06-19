"""Mocking examples

https://docs.python.org/3/library/unittest.mock.html

MagicMock is just a Mock variant that has all the magic methods defined for you.

* @patch() will patch an entire class
* @patch.object() will patch an individual attribute
* @match.dict() will patch a dictionary

In all @patch cases, the original object will be restored after the function or
with block is exited.

IMPORTANT:

If you are using @patch - you must patch the name of the object that is being
used in the system under test. Read this:
https://docs.python.org/3/library/unittest.mock.html#where-to-patch

"""
import os
from unittest.mock import MagicMock, patch

import pytest


class MyTest:
    def add(self, var: int) -> int:
        return var + 10


def test_magicmock() -> None:

    mt = MyTest()
    assert mt.add(10) == 20

    # Mock a single method to return a value
    mt.add = MagicMock(return_value=100)
    assert mt.add(10) == 100

    mt.add.assert_called_once()
    mt.add.assert_called_with(10)
    mt.add.assert_called_once_with(10)

    # Mock a method to return an exception)
    mt.add.side_effect = BaseException("boom")
    with pytest.raises(BaseException) as bex:
        mt.add(10)
    assert str(bex.value) == "boom"


@patch("test_unittest_mock.MyTest", autospec=True)
def test_patch(mt: MagicMock) -> None:
    assert mt is MyTest

    # IMPORTANT:
    #
    # Patching a class replaces the class with a MagicMock instance. If the
    # class is instantiated in the code under test then it will be the
    # return_value of the mock that will be used.
    #
    # To configure return values on methods of instances on the patched class
    # you must do this on the `return_value` of the mock.

    mt.return_value.add.return_value = 100

    obj = MyTest()
    assert obj.add(1) == 100


@patch.dict("os.environ", {"NEW_KEY": "newval"})
def test_patch_env() -> None:
    assert os.environ["NEW_KEY"] == "newval"
