"""Mocking examples

https://docs.python.org/3/library/unittest.mock.html

MagicMock is a Mock variant that has all the magic methods defined for you.

* @patch() will patch an entire class
* @patch.object() will patch an individual attribute
* @match.dict() will patch a dictionary

In all @patch cases, the original object will be restored after the function or
with block is exited.

When you create a Mock (or MagicMock), you can give it a "spec". This spec will
ensure only the attributes of the `spec` are able to be called. If spec is an
object (rather than a random list of strings), it also allows the class to pass
"isinstance" checks.

IMPORTANT:

If you are using @patch - you must patch the name of the object that is being
used in the system under test. Read this:
https://docs.python.org/3/library/unittest.mock.html#where-to-patch

"""
import os
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


class MyTest:
    def __init__(self) -> None:
        self._name = "default"

    def add(self, var: int) -> int:
        return var + 10

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, val: str) -> None:
        self._name = val


def test_magicmock_hierarchy() -> None:
    m = MagicMock()
    m.add(2, 2)
    m.add.assert_called_with(2, 2)

    m.add.child.return_value = MyTest()
    m.add.child.grandchild.return_value = MyTest()

    assert m.add.child().add(10) == 20
    assert m.add.child.grandchild().add(10) == 20


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


def test_isinstance() -> None:
    """
    When using a spec, you ensure only the attributes defined on the spec object
    are created. This ensures any attribute called will result in an attribute
    error, rather than the default Mock behavior of returning a new mock when
    the attribute is first accessed.

    Always use specs with mocks to ensure your mocks behave like the real
    objects as much as possible.
    """
    m = MagicMock(spec=MyTest)
    assert isinstance(m, MyTest)

    m.add.return_value = 100
    assert m.add(10) == 100
    assert m.add(10, 10) == 100


@patch("test_unittest_mock.MyTest", autospec=True)
def test_patch(mt: MagicMock) -> None:
    """
    Patching a class replaces the class with a MagicMock instance. WHen the
    class is instantiated in the code under test, the mock will be used.

    To configure return values on methods of instances on the patched class you
    must do this on the `return_value` of the mock.
    """

    t = MyTest()

    # hmm, this isn't working. Not sure why.
    # assert isinstance(mt, MyTest)

    # constructor invoke
    mt.assert_called_with()

    # mock a method
    mt.return_value.add.return_value = 100
    assert t.add(10) == 100

    mt.return_value.add.assert_called_once()
    mt.return_value.add.assert_called_once_with(10)


def test_patch_object() -> None:
    """
    Patching an object allows you to replace a class's individual attribute with
    a mock object.
    """

    # Patching an attribute
    with patch.object(MyTest, "add", return_value=100) as mock:
        t = MyTest()
        assert t.add(2) == 100

    # Patching a property
    with patch.object(
        MyTest, "name", new_callable=PropertyMock, return_value="test"
    ) as m:
        t = MyTest()
        assert t.name == "test"


@patch.dict("os.environ", {"NEW_KEY": "newval"})
def test_patch_env() -> None:
    assert os.environ["NEW_KEY"] == "newval"
