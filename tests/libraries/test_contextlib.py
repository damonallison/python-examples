"""contextlib provides helper functions for working with contexts (with statements)"""
import dataclasses
import contextlib
from types import TracebackType
import pytest

from typing import Iterator, Optional


@dataclasses.dataclass
class Person:
    name: str


class MyContextManager(contextlib.AbstractContextManager):
    def __init__(self, val: str) -> None:
        self.val = val

    def __enter__(self) -> str:
        return self.val

    def __exit__(
        self,
        exc_type: Optional[BaseException],
        exc_value: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        if exc_type is ValueError:
            # Here we could examine the exception. If we handle the exception,
            # We return True to suppress the exception
            print("We are suppressing value errors")
            return True
        return False


@contextlib.contextmanager
def managed_resource(*args, **kwargs) -> Iterator[Person]:
    # code to acquire resource (like a DB connection)
    try:
        yield Person(name="test")
    except Exception as ex:
        print(f"found an exception {ex}")
        #
        # If you don't re-raise the exception, it will get swallowed.
        raise ex
    finally:
        # release resource
        obj = None


def test_custom_context_manager() -> None:
    val = "hello, world"
    with MyContextManager(val) as v:
        assert type(v) == type(val)
        assert v == val


def test_custom_context_manager_handled_exception() -> None:
    # The context manager suppresses ValueError exceptions
    val = "hello"
    with MyContextManager(val) as v:
        raise ValueError("Oops")

    with pytest.raises(NotImplementedError) as e, MyContextManager(val) as v:
        assert v == val
        raise NotImplementedError("err", "another err")

    assert "another err" in str(e)


def test_context_manager() -> None:
    with managed_resource() as p:
        assert p.name == "test"


def test_context_manager_error() -> None:
    """Errors are caught and reraised in managed_resource"""
    with pytest.raises(ValueError):
        with managed_resource() as p:
            raise ValueError("Oops")
