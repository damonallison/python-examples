"""contextlib provides helper functions for working with contexts (with statementse"""
import dataclasses
import contextlib
import pytest

from typing import Iterator


@dataclasses.dataclass
class Person:
    name: str


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


def test_context_manager() -> None:
    with managed_resource() as p:
        assert p.name == "test"


def test_context_manager_error() -> None:
    """Errors are caught and reraised in managed_resource"""
    with pytest.raises(ValueError):
        with managed_resource() as p:
            raise ValueError("Oops")
