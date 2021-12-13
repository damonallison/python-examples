"""Manage, store, and access context-local state."""
from typing import List
import contextvars
import dataclasses


def get_vars() -> List[str]:
    return [f"{k.name} == {str(v)}" for k, v in contextvars.copy_context().items()]


def test_contextvars() -> None:
    @dataclasses.dataclass
    class Person:
        name: str

        def __str__(self) -> str:
            return self.name

    var = contextvars.ContextVar("o_id")
    var.set(123)
    var2 = contextvars.ContextVar("b_id")
    var2.set(456)
    var3 = contextvars.ContextVar("person")
    var3.set(Person("test"))

    var_strs = get_vars()

    assert "o_id == 123" in var_strs
    assert "b_id == 456" in var_strs
    assert "person == test" in var_strs
