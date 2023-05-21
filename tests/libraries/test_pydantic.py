"""Pydantic attempts to provide "strongly typed" data structures to Python.

Features include:

* Type hinting at runtime
* Data validation

Pydantic has two model object base classes: BaseModel and dataclass. By default,
you'll want to use `BaseModel`.

pydantic.dataclasses are a drop in replacement for Python dataclasses which adds
type validation.

Differences between BaseModel and dataclasses include:

* Mutable field initializers. dataclass requires a default_factory
* BaseModel handles extra fields
"""

#
# `from __future__ import annotations`` allows you to reference a type as an
# annotation within the type declaration.
#
# For example, we define the attr `children` as a list of `Person`. In order to
# refer to ``Person`, we must import annotations.`
#
from __future__ import annotations

import pydantic
import pytest


def test_frozen() -> None:
    class Person(pydantic.BaseModel):
        # Example recursive model
        name: str
        children: list[Person] = []

        class Config:
            frozen = True

    p = Person(name="damon")
    p.children.append(Person(name="kari"))

    with pytest.raises(TypeError):
        p.name = "no"




def test_pydantic() -> None:
    class A(pydantic.BaseModel):
        # By default, Pydantic does not reuse field initializers across
        # instances
        lst: list[int] = pydantic.Field(
            default_factory=list, description="A list", max_items=1
        )

        class Config:
            """Config allows you to customize Pydantic's behavior"""

            # whether to `ignore`, `allow`, or `forbid` extra attributes during
            # model initialization. default: pydantic.Extra.ignore
            extra = pydantic.Extra.allow
            validate_all = True  # whether to validate field defaults

    @pydantic.dataclasses.dataclass
    class B:
        lst: list[int] = pydantic.Field(default_factory=list)

    a, a2 = A(extra="123"), A()
    b, b2 = B(), B()

    a.lst.append(1)
    b.lst.append(2)

    assert a.extra == "123"
    assert a.lst == [1]
    assert a2.lst == []

    assert b.lst == [2]
    assert b2.lst == []

    #
    # Pydantic's field validation and rules are only enforced during
    # initialization set directly. For example, appending to a mutable list will
    # *not* raise validation errors.
    #
    # Is this configurable? Shouldn't validation be ran for each field set by
    # default?
    #
    a.lst.append(2)
    assert len(a.lst) == 2
    a.lst = [1, 2]

    with pytest.raises(ValueError) as ve:
        A(lst=[1, 2])
    assert ve.match("max_items")
