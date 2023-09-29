"""Pydantic is a validation and serialization library.

Features include:

* Type hinting at runtime
* Data validation

Pydantic has two model object base classes: BaseModel and dataclass. By default,
you'll want to use `BaseModel`.

pydantic.dataclasses are a drop in replacement for Python dataclasses which adds
type validation.

Differences between BaseModel and dataclasses include:

* Mutable field initializers. Pydantic will automatically deepcopy a field's
  default value. dataclass requires a default_factory

* BaseModel handles extra fields

Unless you have a specific reason *not* to use BaseModel, always use BaseModel.
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
        model_config = pydantic.ConfigDict(frozen=True)
        # Example recursive model
        name: str
        children: list[Person] = []  # Pydantic will deepcopy() the default argument.

    p = Person(name="damon")
    # Immutability only impacts the model object itself. Mutable values stored
    # on the model are still capable of being edited.
    p.children.append(Person(name="kari"))

    p2 = p.model_copy(deep=True)
    assert p == p2

    with pytest.raises(pydantic.ValidationError):
        p.name = "no"


def test_pydantic() -> None:
    class A(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(
            frozen=True,
            extra="allow",
            validate_default=True,
        )
        # Using default=[] here would work as well - as Pydantic deepcopy(s) the
        # default argument for each new instance, preventing the common python
        # bug caused by mutable default values.
        lst: list[int] = pydantic.Field(
            default_factory=list, description="A list", max_length=1
        )

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

    with pytest.raises(pydantic.ValidationError) as ve:
        A(lst=[1, 2])
    from typing import cast

    ve = cast(pydantic.ValidationError, ve.value)
    # We should have a single validation error given len(lst) exceeded it's
    # max_length configuration value.
    assert ve.error_count() == 1
    assert ve.errors()[0]["type"] == "too_long"
