"""
Data classes add auto-generated "dunder" methods to user defined classes.

The @dataclass decorator examines classes for "fields" (attributes) with type
annotations. Type annotations are used to help static type checkers. It works by
creating members for all variables in __annotations__

Data classes are typically used as data structures
"""

from dataclasses import dataclass, field, FrozenInstanceError

import pytest

def test_dataclass() -> None:
    @dataclass(frozen=True)
    class DC:
        first_name: str
        last_name: str
        # Use default_factory() to create new instances of mutable types as
        # default values for fields.
        children: list["DC"] = field(default_factory=list)

        def __post_init__(self):
            """__post_init__ will be called after initialization."""
            pass

    damon = DC(first_name="damon", last_name="allison")
    assert damon.first_name == "damon"
    assert damon.children == []

    # data classes add default __eq__ methods
    damon.children.append(DC("cole", "allison"))
    assert damon.children[0] == DC("cole", "allison")

    with pytest.raises(FrozenInstanceError):
        damon.first_name = "oops"
