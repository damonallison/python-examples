"""Python enums are a set of symbolic names bound to unique, constant values.

https://docs.python.org/3/library/enum.html

* IntEnum: A base class for creating an enum which is a subclass of int
* @unique: Class decorator which ensures only one name is bound to any one value.
"""

import enum

import pydantic
import pytest

# @enum.unique ensures only one name is bound to one value. If this is omitted,
# you could have the same value associated to multiple names.
#
# IntEnum and StrEnum (3.10) are Enum subclasses where it's members can be used
# in the place of an int and str respectively
@enum.unique
class GeniusLevel(enum.Enum):
    UNKNOWN = "unknown"
    LOW = "low"
    MID = "mid"
    HIGH = "high"

    @classmethod
    def create(cls, value: str) -> "GeniusLevel":
        try:
            return cls(value)
        except ValueError:
            return cls("unknown")


class Payload(pydantic.BaseModel):
    genius: GeniusLevel


def test_enum_basics() -> None:
    # accessing an enum's name and value
    assert GeniusLevel.LOW.name == "LOW"
    assert GeniusLevel.LOW.value == "low"
    print(str(GeniusLevel.HIGH))

    # creating an instance from a value
    assert GeniusLevel("mid") == GeniusLevel.MID

    # trying to create an instance with an invalid value raises ValueError
    with pytest.raises(ValueError):
        GeniusLevel("not there")

    # using a custom factory method to create an instance safely
    assert GeniusLevel.create("not there") == GeniusLevel.UNKNOWN


def test_pydantic() -> None:
    p = Payload(genius=GeniusLevel.HIGH)
    p.genius == GeniusLevel.HIGH
    print(p.json())
