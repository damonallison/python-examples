"""Python enums are a set of symbolic names bound to unique, constant values.

https://docs.python.org/3/library/enum.html

* IntEnum: A base class for creating an enum which is a subclass of int
* @unique: Class decorator which ensures only one name is bound to any one value.
"""

import enum
import pydantic


_GENIUS_MAP = {
    0: "unknown",
    1: "low",
    2: "mid",
    3: "high",
}


@enum.unique
class GeniusLevel(str, enum.Enum):
    UNKNOWN = "unknown"
    LOW = "low"
    MID = "mid"
    HIGH = "high"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_int(cls, value: int) -> "GeniusLevel":
        return GeniusLevel(_GENIUS_MAP[value])


class Payload(pydantic.BaseModel):
    genius: GeniusLevel


def test_enum_basics() -> None:
    g = GeniusLevel.MID
    print(f"g.name == {g.name}, g.value == {g.value}")
    p = Payload(genius=g)
    print(g)
    print(str(g))
    print(p.json())

    # From value
    p2 = Payload(genius=GeniusLevel("high"))
    print(p2.json())

    print(GeniusLevel.MID == "mid")

    for v in GeniusLevel:
        print(v)

    print(f"here we are: {GeniusLevel.from_int(2)}")
