"""Python enums are a set of symbolic names bound to unique, constant values.

https://docs.python.org/3/library/enum.html

* IntEnum: A base class for creating an enum which is a subclass of int
* @unique: Class decorator which ensures only one name is bound to any one value.
"""

import enum


# @enum.unique ensures only one name is bound to one value. If this is omitted,
# you could have the same value associated to multiple names.
# @enum.unique
class GeniusLevel(str, enum.Enum):
    UNKNOWN = "unknown"
    LOW = "low"
    MID = "mid"
    HIGH = "high"
    HIGH2 = "high"

    def __str__(self) -> str:
        return self.value



def test_enum_basics() -> None:
    g = GeniusLevel.MID

    assert GeniusLevel.HIGH == GeniusLevel.HIGH2


    # print(f"g.name == {g.name}, g.value == {g.value}")
    # p = Payload(genius=g)
    # print(g)
    # print(str(g))
    # print(p.json())

    # # From value
    # p2 = Payload(genius=GeniusLevel("high"))
    # print(p2.json())

    # print(GeniusLevel.MID == "mid")

    # for v in GeniusLevel:
    #     print(v)
