"""JSON serialization / deserialization examples."""

import json
import pytest

from typing import Any


class Person:
    def __init__(self, f_name: str, l_name: str):
        self.f_name = f_name
        self.l_name = l_name

    def __eq__(self, rhs: Any) -> bool:
        if not isinstance(rhs, Person):
            return False
        return self.f_name == rhs.f_name and self.l_name == rhs.l_name

    def as_person(d: dict) -> "Person":
        if "f_name" in d and "l_name" in d:
            return Person(d["f_name"], d["l_name"])
        return d


class DefaultEncoder(json.JSONEncoder):
    """A generic JSONEncoder that will encode all symbols in the object's symbol
    table."""

    def default(self, obj: Person) -> dict:
        """JSON Encoder"""
        return obj.__dict__


def test_json_encoding() -> None:
    a = ["this", "is", "a", "test"]
    # Pretty printing
    s = json.dumps(
        {"b": True, "a": ["this", "is", "a", "test"]}, sort_keys=True, indent=4)
    print(s)


def test_json_encoding_class_failure() -> None:
    """Person is *not* JSON serializable."""
    p = Person("damon", "allison")
    with pytest.raises(TypeError):
        s = json.dumps(p)

    s = json.dumps(p, cls=DefaultEncoder, indent=4)
    p2 = json.loads(s, object_hook=Person.as_person)

    assert p2 == p
