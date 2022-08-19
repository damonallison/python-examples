from collections import defaultdict

import pytest


def test_dictionary_key_hashability() -> None:
    """Dictionary keys can be any hashable type.

    Strings, numbers, or tuples can all hashable. To test an object for
    hashability, use hash().
    """

    assert hash("str")
    assert hash(1)
    assert hash((1, 2))

    with pytest.raises(TypeError):
        hash([])

    with pytest.raises(TypeError):
        _ = {[]: "nope"}


def test_dictionary_membership() -> None:
    """Dictionaries are hashes (associative arrays)."""

    d = {"first": "damon", "last": "allison"}

    # Test for membership using `in`
    assert "first" in d
    assert "middle" not in d

    # Test for membership and retrieving a value using .get()
    assert d.get("first") == "damon"
    assert d.get("middle") is None
    assert d.get("middle", "default") == "default"

    # Be careful when retrieving an object using bracket syntax.
    #
    # A KeyError is thrown if trying to retrieve an element
    # using bracket syntax which does not exist.
    #
    # Which is why get() is a safer method for retrieving a value.
    with pytest.raises(KeyError):
        d["middle"]

    # Remove using `del`
    del d["first"]
    assert d.get("first") is None
    assert len(d) == 1


def test_dictionary_iteration() -> None:
    """Iterate dictionaries using key, value"""

    d = {"first": "damon", "last": "allison"}

    keys = set()
    vals = set()

    # Use `.keys()`, `values()`, and `items()` to iterate over dict
    for k, v in d.items():
        keys.add(k)
        vals.add(v)

    assert {"first", "last"} == keys == set(d.keys())
    assert {"last", "first"} == keys == set(d.keys())

    assert {"damon", "allison"} == vals == set(d.values())
    assert {"allison", "damon"} == vals == set(d.values())


def test_dictionary_comprehensions() -> None:
    """Dictionaries can be created from dictionary comprehensions"""
    d = {x: x ** 2 for x in range(4)}
    # NOTE: dictionaries are *not* ordered by default
    assert d == {
        0: 0,
        1: 1,
        3: 9,
        2: 4,
    }


def test_defaultdict() -> None:
    """defaultdict() returns a default value rather than throwing a KeyError
    when an element does not exist."""

    # Instiatiate a default dict with a type or function
    # n
    # defaultdict(type) or defaultdict(func)
    wcdd = defaultdict(int)  # int produces 0 defaults
    # uses the default value (0) rather than throwing a `KeyError`
    wcdd["word"] += 1
    assert 1 == wcdd["word"]

    ldd = defaultdict(list)
    ldd["test"].append("one")

    assert len(ldd) == 1
    assert ldd["test"] == ["one"]

    logs: list[str] = []

    class C:
        fname: str

        def __init__(self):
            logs.append("C.__init__()")

    cdd = defaultdict(C)
    cdd["me"].fname = "damon"

    assert cdd["me"].fname == "damon"
    assert logs[0] == "C.__init__()"
