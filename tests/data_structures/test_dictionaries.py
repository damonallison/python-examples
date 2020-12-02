from collections import defaultdict


def test_dictionary_membership() -> None:
    """Dictionaries are key:value pairs.

    Dictionary keys can be any immutable type.
    Strings, numbers, or tuples can all be keys.
    """

    d = {"first": "damon", "last": "allison"}

    # Test for membership using in
    assert "first" in d
    assert "middle" not in d

    # Test for membership and retrieving a value using .get()
    assert d.get("first") == "damon"
    assert d.get("middle") is None

    # Be careful when retrieving an object using bracket syntax.
    #
    # A KeyError is thrown if trying to retrieve an element
    # using bracket syntax which does not exist.
    #
    # Which is why get() is a safer method for retrieving a value.
    try:
        d["middle"]
        assert False
    except KeyError:
        pass

    # Remove using `del`
    del d["first"]
    assert d.get("first") is None


def test_dictionary_iteration() -> None:
    """Iterate dictionaries using key, value"""

    d = {"first": "damon", "last": "allison"}

    keys = set()
    vals = set()

    for k, v in d.items():
        keys.add(k)
        vals.add(v)

    assert {"first", "last"} == keys
    assert {"damon", "allison"} == vals


def test_dictionary_comprehensions() -> None:
    """Dictionaries can be created from dictionary comprehensions"""
    d = {x: x**2 for x in range(4)}
    assert {0: 0, 1: 1, 2: 4, 3: 9} == d


def test_defaultdict() -> None:
    """defaultdict() returns a default value rather than throwing a KeyError when an element does not exist."""
    wc = defaultdict(int)  # int produces 0 defaults
    # uses the default value (0) rather than throwing a `KeyError`
    wc["word"] += 1
    assert 1 == wc["word"]
