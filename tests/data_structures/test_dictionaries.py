from collections import defaultdict


class TestDictionaries:
    def test_dictionary_membership(self) -> None:
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
        assert len(d) == 1

    def test_dictionary_iteration(self) -> None:
        """Iterate dictionaries using key, value"""

        d = {"first": "damon", "last": "allison"}

        keys = set()
        vals = set()

        for k, v in d.items():
            keys.add(k)
            vals.add(v)

        assert {"first", "last"} == keys
        assert {"last", "first"} == keys

        assert {"damon", "allison"} == vals
        assert {"allison", "damon"} == vals

    def test_dictionary_comprehensions(self) -> None:
        """Dictionaries can be created from dictionary comprehensions"""
        d = {x: x ** 2 for x in range(4)}
        # NOTE: dictionaries are *not* ordered by default
        assert d == {
            0: 0,
            1: 1,
            3: 9,
            2: 4,
        }

    def test_defaultdict(self) -> None:
        """defaultdict() returns a default value rather than throwing a KeyError when an element does not exist."""
        wc = defaultdict(int)  # int produces 0 defaults
        # uses the default value (0) rather than throwing a `KeyError`
        wc["word"] += 1
        assert 1 == wc["word"]
