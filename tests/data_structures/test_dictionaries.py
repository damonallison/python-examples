import unittest


class TestDictionaries(unittest.TestCase):
    """Dictionaries (HashMap) allow us to store elements with a key."""

    def test_dictionary_membership(self):
        """Dictionaries are key:value pairs.

        Dictionary keys can be any immutable type.
        Strings, numbers, or tuples can all be keys.
        """

        d = {"first": "damon", "last": "allison"}

        # Test for membership using in
        self.assertTrue("first" in d, msg="Test for membership.")

        # Test for membership and retrieving a value using .get()
        self.assertEqual("damon", d.get("first"))
        self.assertEqual(None, d.get("middle"))

        # Be careful when retrieving an object using bracket syntax.
        #
        # A KeyError is thrown if trying to retrieve an element
        # using bracket syntax which does not exist.
        #
        # Which is why get() is a safer method for retrieving a value.
        try:
            d["middle"]
            self.fail("Shouldn't get here")
        except KeyError:
            pass

        # Remove using `del`
        del d["first"]
        self.assertFalse("first" in d)

    def test_dictionary_iteration(self):
        """Iterate dictionaries using key, value"""

        d = {"first": "damon", "last": "allison"}

        keys = set()
        vals = set()

        for k, v in d.items():
            keys.add(k)
            vals.add(v)

        self.assertSetEqual({"first", "last"}, keys)
        self.assertSetEqual({"allison", "damon"}, vals)

    def test_dictionary_comprehensions(self):
        """Dictionaries can be created from dictionary comprehensions"""
        d = {x: x**2 for x in range(4)}
        self.assertDictEqual({0: 0, 1: 1, 2: 4, 3: 9}, d)


if __name__ == "__main__":
    unittest.main()
