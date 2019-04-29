import unittest


class TestSets(unittest.TestCase):

    def test_sets(self):
        """Sets are created using set() or {e1, e2, ...}."""

        colors = {"orange", "yellow", "green"}

        self.assertEqual(3, len(colors))

        # set membership
        self.assertTrue("orange" in colors)

        # add
        colors.add("black")
        colors.remove("black")

        # Set comprehension is also supported
        filtered = {x for x in colors if len(x) > 5}

        self.assertEqual({"yellow", "orange"}, filtered)

    def test_set_from_list(self):
        """Use set() to create a set from a list"""
        s = set(["a", "a", "b", "c"])

        self.assertSetEqual({"b", "c", "a"}, s)
        self.assertTrue({"b", "c", "a"} == s)

    def test_set_operations(self):
        """Example of set operations - union, difference, intersection"""

        s1 = {"red", "green", "blue"}
        s2 = {"red", "blue", "white"}

        # union - the set of unique elements across both sets
        self.assertSetEqual({"red", "green", "white", "blue"},
                            s1.union(s2))

        # intersection - the set of elements in s1 also in s2
        self.assertSetEqual({"red", "blue"}, s1.intersection(s2))

        # difference - the set of elements only found in list 1
        self.assertSetEqual({"green"}, s1.difference(s2))

        # Removes a random element from the set.
        s1.pop()
        self.assertTrue(2, len(s1))
