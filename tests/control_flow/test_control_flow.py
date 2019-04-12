import unittest


class TestControlFlow(unittest.TestCase):

    def test_if_elif(self):

        x = 100

        if x > 10 and x < 50:
            pass
        elif x > 100:
            pass
        else:
            self.assertTrue(x == 100)

    def test_complex_if(self):

        height = 72
        weight = 150

        # You can execute a complex expression
        if 18.5 < weight / height**2 < 25:
            pass
        else:
            pass

        # using logical operators (and, or, not)
        # use parsne
        self.assertTrue(height == 72 and
                        not weight > 200 and
                        (height > 100 or height < 80))

    def test_is_truthy(self):
        """Here are most of the built-in objects that are considered False in Python:

        * constants defined to be false: None and False
        * zero of any numeric type: 0, 0.0, 0j, Decimal(0), Fraction(0, 1)
        * empty sequences and collections: '"", (), [], {}, set(), range(0)

        Anything else is treated as True - it doesn't have to be a boolean value.
        """

        self.assertFalse({})
        self.assertFalse(0.0)

        x = 100

        if x:
            pass
        else:
            self.fail("x is True")

    def test_iterable(self):
        """Python for loops work with iterables.

        Strings, list, tuples, dictionaries, lists, files are all example
        iterables.

        Objects with an iter method can be used as an iterable.
        """

        # range() can create a sequence of numbers range(start, stop, step)
        for i in range(10):
            pass

        # Always use a copy of a list for iteration if you are going to mutate
        # the list.
        a = ["damon", "allison"]
        for idx, val in enumerate(a[:]):
            a[idx] = val.title()

        self.assertListEqual(["Damon", "Allison"], a)

        # Iterate thru a dictionary with items
        d = {"first": "damon", "last": "allison"}
        for key, val in d.items():
            pass

    def test_enumerate(self):
        """enumerate() returns tuples of the indicies and values of a list."""

        lst = ["damon", "ryan", "allison"]
        expected = []
        for idx, val in enumerate(lst):
            expected.append((idx, val))

        self.assertListEqual([(0, "damon"), (1, "ryan"), (2, "allison")],
                             expected)

    def test_zip(self):
        """zip() returns an iterator that combines multiple iterables
        into a sequence of tuples.

        Each tuple contains the elements in that position from all the
        iterables.

        Once one list is exhaused, zip stops.

        The object that zip() returns is a zip() object.

        See:
        https://docs.python.org/3.3/library/functions.html#zip

        """

        # Here are two iterables that we are going to pass to zip().
        # Notice how ages only has 2 elements. This will force zip() to stop
        # after two elements, leaving "joe" out of the resulting zip() result.

        # If you care about the trailing, unmatched values from longer
        # iterables, use itertools.zip_longest() instead.

        names = ("damon", "jeff", "joe")
        ages = [20, 32]

        # zip() will return a zip() object, which is an iterator. You need to
        # cast the iterator into a concrete type (tuple, list, set, dict) to
        # realize the iterable.

        self.assertTupleEqual((("damon", 20), ("jeff", 32)),
                              tuple(zip(names, ages)))
        self.assertListEqual([("damon", 20), ("jeff", 32)],
                             list(zip(names, ages)))

        # You can use * to "unzip" a list. In this case, we will unzip people
        # to send zip() the inner tuples.
        people = (("damon", "jeff"), (20, 32))
        self.assertTupleEqual((("damon", 20), ("jeff", 32)),
                              tuple(zip(*people)))

if __name__ == '__main__':
    unittest.main(verbosity=2)