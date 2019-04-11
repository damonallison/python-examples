import unittest


class TestTuples(unittest.TestCase):
    """Tuples are immutable, ordered sequence of elements."""

    def test_tuples(self):
        """Tuples are immutable, ordered sequences."""

        t = (1, 2)

        self.assertEqual(2, len(t))
        self.assertEqual(1, t[0])

        # Parens can be omitted when defining a tuple var
        t2 = t[1], 3, 4

        self.assertTupleEqual((2, 3, 4), t2)



    def test_tuple_assignment(self):
        """Tuples can be used to assign multiple variables at once."""

        coords = (44.23432, -123.234322)
        lat, lon = coords  # Unpack the tuple

        self.assertEqual(44.23432, lat)
        self.assertEqual(-123.234322, lon)

        self.assertEqual(coords[0], lat)

    def test_single_element_tuples(self):
        """Single element tuples must be defined with a trailing comma.
        t = 1,
        """

        # Typical tuple assignment
        t = (1, 2, 3)
        self.assertEqual((1, 2, 3), t)

        # You need the comment after the element for a single element tuple.
        t1 = 1,
        self.assertEqual((1,), t1)

        # Access tuple elements using list notation. t[pos]
        for x in range(0, len(t)):
            self.assertEqual(x, t[x] - 1)
