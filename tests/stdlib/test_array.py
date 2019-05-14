import unittest
from array import array


class ArrayTests(unittest.TestCase):

    def test_array(self) -> None:
        """array is like list [] but only stores data of a single type,
        represented by a typecode.

        Array is used to store data more compactly.
        """

        # array of signed integer (minimum of 2 bytes each)
        a = array('h', [10, 20, 30])

        lst = []
        for item in a:
            lst.append(item)

        self.assertEqual(list(a), lst)
