"""Examples of creating classes in python."""

import unittest

from .custom_objects.person import Person

class ObjectTests(unittest.TestCase):
    """Examples of creating and using `custom` objects."""
    @classmethod
    def setUpClass(cls) -> None:
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def setUp(self):
        pass

    def tearDown(self) -> None:
        pass

    def test_none(self) -> None:
        """There are two ways to check for `None`. Always use `is None`

        1. `obj is None`
        2. `obj == None`

        #2 `==` will use the class's implementation of `==` if it exists. Therefore, to always determine if a
        variable is truly `None`, use `is None`.
        """
        x = None
        self.assertTrue(x is None, msg="Always use `is None` to check for None")

    def test_type(self) -> None:
        """Type check variables"""

        p = Person("damon", "allison")
        self.assertEqual("damon", p.first_name)


if __name__ == '__main__':
    unittest.main()
