"""Examples of creating classes in python."""

import unittest

from .custom_objects.person import Person
from .custom_objects.manager import Manager

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
        self.assertTrue(x == None)

    def test_type_check(self) -> None:
        """Use isinstance() to check for a type. issubclass(obj,"""


        p = Person("test", "user")
        self.assertTrue(isinstance(p, Person))


    def test_object_creation(self) -> None:
        """Shows creating objects and calling methods."""

        #
        # Objects can have class level state.
        #
        Person.iq = 50
        self.assertEqual(50, Person.iq)

        p = Person("damon", "allison")
        self.assertEqual("damon", p.first_name)
        self.assertEqual("allison", p.last_name)

        # You can access class level state thru an instance (yuk).
        self.assertEqual(50, p.iq)
        self.assertEqual(50, Person.iq)

        # You can dynamically modify class instances at runtime (yuk).
        p2 = Person("cole", "allison")
        p2.test = 100
        self.assertEqual(100, p2.test)

        self.assertEqual(50, p.iq)

        #
        # Methods can be called in two ways:
        #
        # 1. By calling the class function with an instance.
        #
        self.assertEqual("cole allison", Person.full_name(p2))
        #
        # 2. By calling the instance method.
        #
        self.assertEqual("cole allison", p2.full_name())


    def test_inheritance(self) -> None:

        m = Manager("damon", "allison")

        self.assertTrue(isinstance(m, Manager))
        self.assertTrue(isinstance(m, Person))

        # Example showing that __class__ is used retrieve the class object for a variable.
        self.assertTrue(issubclass(m.__class__, Person))
        self.assertTrue(issubclass(m.__class__, Manager))

        # Test method overriding
        self.assertEqual("Manager damon allison", m.full_name())

    def test_iteration(self) -> None:

        p = Person("damon", "allison")
        p.children = [Person("grace", "allison"), Person("lily", "allison"), Person("cole", "allison")]

        children = []
        for child in p:
            children.append(child)

        self.assertEqual("grace", children[0].first_name)
        self.assertEqual("lily", children[1].first_name)
        self.assertEqual("cole", children[2].first_name)

    def test_generator(self) -> None:

        p = Person("damon", "allison")
        p.children = [Person("grace", "allison"), Person("lily", "allison"), Person("cole", "allison")]

        names = []
        for name in p.child_first_names():
            names.append(name)

        self.assertEqual("grace", names[0])
        self.assertEqual("lily", names[1])
        self.assertEqual("cole", names[2])

        #
        # Generator expressions are similar to list comprehensions
        # but with parentheses rather than brackets.
        #
        names = list(child.first_name for child in p.children)
        self.assertEqual("grace", names[0])
        self.assertEqual("lily", names[1])
        self.assertEqual("cole", names[2])




if __name__ == '__main__':
    unittest.main()
