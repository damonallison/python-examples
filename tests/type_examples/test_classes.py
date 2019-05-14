"""Examples of using custom classes in python.

Class objects support two kinds of operations: attribute references (data and
method references) and instiantiation.

Classes can be created anywhere - within functions, within an `if` statement,
etc. Classes introduce a new scope and the class object itself is added to it's
enclosing scope (typically a module).

Classes, like python itself, are dynamic. They are created at runtime and can be
modified further after creation.

You cannot hide data in Python. All hiding is based on convention. (Underscore
prefixes, etc)

* Members are "public". Methods are virtual (can be overridden).
* Methods are declared with a specific `self` first argument representing the
  object, which is provided implicitly during method invocation.
* Built-in types can be used as base classes.
* Like C++, most built-in operators with special syntax (arithmetic operators)
  can be redefined for class instances (==, +=)
"""

import unittest

from .classes.printer import Printer
from .classes.person import Person
from .classes.manager import Manager


class ClassesTest(unittest.TestCase):
    """Examples of creating and using classes."""

    def test_basic_object_usage(self) -> None:
        """Shows creating objects and calling methods."""

        #
        # Objects can have class level ("static" or "global") state. Of course,
        # this is traditionally a **bad** idea.
        #
        Person.iq = 50
        self.assertEqual(50, Person.iq)

        p = Person("damon", "allison")
        self.assertEqual("damon", p.first_name)
        self.assertEqual("allison", p.last_name)
        self.assertEqual("damon allison", p.full_name())

        # Referencing the class variable thru an instance pointer. This is
        # generally frowned upon. Don't mix class and instance state.
        # You'd want to use "Person.iq" here.
        self.assertEqual(50, p.iq)

        # Change the default class variable. All instances are impacted.
        Person.iq = 52
        self.assertEqual(52, p.iq)
        self.assertEqual(52, Person.iq)

        # You can dynamically modify class instances at runtime (yuk).
        # Here, we add a "test" data member to p2.
        p2 = Person("cole", "allison")
        p2.test = 100
        self.assertEqual(100, p2.test)
        self.assertEqual(52, p2.iq)  # Still have access to static state.

        # Methods can be called in two ways:
        #
        # 1. By calling the class function with an instance.
        #
        self.assertEqual("cole allison", Person.full_name(p2))
        #
        # 2. By calling the instance method (pythonic).
        #
        self.assertEqual("cole allison", p2.full_name())

    def test_docstring(self) -> None:
        """Classes have a __doc__ attribute that will return the docstring"""

        p = Person("damon", "allison")

        self.assertIsNotNone(p.__doc__)
        # print(p.__doc__)

    def test_check_type(self) -> None:
        """Example showing how to check for type instances using isinstance()"""
        m = Manager("damon", "allison")

        self.assertTrue(isinstance(m, Printer))
        self.assertTrue(isinstance(m, Manager))
        self.assertTrue(isinstance(m, Person))
        self.assertTrue(isinstance(m, object))

        # Example showing that __class__ is used retrieve the class object for a
        # variable.
        self.assertTrue(issubclass(m.__class__, Printer))
        self.assertTrue(issubclass(m.__class__, Manager))
        self.assertTrue(issubclass(m.__class__, Person))
        self.assertTrue(issubclass(m.__class__, object))

        # Test method overriding
        self.assertEqual("Manager damon allison", m.full_name())

    def test_equality(self) -> None:
        """Object equality

        Python has two similar comparison operators: `==` and `is`.

        `==` is for object equality (calls __eq__)
        `is` is for object identity (two objects are the *same* object)
        """

        class A:
            def __eq__(self, rhs):
                return False

        x = A()
        self.assertTrue(x is x)
        self.assertFalse(x == x)

        # == will use the class's implementation of `__eq__` if it exists.
        # Therefore, to always determine if a variable is truly `None`, use `is
        # None`.
        x = None
        self.assertTrue(x is None,
                        msg="Always use `is None` to check for None")

    def test_type_check(self) -> None:
        """Use isinstance() to check for a type, issubclass() to check for inheritance."""

        m = Manager("test", "user")
        self.assertTrue(isinstance(m, Person))
        self.assertTrue(isinstance(m, Manager))
        self.assertTrue(isinstance(m, Printer))

        self.assertTrue(issubclass(m.__class__, Manager))
        self.assertTrue(issubclass(m.__class__, Person))
        self.assertTrue(issubclass(m.__class__, object))

    def test_iteration(self) -> None:

        p = Person("damon", "allison")
        p.children = [Person("grace", "allison"),
                      Person("lily", "allison"),
                      Person("cole", "allison")]

        children = []

        # Person defines __iter__ and __next__, thus it supports iteration.
        for child in p:
            children.append(child)

        self.assertEqual("grace", children[0].first_name)
        self.assertEqual("lily", children[1].first_name)
        self.assertEqual("cole", children[2].first_name)

    def test_generator(self) -> None:
        """Person.child_first_names() is a generator function. Generators return iterators."""

        p = Person("damon", "allison")
        p.children = [Person("grace", "allison"),
                      Person("lily", "allison"),
                      Person("cole", "allison")]

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

    def test_context_manager(self) -> None:
        """Python's context managers allow you to support
        python's `with` statement."""

        with Person("damon", "allison") as p:
            self.assertEqual("damon", p.first_name)


if __name__ == '__main__':
    unittest.main()
