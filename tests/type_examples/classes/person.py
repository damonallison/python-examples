"""A simple person class."""

import logging


class Person:
    """A simple person class

    Objects in python are *very* simple. Each class definition contains two
    types of attributes: data attributes and functions (methods).

    Python's calling convention
    """

    # A class variable (attribute) shared by all instances.
    #
    # This is a sneaky form of global state. Prefer instance variables over
    # class variables.
    #
    iq = 0

    def __init__(self, first_name: str, last_name: str = "default"):
        """A class can have a single __init__ constructor function.

        Unlike other languages, python does not allow you to define multiple
        constructors. Idiomatic python will use default arguments for
        "convenience" (not required) attributes.
        """

        self.first_name = first_name
        self.last_name = last_name
        self.index = 0
        self.children = []

    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    #
    # Adding iterator behavior to classes.
    #
    def __iter__(self):
        """iter() should return an object with a `__next__` method"""

        self.index = 0
        return self

    def __next__(self):
        if self.children is None or self.index >= len(self.children):
            raise StopIteration

        idx = self.index
        self.index = self.index + 1
        return self.children[idx]

    #
    # Generators are another way to write iterators.
    #
    # Generators are typically cleaner than iterators since both __iter__ and
    # __next__ are created implicitly, it also raises `StopIteration` when the
    # iterator is exhausted.
    #
    def child_first_names(self):
        for child in self.children:
            yield child.first_name

    #
    # Adds support for "with". Python calls the with construct a "context
    # manager".
    #
    # Use context managers when you need to ensure non-local resources are
    # closed properly.
    #
    # Examples of non-local resources include:
    #
    # * Files
    # * Context handles / environment pointers
    # * Database connections
    # * Sockets
    #
    #
    # Example:
    #
    # with Person("damon", "allison") as p:
    #     pass
    #
    def __enter__(self):
        logging.warning("__enter__ person context")
        return self

    def __exit__(self, *args):
        logging.warning("__exit__ person context")
