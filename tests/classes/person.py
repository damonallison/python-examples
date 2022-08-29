"""A simple person class."""
import logging
from typing import Any, ClassVar, List, Sequence


class GenericIterator:
    """An iterator object which conforms to the "iterator protocol"."""

    def __init__(self, seq: Sequence):
        self._index = 0
        self._seq = seq

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self._seq):
            raise StopIteration

        curr = self._seq[self._index]
        self._index += 1
        return curr


class Person:
    """A simple person class

    Objects in python are *very* simple. Each class definition contains two
    types of attributes: data attributes and functions (methods).
    """

    #
    # A class variable (attribute) shared by all instances.
    #
    # This is a sneaky form of global state. Prefer instance variables over
    # class variables.
    #
    iq: ClassVar[int] = 0

    first_name: str
    last_name: str
    children: List["Person"]

    def __init__(self, first_name: str, last_name: str = "default"):
        """A class can have a single __init__ constructor function.

        Unlike other languages, python does not allow you to define multiple
        constructors. Idiomatic python will use default arguments for
        "convenience" (not required) attributes.
        """

        self.first_name = first_name
        self.last_name = last_name
        self.children = []

    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    #
    # Object identity
    #
    def __eq__(self, rhs: Any) -> bool:
        """__eq__ implements value equality (x == y)"""
        if not isinstance(rhs, Person):
            return False
        return self.first_name == rhs.first_name and self.last_name == rhs.last_name

    def __hash__(self) -> int:
        """__hash__ allows an object to be used in hashed collections like set
        and dict.

        Objects that are __eq__ must have the same __hash__.

        Only immutable objects should implement __hash__. This class is *not*
        immutable, this is just here for illustration purposes.
        """
        return hash(self.first_name, self.last_name)

    #
    # Formatting
    #
    def __repr__(self) -> str:
        """__repr__ is called by repr() to print the "official" string
        representation of the object.

        An object can also define __str__(), which is called by str(), format()
        and print().

        __str__ differs from __repr__ in that __str__ can be more informal and
        human readable.

        if __str__ is *not* defined on an object, __repr__ is used.
        """
        return f"Person('{self.first_name}', '{self.last_name}')"

    #
    # Container
    #

    # Containers are used to implement a sequence or mapping.
    # https://docs.python.org/3/reference/datamodel.html#emulating-container-types
    #
    # Sequences are int based, think lists, mappings are key / value based,
    # think dict.
    #
    # Mapping containers should provide implementations of keys(), values(),
    # items(), get(), clear(), setdefault(), pop(), popitem(), copy() and
    # update(). The abstract base class collections.abc.MutableMapping will
    # provide a default implementation for many of these methods.
    #
    #  * Immutable "container" : __len__, __get_item__
    #  * Mutable "container" : immutable container + __setitem__, __del_item__
    #  * Iterable : __iter__, returning conformance to __iter__ and next

    #
    # Python has magic methods for treating an object as a container.
    #
    # Here, we are treating Person as a container for "children", keyed by child
    # first name.

    def __len__(self):
        """called by len()"""
        return len(self.children)

    #
    # NOTE: slicing is done with the following three methods.
    #

    def __getitem__(self, key: int):
        """__getitem__ defines behavior for when an item is accessed, using the notation self[key].

        It should raise appropriate exceptions:
            * TypeError if the type of key is wrong
            * KeyError is there is no value for the key
        """
        return self.children[key]

    def __setitem__(self, key: int, val):
        self.children.__setitem__(key, val)

    #
    # Iteration protocol
    #
    # Containers that would like to implement iteration (all should), must
    # define the __iter__ function and return an object which implements the
    # "iterator" protocol (__iter__ and __next__).
    #
    def __iter__(self):
        """iter() should return an object with a `__next__` method"""
        return GenericIterator(self.children.copy())

    # A generator function ("generator") is a cleaner way to write iterators.
    # Generators maintain internal state and implement the iterator protocol
    # (i.e., __iter__, __next__) for you.
    #
    # A generator is simply a function which uses the yield keyword, returning
    # the "next" object to the caller.
    #
    # When the generator function ends, it raises a "StopIteration" error,
    # terminating the iterator.

    def child_first_names(self):
        for child in self.children.copy():
            yield child.first_name

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
