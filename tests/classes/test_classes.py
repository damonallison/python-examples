"""Examples of classes in python.

Class objects support two kinds of operations: attribute references (data and
method references) and instantiation.

Classes can be created anywhere - within functions, within an `if` statement,
etc. Classes introduce a new scope (namespace) and the class object itself is
added to it's enclosing scope (typically a module).

Classes, like python itself, are dynamic. They are created at runtime and can be
modified further after creation (yuk). For example, attributes and methods can
be added to classes anytime.

You cannot hide data in Python. All hiding is based on convention. (Underscore
prefixes, etc)

* Members are "public". Methods are virtual (can be overridden).
* Methods are declared with a specific `self` first argument representing the
  object, which is provided implicitly during method invocation.
* Built-in types can be used as base classes.
* Like C++, most built-in operators with special syntax (arithmetic operators)
  can be redefined for class instances (==, +=)


Namespaces
----------

* Innermost scope (contains local names)
* The scopes of enclosing functions, which are searched up the callstack.
  * Contains non-local (but non-global) names.
* Current module's global names
* Builtins

* `global` is used to declare varaibles in the module's namespace.
* `nonlocal` is used to reference variables in a parent (not module) namespace.

Classes add another
"""

import builtins

from .logger import Logger
from .person import Person
from .manager import Manager


def test_namespaces() -> None:
    # An example of adding a "static" (class level) member - shared by all
    # instances of the class.
    Person.iq2 = 100
    assert 100 == Person.iq2

    p = Person("damon", "allison")

    assert 100 == p.iq2
    assert "damon allison" == p.full_name()

    Person.iq2 = 101
    assert 101 == p.iq2

    # You can modify any namespace at any time. Here, we modify the
    # "builtins" namespace. Facepalm.
    builtins.anew = "test"
    assert "test" == builtins.anew


def test_check_type() -> None:
    """Example showing how to check for type instances using type(), isinstance() and issubclass()"""
    m = Manager("damon", "allison")

    assert type(m) == Manager

    assert isinstance(m, Logger)
    assert isinstance(m, Manager)
    assert isinstance(m, Person)
    assert isinstance(m, object)

    assert issubclass(type(m), Manager)
    assert issubclass(type(m), Logger)
    assert issubclass(type(m), Person)
    assert issubclass(type(m), object)


def test_equality() -> None:
    """Object equality

    Python has two similar comparison operators: `==` and `is`.

    * `==` is for object equality (calls __eq__)
    * `is` is for object identity (two objects are the *same* object)
    """

    class A:
        def __eq__(self, rhs):
            return False

    x = A()
    assert x is x, "reference equality"
    # == will use the class's implementation of `__eq__` if it exists.
    assert x != x, "value equality"

    x = None
    assert x is None, "always use `is None` to check for None (pythonic)"


def test_formatting() -> None:
    """__repr__ defines a string representation for a class"""

    assert "Person('damon', 'allison')" == str(Person("damon", "allison"))
    assert "Manager('damon', 'allison')" == str(Manager("damon", "allison"))


def test_container() -> None:
    """Person is a container (of "child" Person objects). It allows you to
    perform container operations like indexing and slicing."""

    p = Person("damon", "allison")
    p.children = [
        Person("grace", "allison"),
        Person("lily", "allison"),
        Person("cole", "allison"),
    ]

    # note we are taking the length of the *person*, not the children collection
    assert 3 == len(p)

    # containers also supports slicing.
    cc = p[0:2]
    assert 2 == len(cc)
    assert "grace" == cc[0].first_name
    assert "lily" == cc[1].first_name

    # here we are obtaining an iterator for p and exhausting the iterator to
    # create cc.
    cc = list(p)

    assert 3 == len(cc)
    assert "grace" == cc[0].first_name
    assert "cole" == cc[2].first_name

    cc = []
    for c in p:
        cc.append(c)

    assert 3 == len(cc)


def test_generator() -> None:
    """Person.child_first_names() is a generator function.

    Generators return iterators. In practice, generators are cleaner than
    iterators since you don't need to implement __iter__, __next__ and keep
    iterator state.
    """

    p = Person("damon", "allison")
    p.children = [
        Person("grace", "allison"),
        Person("lily", "allison"),
        Person("cole", "allison"),
    ]

    names = []

    # child_first_names is a generator. list() will exhaust the generator
    names = list(p.child_first_names())
    assert 3 == len(names)
    assert "grace" == names[0]
    assert "cole" == names[2]


def test_context_manager() -> None:
    """Python's context managers allow you to support
    python's `with` statement."""
    with Person("damon", "allison") as p:
        assert "damon", p.first_name
