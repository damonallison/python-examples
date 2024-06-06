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

Classes add another namespace.
"""

from typing import Self

from copy import copy, deepcopy
import inspect

from tests.classes.logger import Logger
from tests.classes.person import Person
from tests.classes.manager import Manager


def test_type_hierarchy() -> None:
    """Shows how to create and inspect types as part of a type hierarchy.

    Python classes store attributes in an internal __dict__ variable.

    The dir() function retrieves returns a list of all attributes and methods.
    You can use the `inspect` module to determine a member is a method
    """

    class A:
        def __init__(self, value: str) -> None:
            self._value = value

        @property
        def value(self) -> str:
            return self._value

        @value.setter
        def value(self, value: str) -> None:
            self._value = value

    class B(A):
        pass

    # verify the type hierarchy
    assert isinstance(A, type)
    assert isinstance(B, type)
    assert issubclass(B, A)
    # there is only one instance of each class type
    assert id(A) == id(A)

    # verify the instance hierarchy
    b = B("test")
    assert not isinstance(B("test"), type)
    assert isinstance(b, B)
    assert isinstance(b, A)
    assert isinstance(b, object)
    assert issubclass(b.__class__, A)

    # list all methods
    assert ["__init__"] == [
        name for name, _ in inspect.getmembers(b, predicate=inspect.ismethod)
    ]

    # list all properties
    properties = [
        name
        for name, _ in inspect.getmembers(type(b), lambda x: isinstance(x, property))
    ]
    assert ["value"] == properties

    print(dir(b))
    print(B.__dict__)
    print(B.__annotations__)
    print(B.__base__)
    print(B.__bases__)

    print(A.__subclasses__())
    print(A.__name__)
    print(A.__qualname__)

    a = A("test")
    assert a.__dict__ == {"_value": "test"}
    assert a.value == "test"


def test_super() -> None:
    """Constructor "chaining".

    When defining class hierarches, ensure you call super().__init__() if you
    want to run the superclass's constructor (which you probably always want.
    """
    logs: list[str] = []

    class A:
        def __init__(self):
            logs.append(f"A.__init__")
            super().__init__()

        def f(self) -> str:
            return "a"

    class B:
        def __init__(self):
            logs.append(f"B.__init__")
            super().__init__()

        def f(self) -> str:
            return "b"

        def f2(self) -> str:
            return super().f()

    class C(B, A):
        def __init__(self):
            logs.append("C.__init__")
            super().__init__()

    c = C()
    assert logs == ["C.__init__", "B.__init__", "A.__init__"]

    # Python's method resolution algorithm is depth first, left to right.
    # Therefore, it finds `B` before `A`.
    assert c.f() == "b"
    assert c.f2() == "a"

    assert hasattr(c, "f") and hasattr(c, "f2")
    # Here, we see that B.f resolves before A.f
    assert C.f2 == B.f2 and C.f == B.f

    class D(A, B): ...

    d = D()
    assert hasattr(d, "f") and hasattr(d, "f2")
    # Here, we see that A.f resolves before B.f
    assert D.f == A.f and D.f2 == B.f2


def test_class_variables() -> None:
    # `iq` is an example class variable.
    #
    # Class variables are shared by all instances. Instance variables of the
    # same name take priority over class variables.
    Person.iq = 100
    assert Person.iq == 100

    p = Person("damon", "allison")
    assert p.full_name() == "damon allison"

    assert p.iq == 100
    assert Person("kari", "allison").iq == 100

    Person.iq = 101
    assert Person.iq == 101 and p.iq == 101


def test_inheritance() -> None:
    """Example showing how to check an object's type..

    Uses type(), isinstance() and issubclass().
    """
    m = Manager("damon", "allison")

    # The manager is a logger.
    m.log("hello world")
    m.log("hello again")
    assert m.history() == ["hello again", "hello world"]

    # Note that while Python has name mangling, the mangled names can still
    # be directly accessed.
    m._Logger__original_log("a test")
    assert "a test" in m.history()

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
        def __init__(self, value: str) -> None:
            self.value = value

    #
    # TODO(@damon): What if __eq__ doesn't exist? Does it fall back to
    # reference equality?
    a1 = A("test")
    a2 = A("test")
    assert a1 is a1  # reference equality
    assert a1 != a2  # value equality (falls back to reference equality)

    class B(A):
        def __eq__(self, rhs: Self):
            return self.value == rhs.value

    b1 = B("test")
    b2 = B("test")

    assert b1 is b1
    assert b1 is not b2

    assert b1 == b2

    x = None
    assert x is None, "always use `is None` to check for None (pythonic)"


def test_formatting() -> None:
    """__str__ and __repr__ define string representations for a class."""
    p = Person("damon", "allison")

    expected = "Person('damon', 'allison')"
    assert str(p) == expected
    assert repr(p) == expected


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

    actual: list[str] = []

    for name in p.child_first_names():
        actual.append(name)

    expected = ["grace", "lily", "cole"]
    assert actual == expected

    # automatically exhaust the generator (retrieve all elements) by
    # converting it into a list
    actual = list(p.child_first_names())
    assert actual == expected

    # manually using a generator object
    gen = p.child_first_names()
    assert next(gen) == "grace"
    assert next(gen) == "lily"
    assert next(gen) == "cole"

    # close the generator to return a value
    try:
        gen.send(None)
    except StopIteration as e:
        assert e.value == 3


def test_context_manager() -> None:
    """Python's context managers allow you to support
    python's `with` statement.

    Context managers require the object to support __enter__ and __exit__.
    """
    with Person("damon", "allison") as p:
        assert "damon", p.first_name


def test_object_copy() -> None:
    # All func arguments are "pass by value"
    class C:
        #
        # Custom objects can implement __copy__ and __deepcopy__ to control the copying behavior.
        # See
        #
        def __init__(self, name: str = "default", children: list[str] = None):
            self.name = name
            self.children = children or []

        def __copy__(self) -> "C":
            return C(self.name, self.children)

        def __deepcopy__(self, memo: dict) -> "C":
            return C(self.name, deepcopy(self.children))

    def mutate_c(c: C):
        c.name += " mutated"
        c.children.append("mutated")

    c = C(children=["test"])
    assert hasattr(c, "name")
    assert hasattr(c, "children")

    assert c.name == "default"
    assert c.children == ["test"]

    mutate_c(c)
    assert c.name == "default mutated"
    assert c.children == ["test", "mutated"]

    c2 = copy(c)
    mutate_c(c2)
    assert c.name == "default mutated"
    assert c.children == ["test", "mutated", "mutated"]  # updated - copy is shallow

    assert c2.name == "default mutated mutated"
    assert c2.children == ["test", "mutated", "mutated"]

    d = C(name="test", children=["test"])
    d2 = deepcopy(d)

    mutate_c(d)
    assert d.name == "test mutated"
    assert d.children == ["test", "mutated"]

    assert d2.name == "test"
    assert d2.children == ["test"]  # not mutated - copy is deep (recursive)
