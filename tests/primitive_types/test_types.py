"""
Using type objects.
"""
from typing import Generic, Self, TypeVar

import copy

# A type variable (generic type)
T = TypeVar("T")


def test_types() -> None:
    x = 5

    assert type(x) == int

    # Creating instances dynamically

    class MyClass:
        def __init__(self, value: int) -> None:
            self.value = value

    # Creating a new type
    t = type("MyDynamicClass", (MyClass,), {})

    obj = t(42)
    assert type(obj) == t
    assert obj.value == 42

    # Introspecting a type
    assert t.__name__ == "MyDynamicClass"
    assert MyClass in t.__bases__  # A tuple of base types from which the type inherits.
    t.__module__ == "test_types"  # The module name where the type is defined.


def test_generic_types() -> None:
    """Using generic types.

    Generic types are types which can be parameterized with one or more type
    parameters. They allow you to define functions, classes, or data structures
    that can operate on different types in a type-safe manner, without
    sacrificing the flexability and reusability of the code.
    """

    class Stack(Generic[T]):
        def __init__(self):
            self.items: list[T] = []

        def push(self, item: T) -> None:
            self.items.append(item)

        def pop(self) -> T:
            return self.items.pop()

    s = Stack[int]()
    # Notice code completion will resolve T -> int
    s.push(10)
    assert s.pop() == 10

    s2 = Stack[str]()
    # Notice code completion will resolve T -> str
    s2.push("test")
    assert s2.pop() == "test"


def test_copy() -> None:
    class A:
        def __init__(self, name: str, children: list[Self] = None) -> None:
            self.name = name
            self.children = children

        def __eq__(self, rhs: Self) -> bool:
            if self.name != rhs.name:
                return False
            if self.children != rhs.children:
                return False
            return True

    a1 = A("damon", [A("grace")])

    a2 = copy.copy(a1)

    assert a1 is not a2
    assert a1 == a2

    # Because the copy is shallow, only a pointer to the list is copied.
    a2.children.append(A("lily"))
    assert a1 == a2

    a3 = copy.deepcopy(a1)
    assert a1 == a3

    a3.children.append(A("lily"))
    assert a1 != a3


def test_equality() -> None:
    """Implement the dunder methods __lt__ and friends to allow for sorting."""

    class Person:
        def __init__(self, name: str, age: int):
            self.name = name
            self.age = age

        def __lt__(self, rhs: Self) -> bool:
            return self.age < rhs.age

        def __gt__(self, rhs: Self) -> bool:
            return self.age > rhs.age

        def __le__(self, rhs: Self) -> bool:
            return not self.__gt__(rhs)

        def __ge__(self, rhs: Self) -> bool:
            return not self.__lt__(rhs)

    p1 = Person("Aaron", 10)
    p2 = Person("Brett", 5)

    std: list[Person] = sorted([p1, p2])

    assert std[0].name == "Brett"
    assert std[1].name == "Aaron"
