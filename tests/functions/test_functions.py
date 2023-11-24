"""
Functions

Python has the concept of immutable (str, int) and mutable types (list, dict)

Variables are lexically scoped. Each function has a symbol table. Variables are
looked for in the local symbol table, then the symbol table of enclosing
functions, the global symbol table (module), then built-ins.

Global and nonlocal variables cannot be updated within a function without using
a `global` or `nonlocal` statement.

Parameters are always passed by value.
"""

from typing import Any, Callable, Dict, Iterator, Self, Tuple, TypeVar

T = TypeVar("T")


class SideEffectIterator(Iterator[T]):
    """An iterator object which conforms to the "iterator protocol".

    The iterator protocol:

    __iter__ = returns the iterator object (usually self)

    __next__ = the next value in the iteration, raising StopIteration when the
    iterator is exhausted

    This iterator is not necessary as all Iterator[T] objects already support
    iteration.


    """

    def __init__(self, seq: Iterator[T], side_effect: Callable[[T], None]) -> None:
        self._index = 0
        self._seq = seq
        self._side_effect = side_effect

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> T:
        if self._index >= len(self._seq):
            raise StopIteration

        curr = self._seq[self._index]
        self._side_effect(curr)
        self._index += 1
        return curr


def test_iterators() -> None:
    logs: list[str] = []

    def log(i: int) -> None:
        logs.append(str(i))

    i = SideEffectIterator[int](range(5), log)
    assert tuple(i) == (0, 1, 2, 3, 4)
    assert logs == [str(i) for i in range(5)]

    logs = []

    # Once exhausted, create a new iterator. Trying to use the existing iterator
    # will raise a StopError immediately and return no elements.
    i = SideEffectIterator[str]("damon", lambda x: logs.append(x))
    assert "".join([x for x in i]) == "damon"
    # the iterator of a string will return each character
    assert logs == list(iter("damon"))


def test_list_comprehension() -> None:
    line_list = [" line 1", "", "line 3\n", "\n"]
    assert ["line 1", "line 3"] == [
        y for y in [x.strip() for x in line_list] if len(y) > 0
    ]


def test_genrator_expressions() -> None:
    """A generator expressions return an iterator"""
    line_list = [" line 1", "", "line 3\n", "\n"]

    gen = (y for y in [x.strip() for x in line_list] if len(y) > 0)
    assert isinstance(gen, Iterator)

    assert list(gen) == ["line 1", "line 3"]


def test_nested_functions() -> None:
    """Functions can be nested within functions."""

    def inc_and_add(x: int, y: int) -> int:
        nonlocal one
        one += 1

        x += 1
        y += 1
        return x + y

    one = 1
    two = 2
    assert inc_and_add(one, two) == 5

    # One was mutated from within inc_and_add
    #
    # If you are using the nonlocal and global statements, you are looking for
    # trouble.
    assert one == 2
    assert two == 2


def test_defaults() -> None:
    """Tests functions with default parameters."""

    def fun_defaults(name: str, num: int = 5) -> list[str]:
        """Function arguments can have default values."""
        ret = []
        for i in range(num):
            ret.append(name)
        return ret

    exp = ["damon", "damon"]

    assert ["damon", "damon"] == fun_defaults(name="damon", num=2)

    assert ["damon", "damon", "damon", "damon", "damon"] == fun_defaults(name="damon")


def test_positional_keyword_arguments() -> None:
    def f(one: str, /, two: str, three: str, *, four: str) -> str:
        """When defining functions, two special parameters exist - "/" and "*".

        "/" specifies the prior arguments must be passed by position.
        "*" specifies the following arugments *must* be passed by keyword.

        General guidance
        ----------------
        * Use positional only if you want to hide the param names from the
          caller or want to enforce argument order.
        * Use keyword only when names have meaning and you want to enforce
          the caller specify the param name.
        * For an API, use positional to prevent breaking API changes.
          Positional only allows the param name to change in the future.
        """
        return " ".join([one, two, three, four])

    assert "one two three four" == f("one", "two", "three", four="four")
    assert "one two three four" == f("one", "two", three="three", four="four")
    assert "one two three four" == f("one", two="two", three="three", four="four")


def test_variable_arguments() -> None:
    def fun_varargs(
        *args: Tuple[Any], **kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any], Dict[str, Any]]:
        """Argument handling.

        If a parameter with the form *name is present, it will be set to a tuple
        with any "extra" positional parameters.

        If a parameter with the form **name is present, it will be set to a dict
        with any "extra" keyword parameters.
        """
        return (args, kwargs)

    (args, kwargs) = fun_varargs(1, 2, 3, first="damon", last="allison")

    assert (1, 2, 3) == args
    assert {"first": "damon", "last": "allison"} == kwargs

    args = (1, 2, 3)
    kws = {"first": "damon", "last": "allison"}

    # * unpacks a list/tuple and sends all elements to a function as positional arguments
    # ** unpacks a dict, sending all elements to a function as keyword arguments
    (a, kw) = fun_varargs(*args, **kws)

    assert args == a
    assert kws == kw

    # This will unpack the (0, 2) tuple and send the values as arguments
    # to range()
    assert [0, 1] == list(range(*(0, 2)))


def test_argument_packing() -> None:
    def f(
        *args: tuple[Any], **kwargs: dict[Any, Any]
    ) -> tuple[tuple[Any], dict[Any, Any]]:
        """args and kwargs "pack" all variables into a tuple (args) or dict
        (kwargs).

        When calling the function, the tuple is "unpacked" with * and a
        dictionary is "unpacked" with **.
        """
        return args, kwargs

    # unpacks a tuple (or list) into multiple positional parameters and a dict
    # into multiple keyword parameters.
    (pos, kw) = f(*("one", "two"), **{"one": 1, "two": 2})
    assert pos == ("one", "two")
    assert kw == {"two": 2, "one": 1}


def test_lambdas() -> None:
    """Lambdas are syntactic sugar for a single lien function."""

    def inner_func(val: str) -> str:
        """Example of an inner function"""
        return f"echo {val}"

    assert "echo damon" == inner_func("damon")
    assert "echo damon" == (lambda x: f"echo {str(x)}")("damon")

    assert 200 == (lambda x, y: x * y)(10, 20)


def test_generator() -> None:
    def gen_up_to(val: int):
        """Generator functions are functions which behave like (return)
        iterators.

        Generators are preferable to lists in certain scenarios:

        They are lazy, do not take up memory, could be infinite.

        Generators can only be iterated over once.
        """
        for x in range(val):
            yield x

    # use list or tuple to exhaust the iterator
    assert [0, 1] == list(gen_up_to(2))
    assert (0, 1) == tuple(gen_up_to(2))

    # using a list comprehension to exhaust the iterator
    assert [0, 1] == [x for x in gen_up_to(2)]

    # You can create generator expressions like you would a list
    # comprehension. Generator expressions are wrapped in ()
    square = (x**2 for x in range(3))
    assert (0, 1, 4) == tuple(square)
    assert (0, 1, 4) == tuple(x**2 for x in range(3))


def test_map() -> None:
    def add(x: int, y: int) -> int:
        return x + y

    def is_odd(x: int) -> bool:
        return x % 2 != 0

    assert list(map(add, [1, 2, 3], [2, 2, 2])) == [3, 4, 5]
    assert list(filter(is_odd, map(add, [1, 2, 3], [1, 1, 1]))) == [3]
