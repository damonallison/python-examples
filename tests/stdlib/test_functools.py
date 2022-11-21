"""functools - higher order functions and operations on callable objects

https://docs.python.org/3/library/functools.html

Callable objects are is an object which behaves like a function, but may not be
a function. For example, a class with a __call__ method is a callable. For
example, `enumerate` is a callable object.

functools.lru_cache performs memoization. Note that the lru_cache caches results
in a dictionary, the positional and kwargs must be hashable. Ensure you set an
upper bound, or realize the cache will grow unbounded.
"""
from typing import Callable

import functools


def test_callable() -> None:
    class MyGreeter:
        def __init__(self, name: str) -> None:
            self.name = name

        def __call__(self, greeting: str) -> str:
            return f"{greeting}, {self.name}"

    mg = MyGreeter("damon")
    assert mg("hello") == "hello, damon"


def test_cache() -> None:
    calls: list[str] = []

    # cache is memoization. same as functools.lru_cache(max_size=None)
    @functools.cache
    def echo(value: str) -> str:
        calls.append(value)
        return value

    assert echo("damon") == "damon"
    assert len(calls) == 1

    assert echo("damon") == "damon"
    assert len(calls) == 1


def test_cached_property() -> None:
    calls: list[str] = []

    class Person:
        def __init__(self, fname: str, lname: str) -> None:
            self.fname = fname
            self.lname = lname

        # functools.cached_property turns a class method into a property whose
        # value is computed once. Great for immutable properties which are
        # expensive to calculate (i.e., database and/or network operations)
        @functools.cached_property
        def full_name(self) -> str:
            full = f"{self.fname} {self.lname}"
            calls.append(full)
            return full

    p = Person("damon", "allison")

    assert p.full_name == "damon allison"
    assert len(calls) == 1

    assert p.full_name == "damon allison"
    assert len(calls) == 1


def test_partial() -> None:
    def echo_name(fname: str, lname: str) -> str:
        return f"{fname} {lname}"

    p = functools.partial(echo_name, lname="allison")
    assert p(fname="damon") == "damon allison"
    assert p(fname="kari") == "kari allison"


def test_wraps() -> None:
    calls: list[str] = []

    def my_decorator(f: Callable[[str], str]) -> Callable[[str], str]:
        @functools.wraps(f)
        def wrapper(arg: str) -> str:
            calls.append(arg)
            return f(arg)

        return wrapper

    @my_decorator
    def example(name: str) -> str:
        "example docstring"
        return f"your name is {name}"

    example("damon") == "your name is damon"
    len(calls) == 1
    calls[0] == "damon"

    # The metadata associated with my_decorator will be that of the wrapped
    # function, not my_decorator

    assert example.__name__ == "example"
    assert example.__doc__ == "example docstring"
