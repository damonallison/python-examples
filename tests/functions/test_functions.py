from typing import Any, List, Dict, Tuple
import unittest


def test_nested_functions() -> None:
    """Functions can be nested within functions."""
    def add(x: int, y: int):
        return x + y

    assert 4 == add(2, 2)


def test_defaults() -> None:
    """Tests functions with default parameters."""

    def fun_defaults(name: str, num: int = 5) -> List[str]:
        """Function arguments can have default values."""
        ret = []
        for i in range(num):
            ret.append(name)
        return ret

    exp = ["damon", "damon"]

    assert ["damon", "damon"] == fun_defaults(name="damon", num=2)

    assert ["damon", "damon", "damon", "damon",
            "damon"] == fun_defaults(name="damon")


def test_positional_keyword_arguments() -> None:
    def f(one: str, /, two: str, three: str, *, four: str) -> str:
        """When defining functions, two special parameters exist - "/" and "*".

            "/" specifies the prior arguments must be passed by position.
            "*" specifies the following arugments *must* be passed by keyword.

            General guidance
            ----------------
            * Use positional only if you want to hide the param names from the caller
                or want to enforce argument order.
            * Use keyword only when names have meaning and you want to enforce the
                caller specify the param name.
            * For an API, use positional to prevent breaking API changes. Positional only
                allows the param name to change in the future.
            """
        return " ".join([one, two, three, four])

    assert "one two three four" == f("one", "two", "three", four="four")
    assert "one two three four" == f("one", "two", three="three", four="four")
    assert "one two three four" == f(
        "one", two="two", three="three", four="four")


def test_variable_arguments() -> None:
    def fun_varargs(*args: Tuple[Any], **kwargs: Dict[str, Any]) -> Tuple[Tuple[Any], Dict[str, Any]]:
        return (args, kwargs)

    (args, kwargs) = fun_varargs(1,
                                 2,
                                 3,
                                 first="damon",
                                 last="allison")

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


def test_lambdas() -> None:
    """Python lambdas are restricted to a single statement.
    They are syntactic sugar for a function definition.
    """

    def inner_func(val: str) -> str:
        """Example of an inner function"""
        return f"echo {val}"

    assert "echo damon" == inner_func("damon")
    assert "echo damon" == (lambda x: f"echo {str(x)}")("damon")

    assert 200 == (lambda x, y: x * y)(10, 20)


def test_generator() -> None:
    def gen_up_to(val):
        """Generators are preferable to lists in certain scenarios.

        They are lazy, do not take up memory, could be infinite.

        They can only be iterated over once.
        """
        for x in range(val):
            yield x

    assert [0, 1] == list(gen_up_to(2))

    # You can create generator expressions like you would a list
    # comprehension. Generator expressions tend to be more memory friendly
    # than equivalent list comprehensions.
    gen = (x**2 for x in range(3))
    assert (0, 1, 4) == tuple(gen)
