import logging
import functools
from typing import Any

"""Decorators allow you to wrap function invocation, extending it's behavior. Or
wrap a class, changing the way __init__ is handled (i.e., singleton).

Decorators are useful for:

* Logging
* Authentication
* Setup / teardown
* Metrics

Decorators can be applied at the function or class level.

Builtin decorators include:

@classmethod - accepts a cls parameter and can use / modify class state. The
intrepreter injects the class parameter automatically. Class methods could be
used for creating a factory.

@staticmethod - flags the function as a static method. Python allows you to call
a static method from a class instance. Python will *not* add `self` to the
method invocation.

Function decorators receive functions as their arguments. Class decorators
receive classes as their argument.

"""

# logtimes calls a function a repeated number of times, wrapping each invocation
# with log statements.
#
# logtimes is a function that *returns* a decorator.
#
# When decorators are used *without* parentheses, the decorator is returned
# directly. See @logtwice
#
# When decorators are used *with* parentheses, a decorator must be returned from
# the function invocation. See @logtimes(num_times = 2)
def logtimes(num_times: int = 4):
    def logfn(func):
        # functools.wraps preserves the original function data for introspection.
        @functools.wraps(func)
        def withlog(*args, **kwargs):
            for _ in range(num_times):
                logging.info("starting")
                ret = func(*args, **kwargs)
                logging.info("ending")
            return ret

        return withlog

    return logfn


def runtwice(func):
    @functools.wraps(func)
    def dec(*args, **kwargs):
        for _ in range(2):
            logging.info("running")
            ret = func(*args, **kwargs)
        return ret

    return dec


# count_calls returns a class instance as a decorator. This is idiomatic when
# you want to maintain state within the decorator.
#
# You must implement the __call__ method, which is invoked when the decorated
# function is invoked.
def count_calls(up_to: int = 10):
    class CountCalls:
        def __init__(self, func):
            functools.update_wrapper(self, func)
            self.func = func
            self.num_calls = 0

        def __call__(self, *args, **kwargs):
            self.num_calls += 1
            return self.func(*args, **kwargs)

    return CountCalls


# Decorators can be applied to classes. Rather than receiving a `func` argument,
# it receives a `cls` argument.
def singleton(cls):
    @functools.wraps(cls)
    def wrapper_singleton(*args, **kwargs):
        if not wrapper_singleton.instance:
            wrapper_singleton.instance = cls(*args, **kwargs)
        return wrapper_singleton.instance

    wrapper_singleton.instance = None
    return wrapper_singleton


@logtimes(num_times=2)
def echowithlog(name: str) -> str:
    return f"echoing {name}"


@runtwice
def echotwice(name: str) -> str:
    return f"echoing {name}"


@count_calls(up_to=5)
def say_hello(name: str) -> str:
    return f"hello {name}"


class Person:
    # Class state
    greeting = "hello"

    def __init__(self, fname: str, lname: str, section: str) -> None:
        self.fname = fname
        self.lname = lname
        self.section = section

    @classmethod
    def section1(cls: Any, fname: str, lname: str) -> Any:
        # TODO: what annotation should be used for `cls` and `return`?
        return cls(fname, lname, "1")

    def full_name(self) -> str:
        return f"{self.fname} {self.lname}"

    @staticmethod
    def greet_static(name: str) -> None:
        return f"{Person.greeting} {name}"


@singleton
class Controller:
    def __init__(self, name: str):
        self.name = name


class TestDecorators:
    """Decorators wrap a function, modifying it's behavior."""

    def test_func_decorators(self, caplog) -> None:
        caplog.set_level(logging.INFO)
        assert echotwice("damon") == "echoing damon"

        assert len(caplog.records) == 2
        assert caplog.records[0].message == "running"
        assert caplog.records[1].message == "running"

    def test_func_decorators_with_params(self, caplog) -> None:
        caplog.set_level(logging.INFO)
        assert echowithlog("damon") == "echoing damon"

        assert len(caplog.records) == 4
        assert caplog.records[0].message == "starting"
        assert caplog.records[1].message == "ending"
        assert caplog.records[2].message == "starting"
        assert caplog.records[3].message == "ending"

    def test_func_class_decorators(self) -> None:
        # Using a factory @classmethod
        cole = Person.section1("cole", "allison")
        assert cole.section == "1"

        # Using a @staticmethod from an object instance
        damon = Person.section1("damon", "allison")
        assert damon.greet_static(damon.full_name()) == "hello damon allison"

    def test_class_decorators(self) -> None:
        c = Controller("damon")
        assert c.name == "damon"
        c = Controller("cole")
        assert c.name == "damon"

    def test_stateful_decorators(self) -> None:
        assert say_hello("damon") == "hello damon"
        assert say_hello("cole") == "hello cole"
        assert say_hello.num_calls == 2
