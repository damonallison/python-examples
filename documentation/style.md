# Python Style Guide Cliff Notes

[Google's Python Style Guide](https://google.github.io/styleguide/pyguide.html)

* Import packages and modules only, not individual classes and/or functions.
* Don't use relative imports. Use full package names.

## Comments

```python
# TODO(@damon): The doer should be in (parens). The goal is a greppable, consistent way to find TODOs.
```

## Docstrings

```python
"""Common math functions.

Every module should have license boilerplate and contain an overall description
in the module's docstring.

    Example:

    foo = ClassFoo()
    x = foo.some_func()


"""
def add(x: float, y: float) -> float:
    """Adds two numbers.

    The first line should be in "descriptive style" like "Fetches data from a database."
    rather than imperative "Fetch data from a database.".

    Args:
        x: The first value to add. If the docstring for a parameter is long,
          the next line should be indented two or four characters more than
          the parameter name, like this does.
        y: The second value to add. Note that types are not needed in the
          description if the types are annotated as part of the function.

    Returns:
        The sum of the arguments.

    Raises:
        IOError: If this function actually raised an IOError, you'd describe
          why it does here.
    """
    return x + y

class MyClass:
    """An example class.

    Attributes should be listed in the same fashion as args.

    Attributes:
        likes_spam: A flag indicating if the user likes spam.
        eggs: The number of eggs the user has eaten.

    """
    def __init__(self, likes_spam: bool = False):
        self.likes_spam = likes_spam
        self.eggs = 0
```
## pylint

Use it

```python
# Disable individual warnings i.e., # pylint: disable=...
dict = 'something awful'  # Bad Idea... pylint: disable=redefined-builtin
```

## Exceptions

* Use `ValueError` to enforce correct parameter values. Use `assert` only to
  enforce internal invariants.

## Global Variables

* Avoid them. If necessary, wrap state in function accessors (mockability).
* Module level constants are encouraged.

## Decorators

* Don't use `staticmethod`. Use `classmethod` judiciously.
* Decorators evaluate on import. Ensure decorators are rock solid and don't rely
  on external state (files, etc)

## Logging

For logging functions that expect a pattern-string (with `%` placeholders), call
them with a string literal, not an f string. Some logging frameworks collect the
unexpanded pattern string as a queryable field. Also, you won't spend time
concatenating a string that the logger won't output.

```python

logger = logging.get_logger(self.__name__)

# Don't use an f-string with loggers
logger.error("Cannot write to home directory at %r", homedir)
```
