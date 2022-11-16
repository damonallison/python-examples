# flake8

```python
x = 1  # noqa: E371
```

# pylint

```python
x = 1 . # pylint: disable=one,two
```


# mypy

```python
x = 1 . # type: ignore
```

Typshed contains type stubs from the
[typeshed](https://github.com/python/typeshed) project for builtins, the
standard library, and selected 3rd party packages.


```python

from typing import Sequence, Union, List, Dict, Optional

def greet_all(names: Iterable[str]) -> None:
    '''names can be any iterable.

    Iterable[] more flexible than accepting List[str].'''
    for n in names:
        print(n)

def normalize_id(user_id: Union[int, str]) -> str:
    if isinstance(user_id, int):
        return 'user-{}'.format(100000 + user_id)
    else:
        return user_id

def greeting(name: Optional[str] = None) -> str:
    # Optional[str] means the same thing as Union[str, None]
    if name is None:
        name = 'stranger'
    return 'Hello, ' + name


```