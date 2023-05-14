"""Module / package documentation and tests.


Modules are simply files. A module name is it's file name (without the .py
extension). A package is a directory of modules (files) with an __init__.py
file.

Modules: https://docs.python.org/3/tutorial/modules.html

Each module has a global symbol table. Importing a module adds that module to
the current module's global symbol table.

There are two forms of the import statement. You can import other modules in
their entirety or individual members of a module.

This will import the module in its entirety into your module:

```
import mod [as alias]
```

The following form will import either a module or an individual member from
within a module. `import` first tests whether a member (name) is defined in the
given module. If so, it imports that member. If not, it assumes the member is a
module and attempts to load it.

```
from pkg.subpkg import name [as n]
```

There is a special form of import (`import *)` that allows you to import a set
of modules from a given package. The set of modules that will be imported are
defined in the package's __init__.py file in a special __all__ variable.

```
from pkg import *
```

Assume that pkg.py has __all__defined as:

__all__ = ["test", "test2"]

Generally, **don't use** `from pkg import *` unless you are on the command line!
You will blindly import modules locally, which may overwrite other modules.


Module search path:

1. Built-in modules (sys)
2. sys.path - sys.path is built from
   1. The directory containing the __main__ script
   2. $PYTHONPATH
   3. Installation dependent defaults

--

* The standard library is located in the python installation. An example from
  homebrew:
  /usr/local/Cellar/python3/3.6.4_2/Frameworks/Python.framework/Versions/3.6/lib/python3.6

* PIP installs to the 'site-packages' folder, which is located at:
  /usr/local/lib/python3.6/site-packages
"""

import pytest

#
# Import a module
#
# This imports the module `exceptions.custom_error`.
# It must be referenced with it's full name.
#
# For Example:
# err = tests.exceptions.custom_error.CustomError(state="oops")
#
import tests.exceptions.custom_error

#
# Import a module with an alias
#
# Using "as" allows us to use an alias rather than having to refer to the
# entire function name: tests.modules.calculator.add()
#
import tests.modules.calculator as calc

#
# Import a module or module member.
#
# You can also apply an optional alias.
#
# Allows you to use "CustomError" without the entire package.
#
from tests.exceptions.custom_error import CustomError  # as CE

#
# Intra-package (relative) references
#
# You can use relative paths to import submodules. These imports use "." and ".." to
# indicate the current and parent packages involved in the relative import.
#
from .calculator import add
from .pkg1.mod1 import Mod1Calculator
from .pkg2.mod2 import Mod2Calculator
from ..exceptions.custom_derived_error import CustomDerivedError

#
# Note that when importing module level variables,
# you must import the module and reference the variable the module reference
#
# YES:
# from pkg import mod
# mod.GLOBAL_VAR += 1
#
# NO:
# from pkg.mod import GLOBAL_VAR
#
# Using `from .classes.person import name_call_count` will create a new local
# `name_call_count` variable setting it's value to the current value of the
# person module's value - not what you want!
#
from tests.modules.pkg1 import mod1
from tests.modules.pkg2 import mod2

#
# Here, a local `call_count` variable is created. Because `call_count` is a
# primitive type, a copy of the value is added to this module's namespace.
# Therefore, `call_count` does *not* reference `.pkg1.mod1.call_count` as you
# would expect.
#
from tests.modules.core.appconfig import CALL_COUNT


def test_module_imports() -> None:
    assert isinstance(
        tests.exceptions.custom_error.CustomError(state="oops"),
        tests.exceptions.custom_error.CustomError,
    )
    assert isinstance(CustomError(state="oops"), CustomError)


def test_module_alias() -> None:
    """We imported the calculator module with alias c"""
    assert 4 == calc.add(2, 2)


def test_global_variable_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Because call_count is imported directly into this module,
    call_count actually points to a *new* variable in this module's
    namespace. It is *not* the same as the call_count variable in mod1"""

    CALL_COUNT = 0
    mod1.CALL_COUNT = 0
    mod2.CALL_COUNT = 0

    assert CALL_COUNT == 0
    assert mod1.CALL_COUNT == 0
    assert mod2.CALL_COUNT == 0

    mod1.CALL_COUNT = 1

    assert CALL_COUNT == 0
    assert mod1.CALL_COUNT == 1
    assert mod2.CALL_COUNT == 0

    Mod1Calculator().add(2, 2)

    assert CALL_COUNT == 0
    assert mod1.CALL_COUNT == 2
    assert mod2.CALL_COUNT == 0


def test_function_import() -> None:
    # add was added directly to our module's namespace
    assert add(2, 2) == 4

    assert Mod1Calculator().add(2, 2) == 4
    assert Mod2Calculator().add(2, 2) == 4
    assert isinstance(CustomDerivedError(state="oops"), CustomDerivedError)
