"""Module / package documentation and tests.


Modules are simply files. A module name is it's file name (without the .py
extension).

A package is a directory of modules (files) with an __init__.py file.

__init__.py can be empty or initialize the __all__ variable.

__all__ A list of modules that will be imported when `from <module> import *` is
used. Assume the following is entered into a `pkg` file:

    __all__ = ["test", "test2"]

    # This will import both the "test" and "test2" modules locally

    from pkg import *

Generally, **don't use** `from pkg import *` unless you are on the command line!
You will blindly import symbols locally, which may overwrite other symbols.

Modules are searched for in the following order:

1. Built-in modules (sys)
2. sys.path - sys.path is built from
   1. The directory containing the __main__ script
   2. $PYTHONPATH
   3. Installation dependent default

* The standard library is located in the python installation. An example from
  homebrew:
  /usr/local/Cellar/python3/3.6.4_2/Frameworks/Python.framework/Versions/3.6/lib/python3.6

* PIP installs to the 'site-packages' folder, which is located at:
  /usr/local/lib/python3.6/site-packages
"""

#
# Imports the `unittest` module. Since this module is part of the stdlib, it
# will be found in sys.path
#
import unittest

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
import tests.modules.calculator as c

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
# You can use relative paths to import submodules. These imports use "." to
# indicate the current and parent packages involved in the relative import.
#
from .calculator import add
from .pkg1.mod1 import Mod1Calculator
from .pkg2.mod2 import Mod2Calculator
from ..exceptions.custom_derived_error import CustomDerivedError


class TestModules(unittest.TestCase):
    """Examples showing module imports."""

    def test_module_imports(self):
        self.assertTrue(
            isinstance(
                tests.exceptions.custom_error.CustomError(state="oops"),
                tests.exceptions.custom_error.CustomError))

    def test_module_alias(self):
        """We imported the calculator module with alias c"""
        self.assertTrue(4, c.add(2, 2))

    def test_intra_package_references(self):
        self.assertEqual(4, add(2, 2))
        self.assertEqual(4, Mod1Calculator().add(2, 2))
        self.assertEqual(4, Mod2Calculator().add(2, 2))
        self.assertTrue(
            isinstance(CustomDerivedError(state="oops"), CustomDerivedError))

if __name__ == '__main__':
    unittest.main()