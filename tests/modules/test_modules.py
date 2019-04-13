"""Module / package documentation and tests.


Modules are simply files. A module name is it's file name.

Packages are directory of modules (files) with an __init__.py file.

__init__.py can be empty or initialize the __all__ variable.

__all__ : a list of module names that should be imported when importing *.
    * `__all__ = ["Test", "Test2"]`
    * The following statement will import the Test1 and Test2 modules.
        `from pkg import *`

* Generally, don't use `from pkg import *` unless you are on the command line!
* It can have unwanted side effects.

The module search path:
    * First, look @ built-in modules.
    * Next, look in sys.path.

* The standard library is located in the python installation. An example from homebrew:
    /usr/local/Cellar/python3/3.6.4_2/Frameworks/Python.framework/Versions/3.6/lib/python3.6

* PIP installs to the 'site-packages' folder, which is located at:
    /usr/local/lib/python3.6/site-packages
"""

#
# Imports the `unittest` module in it's entirety.
#
import unittest

#
# Allows us to use an alias rather than having to refer to the entire function
# name:
# tests.modules.calculator.add()
#
import tests.modules.calculator as c

#
# Uses a named import. Makes the objects you import available directly in the
# script. You don't need to prefix the object name when using it.
# add(2, 2)
from tests.modules.calculator import add

#
# This loads the module `exceptions.custom_error`.
# It must be referenced with it's full name.
#
# For Example:
# err = tests.exceptions.custom_error.CustomError(state="oops")
#
import tests.exceptions.custom_error

#
# Import the submodule without it's package prefix.
# It can be used without it's package prefix.
#
# For example:
# err = CustomDerivedError(state="oops")
#
from ..exceptions.custom_derived_error import CustomDerivedError


#
# If you are in a module, you can use an "intra package import"
#
# This will traverse up to this package's parent and traverse down.
#
# Each "." brings you up a parent directory. You do not use "../.." to refer to parents.
#
# from ..package.child import Object
#
class TestModules(unittest.TestCase):
    """Examples showing module imports."""

    @classmethod
    def setUpClass(cls) -> None:
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_module_alias(self):
        # add is imported into this modules directly
        self.assertEqual(4, add(2, 2))

        # it's also available via the 'c' alias
        self.assertTrue(4, c.add(2, 2))

    def test_module_imports(self) -> None:
        """Shows which modules are loaded into the local symbol table."""

        #
        # Because we didn't use the `from` form when importing this module,
        # we must fully qualify it's name.
        #
        self.assertTrue(
            isinstance(
                tests.exceptions.custom_error.CustomError(state="oops"),
                tests.exceptions.custom_error.CustomError),
            msg="Unable to find the CustomError module"
        )

        #
        # Because we imported using the `from` form,
        # we can use the class name directly.
        #
        self.assertTrue(
            isinstance(
                CustomDerivedError(state="oops"),
                CustomDerivedError),
            msg="Unable to find the CustomDerivedError module"
        )

if __name__ == '__main__':
    unittest.main()