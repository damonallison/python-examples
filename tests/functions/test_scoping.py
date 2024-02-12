"""Python scopes and namespaces:

Namespaces
----------

A namespace is a mapping from names to objects. Examples of namespaces include
built-ins, modules, and functions.

* (built-in) The set of built-in names.
* (module) Global names in a module.
* (function) Local names in a function.

Scopes
------

Scopes are textual regions of a Python program where a namespace is directly
accessible. There are at least 3 scopes whose namespaces are directly
accessible.

* Inner most scope of a function (local).
* The scopes of any enclosed functions (nonlocal).
* The Module's global names (global).
* Built-ins

If no `global` statement is provided, assignments to names always go in the
innermost scope.

The `global` statement can be used to indicate that particular variables live in
the global scope and should be rebound there. The `nonlocal` statement indicates
that particular variables live in an enclosing scope and should be rebound
there.
"""

import unittest

#
# Module level (global) scope.
#
# Even though you *can* use global scope, just don't. Consider encapsulating
# state into an object.
#
value = 100
module_var = 1


class ScopeTest:
    value = 10

    def get_value(scope: str) -> str:
        if scope.lower().find("global") >= 0:
            return globals()["value"]
        return value


def test_scope() -> None:
    pass


# TODO: finish me...


class ScopingTests(unittest.TestCase):
    """Examples showing python scoping"""

    #
    # Class level (enclosing scope) variable
    #
    value = 10

    def test_scopes(self) -> None:
        """Python has at least three, usually four, nested scopes whose namespaces
           are directly accessible.

        * The innermost (typically function) scope.
        * The scopes of any enclosing functions.
        * The module's "global" scope.
        * The built-in namespace.

        Keep in mind that "global" is at a module context.

        This test shows how scoping works.
        """

        def local_scope():
            """A new `value` variable is created in the local scope."""

            value = 1000
            self.assertEqual(1000, value)

        def class_scope():
            self.value += 1

        def global_scope():
            """The global statement indicates that the variable lives
            in the global (module) scope and should be rebound here."""

            global value
            value = value + 1

        # Initial values
        self.assertTrue(10, self.value)
        self.assertTrue(100, value)

        # Local scope does not touch class or module level `value`.
        local_scope()
        self.assertTrue(10, self.value)
        self.assertTrue(100, value)

        # Nonlocal updates class level `value`.
        class_scope()
        self.assertTrue(11, self.value)
        self.assertTrue(100, value)

        # Global updates module level `value`.
        global_scope()
        self.assertTrue(11, self.value)
        self.assertTrue(101, value)

    def test_nonlocal(self) -> None:
        """Nonlocal indicates a particular variable is lives
        in an enclosing scope and should be rebound locally."""
        x = 10

        class MyClass:
            name = "test"

            def increment_non_local(self):
                """x lives in the test_nonlocal() scope. `nonlocal`
                rebinds it here."""

                nonlocal x
                x += 1

            def increment_global(self):
                """module_var is declared at the global (module) scope.
                `global` rebinds it here.

                If `module_var` was *not* already a variable, `global`
                would have created it at the global level.”
                """

                global module_var
                module_var += 1

        mc = MyClass()
        mc.name = "damon"
        self.assertEqual("damon", mc.name)

        self.assertEqual(10, x)
        mc.increment_non_local()
        self.assertEqual(11, x)

        self.assertEqual(1, module_var)
        mc.increment_global()
        self.assertEqual(2, module_var)
