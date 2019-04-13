import unittest

#
# Module level (global) scope
#
value = 100


class FunctionTests(unittest.TestCase):
    """Examples showing python scoping"""
    #
    # Class level (enclosing scope) variable
    #
    value = 10

    def test_scopes(self) -> None:
        """Python has 3 scopes.

        local scope
        class scope
        global scope

        This test shows how scoping works."""

        def local_scope():
            """Value is created locally.

            Class and module level `value`+ objects are not touched.
            """
            value = 100
            self.assertEqual(100, value)

        def nonlocal_scope():
            """self.value is bound to the class level (enclosing scope)."""
            self.value = self.value + 1

        def global_scope():
            """Value is bound to the module level (global) `value` instance."""
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
        nonlocal_scope()
        self.assertTrue(11, self.value)
        self.assertTrue(100, value)

        # Global updates module level `value`.
        global_scope()
        self.assertTrue(11, self.value)
        self.assertTrue(101, value)

    def fun_defaults(self, name: str, num: int=5) -> [str]:
        """Function arguments can have default values."""

        ret = []
        for i in range(0, num):
            ret.append(name)
        return ret

    def fun_keyword_args(self, *args: [str], **kwargs: {str : str}) -> ([str], {str : str}):
        """Functions can take arbitrary numbers of arguments and keyword arguments.

        Normally, *args is the last parameter. Anything after *args must be keyword arguments.
        """

        a = []
        for arg in args:
            a.append(arg)

        kw = {}
        for kwarg in kwargs:
            kw[kwarg] = kwargs[kwarg]

        return (a, kw)

    def test_defaults(self) -> None:
        """Tests functions with default parameters."""

        self.assertEqual(["damon", "damon"], self.fun_defaults("damon", num=2))
        self.assertEqual(["damon", "damon", "damon", "damon", "damon"],
                         self.fun_defaults("damon"))

    def test_args(self) -> None:
        """Tests functions with variable arguments and
        variable keyword arguments.
        """

        (args, kwargs) = self.fun_keyword_args(1,
                                               2,
                                               3,
                                               first="damon",
                                               last="allison")

        self.assertEqual([1, 2, 3], args)

        self.assertEqual(
            kwargs,
            {"first": "damon", "last": "allison"})

        self.assertEqual(
            kwargs,
            {"last": "allison", "first": "damon"},
            msg="Dictionary ordering doesn't matter for equality checks")

    def test_unpacking_tuple(self) -> None:
        """Lists and dictionaries can be unpacked
        and sent into a function as parameters.
        """
        args = [1, 2, 3]
        kws = {"first": "damon", "last": "allison"}

        # Unpacks args using * and ** and send to `fun_keyword_args`
        (a, kw) = self.fun_keyword_args(*args, **kws)
        self.assertEqual(args, a)
        self.assertEqual(kws, kw)

    def test_lambdas(self) -> None:
        """Python lambdas are restricted to a single statement.
        They are syntactic sugar for a function definition.
        """

        def inner_func(val: str) -> str:
            """Example of an inner function"""
            return f"echo {val}"

        self.assertEqual("echo damon", inner_func("damon"))
        self.assertEqual("echo damon", (lambda x: f"echo {str(x)}")("damon"))

        self.assertEqual(200, (lambda x, y: x * y)(10, 20))

    def gen_to(self, val):
        """Generators are preferable to lists in certain scenarios.

        They are lazy, do not take up memory, could be infinite.

        They can only be iterated over once.
        """
        for x in range(val):
            yield x

    def test_generator(self):
        lst = []
        for x in self.gen_to(2):
            lst.append(x)
        self.assertListEqual([0, 1], lst)

        # You can create generator expressions like you would a list
        # comprehension
        gen = (x**2 for x in range(3))
        self.assertTupleEqual((0, 1, 4), tuple(gen))

