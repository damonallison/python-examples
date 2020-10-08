import unittest


class FunctionTests(unittest.TestCase):

    def test_nested_functions(self):
        """Functions can be nested within functions."""
        def add(x: int, y: int):
            return x + y

        self.assertEqual(4, add(2, 2))

    def fun_defaults(self, name: str, num: int = 5) -> [str]:
        """Function arguments can have default values."""
        ret = []
        for i in range(num):
            ret.append(name)
        return ret

    def fun_keyword_args(self, *args: [str], **kwargs: {str: str}) -> ([str], {str: str}):
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

    def test_positional_keyword_params(self) -> None:

        def f(one: str, /, two: str, three: str, *, four: str) -> None:
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
            print("positional_params", one, two, three, four)

        f(1, "two", "three", four="four")
        f("one", "two", three="three", four="four")
        f("one", two="two", three="three", four="four")

    def test_defaults(self) -> None:
        """Tests functions with default parameters."""
        exp = ["damon", "damon"]
        self.assertEqual(exp, self.fun_defaults(name="damon", num=2))

        self.assertEqual(["damon", "damon"],
                         self.fun_defaults("damon", num=2))

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

        self.assertDictEqual(
            kwargs,
            {"first": "damon", "last": "allison"})

        self.assertDictEqual(
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
        #
        # Lists and tuples are unpacked with *, while dictionaries are unpacked
        # with **
        (a, kw) = self.fun_keyword_args(*args, **kws)
        self.assertEqual(args, a)
        self.assertEqual(kws, kw)

        # This will unpack the (0, 2) tuple and send the values as arguments
        # to range()
        self.assertEqual([0, 1], list(range(*(0, 2))))

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
        # comprehension. Generator expressions tend to be more memory friendly
        # than equivalent list comprehensions.
        gen = (x**2 for x in range(3))
        self.assertTupleEqual((0, 1, 4), tuple(gen))
