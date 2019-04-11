import unittest


class ListTests(unittest.TestCase):
    """Tests for built in Python data structures.

    list
    tuple
    set
    dictionary
    """

    def test_mutability(self):
        """Lists are mutable "reference" types"""

        a = [1, 2]
        b = a
        a[0] = 3

        self.assertEqual(a, b)
        self.assertEqual(3, b[0])  # a and b point to same object in memory.

    def test_copy(self):
        """Lists are ordered, mutable.

        Lists are not strongly typed. Lists can contain elements
        of multiple types.

        To copy, use [:]. Always use [:] for iteration.
        """

        lst = ["damon", "kari", 10, ["another", "list"]]
        copy = lst[:]

        self.assertFalse(lst is copy,
                         msg="The objects are not referentially equal.")
        self.assertEqual(lst, copy,
                         msg="The lists are logically equivalent")

    def test_list_sorting(self):
        """Example showing max(), min(), sorted()

        max() retrieves the max element (as defined by >)
        min() retrieves the min element (as defined by <)
        sorted() will sort according to <
        """

        a = [10, 20, 1, 2, 3]

        self.assertEqual(20, max(a))
        self.assertEqual(1, min(a))
        self.assertEqual([1, 2, 3, 10, 20], sorted(a))
        self.assertEqual([20, 10, 3, 2, 1], sorted(a, reverse=True))

    def test_list_append(self):
        """Use .append(elt) and .extend(iterable) to append to a list."""

        # Append will add a single value.
        lst = ["damon"]
        lst.append(42)

        # + will concatentate the two lists (use extend() instead for clarity)
        lst = lst + ["cole", 11]
        expected = ["damon", 42, "cole", 11]

        self.assertEqual(expected, lst)

        # .extend(iterable) will append all items from iterable.
        # This is preferred over using `+` since it's clear what
        # you expect.
        expected.extend(["grace", 13])
        lst.extend(["grace", 13])
        self.assertEqual(expected, lst)

    def test_list_joining(self):
        """Joining allows you to combine lists of strings"""

        names = ["damon", "ryan", "allison"]
        self.assertEqual(" ".join(names), "damon ryan allison")

    def test_slicing_sequences(self):
        """Strings are sequences and can be indexed."""

        s = "Damon"
        self.assertEqual(len(s), 5)
        self.assertEqual(s[0:3].lower() + s[-1].lower(), "damn")

    def test_iteration(self):
        """`for` iterates over the elements in a sequence."""

        lst = ["damon", "kari", "grace", "lily", "cole"]
        expected = []

        # Remember to always iterate over a *copy* of the list
        for name in lst[:]:
            expected.append(name)
        self.assertEqual(expected, lst)

        # To iterate over the indices of a sequence, use range(len(lst))
        expected = []
        for i in range(len(lst)):
            expected.append(lst[i])
        self.assertEqual(expected, lst)

        # To iterate over indices and values simultaneously, use enumerate()
        pos = []
        val = []
        for i, v in enumerate(["tic", "tac", "toe"]):
            pos.append(i)
            val.append(v)

        self.assertEqual([0, 1, 2], pos)
        self.assertEqual(["tic", "tac", "toe"], val)

        # Loop statements can have an `else` clause, which executes
        # when the loop terminates without encoutering a `break` statement
        primes = []
        for n in range(2, 6):
            for x in range(2, n):
                if n % x == 0:
                    break
            else:
                primes.append(n)
        self.assertEqual([2, 3, 5], primes)


        # zip() allows you to loop multiple sequences simultaneously.
        # zip() will stop when any list is exhausted.
        questions = ["who", "what", "when", "where", "why"]
        answers = ["humperdink", "potion", "mideival", "sweden"]
        q2 = []
        a2 = []
        for q, a in zip(questions, answers):
            q2.append(q)
            a2.append(a)

        self.assertEqual(["who", "what", "when", "where"], q2)
        self.assertEqual(answers, a2)

    def test_list_comprehensions(self):
        """List comprehensions are a concise way to create lists."""

        squares = [x**2 for x in range(1, 4)]
        self.assertEqual([1, 4, 9], squares)

        evens = [x for x in range(1, 11) if x % 2 == 0]
        self.assertEqual([2, 4, 6, 8, 10], evens)

        squares = [(x, x ** 2) for x in [0, 1, 2, 3]]

        for x in range(len(squares)):
            self.assertEqual(squares[x][0], x)
            self.assertEqual(squares[x][1], x**2)
