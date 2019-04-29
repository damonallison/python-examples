import unittest


class TestControlFlow(unittest.TestCase):

    def test_if_elif(self):

        x = 100

        if x > 10 and x < 50:
            self.fail()
        elif x > 100:
            self.fail()
        else:
            self.assertTrue(x == 100)

    def test_complex_if(self):

        height = 72
        weight = 150

        # You can execute a complex expression
        if 18.5 < weight / height**2 < 25:
            pass
        else:
            pass

        # using logical operators (and, or, not)
        # use parens to disambiguate complex expressions
        self.assertTrue(height == 72 and
                        not weight > 200 and
                        (height > 100 or height < 80))

    def test_is_truthy(self):
        """Here are most of the built-in objects that are considered False in Python:

        * constants defined to be false: None and False
        * zero of any numeric type: 0, 0.0, 0j, Decimal(0), Fraction(0, 1)
        * empty sequences and collections: '"", (), [], {}, set(), range(0)

        Anything else is treated as True - it doesn't have to be a boolean value.
        """

        self.assertFalse({})
        self.assertFalse([])
        self.assertFalse("")
        self.assertFalse(set())
        self.assertFalse(0.0)
        self.assertFalse(0)

        self.assertTrue("test")  # anything non-false will be true
        self.assertTrue([""])

        # Any value can be used in logical statements.
        x = 100
        if x:
            pass
        else:
            self.fail("x is True")

    def test_while_for_else(self):
        """Loop statements (for and while) can also have an else clause.

        The else block is called when the loop finishes naturally. It will *not*
        be invoked when the loop is terminated with `break`"""

        called = False
        iter = 0
        while iter < 10:
            iter += 1
        else:
            called = True

        self.assertEqual(10, iter)
        self.assertTrue(called)

        called = False
        lst = []
        for i in range(5):
            lst.append(i)
        else:
            called = True
        self.assertEqual([0, 1, 2, 3, 4], lst)
        self.assertTrue(called)

        called = False
        for i in range(10):
            if i == 1:
                break
        else:
            called = True

        # `else` will not be called since the iteration was prematurely
        # terminated.
        self.assertFalse(called)

if __name__ == '__main__':
    unittest.main(verbosity=2)