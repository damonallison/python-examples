import unittest


class TestOperators(unittest.TestCase):
    """Python operator examples"""

    def test_arithmetic_operators(self):
        """Python arithmetic operators

        +  addition
        -  subtraction
        *  multiplication
        /  division (always returns a floating point value)
        %  modulus
        ** exponent
        // floor division - always rounds down to next whole integer
                            (negative too)
        """

        # Python handles prescedence within expressions
        val = 2 + 2 * 6
        self.assertEqual(14, val)

        # Division returns the exact value, not integer
        self.assertAlmostEqual(4 / 3, 1.33, 2)
        self.assertAlmostEqual(4.0 / 3.0, 1.33, 2)

        # // will always round *down* to the nearest integer
        self.assertEquals(7 // 2, 3)
        self.assertEquals(-7 // 2, -4)

        # Exponentiation
        self.assertEqual(3 ** 3, 27)

        # Modulo
        self.assertAlmostEqual(4.4 % 1, .4, 2)

    def test_assignment_operators(self):

        # Multiple assignment
        x, y = 10, 20

        self.assertEqual(10, x)
        self.assertEqual(20, y)

        # All arithmetic operators have a corresponding assignment equivalent.
        #
        # += -= *= /=

        val = 10
        val += 10
        self.assertEqual(val, 20)

        val -= 10
        self.assertEqual(val, 10)

        val *= 10
        self.assertEqual(val, 100)

        val **= 2
        self.assertEqual(val, 10000)

        val /= 800
        self.assertAlmostEquals(val, 12.5, 1)

    def test_comparison_operators(self):

        x = True

        self.assertEqual(bool, type(x))
        self.assertEqual(True, x)
        # bool is also equal to 1 and 0.
        self.assertEqual(1, x)

        # bool can also be used in logical comparison operators
        self.assertTrue(x == 1)

        # bool are used in logical operators
        # and
        # or
        # not (inverses a boolean condition)
        self.assertTrue(x == 1 and x and x < 10)
