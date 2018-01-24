"""Tests for Python I/O"""
import unittest
import os.path


class TestIO(unittest.TestCase):
    """Tests for Python I/O."""

    testFileName = "test.txt"

    @classmethod
    def setUpClass(cls) -> None:
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def setUp(self) -> None:
        if os.path.exists(self.testFileName):
            os.remove(self.testFileName)

    def tearDown(self) -> None:
        # Ensure any resources created by this test suite are deleted.
        if os.path.exists(self.testFileName):
            os.remove(self.testFileName)

    def test_basic_printing(self) -> None:
        """str() == human readable output. repr() == interpreter readable output"""

        first_name = "damon"
        last_name = "allison"

        self.assertTrue(str(first_name) == "damon")
        self.assertTrue(repr(first_name) == "'damon'")

        #
        # format using positional arguments
        #
        self.assertEqual("damon allison", "{0} {1}".format(first_name, last_name))

        #
        # format using keyword arguments
        #
        self.assertEqual("damon allison", "{first} {last}".format(first=first_name, last=last_name))
        self.assertEqual("$100 is $100.00", "${var} is ${var:.2f}".format(var=100))

    def test_reading_writing_files(self) -> None:
        """With will automatically close the file"""

        # Adding the newlines is extremely lame.
        lines = ["Hello\n", "World\n"]

        with open(self.testFileName, mode="w") as f:
            f.writelines(lines)

        with open(self.testFileName, mode="r") as f:
            readLines = f.readlines()

        self.assertEqual(lines, [x for x in readLines])


if __name__ == '__main__':
    unittest.main()
