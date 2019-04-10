import unittest

class TestString(unittest.TestCase):
    """String tests"""

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

    def test_string_1(self):
        x = "damon"
        self.assertEqual("damon", x, msg="wtf")
