import unittest


class TestUdacityIntroToPythonLesson7(unittest.TestCase):
    """Scripting"""

    def test_scripting(self):
        how_many_snakes = 1
        snake_string = """
Welcome to Python3!

            ____
            / . .\\
            \  ---<
            \  /
__________/ /
-=:___________/

<3, Juno
"""
        print(snake_string * how_many_snakes)

if __name__ == "__main__":
    unittest.main()
