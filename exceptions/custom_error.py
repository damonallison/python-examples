"""An example exception"""

class CustomError(Exception):
    """A test exception"""

    def __init__(self, state: str):
        """A default constructor which takes some state.

        Attributes:
            state -- a random state string
        """
        self.state = state