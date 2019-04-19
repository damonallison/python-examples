"""An example exception.

Custom exceptions must ultimately derive from Exception, either directly or
indirectly.

Exceptions can do anything classes can do, however they are typically simple,
just containing a small amount of state to relay to anyone handling the
exception.

Most exceptions end in "Error", similar to how the standard exceptions are
named.
"""


class CustomError(Exception):
    """A test exception"""

    def __init__(self, state: str):
        """A default constructor which takes some state.

        Attributes:
            state -- a random state string
        """
        self.state = state