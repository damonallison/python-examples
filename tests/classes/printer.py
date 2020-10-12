""" A class which prints things.

Used to show python's support for multiple inheritance.
"""


class Printer:

    def __init__(self):
        self.print_history = []

    def log(self, val):
        self.print_history.insert(0, val)

    def print(self, val):
        """Probably not good form to use the name of a built-in here"""

        # By using name mangling, we can keep a reference to an original method.
        # Even if a sublcass overrides (log), we have a pointer to the
        # original.
        self.__original_log(val)
        print(val)

    # Python supports "name mangling" - which can be used to allow a
    # base class to keep a pointer to an original method definition. This
    # allows us to keep a copy of a method.
    __original_log = log
