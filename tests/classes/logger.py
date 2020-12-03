""" A class which prints things.

Used to show python's support for multiple inheritance.
"""
from typing import List


class Logger:
    def __init__(self):
        self.history = []

    def log(self, val: str) -> None:
        self.history.insert(0, val)

    def history(self) -> List[str]:
        h = self.history[:]
        self.__original_log("retrieved history")
        return h

    # Python supports "name mangling" - which can be used to allow a
    # base class to keep a pointer to an original method definition. This
    # allows us to keep a copy of a method.
    __original_log = log
