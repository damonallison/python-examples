""" A class which prints things.

Used to show python's support for multiple inheritance.
"""
from typing import List


class Logger:
    def __init__(self) -> None:
        self.logs: list[str] = []

    def log(self, val: str) -> None:
        self.logs.insert(0, val)

    def history(self) -> List[str]:
        h = self.logs[:]
        self.__original_log("retrieved history")
        return h

    # Python supports "name mangling" - which can be used to keep a pointer to
    # an original method definition. Even if the method was overridden in a
    # superclass, the original method can be called.
    #
    # Here, we store a copy of the original log function. If log() is
    # overridden, history() will still refer to the original log function.
    __original_log = log
