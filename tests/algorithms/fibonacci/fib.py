import logging

from typing import List

"""Finonacci"""


def fib_to(num: int) -> List[int]:
    """Returns fibonacci numbers up to num"""
    logging.debug(f"fib_to({num})")
    result: List[int] = []
    a, b = 0, 1
    while b < num:
        result.append(b)
        # Python will evaluate all expressions first, before doing the
        # assignment.
        a, b = b, a + b
    return result


def fibrec(pos: int) -> int:
    """Returns the `num` position in the fibonacci sequence using recursion.

    Warning: This is crazy expensive. Do not call with num > 30
    """
    if pos <= 1:
        return 1
    return fibrec(pos - 2) + fibrec(pos - 1)
