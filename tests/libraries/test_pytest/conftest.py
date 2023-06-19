from typing import Iterator
import pytest


class Calculator:
    def __init__(self):
        self.count = 0

    def _increment(self) -> None:
        self.count += 1

    def add(self, i: int, j: int) -> int:
        self._increment()
        return i + j

    def sub(self, i: int, j: int) -> int:
        self._increment()
        return i - j

    def reset(self) -> None:
        self.count = 0


c = Calculator()


@pytest.fixture
def calculator() -> Iterator[Calculator]:
    yield c
