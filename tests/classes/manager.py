"""A manager class used to demonstrate inheritance in Python"""

from typing import List

from .person import Person
from .logger import Logger


class Manager(Person, Logger):
    """A class used to demonstrate (multiple) inheritance."""

    def __init__(
        self, first_name: str, last_name: str, employees: List[Person] = []
    ) -> None:
        Person.__init__(self, first_name, last_name)
        Logger.__init__(self)
        self.employees = employees

    def full_name(self) -> str:
        """Overrides Person.full_name."""
        return "Manager " + super().full_name()

    def __repr__(self) -> str:
        return f"Manager('{self.first_name}', '{self.last_name}')"
