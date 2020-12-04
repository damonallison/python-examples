"""A manager class used to demonstrate inheritance in Python"""

from .person import Person
from .logger import Logger


class Manager(Person, Logger):
    """A class used to demonstrate (multiple) inheritance."""

    def full_name(self) -> str:
        """Overrides Person.full_name."""

        return "Manager " + super().full_name()

    def __repr__(self) -> str:
        return f"Manager('{self.first_name}', '{self.last_name}')"
