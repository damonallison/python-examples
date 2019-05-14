"""A manager class used to demonstrate inheritance in Python"""

from .person import Person
from .printer import Printer


class Manager(Person, Printer):
    """A class used to demonstrate (multiple) inheritance."""

    def full_name(self) -> str:
        """Overrides Person.full_name."""

        return "Manager " + super().full_name()
