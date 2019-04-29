"""A manager class used to demonstrate inheritance in Python"""

from .person import Person


class Manager(Person):
    """A class used to demonstrate inheritance."""

    def full_name(self) -> str:
        """Overrides Person.full_name."""

        return "Manager " + Person.full_name(self)
