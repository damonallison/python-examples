"""A simple person class"""

class Person:
    """A simple person class"""

    def __init__(self, first_name: str, last_name: str):
        self.first_name = first_name
        self.last_name = last_name

    def full_name(self) -> str:
        return self.first_name + " " + self.last_name


