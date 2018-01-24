"""A simple person class"""

class Person:
    """A simple person class"""

    # A class variable (attribute) shared by all instances.
    iq = 0

    def __init__(self, first_name: str, last_name: str):
        self.first_name = first_name
        self.last_name = last_name
        self.index = 0
        self.children = []

    def full_name(self) -> str:
        return self.first_name + " " + self.last_name

    #
    # Adding iterator behavior to classes.
    #
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.children == None or self.index >= len(self.children):
            raise StopIteration

        idx = self.index
        self.index = self.index + 1
        return self.children[idx]


    #
    # Generators are another way to write iterators.
    #
    def child_first_names(self):
        for child in self.children:
            yield child.first_name