from ..pkg1.mod1 import Mod1Calculator


class Mod2Calculator:
    """Shows using the relative package import."""
    def add(self, x, y):
        return Mod1Calculator().add(x, y)