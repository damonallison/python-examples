"""Builtins is a python module which is automatically available without needing to `import`.
"""

import builtins

def test_builtins() -> None:
    assert builtins.list([10]) == list([10])

    # Builtins is a module like any other. It can be mangled, del() and added
    # to. Yuk.

    # Don't ever do this.
    #
    # Note we are using a function here rather than a lambda since lambdas do
    # not easily support type hinting.
    def echo(x: str) -> None:
        return x
    builtins.echo = echo

    assert "echo" in dir(builtins)
    assert builtins.echo("damon") == "damon"

    del builtins.echo
    assert "echo" not in dir(builtins)
