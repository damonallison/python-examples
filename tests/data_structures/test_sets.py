def test_sets() -> None:
    """Sets are created using set() or {e1, e2, ...}."""

    colors = {"orange", "yellow", "green"}

    assert 3 == len(colors)

    # set membership
    assert "orange" in colors
    assert "blue" not in colors

    colors.add("black")
    assert "black" in colors
    colors.remove("black")
    assert "black" not in colors

    # Set comprehension is also supported
    filtered = {x for x in colors if len(x) > 5}
    assert {"yellow", "orange"} == filtered


def test_set_from_list() -> None:
    """Use set() to create a set from a list"""
    s = set(["a", "a", "b", "c"])

    assert {"b", "c", "a"} == s
    assert {"a", "c", "b"} == s


def test_set_operations() -> None:
    """Example of set operations - union, difference, intersection"""

    s1 = {"red", "green", "blue"}
    s2 = {"red", "blue", "white"}

    assert "red" in s1 and "red" in s2

    # union - the set of unique elements across both sets
    assert {"red", "green", "white", "blue"} == s1.union(s2)

    # intersection - the set of elements in s1 also in s2
    assert {"red", "blue"} == s1.intersection(s2)

    # difference - the set of elements only found in list 1
    assert {"green"} == s1.difference(s2)

    # Removes a random element from the set.
    s1.pop()
    assert len(s1) == 2
