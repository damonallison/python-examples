"""
The collections module provides specialized container data types
"""

import collections


def test_counter() -> None:
    # counter is a dict subclass for counting hashable objects

    c = collections.Counter()
    for word in ["red", "blue", "red"]:
        c[word] += 1

    assert c["red"] == 2
    assert c["blue"] == 1
    assert c["green"] == 0

    assert c.most_common()[0][0] == "red"
    assert set(list(c)) == {"red", "blue"}  # list() returns unique keys only


def test_defaultdict() -> None:
    # defaultdict is a dict subclass which provides a default the first time a
    # dictionary key is accessed.
    #
    # int() is the `default_factory` function. it will be called when an element
    # doesn't exist in the dict.
    dd = collections.defaultdict(int)

    assert dd["test"] == 0

    dd["test"] += 1
    assert dd["test"] == 1

    dd = collections.defaultdict(list)

    assert dd["list"] == []
    dd["list"].append(1)
    assert dd["list"] == [1]

    # Because it was accessed, it is added to the dict
    assert dd["list2"] == []

    assert set(dd.keys()) == {"list", "list2"}
