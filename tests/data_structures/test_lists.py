import copy

"""Python's list() is an ordered, mutable data structure which can elements of different types."""


def test_list_manipulation() -> None:
    lst = [0]

    # Append an element to the end of a list
    lst.append(1)
    assert [0, 1] == lst

    lst.extend([3, 4])
    assert [0, 1, 3, 4] == lst

    lst.insert(2, 2)
    assert [0, 1, 2, 3, 4] == lst

    # Remove raises a value error if the value doesn't exist
    try:
        lst.remove(4)
    except ValueError:
        assert False
    assert [0, 1, 2, 3] == lst

    # By default, pop removes the last element in the list
    assert 3 == lst.pop()
    assert [0, 1, 2] == lst

    # Test membership
    assert 2 in lst
    assert 3 not in lst


def test_copy() -> None:
    """To copy, use [:] or copy().

    Always copy when iterating when you are modifying the list. [:] is idomatic python.
    """

    lst = ["damon", "kari", 10, ["another", "list"]]
    cp = lst.copy()

    assert lst is not cp, "the objects should not be referentially equal"
    assert lst == cp, "the objects should be logically (==) equal"

    #
    # Copies are shallow.
    #
    # Here, we are copying value types. lst2 will have a separate copy of
    # each element.
    #
    lst = [1, 2]
    cp = lst[:]

    cp[0] = 3
    assert lst == [1, 2]
    assert cp == [3, 2]

    #
    # Here, we are copying reference types. lst2 will contain a pointer to the
    # same lst[0] element. Note that [:] and list.copy() are both "shallow"
    # copies.
    #
    lst = [[1], 2]
    cp = lst.copy()
    cp[0][0] = 3

    assert lst == [[3], 2]

    # deepcopy will recursively copy an object. There are some values which
    # can't be deep copied (like functions, file handles). In those cases, the
    # reference is "copied".

    def add(x: int, y: int) -> int:
        return x + y

    lst = [[1], 2, add]
    cp = copy.deepcopy(lst)

    cp[0][0] = 3
    assert lst == [[1], 2, add]
    assert cp == [[3], 2, add]


def test_sorting() -> None:
    """Example showing max(), min(), sorted()

    max() retrieves the max element (as defined by >)
    min() retrieves the min element (as defined by <)
    sorted() will sort according to < and return a *new* list.
    """

    a = [10, 20, 1, 2, 3]

    assert 20 == max(a)
    assert 1 == min(a)
    assert [1, 2, 3, 10, 20] == sorted(a)
    assert [20, 10, 3, 2, 1] == sorted(a, reverse=True)

    # Sorted returns a copy
    assert a == [10, 20, 1, 2, 3]

    b = copy.deepcopy(a)
    a.sort()  # sort() will sort in place.

    assert a == [1, 2, 3, 10, 20]
    assert b == [10, 20, 1, 2, 3]


def test_iteration() -> None:
    """`for` iterates over the elements in a sequence."""

    lst = ["damon", "kari", "grace", "lily", "cole"]
    expected = []

    # Remember to always iterate over a *copy* of the list if you are mutating the list
    for name in lst.copy():
        expected.append(name)

    assert lst == expected

    # To iterate over just the indices of a sequence, use range(len(lst)).
    expected = []
    for i in range(len(lst)):
        expected.append(lst[i])

    assert lst == expected

    # To iterate over indices and values simultaneously, use enumerate()
    # enumerate() returns tuples of the indicies and values of a list.
    pos = []
    val = []
    for i, v in enumerate(["tic", "tac", "toe"]):
        pos.append(i)
        val.append(v)

    assert [0, 1, 2] == pos
    assert ["tic", "tac", "toe"] == val

    # Loop statements can have an `else` clause, which executes
    # when the loop terminates without encoutering a `break` statement
    primes = []
    for n in range(2, 6):
        for x in range(2, n):
            if n % x == 0:
                break
        else:
            primes.append(n)

    assert [2, 3, 5] == primes


def test_zip() -> None:
    """zip() returns an iterator that combines multiple iterables
    into a sequence of tuples.

    Each tuple contains the elements in that position from all the
    iterables.

    Once one list is exhaused, zip stops.

    The object that zip() returns is a zip() object.

    See:
    https://docs.python.org/3.3/library/functions.html#zip

    """

    # Here are two iterables that we are going to pass to zip().
    # Notice how ages only has 2 elements. This will force zip() to stop
    # after two elements, leaving "joe" out of the resulting zip() result.

    # If you care about the trailing, unmatched values from longer
    # iterables, use itertools.zip_longest() instead.

    names = ("damon", "jeff", "joe")
    ages = [20, 32]

    # zip() will return a zip() object, which is an iterator. You need to
    # cast the iterator into a concrete type (tuple, list, set, dict) to
    # realize the iterable.

    assert (("damon", 20), ("jeff", 32)) == tuple(zip(names, ages))
    assert [("damon", 20), ("jeff", 32)] == list(zip(names, ages))

    # You can use * to "unzip" a list or tuple. In this case, we will unzip
    # people to send zip() the inner tuples.
    people = (("damon", "jeff"), (20, 32))
    assert (("damon", 20), ("jeff", 32)) == tuple(zip(*people))


def test_list_comprehensions() -> None:
    """List comprehensions are a concise way to transform lists.

    lst = [<operation> for <elt> in <iterable> [if <condition>]]

    <operation> : executed for each element in the iterable.
    <elt> : the current element being executed
    <iterable> : the iterable to run the list comprehension on
    <condition> : the condition to apply. If condition returns false,
                    the element is skipped.
    """

    squares = [x ** 2 for x in range(1, 4)]
    assert [1, 4, 9] == squares

    evens = [x for x in range(1, 11) if x % 2 == 0]
    assert [2, 4, 6, 8, 10] == evens

    squares = [(x, x ** 2) for x in [0, 1, 2, 3]]
    assert [(0, 0), (1, 1), (2, 4), (3, 9)] == squares

    # Note that the condition can only include an if statement.
    # If you want to use an else, you'll need to make the condition part of
    # the operation.

    # Here, we'll only compute squares for even numbers and default
    # odds to 0.
    even_squares = [x ** 2 if x % 2 == 0 else 0 for x in range(8)]
    assert [0, 0, 4, 0, 16, 0, 36, 0] == even_squares

    # You can have multiple for and if statements in the same list
    # comprehension.
    concatenated = [(x, y) for x in range(1, 3) for y in range(1, 3) if x != y]
    assert [(1, 2), (2, 1)] == concatenated

    # Flatten a list using a listcomp
    vec = [[1, 2, 3], [4, 5, 6]]
    assert [1, 2, 3, 4, 5, 6] == [x for elt in vec for x in elt]
