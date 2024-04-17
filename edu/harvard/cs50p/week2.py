"""
Week 2: Loops

Note that as we loop, we do not introduce a new variable scope like we do a
function.

* while: evaluate a boolean again and again.
* ensure your while loops have a logical end state and focus on edge cases
  (boundaries)
* make everything zero based (avoid off by one errors)
* `_` indicates the variable is unused

* function encapsulation / abstraction (interface based programming).

"""


def test_while(count: int) -> None:
    while count > 0:
        print(f"{count}...", end="")
        count -= 1  # same as `count = count - 1``
    print("boom!")


def test_for(count: int) -> None:
    for i in range(count, 0, -1):
        print(f"{i}...", end="")
    print("boom!")


def test_operator(count: int) -> None:
    print(("test " * count).strip())


def test_infinite() -> None:
    """`break` and `continue` are control flow statements (while / for)"""
    while True:
        n = int(input("what's n? "))
        if n > 0:
            break
    test_for(n)


# lists


def test_lists() -> None:
    students = ["Sam", "Grace", "Lily"]

    # len() is a python built-in which takes an arbitrary container (list,
    # tuple, dict, iterable, str) and returns it's length.
    assert len(students) == 3

    # list accessors - accessing values in a list
    #
    # [pos] - returns element at pos starting from the beginning of the list.
    # raises an exception if the item is out of bounds [-pos]
    #
    # [-pos] - returns elemnt at pos starting from the end of the list. raises
    # an error if the index is out of range.
    #
    # [start:end] - returns a new list for the range starting at pos `start` up
    # to (but not including) `end`. Leaving start empty (i.e., [:end]) assumes
    # start = 0. Leaving end empty (i.e., [0:]) assumes the end (i.e., len[))
    #
    assert students[0] == "Sam"
    assert students[1:2] == ["Grace"]
    assert students[1:1] == []  # no range - returns an empty list
    assert students[1:0] == []  # backwards - returns an empty list
    assert students[:-1] == ["Sam", "Grace"]

    # adding
    students.append("Cole")
    assert students == ["Sam", "Grace", "Lily", "Cole"]

    # removing
    students.remove("Lily")

    assert students == ["Sam", "Grace", "Cole"]

    # iterating a list by value
    for student in students:
        print(student)

    # iterating using a while (typically not used - use range(len()) or
    # enumerate()
    i = 0
    while i < len(students):
        print(f"{i} == {students[i]}")
        i += 1

    # iterating by position
    for i in range(len(students)):
        print(f"{i} == {students[i]}")

    # iterating by position and value with enumerate()
    for i, v in enumerate(students):
        print(f"{i} == {v}")


# dictionaries


def test_dictionaries() -> None:
    # note: nested data structures
    d = {
        "students": ["Sam", "Grace", "Lily"],
        "parents": ["Joe", "Damon"],
    }

    # python iterates dictionary keys by default
    for k in d:
        print(f"{k} == {d[k]}")

    # checking for existence
    if "students" in d:
        print("we have students!")
    assert "students" in d

    # get a value
    assert d["parents"] == ["Joe", "Damon"]

    # add
    d["new_elenent"] = 10

    # remove
    if "new_element" in d:
        del d["new_element"]

    # loop

    # defaults to keys only
    for k in d:
        print(k)

    # items() iterates (key, value) pairs
    for k, v in d.items():
        print(f"{k} == {v}")


# nested loops


def print_row(width: int) -> None:
    # this is actually a loop as it's doing the repetition via the `*`
    print("#" * width)


def print_column(height: int) -> None:
    for i in range(height):
        print("#")


def print_square(size: int) -> None:
    for _ in range(size):
        print_row(size)


def print_square_nested(size: int) -> None:
    for _ in range(size):
        for _ in range(size):
            print("#", end="")
        print()  # print new line


#
# Exercises
#


def camel_case(name: str) -> None:
    # camelCase
    # snake_case

    # implement a program that prompts the user for the name of a variable in
    # camel case and outputs the corresponding name in snake case.

    for c in name:
        if c.islower():
            print(c, end="")
        else:
            print(f"_{c.lower()}", end="")
    print()


def coke_machine() -> None:
    total = 0
    while total < 50:
        print(f"amount due: {50 - total}")
        val = int(input("insert coin: "))
        total += val
    print(f"changed owed: {total - 50}")


def twttr(val: str) -> None:
    # implement a program that prompts the user for a str of text (passed in)
    # and ouputs that same text but without all vowels whether imputted in upper
    # or lower case.

    vowels = "aeiou"
    for c in val:
        if c.lower() not in vowels:
            print(c, end="")
    print()


def validate_vanity_plate(plate: str) -> bool:
    """
    validate `vanity license plates` (the input string) matches the following rules:

    * starts with 2 letters
    * min of 2 chars, max of 6
    * numbers must come at the end
    * the first number used cannot be a zero
    * no periods, spaces, or punctuation marks are allowed
    """

    # length first
    if len(plate) < 2 or len(plate) > 6:
        return False

    if not plate[0].isalpha() or not plate[1].isalpha():
        return False

    found_number = False
    for c in plate:
        if not (c.isalpha() or c.isnumeric()):
            return False
        if c.isnumeric():
            if not found_number and c == "0":
                # first number cannot be zero
                return False
            found_number = True
        elif found_number:
            # alpha occurs after number
            return False

    return True


def nutrition_facts(fruit: str) -> int | None:
    """
    Implement a program that prompts users to input a fruit (case-insensitively)
    and output the number of calories for one portion of that fruit
    """

    fruits = {"apple": 130, "banana": 110}

    return fruits.get(fruit.strip().lower())


def test_exc(val: int) -> None:
    if val > 10:
        raise ValueError("oops, number too big")
