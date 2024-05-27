"""
Exceptions

* SyntaxErrors - python couldn't parse the source. The code won't run.
* Exceptions - runtime error. Something unexpected happened or couldn't be
  completed.

Guidelines:

* Look for interesting "corner cases": None, empty list, negative values, etc.
* Only `try` specific, small blocks of code which might fail. Ton't


"""


class NotAnIntException(ValueError):
    def __init__(self, input: str) -> None:
        self.input = input


def validate_int(val: str) -> int:
    # Assume we require an int

    # give x a default value
    x = 0
    try:
        x = int(val)  # raises a ValueError if `val` cannot be converted to an int
    except ValueError:
        # "catches" an exception.
        print("x is not an int.")
    except Exception as ex:
        # with `as name` exposes the exception object we can use to examine the
        # exception.

        # don't do this! always catch the most specific exception. Don't be
        # lazy, determine what exceptions will be raised.
        print(f"something unknown happened: {ex}")
    else:
        # else will execute when the try block terminates successfully (doesn't
        # raise an exception)
        print("validation succeeeded")
    finally:
        # finally will *always* execute, even if exceptions are handled and
        # re-raised
        print("validation completed")

    return x


def fuel_gauge(fraction: str) -> str:
    parts = fraction.split("/")
    if len(parts) != 2:
        raise ValueError("invalid fraction")
    try:
        num_parts = [int(part) for part in parts]
    except ValueError:
        raise ValueError("non-numeric fraction")

    try:
        percent = num_parts[0] / num_parts[1]
    except ZeroDivisionError as zde:
        raise zde

    if percent >= 0.99:
        return "F"
    elif percent <= 0.01:
        return "E"
    else:
        return f"{int(percent * 100)}%"


def felipes_taqueria(items: list[str]) -> float:
    menu = {
        "Baja Taco": 4.25,
        "Burrito": 7.50,
        "Bowl": 8.50,
        "Nachos": 11.00,
        "Quesadilla": 8.50,
        "Super Burrito": 8.50,
        "Super Quesadilla": 9.50,
        "Taco": 3.00,
        "Tortilla Salad": 8.00,
    }

    total = 0.0
    for menu_item in [item.title() for item in items]:
        if menu_item not in menu:
            raise ValueError(f"Unknown item: {menu_item}")
        total += menu[menu_item]
    return total


def live_input() -> list[str]:
    items: list[str] = []
    while True:
        try:
            items.append(input("enter item:"))
        except EOFError:
            break
    return items


def sort_groceries(list: list[str]) -> list[str]:
    groceries: dict[str, int] = {}
    for item in list:
        if item not in groceries:
            groceries[item] = 1
        else:
            groceries[item] += 1

    return [f"{groceries[item]} {item}" for item in sorted(groceries.keys())]


def outdated(d: str) -> str:
    """Input: 1/2/2020 or January 2, 2020"""

    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    # parse string into [month, day, year] strings
    if "/" in d:
        parts = d.split("/")
        if len(parts) != 3:
            raise ValueError("invalid date")
    else:
        parts = [part.replace(",", "").strip() for part in d.split()]

        if len(parts) != 3:
            raise ValueError("invalid date")

        parts[0] = parts[0].title()
        if parts[0] not in months:
            raise ValueError("invalid month")
        parts[0] = str(months.index(parts[0]) + 1)
    try:
        int_parts = [int(part) for part in parts]
    except ValueError as ve:
        raise ve

    if int_parts[0] < 1 or int_parts[0] > 12:
        raise ValueError("invalid month")
    if int_parts[1] < 1 or int_parts[1] > 31:
        raise ValueError("invalid day")
    if int_parts[2] < 0:
        raise ValueError("invalid year")

    # MM/DD/YYYY -> YYYY-MM-DD
    return f"{int_parts[2]:04}-{int_parts[0]:02}-{int_parts[1]:02}"
