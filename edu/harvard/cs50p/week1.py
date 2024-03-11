"""
Week 1: Conditionals
    https://cs50.harvard.edu/python/2022/weeks/1

* Assignment (=) vs. Equality (==)
* How are operators implemented on an object?
* Boolean expression (named after George Boole).
* Simplicity is a virtue. Occam's razor: simpliest is usually the best.
* Pythonic == "idomatic": all languages have idioms.

"""


def compare() -> None:
    x = int(input("Enter x: "))
    y = int(input("Enter y: "))

    def first_try() -> None:
        # They are all mutually exclusive, yet all conditionals have to run.
        # Note that x and y resolve to the outer scope of compare()
        if x < y:
            print("x < y")
        if x > y:
            print("x > y")
        if x == y:
            print("x == y")

    def second_try() -> None:
        if x < y:
            print("x < y")
        elif x > y:
            print("x > y")
        else:
            print("x == y")

    def third_try() -> None:
        # Note conditionals are short circuited (first True ends the
        # conditional)
        if x < y or x > y:
            print("x != y")
        else:
            print("x == y")

    first_try()
    second_try()
    third_try()


def grade() -> None:
    def first_try() -> None:
        # Bug if score > 100
        if score >= 90 and score <= 100:
            print("A")
        elif score >= 80 and score < 90:
            print("B")
        elif score >= 70 and score < 80:
            print("C")
        elif score >= 60 and score < 70:
            print("D")
        else:
            print("F")

    def second_try() -> None:
        # Simplifying the conditionals
        if score >= 90:
            print("A")
        elif score >= 80:
            print("B")
        elif score >= 70:
            print("C")
        elif score >= 60:
            print("D")
        else:
            print("F")

    score = int(input("enter score: "))
    first_try()
    second_try()


def parity() -> None:
    # modulo operator
    def is_even(x: int) -> bool:
        return x % 2 == 0

    print(f"1.is_even(): {is_even(1)}")
    print(f"2.is_even(): {is_even(2)}")


def house() -> None:
    name = input("what's your name? ")
    match name:
        # Note the authors of match did not use `or` for some reason.
        # | is short-circuited
        case "Harry" | "Hermoine" | "Ron":
            print("Griffyndor")
        case _:  # default case
            print("who?")


#
# Problems
#


def deep_thought() -> None:
    """Implement a program that prompts the user, outputting Yes if 42"""

    answer = input(
        "What is the answer to the great question of life, the universse, and everything?"
    )
    if answer == "42" or answer == "forty-two" or answer == "forty two":
        print("Yes")
    else:
        print("No")


def bank() -> None:
    """Implement a program that prompts the user for a greeting.

    * If the greeting starts with hello, output $0.
    * If it starts with an "h", output $20
    * Else output $100
    """
    greeting = input("Greeting: ").lower().strip()
    if greeting.startswith("hello"):
        print("$0")
    elif greeting.startswith("h"):
        print("$20")
    else:
        print("$100")


def file_extensions() -> str:
    """Return the proper media type for a user-entered filename"""
    filename = input("file name: ").lower().strip()
    if filename.endswith(".gif"):
        return "image/gif"
    elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
        return "image/jpeg"
    elif filename.endswith(".pdf"):
        return "application/pdf"
    elif filename.endswith(".png"):
        return "image/png"
    elif filename.endswith(".txt"):
        return "text/plain"
    elif filename.endswith(".zip"):
        return "application/zip"
    else:
        return "application/octet-stream"


def math_interpreter() -> None:
    expr = input("expression: ")
    lhs, op, rhs = expr.split(" ")
    lhs = int(lhs)
    rhs = int(rhs)
    if op == "+":
        print(lhs + rhs)
    elif op == "-":
        print(lhs - rhs)
    elif op == "*":
        print(lhs * rhs)
    elif op == "/":
        print(lhs / rhs)


def meal_time():
    def convert(time: str) -> float:
        hour, min = time.split(":")
        return int(hour) + (int(min) / 60)

    tflt = convert(input("what time is it: "))
    if tflt >= 7.0 and tflt <= 8.0:
        print("breakfast")
    elif tflt >= 12.0 and tflt <= 13.0:
        print("lunch")
    elif tflt >= 18.0 and tflt <= 19.0:
        print("dinner")
