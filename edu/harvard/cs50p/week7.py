import re


def is_ipv4(value: str) -> bool:
    match = re.search(r"[0-9]\.[0-9]+\.[0-9]+\.[0-9]+", value)
    if not match:
        return False

    # "\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"

    octets = match.string.split(".")
    if len(octets) != 4:
        return False

    # using list comprehension
    # return all([int(octet) >= 0 and int(octet) <= 255 for octet in octets])

    # using iteration
    for octet in octets:
        i_octet = int(octet)
        if i_octet < 0 or i_octet > 255:
            return False
    return True


def test_is_ipv4() -> None:
    assert is_ipv4("1.1.1.1")
    assert not is_ipv4(("1.1.1.999"))


class Person:
    def __init__(self, name: str, age: int) -> None:
        self.name = name
        self.age = age

    def hi(name: str) -> str:
        return f"hi {name}"

    def greet(self) -> str:
        return f"hello, {self.name}. you are {self.age}"
