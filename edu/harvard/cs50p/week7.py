import re


def is_ipv4(value: str) -> bool:
    """A simple IPv4 validator which uses a combination of regex and manual rules"""
    match = re.search(r"^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$", value)
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


def youtube(html: str) -> list[str]:
    """Retrieves embedded youtube URLs"""

    urls: list[str] = []
    # find iframe matches
    iframes = re.findall("<iframe.*</iframe>", html)
    for iframe in iframes:
        m = re.search(r"src=\"([^\s]*)\"", iframe)
        if m is not None:
            urls.append(m.group(1))
        else:
            print("warning: missing src")

    return urls


def nine_to_five(value: str) -> str:
    """
    Implement a function that expects a str in either of the formats below and
    returns the corresponding str in 24 hour format

    * 9:00 AM to 5:00 PM
    * 9 AM to 5 PM
    """
    pattern = r"^(\d+)(?::00)? ([AP]M) to (\d+)(?::00)? ([AP]M)$"
    match = re.search(pattern, value)
    if match is None:
        raise ValueError()

    def canonicalize(hour: int, am: bool) -> int:
        if hour == 12:
            if am:
                return 0
            else:
                return 12
        if not am:
            return 12 + hour
        return hour

    start = int(match.group(1))
    start_am = match.group(2) == "AM"
    end = int(match.group(3))
    end_am = match.group(4) == "AM"

    return f"{canonicalize(start, start_am):02}:00 to {canonicalize(end, end_am):02}:00"


def count_ums(value: str) -> int:
    """Count the number of times "um" is used as it's own word"""

    pattern = r"[^a-zA-Z0-9]um[^a-zA-Z0-9]"
    return len(re.findall(pattern, value))


def validate_email(email: str) -> bool:
    """
    Use a prebuilt validation library to validate an email address rather than
    rolling your own regex.

    The goal with this example is to show the student that pre-built libraries
    exist which do many of the common regular expression features.
    """
    import validators

    return validators.email(email)


class Person:
    def __init__(self, name: str, age: int) -> None:
        self.name = name
        self.age = age

    def hi(name: str) -> str:
        return f"hi {name}"

    def greet(self) -> str:
        return f"hello, {self.name}. you are {self.age}"
