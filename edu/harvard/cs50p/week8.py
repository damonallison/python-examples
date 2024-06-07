from datetime import date, timedelta
import tempfile

import inflect
import pytest


def how_old_in_minutes(dob: date) -> str:
    """Print out in english how old someone is in minutes"""
    delta: timedelta = date.today() - dob
    minutes = int(delta.total_seconds() / 60.0)
    return inflect.engine().number_to_words(minutes) + " minutes"


class Jar:
    def __init__(self, capacity: int = 12) -> None:
        if capacity < 0:
            raise ValueError("invalid capacity")
        self._capacity = capacity
        self._size = 0

    def __str__(self) -> str:
        return "🍪" * self._size

    def deposit(self, n: int) -> None:
        if n <= 0:
            raise ValueError("invalid n")
        if self._size + n > self._capacity:
            raise ValueError("too many cookies")
        self._size += n

    def withdraw(self, n: int) -> None:
        if n <= 0:
            raise ValueError("invalid n")
        if self._size - n < 0:
            raise ValueError("not enough cookies")
        self._size -= n

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return self._size


def shirtificate(name: str) -> None:
    from fpdf import FPDF, XPos, YPos

    class ShirtPDF(FPDF):
        def header(self) -> None:
            self.set_font("helvetica", style="", size=24)
            self.set_xy(x=0, y=30)
            self.cell(
                w=self.w,
                h=None,
                text="CS50 Shirtificate",
                align="C",
                new_x=XPos.LEFT,
                new_y=YPos.NEXT,
            )

        def print_shirt(self, name: str) -> None:
            self.ln(50)
            self.set_xy(50, self.y)

            self.image(
                "./edu/harvard/cs50p/week8-images/shirtificate.png",
                keep_aspect_ratio=True,
                x=30,
                h=150,
            )
            self.set_xy(0, 145)
            self.set_font_size(16)
            self.set_text_color(255, 255, 255)
            self.cell(
                w=self.w,
                h=None,
                text=f"{name} took CS50",
                align="C",
            )

    pdf = ShirtPDF()

    pdf.add_page()
    pdf.print_shirt(name)
    pdf.output("/tmp/shirt.pdf")


#
# Tests
#
# poetry run pytest -s week8.py


def test_happy_path() -> None:
    j = Jar()
    assert j.size == 0
    assert j.capacity == 12

    j.deposit(1)
    assert j.size == 1
    assert j.capacity == 12

    j.withdraw(1)
    assert j.size == 0
    assert j.capacity == 12


def test_invalid_capacity() -> None:
    with pytest.raises(ValueError) as ve:
        j = Jar(-1)
    assert ve.match("invalid capacity")


def test_invalid_deposit() -> None:
    j = Jar(1)
    with pytest.raises(ValueError) as ve:
        j.deposit(2)
    assert ve.match("too many cookies")

    with pytest.raises(ValueError) as ve:
        j.deposit(-1)
    assert ve.match("invalid n")


def test_invalid_withdraw() -> None:
    j = Jar(1)
    j.deposit(1)

    with pytest.raises(ValueError) as ve:
        j.withdraw(2)
    assert ve.match("not enough cookies")

    with pytest.raises(ValueError) as ve:
        j.withdraw(-1)
    assert ve.match("invalid n")
