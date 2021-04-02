import pytest

def test_arithmetic_operators():
    """Python arithmetic operators

    +  addition
    -  subtraction
    *  multiplication
    /  division (always returns a floating point value)
    %  modulus
    ** exponent
    // floor division - always rounds down to next whole integer
                        (negative too)
    """

    # Python handles prescedence within expressions
    assert 2 + 2 * 6 == 14

    # Division returns the exact value, not integer (Python 3)

    assert 4 / 3 == pytest.approx(1.33, 0.1)
    assert 4.0 / 3.0 == pytest.approx(1.33, 0.1)

    # Integer division will always round *down* to the nearest integer
    assert 7 // 2 == 3
    assert -7 // 2 == -4

    # Exponentiation
    assert 3 ** 3 == 27

    # Modulo
    assert 4.4 % 1 == pytest.approx(.4)

def test_assignment_operators():
    # Multiple assignment
    x, y = 10, 20
    assert x == 10 and y == 20

    # All arithmetic operators have a corresponding assignment equivalent.
    #
    # += -= *= /=

    val = 10
    val += 10
    assert val == 20

    val -= 10
    assert val == 10

    val *= 10
    assert val == 100

    val **= 2
    assert val == 10000

    val /= 800
    assert val == pytest.approx(12.5)
