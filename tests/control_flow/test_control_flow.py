class TestControlFlow:
    def test_if_elif(self) -> None:
        x = 100
        if x > 10 and x < 50:
            assert False
        elif x > 100:
            assert False
        else:
            assert x == 100

    def test_complex_if(self) -> None:
        height = 75
        weight = 150

        print(f"weight / height ** 2 == {weight / height ** 2}")
        # You can execute a complex expression
        if 1 < weight / height > 1.5:
            pass
        else:
            assert False

        # using logical operators (and, or, not)
        # use parens to disambiguate complex expressions
        assert height == 75 and not weight > 200 and (height > 100 or height < 80)

    def test_is_truthy(self) -> None:
        """Here are most of the built-in objects that are considered False in Python:

        * constants defined to be false: None and False
        * zero of any numeric type: 0, 0.0, 0j, Decimal(0), Fraction(0, 1)
        * empty sequences and collections: '"", (), [], {}, set(), range(0)

        Anything else is treated as True - it doesn't have to be a boolean value.
        """
        if set():
            assert False
        if {}:
            assert False
        if []:
            assert False
        if "":
            assert False
        if 0:
            assert False
        if 0.0:
            assert False

        # Any value can be used in logical statements.
        x = 100
        if not x:
            assert False

    def test_while_for_else(self) -> None:
        """Loop statements (for and while) can also have an else clause.

        The else clause is called when the loop finishes naturally. It will *not*
        be invoked when the loop is terminated with `break`"""

        called = False
        iter = 0
        while iter < 10:
            iter += 1
        else:
            called = True

        assert called and iter == 10

        called = False
        for i in range(10):
            if i == 1:
                break
        else:
            called = True

        # `else` will not be called since the iteration was prematurely
        # terminated.
        assert i == 1 and not called
