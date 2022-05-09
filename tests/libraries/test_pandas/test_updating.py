"""Examples of manipulating Pandas DataFrame(s)."""
import pandas as pd


def test_min_max_limiting() -> None:
    """Shows how to enforce minimum and maximum values at the column or
    dataframe level.

    At the column level, use `.loc` with a boolean mask to select the correct
    rows and assign the min value.

    At the DataFrame level, use .clip() to enforce the upper and/or lower bound.
    """
    d = {
        "one": [-1, 0, 1],
        "two": [2, 3, -1],
    }
    # Update a single column
    df = pd.DataFrame(d)
    #
    # .loc accepts a boolean mask and set of columns to return.
    #
    df.loc[df["one"] < 0, ["one"]] = 0
    #
    #   one   two
    #   -1     2
    #    0     3
    #    1    -1
    #
    assert df.iloc[0, 0] == 0
    assert df.iloc[2, 1] == -1

    # You can use `clip` to enforce minimum and maximum values for an entire df.
    df = df.clip(lower=0.0)
    assert df.iloc[0, 0] == 0.0
    assert df.iloc[2, 1] == 0.0
