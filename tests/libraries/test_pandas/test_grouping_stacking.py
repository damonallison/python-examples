import numpy as np
import pandas as pd


def test_groupby() -> None:
    """groupby will return a list of [(key, dataframe)] pairs."""
    df = pd.DataFrame(
        {
            "one": [1, 1, 2, 2, 3, 3],
            "two": [1, 2, 3, 4, 5, 6],
        }
    )

    groups = df.groupby(["one"])
    keys = [elt[0] for elt in groups]
    dfs = [elt[1] for elt in groups]

    assert all([type(key) == int for key in keys])
    assert all([type(df) == pd.DataFrame for df in dfs])

    assert keys == [1, 2, 3]

    # A df is created for each group.
    assert np.array_equal(dfs[0]["one"].values, np.array([1, 1]))
    assert np.array_equal(dfs[0]["two"].values, np.array([1, 2]))

    assert np.array_equal(dfs[1]["one"].values, np.array([2, 2]))
    assert np.array_equal(dfs[1]["two"].values, np.array([3, 4]))

    assert np.array_equal(dfs[2]["one"].values, np.array([3, 3]))
    assert np.array_equal(dfs[2]["two"].values, np.array([5, 6]))

    # Aggregate functions can be applied to each group's columns.
    #
    # The df index is the groupby column, the values are the aggregate function
    # applied to each group's df for each column.
    sums_df = df.groupby(["one"]).sum()

    assert sums_df.loc[1, "two"] == 3
    assert sums_df.loc[2, "two"] == 7
    assert sums_df.loc[3, "two"] == 11

    assert np.array_equal(sums_df["two"].values, np.array([3, 7, 11]))
