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


def test_stacking() -> None:
    """Stacking pivots columns into rows. Unstacking pivots from rows -> columns."""

    df = pd.DataFrame({"one": [1, 2, 3], "two": [4, 5, 6]}, index=["a", "b", "c"])

    # Pivot columns into rows, producing a pd.Series
    stacked = df.stack()

    # a one 1
    #   two 4
    # b one 2
    #   two 4
    # c one 3
    #   two 6
    # dtype: int64

    assert stacked.loc[("a", "one")] == 1
    assert stacked.loc[("c", "two")] == 6

    unstacked = stacked.unstack()  # pivot inner most index level to columns

    assert unstacked.loc["a", "one"] == 1
    assert unstacked.loc["c", "two"] == 6


def test_pivot() -> None:
    # Pivoting allows you to go from "long" to "wide" and vice-versa

    # Long
    #
    # order_id      col_name        col_val
    # 1             retailer_id     10
    # 1             shopper_id      11
    # 2             retailer_id     20
    # 2             shopper_id      21

    # Pivoted
    #
    # order_id      retailer_id     shopper_id
    # 1             10              11
    # 2             20              21

    df = pd.DataFrame(
        {
            "order_id": [1, 1, 2, 2],
            "col_name": ["retailer_id", "shopper_id", "retailer_id", "shopper_id"],
            "col_val": [10, 11, 20, 21],
        }
    )

    # col_name  retailer_id  shopper_id
    # order_id
    # 1                  10          11
    # 2                  20          21
    pivoted = df.pivot(index="order_id", columns="col_name", values="col_val")

    assert pivoted.loc[1, "retailer_id"] == 10
    assert pivoted.loc[2, "shopper_id"] == 21
