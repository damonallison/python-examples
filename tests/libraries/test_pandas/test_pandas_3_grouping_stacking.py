"""
Group operations follow a split-apply-combine process.

1. Data is split into groups based on keys.
2. A function is applied to each group.
3. The results are combined.
"""

import numpy as np
import pandas as pd


def test_groupby() -> None:
    df = pd.DataFrame(
        {
            "one": [1, 1, 2, 2, 3, 3],
            "two": [1, 2, 3, 4, 5, 6],
            "three": [100, 100, 100, 200, 200, 200],
        }
    )

    # No apply operation has been done on the GroupBy object, so the GroupBy
    # is returned.
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
    #
    # Calling sum() performs the "apply" on the GroupBy object. This returns a
    # new DF with row indices as the group by key, the value as the column(s).
    sums_df = df.groupby(["one"]).sum()

    assert sums_df.loc[1, "two"] == 3
    assert sums_df.loc[2, "two"] == 7
    assert sums_df.loc[3, "two"] == 11

    assert np.array_equal(sums_df["two"].values, np.array([3, 7, 11]))

    #               two
    # one   three
    # 1     100     3
    # 2     100     3
    # 2     200     4
    # 3     200     11
    #

    # Returns a dataframe
    gb_mc = df.groupby(["one", "three"]).sum()
    assert gb_mc.loc[(1, 100), "two"] == 3
    assert gb_mc.loc[(2, 100), "two"] == 3
    assert gb_mc.loc[(2, 200), "two"] == 4
    assert gb_mc.loc[(3, 200), "two"] == 11


def test_group_by_function() -> None:
    df = pd.DataFrame(
        {
            "one": [-10, 0, 5, 10, 25, 50, 100, 1000],
        }
    )

    def ranker(val: int) -> float:
        # The grouping function will be called once with each index value.
        val = df.loc[val, "one"]
        if val <= 0.0:
            return 0.0
        if val <= 10.0:
            return 5.0
        if val <= 50.0:
            return 25.0
        return 100.0

    # The apply function (count) will return a df with the index being the groupby key,
    #
    groups = df.groupby(ranker).count()
    assert groups.index.to_list() == [0.0, 5.0, 25.0, 100.0]
    assert groups["one"].to_list() == [2] * 4


def test_custom_aggregation() -> None:
    def inflate_sum(arr: pd.Series) -> float:
        # You could think of this function as an "apply_tax" function, which
        # returns a value with a different tax rate based on the incoming
        # values.
        return (arr * 1.1).sum()

    df = pd.DataFrame(
        {
            "one": [1, 1, 1, 2, 2, 2],
            "two": [10, 20, 30, 40, 50, 60],
        }
    )
    assert df.groupby("one").aggregate(inflate_sum)["two"].to_list() == [66.0, 165.0]

    # You can also pass multiple aggregation functions, getting back a column
    # for each source column / aggregation combination (in a multiindex)
    aggs = df.groupby("one").aggregate(["mean", "max", inflate_sum])

    assert aggs.loc[:, ("two", "mean")].to_list() == [20.0, 50.0]
    assert aggs.loc[:, ("two", "max")].to_list() == [30, 60]
    assert aggs.loc[:, ("two", "inflate_sum")].to_list() == [66.0, 165.0]

    # A bit cleaner solution, if you only want a single column, would be to
    # perform the aggregate functions on the column (series) alone.
    aggs = df.groupby("one")["two"].aggregate(["mean", "max", inflate_sum])

    assert aggs["mean"].to_list() == [20.0, 50.0]
    assert aggs["max"].to_list() == [30, 60]
    assert aggs["inflate_sum"].to_list() == [66.0, 165.0]


def test_apply() -> None:
    # Apply splits the object into pieces (rows or columns) and concatenates the
    # results. It's the most general form of groupby

    called_with: list[str] = []

    def my_apply(something: pd.Series) -> str:
        print(f"type(something) == {type(something)} {something}")
        called_with.append(something.name)
        return something * 1.1

    df = pd.DataFrame(
        {
            "one": [1, 1, 2, 2, 3],
            "two": [10, 20, 30, 40, 50],
        }
    )
    #
    # axis=0 means apply the function for each column (all rows)
    #
    # axis=1 means apply the function for each row (all columns)
    #
    df.apply(my_apply)
    assert called_with == df.columns.to_list() == ["one", "two"]

    called_with.clear()
    df = df.apply(my_apply, axis=1)
    assert called_with == df.index.to_list() == [0, 1, 2, 3, 4]


def test_transform() -> None:
    # Transform is similar to apply, however a bit more restrictive.
    #
    # The function you use can:
    #
    # * Produce a scalar value to to beoadcast to the shape of the group.
    # * Return an object (series or DF?) in the same shape as the group.
    # * It must not mutate it's input.

    def get_group_mean(group: pd.DataFrame) -> float:
        print(f"type(group) == {type(group)} {group}")
        return group.mean()

    df = pd.DataFrame(
        {
            "key": ["a", "b", "c"] * 4,
            "value": np.arange(12.0),
        }
    )
    g = df.groupby("key")["value"]
    print(g.transform(get_group_mean))



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
