"""Joining and merging.

Pandas provides SQL like join operations with the `merge` function. Merges will
merge DataFrame(s) based on columns or index values.

Join is a convenience function to simplify merging by index. Typically, you'll
want to use `merge` wherever possible. `join` does a left join, `merge` does an
inner join. You can only specify one merge column with `join`.

"""
import pandas as pd


def test_concat() -> None:
    """Concat joins multiple dataframes together.

    In the documentation, concatenation is also called "stacking".
    """
    df1 = pd.DataFrame({"one": [1, 2, 3], "two": [4, 5, 6], "three": [1, 2, 3]})
    df2 = pd.DataFrame({"one": [7, 8, 9], "two": [10, 11, 12], "four": [1, 2, 3]})

    #
    # join="outer" (default): "union" the other axis (columns). No information
    # loss.
    #
    # join="inner": only return columns in both DataFrames (intersection).
    #
    # ignore_index=True discards the indexes from each DataFrame and
    # concatenates the data in the columns only, assigning a new default index.
    #
    combined = pd.concat([df1, df2], axis=0, join="outer", ignore_index=True)

    assert len(combined) == 6
    assert sorted(combined.columns.to_list()) == ["four", "one", "three", "two"]

    #
    # Inner will "intersect" the other axis (columns). Columns not in all DFs are lost.
    #
    combined = pd.concat([df1, df2], axis=0, join="inner", ignore_index=True)

    assert len(combined) == 6
    assert sorted(combined.columns.to_list()) == ["one", "two"]


def test_concat_columns() -> None:
    df1 = pd.DataFrame({"one": [1, 2, 3]}, index=["a", "b", "c"])
    df2 = pd.DataFrame({"one": [3, 4]}, index=["c", "d"])

    #
    # When concatenating columns, you may need to identify the origin DataFrame.
    # Using `keys` will create a hierarchial column index with the keys as the
    # outermost index.
    #
    # verify_integrity ensures that all column names are unique. Since we are using the
    combined = pd.concat([df1, df2], axis=1, keys=["df1", "df2"], verify_integrity=True)
    #
    #   df1       df2
    #   one       one
    # a 1.0       nan
    # b 2.0       nan
    # c 3.0       3.0
    # d nan       4.0
    #

    assert combined.loc["a", ("df1", "one")] == 1.0
    assert pd.isna(combined.loc["a", ("df2", "one")])


def test_join() -> None:
    """pd.merge is the main entry point for all join (merge) operations.

    NOTE: The order of the merge column output is *unspecified*.

    `inner` is the default join type. `left`, `right`, and `outer` (union of all
    keys) are also available, similar to SQL.
    """

    df_states = pd.DataFrame({"state": ["MN", "CA", "TX"], "population": [1, 2, 3]})
    df_cities = pd.DataFrame(
        {"state": ["MN", "MN", "CA", "CA"], "city": ["mpls", "mg", "oakland", "la"]}
    )

    # If the `on` columns are not specified, overlapping column names are used
    # as the join keys. It's best practice to specify the join keys.
    combined = pd.merge(df_states, df_cities, how="inner", on="state")
    # print(combined)

    assert combined.shape == (4, 3)
    assert combined.loc[combined["city"] == "mpls"].iloc[0]["population"] == 1
    assert combined.loc[combined["city"] == "oakland"].iloc[0]["population"] == 2


def test_join_df_series() -> None:
    states = pd.Series([1, 2, 3, 4], index=["MN", "MN", "CA", "TX"], name="population")
    df_cities = pd.DataFrame(
        {"state": ["MN", "MN", "CA", "CA"], "city": ["mpls", "mg", "oakland", "la"]}
    )

    # pd.merge is the standard entry point for all merge operations.
    combined = pd.merge(
        states,
        df_cities,
        how="inner",
        left_index=True,  # use the left index as it's join key
        right_on="state",  # use the right "state" column as it's join key
    )
    assert len(combined) == 6

    # Each MN city should have *two* populations
    assert combined.loc[(combined["city"] == "mpls")]["population"].values.tolist() == [
        1,
        2,
    ]
    assert combined.loc[(combined["city"] == "mg")]["population"].values.tolist() == [
        1,
        2,
    ]
    assert combined.loc[(combined["city"] == "oakland")][
        "population"
    ].values.tolist() == [3]


def test_left_join() -> None:
    df_states = pd.DataFrame({"state": ["MN", "CA", "TX"], "population": [1, 2, 3]})
    df_cities = pd.DataFrame(
        {
            "state": ["MN", "MN", "CA", "CA", "NY"],
            "city": ["mpls", "mg", "oakland", "la", "new york"],
        }
    )

    # a left join is like SQL's `LEFT OUTER JOIN`
    #
    # all keys from the left frame are used. If there are no matching rows in
    # the right frame, NaN is used
    combined = pd.merge(df_states, df_cities, how="left", on="state")
    # TX doesn't have a city
    assert pd.isna(combined.loc[(combined["state"] == "TX")]["city"].iloc[0])

    # a right join is like SQL's `RIGHT OUTER JOIN`
    combined = pd.merge(df_states, df_cities, how="right", on="state")
    # NY doesn't have a population
    assert pd.isna(combined.loc[combined["state"] == "NY"]["population"].iloc[0])
