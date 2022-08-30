import pandas as pd
import numpy as np


def test_concat() -> None:
    """Concat joins multiple dataframes together"""
    df1 = pd.DataFrame({"one": [1, 2, 3], "two": [4, 5, 6], "three": [1, 2, 3]})
    df2 = pd.DataFrame({"one": [7, 8, 9], "two": [10, 11, 12], "four": [1, 2, 3]})

    #
    # Outer will "union" the other axis (columns). No information loss. This is
    # the default.
    #
    # ignore_index=True will *not* use the index on the concatentation axis
    # (row). All rows will be appended. A new RangeIndex from [0-n) will be
    # created for the combined DF.
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


def test_join() -> None:
    df_states = pd.DataFrame({"state": ["MN", "CA", "TX"], "population": [1, 2, 3]})
    df_cities = pd.DataFrame(
        {"state": ["MN", "MN", "CA", "CA"], "city": ["mpls", "mg", "oakland", "la"]}
    )

    # pd.merge is the standard entry point for all merge operations.
    combined = pd.merge(df_states, df_cities, how="inner", on="state")

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
