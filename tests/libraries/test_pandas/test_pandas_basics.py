"""Examples of indexing and selecting data in pandas

Axis labels (indices) provide a way to identify data elements in a DataFrame.
Slicing uses indices to select rows / columns.

Note that `[]` and `.` are intuitive, however does have some optimization
limits. Using data access methods in production code (`.loc` and `.iloc`) is
recommended.

IMPORTANT: Some indexing operations will return views

* `loc`: selection by label
  * df.loc["label"] (single label)
  * df.loc[["a", "b", "c"]] (array)
  * df.loc[[True, False, True]] (boolean mask)
  * df.loc[a:f] (slice (both start and end are included!))
  * df.loc[func] (accepts a callable which accepts a DF and returns valid output for indexing - one of the above)

* iloc`: selection by index (0 based)

"""
import numpy as np
import pandas as pd
import pytest


def test_creation() -> None:
    #
    # Creating a DF from a list-like object
    #
    df = pd.DataFrame([1, 2, 3], columns=["test"])
    assert df["test"].equals(pd.Series([1, 2, 3]))

    assert isinstance(df.index, pd.RangeIndex)
    assert isinstance(df.columns, pd.Index)

    assert df.index.equals(pd.RangeIndex(start=0, stop=3, step=1))
    assert df.columns.equals(pd.Index(["test"]))
    assert df.dtypes.equals(pd.Series([int], index=["test"]))

    #
    # Creating a DF from a dict
    #
    df = pd.DataFrame({"test": [1, 2, 3]})

    assert isinstance(df.index, pd.RangeIndex)
    assert isinstance(df.columns, pd.Index)

    assert df.index.equals(pd.RangeIndex(start=0, stop=3, step=1))
    assert df.columns.equals(pd.Index(["test"]))
    assert df.dtypes.equals(pd.Series([int], index=["test"]))

    #
    #
    #


def test_selection() -> None:
    """Pandas object selection based on index.

    Pandas generally supports three types of multi-axis indexing:

    * `.loc`   Label based. Raises KeyError if the index is not found
    * `.iloc`  Integer position based
    * []       Python __getitem__

    The primary function of [] is selecting out lower dimensional slices. (i.e.,
    selecting a row or column).

    `iloc` and `loc` are the preferred ways to index a DataFrame.

    """

    df = pd.read_csv("./tests/libraries/test_pandas/data/driver_features_v4.csv")

    # The index does *not* need to be unique
    df = df.set_index("driver_id")

    #
    # Select a column by label (as a Series)
    #
    # [] is typically used for selecting columns.
    #
    # Note the index will be included in the scalar and/or new DF as well.
    s = df["experience"].sort_index(axis=0)

    assert isinstance(s, pd.Series)
    assert s.loc[15] == 11860

    assert np.array_equal(s.loc[[15, 131]].values, np.array([11860, 896]))

    # Selecting a column using Python attributes
    #
    # *Don't* do this. A DataFrame should be thought of as a table data
    # structure. Accessing it via a property attr makes it feel like `driver_id`
    # is part of every DataFrame, which it isn't.
    #
    # The attribute will also not be available if it conflicts with an existing
    # method name (min, max, mean, etc). Or if the index is not a valid
    # identifier (i.e., integer values)
    #
    # In general, don't use attribute based accessors!
    s = df.experience
    assert isinstance(s, pd.Series)
    assert s[914189] == 26

    #
    # Selecting a row by label location (loc).
    #
    s = df.loc[914189]
    assert s["experience"] == 26

    #
    # Selecting a row by index location (iloc)
    #
    s = df.iloc[0]
    assert s["experience"] == 26

    #
    # Selecting a range by label ranges (includes both of the labels)
    #
    # Will throw a KeyError if one of the bounds is duplicated.
    #
    # NOTE: Warning - if one of the labels selected identifies more than one row / column,
    # a `KeyError` would be raised.
    assert np.array_equal(df.loc[914189:776076]["experience"].values, [26, 8])

    #
    # Selecting a range by ordinal position (excludes the upper bound)
    #
    assert np.array_equal(df.iloc[0:2]["experience"].values, [26, 8])

    with pytest.raises(IndexError):
        df.iloc[999999]

    #
    # Selection by callable (can be used to filter, but use boolean indexing
    # instead. The callable must return one other valid indexer (scalar, array,
    # or boolean mask)
    #
    # NOTE: pd.NA values are propogated as False in boolean masks
    #
    # [True, False, False, pd.NA] # Selects only the first element.
    rows = df.loc[lambda row: row["experience"] > 25]
    assert rows.loc[914189]["experience"] == 26

    #
    # Filtering using boolean indexing
    #
    # Selects indexes using a boolean vector (mask).
    #
    # Expressions can be combined using
    # & and
    # | or
    # ~ not
    #
    # Any expression with more than one condition must be grouped in
    # parentheses
    rows = df.loc[(df["experience"] > 25) & (df["driver_minutes_per_order_line"] > 1.0)]
    assert rows.loc[914189]["experience"] == 26

    #
    # Using .query() to select based on an expression.
    #
    # This is generally *not* preferred. Use boolean indexing as shown above.
    #
    rows = df.query("experience > 15 & driver_minutes_per_order_line > 1.0")
    assert rows.loc[914189]["experience"] == 26

    #
    # Filtering can be used to select data based on indices.
    #
    # Boolean indexing is generally preferred.
    #
    # Filter is primarily used for pattern matching on indexes.
    #
    # For example, keeping only the columns whose index match a regular
    # expression. Here, we select all columns who's index (column name) ends in
    # `min`.
    rows = df.filter(items=[914189], axis=0)
    assert len(rows)
    assert rows.iloc[0]["experience"] == 26


def test_indexing() -> None:
    """Indexes provide metadata about axes.

    An index allows us to select data using non-ordinal values (i.e.,
    strings).

    """

    # By default, an int64-like RangeIndex (0-n) is applied.
    df = pd.DataFrame(data={"one": [1, 2, 3], "two": [4, 5, 6]})

    assert isinstance(df.index, pd.RangeIndex)
    assert np.array_equal(df.index.values, [0, 1, 2])

    # Set axis 0's (row) index to a custom index using `.index``
    df.index = ["first", "second", "third"]

    # Set axis 1's (col) index to new values
    df.columns = ["one_updated", "two_updated"]

    #                   one_updated  two_updated
    # first             1            4
    # second            2            5
    # third             3            6

    assert isinstance(df.index, pd.Index)
    assert df.loc["first", "one_updated"] == 1
    assert np.array_equal(df.loc["first"].values, np.array([1, 4]))


def test_multindex() -> None:
    """Hierarchial indexing (MultiIndex) allows you to with with higher
    dimensional data.
    """

    # You can create MultiIndex from tuples or from another DataFrame

    dfKeys = pd.DataFrame(
        {
            "first": ["bar", "baz", "foo", "qux"],
            "second": ["one", "two", "one", "two"],
        }
    )
    mi = pd.MultiIndex.from_frame(dfKeys, names=dfKeys.columns)

    # indices have levels whiich can be accessed by ordinal or name
    assert np.array_equal(mi.get_level_values(0).values, dfKeys["first"].values)
    assert np.array_equal(mi.get_level_values("second").values, dfKeys["second"].values)

    df = pd.DataFrame(
        np.arange(16).reshape(4, 4), index=mi, columns=["A", "B", "C", "D"]
    )

    # Use partial indexing to get all rows with bar as the first index level
    assert len(df.loc["bar"]) == 1

    # Retrieve all columns for a particular row by full index
    assert len(df.loc[("bar", "one")]) == 4

    # Select an individual cell
    assert df.loc[("bar", "one"), "A"] == 0
    assert df.loc[("bar", "one"), "B"] == 1

    # Using xs (cross-section) simplifies selecting data at particular level
    # Select all rows with "one" at the second level
    assert len(df.xs("one", level="second")) == 2

    # Ensure the index is sorted or range operations will not work
    df.sort_index(axis=0, inplace=True)

    # Using slices to select ranges on a partial multi-index.
    #
    # Here, we are selecting all rows between and including `bar` and `foo`
    # at the first multi-index level
    assert len(df.loc["bar":"foo"]) == 3


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
