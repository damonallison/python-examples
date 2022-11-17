"""Pandas provides data manipulation tools for working with 2D data.

Pandas has two main data structures:

1. Series: A list of values with an index.
2. DataFrame: A 2D data set with row and column indices.

Series
------


* The line between numpy / pandas


Examples of indexing and selecting data in pandas:
-------------------------------------------------

** IMPORTANT **

Always use `.loc` and `.iloc` to perform selection. Using df[] will use index
based selection (loc) if the index is numeric, otherwise it will use ordinal
based selection (iloc). To ensure you're always obtaining the values you intend
to, use `.loc` and `.iloc`.

** IMPORTANT **

When slicing based on labels (loc), both endpoints are included. This is
different than python!

IMPORTANT: Some indexing operations (non-"advanced") will return views. Advanced
indexing operations will return copies. "Advanced" indexing includes:

* Boolean masking:
    df.loc[df["col"] > 2]

* Selection with lists:
    > df.loc[[0, 1, 2]]


* `loc`: selection by label * df.loc["label"] (single label) * df.loc[["a", "b",
  "c"]] (array) ("advanced" indexing - returns a copy) * df.loc[[True, False,
  True]] (boolean mask) ("advanced" indexing - returns a copy) * df.loc[a:f]
  (slice (both start and end are included!)) * df.loc[func] (accepts a callable
  which accepts a DF and returns valid output for indexing - one of the above)

* iloc`: selection by index (0 based)

"""

from datetime import datetime
import json
import pathlib

import numpy as np
import pandas as pd
import pytest


def test_series() -> None:
    "Series is a single array of values, each with an index value."
    s = pd.Series([10, 20, 30], index=["A", "B", "C"])

    assert isinstance(s.index, pd.Index)

    # Data is selected by index (label)
    assert "A" in s
    assert "D" not in s

    # Mathematical operations operate on data by index value. This is called
    # "data alignment". This is similar to how relational SQL does joins based
    # on join column values.
    s2 = pd.Series([100, 200, 300], index=["B", "C", "D"])
    s3 = s + s2

    # "A" and "D" do not exist in both series, therefore N/A
    assert pd.isna(s3["A"]) and pd.isna(s3["D"])
    assert (s3["B"], s3["C"]) == (120, 230)

    # An entire index can be updated, but an index value cannot be mutated
    s2.index = ["A", "B", "C"]
    assert (s + s2).values.tolist() == [110, 220, 330]

    with pytest.raises(TypeError):
        s2.index[0] = "0"


def test_series_from_dict() -> None:
    """You can think of a series as a labeled set of values, like an ordered
    dictionary."""

    states = {"Minnesota": 3500, "Texas": 11222}
    s = pd.Series(states)

    assert isinstance(s.index, pd.Index)
    assert s.index.values.tolist() == ["Minnesota", "Texas"]
    assert s.size == 2
    assert s["Minnesota"] == 3500

    s = pd.Series(states, index=["Minnesota", "Texas", "New York"])
    assert pd.isna(s["New York"])

    assert "Not there" not in s.index
    assert "Not there" not in s
    assert "Minnesota" in s


def test_dataframe_creation() -> None:
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
    # Creating a DF from a dict. Each list must be equal length.
    #
    df = pd.DataFrame(
        {
            "test": [1, 2, 3],
            "test2": [2, 3, 4],
        }
    )

    assert isinstance(df.index, pd.RangeIndex)
    assert isinstance(df.columns, pd.Index)

    assert df.index.equals(pd.RangeIndex(start=0, stop=3, step=1))
    assert df.columns.equals(pd.Index(["test", "test2"]))
    assert df.dtypes.equals(pd.Series([int, int], index=["test", "test2"]))


def test_dataframe_indexing() -> None:
    """Indexing is the process of selecting data based off index values.

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
    assert s.loc[15] == 11860  # Index based (driver_id), not ordinal.

    # A form of "Advanced indexing" - selecting multiple series values based off
    # a list. (i.e., [15, 131])
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
    # While you *could* use python indexing to select values from a series, the
    # behavior changes based on the index type. With integer indexes, s[0] will
    # select the row with the index value 0. With string indexes, s[0] will
    # always select the first value (by ordinal).
    #
    # In order to ensure you're using the index value or ordinal to select
    # values, always use `loc` and `iloc`, which uses index values and ordinal
    # values, respectively.
    #

    #
    # Selecting a row by label value (loc).
    #
    s = df.loc[914189]
    assert s["experience"] == 26

    #
    # Selecting a row by index ordinal (iloc)
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


def test_dataframe_slicing() -> None:
    """IMPORTANT: Slicing by indexes will return *BOTH ENDS* of the range.
    Slicing by ordinal will not."""

    df = pd.DataFrame(["this", "is", "a", "test"], columns=["one"])

    assert df.loc[0:1, "one"].to_list() == ["this", "is"]
    assert df.iloc[0:1, 0].to_list() == ["this"]


def test_dadtaframe_indexes() -> None:
    """Indexes provide metadata about axes.

    An index allows us to select data using non-ordinal values (i.e., strings).
    Indexes can have duplicate values and may or may not be monotonically
    increasing.
    """
    # By default, an int64-like RangeIndex (0-n) is applied.
    df = pd.DataFrame(data={"one": [1, 2, 3], "two": [4, 5, 6]})

    # By default, a 0 based RangeIndex is created for each DF
    assert isinstance(df.index, pd.RangeIndex)
    assert np.array_equal(df.index.values, [0, 1, 2])
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique

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


def test_reindexing() -> None:
    """Reindexing: rearranging the values to align with a new index"""
    df = pd.DataFrame(["cole", "lily", "grace"], index=[3, 2, 1], columns=["kids"])

    # Calling reindex arranges the data acccording to the new index, introducing
    # pd.na values if any index values are not present.
    df = df.reindex([1, 2, 3, 4])

    assert df.loc[[1, 2, 3], "kids"].tolist() == ["grace", "lily", "cole"]
    assert pd.isna(df.loc[4, "kids"])

    # Indexes do not have to be unique. `reset_index` will reset the index. By
    # default, the old index is added to the frame as a column called "index".
    # To prevent this, use drop=True.

    df = pd.DataFrame(
        ["damon", "kari", "grace", "lily", "cole"],
        index=["p", "p", "c", "c", "c"],
        columns=["name"],
    )
    df = df.reset_index(drop=True)
    assert df.columns == pd.Index(["name"])


def test_multindex() -> None:
    """Hierarchial indexing (MultiIndex) allows you to with with higher
    dimensional data in a low dimensional form.
    """

    # You can create MultiIndex from tuples or from another DataFrame

    dfKeys = pd.DataFrame(
        {
            "first": ["bar", "baz", "foo", "qux"],
            "second": ["one", "two", "one", "two"],
        }
    )
    mi = pd.MultiIndex.from_frame(dfKeys, names=dfKeys.columns)
    assert mi.nlevels == 2
    assert set(mi.names) == {"first", "second"}

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
    # print(df.xs("one", level="second"))

    # Ensure the index is sorted or range operations will not work
    df.sort_index(axis=0, inplace=True)

    # Using slices to select ranges on a partial multi-index.
    #
    # Here, we are selecting all rows between and including `bar` and `foo`
    # at the first multi-index level
    assert len(df.loc["bar":"foo"]) == 3


def test_arithmetic_operations() -> None:
    """Shows how arithmetic operations work between series and frames, and two
    frames.

    When performing arithmetic operations, the "axis" value used is the "axis to
    match on".
    """

    #   one     two
    #   0       1
    #   2       3
    #   4       5
    df = pd.DataFrame(np.arange(6).reshape((3, 2)), columns=["one", "two"])

    series = df.iloc[0]  # first row == (0, 1)

    assert series.index.values.tolist() == ["one", "two"]
    assert series.values.tolist() == [0, 1]

    # Arithmetic operations between frames and series match the index of the
    # series (column names) on the columns of the frame, broadcasting over the
    # rows by default.

    df2 = df.sub(series)  # axis=1

    #   one     two
    #   0       0
    #   2       2
    #   4       4
    assert df2.values.flatten().tolist() == [0, 0, 2, 2, 4, 4]

    # If you want to match on rows, use axis=0. This will match the index of the
    # series (row indices) on the rows of the frame, broadcasting over the
    # columns by default.
    series = df.loc[:, "one"]

    df2 = df.sub(series, axis=0)
    #   one     two
    #   0       1
    #   0       1
    #   0       1
    assert df2.values.flatten().tolist() == [0, 1, 0, 1, 0, 1]


def test_descriptive_statistics() -> None:

    df = pd.DataFrame([[1.0, 2.0, np.nan], [3.0, np.nan, 4.0]])

    assert df.sum(axis=0).tolist() == [4.0, 2.0, 4.0]  # row totals across all columns
    assert df.sum(axis=1).tolist() == [3.0, 7.0]  # column totals across all rows


def test_function_application() -> None:

    df = pd.DataFrame(np.arange(6).reshape((3, 2)), columns=["one", "two"])

    # numpy ufuncs work with dataframes
    df: pd.DataFrame = np.add(df, 1)
    assert np.array_equal(df.values.flatten(), np.arange(start=1, stop=7))

    # Apply a function to each column or row

    def add_one(x: pd.Series) -> pd.Series:
        # print(f"add_one applying to:\n{x}")
        return np.add(x, 1)

    #
    # The "axis" parameter determines which axis to use when applying the
    # function. This is confusing, so pay attention!
    #
    # axis=0 means "apply to all rows". Each column is sent to apply.
    #
    # axis=1 means "apply to all columns". Each row is sent to apply.
    #
    df = df.apply(add_one)  # by default, apply to each column
    assert np.array_equal(df.values.flatten(), np.arange(start=2, stop=8))

    df = df.apply(add_one, axis=1)  # apply to each row
    assert np.array_equal(df.values.flatten(), np.arange(start=3, stop=9))

    # Apply a function to each element with `applymap`
    def add_one_to_scalar(x: int) -> int:
        return x + 1

    df = df.applymap(add_one_to_scalar)
    assert np.array_equal(df.values.flatten(), np.arange(start=4, stop=10))


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


def test_pandas_itertuples() -> None:
    """itertuples return each row as a tuple. Providing a `name` will
    return a namedtuple.
    """
    d = {
        "driver_id": [1, 2],
        "first": ["damon", "cole"],
        "last": ["allison", "allison"],
    }
    df = pd.DataFrame(d)

    for i in range(5):
        print(df.shape)
        df = df.append(df, ignore_index=True)

    cache = {}
    for t in df.itertuples(index=False, name="Driver"):
        cache[t.driver_id] = t

    print(df.shape)
    print(df.head())


def test_json_serialization(tmp_path: pathlib.Path) -> None:
    d = {
        "col1": [1, 2],
        "col2": ["test", None],
    }

    # "columns" is the default
    df = pd.DataFrame.from_dict(d, orient="columns")

    file = tmp_path / "test_json_serialization.json"
    df.to_json(file, orient="split")
    print(file.read_text(encoding="UTF-8"))

    df2 = pd.read_json(file.read_text(encoding="UTF-8"), orient="split")
    assert df2.equals(df)

    j = """
    {
        "columns": ["col1", "col2"],
        "data": [[1, "test"], [2, null]]
    }
    """
    df2 = pd.read_json(j, orient="split")
    assert df2.equals(df)
