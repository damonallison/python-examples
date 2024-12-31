"""Polars is a DF library written in Rust which uses Apache Arrow as it's foundation"""

import json
import os
from pathlib import Path

import polars as pl

"""
TODO:

* contexts
* expressions

* data types (dates)

"""
def test_polars_create() -> None:
    """The core data structures in polars are series and dataframes."""

    # data types are inferred by default, overridable with `dtype`
    s = pl.Series("a", [1, 2, 3, 4], dtype=pl.Int8)
    assert s.dtype == pl.Int8

    assert s.shape == (4,)
    assert s.dtype == pl.Int8

    df = pl.DataFrame({"one": [1, 2, 3], "two": [4, 5, 6]})

    # summary statistics
    print(df.head())
    print(df.describe())
    assert df.schema == pl.Schema(schema={"one": pl.Int64, "two": pl.Int64})

    assert df.columns == ["one", "two"]
    assert df.get_column("one").equals(pl.Series("one", [1, 2, 3]))

def test_expressions_contexts() -> None:
    df = pl.DataFrame({"person": ["damon"], "height": [1.8], "weight": [69.4]})
    # Expressions are lazy representations of data transformations. expressions
    # are abstract computations that can be saved in a variable, manipulated
    # further, and evaluated later (in a context).
    bmi = df["weight"] / (df["height"] ** 2)
    assert isinstance(bmi, pl.Series)
    assert int(bmi[0]) == 21

    # Expressions are executed in "contexts" to produce a result.
    # Examples of contexts include: select, with_columns, filter, group_by

    # with_columns creates a new df with the columns from the original df and
    # new columns according to the given input expressions.
    df = df.with_columns(bmi=bmi)
    assert df.columns == ["person", "height", "weight", "bmi"]

    # select will return a new df with only the given expressions
    df2 = df.select(bmi=bmi)
    assert df2.columns == ["bmi"]
    print(df.head())
    print(df2.head())

    # filter: filters the rows of a dataframe based on one or more boolean
    # expressions.

    df3 = df.filter(
        pl.col("height") > 1.0
    )
    assert len(df3) == 1
    assert df3.shape[0] == 1


def test_lazy_api(tmp_path: Path) -> None:
    # The lazy API is preferred in most cases.
    #
    # Why?
    #
    # Predicate pushdown. Apply filters when reading, thus only reading in data
    # passing the predicate.
    #
    # Projection pushdown. Select only the columns needed while reading the
    # dataset.

    # assume we have some huge data here
    huge_data = {"one": range(1000), "two": range(1000)}

    # Write an "ndjson" (newline delimited JSON / JSON Lines)
    #
    # Each line represents a separate JSON object.
    tmp_file = tmp_path / "test.json"
    with open(tmp_file, "w") as f:
        for i in range(len(huge_data["one"])):
            f.write(json.dumps({
                "one": huge_data["one"][i],
                "two": huge_data["two"][i],
            }) + os.linesep)

    # Build a query. Calling collect on the query will execute it.
    q = (
        pl.scan_ndjson(tmp_file)
        .filter(pl.col("one") % 2 == 0)
    )

    # See what types of optimizations polars performs on your queries
    print(q.explain())
    df = q.collect()
    assert len(df) == 500




