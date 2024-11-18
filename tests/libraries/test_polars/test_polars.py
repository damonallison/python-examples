"""Polars is a DF library written in Rust which uses Apache Arrow as it's foundation"""

import polars as pl


def test_polars_create() -> None:
    s = pl.Series("a", [1, 2, 3, 4])

    assert s.shape == (4,)
    assert s.dtype == pl.Int64

    df = pl.DataFrame({"one": [1, 2, 3], "two": [4, 5, 6]})

    assert df.columns == ["one", "two"]
    assert df.get_column("one").equals(pl.Series("one", [1, 2, 3]))
