"""
Pandas supports a *lot* of custom file formats. Among the most imporant:

* CSV: ubiquitous, but limited. No metadata storage.

* Parquet: column-oriented, typed, compressed, and binary. It handles custom
  types (dates, times, arrays) very well and is the recommended choice.

* HDF5: "Hierarchial Data Format" - meant for storing large amounts of
  scientific array data. Supports storing multiple data sets w/ metadata, on the
  fly compression, and compact storage for repeated values.
"""
import ast
from datetime import date, datetime, time
import logging


import numpy as np
import pandas as pd
from pandas.api import types as pdtypes
import pytest
import pytz
import sqlalchemy as sqla
import tempfile


logger = logging.getLogger(__name__)


@pytest.mark.skip(reason="requires postgresql (docker compose up)")
def test_sqlalchemy() -> None:
    db = sqla.create_engine("postgresql://postgres:postgres@localhost:5432")

    try:
        db.execute("drop table if exists test_pandas_test_sqlalchemy")
        db.execute(
            """
            create table if not exists test_pandas_test_sqlalchemy (
                type_integer int,
                type_text text,
                type_boolean boolean,
                type_date date,
                type_time time,
                type_timestamp timestamp,
                type_timestamptz timestamptz,
                type_jsonb jsonb
            )
            """
        )
        db.execute(
            """
                insert into test_pandas_test_sqlalchemy
                (
                    type_integer,
                    type_text,
                    type_boolean,
                    type_date,
                    type_time,
                    type_timestamp,
                    type_timestamptz,
                    type_jsonb
                )
                values
                (
                    1,
                    'test1',
                    true,
                    '2022-09-06',
                    '01:02:03',
                    '2022-09-06 01:02:03',
                    '2022-09-06 01:02:03+01:00',
                    '[{"hello": "world"}]'
                )
            """
        )
        # db.execute("insert into test_pandas_test_sqlalchemy values (2, 'test2')")
        # db.execute("insert into test_pandas_test_sqlalchemy values (3, 'test3')")

        df = pd.read_sql(
            "select * from test_pandas_test_sqlalchemy order by type_integer asc",
            con=db,
        )

        assert np.issubdtype(df["type_integer"].dtype, np.integer)
        assert np.issubdtype(df["type_text"].dtype, object)
        assert np.issubdtype(df["type_boolean"].dtype, bool)
        assert np.issubdtype(df["type_date"].dtype, date)
        assert np.issubdtype(df["type_time"].dtype, time)
        assert pdtypes.is_datetime64_any_dtype(df["type_timestamp"].dtype)
        assert pdtypes.is_datetime64_any_dtype(df["type_timestamptz"].dtype)

        print(type(df["type_jsonb"][0]))

        assert df["type_integer"][0] == 1
        assert df["type_text"][0] == "test1"
        assert df["type_boolean"][0] == True
        assert df["type_date"][0] == date(2022, 9, 6)
        assert df["type_time"][0] == time(1, 2, 3)
        assert df["type_timestamp"][0] == datetime(2022, 9, 6, 1, 2, 3)
        assert df["type_timestamptz"][0] == datetime(
            2022, 9, 6, 0, 2, 3, 0, tzinfo=pytz.UTC
        )
    finally:
        db.execute("drop table if exists test_pandas_test_sqlalchemy")


def test_csv() -> None:
    """Tests reading and writing to csv.

    When writing to CSV, type information is not persisted as it is with Parquet
    or other binary formats.

    Therefore, custom processing needs to be done to convert columns into the
    desired type. For example, dates, times, datetimes, and arrays will be read
    in as strings and need custom processing to be converted into their desired
    type.

    Pandas has support for handling datetime columns. Time columns and arrays
    must be manually
    """
    df = pd.DataFrame(
        {
            "s": ["one", "two", "three"],
            "d": [date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3)],
            "t": [time(1, 1, 1, 0), time(2, 2, 2, 0), time(3, 3, 3, 0)],
            "dt": [
                datetime(2022, 1, 1, 1, 1, 1, 0),  # tz naive
                datetime(2022, 2, 1, 1, 1, 1, 0, tzinfo=pytz.UTC),  # tz is stripped
                datetime(
                    2022, 3, 1, 1, 1, 1, 0, tzinfo=pytz.FixedOffset(-60)
                ),  # tz is stripped
            ],
            # Arrays are serialzied / deserialized as strings (str)
            "ai": [[1.1, 2.1], [1.1, 2.2], [3.3, 3.3]],
            "as": [["one", "two"], ["three", "four"], ["five", "six"]],
        }
    )

    with tempfile.TemporaryFile() as f:
        #
        # When writing to csv, you can specify `date_format` - the format string
        # for datetime objects. In most cases, the default (str) is sufficient.
        # You only need to override the default if want to write dates in a
        # custom format.
        #
        df.to_csv(f, index=False)

        f.seek(0)

        #
        # When reading dates, there are multiple datetime parameters that can be
        # used to control date parsing
        #
        # * parse_dates takes a boolean (default False) or a list of columns to
        #   attempt to convert to datetime.
        #
        # * date_parser: func to use for parsing dates
        #
        df = pd.read_csv(
            filepath_or_buffer=f,
            parse_dates=["d"],
            converters={
                # Custom column convertes for handling time and array
                "t": lambda x: pd.to_datetime(x, format="%H:%M:%S").time(),
                "ai": lambda x: ast.literal_eval(x),
                "as": lambda x: ast.literal_eval(x),
            },
        )
        # df["t"] = pd.to_datetime(df["t"], format="%H:%M:%S").dt.time

    # If a column or index cannot be represented as an array of datetime, say
    # because of an unparsable value or a mixture of timezones, the column or
    # index will be returned as an unaltered `object` data type. For
    # non-standard datetime parsing, use pd.to_datetime() after read_csv()

    df["dt"] = pd.to_datetime(df["dt"], format="ISO8601")

    row0 = df.loc[0]
    assert isinstance(row0["s"], str)
    assert isinstance(row0["d"], date)
    assert isinstance(row0["t"], time)
    assert isinstance(row0["dt"], datetime)

    assert isinstance(row0["ai"], list) and isinstance(row0["ai"][0], float)
    assert isinstance(row0["as"], list) and isinstance(row0["as"][0], str)

    # # Notice the offsets were *not* persisted in the 2nd and 3rd DTs
    assert df["dt"][0] == datetime(2022, 1, 1, 1, 1, 1, 0)
    assert df["dt"][1] == datetime(2022, 2, 1, 1, 1, 1, 0, pytz.UTC)
    assert df["dt"][2] == datetime(2022, 3, 1, 1, 1, 1, 0, pytz.FixedOffset(-60))

    assert df["ai"][0] == [1.1, 2.1]
    assert df["as"][0] == ["one", "two"]


def test_parquet() -> None:
    """Tests reading and writing to parquet.

    https://arrow.apache.org/docs/python/pandas.html
    """
    df = pd.DataFrame(
        {
            "s": ["one", "two", "three"],
            "d": [date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3)],
            "t": [time(1, 1, 1, 0), time(2, 2, 2, 0), time(3, 3, 3, 0)],
            "dt": [
                datetime(2022, 1, 1, 1, 1, 1, 0),  # tz naive
                datetime(2022, 2, 1, 1, 1, 1, 0, tzinfo=pytz.UTC),  # tz is stripped
                datetime(
                    2022, 3, 1, 1, 1, 1, 0, tzinfo=pytz.FixedOffset(-60)
                ),  # tz is stripped
            ],
            "ai": [[1.1, 2.1], [1.2, 2.2], [3.3, 3.3]],
            "as": [["one", "two"], ["three", "four"], ["five", "six"]],
        }
    )
    with tempfile.TemporaryFile() as f:
        df.to_parquet(f)
        df = pd.read_parquet(f)

    assert isinstance(df["s"][0], str)
    assert isinstance(df["d"][0], date)
    assert isinstance(df["t"][0], time)
    assert isinstance(df["dt"][0], pd.Timestamp)
    assert isinstance(df["ai"][0], np.ndarray)
    assert isinstance(df["as"][0], np.ndarray)

    # Notice the offsets were *not* persisted in the 2nd and 3rd DTs
    assert df["dt"][0] == pd.Timestamp(datetime(2022, 1, 1, 1, 1, 1, 0))
    assert df["dt"][1] == pd.Timestamp(datetime(2022, 2, 1, 1, 1, 1, 0))
    assert df["dt"][2] == pd.Timestamp(datetime(2022, 3, 1, 2, 1, 1, 0))

    assert df["ai"][0].tolist() == [1.1, 2.1]
    assert df["as"][0].tolist() == ["one", "two"]
