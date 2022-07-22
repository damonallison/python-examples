import datetime

import numpy as np
import pandas as pd
import pytz


def test_pandas_types() -> None:
    """Shows the various numpy and pandas data types."""
    df = pd.DataFrame(
        {
            "type_bool": [
                True,
                False,
            ],
            "type_int8": [
                1,
                2,
            ],
            "type_int32": [
                268435456,  # 2^30
                268435457,
            ],
            "type_int64": [
                17179869184,  # 2^34
                17179869184,
            ],
            "type_float": [
                1.0,
                2.0,
            ],
            "type_date": [
                datetime.date(2000, 1, 1),
                datetime.date(2022, 1, 1),
            ],
            "type_time": [
                datetime.time(1, 2, 3, 0),
                datetime.time(2, 3, 4, 0),
            ],
            "type_datetime": [
                datetime.datetime(2000, 1, 1, 1, 2, 3, 0),
                datetime.datetime(2000, 1, 1, 1, 2, 3, 0),
            ],
            "type_datetime_tz": [
                datetime.datetime(2000, 1, 1, 1, 2, 3, 0, pytz.UTC),
                datetime.datetime(2000, 1, 1, 1, 2, 3, 0, pytz.UTC),
            ],
            "type_str": [
                "one",
                "two",
            ],
        }
    )

    # numpy maps python types to np types:
    #
    # * bool -> np.bool_
    # * int => np.int_
    # * float => np.float_
    # * complex => np.complex_

    assert df["type_bool"].dtype == np.bool_
    assert df["type_bool"].dtype == bool

    assert df["type_int8"].dtype == np.int_
    assert df["type_int8"].dtype == int

    assert df["type_int32"].dtype == np.int_
    assert df["type_int32"].dtype == int

    assert df["type_int64"].dtype == np.int_
    assert df["type_int64"].dtype == int

    assert df["type_float"].dtype == np.float_
    assert df["type_float"].dtype == float

    assert df["type_date"].dtype == np.object_
    assert df["type_date"].dtype == object

    assert df["type_time"].dtype == np.object_
    assert df["type_time"].dtype == object

    assert df["type_datetime"].dtype == "datetime64[ns]"
    assert df["type_datetime_tz"].dtype == "datetime64[ns, UTC]"

    assert df["type_str"].dtype == np.object_
    assert df["type_str"].dtype == object

    print(df.head())

    df["type_date_datetime64"] = pd.to_datetime(df["type_date"])
    df["type_datetime_datetime64"] = pd.to_datetime(df["type_datetime"])
    df["type_datetime_tz_datetime64"] = pd.to_datetime(df["type_datetime_tz"])

    # Is there an actual type object for datetime64[ns]
    # assert df["type_date_datetime64"].dtype == pd.DatetimeTZDtype
    # assert df["type_date_datetime64"].dtype == np.datetime64
    assert df["type_date_datetime64"].dtype == "datetime64[ns]"

    #
    # Time
    #
    # df["type_time_datetime64"] = pd.to_datetime(df["type_time"])
    # df["type_date_datetime64"] = df["type_date"].astype("datetime64[ns]")
    # df["type_time_datetime64"] = df["type_time"].astype("datetime64[ns]")

    print("printing info")
    print(df.info())


def test_datetime() -> None:
    """Deprecated since version 1.11.0: NumPy does not store timezone
    information. For backwards compatibility, datetime64 still parses timezone
    offsets, which it handles by converting to UTC±00:00 (Zulu time). This
    behaviour is deprecated and will raise an error in the future."""
