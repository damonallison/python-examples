import datetime
import math

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_string_dtype
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

    assert np.issubdtype(df["type_bool"].dtype, np.bool_)
    assert np.issubdtype(df["type_int8"].dtype, np.integer)
    assert np.issubdtype(df["type_int32"].dtype, np.integer)
    assert np.issubdtype(df["type_int64"].dtype, np.integer)
    assert np.issubdtype(df["type_float"].dtype, np.floating)
    assert np.issubdtype(df["type_date"].dtype, np.object_)
    assert np.issubdtype(df["type_time"].dtype, np.object_)

    assert is_datetime64_any_dtype(df["type_datetime"])
    assert is_datetime64_any_dtype(df["type_datetime_tz"])

    assert np.issubdtype(df["type_str"].dtype, np.object_)

    df["type_date_datetime64"] = pd.to_datetime(df["type_date"])
    df["type_datetime_datetime64"] = pd.to_datetime(df["type_datetime"])
    df["type_datetime_tz_datetime64"] = pd.to_datetime(df["type_datetime_tz"])

    assert is_datetime64_any_dtype(df["type_date_datetime64"])
    assert is_datetime64_any_dtype(df["type_datetime_datetime64"])
    assert is_datetime64_any_dtype(df["type_datetime_tz_datetime64"])


def test_datetime() -> None:
    """Deprecated since version 1.11.0: NumPy does not store timezone
    information. For backwards compatibility, datetime64 still parses timezone
    offsets, which it handles by converting to UTC±00:00 (Zulu time). This
    behaviour is deprecated and will raise an error in the future."""


def test_na() -> None:
    #
    # Pandas adopted R's convention by referring to missing data as "NA".
    #
    # pd.NA, np.nan, and None are treated as NA.
    #
    assert pd.isna(pd.NA)
    assert pd.isna(np.nan)
    assert pd.isna(None)

    data = pd.Series(["test", pd.NA, np.nan, None])
    assert np.issubdtype(data.dtype, np.object_)

    # When working with string columns that potentially have NA values, use the
    # `.str` method, which skips over N/A values
    assert data.str.upper().to_list() == ["TEST", pd.NA, np.nan, None]

    data = data.astype("string")
    assert is_string_dtype(data)

    assert data.isna().to_list() == [False, True, True, True]
    assert len(data.dropna()) == 1

    # Data imputation
    data = pd.Series([1.0, 2.0, 3.0, 4.0, pd.NA, None, np.nan])
    assert math.isclose(data.mean(), 2.5, abs_tol=0.01)

    data_imputed = data.fillna(data.mean())
    assert data_imputed[data.isna()].to_list() == [2.5, 2.5, 2.5]
