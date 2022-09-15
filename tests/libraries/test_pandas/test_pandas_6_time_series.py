"""
A timeseries in pandas is a Series or DataFrame that has a DateTimeIndex.

* What is the relationship between `datetime`, numpy's `datetime64`, and pandas Timestamp?
*
* datetime does not have nanosecond precision
* datetime64 has something strange w/ timezone handling
* pd.Timestamp has nanosecond precision and can be made timezone aware
"""
import datetime
import numpy as np
import pandas as pd
import pandas.api.types as pd_types
import pytz


def test_basics() -> None:
    # Basics of working with timestamp columns

    df = pd.DataFrame(
        data=np.random.standard_normal(10),
        index=pd.date_range("2022-09-01", periods=10),
    )

    assert pd_types.is_datetime64_any_dtype(df.index)

    # Individual values are pd.Timestamp values
    assert isinstance(df.index[0], pd.Timestamp)
    assert df.index[0] == pd.Timestamp("2022-09-01")

    # Slicing (careful: includes both range ends)
    assert len(df["2022-09-01":"2022-09-05"]) == 5
    df["2022-09-01":"2022-09-05"].first == pd.Timestamp("2022-09-01")
    df["2022-09-01":"2022-09-05"].last == pd.Timestamp("2022-09-05")

    # In general, you can use pd.Timestamp
    assert datetime.datetime(2022, 9, 1) == pd.Timestamp("2022-09-01")

    df2 = pd.DataFrame(
        {
            "dt": [datetime.datetime(2022, 9, 1), datetime.datetime(2022, 9, 2)],
            "pd_ts": [pd.Timestamp("2022-09-01"), pd.Timestamp("2022-09-02")],
        }
    )
    assert pd_types.is_datetime64_any_dtype(df2["dt"])
    assert pd_types.is_datetime64_any_dtype(df2["pd_ts"])

    print(type(df2["dt"][0]))
    print(type(df2["pd_ts"][0]))


def test_timezone_handling() -> None:
    # Use tz_convert (if the datetime is already tz-aware) to convert
    # pd.Timestamps into a new tz.
    #
    # Use tz_localize to convert tz naive timestamps to a tz

    ts = pd.date_range("2022-09-01", periods=10)

    assert ts.tz is None

    ts = ts.tz_localize("UTC")
    assert ts.tz == pytz.UTC

    ts = ts.tz_convert("America/New_York")
    assert ts.tz == pytz.timezone("America/New_York")
