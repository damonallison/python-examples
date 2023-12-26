"""
A timeseries in pandas is a Series or DataFrame that has a DateTimeIndex.

* What is the relationship between `datetime`, numpy's `datetime64`, and pandas
  Timestamp?
*
* datetime does not have nanosecond precision
* datetime64 has something strange w/ timezone handling
* pd.Timestamp has nanosecond precision and can be made timezone aware

Pandas supports "Periods", which allow you to create and aggregate data by fixed
periods. Pandas supports accounting and fiscal year periods like "A-SEP"
(annual, starting in september), wihch helps accounting and fiscal year /
quarter analysis.

Pandas supports rolling window functions (rolling and expanding).

"""
import datetime
import math
import numpy as np
import pandas as pd
import pandas.api.types as pd_types
import pytz

import matplotlib.pyplot as plt


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

    ts = ts.tz_localize(datetime.timezone.utc)
    assert ts.tz == datetime.timezone.utc

    ts = ts.tz_convert("America/New_York")
    assert ts.tz == pytz.timezone("America/New_York")

    s = pd.Series(np.random.standard_normal(len(ts)), index=ts)

    # If two time series with different TZs are combined, the result will be
    # UTC. Pandas stores Timestamp values as UTC under the hood.
    s1 = s[:5].tz_convert("America/Chicago")
    s2 = s[5:]
    assert (s1 + s2).index.tz == datetime.timezone.utc


def test_periods() -> None:
    # Periods represent time spans.
    #
    # Periods are meant for accounting-like calculations which reside on fiscal
    # years and quarters which may not line up with calendar year / quarters.

    pi = pd.period_range("2022-09-01", "2022-12-31", freq="D")
    assert pd_types.is_period_dtype(pi)

    # Resampling is the process of converting a time series to a different
    # frequency. Moving to a lower frequency (say day -> month) is called
    # downsampling. Higher, upsampling.

    # Downsampling
    s = pd.Series(np.random.standard_normal(len(pi)), index=pi)
    sm = s.resample(rule="M", kind="period").mean()

    assert len(sm) == 4
    assert (
        sm.index.to_list()
        == pd.period_range("2022-09-01", "2022-12-31", freq="M").to_list()
    )

    # Upsampling requires interpolation to fill in the missing gaps.
    # This example uses ffill() (forward fill) to interpolate.

    sd = sm.resample(rule="D", kind="period").ffill()
    assert (sd["2022-09-01":"2022-09-30"] == sd["2022-09-01"]).all()
    assert (sd["2022-10-01":"2022-10-31"] == sd["2022-10-01"]).all()


def test_window_functions() -> None:
    # 3 hours
    s = pd.Series(
        np.random.standard_normal(10800),
        pd.date_range("2022-09-01", periods=10800, freq="1s"),
    )

    # Notice the first 249 data points are nan
    s_rolling = s.rolling(250).mean()
    assert pd.isnull(s_rolling.iloc[0:249]).all()

    assert math.isclose(s_rolling.iloc[249], s.iloc[0:250].mean())

    # .expanding() will start the tine window from the same point as the series
    # and increase the size of the window until it encompasses the whole series.

    s_expanding = s.expanding().mean()

    for i in range(len(s)):
        assert math.isclose(s_expanding.iloc[i], s.iloc[0 : i + 1].mean())
