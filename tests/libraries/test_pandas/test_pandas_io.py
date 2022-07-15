from datetime import date, datetime, time
import logging
import pytz
from tempfile import TemporaryFile

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
            "ai": [[1, 2], [1.0, 2.0], [3.0, 3.0]],
            "as": [["one", "two"], ["three", "four"], ["five", "six"]],
        }
    )
    with TemporaryFile() as f:
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

    assert df["ai"][0].tolist() == [1, 2]
    assert df["as"][0].tolist() == ["one", "two"]
