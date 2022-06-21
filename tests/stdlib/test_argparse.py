from typing import Optional

import argparse
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Args:
    backfill_date: datetime

    def __str__(self):
        return f"Args(backfill_date={self.backfill_date}"


def parse_date_with_default(s: str, default: Optional[datetime]) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return default


def test_parse_args() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backfill-date",
        dest="backfill_date",
        type=lambda s: parse_date_with_default(s, datetime.now()),
        help='The date to start backfill from. Format "YYYY-MM-dd"',
    )
    args = parser.parse_args(["--backfill-date", "2020-01-01"])
    a = Args(backfill_date=args.backfill_date)
    assert a.backfill_date == datetime(2020, 1, 1)

    args = parser.parse_args(["--backfill-date", "Oops"])
    a = Args(backfill_date=args.backfill_date)
    assert isinstance(a.backfill_date, datetime)
    assert a.backfill_date.date() == datetime.now().date()

    a.backfill_date.replace(second=0)
