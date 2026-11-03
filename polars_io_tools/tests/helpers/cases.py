"""Building blocks for table-driven tests, and test data that several test modules share.

A table-driven test lists its rows as instances of a small dataclass, one per table, and passes them to
``pytest.mark.parametrize("case", as_params(rows))``, so each row becomes one test with its own name in the report and
the test reads the row's fields by name. Nothing in this module asserts.
"""

from collections.abc import Iterable
from dataclasses import KW_ONLY, dataclass
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest


def as_params(cases: Iterable) -> list:
    """The rows as ``pytest.param`` values, each named by the row's ``id`` and carrying its ``marks``, if it has any."""
    return [pytest.param(case, id=case.id, marks=getattr(case, "marks", ())) for case in cases]


# Dtypes and values
NEW_YORK = "America/New_York"
# Categories not in alphabetical order, so a test notices if the order is lost.
UNSORTED_ENUM = pl.Enum(["c", "a", "b"])
INT64_MIN_MAX = (-(2**63), 2**63 - 1)


def epoch_ns(moment: datetime) -> int:
    """Nanoseconds since 1970 of a datetime without a time zone, computed with integers only."""
    return (moment - datetime(1970, 1, 1)) // timedelta(microseconds=1) * 1000


HOUR_US = 3_600_000_000
# 01:30 in New York on 2024-11-03, the first time (EDT). One hour later the clocks go back and it is 01:30 again (EST).
NEW_YORK_FIRST_0130_US = epoch_ns(datetime(2024, 11, 3, 5, 30)) // 1000
# Microseconds since 1970 around New York's 2024 clock changes: the hour skipped in March, and in November the hour
# that happens twice (once before and once after the clocks go back).
NEW_YORK_2024_CLOCK_CHANGES_US = [1710054000_000000, 1710054000_000000 - 1, 1730611800_000000, 1730615400_000000, 1730613600_000000]
# Strings that often break text handling: a path separator, the empty string, the word None and a non-ASCII letter.
AWKWARD_STRINGS = ["a", "b", "a/b", "", "None", "é"]

# The column dtypes the property tests draw from.
COLUMN_DTYPES = [
    pl.Int8(),
    pl.Int64(),
    pl.UInt64(),
    pl.Float32(),
    pl.Float64(),
    pl.Boolean(),
    pl.Decimal(38, 2),
    pl.String(),
    UNSORTED_ENUM,
    pl.Categorical(),
    pl.Date(),
    pl.Datetime("ms"),
    pl.Datetime("us"),
    pl.Datetime("ns"),
    pl.Datetime("ns", NEW_YORK),
    pl.Datetime("us", "UTC"),
    pl.Duration("us"),
    pl.Duration("ns"),
    pl.Time(),
]
DATE_AND_DATETIME_DTYPES = [dtype for dtype in COLUMN_DTYPES if isinstance(dtype, (pl.Date, pl.Datetime))]


# Frames
def daily_frame(n: int = 14, start: date = date(2024, 1, 1), dtype: pl.DataType | type[pl.DataType] = pl.Date, base: int = 1) -> pl.DataFrame:
    """A frame of ``n`` consecutive days from ``start`` in a ``date`` column cast to ``dtype``, and a ``val`` column
    counting up from ``base``."""
    days = [start + timedelta(days=i) for i in range(n)]
    return pl.DataFrame({"date": days, "val": list(range(base, base + n))}).with_columns(pl.col("date").cast(dtype))


def create_sample_data() -> pl.DataFrame:
    """A 10-row frame with one column of each common dtype, for filter tests.

    ``id`` counts 1 to 10, ``float_val`` goes from 0.5 to 5.0 in equal steps and ``int_val`` from 10 to 100 in tens.
    ``string_val`` holds letters A to F with repeats, and ``cat_val`` is the same letters as a Categorical.
    ``long_string`` is ``text_`` followed by the row number written five times. ``date_val`` holds 10 days from
    2023-01-01, and ``datetime_val`` 10 hours from 2023-01-01 12:00. ``bool_val`` alternates, starting with true.
    ``struct`` holds ``{"name": "num_<i>", "value": <i>}`` for i from 0 to 9, and ``nullable`` is i, null when i is a
    multiple of 3.
    """
    # Generate date range
    dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(10)]
    datetimes = [datetime(2023, 1, 1, 12, 0, 0) + timedelta(hours=i) for i in range(10)]

    # Create the sample dataframe with multiple data types
    df = pl.DataFrame(
        {
            # Numeric types
            "id": list(range(1, 11)),
            "float_val": np.linspace(0.5, 5.0, 10),
            "int_val": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            # String types
            "string_val": ["A", "B", "C", "A", "B", "C", "D", "E", "A", "F"],
            "long_string": ["text_" + str(i) * 5 for i in range(10)],
            # Temporal types
            "date_val": dates,
            "datetime_val": datetimes,
            # Boolean
            "bool_val": [i % 2 == 0 for i in range(10)],
            "struct": [{"name": f"num_{i}", "value": i} for i in range(10)],
            # Null values mixed in
            "nullable": [i if i % 3 != 0 else None for i in range(10)],
        }
    )

    # Add a categorical column
    df = df.with_columns(pl.col("string_val").cast(pl.Categorical).alias("cat_val"))

    return df


# Four groups with one value each.
GROUP_FRAME = pl.DataFrame({"group": ["A", "B", "C", "D"], "val": [1, 2, 3, 4]})


# Common kinds of filter on the ``date`` column of ``daily_frame()``, named by the kind of filter.
def _january_2024(day: int) -> date:
    return date(2024, 1, day)


@dataclass(frozen=True)
class DatePredicateCase:
    """A filter on the ``date`` column of ``daily_frame()``, named by its kind."""

    id: str
    _: KW_ONLY
    predicate: pl.Expr | None


DATE_PREDICATES = [
    DatePredicateCase("ge", predicate=pl.col("date") >= _january_2024(5)),
    DatePredicateCase("le", predicate=pl.col("date") <= _january_2024(5)),
    DatePredicateCase("eq", predicate=pl.col("date") == _january_2024(5)),
    DatePredicateCase("between", predicate=pl.col("date").is_between(_january_2024(3), _january_2024(7))),
    DatePredicateCase("is_in", predicate=pl.col("date").is_in([_january_2024(2), _january_2024(9)])),
    DatePredicateCase("or_disjoint", predicate=(pl.col("date") < _january_2024(3)) | (pl.col("date") > _january_2024(12))),
    DatePredicateCase("ne_in_window", predicate=pl.col("date").is_between(_january_2024(2), _january_2024(8)) & (pl.col("date") != _january_2024(5))),
    DatePredicateCase("contradiction", predicate=(pl.col("date") > _january_2024(9)) & (pl.col("date") < _january_2024(3))),
    DatePredicateCase("with_other_col", predicate=(pl.col("date") >= _january_2024(4)) & (pl.col("val") % 2 == 0)),
    DatePredicateCase("cast_date", predicate=pl.col("date").cast(pl.Date) <= _january_2024(6)),
    DatePredicateCase("none", predicate=None),
]
