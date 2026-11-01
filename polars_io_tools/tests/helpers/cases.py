"""Test data shared across modules: table rows, dtypes and values, standard frames, and predicate shapes.

Nothing here asserts. A table-driven test feeds ``Case`` rows to ``pytest.mark.parametrize`` through ``params``;
what ``expect`` means is up to the test or assertion that consumes the row.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import polars as pl
import pytest


@dataclass(frozen=True)
class Case:
    """One row of a table-driven test, identified by ``id`` in the test report."""

    id: str
    pred: Any
    expect: Any = None
    kw: dict = field(default_factory=dict)
    marks: tuple = ()


def params(cases: list[Case], *fields: str) -> list:
    """``pytest.param`` per row with the row's id and marks; a field is a ``Case`` attribute or a key of ``kw``."""
    return [pytest.param(*(c.kw[f] if f in c.kw else getattr(c, f) for f in fields), id=c.id, marks=c.marks) for c in cases]


# Dtypes and values
NY = "America/New_York"
ENUM = pl.Enum(["c", "a", "b"])
INT64 = (-(2**63), 2**63 - 1)
# Microsecond ticks around the 2024 DST transitions in New York: the spring gap and both readings of the fall fold.
DST_US = [1710054000_000000, 1710054000_000000 - 1, 1730611800_000000, 1730615400_000000, 1730613600_000000]
TEXT = ["a", "b", "a/b", "", "None", "é"]

COLUMN_DTYPES = [
    pl.Int8(),
    pl.Int64(),
    pl.UInt64(),
    pl.Float32(),
    pl.Float64(),
    pl.Boolean(),
    pl.Decimal(38, 2),
    pl.String(),
    ENUM,
    pl.Categorical(),
    pl.Date(),
    pl.Datetime("ms"),
    pl.Datetime("us"),
    pl.Datetime("ns"),
    pl.Datetime("ns", NY),
    pl.Datetime("us", "UTC"),
    pl.Duration("us"),
    pl.Duration("ns"),
    pl.Time(),
]
TEMPORAL_DTYPES = [d for d in COLUMN_DTYPES if isinstance(d, (pl.Date, pl.Datetime))]


# Standard frames
def daily_frame(n: int = 14, start: date = date(2024, 1, 1), dtype: pl.DataType | type[pl.DataType] = pl.Date, base: int = 1) -> pl.DataFrame:
    """``n`` consecutive days in a ``date`` column of ``dtype``, with ``val`` counting up from ``base``."""
    days = [start + timedelta(days=i) for i in range(n)]
    return pl.DataFrame({"date": days, "val": list(range(base, base + n))}).with_columns(pl.col("date").cast(dtype))


GROUP_FRAME = pl.DataFrame({"group": ["A", "B", "C", "D"], "val": [1, 2, 3, 4]})


# Predicate shapes on the ``date`` column of ``daily_frame``
def _d(day: int) -> date:
    return date(2024, 1, day)


DATE_PREDICATES = [
    Case("ge", pl.col("date") >= _d(5)),
    Case("le", pl.col("date") <= _d(5)),
    Case("eq", pl.col("date") == _d(5)),
    Case("between", pl.col("date").is_between(_d(3), _d(7))),
    Case("is_in", pl.col("date").is_in([_d(2), _d(9)])),
    Case("or_disjoint", (pl.col("date") < _d(3)) | (pl.col("date") > _d(12))),
    Case("ne_in_window", pl.col("date").is_between(_d(2), _d(8)) & (pl.col("date") != _d(5))),
    Case("contradiction", (pl.col("date") > _d(9)) & (pl.col("date") < _d(3))),
    Case("with_other_col", (pl.col("date") >= _d(4)) & (pl.col("val") % 2 == 0)),
    Case("dt_date", pl.col("date").cast(pl.Date) <= _d(6)),
    Case("none", None),
]
