"""Tests for partitions.py: the read partitions ``by_value``, ``by_range``, ``by_time`` and their helpers. They check
which partitions a filter gives, and that together the partitions hold every row the filter keeps."""

import functools
import operator
from dataclasses import KW_ONLY, dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal

import polars as pl
import portion
import pytest
from hypothesis import assume, given, strategies as st
from polars.testing import assert_frame_equal

from polars_io_tools.io_sources.partitions import _date_windows, by_range, by_time, by_value
from polars_io_tools.testing import PredicateTracker
from polars_io_tools.tests.helpers.cases import as_params
from polars_io_tools.tests.helpers.polars_versions import xfail_if_polars_has
from polars_io_tools.tests.helpers.property_checks import (
    SETTINGS,
    filter_a_source_receives,
    mask,
    no_panic,
    note_case,
    rows_plain_polars_keeps,
    show,
)
from polars_io_tools.tests.helpers.strategies import (
    frames_and_predicates,
)

_NAN = float("nan")


@dataclass(frozen=True)
class IntervalStrCase:
    """An interval, and the interval string ``_interval_str`` makes of it."""

    id: str
    _: KW_ONLY
    value: int | str
    expected: str


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            IntervalStrCase("1mo-1mo", value="1mo", expected="1mo"),
            IntervalStrCase("2w-2w", value="2w", expected="2w"),
            IntervalStrCase("5d-5d", value="5d", expected="5d"),
            IntervalStrCase("1q-1q", value="1q", expected="1q"),
            IntervalStrCase("3y-3y", value="3y", expected="3y"),
            IntervalStrCase("mo-1mo", value="mo", expected="1mo"),
            IntervalStrCase("month-1mo", value="month", expected="1mo"),
            IntervalStrCase("quarter-1q", value="quarter", expected="1q"),
            IntervalStrCase("7-7d", value=7, expected="7d"),
        ]
    ),
)
def test_interval_str_valid(case: IntervalStrCase):
    from polars_io_tools.io_sources.partitions import _interval_str

    assert _interval_str(case.value) == case.expected


@pytest.mark.parametrize("value", ["", "1", "3h", "-2d", 0, -5, "1mth", True])
def test_interval_str_invalid(value):
    from polars_io_tools.io_sources.partitions import _interval_str

    with pytest.raises((ValueError, TypeError)):
        _interval_str(value)


@pytest.mark.xfail(strict=True, raises=RecursionError, reason="the range extraction recurses once per term of a long chain")
@pytest.mark.parametrize("partitioner", [by_range("x", 1), by_time("t")], ids=["by_range", "by_time"])
def test_long_chain_on_the_partition_column(partitioner):
    col = pl.col(partitioner.on)
    pred = functools.reduce(operator.or_, [col == (i if partitioner.on == "x" else date(2024, 1, 1) + timedelta(days=i)) for i in range(1_000)])
    partitioner.build(pred)


@xfail_if_polars_has("ns_literal_min_overflows")
def test_by_time_on_a_nanosecond_value_near_its_minimum():
    t = pl.Series([-(2**63)]).cast(pl.Datetime("ns"))
    by_time("t", "1d").build(pl.col("t").is_in(t.implode()))


@SETTINGS
@given(case=frames_and_predicates(), every=st.sampled_from(["1d", "1mo", "1y"]), width=st.sampled_from([1, 7.5, 10**6]))
def test_partitions_cover_the_rows_a_predicate_keeps(case, every, width):
    frame, pred = case
    note_case(frame, pred)
    keep = rows_plain_polars_keeps(frame, pred)
    assume(keep is not None and keep.any())
    pushed = filter_a_source_receives(frame, pred)
    assume(pushed is not None)
    for col, dtype in frame.schema.items():
        partitioners = [by_value(col)]
        if dtype == pl.Date or (isinstance(dtype, pl.Datetime) and dtype.time_zone is None):
            partitioners.append(by_time(col, every))
        if dtype.is_numeric():
            partitioners.append(by_range(col, width))
        for partitioner in partitioners:
            partitions = no_panic(partitioner.build)(pushed)
            if partitions is None:
                continue
            covered = functools.reduce(operator.or_, [mask(frame, p.predicate) for p in partitions], pl.Series([False] * frame.height))
            assert covered.filter(keep).all(), (partitioner, [show(p.predicate) for p in partitions])


def _daily_windows(start: datetime, n: int) -> list[tuple[datetime, datetime]]:
    return [(start + timedelta(days=i), start + timedelta(days=i + 1)) for i in range(n)]


@dataclass(frozen=True)
class DateWindowsCase:
    """An interval and a window length, and the windows ``_date_windows`` splits the interval into."""

    id: str
    _: KW_ONLY
    interval: portion.Interval
    every: str
    expect: list


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            # The window that starts at an excluded upper bound is not read.
            DateWindowsCase(
                "exclusive_upper_drops_boundary_window",
                interval=portion.closedopen(datetime(2024, 1, 2), datetime(2024, 1, 10)),
                every="1d",
                expect=_daily_windows(datetime(2024, 1, 2), 8),
            ),
            DateWindowsCase(
                "inclusive_upper_keeps_boundary_window",
                interval=portion.closed(datetime(2024, 1, 2), datetime(2024, 1, 10)),
                every="1d",
                expect=_daily_windows(datetime(2024, 1, 2), 9),
            ),
            DateWindowsCase(
                "monthly_floors_to_calendar_boundary",
                interval=portion.closed(datetime(2024, 1, 15), datetime(2024, 3, 10)),
                every="1mo",
                expect=[
                    (datetime(2024, 1, 1), datetime(2024, 2, 1)),
                    (datetime(2024, 2, 1), datetime(2024, 3, 1)),
                    (datetime(2024, 3, 1), datetime(2024, 4, 1)),
                ],
            ),
            # January and December only: two monthly windows, not twelve across the gap between them.
            DateWindowsCase(
                "disjoint_interval_skips_the_gaps",
                interval=portion.closedopen(datetime(2024, 1, 1), datetime(2024, 1, 15))
                | portion.closedopen(datetime(2024, 12, 1), datetime(2024, 12, 15)),
                every="1mo",
                expect=[(datetime(2024, 1, 1), datetime(2024, 2, 1)), (datetime(2024, 12, 1), datetime(2025, 1, 1))],
            ),
        ]
    ),
)
def test_date_windows(case: DateWindowsCase):
    assert _date_windows(case.interval, case.every) == case.expect


def _partitioned_rows(df: pl.DataFrame, partitioner, pred: pl.Expr) -> pl.DataFrame:
    """The rows of ``df`` matching ``pred``, read through the partitions built from the filter Polars hands to a source."""
    tracker = PredicateTracker(df)
    tracker.lazy_frame.filter(pred).collect()
    parts = partitioner.build(tracker.last_predicate)
    if parts is None:  # unpartitioned read
        return df.filter(pred)
    return pl.concat([df.clear(), *(df.filter(p.predicate) for p in parts)]).filter(pred)


@dataclass(frozen=True)
class RangePartitionsCase:
    """Values and a filter; a read by range partitions must keep the rows the filter keeps."""

    id: str
    _: KW_ONLY
    values: list
    pred: pl.Expr


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            RangePartitionsCase("wrapping_int8_cast", values=[1, 5, 257, -3], pred=pl.col("x").cast(pl.Int8, wrap_numerical=True).is_between(0, 10)),
            RangePartitionsCase("wrapping_uint8_cast", values=[1, 5, 257, -3], pred=pl.col("x").cast(pl.UInt8, wrap_numerical=True).is_between(0, 1)),
            RangePartitionsCase("truncating_int_cast", values=[-0.5, 1.0, 3.5, 7.0], pred=pl.col("x").cast(pl.Int64).is_between(0, 4)),
            RangePartitionsCase("string_cast", values=[1, 2, 7], pred=pl.col("x").cast(pl.String).is_between(pl.lit("1"), pl.lit("3"))),
            RangePartitionsCase("null_membership", values=[1, None, 7], pred=pl.col("x").is_in([1, None], nulls_equal=True)),
            RangePartitionsCase("nan_eq", values=[1.0, _NAN, 3.0], pred=pl.col("x") == _NAN),
            RangePartitionsCase("nan_is_in", values=[1.0, _NAN, 3.0], pred=pl.col("x").is_in([1.0, _NAN])),
            RangePartitionsCase("nan_is_between", values=[1.0, _NAN, 3.0], pred=pl.col("x").is_between(0, _NAN)),
            RangePartitionsCase("nan_or", values=[1.0, _NAN, 3.0], pred=pl.col("x").is_between(0, 10) | (pl.col("x") == _NAN)),
            RangePartitionsCase("is_in", values=[1, 2, 7], pred=pl.col("x").is_in([1, 2])),
        ]
    ),
)
def test_partitions_by_range_keep_matching_rows(case: RangePartitionsCase):
    df = pl.DataFrame({"x": case.values})
    assert_frame_equal(_partitioned_rows(df, by_range("x", 5), case.pred), df.filter(case.pred), check_row_order=False)


def test_partitions_by_range_decimal_bounds_with_float_width():
    df = pl.DataFrame({"x": pl.Series([Decimal("1.0"), Decimal("1.5"), Decimal("3.0")], dtype=pl.Decimal(10, 1))})
    pred = pl.col("x").is_between(Decimal("1.0"), Decimal("2.0"))
    parts = by_range("x", 0.5).build(pred)
    assert_frame_equal(pl.concat([df.filter(p.predicate) for p in parts]).filter(pred), df.filter(pred), check_row_order=False)


@dataclass(frozen=True)
class LastTimeWindowCase:
    """A value near the largest date or datetime, and a window length; a read by time windows must keep it."""

    id: str
    _: KW_ONLY
    value: date | datetime
    every: str


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            LastTimeWindowCase("datetime_daily", value=datetime(9999, 12, 31), every="1d"),
            LastTimeWindowCase("date_daily", value=date(9999, 12, 31), every="1d"),
            LastTimeWindowCase("date_monthly", value=date(9999, 12, 31), every="1mo"),
        ]
    ),
)
def test_partitions_by_time_window_past_maximum(case: LastTimeWindowCase):
    df = pl.DataFrame({"t": [case.value]})
    pred = pl.col("t") == case.value
    assert_frame_equal(_partitioned_rows(df, by_time("t", case.every), pred), df.filter(pred))


def test_partitions_by_time_null_membership():
    df = pl.DataFrame({"t": [datetime(2024, 1, 1), None, datetime(2024, 1, 2)]})
    pred = pl.col("t").is_in([datetime(2024, 1, 1), None], nulls_equal=True)
    assert_frame_equal(_partitioned_rows(df, by_time("t", "1d"), pred), df.filter(pred), check_row_order=False)
