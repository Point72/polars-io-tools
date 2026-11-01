"""Range extraction: tables of predicate -> interval for ``convert_expr_to_range`` and ``convert_expr_to_datetime_range``,
tables for the interval helpers, and superset checks that every row a predicate keeps stays inside its range.
"""

import operator
from datetime import UTC, date, datetime, time, timedelta, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

import polars as pl
import portion as P
import pytest
from polars.testing import assert_frame_equal
from portion import Bound, Interval

from polars_io_tools.io_sources.join import filtered_join_asof
from polars_io_tools.io_sources.partitions import _date_windows, by_range, by_time
from polars_io_tools.io_sources.pushdown_combine import FilterSpec, pushdown_combine
from polars_io_tools.io_sources.range_visitor import (
    IntervalVisitor,
    _convert_atomic_interval_to_polars_expr,
    _convert_interval_to_polars_expr,
    _extend_to_full_dates,
    _lookahead_interval,
    _lookback_interval,
    convert_expr_to_datetime_range,
    convert_expr_to_range,
)
from polars_io_tools.testing import PredicateTracker
from polars_io_tools.tests.helpers.catalogue import Case, params
from polars_io_tools.tests.helpers.oracles import assert_keeps_rows, assert_matches_native

ALL = P.open(-P.inf, P.inf)
EMPTY = P.empty()
NY = ZoneInfo("America/New_York")
_NAN = float("nan")
B = pl.Boolean
c = pl.col


def dt(*a) -> datetime:
    return datetime(*a) if len(a) > 3 else datetime.combine(date(*a), time())


def R(id, pred, expect, col="value_col", **kwargs):
    return Case(id, pred, expect, {"col": col, "kwargs": kwargs})


def _unit_point(x):
    return Interval.from_atomic(Bound.CLOSED, x, x + 1, Bound.OPEN)


# --- convert_expr_to_range --------------------------------------------------------------------------------------------

v, m = c("value_col"), c("mixed_col")

RANGE_CASES = [
    R("simple_equality_range", v == 42, P.singleton(42)),
    R("simple_equality_range-point_func", v == 42, P.closedopen(42, 43), create_point_func=_unit_point),
    R("numeric_range", (v >= 10) & (v < 100), P.closedopen(10, 100)),
    R("mixed_int_float_range", (m >= 10) & (m < 20.5), P.closedopen(10, 20.5), col="mixed_col"),
    R("or_int_float_range", (m == 15) | (m == 20.5), P.singleton(15) | P.singleton(20.5), col="mixed_col"),
    R(
        "or_int_float_range-validate_func",
        (m == 15) | (m == 20.5),
        P.singleton(15) | P.singleton(20),
        col="mixed_col",
        validate_value_func=lambda x: int(x) if isinstance(x, float) else x,
        create_point_func=lambda x: P.singleton(int(x)),
    ),
    R("in_operator_range", v.is_in([10, 20, 30]), P.singleton(10) | P.singleton(20) | P.singleton(30)),
    R(
        "in_operator_range-point_func",
        v.is_in([10, 20, 30]),
        P.closedopen(10, 11) | P.closedopen(20, 21) | P.closedopen(30, 31),
        create_point_func=_unit_point,
    ),
    R("complex_nested_expression", (m >= 15) & ((m < 30.5) | (m == 50.5)), P.closedopen(15, 30.5) | P.singleton(50.5), col="mixed_col"),
    R("unbounded_ranges-lower", m >= 15, P.closedopen(15, P.inf), col="mixed_col"),
    R("unbounded_ranges-upper", m < 30.5, P.open(-P.inf, 30.5), col="mixed_col"),
    R("value_validation", v >= 42.7, P.closedopen(43, P.inf), validate_value_func=round),
    R(
        "different_timezone_ranges",
        (c("timestamp") >= datetime(2023, 5, 15, 12, tzinfo=UTC)) & (c("timestamp") < datetime(2023, 5, 15, 8, tzinfo=timezone(timedelta(hours=-4)))),
        EMPTY,
        col="timestamp",
        validate_value_func=lambda t: t if t.tzinfo is None else t.astimezone(UTC).replace(tzinfo=None),
    ),
    R(
        "in_with_mixed_types_combined_with_range",
        m.is_in([date(2023, 1, 15), date(2023, 2, 15), date(2023, 3, 15)]) & (m >= date(2023, 2, 1)),
        P.singleton(date(2023, 2, 15)) | P.singleton(date(2023, 3, 15)),
        col="mixed_col",
    ),
    # ternaries union both branches and ignore the condition
    R(
        "simple_ternary_with_date_ranges-column_condition",
        pl.when(c("is_us_market"))
        .then((c("date") >= date(2023, 1, 1)) & (c("date") < date(2023, 3, 1)))
        .otherwise((c("date") >= date(2023, 6, 1)) & (c("date") < date(2023, 9, 1))),
        P.closedopen(date(2023, 1, 1), date(2023, 3, 1)) | P.closedopen(date(2023, 6, 1), date(2023, 9, 1)),
        col="date",
    ),
    R(
        "simple_ternary_with_date_ranges-date_condition",
        pl.when(c("is_us_market") & c("date") >= date(2023, 2, 1))
        .then((c("date") >= date(2023, 1, 1)) & (c("date") < date(2023, 3, 1)))
        .otherwise((c("date") >= date(2023, 6, 1)) & (c("date") < date(2023, 9, 1))),
        P.closedopen(date(2023, 1, 1), date(2023, 3, 1)) | P.closedopen(date(2023, 6, 1), date(2023, 9, 1)),
        col="date",
    ),
    R(
        "ternary_one_branch_with_date_range",
        pl.when(c("is_weekend")).then((c("date") >= date(2023, 1, 1)) & (c("date") < date(2023, 4, 1))).otherwise(pl.lit(True)),
        ALL,
        col="date",
    ),
    R(
        "ternary_combined_with_other_conditions",
        pl.when(c("region") == "US")
        .then((c("date") >= date(2023, 1, 1)) & (c("date") < date(2023, 4, 1)))
        .otherwise((c("date") >= date(2023, 6, 1)) & (c("date") < date(2023, 9, 1)))
        & (c("date") < date(2023, 12, 31)),
        P.closedopen(date(2023, 1, 1), date(2023, 4, 1)) | P.closedopen(date(2023, 6, 1), date(2023, 9, 1)),
        col="date",
    ),
    # casts: a Boolean cast is transparent, other casts hide the comparison
    R("boolean_cast_processes_input_node", (v >= 10).cast(B), P.closedopen(10, P.inf)),
    R("non_boolean_cast_ignores_input_node", (v >= 10).cast(pl.Utf8), ALL),
    R("target_column_check_in_cast", (c("other_col") >= 10).cast(B), ALL),
    R("nested_boolean_casts_process_all_levels", (v >= 10).cast(B).cast(B), P.closedopen(10, P.inf)),
    R("boolean_cast_with_complex_input", ((v >= 10) & (v < 100)).cast(B), P.closedopen(10, 100)),
    R("boolean_cast_in_conjunction", (v >= 10).cast(B) & (v < 100), P.closedopen(10, 100)),
    R("cast_with_function", v.is_in([10, 20, 30]).cast(B), P.singleton(10) | P.singleton(20) | P.singleton(30)),
    R(
        "cast_with_ternary",
        pl.when(c("condition")).then((v >= 10) & (v < 50)).otherwise((v >= 100) & (v < 150)).cast(B),
        P.closedopen(10, 50) | P.closedopen(100, 150),
    ),
    R("mixed_boolean_casts_in_expr", (v >= 10).cast(B) & (v < 100).cast(pl.Utf8), P.closedopen(10, P.inf)),
    R("boolean_cast_column_directly", v.cast(B), ALL),
    # aliases are transparent
    R("range_alias_simple_equality", c("original_col").alias("renamed_col") == 42, P.singleton(42), col="original_col"),
    R("alias_with_range_constraints", (v.alias("renamed_value") >= 10) & (v.alias("renamed_value") < 100), P.closedopen(10, 100)),
    R(
        "alias_with_in_function",
        c("test_col").alias("test_alias").is_in([10, 20, 30]),
        P.singleton(10) | P.singleton(20) | P.singleton(30),
        col="test_col",
    ),
    R("alias_with_between_function", c("range_col").alias("range_alias").is_between(5, 15), P.closed(5, 15), col="range_col"),
    R(
        "complex_nested_alias_cast_combinations-cast_alias_alias",
        c("original").cast(pl.Float64).alias("first").alias("second") == 42,
        P.singleton(42),
        col="original",
    ),
    R(
        "complex_nested_alias_cast_combinations-alias_cast_alias",
        c("original").alias("first").cast(pl.Float64).alias("second") == 42,
        P.singleton(42),
        col="original",
    ),
    R(
        "complex_nested_alias_cast_combinations-between",
        c("value").cast(pl.Float64).alias("v1").alias("v2").is_between(10, 20),
        P.closed(10, 20),
        col="value",
    ),
    # a NaN literal cannot bound a column
    R("unusable_nan-eq", c("x") == _NAN, ALL, col="x"),
    R("unusable_nan-is_in", c("x").is_in([1.0, _NAN]), ALL, col="x"),
    R("unusable_nan-is_between", c("x").is_between(0, _NAN), P.closed(0, P.inf), col="x"),
    R("unusable_nan-or", c("x").is_between(0, 10) | (c("x") == _NAN), ALL, col="x"),
    # an integer column is compared with a float as Float64, so a float of magnitude 2**53 or more cannot bound it
    R("float_rounds_int-is_between", c("x").is_between(1.0, float(2**53)) & (c("x") <= 5), P.closed(1.0, 5), col="x"),
]


@pytest.mark.parametrize(("pred", "col", "kwargs", "expect"), params(RANGE_CASES, "pred", "col", "kwargs", "expect"))
def test_range(pred, col, kwargs, expect):
    assert convert_expr_to_range(pred, col, **kwargs) == expect


# --- convert_expr_to_datetime_range -----------------------------------------------------------------------------------

d = c("date_col")
_DATES = [date(2023, 1, 15), date(2023, 2, 15), date(2023, 3, 15)]
_NY_END = datetime(2023, 1, 1, 17, tzinfo=NY)
# 2024-01-02 00:00:00 plus 999 ns, the widened upper bound Polars pushes for an inclusive ``<=`` on a ns column.
_NS_BASE = pl.Series([datetime(2024, 1, 2)], dtype=pl.Datetime("ns")).to_physical()[0]
_NS_UPPER = pl.lit(_NS_BASE + 999).cast(pl.Datetime("ns"))
_NS_US_WINDOW = P.closed(dt(2024, 1, 2), datetime(2024, 1, 2, 0, 0, 0, 1))


def _utc(t: datetime) -> datetime:
    return t.astimezone(UTC).replace(tzinfo=None)


def DT(id, pred, expect, col="date_col", **kwargs):
    return R(id, pred, expect, col, **kwargs)


DATETIME_CASES = [
    DT("single_equality-whole_day", d == date(2023, 5, 15), P.closedopen(dt(2023, 5, 15), dt(2023, 5, 16)), coerce_date_to_datetime=False),
    DT("single_equality-midnight", d == date(2023, 5, 15), P.singleton(dt(2023, 5, 15)), coerce_date_to_datetime=True),
    DT("date_range", (d >= date(2023, 1, 1)) & (d < date(2023, 12, 31)), P.closedopen(dt(2023, 1, 1), dt(2023, 12, 31))),
    DT(
        "datetime_range",
        (c("timestamp") > datetime(2023, 5, 15, 9)) & (c("timestamp") <= datetime(2023, 5, 15, 17)),
        P.openclosed(dt(2023, 5, 15, 9), dt(2023, 5, 15, 17)),
        col="timestamp",
    ),
    # Closed: Polars truncates a finer literal to a coarser column's unit.
    *[
        DT(
            f"cast_date_and_datetime-strict={strict}",
            (d >= date(2023, 1, 1)) & (d.cast(pl.Datetime, strict=strict) < datetime(2023, 12, 31)),
            P.closed(dt(2023, 1, 1), dt(2023, 12, 31)),
            coerce_date_to_datetime=True,
        )
        for strict in (True, False)
    ],
    # A date literal does not bound a column cast to a sub-day unit.
    *[
        DT(
            f"cast_datetime_time_unit-coerce={coerce}",
            c("datetime_col").cast(pl.Datetime(time_unit="ns")) == date(2023, 12, 31),
            ALL,
            col="datetime_col",
            coerce_date_to_datetime=coerce,
        )
        for coerce in (True, False)
    ],
    DT(
        "datetime_cast_time",
        c("datetime_col").is_between(dt(2023, 1, 1, 16), dt(2023, 1, 1, 18)) & c("datetime_col").cast(pl.Time()).is_between(time(12, 15), time(19)),
        P.closed(dt(2023, 1, 1, 16), dt(2023, 1, 1, 18)),
        col="datetime_col",
        coerce_date_to_datetime=True,
    ),
    DT(
        "cast_datetime_time_zone",
        c("datetime_col").cast(pl.Datetime("ns", "America/New_York")).is_between(_NY_END - timedelta(hours=1), _NY_END + timedelta(hours=1)),
        P.closed(_utc(_NY_END - timedelta(hours=1)), _utc(_NY_END + timedelta(hours=1)) + timedelta(microseconds=1)),
        col="datetime_col",
        coerce_date_to_datetime=True,
    ),
    DT("or_conditions", (d == date(2023, 1, 15)) | (d == date(2023, 2, 15)), P.closed(dt(2023, 1, 15), dt(2023, 2, 15))),
    DT("unbounded_min", d <= date(2023, 12, 31), P.openclosed(-P.inf, dt(2023, 12, 31))),
    DT("unbounded_max", d >= date(2023, 1, 1), P.closedopen(dt(2023, 1, 1), P.inf)),
    DT("in_operator-whole_days", d.is_in(_DATES), P.closedopen(dt(2023, 1, 15), dt(2023, 3, 16)), coerce_date_to_datetime=False),
    DT("in_operator-midnights", d.is_in(_DATES), P.closed(dt(2023, 1, 15), dt(2023, 3, 15)), coerce_date_to_datetime=True),
    DT(
        "in_with_other_conditions-and",
        d.is_in(_DATES) & (d >= date(2023, 2, 1)),
        P.closedopen(dt(2023, 2, 15), dt(2023, 3, 16)),
        coerce_date_to_datetime=False,
    ),
    DT("in_with_other_conditions-or", d.is_in(_DATES) | (d >= date(2023, 2, 1)), P.closedopen(dt(2023, 1, 15), P.inf), coerce_date_to_datetime=False),
    # != cannot be expressed as a range and is ignored
    DT("not_equal", (d >= date(2023, 5, 10)) & (d <= date(2023, 5, 20)) & (d != date(2023, 5, 15)), P.closed(dt(2023, 5, 10), dt(2023, 5, 20))),
    DT(
        "contradiction_equal_not_equal",
        (d == date(2023, 5, 15)) & (d != date(2023, 5, 15)),
        P.closedopen(dt(2023, 5, 15), dt(2023, 5, 16)),
        coerce_date_to_datetime=False,
    ),
    DT("contradiction_range-is_between", c("time_col").is_between(datetime(2024, 2, 2), datetime(2023, 2, 3), closed="left"), EMPTY, col="time_col"),
    DT("contradiction_range-ge_lt", (c("time_col") >= date(2023, 5, 15)) & (c("time_col") < date(2023, 5, 10)), EMPTY, col="time_col"),
    DT(
        "multiple_equalities_consistent",
        (d == date(2023, 5, 15)) & (d == date(2023, 5, 15)),
        P.closedopen(dt(2023, 5, 15), dt(2023, 5, 16)),
        coerce_date_to_datetime=False,
    ),
    DT("multiple_equalities_contradictory-midnight", (d == date(2023, 5, 15)) & (d == date(2023, 5, 16)), EMPTY),
    DT("multiple_equalities_contradictory-whole_day", (d == date(2023, 5, 15)) & (d == date(2023, 5, 16)), EMPTY, coerce_date_to_datetime=False),
    DT("column_not_in_expression", (c("other_col1") == "value1") | (c("other_col2") == "value2"), ALL),
    DT("column_with_other_columns", (c("other_col") == "some_value") & (d >= date(2023, 5, 15)), P.closedopen(dt(2023, 5, 15), P.inf)),
    DT("is_null", d.is_null(), ALL),
    DT(
        "datetime_comparisons_with_timezone",
        (c("timestamp") >= datetime(2023, 5, 15, 9, tzinfo=UTC)) & (c("timestamp") <= datetime(2023, 5, 15, 17, tzinfo=UTC)),
        P.closed(dt(2023, 5, 15, 9), dt(2023, 5, 15, 17)),
        col="timestamp",
    ),
    # an OR of ranges yields their enclosure by default
    DT(
        "complex_or_conditions",
        ((d >= date(2023, 1, 1)) & (d < date(2023, 6, 1))) | ((d >= date(2023, 5, 1)) & (d < date(2023, 12, 31))),
        P.closedopen(dt(2023, 1, 1), dt(2023, 12, 31)),
    ),
    DT(
        "disjoint_ranges",
        ((d >= date(2023, 1, 1)) & (d < date(2023, 3, 1))) | ((d >= date(2023, 6, 1)) & (d < date(2023, 9, 1))),
        P.closedopen(dt(2023, 1, 1), dt(2023, 9, 1)),
    ),
    DT(
        "datetime_vs_date",
        (m >= date(2023, 5, 15)) & (m < datetime(2023, 5, 15, 12)),
        P.closedopen(dt(2023, 5, 15), dt(2023, 5, 15, 12)),
        col="mixed_col",
    ),
    DT(
        "datetime_with_multiple_expr",
        c("Date").is_between(dt(2023, 5, 15, 12), dt(2023, 5, 16, 12)) & c("symol").is_in(["A", "B"]),
        P.closed(dt(2023, 5, 15, 12), dt(2023, 5, 16, 12)),
        col="Date",
    ),
    DT(
        "datetime_microseconds",
        (c("timestamp") > dt(2023, 5, 15, 9)) & (c("timestamp") <= datetime(2023, 5, 15, 9, 0, 0, 1)),
        P.openclosed(dt(2023, 5, 15, 9), datetime(2023, 5, 15, 9, 0, 0, 1)),
        col="timestamp",
    ),
    DT(
        "adjacent_date_ranges",
        ((d >= date(2023, 1, 1)) & (d < date(2023, 2, 1))) | ((d >= date(2023, 2, 1)) & (d < date(2023, 3, 1))),
        P.closedopen(dt(2023, 1, 1), dt(2023, 3, 1)),
    ),
    DT(
        "contradiction_in_one_conjunction-midnight",
        ((d >= date(2023, 5, 15)) & (d < date(2023, 5, 10))) | (d == date(2023, 6, 1)),
        P.singleton(dt(2023, 6, 1)),
    ),
    DT(
        "contradiction_in_one_conjunction-whole_day",
        ((d >= date(2023, 5, 15)) & (d < date(2023, 5, 10))) | (d == date(2023, 6, 1)),
        P.closedopen(dt(2023, 6, 1), dt(2023, 6, 2)),
        coerce_date_to_datetime=False,
    ),
    DT("in_empty_list", d.is_in([]), EMPTY),
    DT(
        "not_in_combined_with_range",
        (d >= date(2023, 1, 1)) & (d <= date(2023, 12, 31)) & ~d.is_in([date(2023, 7, 4)]),
        P.closed(dt(2023, 1, 1), dt(2023, 12, 31)),
    ),
    DT(
        "datetime_range_with_complex_expr",
        ((c("date") >= date(2023, 1, 1)) & (c("date") < date(2023, 6, 1))) | ((c("date") >= date(2023, 8, 1)) & (c("date") < date(2023, 12, 31))),
        P.closedopen(dt(2023, 1, 1), dt(2023, 12, 31)),
        col="date",
    ),
    DT(
        "multiple_in_expressions",
        d.is_in([date(2023, 1, 15), date(2023, 2, 15)]) | d.is_in([date(2023, 3, 15), date(2023, 4, 15)]),
        P.closedopen(dt(2023, 1, 15), dt(2023, 4, 16)),
        coerce_date_to_datetime=False,
    ),
    DT(
        "nested_expressions",
        (d >= date(2023, 1, 1)) & ((d < date(2023, 3, 1)) | ((d >= date(2023, 6, 1)) & (d < date(2023, 9, 1)))),
        P.closedopen(dt(2023, 1, 1), dt(2023, 9, 1)),
    ),
    # mixed date and datetime literals
    DT("contradictory_mixed_types", (m >= date(2023, 5, 15)) & (m < datetime(2023, 5, 10, 12)), EMPTY, col="mixed_col"),
    DT(
        "is_between_mixed_types-both",
        m.is_between(date(2023, 5, 15), datetime(2023, 5, 16, 12)),
        P.closed(dt(2023, 5, 15), dt(2023, 5, 16, 12)),
        col="mixed_col",
        get_enclosure=False,
    ),
    DT(
        "is_between_mixed_types-left",
        m.is_between(date(2023, 5, 15), datetime(2023, 5, 16, 12), closed="left"),
        P.closedopen(dt(2023, 5, 15), dt(2023, 5, 16, 12)),
        col="mixed_col",
        get_enclosure=False,
    ),
    DT(
        "complex_or_with_mixed_types",
        ((m >= date(2023, 1, 1)) & (m < datetime(2023, 3, 1))) | ((m >= datetime(2023, 5, 1, 12)) & (m < date(2023, 6, 1))),
        P.closedopen(dt(2023, 1, 1), dt(2023, 3, 1)) | P.closedopen(dt(2023, 5, 1, 12), dt(2023, 6, 1)),
        col="mixed_col",
        get_enclosure=False,
    ),
    DT(
        "complex_overlapping_ranges",
        ((m >= date(2023, 1, 1)) & (m < datetime(2023, 3, 1)))
        | ((m >= date(2023, 2, 15)) & (m < datetime(2023, 4, 1)))
        | ((m >= date(2023, 6, 1)) & (m < datetime(2023, 7, 1))),
        P.closedopen(dt(2023, 1, 1), dt(2023, 4, 1)) | P.closedopen(dt(2023, 6, 1), dt(2023, 7, 1)),
        col="mixed_col",
        get_enclosure=False,
    ),
    # casts
    DT("boolean_cast_with_datetime", ((d >= date(2023, 1, 1)) & (d < date(2023, 12, 31))).cast(B), P.closedopen(dt(2023, 1, 1), dt(2023, 12, 31))),
    *[
        DT(
            f"cast_date_to_datetime_with_comparison-strict={strict}",
            (d >= date(2023, 1, 1)) & (d.cast(pl.Datetime, strict=strict) <= datetime(2023, 12, 31, 12)),
            P.closed(dt(2023, 1, 1), datetime(2023, 12, 31, 12, 0, 0, 1)),
        )
        for strict in (True, False)
    ],
    DT("cast_with_time_unit_and_timezone", d.cast(pl.Datetime("us", "UTC")) >= datetime(2023, 1, 1, tzinfo=UTC), P.closedopen(dt(2023, 1, 1), P.inf)),
    DT(
        "boolean_cast_of_datetime_comparison",
        (c("dt_col") == dt(2023, 5, 15, 12)).cast(B),
        P.singleton(dt(2023, 5, 15, 12)),
        col="dt_col",
        coerce_date_to_datetime=True,
    ),
    DT("cast_boolean_to_datetime", c("bool_col").cast(pl.Datetime), ALL, col="bool_col"),
    DT(
        "nested_cast_to_boolean",
        (c("a").is_in([1, 5, 2]) & c("a").is_in([5, 2]).not_())
        .cast(B)
        .and_(((c("date") >= date(2023, 10, 1)) & (c("date") <= date(2023, 10, 3))).cast(B)),
        P.closed(dt(2023, 10, 1), dt(2023, 10, 3)),
        col="date",
        get_enclosure=False,
    ),
    DT(
        "alias_datetime_range_extraction",
        (d.alias("date_alias") >= date(2023, 1, 1)) & (d.alias("date_alias") <= date(2023, 12, 31)),
        P.closed(dt(2023, 1, 1), dt(2023, 12, 31)),
    ),
    # flooring casts: the bound covers everything that floors onto the literal
    DT(
        "microsecond_cast_bound_covers_sub_microseconds-eq",
        c("t").cast(pl.Datetime("us")) == dt(2024, 1, 2, 12, 30),
        P.closed(dt(2024, 1, 2, 12, 30), datetime(2024, 1, 2, 12, 30, 0, 1)),
        col="t",
    ),
    DT(
        "microsecond_cast_bound_covers_sub_microseconds-le",
        c("t").cast(pl.Datetime("us")) <= dt(2024, 1, 2, 12, 30),
        P.openclosed(-P.inf, datetime(2024, 1, 2, 12, 30, 0, 1)),
        col="t",
    ),
    DT(
        "floored_membership_at_maximum_is_unbounded", c("t").cast(pl.Datetime("us")).is_in([datetime.max]), P.closedopen(datetime.max, P.inf), col="t"
    ),
    # literals that cannot bound a column leave it unconstrained
    DT("unusable_datetime-gt_none", c("t") > None, ALL, col="t"),
    DT("unusable_datetime-eq_missing_none", c("t").eq_missing(None), ALL, col="t"),
    DT("unusable_datetime-is_in_with_null", c("t").is_in([datetime(2024, 1, 1), None], nulls_equal=True), ALL, col="t"),
    DT("unusable_datetime-is_in_only_null", c("t").is_in([None], nulls_equal=True), ALL, col="t"),
    DT("single_element_series_is_in", c("t").is_in(pl.Series([datetime(2024, 1, 1)])), P.singleton(dt(2024, 1, 1)), col="t"),
    DT(
        "is_in_with_failed_literal_leaves_column_unconstrained",
        c("t").is_in(pl.Series([1_704_153_600_000_000_000], dtype=pl.Int64).cast(pl.Datetime("us"))),
        ALL,
        col="t",
    ),
    # A nanosecond bound with a sub-microsecond part is rounded away from the kept side to whole microseconds, so the
    # range stays finite (issue #92); an equality or is_in member widens to its enclosing microsecond window.
    DT(
        "is_between_window_is_not_lost",
        c("t").is_between(datetime(2024, 1, 1), _NS_UPPER),
        P.closed(dt(2024, 1, 1), _NS_US_WINDOW.upper),
        col="t",
        get_enclosure=False,
    ),
    DT("equality_widens_to_microsecond_window", c("t") == _NS_UPPER, _NS_US_WINDOW, col="t", get_enclosure=False),
    DT(
        "is_in_keeps_exact_member_as_point_and_widens_sub_microsecond_member",
        c("t").is_in(pl.Series([_NS_BASE - 1_000_000_000, _NS_BASE + 999], dtype=pl.Int64).cast(pl.Datetime("ns"))),
        P.singleton(dt(2024, 1, 1, 23, 59, 59)) | _NS_US_WINDOW,
        col="t",
        get_enclosure=False,
    ),
]


@pytest.mark.parametrize(("pred", "col", "kwargs", "expect"), params(DATETIME_CASES, "pred", "col", "kwargs", "expect"))
def test_datetime_range(pred, col, kwargs, expect):
    assert convert_expr_to_datetime_range(pred, col, **kwargs) == expect


@pytest.mark.parametrize("constraints", [None, P.empty(), P.closedopen(date(2023, 1, 1), date(2023, 1, 2))], ids=["none", "empty", "interval"])
def test_range_visitor_copy(constraints):
    visitor = IntervalVisitor("date_col", None, None)
    if constraints is not None:
        visitor.constraints = constraints
    new_visitor = visitor.copy()
    assert new_visitor is not visitor
    assert new_visitor.target_column == visitor.target_column
    assert new_visitor.constraints == visitor.constraints


# --- interval helpers -------------------------------------------------------------------------------------------------

_TWO_DAYS = P.closed(dt(2023, 1, 1), dt(2023, 1, 3)) | P.closed(dt(2023, 1, 5), dt(2023, 1, 7))


@pytest.mark.parametrize(
    ("shift", "interval", "delta", "expect"),
    [
        pytest.param(
            _lookback_interval,
            P.closed(dt(2023, 1, 5), dt(2023, 1, 10)),
            1,
            P.closed(dt(2023, 1, 4), dt(2023, 1, 10)),
            id="lookback-simple_closed_interval",
        ),
        pytest.param(
            _lookback_interval, P.open(dt(2023, 1, 5), dt(2023, 1, 10)), 2, P.open(dt(2023, 1, 3), dt(2023, 1, 10)), id="lookback-open_interval"
        ),
        pytest.param(
            _lookback_interval, P.openclosed(-P.inf, dt(2023, 1, 10)), 1, P.openclosed(-P.inf, dt(2023, 1, 10)), id="lookback-infinite_lower_bound"
        ),
        pytest.param(
            _lookback_interval,
            _TWO_DAYS,
            1,
            P.closed(dt(2022, 12, 31), dt(2023, 1, 3)) | P.closed(dt(2023, 1, 4), dt(2023, 1, 7)),
            id="lookback-compound_interval",
        ),
        pytest.param(
            _lookback_interval,
            P.closed(date(2023, 1, 5), date(2023, 1, 10)),
            3,
            P.closed(date(2023, 1, 2), date(2023, 1, 10)),
            id="lookback-date_bounds",
        ),
        pytest.param(
            _lookahead_interval,
            P.closed(dt(2023, 1, 5), dt(2023, 1, 10)),
            1,
            P.closed(dt(2023, 1, 5), dt(2023, 1, 11)),
            id="lookahead-simple_closed_interval",
        ),
        pytest.param(
            _lookahead_interval, P.open(dt(2023, 1, 5), dt(2023, 1, 10)), 2, P.open(dt(2023, 1, 5), dt(2023, 1, 12)), id="lookahead-open_interval"
        ),
        pytest.param(
            _lookahead_interval, P.closedopen(dt(2023, 1, 10), P.inf), 1, P.closedopen(dt(2023, 1, 10), P.inf), id="lookahead-infinite_upper_bound"
        ),
        pytest.param(
            _lookahead_interval,
            _TWO_DAYS,
            1,
            P.closed(dt(2023, 1, 1), dt(2023, 1, 4)) | P.closed(dt(2023, 1, 5), dt(2023, 1, 8)),
            id="lookahead-compound_interval",
        ),
        pytest.param(
            _lookahead_interval,
            P.closed(date(2023, 1, 5), date(2023, 1, 10)),
            3,
            P.closed(date(2023, 1, 5), date(2023, 1, 13)),
            id="lookahead-date_bounds",
        ),
    ],
)
def test_shift_interval(shift, interval, delta, expect):
    assert shift(interval, timedelta(days=delta)) == expect


@pytest.mark.parametrize(
    ("interval", "expect"),
    [
        pytest.param(
            P.closed(datetime(2023, 1, 5, 12, 30, 45), datetime(2023, 1, 10, 14, 20, 15)),
            P.closed(date(2023, 1, 5), date(2023, 1, 10)),
            id="datetime_bounds",
        ),
        pytest.param(P.closed(date(2023, 1, 5), date(2023, 1, 10)), P.closed(date(2023, 1, 5), date(2023, 1, 10)), id="date_bounds"),
        pytest.param(ALL, ALL, id="infinite_bounds"),
        pytest.param(
            P.closed(dt(2023, 1, 1, 8, 30), dt(2023, 1, 3, 16, 45)) | P.closed(dt(2023, 1, 5, 9, 15), dt(2023, 1, 7, 17, 30)),
            P.closed(date(2023, 1, 1), date(2023, 1, 3)) | P.closed(date(2023, 1, 5), date(2023, 1, 7)),
            id="compound_interval",
        ),
        # A Date value ``d`` satisfies ``d < upper`` when ``upper`` falls after midnight on ``d``.
        pytest.param(
            P.closedopen(dt(2023, 12, 1), dt(2024, 1, 1, 12)),
            P.closed(date(2023, 12, 1), date(2024, 1, 1)),
            id="open_upper_bound_keeps_date_after_midnight",
        ),
    ],
)
def test_extend_to_full_dates(interval, expect):
    assert _extend_to_full_dates(interval) == expect


_JAN = [dt(2023, 1, n) for n in (4, 5, 7, 10, 11)]
_SPAN = [dt(2022, 12, 31), *(dt(2023, 1, n) for n in range(1, 9))]


@pytest.mark.parametrize(
    ("convert", "interval", "dates", "kept"),
    [
        pytest.param(_convert_atomic_interval_to_polars_expr, P.open(dt(2023, 1, 5), dt(2023, 1, 10)), _JAN, [2], id="atomic-fully_open_interval"),
        pytest.param(
            _convert_atomic_interval_to_polars_expr, P.closed(dt(2023, 1, 5), dt(2023, 1, 10)), _JAN, [1, 2, 3], id="atomic-fully_closed_interval"
        ),
        pytest.param(
            _convert_atomic_interval_to_polars_expr,
            P.openclosed(dt(2023, 1, 5), dt(2023, 1, 10)),
            _JAN,
            [2, 3],
            id="atomic-left_open_right_closed_interval",
        ),
        pytest.param(
            _convert_atomic_interval_to_polars_expr,
            P.closedopen(dt(2023, 1, 5), dt(2023, 1, 10)),
            _JAN,
            [1, 2],
            id="atomic-left_closed_right_open_interval",
        ),
        pytest.param(
            _convert_atomic_interval_to_polars_expr,
            P.openclosed(-P.inf, dt(2023, 1, 10)),
            [dt(2022, 1, 1), dt(2023, 1, 4), dt(2023, 1, 10), dt(2023, 1, 11)],
            [0, 1, 2],
            id="atomic-infinite_lower_bound",
        ),
        pytest.param(
            _convert_atomic_interval_to_polars_expr,
            P.closedopen(dt(2023, 1, 5), P.inf),
            [dt(2023, 1, 4), dt(2023, 1, 5), dt(2023, 1, 10), dt(2024, 1, 1)],
            [1, 2, 3],
            id="atomic-infinite_upper_bound",
        ),
        pytest.param(_convert_atomic_interval_to_polars_expr, ALL, [], None, id="atomic-full_infinite_interval"),
        pytest.param(
            _convert_atomic_interval_to_polars_expr,
            P.singleton(dt(2023, 1, 5)),
            [dt(2023, 1, n) for n in (4, 5, 6)],
            [1],
            id="atomic-singleton_interval",
        ),
        pytest.param(_convert_interval_to_polars_expr, P.closed(dt(2023, 1, 5), dt(2023, 1, 10)), _JAN, [1, 2, 3], id="simple_atomic_interval"),
        pytest.param(_convert_interval_to_polars_expr, _TWO_DAYS, _SPAN, [1, 2, 3, 5, 6, 7], id="compound_interval"),
        pytest.param(_convert_interval_to_polars_expr, EMPTY, [], False, id="empty_interval"),
        pytest.param(_convert_interval_to_polars_expr, P.closed(-P.inf, P.inf), [], None, id="universe_interval"),
        pytest.param(
            _convert_interval_to_polars_expr,
            P.openclosed(dt(2023, 1, 1), dt(2023, 1, 3)) | P.closedopen(dt(2023, 1, 5), dt(2023, 1, 7)),
            _SPAN,
            [2, 3, 5, 6],
            id="mixed_bounds_compound",
        ),
    ],
)
def test_interval_to_polars_expr(convert, interval, dates, kept):
    """The expression keeps the rows at indices ``kept``; ``None`` (no restriction) and ``False`` (nothing) are literal."""
    expr = convert(interval, "date")
    if kept is None or kept is False:
        assert expr is kept
    else:
        assert pl.DataFrame({"date": dates}).filter(expr)["date"].to_list() == [dates[i] for i in kept]


def _days(start: datetime, n: int) -> list[tuple[datetime, datetime]]:
    return [(start + timedelta(days=i), start + timedelta(days=i + 1)) for i in range(n)]


@pytest.mark.parametrize(
    ("interval", "every", "expect"),
    [
        # the window starting at an exclusive upper bound is not read; a closed="left" regression
        pytest.param(P.closedopen(dt(2024, 1, 2), dt(2024, 1, 10)), "1d", _days(dt(2024, 1, 2), 8), id="exclusive_upper_drops_boundary_window"),
        pytest.param(P.closed(dt(2024, 1, 2), dt(2024, 1, 10)), "1d", _days(dt(2024, 1, 2), 9), id="inclusive_upper_keeps_boundary_window"),
        pytest.param(
            P.closed(dt(2024, 1, 15), dt(2024, 3, 10)),
            "1mo",
            [(dt(2024, 1, 1), dt(2024, 2, 1)), (dt(2024, 2, 1), dt(2024, 3, 1)), (dt(2024, 3, 1), dt(2024, 4, 1))],
            id="monthly_floors_to_calendar_boundary",
        ),
        # January and December only: two monthly windows, not twelve across the gap between them.
        pytest.param(
            P.closedopen(dt(2024, 1, 1), dt(2024, 1, 15)) | P.closedopen(dt(2024, 12, 1), dt(2024, 12, 15)),
            "1mo",
            [(dt(2024, 1, 1), dt(2024, 2, 1)), (dt(2024, 12, 1), dt(2025, 1, 1))],
            id="disjoint_interval_skips_the_gaps",
        ),
    ],
)
def test_date_windows(interval, every, expect):
    assert _date_windows(interval, every) == expect


# --- every row a predicate keeps stays inside its range ---------------------------------------------------------------


def _assert_range_keeps_rows(df: pl.DataFrame, pred: pl.Expr, col: str, interval: P.Interval) -> None:
    expr = _convert_interval_to_polars_expr(interval, col)
    assert_keeps_rows(df, pred, pl.lit(expr) if isinstance(expr, bool) else expr)


@pytest.mark.parametrize(
    "pred",
    [
        pytest.param(c("t").dt.date() == date(2024, 1, 1), id="eq"),
        pytest.param(c("t").dt.date() <= date(2024, 1, 1), id="le"),
        pytest.param(c("t").cast(pl.Date).is_in([date(2024, 1, 1)]), id="cast_is_in"),
        pytest.param(c("t").dt.date().is_between(date(2023, 12, 1), date(2024, 1, 1)), id="is_between"),
    ],
)
def test_day_floored_column_bound_covers_whole_day(pred):
    df = pl.DataFrame({"t": [datetime(2023, 12, 31, 23), datetime(2024, 1, 1, 13), datetime(2024, 1, 2, 1)]})
    interval = convert_expr_to_datetime_range(pred, "t")
    assert all(t in interval for t in df.filter(pred)["t"])


_NS_ROWS = pl.DataFrame({"t": pl.Series([_NS_BASE + o for o in [-2000, -1, 0, 1, 500, 999, 1000, 2000]], dtype=pl.Int64).cast(pl.Datetime("ns"))})


@pytest.mark.parametrize(
    "pred",
    [
        pytest.param(c("t").is_between(datetime(2024, 1, 1), _NS_UPPER), id="is_between"),
        pytest.param(c("t").is_between(datetime(2024, 1, 1), _NS_UPPER, closed="right"), id="is_between_right"),
        pytest.param(c("t") <= _NS_UPPER, id="le"),
        pytest.param(c("t") < _NS_UPPER, id="lt"),
        pytest.param(c("t") >= _NS_UPPER, id="ge"),
        pytest.param(c("t") > _NS_UPPER, id="gt"),
        pytest.param(c("t") == _NS_UPPER, id="eq"),
    ],
)
def test_range_keeps_every_matching_row(pred):
    interval = convert_expr_to_datetime_range(pred, "t", get_enclosure=False)
    assert not interval.empty and interval != ALL
    _assert_range_keeps_rows(_NS_ROWS, pred, "t", interval)


@pytest.mark.parametrize(
    "pred",
    [
        pytest.param(~(c("t") < _NS_UPPER), id="not_lt"),
        pytest.param(~(c("t") <= _NS_UPPER), id="not_le"),
        pytest.param(~(c("t") > _NS_UPPER), id="not_gt"),
        pytest.param(~(c("t") >= _NS_UPPER), id="not_ge"),
        pytest.param(c("t") != _NS_UPPER, id="ne"),
        pytest.param(~c("t").is_between(datetime(2024, 1, 1), _NS_UPPER), id="not_is_between"),
        pytest.param(~(~(c("t") < _NS_UPPER)), id="not_not_lt"),
    ],
)
def test_negation_does_not_narrow_below_matching_rows(pred):
    """The visitor never complements an interval: a negated subexpression is unconstrained, which keeps every row."""
    _assert_range_keeps_rows(_NS_ROWS, pred, "t", convert_expr_to_datetime_range(pred, "t", get_enclosure=False))


@pytest.mark.parametrize("op", [operator.ge, operator.gt, operator.eq, operator.le, operator.lt], ids=lambda op: op.__name__)
def test_sub_microsecond_negative_duration_bound_keeps_matching_rows(op):
    """Polars converts a negative ``Duration("ns")`` toward zero, so such a bound stays unconstrained rather than rounded."""
    df = pl.DataFrame({"d": pl.Series([-1200, -1001, -1, 0, 1, 1200], dtype=pl.Int64).cast(pl.Duration("ns"))})
    pred = op(c("d"), pl.lit(-1200).cast(pl.Duration("ns")))
    _assert_range_keeps_rows(df, pred, "d", convert_expr_to_range(pred, "d"))


# 2024-11-03 06:30:00 UTC plus 999 ns: 01:30 in the repeated (fall-back) hour in New York, the second pass (fold=1).
_NY_NS = pl.Datetime("ns", "America/New_York")
_NY_REPEATED = pl.lit(pl.Series([datetime(2024, 11, 3, 6, 30)], dtype=pl.Datetime("ns")).to_physical()[0] + 999).cast(_NY_NS)
_NY_START = pl.lit(datetime(2024, 11, 3, 4, 30, tzinfo=UTC)).cast(_NY_NS)


@pytest.mark.parametrize(
    "pred",
    [
        pytest.param(c("t") == _NY_REPEATED, id="eq"),
        pytest.param(c("t").is_in(pl.concat_list(_NY_REPEATED)), id="is_in"),
        pytest.param(c("t") < _NY_REPEATED, id="lt"),
        pytest.param(c("t") <= _NY_REPEATED, id="le"),
        *[pytest.param(c("t").is_between(_NY_START, _NY_REPEATED, closed=cl), id=f"is_between_{cl}") for cl in ("both", "left", "right", "none")],
    ],
)
def test_rounding_up_in_repeated_dst_hour_keeps_matching_rows(pred):
    """Adding a microsecond to an aware datetime is wall-clock arithmetic that resets ``fold``; it must not move the bound back an hour."""
    times = [datetime(2024, 11, 3, h, mi, tzinfo=UTC) for h, mi in [(4, 30), (5, 0), (5, 30), (6, 0), (6, 29), (6, 30)]]
    df = pl.DataFrame({"t": pl.Series(times).cast(_NY_NS).dt.offset_by("999ns")}).vstack(pl.DataFrame({"t": pl.Series(times).cast(_NY_NS)}))
    interval = convert_expr_to_datetime_range(pred, "t", get_enclosure=False)
    matching = df.filter(pred)["t"].to_physical().to_list()
    assert matching
    # Compare in UTC nanoseconds: the range holds naive UTC bounds.
    to_ns = lambda b: (b - datetime(1970, 1, 1)) // timedelta(microseconds=1) * 1000
    interval_ns = interval.apply(lambda a: a.replace(lower=to_ns, upper=to_ns))
    assert all(t in interval_ns for t in matching)


_ENUM = pl.Enum(["b", "a"])


@pytest.mark.parametrize(
    "pred",
    [
        pytest.param(c("x") > pl.lit("b", dtype=_ENUM), id="gt_enum_literal"),
        pytest.param(c("x") > "b", id="gt_str"),
        pytest.param(c("x") < "a", id="lt_str"),
        pytest.param(c("x").is_between(pl.lit("b", dtype=_ENUM), pl.lit("a", dtype=_ENUM)), id="is_between"),
    ],
)
def test_enum_ordering_comparison_keeps_rows(pred):
    """Enum values order by category, so ``"a" > "b"`` for ``Enum(["b", "a"])``."""
    df = pl.DataFrame({"x": pl.Series(["b", "a"], dtype=_ENUM)})
    interval = convert_expr_to_range(pred, "x")
    assert all(x in interval for x in df.filter(pred)["x"])


# --- entry points built on ranges keep what native Polars keeps -------------------------------------------------------


@pytest.mark.parametrize(
    ("dtype", "t", "pred", "kwargs"),
    [
        # A ``Datetime("ms")`` column cast to ``us`` is exact; its bound must survive truncation to milliseconds.
        pytest.param(
            pl.Datetime("ms"),
            datetime(2024, 1, 1, 12),
            c("t").cast(pl.Datetime("us")) == datetime(2024, 1, 1, 12),
            {"tolerance": timedelta(seconds=1)},
            id="ms_cast_to_us",
        ),
        pytest.param(
            pl.Datetime("us"),
            datetime(9999, 12, 31),
            c("t") == datetime(9999, 12, 31),
            {"strategy": "forward", "tolerance": timedelta(days=1)},
            id="forward_at_max",
        ),
        pytest.param(
            pl.Datetime("us"),
            datetime(1, 1, 1),
            c("t") == datetime(1, 1, 1),
            {"strategy": "backward", "tolerance": timedelta(days=1)},
            id="backward_at_min",
        ),
    ],
)
def test_filtered_join_asof_keeps_matches(dtype, t, pred, kwargs):
    left = pl.DataFrame({"t": pl.Series([t], dtype=dtype)}).lazy().set_sorted("t")
    right = left.with_columns(v=pl.lit(7)).set_sorted("t")
    assert_matches_native(
        filtered_join_asof(left, right, on="t", log_explain=False, **kwargs),
        left.join_asof(right, on="t", **kwargs),
        pred & (c("v") == 7),
        check_row_order=True,
    )


@pytest.mark.parametrize(
    ("value", "pred", "spec"),
    [
        pytest.param(date.max, c("d") <= date.max, FilterSpec(lookahead=timedelta(days=1)), id="le_date_max_lookahead"),
        pytest.param(date.max, c("d") == date.max, FilterSpec(lookahead=timedelta(days=1)), id="eq_date_max_lookahead"),
        pytest.param(date.min, c("d") >= date.min, FilterSpec(lookback=timedelta(days=1)), id="ge_date_min_lookback"),
        pytest.param(
            datetime(9999, 12, 31), c("d") == datetime(9999, 12, 31), FilterSpec(lookahead=timedelta(days=1)), id="eq_datetime_max_lookahead"
        ),
    ],
)
def test_pushdown_combine_expansion_past_extremes(value, pred, spec):
    df = pl.DataFrame({"d": [value], "v": [1]})
    lf = pushdown_combine(sources={"data": (df.lazy(), {"d": spec})}, combine=lambda s: s["data"])
    assert_matches_native(lf, df.lazy(), pred, check_row_order=True)


def _partitioned_rows(df: pl.DataFrame, partitioner, pred: pl.Expr) -> pl.DataFrame:
    """Rows of ``df`` matching ``pred`` when read through the partitions built from the pushed-down predicate."""
    tracker = PredicateTracker(df)
    tracker.lazy_frame.filter(pred).collect()
    parts = partitioner.build(tracker.last_predicate)
    if parts is None:  # unpartitioned read
        return df.filter(pred)
    return pl.concat([df.clear(), *(df.filter(p.predicate) for p in parts)]).filter(pred)


@pytest.mark.parametrize(
    ("values", "pred"),
    [
        pytest.param([1, 5, 257, -3], c("x").cast(pl.Int8, wrap_numerical=True).is_between(0, 10), id="wrapping_int8_cast"),
        pytest.param([1, 5, 257, -3], c("x").cast(pl.UInt8, wrap_numerical=True).is_between(0, 1), id="wrapping_uint8_cast"),
        pytest.param([-0.5, 1.0, 3.5, 7.0], c("x").cast(pl.Int64).is_between(0, 4), id="truncating_int_cast"),
        pytest.param([1, 2, 7], c("x").cast(pl.String).is_between(pl.lit("1"), pl.lit("3")), id="string_cast"),
        pytest.param([1, None, 7], c("x").is_in([1, None], nulls_equal=True), id="null_membership"),
        pytest.param([1.0, _NAN, 3.0], c("x") == _NAN, id="nan_eq"),
        pytest.param([1.0, _NAN, 3.0], c("x").is_in([1.0, _NAN]), id="nan_is_in"),
        pytest.param([1.0, _NAN, 3.0], c("x").is_between(0, _NAN), id="nan_is_between"),
        pytest.param([1.0, _NAN, 3.0], c("x").is_between(0, 10) | (c("x") == _NAN), id="nan_or"),
        pytest.param([1, 2, 7], c("x").is_in([1, 2]), id="is_in"),
    ],
)
def test_partitions_by_range_keep_matching_rows(values, pred):
    df = pl.DataFrame({"x": values})
    assert_frame_equal(_partitioned_rows(df, by_range("x", 5), pred), df.filter(pred), check_row_order=False)


def test_partitions_by_range_decimal_bounds_with_float_width():
    df = pl.DataFrame({"x": pl.Series([Decimal("1.0"), Decimal("1.5"), Decimal("3.0")], dtype=pl.Decimal(10, 1))})
    pred = c("x").is_between(Decimal("1.0"), Decimal("2.0"))
    parts = by_range("x", 0.5).build(pred)
    assert_frame_equal(pl.concat([df.filter(p.predicate) for p in parts]).filter(pred), df.filter(pred), check_row_order=False)


@pytest.mark.parametrize(
    ("value", "every"),
    [(datetime(9999, 12, 31), "1d"), (date(9999, 12, 31), "1d"), (date(9999, 12, 31), "1mo")],
    ids=["datetime_daily", "date_daily", "date_monthly"],
)
def test_partitions_by_time_window_past_maximum(value, every):
    df = pl.DataFrame({"t": [value]})
    pred = c("t") == value
    assert_frame_equal(_partitioned_rows(df, by_time("t", every), pred), df.filter(pred))


def test_partitions_by_time_null_membership():
    df = pl.DataFrame({"t": [datetime(2024, 1, 1), None, datetime(2024, 1, 2)]})
    pred = c("t").is_in([datetime(2024, 1, 1), None], nulls_equal=True)
    assert_frame_equal(_partitioned_rows(df, by_time("t", "1d"), pred), df.filter(pred), check_row_order=False)
