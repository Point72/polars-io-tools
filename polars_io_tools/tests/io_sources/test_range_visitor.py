"""convert_expr_to_range and convert_expr_to_datetime_range: the range of values a filter allows in one column.

Ranges are ``portion`` intervals. A range may be wider than the filter needs but must hold every value of a row the
filter keeps; several tests check that on real rows. Also tests the helpers that shift, widen and split ranges.
"""

import operator
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from datetime import UTC, date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import polars as pl
import portion
import pytest
from portion import Bound, Interval

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
from polars_io_tools.tests.helpers.assertions import assert_pushed_filter_keeps_rows
from polars_io_tools.tests.helpers.cases import as_params

ANY_VALUE = portion.open(-portion.inf, portion.inf)
EMPTY = portion.empty()
NEW_YORK = ZoneInfo("America/New_York")
_NAN = float("nan")


@dataclass(frozen=True)
class RangeCase:
    """A filter, and the range of values it allows in one column."""

    id: str
    _: KW_ONLY
    predicate: pl.Expr
    expected_range: Interval
    column: str = "value_col"
    # passed on to ``convert_expr_to_range``
    create_point_func: Callable | None = None
    validate_value_func: Callable | None = None


def _from_value_to_next_integer(x):
    return Interval.from_atomic(Bound.CLOSED, x, x + 1, Bound.OPEN)


# --- convert_expr_to_range --------------------------------------------------------------------------------------------

# Filters on numbers (and some dates), and the range each one allows.
RANGE_CASES = [
    RangeCase("simple_equality_range", predicate=pl.col("value_col") == 42, expected_range=portion.singleton(42)),
    RangeCase(
        "simple_equality_range-point_func",
        predicate=pl.col("value_col") == 42,
        expected_range=portion.closedopen(42, 43),
        create_point_func=_from_value_to_next_integer,
    ),
    RangeCase("numeric_range", predicate=(pl.col("value_col") >= 10) & (pl.col("value_col") < 100), expected_range=portion.closedopen(10, 100)),
    RangeCase(
        "mixed_int_float_range",
        predicate=(pl.col("mixed_col") >= 10) & (pl.col("mixed_col") < 20.5),
        expected_range=portion.closedopen(10, 20.5),
        column="mixed_col",
    ),
    RangeCase(
        "or_int_float_range",
        predicate=(pl.col("mixed_col") == 15) | (pl.col("mixed_col") == 20.5),
        expected_range=portion.singleton(15) | portion.singleton(20.5),
        column="mixed_col",
    ),
    RangeCase(
        "or_int_float_range-validate_func",
        predicate=(pl.col("mixed_col") == 15) | (pl.col("mixed_col") == 20.5),
        expected_range=portion.singleton(15) | portion.singleton(20),
        column="mixed_col",
        validate_value_func=lambda x: int(x) if isinstance(x, float) else x,
        create_point_func=lambda x: portion.singleton(int(x)),
    ),
    RangeCase(
        "in_operator_range",
        predicate=pl.col("value_col").is_in([10, 20, 30]),
        expected_range=portion.singleton(10) | portion.singleton(20) | portion.singleton(30),
    ),
    RangeCase(
        "in_operator_range-point_func",
        predicate=pl.col("value_col").is_in([10, 20, 30]),
        expected_range=portion.closedopen(10, 11) | portion.closedopen(20, 21) | portion.closedopen(30, 31),
        create_point_func=_from_value_to_next_integer,
    ),
    RangeCase(
        "complex_nested_expression",
        predicate=(pl.col("mixed_col") >= 15) & ((pl.col("mixed_col") < 30.5) | (pl.col("mixed_col") == 50.5)),
        expected_range=portion.closedopen(15, 30.5) | portion.singleton(50.5),
        column="mixed_col",
    ),
    RangeCase("unbounded_ranges-lower", predicate=pl.col("mixed_col") >= 15, expected_range=portion.closedopen(15, portion.inf), column="mixed_col"),
    RangeCase("unbounded_ranges-upper", predicate=pl.col("mixed_col") < 30.5, expected_range=portion.open(-portion.inf, 30.5), column="mixed_col"),
    RangeCase(
        "value_validation", predicate=pl.col("value_col") >= 42.7, expected_range=portion.closedopen(43, portion.inf), validate_value_func=round
    ),
    RangeCase(
        "different_timezone_ranges",
        predicate=(pl.col("timestamp") >= datetime(2023, 5, 15, 12, tzinfo=UTC))
        & (pl.col("timestamp") < datetime(2023, 5, 15, 8, tzinfo=timezone(timedelta(hours=-4)))),
        expected_range=EMPTY,
        column="timestamp",
        validate_value_func=lambda t: t if t.tzinfo is None else t.astimezone(UTC).replace(tzinfo=None),
    ),
    RangeCase(
        "in_with_mixed_types_combined_with_range",
        predicate=pl.col("mixed_col").is_in([date(2023, 1, 15), date(2023, 2, 15), date(2023, 3, 15)]) & (pl.col("mixed_col") >= date(2023, 2, 1)),
        expected_range=portion.singleton(date(2023, 2, 15)) | portion.singleton(date(2023, 3, 15)),
        column="mixed_col",
    ),
    # when/then/otherwise: the range covers both branches; the condition is ignored.
    RangeCase(
        "simple_ternary_with_date_ranges-column_condition",
        predicate=pl.when(pl.col("is_us_market"))
        .then((pl.col("date") >= date(2023, 1, 1)) & (pl.col("date") < date(2023, 3, 1)))
        .otherwise((pl.col("date") >= date(2023, 6, 1)) & (pl.col("date") < date(2023, 9, 1))),
        expected_range=portion.closedopen(date(2023, 1, 1), date(2023, 3, 1)) | portion.closedopen(date(2023, 6, 1), date(2023, 9, 1)),
        column="date",
    ),
    RangeCase(
        "simple_ternary_with_date_ranges-date_condition",
        predicate=pl.when(pl.col("is_us_market") & pl.col("date") >= date(2023, 2, 1))
        .then((pl.col("date") >= date(2023, 1, 1)) & (pl.col("date") < date(2023, 3, 1)))
        .otherwise((pl.col("date") >= date(2023, 6, 1)) & (pl.col("date") < date(2023, 9, 1))),
        expected_range=portion.closedopen(date(2023, 1, 1), date(2023, 3, 1)) | portion.closedopen(date(2023, 6, 1), date(2023, 9, 1)),
        column="date",
    ),
    RangeCase(
        "ternary_one_branch_with_date_range",
        predicate=pl.when(pl.col("is_weekend"))
        .then((pl.col("date") >= date(2023, 1, 1)) & (pl.col("date") < date(2023, 4, 1)))
        .otherwise(pl.lit(True)),
        expected_range=ANY_VALUE,
        column="date",
    ),
    RangeCase(
        "ternary_combined_with_other_conditions",
        predicate=pl.when(pl.col("region") == "US")
        .then((pl.col("date") >= date(2023, 1, 1)) & (pl.col("date") < date(2023, 4, 1)))
        .otherwise((pl.col("date") >= date(2023, 6, 1)) & (pl.col("date") < date(2023, 9, 1)))
        & (pl.col("date") < date(2023, 12, 31)),
        expected_range=portion.closedopen(date(2023, 1, 1), date(2023, 4, 1)) | portion.closedopen(date(2023, 6, 1), date(2023, 9, 1)),
        column="date",
    ),
    # A cast to Boolean is ignored; any other cast hides the comparison, so the range is unlimited.
    RangeCase(
        "boolean_cast_processes_input_node",
        predicate=(pl.col("value_col") >= 10).cast(pl.Boolean),
        expected_range=portion.closedopen(10, portion.inf),
    ),
    RangeCase("non_boolean_cast_ignores_input_node", predicate=(pl.col("value_col") >= 10).cast(pl.Utf8), expected_range=ANY_VALUE),
    RangeCase("target_column_check_in_cast", predicate=(pl.col("other_col") >= 10).cast(pl.Boolean), expected_range=ANY_VALUE),
    RangeCase(
        "nested_boolean_casts_process_all_levels",
        predicate=(pl.col("value_col") >= 10).cast(pl.Boolean).cast(pl.Boolean),
        expected_range=portion.closedopen(10, portion.inf),
    ),
    RangeCase(
        "boolean_cast_with_complex_input",
        predicate=((pl.col("value_col") >= 10) & (pl.col("value_col") < 100)).cast(pl.Boolean),
        expected_range=portion.closedopen(10, 100),
    ),
    RangeCase(
        "boolean_cast_in_conjunction",
        predicate=(pl.col("value_col") >= 10).cast(pl.Boolean) & (pl.col("value_col") < 100),
        expected_range=portion.closedopen(10, 100),
    ),
    RangeCase(
        "cast_with_function",
        predicate=pl.col("value_col").is_in([10, 20, 30]).cast(pl.Boolean),
        expected_range=portion.singleton(10) | portion.singleton(20) | portion.singleton(30),
    ),
    RangeCase(
        "cast_with_ternary",
        predicate=pl.when(pl.col("condition"))
        .then((pl.col("value_col") >= 10) & (pl.col("value_col") < 50))
        .otherwise((pl.col("value_col") >= 100) & (pl.col("value_col") < 150))
        .cast(pl.Boolean),
        expected_range=portion.closedopen(10, 50) | portion.closedopen(100, 150),
    ),
    RangeCase(
        "mixed_boolean_casts_in_expr",
        predicate=(pl.col("value_col") >= 10).cast(pl.Boolean) & (pl.col("value_col") < 100).cast(pl.Utf8),
        expected_range=portion.closedopen(10, portion.inf),
    ),
    RangeCase("boolean_cast_column_directly", predicate=pl.col("value_col").cast(pl.Boolean), expected_range=ANY_VALUE),
    # Aliases are ignored.
    RangeCase(
        "range_alias_simple_equality",
        predicate=pl.col("original_col").alias("renamed_col") == 42,
        expected_range=portion.singleton(42),
        column="original_col",
    ),
    RangeCase(
        "alias_with_range_constraints",
        predicate=(pl.col("value_col").alias("renamed_value") >= 10) & (pl.col("value_col").alias("renamed_value") < 100),
        expected_range=portion.closedopen(10, 100),
    ),
    RangeCase(
        "alias_with_in_function",
        predicate=pl.col("test_col").alias("test_alias").is_in([10, 20, 30]),
        expected_range=portion.singleton(10) | portion.singleton(20) | portion.singleton(30),
        column="test_col",
    ),
    RangeCase(
        "alias_with_between_function",
        predicate=pl.col("range_col").alias("range_alias").is_between(5, 15),
        expected_range=portion.closed(5, 15),
        column="range_col",
    ),
    RangeCase(
        "complex_nested_alias_cast_combinations-cast_alias_alias",
        predicate=pl.col("original").cast(pl.Float64).alias("first").alias("second") == 42,
        expected_range=portion.singleton(42),
        column="original",
    ),
    RangeCase(
        "complex_nested_alias_cast_combinations-alias_cast_alias",
        predicate=pl.col("original").alias("first").cast(pl.Float64).alias("second") == 42,
        expected_range=portion.singleton(42),
        column="original",
    ),
    RangeCase(
        "complex_nested_alias_cast_combinations-between",
        predicate=pl.col("value").cast(pl.Float64).alias("v1").alias("v2").is_between(10, 20),
        expected_range=portion.closed(10, 20),
        column="value",
    ),
    # A NaN cannot limit a column.
    RangeCase("unusable_nan-eq", predicate=pl.col("x") == _NAN, expected_range=ANY_VALUE, column="x"),
    RangeCase("unusable_nan-is_in", predicate=pl.col("x").is_in([1.0, _NAN]), expected_range=ANY_VALUE, column="x"),
    RangeCase("unusable_nan-is_between", predicate=pl.col("x").is_between(0, _NAN), expected_range=portion.closed(0, portion.inf), column="x"),
    RangeCase("unusable_nan-or", predicate=pl.col("x").is_between(0, 10) | (pl.col("x") == _NAN), expected_range=ANY_VALUE, column="x"),
    # an integer column is compared with a float as Float64, so a float of magnitude 2**53 or more cannot bound it
    RangeCase(
        "float_rounds_int-is_between",
        predicate=pl.col("x").is_between(1.0, float(2**53)) & (pl.col("x") <= 5),
        expected_range=portion.closed(1.0, 5),
        column="x",
    ),
]


@pytest.mark.parametrize("case", as_params(RANGE_CASES))
def test_range(case: RangeCase):
    assert convert_expr_to_range(case.predicate, case.column, case.create_point_func, case.validate_value_func) == case.expected_range


# --- convert_expr_to_datetime_range -----------------------------------------------------------------------------------

_15TH_OF_JAN_FEB_MAR_2023 = [date(2023, 1, 15), date(2023, 2, 15), date(2023, 3, 15)]
_NEW_YORK_5PM_JAN_1_2023 = datetime(2023, 1, 1, 17, tzinfo=NEW_YORK)
# 2024-01-02 00:00:00 plus 999 ns, the widened upper bound Polars hands on for an inclusive ``<=`` on a ns column.
_JAN_2_2024_NS = pl.Series([datetime(2024, 1, 2)], dtype=pl.Datetime("ns")).to_physical()[0]
_JAN_2_2024_PLUS_999_NS = pl.lit(_JAN_2_2024_NS + 999).cast(pl.Datetime("ns"))
_MICROSECOND_FROM_JAN_2_2024 = portion.closed(datetime(2024, 1, 2), datetime(2024, 1, 2, 0, 0, 0, 1))


def _as_naive_utc(t: datetime) -> datetime:
    return t.astimezone(UTC).replace(tzinfo=None)


@dataclass(frozen=True)
class DatetimeRangeCase:
    """A filter, and the range of date and time values it allows in one column."""

    id: str
    _: KW_ONLY
    predicate: pl.Expr
    expected_range: Interval
    column: str = "date_col"
    # passed on to ``convert_expr_to_datetime_range``
    coerce_date_to_datetime: bool = True
    get_enclosure: bool = True


# Filters on dates and datetimes, and the range of datetimes each one allows.
DATETIME_CASES = [
    DatetimeRangeCase(
        "single_equality-whole_day",
        predicate=pl.col("date_col") == date(2023, 5, 15),
        expected_range=portion.closedopen(datetime(2023, 5, 15), datetime(2023, 5, 16)),
        coerce_date_to_datetime=False,
    ),
    DatetimeRangeCase(
        "single_equality-midnight",
        predicate=pl.col("date_col") == date(2023, 5, 15),
        expected_range=portion.singleton(datetime(2023, 5, 15)),
        coerce_date_to_datetime=True,
    ),
    DatetimeRangeCase(
        "date_range",
        predicate=(pl.col("date_col") >= date(2023, 1, 1)) & (pl.col("date_col") < date(2023, 12, 31)),
        expected_range=portion.closedopen(datetime(2023, 1, 1), datetime(2023, 12, 31)),
    ),
    DatetimeRangeCase(
        "datetime_range",
        predicate=(pl.col("timestamp") > datetime(2023, 5, 15, 9)) & (pl.col("timestamp") <= datetime(2023, 5, 15, 17)),
        expected_range=portion.openclosed(datetime(2023, 5, 15, 9), datetime(2023, 5, 15, 17)),
        column="timestamp",
    ),
    # The range is closed: Polars cuts a finer literal down to the coarser unit of the column.
    *[
        DatetimeRangeCase(
            f"cast_date_and_datetime-strict={strict}",
            predicate=(pl.col("date_col") >= date(2023, 1, 1)) & (pl.col("date_col").cast(pl.Datetime, strict=strict) < datetime(2023, 12, 31)),
            expected_range=portion.closed(datetime(2023, 1, 1), datetime(2023, 12, 31)),
            coerce_date_to_datetime=True,
        )
        for strict in (True, False)
    ],
    # A date literal does not bound a column cast to a sub-day unit.
    *[
        DatetimeRangeCase(
            f"cast_datetime_time_unit-coerce={coerce}",
            predicate=pl.col("datetime_col").cast(pl.Datetime(time_unit="ns")) == date(2023, 12, 31),
            expected_range=ANY_VALUE,
            column="datetime_col",
            coerce_date_to_datetime=coerce,
        )
        for coerce in (True, False)
    ],
    DatetimeRangeCase(
        "datetime_cast_time",
        predicate=pl.col("datetime_col").is_between(datetime(2023, 1, 1, 16), datetime(2023, 1, 1, 18))
        & pl.col("datetime_col").cast(pl.Time()).is_between(time(12, 15), time(19)),
        expected_range=portion.closed(datetime(2023, 1, 1, 16), datetime(2023, 1, 1, 18)),
        column="datetime_col",
        coerce_date_to_datetime=True,
    ),
    DatetimeRangeCase(
        "cast_datetime_time_zone",
        predicate=pl.col("datetime_col")
        .cast(pl.Datetime("ns", "America/New_York"))
        .is_between(_NEW_YORK_5PM_JAN_1_2023 - timedelta(hours=1), _NEW_YORK_5PM_JAN_1_2023 + timedelta(hours=1)),
        expected_range=portion.closed(
            _as_naive_utc(_NEW_YORK_5PM_JAN_1_2023 - timedelta(hours=1)),
            _as_naive_utc(_NEW_YORK_5PM_JAN_1_2023 + timedelta(hours=1)) + timedelta(microseconds=1),
        ),
        column="datetime_col",
        coerce_date_to_datetime=True,
    ),
    DatetimeRangeCase(
        "or_conditions",
        predicate=(pl.col("date_col") == date(2023, 1, 15)) | (pl.col("date_col") == date(2023, 2, 15)),
        expected_range=portion.closed(datetime(2023, 1, 15), datetime(2023, 2, 15)),
    ),
    DatetimeRangeCase(
        "unbounded_min", predicate=pl.col("date_col") <= date(2023, 12, 31), expected_range=portion.openclosed(-portion.inf, datetime(2023, 12, 31))
    ),
    DatetimeRangeCase(
        "unbounded_max", predicate=pl.col("date_col") >= date(2023, 1, 1), expected_range=portion.closedopen(datetime(2023, 1, 1), portion.inf)
    ),
    DatetimeRangeCase(
        "in_operator-whole_days",
        predicate=pl.col("date_col").is_in(_15TH_OF_JAN_FEB_MAR_2023),
        expected_range=portion.closedopen(datetime(2023, 1, 15), datetime(2023, 3, 16)),
        coerce_date_to_datetime=False,
    ),
    DatetimeRangeCase(
        "in_operator-midnights",
        predicate=pl.col("date_col").is_in(_15TH_OF_JAN_FEB_MAR_2023),
        expected_range=portion.closed(datetime(2023, 1, 15), datetime(2023, 3, 15)),
        coerce_date_to_datetime=True,
    ),
    DatetimeRangeCase(
        "in_with_other_conditions-and",
        predicate=pl.col("date_col").is_in(_15TH_OF_JAN_FEB_MAR_2023) & (pl.col("date_col") >= date(2023, 2, 1)),
        expected_range=portion.closedopen(datetime(2023, 2, 15), datetime(2023, 3, 16)),
        coerce_date_to_datetime=False,
    ),
    DatetimeRangeCase(
        "in_with_other_conditions-or",
        predicate=pl.col("date_col").is_in(_15TH_OF_JAN_FEB_MAR_2023) | (pl.col("date_col") >= date(2023, 2, 1)),
        expected_range=portion.closedopen(datetime(2023, 1, 15), portion.inf),
        coerce_date_to_datetime=False,
    ),
    # != cannot be written as a range and is ignored
    DatetimeRangeCase(
        "not_equal",
        predicate=(pl.col("date_col") >= date(2023, 5, 10)) & (pl.col("date_col") <= date(2023, 5, 20)) & (pl.col("date_col") != date(2023, 5, 15)),
        expected_range=portion.closed(datetime(2023, 5, 10), datetime(2023, 5, 20)),
    ),
    DatetimeRangeCase(
        "contradiction_equal_not_equal",
        predicate=(pl.col("date_col") == date(2023, 5, 15)) & (pl.col("date_col") != date(2023, 5, 15)),
        expected_range=portion.closedopen(datetime(2023, 5, 15), datetime(2023, 5, 16)),
        coerce_date_to_datetime=False,
    ),
    DatetimeRangeCase(
        "contradiction_range-is_between",
        predicate=pl.col("time_col").is_between(datetime(2024, 2, 2), datetime(2023, 2, 3), closed="left"),
        expected_range=EMPTY,
        column="time_col",
    ),
    DatetimeRangeCase(
        "contradiction_range-ge_lt",
        predicate=(pl.col("time_col") >= date(2023, 5, 15)) & (pl.col("time_col") < date(2023, 5, 10)),
        expected_range=EMPTY,
        column="time_col",
    ),
    DatetimeRangeCase(
        "multiple_equalities_consistent",
        predicate=(pl.col("date_col") == date(2023, 5, 15)) & (pl.col("date_col") == date(2023, 5, 15)),
        expected_range=portion.closedopen(datetime(2023, 5, 15), datetime(2023, 5, 16)),
        coerce_date_to_datetime=False,
    ),
    DatetimeRangeCase(
        "multiple_equalities_contradictory-midnight",
        predicate=(pl.col("date_col") == date(2023, 5, 15)) & (pl.col("date_col") == date(2023, 5, 16)),
        expected_range=EMPTY,
    ),
    DatetimeRangeCase(
        "multiple_equalities_contradictory-whole_day",
        predicate=(pl.col("date_col") == date(2023, 5, 15)) & (pl.col("date_col") == date(2023, 5, 16)),
        expected_range=EMPTY,
        coerce_date_to_datetime=False,
    ),
    DatetimeRangeCase(
        "column_not_in_expression", predicate=(pl.col("other_col1") == "value1") | (pl.col("other_col2") == "value2"), expected_range=ANY_VALUE
    ),
    DatetimeRangeCase(
        "column_with_other_columns",
        predicate=(pl.col("other_col") == "some_value") & (pl.col("date_col") >= date(2023, 5, 15)),
        expected_range=portion.closedopen(datetime(2023, 5, 15), portion.inf),
    ),
    DatetimeRangeCase("is_null", predicate=pl.col("date_col").is_null(), expected_range=ANY_VALUE),
    DatetimeRangeCase(
        "datetime_comparisons_with_timezone",
        predicate=(pl.col("timestamp") >= datetime(2023, 5, 15, 9, tzinfo=UTC)) & (pl.col("timestamp") <= datetime(2023, 5, 15, 17, tzinfo=UTC)),
        expected_range=portion.closed(datetime(2023, 5, 15, 9), datetime(2023, 5, 15, 17)),
        column="timestamp",
    ),
    # By default an OR of ranges gives one range from the lowest start to the highest end.
    DatetimeRangeCase(
        "complex_or_conditions",
        predicate=((pl.col("date_col") >= date(2023, 1, 1)) & (pl.col("date_col") < date(2023, 6, 1)))
        | ((pl.col("date_col") >= date(2023, 5, 1)) & (pl.col("date_col") < date(2023, 12, 31))),
        expected_range=portion.closedopen(datetime(2023, 1, 1), datetime(2023, 12, 31)),
    ),
    DatetimeRangeCase(
        "disjoint_ranges",
        predicate=((pl.col("date_col") >= date(2023, 1, 1)) & (pl.col("date_col") < date(2023, 3, 1)))
        | ((pl.col("date_col") >= date(2023, 6, 1)) & (pl.col("date_col") < date(2023, 9, 1))),
        expected_range=portion.closedopen(datetime(2023, 1, 1), datetime(2023, 9, 1)),
    ),
    DatetimeRangeCase(
        "datetime_vs_date",
        predicate=(pl.col("mixed_col") >= date(2023, 5, 15)) & (pl.col("mixed_col") < datetime(2023, 5, 15, 12)),
        expected_range=portion.closedopen(datetime(2023, 5, 15), datetime(2023, 5, 15, 12)),
        column="mixed_col",
    ),
    DatetimeRangeCase(
        "datetime_with_multiple_expr",
        predicate=pl.col("Date").is_between(datetime(2023, 5, 15, 12), datetime(2023, 5, 16, 12)) & pl.col("symol").is_in(["A", "B"]),
        expected_range=portion.closed(datetime(2023, 5, 15, 12), datetime(2023, 5, 16, 12)),
        column="Date",
    ),
    DatetimeRangeCase(
        "datetime_microseconds",
        predicate=(pl.col("timestamp") > datetime(2023, 5, 15, 9)) & (pl.col("timestamp") <= datetime(2023, 5, 15, 9, 0, 0, 1)),
        expected_range=portion.openclosed(datetime(2023, 5, 15, 9), datetime(2023, 5, 15, 9, 0, 0, 1)),
        column="timestamp",
    ),
    DatetimeRangeCase(
        "adjacent_date_ranges",
        predicate=((pl.col("date_col") >= date(2023, 1, 1)) & (pl.col("date_col") < date(2023, 2, 1)))
        | ((pl.col("date_col") >= date(2023, 2, 1)) & (pl.col("date_col") < date(2023, 3, 1))),
        expected_range=portion.closedopen(datetime(2023, 1, 1), datetime(2023, 3, 1)),
    ),
    DatetimeRangeCase(
        "contradiction_in_one_conjunction-midnight",
        predicate=((pl.col("date_col") >= date(2023, 5, 15)) & (pl.col("date_col") < date(2023, 5, 10))) | (pl.col("date_col") == date(2023, 6, 1)),
        expected_range=portion.singleton(datetime(2023, 6, 1)),
    ),
    DatetimeRangeCase(
        "contradiction_in_one_conjunction-whole_day",
        predicate=((pl.col("date_col") >= date(2023, 5, 15)) & (pl.col("date_col") < date(2023, 5, 10))) | (pl.col("date_col") == date(2023, 6, 1)),
        expected_range=portion.closedopen(datetime(2023, 6, 1), datetime(2023, 6, 2)),
        coerce_date_to_datetime=False,
    ),
    DatetimeRangeCase("in_empty_list", predicate=pl.col("date_col").is_in([]), expected_range=EMPTY),
    DatetimeRangeCase(
        "not_in_combined_with_range",
        predicate=(pl.col("date_col") >= date(2023, 1, 1))
        & (pl.col("date_col") <= date(2023, 12, 31))
        & ~pl.col("date_col").is_in([date(2023, 7, 4)]),
        expected_range=portion.closed(datetime(2023, 1, 1), datetime(2023, 12, 31)),
    ),
    DatetimeRangeCase(
        "datetime_range_with_complex_expr",
        predicate=((pl.col("date") >= date(2023, 1, 1)) & (pl.col("date") < date(2023, 6, 1)))
        | ((pl.col("date") >= date(2023, 8, 1)) & (pl.col("date") < date(2023, 12, 31))),
        expected_range=portion.closedopen(datetime(2023, 1, 1), datetime(2023, 12, 31)),
        column="date",
    ),
    DatetimeRangeCase(
        "multiple_in_expressions",
        predicate=pl.col("date_col").is_in([date(2023, 1, 15), date(2023, 2, 15)]) | pl.col("date_col").is_in([date(2023, 3, 15), date(2023, 4, 15)]),
        expected_range=portion.closedopen(datetime(2023, 1, 15), datetime(2023, 4, 16)),
        coerce_date_to_datetime=False,
    ),
    DatetimeRangeCase(
        "nested_expressions",
        predicate=(pl.col("date_col") >= date(2023, 1, 1))
        & ((pl.col("date_col") < date(2023, 3, 1)) | ((pl.col("date_col") >= date(2023, 6, 1)) & (pl.col("date_col") < date(2023, 9, 1)))),
        expected_range=portion.closedopen(datetime(2023, 1, 1), datetime(2023, 9, 1)),
    ),
    # mixed date and datetime literals
    DatetimeRangeCase(
        "contradictory_mixed_types",
        predicate=(pl.col("mixed_col") >= date(2023, 5, 15)) & (pl.col("mixed_col") < datetime(2023, 5, 10, 12)),
        expected_range=EMPTY,
        column="mixed_col",
    ),
    DatetimeRangeCase(
        "is_between_mixed_types-both",
        predicate=pl.col("mixed_col").is_between(date(2023, 5, 15), datetime(2023, 5, 16, 12)),
        expected_range=portion.closed(datetime(2023, 5, 15), datetime(2023, 5, 16, 12)),
        column="mixed_col",
        get_enclosure=False,
    ),
    DatetimeRangeCase(
        "is_between_mixed_types-left",
        predicate=pl.col("mixed_col").is_between(date(2023, 5, 15), datetime(2023, 5, 16, 12), closed="left"),
        expected_range=portion.closedopen(datetime(2023, 5, 15), datetime(2023, 5, 16, 12)),
        column="mixed_col",
        get_enclosure=False,
    ),
    DatetimeRangeCase(
        "complex_or_with_mixed_types",
        predicate=((pl.col("mixed_col") >= date(2023, 1, 1)) & (pl.col("mixed_col") < datetime(2023, 3, 1)))
        | ((pl.col("mixed_col") >= datetime(2023, 5, 1, 12)) & (pl.col("mixed_col") < date(2023, 6, 1))),
        expected_range=portion.closedopen(datetime(2023, 1, 1), datetime(2023, 3, 1))
        | portion.closedopen(datetime(2023, 5, 1, 12), datetime(2023, 6, 1)),
        column="mixed_col",
        get_enclosure=False,
    ),
    DatetimeRangeCase(
        "complex_overlapping_ranges",
        predicate=((pl.col("mixed_col") >= date(2023, 1, 1)) & (pl.col("mixed_col") < datetime(2023, 3, 1)))
        | ((pl.col("mixed_col") >= date(2023, 2, 15)) & (pl.col("mixed_col") < datetime(2023, 4, 1)))
        | ((pl.col("mixed_col") >= date(2023, 6, 1)) & (pl.col("mixed_col") < datetime(2023, 7, 1))),
        expected_range=portion.closedopen(datetime(2023, 1, 1), datetime(2023, 4, 1))
        | portion.closedopen(datetime(2023, 6, 1), datetime(2023, 7, 1)),
        column="mixed_col",
        get_enclosure=False,
    ),
    # casts
    DatetimeRangeCase(
        "boolean_cast_with_datetime",
        predicate=((pl.col("date_col") >= date(2023, 1, 1)) & (pl.col("date_col") < date(2023, 12, 31))).cast(pl.Boolean),
        expected_range=portion.closedopen(datetime(2023, 1, 1), datetime(2023, 12, 31)),
    ),
    *[
        DatetimeRangeCase(
            f"cast_date_to_datetime_with_comparison-strict={strict}",
            predicate=(pl.col("date_col") >= date(2023, 1, 1)) & (pl.col("date_col").cast(pl.Datetime, strict=strict) <= datetime(2023, 12, 31, 12)),
            expected_range=portion.closed(datetime(2023, 1, 1), datetime(2023, 12, 31, 12, 0, 0, 1)),
        )
        for strict in (True, False)
    ],
    DatetimeRangeCase(
        "cast_with_time_unit_and_timezone",
        predicate=pl.col("date_col").cast(pl.Datetime("us", "UTC")) >= datetime(2023, 1, 1, tzinfo=UTC),
        expected_range=portion.closedopen(datetime(2023, 1, 1), portion.inf),
    ),
    DatetimeRangeCase(
        "boolean_cast_of_datetime_comparison",
        predicate=(pl.col("dt_col") == datetime(2023, 5, 15, 12)).cast(pl.Boolean),
        expected_range=portion.singleton(datetime(2023, 5, 15, 12)),
        column="dt_col",
        coerce_date_to_datetime=True,
    ),
    DatetimeRangeCase("cast_boolean_to_datetime", predicate=pl.col("bool_col").cast(pl.Datetime), expected_range=ANY_VALUE, column="bool_col"),
    DatetimeRangeCase(
        "nested_cast_to_boolean",
        predicate=(pl.col("a").is_in([1, 5, 2]) & pl.col("a").is_in([5, 2]).not_())
        .cast(pl.Boolean)
        .and_(((pl.col("date") >= date(2023, 10, 1)) & (pl.col("date") <= date(2023, 10, 3))).cast(pl.Boolean)),
        expected_range=portion.closed(datetime(2023, 10, 1), datetime(2023, 10, 3)),
        column="date",
        get_enclosure=False,
    ),
    DatetimeRangeCase(
        "alias_datetime_range_extraction",
        predicate=(pl.col("date_col").alias("date_alias") >= date(2023, 1, 1)) & (pl.col("date_col").alias("date_alias") <= date(2023, 12, 31)),
        expected_range=portion.closed(datetime(2023, 1, 1), datetime(2023, 12, 31)),
    ),
    # A cast to a coarser unit rounds down, so the range covers every value that rounds down onto the literal.
    DatetimeRangeCase(
        "microsecond_cast_bound_covers_sub_microseconds-eq",
        predicate=pl.col("t").cast(pl.Datetime("us")) == datetime(2024, 1, 2, 12, 30),
        expected_range=portion.closed(datetime(2024, 1, 2, 12, 30), datetime(2024, 1, 2, 12, 30, 0, 1)),
        column="t",
    ),
    DatetimeRangeCase(
        "microsecond_cast_bound_covers_sub_microseconds-le",
        predicate=pl.col("t").cast(pl.Datetime("us")) <= datetime(2024, 1, 2, 12, 30),
        expected_range=portion.openclosed(-portion.inf, datetime(2024, 1, 2, 12, 30, 0, 1)),
        column="t",
    ),
    DatetimeRangeCase(
        "floored_membership_at_maximum_is_unbounded",
        predicate=pl.col("t").cast(pl.Datetime("us")).is_in([datetime.max]),
        expected_range=portion.closedopen(datetime.max, portion.inf),
        column="t",
    ),
    # Literals that cannot limit a column leave its range unlimited.
    DatetimeRangeCase("unusable_datetime-gt_none", predicate=pl.col("t") > None, expected_range=ANY_VALUE, column="t"),
    DatetimeRangeCase("unusable_datetime-eq_missing_none", predicate=pl.col("t").eq_missing(None), expected_range=ANY_VALUE, column="t"),
    DatetimeRangeCase(
        "unusable_datetime-is_in_with_null",
        predicate=pl.col("t").is_in([datetime(2024, 1, 1), None], nulls_equal=True),
        expected_range=ANY_VALUE,
        column="t",
    ),
    DatetimeRangeCase(
        "unusable_datetime-is_in_only_null", predicate=pl.col("t").is_in([None], nulls_equal=True), expected_range=ANY_VALUE, column="t"
    ),
    DatetimeRangeCase(
        "single_element_series_is_in",
        predicate=pl.col("t").is_in(pl.Series([datetime(2024, 1, 1)])),
        expected_range=portion.singleton(datetime(2024, 1, 1)),
        column="t",
    ),
    DatetimeRangeCase(
        "is_in_with_failed_literal_leaves_column_unconstrained",
        predicate=pl.col("t").is_in(pl.Series([1_704_153_600_000_000_000], dtype=pl.Int64).cast(pl.Datetime("us"))),
        expected_range=ANY_VALUE,
        column="t",
    ),
    # A nanosecond bound with a sub-microsecond part is rounded away from the kept side to whole microseconds, so the
    # range stays finite (issue #92); an equality or is_in member widens to its enclosing microsecond window.
    DatetimeRangeCase(
        "is_between_window_is_not_lost",
        predicate=pl.col("t").is_between(datetime(2024, 1, 1), _JAN_2_2024_PLUS_999_NS),
        expected_range=portion.closed(datetime(2024, 1, 1), _MICROSECOND_FROM_JAN_2_2024.upper),
        column="t",
        get_enclosure=False,
    ),
    DatetimeRangeCase(
        "equality_widens_to_microsecond_window",
        predicate=pl.col("t") == _JAN_2_2024_PLUS_999_NS,
        expected_range=_MICROSECOND_FROM_JAN_2_2024,
        column="t",
        get_enclosure=False,
    ),
    DatetimeRangeCase(
        "is_in_keeps_exact_member_as_point_and_widens_sub_microsecond_member",
        predicate=pl.col("t").is_in(pl.Series([_JAN_2_2024_NS - 1_000_000_000, _JAN_2_2024_NS + 999], dtype=pl.Int64).cast(pl.Datetime("ns"))),
        expected_range=portion.singleton(datetime(2024, 1, 1, 23, 59, 59)) | _MICROSECOND_FROM_JAN_2_2024,
        column="t",
        get_enclosure=False,
    ),
]


@pytest.mark.parametrize("case", as_params(DATETIME_CASES))
def test_datetime_range(case: DatetimeRangeCase):
    got = convert_expr_to_datetime_range(case.predicate, case.column, case.coerce_date_to_datetime, case.get_enclosure)
    assert got == case.expected_range


@pytest.mark.parametrize(
    "constraints", [None, portion.empty(), portion.closedopen(date(2023, 1, 1), date(2023, 1, 2))], ids=["none", "empty", "interval"]
)
def test_range_visitor_copy(constraints):
    visitor = IntervalVisitor("date_col", None, None)
    if constraints is not None:
        visitor.constraints = constraints
    new_visitor = visitor.copy()
    assert new_visitor is not visitor
    assert new_visitor.target_column == visitor.target_column
    assert new_visitor.constraints == visitor.constraints


# --- interval helpers -------------------------------------------------------------------------------------------------

_JAN_1_TO_3_AND_5_TO_7 = portion.closed(datetime(2023, 1, 1), datetime(2023, 1, 3)) | portion.closed(datetime(2023, 1, 5), datetime(2023, 1, 7))


@dataclass(frozen=True)
class ShiftIntervalCase:
    """An interval shifted by a number of days, and the result."""

    id: str
    _: KW_ONLY
    # the lookback or lookahead shift
    shift: Callable
    interval: Interval
    # days
    delta: int
    expect: Interval


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            ShiftIntervalCase(
                "lookback-simple_closed_interval",
                shift=_lookback_interval,
                interval=portion.closed(datetime(2023, 1, 5), datetime(2023, 1, 10)),
                delta=1,
                expect=portion.closed(datetime(2023, 1, 4), datetime(2023, 1, 10)),
            ),
            ShiftIntervalCase(
                "lookback-open_interval",
                shift=_lookback_interval,
                interval=portion.open(datetime(2023, 1, 5), datetime(2023, 1, 10)),
                delta=2,
                expect=portion.open(datetime(2023, 1, 3), datetime(2023, 1, 10)),
            ),
            ShiftIntervalCase(
                "lookback-infinite_lower_bound",
                shift=_lookback_interval,
                interval=portion.openclosed(-portion.inf, datetime(2023, 1, 10)),
                delta=1,
                expect=portion.openclosed(-portion.inf, datetime(2023, 1, 10)),
            ),
            ShiftIntervalCase(
                "lookback-compound_interval",
                shift=_lookback_interval,
                interval=_JAN_1_TO_3_AND_5_TO_7,
                delta=1,
                expect=portion.closed(datetime(2022, 12, 31), datetime(2023, 1, 3)) | portion.closed(datetime(2023, 1, 4), datetime(2023, 1, 7)),
            ),
            ShiftIntervalCase(
                "lookback-date_bounds",
                shift=_lookback_interval,
                interval=portion.closed(date(2023, 1, 5), date(2023, 1, 10)),
                delta=3,
                expect=portion.closed(date(2023, 1, 2), date(2023, 1, 10)),
            ),
            ShiftIntervalCase(
                "lookahead-simple_closed_interval",
                shift=_lookahead_interval,
                interval=portion.closed(datetime(2023, 1, 5), datetime(2023, 1, 10)),
                delta=1,
                expect=portion.closed(datetime(2023, 1, 5), datetime(2023, 1, 11)),
            ),
            ShiftIntervalCase(
                "lookahead-open_interval",
                shift=_lookahead_interval,
                interval=portion.open(datetime(2023, 1, 5), datetime(2023, 1, 10)),
                delta=2,
                expect=portion.open(datetime(2023, 1, 5), datetime(2023, 1, 12)),
            ),
            ShiftIntervalCase(
                "lookahead-infinite_upper_bound",
                shift=_lookahead_interval,
                interval=portion.closedopen(datetime(2023, 1, 10), portion.inf),
                delta=1,
                expect=portion.closedopen(datetime(2023, 1, 10), portion.inf),
            ),
            ShiftIntervalCase(
                "lookahead-compound_interval",
                shift=_lookahead_interval,
                interval=_JAN_1_TO_3_AND_5_TO_7,
                delta=1,
                expect=portion.closed(datetime(2023, 1, 1), datetime(2023, 1, 4)) | portion.closed(datetime(2023, 1, 5), datetime(2023, 1, 8)),
            ),
            ShiftIntervalCase(
                "lookahead-date_bounds",
                shift=_lookahead_interval,
                interval=portion.closed(date(2023, 1, 5), date(2023, 1, 10)),
                delta=3,
                expect=portion.closed(date(2023, 1, 5), date(2023, 1, 13)),
            ),
        ]
    ),
)
def test_shift_interval(case: ShiftIntervalCase):
    assert case.shift(case.interval, timedelta(days=case.delta)) == case.expect


@dataclass(frozen=True)
class FullDatesCase:
    """An interval, and the interval widened to whole days."""

    id: str
    _: KW_ONLY
    interval: Interval
    expect: Interval


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            FullDatesCase(
                "datetime_bounds",
                interval=portion.closed(datetime(2023, 1, 5, 12, 30, 45), datetime(2023, 1, 10, 14, 20, 15)),
                expect=portion.closed(date(2023, 1, 5), date(2023, 1, 10)),
            ),
            FullDatesCase(
                "date_bounds",
                interval=portion.closed(date(2023, 1, 5), date(2023, 1, 10)),
                expect=portion.closed(date(2023, 1, 5), date(2023, 1, 10)),
            ),
            FullDatesCase("infinite_bounds", interval=ANY_VALUE, expect=ANY_VALUE),
            FullDatesCase(
                "compound_interval",
                interval=portion.closed(datetime(2023, 1, 1, 8, 30), datetime(2023, 1, 3, 16, 45))
                | portion.closed(datetime(2023, 1, 5, 9, 15), datetime(2023, 1, 7, 17, 30)),
                expect=portion.closed(date(2023, 1, 1), date(2023, 1, 3)) | portion.closed(date(2023, 1, 5), date(2023, 1, 7)),
            ),
            # A Date value ``d`` satisfies ``d < upper`` when ``upper`` falls after midnight on ``d``.
            FullDatesCase(
                "open_upper_bound_keeps_date_after_midnight",
                interval=portion.closedopen(datetime(2023, 12, 1), datetime(2024, 1, 1, 12)),
                expect=portion.closed(date(2023, 12, 1), date(2024, 1, 1)),
            ),
        ]
    ),
)
def test_extend_to_full_dates(case: FullDatesCase):
    assert _extend_to_full_dates(case.interval) == case.expect


_FIVE_DAYS_IN_JAN = [datetime(2023, 1, n) for n in (4, 5, 7, 10, 11)]
_DEC_31_TO_JAN_8 = [datetime(2022, 12, 31), *(datetime(2023, 1, n) for n in range(1, 9))]


@dataclass(frozen=True)
class IntervalExprCase:
    """An interval converted to a Polars filter, and the rows of ``dates`` the filter keeps."""

    id: str
    _: KW_ONLY
    convert: Callable
    interval: Interval
    dates: list
    # indices into ``dates``; None and False are the literal results for no restriction and for nothing
    kept: list[int] | bool | None


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            IntervalExprCase(
                "atomic-fully_open_interval",
                convert=_convert_atomic_interval_to_polars_expr,
                interval=portion.open(datetime(2023, 1, 5), datetime(2023, 1, 10)),
                dates=_FIVE_DAYS_IN_JAN,
                kept=[2],
            ),
            IntervalExprCase(
                "atomic-fully_closed_interval",
                convert=_convert_atomic_interval_to_polars_expr,
                interval=portion.closed(datetime(2023, 1, 5), datetime(2023, 1, 10)),
                dates=_FIVE_DAYS_IN_JAN,
                kept=[1, 2, 3],
            ),
            IntervalExprCase(
                "atomic-left_open_right_closed_interval",
                convert=_convert_atomic_interval_to_polars_expr,
                interval=portion.openclosed(datetime(2023, 1, 5), datetime(2023, 1, 10)),
                dates=_FIVE_DAYS_IN_JAN,
                kept=[2, 3],
            ),
            IntervalExprCase(
                "atomic-left_closed_right_open_interval",
                convert=_convert_atomic_interval_to_polars_expr,
                interval=portion.closedopen(datetime(2023, 1, 5), datetime(2023, 1, 10)),
                dates=_FIVE_DAYS_IN_JAN,
                kept=[1, 2],
            ),
            IntervalExprCase(
                "atomic-infinite_lower_bound",
                convert=_convert_atomic_interval_to_polars_expr,
                interval=portion.openclosed(-portion.inf, datetime(2023, 1, 10)),
                dates=[datetime(2022, 1, 1), datetime(2023, 1, 4), datetime(2023, 1, 10), datetime(2023, 1, 11)],
                kept=[0, 1, 2],
            ),
            IntervalExprCase(
                "atomic-infinite_upper_bound",
                convert=_convert_atomic_interval_to_polars_expr,
                interval=portion.closedopen(datetime(2023, 1, 5), portion.inf),
                dates=[datetime(2023, 1, 4), datetime(2023, 1, 5), datetime(2023, 1, 10), datetime(2024, 1, 1)],
                kept=[1, 2, 3],
            ),
            IntervalExprCase(
                "atomic-full_infinite_interval", convert=_convert_atomic_interval_to_polars_expr, interval=ANY_VALUE, dates=[], kept=None
            ),
            IntervalExprCase(
                "atomic-singleton_interval",
                convert=_convert_atomic_interval_to_polars_expr,
                interval=portion.singleton(datetime(2023, 1, 5)),
                dates=[datetime(2023, 1, n) for n in (4, 5, 6)],
                kept=[1],
            ),
            IntervalExprCase(
                "simple_atomic_interval",
                convert=_convert_interval_to_polars_expr,
                interval=portion.closed(datetime(2023, 1, 5), datetime(2023, 1, 10)),
                dates=_FIVE_DAYS_IN_JAN,
                kept=[1, 2, 3],
            ),
            IntervalExprCase(
                "compound_interval",
                convert=_convert_interval_to_polars_expr,
                interval=_JAN_1_TO_3_AND_5_TO_7,
                dates=_DEC_31_TO_JAN_8,
                kept=[1, 2, 3, 5, 6, 7],
            ),
            IntervalExprCase("empty_interval", convert=_convert_interval_to_polars_expr, interval=EMPTY, dates=[], kept=False),
            IntervalExprCase(
                "universe_interval", convert=_convert_interval_to_polars_expr, interval=portion.closed(-portion.inf, portion.inf), dates=[], kept=None
            ),
            IntervalExprCase(
                "mixed_bounds_compound",
                convert=_convert_interval_to_polars_expr,
                interval=portion.openclosed(datetime(2023, 1, 1), datetime(2023, 1, 3))
                | portion.closedopen(datetime(2023, 1, 5), datetime(2023, 1, 7)),
                dates=_DEC_31_TO_JAN_8,
                kept=[2, 3, 5, 6],
            ),
        ]
    ),
)
def test_interval_to_polars_expr(case: IntervalExprCase):
    """The expression keeps the rows at indices ``kept``; ``None`` (no restriction) and ``False`` (nothing) are literal."""
    expr = case.convert(case.interval, "date")
    if case.kept is None or case.kept is False:
        assert expr is case.kept
    else:
        assert pl.DataFrame({"date": case.dates}).filter(expr)["date"].to_list() == [case.dates[i] for i in case.kept]


# --- every row a filter keeps stays inside its range ------------------------------------------------------------------


def _assert_range_keeps_rows(df: pl.DataFrame, pred: pl.Expr, col: str, interval: portion.Interval) -> None:
    expr = _convert_interval_to_polars_expr(interval, col)
    assert_pushed_filter_keeps_rows(df, pred, pl.lit(expr) if isinstance(expr, bool) else expr)


@pytest.mark.parametrize(
    "pred",
    [
        pytest.param(pl.col("t").dt.date() == date(2024, 1, 1), id="eq"),
        pytest.param(pl.col("t").dt.date() <= date(2024, 1, 1), id="le"),
        pytest.param(pl.col("t").cast(pl.Date).is_in([date(2024, 1, 1)]), id="cast_is_in"),
        pytest.param(pl.col("t").dt.date().is_between(date(2023, 12, 1), date(2024, 1, 1)), id="is_between"),
    ],
)
def test_day_floored_column_bound_covers_whole_day(pred):
    df = pl.DataFrame({"t": [datetime(2023, 12, 31, 23), datetime(2024, 1, 1, 13), datetime(2024, 1, 2, 1)]})
    interval = convert_expr_to_datetime_range(pred, "t")
    assert all(t in interval for t in df.filter(pred)["t"])


_ROWS_AROUND_JAN_2_2024_NS = pl.DataFrame(
    {"t": pl.Series([_JAN_2_2024_NS + o for o in [-2000, -1, 0, 1, 500, 999, 1000, 2000]], dtype=pl.Int64).cast(pl.Datetime("ns"))}
)


@pytest.mark.parametrize(
    "pred",
    [
        pytest.param(pl.col("t").is_between(datetime(2024, 1, 1), _JAN_2_2024_PLUS_999_NS), id="is_between"),
        pytest.param(pl.col("t").is_between(datetime(2024, 1, 1), _JAN_2_2024_PLUS_999_NS, closed="right"), id="is_between_right"),
        pytest.param(pl.col("t") <= _JAN_2_2024_PLUS_999_NS, id="le"),
        pytest.param(pl.col("t") < _JAN_2_2024_PLUS_999_NS, id="lt"),
        pytest.param(pl.col("t") >= _JAN_2_2024_PLUS_999_NS, id="ge"),
        pytest.param(pl.col("t") > _JAN_2_2024_PLUS_999_NS, id="gt"),
        pytest.param(pl.col("t") == _JAN_2_2024_PLUS_999_NS, id="eq"),
    ],
)
def test_range_keeps_every_matching_row(pred):
    interval = convert_expr_to_datetime_range(pred, "t", get_enclosure=False)
    assert not interval.empty and interval != ANY_VALUE
    _assert_range_keeps_rows(_ROWS_AROUND_JAN_2_2024_NS, pred, "t", interval)


@pytest.mark.parametrize(
    "pred",
    [
        pytest.param(~(pl.col("t") < _JAN_2_2024_PLUS_999_NS), id="not_lt"),
        pytest.param(~(pl.col("t") <= _JAN_2_2024_PLUS_999_NS), id="not_le"),
        pytest.param(~(pl.col("t") > _JAN_2_2024_PLUS_999_NS), id="not_gt"),
        pytest.param(~(pl.col("t") >= _JAN_2_2024_PLUS_999_NS), id="not_ge"),
        pytest.param(pl.col("t") != _JAN_2_2024_PLUS_999_NS, id="ne"),
        pytest.param(~pl.col("t").is_between(datetime(2024, 1, 1), _JAN_2_2024_PLUS_999_NS), id="not_is_between"),
        pytest.param(~(~(pl.col("t") < _JAN_2_2024_PLUS_999_NS)), id="not_not_lt"),
    ],
)
def test_negation_does_not_narrow_below_matching_rows(pred):
    """The range of a negated part of the filter is unlimited (the converter never takes the complement), so no row is lost."""
    _assert_range_keeps_rows(_ROWS_AROUND_JAN_2_2024_NS, pred, "t", convert_expr_to_datetime_range(pred, "t", get_enclosure=False))


@pytest.mark.parametrize("op", [operator.ge, operator.gt, operator.eq, operator.le, operator.lt], ids=lambda op: op.__name__)
def test_sub_microsecond_negative_duration_bound_keeps_matching_rows(op):
    """Polars rounds a negative ``Duration("ns")`` toward zero, so such a bound is left unlimited instead of rounded."""
    df = pl.DataFrame({"d": pl.Series([-1200, -1001, -1, 0, 1, 1200], dtype=pl.Int64).cast(pl.Duration("ns"))})
    pred = op(pl.col("d"), pl.lit(-1200).cast(pl.Duration("ns")))
    _assert_range_keeps_rows(df, pred, "d", convert_expr_to_range(pred, "d"))


# 2024-11-03 06:30:00 UTC plus 999 ns: 01:30 in New York in the repeated hour when clocks go back, the second time
# (``fold=1`` in Python).
_NEW_YORK_NS = pl.Datetime("ns", "America/New_York")
_NEW_YORK_REPEATED_HOUR_PLUS_999_NS = pl.lit(pl.Series([datetime(2024, 11, 3, 6, 30)], dtype=pl.Datetime("ns")).to_physical()[0] + 999).cast(
    _NEW_YORK_NS
)
_NEW_YORK_0030_ON_CLOCK_CHANGE_DAY = pl.lit(datetime(2024, 11, 3, 4, 30, tzinfo=UTC)).cast(_NEW_YORK_NS)


@pytest.mark.parametrize(
    "pred",
    [
        pytest.param(pl.col("t") == _NEW_YORK_REPEATED_HOUR_PLUS_999_NS, id="eq"),
        pytest.param(pl.col("t").is_in(pl.concat_list(_NEW_YORK_REPEATED_HOUR_PLUS_999_NS)), id="is_in"),
        pytest.param(pl.col("t") < _NEW_YORK_REPEATED_HOUR_PLUS_999_NS, id="lt"),
        pytest.param(pl.col("t") <= _NEW_YORK_REPEATED_HOUR_PLUS_999_NS, id="le"),
        *[
            pytest.param(
                pl.col("t").is_between(_NEW_YORK_0030_ON_CLOCK_CHANGE_DAY, _NEW_YORK_REPEATED_HOUR_PLUS_999_NS, closed=cl), id=f"is_between_{cl}"
            )
            for cl in ("both", "left", "right", "none")
        ],
    ],
)
def test_rounding_up_in_repeated_dst_hour_keeps_matching_rows(pred):
    """Adding a microsecond to an aware datetime is wall-clock arithmetic that forgets which of the two 01:30s of the
    repeated hour when clocks go back it was (``fold``); it must not move the bound back an hour."""
    times = [datetime(2024, 11, 3, h, mi, tzinfo=UTC) for h, mi in [(4, 30), (5, 0), (5, 30), (6, 0), (6, 29), (6, 30)]]
    df = pl.DataFrame({"t": pl.Series(times).cast(_NEW_YORK_NS).dt.offset_by("999ns")}).vstack(
        pl.DataFrame({"t": pl.Series(times).cast(_NEW_YORK_NS)})
    )
    interval = convert_expr_to_datetime_range(pred, "t", get_enclosure=False)
    matching = df.filter(pred)["t"].to_physical().to_list()
    assert matching
    # Compare in UTC nanoseconds: the range holds naive UTC bounds.
    to_ns = lambda b: (b - datetime(1970, 1, 1)) // timedelta(microseconds=1) * 1000
    interval_ns = interval.apply(lambda a: a.replace(lower=to_ns, upper=to_ns))
    assert all(t in interval_ns for t in matching)


_B_BEFORE_A_ENUM = pl.Enum(["b", "a"])


@pytest.mark.parametrize(
    "pred",
    [
        pytest.param(pl.col("x") > pl.lit("b", dtype=_B_BEFORE_A_ENUM), id="gt_enum_literal"),
        pytest.param(pl.col("x") > "b", id="gt_str"),
        pytest.param(pl.col("x") < "a", id="lt_str"),
        pytest.param(pl.col("x").is_between(pl.lit("b", dtype=_B_BEFORE_A_ENUM), pl.lit("a", dtype=_B_BEFORE_A_ENUM)), id="is_between"),
    ],
)
def test_enum_ordering_comparison_keeps_rows(pred):
    """Enum values order by category, so ``"a" > "b"`` for ``Enum(["b", "a"])``."""
    df = pl.DataFrame({"x": pl.Series(["b", "a"], dtype=_B_BEFORE_A_ENUM)})
    interval = convert_expr_to_range(pred, "x")
    assert all(x in interval for x in df.filter(pred)["x"])
