"""pushdown_combine: ``CombineCase`` rows run by one test, and tables for FilterSpec, IntervalFilterSpec and helpers.

A row names its sources (a frame and its filter specs), the combine function and the user's filter. The result must
equal the combine of the unfiltered sources, filtered by the user's filter, unless the row sets
``compare_with_polars=False``; with the identity combine this also means the source kept every row the filter keeps.
A row may also state what a source received. Sources a row inspects, and the source of the identity combine, sit behind
a ``PredicateTracker``, which records what they receive.

The property tests at the end (using the Hypothesis library) run combines of several shapes with random data and
filters, and compare them with plain Polars.
"""

import logging
import sys
import warnings
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest
from hypothesis import assume, given, strategies as st
from polars.testing import assert_frame_equal

import polars_io_tools as cpl
from polars_io_tools.io_sources.base import BinaryExprNode, FunctionNode
from polars_io_tools.io_sources.enum import BooleanFunctionType, OperatorType
from polars_io_tools.io_sources.pushdown_combine import (
    FilterSpec,
    IntervalFilterSpec,
    _apply_value_mapping,
    _compute_output_schema,
    _get_source_col,
    _map_values,
    _union_discrete_filter,
    pushdown_combine,
)
from polars_io_tools.testing import PredicateAnalyzer, PredicateTracker, io_source_assert
from polars_io_tools.tests.helpers.assertions import assert_matches_plain_polars, assert_pushed_time_bounds, failure_note
from polars_io_tools.tests.helpers.cases import GROUP_FRAME, as_params, daily_frame, epoch_ns
from polars_io_tools.tests.helpers.pickling import pickle_round_trip
from polars_io_tools.tests.helpers.polars_versions import polars_has
from polars_io_tools.tests.helpers.property_checks import SETTINGS

UTC_ZONE = ZoneInfo("UTC")
NEW_YORK = ZoneInfo("America/New_York")
CLOSED_OPTIONS = ["both", "left", "right", "none"]


def make_date(day: int, month: int = 1, year: int = 2024) -> date:
    return date(year, month, day)


def days(n: float) -> timedelta:
    return timedelta(days=n)


# Specs, values and combine functions -----------------------------------------------------------------------------------
REGIONS = {"NA": "NORTH_AMERICA", "EU": "EUROPE", "APAC": "ASIA_PACIFIC", "LATAM": "LATIN_AMERICA"}


def to_codes(*codes: str) -> dict:
    return {REGIONS[code]: code for code in codes}


def region_frame(*codes: str) -> pl.DataFrame:
    return pl.DataFrame({"region_code": list(codes), "val": list(range(1, len(codes) + 1))})


def identity(s):
    return s["data"]


def regions(s):
    return s["data"].with_columns(pl.col("region_code").replace(REGIONS).alias("region")).drop("region_code")


def lag(n: int):
    return lambda s: s["data"].with_columns(pl.col("val").shift(n).alias(f"lag{n}"))


def lead(n: int):
    return lambda s: s["data"].with_columns(pl.col("val").shift(-n).alias(f"lead{n}"))


def join(on, how: str = "inner", **kw):
    return lambda s: s["left"].join(s["right"], on=on, how=how, **kw)


def single_source(df: pl.DataFrame, **specs) -> dict:
    """One source named ``data``, serving ``df``, with a FilterSpec per column in ``specs``."""
    return {"data": (df, specs)}


def left_and_right_sources(left: pl.DataFrame, right: pl.DataFrame, left_specs: dict, right_specs: dict | None = None) -> dict:
    """Sources named ``left`` and ``right``; the right one uses ``left_specs`` unless ``right_specs`` is given."""
    return {"left": (left, left_specs), "right": (right, left_specs if right_specs is None else right_specs)}


@dataclass(frozen=True)
class CombineCase:
    """Sources, a combine function and the user's filter, and what the result and the sources should be.

    ``sources`` maps a source name to a frame and its FilterSpecs. ``predicate`` may be a list of filters applied in
    turn to the same combined frame. ``time_bounds``, ``allowed_values`` and ``allowed_values_node`` are keyed by source
    name, then column.
    """

    id: str
    _: KW_ONLY
    sources: dict
    combine: Callable
    predicate: Any = None
    compare_with_polars: bool = True
    # defaults to True for the identity combine, which keeps the source's row order
    check_row_order: bool | None = None
    # sources that must receive a filter
    sources_receiving_filter: tuple[str, ...] = ()
    # (lower, upper) bounds of the date or time filter a source receives on a column, as in ``assert_pushed_time_bounds``
    time_bounds: dict[str, dict[str, tuple]] = field(default_factory=dict)
    # the values the first equality or ``is_in`` filter on a column allows
    allowed_values: dict[str, dict[str, set]] = field(default_factory=dict)
    # "eq" or "is_in": the form of that filter
    allowed_values_node: dict[str, dict[str, str]] = field(default_factory=dict)
    # columns that must receive no equality or ``is_in`` filter
    columns_without_value_filter: dict[str, set[str]] = field(default_factory=dict)
    # receives the result and the trackers by source name, and asserts on them
    result_check: Callable | None = None
    # applied to the filtered frame and to the reference
    post_step: Callable | None = None
    combine_kwargs: dict | None = None
    sources_as_kwargs: bool = False


# Checks on the result and the trackers
def pushed_text(source: str, has=(), lacks=()):
    """A result check: the filter ``source`` received, as text, holds every item of ``has`` and none of ``lacks``."""

    def check(_got, trackers):
        pushed = str(trackers[source].last_predicate)
        for text in has:
            assert text in pushed, pushed
        for text in lacks:
            assert text not in pushed, pushed

    return check


def assert_discrete(tracker: PredicateTracker, col: str, values: set, node: str | None = None) -> None:
    """The first equality or ``is_in`` filter on ``col`` allows ``values``; ``node`` (``"eq"`` or ``"is_in"``) also
    checks its form.

    Only the first filter is read: Polars may push a narrower filter of its own for one side of a self-join.
    """
    analyzer = tracker.get_analyzer()
    found = analyzer.find_discrete_filter(col)
    assert analyzer.extract_discrete_values(found) == values, tracker.last_predicate
    if node == "eq":
        assert isinstance(found, BinaryExprNode) and found.op == OperatorType.EQ, found
    elif node == "is_in":
        assert isinstance(found, FunctionNode) and found.function_type == BooleanFunctionType.IS_IN, found


# Frames
TEN_DAYS_FROM_ZERO = daily_frame(10, base=0)
FOURTEEN_DAYS = daily_frame(14)
TEN_DAYS_OTHER_COLUMN = TEN_DAYS_FROM_ZERO.rename({"val": "other"}).with_columns(pl.col("other") + 10)
GROUPS = pl.DataFrame({"date": daily_frame(10)["date"], "group": ["A", "B"] * 5, "val": range(10)})
GROUP_PAIR = (GROUP_FRAME, GROUP_FRAME.rename({"val": "other"}))
INTRADAY = pl.DataFrame(
    {"timestamp": [datetime(2024, 1, day, hour) for day in (1, 2, 3) for hour in (10, 14)], "value": [1, 2, 3, 4, 5, 6]},
)
FIVE_DAYS = daily_frame(5)


def as_date(s):
    return s["main"].with_columns(pl.col("timestamp").cast(pl.Date).alias("data_date"))


def intraday(pred, **spec):
    return {"sources": {"main": (INTRADAY, {"data_date": FilterSpec(source_col="timestamp", **spec)})}, "combine": as_date, "predicate": pred}


def scaled(sources, multiplier):
    return sources["data"].with_columns((pl.col("val") * multiplier).alias("scaled_val"))


def with_mapping(sources, code_to_region):
    return sources["data"].with_columns(pl.col("code").replace(code_to_region).alias("region"))


def multiple_args(sources, multiplier, suffix, constant):
    return sources["data"].with_columns((pl.col("val") * multiplier).alias(f"scaled_{suffix}"), pl.lit(constant).alias("const_col"))


def join_how(sources, join_how):
    return sources["left"].join(sources["right"], on="key", how=join_how)


def primary_with_reference(sources, category_to_code):
    primary = sources["primary"].with_columns(pl.col("category").replace(category_to_code).alias("category_code"))
    return primary.join(sources["reference"], on=["date", "category_code"], how="left")


def keyword_sources_join(left, right):
    return left.join(right, on="date")


def keyword_sources_scaled(data, multiplier, suffix):
    return data.with_columns((pl.col("val") * multiplier).alias(f"scaled_{suffix}"))


def keyword_sources_lag(data):
    return data.with_columns(pl.col("val").shift(3).alias("lag3"))


def keyword_sources_three(prices, volumes, metadata):
    return prices.join(volumes, on="date").join(metadata, on="date")


def keyword_sources_regions(data):
    return regions({"data": data})


def keyword_sources_primary_with_reference(primary, reference, category_mapping):
    return primary_with_reference({"primary": primary, "reference": reference}, category_mapping)


KEY_LEFT = pl.DataFrame({"date": FIVE_DAYS["date"], "key": ["A", "B", "C", "D", "E"], "left_val": [1, 2, 3, 4, 5]})
KEY_RIGHT = pl.DataFrame({"key": ["A", "C", "E"], "right_val": [100, 300, 500]})
PRIMARY = pl.DataFrame({"date": FIVE_DAYS["date"], "category": ["CATEGORY_A"] * 5, "item_id": range(5)})
REFERENCE = pl.DataFrame({"date": FIVE_DAYS["date"], "category_code": ["A"] * 5, "value": [0.05 + i * 0.001 for i in range(5)]})
CATEGORY_TO_CODE = {"CATEGORY_A": "A", "CATEGORY_B": "B"}
PRIMARY_REFERENCE = {
    "primary": (PRIMARY, {"date": FilterSpec(), "category": FilterSpec()}),
    "reference": (
        REFERENCE,
        {"date": FilterSpec(lookback=days(2)), "category": FilterSpec(source_col="category_code", value_mapping=CATEGORY_TO_CODE)},
    ),
}


def three_way(s):
    primary = s["primary"].join(s["mapping"], on="category", how="left")
    return primary.join(s["reference"], on=["date", "category_code"], how="left")


THREE_WAY = {
    "primary": (
        pl.DataFrame({"date": daily_frame(10)["date"], "category": ["CAT_A"] * 10, "item_id": range(10)}),
        {"date": FilterSpec(), "category": FilterSpec()},
    ),
    "reference": (
        pl.DataFrame(
            {
                "date": daily_frame(19, start=make_date(27, 12, 2023))["date"],
                "category_code": ["A"] * 19,
                "value": [0.05 + i * 0.001 for i in range(19)],
            }
        ),
        {"date": FilterSpec(lookback=days(4)), "category": FilterSpec(source_col="category_code", value_mapping={"CAT_A": "A", "CAT_B": "B"})},
    ),
    "mapping": (pl.DataFrame({"category": ["CAT_A", "CAT_B"], "category_code": ["A", "B"]}), {"category": FilterSpec()}),
}


# IntervalFilterSpec: a schedule of [start, end] windows joined to a spine that carries the requested ``date``.
def schedule(starts, ends, vals=None, roots=None) -> pl.DataFrame:
    return pl.DataFrame({"root": roots or ["A"] * len(starts), "start": starts, "end": ends, "val": vals or list(range(1, len(starts) + 1))})


def spine(*dates, roots=None) -> pl.DataFrame:
    return pl.DataFrame({"root": roots or ["A"] * len(dates), "date": list(dates)})


def windows(sched, spine_df, **spec):
    return {"sched": (sched, {"date": IntervalFilterSpec(start_col="start", end_col="end", **spec)}), "spine": (spine_df, {"date": FilterSpec()})}


IN_WINDOW = (pl.col("start") <= pl.col("date")) & (pl.col("date") <= pl.col("end"))


def interval_join(membership=IN_WINDOW):
    """Join the spine to the schedule and keep the rows ``membership`` keeps; ``None`` keeps every pair."""

    def combine(s):
        joined = s["spine"].join(s["sched"], on="root")
        return joined if membership is None else joined.filter(membership)

    return combine


def previous_weekday(day):
    return day - days(3 if day.weekday() == 0 else 1)


SCHEDULE_JAN_11_TO_20 = schedule([make_date(11)], [make_date(20)], [2])
SCHEDULE_JAN_11_TO_20_DATETIME = SCHEDULE_JAN_11_TO_20.with_columns(pl.col("start", "end").cast(pl.Datetime("us")))
SCHEDULE_AUG_AND_SEP = schedule([make_date(1, 8), make_date(31, 8)], [make_date(30, 8), make_date(30, 9)])
SPINE_SEP_1_2_3 = spine(make_date(1, 9), make_date(2, 9), make_date(3, 9))
SEP_MEMBERSHIP = pl.col("date").map_elements(previous_weekday, return_dtype=pl.Date).is_between(pl.col("start"), pl.col("end"))
INTRADAY_SCHEDULE = schedule(
    [datetime(2024, 1, 1, 9), datetime(2024, 1, 15, 15)], [datetime(2024, 1, 15, 9), datetime(2024, 1, 20, 9)], roots=["A", "B"]
)


# Two or more output columns from one source column union their discrete filters.
THREE_GROUPS = pl.DataFrame({"data_date": [make_date(1)] * 3, "group": ["A", "B", "C"], "val": [1.0, 2.0, 3.0]})
PAIR_SPECS = {"data_date": FilterSpec(), "group1": FilterSpec(source_col="group"), "group2": FilterSpec(source_col="group")}


def group_pairs(s):
    left = s["data"].rename({"group": "group1", "val": "val1"})
    right = s["data"].select("data_date", pl.col("group").alias("group2"), pl.col("val").alias("val2"))
    return left.join(right, on="data_date").filter(pl.col("group1") != pl.col("group2"))


def cross_aliases(col: str, *names: str, keep: tuple = ()):
    """Cross-join one source with itself, ``col`` renamed to each of ``names``; ``keep`` columns stay on the first side."""

    def combine(s):
        out = s["data"].select(pl.col(col).alias(names[0]), *keep)
        for name in names[1:]:
            out = out.join(s["data"].select(pl.col(col).alias(name)), how="cross")
        return out

    return combine


def mapped_cross(s):
    code_to_name = {"a": "A", "b": "B", "c": "C"}
    left = s["data"].with_columns(pl.col("code").replace(code_to_name).alias("name1")).drop("code")
    right = s["data"].with_columns(pl.col("code").replace(code_to_name).alias("name2")).drop("code")
    return left.join(right, how="cross")


GROUP_OTHER = pl.DataFrame({"group": ["A", "B", "C"], "other": ["x", "y", "z"]})
GROUP_OTHER_SPECS = {"group1": FilterSpec(source_col="group"), "group2": FilterSpec(source_col="group"), "other": FilterSpec()}
GROUP_OTHER_PRED = (pl.col("group1") == "A") & (pl.col("group2") == "B") & (pl.col("other") == "x")
THREE_SPECS = {f"group{i}": FilterSpec(source_col="group") for i in (1, 2, 3)}
NAME_MAPPING = {"A": "a", "B": "b", "C": "c"}
CODES = pl.DataFrame({"code": ["a", "b", "c"]})
THREE_DATES = pl.DataFrame({"d": [make_date(1), make_date(2), make_date(3)], "val": [1, 2, 3]})

# Rows a pushed predicate could drop through a cast, a time zone, a unit or a null.
_TIMESTAMPS_AROUND_NEW_YEAR = pl.DataFrame(
    {"t": [datetime(2023, 12, 31, 23), datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 0, 0, 500), datetime(2024, 1, 1, 13), None]}
)
_DEC_31_JAN_1_JAN_2 = pl.DataFrame({"t": [make_date(31, 12, 2023), make_date(1), make_date(2)]})
_THREE_DECIMALS = pl.DataFrame({"t": pl.Series([Decimal("1.25"), Decimal("2.50"), Decimal("3.75")], dtype=pl.Decimal(10, 2))})
_TIMESTAMPS_IN_NEW_YORK = _TIMESTAMPS_AROUND_NEW_YEAR.with_columns(pl.col("t").dt.replace_time_zone("UTC").dt.convert_time_zone("America/New_York"))
# A name, a frame, a filter on its column ``t`` and the column's FilterSpec; the combined result must keep every
# matching row.
MATCHES_FILTER = [
    ("date_cast_lt_noon", _TIMESTAMPS_AROUND_NEW_YEAR, pl.col("t").cast(pl.Date) < datetime(2024, 1, 1, 12), FilterSpec()),
    ("dt_date_lt_noon", _TIMESTAMPS_AROUND_NEW_YEAR, pl.col("t").dt.date() < datetime(2024, 1, 1, 12), FilterSpec()),
    ("ms_cast_le_midnight", _TIMESTAMPS_AROUND_NEW_YEAR, pl.col("t").cast(pl.Datetime("ms")) <= datetime(2024, 1, 1), FilterSpec()),
    ("is_in_with_null_nulls_equal", _TIMESTAMPS_AROUND_NEW_YEAR, pl.col("t").is_in([datetime(2024, 1, 1), None], nulls_equal=True), FilterSpec()),
    ("is_in_only_null_nulls_equal", _TIMESTAMPS_AROUND_NEW_YEAR, pl.col("t").is_in([None], nulls_equal=True), FilterSpec()),
    ("date_lt_noon", _DEC_31_JAN_1_JAN_2, pl.col("t") < datetime(2024, 1, 1, 12), FilterSpec()),
    ("date_lt_noon_with_lookback", _DEC_31_JAN_1_JAN_2, pl.col("t") < datetime(2024, 1, 1, 12), FilterSpec(lookback=days(1))),
    (
        "date_between_closed_left",
        _DEC_31_JAN_1_JAN_2,
        pl.col("t").is_between(datetime(2023, 12, 1), datetime(2024, 1, 1, 12), closed="left"),
        FilterSpec(),
    ),
    ("new_york_ge_midnight", _TIMESTAMPS_IN_NEW_YORK, pl.col("t") >= datetime(2024, 1, 1, tzinfo=NEW_YORK), FilterSpec()),
    ("new_york_lt_midnight", _TIMESTAMPS_IN_NEW_YORK, pl.col("t") < datetime(2024, 1, 1, tzinfo=NEW_YORK), FilterSpec()),
    (
        "new_york_between_days",
        _TIMESTAMPS_IN_NEW_YORK,
        pl.col("t").is_between(datetime(2023, 12, 31, tzinfo=NEW_YORK), datetime(2024, 1, 1, tzinfo=NEW_YORK)),
        FilterSpec(),
    ),
    ("new_york_eq_utc_midnight", _TIMESTAMPS_IN_NEW_YORK, pl.col("t") == datetime(2023, 12, 31, 19, tzinfo=NEW_YORK), FilterSpec()),
    ("new_york_is_in_utc_midnight", _TIMESTAMPS_IN_NEW_YORK, pl.col("t").is_in([datetime(2023, 12, 31, 19, tzinfo=NEW_YORK)]), FilterSpec()),
    ("new_york_date_cast_le_dec_31", _TIMESTAMPS_IN_NEW_YORK, pl.col("t").cast(pl.Date) <= make_date(31, 12, 2023), FilterSpec()),
    ("new_york_dt_date_eq_jan_1", _TIMESTAMPS_IN_NEW_YORK, pl.col("t").dt.date() == make_date(1), FilterSpec()),
    (
        "decimal_is_in_list",
        _THREE_DECIMALS,
        pl.col("t").is_in(pl.lit([Decimal("1.25"), Decimal("2.50")], dtype=pl.List(pl.Decimal(10, 2)))),
        FilterSpec(),
    ),
]
# A name, a filter on a float column, and the float dtype the column is cast to.
FLOAT_CASTS = [
    ("float32_eq_2_pow_24", pl.col("x") == 2.0**24, pl.Float32),
    ("float32_eq_2_pow_24_or_5", (pl.col("x") == 2.0**24) | (pl.col("x") == 5.0), pl.Float32),
    ("float32_eq_1e300", pl.col("x") == 1e300, pl.Float32),
    ("float64_eq_2_pow_24", pl.col("x") == 2.0**24, pl.Float64),
    ("float64_eq_1e300", pl.col("x") == 1e300, pl.Float64),
    ("float32_eq_float_max", pl.col("x") == sys.float_info.max, pl.Float32),
    ("float64_eq_float_max", pl.col("x") == sys.float_info.max, pl.Float64),
    ("float64_eq_2_pow_24_or_5", (pl.col("x") == 2.0**24) | (pl.col("x") == 5.0), pl.Float64),
]
NULL_PREDICATES = {
    "is_null": pl.col("a").is_null(),
    "eq_missing": pl.col("a").eq_missing(None),
    "is_in_nulls_equal": pl.col("a").is_in([1, None], nulls_equal=True),
}
NULLABLE = pl.DataFrame({"a": [1, None, 3], "b": [1, None, 3]})


def cast_to(dtype):
    return lambda s: s["data"].with_columns(pl.col("x").cast(dtype))


# A value_mapping with a window must push a window around the mapped values, never only the exact mapped value.
TWENTY_DAYS = pl.DataFrame({"data_date": daily_frame(20)["date"], "v": list(range(20))})


def next_day_rolling(s):
    return s["data"].with_columns(pl.col("v").rolling_sum(3).alias("rolling_sum"), (pl.col("data_date") + pl.duration(days=1)).alias("date"))


def same_day_rolling(s):
    return s["data"].with_columns(pl.col("v").rolling_sum(2).alias("rolling_sum"), pl.col("data_date").alias("date"))


def mapped_self_join(s):
    side = s["data"].with_columns(pl.col("v").rolling_sum(2).alias("rolling_sum"), (pl.col("data_date") + pl.duration(days=1)).alias("d"))
    return side.select(pl.col("d").alias("d1"), pl.col("rolling_sum").alias("rolling_sum_1")).join(
        side.select(pl.col("d").alias("d2"), pl.col("rolling_sum").alias("rolling_sum_2")), how="cross"
    )


def window(**kw) -> FilterSpec:
    return FilterSpec(source_col="data_date", **kw)


def weekday_window(lookback: int) -> FilterSpec:
    return window(lookback=days(lookback), value_mapping=previous_weekday)


def day_of_month_key(s):
    return s["data"].with_columns(pl.col("v").rolling_sum(2).alias("rolling_sum"), pl.col("data_date").dt.day().cast(pl.Int64).add(1).alias("k"))


def unbounded_sharer(s):
    return s["data"].select(pl.col("d").alias("a"), pl.col("v").cum_sum().alias("c")).join(s["data"].select(pl.col("d").alias("b")), how="cross")


TEN_DAYS = pl.DataFrame({"d": daily_frame(10)["date"], "v": list(range(10))})
# 01:30 on the day New York's clocks go back, before and after they go back.
NEW_YORK_REPEATED_HOUR = [datetime(2024, 11, 3, 1, 30, tzinfo=NEW_YORK, fold=0), datetime(2024, 11, 3, 1, 30, tzinfo=NEW_YORK, fold=1)]
NEAR_MAX = [
    datetime(9999, 12, 31, 23, 59, 59, 999998, tzinfo=timezone(timedelta(hours=-5))),
    datetime(9999, 12, 31, 23, 59, 59, 999999, tzinfo=timezone(timedelta(hours=-5))),
]
US_MEMBERS = pl.Series([datetime(2024, 1, 1), datetime(2024, 1, 2)], dtype=pl.Datetime("us", "UTC"))


def five_seconds(unit: str, time_zone: str | None) -> pl.Series:
    """Datetimes one second apart from 2024-01-01 00:00:00, in ``unit`` and ``time_zone``."""
    tz = ZoneInfo(time_zone) if time_zone else None
    return pl.Series([datetime(2024, 1, 1, 0, 0, s, tzinfo=tz) for s in range(5)], dtype=pl.Datetime(unit, time_zone))


def result_values_are(values: set):
    """A result check: the ``val`` column of the result holds exactly ``values``."""

    def check(got, _trackers):
        assert set(got["val"]) == values, got

    return check


SCENARIOS = [
    # Propagation through a join
    CombineCase(
        "no_filters",
        sources=left_and_right_sources(pl.DataFrame({"id": [1, 2], "val": [10, 20]}), pl.DataFrame({"id": [1, 2], "other": [100, 200]}), {}),
        combine=join("id"),
    ),
    CombineCase(
        "simple_date_filter",
        sources=left_and_right_sources(TEN_DAYS_FROM_ZERO, TEN_DAYS_OTHER_COLUMN, {"date": FilterSpec()}),
        combine=join("date", maintain_order="left"),
        predicate=pl.col("date") >= make_date(5),
        check_row_order=True,
    ),
    CombineCase(
        "dt_date_filter_prunes_datetime_source",
        sources=single_source(daily_frame(90, dtype=pl.Datetime("us")), date=FilterSpec()),
        combine=identity,
        predicate=(pl.col("date").dt.date() >= make_date(1)) & (pl.col("date").dt.date() <= make_date(31)),
        sources_receiving_filter=("data",),
    ),
    CombineCase(
        "simple_equality_filter",
        sources=left_and_right_sources(*GROUP_PAIR, {"group": FilterSpec()}),
        combine=join("group"),
        predicate=pl.col("group") == "B",
    ),
    CombineCase(
        "is_in_filter",
        sources=left_and_right_sources(*GROUP_PAIR, {"group": FilterSpec()}),
        combine=join("group"),
        predicate=pl.col("group").is_in(["A", "C"]),
    ),
    # Lookback and lookahead
    CombineCase(
        "lookback_with_date_equality",
        sources=single_source(FOURTEEN_DAYS, date=FilterSpec(lookback=days(4))),
        combine=lag(3),
        predicate=pl.col("date") == make_date(5),
    ),
    CombineCase(
        "lookahead_with_date_equality",
        sources=single_source(FOURTEEN_DAYS, date=FilterSpec(lookahead=days(4))),
        combine=lead(3),
        predicate=pl.col("date") == make_date(5),
    ),
    CombineCase(
        "lookback_and_lookahead_with_date_equality",
        sources=single_source(FOURTEEN_DAYS, date=FilterSpec(lookback=days(3), lookahead=days(3))),
        combine=lambda s: s["data"].with_columns(pl.col("val").shift(2).alias("lag2"), pl.col("val").shift(-2).alias("lead2")),
        predicate=pl.col("date") == make_date(7),
    ),
    CombineCase(
        "lookback_expands_date_range",
        sources=left_and_right_sources(
            daily_frame(10, base=100), TEN_DAYS_OTHER_COLUMN, {"date": FilterSpec(lookback=days(3))}, {"date": FilterSpec()}
        ),
        combine=join("date", maintain_order="left"),
        predicate=pl.col("date") >= make_date(5),
        check_row_order=True,
        sources_receiving_filter=("left",),
    ),
    CombineCase(
        "lookback_for_rolling_computation",
        sources=single_source(TEN_DAYS_FROM_ZERO.with_columns(pl.col("val").cast(pl.Float64)), date=FilterSpec(lookback=days(3))),
        combine=lambda s: s["data"].sort("date").with_columns(pl.col("val").rolling_sum(window_size=3, min_samples=1).alias("rolling_sum")),
        predicate=pl.col("date") >= make_date(5),
    ),
    CombineCase(
        "lookahead_expands_date_range",
        sources=single_source(daily_frame(10, base=100), date=FilterSpec(lookahead=days(2))),
        combine=identity,
        predicate=pl.col("date") <= make_date(5),
    ),
    CombineCase(
        "combined_lookback_lookahead",
        sources=single_source(daily_frame(10, base=100), date=FilterSpec(lookback=days(2), lookahead=days(2))),
        combine=identity,
        predicate=pl.col("date").is_between(make_date(4), make_date(7)),
    ),
    # Pushed bounds and values
    CombineCase(
        "date_gte_filter_structure",
        sources=single_source(TEN_DAYS_FROM_ZERO, date=FilterSpec()),
        combine=identity,
        predicate=pl.col("date") >= make_date(5),
        time_bounds={"data": {"date": (make_date(5), ...)}},
    ),
    CombineCase(
        "date_between_filter_structure",
        sources=single_source(TEN_DAYS_FROM_ZERO, date=FilterSpec()),
        combine=identity,
        predicate=pl.col("date").is_between(make_date(3), make_date(7)),
        time_bounds={"data": {"date": (make_date(3), make_date(7))}},
    ),
    CombineCase(
        "lookback_expands_date_range_correctly",
        sources=single_source(TEN_DAYS_FROM_ZERO, date=FilterSpec(lookback=days(3))),
        combine=identity,
        predicate=pl.col("date") >= make_date(5),
        time_bounds={"data": {"date": (make_date(2), ...)}},
    ),
    CombineCase(
        "lookback_with_is_between_and_complex_combine",
        sources=single_source(FOURTEEN_DAYS, date=FilterSpec(lookback=days(4))),
        combine=lag(3),
        predicate=pl.col("date").is_between(make_date(5), make_date(10)),
        time_bounds={"data": {"date": (make_date(1), make_date(10))}},
    ),
    # Polars may merge the widened filter into the user's one when the combine is the identity.
    CombineCase(
        "lookback_optimization_with_identity_combine",
        sources=single_source(FOURTEEN_DAYS, date=FilterSpec(lookback=days(4))),
        combine=identity,
        predicate=pl.col("date").is_between(make_date(5), make_date(10)),
        time_bounds={"data": {"date": ((make_date(1), make_date(5)), make_date(10))}},
    ),
    CombineCase(
        "lookahead_expands_upper_bound",
        sources=single_source(FOURTEEN_DAYS, date=FilterSpec(lookahead=days(3))),
        combine=identity,
        predicate=pl.col("date") <= make_date(7),
        time_bounds={"data": {"date": (..., make_date(10))}},
    ),
    CombineCase(
        "discrete_equality_filter_structure",
        sources=single_source(GROUP_FRAME, group=FilterSpec()),
        combine=identity,
        predicate=pl.col("group") == "B",
        allowed_values={"data": {"group": {"B"}}},
        allowed_values_node={"data": {"group": "eq"}},
    ),
    CombineCase(
        "discrete_is_in_filter_structure",
        sources=single_source(GROUP_FRAME, group=FilterSpec()),
        combine=identity,
        predicate=pl.col("group").is_in(["A", "C"]),
        allowed_values={"data": {"group": {"A", "C"}}},
        allowed_values_node={"data": {"group": "is_in"}},
    ),
    # Value mapping and source column renames
    CombineCase(
        "dict_value_mapping",
        sources=single_source(region_frame("NA", "EU"), region=FilterSpec(source_col="region_code", value_mapping=to_codes("NA", "EU"))),
        combine=regions,
        predicate=pl.col("region") == "NORTH_AMERICA",
        allowed_values={"data": {"region_code": {"NA"}}},
        allowed_values_node={"data": {"region_code": "eq"}},
    ),
    CombineCase(
        "callable_value_mapping",
        sources=single_source(pl.DataFrame({"name": ["foo", "bar", "baz"], "val": [1, 2, 3]}), name=FilterSpec(value_mapping=str.lower)),
        combine=lambda s: s["data"].with_columns(pl.col("name").str.to_uppercase()),
        predicate=pl.col("name") == "FOO",
    ),
    CombineCase(
        "is_in_with_value_mapping",
        sources=single_source(
            region_frame("NA", "EU", "APAC"), region=FilterSpec(source_col="region_code", value_mapping=to_codes("NA", "EU", "APAC"))
        ),
        combine=regions,
        predicate=pl.col("region").is_in(["NORTH_AMERICA", "ASIA_PACIFIC"]),
        allowed_values={"data": {"region_code": {"NA", "APAC"}}},
        allowed_values_node={"data": {"region_code": "is_in"}},
    ),
    # An unmapped value is not pushed: the source supplies every row and the user's filter decides.
    CombineCase(
        "unmapped_value_not_pushed_down",
        sources=single_source(
            region_frame("NA", "EU", "APAC", "LATAM"), region=FilterSpec(source_col="region_code", value_mapping=to_codes("NA", "EU"))
        ),
        combine=regions,
        predicate=pl.col("region") == "LATIN_AMERICA",
    ),
    CombineCase(
        "mixed_mapped_and_unmapped_values_in_is_in",
        sources=single_source(
            region_frame("NA", "EU", "APAC", "LATAM"), region=FilterSpec(source_col="region_code", value_mapping=to_codes("NA", "EU"))
        ),
        combine=regions,
        predicate=pl.col("region").is_in(["NORTH_AMERICA", "LATIN_AMERICA"]),
    ),
    CombineCase(
        "different_source_column_name",
        sources=single_source(TEN_DAYS_FROM_ZERO.rename({"date": "DataDate"}), date=FilterSpec(source_col="DataDate")),
        combine=lambda s: s["data"].rename({"DataDate": "date"}),
        predicate=pl.col("date") == make_date(2),
        allowed_values={"data": {"DataDate": {make_date(2)}}},
        allowed_values_node={"data": {"DataDate": "eq"}},
    ),
    # Several sources with their own specs
    CombineCase(
        "different_lookback_per_source",
        sources={
            "source1": (TEN_DAYS_FROM_ZERO.rename({"val": "val1"}), {"date": FilterSpec(lookback=days(5))}),
            "source2": (TEN_DAYS_OTHER_COLUMN, {"date": FilterSpec(lookback=days(2))}),
        },
        combine=lambda s: s["source1"].join(s["source2"], on="date", maintain_order="right"),
        predicate=pl.col("date") >= make_date(7),
        check_row_order=True,
        time_bounds={"source1": {"date": (make_date(2), ...)}, "source2": {"date": (make_date(5), ...)}},
    ),
    CombineCase(
        "mixed_filter_types_per_source",
        sources={
            "left": (GROUPS.head(5), {"date": FilterSpec(lookback=days(2)), "group": FilterSpec()}),
            "right": (GROUPS.head(5).rename({"val": "val2"}), {"date": FilterSpec(), "group": FilterSpec()}),
        },
        combine=join(["date", "group"], maintain_order="left"),
        predicate=(pl.col("date") >= make_date(3)) & (pl.col("group") == "A"),
        check_row_order=True,
    ),
    CombineCase(
        "pushdown_combine_join_with_mapping_and_lookback",
        sources=THREE_WAY,
        combine=three_way,
        predicate=pl.col("date").is_between(make_date(5), make_date(7)) & (pl.col("category") == "CAT_A"),
    ),
    # Edge cases
    CombineCase(
        "empty_result", sources=single_source(daily_frame(1), date=FilterSpec()), combine=identity, predicate=pl.col("date") == make_date(31, 12)
    ),
    CombineCase(
        "no_filter_specs", sources=single_source(pl.DataFrame({"id": [1, 2], "val": [10, 20]})), combine=identity, predicate=pl.col("id") == 1
    ),
    CombineCase(
        "source_col_not_in_schema",
        sources=single_source(pl.DataFrame({"id": [1, 2], "val": [10, 20]}), nonexistent=FilterSpec(source_col="also_nonexistent")),
        combine=identity,
    ),
    CombineCase(
        "multiple_filters_combined",
        sources=single_source(GROUPS, date=FilterSpec(), group=FilterSpec()),
        combine=identity,
        predicate=(pl.col("date") >= make_date(5)) & (pl.col("group") == "A"),
    ),
    CombineCase(
        "datetime_column",
        sources=single_source(
            pl.DataFrame({"ts": [datetime(2024, 1, 1, h) for h in range(24)], "val": range(24)}), ts=FilterSpec(lookback=timedelta(hours=3))
        ),
        combine=identity,
        predicate=pl.col("ts") >= datetime(2024, 1, 1, 12),
    ),
    CombineCase(
        "column_selection_with_columns",
        sources=single_source(pl.DataFrame({"date": [make_date(1), make_date(2)], "a": [1, 2], "b": [10, 20], "c": [100, 200]}), date=FilterSpec()),
        combine=identity,
        post_step=lambda lf: lf.select("date", "a"),
    ),
    CombineCase("row_limit", sources=single_source(TEN_DAYS_FROM_ZERO, date=FilterSpec()), combine=identity, post_step=lambda lf: lf.head(3)),
    CombineCase(
        "filter_on_column_without_spec",
        sources=single_source(FIVE_DAYS.with_columns(unspecified_col=pl.Series(list("ABCDE"))), date=FilterSpec()),
        combine=identity,
        predicate=pl.col("unspecified_col") == "C",
    ),
    CombineCase(
        "complex_predicate_with_or",
        sources=single_source(TEN_DAYS_FROM_ZERO, date=FilterSpec()),
        combine=identity,
        predicate=(pl.col("date") == make_date(3)) | (pl.col("date") == make_date(7)),
    ),
    CombineCase(
        "nested_filters_with_and",
        sources=single_source(GROUPS, date=FilterSpec(lookback=days(2)), group=FilterSpec()),
        combine=identity,
        predicate=(pl.col("date") >= make_date(5)) & (pl.col("group") == "A"),
    ),
    CombineCase(
        "filter_with_null_values",
        sources=single_source(pl.DataFrame({"date": [make_date(1), make_date(2), None, make_date(4)], "val": [1, 2, 3, 4]}), date=FilterSpec()),
        combine=identity,
        predicate=pl.col("date") >= make_date(2),
    ),
    CombineCase(
        "multiple_collects",
        sources=single_source(TEN_DAYS_FROM_ZERO, date=FilterSpec()),
        combine=identity,
        predicate=[pl.col("date") >= make_date(5), pl.col("date") <= make_date(3)],
    ),
    CombineCase("empty_source", sources=single_source(daily_frame(0), date=FilterSpec()), combine=identity, predicate=pl.col("date") >= make_date(1)),
    CombineCase(
        "very_large_lookback",
        sources=single_source(TEN_DAYS_FROM_ZERO, date=FilterSpec(lookback=days(365))),
        combine=identity,
        predicate=pl.col("date") >= make_date(5),
    ),
    CombineCase(
        "combine_with_additional_columns",
        sources=single_source(FIVE_DAYS, date=FilterSpec()),
        combine=lambda s: s["data"].with_columns((pl.col("val") * 2).alias("val_doubled"), pl.lit("constant").alias("const_col")),
        predicate=pl.col("date") >= make_date(3),
    ),
    CombineCase(
        "combine_drops_columns",
        sources=single_source(FIVE_DAYS.with_columns(drop_col=pl.Series(list("abcde"))), date=FilterSpec()),
        combine=lambda s: s["data"].drop("drop_col"),
        predicate=pl.col("date") >= make_date(3),
    ),
    # combine_kwargs and sources_as_kwargs
    CombineCase(
        "combine_kwargs_basic",
        sources=single_source(FIVE_DAYS, date=FilterSpec()),
        combine=scaled,
        predicate=pl.col("date") >= make_date(3),
        combine_kwargs={"multiplier": 10},
    ),
    CombineCase(
        "combine_kwargs_with_mapping_dict",
        sources=single_source(pl.DataFrame({"date": [make_date(1), make_date(2)], "code": ["US", "CN"], "val": [100, 200]}), date=FilterSpec()),
        combine=with_mapping,
        combine_kwargs={"code_to_region": {"US": "North America", "CN": "Asia"}},
    ),
    CombineCase(
        "combine_kwargs_multiple_args",
        sources=single_source(FIVE_DAYS, date=FilterSpec()),
        combine=multiple_args,
        combine_kwargs={"multiplier": 2, "suffix": "doubled", "constant": "hello"},
        check_row_order=True,
    ),
    CombineCase("combine_kwargs_with_none", sources=single_source(FIVE_DAYS, date=FilterSpec()), combine=identity, combine_kwargs=None),
    CombineCase("combine_kwargs_empty_dict", sources=single_source(FIVE_DAYS, date=FilterSpec()), combine=identity, combine_kwargs={}),
    CombineCase(
        "combine_kwargs_complex_join_pattern",
        sources=PRIMARY_REFERENCE,
        combine=primary_with_reference,
        predicate=(pl.col("date") >= make_date(2)) & (pl.col("category") == "CATEGORY_A"),
        combine_kwargs={"category_to_code": CATEGORY_TO_CODE},
    ),
    CombineCase(
        "sources_as_kwargs_basic",
        sources=left_and_right_sources(FIVE_DAYS.rename({"val": "left_val"}), FIVE_DAYS.rename({"val": "right_val"}), {"date": FilterSpec()}),
        combine=keyword_sources_join,
        predicate=pl.col("date") >= make_date(3),
        sources_as_kwargs=True,
    ),
    CombineCase(
        "sources_as_kwargs_with_combine_kwargs",
        sources=single_source(FIVE_DAYS, date=FilterSpec()),
        combine=keyword_sources_scaled,
        combine_kwargs={"multiplier": 10, "suffix": "x10"},
        sources_as_kwargs=True,
        check_row_order=True,
    ),
    CombineCase(
        "sources_as_kwargs_with_lookback",
        sources=single_source(FOURTEEN_DAYS, date=FilterSpec(lookback=days(4))),
        combine=keyword_sources_lag,
        predicate=pl.col("date") == make_date(5),
        sources_as_kwargs=True,
    ),
    CombineCase(
        "sources_as_kwargs_multiple_sources",
        sources={
            "prices": (FIVE_DAYS.rename({"val": "price"}), {"date": FilterSpec()}),
            "volumes": (FIVE_DAYS.rename({"val": "volume"}), {"date": FilterSpec()}),
            "metadata": (FIVE_DAYS.with_columns(category=pl.Series(list("AABBA"))).drop("val"), {"date": FilterSpec()}),
        },
        combine=keyword_sources_three,
        predicate=pl.col("date") >= make_date(3),
        sources_as_kwargs=True,
    ),
    CombineCase("sources_as_kwargs_false_is_default", sources=single_source(FIVE_DAYS, date=FilterSpec()), combine=identity),
    CombineCase(
        "sources_as_kwargs_with_value_mapping",
        sources=single_source(region_frame("NA", "EU"), region=FilterSpec(source_col="region_code", value_mapping=to_codes("NA", "EU"))),
        combine=keyword_sources_regions,
        predicate=pl.col("region") == "NORTH_AMERICA",
        sources_as_kwargs=True,
    ),
    CombineCase(
        "sources_as_kwargs_complex_join_pattern",
        sources={
            "primary": (PRIMARY.with_columns(category=pl.lit("A")), {"date": FilterSpec(), "category": FilterSpec()}),
            "reference": (
                REFERENCE,
                {"date": FilterSpec(lookback=days(2)), "category": FilterSpec(source_col="category_code", value_mapping={"A": "A", "B": "B"})},
            ),
        },
        combine=keyword_sources_primary_with_reference,
        predicate=(pl.col("date") >= make_date(2)) & (pl.col("category") == "A"),
        combine_kwargs={"category_mapping": {"A": "A", "B": "B"}},
        sources_as_kwargs=True,
    ),
    # A Date filter linked to a Datetime source keeps every intraday row of the requested days.
    CombineCase("eq_covers_full_day", **intraday(pl.col("data_date") == make_date(2))),
    CombineCase("eq_with_lookback_covers_full_day", **intraday(pl.col("data_date") == make_date(2), lookback=days(1))),
    # A sub-day lookback from a Date literal still crosses the day boundary, and the upper bound covers the whole day.
    CombineCase(
        "eq_with_subday_lookback_fetches_prior_evening",
        sources={
            "main": (
                pl.DataFrame(
                    {
                        "timestamp": [datetime(2024, 1, 1, 20), datetime(2024, 1, 2, 4), datetime(2024, 1, 2, 10), datetime(2024, 1, 3, 4)],
                        "value": [10, 20, 30, 40],
                    }
                ),
                {"data_date": FilterSpec(source_col="timestamp", lookback=timedelta(hours=18))},
            )
        },
        combine=as_date,
        predicate=pl.col("data_date") == make_date(2),
        result_check=pushed_text("main", has=("2024-01-01 06:00:00", "2024-01-03 00:00:00")),
    ),
    CombineCase("le_covers_full_day", **intraday(pl.col("data_date") <= make_date(2))),
    CombineCase("ge_starts_at_midnight", **intraday(pl.col("data_date") >= make_date(2))),
    CombineCase("lt_excludes_full_day", **intraday(pl.col("data_date") < make_date(2))),
    CombineCase("gt_starts_next_day", **intraday(pl.col("data_date") > make_date(2))),
    CombineCase("is_in_covers_full_days", **intraday(pl.col("data_date").is_in([make_date(2), make_date(3)]))),
    CombineCase(
        "datetime_filter_on_datetime_source_unchanged",
        sources={"main": (INTRADAY, {"timestamp": FilterSpec()})},
        combine=lambda s: s["main"],
        predicate=pl.col("timestamp") >= datetime(2024, 1, 2, 12),
        check_row_order=True,
    ),
    # IntervalFilterSpec pushes start <= hi and end >= lo onto the schedule; membership is the combine's.
    CombineCase(
        "selects_covering_windows",
        sources=windows(schedule([make_date(1), make_date(11), make_date(21)], [make_date(10), make_date(20), make_date(30)]), spine(make_date(15))),
        combine=interval_join(),
        predicate=pl.col("date") == make_date(15),
    ),
    CombineCase(
        "long_window_kept",
        sources=windows(schedule([make_date(1, 1, 2020)], [make_date(1, 1, 2030)]), spine(make_date(15))),
        combine=interval_join(),
        predicate=pl.col("date") == make_date(15),
    ),
    CombineCase(
        "bounds_pushed_to_source",
        sources=windows(SCHEDULE_JAN_11_TO_20, spine(make_date(12), make_date(15))),
        combine=interval_join(),
        predicate=pl.col("date").is_between(make_date(12), make_date(15)),
        time_bounds={"sched": {"end": (make_date(12), ...), "start": (..., make_date(15))}},
    ),
    # A one-day timedelta value_mapping moves both bounds the schedule source receives back by a day.
    CombineCase(
        "interval_timedelta_mapping_shifts_bounds-open",
        sources=windows(SCHEDULE_JAN_11_TO_20, spine(make_date(13)), value_mapping=days(-1)),
        combine=interval_join(None),
        predicate=pl.col("date") >= make_date(12),
        result_check=pushed_text("sched", has=('(col("end")) >= (2024-01-11)',), lacks=('col("start")',)),
    ),
    CombineCase(
        "interval_timedelta_mapping_shifts_bounds-over_cap",
        sources=windows(SCHEDULE_JAN_11_TO_20, spine(make_date(13)), value_mapping=days(-1)),
        combine=interval_join(None),
        predicate=pl.col("date").is_between(make_date(1, 1, 1990), make_date(15)),
        result_check=pushed_text("sched", has=('(col("end")) >= (1989-12-31)', '(col("start")) <= (2024-01-14)')),
    ),
    CombineCase(
        "interval_timedelta_mapping_shifts_bounds-datetime",
        sources=windows(SCHEDULE_JAN_11_TO_20_DATETIME, spine(datetime(2024, 1, 13)), value_mapping=days(-1)),
        combine=interval_join(None),
        predicate=pl.col("date").is_between(datetime(2024, 1, 12, 6), datetime(2024, 1, 15, 6)),
        result_check=pushed_text("sched", has=('(col("end")) >= (2024-01-11 06:00:00)', '(col("start")) <= (2024-01-14 06:00:00)')),
    ),
    # A partial day shifts a Date request outward to whole days.
    *[
        CombineCase(
            f"interval_timedelta_mapping_rounds_dates_outward-{hours}h",
            sources=windows(SCHEDULE_JAN_11_TO_20, spine(make_date(13)), value_mapping=timedelta(hours=hours)),
            combine=interval_join(),
            predicate=pl.col("date") == make_date(13),
            result_check=pushed_text("sched", has=(f'(col("end")) >= ({end_lower})', f'(col("start")) <= ({start_upper})')),
        )
        for hours, end_lower, start_upper in [(12, make_date(13), make_date(14)), (-12, make_date(12), make_date(13))]
    ],
    # Rounding a shift of nearly the whole timedelta range out to whole days overflows; the bound is left unbounded.
    *[
        CombineCase(
            f"interval_timedelta_mapping_out_of_range-{name}",
            sources=windows(SCHEDULE_JAN_11_TO_20, spine(make_date(13)), value_mapping=shift),
            combine=interval_join(),
            predicate=pl.col("date") == make_date(13),
        )
        for name, shift in [("max", timedelta.max), ("min", timedelta.min)]
    ],
    # Windows closed on ``closed`` sides, joined to a requested day: the window starting that day (val 1) is kept if its
    # start is closed, and the window ending that day (val 2) if its end is closed.
    *[
        CombineCase(
            f"date_endpoints-{closed}",
            sources=windows(
                schedule([make_date(15), make_date(1)], [make_date(20), make_date(15)], roots=["A", "B"]),
                spine(make_date(15), make_date(15), roots=["A", "B"]),
                closed=closed,
            ),
            combine=interval_join(pl.lit(True)),
            predicate=pl.col("date") == make_date(15),
            compare_with_polars=False,
            result_check=result_values_are(kept),
        )
        for closed, kept in [("both", {1, 2}), ("left", {1}), ("right", {2}), ("none", set())]
    ],
    # A Date request on the 15th keeps windows that end or start intraday on the 15th, under every closed mode.
    *[
        CombineCase(
            f"datetime_intraday_boundary_not_pruned-{closed}",
            sources=windows(INTRADAY_SCHEDULE, spine(make_date(15), make_date(15), roots=["A", "B"]), closed=closed),
            combine=interval_join(None),
            predicate=pl.col("date") == make_date(15),
        )
        for closed in CLOSED_OPTIONS
    ],
    # Disjoint requested ranges collapse to their hull: a window in the gap survives the pushdown.
    CombineCase(
        "disjoint_request_collapses_to_hull",
        sources=windows(schedule([make_date(20)], [make_date(25)]), spine(make_date(22))),
        combine=interval_join(None),
        predicate=pl.col("date").is_between(make_date(1), make_date(5)) | pl.col("date").is_between(make_date(30), make_date(31)),
        time_bounds={"sched": {"end": (make_date(1), ...)}},
    ),
    # Monday maps to the previous Friday, other days to the day before: Sun 1 Sep maps after Mon 2 Sep.
    CombineCase(
        "non_monotone_value_mapping_keeps_interior_days",
        sources=windows(SCHEDULE_AUG_AND_SEP, SPINE_SEP_1_2_3, value_mapping=previous_weekday),
        combine=interval_join(SEP_MEMBERSHIP),
        predicate=pl.col("date").is_between(make_date(1, 9), make_date(3, 9)),
    ),
    # An interior day mapped to date.max leaves the upper bound on Datetime starts open.
    CombineCase(
        "value_mapping_to_last_date_leaves_datetime_start_unbounded",
        sources=windows(
            pl.DataFrame({"root": ["A"], "start": [datetime(2024, 1, 1)], "end": [datetime(9999, 12, 31)], "val": [1]}),
            spine(make_date(1), make_date(2), make_date(3)),
            value_mapping=lambda day: date.max if day == make_date(2) else make_date(1),
        ),
        combine=interval_join(None),
        predicate=pl.col("date").is_between(make_date(1), make_date(3)),
        result_check=pushed_text("sched", has=('col("end")',), lacks=('col("start")',)),
    ),
    CombineCase(
        "value_mapping_to_none_pushes_nothing",
        sources=windows(
            schedule([make_date(1), make_date(3)], [make_date(1), make_date(3)], [1, 3]),
            spine(make_date(1), make_date(2), make_date(3)),
            value_mapping=lambda day: None if day == make_date(2) else day,
        ),
        combine=interval_join(),
        predicate=pl.col("date").is_between(make_date(1), make_date(3)),
    ),
    CombineCase(
        "value_mapping_that_raises_pushes_nothing",
        sources=windows(
            schedule([make_date(1), make_date(3)], [make_date(1), make_date(3)], [1, 3]),
            spine(make_date(1), make_date(2), make_date(3)),
            value_mapping={make_date(1): make_date(1), make_date(3): make_date(3)}.__getitem__,
        ),
        combine=interval_join(),
        predicate=pl.col("date").is_between(make_date(1), make_date(3)),
    ),
    CombineCase(
        "interval_filter_with_tz_aware_endpoints",
        sources={
            "spine": (pl.DataFrame({"d": [datetime(2024, 1, 2, tzinfo=UTC_ZONE)]}), {"d": FilterSpec()}),
            "sched": (
                pl.DataFrame({"start": [datetime(2024, 1, 1, tzinfo=UTC_ZONE)], "end": [datetime(2024, 1, 3, tzinfo=UTC_ZONE)], "x": [1]}),
                {"d": IntervalFilterSpec("start", "end")},
            ),
        },
        combine=lambda s: s["spine"].join_where(s["sched"], pl.col("d") >= pl.col("start"), pl.col("d") <= pl.col("end")).select("d", "x"),
        predicate=pl.col("d") == datetime(2024, 1, 2, tzinfo=UTC_ZONE),
    ),
    # Shared source columns: a source row may serve any output column mapped to it, so the pushed values are the union.
    CombineCase(
        "lone_side_filter_returns_all_partners",
        sources=single_source(THREE_GROUPS, **PAIR_SPECS),
        combine=group_pairs,
        predicate=pl.col("group1") == "A",
    ),
    CombineCase(
        "both_sides_filter_unions_to_is_in",
        sources=single_source(THREE_GROUPS, **PAIR_SPECS),
        combine=group_pairs,
        predicate=(pl.col("group1") == "A") & (pl.col("group2") == "B"),
        allowed_values={"data": {"group": {"A", "B"}}},
    ),
    CombineCase(
        "is_in_both_sides_unions",
        sources=single_source(THREE_GROUPS, **PAIR_SPECS),
        combine=group_pairs,
        predicate=pl.col("group1").is_in(["A", "B"]) & pl.col("group2").is_in(["B", "C"]),
        allowed_values={"data": {"group": {"A", "B", "C"}}},
    ),
    CombineCase(
        "three_shared_columns_union-all_constrained",
        sources=single_source(THREE_GROUPS.select("group"), **THREE_SPECS),
        combine=cross_aliases("group", "group1", "group2", "group3"),
        predicate=(pl.col("group1") == "A") & (pl.col("group2") == "B") & (pl.col("group3") == "C"),
        allowed_values={"data": {"group": {"A", "B", "C"}}},
    ),
    CombineCase(
        "three_shared_columns_union-one_unconstrained",
        sources=single_source(THREE_GROUPS.select("group"), **THREE_SPECS),
        combine=cross_aliases("group", "group1", "group2", "group3"),
        predicate=(pl.col("group1") == "A") & (pl.col("group2") == "B"),
        columns_without_value_filter={"data": {"group"}},
    ),
    CombineCase(
        "value_mapping_union_of_mapped_values",
        sources=single_source(
            CODES, name1=FilterSpec(source_col="code", value_mapping=NAME_MAPPING), name2=FilterSpec(source_col="code", value_mapping=NAME_MAPPING)
        ),
        combine=cross_aliases("code", "name1", "name2"),
        predicate=(pl.col("name1") == "A") & (pl.col("name2") == "B"),
        allowed_values={"data": {"code": {"a", "b"}}},
    ),
    # name2 has no mapping for "C", so the union is skipped and the user's filter decides.
    CombineCase(
        "value_mapping_unmapped_member_still_correct",
        sources=single_source(
            CODES,
            name1=FilterSpec(source_col="code", value_mapping=NAME_MAPPING),
            name2=FilterSpec(source_col="code", value_mapping={"A": "a", "B": "b"}),
        ),
        combine=mapped_cross,
        predicate=(pl.col("name1") == "A") & (pl.col("name2") == "C"),
    ),
    # Only the pushed filters are checked: the filter on ``other`` reaches the whole source, so the group2 side loses "B"
    # and the result is empty where plain Polars returns one row.
    CombineCase(
        "non_shared_column_in_same_source_unaffected",
        sources=single_source(GROUP_OTHER, **GROUP_OTHER_SPECS),
        combine=cross_aliases("group", "group1", "group2", keep=("other",)),
        predicate=GROUP_OTHER_PRED,
        compare_with_polars=False,
        allowed_values={"data": {"group": {"A", "B"}, "other": {"x"}}},
    ),
    # A lookback on d1 suppresses its discrete membership, so d2 is the lone member of the union on d. The result is not
    # checked: the lookback range and d == Jan 2 intersect on the shared column and starve d1 == Jan 3 (a documented
    # limitation of mixing temporal and discrete specs on one source column).
    CombineCase(
        "temporal_suppressed_column_excluded_from_union",
        sources=single_source(THREE_DATES, d1=FilterSpec(source_col="d", lookback=days(2)), d2=FilterSpec(source_col="d")),
        combine=cross_aliases("d", "d1", "d2"),
        predicate=(pl.col("d1") == make_date(3)) & (pl.col("d2") == make_date(2)),
        compare_with_polars=False,
        time_bounds={"data": {"d": (..., ...)}},
        allowed_values={"data": {"d": {make_date(2)}}},
    ),
    # Unorderable members (None and strings) must not be sorted into the pushed is_in.
    CombineCase(
        "union_with_unorderable_values_does_not_crash",
        sources=single_source(
            pl.DataFrame({"group": ["A", "B", None]}), group1=FilterSpec(source_col="group"), group2=FilterSpec(source_col="group")
        ),
        combine=cross_aliases("group", "group1", "group2"),
        predicate=(pl.col("group1").is_null() | (pl.col("group1") == "A")) & (pl.col("group2") == "B"),
    ),
    # Casts: 257 wraps to 1 under Int8, and 2**24 + 1 rounds to 2.0**24 under Float32.
    CombineCase(
        "narrowing_cast_does_not_drop_rows",
        sources=single_source(pl.DataFrame({"a": [1, 2, 257]}), a=FilterSpec()),
        combine=identity,
        predicate=pl.col("a").cast(pl.Int8, wrap_numerical=True) == 1,
    ),
    *[
        CombineCase(
            f"float_cast_in_combine_keeps_rounded_rows-{name}",
            sources=single_source(pl.DataFrame({"x": [2**24 + 1, 5, None]}), x=FilterSpec()),
            combine=cast_to(dtype),
            predicate=pred,
        )
        for name, pred, dtype in FLOAT_CASTS
    ],
    # A Float64 value next to a Float32 value rounds to it under the cast.
    *[
        CombineCase(
            f"float32_cast_of_float64_keeps_rounded_rows-{name}",
            sources=single_source(pl.DataFrame({"x": [2.0**24 + 1, 0.1, None]}), x=FilterSpec()),
            combine=cast_to(pl.Float32),
            predicate=pred,
        )
        for name, pred in [
            ("eq_2_pow_24", pl.col("x") == 2.0**24),
            ("eq_float32_of_0_1", pl.col("x") == float(np.float32(0.1))),
            ("is_in_2_pow_24", pl.col("x").is_in([2.0**24, -1.0])),
        ]
    ],
    # A value set containing None keeps null rows, also when two output columns share a source column.
    *[
        CombineCase(
            f"discrete_filter_keeps_null_rows-{name}",
            sources={"single": (NULLABLE, {"a": FilterSpec()}), "shared": (NULLABLE, {"a": FilterSpec(), "b": FilterSpec(source_col="a")})},
            combine=lambda s: pl.concat([s["single"], s["shared"]]),
            predicate=pred,
        )
        for name, pred in NULL_PREDICATES.items()
    ],
    *[
        CombineCase(f"matches_filter-{name}", sources=single_source(df, t=spec), combine=identity, predicate=pred)
        for name, df, pred, spec in MATCHES_FILTER
    ],
    *[
        CombineCase(
            f"temporal_filter_applies_value_mapping-{name}",
            sources=single_source(pl.DataFrame({"d": [make_date(1), make_date(2)]}), d=FilterSpec(value_mapping=mapping)),
            combine=lambda s: s["data"].with_columns(pl.col("d") + days(1)),
            predicate=pl.col("d") == make_date(2),
            check_row_order=True,
        )
        for name, mapping in [("dict", {make_date(2): make_date(1)}), ("callable", lambda day: day - days(1))]
    ],
    # Filtering the right side of a left join would null-extend rows that then satisfy the predicate.
    *[
        CombineCase(
            f"null_accepting_filter_not_pushed_below_left_join-{name}",
            sources=left_and_right_sources(pl.DataFrame({"id": [1, 2]}), pl.DataFrame({"id": [1, 2], "v": [10, None]}), {}, {"v": FilterSpec()}),
            combine=join("id", how="left"),
            predicate=pred,
        )
        for name, pred in [("is_null", pl.col("v").is_null()), ("is_null_or_eq", pl.col("v").is_null() | (pl.col("v") == 5))]
    ],
    # A value_mapping with a window pushes a window around the mapped values.
    *[
        CombineCase(
            f"window_pushed_around_mapped_values-{name}",
            sources=single_source(TWENTY_DAYS, date=weekday_window(7)),
            combine=next_day_rolling,
            predicate=pred,
            time_bounds={"data": {"data_date": (..., ...)}},
        )
        for name, pred in [
            ("eq", pl.col("date") == make_date(8)),
            ("is_in", pl.col("date").is_in([make_date(8), make_date(15)])),
            ("between", pl.col("date").is_between(make_date(8), make_date(10))),
        ]
    ],
    CombineCase(
        "open_range_through_callable_pushes_window_around_requested_range",
        sources=single_source(TWENTY_DAYS, date=weekday_window(7)),
        combine=next_day_rolling,
        predicate=pl.col("date") >= make_date(8),
        time_bounds={"data": {"data_date": (make_date(1), None)}},
    ),
    *[
        CombineCase(
            f"incomplete_or_failing_mapping_keeps_rows-{name}",
            sources=single_source(TWENTY_DAYS, date=window(lookback=days(2), value_mapping=mapping)),
            combine=next_day_rolling,
            predicate=pl.col("date") == make_date(8),
        )
        for name, mapping in [("dict", {make_date(9): make_date(8)}), ("none", lambda day: None), ("raises", lambda day: 1 / 0)]
    ],
    CombineCase(
        "unconstrained_identity_sibling_does_not_block_the_window",
        sources=single_source(TWENTY_DAYS, data_date=FilterSpec(), date=weekday_window(2)),
        combine=next_day_rolling,
        predicate=pl.col("date") == make_date(8),
        time_bounds={"data": {"data_date": (..., ...)}},
    ),
    CombineCase(
        "mapped_aliases_in_a_self_join_union",
        sources=single_source(TWENTY_DAYS, **{n: window(lookback=days(1), value_mapping=lambda day: day - days(1)) for n in ("d1", "d2")}),
        combine=mapped_self_join,
        predicate=(pl.col("d1") == make_date(3)) & (pl.col("d2") == make_date(12)),
    ),
    CombineCase(
        "integer_output_key_is_not_used_as_a_date",
        sources=single_source(TWENTY_DAYS, k=window(lookback=days(2), value_mapping={8: make_date(7)})),
        combine=day_of_month_key,
        predicate=pl.col("k") == 8,
    ),
    CombineCase(
        "dict_covering_the_range_is_enumerated",
        sources=single_source(TWENTY_DAYS, date=window(lookback=days(1), value_mapping={make_date(8): make_date(8), make_date(9): make_date(9)})),
        combine=same_day_rolling,
        predicate=pl.col("date").is_between(make_date(8), make_date(9)),
        time_bounds={"data": {"data_date": (make_date(7), make_date(9))}},
    ),
    CombineCase(
        "lookahead_around_mapped_values",
        sources=single_source(TWENTY_DAYS, date=window(lookahead=days(2), value_mapping=lambda day: day)),
        combine=lambda s: s["data"].with_columns(pl.col("v").shift(-2).alias("lead"), pl.col("data_date").alias("date")),
        predicate=pl.col("date") == make_date(10),
    ),
    CombineCase(
        "range_beyond_the_enumeration_cap_gets_the_window_around_the_range",
        sources=single_source(TWENTY_DAYS, date=window(lookback=days(1), value_mapping=lambda day: day)),
        combine=same_day_rolling,
        predicate=pl.col("date").is_between(make_date(1, 1, 1990), make_date(10)),
        time_bounds={"data": {"data_date": (make_date(31, 12, 1989), make_date(10))}},
    ),
    # A window that cannot be applied pushes nothing rather than the unwidened value.
    CombineCase(
        "time_zone_aware_column",
        sources=single_source(
            pl.DataFrame({"t": [datetime(2024, 1, day, tzinfo=UTC_ZONE) for day in range(1, 11)], "v": range(10)}), t=FilterSpec(lookback=days(3))
        ),
        combine=lambda s: s["data"].with_columns(pl.col("v").shift(1).alias("lag")),
        predicate=pl.col("t") == datetime(2024, 1, 5, tzinfo=UTC_ZONE),
    ),
    CombineCase(
        "unbounded_window",
        sources=single_source(TEN_DAYS, d=FilterSpec(lookback=timedelta.max)),
        combine=lambda s: s["data"].with_columns(pl.col("v").cum_sum().alias("c")),
        predicate=pl.col("d") == make_date(5),
    ),
    CombineCase(
        "unbounded_window_sharer_keeps_the_whole_source_column",
        sources=single_source(
            TEN_DAYS.with_columns(pl.col("v") + 1),
            a=FilterSpec(source_col="d", lookback=timedelta.max),
            b=FilterSpec(source_col="d", value_mapping=lambda x: x),
        ),
        combine=unbounded_sharer,
        predicate=(pl.col("a") >= make_date(5)) & (pl.col("b") == make_date(8)),
    ),
    # Temporal values and units
    CombineCase(
        "mapped_values_keep_both_readings_of_a_repeated_hour",
        sources=single_source(
            pl.DataFrame({"t": NEW_YORK_REPEATED_HOUR}, schema={"t": pl.Datetime("us", "America/New_York")}),
            a=FilterSpec(source_col="t", value_mapping=lambda a: a.astimezone(NEW_YORK)),
        ),
        combine=lambda s: s["data"].with_columns(pl.col("t").dt.convert_time_zone("UTC").alias("a")),
        predicate=pl.col("a").is_in([datetime(2024, 11, 3, 5, 30, tzinfo=UTC_ZONE), datetime(2024, 11, 3, 6, 30, tzinfo=UTC_ZONE)]),
    ),
    # The collection has the column's own dtype: Polars rejects a microsecond list against a millisecond column.
    *[
        CombineCase(
            f"is_in_on_any_datetime_unit-{time_zone}-{unit}",
            sources=single_source(five_seconds(unit, time_zone).to_frame("t"), t=FilterSpec()),
            combine=identity,
            predicate=pl.col("t").is_in(five_seconds(unit, time_zone).gather([1, 3]).implode()),
        )
        for time_zone in (None, "UTC")
        for unit in ("ms", "us", "ns")
    ],
    CombineCase(
        "value_set_keys_do_not_overflow_near_the_date_limit",
        sources=single_source(
            pl.DataFrame({"t": NEAR_MAX, "k": [1, 2]}), k=FilterSpec(source_col="t", value_mapping={1: NEAR_MAX[0], 2: NEAR_MAX[1]})
        ),
        combine=identity,
        predicate=pl.col("k").is_in([1, 2]),
    ),
    CombineCase(
        "pushed_equality_keeps_the_predicates_comparison_unit",
        sources=single_source(
            pl.Series("t", [epoch_ns(datetime(2024, 1, 1)), epoch_ns(datetime(2024, 1, 1)) + 1]).cast(pl.Datetime("ns", "UTC")).to_frame(),
            a=FilterSpec(source_col="t"),
        ),
        combine=lambda s: s["data"].with_columns(pl.col("t").dt.cast_time_unit("us").alias("a")),
        predicate=pl.col("a") == datetime(2024, 1, 1, tzinfo=UTC),
    ),
    CombineCase(
        "membership_with_another_unit_keeps_native_rows",
        sources=single_source(US_MEMBERS.to_physical().cast(pl.Datetime("ns", "UTC")).to_frame("t"), t=FilterSpec()),
        combine=lambda s: s["data"].collect().lazy(),
        predicate=pl.col("t").is_in(US_MEMBERS.implode()),
    ),
    # A value_mapping onto a Datetime source column that returns dates asks for whole days, not midnights.
    *[
        CombineCase(
            f"mapped_date_on_datetime_keeps_the_whole_day-{name}-{zone}",
            sources=single_source(frame, day=FilterSpec(source_col="t", value_mapping=lambda day: day)),
            combine=lambda s: s["data"].with_columns(pl.col("t").dt.date().alias("day")),
            predicate=pred,
            sources_receiving_filter=("data",),
        )
        for zone, frame in [("naive", _TIMESTAMPS_AROUND_NEW_YEAR), ("new_york", _TIMESTAMPS_IN_NEW_YORK)]
        for name, pred in [("eq", pl.col("day") == make_date(1)), ("is_in", pl.col("day").is_in([make_date(1), make_date(2)]))]
    ],
]


def _filter(lf: pl.LazyFrame, pred) -> pl.LazyFrame:
    return lf if pred is None else lf.filter(pred)


def _matches_plain_polars(actual: pl.LazyFrame, reference: pl.LazyFrame, row_order: bool) -> pl.DataFrame | None:
    """``assert_matches_plain_polars``; without ``row_order``, the cheaper ordered comparison is tried first."""
    if not row_order:
        try:
            return assert_matches_plain_polars(actual, reference, None, check_row_order=True)
        except AssertionError:
            pass
    return assert_matches_plain_polars(actual, reference, None, check_row_order=row_order)


@pytest.mark.parametrize("case", as_params(SCENARIOS))
def test_pushdown_combine(case: CombineCase):
    sources, is_identity = case.sources, case.combine is identity
    inspected = {*case.sources_receiving_filter, *case.time_bounds, *case.allowed_values, *case.columns_without_value_filter}
    if is_identity or case.result_check is not None:
        inspected |= set(sources)
    trackers = {name: PredicateTracker(df) for name, (df, _) in sources.items() if name in inspected}
    frames = {name: trackers[name].lazy_frame if name in trackers else df.lazy() for name, (df, _) in sources.items()}
    lf = pushdown_combine(
        {name: (frames[name], specs) for name, (_, specs) in sources.items()},
        case.combine,
        combine_kwargs=case.combine_kwargs,
        sources_as_kwargs=case.sources_as_kwargs,
    )
    unfiltered = {name: df.lazy() for name, (df, _) in sources.items()}
    combine_kwargs = case.combine_kwargs or {}
    reference = case.combine(**unfiltered, **combine_kwargs) if case.sources_as_kwargs else case.combine(unfiltered, **combine_kwargs)
    post = case.post_step or (lambda frame: frame)
    row_order = is_identity if case.check_row_order is None else case.check_row_order
    preds = case.predicate if isinstance(case.predicate, list) else [case.predicate]
    for p in preds:
        actual = post(_filter(lf, p))
        if case.compare_with_polars:
            with failure_note(f"filtered by {p}"):
                got = _matches_plain_polars(actual, post(_filter(reference, p)), row_order)
        else:
            got = actual.collect()
    for name in case.sources_receiving_filter:
        assert trackers[name].last_predicate is not None, f"nothing pushed to {name}"
    for name, columns in case.time_bounds.items():
        for col, (lower, upper) in columns.items():
            assert_pushed_time_bounds(trackers[name], col, lower, upper)
    for name, columns in case.allowed_values.items():
        for col, values in columns.items():
            assert_discrete(trackers[name], col, values, case.allowed_values_node.get(name, {}).get(col))
    for name, columns in case.columns_without_value_filter.items():
        pushed = trackers[name].last_predicate
        for col in columns:
            assert pushed is None or PredicateAnalyzer(pushed).find_discrete_filter(col) is None, pushed
    if case.result_check is not None:
        case.result_check(got, trackers)


def test_combine_kwargs_with_join_parameters():
    """Two frames built over the same sources, each with its own combine_kwargs."""
    left, right = PredicateTracker(KEY_LEFT), PredicateTracker(KEY_RIGHT)
    for how in ("inner", "left"):
        lf = pushdown_combine(
            {"left": (left.lazy_frame, {"date": FilterSpec()}), "right": (right.lazy_frame, {})}, join_how, combine_kwargs={"join_how": how}
        )
        assert_matches_plain_polars(lf, KEY_LEFT.lazy().join(KEY_RIGHT.lazy(), on="key", how=how), None)


def test_string_filter_is_pushed_without_a_datetime_range_attempt(caplog):
    """The range visitor logs each literal it cannot read as a datetime; a filter on a string column must not reach it."""
    left, right = pl.LazyFrame({"id": ["A", "B", "C"], "val": [1, 2, 3]}), pl.LazyFrame({"id": ["A", "B", "C"], "other": [10, 20, 30]})
    lf = pushdown_combine(
        {"left": (left, {"id": FilterSpec()}), "right": (right, {"id": FilterSpec()})}, lambda s: s["left"].join(s["right"], on="id")
    )
    with caplog.at_level(logging.INFO, logger="polars_io_tools.io_sources.range_visitor"):
        result = lf.filter(pl.col("id") == "B").collect()
    assert [r.getMessage() for r in caplog.records if r.name == "polars_io_tools.io_sources.range_visitor"] == []
    assert_frame_equal(result, pl.DataFrame({"id": ["B"], "val": [2], "other": [20]}))


def test_lone_side_pushes_no_group_predicate():
    """With one side of a shared pair unconstrained, at least one scan carries no discrete filter on the shared column."""
    predicates = []
    source = io_source_assert(THREE_GROUPS, predicates.append)
    lf = pushdown_combine({"data": (source, PAIR_SPECS)}, group_pairs)
    assert_matches_plain_polars(lf, group_pairs({"data": THREE_GROUPS.lazy()}), pl.col("group1") == "A")
    assert predicates
    assert any(p is None or PredicateAnalyzer(p).find_discrete_filter("group") is None for p in predicates)


def test_many_float32_values_push_one_membership():
    values = {float(2**30 + 64 * i) for i in range(10_000)}
    expr = _union_discrete_filter("x", [("x", FilterSpec())], {"x": values}, pl.Int64, lambda: frozenset({"x"}))
    assert len(str(expr)) < 1_000
    rows = [2**30 + 64, 2**30 + 65, 2**30 + 64 * 10_000, 2**30 + 64 * 10_001, 5]
    targets = {np.float32(v) for v in values}
    assert pl.DataFrame({"x": rows}).filter(expr)["x"].to_list() == [x for x in rows if np.float32(x) in targets]


def test_non_numeric_source_does_not_resolve_float32_outputs():
    def float32_outputs():
        raise AssertionError("output schema resolved")

    expr = _union_discrete_filter("x", [("x", FilterSpec())], {"x": {"a"}}, pl.String, float32_outputs)
    assert pl.DataFrame({"x": ["a", "b"]}).filter(expr)["x"].to_list() == ["a"]


def test_integral_float_without_float32_consumer_pushes_equality():
    expr = _union_discrete_filter("x", [("x", FilterSpec())], {"x": {2.0**24}}, pl.Int64)
    # Neither the integer next to the value nor nulls are kept.
    assert pl.DataFrame({"x": [2**24, 2**24 + 1, None]}).filter(expr)["x"].to_list() == [2**24]


@pytest.mark.parametrize("pred", [pl.lit(False), pl.lit(None, dtype=pl.Boolean)], ids=["false", "null"])
def test_constant_false_predicate_reads_nothing(pred):
    from polars.io.plugins import register_io_source

    rows_read = []

    def generator(with_columns, predicate, n_rows, batch_size):
        out = TEN_DAYS if n_rows is None else TEN_DAYS.head(n_rows)
        rows_read.append(out.height)
        yield out

    source = register_io_source(generator, schema=TEN_DAYS.schema)
    assert (
        pushdown_combine(
            {"s": (source, {"d": FilterSpec(lookback=days(1))})}, lambda s: s["s"].with_columns(pl.col("v").rolling_sum(2).alias("rolling_sum"))
        )
        .filter(pred)
        .collect()
        .height
        == 0
    )
    assert sum(rows_read) == 0


# Specs and helpers ------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SpecFieldsCase:
    """A filter spec and the values some of its fields hold."""

    id: str
    _: KW_ONLY
    spec: FilterSpec | IntervalFilterSpec
    # field name -> value
    fields: dict


SPEC_FIELDS = [
    SpecFieldsCase(
        "filter_spec-defaults",
        spec=FilterSpec(),
        fields={"source_col": None, "lookback": timedelta(), "lookahead": timedelta(), "value_mapping": None},
    ),
    SpecFieldsCase("source_col_override", spec=FilterSpec(source_col="DataDate"), fields={"source_col": "DataDate"}),
    SpecFieldsCase("lookback_override", spec=FilterSpec(lookback=days(5)), fields={"lookback": days(5)}),
    SpecFieldsCase("lookahead_override", spec=FilterSpec(lookahead=days(3)), fields={"lookahead": days(3)}),
    SpecFieldsCase("value_mapping_dict", spec=FilterSpec(value_mapping={"A": "a", "B": "b"}), fields={"value_mapping": {"A": "a", "B": "b"}}),
    SpecFieldsCase("value_mapping_callable", spec=FilterSpec(value_mapping=str.upper), fields={"value_mapping": str.upper}),
    SpecFieldsCase(
        "combined_options",
        spec=FilterSpec(source_col="col", lookback=days(5), lookahead=days(3), value_mapping={"x": "y"}),
        fields={"source_col": "col", "lookback": days(5), "lookahead": days(3), "value_mapping": {"x": "y"}},
    ),
    SpecFieldsCase(
        "interval_spec-defaults",
        spec=IntervalFilterSpec(start_col="s", end_col="e"),
        fields={"start_col": "s", "end_col": "e", "closed": "both", "value_mapping": None},
    ),
    *[
        SpecFieldsCase(
            f"valid_closed_accepted-{closed}", spec=IntervalFilterSpec(start_col="s", end_col="e", closed=closed), fields={"closed": closed}
        )
        for closed in CLOSED_OPTIONS
    ],
]


@pytest.mark.parametrize("case", as_params(SPEC_FIELDS))
def test_spec_fields(case: SpecFieldsCase):
    assert {name: getattr(case.spec, name) for name in case.fields} == case.fields


def test_invalid_closed_raises():
    with pytest.raises(ValueError, match="closed must be one of"):
        IntervalFilterSpec(start_col="s", end_col="e", closed="closed")


@dataclass(frozen=True)
class ValueMappingCase:
    """Values and a value mapping, and what ``_apply_value_mapping`` makes of them."""

    id: str
    _: KW_ONLY
    values: set
    mapping: dict | Callable | None
    mapped: set
    # the values the mapping leaves out, which are not pushed
    unmapped: set


VALUE_MAPPINGS = [
    ValueMappingCase("none_mapping", values={"a", "b", "c"}, mapping=None, mapped={"a", "b", "c"}, unmapped=set()),
    ValueMappingCase(
        "dict_mapping_all_keys_present",
        values={"k1", "k2"},
        mapping={"k1": "v1", "k2": "v2"},
        mapped={"v1", "v2"},
        unmapped=set(),
    ),
    ValueMappingCase("dict_mapping_partial_keys", values={"k1", "missing"}, mapping={"k1": "v1"}, mapped={"v1"}, unmapped={"missing"}),
    ValueMappingCase(
        "dict_mapping_no_keys_present", values={"missing1", "missing2"}, mapping={"k1": "v1"}, mapped=set(), unmapped={"missing1", "missing2"}
    ),
    ValueMappingCase("callable_mapping", values={"foo", "bar"}, mapping=str.upper, mapped={"FOO", "BAR"}, unmapped=set()),
    ValueMappingCase("callable_mapping_custom_function", values={1, 2, 3}, mapping=lambda x: x * 2, mapped={2, 4, 6}, unmapped=set()),
    ValueMappingCase("empty_values", values=set(), mapping={"a": "b"}, mapped=set(), unmapped=set()),
    # A set or frozenset result is one-to-many: it stands for each of its members.
    ValueMappingCase(
        "dict_set_result_is_flattened", values={"A"}, mapping={"A": frozenset({"a1", "a2", "a3"})}, mapped={"a1", "a2", "a3"}, unmapped=set()
    ),
    ValueMappingCase(
        "callable_set_result_is_flattened", values={"A", "B"}, mapping={"A": {"a1", "a2"}, "B": {"b1"}}.get, mapped={"a1", "a2", "b1"}, unmapped=set()
    ),
    ValueMappingCase("empty_set_result_is_mapped_to_nothing", values={"Z"}, mapping={"Z": frozenset()}, mapped=set(), unmapped=set()),
]


@pytest.mark.parametrize("case", as_params(VALUE_MAPPINGS))
def test_apply_value_mapping(case: ValueMappingCase):
    assert _apply_value_mapping(case.values, case.mapping) == (case.mapped, case.unmapped)


def test_list_and_tuple_results_stay_single_values():
    """Only a set or frozenset is one-to-many; a list or tuple is one (for example List-dtype) value."""
    assert _map_values({"A"}, {"A": ["x", "y"]}) == ([["x", "y"]], set())
    assert _map_values({"A"}, {"A": ("x", "y")}) == ([("x", "y")], set())


# One-to-many value mappings: a set or frozenset result stands for any of its members. Group "A" is derived from
# members a1, a2 and a3, "B" from b1 and b2, and "Z" from none.
MEMBERS_FRAME = pl.DataFrame({"member": ["a1", "a2", "a3", "b1", "b2"], "v": [1, 2, 3, 4, 5]})
MEMBERS_OF_GROUP = {"A": frozenset({"a1", "a2", "a3"}), "B": frozenset({"b1", "b2"}), "Z": frozenset()}
GROUP_OF_MEMBER = {"a1": "A", "a2": "A", "a3": "A", "b1": "B", "b2": "B"}


def _with_group(s):
    return s["src"].with_columns(pl.col("member").replace_strict(GROUP_OF_MEMBER, default=None).alias("group"))


def _combined_matches_plain(sources: dict, combine, pred) -> pl.DataFrame | None:
    reference = combine({name: lf for name, (lf, _) in sources.items()})
    return assert_matches_plain_polars(pushdown_combine(sources, combine), reference, pred)


def _discrete_values_pushed(tracker: PredicateTracker, column: str) -> set:
    analyzer = tracker.get_analyzer()
    return analyzer.extract_discrete_values(analyzer.find_discrete_filter(column))


@pytest.mark.parametrize("mapping", [MEMBERS_OF_GROUP, MEMBERS_OF_GROUP.__getitem__], ids=["dict", "callable"])
def test_set_valued_mapping_pushes_is_in_over_the_members(mapping):
    tracker = PredicateTracker(MEMBERS_FRAME)
    lf = pushdown_combine({"src": (tracker.lazy_frame, {"group": FilterSpec(source_col="member", value_mapping=mapping)})}, _with_group)
    assert set(lf.filter(pl.col("group") == "A").collect()["member"]) == {"a1", "a2", "a3"}
    assert _discrete_values_pushed(tracker, "member") == {"a1", "a2", "a3"}
    assert tracker.call_count == 1


@pytest.mark.parametrize(
    "pred", [pl.col("group") == "A", pl.col("group").is_in(["A", "B"]), pl.col("group") == "Z"], ids=["eq", "is_in", "empty_set"]
)
def test_set_valued_mapping_matches_plain_polars(pred):
    _combined_matches_plain(
        {"src": (MEMBERS_FRAME.lazy(), {"group": FilterSpec(source_col="member", value_mapping=MEMBERS_OF_GROUP)})}, _with_group, pred
    )


def test_empty_set_pushes_a_filter_that_keeps_nothing():
    tracker = PredicateTracker(MEMBERS_FRAME)
    lf = pushdown_combine({"src": (tracker.lazy_frame, {"group": FilterSpec(source_col="member", value_mapping=MEMBERS_OF_GROUP)})}, _with_group)
    assert lf.filter(pl.col("group") == "Z").collect().height == 0
    assert tracker.call_count == 1
    assert tracker.last_predicate is not None and tracker.last_predicate.meta.eq(pl.lit(False))


def _member_with_alias(s):
    left = s["src"].with_columns(pl.col("member").replace({"a1": "A", "a2": "A", "b1": "B"}).alias("group1"))
    return left.join(s["src"].select(pl.col("member").alias("member2")), how="cross")


def test_set_valued_mapping_unions_with_an_identity_sibling():
    specs = {"group1": FilterSpec(source_col="member", value_mapping={"A": frozenset({"a1", "a2"})}), "member2": FilterSpec(source_col="member")}
    tracker = PredicateTracker(pl.DataFrame({"member": ["a1", "a2", "b1"]}))
    pushdown_combine({"src": (tracker.lazy_frame, specs)}, _member_with_alias).filter(
        (pl.col("group1") == "A") & (pl.col("member2") == "b1")
    ).collect()
    assert _discrete_values_pushed(tracker, "member") == {"a1", "a2", "b1"}


def test_empty_set_with_an_unconstrained_sibling_pushes_nothing():
    specs = {"group1": FilterSpec(source_col="member", value_mapping={"A": frozenset()}), "member2": FilterSpec(source_col="member")}
    tracker = PredicateTracker(pl.DataFrame({"member": ["a1", "a2", "b1"]}))
    pushdown_combine({"src": (tracker.lazy_frame, specs)}, _member_with_alias).filter(pl.col("group1") == "A").collect()
    assert tracker.last_predicate is None


def test_set_valued_mapping_and_a_filter_on_another_column_are_both_pushed():
    def combine(s):
        return s["src"].with_columns(pl.col("member").replace({"a1": "A", "a2": "A", "b1": "B"}).alias("group"))

    specs = {"group": FilterSpec(source_col="member", value_mapping={"A": frozenset({"a1", "a2"})}), "region": FilterSpec()}
    tracker = PredicateTracker(pl.DataFrame({"member": ["a1", "a2", "b1"], "region": ["x", "y", "x"]}))
    pushdown_combine({"src": (tracker.lazy_frame, specs)}, combine).filter((pl.col("group") == "A") & (pl.col("region") == "x")).collect()
    assert _discrete_values_pushed(tracker, "member") == {"a1", "a2"}
    assert _discrete_values_pushed(tracker, "region") == {"x"}


def test_none_in_a_set_also_matches_null():
    def combine(s):
        group = pl.when((pl.col("member") == "a1") | pl.col("member").is_null()).then(pl.lit("A")).otherwise(pl.lit("B")).alias("group")
        return s["src"].with_columns(group)

    frame = pl.DataFrame({"member": ["a1", "a2", None], "v": [1, 2, 3]})
    spec = FilterSpec(source_col="member", value_mapping={"A": frozenset({None, "a1"})})
    result = _combined_matches_plain({"src": (frame.lazy(), {"group": spec})}, combine, pl.col("group") == "A")
    assert sorted(result["v"].to_list()) == [1, 3]


def test_a_missing_key_next_to_an_empty_set_pushes_nothing():
    tracker = PredicateTracker(MEMBERS_FRAME)
    lf = pushdown_combine({"src": (tracker.lazy_frame, {"group": FilterSpec(source_col="member", value_mapping=MEMBERS_OF_GROUP)})}, _with_group)
    lf.filter(pl.col("group").is_in(["Z", "MISSING"])).collect()
    assert tracker.last_predicate is None


def test_set_valued_mapping_on_a_windowed_column_pushes_nothing():
    def combine(s):
        return s["s"].with_columns(pl.col("v").rolling_sum(2).alias("r"), pl.col("data_date").alias("date"))

    tracker = PredicateTracker(pl.DataFrame({"data_date": [date(2024, 1, d) for d in range(1, 11)], "v": list(range(1, 11))}))
    spec = FilterSpec(source_col="data_date", lookback=timedelta(days=2), value_mapping=lambda d: {d})
    _combined_matches_plain({"s": (tracker.lazy_frame, {"date": spec})}, combine, pl.col("date") == date(2024, 1, 8))
    assert tracker.last_predicate is None


# An IntervalFilterSpec mapping that maps a requested day to a set has no single day to compare against, so nothing is
# pushed for the interval.
SET_BOUND_SCHEDULE = pl.DataFrame(
    {"root": ["A", "A"], "start": [date(2024, 1, 1), date(2024, 1, 20)], "end": [date(2024, 1, 10), date(2024, 1, 30)], "val": [1, 2]}
)


@pytest.mark.parametrize(
    ("mapping", "pred"),
    [
        (lambda day: {day}, pl.col("date") == date(2024, 1, 5)),
        (
            {date(2024, 1, 4): frozenset({date(2024, 1, 4)}), date(2024, 1, 5): date(2024, 1, 5), date(2024, 1, 6): date(2024, 1, 6)},
            pl.col("date").is_between(date(2024, 1, 4), date(2024, 1, 6)),
        ),
    ],
    ids=["every_day", "one_day"],
)
def test_interval_mapping_to_a_set_pushes_nothing(mapping, pred):
    tracker = PredicateTracker(SET_BOUND_SCHEDULE)
    spine = pl.DataFrame({"root": ["A"], "date": [date(2024, 1, 5)]})
    sources = {
        "sched": (tracker.lazy_frame, {"date": IntervalFilterSpec(start_col="start", end_col="end", value_mapping=mapping)}),
        "spine": (spine.lazy(), {"date": FilterSpec()}),
    }
    _combined_matches_plain(sources, interval_join(), pred)
    assert tracker.last_predicate is None


def _previous_day(d):
    return d - timedelta(days=1)


# same_row_as: output columns read from the same source row. ``date`` is ``data_date`` + 1 day.
TEN_DAYS = pl.DataFrame({"data_date": [make_date(d) for d in range(1, 11)], "v": list(range(1, 11))})


def _with_next_day(s):
    return s["s"].with_columns(pl.col("data_date").dt.offset_by("1d").alias("date"))


def _day_and_next_day(**next_day_spec) -> dict:
    return {"data_date": FilterSpec(), "date": FilterSpec(source_col="data_date", value_mapping=_previous_day, **next_day_spec)}


@dataclass(frozen=True)
class SameRowCase:
    """A filter on ``date`` and ``data_date`` of TEN_DAYS, and the source days the pushed filter keeps (None: nothing pushed)."""

    id: str
    _: KW_ONLY
    pred: pl.Expr
    kept_days: set | None


SAME_ROW = [
    SameRowCase("eq", pred=pl.col("date") == make_date(5), kept_days={4}),
    SameRowCase("is_in", pred=pl.col("date").is_in([make_date(5), make_date(8)]), kept_days={4, 7}),
    SameRowCase("both_filtered", pred=(pl.col("date") == make_date(5)) & (pl.col("data_date") == make_date(4)), kept_days={4}),
    SameRowCase("both_filtered_disjoint", pred=(pl.col("date") == make_date(5)) & (pl.col("data_date") == make_date(7)), kept_days=set()),
    SameRowCase("identity_only", pred=pl.col("data_date") == make_date(4), kept_days={4}),
    # Either column may hold for the row, so neither filter alone narrows the read.
    SameRowCase("or_across_columns", pred=(pl.col("date") == make_date(5)) | (pl.col("data_date") == make_date(8)), kept_days=None),
    # A range on a mapped column without a lookback is not pushed, as for the mapped column alone.
    SameRowCase("range", pred=pl.col("date").is_between(make_date(5), make_date(6)), kept_days=None),
]


@pytest.mark.parametrize("case", as_params(SAME_ROW))
def test_same_row_as_pushes_the_filters_of_one_row(case: SameRowCase):
    tracker = PredicateTracker(TEN_DAYS)
    sources = {"s": (tracker.lazy_frame, _day_and_next_day(same_row_as="data_date"))}
    assert_matches_plain_polars(pushdown_combine(sources, _with_next_day), _with_next_day({"s": TEN_DAYS.lazy()}), case.pred)
    kept = None if tracker.last_predicate is None else {d.day for d in TEN_DAYS.filter(tracker.last_predicate)["data_date"]}
    assert kept == case.kept_days


def test_without_same_row_as_a_filter_on_the_mapped_column_pushes_nothing():
    tracker = PredicateTracker(TEN_DAYS)
    with pytest.warns(UserWarning, match="same_row_as"):
        lf = pushdown_combine({"s": (tracker.lazy_frame, _day_and_next_day())}, _with_next_day)
    assert_matches_plain_polars(lf, _with_next_day({"s": TEN_DAYS.lazy()}), pl.col("date") == make_date(5))
    assert tracker.last_predicate is None


def test_same_row_as_links_through_a_chain():
    specs = {
        "data_date": FilterSpec(),
        "date": FilterSpec(source_col="data_date", value_mapping=_previous_day, same_row_as="data_date"),
        "day_after": FilterSpec(source_col="data_date", value_mapping=lambda d: d - timedelta(days=2), same_row_as="date"),
    }

    def combine(s):
        return _with_next_day(s).with_columns(pl.col("data_date").dt.offset_by("2d").alias("day_after"))

    tracker = PredicateTracker(TEN_DAYS)
    pred = (pl.col("day_after") == make_date(6)) & (pl.col("date") == make_date(5))
    assert_matches_plain_polars(pushdown_combine({"s": (tracker.lazy_frame, specs)}, combine), combine({"s": TEN_DAYS.lazy()}), pred)
    assert set(TEN_DAYS.filter(tracker.last_predicate)["data_date"]) == {make_date(4)}


def test_a_row_without_a_filter_still_pushes_nothing_for_separate_rows():
    # ``date`` and ``data_date`` are one row; ``other`` is read from another row of the same source (a self-join).
    specs = {**_day_and_next_day(same_row_as="data_date"), "other": FilterSpec(source_col="data_date", value_mapping=_previous_day)}

    def combine(s):
        return _with_next_day(s).join(s["s"].select(pl.col("data_date").dt.offset_by("1d").alias("other")), how="cross")

    reference = combine({"s": TEN_DAYS.lazy()})
    for pred, kept_days in [(pl.col("date") == make_date(5), None), ((pl.col("date") == make_date(5)) & (pl.col("other") == make_date(8)), {4, 7})]:
        tracker = PredicateTracker(TEN_DAYS)
        assert_matches_plain_polars(pushdown_combine({"s": (tracker.lazy_frame, specs)}, combine), reference, pred)
        kept = None if tracker.last_predicate is None else {d.day for d in TEN_DAYS.filter(tracker.last_predicate)["data_date"]}
        assert kept == kept_days


def test_same_row_as_on_a_datetime_source_keeps_the_whole_day():
    frame = pl.DataFrame({"data_date": [datetime(2024, 1, 4, 12), datetime(2024, 1, 5, 9)], "v": [1, 2]})

    def combine(s):
        return s["s"].with_columns(pl.col("data_date").dt.date().dt.offset_by("1d").alias("date"))

    tracker = PredicateTracker(frame)
    sources = {"s": (tracker.lazy_frame, _day_and_next_day(same_row_as="data_date"))}
    result = assert_matches_plain_polars(pushdown_combine(sources, combine), combine({"s": frame.lazy()}), pl.col("date") == make_date(5))
    assert result["v"].to_list() == [1]
    assert frame.filter(tracker.last_predicate)["v"].to_list() == [1]


def test_same_row_as_pushes_nothing_when_a_linked_column_accepts_null():
    # A left join pads id 2 with nulls; dropping the source row of id 1 would pad id 1 too, and the final filter
    # keeps padded rows.
    def combine(s):
        joined = s["left"].join(s["right"], on="id", how="left")
        return joined.with_columns(pl.col("data_date").fill_null(make_date(10)).dt.offset_by("1d").alias("date"))

    right = pl.DataFrame({"id": [1], "data_date": [make_date(4)]})
    specs = {
        "data_date": FilterSpec(),
        "date": FilterSpec(source_col="data_date", value_mapping={make_date(11): frozenset({None, make_date(10)})}, same_row_as="data_date"),
    }
    sources = {"left": (pl.LazyFrame({"id": [1, 2]}), {}), "right": (right.lazy(), specs)}
    pred = pl.col("data_date").is_null() & (pl.col("date") == make_date(11))
    result = assert_matches_plain_polars(pushdown_combine(sources, combine), combine({n: lf for n, (lf, _) in sources.items()}), pred)
    assert result["id"].to_list() == [2]


def test_same_row_as_pushes_nothing_when_another_row_accepts_null():
    # ``other`` is read from the next row by a left self-join; its mapping accepts the null that pads the last row.
    frame = pl.DataFrame({"id": [1, 2], "data_date": [make_date(4), make_date(7)]})

    def combine(s):
        nxt = s["s"].select((pl.col("id") - 1).alias("id"), pl.col("data_date").alias("next_data_date"))
        joined = _with_next_day(s).join(nxt, on="id", how="left")
        return joined.with_columns(pl.col("next_data_date").fill_null(make_date(10)).dt.offset_by("1d").alias("other")).drop("next_data_date")

    specs = {
        **_day_and_next_day(same_row_as="data_date"),
        "other": FilterSpec(source_col="data_date", value_mapping={make_date(11): frozenset({None, make_date(10)})}),
    }
    pred = (pl.col("date") == make_date(5)) & (pl.col("other") == make_date(11))
    result = assert_matches_plain_polars(pushdown_combine({"s": (frame.lazy(), specs)}, combine), combine({"s": frame.lazy()}), pred)
    assert result.is_empty()


def _raises(day):
    raise ValueError("mapping failed")


@pytest.mark.parametrize("same_row_as", [None, "data_date"], ids=["alone", "linked"])
def test_a_value_mapping_that_raises_pushes_nothing(same_row_as):
    specs = {"date": FilterSpec(source_col="data_date", value_mapping=_raises, same_row_as=same_row_as)}
    if same_row_as is not None:
        specs["data_date"] = FilterSpec()
    tracker = PredicateTracker(TEN_DAYS)
    lf = pushdown_combine({"s": (tracker.lazy_frame, specs)}, _with_next_day)
    assert_matches_plain_polars(lf, _with_next_day({"s": TEN_DAYS.lazy()}), pl.col("date") == make_date(5))
    assert tracker.last_predicate is None


@dataclass(frozen=True)
class SameRowErrorCase:
    """Specs of one source entry with a ``same_row_as`` that ``pushdown_combine`` must reject."""

    id: str
    _: KW_ONLY
    specs: dict
    match: str


SAME_ROW_ERRORS = [
    SameRowErrorCase("itself", specs={"date": FilterSpec(source_col="data_date", same_row_as="date")}, match="another FilterSpec"),
    SameRowErrorCase("missing", specs={"date": FilterSpec(source_col="data_date", same_row_as="nope")}, match="another FilterSpec"),
    SameRowErrorCase(
        "another_source_column", specs={"v": FilterSpec(), "date": FilterSpec(source_col="data_date", same_row_as="v")}, match="another FilterSpec"
    ),
    SameRowErrorCase(
        "interval_spec",
        specs={"w": IntervalFilterSpec("data_date", "data_date"), "date": FilterSpec(source_col="data_date", same_row_as="w")},
        match="another FilterSpec",
    ),
    # A windowed column can need rows that the filter of the other column of its row would drop.
    SameRowErrorCase(
        "windowed_reader",
        specs={"data_date": FilterSpec(lookback=timedelta(days=2)), "date": FilterSpec(source_col="data_date", same_row_as="data_date")},
        match="lookback or lookahead",
    ),
]


@pytest.mark.parametrize("case", as_params(SAME_ROW_ERRORS))
def test_invalid_same_row_as_is_rejected(case: SameRowErrorCase):
    with pytest.raises(ValueError, match=case.match):
        pushdown_combine({"s": (TEN_DAYS.lazy(), case.specs)}, lambda s: s["s"])


@pytest.mark.parametrize(
    "specs",
    [
        {"data_date": FilterSpec(), "date": FilterSpec(source_col="data_date")},
        {"data_date": FilterSpec(), "date": FilterSpec(source_col="data_date", value_mapping=_previous_day, lookback=timedelta(days=2))},
        _day_and_next_day(same_row_as="data_date"),
    ],
    ids=["plain_alias", "windowed_mapped_alias", "linked"],
)
def test_no_warning_without_an_unlinked_mapped_column(specs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pushdown_combine({"s": (TEN_DAYS.lazy(), specs)}, _with_next_day)


def test_no_warning_when_each_row_is_its_own_source_entry():
    specs = {"date": FilterSpec(source_col="data_date", value_mapping=_previous_day)}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pushdown_combine({"a": (TEN_DAYS.lazy(), {"data_date": FilterSpec()}), "b": (TEN_DAYS.lazy(), specs)}, lambda s: s["a"])


@dataclass(frozen=True)
class SourceColCase:
    """A filter spec, and the source column of the output column ``date``."""

    id: str
    _: KW_ONLY
    spec: FilterSpec
    expected: str


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            SourceColCase("none_source_col_returns_output_col", spec=FilterSpec(), expected="date"),
            SourceColCase("explicit_source_col", spec=FilterSpec(source_col="DataDate"), expected="DataDate"),
        ]
    ),
)
def test_get_source_col(case: SourceColCase):
    assert _get_source_col("date", case.spec) == case.expected


@dataclass(frozen=True)
class OutputSchemaCase:
    """A combine function and the schema of its output."""

    id: str
    _: KW_ONLY
    combine: Callable
    expected_schema: dict[str, pl.DataType]


OUTPUT_SCHEMAS = [
    OutputSchemaCase(
        "simple_join_schema", combine=lambda s: s["left"].join(s["right"], on="a"), expected_schema={"a": pl.Int64, "b": pl.Int64, "c": pl.Int64}
    ),
    OutputSchemaCase(
        "with_columns_schema",
        combine=lambda s: s["left"].select("a").with_columns(pl.lit(42).alias("new_col")),
        expected_schema={"a": pl.Int64, "new_col": pl.Int32},
    ),
    OutputSchemaCase("select_schema", combine=lambda s: s["left"].select("a", "b"), expected_schema={"a": pl.Int64, "b": pl.Int64}),
]


@pytest.mark.parametrize("case", as_params(OUTPUT_SCHEMAS))
def test_compute_output_schema(case: OutputSchemaCase):
    sources = {"left": (pl.LazyFrame({"a": [1], "b": [2]}), {}), "right": (pl.LazyFrame({"a": [1], "c": [3]}), {})}
    assert dict(_compute_output_schema(sources, case.combine)) == case.expected_schema


@pytest.mark.xfail(strict=True, raises=pl.exceptions.PanicException, reason="a Polars panic on an out-of-range literal breaks the parser")
def test_out_of_range_datetime_literal_does_not_crash_pushdown_combine():
    frame = pl.DataFrame({"t": pl.Series([1, None]).cast(pl.Datetime("us"))})
    huge = pl.lit(-(2**63), dtype=pl.Int64).cast(pl.Datetime("ms"))
    pred = (pl.col("t").cast(pl.Datetime("ms"), strict=False) < huge) | (pl.col("t") > 0)
    combined = pushdown_combine({"s": (frame.lazy(), {"t": FilterSpec()})}, lambda s: s["s"])
    assert_frame_equal(combined.filter(pred).collect(), frame.filter(pred))


_JAN_1_2024 = date(2024, 1, 1)


_NAMES = {0: "zero", 1: "one", 2: "two", 3: "three"}


def _previous_business_day(d):
    return d - timedelta(days={0: 3, 6: 2}.get(d.weekday(), 1))


def _offset_monday(d):
    """Not monotone: Sunday maps to Saturday, after Monday's Friday."""
    return d - timedelta(days=3 if d.weekday() == 0 else 1)


def _offset_monday_expr(e: pl.Expr) -> pl.Expr:
    return pl.when(e.dt.weekday() == 1).then(e.dt.offset_by("-3d")).otherwise(e.dt.offset_by("-1d"))


def _last_day_on_sunday(d):
    """Sunday maps to the last representable day, the others to themselves."""
    return date.max if d.weekday() == 6 else d


def _last_day_on_sunday_expr(e: pl.Expr) -> pl.Expr:
    return pl.when(e.dt.weekday() == 7).then(pl.lit(date.max)).otherwise(e)


def _rolling(frame):
    return frame.sort("key", "data_date").with_columns(pl.col("v").rolling_sum_by("data_date", window_size="3d").over("key").alias("rolling_sum"))


def _overlapping(s, mapped: pl.Expr) -> pl.LazyFrame:
    """Calendar days joined with the windows holding each day's mapped day."""
    days = s["c"].with_columns(mapped.alias("m"))
    return days.join(s["w"], how="cross").filter((pl.col("start") <= pl.col("m")) & (pl.col("m") <= pl.col("end"))).drop("m")


# Each shape: specs per source (source name -> (frame, specs)) and a combine whose output rows depend only on the source
# rows the specs declare (the source rows matching the output value, widened by any lookback, lookahead or value
# mapping). Output columns: date (Date), key (Int64) and, for the lookup shape, name (String).
_SHAPES = {
    "rowwise": (
        {"s": ("src", {"date": FilterSpec(source_col="data_date"), "key": FilterSpec()})},
        lambda s: s["s"].rename({"data_date": "date"}),
    ),
    "rolling": (
        {"s": ("src", {"date": FilterSpec(source_col="data_date", lookback=timedelta(days=3)), "key": FilterSpec()})},
        lambda s: _rolling(s["s"]).rename({"data_date": "date"}),
    ),
    "business_day_mapped": (
        # FilterSpec: an open range's window must also cover the mapping's offset, so lookback = offset (3d) + rolling window (3d).
        {"s": ("src", {"date": FilterSpec(source_col="data_date", lookback=timedelta(days=3 + 3), value_mapping=_previous_business_day)})},
        lambda s: _rolling(s["s"]).with_columns(pl.col("data_date").dt.add_business_days(1, roll="backward").alias("date")),
    ),
    "inner_join": (
        {
            "s": ("src", {"date": FilterSpec(source_col="data_date"), "key": FilterSpec()}),
            "o": ("other", {"date": FilterSpec(), "key": FilterSpec()}),
        },
        lambda s: s["s"].rename({"data_date": "date"}).join(s["o"], on=["date", "key"]),
    ),
    "left_join": (
        {
            "s": ("src", {"date": FilterSpec(source_col="data_date"), "key": FilterSpec()}),
            "o": ("other", {"date": FilterSpec(), "key": FilterSpec()}),
        },
        lambda s: s["s"].rename({"data_date": "date"}).join(s["o"], on=["date", "key"], how="left"),
    ),
    "asof_join": (
        {
            "s": ("src", {"date": FilterSpec(source_col="data_date"), "key": FilterSpec()}),
            "o": ("other", {"date": FilterSpec(lookback=timedelta(days=2)), "key": FilterSpec()}),
        },
        lambda s: (
            s["s"]
            .rename({"data_date": "date"})
            .sort("date")
            # An unstable sort orders rows of equal date by the other rows present, so ties are broken by w.
            .join_asof(s["o"].sort("date", "w"), on="date", by="key", strategy="backward", tolerance=timedelta(days=2))
        ),
    ),
    "group_by": (
        {"s": ("src", {"date": FilterSpec(source_col="data_date"), "key": FilterSpec()})},
        lambda s: s["s"].group_by(pl.col("data_date").alias("date"), "key").agg(pl.col("v").sum(), pl.len()),
    ),
    "explode": (
        {"s": ("src", {"date": FilterSpec(source_col="data_date"), "key": FilterSpec()})},
        lambda s: s["s"].rename({"data_date": "date"}).explode("tags"),
    ),
    "inner_filter": (
        {"s": ("src", {"date": FilterSpec(source_col="data_date", lookback=timedelta(days=3)), "key": FilterSpec()})},
        lambda s: _rolling(s["s"]).filter(pl.col("v") % 2 == 0).rename({"data_date": "date"}),
    ),
    "self_join_two_entries": (
        {"a": ("src", {"date": FilterSpec(source_col="data_date")}), "b": ("src", {"key": FilterSpec()})},
        lambda s: s["a"].select(pl.col("data_date").alias("date"), "v").join(s["b"].select("key", pl.col("v").alias("w")), how="cross"),
    ),
    "lookup": (
        {
            "s": (
                "src",
                {"date": FilterSpec(source_col="data_date"), "name": FilterSpec(source_col="key", value_mapping={v: k for k, v in _NAMES.items()})},
            )
        },
        lambda s: s["s"].rename({"data_date": "date"}).with_columns(pl.col("key").replace_strict(_NAMES, default=None).alias("name")),
    ),
    "interval": (
        {"c": ("calendar", {"date": FilterSpec()}), "w": ("windows", {"date": IntervalFilterSpec("start", "end"), "key": FilterSpec()})},
        lambda s: _overlapping(s, pl.col("date")),
    ),
    "interval_offset": (
        {
            "c": ("calendar", {"date": FilterSpec()}),
            "w": ("windows", {"date": IntervalFilterSpec("start", "end", value_mapping=timedelta(days=-1)), "key": FilterSpec()}),
        },
        lambda s: _overlapping(s, pl.col("date").dt.offset_by("-1d")),
    ),
    "interval_mapped": (
        {
            "c": ("calendar", {"date": FilterSpec()}),
            "w": ("windows", {"date": IntervalFilterSpec("start", "end", value_mapping=_offset_monday), "key": FilterSpec()}),
        },
        lambda s: _overlapping(s, _offset_monday_expr(pl.col("date"))),
    ),
    "same_row_mapped": (
        # date is data_date + 1 day, read from the same source row.
        {
            "s": (
                "src",
                {
                    "data_date": FilterSpec(),
                    "date": FilterSpec(source_col="data_date", value_mapping=_previous_day, same_row_as="data_date"),
                    "key": FilterSpec(),
                },
            )
        },
        lambda s: s["s"].with_columns(pl.col("data_date").dt.offset_by("1d").alias("date")),
    ),
    "interval_mapped_to_last_day": (
        {
            "c": ("calendar", {"date": FilterSpec()}),
            "w": ("windows", {"date": IntervalFilterSpec("start", "end", value_mapping=_last_day_on_sunday), "key": FilterSpec()}),
        },
        lambda s: _overlapping(s, _last_day_on_sunday_expr(pl.col("date"))),
    ),
}


def _old_polars_null_key_bug(shape, frames):
    """Polars 1.37 matches a null ``by`` key in ``join_asof``, and returns wrong rows for a cross join of two IO
    sources filtered on a null key; such inputs are skipped there."""
    nulls = frames["src"]["key"].null_count() or frames["other"]["key"].null_count()
    if not nulls:
        return False
    if shape == "asof_join":
        return not polars_has("join_asof_null_by_unmatched")
    return shape == "self_join_two_entries" and not polars_has("cross_joins_filtered_sources")


@st.composite
def _frames(draw):
    def rows(n_max):
        return draw(st.lists(st.tuples(st.integers(0, 20), st.one_of(st.none(), st.integers(0, 3)), st.integers(-5, 5)), max_size=n_max))

    def frame(n_max):
        drawn = rows(n_max)
        return pl.DataFrame(
            {
                "data_date": [_JAN_1_2024 + timedelta(days=d) for d, _, _ in drawn],
                "key": [k for _, k, _ in drawn],
                "v": [v for _, _, v in drawn],
                "tags": [[v, v + 1][: 1 + abs(v) % 2] for _, _, v in drawn],
            },
            schema={"data_date": pl.Date, "key": pl.Int64, "v": pl.Int64, "tags": pl.List(pl.Int64)},
        )

    src, other = frame(10), frame(8)
    windows = rows(6)
    # Window endpoints are Dates or midnight Datetimes; a Date request on Datetime endpoints covers the whole day.
    endpoint = draw(st.sampled_from([pl.Date, pl.Datetime("us")]))
    return {
        "src": src,
        "other": other.select(pl.col("data_date").alias("date"), "key", pl.col("v").alias("w")),
        "calendar": pl.DataFrame({"date": [_JAN_1_2024 + timedelta(days=d) for d in range(-2, 24)]}),
        "windows": pl.DataFrame(
            {
                "start": [_JAN_1_2024 + timedelta(days=d) for d, _, _ in windows],
                "end": [_JAN_1_2024 + timedelta(days=d + abs(w)) for d, _, w in windows],
                "key": [k for _, k, _ in windows],
            },
            schema={"start": pl.Date, "end": pl.Date, "key": pl.Int64},
        ).with_columns(pl.col("start", "end").cast(endpoint)),
    }


def _day(draw):
    return _JAN_1_2024 + timedelta(days=draw(st.integers(-2, 23)))


@st.composite
def _combine_predicates(draw, columns):
    def atom():
        col = draw(st.sampled_from(sorted(columns)))
        if col in ("date", "data_date"):
            d, c = _day(draw), pl.col(col)
            return draw(
                st.sampled_from(
                    [
                        c == d,
                        c.is_in([d, d + timedelta(days=draw(st.integers(1, 9)))]),
                        c >= d,
                        c < d,
                        c.is_between(d, d + timedelta(days=draw(st.integers(0, 5)))),
                        c.is_null() | (c == d),
                    ]
                )
            )
        if col == "key":
            k = draw(st.integers(0, 3))
            return draw(st.sampled_from([pl.col("key") == k, pl.col("key").is_in([k, 3 - k]), pl.col("key") > k, pl.col("key").is_null()]))
        return pl.col("name") == _NAMES[draw(st.integers(0, 3))]

    pred = atom()
    for _ in range(draw(st.integers(0, 2))):
        pred = pred & atom() if draw(st.booleans()) else pred | atom()
    return pred


def _combine_without_pushdown(entries, combine, frames):
    return combine({entry: frames[frame].lazy() for entry, (frame, _) in entries.items()})


@SETTINGS
@given(data=st.data(), shape=st.sampled_from(sorted(_SHAPES)))
def test_generated_combine_matches_native(data, shape):
    entries, combine = _SHAPES[shape]
    frames = data.draw(_frames())
    assume(not _old_polars_null_key_bug(shape, frames))
    output = _combine_without_pushdown(entries, combine, frames).collect_schema()
    pred = data.draw(_combine_predicates({"date", "data_date", "key", "name"} & set(output.names())))
    sources = {entry: (PredicateTracker(frames[frame]).lazy_frame, specs) for entry, (frame, specs) in entries.items()}
    result = pushdown_combine(sources, combine).filter(pred).collect()
    assert_frame_equal(result, _combine_without_pushdown(entries, combine, frames).filter(pred).collect(), check_row_order=False)


def _source_rows_needed_for(spec, value):
    """The source rows an output value depends on under ``spec``, as a predicate on a source row."""
    if isinstance(spec, IntervalFilterSpec):
        mapping = spec.value_mapping
        m = value + mapping if isinstance(mapping, timedelta) else mapping(value) if callable(mapping) else value

        def inside(w):
            start, end = w[spec.start_col], w[spec.end_col]
            if start is None or end is None:
                return False
            # Polars compares a Date with a Datetime at midnight.
            at = datetime(m.year, m.month, m.day) if isinstance(start, datetime) and not isinstance(m, datetime) else m
            return start <= at <= end

        return inside
    mapping = spec.value_mapping
    if mapping is None:
        anchors = [value]
    elif callable(mapping):
        anchors = [mapping(value)] + ([value] if spec.lookback or spec.lookahead else [])
    else:
        anchors = [mapping[value]] if value in mapping else []
    if not (spec.lookback or spec.lookahead):
        return lambda w: w is not None and w in anchors
    return lambda w: w is not None and any(a - spec.lookback <= w <= a + spec.lookahead for a in anchors)


@SETTINGS
@given(data=st.data(), shape=st.sampled_from(sorted(_SHAPES)))
def test_generated_combine_depends_only_on_declared_source_rows(data, shape):
    """Deleting source rows that an output row's specs do not declare keeps the number of copies of that row.

    This checks the shapes above declare their specs correctly, so that the comparison with plain Polars tests the
    pushdown rather than the shapes."""
    entries, combine = _SHAPES[shape]
    frames = data.draw(_frames())
    assume(not _old_polars_null_key_bug(shape, frames))
    output = _combine_without_pushdown(entries, combine, frames).collect()
    if output.is_empty():
        return
    # Joins and group_by emit rows in any order; the draws below must not depend on it.
    output = output.sort([c for c, d in output.schema.items() if not d.is_nested()], nulls_last=True, maintain_order=True)
    row = output.row(data.draw(st.integers(0, output.height - 1)), named=True)
    if data.draw(st.booleans()):
        # A tuple that may be absent: the same keys with another date.
        row = {**row, "date": _day(data.draw)} if "date" in row else row
    deleted = {}
    for entry, (frame, specs) in entries.items():
        df = frames[frame]
        # A null output value constrains nothing: a filter accepting null pushes nothing for that column.
        inside = [
            all(
                row.get(o) is None
                or (
                    _source_rows_needed_for(sp, row[o])(w)
                    if isinstance(sp, IntervalFilterSpec)
                    else _source_rows_needed_for(sp, row[o])(w[sp.source_col or o])
                )
                for o, sp in specs.items()
            )
            for w in df.iter_rows(named=True)
        ]
        keep = [i or data.draw(st.booleans()) for i in inside]
        deleted[entry] = df.filter(pl.Series(keep, dtype=pl.Boolean)) if keep else df
    after = combine({entry: deleted[entry].lazy() for entry in entries}).collect()

    def count(frame):
        return frame.filter(pl.all_horizontal([pl.col(c).eq_missing(pl.lit(v, dtype=frame.schema[c])) for c, v in row.items()])).height

    assert count(after) == count(output)


@dataclass(frozen=True)
class ExpansionPastExtremesCase:
    """A value at the edge of its dtype's range, a filter, and a spec that widens the filter past that edge."""

    id: str
    _: KW_ONLY
    value: object
    pred: pl.Expr
    spec: FilterSpec


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            ExpansionPastExtremesCase(
                "le_date_max_lookahead", value=date.max, pred=pl.col("d") <= date.max, spec=FilterSpec(lookahead=timedelta(days=1))
            ),
            ExpansionPastExtremesCase(
                "eq_date_max_lookahead", value=date.max, pred=pl.col("d") == date.max, spec=FilterSpec(lookahead=timedelta(days=1))
            ),
            ExpansionPastExtremesCase(
                "ge_date_min_lookback", value=date.min, pred=pl.col("d") >= date.min, spec=FilterSpec(lookback=timedelta(days=1))
            ),
            ExpansionPastExtremesCase(
                "eq_datetime_max_lookahead",
                value=datetime(9999, 12, 31),
                pred=pl.col("d") == datetime(9999, 12, 31),
                spec=FilterSpec(lookahead=timedelta(days=1)),
            ),
        ]
    ),
)
def test_pushdown_combine_expansion_past_extremes(case: ExpansionPastExtremesCase):
    df = pl.DataFrame({"d": [case.value], "v": [1]})
    lf = pushdown_combine(sources={"data": (df.lazy(), {"d": case.spec})}, combine=lambda s: s["data"])
    assert_matches_plain_polars(lf, df.lazy(), case.pred, check_row_order=True)


@pytest.mark.parametrize(
    "predicate",
    [pl.col("day") <= date.max, pl.col("day") == date.max, pl.col("day") > date.max],
    ids=["inclusive_upper", "equality", "exclusive_lower"],
)
def test_pushdown_max_date_on_datetime_source(predicate):
    # date.max is a common open-ended validity sentinel. A closed bound must include intraday values on that date.
    frame = pl.LazyFrame({"ts": [datetime(9999, 12, 30, 12), datetime(9999, 12, 31, 12)]})

    def combine(sources):
        return sources["data"].with_columns(pl.col("ts").dt.date().alias("day"))

    mapped = pushdown_combine(sources={"data": (frame, {"day": FilterSpec(source_col="ts")})}, combine=combine)
    assert_matches_plain_polars(mapped, combine({"data": frame}), predicate, check_row_order=True)


class TestPushdownCombinePickle:
    """Tests for pushdown_combine pickle support."""

    def test_pushdown_combine_pickle_basic(self):
        """pushdown_combine LazyFrames can be pickled and unpickled."""
        left_df = pl.DataFrame(
            {
                "date": [date(2025, 1, 1), date(2025, 1, 2)],
                "id": [1, 2],
                "value_left": [10, 20],
            }
        )
        right_df = pl.DataFrame(
            {
                "date": [date(2025, 1, 1), date(2025, 1, 2)],
                "id": [1, 2],
                "value_right": [100, 200],
            }
        )

        lf = cpl.pushdown_combine(
            sources={
                "left": (left_df.lazy(), {"date": cpl.FilterSpec(), "id": cpl.FilterSpec()}),
                "right": (right_df.lazy(), {"date": cpl.FilterSpec(), "id": cpl.FilterSpec()}),
            },
            combine=lambda s: s["left"].join(s["right"], on=["date", "id"]),
        )
        lf_unpickled = pickle_round_trip(lf)

        expected = left_df.join(right_df, on=["date", "id"])
        result = lf_unpickled.collect()
        assert_frame_equal(result, expected, check_row_order=False)
