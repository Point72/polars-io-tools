"""pushdown_combine: scenario rows run by one test, and tables for FilterSpec, IntervalFilterSpec and the helpers.

A scenario names its sources (a frame and its specs), the combine function and the user's predicate. The result must
match combine over the unfiltered sources filtered by the predicate, unless the row sets ``compare_native=False``; with
the identity combine this also means the source kept every row the predicate keeps. A row may pin what a source was
asked for: ``pushed`` sources received a predicate, ``bounds`` are the temporal bounds on a column, ``discrete`` the
values on a column, and ``no_discrete`` a column with no discrete filter. ``check`` runs on the result and the trackers.
Sources a row inspects, and the source of the identity combine, sit behind a ``PredicateTracker``.
"""

import sys
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from polars_io_tools.io_sources.base import BinaryExprNode, FunctionNode
from polars_io_tools.io_sources.enum import BooleanFunctionType, OperatorType
from polars_io_tools.io_sources.pushdown_combine import (
    FilterSpec,
    IntervalFilterSpec,
    _apply_value_mapping,
    _compute_output_schema,
    _get_source_col,
    _union_discrete_filter,
    pushdown_combine,
)
from polars_io_tools.testing import PredicateAnalyzer, PredicateTracker, io_source_assert
from polars_io_tools.tests.helpers.assertions import assert_matches_native, assert_pushed_bounds
from polars_io_tools.tests.helpers.cases import GROUP_FRAME, Case, daily_frame, params

c = pl.col
UTC_ZONE = ZoneInfo("UTC")
NY = ZoneInfo("America/New_York")
CLOSED = ["both", "left", "right", "none"]


def d(day: int, month: int = 1, year: int = 2024) -> date:
    return date(year, month, day)


def days(n: float) -> timedelta:
    return timedelta(days=n)


# Specs, values and combine functions -----------------------------------------------------------------------------------
F = FilterSpec
REGIONS = {"NA": "NORTH_AMERICA", "EU": "EUROPE", "APAC": "ASIA_PACIFIC", "LATAM": "LATIN_AMERICA"}


def to_codes(*codes: str) -> dict:
    return {REGIONS[code]: code for code in codes}


def region_frame(*codes: str) -> pl.DataFrame:
    return pl.DataFrame({"region_code": list(codes), "val": list(range(1, len(codes) + 1))})


def identity(s):
    return s["data"]


def regions(s):
    return s["data"].with_columns(c("region_code").replace(REGIONS).alias("region")).drop("region_code")


def lag(n: int):
    return lambda s: s["data"].with_columns(c("val").shift(n).alias(f"lag{n}"))


def lead(n: int):
    return lambda s: s["data"].with_columns(c("val").shift(-n).alias(f"lead{n}"))


def join(on, how: str = "inner", **kw):
    return lambda s: s["left"].join(s["right"], on=on, how=how, **kw)


def one(df: pl.DataFrame, **specs) -> dict:
    return {"data": (df, specs)}


def two(left: pl.DataFrame, right: pl.DataFrame, left_specs: dict, right_specs: dict | None = None) -> dict:
    return {"left": (left, left_specs), "right": (right, left_specs if right_specs is None else right_specs)}


@dataclass(frozen=True)
class Scenario:
    """Sources (a frame and its specs per name), the combine function and the user's predicate.

    ``predicate`` may be a list of predicates collected in turn from the same frame. ``bounds``, ``discrete`` and
    ``discrete_node`` are keyed by source name, then column.
    """

    id: str
    sources: dict
    combine: Callable
    predicate: Any = None
    _: KW_ONLY
    compare_native: bool = True
    # defaults to True for the identity combine, which keeps the source's row order
    check_row_order: bool | None = None
    # sources that must receive a predicate
    pushed: tuple[str, ...] = ()
    # (lower, upper) temporal bounds, as in ``assert_pushed_bounds``
    bounds: dict[str, dict[str, tuple]] = field(default_factory=dict)
    # the values the first discrete filter admits
    discrete: dict[str, dict[str, set]] = field(default_factory=dict)
    # "eq" or "is_in": the node shape of that filter
    discrete_node: dict[str, dict[str, str]] = field(default_factory=dict)
    # columns with no discrete filter
    no_discrete: dict[str, set[str]] = field(default_factory=dict)
    # runs on the result and the trackers
    check: Callable | None = None
    # (exception type, match) that collecting raises
    raises: tuple | None = None
    # applied to the filtered frame and to the reference
    post: Callable | None = None
    combine_kwargs: dict | None = None
    sources_as_kwargs: bool = False
    marks: tuple = ()


# Checks on the result and the trackers
def pushed_text(source: str, has=(), lacks=()):
    def check(_got, trackers):
        pushed = str(trackers[source].last_predicate)
        for text in has:
            assert text in pushed, pushed
        for text in lacks:
            assert text not in pushed, pushed

    return check


def assert_discrete(tracker: PredicateTracker, col: str, values: set, node: str | None = None) -> None:
    """The first discrete filter on ``col`` admits ``values``; ``node`` is ``"eq"`` or ``"is_in"`` to pin its shape.

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
DAILY10 = daily_frame(10, base=0)
DAILY14 = daily_frame(14)
OTHER10 = DAILY10.rename({"val": "other"}).with_columns(c("other") + 10)
GROUPS = pl.DataFrame({"date": daily_frame(10)["date"], "group": ["A", "B"] * 5, "val": range(10)})
GROUP_PAIR = (GROUP_FRAME, GROUP_FRAME.rename({"val": "other"}))
INTRADAY = pl.DataFrame(
    {"timestamp": [datetime(2024, 1, day, hour) for day in (1, 2, 3) for hour in (10, 14)], "value": [1, 2, 3, 4, 5, 6]},
)
FIVE = daily_frame(5)


def as_date(s):
    return s["main"].with_columns(c("timestamp").cast(pl.Date).alias("data_date"))


def intraday(pred, **spec):
    return {"sources": {"main": (INTRADAY, {"data_date": F(source_col="timestamp", **spec)})}, "combine": as_date, "predicate": pred}


def scaled(sources, multiplier):
    return sources["data"].with_columns((c("val") * multiplier).alias("scaled_val"))


def with_mapping(sources, code_to_region):
    return sources["data"].with_columns(c("code").replace(code_to_region).alias("region"))


def multiple_args(sources, multiplier, suffix, constant):
    return sources["data"].with_columns((c("val") * multiplier).alias(f"scaled_{suffix}"), pl.lit(constant).alias("const_col"))


def join_how(sources, join_how):
    return sources["left"].join(sources["right"], on="key", how=join_how)


def primary_with_reference(sources, category_to_code):
    primary = sources["primary"].with_columns(c("category").replace(category_to_code).alias("category_code"))
    return primary.join(sources["reference"], on=["date", "category_code"], how="left")


def kw_join(left, right):
    return left.join(right, on="date")


def kw_scaled(data, multiplier, suffix):
    return data.with_columns((c("val") * multiplier).alias(f"scaled_{suffix}"))


def kw_lag(data):
    return data.with_columns(c("val").shift(3).alias("lag3"))


def kw_three(prices, volumes, metadata):
    return prices.join(volumes, on="date").join(metadata, on="date")


def kw_regions(data):
    return regions({"data": data})


def kw_primary_with_reference(primary, reference, category_mapping):
    return primary_with_reference({"primary": primary, "reference": reference}, category_mapping)


KEY_LEFT = pl.DataFrame({"date": FIVE["date"], "key": ["A", "B", "C", "D", "E"], "left_val": [1, 2, 3, 4, 5]})
KEY_RIGHT = pl.DataFrame({"key": ["A", "C", "E"], "right_val": [100, 300, 500]})
PRIMARY = pl.DataFrame({"date": FIVE["date"], "category": ["CATEGORY_A"] * 5, "item_id": range(5)})
REFERENCE = pl.DataFrame({"date": FIVE["date"], "category_code": ["A"] * 5, "value": [0.05 + i * 0.001 for i in range(5)]})
CATEGORY_TO_CODE = {"CATEGORY_A": "A", "CATEGORY_B": "B"}
PRIMARY_REFERENCE = {
    "primary": (PRIMARY, {"date": F(), "category": F()}),
    "reference": (REFERENCE, {"date": F(lookback=days(2)), "category": F(source_col="category_code", value_mapping=CATEGORY_TO_CODE)}),
}


def three_way(s):
    primary = s["primary"].join(s["mapping"], on="category", how="left")
    return primary.join(s["reference"], on=["date", "category_code"], how="left")


THREE_WAY = {
    "primary": (pl.DataFrame({"date": daily_frame(10)["date"], "category": ["CAT_A"] * 10, "item_id": range(10)}), {"date": F(), "category": F()}),
    "reference": (
        pl.DataFrame(
            {"date": daily_frame(19, start=d(27, 12, 2023))["date"], "category_code": ["A"] * 19, "value": [0.05 + i * 0.001 for i in range(19)]}
        ),
        {"date": F(lookback=days(4)), "category": F(source_col="category_code", value_mapping={"CAT_A": "A", "CAT_B": "B"})},
    ),
    "mapping": (pl.DataFrame({"category": ["CAT_A", "CAT_B"], "category_code": ["A", "B"]}), {"category": F()}),
}


# IntervalFilterSpec: a schedule of [start, end] windows joined to a spine that carries the requested ``date``.
def schedule(starts, ends, vals=None, roots=None) -> pl.DataFrame:
    return pl.DataFrame({"root": roots or ["A"] * len(starts), "start": starts, "end": ends, "val": vals or list(range(1, len(starts) + 1))})


def spine(*dates, roots=None) -> pl.DataFrame:
    return pl.DataFrame({"root": roots or ["A"] * len(dates), "date": list(dates)})


def windows(sched, spine_df, **spec):
    return {"sched": (sched, {"date": IntervalFilterSpec(start_col="start", end_col="end", **spec)}), "spine": (spine_df, {"date": F()})}


def interval_join(membership=...):
    """Join the spine to the schedule; ``...`` keeps rows whose date is in their window, ``None`` keeps every pair."""

    def combine(s):
        joined = s["spine"].join(s["sched"], on="root")
        if membership is None:
            return joined
        return joined.filter((c("start") <= c("date")) & (c("date") <= c("end")) if membership is ... else membership)

    return combine


def previous_weekday(day):
    return day - days(3 if day.weekday() == 0 else 1)


SCHED_11_20 = schedule([d(11)], [d(20)], [2])
SCHED_11_20_DT = SCHED_11_20.with_columns(c("start", "end").cast(pl.Datetime("us")))
SEP_SCHED = schedule([d(1, 8), d(31, 8)], [d(30, 8), d(30, 9)])
SEP_SPINE = spine(d(1, 9), d(2, 9), d(3, 9))
SEP_MEMBERSHIP = c("date").map_elements(previous_weekday, return_dtype=pl.Date).is_between(c("start"), c("end"))
INTRADAY_SCHED = schedule(
    [datetime(2024, 1, 1, 9), datetime(2024, 1, 15, 15)], [datetime(2024, 1, 15, 9), datetime(2024, 1, 20, 9)], roots=["A", "B"]
)


def endpoint_row(closed: str, keep_start_eq_hi: bool, keep_end_eq_lo: bool) -> Scenario:
    """A window starting on the requested day is kept iff its start is closed (val 1); one ending there iff its end is (val 2)."""
    kept = {1} if keep_start_eq_hi else set()
    kept |= {2} if keep_end_eq_lo else set()

    def check(got, _trackers):
        assert set(got["val"]) == kept, got

    sched = schedule([d(15), d(1)], [d(20), d(15)], roots=["A", "B"])
    spine_df = spine(d(15), d(15), roots=["A", "B"])
    return Scenario(
        f"date_endpoints-{closed}",
        windows(sched, spine_df, closed=closed),
        interval_join(pl.lit(True)),
        c("date") == d(15),
        compare_native=False,
        check=check,
    )


def shift_bounds_row(id, pred, end_lower, start_upper):
    """A timedelta value_mapping shifts both pushed interval bounds by a day."""
    is_datetime = isinstance(end_lower, datetime)
    sched, spine_df = (SCHED_11_20_DT, spine(datetime(2024, 1, 13))) if is_datetime else (SCHED_11_20, spine(d(13)))
    start = (f'(col("start")) <= ({start_upper})',) if start_upper is not None else ()
    return Scenario(
        f"interval_timedelta_mapping_shifts_bounds-{id}",
        windows(sched, spine_df, value_mapping=days(-1)),
        interval_join(None),
        pred,
        check=pushed_text("sched", has=(f'(col("end")) >= ({end_lower})', *start), lacks=() if start else ('col("start")',)),
    )


# Two or more output columns from one source column union their discrete filters.
THREE_GROUPS = pl.DataFrame({"data_date": [d(1)] * 3, "group": ["A", "B", "C"], "val": [1.0, 2.0, 3.0]})
PAIR_SPECS = {"data_date": F(), "group1": F(source_col="group"), "group2": F(source_col="group")}


def group_pairs(s):
    left = s["data"].rename({"group": "group1", "val": "val1"})
    right = s["data"].select("data_date", c("group").alias("group2"), c("val").alias("val2"))
    return left.join(right, on="data_date").filter(c("group1") != c("group2"))


def cross_aliases(col: str, *names: str, keep: tuple = ()):
    """Cross-join one source with itself, ``col`` renamed to each of ``names``; ``keep`` columns stay on the first side."""

    def combine(s):
        out = s["data"].select(c(col).alias(names[0]), *keep)
        for name in names[1:]:
            out = out.join(s["data"].select(c(col).alias(name)), how="cross")
        return out

    return combine


def mapped_cross(s):
    code_to_name = {"a": "A", "b": "B", "c": "C"}
    left = s["data"].with_columns(c("code").replace(code_to_name).alias("name1")).drop("code")
    right = s["data"].with_columns(c("code").replace(code_to_name).alias("name2")).drop("code")
    return left.join(right, how="cross")


GROUP_OTHER = pl.DataFrame({"group": ["A", "B", "C"], "other": ["x", "y", "z"]})
GROUP_OTHER_SPECS = {"group1": F(source_col="group"), "group2": F(source_col="group"), "other": F()}
GROUP_OTHER_PRED = (c("group1") == "A") & (c("group2") == "B") & (c("other") == "x")
THREE_SPECS = {f"group{i}": F(source_col="group") for i in (1, 2, 3)}
NAME_MAPPING = {"A": "a", "B": "b", "C": "c"}
CODES = pl.DataFrame({"code": ["a", "b", "c"]})
THREE_DATES = pl.DataFrame({"d": [d(1), d(2), d(3)], "val": [1, 2, 3]})

# Rows a pushed predicate could drop through a cast, a time zone, a unit or a null.
_TS = pl.DataFrame({"t": [datetime(2023, 12, 31, 23), datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 0, 0, 500), datetime(2024, 1, 1, 13), None]})
_DATES = pl.DataFrame({"t": [d(31, 12, 2023), d(1), d(2)]})
_DEC = pl.DataFrame({"t": pl.Series([Decimal("1.25"), Decimal("2.50"), Decimal("3.75")], dtype=pl.Decimal(10, 2))})
_TS_NY = _TS.with_columns(c("t").dt.replace_time_zone("UTC").dt.convert_time_zone("America/New_York"))
MATCHES_FILTER = [
    (_TS, c("t").cast(pl.Date) < datetime(2024, 1, 1, 12), F()),
    (_TS, c("t").dt.date() < datetime(2024, 1, 1, 12), F()),
    (_TS, c("t").cast(pl.Datetime("ms")) <= datetime(2024, 1, 1), F()),
    (_TS, c("t").is_in([datetime(2024, 1, 1), None], nulls_equal=True), F()),
    (_TS, c("t").is_in([None], nulls_equal=True), F()),
    (_DATES, c("t") < datetime(2024, 1, 1, 12), F()),
    (_DATES, c("t") < datetime(2024, 1, 1, 12), F(lookback=days(1))),
    (_DATES, c("t").is_between(datetime(2023, 12, 1), datetime(2024, 1, 1, 12), closed="left"), F()),
    (_TS_NY, c("t") >= datetime(2024, 1, 1, tzinfo=NY), F()),
    (_TS_NY, c("t") < datetime(2024, 1, 1, tzinfo=NY), F()),
    (_TS_NY, c("t").is_between(datetime(2023, 12, 31, tzinfo=NY), datetime(2024, 1, 1, tzinfo=NY)), F()),
    (_TS_NY, c("t") == datetime(2023, 12, 31, 19, tzinfo=NY), F()),
    (_TS_NY, c("t").is_in([datetime(2023, 12, 31, 19, tzinfo=NY)]), F()),
    (_TS_NY, c("t").cast(pl.Date) <= d(31, 12, 2023), F()),
    (_TS_NY, c("t").dt.date() == d(1), F()),
    (_DEC, c("t").is_in(pl.lit([Decimal("1.25"), Decimal("2.50")], dtype=pl.List(pl.Decimal(10, 2)))), F()),
]
FLOAT_CASTS = [
    (c("x") == 16777216.0, pl.Float32),
    ((c("x") == 16777216.0) | (c("x") == 5.0), pl.Float32),
    (c("x") == 1e300, pl.Float32),
    (c("x") == 16777216.0, pl.Float64),
    (c("x") == 1e300, pl.Float64),
    (c("x") == sys.float_info.max, pl.Float32),
    (c("x") == sys.float_info.max, pl.Float64),
    ((c("x") == 16777216.0) | (c("x") == 5.0), pl.Float64),
]
NULL_PREDICATES = {"is_null": c("a").is_null(), "eq_missing": c("a").eq_missing(None), "is_in_nulls_equal": c("a").is_in([1, None], nulls_equal=True)}
NULLABLE = pl.DataFrame({"a": [1, None, 3], "b": [1, None, 3]})


def cast_to(dtype):
    return lambda s: s["data"].with_columns(c("x").cast(dtype))


# A value_mapping with a window must push a window around the mapped values, never only the exact mapped value.
DAYS20 = pl.DataFrame({"data_date": daily_frame(20)["date"], "v": list(range(20))})


def next_day_rolling(s):
    return s["data"].with_columns(c("v").rolling_sum(3).alias("r"), (c("data_date") + pl.duration(days=1)).alias("date"))


def same_day_rolling(s):
    return s["data"].with_columns(c("v").rolling_sum(2).alias("r"), c("data_date").alias("date"))


def mapped_self_join(s):
    side = s["data"].with_columns(c("v").rolling_sum(2).alias("r"), (c("data_date") + pl.duration(days=1)).alias("d"))
    return side.select(c("d").alias("d1"), c("r").alias("r1")).join(side.select(c("d").alias("d2"), c("r").alias("r2")), how="cross")


def window(**kw) -> FilterSpec:
    return F(source_col="data_date", **kw)


def weekday_window(lookback: int) -> FilterSpec:
    return window(lookback=days(lookback), value_mapping=previous_weekday)


def day_of_month_key(s):
    return s["data"].with_columns(c("v").rolling_sum(2).alias("r"), c("data_date").dt.day().cast(pl.Int64).add(1).alias("k"))


def unbounded_sharer(s):
    return s["data"].select(c("d").alias("a"), c("v").cum_sum().alias("c")).join(s["data"].select(c("d").alias("b")), how="cross")


TENDAYS = pl.DataFrame({"d": daily_frame(10)["date"], "v": list(range(10))})
FOLD = [datetime(2024, 11, 3, 1, 30, tzinfo=NY, fold=0), datetime(2024, 11, 3, 1, 30, tzinfo=NY, fold=1)]
NEAR_MAX = [
    datetime(9999, 12, 31, 23, 59, 59, 999998, tzinfo=timezone(timedelta(hours=-5))),
    datetime(9999, 12, 31, 23, 59, 59, 999999, tzinfo=timezone(timedelta(hours=-5))),
]
US_MEMBERS = pl.Series([datetime(2024, 1, 1), datetime(2024, 1, 2)], dtype=pl.Datetime("us", "UTC"))


def is_in_units_row(unit: str, time_zone: str | None) -> Scenario:
    tz = ZoneInfo(time_zone) if time_zone else None
    values = pl.Series([datetime(2024, 1, 1, 0, 0, s, tzinfo=tz) for s in range(5)], dtype=pl.Datetime(unit, time_zone))
    # The collection has the column's own dtype: Polars rejects a microsecond list against a millisecond column.
    return Scenario(
        f"is_in_on_any_datetime_unit-{time_zone}-{unit}", one(values.to_frame("t"), t=F()), identity, c("t").is_in(values.gather([1, 3]).implode())
    )


SCENARIOS = [
    # Propagation through a join
    Scenario("no_filters", two(pl.DataFrame({"id": [1, 2], "val": [10, 20]}), pl.DataFrame({"id": [1, 2], "other": [100, 200]}), {}), join("id")),
    Scenario(
        "simple_date_filter", two(DAILY10, OTHER10, {"date": F()}), join("date", maintain_order="left"), c("date") >= d(5), check_row_order=True
    ),
    Scenario(
        "dt_date_filter_prunes_datetime_source",
        one(daily_frame(90, dtype=pl.Datetime("us")), date=F()),
        identity,
        (c("date").dt.date() >= d(1)) & (c("date").dt.date() <= d(31)),
        pushed=("data",),
    ),
    Scenario("simple_equality_filter", two(*GROUP_PAIR, {"group": F()}), join("group"), c("group") == "B"),
    Scenario("is_in_filter", two(*GROUP_PAIR, {"group": F()}), join("group"), c("group").is_in(["A", "C"])),
    # Lookback and lookahead
    Scenario("lookback_with_date_equality", one(DAILY14, date=F(lookback=days(4))), lag(3), c("date") == d(5)),
    Scenario("lookahead_with_date_equality", one(DAILY14, date=F(lookahead=days(4))), lead(3), c("date") == d(5)),
    Scenario(
        "lookback_and_lookahead_with_date_equality",
        one(DAILY14, date=F(lookback=days(3), lookahead=days(3))),
        lambda s: s["data"].with_columns(c("val").shift(2).alias("lag2"), c("val").shift(-2).alias("lead2")),
        c("date") == d(7),
    ),
    Scenario(
        "lookback_expands_date_range",
        two(daily_frame(10, base=100), OTHER10, {"date": F(lookback=days(3))}, {"date": F()}),
        join("date", maintain_order="left"),
        c("date") >= d(5),
        check_row_order=True,
        pushed=("left",),
    ),
    Scenario(
        "lookback_for_rolling_computation",
        one(DAILY10.with_columns(c("val").cast(pl.Float64)), date=F(lookback=days(3))),
        lambda s: s["data"].sort("date").with_columns(c("val").rolling_sum(window_size=3, min_samples=1).alias("rolling_sum")),
        c("date") >= d(5),
    ),
    Scenario("lookahead_expands_date_range", one(daily_frame(10, base=100), date=F(lookahead=days(2))), identity, c("date") <= d(5)),
    Scenario(
        "combined_lookback_lookahead",
        one(daily_frame(10, base=100), date=F(lookback=days(2), lookahead=days(2))),
        identity,
        c("date").is_between(d(4), d(7)),
    ),
    # Pushed bounds and values
    Scenario("date_gte_filter_structure", one(DAILY10, date=F()), identity, c("date") >= d(5), bounds={"data": {"date": (d(5), ...)}}),
    Scenario(
        "date_between_filter_structure", one(DAILY10, date=F()), identity, c("date").is_between(d(3), d(7)), bounds={"data": {"date": (d(3), d(7))}}
    ),
    Scenario(
        "lookback_expands_date_range_correctly",
        one(DAILY10, date=F(lookback=days(3))),
        identity,
        c("date") >= d(5),
        bounds={"data": {"date": (d(2), ...)}},
    ),
    Scenario(
        "lookback_with_is_between_and_complex_combine",
        one(DAILY14, date=F(lookback=days(4))),
        lag(3),
        c("date").is_between(d(5), d(10)),
        bounds={"data": {"date": (d(1), d(10))}},
    ),
    # Polars may fold the widened filter into the user's one when the combine is the identity.
    Scenario(
        "lookback_optimization_with_identity_combine",
        one(DAILY14, date=F(lookback=days(4))),
        identity,
        c("date").is_between(d(5), d(10)),
        bounds={"data": {"date": ((d(1), d(5)), d(10))}},
    ),
    Scenario(
        "lookahead_expands_upper_bound", one(DAILY14, date=F(lookahead=days(3))), identity, c("date") <= d(7), bounds={"data": {"date": (..., d(10))}}
    ),
    Scenario(
        "discrete_equality_filter_structure",
        one(GROUP_FRAME, group=F()),
        identity,
        c("group") == "B",
        discrete={"data": {"group": {"B"}}},
        discrete_node={"data": {"group": "eq"}},
    ),
    Scenario(
        "discrete_is_in_filter_structure",
        one(GROUP_FRAME, group=F()),
        identity,
        c("group").is_in(["A", "C"]),
        discrete={"data": {"group": {"A", "C"}}},
        discrete_node={"data": {"group": "is_in"}},
    ),
    # Value mapping and source column renames
    Scenario(
        "dict_value_mapping",
        one(region_frame("NA", "EU"), region=F(source_col="region_code", value_mapping=to_codes("NA", "EU"))),
        regions,
        c("region") == "NORTH_AMERICA",
        discrete={"data": {"region_code": {"NA"}}},
        discrete_node={"data": {"region_code": "eq"}},
    ),
    Scenario(
        "callable_value_mapping",
        one(pl.DataFrame({"name": ["foo", "bar", "baz"], "val": [1, 2, 3]}), name=F(value_mapping=str.lower)),
        lambda s: s["data"].with_columns(c("name").str.to_uppercase()),
        c("name") == "FOO",
    ),
    Scenario(
        "is_in_with_value_mapping",
        one(region_frame("NA", "EU", "APAC"), region=F(source_col="region_code", value_mapping=to_codes("NA", "EU", "APAC"))),
        regions,
        c("region").is_in(["NORTH_AMERICA", "ASIA_PACIFIC"]),
        discrete={"data": {"region_code": {"NA", "APAC"}}},
        discrete_node={"data": {"region_code": "is_in"}},
    ),
    # An unmapped value is not pushed: the source supplies every row and the user's filter decides.
    Scenario(
        "unmapped_value_not_pushed_down",
        one(region_frame("NA", "EU", "APAC", "LATAM"), region=F(source_col="region_code", value_mapping=to_codes("NA", "EU"))),
        regions,
        c("region") == "LATIN_AMERICA",
    ),
    Scenario(
        "mixed_mapped_and_unmapped_values_in_is_in",
        one(region_frame("NA", "EU", "APAC", "LATAM"), region=F(source_col="region_code", value_mapping=to_codes("NA", "EU"))),
        regions,
        c("region").is_in(["NORTH_AMERICA", "LATIN_AMERICA"]),
    ),
    Scenario(
        "different_source_column_name",
        one(DAILY10.rename({"date": "DataDate"}), date=F(source_col="DataDate")),
        lambda s: s["data"].rename({"DataDate": "date"}),
        c("date") == d(2),
        discrete={"data": {"DataDate": {d(2)}}},
        discrete_node={"data": {"DataDate": "eq"}},
    ),
    # Several sources with their own specs
    Scenario(
        "different_lookback_per_source",
        {"source1": (DAILY10.rename({"val": "val1"}), {"date": F(lookback=days(5))}), "source2": (OTHER10, {"date": F(lookback=days(2))})},
        lambda s: s["source1"].join(s["source2"], on="date", maintain_order="right"),
        c("date") >= d(7),
        check_row_order=True,
        bounds={"source1": {"date": (d(2), ...)}, "source2": {"date": (d(5), ...)}},
    ),
    Scenario(
        "mixed_filter_types_per_source",
        {
            "left": (GROUPS.head(5), {"date": F(lookback=days(2)), "group": F()}),
            "right": (GROUPS.head(5).rename({"val": "val2"}), {"date": F(), "group": F()}),
        },
        join(["date", "group"], maintain_order="left"),
        (c("date") >= d(3)) & (c("group") == "A"),
        check_row_order=True,
    ),
    Scenario("pushdown_combine_join_with_mapping_and_lookback", THREE_WAY, three_way, c("date").is_between(d(5), d(7)) & (c("category") == "CAT_A")),
    # Edge cases
    Scenario("empty_result", one(daily_frame(1), date=F()), identity, c("date") == d(31, 12)),
    Scenario("no_filter_specs", one(pl.DataFrame({"id": [1, 2], "val": [10, 20]})), identity, c("id") == 1),
    Scenario("source_col_not_in_schema", one(pl.DataFrame({"id": [1, 2], "val": [10, 20]}), nonexistent=F(source_col="also_nonexistent")), identity),
    Scenario("multiple_filters_combined", one(GROUPS, date=F(), group=F()), identity, (c("date") >= d(5)) & (c("group") == "A")),
    Scenario(
        "datetime_column",
        one(pl.DataFrame({"ts": [datetime(2024, 1, 1, h) for h in range(24)], "val": range(24)}), ts=F(lookback=timedelta(hours=3))),
        identity,
        c("ts") >= datetime(2024, 1, 1, 12),
    ),
    Scenario(
        "column_selection_with_columns",
        one(pl.DataFrame({"date": [d(1), d(2)], "a": [1, 2], "b": [10, 20], "c": [100, 200]}), date=F()),
        identity,
        post=lambda lf: lf.select("date", "a"),
    ),
    Scenario("row_limit", one(DAILY10, date=F()), identity, post=lambda lf: lf.head(3)),
    Scenario(
        "filter_on_column_without_spec",
        one(FIVE.with_columns(unspecified_col=pl.Series(list("ABCDE"))), date=F()),
        identity,
        c("unspecified_col") == "C",
    ),
    Scenario("complex_predicate_with_or", one(DAILY10, date=F()), identity, (c("date") == d(3)) | (c("date") == d(7))),
    Scenario("nested_filters_with_and", one(GROUPS, date=F(lookback=days(2)), group=F()), identity, (c("date") >= d(5)) & (c("group") == "A")),
    Scenario(
        "filter_with_null_values", one(pl.DataFrame({"date": [d(1), d(2), None, d(4)], "val": [1, 2, 3, 4]}), date=F()), identity, c("date") >= d(2)
    ),
    Scenario("multiple_collects", one(DAILY10, date=F()), identity, [c("date") >= d(5), c("date") <= d(3)]),
    Scenario("empty_source", one(daily_frame(0), date=F()), identity, c("date") >= d(1)),
    Scenario("very_large_lookback", one(DAILY10, date=F(lookback=days(365))), identity, c("date") >= d(5)),
    Scenario(
        "combine_with_additional_columns",
        one(FIVE, date=F()),
        lambda s: s["data"].with_columns((c("val") * 2).alias("val_doubled"), pl.lit("constant").alias("const_col")),
        c("date") >= d(3),
    ),
    Scenario(
        "combine_drops_columns",
        one(FIVE.with_columns(drop_col=pl.Series(list("abcde"))), date=F()),
        lambda s: s["data"].drop("drop_col"),
        c("date") >= d(3),
    ),
    # combine_kwargs and sources_as_kwargs
    Scenario("combine_kwargs_basic", one(FIVE, date=F()), scaled, c("date") >= d(3), combine_kwargs={"multiplier": 10}),
    Scenario(
        "combine_kwargs_with_mapping_dict",
        one(pl.DataFrame({"date": [d(1), d(2)], "code": ["US", "CN"], "val": [100, 200]}), date=F()),
        with_mapping,
        combine_kwargs={"code_to_region": {"US": "North America", "CN": "Asia"}},
    ),
    Scenario(
        "combine_kwargs_multiple_args",
        one(FIVE, date=F()),
        multiple_args,
        combine_kwargs={"multiplier": 2, "suffix": "doubled", "constant": "hello"},
        check_row_order=True,
    ),
    Scenario("combine_kwargs_with_none", one(FIVE, date=F()), identity, combine_kwargs=None),
    Scenario("combine_kwargs_empty_dict", one(FIVE, date=F()), identity, combine_kwargs={}),
    Scenario(
        "combine_kwargs_complex_join_pattern",
        PRIMARY_REFERENCE,
        primary_with_reference,
        (c("date") >= d(2)) & (c("category") == "CATEGORY_A"),
        combine_kwargs={"category_to_code": CATEGORY_TO_CODE},
    ),
    Scenario(
        "sources_as_kwargs_basic",
        two(FIVE.rename({"val": "left_val"}), FIVE.rename({"val": "right_val"}), {"date": F()}),
        kw_join,
        c("date") >= d(3),
        sources_as_kwargs=True,
    ),
    Scenario(
        "sources_as_kwargs_with_combine_kwargs",
        one(FIVE, date=F()),
        kw_scaled,
        combine_kwargs={"multiplier": 10, "suffix": "x10"},
        sources_as_kwargs=True,
        check_row_order=True,
    ),
    Scenario("sources_as_kwargs_with_lookback", one(DAILY14, date=F(lookback=days(4))), kw_lag, c("date") == d(5), sources_as_kwargs=True),
    Scenario(
        "sources_as_kwargs_multiple_sources",
        {
            "prices": (FIVE.rename({"val": "price"}), {"date": F()}),
            "volumes": (FIVE.rename({"val": "volume"}), {"date": F()}),
            "metadata": (FIVE.with_columns(category=pl.Series(list("AABBA"))).drop("val"), {"date": F()}),
        },
        kw_three,
        c("date") >= d(3),
        sources_as_kwargs=True,
    ),
    Scenario("sources_as_kwargs_false_is_default", one(FIVE, date=F()), identity),
    Scenario(
        "sources_as_kwargs_with_value_mapping",
        one(region_frame("NA", "EU"), region=F(source_col="region_code", value_mapping=to_codes("NA", "EU"))),
        kw_regions,
        c("region") == "NORTH_AMERICA",
        sources_as_kwargs=True,
    ),
    Scenario(
        "sources_as_kwargs_complex_join_pattern",
        {
            "primary": (PRIMARY.with_columns(category=pl.lit("A")), {"date": F(), "category": F()}),
            "reference": (REFERENCE, {"date": F(lookback=days(2)), "category": F(source_col="category_code", value_mapping={"A": "A", "B": "B"})}),
        },
        kw_primary_with_reference,
        (c("date") >= d(2)) & (c("category") == "A"),
        combine_kwargs={"category_mapping": {"A": "A", "B": "B"}},
        sources_as_kwargs=True,
    ),
    # A Date filter linked to a Datetime source keeps every intraday row of the requested days.
    Scenario("eq_covers_full_day", **intraday(c("data_date") == d(2))),
    Scenario("eq_with_lookback_covers_full_day", **intraday(c("data_date") == d(2), lookback=days(1))),
    # A sub-day lookback from a Date literal still crosses the day boundary, and the upper bound covers the whole day.
    Scenario(
        "eq_with_subday_lookback_fetches_prior_evening",
        sources={
            "main": (
                pl.DataFrame(
                    {
                        "timestamp": [datetime(2024, 1, 1, 20), datetime(2024, 1, 2, 4), datetime(2024, 1, 2, 10), datetime(2024, 1, 3, 4)],
                        "value": [10, 20, 30, 40],
                    }
                ),
                {"data_date": F(source_col="timestamp", lookback=timedelta(hours=18))},
            )
        },
        combine=as_date,
        predicate=c("data_date") == d(2),
        check=pushed_text("main", has=("2024-01-01 06:00:00", "2024-01-03 00:00:00")),
    ),
    Scenario("le_covers_full_day", **intraday(c("data_date") <= d(2))),
    Scenario("ge_starts_at_midnight", **intraday(c("data_date") >= d(2))),
    Scenario("lt_excludes_full_day", **intraday(c("data_date") < d(2))),
    Scenario("gt_starts_next_day", **intraday(c("data_date") > d(2))),
    Scenario("is_in_covers_full_days", **intraday(c("data_date").is_in([d(2), d(3)]))),
    Scenario(
        "datetime_filter_on_datetime_source_unchanged",
        {"main": (INTRADAY, {"timestamp": F()})},
        lambda s: s["main"],
        c("timestamp") >= datetime(2024, 1, 2, 12),
        check_row_order=True,
    ),
    # IntervalFilterSpec pushes start <= hi and end >= lo onto the schedule; membership is the combine's.
    Scenario(
        "selects_covering_windows", windows(schedule([d(1), d(11), d(21)], [d(10), d(20), d(30)]), spine(d(15))), interval_join(), c("date") == d(15)
    ),
    Scenario("long_window_kept", windows(schedule([d(1, 1, 2020)], [d(1, 1, 2030)]), spine(d(15))), interval_join(), c("date") == d(15)),
    Scenario(
        "bounds_pushed_to_source",
        windows(SCHED_11_20, spine(d(12), d(15))),
        interval_join(),
        c("date").is_between(d(12), d(15)),
        bounds={"sched": {"end": (d(12), ...), "start": (..., d(15))}},
    ),
    shift_bounds_row("open", c("date") >= d(12), d(11), None),
    shift_bounds_row("over_cap", c("date").is_between(d(1, 1, 1990), d(15)), d(31, 12, 1989), d(14)),
    shift_bounds_row(
        "datetime", c("date").is_between(datetime(2024, 1, 12, 6), datetime(2024, 1, 15, 6)), datetime(2024, 1, 11, 6), datetime(2024, 1, 14, 6)
    ),
    Scenario(
        "interval_timedelta_mapping_must_be_whole_days_for_dates",
        windows(SCHED_11_20, spine(d(13)), value_mapping=timedelta(hours=12)),
        interval_join(),
        c("date") == d(13),
        raises=(Exception, "whole number of days"),
    ),
    endpoint_row("both", True, True),
    endpoint_row("left", True, False),
    endpoint_row("right", False, True),
    endpoint_row("none", False, False),
    # A Date request on the 15th keeps windows that end or start intraday on the 15th, under every closed mode.
    *[
        Scenario(
            f"datetime_intraday_boundary_not_pruned-{closed}",
            windows(INTRADAY_SCHED, spine(d(15), d(15), roots=["A", "B"]), closed=closed),
            interval_join(None),
            c("date") == d(15),
        )
        for closed in CLOSED
    ],
    # Disjoint requested ranges collapse to their hull: a window in the gap survives the pushdown.
    Scenario(
        "disjoint_request_collapses_to_hull",
        windows(schedule([d(20)], [d(25)]), spine(d(22))),
        interval_join(None),
        c("date").is_between(d(1), d(5)) | c("date").is_between(d(30), d(31)),
        bounds={"sched": {"end": (d(1), ...)}},
    ),
    # Monday maps to the previous Friday, other days to the day before: Sun 1 Sep maps after Mon 2 Sep.
    Scenario(
        "non_monotone_value_mapping_keeps_interior_days",
        windows(SEP_SCHED, SEP_SPINE, value_mapping=previous_weekday),
        interval_join(SEP_MEMBERSHIP),
        c("date").is_between(d(1, 9), d(3, 9)),
    ),
    Scenario(
        "value_mapping_to_none_pushes_nothing",
        windows(schedule([d(1), d(3)], [d(1), d(3)], [1, 3]), spine(d(1), d(2), d(3)), value_mapping=lambda day: None if day == d(2) else day),
        interval_join(),
        c("date").is_between(d(1), d(3)),
    ),
    Scenario(
        "interval_filter_with_tz_aware_endpoints",
        {
            "spine": (pl.DataFrame({"d": [datetime(2024, 1, 2, tzinfo=UTC_ZONE)]}), {"d": F()}),
            "sched": (
                pl.DataFrame({"start": [datetime(2024, 1, 1, tzinfo=UTC_ZONE)], "end": [datetime(2024, 1, 3, tzinfo=UTC_ZONE)], "x": [1]}),
                {"d": IntervalFilterSpec("start", "end")},
            ),
        },
        lambda s: s["spine"].join_where(s["sched"], c("d") >= c("start"), c("d") <= c("end")).select("d", "x"),
        c("d") == datetime(2024, 1, 2, tzinfo=UTC_ZONE),
    ),
    # Shared source columns: a source row may serve any output column mapped to it, so the pushed values are the union.
    Scenario("lone_side_filter_returns_all_partners", one(THREE_GROUPS, **PAIR_SPECS), group_pairs, c("group1") == "A"),
    Scenario(
        "both_sides_filter_unions_to_is_in",
        one(THREE_GROUPS, **PAIR_SPECS),
        group_pairs,
        (c("group1") == "A") & (c("group2") == "B"),
        discrete={"data": {"group": {"A", "B"}}},
    ),
    Scenario(
        "is_in_both_sides_unions",
        one(THREE_GROUPS, **PAIR_SPECS),
        group_pairs,
        c("group1").is_in(["A", "B"]) & c("group2").is_in(["B", "C"]),
        discrete={"data": {"group": {"A", "B", "C"}}},
    ),
    Scenario(
        "three_shared_columns_union-all_constrained",
        one(THREE_GROUPS.select("group"), **THREE_SPECS),
        cross_aliases("group", "group1", "group2", "group3"),
        (c("group1") == "A") & (c("group2") == "B") & (c("group3") == "C"),
        discrete={"data": {"group": {"A", "B", "C"}}},
    ),
    Scenario(
        "three_shared_columns_union-one_unconstrained",
        one(THREE_GROUPS.select("group"), **THREE_SPECS),
        cross_aliases("group", "group1", "group2", "group3"),
        (c("group1") == "A") & (c("group2") == "B"),
        no_discrete={"data": {"group"}},
    ),
    Scenario(
        "value_mapping_union_of_mapped_values",
        one(CODES, name1=F(source_col="code", value_mapping=NAME_MAPPING), name2=F(source_col="code", value_mapping=NAME_MAPPING)),
        cross_aliases("code", "name1", "name2"),
        (c("name1") == "A") & (c("name2") == "B"),
        discrete={"data": {"code": {"a", "b"}}},
    ),
    # name2 has no mapping for "C", so the union is skipped and the user's filter decides.
    Scenario(
        "value_mapping_unmapped_member_still_correct",
        one(CODES, name1=F(source_col="code", value_mapping=NAME_MAPPING), name2=F(source_col="code", value_mapping={"A": "a", "B": "b"})),
        mapped_cross,
        (c("name1") == "A") & (c("name2") == "C"),
    ),
    # Only the pushed filters are checked: the filter on ``other`` reaches the whole source, so the group2 side loses "B"
    # and the result is empty where plain Polars returns one row.
    Scenario(
        "non_shared_column_in_same_source_unaffected",
        one(GROUP_OTHER, **GROUP_OTHER_SPECS),
        cross_aliases("group", "group1", "group2", keep=("other",)),
        GROUP_OTHER_PRED,
        compare_native=False,
        discrete={"data": {"group": {"A", "B"}, "other": {"x"}}},
    ),
    # A lookback on d1 suppresses its discrete membership, so d2 is the lone member of the union on d. The result is not
    # checked: the lookback range and d == Jan 2 intersect on the shared column and starve d1 == Jan 3 (a documented
    # limitation of mixing temporal and discrete specs on one source column).
    Scenario(
        "temporal_suppressed_column_excluded_from_union",
        one(THREE_DATES, d1=F(source_col="d", lookback=days(2)), d2=F(source_col="d")),
        cross_aliases("d", "d1", "d2"),
        (c("d1") == d(3)) & (c("d2") == d(2)),
        compare_native=False,
        bounds={"data": {"d": (..., ...)}},
        discrete={"data": {"d": {d(2)}}},
    ),
    # Unorderable members (None and strings) must not be sorted into the pushed is_in.
    Scenario(
        "union_with_unorderable_values_does_not_crash",
        one(pl.DataFrame({"group": ["A", "B", None]}), group1=F(source_col="group"), group2=F(source_col="group")),
        cross_aliases("group", "group1", "group2"),
        (c("group1").is_null() | (c("group1") == "A")) & (c("group2") == "B"),
    ),
    # Casts: 257 wraps to 1 under Int8, and 16777217 rounds to 16777216.0 under Float32.
    Scenario(
        "narrowing_cast_does_not_drop_rows", one(pl.DataFrame({"a": [1, 2, 257]}), a=F()), identity, c("a").cast(pl.Int8, wrap_numerical=True) == 1
    ),
    *[
        Scenario(f"float_cast_in_combine_keeps_rounded_rows-{i}", one(pl.DataFrame({"x": [16777217, 5, None]}), x=F()), cast_to(dtype), pred)
        for i, (pred, dtype) in enumerate(FLOAT_CASTS)
    ],
    # A value set containing None keeps null rows, also when two output columns share a source column.
    *[
        Scenario(
            f"discrete_filter_keeps_null_rows-{name}",
            {"single": (NULLABLE, {"a": F()}), "shared": (NULLABLE, {"a": F(), "b": F(source_col="a")})},
            lambda s: pl.concat([s["single"], s["shared"]]),
            pred,
        )
        for name, pred in NULL_PREDICATES.items()
    ],
    *[Scenario(f"matches_filter-{i}", one(df, t=spec), identity, pred) for i, (df, pred, spec) in enumerate(MATCHES_FILTER)],
    *[
        Scenario(
            f"temporal_filter_applies_value_mapping-{name}",
            one(pl.DataFrame({"d": [d(1), d(2)]}), d=F(value_mapping=mapping)),
            lambda s: s["data"].with_columns(c("d") + days(1)),
            c("d") == d(2),
            check_row_order=True,
        )
        for name, mapping in [("dict", {d(2): d(1)}), ("callable", lambda day: day - days(1))]
    ],
    # Filtering the right side of a left join would null-extend rows that then satisfy the predicate.
    *[
        Scenario(
            f"null_accepting_filter_not_pushed_below_left_join-{name}",
            two(pl.DataFrame({"id": [1, 2]}), pl.DataFrame({"id": [1, 2], "v": [10, None]}), {}, {"v": F()}),
            join("id", how="left"),
            pred,
        )
        for name, pred in [("is_null", c("v").is_null()), ("is_null_or_eq", c("v").is_null() | (c("v") == 5))]
    ],
    # A value_mapping with a window pushes a window around the mapped values.
    *[
        Scenario(
            f"window_pushed_around_mapped_values-{name}",
            one(DAYS20, date=weekday_window(7)),
            next_day_rolling,
            pred,
            bounds={"data": {"data_date": (..., ...)}},
        )
        for name, pred in [("eq", c("date") == d(8)), ("is_in", c("date").is_in([d(8), d(15)])), ("between", c("date").is_between(d(8), d(10)))]
    ],
    Scenario(
        "open_range_through_callable_pushes_window_around_requested_range",
        one(DAYS20, date=weekday_window(7)),
        next_day_rolling,
        c("date") >= d(8),
        bounds={"data": {"data_date": (d(1), None)}},
    ),
    *[
        Scenario(
            f"incomplete_or_failing_mapping_keeps_rows-{name}",
            one(DAYS20, date=window(lookback=days(2), value_mapping=mapping)),
            next_day_rolling,
            c("date") == d(8),
        )
        for name, mapping in [("dict", {d(9): d(8)}), ("none", lambda day: None), ("raises", lambda day: 1 / 0)]
    ],
    Scenario(
        "unconstrained_identity_sibling_does_not_block_the_window",
        one(DAYS20, data_date=F(), date=weekday_window(2)),
        next_day_rolling,
        c("date") == d(8),
        bounds={"data": {"data_date": (..., ...)}},
    ),
    Scenario(
        "mapped_aliases_in_a_self_join_union",
        one(DAYS20, **{n: window(lookback=days(1), value_mapping=lambda day: day - days(1)) for n in ("d1", "d2")}),
        mapped_self_join,
        (c("d1") == d(3)) & (c("d2") == d(12)),
    ),
    Scenario(
        "integer_output_key_is_not_used_as_a_date", one(DAYS20, k=window(lookback=days(2), value_mapping={8: d(7)})), day_of_month_key, c("k") == 8
    ),
    Scenario(
        "dict_covering_the_range_is_enumerated",
        one(DAYS20, date=window(lookback=days(1), value_mapping={d(8): d(8), d(9): d(9)})),
        same_day_rolling,
        c("date").is_between(d(8), d(9)),
        bounds={"data": {"data_date": (d(7), d(9))}},
    ),
    Scenario(
        "lookahead_around_mapped_values",
        one(DAYS20, date=window(lookahead=days(2), value_mapping=lambda day: day)),
        lambda s: s["data"].with_columns(c("v").shift(-2).alias("lead"), c("data_date").alias("date")),
        c("date") == d(10),
    ),
    Scenario(
        "range_beyond_the_enumeration_cap_gets_the_window_around_the_range",
        one(DAYS20, date=window(lookback=days(1), value_mapping=lambda day: day)),
        same_day_rolling,
        c("date").is_between(d(1, 1, 1990), d(10)),
        bounds={"data": {"data_date": (d(31, 12, 1989), d(10))}},
    ),
    # A window that cannot be applied pushes nothing rather than the unwidened value.
    Scenario(
        "time_zone_aware_column",
        one(pl.DataFrame({"t": [datetime(2024, 1, day, tzinfo=UTC_ZONE) for day in range(1, 11)], "v": range(10)}), t=F(lookback=days(3))),
        lambda s: s["data"].with_columns(c("v").shift(1).alias("lag")),
        c("t") == datetime(2024, 1, 5, tzinfo=UTC_ZONE),
    ),
    Scenario(
        "unbounded_window", one(TENDAYS, d=F(lookback=timedelta.max)), lambda s: s["data"].with_columns(c("v").cum_sum().alias("c")), c("d") == d(5)
    ),
    Scenario(
        "unbounded_window_sharer_keeps_the_whole_source_column",
        one(TENDAYS.with_columns(c("v") + 1), a=F(source_col="d", lookback=timedelta.max), b=F(source_col="d", value_mapping=lambda x: x)),
        unbounded_sharer,
        (c("a") >= d(5)) & (c("b") == d(8)),
    ),
    # Temporal values and units
    Scenario(
        "mapped_values_keep_both_readings_of_a_repeated_hour",
        one(
            pl.DataFrame({"t": FOLD}, schema={"t": pl.Datetime("us", "America/New_York")}),
            a=F(source_col="t", value_mapping=lambda a: a.astimezone(NY)),
        ),
        lambda s: s["data"].with_columns(c("t").dt.convert_time_zone("UTC").alias("a")),
        c("a").is_in([datetime(2024, 11, 3, 5, 30, tzinfo=UTC_ZONE), datetime(2024, 11, 3, 6, 30, tzinfo=UTC_ZONE)]),
    ),
    *[is_in_units_row(unit, time_zone) for time_zone in (None, "UTC") for unit in ("ms", "us", "ns")],
    Scenario(
        "value_set_keys_do_not_overflow_near_the_date_limit",
        one(pl.DataFrame({"t": NEAR_MAX, "k": [1, 2]}), k=F(source_col="t", value_mapping={1: NEAR_MAX[0], 2: NEAR_MAX[1]})),
        identity,
        c("k").is_in([1, 2]),
    ),
    Scenario(
        "pushed_equality_keeps_the_predicates_comparison_unit",
        one(pl.Series("t", [1704067200000000000, 1704067200000000001]).cast(pl.Datetime("ns", "UTC")).to_frame(), a=F(source_col="t")),
        lambda s: s["data"].with_columns(c("t").dt.cast_time_unit("us").alias("a")),
        c("a") == datetime(2024, 1, 1, tzinfo=UTC),
    ),
    Scenario(
        "membership_with_another_unit_keeps_native_rows",
        one(US_MEMBERS.to_physical().cast(pl.Datetime("ns", "UTC")).to_frame("t"), t=F()),
        lambda s: s["data"].collect().lazy(),
        c("t").is_in(US_MEMBERS.implode()),
    ),
]


def _call(scenario: Scenario, frames: dict) -> pl.LazyFrame:
    extra = scenario.combine_kwargs or {}
    return scenario.combine(**frames, **extra) if scenario.sources_as_kwargs else scenario.combine(frames, **extra)


def _filter(lf: pl.LazyFrame, pred) -> pl.LazyFrame:
    return lf if pred is None else lf.filter(pred)


def _matches_native(actual: pl.LazyFrame, reference: pl.LazyFrame, row_order: bool) -> pl.DataFrame | None:
    """``assert_matches_native``; without ``row_order``, the cheaper ordered comparison is tried first."""
    if not row_order:
        try:
            return assert_matches_native(actual, reference, None, check_row_order=True)
        except AssertionError:
            pass
    return assert_matches_native(actual, reference, None, check_row_order=row_order)


@pytest.mark.parametrize("scenario", [pytest.param(sc, id=sc.id, marks=sc.marks) for sc in SCENARIOS])
def test_pushdown_combine(scenario):
    sources, is_identity = scenario.sources, scenario.combine is identity
    inspected = {*scenario.pushed, *scenario.bounds, *scenario.discrete, *scenario.no_discrete}
    if is_identity or scenario.check is not None:
        inspected |= set(sources)
    trackers = {name: PredicateTracker(df) for name, (df, _) in sources.items() if name in inspected}
    frames = {name: trackers[name].lazy_frame if name in trackers else df.lazy() for name, (df, _) in sources.items()}
    lf = pushdown_combine(
        {name: (frames[name], specs) for name, (_, specs) in sources.items()},
        scenario.combine,
        combine_kwargs=scenario.combine_kwargs,
        sources_as_kwargs=scenario.sources_as_kwargs,
    )
    reference = _call(scenario, {name: df.lazy() for name, (df, _) in sources.items()})
    post = scenario.post or (lambda frame: frame)
    row_order = is_identity if scenario.check_row_order is None else scenario.check_row_order
    preds = scenario.predicate if isinstance(scenario.predicate, list) else [scenario.predicate]
    for p in preds:
        actual = post(_filter(lf, p))
        if scenario.raises is not None:
            with pytest.raises(scenario.raises[0], match=scenario.raises[1]):
                actual.collect()
            return
        if scenario.compare_native:
            got = _matches_native(actual, post(_filter(reference, p)), row_order)
        else:
            got = actual.collect()
    for name in scenario.pushed:
        assert trackers[name].last_predicate is not None, f"nothing pushed to {name}"
    for name, columns in scenario.bounds.items():
        for col, (lower, upper) in columns.items():
            assert_pushed_bounds(trackers[name], col, lower, upper)
    for name, columns in scenario.discrete.items():
        for col, values in columns.items():
            assert_discrete(trackers[name], col, values, scenario.discrete_node.get(name, {}).get(col))
    for name, columns in scenario.no_discrete.items():
        pushed = trackers[name].last_predicate
        for col in columns:
            assert pushed is None or PredicateAnalyzer(pushed).find_discrete_filter(col) is None, pushed
    if scenario.check is not None:
        scenario.check(got, trackers)


def test_combine_kwargs_with_join_parameters():
    """Two frames built over the same sources, each with its own combine_kwargs."""
    left, right = PredicateTracker(KEY_LEFT), PredicateTracker(KEY_RIGHT)
    for how in ("inner", "left"):
        lf = pushdown_combine({"left": (left.lazy_frame, {"date": F()}), "right": (right.lazy_frame, {})}, join_how, combine_kwargs={"join_how": how})
        assert_matches_native(lf, KEY_LEFT.lazy().join(KEY_RIGHT.lazy(), on="key", how=how), None)


def test_lone_side_pushes_no_group_predicate():
    """With one side of a shared pair unconstrained, at least one scan carries no discrete filter on the shared column."""
    predicates = []
    source = io_source_assert(THREE_GROUPS, predicates.append)
    lf = pushdown_combine({"data": (source, PAIR_SPECS)}, group_pairs)
    assert_matches_native(lf, group_pairs({"data": THREE_GROUPS.lazy()}), c("group1") == "A")
    assert predicates
    assert any(p is None or PredicateAnalyzer(p).find_discrete_filter("group") is None for p in predicates)


def test_integral_float_without_float32_consumer_pushes_equality():
    expr = _union_discrete_filter("x", [("x", F())], {"x": {16777216.0}}, pl.Int64)
    # Neither the integer next to the value nor nulls are kept.
    assert pl.DataFrame({"x": [16777216, 16777217, None]}).filter(expr)["x"].to_list() == [16777216]


@pytest.mark.parametrize("pred", [pl.lit(False), pl.lit(None, dtype=pl.Boolean)], ids=["false", "null"])
def test_constant_false_predicate_reads_nothing(pred):
    from polars.io.plugins import register_io_source

    rows_read = []

    def generator(with_columns, predicate, n_rows, batch_size):
        out = TENDAYS if n_rows is None else TENDAYS.head(n_rows)
        rows_read.append(out.height)
        yield out

    source = register_io_source(generator, schema=TENDAYS.schema)
    assert (
        pushdown_combine({"s": (source, {"d": F(lookback=days(1))})}, lambda s: s["s"].with_columns(c("v").rolling_sum(2).alias("r")))
        .filter(pred)
        .collect()
        .height
        == 0
    )
    assert sum(rows_read) == 0


# Specs and helpers ------------------------------------------------------------------------------------------------------
SPEC_FIELDS = [
    Case("filter_spec-defaults", F(), {"source_col": None, "lookback": timedelta(), "lookahead": timedelta(), "value_mapping": None}),
    Case("source_col_override", F(source_col="DataDate"), {"source_col": "DataDate"}),
    Case("lookback_override", F(lookback=days(5)), {"lookback": days(5)}),
    Case("lookahead_override", F(lookahead=days(3)), {"lookahead": days(3)}),
    Case("value_mapping_dict", F(value_mapping={"A": "a", "B": "b"}), {"value_mapping": {"A": "a", "B": "b"}}),
    Case("value_mapping_callable", F(value_mapping=str.upper), {"value_mapping": str.upper}),
    Case(
        "combined_options",
        F(source_col="col", lookback=days(5), lookahead=days(3), value_mapping={"x": "y"}),
        {"source_col": "col", "lookback": days(5), "lookahead": days(3), "value_mapping": {"x": "y"}},
    ),
    Case(
        "interval_spec-defaults",
        IntervalFilterSpec(start_col="s", end_col="e"),
        {"start_col": "s", "end_col": "e", "closed": "both", "value_mapping": None},
    ),
    *[
        Case(f"valid_closed_accepted-{closed}", IntervalFilterSpec(start_col="s", end_col="e", closed=closed), {"closed": closed})
        for closed in CLOSED
    ],
]


@pytest.mark.parametrize(("spec", "fields"), params(SPEC_FIELDS, "pred", "expect"))
def test_spec_fields(spec, fields):
    assert {name: getattr(spec, name) for name in fields} == fields


def test_invalid_closed_raises():
    with pytest.raises(ValueError, match="closed must be one of"):
        IntervalFilterSpec(start_col="s", end_col="e", closed="closed")


# (values, mapping) -> (mapped, unmapped); an unmapped value is not pushed.
VALUE_MAPPINGS = [
    Case("none_mapping", ({"a", "b", "c"}, None), ({"a", "b", "c"}, set())),
    Case("dict_mapping_all_keys_present", ({"USE4S", "CNE5S"}, {"USE4S": "US4S", "CNE5S": "CN5S"}), ({"US4S", "CN5S"}, set())),
    Case("dict_mapping_partial_keys", ({"USE4S", "UNKNOWN"}, {"USE4S": "US4S"}), ({"US4S"}, {"UNKNOWN"})),
    Case("dict_mapping_no_keys_present", ({"UNKNOWN1", "UNKNOWN2"}, {"USE4S": "US4S"}), (set(), {"UNKNOWN1", "UNKNOWN2"})),
    Case("callable_mapping", ({"foo", "bar"}, str.upper), ({"FOO", "BAR"}, set())),
    Case("callable_mapping_custom_function", ({1, 2, 3}, lambda x: x * 2), ({2, 4, 6}, set())),
    Case("empty_values", (set(), {"a": "b"}), (set(), set())),
]


@pytest.mark.parametrize(("args", "expect"), params(VALUE_MAPPINGS, "pred", "expect"))
def test_apply_value_mapping(args, expect):
    assert _apply_value_mapping(*args) == expect


@pytest.mark.parametrize(
    ("spec", "expect"), [(F(), "date"), (F(source_col="DataDate"), "DataDate")], ids=["none_source_col_returns_output_col", "explicit_source_col"]
)
def test_get_source_col(spec, expect):
    assert _get_source_col("date", spec) == expect


OUTPUT_SCHEMAS = [
    Case("simple_join_schema", lambda s: s["left"].join(s["right"], on="a"), {"a": pl.Int64, "b": pl.Int64, "c": pl.Int64}),
    Case("with_columns_schema", lambda s: s["left"].select("a").with_columns(pl.lit(42).alias("new_col")), {"a": pl.Int64, "new_col": pl.Int32}),
    Case("select_schema", lambda s: s["left"].select("a", "b"), {"a": pl.Int64, "b": pl.Int64}),
]


@pytest.mark.parametrize(("combine", "expect"), params(OUTPUT_SCHEMAS, "pred", "expect"))
def test_compute_output_schema(combine, expect):
    sources = {"left": (pl.LazyFrame({"a": [1], "b": [2]}), {}), "right": (pl.LazyFrame({"a": [1], "c": [3]}), {})}
    assert dict(_compute_output_schema(sources, combine)) == expect
