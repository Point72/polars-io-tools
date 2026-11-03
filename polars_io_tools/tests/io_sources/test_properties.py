"""Property tests against native Polars: whatever a predicate keeps, the pushdown keeps.

Each test draws frames with nulls and a wide mix of dtypes, filters them with random predicates, and checks the
superset contract of one entry point: every row native Polars keeps is kept, and a valid predicate does not raise.
For value extractions, every kept row's value is covered. An adapter that reports "unconstrained" passes trivially.
"""

import functools
import itertools
import logging
import math
import operator
import shutil
import time
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import narwhals as nw
import polars as pl
import portion
import pytest
from hypothesis import assume, example, given, note, settings, strategies as st
from polars.io.plugins import register_io_source
from polars.testing import assert_frame_equal

from polars_io_tools.io_sources.base import get_parsed_expr
from polars_io_tools.io_sources.concat_named import concat_named
from polars_io_tools.io_sources.delta_io import scan_delta, sink_delta
from polars_io_tools.io_sources.dnf_visitor import MAX_DNF_TUPLES, _is_contradiction, convert_expr_to_dnf
from polars_io_tools.io_sources.lazy_cache_memory import cache_memory
from polars_io_tools.io_sources.lazy_cache_parquet import CacheMode, cache_parquet
from polars_io_tools.io_sources.lazy_narwhals_reader import scan_narwhals
from polars_io_tools.io_sources.partitions import by_range, by_time, by_value
from polars_io_tools.io_sources.pushdown_combine import FilterSpec, IntervalFilterSpec, pushdown_combine
from polars_io_tools.io_sources.pushdown_pivot import pushdown_pivot
from polars_io_tools.io_sources.pushdown_unpivot import pushdown_unpivot
from polars_io_tools.io_sources.range_visitor import convert_expr_to_datetime_range, convert_expr_to_range
from polars_io_tools.io_sources.restrict_visitor import restrict_expr_to_columns
from polars_io_tools.io_sources.set_visitor import convert_expr_to_valid_values
from polars_io_tools.testing import PredicateTracker
from polars_io_tools.tests.helpers.cases import NY
from polars_io_tools.tests.helpers.polars_versions import polars_has, xfail_if, xfail_unless
from polars_io_tools.tests.helpers.strategies import (
    IMPLODE,
    POLARS_ERRORS,
    frames_and_predicates,
    native,
    predicate,
)

# ---------------------------------------------------------------------------
# Expected rows: rows native Polars keeps, and rows library output keeps
# ---------------------------------------------------------------------------


def _filtered(frame: pl.DataFrame, expr: pl.Expr) -> pl.Series:
    kept = frame.with_row_index("__row").filter(expr)["__row"]
    return pl.int_range(frame.height, dtype=kept.dtype, eager=True).is_in(kept.implode() if IMPLODE else kept)


def _projected(frame: pl.DataFrame, expr: pl.Expr) -> pl.Series:
    out = frame.with_columns(expr.alias("__m"))["__m"]
    if out.dtype not in (pl.Boolean, pl.Null):
        raise TypeError(out.dtype)
    return out.cast(pl.Boolean).fill_null(False)


def mask(frame: pl.DataFrame, expr: pl.Expr) -> pl.Series:
    """The rows of ``frame`` a predicate built by library code keeps, filtered or projected.

    Polars versions simplify some predicates differently in a filter than in a projection; a row either keeps counts.
    """
    return _filtered(frame, expr) | _projected(frame, expr)


def show(expr: pl.Expr) -> str:
    """``str(expr)``; Polars panics formatting a temporal literal Python cannot hold."""
    try:
        return str(expr)
    except POLARS_ERRORS:
        return repr(expr)


def native_keep(frame: pl.DataFrame, pred: pl.Expr) -> pl.Series | None:
    """The rows native Polars keeps for ``pred``; None if Polars rejects the predicate, or if a filter and a projection
    disagree (a Polars bug, such as 1.44 simplifying ``x.is_in([]).is_null()`` to false in a filter only)."""
    try:
        filtered, projected = _filtered(frame, pred), _projected(frame, pred)
    except POLARS_ERRORS:
        return None
    return filtered if filtered.equals(projected) else None


def delivered(frame: pl.DataFrame, pred: pl.Expr) -> pl.Expr | None:
    """The predicate Polars hands an IO source scanning ``frame`` filtered by ``pred``."""
    received: list = []

    def source(with_columns, predicate, n_rows, batch_size):
        received.append(predicate)
        yield frame

    try:
        register_io_source(source, schema=frame.schema).filter(pred).collect()
    except POLARS_ERRORS:
        return None
    return received[0] if received else None


_COMPARISONS = {
    "=": operator.eq,
    "==": operator.eq,
    "!=": operator.ne,
    "<>": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "!>": operator.le,
    "!>=": operator.lt,
    "!<": operator.ge,
    "!<=": operator.gt,
}


def dnf_expr(dnf: list, schema: dict | None = None) -> pl.Expr:
    """A DNF of ``(column, op, value)`` terms as a Polars predicate with SQL null semantics.

    Values are compared as dynamic Python literals, as a consumer without the schema does, or typed with the column's
    dtype when ``schema`` is given.
    """

    def lit(c, value):
        return pl.lit(value) if schema is None else pl.lit(value, dtype=schema[c])

    def term(c: str, op: str, value) -> pl.Expr:
        e = pl.col(c)
        if op in ("in", "!in", "not in"):
            values = list(value) if isinstance(value, (list, tuple, set, frozenset)) else [value]
            members = pl.any_horizontal([e == lit(c, v) for v in values if v is not None] or [pl.lit(False)])
            return members if op == "in" else ~members
        if op == "is" or (op in ("=", "==") and value is None):
            return e.is_null() if value is None else e == lit(c, value)
        if op in ("!is", "is not"):
            return e.is_not_null() if value is None else e != lit(c, value)
        if op in ("~", "=~", "!~"):
            match = e.str.contains(value)
            return ~match if op == "!~" else match
        return _COMPARISONS[op](e, lit(c, value))

    clauses = [pl.all_horizontal([term(*t) for t in clause] or [pl.lit(True)]).fill_null(False) for clause in dnf]
    return pl.any_horizontal(clauses) if clauses else pl.lit(False)


SETTINGS = settings(deadline=None)


# ---------------------------------------------------------------------------
# Predicate analysis: DNF, contradiction, value sets, ranges, restriction
# ---------------------------------------------------------------------------


def _is_nan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)


def _naive_utc(v):
    if isinstance(v, datetime):
        return v.astimezone(UTC).replace(tzinfo=None) if v.tzinfo is not None else v
    return datetime.combine(v, datetime.min.time())


def _midnight(v):
    return datetime.combine(v, datetime.min.time()) if isinstance(v, date) and not isinstance(v, datetime) else v


def _within(value, interval) -> bool:
    """``value in interval``, reading a date as its midnight against datetime bounds, and a naive value as UTC against
    zoned bounds."""
    try:
        return value in interval
    except TypeError:
        pass
    try:
        return _naive_utc(value) in interval
    except TypeError:
        pass
    try:
        return _naive_utc(value).replace(tzinfo=UTC) in interval
    except TypeError:  # date bounds: their midnights
        return _naive_utc(value) in interval.apply(lambda i: i.replace(lower=_midnight(i.lower), upper=_midnight(i.upper)))


def _members(series: pl.Series, values: set) -> pl.Series:
    """Whether each value of ``series`` is one of ``values`` as a consumer filtering on ``col == value`` sees it."""
    frame = series.to_frame("c")
    try:
        return mask(frame, pl.any_horizontal([pl.col("c") == v for v in values if v is not None] + [pl.col("c").is_null() & (None in values)]))
    except POLARS_ERRORS:  # values a consumer cannot compare with the column
        return pl.Series([v in values for v in series.to_list()])


def _inside(series: pl.Series, interval) -> pl.Series:
    """Whether each non-null, non-NaN value of ``series`` lies in ``interval`` as a consumer filtering on its bounds
    sees it."""
    c = pl.col("c")

    def bound(atomic) -> pl.Expr:
        lower = pl.lit(True) if atomic.lower == -portion.inf else (c >= atomic.lower if atomic.left == portion.CLOSED else c > atomic.lower)
        upper = pl.lit(True) if atomic.upper == portion.inf else (c <= atomic.upper if atomic.right == portion.CLOSED else c < atomic.upper)
        return lower & upper

    skip = c.is_null() | (c.is_nan() if series.dtype.is_float() else pl.lit(False))
    try:
        return mask(series.to_frame("c"), skip | pl.any_horizontal([bound(a) for a in interval] or [pl.lit(False)]))
    except POLARS_ERRORS:  # bounds a consumer cannot compare with the column
        return pl.Series([v is None or _is_nan(v) or v in interval for v in series.to_list()])


def _dnf_mask(frame: pl.DataFrame, dnf: list) -> pl.Series | None:
    """The rows a consumer of ``dnf`` keeps with dynamic or typed literals (either counts); None if Polars rejects both.

    Polars compares a column with a dynamic literal of another time unit in the coarser unit on some versions.
    """
    masks = []
    for schema in (None, dict(frame.schema)):
        try:
            masks.append(mask(frame, dnf_expr(dnf, schema)))
        except POLARS_ERRORS:
            continue
    return functools.reduce(operator.or_, masks) if masks else None


def check_adapters(frame: pl.DataFrame, pred: pl.Expr, keep: pl.Series, schema: dict | None) -> None:
    """Assert every extraction adapter, given ``schema`` (or none), covers the rows of ``frame`` where ``keep`` is true."""
    full = dict(frame.schema)
    kept = frame.filter(keep)
    # Every caller of the DNF and the contradiction check passes the schema; without it, an integer column used as a
    # Boolean reads as one.
    if kept.height and schema is not None:
        assert not _is_contradiction(pred, schema), pred
    dnf = convert_expr_to_dnf(pred, schema) if schema is not None else None
    if dnf is not None:
        matched = _dnf_mask(frame, dnf)
        assert matched is None or matched.filter(keep).all(), (pred, dnf)
    for col in frame.columns:
        dtype = full[col]
        try:
            kept_values = kept[col].to_list()
        except POLARS_ERRORS:  # values Python cannot hold
            continue
        values = convert_expr_to_valid_values(pred, col, schema)
        if values is not None:
            assert _members(kept[col], values).all(), (pred, col, values)
        if schema is None:
            continue
        if isinstance(dtype, (pl.Date, pl.Datetime)):
            for enclosure, preserve in ((True, False), (False, False), (False, True)):
                try:
                    interval = convert_expr_to_datetime_range(pred, col, get_enclosure=enclosure, preserve_dates=preserve)
                except TypeError:  # date and datetime bounds that do not compare; callers fall back without preserve_dates
                    assert preserve
                    continue
                as_bound = (lambda v: v) if preserve and dtype == pl.Date else _naive_utc
                assert all(_within(as_bound(v), interval) for v in kept_values if v is not None), (pred, col, interval)
        elif dtype.is_numeric():
            interval = convert_expr_to_range(pred, col)
            assert _inside(kept[col], interval).all(), (pred, col, interval)
    if schema is None:
        return
    for size in range(1, len(frame.columns) + 1):
        for cols in itertools.combinations(frame.columns, size):
            restricted = restrict_expr_to_columns(pred, cols)
            if restricted is not None:
                assert set(restricted.meta.root_names()) <= set(cols), restricted
                try:
                    matched = mask(frame, restricted)
                except POLARS_ERRORS:  # Polars 1.37 rejects some sub-expressions it accepts within the predicate
                    continue
                assert matched.filter(keep).all(), (pred, cols, restricted)


def no_panic(fn):
    """Report a Rust panic, which is not an ``Exception``, as a failure Hypothesis can shrink."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except pl.exceptions.PanicException as e:
            raise AssertionError(f"panic: {e}") from e

    return wrapper


def note_case(frame: pl.DataFrame, pred: pl.Expr) -> None:
    note(f"physical={ {c: frame[c].to_physical().to_list() for c in frame.columns} } schema={frame.schema} pred={show(pred)}")


@no_panic
def check_predicate(frame: pl.DataFrame, pred: pl.Expr) -> None:
    """Assert every adapter, with and without the schema, covers the rows ``pred`` keeps on ``frame``, and likewise for
    the predicate Polars delivers to an IO source; skip predicates Polars rejects."""
    keep = native_keep(frame, pred)
    assume(keep is not None)
    for known in (dict(frame.schema), None):
        check_adapters(frame, pred, keep, known)
    pushed = delivered(frame, pred)
    if pushed is not None:
        # Polars may rewrite the predicate it delivers (``is_in([])`` becomes ``false``), so it is checked against its own native result.
        pushed_keep = native_keep(frame, pushed)
        if pushed_keep is not None:
            for known in (dict(frame.schema), None):
                check_adapters(frame, pushed, pushed_keep, known)


@SETTINGS
@given(case=frames_and_predicates())
def test_random_predicates_are_covered(case):
    note_case(*case)
    check_predicate(*case)


_EARLY, _LATE = (pl.Series([t], dtype=pl.Int64).cast(pl.Datetime("ns", NY)) for t in (1730611800_000000_000, 1730615400_000000_000))


@pytest.mark.parametrize(
    ("frame", "pred"),
    [
        pytest.param(pl.DataFrame({"x": [1, None]}), pl.lit(False), id="false"),
        pytest.param(
            pl.DataFrame({"x": pl.Series([Decimal("0.1")], dtype=pl.Decimal(10, 1))}),
            (pl.col("x") == 0.1) & (pl.col("x") == Decimal("0.1")),
            id="decimal_float",
            marks=pytest.mark.xfail(strict=True, reason="a float and a Decimal literal of equal value are taken as a contradiction"),
        ),
        pytest.param(pl.DataFrame({"x": pl.concat([_EARLY, _LATE])}), (pl.col("x") == _EARLY[0]) & (pl.col("x") != _LATE[0]), id="dst_fold"),
        pytest.param(pl.DataFrame({"x": pl.Series([0.1], dtype=pl.Float32)}), pl.col("x") == 0.1, id="float32"),
        pytest.param(pl.DataFrame({"x": [2, 0, None]}), pl.col("x").cast(pl.Boolean) & (pl.col("x") == 2), id="int_cast_bool"),
        pytest.param(pl.DataFrame({"x": [1, 2, 3], "y": [2, 2, 1]}), ((pl.col("x") & pl.col("y")) > 0) & (pl.col("x") > 1), id="bitwise"),
        pytest.param(
            pl.DataFrame({"k": [None, 5, 0], "x": [1, 300, 4]}),
            pl.when(pl.col("k") > 1).then(pl.col("x") > 100).otherwise(pl.col("k").is_null()),
            id="null_condition",
        ),
        pytest.param(
            pl.DataFrame({"t": pl.Series([1704067200_000000_500], dtype=pl.Int64).cast(pl.Datetime("ns"))}),
            pl.col("t") == datetime(2024, 1, 1),
            id="ns_floor",
        ),
        pytest.param(pl.DataFrame({"x": [1]}), pl.col("x").cast(pl.String) == "1", id="cast_source"),
        pytest.param(
            pl.DataFrame({"x": [2**53 + 1]}),
            (pl.col("x") == float(2**53)) & (pl.col("x") > 2**53),
            id="float_rounds_int",
        ),
        pytest.param(pl.DataFrame({"x": pl.Series([20240100.0], dtype=pl.Float32)}), pl.col("x") == 20240101, id="float32_int"),
        pytest.param(pl.DataFrame({"x": _EARLY.dt.offset_by("29m59s999999us")}), pl.col("x") <= _LATE[0], id="zoned_fold_bound"),
        pytest.param(
            pl.DataFrame({"x": pl.Series([Decimal(-(10**27))], dtype=pl.Decimal(38, 2))}),
            pl.col("x") <= pl.lit(Decimal("0.01"), dtype=pl.Decimal(38, 2)),
            id="decimal_overflow",
        ),
        pytest.param(pl.DataFrame({"x": _LATE}), pl.col("x") > date(1677, 9, 20), id="beyond_nanoseconds"),
        pytest.param(pl.DataFrame({"x": [float("nan"), 1.0, None]}), ~(pl.col("x") <= 5.0), id="nan"),
        pytest.param(pl.DataFrame({"a": [True, False, None]}), (pl.col("a") > 0) | (pl.col("a") == 5), id="bool_int"),
    ],
)
def test_named_predicates_are_covered(frame, pred):
    check_predicate(frame, pred)


_TERMS = 10_000
_ROWS = {"x": [0, 1, 5, _TERMS, None, 3], "y": [1, 0, None, 3, _TERMS, -1]}


@pytest.mark.parametrize(
    ("build", "keeps"),
    [
        pytest.param(
            lambda: functools.reduce(operator.or_, [pl.col("x") == i for i in range(_TERMS)]),
            lambda x, y: x is not None and 0 <= x < _TERMS,
            id="or_chain",
        ),
        pytest.param(
            lambda: functools.reduce(operator.and_, [pl.col("x") != i for i in range(_TERMS)]),
            lambda x, y: x is not None and not 0 <= x < _TERMS,
            id="and_chain",
        ),
        pytest.param(
            lambda: functools.reduce(operator.and_, [(pl.col("x") == i) | (pl.col("y") < i + 1) for i in range(_TERMS // 2)]),
            lambda x, y: all((x is not None and x == i) or (y is not None and y < i + 1) for i in range(_TERMS // 2)),
            id="and_of_or",
        ),
        pytest.param(
            lambda: ~functools.reduce(operator.or_, [(pl.col("x") > i) & (pl.col("y") < i) for i in range(_TERMS // 2)]),
            lambda x, y: all((x is not None and x <= i) or (y is not None and y >= i) for i in range(_TERMS // 2)),
            id="negated_or_of_and",
        ),
    ],
)
def test_ten_thousand_terms_complete_and_stay_sound(build, keeps):
    """Polars itself needs about a minute to evaluate such a chain, so the kept rows come from Python."""
    frame = pl.DataFrame(_ROWS)
    schema = dict(frame.schema)
    kept = pl.Series([keeps(x, y) for x, y in frame.iter_rows()])
    assert kept.any()
    start = time.monotonic()
    expr = build()
    node = get_parsed_expr(expr)
    assert not _is_contradiction(node, schema)
    dnf = convert_expr_to_dnf(node, schema)
    assert dnf is None or _dnf_mask(frame, dnf).filter(kept).all()
    for col in ("x", "y"):
        values = convert_expr_to_valid_values(node, col, schema)
        assert values is None or _members(frame.filter(kept)[col], values).all()
        restricted = restrict_expr_to_columns(node, [col])
        if restricted is not None:
            assert set(restricted.meta.root_names()) == {col}
            # Polars needs about a minute to evaluate the unbalanced original, so an unchanged predicate is not evaluated.
            assert restricted.meta.eq(expr) or mask(frame, restricted).filter(kept).all(), (col, restricted)
    assert time.monotonic() - start < 60


@pytest.mark.xfail(strict=True, raises=RecursionError, reason="the range extraction recurses once per term of a long chain")
@pytest.mark.parametrize("partitioner", [by_range("x", 1), by_time("t")], ids=["by_range", "by_time"])
def test_long_chain_on_the_partition_column(partitioner):
    col = pl.col(partitioner.on)
    pred = functools.reduce(operator.or_, [col == (i if partitioner.on == "x" else date(2024, 1, 1) + timedelta(days=i)) for i in range(1_000)])
    partitioner.build(pred)


def test_dnf_of_many_disjunctions_stays_within_budget():
    pred = functools.reduce(operator.and_, [(pl.col(f"a{i}") == 1) | (pl.col(f"b{i}") == 1) for i in range(16)])
    dnf = convert_expr_to_dnf(pred, {c: pl.Int64 for i in range(16) for c in (f"a{i}", f"b{i}")})
    assert dnf is None or len(dnf) <= 10_000


def test_dnf_of_wide_clauses_stays_within_budget():
    """Conjuncts that survive the budget are kept, and the resulting superset is not negated."""
    pred = pl.all_horizontal([pl.col(f"c{i}") == 1 for i in range(1_000)] + [(pl.col(f"a{i}") == 1) | (pl.col(f"b{i}") == 1) for i in range(13)])
    schema = {c: pl.Int64 for c in pred.meta.root_names()}
    dnf = convert_expr_to_dnf(pred, schema)
    assert dnf is not None and sum(len(clause) for clause in dnf) <= MAX_DNF_TUPLES
    assert all(("c999", "=", 1) in clause for clause in dnf)
    assert convert_expr_to_dnf(~pred, schema) is None


def _any_of(col: str, n: int) -> pl.Expr:
    return functools.reduce(operator.or_, [pl.col(col) == i for i in range(n)])


@pytest.mark.parametrize(
    ("last", "bounds"),
    [
        (_any_of("c", 50), [("c", "in", list(range(50)))]),
        (functools.reduce(operator.or_, [pl.col("c").is_between(i, i + 1) for i in range(50)]), [("c", ">=", 0), ("c", "<=", 50)]),
    ],
    ids=["members", "range"],
)
def test_an_and_over_the_budget_keeps_the_bounds_of_the_side_it_drops(last, bounds):
    pred = _any_of("a", 50) & _any_of("b", 50) & last
    values = pl.DataFrame({"a": [0, 1, 49, 50, 51, -1, None]}, schema={"a": pl.Int64})
    frame = values.join(values.rename({"a": "b"}), how="cross").join(values.rename({"a": "c"}), how="cross")
    schema = dict(frame.schema)
    dnf = convert_expr_to_dnf(pred, schema)
    assert dnf is not None and sum(len(clause) for clause in dnf) <= MAX_DNF_TUPLES
    assert all(clause[-len(bounds) :] == bounds for clause in dnf)
    # The other tuples of each clause are those of an exact AND of the first two disjunctions.
    assert _dnf_mask(frame, [bounds]).filter(mask(frame, last)).all()
    assert convert_expr_to_dnf(~pred, schema) is None


@pytest.mark.xfail(strict=True, raises=pl.exceptions.PanicException, reason="a Polars panic on an out-of-range literal breaks the parser")
def test_out_of_range_datetime_literal_does_not_crash_pushdown_combine():
    frame = pl.DataFrame({"t": pl.Series([1, None]).cast(pl.Datetime("us"))})
    huge = pl.lit(-(2**63), dtype=pl.Int64).cast(pl.Datetime("ms"))
    pred = (pl.col("t").cast(pl.Datetime("ms"), strict=False) < huge) | (pl.col("t") > 0)
    combined = pushdown_combine({"s": (frame.lazy(), {"t": FilterSpec()})}, lambda s: s["s"])
    assert_frame_equal(combined.filter(pred).collect(), frame.filter(pred))


@pytest.mark.xfail(strict=True, reason="the day bounds of a nanosecond value near the Int64 minimum overflow")
def test_cache_parquet_on_a_nanosecond_date_column_near_its_minimum(tmp_path):
    frame = pl.DataFrame({"t": pl.Series([-(2**63), 0]).cast(pl.Datetime("ns")), "i": [0, 1]})
    pred = pl.col("i") >= 0
    assert_frame_equal(cache_parquet(frame.lazy(), tmp_path, "t", time_unit="daily").filter(pred).collect(), frame, check_row_order=False)


@xfail_if("ns_literal_min_overflows")
def test_by_time_on_a_nanosecond_value_near_its_minimum():
    t = pl.Series([-(2**63)]).cast(pl.Datetime("ns"))
    by_time("t", "1d").build(pl.col("t").is_in(t.implode()))


# ---------------------------------------------------------------------------
# pushdown_combine: generated combines against native Polars
# ---------------------------------------------------------------------------

_BASE = date(2024, 1, 1)
_NAMES = {0: "zero", 1: "one", 2: "two", 3: "three"}


def _previous_business_day(d):
    return d - timedelta(days={0: 3, 6: 2}.get(d.weekday(), 1))


def _offset_monday(d):
    """Not monotone: Sunday maps to Saturday, after Monday's Friday."""
    return d - timedelta(days=3 if d.weekday() == 0 else 1)


def _offset_monday_expr(e: pl.Expr) -> pl.Expr:
    return pl.when(e.dt.weekday() == 1).then(e.dt.offset_by("-3d")).otherwise(e.dt.offset_by("-1d"))


def _rolling(frame):
    return frame.sort("key", "data_date").with_columns(pl.col("v").rolling_sum_by("data_date", window_size="3d").over("key").alias("r"))


def _overlapping(s, mapped: pl.Expr) -> pl.LazyFrame:
    """Calendar days joined with the windows holding each day's mapped day."""
    days = s["c"].with_columns(mapped.alias("m"))
    return days.join(s["w"], how="cross").filter((pl.col("start") <= pl.col("m")) & (pl.col("m") <= pl.col("end"))).drop("m")


# Each shape: specs per entry (entry -> (frame, specs)) and a combine whose output rows depend only on source rows inside
# the declared neighborhoods. Output columns: date (Date), key (Int64) and, for the lookup shape, name (String).
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
}


def _old_polars_null_key_bug(shape, frames):
    """Polars 1.37 matches a null ``by`` key in ``join_asof``, and returns wrong rows for a cross join of two IO
    sources filtered on a null key; such inputs are skipped there."""
    nulls = frames["src"]["key"].null_count() or frames["other"]["key"].null_count()
    return not polars_has("join_asof_null_by_unmatched") and shape in ("asof_join", "self_join_two_entries") and bool(nulls)


@st.composite
def _frames(draw):
    def rows(n_max):
        return draw(st.lists(st.tuples(st.integers(0, 20), st.one_of(st.none(), st.integers(0, 3)), st.integers(-5, 5)), max_size=n_max))

    def frame(n_max):
        drawn = rows(n_max)
        return pl.DataFrame(
            {
                "data_date": [_BASE + timedelta(days=d) for d, _, _ in drawn],
                "key": [k for _, k, _ in drawn],
                "v": [v for _, _, v in drawn],
                "tags": [[v, v + 1][: 1 + abs(v) % 2] for _, _, v in drawn],
            },
            schema={"data_date": pl.Date, "key": pl.Int64, "v": pl.Int64, "tags": pl.List(pl.Int64)},
        )

    src, other = frame(10), frame(8)
    windows = rows(6)
    return {
        "src": src,
        "other": other.select(pl.col("data_date").alias("date"), "key", pl.col("v").alias("w")),
        "calendar": pl.DataFrame({"date": [_BASE + timedelta(days=d) for d in range(-2, 24)]}),
        "windows": pl.DataFrame(
            {
                "start": [_BASE + timedelta(days=d) for d, _, _ in windows],
                "end": [_BASE + timedelta(days=d + abs(w)) for d, _, w in windows],
                "key": [k for _, k, _ in windows],
            },
            schema={"start": pl.Date, "end": pl.Date, "key": pl.Int64},
        ),
    }


def _day(draw):
    return _BASE + timedelta(days=draw(st.integers(-2, 23)))


@st.composite
def _combine_predicates(draw, columns):
    def atom():
        col = draw(st.sampled_from(sorted(columns)))
        if col == "date":
            d = _day(draw)
            return draw(
                st.sampled_from(
                    [
                        pl.col("date") == d,
                        pl.col("date").is_in([d, d + timedelta(days=draw(st.integers(1, 9)))]),
                        pl.col("date") >= d,
                        pl.col("date") < d,
                        pl.col("date").is_between(d, d + timedelta(days=draw(st.integers(0, 5)))),
                        pl.col("date").is_null() | (pl.col("date") == d),
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


def _native(entries, combine, frames):
    return combine({entry: frames[frame].lazy() for entry, (frame, _) in entries.items()})


@SETTINGS
@given(data=st.data(), shape=st.sampled_from(sorted(_SHAPES)))
def test_generated_combine_matches_native(data, shape):
    entries, combine = _SHAPES[shape]
    frames = data.draw(_frames())
    assume(not _old_polars_null_key_bug(shape, frames))
    output = _native(entries, combine, frames).collect_schema()
    pred = data.draw(_combine_predicates({"date", "key", "name"} & set(output.names())))
    sources = {entry: (PredicateTracker(frames[frame]).lazy_frame, specs) for entry, (frame, specs) in entries.items()}
    result = pushdown_combine(sources, combine).filter(pred).collect()
    assert_frame_equal(result, _native(entries, combine, frames).filter(pred).collect(), check_row_order=False)


def _neighborhood(spec, value):
    """The source rows an output value depends on under ``spec``, as a predicate on a source row."""
    if isinstance(spec, IntervalFilterSpec):
        mapping = spec.value_mapping
        m = value + mapping if isinstance(mapping, timedelta) else mapping(value) if callable(mapping) else value
        return lambda w: w[spec.start_col] is not None and w[spec.end_col] is not None and w[spec.start_col] <= m <= w[spec.end_col]
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
def test_generated_combine_depends_only_on_declared_neighborhoods(data, shape):
    """Deleting source rows outside an output row's declared neighborhoods keeps that row's multiplicity.

    This checks the shapes above declare their specs correctly, so that the comparison with native Polars tests the
    pushdown rather than the shapes."""
    entries, combine = _SHAPES[shape]
    frames = data.draw(_frames())
    assume(not _old_polars_null_key_bug(shape, frames))
    output = _native(entries, combine, frames).collect()
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
                or (_neighborhood(sp, row[o])(w) if isinstance(sp, IntervalFilterSpec) else _neighborhood(sp, row[o])(w[sp.source_col or o]))
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


# ---------------------------------------------------------------------------
# Caches: random predicates, cold then warm, against a plain filter
# ---------------------------------------------------------------------------

_NY_FOLD_US = [1_730_611_800_000_000, 1_730_615_400_000_000]  # 2024-11-03 01:30 in New York, before and after the fold
_PARTITION_POOLS = {
    "int": pl.Series([1, 2, 3, None]),
    "uint64": pl.Series([0, 2**64 - 1, 2**63, None], dtype=pl.UInt64),
    "string": pl.Series(["a", "a/b", "None", "~2F", "", None]),
    "categorical": pl.Series(["a", "a/b", "", None], dtype=pl.Categorical),
    "enum": pl.Series(["c", "a", None], dtype=pl.Enum(["c", "a", "b"])),
    "decimal": pl.Series([Decimal("0.10"), Decimal("-1.25"), None], dtype=pl.Decimal(38, 2)),
    "datetime_us_ny": pl.Series([*_NY_FOLD_US, _NY_FOLD_US[0] + 1, None]).cast(pl.Datetime("us", NY)),
    "datetime_ms": pl.Series([1_704_067_200_000, 1_704_067_200_001, None]).cast(pl.Datetime("ms")),
    "date": pl.Series([19_723, 19_724, None], dtype=pl.Int32).cast(pl.Date),
    "time": pl.Series([0, 1, 1000, None]).cast(pl.Time),
    "float": pl.Series([0.0, -0.0, float("nan"), 1.5, None]),
    "bool": pl.Series([True, False, None]),
}
# Known bug, see test_cache_parquet_on_sub_second_time_keys: older Polars names Time partitions to the second.
_CACHE_PARQUET_POOLS = (
    _PARTITION_POOLS if polars_has("time_partition_subsecond") else _PARTITION_POOLS | {"time": pl.Series([0, 10**9, 61 * 10**9, None]).cast(pl.Time)}
)
_DAYS = [date(2024, 1, 31), date(2024, 2, 1), date(2024, 2, 29), date(2024, 3, 1), None]


@st.composite
def _partitioned_cases(draw, pools=_PARTITION_POOLS):
    """A frame with a date column, a partition column ``x`` of values from one of ``pools`` and a row id, and three
    predicates on them."""
    kind = draw(st.sampled_from(sorted(pools)))
    pool = pools[kind]
    zoned = draw(st.booleans())
    n = draw(st.integers(1, 12))
    xs = draw(st.lists(st.integers(0, len(pool) - 1), min_size=n, max_size=n))
    ds = draw(st.lists(st.integers(0, len(_DAYS) - 1), min_size=n, max_size=n))
    day = pl.Series([_DAYS[i] for i in ds], dtype=pl.Date)
    if zoned:
        day = day.cast(pl.Datetime("us")).dt.offset_by("23h30m").dt.replace_time_zone(NY)
    frame = pl.DataFrame({"date": day, "x": pool.gather(xs), "i": list(range(n))})

    def x_atom():
        literal = pl.lit(pool.slice(draw(st.integers(0, len(pool) - 2)), 1)).first()
        op = draw(st.sampled_from(["==", "!=", "in", "null", ">=", "ne_missing"]))
        if op == "==":
            return pl.col("x") == literal
        if op == "!=":
            return pl.col("x") != literal
        if op == "ne_missing":
            return pl.col("x").ne_missing(literal)
        if op == "null":
            return pl.col("x").is_null()
        if op == ">=" and kind not in ("string", "categorical", "bool"):
            return pl.col("x") >= literal
        members = pool.gather(draw(st.lists(st.integers(0, len(pool) - 2), min_size=1, max_size=3)))
        return pl.col("x").is_in(members.implode())

    def date_atom():
        d = _DAYS[draw(st.integers(0, len(_DAYS) - 2))]
        literal = pl.lit(d) if not zoned else pl.lit(datetime.combine(d, datetime.min.time())).dt.replace_time_zone(NY).dt.offset_by("12h")
        op = draw(st.sampled_from([">=", "<", "==", "between", "null"]))
        col = pl.col("date") if op != "==" or not zoned else pl.col("date").dt.date()
        if op == "==":
            return col == pl.lit(d)
        if op == "null":
            return col.is_null()
        if op == "between":
            return col.is_between(literal, literal.dt.offset_by(f"{draw(st.integers(0, 40))}d"))
        return col >= literal if op == ">=" else col < literal

    def predicate(depth=0):
        choice = draw(st.integers(0, 4 if depth < 2 else 1))
        if choice == 0:
            return x_atom()
        if choice == 1:
            return date_atom()
        if choice == 2:
            return predicate(depth + 1) & predicate(depth + 1)
        if choice == 3:
            return predicate(depth + 1) | predicate(depth + 1)
        return ~predicate(depth + 1)

    return frame, [predicate() for _ in range(3)]


@no_panic
def _check_cache_keeps_filtered_rows(lf: pl.LazyFrame, frame: pl.DataFrame, predicates: list) -> None:
    """Each predicate, applied in turn to the cached frame (cold, then warm), keeps the rows a plain filter keeps."""
    for pred in [*predicates, pl.lit(True), *predicates]:
        try:
            expected = frame.lazy().filter(pred).collect()
        except POLARS_ERRORS:
            continue
        note(f"physical={ {c: frame[c].to_physical().to_list() for c in frame.columns} } schema={frame.schema} pred={show(pred)}")
        assert_frame_equal(lf.filter(pred).collect(), expected, check_row_order=False)


@SETTINGS
@given(case=_partitioned_cases(_CACHE_PARQUET_POOLS), time_unit=st.sampled_from(["daily", "monthly"]))
# A query that caches only null keys once broke later queries.
@example(
    case=(
        pl.DataFrame({"date": [date(2024, 1, 31), date(2024, 2, 1)], "x": [None, 1], "i": [0, 1]}),
        [pl.col("date") < date(2024, 2, 1), pl.col("x") == 1],
    ),
    time_unit="monthly",
)
# A one-sided date query on a cache that holds only the null-date partition.
@example(
    case=(pl.DataFrame({"date": [None], "x": [1], "i": [0]}, schema_overrides={"date": pl.Date}), [pl.col("date") >= date(2024, 2, 1)]),
    time_unit="daily",
)
def test_cache_parquet_keeps_the_rows_a_filter_keeps(tmp_path_factory, case, time_unit):
    frame, predicates = case
    root = tmp_path_factory.mktemp("cache")
    try:
        for cache_mode in (CacheMode.CACHE, CacheMode.CACHE, CacheMode.READ_ONLY):
            _check_cache_keeps_filtered_rows(
                cache_parquet(frame.lazy(), root, "date", time_unit=time_unit, extra_partition_cols="x", cache_mode=cache_mode),
                frame,
                predicates,
            )
    finally:
        shutil.rmtree(root)


@SETTINGS
@given(case=_partitioned_cases(), partition_cols=st.sampled_from([("x",), ("date", "x"), ()]))
# A literal the predicate analysis cannot read, so the query names no partition.
@example(
    case=(pl.DataFrame({"x": [0.0, float("nan")], "i": [0, 1]}), [pl.lit(True), pl.col("x") == pl.lit(pl.Series([0.0])).first(), pl.lit(True)]),
    partition_cols=("x",),
)
def test_cache_memory_keeps_the_rows_a_filter_keeps(case, partition_cols):
    frame, predicates = case
    _check_cache_keeps_filtered_rows(cache_memory(frame.lazy(), schema=frame.schema, partition_cols=partition_cols), frame, predicates)


_NAN_KEY = (pl.DataFrame({"x": [float("nan")], "i": [0]}), [pl.col("x").is_nan()])


@SETTINGS
@given(case=_partitioned_cases(), partition_cols=st.sampled_from([("x",), ("date", "x"), ()]), column_granularity=st.booleans())
@example(case=_NAN_KEY, partition_cols=("x",), column_granularity=True)
@example(case=_NAN_KEY, partition_cols=("x",), column_granularity=False)
def test_cache_keeps_the_rows_a_filter_keeps(case, partition_cols, column_granularity):
    frame, predicates = case
    _check_cache_keeps_filtered_rows(
        frame.lazy().piot.cache({}, order_by="i", partition_cols=partition_cols, column_granularity=column_granularity),
        frame,
        predicates,
    )


@SETTINGS
@given(case=frames_and_predicates(max_columns=2, depth=2))
# Polars delivers this predicate as ``c0 && (c0 > 0).cast(Int8)``, whose conjunct ``c0`` is not Boolean.
@example(
    case=(pl.DataFrame({"c0": pl.Series([0, 1, 2, 3, -1, None], dtype=pl.Int8)}), pl.all_horizontal(pl.col("c0"), pl.col("c0") > 0).cast(pl.Boolean))
)
def test_cache_memory_keeps_the_rows_of_random_predicates(case):
    """Partitioned on every column of a wide dtype mix."""
    frame, pred = case
    frame = frame.with_row_index("i")
    _check_cache_keeps_filtered_rows(cache_memory(frame.lazy(), schema=frame.schema, partition_cols=frame.columns[1:]), frame, [pred])


def test_cache_parquet_after_caching_only_null_partition_keys(tmp_path):
    frame = pl.DataFrame({"date": [date(2024, 1, 31), date(2024, 2, 1)], "x": [None, 1], "i": [0, 1]})
    lf = cache_parquet(frame.lazy(), tmp_path, "date", extra_partition_cols="x")
    for pred in [pl.col("date") < date(2024, 2, 1), pl.col("x") == 1]:
        assert_frame_equal(lf.filter(pred).collect(), frame.filter(pred))


@pytest.mark.parametrize("column_granularity", [True, False])
def test_cache_on_a_nan_partition_key(column_granularity):
    frame = pl.DataFrame({"x": [float("nan")], "i": [0]})
    lf = frame.lazy().piot.cache({}, order_by="i", partition_cols=("x",), column_granularity=column_granularity)
    for pred in [pl.col("x").is_nan(), pl.lit(True)]:
        assert_frame_equal(lf.filter(pred).collect(), frame.filter(pred))


def test_cache_memory_on_a_nan_partition_key():
    frame = pl.DataFrame({"x": [0.0, float("nan")], "i": [0, 1]})
    lf = cache_memory(frame.lazy(), schema=frame.schema, partition_cols="x")
    # A literal the predicate analysis cannot read, so the query names no partition.
    for pred in [pl.lit(True), pl.col("x") == pl.lit(pl.Series([0.0])).first(), pl.lit(True)]:
        assert_frame_equal(lf.filter(pred).collect(), frame.filter(pred), check_row_order=False)


@xfail_unless("time_partition_subsecond", reason="Polars 1.37 names Time partitions to the second")
def test_cache_parquet_on_sub_second_time_keys(tmp_path):
    frame = pl.DataFrame({"date": [date(2024, 1, 31)] * 2, "x": pl.Series([0, 1000]).cast(pl.Time), "i": [0, 1]})
    lf = cache_parquet(frame.lazy(), tmp_path, "date", time_unit="daily", extra_partition_cols="x")
    for pred in [pl.col("x") == pl.lit(frame["x"].slice(0, 1)).first(), pl.lit(True)]:
        assert_frame_equal(lf.filter(pred).collect(), frame.filter(pred), check_row_order=False)


@pytest.mark.xfail(strict=True, raises=(pl.exceptions.PanicException, pl.exceptions.ComputeError), reason="keys Python cannot hold")
@pytest.mark.parametrize("wrap", ["cache_memory", "cache"])
def test_cache_on_a_partition_key_beyond_python_datetime(wrap):
    frame = pl.DataFrame({"t": pl.Series([300_000_000_000_000, 0]).cast(pl.Datetime("ms")), "i": [0, 1]})  # the year 11476
    if wrap == "cache":
        lf = frame.lazy().piot.cache({}, order_by="i", partition_cols=("t",))
    else:
        lf = cache_memory(frame.lazy(), schema=frame.schema, partition_cols="t")
    assert_frame_equal(lf.filter(pl.col("i") >= 0).collect(), frame, check_row_order=False)


def _via_delta(lf, path):
    sink_delta(lf, str(path / "delta"))
    return scan_delta(str(path / "delta"), credential_provider=None)


_UNPIVOTED = {"index": ["date", "i"], "on": ["c0"]}
_ENTRY_POINTS = {
    "cache_memory": lambda lf, path: cache_memory(lf, schema=lf.collect_schema(), partition_cols="c0"),
    "cache": lambda lf, path: lf.piot.cache({}, order_by="i", partition_cols=("c0",)),
    "cache_parquet": lambda lf, path: cache_parquet(lf, path, "date", extra_partition_cols="c0"),
    "probe": lambda lf, path: lf.piot.probe(),
    "debug": lambda lf, path: lf.piot.debug(log_level=logging.DEBUG),
    "concat_named": lambda lf, path: concat_named({("a",): lf}, ["name"]).drop("name"),
    "pushdown_combine": lambda lf, path: pushdown_combine({"s": (lf, {"c0": FilterSpec()})}, lambda s: s["s"]),
    "filtered_join": lambda lf, path: lf.piot.filtered_join(lf.select("i"), on="i"),
    "filtered_join_asof": lambda lf, path: lf.piot.filtered_join_asof(lf.select("i"), on="i"),
    "pushdown_pivot": lambda lf, path: pushdown_pivot(lf.unpivot(**_UNPIVOTED), "variable", ["c0"], index=["date", "i"], values="value"),
    "pushdown_unpivot": lambda lf, path: pushdown_unpivot(lf, **_UNPIVOTED).drop("variable").rename({"value": "c0"}),
    "scan_narwhals": lambda lf, path: scan_narwhals(nw.from_native(lf), fetch_size=10),
    "scan_delta": _via_delta,
}


_INTEGER_OPERANDS = {
    "cast": pl.all_horizontal(pl.col("c0"), pl.col("c0") > 0).cast(pl.Boolean),
    "column": pl.all_horizontal(pl.col("c0"), pl.col("c0")).cast(pl.Boolean),
}


@pytest.mark.parametrize("operand", sorted(_INTEGER_OPERANDS))
@pytest.mark.parametrize("entry_point", sorted(_ENTRY_POINTS))
def test_an_integer_operand_of_all_horizontal_is_filtered(entry_point, operand, tmp_path):
    """Polars passes these predicates to an IO source as ``c0 && (c0 > 0).cast(Int8)`` and ``c0 && i``.

    The logical operands are not Boolean (``logical_operands_reach_source_uncast``).
    """
    frame = pl.DataFrame({"date": [date(2024, 1, 1)] * 6, "i": range(6), "c0": pl.Series([0, 1, 2, 3, -1, None], dtype=pl.Int8)})
    pred = _INTEGER_OPERANDS[operand]
    lf = _ENTRY_POINTS[entry_point](frame.lazy(), tmp_path)
    assert_frame_equal(lf.filter(pred).collect(), frame.filter(pred), check_row_order=False)


@pytest.mark.parametrize("entry_point", ["debug", "concat_named", "pushdown_combine"])
def test_a_boolean_operand_keeps_the_predicate_pushable(entry_point, tmp_path):
    """An operation on a Boolean column reaches a nested source as it was written, not wrapped."""
    frame = pl.DataFrame({"date": [date(2024, 1, 1)] * 3, "i": range(3), "c0": [True, False, None]})
    tracker = PredicateTracker(frame)
    pred = pl.col("c0") & (pl.col("i") > 0)
    lf = _ENTRY_POINTS[entry_point](tracker.lazy_frame, tmp_path)
    assert_frame_equal(lf.filter(pred).collect(), frame.filter(pred), check_row_order=False)
    assert "fill_null" not in str(tracker.last_predicate)


@pytest.mark.xfail(strict=True, raises=pl.exceptions.ComputeError, reason="Polars passes a bitwise NOT of the integer operand to the IO source")
def test_a_negated_integer_operand_of_all_horizontal_is_filtered():
    frame = pl.DataFrame({"i": range(6), "c0": pl.Series([0, 1, 2, 3, -1, None], dtype=pl.Int8)})
    lf = cache_memory(frame.lazy(), schema=frame.schema, partition_cols="c0")
    expected = frame.filter(~((pl.col("c0") != 0) & (pl.col("c0") > 0)))
    assert_frame_equal(lf.filter(~pl.all_horizontal(pl.col("c0"), pl.col("c0") > 0)).collect(), expected, check_row_order=False)


# ---------------------------------------------------------------------------
# scan_delta: partition pruning against a read of every file
# ---------------------------------------------------------------------------

_HOUR_US = 3_600 * 10**6
_DELTA_VALUES = {
    pl.Int64(): st.integers(-5, 300),
    pl.Int8(): st.integers(-5, 100),
    pl.Boolean(): st.booleans(),
    pl.String(): st.sampled_from(["", "a", "ab", "b", "B", " x", "é", "a/b", "None"]),
    pl.Date(): st.integers(19_000, 19_040),
    pl.Datetime("us"): st.integers(0, 60).map(lambda h: 1_704_067_200 * 10**6 + h * _HOUR_US),
    pl.Datetime("us", "UTC"): st.integers(0, 60).map(lambda h: 1_704_067_200 * 10**6 + h * _HOUR_US),
    pl.Datetime("ns"): st.tuples(st.integers(0, 60), st.sampled_from([0, 500])).map(
        lambda p: (1_704_067_200 * 10**6 + p[0] * _HOUR_US) * 1000 + p[1]
    ),
    pl.Float64(): st.sampled_from([0.0, 1.5, -2.0, math.nan]),
    pl.Decimal(10, 2): st.integers(0, 500),  # deltalake cannot write a negative Decimal partition value
}


@st.composite
def _delta_cases(draw):
    dtypes = draw(st.lists(st.sampled_from(list(_DELTA_VALUES)), min_size=1, max_size=3))
    schema = {f"c{i}": d for i, d in enumerate(dtypes)}
    pools = {name: draw(st.lists(_DELTA_VALUES[d], min_size=1, max_size=3)) for name, d in schema.items()}
    columns = {name: native(d, draw(st.lists(st.sampled_from([*pools[name], None]), min_size=5, max_size=5))) for name, d in schema.items()}
    frame = pl.DataFrame({**columns, "v": list(range(5))})
    partition_by = draw(st.lists(st.sampled_from(list(schema)), min_size=1, max_size=2, unique=True))
    return frame, partition_by, predicate(draw, schema, pools, depth=2)


@SETTINGS
@given(case=_delta_cases())
def test_scan_delta_pruning_keeps_the_rows_a_filter_keeps(tmp_path_factory, case):
    """Partition pruning keeps every row the predicate keeps and only rows a read of every file gives."""
    pytest.importorskip("deltalake")
    frame, partition_by, pred = case
    note(f"partition_by={partition_by} pred={show(pred)}")
    root = tmp_path_factory.mktemp("delta")
    table_path = str(root / "table")
    try:
        sink_delta(frame.lazy(), table_path, mode="overwrite", delta_write_options={"partition_by": partition_by})
        unpruned = scan_delta(table_path, credential_provider=None, pushdown_predicate_deltalake=False)
        try:
            kept, unpruned_kept = set(unpruned.collect().filter(pred)["v"]), set(unpruned.filter(pred).collect()["v"])
        except POLARS_ERRORS:
            return
        # Filtering a scan of hive partitions may keep null partition values that filtering the collected frame drops.
        assert kept <= set(scan_delta(table_path, credential_provider=None).filter(pred).collect()["v"]) <= unpruned_kept
    finally:
        shutil.rmtree(root)


# ---------------------------------------------------------------------------
# Partitioners: the partitions built from a delivered predicate cover its rows
# ---------------------------------------------------------------------------


@SETTINGS
@given(case=frames_and_predicates(), every=st.sampled_from(["1d", "1mo", "1y"]), width=st.sampled_from([1, 7.5, 10**6]))
def test_partitions_cover_the_rows_a_predicate_keeps(case, every, width):
    frame, pred = case
    note_case(frame, pred)
    keep = native_keep(frame, pred)
    assume(keep is not None and keep.any())
    pushed = delivered(frame, pred)
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
