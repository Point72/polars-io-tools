"""piot.cache: per-column, per-partition caching against a source that counts the columns it is asked for.

Step tables give a sequence of queries, the source columns each must fetch and the cache size after it; every
result is compared with the same query on the uncached frame.
"""

import os
import pickle
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import NamedTuple

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_io_tools.io_sources  # noqa: F401
from polars_io_tools.io_sources.lazy_cache import _is_contradiction
from polars_io_tools.io_sources.util import register_io_source_with_is_pure
from polars_io_tools.tests.helpers.catalogue import Case, params


@dataclass
class CallRecord:
    """Record of a single call to the IO source."""

    with_columns: list[str] | None
    predicate_str: str | None
    n_rows: int | None


@dataclass
class TrackableSource:
    """An IO source over ``data`` that records the columns and predicate of every call."""

    data: pl.DataFrame
    calls: list[CallRecord] = field(default_factory=list)

    def reset(self):
        self.calls.clear()

    def get_column_counts(self) -> dict[str, int]:
        """How often each column was requested; a call without a projection requests every column."""
        counts: dict[str, int] = {}
        for call in self.calls:
            for col in call.with_columns or list(self.data.columns):
                counts[col] = counts.get(col, 0) + 1
        return counts

    def as_lazy_frame(self) -> pl.LazyFrame:
        tracker = self

        def source_generator(with_columns, predicate, n_rows, batch_size) -> Iterator[pl.DataFrame]:
            tracker.calls.append(CallRecord(with_columns, str(predicate) if predicate is not None else None, n_rows))
            df = tracker.data.lazy()
            if predicate is not None:
                df = df.filter(predicate)
            if with_columns is not None:
                df = df.select(with_columns)
            if n_rows is not None:
                df = df.head(n_rows)
            if batch_size is None:
                yield df.collect()
            else:
                yield from df.collect().iter_slices(n_rows=batch_size)

        return register_io_source_with_is_pure(source_generator, schema=self.data.schema, validate_schema=False)


@pytest.fixture
def source():
    data = pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "y": ["a", "b", "c", "a", "b", "c"],
            "p": [True, True, True, False, False, False],
            "x2": [10, 20, 30, 40, 50, 60],
            "y2": ["A", "B", "C", "A", "B", "C"],
            "p2": [False, False, False, True, True, True],
        }
    )
    src = TrackableSource(data=data)
    yield src
    src.reset()


@pytest.fixture
def df(source):
    return source.as_lazy_frame()


@pytest.mark.parametrize(
    ("pred", "select", "empty"),
    [
        (None, ["x2", "y2"], False),
        (pl.col("p"), "x2", False),
        (pl.col("p") & ~pl.col("p"), "x2", True),
        (pl.col("p2"), "x2", False),
        (pl.col("p2") & ~pl.col("p2"), "x2", True),
    ],
    ids=["source_basic", "filter", "logically_false", "derived_filter", "logically_false_derived"],
)
def test_source(source, df, pred, select, empty):
    """The tracking source is called once per query, sees the projection and pushed predicate, and filters correctly."""
    out = (df if pred is None else df.filter(pred)).select(select).collect()
    assert len(source.calls) == 1
    call = source.calls[0]
    if pred is None:
        assert set(call.with_columns) == {"x2", "y2"}
    else:
        assert call.predicate_str is not None and pred.meta.root_names()[0] in call.predicate_str
        assert_frame_equal(out, source.data.filter(pred).select(select))
        assert out.is_empty() is empty


class Scenario(NamedTuple):
    filter: pl.Expr = pl.lit(True)
    select: pl.Expr = pl.all()
    head: int | None = None
    partition_cols: tuple[str, ...] = ()

    # Results to test against
    counts: dict[str, int] = {}
    cache_size: int = 0

    # For partitioned data, we "complete" the cache by collecting the frame for all columns and no filters
    complete_counts: dict[str, int] = {}
    complete_cache_size: int = 0


SCENARIOS = [
    # Scenario 0
    Scenario(head=3, counts={"x2": 1, "y2": 1, "p2": 1}, cache_size=6, complete_cache_size=6),
    # Scenario 1
    Scenario(
        filter=pl.col("p2"),
        select=pl.col("x2").max(),
        counts={"x2": 1, "p2": 1},
        cache_size=3,
        complete_counts={"y2": 1},
        complete_cache_size=6,
    ),
    # Scenario 2
    Scenario(
        partition_cols=("p",),
        counts={"x2": 1, "y2": 1, "p2": 1},
        cache_size=12,
        complete_counts={"x2": 1, "y2": 1, "p2": 1},
        complete_cache_size=12,
    ),
    # Scenario 3
    Scenario(
        partition_cols=("y",),
        counts={"x2": 1, "y2": 1, "p2": 1},
        cache_size=18,
        complete_counts={"x2": 1, "y2": 1, "p2": 1},
        complete_cache_size=18,
    ),
    # Scenario 4
    Scenario(
        partition_cols=("y", "p"),
        counts={"x2": 1, "y2": 1, "p2": 1},
        cache_size=36,
        complete_counts={"x2": 1, "y2": 1, "p2": 1},
        complete_cache_size=36,
    ),
    # Scenario 5
    Scenario(
        filter=pl.col("p"),
        select=pl.col("x2").max(),
        partition_cols=("p",),
        counts={"x2": 1},
        cache_size=3,
        complete_counts={"x2": 1, "y2": 2, "p2": 2},
        complete_cache_size=12,
    ),
    # Scenario 6
    Scenario(
        filter=pl.col("p") & (pl.col("x2") >= 0),  # Add a predicate that doesn't matter
        select=pl.all(),
        partition_cols=("p",),
        counts={"x2": 1, "y2": 1, "p2": 1},
        cache_size=6,
        complete_counts={"x2": 1, "y2": 1, "p2": 1},
        complete_cache_size=12,
    ),
    # Scenario 7
    Scenario(
        filter=pl.col("p"),
        select=pl.all(),
        partition_cols=("y", "p"),
        counts={"x2": 1, "y2": 1, "p2": 1},
        cache_size=18,
        complete_counts={"x2": 1, "y2": 1, "p2": 1},
        complete_cache_size=36,
    ),
]


@pytest.mark.parametrize("cache_mode", ["cache", "rebuild", "ignore"])
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenarios(source, df, cache_mode, scenario):
    """A query populates the cache, a fresh frame over the same cache returns the same, and a full read completes it."""
    cache = {}

    def run(partition_cols):
        # The source's call log is part of the serialized plan, and so of the cache key.
        source.reset()
        lf = df.piot.cache(cache, order_by="x", partition_cols=partition_cols, cache_mode=cache_mode)
        return lf, lf.filter(scenario.filter).select(scenario.select).head(scenario.head).collect()

    df_cache, out_cached = run(scenario.partition_cols)
    assert len(source.calls) >= 1
    assert len(cache) == (0 if cache_mode == "ignore" else scenario.cache_size)
    out = df.filter(scenario.filter).select(scenario.select).head(scenario.head).collect()
    assert_frame_equal(out_cached, out, check_row_order=not scenario.partition_cols)

    # A new frame over the same cache, with the partition columns reversed, holds no state of its own.
    df_cache, out_cached2 = run(list(reversed(scenario.partition_cols)))
    assert len(cache) == (0 if cache_mode == "ignore" else scenario.cache_size)
    assert_frame_equal(out_cached2, out_cached)

    out_cached3 = df_cache.collect()
    assert len(cache) == (0 if cache_mode == "ignore" else scenario.complete_cache_size)
    assert_frame_equal(out_cached3, df.collect(), check_row_order=not scenario.partition_cols)


# Steps: each query on one cache, the columns it must fetch from the source, and the cache size after it.

ANY = range(99)
SOME = range(1, 99)


@dataclass(frozen=True)
class Q:
    """A query; ``counts`` maps a source column to how often it is fetched (``None``: the source is not called)."""

    filter: pl.Expr | None = None
    select: object = None
    counts: dict | None = None
    cache_size: int | None = None
    ordered: bool = True


_y, _p = pl.col("y"), pl.col("p")
_ya, _yb = _y == "a", _y == "b"
_y_ab = _y.is_in(["a", "b"])
_ALL6 = {c: 1 for c in ("x", "y", "p", "x2", "y2", "p2")}
_ALL5 = {c: 1 for c in ("x", "p", "x2", "y2", "p2")}


def STEPS(id, queries, partition_cols=(), column_granularity=True):
    return Case(id, queries, None, {"partition_cols": partition_cols, "column_granularity": column_granularity})


CACHE_STEPS = [
    STEPS(
        "df_partitioned",
        [
            # x2, plus y (partition) and x (order_by), for y=a
            Q(_ya, "x2", {"x2": 1, "y2": 0, "p2": 0}, 3),
            Q(_ya, "p2", {"x2": 0, "y2": 0, "p2": 1}),
            Q(_yb, "y2", {"x2": 0, "y2": 1, "p2": 0}),
            Q(_yb, ["x2", "y2"], {"x2": 1, "y2": 0, "p2": 0}),
            # only y2 for y=a and p2 for y=b are missing
            Q(_y_ab, None, {"x2": 0, "y2": 1, "p2": 1}, ordered=False),
        ],
        ("y",),
    ),
    STEPS(
        "df_multi_partitioned",
        [
            # x2, y, p and x for y=a/p=True and y=a/p=False
            Q(_ya, "x2", {"x2": 1, "y2": 0, "p2": 0}, 8),
            Q(_ya, "x2", {"x2": ANY}, 8),
            Q(_ya, "p2", {"x2": 0, "y2": 0, "p2": SOME}),
            Q(_yb, "y2", {"x2": 0, "y2": 1, "p2": 0}),
            Q(_yb, ["x2", "y2"], {"x2": SOME, "p2": 0}),
            Q(_y_ab, None, {}, ordered=False),
        ],
        ("y", "p"),
    ),
    STEPS(
        "df_multi_partitioned_contradiction",
        [
            Q([_ya, _p == pl.lit(True)], "x2", {"x2": 1, "y2": 0, "p2": 0}, 4),
            Q([_ya, _p == pl.lit(True)], "x2", {"x2": 0}, 4),
            Q([_yb, _p == pl.lit(False)], "x2", {"x2": 1, "y2": 0, "p2": 0}, 8),
            Q([_yb, _p == pl.lit(True)], "x2", {"x2": 1, "y2": 0, "p2": 0}, 12),
            Q([_yb, _p.eq(True)], "x2", {"x2": 0}, 12),
            # needs y=c/p=True
            Q(_p.eq(True), "x2", {"x2": 1, "y2": 0, "p2": 0}, 16),
            Q(_y_ab & _p.eq(True), "x2", {"x2": 0}, 16),
            # needs y=a/p=False
            Q(_y_ab & _p.is_in([False, True]), "x2", {"x2": 1, "y2": 0, "p2": 0}, 20, ordered=False),
            Q(_y_ab & _p.is_in([False, True]), "x2", {"x2": 0}, 20, ordered=False),
        ],
        ("y", "p"),
    ),
    # With column_granularity=False the first touch of a block fetches its whole schema.
    STEPS("column_granularity_false_fills_all_columns", [Q(None, "x2", _ALL6, 6), Q(None, ["y2", "p2"], None)], column_granularity=False),
    STEPS(
        "column_granularity_false_preserves_partitioning",
        [Q(_ya, "x2", _ALL5), Q(_ya, "p2", None), Q(_yb, "x2", _ALL5)],
        ("y",),
        column_granularity=False,
    ),
]


@pytest.mark.parametrize(("queries", "partition_cols", "column_granularity"), params(CACHE_STEPS, "pred", "partition_cols", "column_granularity"))
def test_cache_steps(source, df, queries, partition_cols, column_granularity):
    cache = {}
    df_cache = df.piot.cache(cache, order_by="x", partition_cols=partition_cols, column_granularity=column_granularity)
    for q in queries:
        source.reset()

        def query(lf, q=q):
            lf = lf if q.filter is None else lf.filter(*q.filter) if isinstance(q.filter, list) else lf.filter(q.filter)
            return (lf if q.select is None else lf.select(q.select)).collect()

        out = query(df_cache)
        counts = source.get_column_counts()
        if q.counts is None:
            assert source.calls == []
        for col, want in q.counts.items() if q.counts else ():
            assert counts.get(col, 0) in (want if isinstance(want, range) else (want,)), (col, counts)
        if q.cache_size is not None:
            assert len(cache) == q.cache_size
        assert_frame_equal(out, query(df), check_row_order=q.ordered)


def test_streaming(df):
    df_cache = df.piot.cache({}, order_by="x")
    assert_frame_equal(df.collect(), df_cache.collect(engine="streaming"), check_row_order=False)
    with pl.Config() as cfg:
        cfg.set_streaming_chunk_size(2)
        assert_frame_equal(df.collect(), df_cache.collect(engine="streaming"), check_row_order=False)


@pytest.mark.parametrize("cache_arg", [None, {}, "diskcache"])
def test_cache_types(cache_arg, source, df, tmp_path):
    """The default cache, a dict and a diskcache each serve a repeated query without calling the source."""
    cache = cache_arg
    if cache_arg == "diskcache":
        diskcache = pytest.importorskip("diskcache")
        cache = diskcache.Cache(tmp_path / "cache")
    df_cache = df.piot.cache(cache, order_by="x")
    for expected_calls in (1, 0):
        source.reset()
        df_cache.select("x2").collect()
        assert len(source.calls) == expected_calls
    source.reset()
    df.piot.cache(cache, order_by="x").select("x2").collect()
    assert len(source.calls) == 0
    if cache is not None:
        assert len(cache)


def test_docstring():
    assert len(pl.LazyFrame.piot.cache.__doc__.split("\n")) > 10


# _is_contradiction: (predicate, schema) -> whether no row can match


def C(id, expr, expected, schema=None):
    return Case(id, expr, expected, {"schema": schema})


x, a, b, c = pl.col("x"), pl.col("a"), pl.col("b"), pl.col("c")
_x_5_6_excl = x.is_between(5, 6, closed="both") & x.ne(5) & x.ne(6)
_dt_excl = x.is_between(datetime(2024, 1, 1), datetime(2024, 1, 2), closed="both") & x.ne(datetime(2024, 1, 1)) & x.ne(datetime(2024, 1, 2))
_d_excl = x.is_between(date(2024, 1, 1), date(2024, 1, 2), closed="both") & x.ne(date(2024, 1, 1)) & x.ne(date(2024, 1, 2))
_d3 = x.ne(date(2024, 1, 1)) & x.ne(date(2024, 1, 2)) & x.ne(date(2024, 1, 3))

CONTRADICTIONS = [
    C("single_clause_contradiction", (a == 5) & (a == 6), True),
    C("single_clause_not_contradiction", (a == 5) & (a > 0), False),
    C("single_clause_eq_and_gt", (a == 5) & (a > 5), True),
    C("all_clauses_contradictions", ((a == 5) & (a == 6)) | ((b < 0) & (b > 10)), True),
    C("one_clause_not_contradiction", ((a == 5) & (a == 6)) | ((b > 0) & (b < 10)), False),
    C("no_clause_contradiction", ((a == 5) & (a > 0)) | ((b > 0) & (b < 10)), False),
    C("literal_true", pl.lit(True), False),
    C("literal_false", pl.lit(False), True),
    C("nested_all_contradictions", (((a == 1) & (a == 2)) | ((b == 3) & (b == 4))) & ((c > 5) & (c < 0)), True),
    C("nested_one_path_not_contradiction", (((a == 1) & (a > 0)) | ((b == 3) & (b == 4))) & ((c > 0) & (c < 10)), False),
    C("is_null_and_value", x.is_null() & (x == 5), True),
    C("is_null_or_value", x.is_null() | (x == 5), False),
    C("is_not_null_and_is_null", x.is_not_null() & x.is_null(), True),
    C("is_not_null_and_value", x.is_not_null() & (x == 5), False),
    C("not_literal_false", ~pl.lit(False), False),
    C("always_true_or", (a > 0) | (a <= 0), False),
    C("is_between_disjoint", x.is_between(1, 5) & x.is_between(10, 15), True),
    C("is_between_and_not_disjoint", x.is_between(1, 5) & ~x.is_between(10, 15), False),
    C("is_between_and_not_same", x.is_between(1, 5) & ~x.is_between(1, 5), True),
    C("is_between_overlapping", x.is_between(1, 10) & x.is_between(5, 15), False),
    C("is_between_value_outside", x.is_between(1, 5) & (x == 10), True),
    C("is_between_value_inside", x.is_between(1, 5) & (x == 3), False),
    C("is_between_reversed_bounds", x.is_between(5, 1) & (x == 3), True),
    C("is_between_inclusive_bound", x.is_between(1, 5, closed="both") & (x == 5), False),
    C("is_between_exclusive_upper_bound", x.is_between(1, 5, closed="left") & (x == 5), True),
    C("is_between_open_bounds_or_either_bound", x.is_between(5, 6, closed="none") & ((x == 5) | (x == 6)) & x.is_between(4, 7, closed="none"), True),
    C("is_in_disjoint", x.is_in([1, 2, 3]) & x.is_in([4, 5, 6]), True),
    C("is_in_overlapping", x.is_in([1, 2, 3]) & x.is_in([3, 4, 5]), False),
    C("is_in_value_outside", x.is_in([1, 2, 3]) & (x == 4), True),
    C("is_in_value_inside", x.is_in([1, 2, 3]) & (x == 2), False),
    C("is_in_and_not_same", x.is_in([1, 2, 3]) & x.is_in([1, 2, 3]).not_(), True),
    C("is_in_and_not_other", x.is_in([1, 2, 3]) & x.is_in([4, 5, 6]).not_(), False),
    C("is_in_excluded_value", x.is_in([1, 2, 3]) & ~x.is_in([3, 4, 5]) & (x == 3), True),
    C("is_in_empty", x.is_in([]) & (x == 5), True),
    C("is_in_value_gt_exclusive", x.is_in([5]) & (x > 5), True),
    C("is_in_value_eq", x.is_in([5]) & (x == 5), False),
    # Without a schema the bounds cannot be enumerated.
    C("excluded_bounds_no_schema", _x_5_6_excl, False),
    C("excluded_bounds_int", _x_5_6_excl, True, pl.Schema({"x": pl.Int32})),
    C("excluded_bounds_float", _x_5_6_excl, False, pl.Schema({"x": pl.Float64})),
    C("excluded_bounds_datetime", _dt_excl, False, pl.Schema({"x": pl.Datetime()})),
    C("excluded_bounds_date", _d_excl, True, pl.Schema({"x": pl.Date()})),
    C("excluded_dates_closed_left", x.is_between(date(2024, 1, 1), date(2024, 1, 4), closed="left") & _d3, True, pl.Schema({"x": pl.Date()})),
    C("excluded_dates_closed_right", x.is_between(date(2024, 1, 1), date(2024, 1, 4), closed="right") & _d3, False, pl.Schema({"x": pl.Date()})),
]


@pytest.mark.parametrize(("expr", "schema", "expected"), params(CONTRADICTIONS, "pred", "schema", "expect"))
def test_lazy_cache_is_contradiction(expr, schema, expected):
    assert (_is_contradiction(expr) if schema is None else _is_contradiction(expr, schema)) == expected


def test_pickle(df):
    """Test that the namespace sticks even after pickling/unpickling"""
    df = pickle.loads(pickle.dumps(df))
    assert hasattr(df, "piot")
    assert hasattr(df.piot, "cache")


def test_nondeterministic_source_stays_aligned_with_order_by():
    """A non-deterministic (row-shuffling) source stays aligned when order_by is given.

    Columns are cached independently, so a source that emits a different row order on each
    collect could previously misalign cached columns. Passing a unique ``order_by`` forces
    every cached block into the same canonical order, guaranteeing correct recombination.
    """

    def create_nondeterministic_source(data: pl.DataFrame) -> pl.LazyFrame:
        """Creates a LazyFrame backed by a source that shuffles rows on each collect."""
        schema = data.schema
        base_data = data

        def source_generator(
            with_columns: list[str] | None,
            predicate: pl.Expr | None,
            n_rows: int | None,
            batch_size: int | None,
        ) -> Iterator[pl.DataFrame]:
            # Shuffle the data each time - this simulates non-deterministic ordering
            shuffled = base_data.sample(fraction=1.0, shuffle=True)

            df = shuffled.lazy()
            if predicate is not None:
                df = df.filter(predicate)
            if with_columns is not None:
                df = df.select(with_columns)
            if n_rows is not None:
                df = df.head(n_rows)

            if batch_size is None:
                yield df.collect()
            else:
                yield from df.collect().iter_slices(n_rows=batch_size)

        return register_io_source_with_is_pure(source_generator, schema=schema, validate_schema=False)

    # Create data where we can verify alignment
    data = pl.DataFrame(
        {
            "key": list(range(1000)),
            "left_val": list(range(1000)),
            "right_val": [k + 1000000 for k in range(1000)],
        }
    )

    # Create non-deterministic source
    lf = create_nondeterministic_source(data)

    cache = {}
    df_cached = lf.piot.cache(cache, order_by="key")

    # First collect: cache only left_val (key co-fetched for ordering)
    _ = df_cached.select(["key", "left_val"]).collect()

    # Second collect: cache only right_val (source shuffles differently!)
    _ = df_cached.select(["key", "right_val"]).collect()

    # Third collect: get all columns together from cache
    result = df_cached.select(["key", "left_val", "right_val"]).collect()

    # Verify alignment: right_val should be left_val + 1000000 if rows are aligned
    misaligned = result.filter(pl.col("right_val") != (pl.col("left_val") + 1000000))

    # With a unique order_by, the canonical ordering keeps the columns aligned
    assert misaligned.is_empty(), "Columns must stay aligned once a unique order_by is provided."


def test_piot_cache_no_warning():
    """cache() no longer emits an ordering warning now that order_by guarantees alignment."""
    import warnings

    df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}).lazy()
    cache = {}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = df.piot.cache(cache, order_by="x").collect()

    assert not any("consistent row ordering" in str(warning.message) for warning in w), "No ordering warning should be emitted."


def test_piot_cache_non_unique_order_by_raises():
    """A non-unique order_by is rejected because it cannot guarantee a total order."""
    df = pl.DataFrame({"k": [1, 1, 2], "v": [10, 20, 30]}).lazy()
    with pytest.raises(pl.exceptions.ComputeError, match="does not uniquely identify rows"):
        df.piot.cache({}, order_by="k").select(["k", "v"]).collect()


def test_piot_cache_validate_false_skips_uniqueness_check():
    """validate=False bypasses the uniqueness check for callers that guarantee it themselves."""
    df = pl.DataFrame({"k": [1, 1, 2], "v": [10, 20, 30]}).lazy()
    out = df.piot.cache({}, order_by="k", validate=False).select(["v"]).collect()
    assert out.height == 3


def test_piot_cache_order_by_not_in_schema_raises():
    """An order_by column missing from the schema fails fast at definition time."""
    df = pl.DataFrame({"x": [1, 2, 3]}).lazy()
    with pytest.raises(ValueError, match="order_by columns not found"):
        df.piot.cache({}, order_by="missing")


def test_piot_cache_shared_cache_different_partition_cols():
    """A shared cache reused with different partition_cols must not collide/crash.

    The partition layout is part of the cache key, so coarse- and fine-grained partitionings
    of the same frame live in separate namespaces instead of reading each other's keys.
    """
    df = pl.DataFrame(
        {
            "p": ["a", "a", "b", "b"],
            "q": [1, 2, 1, 2],
            "x": [1, 2, 3, 4],
            "v": [10, 20, 30, 40],
        }
    ).lazy()
    cache = {}
    fine = df.piot.cache(cache, order_by="x", partition_cols=("p", "q")).select("v").collect()
    coarse = df.piot.cache(cache, order_by="x", partition_cols=("p",)).select("v").collect()
    assert sorted(fine["v"].to_list()) == [10, 20, 30, 40]
    assert sorted(coarse["v"].to_list()) == [10, 20, 30, 40]


def test_unfiltered_select_excludes_cached_partitions(source, df):
    """An unfiltered read over a fully-cached partitioned frame must not re-scan everything.

    It should restrict the discovery query to exclude the already-cached partitions instead of
    issuing a full-source scan (predicate=None).
    """
    cache = {}
    df_cache = df.piot.cache(cache, order_by="x", partition_cols=("y",))

    # Fully populate: every column of every partition.
    full = df_cache.collect()
    source.reset()

    # Unfiltered read again: the discovery query must carry a partition-exclusion predicate.
    out = df_cache.collect()
    preds = [c.predicate_str for c in source.calls]
    assert preds, "expected a partition-discovery query"
    assert all(p is not None for p in preds), f"expected exclusion predicate, got a full re-scan: {preds}"
    assert any("y" in (p or "") for p in preds), f"exclusion should reference partition col y: {preds}"

    assert_frame_equal(out.sort(["y", "x"]), full.sort(["y", "x"]))
    assert_frame_equal(out.sort(["y", "x"]), df.collect().sort(["y", "x"]))


def test_null_partition_unfiltered_incremental_fetch():
    """Regression: a null-valued partition must survive an unfiltered read that fetches more columns.

    First caching only ``x`` populates the null partition; a later unfiltered ``collect`` excludes
    already-cached partitions from discovery, so the missing columns for the null partition are
    fetched by the fixed-partition scan — which must use null-safe equality (``eq_missing``) or it
    returns no rows and the assembled block has mismatched column heights.
    """
    data = pl.DataFrame(
        {"g": [None, None, "a", "a"], "x": [1, 2, 3, 4], "yv": [10, 20, 30, 40]},
        schema={"g": pl.String, "x": pl.Int64, "yv": pl.Int64},
    ).lazy()
    cache = {}
    cached = data.piot.cache(cache, order_by="x", partition_cols=("g",))

    cached.select("x").collect()  # populate x for g=None and g="a"
    out = cached.collect()  # unfiltered: must fetch yv for the null partition without error

    assert_frame_equal(
        out.sort(["g", "x"], nulls_last=True),
        data.collect().sort(["g", "x"], nulls_last=True),
    )


def test_column_granularity_false_and_true_share_cache(source, df):
    """A column_granularity=False fill populates keys a granularity=True reader hits."""
    cache = {}
    # Eager fill of all columns via the False reader.
    df.piot.cache(cache, order_by="x", column_granularity=False).select("x2").collect()

    source.reset()
    # Default (per-column) reader finds every column already cached -> no query.
    out = df.piot.cache(cache, order_by="x").select(["x2", "y2"]).collect()
    assert source.calls == []
    assert_frame_equal(out, df.select(["x2", "y2"]).collect())


@pytest.mark.parametrize("num_threads", [1, 2])
def test_concurrent_collections_shared_cache_do_not_race(num_threads):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import faulthandler; faulthandler.dump_traceback_later(30, exit=True); "
                "from polars_io_tools.tests.io_sources.test_lazy_cache import _collect_concurrently; "
                "_collect_concurrently()"
            ),
        ],
        env={**os.environ, "POLARS_MAX_THREADS": str(num_threads)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _collect_concurrently():
    """Concurrent collections sharing one cache must not raise or corrupt it.

    Barrier-synchronized threads each collect a distinct partition with
    ``column_granularity=False`` so that, at any moment, some threads iterate the shared cache
    during discovery while others publish into it. Without synchronization this raises
    ``RuntimeError: dictionary changed size during iteration``.
    """
    import threading

    n_partitions = 400
    universe = pl.DataFrame(
        {
            "sym": [f"s{i}" for i in range(n_partitions)],
            "x": list(range(n_partitions)),
            "y": [i * 2 for i in range(n_partitions)],
        }
    ).lazy()

    cache: dict = {}
    n_threads = 8
    universes = [universe.clone() for _ in range(n_threads)]
    per_thread = n_partitions // n_threads
    barrier = threading.Barrier(n_threads)
    errors: list[Exception] = []
    results: dict[int, pl.DataFrame] = {}

    def worker(tid: int) -> None:
        try:
            barrier.wait()
            for j in range(per_thread):
                sym = f"s{tid * per_thread + j}"
                out = (
                    universes[tid]
                    .piot.cache(cache, order_by="sym", partition_cols=("sym",), column_granularity=False)
                    .filter(pl.col("sym") == sym)
                    .collect()
                )
                results[tid * per_thread + j] = out
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent collections raised: {errors[:3]}"
    # Every partition was collected correctly, and every column landed in the cache.
    combined = pl.concat([results[i] for i in range(n_partitions)]).sort("x")
    assert_frame_equal(combined, universe.collect().sort("x"))
    assert len(cache) == n_partitions * len(universe.collect_schema())


def test_explicit_df_key_replaces_plan_identity(source, df):
    """With ``df_key``, frames with different plans share cache entries; without it they do not."""
    cache = {}
    same_data_other_plan = df.select(pl.all())

    df.piot.cache(cache, order_by="x", df_key="frame-v1").select("x2").collect()
    source.reset()
    out = same_data_other_plan.piot.cache(cache, order_by="x", df_key="frame-v1").select("x2").collect()
    assert source.calls == []
    assert_frame_equal(out, df.select("x2").collect().sort("x2"), check_row_order=False)

    source.reset()
    same_data_other_plan.piot.cache(cache, order_by="x").select("x2").collect()
    assert source.calls

    source.reset()
    df.piot.cache(cache, order_by="x", df_key="frame-v2").select("x2").collect()
    assert source.calls


def test_explicit_df_key_still_separates_layouts(df):
    """``order_by`` and ``partition_cols`` stay part of the key when ``df_key`` is given."""
    cache = {}
    df.piot.cache(cache, order_by="x", df_key="k").select("x2").collect()
    df.piot.cache(cache, order_by=["x", "y"], df_key="k").select("x2").collect()
    df.piot.cache(cache, order_by="x", partition_cols=("y",), df_key="k").select("x2").collect()
    assert len({k.df_key for k in cache}) == 3


@pytest.mark.skipif(
    getattr(pl.Categorical(), "ordering", "lexical") != "physical",
    reason="Physical categorical ordering is not supported by this Polars version",
)
@pytest.mark.parametrize("order_by", ["key", ("part", "key")])
@pytest.mark.parametrize("partition_cols", [(), ("part",)])
@pytest.mark.parametrize("column_granularity", [False, True])
def test_explicit_df_key_rejects_physical_categorical_ordering(order_by, partition_cols, column_granularity):
    frame = pl.DataFrame({"key": ["b", "a"], "part": ["p", "p"], "value": [20, 10]}).with_columns(pl.col("key").cast(pl.Categorical)).lazy()
    cache = {}
    with pytest.raises(ValueError, match="physical categorical"):
        frame.piot.cache(
            cache,
            order_by=order_by,
            partition_cols=partition_cols,
            column_granularity=column_granularity,
            validate=False,
            df_key="same",
        )
    assert cache == {}
    assert_frame_equal(frame.piot.cache(cache, order_by=order_by).collect(), frame.sort(order_by).collect())
    assert frame.piot.cache(cache, order_by=order_by, cache_mode="ignore", df_key="same") is frame


@pytest.mark.skipif(
    getattr(pl.Categorical(), "ordering", "lexical") != "physical",
    reason="Physical categorical ordering is not supported by this Polars version",
)
@pytest.mark.parametrize(
    "key_expr",
    [
        pl.struct(pl.lit(1).alias("number"), pl.col("key").alias("category")),
        pl.concat_list(pl.col("key")),
        pl.concat_list(pl.col("key")).list.to_array(1),
        pl.struct(pl.concat_list(pl.col("key")).alias("categories")),
        pl.concat_list(pl.struct(pl.col("key").alias("category"))),
        pl.struct(pl.concat_list(pl.col("key")).list.to_array(1).alias("categories")),
    ],
    ids=["struct", "list", "array", "struct-list", "list-struct", "struct-array"],
)
def test_explicit_df_key_rejects_nested_physical_categorical_ordering(key_expr):
    frame = (
        pl.DataFrame({"key": ["b", "a"], "value": [20, 10]})
        .lazy()
        .with_columns(pl.col("key").cast(pl.Categorical))
        .with_columns(key_expr.alias("key"))
    )
    cache = {}
    with pytest.raises(ValueError, match="physical categorical"):
        frame.piot.cache(cache, order_by="key", validate=False, df_key="same")
    assert cache == {}
    assert frame.piot.cache(cache, order_by="key", cache_mode="ignore", df_key="same") is frame


@pytest.mark.skipif(
    getattr(pl.Categorical(), "ordering", "lexical") != "physical",
    reason="Physical categorical ordering is not supported by this Polars version",
)
@pytest.mark.parametrize(
    "key_expr",
    [pl.col("key"), pl.struct(pl.col("key").alias("category")), pl.concat_list(pl.col("key"))],
    ids=["scalar", "struct", "list"],
)
@pytest.mark.parametrize("column_granularity", [False, True])
@pytest.mark.parametrize("cache_mode", ["cache", "rebuild"])
def test_explicit_df_key_rejects_materialized_physical_categorical_ordering(key_expr, column_granularity, cache_mode):
    base = pl.DataFrame({"text": ["v3", "v1", "v2", "v0", "v5", "v4"], "value": [3, 1, 2, 0, 5, 4]}).lazy()
    frame = pl.concat(
        [
            base.with_columns(key=pl.col("text").cast(pl.Categorical("lexical"))).head(3),
            base.with_columns(key=pl.col("text").cast(pl.Categorical("physical"))).tail(3),
        ]
    ).with_columns(key_expr.alias("key"))
    cache = {}
    cached = frame.piot.cache(cache, order_by="key", df_key="same", column_granularity=column_granularity, cache_mode=cache_mode, validate=False)
    with pytest.raises(pl.exceptions.ComputeError, match="physical categorical"):
        cached.select("value").collect()
    assert cache == {}


@pytest.mark.parametrize(
    "key_expr",
    [pl.col("key"), pl.struct(pl.col("key").alias("category")), pl.concat_list(pl.col("key"))],
    ids=["scalar", "struct", "list"],
)
@pytest.mark.parametrize("partition_cols", [(), ("part",)])
@pytest.mark.parametrize("column_granularity", [False, True])
def test_explicit_df_key_lexical_categorical_partial_fill_stays_aligned(key_expr, partition_cols, column_granularity):
    dtype = pl.Categorical()
    if getattr(dtype, "ordering", "lexical") == "physical":
        dtype = pl.Categorical("lexical")
    first = (
        pl.DataFrame({"key": ["b", "a"], "part": ["p", "p"], "left": [20, 10], "right": [200, 100]})
        .with_columns(pl.col("key").cast(dtype))
        .with_columns(key_expr.alias("key"))
        .sort("left")
        .lazy()
    )
    second = (
        pl.DataFrame({"key": ["a", "b"], "part": ["p", "p"], "left": [10, 20], "right": [100, 200]})
        .with_columns(pl.col("key").cast(dtype))
        .with_columns(key_expr.alias("key"))
        .sort("left")
        .lazy()
    )
    cache = {}
    options = {"order_by": "key", "partition_cols": partition_cols, "column_granularity": column_granularity, "df_key": "same"}
    first.piot.cache(cache, **options).select("left").collect()
    out = second.piot.cache(cache, **options).select("left", "right").collect()
    assert_frame_equal(out, pl.DataFrame({"left": [10, 20], "right": [100, 200]}))


_FOLD_US = 1_730_611_800_000_000  # 2024-11-03 05:30 UTC, 01:30 EDT; one hour later is 01:30 EST


@pytest.mark.parametrize(
    "dtype, start, step",
    [
        (pl.Datetime("ns"), 1000, 1),
        (pl.Duration("ns"), 1000, 1),
        (pl.Time, 1000, 1),
        (pl.Datetime("us", "America/New_York"), _FOLD_US, 3_600_000_000),
    ],
)
def test_lossy_key_partitions_do_not_collide(dtype, start, step):
    df = pl.DataFrame({"k": pl.Series([start, start + step, start + 2 * step, None]).cast(dtype), "v": [1, 2, 3, 4]})
    lf = df.lazy().piot.cache({}, order_by="v", partition_cols=["k"])
    for pred in [pl.col("k") == pl.lit(start + step).cast(dtype), pl.col("k") == pl.lit(start).cast(dtype), pl.col("v") > 0]:
        assert_frame_equal(lf.filter(pred).collect().sort("v"), df.filter(pred))
