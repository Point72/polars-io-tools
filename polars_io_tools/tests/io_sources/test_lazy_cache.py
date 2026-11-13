"""piot.cache: per-column, per-partition caching against a source that counts the columns it is asked for.

Step tables give a sequence of queries, the source columns each must fetch and the cache size after it; every
result is compared with the same query on the uncached frame.

The property test at the end (using the Hypothesis library) runs random filters on a cache, cold then warm, and
compares each with a plain filter.
"""

import os
import pickle
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import KW_ONLY, dataclass, field

import cloudpickle
import polars as pl
import pytest
from hypothesis import example, given, strategies as st
from polars.testing import assert_frame_equal

import polars_io_tools.io_sources  # noqa: F401
from polars_io_tools.io_sources import util
from polars_io_tools.io_sources.util import _exclusion_from_frame, register_io_source_with_is_pure
from polars_io_tools.tests.helpers.assertions import failure_note
from polars_io_tools.tests.helpers.cases import HOUR_US, NEW_YORK_FIRST_0130_US, as_params
from polars_io_tools.tests.helpers.pickling import pickle_round_trip
from polars_io_tools.tests.helpers.property_checks import SETTINGS, check_cache_keeps_filtered_rows
from polars_io_tools.tests.helpers.strategies import partitioned_frames_and_predicates


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

    def get_column_counts(self, since: int = 0) -> dict[str, int]:
        """How often each column was requested by the calls from number ``since`` on; a call without a projection
        requests every column."""
        counts: dict[str, int] = {}
        for call in self.calls[since:]:
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
def tracked_frame(source):
    return source.as_lazy_frame()


@dataclass(frozen=True)
class TrackingSourceCase:
    """A query on the tracking source, and whether it returns no rows."""

    id: str
    _: KW_ONLY
    pred: pl.Expr | None
    select: str | list[str]
    empty: bool


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            TrackingSourceCase("source_basic", pred=None, select=["x2", "y2"], empty=False),
            TrackingSourceCase("filter", pred=pl.col("p"), select="x2", empty=False),
            TrackingSourceCase("logically_false", pred=pl.col("p") & ~pl.col("p"), select="x2", empty=True),
            TrackingSourceCase("derived_filter", pred=pl.col("p2"), select="x2", empty=False),
            TrackingSourceCase("logically_false_derived", pred=pl.col("p2") & ~pl.col("p2"), select="x2", empty=True),
        ]
    ),
)
def test_source(source, tracked_frame, case: TrackingSourceCase):
    """The tracking source is called once per query, sees the projection and pushed predicate, and filters correctly."""
    out = (tracked_frame if case.pred is None else tracked_frame.filter(case.pred)).select(case.select).collect()
    assert len(source.calls) == 1
    call = source.calls[0]
    if case.pred is None:
        assert set(call.with_columns) == {"x2", "y2"}
    else:
        assert call.predicate_str is not None and case.pred.meta.root_names()[0] in call.predicate_str
        assert_frame_equal(out, source.data.filter(case.pred).select(case.select))
        assert out.is_empty() is case.empty


@dataclass(frozen=True)
class FreshCacheQueryCase:
    """A query on a fresh cache: ``filter``, then ``select``, then ``head``, with the cache split by ``partition_cols``.

    ``cache_size`` is the number of cache entries after the query, and ``complete_cache_size`` after reading every row
    and column. ``counts`` is how often each of the columns x2, y2 and p2 is fetched from the source by the query, and
    ``complete_counts`` by the full read; the full read is only counted with ``cache_mode="cache"``, the one mode that
    reuses what the query fetched.
    """

    id: str
    _: KW_ONLY
    filter: pl.Expr = field(default_factory=lambda: pl.lit(True))
    select: pl.Expr = field(default_factory=pl.all)
    head: int | None = None
    partition_cols: tuple[str, ...] = ()

    # Results to test against
    counts: dict[str, int] = field(default_factory=dict)
    cache_size: int = 0

    # A full read (every column, no filter) after the query fills the rest of the cache.
    complete_counts: dict[str, int] = field(default_factory=dict)
    complete_cache_size: int = 0


SCENARIOS = [
    FreshCacheQueryCase("head_of_three", head=3, counts={"x2": 1, "y2": 1, "p2": 1}, cache_size=6, complete_cache_size=6),
    FreshCacheQueryCase(
        "filter_then_aggregate",
        filter=pl.col("p2"),
        select=pl.col("x2").max(),
        counts={"x2": 1, "p2": 1},
        cache_size=3,
        complete_counts={"y2": 1},
        complete_cache_size=6,
    ),
    FreshCacheQueryCase(
        "partitioned_by_p",
        partition_cols=("p",),
        counts={"x2": 1, "y2": 1, "p2": 1},
        cache_size=12,
        complete_counts={"x2": 1, "y2": 1, "p2": 1},
        complete_cache_size=12,
    ),
    FreshCacheQueryCase(
        "partitioned_by_y",
        partition_cols=("y",),
        counts={"x2": 1, "y2": 1, "p2": 1},
        cache_size=18,
        complete_counts={"x2": 1, "y2": 1, "p2": 1},
        complete_cache_size=18,
    ),
    FreshCacheQueryCase(
        "partitioned_by_y_and_p",
        partition_cols=("y", "p"),
        counts={"x2": 1, "y2": 1, "p2": 1},
        cache_size=36,
        complete_counts={"x2": 1, "y2": 1, "p2": 1},
        complete_cache_size=36,
    ),
    FreshCacheQueryCase(
        "partition_filter_then_aggregate",
        filter=pl.col("p"),
        select=pl.col("x2").max(),
        partition_cols=("p",),
        counts={"x2": 1},
        cache_size=3,
        complete_counts={"x2": 1, "y2": 2, "p2": 2},
        complete_cache_size=12,
    ),
    FreshCacheQueryCase(
        "partition_filter_with_value_filter",
        filter=pl.col("p") & (pl.col("x2") >= 0),  # Add a predicate that doesn't matter
        select=pl.all(),
        partition_cols=("p",),
        counts={"x2": 1, "y2": 1, "p2": 1},
        cache_size=6,
        complete_counts={"x2": 1, "y2": 1, "p2": 1},
        complete_cache_size=12,
    ),
    FreshCacheQueryCase(
        "partition_filter_partitioned_by_y_and_p",
        filter=pl.col("p"),
        select=pl.all(),
        partition_cols=("y", "p"),
        counts={"x2": 1, "y2": 1, "p2": 1},
        cache_size=18,
        complete_counts={"x2": 1, "y2": 1, "p2": 1},
        complete_cache_size=36,
    ),
]


def _value_column_counts(source: TrackableSource, since: int = 0) -> dict[str, int]:
    """``source.get_column_counts(since)`` for the columns x2, y2 and p2 only."""
    return {col: n for col, n in source.get_column_counts(since).items() if col in ("x2", "y2", "p2")}


@pytest.mark.parametrize("cache_mode", ["cache", "rebuild", "ignore"])
@pytest.mark.parametrize("case", as_params(SCENARIOS))
def test_scenarios(source, tracked_frame, cache_mode, case: FreshCacheQueryCase):
    """A query populates the cache, a fresh frame over the same cache returns the same, and a full read completes it."""
    cache = {}

    def run(partition_cols):
        # The source's call log is part of the serialized plan, and so of the cache key.
        source.reset()
        lf = tracked_frame.piot.cache(cache, order_by="x", partition_cols=partition_cols, cache_mode=cache_mode)
        return lf, lf.filter(case.filter).select(case.select).head(case.head).collect()

    df_cache, out_cached = run(case.partition_cols)
    assert len(source.calls) >= 1
    assert _value_column_counts(source) == case.counts
    assert len(cache) == (0 if cache_mode == "ignore" else case.cache_size)
    out = tracked_frame.filter(case.filter).select(case.select).head(case.head).collect()
    assert_frame_equal(out_cached, out, check_row_order=not case.partition_cols)

    # A new frame over the same cache, with the partition columns reversed, holds no state of its own.
    df_cache, out_cached2 = run(list(reversed(case.partition_cols)))
    assert len(cache) == (0 if cache_mode == "ignore" else case.cache_size)
    assert_frame_equal(out_cached2, out_cached)

    calls_before_full_read = len(source.calls)
    out_cached3 = df_cache.collect()
    assert len(cache) == (0 if cache_mode == "ignore" else case.complete_cache_size)
    if cache_mode == "cache":
        assert _value_column_counts(source, since=calls_before_full_read) == case.complete_counts
    assert_frame_equal(out_cached3, tracked_frame.collect(), check_row_order=not case.partition_cols)


# Fetch counts that accept any number of fetches, or at least one.
ANY_NUMBER_OF_TIMES = range(99)
AT_LEAST_ONCE = range(1, 99)


@dataclass(frozen=True, kw_only=True)
class CacheQuery:
    """One query on a shared cache, and what it must fetch from the source.

    ``filter`` (one expression or a list) and ``select`` make the query. ``fetches`` maps a source column to how often
    the query fetches it (a number or a range); ``served_from_cache`` means the source is not called. ``cache_size`` is the number of cache entries afterwards. ``compare_row_order`` compares the result row by row.
    """

    filter: pl.Expr | list | None = None
    select: object = None
    fetches: dict = field(default_factory=dict)
    served_from_cache: bool = False
    cache_size: int | None = None
    compare_row_order: bool = True


_Y_IS_A, _Y_IS_B = pl.col("y") == "a", pl.col("y") == "b"
_Y_IS_A_OR_B = pl.col("y").is_in(["a", "b"])
_EVERY_COLUMN_ONCE = {column: 1 for column in ("x", "y", "p", "x2", "y2", "p2")}
_EVERY_COLUMN_BUT_Y_ONCE = {column: 1 for column in ("x", "p", "x2", "y2", "p2")}


@dataclass(frozen=True)
class CacheStepsCase:
    """Queries run in turn on one cache; each result must equal the uncached query."""

    id: str
    _: KW_ONLY
    queries: list[CacheQuery]
    partition_cols: tuple[str, ...] = ()
    column_granularity: bool = True


# Query sequences on one cache, the columns each query fetches from the source, and the cache size after it.
CACHE_STEPS = [
    CacheStepsCase(
        "df_partitioned",
        queries=[
            # x2, plus y (partition) and x (order_by), for y=a
            CacheQuery(filter=_Y_IS_A, select="x2", fetches={"x2": 1, "y2": 0, "p2": 0}, cache_size=3),
            CacheQuery(filter=_Y_IS_A, select="p2", fetches={"x2": 0, "y2": 0, "p2": 1}),
            CacheQuery(filter=_Y_IS_B, select="y2", fetches={"x2": 0, "y2": 1, "p2": 0}),
            CacheQuery(filter=_Y_IS_B, select=["x2", "y2"], fetches={"x2": 1, "y2": 0, "p2": 0}),
            # only y2 for y=a and p2 for y=b are missing
            CacheQuery(filter=_Y_IS_A_OR_B, select=None, fetches={"x2": 0, "y2": 1, "p2": 1}, compare_row_order=False),
        ],
        partition_cols=("y",),
    ),
    CacheStepsCase(
        "df_multi_partitioned",
        queries=[
            # x2, y, p and x for y=a/p=True and y=a/p=False
            CacheQuery(filter=_Y_IS_A, select="x2", fetches={"x2": 1, "y2": 0, "p2": 0}, cache_size=8),
            CacheQuery(filter=_Y_IS_A, select="x2", fetches={"x2": ANY_NUMBER_OF_TIMES}, cache_size=8),
            CacheQuery(filter=_Y_IS_A, select="p2", fetches={"x2": 0, "y2": 0, "p2": AT_LEAST_ONCE}),
            CacheQuery(filter=_Y_IS_B, select="y2", fetches={"x2": 0, "y2": 1, "p2": 0}),
            CacheQuery(filter=_Y_IS_B, select=["x2", "y2"], fetches={"x2": AT_LEAST_ONCE, "p2": 0}),
            CacheQuery(filter=_Y_IS_A_OR_B, select=None, fetches={}, compare_row_order=False),
        ],
        partition_cols=("y", "p"),
    ),
    CacheStepsCase(
        "df_multi_partitioned_contradiction",
        queries=[
            CacheQuery(filter=[_Y_IS_A, pl.col("p") == pl.lit(True)], select="x2", fetches={"x2": 1, "y2": 0, "p2": 0}, cache_size=4),
            CacheQuery(filter=[_Y_IS_A, pl.col("p") == pl.lit(True)], select="x2", fetches={"x2": 0}, cache_size=4),
            CacheQuery(filter=[_Y_IS_B, pl.col("p") == pl.lit(False)], select="x2", fetches={"x2": 1, "y2": 0, "p2": 0}, cache_size=8),
            CacheQuery(filter=[_Y_IS_B, pl.col("p") == pl.lit(True)], select="x2", fetches={"x2": 1, "y2": 0, "p2": 0}, cache_size=12),
            CacheQuery(filter=[_Y_IS_B, pl.col("p").eq(True)], select="x2", fetches={"x2": 0}, cache_size=12),
            # needs y=c/p=True
            CacheQuery(filter=pl.col("p").eq(True), select="x2", fetches={"x2": 1, "y2": 0, "p2": 0}, cache_size=16),
            CacheQuery(filter=_Y_IS_A_OR_B & pl.col("p").eq(True), select="x2", fetches={"x2": 0}, cache_size=16),
            # needs y=a/p=False
            CacheQuery(
                filter=_Y_IS_A_OR_B & pl.col("p").is_in([False, True]),
                select="x2",
                fetches={"x2": 1, "y2": 0, "p2": 0},
                cache_size=20,
                compare_row_order=False,
            ),
            CacheQuery(
                filter=_Y_IS_A_OR_B & pl.col("p").is_in([False, True]), select="x2", fetches={"x2": 0}, cache_size=20, compare_row_order=False
            ),
        ],
        partition_cols=("y", "p"),
    ),
    # With column_granularity=False the first touch of a block fetches its whole schema.
    CacheStepsCase(
        "column_granularity_false_fills_all_columns",
        queries=[
            CacheQuery(filter=None, select="x2", fetches=_EVERY_COLUMN_ONCE, cache_size=6),
            CacheQuery(filter=None, select=["y2", "p2"], served_from_cache=True),
        ],
        column_granularity=False,
    ),
    CacheStepsCase(
        "column_granularity_false_preserves_partitioning",
        queries=[
            CacheQuery(filter=_Y_IS_A, select="x2", fetches=_EVERY_COLUMN_BUT_Y_ONCE),
            CacheQuery(filter=_Y_IS_A, select="p2", served_from_cache=True),
            CacheQuery(filter=_Y_IS_B, select="x2", fetches=_EVERY_COLUMN_BUT_Y_ONCE),
        ],
        partition_cols=("y",),
        column_granularity=False,
    ),
]


def _run(lf: pl.LazyFrame, step: CacheQuery) -> pl.DataFrame:
    lf = lf if step.filter is None else lf.filter(step.filter)
    return (lf if step.select is None else lf.select(step.select)).collect()


@pytest.mark.parametrize("case", as_params(CACHE_STEPS))
def test_cache_steps(source, tracked_frame, case: CacheStepsCase):
    cache = {}
    df_cache = tracked_frame.piot.cache(cache, order_by="x", partition_cols=case.partition_cols, column_granularity=case.column_granularity)
    for number, step in enumerate(case.queries, start=1):
        source.reset()
        out = _run(df_cache, step)
        counts = source.get_column_counts()
        if step.served_from_cache:
            assert source.calls == [], f"query {number} called the source"
        for col, want in step.fetches.items():
            assert counts.get(col, 0) in (want if isinstance(want, range) else (want,)), (number, col, counts)
        if step.cache_size is not None:
            assert len(cache) == step.cache_size, f"query {number}: {len(cache)} cache entries"
        with failure_note(f"query {number}"):
            assert_frame_equal(out, _run(tracked_frame, step), check_row_order=step.compare_row_order)


def test_cache_collects_with_the_streaming_engine(tracked_frame):
    df_cache = tracked_frame.piot.cache({}, order_by="x")
    assert_frame_equal(tracked_frame.collect(), df_cache.collect(engine="streaming"), check_row_order=False)
    with pl.Config() as cfg:
        cfg.set_streaming_chunk_size(2)
        assert_frame_equal(tracked_frame.collect(), df_cache.collect(engine="streaming"), check_row_order=False)


@pytest.mark.parametrize("cache_arg", [None, {}, "diskcache"], ids=["default_cache", "dict", "diskcache"])
def test_cache_types(cache_arg, source, tracked_frame, tmp_path):
    """The default cache, a dict and a diskcache each serve a repeated query without calling the source."""
    cache = cache_arg
    if cache_arg == "diskcache":
        diskcache = pytest.importorskip("diskcache")
        cache = diskcache.Cache(tmp_path / "cache")
    df_cache = tracked_frame.piot.cache(cache, order_by="x")
    for expected_calls in (1, 0):
        source.reset()
        df_cache.select("x2").collect()
        assert len(source.calls) == expected_calls
    source.reset()
    tracked_frame.piot.cache(cache, order_by="x").select("x2").collect()
    assert len(source.calls) == 0
    if cache is not None:
        assert len(cache)


def test_docstring():
    assert len(pl.LazyFrame.piot.cache.__doc__.split("\n")) > 10


def test_pickle(tracked_frame):
    """Test that the namespace sticks even after pickling/unpickling"""
    tracked_frame = pickle.loads(pickle.dumps(tracked_frame))
    assert hasattr(tracked_frame, "piot")
    assert hasattr(tracked_frame.piot, "cache")


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


def test_unfiltered_select_excludes_cached_partitions(source, tracked_frame):
    """An unfiltered read over a fully-cached partitioned frame must not re-scan everything.

    It should restrict the discovery query to exclude the already-cached partitions instead of
    issuing a full-source scan (predicate=None).
    """
    cache = {}
    df_cache = tracked_frame.piot.cache(cache, order_by="x", partition_cols=("y",))

    # Fully populate: every column of every partition.
    full = df_cache.collect()
    source.reset()

    # Unfiltered read again: the discovery query must carry a partition-exclusion predicate.
    out = df_cache.collect()
    preds = [call.predicate_str for call in source.calls]
    assert preds, "expected a partition-discovery query"
    assert all(p is not None for p in preds), f"expected exclusion predicate, got a full re-scan: {preds}"
    assert any("y" in (p or "") for p in preds), f"exclusion should reference partition col y: {preds}"

    assert_frame_equal(out.sort(["y", "x"]), full.sort(["y", "x"]))
    assert_frame_equal(out.sort(["y", "x"]), tracked_frame.collect().sort(["y", "x"]))


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


def test_column_granularity_false_and_true_share_cache(source, tracked_frame):
    """A column_granularity=False fill populates keys a granularity=True reader hits."""
    cache = {}
    # Eager fill of all columns via the False reader.
    tracked_frame.piot.cache(cache, order_by="x", column_granularity=False).select("x2").collect()

    source.reset()
    # Default (per-column) reader finds every column already cached -> no query.
    out = tracked_frame.piot.cache(cache, order_by="x").select(["x2", "y2"]).collect()
    assert source.calls == []
    assert_frame_equal(out, tracked_frame.select(["x2", "y2"]).collect())


@pytest.mark.parametrize("num_threads", [1, 2], ids=["one_thread", "two_threads"])
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


def test_explicit_df_key_replaces_plan_identity(source, tracked_frame):
    """With ``df_key``, frames with different plans share cache entries; without it they do not."""
    cache = {}
    same_data_other_plan = tracked_frame.select(pl.all())

    tracked_frame.piot.cache(cache, order_by="x", df_key="frame-v1").select("x2").collect()
    source.reset()
    out = same_data_other_plan.piot.cache(cache, order_by="x", df_key="frame-v1").select("x2").collect()
    assert source.calls == []
    assert_frame_equal(out, tracked_frame.select("x2").collect().sort("x2"), check_row_order=False)

    source.reset()
    same_data_other_plan.piot.cache(cache, order_by="x").select("x2").collect()
    assert source.calls

    source.reset()
    tracked_frame.piot.cache(cache, order_by="x", df_key="frame-v2").select("x2").collect()
    assert source.calls


def test_explicit_df_key_still_separates_layouts(tracked_frame):
    """``order_by`` and ``partition_cols`` stay part of the key when ``df_key`` is given."""
    cache = {}
    tracked_frame.piot.cache(cache, order_by="x", df_key="k").select("x2").collect()
    tracked_frame.piot.cache(cache, order_by=["x", "y"], df_key="k").select("x2").collect()
    tracked_frame.piot.cache(cache, order_by="x", partition_cols=("y",), df_key="k").select("x2").collect()
    assert len({k.df_key for k in cache}) == 3


@pytest.mark.skipif(
    getattr(pl.Categorical(), "ordering", "lexical") != "physical",
    reason="Physical categorical ordering is not supported by this Polars version",
)
@pytest.mark.parametrize("order_by", ["key", ("part", "key")], ids=["key", "part_then_key"])
@pytest.mark.parametrize("partition_cols", [(), ("part",)], ids=["unpartitioned", "by_part"])
@pytest.mark.parametrize("column_granularity", [False, True], ids=["whole_rows", "per_column"])
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
@pytest.mark.parametrize("column_granularity", [False, True], ids=["whole_rows", "per_column"])
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
@pytest.mark.parametrize("partition_cols", [(), ("part",)], ids=["unpartitioned", "by_part"])
@pytest.mark.parametrize("column_granularity", [False, True], ids=["whole_rows", "per_column"])
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


@dataclass(frozen=True)
class CacheLossyKeyCase:
    """A partition column dtype whose values can share a key, and three values ``step`` apart from ``start``."""

    id: str
    _: KW_ONLY
    dtype: pl.DataType | type[pl.DataType]
    # cast to ``dtype``
    start: int
    step: int


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            CacheLossyKeyCase("ns_datetime_keys_1ns_apart", dtype=pl.Datetime("ns"), start=1000, step=1),
            CacheLossyKeyCase("ns_duration_keys_1ns_apart", dtype=pl.Duration("ns"), start=1000, step=1),
            CacheLossyKeyCase("time_keys_1ns_apart", dtype=pl.Time, start=1000, step=1),
            CacheLossyKeyCase(
                "new_york_repeated_hour_keys_1h_apart", dtype=pl.Datetime("us", "America/New_York"), start=NEW_YORK_FIRST_0130_US, step=HOUR_US
            ),
        ]
    ),
)
def test_lossy_key_partitions_do_not_collide(case: CacheLossyKeyCase):
    df = pl.DataFrame({"k": pl.Series([case.start, case.start + case.step, case.start + 2 * case.step, None]).cast(case.dtype), "v": [1, 2, 3, 4]})
    lf = df.lazy().piot.cache({}, order_by="v", partition_cols=["k"])
    for pred in [pl.col("k") == pl.lit(case.start + case.step).cast(case.dtype), pl.col("k") == pl.lit(case.start).cast(case.dtype), pl.col("v") > 0]:
        assert_frame_equal(lf.filter(pred).collect().sort("v"), df.filter(pred))


_NAN_KEY = (pl.DataFrame({"x": [float("nan")], "i": [0]}), [pl.col("x").is_nan()])


@SETTINGS
@given(case=partitioned_frames_and_predicates(), partition_cols=st.sampled_from([("x",), ("date", "x"), ()]), column_granularity=st.booleans())
@example(case=_NAN_KEY, partition_cols=("x",), column_granularity=True)
@example(case=_NAN_KEY, partition_cols=("x",), column_granularity=False)
def test_cache_keeps_the_rows_a_filter_keeps(case, partition_cols, column_granularity):
    frame, predicates = case
    check_cache_keeps_filtered_rows(
        frame.lazy().piot.cache({}, order_by="i", partition_cols=partition_cols, column_granularity=column_granularity),
        frame,
        predicates,
    )


@pytest.mark.parametrize("column_granularity", [True, False], ids=["per_column", "whole_rows"])
def test_cache_on_a_nan_partition_key(column_granularity):
    frame = pl.DataFrame({"x": [float("nan")], "i": [0]})
    lf = frame.lazy().piot.cache({}, order_by="i", partition_cols=("x",), column_granularity=column_granularity)
    for pred in [pl.col("x").is_nan(), pl.lit(True)]:
        assert_frame_equal(lf.filter(pred).collect(), frame.filter(pred))


@dataclass(frozen=True)
class LegacyHelperCase:
    """A helper that old pickles reference by its old name, and the ids of the rows its filter keeps."""

    id: str
    _: KW_ONLY
    helper_name: str
    expected_ids: list[int]


@dataclass(frozen=True)
class RenamedHelperCase:
    """A helper that old pickles reference by its old name, and the name the helper has in ``util`` now."""

    id: str
    _: KW_ONLY
    helper_name: str
    new_util_name: str


@dataclass(frozen=True)
class CallbackDependencyCase:
    """A module and a symbol that pickled callbacks reference, which must still load and pickle to itself."""

    id: str
    _: KW_ONLY
    module_name: str
    symbol: str


class TestpiotCachePickle:
    """Tests for cachepickle support."""

    def test_piot_cache_pickle_basic(self, tmp_path):
        """cacheLazyFrames can be pickled and unpickled."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        source_lf = df.lazy()

        lf = source_lf.piot.cache(order_by="a")
        lf_unpickled = pickle_round_trip(lf)

        result = lf_unpickled.collect()
        assert_frame_equal(result, df)

    def test_piot_cache_pickle_after_caching_a_nan_partition(self):
        df = pl.DataFrame({"x": [float("nan")], "i": [0]})
        lf = df.lazy().piot.cache({}, order_by="i", partition_cols=("x",))
        lf.collect()
        assert_frame_equal(pickle_round_trip(lf).collect(), df)

    @pytest.mark.parametrize(
        "case",
        as_params(
            [
                LegacyHelperCase("_generate_expr", helper_name="_generate_expr", expected_ids=[2, 3, 4, 5]),
                LegacyHelperCase("_repeated_grouping", helper_name="_repeated_grouping", expected_ids=[3, 4, 5]),
                LegacyHelperCase("_extract_filter_from_df", helper_name="_extract_filter_from_df", expected_ids=[3, 4, 5]),
            ]
        ),
    )
    def test_legacy_cache_helper_pickle_references(self, case: LegacyHelperCase):
        helper_name = case.helper_name
        payload = f"cpolars_io_tools.io_sources.lazy_cache\n{helper_name}\n.".encode("ascii")
        helper = cloudpickle.loads(payload)
        frame = pl.DataFrame({"id": [1, 2, 3, 4, 5], "desk": ["a", "a", "b", "c", None], "day": [1, 2, 1, 1, 1]})
        known = pl.DataFrame({"desk": ["a", "a"], "day": [1, 2]})
        if helper_name == "_generate_expr":
            predicate = helper({"desk": "a", "day": 1}, frame.schema)
        elif helper_name == "_repeated_grouping":
            grouped = helper(known)
            assert grouped.height == 1
            predicate = _exclusion_from_frame(grouped)
        else:
            predicate = helper(known)
        assert frame.filter(predicate)["id"].to_list() == case.expected_ids

    @pytest.mark.parametrize(
        "case",
        as_params(
            [
                RenamedHelperCase("_generate_expr-_exclude_row_predicate", helper_name="_generate_expr", new_util_name="_exclude_row_predicate"),
                RenamedHelperCase("_repeated_grouping-_repeated_grouping", helper_name="_repeated_grouping", new_util_name="_repeated_grouping"),
                RenamedHelperCase(
                    "_extract_filter_from_df-_exclusion_from_frame", helper_name="_extract_filter_from_df", new_util_name="_exclusion_from_frame"
                ),
            ]
        ),
    )
    def test_legacy_helpers_repickle_without_new_util_names(self, case: RenamedHelperCase, monkeypatch):
        payload = f"cpolars_io_tools.io_sources.lazy_cache\n{case.helper_name}\n.".encode("ascii")
        helper = cloudpickle.loads(payload)
        repickled = cloudpickle.dumps(helper)
        with monkeypatch.context() as old_runtime:
            old_runtime.delattr(util, case.new_util_name)
            restored = cloudpickle.loads(repickled)
        assert restored is helper

    @pytest.mark.parametrize(
        "case",
        as_params(
            [
                CallbackDependencyCase("lazy_cache-_CacheKey", module_name="lazy_cache", symbol="_CacheKey"),
                CallbackDependencyCase("lazy_cache-_validate_order_by_unique", module_name="lazy_cache", symbol="_validate_order_by_unique"),
                CallbackDependencyCase("dnf_visitor-_is_contradiction", module_name="dnf_visitor", symbol="_is_contradiction"),
                CallbackDependencyCase(
                    "restrict_visitor-restrict_expr_to_columns", module_name="restrict_visitor", symbol="restrict_expr_to_columns"
                ),
                CallbackDependencyCase("profiling-profile_io_source_iterator", module_name="profiling", symbol="profile_io_source_iterator"),
                CallbackDependencyCase("profiling-_fallback_n_columns", module_name="profiling", symbol="_fallback_n_columns"),
                CallbackDependencyCase("profiling-_emit_span", module_name="profiling", symbol="_emit_span"),
                CallbackDependencyCase("profiling-_close_inner", module_name="profiling", symbol="_close_inner"),
                CallbackDependencyCase("util-partition_key", module_name="util", symbol="partition_key"),
                CallbackDependencyCase("util-_format_arg_for_error", module_name="util", symbol="_format_arg_for_error"),
                CallbackDependencyCase("util-_format_args_for_error", module_name="util", symbol="_format_args_for_error"),
                CallbackDependencyCase("util-_format_kwargs_for_error", module_name="util", symbol="_format_kwargs_for_error"),
            ]
        ),
    )
    def test_legacy_callback_dependency_pickle_references(self, case: CallbackDependencyCase):
        payload = f"cpolars_io_tools.io_sources.{case.module_name}\n{case.symbol}\n.".encode("ascii")
        dependency = cloudpickle.loads(payload)
        assert cloudpickle.loads(cloudpickle.dumps(dependency)) is dependency
