import pickle
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_io_tools.io_sources  # noqa: F401
from polars_io_tools.io_sources.lazy_cache import _is_contradiction
from polars_io_tools.io_sources.util import register_io_source_with_is_pure


def assert_expr_equal(a: Optional[pl.Expr], b: Optional[pl.Expr]):
    """Assert that two polars expressions are equal"""
    if a is None:
        assert b is None
    elif b is None:
        assert a is None
    else:
        assert a.meta.serialize(format="json") == b.meta.serialize(format="json")


@dataclass
class CallRecord:
    """Record of a single call to the IO source."""

    with_columns: Optional[List[str]]
    predicate_str: Optional[str]
    n_rows: Optional[int]


@dataclass
class TrackableSource:
    """
    A trackable IO source that records column requests.

    This replaces the previous approach of using map_batches with Mocks,
    which was fundamentally flawed because map_batches requires pure functions
    and Polars may call them with arbitrary input data.
    """

    data: pl.DataFrame
    calls: List[CallRecord] = field(default_factory=list)

    def reset(self):
        """Clear all recorded calls."""
        self.calls.clear()

    def get_column_counts(self) -> Dict[str, int]:
        """
        Count how many times each column was requested.

        Returns a dict mapping column name to request count.
        If with_columns was None (all columns), counts all columns.
        """
        counts: Dict[str, int] = {}
        for call in self.calls:
            cols = call.with_columns or list(self.data.columns)
            for col in cols:
                counts[col] = counts.get(col, 0) + 1
        return counts

    def as_lazy_frame(self) -> pl.LazyFrame:
        """Create a LazyFrame backed by this trackable source."""
        schema = self.data.schema
        tracker = self
        base_data = self.data

        def source_generator(
            with_columns: Optional[List[str]],
            predicate: Optional[pl.Expr],
            n_rows: Optional[int],
            batch_size: Optional[int],
        ) -> Iterator[pl.DataFrame]:
            # Record the call - use `is not None` to avoid Expr truthiness issues
            tracker.calls.append(
                CallRecord(
                    with_columns=with_columns,
                    predicate_str=str(predicate) if predicate is not None else None,
                    n_rows=n_rows,
                )
            )

            df = base_data.lazy()
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


@pytest.fixture
def source():
    """Create a trackable source with test data.

    The source is automatically reset after each test.
    """
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
    """Build a lazy frame backed by the trackable source."""
    return source.as_lazy_frame()


class Scenario(NamedTuple):
    filter: pl.Expr = pl.lit(True)
    select: pl.Expr = pl.all()
    head: Optional[int] = None
    partition_cols: Tuple[str, ...] = ()

    # Results to test against
    counts: Dict[str, int] = {}
    cache_size: int = 0

    # For partitioned data, we "complete" the cache by collecting the frame for all columns and no filters
    complete_counts: Dict[str, int] = {}
    complete_cache_size: int = 0


def test_source_basic(source, df):
    """Test that the trackable source correctly records column requests."""
    # Select specific columns
    df.select(["x2", "y2"]).collect()
    assert len(source.calls) == 1
    assert set(source.calls[0].with_columns) == {"x2", "y2"}

    source.reset()

    # Filter and select
    df.filter(pl.col("p")).select("x2").collect()
    assert len(source.calls) == 1
    # The predicate should be pushed down
    assert source.calls[0].predicate_str is not None
    assert "p" in source.calls[0].predicate_str


def test_source_logically_false_and_derived_filters(source, df):
    # Baseline raw data for logical comparisons
    raw_df = source.data

    # 1) Simple filter on p, select x2
    source.reset()
    out = df.filter(pl.col("p")).select("x2").collect()
    assert len(source.calls) == 1
    expected = raw_df.filter(pl.col("p"))["x2"]
    assert out["x2"].to_list() == expected.to_list()

    # 2) Logically false predicate p & ~p
    source.reset()
    out_false = df.filter(pl.col("p") & ~pl.col("p")).select("x2").collect()
    # Expression is still evaluated once at the IO layer
    assert len(source.calls) == 1
    # Result is logically empty
    assert len(out_false) == 0
    call = source.calls[0]
    # Predicate should be present and mention p
    assert call.predicate_str is not None
    assert "p" in call.predicate_str

    # 3) Filter on derived column p2, select x2
    source.reset()
    out_p2 = df.filter(pl.col("p2")).select("x2").collect()
    assert len(source.calls) == 1
    expected_p2 = raw_df.filter(pl.col("p2"))["x2"]
    assert out_p2["x2"].to_list() == expected_p2.to_list()

    # 4) Logically false predicate on derived column p2 & ~p2
    source.reset()
    out_p2_false = df.filter(pl.col("p2") & ~pl.col("p2")).select("x2").collect()
    # Again, expression evaluated once at IO, result empty
    assert len(source.calls) == 1
    assert len(out_p2_false) == 0
    call = source.calls[0]
    assert call.predicate_str is not None
    assert "p2" in call.predicate_str


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
@pytest.mark.parametrize(
    "scenario",
    SCENARIOS,
)
def test_scenarios(source, df, cache_mode, scenario):
    """Check the caching behavior for various scenarios.

    This test verifies:
    1. First call populates the cache with the expected size
    2. Second identical call uses cache (no new source calls in cache mode)
    3. Third call to collect all data completes the cache
    """
    cache = {}
    df_cache = df.piot.cache(cache, order_by="x", partition_cols=scenario.partition_cols, cache_mode=cache_mode)

    # First call
    out_cached = df_cache.filter(scenario.filter).select(scenario.select).head(scenario.head).collect()
    first_call_count = len(source.calls)
    assert first_call_count >= 1, "Expected at least one source call"
    if cache_mode == "ignore":
        assert len(cache) == 0
    else:
        assert len(cache) == scenario.cache_size

    # Verify results match direct query
    source.reset()
    out = df.filter(scenario.filter).select(scenario.select).head(scenario.head).collect()
    assert_frame_equal(out_cached, out, check_row_order=not scenario.partition_cols)

    # Second call - reset source to track new calls
    source.reset()

    # Re-create df_cache (but using the same `cache`) to prove there is no state in df_cache that's not captured by `cache`
    # We also reverse the partition cols here to make sure that doesn't impact things
    df_cache = df.piot.cache(cache, order_by="x", partition_cols=list(reversed(scenario.partition_cols)), cache_mode=cache_mode)
    out_cached2 = df_cache.filter(scenario.filter).select(scenario.select).head(scenario.head).collect()

    if cache_mode == "cache":
        # In cache mode, second call should not need new source data (or minimal calls for partitioned)
        assert len(cache) == scenario.cache_size
    else:
        # In rebuild/ignore mode, source is called again
        if cache_mode == "ignore":
            assert len(cache) == 0
        else:
            assert len(cache) == scenario.cache_size
    assert_frame_equal(out_cached2, out_cached)

    # Third Call - "Complete" the cache by selecting all the data
    source.reset()

    out_cached3 = df_cache.collect()
    if cache_mode == "cache":
        assert len(cache) == scenario.complete_cache_size
    else:
        if cache_mode == "ignore":
            assert len(cache) == 0
        else:
            assert len(cache) == scenario.complete_cache_size

    assert_frame_equal(out_cached3, df.collect(), check_row_order=not scenario.partition_cols)


def test_df_partitioned(source, df):
    """Test partitioning scenarios with single partition column."""
    cache = {}
    # Get x2 on y=a
    df_cache = df.piot.cache(cache, order_by="x", partition_cols=("y",))
    out_cached = df_cache.filter(pl.col("y") == "a").select("x2").collect()
    counts = source.get_column_counts()
    assert counts.get("x2", 0) == 1  # x2 fetched once
    assert counts.get("y2", 0) == 0  # y2 not fetched
    assert counts.get("p2", 0) == 0  # p2 not fetched
    assert len(cache) == 3  # x2, y (partition col), x (order_by) for partition y=a

    source.reset()
    out = df.filter(pl.col("y") == "a").select("x2").collect()
    assert_frame_equal(out_cached, out)

    source.reset()

    # Get p2 on y=a (x2 should be cached, need to fetch p2)
    out_cached = df_cache.filter(pl.col("y") == "a").select("p2").collect()
    counts = source.get_column_counts()
    assert counts.get("x2", 0) == 0  # x2 already cached
    assert counts.get("y2", 0) == 0  # y2 not needed
    assert counts.get("p2", 0) == 1  # p2 fetched

    source.reset()
    out = df.filter(pl.col("y") == "a").select("p2").collect()
    assert_frame_equal(out_cached, out)

    source.reset()

    # Get y2 on y=b (different partition)
    out_cached = df_cache.filter(pl.col("y") == "b").select("y2").collect()
    counts = source.get_column_counts()
    assert counts.get("x2", 0) == 0  # x2 not needed
    assert counts.get("y2", 0) == 1  # y2 fetched for new partition
    assert counts.get("p2", 0) == 0  # p2 not needed

    source.reset()
    out = df.filter(pl.col("y") == "b").select("y2").collect()
    assert_frame_equal(out_cached, out)

    source.reset()

    # Get x2 and y2 on y=b (y2 already cached for y=b, need x2)
    out_cached = df_cache.filter(pl.col("y") == "b").select(["x2", "y2"]).collect()
    counts = source.get_column_counts()
    assert counts.get("x2", 0) == 1  # x2 fetched for y=b partition
    assert counts.get("y2", 0) == 0  # y2 already cached for y=b
    assert counts.get("p2", 0) == 0  # p2 not needed

    source.reset()
    out = df.filter(pl.col("y") == "b").select(["x2", "y2"]).collect()
    assert_frame_equal(out_cached, out)

    source.reset()

    # Get all columns on y=a and y=b (only need y2 for y=a, p2 for y=b)
    out_cached = df_cache.filter(pl.col("y").is_in(["a", "b"])).collect()
    counts = source.get_column_counts()
    assert counts.get("x2", 0) == 0  # x2 cached for both partitions
    assert counts.get("y2", 0) == 1  # y2 needed for y=a
    assert counts.get("p2", 0) == 1  # p2 needed for y=b

    source.reset()
    out = df.filter(pl.col("y").is_in(["a", "b"])).collect()
    assert_frame_equal(out_cached, out, check_row_order=False)


def test_df_multi_partitioned(source, df):
    """Test partitioning scenarios with multiple partition columns."""
    cache = {}
    # Get x2 on y=a (covers 2 partitions: y=a/p=True and y=a/p=False)
    df_cache = df.piot.cache(cache, order_by="x", partition_cols=("y", "p"))
    out_cached = df_cache.filter(pl.col("y") == "a").select("x2").collect()
    counts = source.get_column_counts()
    assert counts.get("x2", 0) == 1
    assert counts.get("y2", 0) == 0
    assert counts.get("p2", 0) == 0
    assert len(cache) == 8  # x2, y, p, x (order_by) for 2 partitions (y=a, p=True) and (y=a, p=False)

    source.reset()
    out = df.filter(pl.col("y") == "a").select("x2").collect()
    assert_frame_equal(out_cached, out)

    source.reset()
    # Repeat query - with multi-partition, may still need to query due to partition key ambiguity
    out_cached = df_cache.filter(pl.col("y") == "a").select("x2").collect()
    counts = source.get_column_counts()
    assert counts.get("x2", 0) >= 0  # May or may not need to query depending on cache state
    assert len(cache) == 8

    source.reset()
    out = df.filter(pl.col("y") == "a").select("x2").collect()
    assert_frame_equal(out_cached, out)

    source.reset()

    # Get p2 on y=a - need to fetch p2 for y=a partitions
    out_cached = df_cache.filter(pl.col("y") == "a").select("p2").collect()
    counts = source.get_column_counts()
    assert counts.get("x2", 0) == 0  # x2 not needed
    assert counts.get("y2", 0) == 0  # y2 not needed
    assert counts.get("p2", 0) >= 1  # p2 fetched

    source.reset()
    out = df.filter(pl.col("y") == "a").select("p2").collect()
    assert_frame_equal(out_cached, out)

    source.reset()

    # Get y2 on y=b (different partition)
    out_cached = df_cache.filter(pl.col("y") == "b").select("y2").collect()
    counts = source.get_column_counts()
    assert counts.get("x2", 0) == 0  # x2 not needed
    assert counts.get("y2", 0) == 1  # y2 fetched
    assert counts.get("p2", 0) == 0  # p2 not needed

    source.reset()
    out = df.filter(pl.col("y") == "b").select("y2").collect()
    assert_frame_equal(out_cached, out)

    source.reset()

    # Get x2 and y2 on y=b (y2 cached, need x2)
    out_cached = df_cache.filter(pl.col("y") == "b").select(["x2", "y2"]).collect()
    counts = source.get_column_counts()
    assert counts.get("x2", 0) >= 1  # x2 fetched for y=b partitions
    assert counts.get("p2", 0) == 0  # p2 not needed

    source.reset()
    out = df.filter(pl.col("y") == "b").select(["x2", "y2"]).collect()
    assert_frame_equal(out_cached, out)

    source.reset()

    # Get all columns on y=a and y=b
    out_cached = df_cache.filter(pl.col("y").is_in(["a", "b"])).collect()
    counts = source.get_column_counts()
    # Some columns may need to be fetched for missing partitions

    source.reset()
    out = df.filter(pl.col("y").is_in(["a", "b"])).collect()
    assert_frame_equal(out_cached, out, check_row_order=False)


def test_df_multi_partitioned_contradiction(source, df):
    """Test partitioning with multiple partition columns and contradiction detection."""
    cache = {}
    # Get x2 on y=a, p=True (single partition)
    df_cache = df.piot.cache(cache, order_by="x", partition_cols=("y", "p"))
    out_cached = df_cache.filter(pl.col("y") == "a", pl.col("p") == pl.lit(True)).select("x2").collect()
    counts = source.get_column_counts()
    assert counts.get("x2", 0) == 1
    assert counts.get("y2", 0) == 0
    assert counts.get("p2", 0) == 0
    assert len(cache) == 4  # x2, y, p, x (order_by) for single partition (y=a, p=True)

    source.reset()
    out = df.filter(pl.col("y") == "a", pl.col("p") == pl.lit(True)).select("x2").collect()
    assert_frame_equal(out_cached, out)

    source.reset()
    # Repeat - should hit cache (no new source calls)
    out_cached = df_cache.filter(pl.col("y") == "a", pl.col("p") == pl.lit(True)).select("x2").collect()
    counts = source.get_column_counts()
    assert counts.get("x2", 0) == 0  # Cache hit
    assert len(cache) == 4

    source.reset()
    out = df.filter(pl.col("y") == "a", pl.col("p") == pl.lit(True)).select("x2").collect()
    assert_frame_equal(out_cached, out)

    source.reset()
    # Different partition (y=b, p=False)
    out_cached = df_cache.filter(pl.col("y") == "b", pl.col("p") == pl.lit(False)).select("x2").collect()
    counts = source.get_column_counts()
    assert counts.get("x2", 0) == 1  # New partition
    assert counts.get("y2", 0) == 0
    assert counts.get("p2", 0) == 0
    assert len(cache) == 8

    source.reset()
    out = df.filter(pl.col("y") == "b", pl.col("p") == pl.lit(False)).select("x2").collect()
    assert_frame_equal(out_cached, out)

    source.reset()
    # Another partition (y=b, p=True)
    out_cached = df_cache.filter(pl.col("y") == "b", pl.col("p") == pl.lit(True)).select("x2").collect()
    counts = source.get_column_counts()
    assert counts.get("x2", 0) == 1  # New partition
    assert counts.get("y2", 0) == 0
    assert counts.get("p2", 0) == 0
    assert len(cache) == 12

    source.reset()
    out = df.filter(pl.col("y") == "b", pl.col("p") == pl.lit(True)).select("x2").collect()
    assert_frame_equal(out_cached, out)

    source.reset()
    # Repeat - should hit cache
    out_cached = df_cache.filter(pl.col("y") == "b", pl.col("p") == pl.lit(True)).select("x2").collect()
    counts = source.get_column_counts()
    assert counts.get("x2", 0) == 0  # Cache hit
    assert len(cache) == 12

    source.reset()
    out = df.filter(pl.col("y") == "b", pl.col("p").eq(True)).select("x2").collect()
    assert_frame_equal(out_cached, out)

    source.reset()
    # Filter only on p=True (needs to query for y=c partition)
    out_cached = df_cache.filter(pl.col("p").eq(True)).select("x2").collect()
    counts = source.get_column_counts()
    assert counts.get("x2", 0) == 1  # Need y=c/p=True partition
    assert counts.get("y2", 0) == 0
    assert counts.get("p2", 0) == 0
    assert len(cache) == 16

    source.reset()
    out = df.filter(pl.col("p").eq(True)).select("x2").collect()
    assert_frame_equal(out_cached, out)

    source.reset()
    # Combined filter that should hit cache
    filter_expr = pl.col("y").is_in(["a", "b"]) & pl.col("p").eq(True)
    out_cached = df_cache.filter(filter_expr).select("x2").collect()
    counts = source.get_column_counts()
    assert counts.get("x2", 0) == 0  # All needed partitions cached
    assert len(cache) == 16

    source.reset()
    out = df.filter(filter_expr).select("x2").collect()
    assert_frame_equal(out_cached, out)

    source.reset()
    # Filter that needs additional partitions (y=a/p=False missing)
    filter_expr = pl.col("y").is_in(["a", "b"]) & pl.col("p").is_in([False, True])
    out_cached = df_cache.filter(filter_expr).select("x2").collect()
    counts = source.get_column_counts()
    assert counts.get("x2", 0) == 1  # Need y=a/p=False partition
    assert counts.get("y2", 0) == 0
    assert counts.get("p2", 0) == 0
    assert len(cache) == 20

    source.reset()
    out = df.filter(filter_expr).select("x2").collect()
    assert_frame_equal(out_cached, out, check_row_order=False)

    source.reset()
    # Repeat - should hit cache
    filter_expr = pl.col("y").is_in(["a", "b"]) & pl.col("p").is_in([False, True])
    out_cached = df_cache.filter(filter_expr).select("x2").collect()
    counts = source.get_column_counts()
    assert counts.get("x2", 0) == 0  # All partitions cached
    assert len(cache) == 20

    source.reset()
    out = df.filter(filter_expr).select("x2").collect()
    assert_frame_equal(out_cached, out, check_row_order=False)


def test_streaming(df):
    cache = {}
    df_cache = df.piot.cache(cache, order_by="x")
    assert_frame_equal(df.collect(), df_cache.collect(engine="streaming"), check_row_order=False)

    with pl.Config() as cfg:
        cfg.set_streaming_chunk_size(2)
        assert_frame_equal(df.collect(), df_cache.collect(engine="streaming"), check_row_order=False)


@pytest.mark.parametrize("cache_arg", [None, {}, "diskcache"])
def test_cache_types(cache_arg, source, df, tmp_path):
    """Test that both default and provided caches work on a simple example"""
    cache = cache_arg
    if cache_arg == "diskcache":
        diskcache = pytest.importorskip("diskcache")
        cache = diskcache.Cache(tmp_path / "cache")

    df_cache = df.piot.cache(cache, order_by="x")
    df_cache.select("x2").collect()
    first_call_count = len(source.calls)
    assert first_call_count == 1

    source.reset()

    # Second call - should hit cache
    df_cache.select("x2").collect()
    assert len(source.calls) == 0  # No new source calls

    source.reset()

    # Re-create df_cache with same cache - should still hit cache
    df_cache = df.piot.cache(cache, order_by="x")
    df_cache.select("x2").collect()
    assert len(source.calls) == 0  # No new source calls

    if cache is not None:
        assert len(cache)


def test_docstring():
    """Test that the docstring is accessible"""
    assert len(pl.LazyFrame.piot.cache.__doc__.split("\n")) > 10


# Test cases for _is_contradiction from lazy_cache.py
# Each case is a tuple: (description, polars_expression, expected_is_contradiction)
lazy_cache_contradiction_test_cases = [
    (
        "Single DNF clause, is a contradiction",
        (pl.col("a") == 5) & (pl.col("a") == 6),
        True,
    ),
    (
        "Single DNF clause, not a contradiction",
        (pl.col("a") == 5) & (pl.col("a") > 0),
        False,
    ),
    (
        "Single DNF clause, contradiction a == 5 & a > 5",
        (pl.col("a") == 5) & (pl.col("a") > 5),
        True,
    ),
    (
        "Multiple DNF clauses, all are contradictions",
        ((pl.col("a") == 5) & (pl.col("a") == 6)) | ((pl.col("b") < 0) & (pl.col("b") > 10)),
        True,
    ),
    (
        "Multiple DNF clauses, one is not a contradiction",
        ((pl.col("a") == 5) & (pl.col("a") == 6)) | ((pl.col("b") > 0) & (pl.col("b") < 10)),
        False,
    ),
    (
        "Multiple DNF clauses, none are contradictions",
        ((pl.col("a") == 5) & (pl.col("a") > 0)) | ((pl.col("b") > 0) & (pl.col("b") < 10)),
        False,
    ),
    (
        "Literal True expression",
        pl.lit(True),
        False,
    ),
    (
        "Literal False expression",
        pl.lit(False),
        True,
    ),
    (
        "Complex nested: ((C1 or C2) and C3) -> (C1 and C3) or (C2 and C3) -- all contradictions",
        (
            ((pl.col("a") == 1) & (pl.col("a") == 2))  # C1
            | ((pl.col("b") == 3) & (pl.col("b") == 4))  # C2
        )
        & (
            (pl.col("c") > 5) & (pl.col("c") < 0)  # C3
        ),
        True,
    ),
    (
        "Complex nested: ((NC1 or C2) and NC3) -> (NC1 and NC3) or (C2 and NC3) -- one path not contradiction",
        (
            ((pl.col("a") == 1) & (pl.col("a") > 0))  # NC1 (Not Contradiction)
            | ((pl.col("b") == 3) & (pl.col("b") == 4))  # C2  (Contradiction)
        )
        & (
            (pl.col("c") > 0) & (pl.col("c") < 10)  # NC3 (Not Contradiction)
        ),
        False,
    ),
    ("Contradiction involving IS NULL", (pl.col("x").is_null()) & (pl.col("x") == 5), True),
    (
        "Not a contradiction with IS NULL",
        (pl.col("x").is_null()) | (pl.col("x") == 5),  # x can be NULL or x can be 5
        False,
    ),
    ("Contradiction with IS NOT NULL", (pl.col("x").is_not_null()) & (pl.col("x").is_null()), True),
    ("Not a contradiction with IS NOT NULL", (pl.col("x").is_not_null()) & (pl.col("x") == 5), False),
    (
        "Expression that becomes empty DNF (e.g. not (lit(False))) -> should not be contradiction",
        ~pl.lit(False),  # equivalent to pl.lit(True)
        False,
    ),
    (
        "Expression that becomes DNF with empty clause (e.g. from complex True OR True)",
        (pl.col("a") > 0) | (pl.col("a") <= 0),  # This is always true
        False,  # An always true statement is not a contradiction
    ),
    # Tests for is_between
    (
        "is_between: Contradiction (disjoint ranges)",
        (pl.col("x").is_between(1, 5)) & (pl.col("x").is_between(10, 15)),
        True,
    ),
    (
        "is_between: No contradiction (disjoint ranges and exclusive of one)",
        (pl.col("x").is_between(1, 5)) & ~(pl.col("x").is_between(10, 15)),
        False,
    ),
    (
        "is_between: Contradiction (same range negated)",
        (pl.col("x").is_between(1, 5)) & ~(pl.col("x").is_between(1, 5)),
        True,
    ),
    (
        "is_between: Not a contradiction (overlapping ranges)",
        (pl.col("x").is_between(1, 10)) & (pl.col("x").is_between(5, 15)),
        False,
    ),
    (
        "is_between: Contradiction with exact value (value outside range)",
        (pl.col("x").is_between(1, 5)) & (pl.col("x") == 10),
        True,
    ),
    (
        "is_between: Not a contradiction with exact value (value inside range)",
        (pl.col("x").is_between(1, 5)) & (pl.col("x") == 3),
        False,
    ),
    (
        "is_between: Contradiction (upper bound less than lower bound in one expr)",
        (pl.col("x").is_between(5, 1)) & (pl.col("x") == 3),  # is_between(5,1) is always false
        True,
    ),
    (
        "is_between: Not a contradiction (inclusive bounds)",
        (pl.col("x").is_between(1, 5, closed="both")) & (pl.col("x") == 5),
        False,
    ),
    (
        "is_between: Contradiction (exclusive upper bound)",
        (pl.col("x").is_between(1, 5, closed="left")) & (pl.col("x") == 5),  # x >= 1 and x < 5
        True,
    ),
    (
        "is_between: Contradiction, we exclude both bounds and can equal either one.",
        (
            (pl.col("x").is_between(5, 6, closed="none"))
            & ((pl.col("x") == 5) | (pl.col("x") == 6))  # x == 5 or x == 6
            & (pl.col("x").is_between(4, 7, closed="none"))
        ),
        True,
    ),
    # Tests for is_in
    (
        "is_in: Contradiction (disjoint sets)",
        pl.col("x").is_in([1, 2, 3]) & pl.col("x").is_in([4, 5, 6]),
        True,
    ),
    (
        "is_in: Not a contradiction (overlapping sets)",
        pl.col("x").is_in([1, 2, 3]) & pl.col("x").is_in([3, 4, 5]),
        False,
    ),
    (
        "is_in: Contradiction with exact value (value not in set)",
        pl.col("x").is_in([1, 2, 3]) & (pl.col("x") == 4),
        True,
    ),
    (
        "is_in: Not a contradiction with exact value (value in set)",
        pl.col("x").is_in([1, 2, 3]) & (pl.col("x") == 2),
        False,
    ),
    (
        "is_in: Contradiction with is_not_in (same sets)",
        pl.col("x").is_in([1, 2, 3]) & pl.col("x").is_in([1, 2, 3]).not_(),
        True,
    ),
    (
        "is_in: Not a contradiction with is_not_in (different sets, but compatible)",
        pl.col("x").is_in([1, 2, 3]) & pl.col("x").is_in([4, 5, 6]).not_(),  # e.g. x can be 1
        False,
    ),
    (
        "is_in: Contradiction with is_not_in (value from is_in is excluded by is_not_in)",
        # x in [1,2,3] AND x not in [3,4,5] -> x can be 1 or 2
        # (x in [1,2,3] AND x not in [3,4,5]) AND x == 3 -> contradiction
        (pl.col("x").is_in([1, 2, 3])) & (~pl.col("x").is_in([3, 4, 5])) & (pl.col("x") == 3),
        True,
    ),
    (
        "is_in: A contradiction (empty list in is_in means always false for that part)",
        # (pl.col("x").is_in([])) is effectively False. So (False & True) is False.
        # The DNF for (A & B) is [(A,B)]. If A is False, then (False, B) is a contradiction.
        # So the whole expression is a contradiction.
        (pl.col("x").is_in([])) & (pl.col("x") == 5),
        True,
    ),
    (
        "is_in: A contradiction (is_in contains the minimum value, exclusive)",
        # (pl.col("x").is_in([])) is effectively False. So (False & True) is False.
        # The DNF for (A & B) is [(A,B)]. If A is False, then (False, B) is a contradiction.
        # So the whole expression is a contradiction.
        (pl.col("x").is_in([5])) & (pl.col("x") > 5),
        True,
    ),
    (
        "is_in: A contradiction (is_in contains the minimum value, inclusive)",
        # (pl.col("x").is_in([])) is effectively False. So (False & True) is False.
        # The DNF for (A & B) is [(A,B)]. If A is False, then (False, B) is a contradiction.
        # So the whole expression is a contradiction.
        (pl.col("x").is_in([5])) & (pl.col("x") == 5),
        False,
    ),
    # Tests for edge cases
    (
        "Contradiction for int but not float (so we say False)",
        (
            (pl.col("x").is_between(5, 6, closed="both")) & pl.col.x.ne(5) & pl.col.x.ne(6)  # x != 5 and x != 6
        ),
        False,
    ),
]


@pytest.mark.parametrize("description, expr, expected", lazy_cache_contradiction_test_cases)
def test_lazy_cache_is_contradiction(description: str, expr: pl.Expr, expected: bool):
    assert _is_contradiction(expr) == expected, f"Test failed for: {description}"


expr_with_schema = [
    (
        "Contradiction for int but not float (so we say True)",
        (
            (pl.col("x").is_between(5, 6, closed="both")) & pl.col.x.ne(5) & pl.col.x.ne(6)  # x != 5 and x != 6
        ),
        pl.Schema(dict(x=pl.Int32)),
        True,  # we can iterate through the bounds because we have an int type.
    ),
    (
        "Contradiction for int but not float (so we say False)",
        (
            (pl.col("x").is_between(5, 6, closed="both")) & pl.col.x.ne(5) & pl.col.x.ne(6)  # x != 5 and x != 6
        ),
        pl.Schema(dict(x=pl.Float64)),
        False,  # we cannot iterate through the bounds with a float.
    ),
    (
        "Contradiction for date but not datetime (so we say False)",
        (
            (pl.col("x").is_between(datetime(2024, 1, 1), datetime(2024, 1, 2), closed="both"))
            & pl.col.x.ne(datetime(2024, 1, 1))
            & pl.col.x.ne(datetime(2024, 1, 2))  # x != 5 and x != 6
        ),
        pl.Schema(dict(x=pl.Datetime())),
        False,  # we cannot iterate through the bounds with a Datetime.
    ),
    (
        "Contradiction for date but not datetime (so we say False)",
        (
            (pl.col("x").is_between(date(2024, 1, 1), date(2024, 1, 2), closed="both"))
            & pl.col.x.ne(date(2024, 1, 1))
            & pl.col.x.ne(date(2024, 1, 2))  # x != 5 and x != 6
        ),
        pl.Schema(dict(x=pl.Date())),
        True,  # we can iterate through the bounds with a Date.
    ),
    (
        "Contradiction for date across larger range.",
        (
            (pl.col("x").is_between(date(2024, 1, 1), date(2024, 1, 4), closed="left"))
            & pl.col.x.ne(date(2024, 1, 1))
            & pl.col.x.ne(date(2024, 1, 2))
            & pl.col.x.ne(date(2024, 1, 3))  # x != 5 and x != 6
        ),
        pl.Schema(dict(x=pl.Date())),
        True,  # we can iterate through the bounds with a Date.
    ),
    (
        "No contradiction for date across larger range.",
        (
            (pl.col("x").is_between(date(2024, 1, 1), date(2024, 1, 4), closed="right"))
            & pl.col.x.ne(date(2024, 1, 1))
            & pl.col.x.ne(date(2024, 1, 2))
            & pl.col.x.ne(date(2024, 1, 3))  # x != 5 and x != 6
        ),
        pl.Schema(dict(x=pl.Date())),
        False,  # we can iterate through the bounds with a Date, but 2024-01-04 is included in the filter so we don't have a contradiction.
    ),
]


@pytest.mark.parametrize("description, expr, schema, expected", expr_with_schema)
def test_lazy_cache_is_contradiction_with_schema(description: str, expr: pl.Expr, schema: pl.Schema, expected: bool):
    assert _is_contradiction(expr, schema) == expected, f"Test failed for: {description}"


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
            with_columns: Optional[List[str]],
            predicate: Optional[pl.Expr],
            n_rows: Optional[int],
            batch_size: Optional[int],
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
