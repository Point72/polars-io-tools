"""cache_memory: one upstream scan per build, results equal to native Polars, and terminal build failures.

Tables give the plan, the expected number of builds or upstream scans, and the expected error; results are
compared with the same query on the uncached frame.
"""

import gc
import threading
import weakref

import polars as pl
import pytest
from polars.exceptions import ComputeError
from polars.io.plugins import register_io_source
from polars.testing import assert_frame_equal

from polars_io_tools.io_sources import cache_memory
from polars_io_tools.tests.helpers.cache_parquet_shared import UpstreamSpy
from polars_io_tools.tests.helpers.cases import Case, params

FRAME = pl.DataFrame({"a": pl.Series([1, 2], dtype=pl.Int64), "b": ["x", "y"]})
PFRAME = pl.DataFrame({"region": ["us", "us", "eu", "eu"], "id": [1, 2, 3, 4], "v": [10, 20, 30, 40]})


def _counted(frame: pl.DataFrame, calls: list):
    def build():
        calls.append(1)
        return frame.lazy()

    return build


def _stateful_source() -> pl.LazyFrame:
    """A plugin source whose generator closes over a lock, so its plan cannot be serialized."""
    data = pl.DataFrame({"id": [1, 2, 3, 4], "value": [10, 20, 30, 40]})
    lock = threading.Lock()

    def generator(with_columns, predicate, n_rows, batch_size):
        with lock:
            frame = data.lazy()
        if predicate is not None:
            frame = frame.filter(predicate)
        if with_columns is not None:
            frame = frame.select(with_columns)
        yield frame.collect()

    return register_io_source(io_source=generator, schema=data.schema)


# Builds and scans: N references to one frame cost one build per demanded partition set.


def _twice_and_two_more(lf):
    return [pl.concat([lf, lf], how="vertical"), lf, lf]


def ONCE(id, source, plan, partition_cols=None, frame=FRAME):
    return Case(id, plan, 1, {"source": source, "partition_cols": partition_cols, "frame": frame})


_us = pl.col("region") == "us"
COLLECTS_ONCE = [
    ONCE("collects_once_across_n_references", "lambda_scan", _twice_and_two_more),
    ONCE("builder_called_once", "builder", lambda lf: [pl.concat([lf, lf], how="vertical"), lf]),
    ONCE("accepts_plain_lazyframe", "lazy_scan", lambda lf: [pl.concat([lf, lf], how="vertical"), lf]),
    ONCE("no_partition_cols_unchanged", "lambda_scan", lambda lf: [pl.concat([lf, lf], how="vertical"), lf], ()),
    ONCE("partitioned_collects_builder_once_per_build", "lambda_scan", lambda lf: [pl.concat([lf.filter(_us), lf.filter(_us)])], "region", PFRAME),
]


@pytest.mark.parametrize(("source", "plan", "partition_cols", "frame"), params(COLLECTS_ONCE, "source", "pred", "partition_cols", "frame"))
def test_collects_once(source, plan, partition_cols, frame):
    """The builder or, for a scan source, the upstream scan runs once however often the frame is referenced."""
    calls: list = []
    if source == "builder":
        arg = _counted(frame, calls)
    else:
        spy = UpstreamSpy(frame, is_pure=True)
        calls = spy.predicates
        arg = (lambda: spy.lazy) if source == "lambda_scan" else spy.lazy
    kwargs = {} if partition_cols is None else {"partition_cols": partition_cols}
    lf = cache_memory(arg, schema=frame.schema, **kwargs)
    for query in plan(lf):
        query.collect()
    assert len(calls) == 1


def test_collects_once_across_collect_all_fanout():
    """Branches collected concurrently by ``pl.collect_all`` share one build, and a second pass reuses it."""
    calls: list = []
    base = pl.DataFrame({"id": [1, 2, 3, 4], "v": [10, 20, 30, 40]})
    lf = cache_memory(_counted(base, calls), schema=base.schema)
    branches = [lf.filter(pl.col("id") == i).select(pl.col("v").sum()) for i in range(1, 5)]
    results = pl.collect_all([*branches, lf.group_by("id").agg(pl.col("v").mean())])
    assert len(calls) == 1
    assert [r.item() for r in results[:4]] == [10, 20, 30, 40]
    pl.collect_all([lf.select(pl.len()), lf.filter(pl.col("id") > 2)])
    assert len(calls) == 1


def BUILDS(id, steps, partition_cols="region", frame=PFRAME):
    """Each step is (query, expected sorted ``v``, builds so far)."""
    return Case(id, steps, None, {"partition_cols": partition_cols, "frame": frame})


_G = pl.DataFrame({"g": ["a", None, "a", None], "v": [1, 2, 3, 4]})
_is_a, _is_null = pl.col("g") == "a", pl.col("g").is_null()
PARTITION_BUILDS = [
    BUILDS(
        "partition_predicate_restricts_build",
        [
            (lambda lf: lf.filter(_us), [10, 20], 1),
            (lambda lf: lf.filter(pl.col("region") == "eu"), [30, 40], 2),
            (lambda lf: lf.filter(_us), [10, 20], 2),
        ],
    ),
    BUILDS("full_build_when_no_partition_predicate", [(lambda lf: lf.select(pl.col("v").sum()), [100], 1), (lambda lf: lf.filter(_us), [10, 20], 1)]),
    BUILDS(
        "null_valued_partition_incremental_builds-value_first",
        [(lambda lf: lf.filter(_is_a), [1, 3], 1), (lambda lf: lf.filter(_is_null), [2, 4], 2)],
        "g",
        _G,
    ),
    BUILDS(
        "null_valued_partition_incremental_builds-null_first",
        [(lambda lf: lf.filter(_is_null), [2, 4], 1), (lambda lf: lf.filter(_is_a), [1, 3], 2)],
        "g",
        _G,
    ),
]


@pytest.mark.parametrize(("steps", "partition_cols", "frame"), params(PARTITION_BUILDS, "pred", "partition_cols", "frame"))
def test_partition_builds(steps, partition_cols, frame):
    """A partition predicate builds only the matching partitions, null partitions included; built ones are reused."""
    calls: list = []
    lf = cache_memory(_counted(frame, calls), schema=frame.schema, partition_cols=partition_cols)
    for query, values, builds in steps:
        assert sorted(query(lf).collect()["v"].to_list()) == values
        assert len(calls) == builds


@pytest.mark.parametrize("collect", [False, True], ids=["callable_schema_resolved_without_running_builder", "callable_schema_resolved_once"])
def test_callable_schema(collect):
    builds, schemas = [], []
    lf = cache_memory(_counted(FRAME, builds), schema=lambda: schemas.append(1) or FRAME.schema)
    assert lf.collect_schema().names() == ["a", "b"]
    if collect:
        lf.collect()
        lf.collect()
    assert (len(builds), len(schemas)) == (int(collect), 1)


# Results equal the same query on the uncached frame.


def NATIVE(id, query, partition_cols=None, frame=FRAME):
    return Case(id, query, None, {"partition_cols": partition_cols, "frame": frame})


NATIVE_QUERIES = [
    NATIVE("varying_predicate_across_collects_correct-lt", lambda lf: lf.filter(pl.col("a") < 2)),
    NATIVE("varying_predicate_across_collects_correct-ge", lambda lf: lf.filter(pl.col("a") >= 2)),
    NATIVE("varying_projection_across_collects-b", lambda lf: lf.select("b")),
    NATIVE("varying_projection_across_collects-a", lambda lf: lf.select("a")),
    # Polars pushes with_columns=[] for pl.len(); every row must still be counted.
    NATIVE("count_only_query_returns_true_row_count", lambda lf: lf.select(pl.len())),
    NATIVE("empty_projection_matches_plain_polars", lambda lf: lf.select([])),
    NATIVE("partitioned_result_correct-eu", lambda lf: lf.filter(pl.col("region") == "eu").sort("id"), "region", PFRAME),
    NATIVE("partitioned_result_correct-all", lambda lf: lf.sort("id"), "region", PFRAME),
    NATIVE("str_partition_cols_normalized-str", lambda lf: lf.filter(_us).sort("id"), "region", PFRAME),
    NATIVE("str_partition_cols_normalized-list", lambda lf: lf.filter(_us).sort("id"), ["region"], PFRAME),
    NATIVE("partition_cols_order_insensitive-rz", lambda lf: lf.filter(_us).sort("id"), ["region", "id"], PFRAME),
    NATIVE("partition_cols_order_insensitive-zr", lambda lf: lf.filter(_us).sort("id"), ["id", "region"], PFRAME),
    NATIVE("count_only_query_partitioned", lambda lf: lf.select(pl.len()), "region", PFRAME),
]


@pytest.mark.parametrize(("query", "partition_cols", "frame"), params(NATIVE_QUERIES, "pred", "partition_cols", "frame"))
def test_matches_native(query, partition_cols, frame):
    kwargs = {} if partition_cols is None else {"partition_cols": partition_cols}
    lf = cache_memory(lambda: frame.lazy(), schema=frame.schema, **kwargs)
    assert_frame_equal(query(lf).collect(), query(frame.lazy()).collect())


def test_partition_col_not_in_declared_schema():
    """A partition column the declaration omits drives the build but is dropped from what is served."""
    lf = cache_memory(lambda: PFRAME.lazy(), schema=PFRAME.drop("region").schema, partition_cols="region")
    assert_frame_equal(lf.sort("id").collect(), PFRAME.drop("region"))


def test_non_serializable_source_memoized():
    """A source whose plan cannot be serialized (so ``cache`` cannot key it) is memoized."""
    with pytest.raises(Exception):
        _stateful_source().serialize()
    lf = cache_memory(_stateful_source, schema=pl.Schema({"id": pl.Int64, "value": pl.Int64}))
    assert pl.concat([lf, lf], how="vertical").collect().height == 8
    assert lf.filter(pl.col("id") == 1).collect()["value"].to_list() == [10]


def test_empty_projection_preserves_row_count(monkeypatch):
    """An empty projection pushed straight to the generator keeps one row per source row.

    Called directly because Polars 1.42.1 pushes a named column for ``pl.len()`` and never reaches this branch.
    """
    import polars_io_tools.io_sources.lazy_cache_memory as mod

    captured = {}
    real = mod.register_io_source_with_is_pure
    monkeypatch.setattr(
        mod,
        "register_io_source_with_is_pure",
        lambda io_source, schema, **kw: captured.setdefault("gen", io_source) and real(io_source, schema, **kw),
    )
    cache_memory(lambda: FRAME.lazy(), schema=FRAME.schema)
    assert sum(df.height for df in captured["gen"](with_columns=[], predicate=None, n_rows=None, batch_size=None)) == FRAME.height


# The built frame is reconciled with the declared schema.


@pytest.mark.parametrize(
    ("transform", "error"),
    [
        (lambda lf: lf.drop("b"), "'b'"),
        (lambda lf: lf.with_columns(pl.col("a").cast(pl.Int32)), "'a'"),
        (lambda lf: lf.with_columns(pl.lit(1).alias("extra")), None),
    ],
    ids=["missing_declared_column_raises", "declared_dtype_mismatch_raises", "extra_column_dropped"],
)
def test_declared_schema_reconciliation(transform, error):
    lf = cache_memory(lambda: transform(FRAME.lazy()), schema=FRAME.schema)
    if error is None:
        assert_frame_equal(lf.collect(), FRAME)
    else:
        # The reconciliation error is raised inside the source generator, so Polars surfaces it as ComputeError.
        with pytest.raises(ComputeError, match=error):
            lf.collect()


def test_non_schema_non_callable_rejected():
    with pytest.raises(TypeError, match="pl.Schema or a callable"):
        cache_memory(FRAME.lazy(), schema={"a": pl.Int64})


# A failed build is terminal: it runs once per instance, its error is re-raised, and a fresh instance retries.


class BadStringError(Exception):
    def __str__(self):
        raise RuntimeError("cannot format")


def FAIL(id, exc, fanout, partition_cols=None, marks=()):
    return Case(id, exc, fanout, {"partition_cols": partition_cols}, marks)


BUILD_FAILURES = [
    FAIL("failed_build_is_terminal_and_cached", ValueError("transient"), False),
    FAIL("failed_build_not_amplified_across_collect_all", ValueError("boom"), True),
    FAIL("keyboard_interrupt_build_is_terminal", KeyboardInterrupt(), False),
    FAIL("failed_build_not_amplified_for_base_exception", KeyboardInterrupt(), True),
    # Deriving the cached message must not call a raising __str__ inside the lock, or later collects hang.
    FAIL("unprintable_exception_does_not_deadlock", BadStringError(), False, marks=(pytest.mark.timeout(15),)),
    FAIL("failed_build_terminal_partitioned", ValueError("boom"), False, "region"),
    FAIL("failed_build_not_amplified_across_collect_all_partitioned", ValueError("boom"), True, "region"),
]


@pytest.mark.parametrize(("exc", "fanout", "partition_cols"), params(BUILD_FAILURES, "pred", "expect", "partition_cols"))
def test_failed_build_is_terminal(exc, fanout, partition_cols):
    attempts: list = []

    def build():
        attempts.append(1)
        raise exc

    kwargs = {} if partition_cols is None else {"partition_cols": partition_cols}
    lf = cache_memory(build, schema=PFRAME.schema, **kwargs)
    first = (KeyboardInterrupt, ComputeError) if isinstance(exc, KeyboardInterrupt) else ComputeError
    match = str(exc) if isinstance(exc, ValueError) else None
    branches = [lf.filter(pl.col("region") == r) for r in ["us", "eu"] * 32]
    if fanout:
        with pytest.raises(first, match=match):
            pl.collect_all(branches)
    else:
        with pytest.raises(first, match=match):
            branches[0].collect()
        with pytest.raises(ComputeError, match=match):
            branches[1].collect()
    assert len(attempts) == 1
    assert_frame_equal(cache_memory(lambda: PFRAME.lazy(), schema=PFRAME.schema, **kwargs).sort("id").collect(), PFRAME)


# Buffers live only in the returned frame: dropping it frees them, and a cached failure holds no frame.


class Sentinel:
    pass


@pytest.mark.parametrize(
    ("partition_cols", "a_dtype"),
    [(None, pl.Int64), ("region", pl.Int64), (None, pl.Int32)],
    ids=["buffer_released_when_frame_dropped", "partition_buffers_released_when_frame_dropped", "failed_build_does_not_retain_frame"],
)
def test_buffers_released(partition_cols, a_dtype):
    holder: dict = {}
    schema = pl.Schema({"region": pl.String, "a": pl.Int64, "obj": pl.Object})

    def build():
        sentinel = Sentinel()
        holder["ref"] = weakref.ref(sentinel)
        # With a_dtype Int32 the build drifts from the declared Int64, so reconciliation fails while the frame is live.
        return pl.DataFrame({"region": ["us", "eu"], "a": [1, 2], "obj": [sentinel, sentinel]}, schema=schema | {"a": a_dtype}).lazy()

    kwargs = {} if partition_cols is None else {"partition_cols": partition_cols}
    lf = cache_memory(build, schema=schema, **kwargs)
    if a_dtype == pl.Int64:
        lf.collect()
        assert holder["ref"]() is not None, "sentinel should be alive while the frame is held"
        del lf
    else:
        for _ in range(2):
            with pytest.raises(ComputeError, match="'a'"):
                lf.collect()
    gc.collect()
    assert holder["ref"]() is None


# Partition keys: built keys are excluded from later builds, without collisions between lossy key spellings.


def test_exclude_built_infers_dtype_with_null_first_keys():
    """A valued key after more than 100 null-valued keys still gets the column's dtype (Polars infers from a prefix)."""
    from polars_io_tools.io_sources.lazy_cache_memory import _exclude_built
    from polars_io_tools.io_sources.util import partition_key

    keys = [partition_key({"a": None, "b": i}) for i in range(150)]
    keys.append(partition_key({"a": 7, "b": 999}))
    expr = _exclude_built(keys)
    assert expr is not None
    frame = pl.DataFrame({"a": [7, 7, None, 3], "b": [999, 1, 5, 5]}, schema={"a": pl.Int64, "b": pl.Int64})
    assert frame.filter(expr).to_dicts() == [{"a": 7, "b": 1}, {"a": 3, "b": 5}]


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
    lf = cache_memory(df.lazy(), schema=df.schema, partition_cols="k")
    for pred in [pl.col("k") == pl.lit(start + step).cast(dtype), pl.col("k") == pl.lit(start).cast(dtype), pl.col("v") > 0]:
        assert_frame_equal(lf.filter(pred).collect().sort("v"), df.filter(pred))


@pytest.mark.parametrize("dtype", [pl.Datetime("ms"), pl.Duration("ms")])
def test_millisecond_partitions_exclude_built_keys(dtype):
    df = pl.DataFrame({"k": pl.Series([1_500, 2_500, 1_000, 2_000]).cast(dtype), "v": [1, 2, 3, 4]})

    def gen(with_columns, predicate, n_rows, batch_size):
        yield df if predicate is None else df.filter(predicate)

    lf = cache_memory(register_io_source(io_source=gen, schema=df.schema), schema=df.schema, partition_cols="k")
    for pred in [pl.col("k") == df["k"][0], pl.col("k").is_in(df["k"][:2].implode()), pl.lit(True)]:
        assert_frame_equal(lf.filter(pred).collect().sort("v"), df.filter(pred))
