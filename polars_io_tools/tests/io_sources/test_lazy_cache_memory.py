"""cache_memory: one upstream scan per build, results equal to plain Polars, and build failures that are not retried.

Tables give the plan, the expected number of builds or upstream scans, and the expected error; results are
compared with the same query on the uncached frame.

The property tests at the end (using the Hypothesis library) run random filters on a cache, cold then warm, and
compare each with a plain filter.
"""

import gc
import textwrap
import threading
import weakref
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field

import polars as pl
import pytest
from hypothesis import example, given, strategies as st
from polars.exceptions import ComputeError
from polars.io.plugins import register_io_source
from polars.testing import assert_frame_equal

from polars_io_tools.io_sources import cache_memory
from polars_io_tools.tests.helpers.cache_helpers import RecordingSource
from polars_io_tools.tests.helpers.cases import HOUR_US, NEW_YORK_FIRST_0130_US, as_params
from polars_io_tools.tests.helpers.property_checks import SETTINGS, check_cache_keeps_filtered_rows
from polars_io_tools.tests.helpers.strategies import (
    frames_and_predicates,
    partitioned_frames_and_predicates,
)
from polars_io_tools.tests.helpers.subprocesses import run_in_subprocess

TWO_ROWS = pl.DataFrame({"a": pl.Series([1, 2], dtype=pl.Int64), "b": ["x", "y"]})
REGIONS = pl.DataFrame({"region": ["us", "us", "eu", "eu"], "id": [1, 2, 3, 4], "v": [10, 20, 30, 40]})


def _counting_builder(frame: pl.DataFrame, calls: list):
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


@dataclass(frozen=True)
class CollectsOnceCase:
    """A frame referenced several times in the queries of a plan, which must be built or scanned once."""

    id: str
    _: KW_ONLY
    # "builder", "lambda_scan" or "lazy_scan": how the frame is given to ``cache_memory``
    source_kind: str
    # receives the cached frame and returns the queries to collect
    plan: Callable[[pl.LazyFrame], list[pl.LazyFrame]]
    partition_cols: str | tuple | None = None
    frame: pl.DataFrame = field(default_factory=lambda: TWO_ROWS)


_REGION_IS_US = pl.col("region") == "us"
COLLECTS_ONCE = [
    CollectsOnceCase("collects_once_across_n_references", source_kind="lambda_scan", plan=lambda lf: [pl.concat([lf, lf], how="vertical"), lf, lf]),
    CollectsOnceCase("builder_called_once", source_kind="builder", plan=lambda lf: [pl.concat([lf, lf], how="vertical"), lf]),
    CollectsOnceCase("accepts_plain_lazyframe", source_kind="lazy_scan", plan=lambda lf: [pl.concat([lf, lf], how="vertical"), lf]),
    CollectsOnceCase(
        "no_partition_cols_unchanged", source_kind="lambda_scan", plan=lambda lf: [pl.concat([lf, lf], how="vertical"), lf], partition_cols=()
    ),
    CollectsOnceCase(
        "partitioned_collects_builder_once_per_build",
        source_kind="lambda_scan",
        plan=lambda lf: [pl.concat([lf.filter(_REGION_IS_US), lf.filter(_REGION_IS_US)])],
        partition_cols="region",
        frame=REGIONS,
    ),
]


def _source(kind: str, frame: pl.DataFrame):
    """What cache_memory is given for the source kind ``kind``, and a list growing by one per build or upstream scan."""
    if kind == "builder":
        calls: list = []
        return _counting_builder(frame, calls), calls
    spy = RecordingSource(frame, is_pure=True)
    return ((lambda: spy.lazy) if kind == "lambda_scan" else spy.lazy), spy.predicates


@pytest.mark.parametrize("case", as_params(COLLECTS_ONCE))
def test_collects_once(case: CollectsOnceCase):
    """The builder or, for a scan source, the upstream scan runs once however often the frame is referenced."""
    arg, calls = _source(case.source_kind, case.frame)
    kwargs = {} if case.partition_cols is None else {"partition_cols": case.partition_cols}
    lf = cache_memory(arg, schema=case.frame.schema, **kwargs)
    for query in case.plan(lf):
        query.collect()
    assert len(calls) == 1


def test_collects_once_across_collect_all_fanout():
    """Branches collected concurrently by ``pl.collect_all`` share one build, and a second pass reuses it."""
    calls: list = []
    base = pl.DataFrame({"id": [1, 2, 3, 4], "v": [10, 20, 30, 40]})
    lf = cache_memory(_counting_builder(base, calls), schema=base.schema)
    branches = [lf.filter(pl.col("id") == i).select(pl.col("v").sum()) for i in range(1, 5)]
    results = pl.collect_all([*branches, lf.group_by("id").agg(pl.col("v").mean())])
    assert len(calls) == 1
    assert [r.item() for r in results[:4]] == [10, 20, 30, 40]
    pl.collect_all([lf.select(pl.len()), lf.filter(pl.col("id") > 2)])
    assert len(calls) == 1


@dataclass(frozen=True)
class PartitionBuildsCase:
    """Queries run in turn on one partitioned cache, and how many partition builds each needs."""

    id: str
    _: KW_ONLY
    # (query, the sorted ``v`` it returns, the number of builds so far)
    steps: list[tuple]
    partition_cols: str = "region"
    frame: pl.DataFrame = field(default_factory=lambda: REGIONS)


_GROUP_A_OR_NULL = pl.DataFrame({"g": ["a", None, "a", None], "v": [1, 2, 3, 4]})
_GROUP_IS_A, _GROUP_IS_NULL = pl.col("g") == "a", pl.col("g").is_null()
PARTITION_BUILDS = [
    PartitionBuildsCase(
        "partition_predicate_restricts_build",
        steps=[
            (lambda lf: lf.filter(_REGION_IS_US), [10, 20], 1),
            (lambda lf: lf.filter(pl.col("region") == "eu"), [30, 40], 2),
            (lambda lf: lf.filter(_REGION_IS_US), [10, 20], 2),
        ],
    ),
    PartitionBuildsCase(
        "full_build_when_no_partition_predicate",
        steps=[(lambda lf: lf.select(pl.col("v").sum()), [100], 1), (lambda lf: lf.filter(_REGION_IS_US), [10, 20], 1)],
    ),
    PartitionBuildsCase(
        "null_valued_partition_incremental_builds-value_first",
        steps=[(lambda lf: lf.filter(_GROUP_IS_A), [1, 3], 1), (lambda lf: lf.filter(_GROUP_IS_NULL), [2, 4], 2)],
        partition_cols="g",
        frame=_GROUP_A_OR_NULL,
    ),
    PartitionBuildsCase(
        "null_valued_partition_incremental_builds-null_first",
        steps=[(lambda lf: lf.filter(_GROUP_IS_NULL), [2, 4], 1), (lambda lf: lf.filter(_GROUP_IS_A), [1, 3], 2)],
        partition_cols="g",
        frame=_GROUP_A_OR_NULL,
    ),
]


@pytest.mark.parametrize("case", as_params(PARTITION_BUILDS))
def test_partition_builds(case: PartitionBuildsCase):
    """A partition predicate builds only the matching partitions, null partitions included; built ones are reused."""
    calls: list = []
    lf = cache_memory(_counting_builder(case.frame, calls), schema=case.frame.schema, partition_cols=case.partition_cols)
    for query, values, builds in case.steps:
        assert sorted(query(lf).collect()["v"].to_list()) == values
        assert len(calls) == builds


@pytest.mark.parametrize("collect", [False, True], ids=["callable_schema_resolved_without_running_builder", "callable_schema_resolved_once"])
def test_callable_schema(collect):
    builds, schemas = [], []
    lf = cache_memory(_counting_builder(TWO_ROWS, builds), schema=lambda: schemas.append(1) or TWO_ROWS.schema)
    assert lf.collect_schema().names() == ["a", "b"]
    if collect:
        lf.collect()
        lf.collect()
    assert (len(builds), len(schemas)) == (int(collect), 1)


# Results equal the same query on the uncached frame.


@dataclass(frozen=True)
class MatchesPlainPolarsCase:
    """A query on a cached frame, which must return what it returns on the frame itself."""

    id: str
    _: KW_ONLY
    query: Callable[[pl.LazyFrame], pl.LazyFrame]
    partition_cols: str | None = None
    frame: pl.DataFrame = field(default_factory=lambda: TWO_ROWS)


NATIVE_QUERIES = [
    MatchesPlainPolarsCase("varying_predicate_across_collects_correct-lt", query=lambda lf: lf.filter(pl.col("a") < 2)),
    MatchesPlainPolarsCase("varying_predicate_across_collects_correct-ge", query=lambda lf: lf.filter(pl.col("a") >= 2)),
    MatchesPlainPolarsCase("varying_projection_across_collects-b", query=lambda lf: lf.select("b")),
    MatchesPlainPolarsCase("varying_projection_across_collects-a", query=lambda lf: lf.select("a")),
    # Polars pushes with_columns=[] for pl.len(); every row must still be counted.
    MatchesPlainPolarsCase("count_only_query_returns_true_row_count", query=lambda lf: lf.select(pl.len())),
    MatchesPlainPolarsCase("empty_projection_matches_plain_polars", query=lambda lf: lf.select([])),
    MatchesPlainPolarsCase(
        "partitioned_result_correct-eu", query=lambda lf: lf.filter(pl.col("region") == "eu").sort("id"), partition_cols="region", frame=REGIONS
    ),
    MatchesPlainPolarsCase("partitioned_result_correct-all", query=lambda lf: lf.sort("id"), partition_cols="region", frame=REGIONS),
    MatchesPlainPolarsCase(
        "str_partition_cols_normalized-str", query=lambda lf: lf.filter(_REGION_IS_US).sort("id"), partition_cols="region", frame=REGIONS
    ),
    MatchesPlainPolarsCase(
        "str_partition_cols_normalized-list", query=lambda lf: lf.filter(_REGION_IS_US).sort("id"), partition_cols=["region"], frame=REGIONS
    ),
    MatchesPlainPolarsCase(
        "partition_cols_order_insensitive-rz", query=lambda lf: lf.filter(_REGION_IS_US).sort("id"), partition_cols=["region", "id"], frame=REGIONS
    ),
    MatchesPlainPolarsCase(
        "partition_cols_order_insensitive-zr", query=lambda lf: lf.filter(_REGION_IS_US).sort("id"), partition_cols=["id", "region"], frame=REGIONS
    ),
    MatchesPlainPolarsCase("count_only_query_partitioned", query=lambda lf: lf.select(pl.len()), partition_cols="region", frame=REGIONS),
]


@pytest.mark.parametrize("case", as_params(NATIVE_QUERIES))
def test_matches_native(case: MatchesPlainPolarsCase):
    kwargs = {} if case.partition_cols is None else {"partition_cols": case.partition_cols}
    lf = cache_memory(lambda: case.frame.lazy(), schema=case.frame.schema, **kwargs)
    assert_frame_equal(case.query(lf).collect(), case.query(case.frame.lazy()).collect())


def test_partition_col_not_in_declared_schema():
    """A partition column the declaration omits drives the build but is dropped from what is served."""
    lf = cache_memory(lambda: REGIONS.lazy(), schema=REGIONS.drop("region").schema, partition_cols="region")
    assert_frame_equal(lf.sort("id").collect(), REGIONS.drop("region"))


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
    cache_memory(lambda: TWO_ROWS.lazy(), schema=TWO_ROWS.schema)
    assert sum(df.height for df in captured["gen"](with_columns=[], predicate=None, n_rows=None, batch_size=None)) == TWO_ROWS.height


# The built frame is reconciled with the declared schema.


@dataclass(frozen=True)
class DeclaredSchemaCase:
    """A change to the built frame, and the error reconciling it with the declared schema raises."""

    id: str
    _: KW_ONLY
    transform: Callable[[pl.LazyFrame], pl.LazyFrame]
    # a pattern of the error message; None: the build matches the schema
    error: str | None


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            DeclaredSchemaCase("missing_declared_column_raises", transform=lambda lf: lf.drop("b"), error="'b'"),
            DeclaredSchemaCase("declared_dtype_mismatch_raises", transform=lambda lf: lf.with_columns(pl.col("a").cast(pl.Int32)), error="'a'"),
            DeclaredSchemaCase("extra_column_dropped", transform=lambda lf: lf.with_columns(pl.lit(1).alias("extra")), error=None),
        ]
    ),
)
def test_declared_schema_reconciliation(case: DeclaredSchemaCase):
    lf = cache_memory(lambda: case.transform(TWO_ROWS.lazy()), schema=TWO_ROWS.schema)
    if case.error is None:
        assert_frame_equal(lf.collect(), TWO_ROWS)
    else:
        # The reconciliation error is raised inside the source generator, so Polars surfaces it as ComputeError.
        with pytest.raises(ComputeError, match=case.error):
            lf.collect()


def test_non_schema_non_callable_rejected():
    with pytest.raises(TypeError, match="pl.Schema or a callable"):
        cache_memory(TWO_ROWS.lazy(), schema={"a": pl.Int64})


# A failed build is terminal: it runs once per instance, its error is re-raised, and a fresh instance retries.


class BadStringError(Exception):
    def __str__(self):
        raise RuntimeError("cannot format")


@dataclass(frozen=True)
class FailedBuildCase:
    """A build that raises, and how 64 filtered branches are collected; the build must run once and every branch must
    fail."""

    id: str
    _: KW_ONLY
    error: BaseException
    # collect the branches with ``pl.collect_all`` instead of one by one
    collect_branches_together: bool
    partition_cols: str | None = None
    marks: tuple = ()


# Build errors, which must not be retried by later collects on the same frame.
BUILD_FAILURES = [
    FailedBuildCase("failed_build_is_terminal_and_cached", error=ValueError("transient"), collect_branches_together=False),
    FailedBuildCase("failed_build_not_amplified_across_collect_all", error=ValueError("boom"), collect_branches_together=True),
    FailedBuildCase("keyboard_interrupt_build_is_terminal", error=KeyboardInterrupt(), collect_branches_together=False),
    FailedBuildCase("failed_build_not_amplified_for_base_exception", error=KeyboardInterrupt(), collect_branches_together=True),
    # Deriving the cached message must not call a raising __str__ inside the lock, or later collects hang.
    FailedBuildCase(
        "unprintable_exception_does_not_deadlock", error=BadStringError(), collect_branches_together=False, marks=(pytest.mark.timeout(15),)
    ),
    FailedBuildCase("failed_build_terminal_partitioned", error=ValueError("boom"), collect_branches_together=False, partition_cols="region"),
    FailedBuildCase(
        "failed_build_not_amplified_across_collect_all_partitioned", error=ValueError("boom"), collect_branches_together=True, partition_cols="region"
    ),
]


@pytest.mark.parametrize("case", as_params(BUILD_FAILURES))
def test_failed_build_is_terminal(case: FailedBuildCase):
    attempts: list = []

    def build():
        attempts.append(1)
        raise case.error

    kwargs = {} if case.partition_cols is None else {"partition_cols": case.partition_cols}
    lf = cache_memory(build, schema=REGIONS.schema, **kwargs)
    first = (KeyboardInterrupt, ComputeError) if isinstance(case.error, KeyboardInterrupt) else ComputeError
    match = str(case.error) if isinstance(case.error, ValueError) else None
    branches = [lf.filter(pl.col("region") == r) for r in ["us", "eu"] * 32]
    if case.collect_branches_together:
        with pytest.raises(first, match=match):
            pl.collect_all(branches)
    else:
        with pytest.raises(first, match=match):
            branches[0].collect()
        with pytest.raises(ComputeError, match=match):
            branches[1].collect()
    assert len(attempts) == 1
    assert_frame_equal(cache_memory(lambda: REGIONS.lazy(), schema=REGIONS.schema, **kwargs).sort("id").collect(), REGIONS)


# Buffers live only in the returned frame: dropping it frees them, and a cached failure holds no frame.


class Sentinel:
    pass


@dataclass(frozen=True)
class BufferReleaseCase:
    """How the cache is partitioned and the dtype the build gives ``a``; the buffers must be released either way."""

    id: str
    _: KW_ONLY
    partition_cols: str | None
    # Int32 makes the build differ from the declared Int64
    a_dtype: pl.DataType | type[pl.DataType]


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            BufferReleaseCase("buffer_released_when_frame_dropped", partition_cols=None, a_dtype=pl.Int64),
            BufferReleaseCase("partition_buffers_released_when_frame_dropped", partition_cols="region", a_dtype=pl.Int64),
            BufferReleaseCase("failed_build_does_not_retain_frame", partition_cols=None, a_dtype=pl.Int32),
        ]
    ),
)
def test_buffers_released(case: BufferReleaseCase):
    holder: dict = {}
    schema = pl.Schema({"region": pl.String, "a": pl.Int64, "obj": pl.Object})

    def build():
        sentinel = Sentinel()
        holder["ref"] = weakref.ref(sentinel)
        # With a_dtype Int32 the build drifts from the declared Int64, so reconciliation fails while the frame is live.
        return pl.DataFrame({"region": ["us", "eu"], "a": [1, 2], "obj": [sentinel, sentinel]}, schema=schema | {"a": case.a_dtype}).lazy()

    kwargs = {} if case.partition_cols is None else {"partition_cols": case.partition_cols}
    lf = cache_memory(build, schema=schema, **kwargs)
    if case.a_dtype == pl.Int64:
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


@dataclass(frozen=True)
class MemoryLossyKeyCase:
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
            MemoryLossyKeyCase("ns_datetime_keys_1ns_apart", dtype=pl.Datetime("ns"), start=1000, step=1),
            MemoryLossyKeyCase("ns_duration_keys_1ns_apart", dtype=pl.Duration("ns"), start=1000, step=1),
            MemoryLossyKeyCase("time_keys_1ns_apart", dtype=pl.Time, start=1000, step=1),
            MemoryLossyKeyCase(
                "new_york_repeated_hour_keys_1h_apart", dtype=pl.Datetime("us", "America/New_York"), start=NEW_YORK_FIRST_0130_US, step=HOUR_US
            ),
        ]
    ),
)
def test_lossy_key_partitions_do_not_collide(case: MemoryLossyKeyCase):
    df = pl.DataFrame({"k": pl.Series([case.start, case.start + case.step, case.start + 2 * case.step, None]).cast(case.dtype), "v": [1, 2, 3, 4]})
    lf = cache_memory(df.lazy(), schema=df.schema, partition_cols="k")
    for pred in [pl.col("k") == pl.lit(case.start + case.step).cast(case.dtype), pl.col("k") == pl.lit(case.start).cast(case.dtype), pl.col("v") > 0]:
        assert_frame_equal(lf.filter(pred).collect().sort("v"), df.filter(pred))


@pytest.mark.parametrize("dtype", [pl.Datetime("ms"), pl.Duration("ms")], ids=["datetime_ms", "duration_ms"])
def test_millisecond_partitions_exclude_built_keys(dtype):
    df = pl.DataFrame({"k": pl.Series([1_500, 2_500, 1_000, 2_000]).cast(dtype), "v": [1, 2, 3, 4]})

    def gen(with_columns, predicate, n_rows, batch_size):
        yield df if predicate is None else df.filter(predicate)

    lf = cache_memory(register_io_source(io_source=gen, schema=df.schema), schema=df.schema, partition_cols="k")
    for pred in [pl.col("k") == df["k"][0], pl.col("k").is_in(df["k"][:2].implode()), pl.lit(True)]:
        assert_frame_equal(lf.filter(pred).collect().sort("v"), df.filter(pred))


@SETTINGS
@given(case=partitioned_frames_and_predicates(), partition_cols=st.sampled_from([("x",), ("date", "x"), ()]))
# A literal the predicate analysis cannot read, so the query names no partition.
@example(
    case=(pl.DataFrame({"x": [0.0, float("nan")], "i": [0, 1]}), [pl.lit(True), pl.col("x") == pl.lit(pl.Series([0.0])).first(), pl.lit(True)]),
    partition_cols=("x",),
)
def test_cache_memory_keeps_the_rows_a_filter_keeps(case, partition_cols):
    frame, predicates = case
    check_cache_keeps_filtered_rows(cache_memory(frame.lazy(), schema=frame.schema, partition_cols=partition_cols), frame, predicates)


@SETTINGS
@given(case=frames_and_predicates(max_columns=2, depth=2))
def test_cache_memory_keeps_the_rows_of_random_predicates(case):
    """Partitioned on every column of a wide dtype mix."""
    frame, pred = case
    frame = frame.with_row_index("i")
    check_cache_keeps_filtered_rows(cache_memory(frame.lazy(), schema=frame.schema, partition_cols=frame.columns[1:]), frame, [pred])


def test_cache_memory_on_a_nan_partition_key():
    frame = pl.DataFrame({"x": [0.0, float("nan")], "i": [0, 1]})
    lf = cache_memory(frame.lazy(), schema=frame.schema, partition_cols="x")
    # A literal the predicate analysis cannot read, so the query names no partition.
    for pred in [pl.lit(True), pl.col("x") == pl.lit(pl.Series([0.0])).first(), pl.lit(True)]:
        assert_frame_equal(lf.filter(pred).collect(), frame.filter(pred), check_row_order=False)


@pytest.mark.xfail(strict=True, raises=(pl.exceptions.PanicException, pl.exceptions.ComputeError), reason="keys Python cannot hold")
@pytest.mark.parametrize("wrap", ["cache_memory", "cache"])
def test_cache_on_a_partition_key_beyond_python_datetime(wrap):
    """Runs both cache_memory and piot.cache, whose partition keys are built the same way."""
    frame = pl.DataFrame({"t": pl.Series([300_000_000_000_000, 0]).cast(pl.Datetime("ms")), "i": [0, 1]})  # the year 11476
    if wrap == "cache":
        lf = frame.lazy().piot.cache({}, order_by="i", partition_cols=("t",))
    else:
        lf = cache_memory(frame.lazy(), schema=frame.schema, partition_cols="t")
    assert_frame_equal(lf.filter(pl.col("i") >= 0).collect(), frame, check_row_order=False)


@pytest.mark.parametrize("threads", [1, 2], ids=["one_thread", "two_threads"])
@pytest.mark.parametrize("scenario", ["join", "fanout"])
def test_nested_memory_cache_build_does_not_deadlock(threads, scenario):
    script = textwrap.dedent(
        f"""
        import os
        os.environ["POLARS_ENGINE_AFFINITY"] = "in-memory"

        import faulthandler
        import polars as pl
        from polars_io_tools.io_sources import cache_memory

        faulthandler.dump_traceback_later(30, exit=True)
        frame = pl.LazyFrame({{"id": [1, 2, 3], "value": [10, 20, 30]}})
        schema = frame.collect_schema()
        builds = [0, 0]

        def build_inner():
            builds[0] += 1
            return frame

        inner = cache_memory(build_inner, schema=schema)

        def build_outer():
            builds[1] += 1
            return inner

        outer = cache_memory(build_outer, schema=schema)
        if {scenario!r} == "join":
            result = outer.join(inner, on="id").filter(pl.col("id") > 1).collect()
            assert result["id"].sort().to_list() == [2, 3], result
        else:
            results = pl.collect_all([outer.filter(pl.col("id") == 1), outer.filter(pl.col("id") == 2), inner.filter(pl.col("id") == 3)])
            assert [result["id"].to_list() for result in results] == [[1], [2], [3]], results
        assert outer.collect()["value"].to_list() == [10, 20, 30]
        assert builds == [1, 1], builds
        faulthandler.cancel_dump_traceback_later()
        print("OK")
        """
    )
    result = run_in_subprocess(script, threads=threads)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "OK" in result.stdout
