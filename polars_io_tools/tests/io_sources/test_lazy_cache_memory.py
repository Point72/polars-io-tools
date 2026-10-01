"""Unit tests for :func:`polars_io_tools.io_sources.lazy_cache_memory.cache_memory`."""

import gc
import threading
import unittest
import weakref

import polars as pl
import pytest
from polars.exceptions import ComputeError
from polars.io.plugins import register_io_source
from polars.testing import assert_frame_equal

from polars_io_tools.io_sources import cache_memory
from polars_io_tools.io_sources.util import collect_lf_in_io_source, register_io_source_with_is_pure


def _counting_source(frame: pl.LazyFrame, counter: list) -> pl.LazyFrame:
    """A plain (non-memoized) register_io_source that appends to ``counter`` on every scan.

    Used to measure how many times the *upstream* is actually executed — the quantity
    ``cache_memory`` is meant to collapse to one, independent of builder-call counting.
    """
    schema = frame.collect_schema()

    def gen(with_columns, predicate, n_rows, batch_size):
        counter.append(1)
        lf = frame
        if predicate is not None:
            lf = lf.filter(predicate)
        if with_columns is not None:
            requested = set(with_columns)
            lf = lf.select([c for c in schema.names() if c in requested])
        if n_rows is not None:
            lf = lf.head(n_rows)
        yield from collect_lf_in_io_source(lf, batch_size)

    return register_io_source_with_is_pure(io_source=gen, schema=schema)


def _stateful_source() -> pl.LazyFrame:
    """A register_io_source-backed frame whose generator closes over a non-picklable lock.

    Reproduces issue #26: ``cache`` (which keys by serializing the plan) cannot handle
    this source, but ``cache_memory`` can.
    """
    data = pl.DataFrame({"id": [1, 2, 3, 4], "value": [10, 20, 30, 40]})
    schema = data.collect_schema()
    lock = threading.Lock()  # non-picklable state captured in the generator's closure

    def generator(with_columns, predicate, n_rows, batch_size):
        with lock:
            frame = data.lazy()
        if predicate is not None:
            frame = frame.filter(predicate)
        if with_columns is not None:
            frame = frame.select(with_columns)
        yield frame.collect()

    return register_io_source(io_source=generator, schema=schema)


class TestCacheMemory(unittest.TestCase):
    """Tests for cache_memory, which collapses N references to one upstream scan.

    ``register_io_source_with_is_pure`` does no declared-vs-yielded validation, so
    cache_memory's own reconciliation is the entire safety net against schema drift.
    """

    @staticmethod
    def _frame() -> pl.LazyFrame:
        return pl.LazyFrame({"a": pl.Series([1, 2], dtype=pl.Int64), "b": ["x", "y"]})

    @staticmethod
    def _pframe() -> pl.LazyFrame:
        """A frame with a partition column ``region`` (two partitions) plus id/value columns."""
        return pl.LazyFrame(
            {
                "region": ["us", "us", "eu", "eu"],
                "id": [1, 2, 3, 4],
                "v": [10, 20, 30, 40],
            }
        )

    @staticmethod
    def _pschema() -> pl.Schema:
        return pl.Schema({"region": pl.String, "id": pl.Int64, "v": pl.Int64})

    def test_collects_once_across_n_references(self):
        """The upstream is scanned exactly once even when the frame is referenced N times.

        Not merely that the builder is called once, but that the *collected buffer* is
        reused, so a consumer that references the frame many times pays one upstream
        execution total.
        """
        scans: list = []
        frame = self._frame()
        lf = cache_memory(lambda: _counting_source(frame, scans), schema=frame.collect_schema())

        # Referenced twice in one plan, plus two independent collects.
        pl.concat([lf, lf], how="vertical").collect()
        lf.collect()
        lf.collect()

        self.assertEqual(len(scans), 1)

    def test_builder_called_once(self):
        """The builder itself is invoked at most once."""
        calls = []

        def build():
            calls.append(1)
            return frame

        frame = self._frame()
        lf = cache_memory(build, schema=frame.collect_schema())

        pl.concat([lf, lf], how="vertical").collect()
        lf.collect()

        self.assertEqual(len(calls), 1)

    def test_accepts_plain_lazyframe(self):
        """A LazyFrame passed directly is wrapped as a builder and still collapses references."""
        scans: list = []
        frame = self._frame()
        source = _counting_source(frame, scans)
        lf = cache_memory(source, schema=frame.collect_schema())

        pl.concat([lf, lf], how="vertical").collect()
        lf.collect()

        self.assertEqual(len(scans), 1)

    def test_varying_predicate_across_collects_correct(self):
        """Different predicates across collects each return the correctly filtered rows."""
        frame = self._frame()
        lf = cache_memory(lambda: frame, schema=frame.collect_schema())

        self.assertEqual(lf.filter(pl.col("a") < 2).collect()["a"].to_list(), [1])
        self.assertEqual(lf.filter(pl.col("a") >= 2).collect()["a"].to_list(), [2])

    def test_varying_projection_across_collects(self):
        """Different projections across collects each return the correct columns."""
        frame = self._frame()
        lf = cache_memory(lambda: frame, schema=frame.collect_schema())

        self.assertEqual(lf.select("b").collect().columns, ["b"])
        self.assertEqual(lf.select("a").collect().columns, ["a"])

    def test_callable_schema_resolved_without_running_builder(self):
        """A callable schema lets collect_schema() resolve columns without running the builder."""
        calls = []
        frame = self._frame()

        def build():
            calls.append(1)
            return frame

        lf = cache_memory(build, schema=lambda: frame.collect_schema())

        self.assertEqual(lf.collect_schema().names(), ["a", "b"])
        self.assertEqual(len(calls), 0)

    def test_missing_declared_column_raises(self):
        """A constructed frame lacking a declared column must raise, naming the column."""
        schema = self._frame().collect_schema()
        lf = cache_memory(lambda: self._frame().drop("b"), schema=schema)

        # The reconciliation ValueError is raised inside the source generator, so Polars
        # surfaces it as ComputeError (message preserved).
        with pytest.raises(ComputeError, match="'b'"):
            lf.collect()

    def test_declared_dtype_mismatch_raises(self):
        """A constructed frame whose dtype drifted from the declaration must raise."""
        schema = self._frame().collect_schema()
        lf = cache_memory(lambda: self._frame().with_columns(pl.col("a").cast(pl.Int32)), schema=schema)

        with pytest.raises(ComputeError, match="'a'"):
            lf.collect()

    def test_extra_column_dropped(self):
        """A constructed frame with an extra column keeps the declared schema constant."""
        schema = self._frame().collect_schema()
        lf = cache_memory(lambda: self._frame().with_columns(pl.lit(1).alias("extra")), schema=schema)

        self.assertEqual(lf.collect().columns, ["a", "b"])

    def test_non_schema_non_callable_rejected(self):
        """A plain dict (neither pl.Schema nor callable) is rejected up front."""
        with pytest.raises(TypeError, match="pl.Schema or a callable"):
            cache_memory(self._frame(), schema={"a": pl.Int64})

    def test_count_only_query_returns_true_row_count(self):
        """A count-only query, where Polars requests zero columns, still counts every row.

        Polars pushes ``with_columns=[]`` for ``pl.len()``. The projection logic must keep
        yielding one row per source row rather than collapsing to a 0-width, 0-row frame.
        """
        frame = self._frame()
        lf = cache_memory(lambda: frame, schema=frame.collect_schema())

        self.assertEqual(lf.select(pl.len()).collect().item(), frame.collect().height)

    def test_empty_projection_matches_plain_polars(self):
        """An explicitly empty projection behaves exactly as on a plain LazyFrame."""
        frame = self._frame()
        lf = cache_memory(lambda: frame, schema=frame.collect_schema())

        self.assertEqual(lf.select([]).collect().shape, frame.select([]).collect().shape)

    def test_non_serializable_source_memoized(self):
        """A non-serializable (lock-closing) plugin source works, unlike ``cache``.

        This is issue #26's core motivation: the plan cannot be serialized, so a
        content-addressed cache cannot key it, but cache_memory needs no key.
        """
        src = _stateful_source()
        # The source itself cannot be serialized (what `cache` would attempt for its key).
        with pytest.raises(Exception):
            src.serialize()

        lf = cache_memory(_stateful_source, schema=pl.Schema({"id": pl.Int64, "value": pl.Int64}))
        out = pl.concat([lf, lf], how="vertical").collect()
        self.assertEqual(out.height, 8)
        self.assertEqual(lf.filter(pl.col("id") == 1).collect()["value"].to_list(), [10])

    def test_collects_once_across_collect_all_fanout(self):
        """A frame fanned out into many branches collected via ``pl.collect_all`` builds once.

        This is the multi-branch / ``collect_all`` use case: several *separate* plans that each
        reference the same cache_memory frame are collected in one (parallel) pass. The
        double-checked lock must still collapse them to a single upstream execution even though
        Polars runs the branches concurrently across its thread pool.
        """
        counter: list = []
        base = pl.DataFrame({"id": [1, 2, 3, 4], "v": [10, 20, 30, 40]})

        def build():
            counter.append(1)
            return base.lazy()

        lf = cache_memory(build, schema=base.collect_schema())

        branches = [lf.filter(pl.col("id") == i).select(pl.col("v").sum()) for i in range(1, 5)]
        branches.append(lf.group_by("id").agg(pl.col("v").mean()))
        results = pl.collect_all(branches)

        self.assertEqual(len(counter), 1)
        # Sanity: each branch still returns the correct filtered slice.
        self.assertEqual(results[0].item(), 10)

        # A second, independent collect_all pass reuses the buffer (no rebuild).
        pl.collect_all([lf.select(pl.len()), lf.filter(pl.col("id") > 2)])
        self.assertEqual(len(counter), 1)

    def test_buffer_released_when_frame_dropped(self):
        """The collected buffer is reclaimed by GC once the returned frame is dropped.

        The buffer lives only in the returned frame's closure, so nothing outlives it. A
        sentinel embedded in the materialized buffer (via an Object-dtype column) must be
        weakref-dead after the frame is dropped and a collection runs.
        """

        class Sentinel:
            pass

        holder: dict = {}

        def build():
            sentinel = Sentinel()
            holder["ref"] = weakref.ref(sentinel)
            return pl.DataFrame(
                {"a": [1, 2], "obj": [sentinel, sentinel]},
                schema={"a": pl.Int64, "obj": pl.Object},
            ).lazy()

        lf = cache_memory(build, schema=pl.Schema({"a": pl.Int64, "obj": pl.Object}))
        lf.collect()  # force the one-time build so the buffer holds the sentinel

        self.assertIsNotNone(holder["ref"](), "sentinel should be alive while the frame is held")

        del lf
        gc.collect()

        self.assertIsNone(holder["ref"](), "buffer (and its contents) must be released once the frame is dropped")

    def test_failed_build_not_amplified_across_collect_all(self):
        """A failing build runs once per episode, not once per fanned-out branch.

        Regression for the fan-out failure path: without episode sharing, every branch parked on
        the lock would re-run the failing builder (N failed builds for N branches in one
        ``collect_all``). Concurrent waiters must instead share the single build's exception.
        """
        attempts: list = []

        def build():
            attempts.append(1)
            raise ValueError("boom")

        lf = cache_memory(build, schema=pl.Schema({"a": pl.Int64}))
        branches = [lf.filter(pl.col("a") == i) for i in range(64)]

        with pytest.raises(ComputeError, match="boom"):
            pl.collect_all(branches)

        # One build episode for the whole collect_all pass, not one per branch.
        self.assertEqual(len(attempts), 1)

    def test_failed_build_is_terminal_and_cached(self):
        """A failed build runs once; the exception is cached and re-raised on later collects.

        Retry is by constructing a fresh cache_memory, matching the success path's
        instance-scoped model. This is what keeps a failing builder from being re-run once
        per branch under fan-out / collect_all.
        """
        state = {"calls": 0}
        frame = self._frame()

        def build():
            state["calls"] += 1
            raise ValueError("transient")

        lf = cache_memory(build, schema=frame.collect_schema())

        with pytest.raises(ComputeError, match="transient"):
            lf.collect()
        with pytest.raises(ComputeError, match="transient"):
            lf.collect()

        # Built once, cached failure re-raised on the second collect.
        self.assertEqual(state["calls"], 1)

        # A fresh instance retries from scratch.
        state["ok"] = frame
        lf2 = cache_memory(lambda: frame, schema=frame.collect_schema())
        self.assertEqual(lf2.collect()["a"].to_list(), [1, 2])

    def test_failed_build_does_not_retain_frame(self):
        """A cached failure must not pin the failed DataFrame (via a stored exception traceback).

        Exercises the exact retention path: reconciliation fails *after* ``build`` returns a frame
        holding a sentinel, so the materialized ``built`` frame is live at the raise site. The
        cached failure must be a lightweight record, not the live exception whose traceback pins it.
        """

        class Sentinel:
            pass

        holder: dict = {}

        def build():
            sentinel = Sentinel()
            holder["ref"] = weakref.ref(sentinel)
            # 'a' drifts Int64 -> Int32 so reconciliation raises inside get_buffer, where ``built``
            # (carrying the sentinel in its Object column) is the live materialized frame.
            return pl.DataFrame(
                {"a": pl.Series([1, 2], dtype=pl.Int32), "obj": [sentinel, sentinel]},
                schema={"a": pl.Int32, "obj": pl.Object},
            ).lazy()

        lf = cache_memory(build, schema=pl.Schema({"a": pl.Int64, "obj": pl.Object}))
        with pytest.raises(ComputeError, match="'a'"):
            lf.collect()
        # A second collect re-raises the cached failure without rebuilding.
        with pytest.raises(ComputeError, match="'a'"):
            lf.collect()

        gc.collect()
        # The frame built during the failed attempt must not be retained by the cached error.
        self.assertIsNone(holder["ref"](), "failed build must not be retained by the cached error")

    def test_keyboard_interrupt_build_is_terminal(self):
        """An interrupted build is terminal (not retried per branch); a fresh instance retries.

        Recording every failure — including a control-flow BaseException — is what keeps a
        fan-out / collect_all from re-running the builder once per parked waiter. The interrupt
        propagates on the first collect; later collects re-raise a neutral cached error.
        """
        state = {"calls": 0}
        frame = self._frame()

        def build():
            state["calls"] += 1
            raise KeyboardInterrupt

        lf = cache_memory(build, schema=frame.collect_schema())

        with pytest.raises((KeyboardInterrupt, ComputeError)):
            lf.collect()
        # Subsequent collect re-raises the cached failure (as ComputeError) without rebuilding.
        with pytest.raises(ComputeError):
            lf.collect()
        self.assertEqual(state["calls"], 1)

        # A fresh instance retries from scratch.
        lf2 = cache_memory(lambda: frame, schema=frame.collect_schema())
        self.assertEqual(lf2.collect()["a"].to_list(), [1, 2])

    @pytest.mark.timeout(15)
    def test_unprintable_exception_does_not_deadlock(self):
        """A builder exception whose ``__str__`` raises must still publish a terminal outcome.

        Regression: deriving the cached message inside the lock would abort the critical section
        before clearing ``building``, hanging every later collect. The message must be derived
        before the lock, with a fallback that does not call ``str``.
        """
        calls: list = []

        class BadStringError(Exception):
            def __str__(self):
                raise RuntimeError("cannot format")

        def build():
            calls.append(1)
            raise BadStringError

        lf = cache_memory(build, schema=pl.Schema({"a": pl.Int64}))

        with pytest.raises(ComputeError):
            lf.collect()
        # Must not hang: the terminal state was published despite the broken __str__.
        with pytest.raises(ComputeError):
            lf.collect()
        self.assertEqual(len(calls), 1)

    def test_failed_build_not_amplified_for_base_exception(self):
        """A control-flow failure also collapses to one build across a collect_all fan-out."""
        attempts: list = []

        def build():
            attempts.append(1)
            raise KeyboardInterrupt

        lf = cache_memory(build, schema=pl.Schema({"a": pl.Int64}))
        branches = [lf.filter(pl.col("a") == i) for i in range(64)]

        with pytest.raises((KeyboardInterrupt, ComputeError)):
            pl.collect_all(branches)

        self.assertEqual(len(attempts), 1)

    def test_empty_projection_preserves_row_count(self):
        """A directly-pushed empty projection must not collapse row cardinality (count-query guard).

        Some Polars versions may push ``with_columns=[]`` for a count-only query. The generator
        must still yield one row per source row. Invoked directly because Polars 1.42.1 pushes a
        named column for ``pl.len()`` and never reaches this branch through normal use.
        """
        frame = self._frame()
        captured: dict = {}

        def build():
            return frame

        # Capture the underlying io_source generator to invoke the empty-projection path directly.
        real_register = register_io_source_with_is_pure

        def _capture(io_source, schema, **kwargs):
            captured["gen"] = io_source
            return real_register(io_source, schema, **kwargs)

        import polars_io_tools.io_sources.lazy_cache_memory as mod

        orig = mod.register_io_source_with_is_pure
        mod.register_io_source_with_is_pure = _capture
        try:
            cache_memory(build, schema=frame.collect_schema())
        finally:
            mod.register_io_source_with_is_pure = orig

        rows = sum(df.height for df in captured["gen"](with_columns=[], predicate=None, n_rows=None, batch_size=None))
        self.assertEqual(rows, frame.collect().height)

    def test_callable_schema_resolved_once(self):
        """A callable schema is resolved exactly once even across schema queries and collects."""
        calls = []
        frame = self._frame()

        def sch():
            calls.append(1)
            return frame.collect_schema()

        lf = cache_memory(lambda: frame, schema=sch)
        lf.collect_schema()
        lf.collect()
        lf.collect()

        self.assertEqual(len(calls), 1)

    # ------------------------------------------------------------------
    # Partitioning (partition_cols) — lazy per-partition build.
    # ------------------------------------------------------------------

    def test_no_partition_cols_unchanged(self):
        """An explicit empty ``partition_cols`` still collapses N references to one scan.

        Guards the single-buffer fast path: the partitioned code must be byte-for-byte
        equivalent to today when no partition columns are given.
        """
        scans: list = []
        frame = self._frame()
        lf = cache_memory(lambda: _counting_source(frame, scans), schema=frame.collect_schema(), partition_cols=())

        pl.concat([lf, lf], how="vertical").collect()
        lf.collect()

        self.assertEqual(len(scans), 1)

    def test_partition_predicate_restricts_build(self):
        """A predicate on a partition column builds only the matching partition.

        Demanding ``region == "us"`` collects only the ``us`` rows; a later ``eu`` demand
        triggers a second build. The builder is called twice total — once per demanded
        partition — never once per partition value.
        """
        calls: list = []

        def build():
            calls.append(1)
            return self._pframe()

        lf = cache_memory(build, schema=self._pschema(), partition_cols="region")

        us = lf.filter(pl.col("region") == "us").sort("id").collect()
        self.assertEqual(us["v"].to_list(), [10, 20])
        self.assertEqual(len(calls), 1)

        eu = lf.filter(pl.col("region") == "eu").sort("id").collect()
        self.assertEqual(eu["v"].to_list(), [30, 40])
        self.assertEqual(len(calls), 2)

        # A repeat of an already-built partition does not rebuild.
        lf.filter(pl.col("region") == "us").collect()
        self.assertEqual(len(calls), 2)

    def test_full_build_when_no_partition_predicate(self):
        """A demand with no partition predicate full-builds once; later demands never rebuild."""
        calls: list = []

        def build():
            calls.append(1)
            return self._pframe()

        lf = cache_memory(build, schema=self._pschema(), partition_cols="region")

        # No partition predicate -> full build of every partition.
        self.assertEqual(lf.select(pl.col("v").sum()).collect().item(), 100)
        self.assertEqual(len(calls), 1)

        # A later partition-restricted demand is served from buffers (full_built short-circuit).
        lf.filter(pl.col("region") == "us").collect()
        self.assertEqual(len(calls), 1)

    def test_partitioned_result_correct(self):
        """Partitioned results match the equivalent un-cached filter (order-insensitive)."""
        frame = self._pframe()
        lf = cache_memory(lambda: self._pframe(), schema=self._pschema(), partition_cols="region")

        eu = lf.filter(pl.col("region") == "eu").sort("id").collect()
        assert_frame_equal(eu, frame.filter(pl.col("region") == "eu").sort("id").collect())

        all_rows = lf.sort("id").collect()
        assert_frame_equal(all_rows, frame.sort("id").collect())

    def test_partition_col_not_in_declared_schema(self):
        """A partition column absent from the declared schema is dropped from the buffers.

        The builder must still produce the partition column (it drives ``partition_by``),
        but it is dropped from the stored buffers so they match the declared schema. Such a
        column cannot be filtered on downstream (Polars validates predicates against the
        advertised schema), so a full build is the usable path.
        """
        declared = pl.Schema({"id": pl.Int64, "v": pl.Int64})  # region omitted
        lf = cache_memory(lambda: self._pframe(), schema=declared, partition_cols="region")

        out = lf.sort("id").collect()
        self.assertEqual(out.columns, ["id", "v"])
        self.assertEqual(out["v"].to_list(), [10, 20, 30, 40])

    def test_null_valued_partition_incremental_builds(self):
        """A null-valued partition builds and serves independently of non-null partitions.

        The build subtraction must use null-safe equality: excluding a non-null partition
        must not filter out the null partition (and vice versa), which naive ``col == value``
        would, since ``col == None`` is null under three-valued logic.
        """

        def src():
            return pl.LazyFrame({"g": ["a", None, "a", None], "v": [1, 2, 3, 4]})

        schema = pl.Schema({"g": pl.String, "v": pl.Int64})

        # Non-null partition first, then the null partition: the null rows must still build.
        lf = cache_memory(src, schema=schema, partition_cols="g")
        self.assertEqual(lf.filter(pl.col("g") == "a").sort("v").collect()["v"].to_list(), [1, 3])
        self.assertEqual(lf.filter(pl.col("g").is_null()).sort("v").collect()["v"].to_list(), [2, 4])

        # Null partition first, then a non-null demand: the non-null rows must not be poisoned.
        lf2 = cache_memory(src, schema=schema, partition_cols="g")
        self.assertEqual(lf2.filter(pl.col("g").is_null()).sort("v").collect()["v"].to_list(), [2, 4])
        self.assertEqual(lf2.filter(pl.col("g") == "a").sort("v").collect()["v"].to_list(), [1, 3])

    def test_str_partition_cols_normalized(self):
        """A bare-string ``partition_cols`` behaves identically to a one-element sequence."""
        lf_str = cache_memory(lambda: self._pframe(), schema=self._pschema(), partition_cols="region")
        lf_seq = cache_memory(lambda: self._pframe(), schema=self._pschema(), partition_cols=["region"])

        assert_frame_equal(
            lf_str.filter(pl.col("region") == "us").sort("id").collect(),
            lf_seq.filter(pl.col("region") == "us").sort("id").collect(),
        )

    def test_partition_cols_order_insensitive(self):
        """``partition_cols`` is order-independent (sorted internally, matching ``cache``)."""

        def src():
            return pl.LazyFrame({"r": ["us", "us", "eu"], "z": ["a", "b", "a"], "v": [1, 2, 3]})

        schema = pl.Schema({"r": pl.String, "z": pl.String, "v": pl.Int64})
        lf_ab = cache_memory(src, schema=schema, partition_cols=["r", "z"])
        lf_ba = cache_memory(src, schema=schema, partition_cols=["z", "r"])

        assert_frame_equal(
            lf_ab.filter(pl.col("r") == "us").sort("v").collect(),
            lf_ba.filter(pl.col("r") == "us").sort("v").collect(),
        )

    def test_count_only_query_partitioned(self):
        """A count-only query over a partitioned instance returns the true total row count."""
        lf = cache_memory(lambda: self._pframe(), schema=self._pschema(), partition_cols="region")

        self.assertEqual(lf.select(pl.len()).collect().item(), 4)

    def test_partitioned_collects_builder_once_per_build(self):
        """N references to the same partition subset in one plan yield one upstream scan.

        The CSE-collapse guarantee still holds per build when partitioning.
        """
        scans: list = []
        frame = self._pframe()
        lf = cache_memory(
            lambda: _counting_source(frame, scans),
            schema=self._pschema(),
            partition_cols="region",
        )

        branch = lf.filter(pl.col("region") == "us")
        pl.concat([branch, branch], how="vertical").collect()

        self.assertEqual(len(scans), 1)

    def test_failed_build_terminal_partitioned(self):
        """A failing build is terminal for the whole partitioned instance and re-raised.

        Matches the unpartitioned contract: one build attempt, the cached failure re-raised
        on every subsequent demand (any partition).
        """
        state = {"calls": 0}

        def build():
            state["calls"] += 1
            raise ValueError("boom")

        lf = cache_memory(build, schema=self._pschema(), partition_cols="region")

        with pytest.raises(ComputeError, match="boom"):
            lf.filter(pl.col("region") == "us").collect()
        with pytest.raises(ComputeError, match="boom"):
            lf.filter(pl.col("region") == "eu").collect()

        self.assertEqual(state["calls"], 1)

    def test_failed_build_not_amplified_across_collect_all_partitioned(self):
        """A failing build runs once across a partitioned collect_all fan-out, not once per branch."""
        attempts: list = []

        def build():
            attempts.append(1)
            raise ValueError("boom")

        lf = cache_memory(build, schema=self._pschema(), partition_cols="region")
        regions = ["us", "eu"] * 32
        branches = [lf.filter(pl.col("region") == r) for r in regions]

        with pytest.raises(ComputeError, match="boom"):
            pl.collect_all(branches)

        self.assertEqual(len(attempts), 1)

    def test_partition_buffers_released_when_frame_dropped(self):
        """Per-partition buffers are reclaimed by GC once the returned frame is dropped.

        The dict-of-buffers state lives only in the returned frame's closure, so a sentinel
        embedded in a materialized partition must be weakref-dead after the frame is dropped.
        """

        class Sentinel:
            pass

        holder: dict = {}

        def build():
            sentinel = Sentinel()
            holder["ref"] = weakref.ref(sentinel)
            return pl.DataFrame(
                {"region": ["us", "eu"], "a": [1, 2], "obj": [sentinel, sentinel]},
                schema={"region": pl.String, "a": pl.Int64, "obj": pl.Object},
            ).lazy()

        lf = cache_memory(
            build,
            schema=pl.Schema({"region": pl.String, "a": pl.Int64, "obj": pl.Object}),
            partition_cols="region",
        )
        lf.collect()  # full build materializes the partition buffers holding the sentinel

        self.assertIsNotNone(holder["ref"](), "sentinel should be alive while the frame is held")

        del lf
        gc.collect()

        self.assertIsNone(holder["ref"](), "partition buffers must be released once the frame is dropped")


def test_exclude_built_infers_dtype_with_null_first_keys():
    """Regression: carrier-frame inference must not fail when early keys are null-valued.

    ``pl.DataFrame`` infers dtypes from a bounded prefix by default, so many ``a=None`` keys
    followed by a valued ``a`` key beyond that prefix used to infer ``Null`` for ``a`` and fail
    to append the later integer. ``infer_schema_length=None`` scans every key instead.
    """
    from polars_io_tools.io_sources.lazy_cache_memory import _exclude_built
    from polars_io_tools.io_sources.util import partition_key

    # Force the order (a sequence, not a set) so the single valued key deterministically lands
    # past Polars' 100-row inference prefix — the case set iteration only hits sometimes.
    keys = [partition_key({"a": None, "b": i}) for i in range(150)]
    keys.append(partition_key({"a": 7, "b": 999}))
    expr = _exclude_built(keys)
    assert expr is not None

    frame = pl.DataFrame(
        {"a": [7, 7, None, 3], "b": [999, 1, 5, 5]},
        schema={"a": pl.Int64, "b": pl.Int64},
    )
    # Keep only rows not matching a built partition: (a=7,b=999) and every (a=None, b in 0..149)
    # are excluded; (a=7,b=1) and (a=3,b=5) are kept.
    assert frame.filter(expr).to_dicts() == [{"a": 7, "b": 1}, {"a": 3, "b": 5}]
