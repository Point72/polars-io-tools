"""Unit tests for :func:`polars_io_tools.io_sources.lazy_cache_memory.cache_memory`."""

import gc
import threading
import unittest
import weakref

import polars as pl
import pytest
from polars.exceptions import ComputeError
from polars.io.plugins import register_io_source

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
