"""Regression tests for the polars-iosource-collect_batches deadlock.

The deadlock occurs when an io_source callback registered via
``register_io_source`` consumes a sub-LazyFrame via ``collect_batches`` while
the polars Rayon pool only has a single worker AND the outer plan contains a
join with a pushed-down predicate. See ``polars_io_tools.io_sources.util.collect_lf_in_io_source``
for the fix.

Tests run in a subprocess because ``POLARS_MAX_THREADS`` is read once at
polars import time and the thread pool cannot be resized afterwards.
"""

import os
import subprocess
import sys
import textwrap

import polars as pl
import pytest

from polars_io_tools.io_sources import lazy_cache_parquet
from polars_io_tools.io_sources.util import register_io_source_with_is_pure

# 60s should be more than enough for these tiny queries; bare deadlock
# repros hang indefinitely so any timeout would catch them.
_TIMEOUT_S = 60


def _run_in_subprocess(script: str, *, threads: int = 1) -> subprocess.CompletedProcess:
    # Inherit the parent env so the subprocess finds libpython, the venv,
    # PYTHONPATH, etc. (we only need to override POLARS_MAX_THREADS).
    env = {**os.environ, "POLARS_MAX_THREADS": str(threads)}
    return subprocess.run(
        [sys.executable, "-P", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
    )


@pytest.mark.parametrize("threads", [1, 2])
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
    result = _run_in_subprocess(script, threads=threads)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "OK" in result.stdout


@pytest.mark.parametrize(
    "otel_setup",
    [
        "",
        """
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(InMemorySpanExporter()))
        trace.set_tracer_provider(provider)
        """,
    ],
    ids=["otel-no-op", "otel-batch-processor"],
)
def test_iosource_collect_batches_does_not_deadlock_at_threads_one(otel_setup):
    """Reproduces the polars-io-tools / polars deadlock and asserts the fix holds.

    Without the helper's thread-pool gate this script hangs forever at
    ``POLARS_MAX_THREADS=1`` due to a Rayon scheduling deadlock when the
    outer engine reaches an io_source whose callback re-enters polars via
    ``collect_batches`` and the outer plan contains a join with a
    pushed-down predicate.
    """
    script = textwrap.dedent(
        """
        import polars as pl
        from polars_io_tools.io_sources.util import (
            collect_lf_in_io_source,
            register_io_source_with_is_pure,
        )

        assert pl.thread_pool_size() == 1, pl.thread_pool_size()

        __OTEL_SETUP__

        # This is the minimal pattern that deadlocks polars 1.39.3+1.40.1
        # without the fix in collect_lf_in_io_source. Removing any one of:
        # the join, the pushed-down predicate, the inner filter+collect_batches,
        # or POLARS_MAX_THREADS=1 avoids the hang.
        SCHEMA = {"key": pl.Utf8, "value": pl.Int64}

        def reader(with_columns, predicate, n_rows, batch_size):
            df = pl.DataFrame(
                {
                    "key": ["a", "b", "c", "d", "e"] * 20_000,
                    "value": list(range(100_000)),
                },
                schema=SCHEMA,
            )
            lf = df.lazy()
            if predicate is not None:
                lf = lf.filter(predicate)
            yield from collect_lf_in_io_source(lf, batch_size or 100_000)

        src = register_io_source_with_is_pure(reader, schema=SCHEMA)
        rhs = pl.LazyFrame(
            {"key": ["a", "b", "c", "d", "e"], "label": ["A", "B", "C", "D", "E"]}
        )

        plan = (
            src.filter(pl.col("key").is_in(["a", "b", "c", "d", "e"]))
               .join(rhs, on="key", how="left")
        )
        out = plan.collect()
        assert out.height == 100_000, out.height
        print("OK", out.height)
        """
    ).replace("__OTEL_SETUP__", textwrap.dedent(otel_setup))

    try:
        result = _run_in_subprocess(script)
    except subprocess.TimeoutExpired as e:
        pytest.fail(f"deadlock: subprocess did not finish in {_TIMEOUT_S}s\nstdout so far: {e.stdout!r}\nstderr so far: {e.stderr!r}")

    assert result.returncode == 0, f"subprocess failed: returncode={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    assert "OK 100000" in result.stdout, result.stdout


def test_helper_collect_method_at_threads_one():
    """``collect_lf_in_io_source`` at a single worker: stream via ``collect_batches`` on fixed
    runtimes (``POLARS_STREAMING_SUPPORTS_NESTED_IO_PLUGIN``), else fall back to full-collect +
    ``iter_slices`` to dodge the (old) collect_batches re-entrancy deadlock. Either way the
    results are correct and the call does not hang."""
    script = textwrap.dedent(
        """
        import polars as pl
        from polars_io_tools.io_sources.util import collect_lf_in_io_source
        from polars_io_tools._compat import (
            POLARS_HAS_COLLECT_BATCHES,
            POLARS_STREAMING_SUPPORTS_NESTED_IO_PLUGIN,
        )

        assert pl.thread_pool_size() == 1, pl.thread_pool_size()

        used = {"collect_batches": 0}
        _orig_cb = getattr(pl.LazyFrame, "collect_batches", None)
        if _orig_cb is not None:
            def _spy(self, *a, **k):
                used["collect_batches"] += 1
                return _orig_cb(self, *a, **k)
            pl.LazyFrame.collect_batches = _spy

        lf = pl.LazyFrame({"a": list(range(1000))}).filter(pl.col("a") < 500)

        # batch_size=None -> single DataFrame (always a plain collect, never collect_batches)
        chunks = list(collect_lf_in_io_source(lf, None))
        assert len(chunks) == 1, len(chunks)
        assert chunks[0].height == 500, chunks[0].height
        assert used["collect_batches"] == 0, used

        # batch_size set -> multiple DataFrames summing to expected
        chunks = list(collect_lf_in_io_source(lf, 100))
        assert sum(c.height for c in chunks) == 500

        expect_batches = POLARS_HAS_COLLECT_BATCHES and POLARS_STREAMING_SUPPORTS_NESTED_IO_PLUGIN
        if expect_batches:
            assert used["collect_batches"] == 1, ("expected collect_batches on fixed runtime", used)
        else:
            assert used["collect_batches"] == 0, ("expected iter_slices fallback on old runtime", used)
        print("OK")
        """
    )
    result = _run_in_subprocess(script)
    assert result.returncode == 0, f"subprocess failed: returncode={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    assert "OK" in result.stdout, result.stdout


def test_cache_parquet_sink_preparation_python_scan_behavior(monkeypatch):
    """Sink preparation streams nested io_sources on fixed runtimes, materializes on old ones.

    On runtimes where nested streaming io_source collection is known-good
    (``POLARS_STREAMING_SUPPORTS_NESTED_IO_PLUGIN``, polars >= 1.43.2),
    ``_prepare_lf_for_sink_from_io_source`` must return the frame lazily (no eager collect of
    the source), so a large nested source streams into the sink instead of materializing. On
    older runtimes it must retain the previous behavior: materialize the source eagerly to
    dodge the partitioned-sink re-entrancy deadlock.
    """
    from polars_io_tools._compat import POLARS_STREAMING_SUPPORTS_NESTED_IO_PLUGIN

    calls = 0

    def reader(with_columns, predicate, n_rows, batch_size):
        nonlocal calls
        calls += 1
        yield pl.DataFrame({"a": [1]})

    monkeypatch.setattr(lazy_cache_parquet.pl, "thread_pool_size", lambda: 8)

    lf = register_io_source_with_is_pure(reader, schema={"a": pl.Int64})
    prepared = lazy_cache_parquet._prepare_lf_for_sink_from_io_source(lf)

    if POLARS_STREAMING_SUPPORTS_NESTED_IO_PLUGIN:
        # Streamed: the source is not collected during preparation.
        assert calls == 0
        assert prepared.collect().to_dict(as_series=False) == {"a": [1]}
    else:
        # Materialized: the source is collected eagerly during preparation.
        assert calls == 1
        assert prepared.collect().to_dict(as_series=False) == {"a": [1]}
        assert calls == 1


def test_cache_parquet_chained_write_does_not_deadlock_at_threads_one():
    """Regression test for chained ``cache_parquet`` writes under one Polars worker."""
    script = textwrap.dedent(
        """
        import datetime
        from tempfile import TemporaryDirectory

        import polars as pl
        import polars_io_tools  # noqa: F401

        assert pl.thread_pool_size() == 1, pl.thread_pool_size()

        def run(base, do_collect):
            src = pl.DataFrame(
                {
                    "a": [1, 2, 3],
                    "date": [datetime.date(2025, 7, 31)] * 3,
                }
            ).lazy()

            ldf = src.piot.cache_parquet(base / "foo_test", "date", time_unit="daily")
            if do_collect:
                ldf.collect()

            ldf1 = ldf.with_columns(pl.lit("foo").alias("foo")).piot.cache_parquet(base / "foo_test1", "date", time_unit="daily")
            ldf2 = ldf.with_columns(pl.lit("bar").alias("bar")).piot.cache_parquet(base / "foo_test2", "date", time_unit="daily")

            out = ldf1.join(ldf2, on="a").collect()
            assert out.height == 3, out

        with TemporaryDirectory() as tmpdir:
            from pathlib import Path

            base = Path(tmpdir)
            run(base / "scenario_with_collect", True)
            run(base / "scenario_no_collect", False)

        print("OK")
        """
    )
    try:
        result = _run_in_subprocess(script)
    except subprocess.TimeoutExpired as e:
        pytest.fail(f"deadlock: subprocess did not finish in {_TIMEOUT_S}s\nstdout so far: {e.stdout!r}\nstderr so far: {e.stderr!r}")

    assert result.returncode == 0, f"subprocess failed: returncode={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    assert "OK" in result.stdout, result.stdout
