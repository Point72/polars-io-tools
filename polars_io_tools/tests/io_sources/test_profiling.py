import time
from collections.abc import Iterator

import polars as pl
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from polars_io_tools.io_sources.profiling import (
    get_source_span_parent,
    profile_io_source_iterator,
    set_source_span_parent,
    wrap_io_source_with_profiling,
)
from polars_io_tools.io_sources.util import register_io_source_with_is_pure

_EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="module", autouse=True)
def _otel_provider():
    provider = trace.get_tracer_provider()
    if not hasattr(provider, "add_span_processor"):
        trace.set_tracer_provider(TracerProvider())
        provider = trace.get_tracer_provider()
    if hasattr(provider, "add_span_processor"):
        provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))
    yield


@pytest.fixture(autouse=True)
def _reset():
    _EXPORTER.clear()
    set_source_span_parent(None)
    yield
    set_source_span_parent(None)


def _profile(iterable, *, fallback_n_columns=None):
    return profile_io_source_iterator(iterable, explain_name="test.reader", fallback_n_columns=fallback_n_columns)


def _spans():
    return [s for s in _EXPORTER.get_finished_spans() if s.name.startswith("io_source.execute")]


def _span():
    spans = _spans()
    assert len(spans) == 1
    return spans[0]


def _module_level_reader():
    # A module-level factory whose inner closure has qualname ``_module_level_reader.<locals>.source_generator``,
    # matching how the real readers define their ``source_generator``.
    def source_generator(*_args, **_kwargs):
        return iter([pl.DataFrame({"value": [1]})])

    return source_generator


def test_multi_batch_accumulates_only_pull_time():
    sleep_s = 0.015

    def source() -> Iterator[pl.DataFrame]:
        for value in range(3):
            time.sleep(sleep_s)
            yield pl.DataFrame({"value": [value]})

    assert len(list(_profile(source()))) == 3

    span = _span()
    assert span.attributes["polars_io_tools.next_elapsed_total_ms"] >= 3 * sleep_s * 0.8 * 1000
    assert span.attributes["polars_io_tools.total_rows"] == 3
    assert span.attributes["polars_io_tools.n_columns"] == 1
    assert span.attributes["polars_io_tools.batch_count"] == 3
    assert span.attributes["polars_io_tools.outcome"] == "exhausted"


def test_slow_terminal_stop_iteration_is_counted():
    terminal_sleep_s = 0.025

    class SlowTerminalIterator:
        def __init__(self) -> None:
            self._yielded = False

        def __iter__(self):
            return self

        def __next__(self):
            if not self._yielded:
                self._yielded = True
                return pl.DataFrame({"value": [1]})
            time.sleep(terminal_sleep_s)
            raise StopIteration

    assert len(list(_profile(SlowTerminalIterator()))) == 1
    span = _span()
    assert span.attributes["polars_io_tools.next_elapsed_total_ms"] >= terminal_sleep_s * 0.8 * 1000
    assert span.attributes["polars_io_tools.outcome"] == "exhausted"


def test_failure_before_first_yield_is_counted_and_propagated_unchanged():
    error = ValueError("source failed")
    failure_sleep_s = 0.025

    class FailingIterator:
        def __iter__(self):
            return self

        def __next__(self):
            time.sleep(failure_sleep_s)
            raise error

    with pytest.raises(ValueError) as exc_info:
        next(_profile(FailingIterator()))

    assert exc_info.value is error
    span = _span()
    assert span.attributes["polars_io_tools.next_elapsed_total_ms"] >= failure_sleep_s * 0.8 * 1000
    assert span.attributes["polars_io_tools.total_rows"] == 0
    assert span.attributes["polars_io_tools.batch_count"] == 0
    assert span.attributes["polars_io_tools.outcome"] == "error"
    assert span.attributes["error.type"] == "ValueError"


def test_failure_reports_partial_rows():
    error = LookupError("failed after a batch")

    def source() -> Iterator[pl.DataFrame]:
        yield pl.DataFrame({"value": [1, 2]})
        time.sleep(0.015)
        raise error

    iterator = _profile(source())
    assert next(iterator).height == 2
    with pytest.raises(LookupError) as exc_info:
        next(iterator)

    assert exc_info.value is error
    span = _span()
    assert span.attributes["polars_io_tools.total_rows"] == 2
    assert span.attributes["polars_io_tools.n_columns"] == 1
    assert span.attributes["polars_io_tools.batch_count"] == 1
    assert span.attributes["polars_io_tools.outcome"] == "error"
    assert span.attributes["polars_io_tools.next_elapsed_total_ms"] >= 10


def test_consumer_sleep_between_pulls_is_excluded():
    iterator = _profile(iter([pl.DataFrame({"value": [1]}), pl.DataFrame({"value": [2]})]))

    next(iterator)
    time.sleep(0.1)
    next(iterator)
    with pytest.raises(StopIteration):
        next(iterator)

    assert _span().attributes["polars_io_tools.next_elapsed_total_ms"] < 50


@pytest.mark.parametrize("pull_first", [False, True])
def test_early_close_propagates_to_custom_iterator_and_emits_once(pull_first):
    class ClosableIterator:
        def __init__(self) -> None:
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            return pl.DataFrame({"value": [1]})

        def close(self):
            self.closed = True

    inner = ClosableIterator()
    iterator = _profile(inner)
    if pull_first:
        next(iterator)
    iterator.close()
    iterator.close()

    assert inner.closed
    span = _span()
    assert span.attributes["polars_io_tools.outcome"] == "closed"
    assert span.attributes["polars_io_tools.total_rows"] == (1 if pull_first else 0)


def test_exactly_once_emission_for_all_outcomes():
    assert list(_profile(iter(()))) == []
    assert _span().attributes["polars_io_tools.outcome"] == "exhausted"
    _EXPORTER.clear()

    closed = _profile(iter([pl.DataFrame({"value": [1]})]))
    next(closed)
    closed.close()
    assert _span().attributes["polars_io_tools.outcome"] == "closed"
    _EXPORTER.clear()

    def failing() -> Iterator[pl.DataFrame]:
        raise ZeroDivisionError("boom")
        yield

    with pytest.raises(ZeroDivisionError):
        next(_profile(failing()))
    assert _span().attributes["polars_io_tools.outcome"] == "error"


def test_emission_failure_never_masks_completion_or_source_error(monkeypatch):
    import opentelemetry.trace as ot

    def boom(*_args, **_kwargs):
        raise RuntimeError("tracer boom")

    monkeypatch.setattr(ot, "get_tracer", boom)

    frames = list(_profile(iter([pl.DataFrame({"value": [1]})])))
    assert frames[0].height == 1

    error = OSError("source error")

    def failing() -> Iterator[pl.DataFrame]:
        raise error
        yield

    with pytest.raises(OSError) as exc_info:
        next(_profile(failing()))
    assert exc_info.value is error


def test_no_yield_uses_known_projection_or_eager_schema_width():
    wrapped = wrap_io_source_with_profiling(
        lambda *_args, **_kwargs: iter(()),
        schema={"a": pl.Int64, "b": pl.Int64},
        explain_name="empty.reader",
    )
    assert list(wrapped(["a"], None, None, None)) == []
    assert _span().attributes["polars_io_tools.n_columns"] == 1
    _EXPORTER.clear()

    wrapped = wrap_io_source_with_profiling(
        lambda *_args, **_kwargs: iter(()),
        schema={"a": pl.Int64, "b": pl.Int64},
        explain_name="empty.reader",
    )
    assert list(wrapped(None, None, None, None)) == []
    assert _span().attributes["polars_io_tools.n_columns"] == 2


def test_empty_first_batch_sets_width_and_zero_rows():
    frames = list(_profile(iter([pl.DataFrame(schema={"a": pl.Int64, "b": pl.String})])))
    assert frames[0].is_empty()

    span = _span()
    assert span.attributes["polars_io_tools.total_rows"] == 0
    assert span.attributes["polars_io_tools.n_columns"] == 2
    assert span.attributes["polars_io_tools.batch_count"] == 1


def test_deferred_schema_is_not_forced_for_telemetry():
    schema_calls = 0

    def schema():
        nonlocal schema_calls
        schema_calls += 1
        return {"a": pl.Int64}

    wrapped = wrap_io_source_with_profiling(
        lambda *_args, **_kwargs: iter(()),
        schema=schema,
        explain_name="empty.reader",
    )
    assert list(wrapped(None, None, None, None)) == []
    assert schema_calls == 0
    assert "polars_io_tools.n_columns" not in _span().attributes


def test_eager_callable_work_is_excluded_from_pull_time():
    def eager_source(*_args, **_kwargs):
        time.sleep(0.05)
        return iter([pl.DataFrame({"value": [1]})])

    wrapped = wrap_io_source_with_profiling(
        eager_source,
        schema={"value": pl.Int64},
        explain_name="eager.reader",
    )
    frames = list(wrapped(None, None, None, None))
    assert frames[0].height == 1
    assert _span().attributes["polars_io_tools.next_elapsed_total_ms"] < 30


def test_source_parent_is_captured_at_emit_time():
    # The parent context is read when the span is emitted (after execution), NOT at construction -- a query observer
    # publishes its context after the LazyFrame/iterator is built, so a parent set only after construction must apply.
    parent = trace.get_tracer("test").start_span("parent")
    iterator = _profile(iter([pl.DataFrame({"value": [1]})]))
    set_source_span_parent(trace.set_span_in_context(parent))  # published after construction
    try:
        list(iterator)
    finally:
        set_source_span_parent(None)
        parent.end()

    span = _span()
    assert span.parent is not None
    assert span.parent.span_id == parent.get_span_context().span_id


def test_source_parent_not_captured_at_construction():
    # A parent present at construction but cleared before execution completes is not captured.
    parent = trace.get_tracer("test").start_span("parent")
    set_source_span_parent(trace.set_span_in_context(parent))
    iterator = _profile(iter([pl.DataFrame({"value": [1]})]))
    set_source_span_parent(None)  # cleared before the source is drained
    list(iterator)
    parent.end()

    assert get_source_span_parent() is None
    assert _span().parent is None


def test_default_source_identity_is_stable_for_callable_objects():
    class CallableSource:
        def __call__(self, *_args, **_kwargs):
            return iter([pl.DataFrame({"value": [1]})])

    wrapped = wrap_io_source_with_profiling(CallableSource(), schema={"value": pl.Int64})
    list(wrapped(None, None, None, None))

    name = _span().attributes["polars_io_tools.explain_name"]
    assert name.endswith("CallableSource")
    assert "0x" not in name


def test_default_source_identity_trims_closure_to_enclosing_function():
    # Mirrors the real readers: a ``source_generator`` closure defined inside ``scan_db`` should resolve to ``scan_db``,
    # not ``scan_db.<locals>.source_generator``.
    wrapped = wrap_io_source_with_profiling(_module_level_reader(), schema={"value": pl.Int64})
    list(wrapped(None, None, None, None))

    assert _span().attributes["polars_io_tools.explain_name"] == "_module_level_reader"


def test_pure_self_join_emits_one_physical_execution():
    calls = 0

    def source(with_columns, predicate, n_rows, batch_size):
        nonlocal calls
        calls += 1
        frame = pl.DataFrame({"id": [1, 2], "value": [10, 20]})
        if with_columns is not None:
            frame = frame.select(with_columns)
        yield frame

    lf = register_io_source_with_is_pure(
        source,
        schema={"id": pl.Int64, "value": pl.Int64},
        explain_name="memory.reader",
    )
    result = lf.join(lf, on="id").collect()

    assert result.height == 2
    assert calls == 1
    span = _span()
    assert span.attributes["polars_io_tools.explain_name"] == "memory.reader"
    assert span.attributes["polars_io_tools.outcome"] == "exhausted"


def _capture_registered_callback(source, monkeypatch, **kwargs):
    # Assemble the callback exactly as register_io_source_with_is_pure hands it to polars, so tests exercise the real
    # wrapper composition (profiling + error-catching) rather than profile_io_source_iterator in isolation.
    captured = {}

    def fake_register(io_source, *, schema, **_kw):
        captured["io_source"] = io_source
        return object()

    monkeypatch.setattr("polars.io.plugins.register_io_source", fake_register)
    register_io_source_with_is_pure(source, schema={"value": pl.Int64}, **kwargs)
    return captured["io_source"]


def test_registration_excludes_eager_work_from_pull_time(monkeypatch):
    # With error-catching enabled (the default), eager work the source does before returning its iterator must still be
    # excluded from pull time -- i.e. profiling wraps the original source, not the error-catching generator.
    def source(*_args, **_kwargs):
        time.sleep(0.05)
        return iter([pl.DataFrame({"value": [1]})])

    callback = _capture_registered_callback(source, monkeypatch)
    list(callback(None, None, None, None))

    assert _span().attributes["polars_io_tools.next_elapsed_total_ms"] < 40


def test_registration_closes_real_iterator_on_exhaustion(monkeypatch):
    # Profiling wraps the original source, so its close() reaches the real iterator on exhaustion even under the
    # error-catching wrapper (whose yield-from would not close a custom iterator on normal completion).
    class RealIter:
        def __init__(self):
            self.closed = False
            self.remaining = 2

        def __iter__(self):
            return self

        def __next__(self):
            if self.remaining <= 0:
                raise StopIteration
            self.remaining -= 1
            return pl.DataFrame({"value": [1]})

        def close(self):
            self.closed = True

    iterator = RealIter()
    callback = _capture_registered_callback(lambda *_a, **_k: iterator, monkeypatch)
    list(callback(None, None, None, None))

    assert iterator.closed


def test_registration_emits_error_span_for_eager_factory_failure(monkeypatch):
    # A non-generator source can raise while building its iterator, before the timed generator exists; that failure must
    # still produce an ``error`` span with ``error.type``.
    def source(*_args, **_kwargs):
        raise ConnectionError("connect failed")

    callback = _capture_registered_callback(source, monkeypatch, wrap_with_error_catching=False)
    with pytest.raises(ConnectionError):
        list(callback(None, None, None, None))

    span = _span()
    assert span.attributes["polars_io_tools.outcome"] == "error"
    assert span.attributes["error.type"] == "ConnectionError"


def test_instrumentation_opt_out_disables_spans_with_zero_overhead(monkeypatch):
    # The OTEL_PYTHON_INSTRUMENTATION_<LIB>_ENABLED opt-out: when disabled the profiling wrapper is not installed, so no
    # span is emitted and the query still runs unchanged.
    from polars_io_tools.io_sources import profiling

    monkeypatch.setenv("OTEL_PYTHON_INSTRUMENTATION_POLARS_IO_TOOLS_ENABLED", "false")
    profiling._instrumentation_enabled.cache_clear()

    def source(with_columns, predicate, n_rows, batch_size):
        yield pl.DataFrame({"id": [1, 2, 3]})

    try:
        result = register_io_source_with_is_pure(source, schema={"id": pl.Int64}, explain_name="scan_x").collect(engine="streaming")
    finally:
        profiling._instrumentation_enabled.cache_clear()

    assert result.height == 3
    assert _spans() == []


def test_registration_wraps_source_and_forwards_kwargs(monkeypatch):
    # Instrumentation is always installed, so the callback handed to register_io_source is the profiling wrapper, not the
    # original; schema and pass-through kwargs are preserved.
    captured = {}
    sentinel = object()

    def fake_register(io_source, *, schema, **kwargs):
        captured["io_source"] = io_source
        captured["schema"] = schema
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr("polars.io.plugins.register_io_source", fake_register)

    def source(with_columns, predicate, n_rows, batch_size):
        return iter([pl.DataFrame({"value": [1]})])

    result = register_io_source_with_is_pure(
        source,
        schema={"value": pl.Int64},
        wrap_with_error_catching=False,
        explain_name="ignored.reader",
        validate_schema=True,
    )

    assert result is sentinel
    assert captured["io_source"] is not source
    assert captured["schema"] == {"value": pl.Int64}
    assert captured["kwargs"]["validate_schema"] is True


def test_clean_exit_propagates_inner_close_error():
    class ClosingRaises:
        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

        def close(self):
            raise RuntimeError("close failed")

    with pytest.raises(RuntimeError, match="close failed"):
        list(_profile(ClosingRaises()))

    # The span is still emitted (before closing), with a clean outcome.
    assert _span().attributes["polars_io_tools.outcome"] == "exhausted"


def test_active_pull_error_not_masked_by_inner_close_error():
    class BothRaise:
        def __iter__(self):
            return self

        def __next__(self):
            raise ValueError("pull failed")

        def close(self):
            raise RuntimeError("close failed")

    with pytest.raises(ValueError, match="pull failed"):
        next(_profile(BothRaise()))

    assert _span().attributes["polars_io_tools.outcome"] == "error"


def test_registration_resolves_identity_from_original_callable():
    # Error-catching wraps the source in a generic closure; the profiling identity must come from the original callable.

    class AlphaSource:
        def __call__(self, with_columns, predicate, n_rows, batch_size):
            frame = pl.DataFrame({"id": [1, 2]})
            yield frame.select(with_columns) if with_columns else frame

    class BetaSource:
        def __call__(self, with_columns, predicate, n_rows, batch_size):
            frame = pl.DataFrame({"id": [3, 4]})
            yield frame.select(with_columns) if with_columns else frame

    for source in (AlphaSource(), BetaSource()):
        register_io_source_with_is_pure(source, schema={"id": pl.Int64}).collect(engine="streaming")

    names = {span.attributes["polars_io_tools.explain_name"] for span in _spans()}
    assert any(name.endswith("AlphaSource") for name in names)
    assert any(name.endswith("BetaSource") for name in names)
    assert not any("error_catching" in name for name in names)


def test_distinct_source_names_produce_distinct_span_identities():
    # Two registrations of the same reader with different identities must emit distinctly named spans, carrying the
    # ``explain_name`` / ``explain_detail`` they are registered under.

    def make_reader():
        def _reader(with_columns, predicate, n_rows, batch_size):
            frame = pl.DataFrame({"id": [1, 2]})
            yield frame.select(with_columns) if with_columns else frame

        return _reader

    for collection in ("trades", "quotes"):
        register_io_source_with_is_pure(
            make_reader(),
            schema={"id": pl.Int64},
            explain_name=f"tickstore.{collection}",
            explain_detail=f"collection={collection}",
        ).collect(engine="streaming")

    spans = {span.name: span for span in _spans()}
    assert set(spans) == {"io_source.execute[tickstore.trades]", "io_source.execute[tickstore.quotes]"}
    for collection in ("trades", "quotes"):
        span = spans[f"io_source.execute[tickstore.{collection}]"]
        assert span.attributes["polars_io_tools.explain_name"] == f"tickstore.{collection}"
        assert span.attributes["polars_io_tools.explain_detail"] == f"collection={collection}"


def test_explain_labels_forwarded_to_register_io_source_when_supported(monkeypatch):
    # When the Polars build exposes explain_name/explain_detail, the resolved identity and detail are forwarded to the
    # scan so its node label matches the profiling span. Feature-detected against register_io_source's signature.
    from polars_io_tools.io_sources import util

    captured = {}

    def fake_register(io_source, *, schema, explain_name=None, explain_detail=None, **kwargs):
        captured["explain_name"] = explain_name
        captured["explain_detail"] = explain_detail
        return object()

    monkeypatch.setattr("polars.io.plugins.register_io_source", fake_register)
    util._register_io_source_supports_explain_labels.cache_clear()

    def source(with_columns, predicate, n_rows, batch_size):
        yield pl.DataFrame({"value": [1]})

    try:
        register_io_source_with_is_pure(
            source,
            schema={"value": pl.Int64},
            wrap_with_error_catching=False,
            explain_name="tickstore.trades",
            explain_detail="collection=trades",
        )
    finally:
        util._register_io_source_supports_explain_labels.cache_clear()

    assert captured["explain_name"] == "tickstore.trades"
    assert captured["explain_detail"] == "collection=trades"
