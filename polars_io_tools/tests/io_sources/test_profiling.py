"""Profiling spans around IO sources.

The main table drives an iterator (bare, wrapped, or as registered with Polars) and checks the one span it emits.
"""

import contextlib
import time
from collections.abc import Iterator

import polars as pl
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from polars_io_tools._compat import POLARS_GE_2
from polars_io_tools.io_sources.profiling import (
    get_source_span_parent,
    profile_io_source_iterator,
    set_source_span_parent,
    wrap_io_source_with_profiling,
)
from polars_io_tools.io_sources.util import register_io_source_with_is_pure
from polars_io_tools.tests.helpers.cases import Case, params

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


# Iterators driven by a consumer -> the one span emitted: its attributes, and what the consumer sees.
class Iter:
    """A hand-written iterator: ``batches`` frames, then ``stop`` (StopIteration or an exception), each step after
    ``sleep`` seconds; ``close`` records the call and raises ``close_error`` if given."""

    def __init__(self, batches=(), stop=StopIteration, sleep=0.0, sleep_before_stop=0.0, close_error=None, endless=False):
        self.batches, self.stop, self.sleep, self.sleep_before_stop = list(batches), stop, sleep, sleep_before_stop
        self.close_error, self.endless, self.closed = close_error, endless, False

    def __iter__(self):
        return self

    def __next__(self):
        if self.endless:
            return pl.DataFrame({"value": [1]})
        time.sleep(self.sleep)
        if self.batches:
            return self.batches.pop(0)
        time.sleep(self.sleep_before_stop)
        raise self.stop

    def close(self):
        self.closed = True
        if self.close_error:
            raise self.close_error


def _one(*values):
    return [pl.DataFrame({"value": [v]}) for v in values]


def _wrapped(source, schema=None, explain_name="empty.reader", with_columns=None):
    return lambda monkeypatch: (
        wrap_io_source_with_profiling(source, schema=schema or {"a": pl.Int64, "b": pl.Int64}, explain_name=explain_name)(
            with_columns, None, None, None
        ),
        None,
    )


def _profiled(inner):
    return lambda monkeypatch: (_profile(inner()), None)


def _profiled_inner(inner):
    def start(monkeypatch):
        it = inner()
        return _profile(it), it

    return start


def _eager_sleeper(*_args, **_kwargs):
    time.sleep(0.05)
    return iter(_one(1))


class CallableSource:
    def __call__(self, *_args, **_kwargs):
        return iter(_one(1))


def _registered(source, **kwargs):
    """The callback register_io_source_with_is_pure hands to Polars, so the real wrapper composition is exercised."""

    def start(monkeypatch):
        captured = {}

        def fake_register(io_source, *, schema, **_kw):
            captured["io_source"] = io_source
            return object()

        monkeypatch.setattr("polars.io.plugins.register_io_source", fake_register)
        inner = source()
        register_io_source_with_is_pure(inner if callable(inner) else (lambda *a, **k: inner), schema={"value": pl.Int64}, **kwargs)

        def pull():  # call the callback inside the consumer, where a failing factory raises
            yield from captured["io_source"](None, None, None, None)

        return pull(), inner

    return start


def _raise_connection(*_args, **_kwargs):
    raise ConnectionError("connect failed")


_ERR_VALUE, _ERR_LOOKUP = ValueError("source failed"), LookupError("failed after a batch")
EXHAUSTED, CLOSED, ERROR = ({"polars_io_tools.outcome": o} for o in ("exhausted", "closed", "error"))
ELAPSED = "polars_io_tools.next_elapsed_total_ms"
ABSENT = object()


def _at_least(ms):
    return lambda v: v >= ms


def _below(ms):
    return lambda v: v < ms


def _stats(rows, batches, columns=1):
    return {"polars_io_tools.total_rows": rows, "polars_io_tools.batch_count": batches, "polars_io_tools.n_columns": columns}


def S(id, start, attrs, drive="list", raises=None, closed=None):
    """``drive``: how the consumer pulls — "list"; "next" (one pull); "pull_then_fail" (one pull, then the next
    raises); "sleepy" (sleeps between pulls); "close" / "pull_close" (closes twice, after no or one pull).
    ``raises``: the exception (class, or the very object) the consumer sees. ``closed``: whether the inner iterator
    was closed. ``attrs`` values may be predicates; ABSENT means the attribute is not set."""
    return Case(id, start, attrs, {"drive": drive, "raises": raises, "closed": closed})


SPANS = [
    S(
        "multi_batch_accumulates_only_pull_time",
        _profiled(lambda: Iter(_one(0, 1, 2), sleep=0.015)),
        {**EXHAUSTED, **_stats(3, 3), ELAPSED: _at_least(3 * 15 * 0.8)},
    ),
    S(
        "slow_terminal_stop_iteration_is_counted",
        _profiled(lambda: Iter(_one(1), sleep_before_stop=0.025)),
        {**EXHAUSTED, ELAPSED: _at_least(25 * 0.8)},
    ),
    S(
        "failure_before_first_yield_is_counted_and_propagated_unchanged",
        _profiled(lambda: Iter(stop=_ERR_VALUE, sleep=0.025)),
        {**ERROR, "polars_io_tools.total_rows": 0, "polars_io_tools.batch_count": 0, "error.type": "ValueError", ELAPSED: _at_least(25 * 0.8)},
        "next",
        _ERR_VALUE,
    ),
    S(
        "failure_reports_partial_rows",
        _profiled(lambda: Iter([pl.DataFrame({"value": [1, 2]})], stop=_ERR_LOOKUP, sleep_before_stop=0.015)),
        {**ERROR, **_stats(2, 1), ELAPSED: _at_least(10)},
        "pull_then_fail",
        _ERR_LOOKUP,
    ),
    S("consumer_sleep_between_pulls_is_excluded", _profiled(lambda: iter(_one(1, 2))), {ELAPSED: _below(50)}, "sleepy"),
    S(
        "early_close_propagates_to_custom_iterator_and_emits_once-False",
        _profiled_inner(lambda: Iter(endless=True)),
        {**CLOSED, "polars_io_tools.total_rows": 0},
        "close",
        closed=True,
    ),
    S(
        "early_close_propagates_to_custom_iterator_and_emits_once-True",
        _profiled_inner(lambda: Iter(endless=True)),
        {**CLOSED, "polars_io_tools.total_rows": 1},
        "pull_close",
        closed=True,
    ),
    S("exactly_once_emission_for_all_outcomes-exhausted", _profiled(lambda: iter(())), EXHAUSTED),
    S("exactly_once_emission_for_all_outcomes-closed", _profiled(lambda: iter(_one(1))), CLOSED, "pull_close"),
    S("exactly_once_emission_for_all_outcomes-error", _profiled(lambda: Iter(stop=ZeroDivisionError("boom"))), ERROR, "next", ZeroDivisionError),
    S("empty_first_batch_sets_width_and_zero_rows", _profiled(lambda: iter([pl.DataFrame(schema={"a": pl.Int64, "b": pl.String})])), _stats(0, 1, 2)),
    # the span is emitted before closing, with a clean outcome
    S("clean_exit_propagates_inner_close_error", _profiled(lambda: Iter(close_error=RuntimeError("close failed"))), EXHAUSTED, raises=RuntimeError),
    S(
        "active_pull_error_not_masked_by_inner_close_error",
        _profiled(lambda: Iter(stop=ValueError("pull failed"), close_error=RuntimeError("close failed"))),
        ERROR,
        "next",
        ValueError,
    ),
    # with no batch the width comes from the projection, else from an eager schema
    S(
        "no_yield_uses_known_projection_or_eager_schema_width-projection",
        _wrapped(lambda *a, **k: iter(()), with_columns=["a"]),
        {"polars_io_tools.n_columns": 1},
    ),
    S("no_yield_uses_known_projection_or_eager_schema_width-schema", _wrapped(lambda *a, **k: iter(())), {"polars_io_tools.n_columns": 2}),
    S("eager_callable_work_is_excluded_from_pull_time", _wrapped(_eager_sleeper, schema={"value": pl.Int64}), {ELAPSED: _below(30)}),
    # a source's name is stable for callable objects and trimmed to the enclosing function for closures
    S(
        "default_source_identity_is_stable_for_callable_objects",
        _wrapped(CallableSource(), schema={"value": pl.Int64}, explain_name=None),
        {"polars_io_tools.explain_name": lambda v: v.endswith("CallableSource") and "0x" not in v},
    ),
    S(
        "default_source_identity_trims_closure_to_enclosing_function",
        _wrapped(_module_level_reader(), schema={"value": pl.Int64}, explain_name=None),
        {"polars_io_tools.explain_name": "_module_level_reader"},
    ),
    # profiling wraps the original source inside the error-catching wrapper: eager work is excluded, a custom
    # iterator is closed on exhaustion, and a failing factory still emits an error span
    S("registration_excludes_eager_work_from_pull_time", _registered(lambda: _eager_sleeper), {ELAPSED: _below(40)}),
    S("registration_closes_real_iterator_on_exhaustion", _registered(lambda: Iter(_one(1, 1))), EXHAUSTED, closed=True),
    S(
        "registration_emits_error_span_for_eager_factory_failure",
        _registered(lambda: _raise_connection, wrap_with_error_catching=False),
        {**ERROR, "error.type": "ConnectionError"},
        raises=ConnectionError,
    ),
]


@pytest.mark.parametrize(("start", "attrs", "drive", "raises", "closed"), params(SPANS, "pred", "expect", "drive", "raises", "closed"))
def test_span(monkeypatch, start, attrs, drive, raises, closed):
    iterator, inner = start(monkeypatch)
    expected_error = pytest.raises(raises if isinstance(raises, type) else type(raises)) if raises is not None else contextlib.nullcontext()
    with expected_error as error:
        if drive == "list":
            list(iterator)
        elif drive == "next":
            next(iterator)
        elif drive == "pull_then_fail":
            assert next(iterator).height == 2
            next(iterator)
        elif drive == "sleepy":
            next(iterator)
            time.sleep(0.1)
            next(iterator)
            with pytest.raises(StopIteration):
                next(iterator)
        else:
            if drive == "pull_close":
                next(iterator)
            iterator.close()
            iterator.close()
    if raises is not None and not isinstance(raises, type):
        assert error.value is raises
    span = _span()
    for key, want in attrs.items():
        if want is ABSENT:
            assert key not in span.attributes
        elif callable(want):
            assert want(span.attributes[key]), (key, span.attributes[key])
        else:
            assert span.attributes[key] == want, key
    if closed is not None:
        assert inner.closed == closed


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


@pytest.mark.parametrize(
    "engine",
    ["auto", "streaming", pytest.param("in-memory", marks=pytest.mark.skipif(not POLARS_GE_2, reason="Polars 1 uses auto for in-memory execution"))],
)
def test_pure_self_join_profiles_physical_executions(engine):
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
    result = lf.join(lf, on="id").collect(engine=engine)

    assert result.sort("id").to_dict(as_series=False) == {"id": [1, 2], "value": [10, 20], "value_right": [10, 20]}
    if engine == ("in-memory" if POLARS_GE_2 else "auto"):
        assert calls == 1
    spans = _EXPORTER.get_finished_spans()
    assert calls >= 1
    assert len(spans) == calls
    assert all(span.attributes["polars_io_tools.explain_name"] == "memory.reader" for span in spans)
    assert all(span.attributes["polars_io_tools.outcome"] == "exhausted" for span in spans)


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
