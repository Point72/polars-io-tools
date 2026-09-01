import polars as pl
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from polars.testing import assert_frame_equal

import polars_io_tools  # noqa: F401  -- registers the `.piot` namespace
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
    yield


def _probe_spans():
    return [s for s in _EXPORTER.get_finished_spans() if s.name.startswith("io_source.execute[probe]")]


def test_probe_is_result_preserving_passthrough():
    df = pl.DataFrame({"id": [1, 2, 3], "v": [10, 20, 30]})
    out = df.lazy().piot.probe().collect(engine="streaming")
    assert_frame_equal(out, df)


def test_probe_forwards_n_rows_pushdown():
    out = pl.LazyFrame({"id": list(range(100))}).piot.probe().head(5).collect(engine="streaming")
    assert out["id"].to_list() == [0, 1, 2, 3, 4]
    assert _probe_spans()[-1].attributes["polars_io_tools.total_rows"] == 5


def test_probe_does_not_dedupe_impure_input():
    # A pass-through cannot claim purity: reusing the same probe over an impure input must not collapse executions.
    state = {"n": 0}

    def impure(with_columns, predicate, n_rows, batch_size):
        state["n"] += 1
        yield pl.DataFrame({"run": [state["n"]]})

    probe = register_io_source_with_is_pure(impure, schema=pl.Schema({"run": pl.Int64}), is_pure=False).piot.probe()
    out = pl.concat([probe, probe]).collect(engine="streaming")
    assert sorted(out["run"].to_list()) == [1, 2]


def test_probe_emits_one_span_with_description():
    df = pl.DataFrame({"id": [1, 2, 3]})
    df.lazy().piot.probe(description="stage1").collect(engine="streaming")

    spans = _probe_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["polars_io_tools.explain_name"] == "probe"
    assert span.attributes["polars_io_tools.explain_detail"] == "stage1"
    assert span.attributes["polars_io_tools.total_rows"] == 3


def test_probe_forwards_predicate_pushdown():
    # A pushed predicate reaches the probe, so only matching rows flow through it (and are counted).
    df = pl.DataFrame({"id": [1, 2, 3, 4], "grp": [0, 1, 0, 1]})
    out = df.lazy().piot.probe().filter(pl.col("grp") == 1).collect(engine="streaming")

    assert out.sort("id")["id"].to_list() == [2, 4]
    assert _probe_spans()[-1].attributes["polars_io_tools.total_rows"] == 2


def test_probe_forwards_is_pure(monkeypatch):
    # is_pure defaults to False (safe for an input of unknown purity) and can be opted into for a known-pure input.
    captured = {}

    def fake_register(io_source, *, schema, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("polars.io.plugins.register_io_source", fake_register)

    pl.LazyFrame({"a": [1]}).piot.probe()
    assert captured["is_pure"] is False

    captured.clear()
    pl.LazyFrame({"a": [1]}).piot.probe(is_pure=True)
    assert captured["is_pure"] is True
