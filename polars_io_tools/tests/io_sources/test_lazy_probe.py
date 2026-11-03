"""LazyFrame.piot.probe: a pass-through that emits a profiling span."""

from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass

import polars as pl
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from polars.testing import assert_frame_equal

import polars_io_tools  # noqa: F401  -- registers the `.piot` namespace
from polars_io_tools.io_sources.util import register_io_source_with_is_pure
from polars_io_tools.tests.helpers.cases import as_params

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


@dataclass(frozen=True)
class ProbeCase:
    """A frame, probe options and a query; the probe passes the rows through and emits one span."""

    id: str
    _: KW_ONLY
    df: pl.DataFrame
    # keyword arguments for ``probe``
    kw: dict
    op: Callable[[pl.LazyFrame], pl.LazyFrame]
    # span attribute -> its value
    attrs: dict


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            ProbeCase(
                "probe_is_result_preserving_passthrough",
                df=pl.DataFrame({"id": [1, 2, 3], "v": [10, 20, 30]}),
                kw={},
                op=lambda lf: lf,
                attrs={"polars_io_tools.total_rows": 3},
            ),
            ProbeCase(
                "probe_forwards_n_rows_pushdown",
                df=pl.DataFrame({"id": list(range(100))}),
                kw={},
                op=lambda lf: lf.head(5),
                attrs={"polars_io_tools.total_rows": 5},
            ),
            # a pushed predicate reaches the probe, so only matching rows flow through it
            ProbeCase(
                "probe_forwards_predicate_pushdown",
                df=pl.DataFrame({"id": [1, 2, 3, 4], "grp": [0, 1, 0, 1]}),
                kw={},
                op=lambda lf: lf.filter(pl.col("grp") == 1),
                attrs={"polars_io_tools.total_rows": 2},
            ),
            ProbeCase(
                "probe_emits_one_span_with_description",
                df=pl.DataFrame({"id": [1, 2, 3]}),
                kw={"description": "stage1"},
                op=lambda lf: lf,
                attrs={"polars_io_tools.explain_name": "probe", "polars_io_tools.explain_detail": "stage1", "polars_io_tools.total_rows": 3},
            ),
        ]
    ),
)
def test_probe(case: ProbeCase):
    """A probe passes the rows through, Polars' pushdowns included, and emits one span counting them."""
    got = case.op(case.df.lazy().piot.probe(**case.kw)).collect(engine="streaming")
    assert_frame_equal(got, case.op(case.df.lazy()).collect(), check_row_order=False)
    (span,) = [s for s in _EXPORTER.get_finished_spans() if s.name.startswith("io_source.execute[probe]")]
    assert {k: span.attributes[k] for k in case.attrs} == case.attrs


def test_probe_does_not_dedupe_impure_input():
    # A pass-through cannot claim purity: reusing the same probe over an impure input must not collapse executions.
    state = {"n": 0}

    def impure(with_columns, predicate, n_rows, batch_size):
        state["n"] += 1
        yield pl.DataFrame({"run": [state["n"]]})

    probe = register_io_source_with_is_pure(impure, schema=pl.Schema({"run": pl.Int64}), is_pure=False).piot.probe()
    out = pl.concat([probe, probe]).collect(engine="streaming")
    assert sorted(out["run"].to_list()) == [1, 2]


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
