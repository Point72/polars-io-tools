from collections.abc import Iterator

import polars as pl

from .util import collect_lf_in_io_source, register_io_source_with_is_pure

__all__ = ("probe",)


def probe(self: pl.LazyFrame, description: str | None = None) -> pl.LazyFrame:
    """A result-preserving pass-through source that emits one OpenTelemetry span per execution.

    Inserting ``.piot.probe()`` at a point in a lazy pipeline turns that point into a measured
    ``io_source.execute[probe]`` span: ``next_elapsed_total_ms`` is the time spent pulling the sub-plan
    below the probe and ``total_rows`` is the number of rows that flow through it. Predicate and
    projection pushdown are forwarded to the input, so the probe is not an optimization barrier for
    them. It does add a ``PythonScan`` node that re-enters Polars via ``collect_lf_in_io_source``, so it
    is a measurement point rather than a zero-cost tap.

    Unlike ``debug``, it performs no logging and does not compute ``explain()``, so it is cheap enough to
    leave in a pipeline. The span is a no-op unless an OpenTelemetry SDK is configured (see
    ``polars_io_tools.io_sources.profiling``).

    Overhead is on the order of milliseconds and grows with the number of rows passing through the probe
    (each batch makes a Rust-Python round trip); it is negligible relative to any non-trivial read but can
    dominate a small, fully-cached one. The overhead is the io-source boundary itself, not the telemetry.
    ``next_elapsed_total_ms`` is caller-observed pull latency, so it can undercount a reader that decodes
    on background threads.

    Args:
        self: The input LazyFrame to pass through unchanged.
        description: Optional free-form description of this probe instance, attached to its OpenTelemetry span (``explain_detail``) -- e.g. the pipeline stage being measured.

    Returns:
        pl.LazyFrame: A LazyFrame equivalent to ``self`` whose execution emits one telemetry span.
    """
    schema = self.collect_schema()

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        df = self
        if predicate is not None:
            df = df.filter(predicate)
        if with_columns is not None:
            df = df.select(with_columns)
        if n_rows is not None:
            df = df.head(n_rows)
        yield from collect_lf_in_io_source(df, batch_size)

    return register_io_source_with_is_pure(source_generator, schema=schema, validate_schema=False, explain_detail=description)
