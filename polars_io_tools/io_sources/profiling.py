"""Automatic OpenTelemetry profiling for registered IO sources.

Each physical iterator execution is measured for its *pull latency* -- ``next_elapsed_total``, the summed wall duration of
every ``next()`` call on the source iterator. This is the fetch cost a streaming engine's own per-node metrics do not see.
It is caller-observed: it excludes work a driver performs on a background thread while the iterator is suspended (e.g.
arrow-odbc's ``fetch_concurrently``), and includes Arrow-to-Polars conversion and any pushed predicate/projection the
source applies.

Sources registered through ``register_io_source_with_is_pure`` are instrumented automatically. Each execution is emitted
as one OpenTelemetry span through the OpenTelemetry API, which is a no-op unless the application has configured an SDK
(mirroring how native library instrumentation behaves); export is delegated to the application's span processor, so use a
batching processor to keep it off the Polars worker. Span attributes are namespaced under ``polars_io_tools.``.

OpenTelemetry is an optional, undeclared dependency: when the ``opentelemetry`` API is not installed, instrumentation is
a no-op. Any application that has configured an OpenTelemetry SDK already provides the API transitively, so no extra
install is required to capture spans.

Instrumentation is on by default and can be disabled -- e.g. to keep OpenTelemetry for the rest of an application while
suppressing these spans -- by setting ``OTEL_PYTHON_INSTRUMENTATION_POLARS_IO_TOOLS_ENABLED=false``, following the
OpenTelemetry Python ``OTEL_PYTHON_INSTRUMENTATION_<LIBRARY>_ENABLED`` opt-out convention. When disabled the wrapper is
not installed at all, so there is zero overhead.
"""

from __future__ import annotations

import functools
import os
import sys
import time
from collections.abc import Callable, Generator, Iterable, Mapping
from typing import Any, Literal, TypeAlias

try:
    from opentelemetry.context import Context

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised without OpenTelemetry installed
    Context: TypeAlias = Any
    _OTEL_AVAILABLE = False

Outcome: TypeAlias = Literal["exhausted", "closed", "error"]

_INSTRUMENTATION_NAME = "polars_io_tools"
_ATTR_PREFIX = f"{_INSTRUMENTATION_NAME}."

_ENABLED_ENV_VAR = "OTEL_PYTHON_INSTRUMENTATION_POLARS_IO_TOOLS_ENABLED"


@functools.cache
def _instrumentation_enabled() -> bool:
    """Whether IO-source instrumentation is active.

    On by default (the native-instrumentation convention); disable it by setting the environment variable
    ``OTEL_PYTHON_INSTRUMENTATION_POLARS_IO_TOOLS_ENABLED=false``. This follows the OpenTelemetry Python
    ``OTEL_PYTHON_INSTRUMENTATION_<LIBRARY>_ENABLED`` opt-out convention, letting an application keep OpenTelemetry for
    the rest of its code while suppressing these spans with zero overhead.
    """
    return os.environ.get(_ENABLED_ENV_VAR, "").strip().lower() != "false"


@functools.cache
def _instrumentation_version() -> str:
    """Return the installed ``polars_io_tools`` version for the tracer's instrumentation scope."""
    try:
        from polars_io_tools import __version__
    except ImportError:  # pragma: no cover - version metadata should always be importable
        return ""
    return __version__


__all__ = (
    "get_source_span_parent",
    "profile_io_source_iterator",
    "set_source_span_parent",
    "source_identity",
    "wrap_io_source_with_profiling",
)


_span_parent: Context | None = None


def set_source_span_parent(ctx: Context | None) -> None:
    """Publish a parent context so source spans nest under it.

    A single process-global, intended for sequential collects; concurrent collects make the parent association ambiguous.
    """
    global _span_parent
    _span_parent = ctx


def get_source_span_parent() -> Context | None:
    """Return the currently published parent context, if any."""
    return _span_parent


def _emit_span(
    *,
    explain_name: str,
    explain_detail: str | None,
    elapsed_ns: int,
    total_rows: int,
    n_columns: int | None,
    batch_count: int,
    outcome: Outcome,
    error_type: str | None,
    start_ns: int,
    end_ns: int,
    parent: Context | None,
) -> None:
    """Emit one ``io_source.execute[<name>]`` span; a no-op without OpenTelemetry, and never raises.

    The span is created inline via the OpenTelemetry API, which is a no-op unless the application configured an SDK;
    export is delegated to the application's span processor, so use a batching processor (e.g. ``BatchSpanProcessor``,
    which Logfire installs by default) to keep it off the Polars worker thread. Telemetry must never affect the query, so
    any error here is swallowed.
    """
    if not _OTEL_AVAILABLE:
        return
    try:
        from opentelemetry import trace

        # Name the span ``io_source.execute[<explain_name>]`` and attach ``explain_name`` so the span carries the
        # identity the source is registered under. ``explain_name`` is expected to be low cardinality (the source kind,
        # e.g. ``scan_db``); per-instance variation belongs in ``explain_detail``.
        span = trace.get_tracer(_INSTRUMENTATION_NAME, _instrumentation_version()).start_span(
            f"io_source.execute[{explain_name}]", context=parent, start_time=start_ns
        )
        if span.is_recording():
            attributes: dict[str, Any] = {
                f"{_ATTR_PREFIX}explain_name": explain_name,
                f"{_ATTR_PREFIX}next_elapsed_total_ms": elapsed_ns / 1_000_000,
                f"{_ATTR_PREFIX}total_rows": total_rows,
                f"{_ATTR_PREFIX}batch_count": batch_count,
                f"{_ATTR_PREFIX}outcome": outcome,
            }
            if explain_detail is not None:
                attributes[f"{_ATTR_PREFIX}explain_detail"] = explain_detail
            if n_columns is not None:
                attributes[f"{_ATTR_PREFIX}n_columns"] = n_columns
            if error_type is not None:
                attributes["error.type"] = error_type  # OTel semantic convention
            span.set_attributes(attributes)
            if outcome == "error":
                from opentelemetry.trace import Status, StatusCode

                span.set_status(Status(StatusCode.ERROR))
        span.end(end_time=end_ns)
    except BaseException:  # noqa: BLE001, S110 - telemetry must never affect the query
        pass


def _close_inner(inner: Any) -> None:
    """Close the inner iterator, surfacing a cleanup failure only on a clean exit.

    When the wrapper is already unwinding an exception (a failing pull or a ``GeneratorExit`` from cancellation), a
    ``close()`` error is swallowed so it cannot mask the active exception; on a clean exit it propagates, matching the
    unwrapped iterator's behaviour.
    """
    close = getattr(inner, "close", None)
    if close is None:
        return
    if sys.exc_info()[0] is None:
        close()
    else:
        try:
            close()
        except BaseException:  # noqa: BLE001, S110 - do not mask the active exception
            pass


def profile_io_source_iterator(
    inner_iterable: Iterable[Any],
    *,
    explain_name: str,
    explain_detail: str | None = None,
    fallback_n_columns: int | None = None,
) -> Generator[Any, None, None]:
    """Wrap one iterator execution in a self-measuring, close-propagating generator.

    The returned generator yields the inner batches unchanged and emits exactly one ``io_source.execute`` span when it
    finishes (``exhausted`` / ``closed`` / ``error``). ``explain_name`` names the span and is attached as an attribute;
    ``explain_detail`` is an optional free-form per-instance description. ``total_rows`` counts rows yielded and may
    exceed a pushed ``n_rows`` when the final batch overshoots it.
    """
    inner = iter(inner_iterable)

    def timed() -> Generator[Any, None, None]:
        elapsed_ns = total_rows = batch_count = 0
        n_columns: int | None = None
        outcome: Outcome = "exhausted"
        error_type: str | None = None
        wall_start_ns: int | None = None
        try:
            # Prime point: suspending inside the try means a close() before the
            # first real pull still runs the finally (emit + propagate close).
            yield
            while True:
                if wall_start_ns is None:
                    wall_start_ns = time.time_ns()
                pull_start = time.perf_counter_ns()
                try:
                    batch = next(inner)
                except StopIteration:
                    elapsed_ns += time.perf_counter_ns() - pull_start
                    break
                except BaseException:
                    elapsed_ns += time.perf_counter_ns() - pull_start
                    outcome = "error"
                    raise
                elapsed_ns += time.perf_counter_ns() - pull_start
                total_rows += batch.height
                if n_columns is None:
                    n_columns = batch.width
                batch_count += 1
                yield batch
        except GeneratorExit:
            outcome = "closed"
            raise
        except BaseException as exc:
            outcome = "error"
            error_type = type(exc).__name__
            raise
        finally:
            now_ns = time.time_ns()
            # Emit before closing so telemetry is recorded regardless of whether the inner ``close()`` succeeds. The
            # parent context is read here (emit time): a consumer publishes it after the iterator is built, so capturing
            # earlier races it.
            _emit_span(
                explain_name=explain_name,
                explain_detail=explain_detail,
                elapsed_ns=elapsed_ns,
                total_rows=total_rows,
                n_columns=n_columns if n_columns is not None else fallback_n_columns,
                batch_count=batch_count,
                outcome=outcome,
                error_type=error_type,
                start_ns=wall_start_ns if wall_start_ns is not None else now_ns,
                end_ns=now_ns,
                parent=get_source_span_parent(),
            )
            _close_inner(inner)

    generator = timed()
    next(generator)  # advance to the prime point
    return generator


def source_identity(io_source: Callable[..., Any], explain_name: str | None = None) -> str:
    """Resolve the ``explain_name`` for a source callable.

    Defaults to the callable's qualified name (or its class for callable instances), with the local-closure suffix
    trimmed: a source defined as ``source_generator`` inside ``scan_db`` resolves to ``scan_db``, the enclosing function
    that names the source kind, rather than ``scan_db.<locals>.source_generator``. Resolve this from the *original*
    callable, before any wrapping that would mask its identity.
    """
    if explain_name is not None:
        return explain_name
    for attribute in ("__qualname__", "__name__"):
        value = getattr(io_source, attribute, None)
        if isinstance(value, str) and value:
            # Drop the inner-closure boilerplate: the enclosing function is the meaningful, low-cardinality identity.
            return value.split(".<locals>.", 1)[0]
    cls = type(io_source)
    return f"{cls.__module__}.{cls.__qualname__}"


def _fallback_n_columns(args: tuple[Any, ...], kwargs: dict[str, Any], schema: Any) -> int | None:
    with_columns = kwargs.get("with_columns", args[0] if args else None)
    if with_columns is not None:
        return len(with_columns)
    if isinstance(schema, Mapping):
        return len(schema)
    return None


def wrap_io_source_with_profiling(
    io_source: Callable[..., Iterable[Any]],
    *,
    schema: Any,
    explain_name: str | None = None,
    explain_detail: str | None = None,
) -> Callable[..., Generator[Any, None, None]]:
    """Wrap an IO-source callable so each execution emits one ``io_source.execute`` span.

    Work performed eagerly by ``io_source`` before it returns its iterator is outside the pull-time measurement, but a
    failure during that eager step still emits an ``error`` span; generator-body work is captured on the first pull.
    ``explain_name`` names the span (defaults to the source function's name); ``explain_detail`` is an optional free-form
    per-instance description.
    """
    name = source_identity(io_source, explain_name)

    @functools.wraps(io_source)
    def profiled(*args: Any, **kwargs: Any) -> Generator[Any, None, None]:
        start_ns = time.time_ns()
        try:
            iterable = io_source(*args, **kwargs)
        except BaseException as exc:
            # A non-generator source can fail while eagerly building its iterator, before the timed generator exists.
            _emit_span(
                explain_name=name,
                explain_detail=explain_detail,
                elapsed_ns=0,
                total_rows=0,
                n_columns=_fallback_n_columns(args, kwargs, schema),
                batch_count=0,
                outcome="error",
                error_type=type(exc).__name__,
                start_ns=start_ns,
                end_ns=time.time_ns(),
                parent=get_source_span_parent(),
            )
            raise
        return profile_io_source_iterator(
            iterable,
            explain_name=name,
            explain_detail=explain_detail,
            fallback_n_columns=_fallback_n_columns(args, kwargs, schema),
        )

    return profiled
