"""In-memory, builder-driven cache for non-serializable LazyFrames.

:func:`cache_memory` is the third member of the caching family, complementing
:func:`~polars_io_tools.io_sources.lazy_cache.cache` and
:func:`~polars_io_tools.io_sources.lazy_cache_parquet.cache_parquet`:

======================  ==============================  ==============================  ============
function                input                           key model                       backend
======================  ==============================  ==============================  ============
``cache``               ``LazyFrame`` (plan)            serialize (content-addressed)   in-memory
``cache_parquet``       builder callable or plan        ``cache_path`` (explicit)       Parquet (disk)
``cache_memory``        builder callable or plan        none (instance / identity)      in-memory
======================  ==============================  ==============================  ============

Unlike ``cache``, which derives its key by serializing the plan, ``cache_memory``
holds the materialized buffer in the returned frame's own closure, so nothing is
serialized, hashed, or looked up. This makes it usable with ``register_io_source``
plugins whose generator closes over non-picklable state (a connection, a lock, an
open iterator) that ``cache`` cannot serialize.

The buffer is collected **once** on first row demand and replayed thereafter, so a
consumer that references the returned frame many times — within one query, or across
several collects — pays a single upstream execution. Polars' Common Subplan
Elimination does not deduplicate independent references to an IO-backed source, so a
plain lazy plan is otherwise re-executed once per reference. This is the primitive to
reach for during complex, multi-branch evaluations where an expensive or
non-serializable source is read from several branches and a Parquet round-trip
(``cache_parquet``) is unwanted.

Lifetime is bounded by the returned frame: the buffer is reachable only through it, so
ordinary garbage collection reclaims the memory when the frame (and any plan embedding
it) is dropped. There is no key, no eviction policy, and nothing to invalidate — a
fresh call builds a fresh buffer.

Thread-safe for the one-time materialization: a double-checked lock prevents concurrent
callers from running the builder twice. After materialization the buffered
``pl.DataFrame`` is immutable and safe to read from multiple threads.

Build-once, uniformly: the builder runs **exactly once** per instance whatever the outcome.
On success the buffer is served thereafter; on any failure — an ordinary exception (e.g. a
data-quality raise) or a control-flow ``BaseException`` such as ``KeyboardInterrupt`` — the
failure is recorded and re-raised (as a fresh neutral error carrying the original type name
and message) on every subsequent collect. So a fan-out across ``pl.collect_all`` sees one
failure, never one re-run of a failing builder per branch, and an interrupted build poisons
the instance rather than silently re-running. There is no in-place retry — retrying means
constructing a fresh ``cache_memory`` (a fresh call builds a fresh buffer), the same
instance-scoped invalidation model as the success path.
"""

import threading
from collections.abc import Callable, Iterator, Sequence

import polars as pl

from .restrict_visitor import restrict_expr_to_columns
from .util import PartitionKey, collect_lf_in_io_source, partition_exclusion_predicate, partition_key, register_io_source_with_is_pure

__all__ = ("cache_memory",)

# Sentinel returned by the build planner to mean "everything demanded is already
# materialized" — distinct from ``None``, which means "do a full (unrestricted) build".
_NOTHING = object()


def _exclude_built(built_keys: set[PartitionKey]) -> pl.Expr | None:
    """A predicate that excludes every already-built partition.

    Returns "not in any already-built partition" (see
    :func:`~polars_io_tools.io_sources.util.partition_exclusion_predicate`), or ``None`` when
    nothing has been built yet (no restriction needed). ``eq_missing`` semantics mean a
    null-valued partition key excludes exactly its own rows.
    """
    keys = [key for key in built_keys if key]
    if not keys:
        return None
    cols = [col for col, _ in keys[0]]
    # infer_schema_length=None scans every key so a partition column that is null in the first
    # keys but valued later still infers the correct dtype (otherwise Polars infers Null and
    # fails to append the later value).
    known = pl.DataFrame([dict(key) for key in keys], infer_schema_length=None)
    return partition_exclusion_predicate(known, cols)


class _CachedBuildError(Exception):
    """Raised on every collect after a ``cache_memory`` instance's one build attempt failed.

    The original failure (its type name and message) is recorded once and re-raised through a
    fresh instance of this neutral type on each subsequent collect. Storing the live exception
    would pin its traceback — and through it the whole failed frame — alive, and re-raising one
    shared object across collects would grow its traceback and cannot faithfully reconstruct an
    arbitrary exception type from its message.
    """


def cache_memory(
    self_or_fn: pl.LazyFrame | Callable[[], pl.LazyFrame],
    *,
    schema: pl.Schema | Callable[[], pl.Schema],
    partition_cols: str | Sequence[str] = (),
    description: str | None = None,
) -> pl.LazyFrame:
    """Collect a builder at most once into an in-memory buffer and replay it thereafter.

    Wrapping performs no I/O and no collection; the builder runs only on first row
    demand. With a callable ``schema``, even ``collect_schema()`` on the result avoids
    running the builder. The first row demand collects the builder into a buffer;
    subsequent demands (more references, more collects) replay it, so a consumer that
    references the returned frame N times pays one upstream execution total.

    Because it materializes the whole frame, downstream predicates and projections are
    applied to the buffer *after* materialization (they are not pushed into the upstream
    source). This is inherent to "collect once and reuse" and differs from a pass-through
    source.

    Args:
        self_or_fn (pl.LazyFrame | Callable[[], pl.LazyFrame]): The source to memoize.
            A zero-argument callable returning a ``LazyFrame`` (executed at most once, on
            first row demand, and free to contain arbitrary Python: staged collects,
            validation that raises, data-dependent control flow), or a ``LazyFrame``,
            which is wrapped as such a callable.
        schema (pl.Schema | Callable[[], pl.Schema]): The schema the returned frame
            advertises. Either a ``pl.Schema`` or a zero-argument callable returning one.
            A callable defers resolution until Polars needs it, so ``collect_schema()`` on
            the result never forces the builder to run — useful when the schema is derived
            from the builder's own lazy plan.
        partition_cols (str | Sequence[str], default ()): Optional column or columns to
            partition the buffer by. When set, a pushed predicate on the partition columns
            limits which partitions are built, so a partition never demanded is never
            materialized; a demand with no such predicate builds every partition. Each
            build collects the (optionally partition-filtered) builder once, so a partition
            is materialized at most once. Repeated demands are recognized by *structural*
            predicate equality, so a logically-equivalent but differently-shaped predicate may
            re-run a build that yields no new partitions (wasted work, never wrong results).
            Partition columns must be produced by the builder but need not appear in
            ``schema``; those absent from ``schema`` are dropped from the output and cannot be
            filtered on (Polars validates predicates against the advertised schema). Across-
            partition row order is unspecified. Empty (default) is identical to the
            single-buffer form.
        description (str | None, default None): Optional free-form description of this source instance, attached to its OpenTelemetry span (``explain_detail``).

    Returns:
        pl.LazyFrame: A LazyFrame with ``schema``, backed by a generator over a one-time
        materialization of the builder (per partition when ``partition_cols`` is set).

    Raises:
        TypeError: If ``schema`` is neither a ``pl.Schema`` nor a callable.
        ValueError: At build time, if the materialized frame is missing a declared column
            or a declared column's dtype does not match. Because the check runs inside the
            io_source generator, Polars surfaces it as
            :class:`polars.exceptions.ComputeError` at collect time (message preserved).
            A build failure is terminal for the whole instance; retry by constructing a
            fresh ``cache_memory``. A ``partition_cols`` entry absent from the constructed
            frame likewise surfaces as ``ComputeError`` at build time.
    """
    if not isinstance(schema, pl.Schema) and not callable(schema):
        raise TypeError(f"cache_memory requires a pl.Schema or a callable, got {type(schema).__name__}")

    partition_cols = (partition_cols,) if isinstance(partition_cols, str) else tuple(sorted(partition_cols))

    build_frame: Callable[[], pl.LazyFrame] = self_or_fn if callable(self_or_fn) else (lambda: self_or_fn)

    # Materialized buffers, one per partition, each already reconciled and projected to the
    # declared schema. Without partition_cols there is exactly one key, the empty tuple ``()``.
    buffers: dict[PartitionKey, pl.DataFrame] = {}
    # Partition keys already materialized — used to subtract already-built partitions from a
    # restricted build so each partition is collected at most once across the instance's lifetime.
    built_keys: set[PartitionKey] = set()
    # Restricted predicates already built. A repeated identical demand is served from buffers
    # without re-collecting, matched by structural predicate equality (``Expr.meta.eq``).
    built_predicates: list[pl.Expr] = []
    # True once an unrestricted build has run: every partition is then known, so later demands
    # (any predicate) are served from ``buffers`` without rebuilding.
    full_built = False
    # A cached build failure, stored as a lightweight ``(exception_type_name, message)`` record
    # rather than the live exception object. Storing the object would pin its traceback — and
    # through it the whole failed DataFrame — alive for the instance's lifetime, and re-raising one
    # shared object grows its traceback on every collect. We re-raise a fresh neutral error instead.
    # A failure is terminal for the whole instance.
    build_error: tuple[str, str] | None = None
    resolved_schema: pl.Schema | None = None
    building = False
    schema_lock = threading.Lock()
    build_cond = threading.Condition()

    def get_schema() -> pl.Schema:
        nonlocal resolved_schema
        # Resolve the (possibly callable) schema exactly once, under its own lock, so a
        # non-deterministic callable can never advertise one schema to Polars while the buffer
        # is reconciled against another. Polars is handed ``get_schema`` (not the raw callable),
        # so the value it sees and the value used for reconciliation are the same memoized object.
        with schema_lock:
            if resolved_schema is None:
                resolved_schema = schema() if callable(schema) else schema
            return resolved_schema

    def _plan_build(restricted_pred: pl.Expr | None):
        """Decide what to build for a demand. Must be called while holding ``build_cond``.

        Returns ``_NOTHING`` if everything demanded is already materialized (serve from
        ``buffers``); ``None`` to do a full (unrestricted) build; or a predicate to build
        just the not-yet-materialized partitions matching the demand.
        """
        if full_built:
            return _NOTHING
        if not partition_cols:
            # Single-buffer fast path: () is the only key.
            return _NOTHING if () in buffers else None
        if restricted_pred is None:
            # No usable partition predicate: we cannot know which partitions the demand needs,
            # so we must build them all.
            return None
        if any(restricted_pred.meta.eq(built) for built in built_predicates):
            return _NOTHING
        # Subtract already-built partitions so each partition is collected at most once.
        not_built = _exclude_built(built_keys)
        return restricted_pred if not_built is None else restricted_pred & not_built

    def get_partitions(restricted_pred: pl.Expr | None) -> list[pl.DataFrame]:
        """Return the materialized partition buffers for a demand, building lazily as needed.

        The partition predicate restricts *which* partitions are built (an optimization only);
        row-level correctness is still enforced by re-applying the full predicate in
        ``source_generator``, so returning a superset of buffers here is safe.
        """
        nonlocal full_built, building, build_error
        with build_cond:
            while True:
                if build_error is not None:
                    # The build ran once and failed; that outcome is terminal for this instance.
                    # Re-raise a fresh neutral error carrying the original type and message rather
                    # than retrying, so a fan-out / collect_all of N branches sees one failure, not
                    # N re-runs of a failing builder. To retry, construct a fresh cache_memory.
                    type_name, message = build_error
                    raise _CachedBuildError(f"{type_name}: {message}")
                todo = _plan_build(restricted_pred)
                if todo is _NOTHING:
                    return list(buffers.values())
                if building:
                    # Another thread is running the one in-flight build. Wait and share its outcome.
                    build_cond.wait()
                    continue
                # No outcome yet and nobody building: become the sole builder.
                building = True
                break

        # Build outside the condition lock so waiters can park on ``build_cond`` while this runs.
        # Only one thread ever reaches here (guarded by ``building`` + the terminal outcome above).
        try:
            lf = build_frame()
            if todo is not None:
                # Restricted build: filter the builder to just the demanded (not-yet-built) partitions.
                lf = lf.filter(todo)
            built = lf.collect()
            # Reconcile the materialized frame against the declared schema ONCE, here at build time.
            # The check is deliberately asymmetric: a missing declared column or a dtype mismatch
            # raises (either would silently corrupt results); an extra column is dropped by the
            # select below so the declared schema stays constant. Polars does not validate yielded
            # frames against the declared schema, so this is the only safety net. Partition columns
            # need not be declared; they are dropped from the stored buffers by the select below but
            # must be present in ``built`` for the partition_by split.
            declared = get_schema()
            actual = built.schema
            for name in declared.names():
                if name not in actual:
                    raise ValueError(f"cache_memory: declared column '{name}' is missing from the constructed frame")
                if actual[name] != declared[name]:
                    raise ValueError(f"cache_memory: column '{name}' declared as {declared[name]} but constructed frame has {actual[name]}")
            if partition_cols:
                new_buffers = {}
                for values, frame in built.partition_by(list(partition_cols), as_dict=True).items():
                    key = partition_key(dict(zip(partition_cols, values)))
                    new_buffers[key] = frame.select(declared.names())
            else:
                new_buffers = {(): built.select(declared.names())}
        except BaseException as exc:
            # Any build failure — an ordinary Exception (e.g. a data-quality raise) or a control-flow
            # BaseException (KeyboardInterrupt/SystemExit) — is terminal for this instance. Recording
            # it (rather than resetting) keeps parked fan-out waiters from each launching a redundant
            # retry, so the builder runs exactly once whatever the outcome. An interrupted build thus
            # poisons the instance; retry by constructing a fresh cache_memory.
            #
            # Derive the message *before* taking the lock: a pathological ``__str__`` that raises must
            # not abort the critical section, which would leave ``building`` stuck True and deadlock
            # every future collect. The work under the lock is then only assignment + notify.
            type_name = type(exc).__name__
            try:
                message = str(exc)
            except BaseException:  # noqa: BLE001 - a broken __str__ must not poison the instance
                message = "<exception message unavailable>"
            with build_cond:
                build_error = (type_name, message)
                building = False
                build_cond.notify_all()
            raise

        with build_cond:
            for key, frame in new_buffers.items():
                buffers.setdefault(key, frame)
                built_keys.add(key)
            if todo is None:
                # An unrestricted build materialized every partition; short-circuit later demands.
                full_built = True
            elif restricted_pred is not None:
                # Remember this demand so an identical repeat is served without re-collecting.
                built_predicates.append(restricted_pred)
            building = False
            build_cond.notify_all()
            # Snapshot under the lock: a concurrent builder may be mutating ``buffers`` (setdefault
            # above), so iterating it outside the lock could raise "dictionary changed size".
            result = list(buffers.values())
        return result

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        # Restrict the pushed predicate to the partition columns to prune which partitions we build.
        restricted_pred = restrict_expr_to_columns(predicate, set(partition_cols)) if predicate is not None and partition_cols else None
        # Each buffer is already reconciled and projected to the declared schema (see get_partitions).
        parts = get_partitions(restricted_pred)
        if len(parts) == 1:
            buf = parts[0]
        elif parts:
            buf = pl.concat(parts, how="vertical")
        else:
            buf = pl.DataFrame(schema=get_schema())
        lf = buf.lazy()
        if predicate is not None:
            lf = lf.filter(predicate)

        # Project after the filter so predicate columns are still available to it. Without this the
        # wrapped source would be asked for every column, which is strictly worse than the un-proxied
        # plan.
        if with_columns is not None:
            requested = set(with_columns)
            names = get_schema().names()
            projected = [c for c in names if c in requested]
            # Defensive: an empty projection (some Polars versions request ``with_columns=[]`` for a
            # count-only query such as ``pl.len()``) must not collapse row cardinality — ``select([])``
            # yields a 0-row, 0-column frame. Keep one declared column so every source row survives;
            # the count consumer only needs the height. (On Polars 1.42.1 count pushes a single named
            # column, so this branch is not exercised in practice — it guards against version drift.)
            if projected:
                lf = lf.select(projected)
            elif names:
                lf = lf.select(names[0])

        if n_rows is not None:
            lf = lf.head(n_rows)

        yield from collect_lf_in_io_source(lf, batch_size)

    # register_io_source_with_is_pure (unlike the plain register_io_source, which raises ComputeError
    # for a callable schema) accepts a zero-arg callable schema and resolves it lazily. We hand it the
    # memoizing ``get_schema`` so Polars and the buffer reconciliation agree on one resolved schema.
    return register_io_source_with_is_pure(io_source=source_generator, schema=get_schema, explain_detail=description)
