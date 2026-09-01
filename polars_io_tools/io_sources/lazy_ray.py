import contextlib
import io
import math
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import cloudpickle
import polars as pl
import polars.selectors as cs
from tqdm import tqdm

from .partitions import (
    KeyPartitions,
    Partitioner,
    ReadPartition,
    as_partition_list,
    by_range,
    by_time,
    by_value,
    cartesian_partitions,
    discrete_partitions,
    retained_columns,
)

try:
    import ray
except ImportError:
    raise ImportError("The `execute_on_ray` function requires the `ray` package. Please install it with `pip install ray`.")


from .util import register_io_source_with_is_pure

# As we did in the `lazy_parquet_cache` module, we expose the functions to users, in case
# they want to use them directly or though Polars' `.pipe()` syntax; however, the canonical
# useage is to call the functions as methods on the `LazyFrame`'s `piot` namespace.
__all__ = (
    "RayPartition",
    "ReadPartition",
    "by_range",
    "by_time",
    "by_value",
    "cartesian_partitions",
    "discrete_partitions",
    "execute_on_ray",
)


@dataclass(frozen=True)
class RayPartition(ReadPartition):
    """A :class:`ReadPartition` carrying per-task Ray options.

    The shared ``predicate`` / ``key`` come from :class:`ReadPartition`; ``remote_options`` are
    Ray-specific overrides shallow-merged over the calling function's uniform base options.
    """

    remote_options: Mapping[str, Any] | None = field(default=None)


_TIME_UNIT_TO_INTERVAL = {"daily": "1d", "monthly": "1mo", "yearly": "1y"}


@ray.remote
def _execute_partition(
    plan_bytes_or_ref: "bytes | ray.ObjectRef",
    predicate_bytes: bytes,
    return_as: Literal["arrow", "ipc", "parquet"] = "arrow",
) -> bytes:
    """
    Deserialise plan and partition predicate, apply the filter, collect, and return the
    result in the requested format. Works whether or not Ray has already dereferenced
    ``plan_bytes_or_ref``.
    """
    # Resolve the reference if we have it; don't worry
    # otherwise, since we've got the raw bytes
    if isinstance(plan_bytes_or_ref, ray.ObjectRef):
        plan_bytes = ray.get(plan_bytes_or_ref)
    else:
        plan_bytes = plan_bytes_or_ref

    lf = cloudpickle.loads(plan_bytes)
    predicate = cloudpickle.loads(predicate_bytes)

    out = lf.filter(predicate).collect()

    if return_as == "arrow":
        return out.to_arrow()
    elif return_as == "parquet":
        buf = io.BytesIO()
        out.write_parquet(buf, use_pyarrow=True)
        return buf.getvalue()
    elif return_as == "ipc":
        buf = io.BytesIO()
        out.write_ipc(buf)
        return buf.getvalue()
    else:
        raise ValueError(f"Unsupported return format: {return_as}")


def _decode_blob(blob: bytes, return_as: str) -> pl.DataFrame:
    if return_as == "arrow":
        return pl.DataFrame(blob)
    elif return_as == "parquet":
        return pl.read_parquet(io.BytesIO(blob))
    elif return_as == "ipc":
        return pl.read_ipc(io.BytesIO(blob))
    else:
        raise ValueError(f"Unsupported return format: {return_as}")


# Shared core

_VALID_RETURN_AS = ("arrow", "ipc", "parquet")


def _output_schema(full_schema: pl.Schema, requested_cols: list[str] | None) -> pl.Schema:
    if requested_cols is None:
        return full_schema
    return pl.Schema({c: full_schema[c] for c in requested_cols})


def _run_on_ray(
    original_lf: pl.LazyFrame,
    make_specs: Callable[["pl.Expr | None"], list[RayPartition]],
    *,
    predicate_required: bool,
    return_as: Literal["arrow", "ipc", "parquet"],
    base_remote_options: dict,
    max_concurrency: int | None,
    preserve_partition_order: bool,
    description: str | None = None,
) -> pl.LazyFrame:
    """
    Register a custom IO source that distributes ``original_lf`` across Ray by fanning out
    one task per partition produced by ``make_specs``.

    ``make_specs`` receives the pushed-down predicate (or ``None``) and returns the list of
    :class:`RayPartition` to execute. The core owns everything partition-scheme agnostic:
    projection/predicate pushdown, predicate-column retention, task fan-out with bounded
    concurrency, result reassembly, ``n_rows``/``batch_size`` handling, and cancellation.
    """
    if return_as not in _VALID_RETURN_AS:
        raise ValueError(f"`return_as` must be one of {_VALID_RETURN_AS}, got {return_as!r}.")
    if max_concurrency is not None and max_concurrency <= 0:
        raise ValueError(f"`max_concurrency` must be a positive integer or None, got {max_concurrency!r}.")

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        full_schema = original_lf.collect_schema()
        requested_cols = list(with_columns) if with_columns is not None else None

        if predicate_required and predicate is None:
            raise ValueError("`execute_on_ray` requires a bounded temporal predicate on the chosen `date_column`.")

        specs = list(make_specs(predicate))
        if not specs:
            yield pl.DataFrame(schema=_output_schema(full_schema, requested_cols))
            return

        # Columns required to evaluate every predicate that runs against the source (each
        # partition predicate on the worker, plus the pushed-down predicate) must survive
        # projection pushdown even if the user did not request them, and are dropped again from
        # yielded frames.
        predicates = [sp.predicate for sp in specs]
        if predicate is not None:
            predicates.append(predicate)
        retained = retained_columns(predicates, requested_cols)

        lf = original_lf
        added_cols: list[str] = []
        if requested_cols is not None:
            if retained is None:
                # A predicate's dependencies could not be resolved -- retain every column and drop
                # the non-requested ones from yielded frames.
                added_cols = [c for c in full_schema.names() if c not in requested_cols]
            else:
                added_cols = [c for c in retained if c not in requested_cols]
                lf = lf.select(retained)

        if predicate is not None:
            lf = lf.filter(predicate)

        plan_bytes = cloudpickle.dumps(lf)

        def make_task(sp: RayPartition) -> ray.ObjectRef:
            opts = dict(base_remote_options)
            if sp.remote_options:
                opts.update(sp.remote_options)
            pred_bytes = cloudpickle.dumps(sp.predicate)
            return _execute_partition.options(**opts).remote(plan_bytes, pred_bytes, return_as)

        def fetch(ref: ray.ObjectRef, idx: int, sp: RayPartition) -> pl.DataFrame:
            try:
                blob = ray.get(ref)
            except Exception as e:
                err_msg = (
                    f"Ray worker failed while executing partition {idx} (key={sp.key!r}) of lazy frame.\n"
                    f"Polars plan for this lazy frame:\n{lf.explain()}"
                    f"\n\nWhile running the above, received error: {e.__class__.__name__}:{e}"
                )
                raise RuntimeError(err_msg) from e
            return _decode_blob(blob, return_as)

        spec_iter = iter(enumerate(specs))
        effective_concurrency = max_concurrency if max_concurrency is not None else len(specs)
        window = min(effective_concurrency, len(specs))

        pbar = tqdm(total=len(specs), desc="execute_on_ray")

        rows_yielded = 0
        any_yielded = False
        stop = False

        def prepare(df: pl.DataFrame) -> tuple[list[pl.DataFrame], bool]:
            """Drop helper columns, honour ``n_rows``, and slice by ``batch_size``.

            Returns the frames to yield and whether the global ``n_rows`` limit is now met.
            """
            nonlocal rows_yielded
            if added_cols:
                df = df.drop(added_cols)
            stop = False
            if n_rows is not None:
                remaining = n_rows - rows_yielded
                if remaining <= 0:
                    return [], True
                if len(df) >= remaining:
                    df = df.head(remaining)
                    stop = True
            rows_yielded += len(df)
            if batch_size is not None and len(df) > batch_size:
                return [df.slice(i, batch_size) for i in range(0, len(df), batch_size)], stop
            return [df], stop

        if preserve_partition_order:
            # Ordered fan-out. Keep a spec-order deque of at most ``window`` in-flight tasks and
            # block only on the head -- the next partition to yield. Later tasks run concurrently;
            # their results stay in Ray's object store behind their refs (spillable) until we
            # reach them, rather than being decoded and buffered on the driver. Outstanding
            # results are bounded to ``window`` refs plus the one partition being materialised.
            inflight: deque[tuple[ray.ObjectRef, int, RayPartition]] = deque()
            try:
                for _ in range(window):
                    idx, sp = next(spec_iter)
                    inflight.append((make_task(sp), idx, sp))
                while inflight:
                    ref, idx, sp = inflight.popleft()
                    df = fetch(ref, idx, sp)
                    del ref  # release the head result promptly once decoded
                    frames, stop = prepare(df)
                    pbar.update()
                    # Refill the freed slot *before* yielding: a generator suspends at ``yield``,
                    # so submitting the replacement afterwards would idle a worker until the
                    # consumer pulls again.
                    if not stop:
                        with contextlib.suppress(StopIteration):
                            nxt_idx, nxt_sp = next(spec_iter)
                            inflight.append((make_task(nxt_sp), nxt_idx, nxt_sp))
                    for fr in frames:
                        any_yielded = True
                        yield fr
                    if stop:
                        break
            finally:
                for ref, _, _ in inflight:
                    with contextlib.suppress(Exception):
                        ray.cancel(ref, force=False)
                pbar.close()
        else:
            # Completion-order fan-out. Yield each partition as soon as it finishes. A ref is
            # owned by ``pending`` until popped on completion, so nothing pins finished results.
            pending: dict[ray.ObjectRef, tuple[int, RayPartition]] = {}
            try:
                for _ in range(window):
                    idx, sp = next(spec_iter)
                    pending[make_task(sp)] = (idx, sp)
                while pending and not stop:
                    ready_refs, _ = ray.wait(list(pending), num_returns=1)
                    for ref in ready_refs:
                        idx, sp = pending.pop(ref)
                        # keep the pipeline full
                        with contextlib.suppress(StopIteration):
                            nxt_idx, nxt_sp = next(spec_iter)
                            pending[make_task(nxt_sp)] = (nxt_idx, nxt_sp)
                        pbar.update()
                        df = fetch(ref, idx, sp)
                        frames, stop = prepare(df)
                        for fr in frames:
                            any_yielded = True
                            yield fr
                        if stop:
                            break
            finally:
                # Cancel any still-pending tasks (n_rows satisfied early, or a failure occurred).
                for ref in pending:
                    with contextlib.suppress(Exception):
                        ray.cancel(ref, force=False)
                pbar.close()

        if not any_yielded:
            yield pl.DataFrame(schema=_output_schema(full_schema, requested_cols))

    return register_io_source_with_is_pure(source_generator, schema=lambda: original_lf.collect_schema(), explain_detail=description)


# Shared preconditions / helpers


def _require_ray() -> None:
    if not ray.is_initialized():
        raise RuntimeError("Ray is not initialised. Please call `ray.init()` before using `execute_on_ray`.")


def _normalize_remote_options(remote_options: dict | None) -> dict:
    remote_options = remote_options or {}
    if not isinstance(remote_options, dict):
        raise TypeError("`remote_options` must be a dict or None.")
    return remote_options


def _resolve_ro(remote_options, key):
    """Resolve per-partition remote options from a callable or a {key: dict} mapping."""
    if remote_options is None:
        return None
    if callable(remote_options):
        return remote_options(key)
    if isinstance(remote_options, Mapping):
        return remote_options.get(key)
    raise TypeError("per-partition `remote_options` must be a callable or a mapping.")


def _canonical_key(value):
    """Canonicalize a key for duplicate detection so that distinct NaN objects (which are
    unequal in a Python set but match under polars equality) collapse to one value."""
    if isinstance(value, tuple):
        return tuple(_canonical_key(v) for v in value)
    if isinstance(value, float) and math.isnan(value):  # NaN
        return "__nan__"
    return value


def _prune_by_key(specs: list[RayPartition], key_cols: list[str], predicate: pl.Expr | None, schema: pl.Schema) -> list[RayPartition]:
    """Drop equality partitions that cannot match the pushed-down ``predicate``.

    Sound and exact: each partition corresponds to a concrete key value on ``key_cols``, so a
    row in that partition has those exact values. When ``predicate`` references *only*
    ``key_cols``, evaluating it -- with polars' real semantics -- on a one-row-per-partition
    frame of the key values is equivalent to evaluating it on any real row of the partition,
    provided both preconditions below hold. We then keep only partitions whose key row
    satisfies it.

    Preconditions for exactness (otherwise fail open, keeping every partition):
    - **Non-float key dtype.** Float ``==`` is not observational (``-0.0``/``+0.0``, NaN), so a
      matched row need not be observationally identical to the key. Every other dtype has
      observational equality, making the key a faithful representative.
    - **No Python UDF in the predicate.** A pushed row-wise native predicate is a
      deterministic function of the row's values; a stateful/non-deterministic UDF is not, so
      one evaluation on the key row would not predict the worker's evaluation.

    Pruning only decides whether to launch a task; the partition predicate is always applied
    on the worker, so a retained partition never changes its result.
    """
    if predicate is None or not specs:
        return specs
    try:
        pred_cols = set(predicate.meta.root_names())
    except Exception:  # noqa: BLE001 -- cannot introspect, fail open
        return specs
    if not pred_cols or not pred_cols.issubset(key_cols):
        return specs
    if any(c not in schema or schema[c].is_float() for c in key_cols):
        return specs
    if _has_python_udf(predicate):
        return specs
    try:
        columns = {}
        for i, c in enumerate(key_cols):
            values = [(sp.key[i] if isinstance(sp.key, tuple) else sp.key) for sp in specs]
            columns[c] = pl.Series(c, values, dtype=schema[c])
        keep_mask = pl.DataFrame(columns).select(predicate.alias("__keep__"))["__keep__"].to_list()
    except Exception:  # noqa: BLE001 -- evaluation failed, fail open
        return specs
    return [sp for sp, keep in zip(specs, keep_mask) if keep is True]


def _has_python_udf(expr: pl.Expr) -> bool:
    """Whether ``expr`` contains a Python UDF (``map_elements`` / ``map_batches``), which can
    be non-deterministic or stateful. Returns True conservatively if it cannot be verified."""
    try:
        return "AnonymousFunction" in expr.meta.serialize(format="json")
    except Exception:  # noqa: BLE001 -- cannot verify, treat as unsafe
        return True


def execute_on_ray(
    self: pl.LazyFrame,
    partitions: "Partitioner | Iterable[ReadPartition] | None" = None,
    *,
    date_column: str | None = None,
    time_unit: Literal["daily", "monthly", "yearly"] | None = None,
    return_as: Literal["arrow", "ipc", "parquet"] = "arrow",
    remote_options: dict | None = None,
    max_concurrency: int | None = 100,
    preserve_partition_order: bool | None = None,
    description: str | None = None,
) -> pl.LazyFrame:
    """
    Execute a Polars LazyFrame on an *already initialised* Ray cluster, distributing the work
    across one Ray task per :class:`ReadPartition`.

    The function returns **another** LazyFrame whose scan node is a custom I/O source. No
    computation happens immediately; evaluation is triggered only when the user calls
    ``.collect()``.

    Args:
        partitions (Partitioner | Iterable[ReadPartition] | None): How to split the work. Pass a
            partitioner from :mod:`polars_io_tools.io_sources.partitions` (``by_time``, ``by_value``,
            ``by_range``) to derive slices from the filter pushed down at scan time, or an explicit
            iterable of :class:`ReadPartition` / :class:`RayPartition` for hand-built slices. A
            partitioner requires a bounded pushed-down predicate on its column.
        date_column, time_unit: Legacy calendar shortcut, equivalent to
            ``partitions=by_time(date_column, {daily: "1d", monthly: "1mo", yearly: "1y"}[time_unit])``.
            Mutually exclusive with ``partitions``.
        return_as ({"arrow", "ipc", "parquet"}, default "arrow"): The format in which the Ray
            worker returns the data.
        remote_options (Optional[dict]): Uniform Ray ``.options()`` for each task, overridden per
            partition by any :class:`RayPartition.remote_options`.
        max_concurrency (Optional[int], default 100): The maximum number of concurrent tasks.
        preserve_partition_order (bool | None): If True, yield partitions in spec order; if False,
            yield in completion order. Ordered mode blocks on the next partition in order while
            later tasks run ahead, keeping outstanding results bounded by ``max_concurrency``
            (their unmaterialised results live in Ray's spillable object store). Defaults to
            completion order for ``partitions``, and to spec order for the legacy calendar shortcut.
        description: Optional free-form description of this source instance, attached to its
            OpenTelemetry span (``explain_detail``).

    Returns:
        pl.LazyFrame: A new LazyFrame whose execution runs on a Ray cluster at ``.collect()``.

    **WARNING:** Chaining multiple ``execute_on_ray`` calls (nesting one distributed source
    inside another's plan) can have unintended consequences -- it relies on predicate pushdown
    surviving intervening operations and can silently re-execute the upstream once per downstream
    partition (N x N). Partition once at the outermost boundary. For multi-stage / cross-shuffle
    distributed pipelines prefer Polars Cloud (https://docs.cloud.pola.rs/polars-cloud/).
    """
    _require_ray()
    base_remote_options = _normalize_remote_options(remote_options)
    original_lf = self

    # Legacy calendar shortcut: translate `date_column`/`time_unit` into a `by_time` partitioner
    # and run it through the single partition path below (kept chronological, as it was before).
    if date_column is not None or time_unit is not None:
        if partitions is not None:
            raise ValueError("Pass either `partitions` or the legacy `date_column`/`time_unit`, not both.")
        if date_column is None or time_unit is None:
            raise ValueError("The legacy calendar shortcut needs both `date_column` and `time_unit`.")
        try:
            every = _TIME_UNIT_TO_INTERVAL[time_unit]
        except KeyError:
            raise ValueError(f"time_unit must be one of {sorted(_TIME_UNIT_TO_INTERVAL)}, got {time_unit!r}.") from None
        partitions = by_time(date_column, every)
        if preserve_partition_order is None:
            preserve_partition_order = True

    if partitions is None:
        raise ValueError("Provide `partitions` (a partitioner or ReadPartition list), or the legacy `date_column`/`time_unit`.")

    if isinstance(partitions, KeyPartitions):
        return _execute_on_ray_by(
            self,
            partitions.partitions,
            partitions.by,
            remote_options=remote_options,
            partition_remote_options=partitions.partition_remote_options,
            return_as=return_as,
            max_concurrency=max_concurrency,
            preserve_partition_order=bool(preserve_partition_order),
            description=description,
        )

    # Materialise an explicit iterable once (a partitioner is re-usable and re-built per collect,
    # but a bare generator would be exhausted after the first collection).
    if not isinstance(partitions, Partitioner):
        partitions = list(partitions)

    def make_specs(predicate: pl.Expr | None) -> list[RayPartition]:
        resolved = as_partition_list(partitions, predicate)
        if resolved is None:
            raise ValueError("Could not derive partitions from the pushed-down predicate (no bounded range on the partition column).")
        return [sp if isinstance(sp, RayPartition) else RayPartition(predicate=sp.predicate, key=sp.key) for sp in resolved]

    return _run_on_ray(
        original_lf,
        make_specs,
        # A partitioner that needs the pushed predicate reports that by returning None from
        # build(), which make_specs turns into a clear error -- so the source never needs to
        # pre-require a predicate (an explicit list or by_value([...]) needs none).
        predicate_required=False,
        return_as=return_as,
        base_remote_options=base_remote_options,
        max_concurrency=max_concurrency,
        preserve_partition_order=bool(preserve_partition_order),
        description=description,
    )


def _partitions_to_frame(partitions, by) -> tuple[pl.DataFrame, list[str] | None, "pl.Expr | None"]:
    """Normalize ``partitions`` into a DataFrame and resolve the key columns / expression.

    Returns ``(pdf, key_cols, key_expr)``. Exactly one of ``key_cols`` / ``key_expr`` is set.
    When ``by`` is an expression, the single key column in ``pdf`` is named ``__key__``.
    """
    # A polars selector is an `Expr` subclass, so it must be detected *before* the general
    # expression branch and resolved through the frame's schema like column selectors.
    is_selector = cs.is_selector(by)

    if isinstance(by, pl.Expr) and not is_selector:
        if isinstance(partitions, pl.DataFrame):
            if partitions.width != 1:
                raise ValueError("When `by` is an expression, `partitions` must be a single column / Series / sequence.")
            values = partitions.to_series()
        elif isinstance(partitions, pl.Series):
            values = partitions
        elif isinstance(partitions, pl.LazyFrame):
            values = partitions.collect().to_series()
        else:
            values = pl.Series("__key__", list(partitions))
        return values.rename("__key__").to_frame(), None, by

    # by is column name(s) or a selector
    if isinstance(partitions, pl.LazyFrame):
        pdf = partitions.collect()
    elif isinstance(partitions, pl.DataFrame):
        pdf = partitions
    elif isinstance(partitions, pl.Series):
        # For a scalar string `by`, align the series name with the key column.
        pdf = (partitions.rename(by) if isinstance(by, str) else partitions).to_frame()
    elif isinstance(by, str):
        pdf = pl.Series(by, list(partitions)).to_frame()
    else:
        raise TypeError("`partitions` must be a DataFrame / LazyFrame when `by` selects multiple columns.")

    if isinstance(by, str):
        key_cols = [by]
    elif isinstance(by, (list, tuple)) and all(isinstance(c, str) for c in by):
        key_cols = list(by)
    else:
        # treat as a selector
        key_cols = pdf.select(by).columns

    return pdf, key_cols, None


def _execute_on_ray_by(
    self: pl.LazyFrame,
    partitions,
    by,
    *,
    remote_options: dict | None = None,
    partition_remote_options=None,
    return_as: Literal["arrow", "ipc", "parquet"] = "arrow",
    max_concurrency: int | None = 100,
    preserve_partition_order: bool = False,
    description: str | None = None,
) -> pl.LazyFrame:
    """
    Distribute ``self`` across Ray by equality on caller-enumerated partition keys.

    ``partitions`` is a small frame with **one row per partition**; each row becomes a task
    whose predicate is ``AND(col_i == row[col_i])`` over the ``by`` columns (a null key value
    becomes ``col.is_null()``). ``by`` may be a column name, a list of names, a polars
    selector, or a ``pl.Expr`` (e.g. ``pl.col("id").hash() % N``) evaluated against the
    execution frame -- so partitioning needs no upstream ``with_columns`` and no schema
    pollution.

    For **grouped / member-list** partitioning (one shard = many ids), use ``by_value`` or
    :func:`discrete_partitions` (``id.is_in(members)``) with :func:`execute_on_ray` rather than
    this scalar-equality convenience.

    Args:
        partitions: A ``DataFrame`` / ``LazyFrame`` (one row per partition), or a
            ``Series`` / sequence of scalar keys (with a scalar ``by``).
        by: Column name(s), a selector, or a ``pl.Expr`` identifying the partition key.
        remote_options (Optional[dict]): Uniform base Ray options.
        partition_remote_options: Per-partition Ray options as either the name of a struct
            column in ``partitions``, or a ``{key: dict}`` mapping keyed by the partition key.
        return_as ({"arrow", "ipc", "parquet"}, default "arrow"): Worker return format.
        max_concurrency (Optional[int], default 100): Maximum concurrent tasks.
        preserve_partition_order (bool, default False): Yield in spec order (blocks on the next
            partition while later tasks run ahead; outstanding results bounded by
            ``max_concurrency``) vs completion order (no order guarantee).

    Returns:
        pl.LazyFrame: A new LazyFrame whose execution runs on a Ray cluster at ``.collect()``.

    Note: hash-bucket keys (a computed ``hash % N``) cannot be pruned at the source, so each
    task scans the universe and filters after -- this bounds working-set memory, not scan
    I/O. Discrete keys on a real, physically-prunable column can prune at the source.

    **WARNING:** Chaining multiple ``execute_on_ray*`` calls can have unintended
    consequences. Partition once at the outermost boundary; for multi-stage distributed
    pipelines prefer Polars Cloud (https://docs.cloud.pola.rs/polars-cloud/).
    """
    _require_ray()
    base_remote_options = _normalize_remote_options(remote_options)
    original_lf = self

    pdf, key_cols, key_expr = _partitions_to_frame(partitions, by)

    option_col = partition_remote_options if isinstance(partition_remote_options, str) else None
    option_map = partition_remote_options if isinstance(partition_remote_options, Mapping) else None
    if option_col is not None and option_col not in pdf.columns:
        raise ValueError(f"`partition_remote_options` column {option_col!r} not found in `partitions`.")
    # The options column must not be treated as a key column.
    if option_col is not None and key_cols is not None:
        key_cols = [c for c in key_cols if c != option_col]
        if not key_cols:
            raise ValueError("`by` must select at least one key column distinct from the options column.")

    # Reject duplicate partition keys up front (cheap, deterministic). NaN keys are
    # canonicalized so that repeated NaNs (which compare unequal in a Python set but match
    # under polars equality) are rejected.
    if key_expr is not None:
        _keys = pdf["__key__"].to_list()
    elif len(key_cols) > 1:
        _keys = [tuple(r) for r in pdf.select(key_cols).iter_rows()]
    else:
        _keys = pdf[key_cols[0]].to_list()
    _seen: set = set()
    for _k in _keys:
        _canon = _canonical_key(_k)
        if _canon in _seen:
            raise ValueError(f"Duplicate partition key {_k!r} in `partitions`.")
        _seen.add(_canon)

    # Resolve the output dtype of the key expression once, for a typed literal.
    key_expr_dtype = None
    if key_expr is not None:
        key_expr_dtype = original_lf.select(key_expr.alias("__key__")).collect_schema()["__key__"]

    def make_specs(predicate: pl.Expr | None) -> list[RayPartition]:
        exec_schema = original_lf.collect_schema()
        specs: list[RayPartition] = []
        for row in pdf.iter_rows(named=True):
            if key_expr is not None:
                val = row["__key__"]
                pred = key_expr.is_null() if val is None else (key_expr == pl.lit(val, dtype=key_expr_dtype))
                key = val
            else:
                conds = []
                key_vals = []
                for c in key_cols:
                    v = row[c]
                    key_vals.append(v)
                    if v is None:
                        conds.append(pl.col(c).is_null())
                    else:
                        conds.append(pl.col(c) == pl.lit(v, dtype=exec_schema[c]))
                pred = conds[0]
                for extra in conds[1:]:
                    pred = pred & extra
                key = tuple(key_vals) if len(key_cols) > 1 else key_vals[0]

            if option_col is not None:
                ro = row[option_col]
            else:
                ro = _resolve_ro(option_map, key)

            specs.append(RayPartition(predicate=pred, key=key, remote_options=ro))
        # Exact key-value pruning (only when the pushed-down predicate touches key columns
        # only). Pruning drops only partitions that cannot match; retained partitions are
        # unchanged.
        if key_cols is not None:
            specs = _prune_by_key(specs, key_cols, predicate, exec_schema)
        return specs

    return _run_on_ray(
        original_lf,
        make_specs,
        predicate_required=False,
        return_as=return_as,
        base_remote_options=base_remote_options,
        max_concurrency=max_concurrency,
        preserve_partition_order=preserve_partition_order,
        description=description,
    )
