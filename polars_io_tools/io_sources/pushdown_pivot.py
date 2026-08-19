"""
``pushdown_pivot``: a pushdown-friendly wrapper around ``LazyFrame.pivot``.

Polars' lazy ``pivot`` (unstable as of 1.39) is implemented internally as a ``group_by`` followed by ``filter().item()``
aggregations — one per value in ``on_columns``. As a result, predicate and projection pushdown behave like they would for a
group-by aggregation:

* Filters on the ``index`` column flow through.
* A pure ``select(*index)`` drops the aggregations and projects the source to just the index column.
* Selecting only a subset of pivoted *output* columns could be rewritten as the upstream row filter
  ``col(on).is_in(subset)`` plus a projection of ``[*index, on, *values]``, but polars currently does not perform this rewrite.

``pushdown_pivot`` re-exposes the pivot output as a registered Python IO source so that polars pushes projections and
predicates *into* the wrapper, which then translates them into upstream-friendly operations.

Design notes:

* The full original predicate is always re-applied post-pivot; any upstream filter we synthesize on ``on`` is treated as an
  additional row reduction that may safely overshoot.
* The mapping ``on_value -> output column names`` is derived at construction time by calling
  ``source.pivot(on, on_columns=[v], ...).collect_schema()`` for each ``v in on_columns``. This avoids string-parsing the
  output column names and works with non-string ``on_columns``, ``column_naming="combine"``, and multi-``values`` pivots.
* When ``maintain_order=True``, the row-order of the output depends on the order of first appearance of values in ``on``;
  restricting ``on`` upstream could change that order, so the upstream filter is suppressed in that case.
* If a downstream predicate references output columns that have been dropped from the upstream filter (because they are not
  in ``with_columns``), we must keep them in the upstream filter too so the predicate can evaluate correctly.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import Any, Literal

import polars as pl

from .restrict_visitor import restrict_expr_to_columns
from .util import register_io_source_with_is_pure

__all__ = ("pushdown_pivot",)

log = logging.getLogger(__name__)


def pushdown_pivot(
    source: pl.LazyFrame,
    on: str,
    on_columns: Sequence[Any],
    *,
    index: str | Sequence[str],
    values: str | Sequence[str],
    aggregate_function: Any = None,
    maintain_order: bool = False,
    separator: str = "_",
    column_naming: Literal["auto", "combine"] = "auto",
    is_pure: bool = True,
    dense_on: bool = False,
) -> pl.LazyFrame:
    """
    Like ``source.pivot(on, on_columns=on_columns, ...)`` but pushdown-friendly.

    Args:
        source (pl.LazyFrame): The source to pivot.
        on (str): The column whose values become new output columns.
        on_columns (Sequence[Any]): Explicit set of values from ``on`` whose output columns should be
            materialized. Required to pin the output schema.
        index (str | Sequence[str]): Column(s) that remain from the input to the output.
        values (str | Sequence[str]): Column(s) of values that move under the new pivoted columns.
        aggregate_function (optional): Currently unsupported; raises ``NotImplementedError`` if given.
        maintain_order (bool, default False): If True, suppress the upstream row-filter on ``on`` so that the
            first-appearance ordering matches a bare pivot.
        separator, column_naming: Forwarded to the underlying ``pivot``.
        is_pure (bool, default True): Whether the underlying source is pure (deterministic). Forwarded to
            ``register_io_source``.
        dense_on (bool, default False): Caller's assertion that every ``index`` value in the source has at least one row for every value in ``on_columns``.
            When the wrapper synthesizes an upstream row filter ``on ∈ S`` (because the downstream selects only some pivoted
            output columns), that filter drops index values that have no row for any value in ``S``. Bare ``pl.LazyFrame.pivot``
            would have kept those index values with nulls in the surviving columns. Two modes:

            * ``dense_on=False`` (default, *smart*): correctness preserved by a second narrow scan
              ``source.select(index).unique()`` joined back to the prefiltered pivot. Always matches bare semantics.
            * ``dense_on=True`` (*fast*): caller asserts density; the recovery scan and join are skipped. Roughly 2× faster but
              silently drops rows on sparse data — only set when the source is known to be dense (e.g. a freshly exploded matrix).

            Has no effect when no upstream ``on`` filter is synthesized (no projection, full projection, or
            ``maintain_order=True``).

    Returns:
        pl.LazyFrame: A LazyFrame whose output is identical to the equivalent ``source.pivot(...)`` call, but which the optimizer can push
            filters and projections into.

    Notes:
        Construction performs ``O(len(on_columns))`` lazy-plan walks (``source.pivot(...).collect_schema()`` once per value) to
        map each ``on``-value to its set of output columns. For pivots with very large ``on_columns`` lists (hundreds+) this can
        dominate construction time; consider chunking large pivots if this becomes a bottleneck.
    """
    if aggregate_function is not None:
        raise NotImplementedError("pushdown_pivot does not yet support aggregate_function; pass None.")
    if not isinstance(on, str):
        raise TypeError(f"pushdown_pivot: `on` must be a single column name, got {on!r}")

    index_cols: list[str] = [index] if isinstance(index, str) else list(index)
    value_cols: list[str] = [values] if isinstance(values, str) else list(values)
    on_values: list[Any] = list(on_columns)

    pivot_kwargs = {
        "index": index_cols,
        "values": value_cols,
        "maintain_order": maintain_order,
        "separator": separator,
        "column_naming": column_naming,
    }

    # Defer all `collect_schema()` walks (the source schema, the full pivot output schema, and the per-`on`-value pivot
    # schemas used for the output→on-value mapping) until Polars actually needs them — typically when it asks for the
    # registered schema at collect time. This avoids forcing schema resolution on the input LazyFrame eagerly when the
    # wrapper is constructed (the convention documented on `register_io_source_with_is_pure` and used by `multi_source`).
    _resolved: dict[str, Any] = {}

    def _resolve() -> dict[str, Any]:
        if _resolved:
            return _resolved
        source_schema = source.collect_schema()
        source_cols = list(source_schema.names())
        for c in [on, *index_cols, *value_cols]:
            if c not in source_schema:
                raise ValueError(f"pushdown_pivot: column {c!r} not in source schema {source_cols!r}")
        declared_schema = source.pivot(on, on_columns=on_values, **pivot_kwargs).collect_schema()
        declared_cols = list(declared_schema.names())
        index_set = set(index_cols)
        on_value_to_output_cols: dict[Any, list[str]] = {}
        for v in on_values:
            sub_schema = source.pivot(on, on_columns=[v], **pivot_kwargs).collect_schema()
            on_value_to_output_cols[v] = [c for c in sub_schema.names() if c not in index_set]
        output_to_on_value: dict[str, Any] = {}
        for v, cols in on_value_to_output_cols.items():
            for c in cols:
                output_to_on_value[c] = v
        _resolved.update(
            source_schema=source_schema,
            source_cols=source_cols,
            declared_schema=declared_schema,
            declared_cols=declared_cols,
            on_value_to_output_cols=on_value_to_output_cols,
            output_col_set=set(output_to_on_value),
        )
        return _resolved

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        state = _resolve()
        source_cols: list[str] = state["source_cols"]
        declared_schema: pl.Schema = state["declared_schema"]
        declared_cols: list[str] = state["declared_cols"]
        on_value_to_output_cols: dict[Any, list[str]] = state["on_value_to_output_cols"]
        output_col_set: set[str] = state["output_col_set"]

        pred_index_safe: pl.Expr | None = None
        pred_root_names: set[str] = set()
        if predicate is not None:
            pred_root_names = set(predicate.meta.root_names())
            pred_index_safe = restrict_expr_to_columns(predicate, set(index_cols))

        wc_set: set[str] | None = set(with_columns) if with_columns is not None else None
        pred_output_refs = pred_root_names & output_col_set
        wc_output_refs: set[str] = (wc_set & output_col_set) if wc_set is not None else output_col_set

        if wc_set is not None:
            needed_output_materialized = wc_output_refs | pred_output_refs
        else:
            needed_output_materialized = output_col_set

        if needed_output_materialized:
            needed_on = [v for v in on_values if set(on_value_to_output_cols[v]) & needed_output_materialized]
        else:
            needed_on = []

        push_on_filter = wc_set is not None and not maintain_order and bool(needed_on) and len(needed_on) < len(on_values)

        source_filter = pred_index_safe
        if push_on_filter:
            on_filter_expr: pl.Expr
            if len(needed_on) == 1:
                on_filter_expr = pl.col(on) == needed_on[0]
            else:
                on_filter_expr = pl.col(on).is_in(needed_on)
            source_filter = on_filter_expr if source_filter is None else (source_filter & on_filter_expr)

        internal_cols: list[str] = list(index_cols)
        if needed_on:
            if on not in internal_cols:
                internal_cols.append(on)
            for v in value_cols:
                if v not in internal_cols:
                    internal_cols.append(v)
        for c in source_cols:
            if c in internal_cols:
                continue
            if c in pred_root_names or (wc_set is not None and c in wc_set):
                internal_cols.append(c)

        subset = source
        if source_filter is not None:
            subset = subset.filter(source_filter)
        subset = subset.select(internal_cols)

        pivot_on_columns = needed_on if push_on_filter else on_values
        if needed_on:
            result_df = subset.collect().pivot(on, on_columns=pivot_on_columns, **pivot_kwargs)
            if push_on_filter and not dense_on:
                # Recover index values dropped by the synthesized `on ∈ S` filter: bare pivot would have produced one row
                # per distinct(source.index) (with nulls in unmatched cells).
                recovery = source
                if pred_index_safe is not None:
                    recovery = recovery.filter(pred_index_safe)
                full_index_df = recovery.select(index_cols).unique().collect()
                result_df = full_index_df.join(result_df, on=index_cols, how="left")
        else:
            result_df = subset.collect().unique(subset=index_cols, maintain_order=True)

        missing_declared = [c for c in declared_cols if c not in result_df.columns]
        if missing_declared:
            null_exprs = [pl.lit(None).cast(declared_schema[c]).alias(c) for c in missing_declared]
            result_df = result_df.with_columns(null_exprs)
        result_df = result_df.select(declared_cols)

        result_lf = result_df.lazy()
        if predicate is not None:
            result_lf = result_lf.filter(predicate)
        if with_columns is not None:
            result_lf = result_lf.select(with_columns)
        if n_rows is not None:
            result_lf = result_lf.head(n_rows)

        out = result_lf.collect()
        if batch_size is None:
            yield out
        else:
            yield from out.iter_slices(n_rows=batch_size)

    return register_io_source_with_is_pure(
        source_generator,
        schema=lambda: _resolve()["declared_schema"],
        is_pure=is_pure,
    )
