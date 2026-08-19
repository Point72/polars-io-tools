"""
``pushdown_unpivot``: a pushdown-friendly wrapper around ``LazyFrame.unpivot``.

Polars' built-in ``unpivot`` blocks several optimization opportunities. In particular:

* A filter on the ``variable`` column (e.g. ``col("variable") == "A"`` or ``col("variable").is_in(["A", "B"])``) can be
  rewritten as an upstream *column projection* keeping only the matching value columns plus the ``index`` columns. Polars
  currently does not perform this rewrite.
* A filter on any ``index`` column can flow through unpivot unchanged. Polars already does this for built-in ``unpivot``
  over a materialized source, but it is blocked for plugin (Python) IO sources.

``pushdown_unpivot`` is a drop-in replacement that re-exposes the unpivot output as a registered Python IO source so that
polars' optimizer pushes projections and predicates *into* the wrapper, which then translates them into upstream-friendly
operations on the underlying source.

The wrapper's contract:

* The full original predicate is *always* re-applied post-unpivot. Any upstream filter we synthesize is treated as an
  additional row-reduction filter that may safely overshoot. This avoids subtle correctness issues with mixed AND/OR
  predicates and predicates referencing ``value``.
* The declared output schema (and in particular the ``value`` column dtype) is fixed at construction time using the full
  ``on`` set, so subset unpivots that would otherwise unify to a different dtype are cast back.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import Any

import polars as pl

from .restrict_visitor import restrict_expr_to_columns
from .set_visitor import convert_expr_to_valid_values
from .util import register_io_source_with_is_pure

__all__ = ("pushdown_unpivot",)

log = logging.getLogger(__name__)


def pushdown_unpivot(
    source: pl.LazyFrame,
    *,
    index: str | Sequence[str],
    on: Sequence[str] | None = None,
    variable_name: str = "variable",
    value_name: str = "value",
    is_pure: bool = True,
) -> pl.LazyFrame:
    """
    Like ``source.unpivot(index=index, on=on, ...)`` but pushdown-friendly.

    Args:
        source (pl.LazyFrame): The source to unpivot.
        index (str | Sequence[str]): The id column(s) to keep on every row.
        on (Sequence[str], optional): The value columns to unpivot. Defaults to all non-index columns of ``source``.
        variable_name (str, default "variable"): Name of the variable column in the output.
        value_name (str, default "value"): Name of the value column in the output.
        is_pure (bool, default True): Whether the underlying source is pure (deterministic). Forwarded to ``register_io_source``.

    Returns:
        pl.LazyFrame: A LazyFrame whose output is identical to the equivalent ``source.unpivot(...)`` call, but which the optimizer can
            push filters and projections into.
    """
    index_cols: list[str] = [index] if isinstance(index, str) else list(index)

    # Defer all `collect_schema()` walks (the source schema, the resolved on-columns list, and the declared unpivot output
    # schema) until Polars actually needs them — typically when it asks for the registered schema at collect time. This
    # avoids forcing schema resolution on the input LazyFrame eagerly when the wrapper is constructed (the convention
    # documented on `register_io_source_with_is_pure` and used by `multi_source`).
    _resolved: dict[str, Any] = {}

    def _resolve() -> dict[str, Any]:
        if _resolved:
            return _resolved
        source_schema = source.collect_schema()
        source_cols = list(source_schema.names())
        if on is None:
            on_cols = [c for c in source_cols if c not in index_cols]
        else:
            on_cols = list(on)
        missing = [c for c in index_cols + on_cols if c not in source_schema]
        if missing:
            raise ValueError(f"pushdown_unpivot: columns {missing!r} not in source schema {source_cols!r}")
        declared_schema = source.unpivot(
            index=index_cols,
            on=on_cols,
            variable_name=variable_name,
            value_name=value_name,
        ).collect_schema()
        _resolved.update(
            source_cols=source_cols,
            on_cols=on_cols,
            declared_schema=declared_schema,
            declared_value_dtype=declared_schema[value_name],
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
        on_cols: list[str] = state["on_cols"]
        declared_schema: pl.Schema = state["declared_schema"]
        declared_value_dtype: pl.DataType = state["declared_value_dtype"]

        pred_index_safe: pl.Expr | None = None
        valid_variable_values: set[Any] | None = None
        pred_root_names: set[str] = set()
        if predicate is not None:
            pred_root_names = set(predicate.meta.root_names())
            pred_index_safe = restrict_expr_to_columns(predicate, set(index_cols))
            try:
                valid_variable_values = convert_expr_to_valid_values(predicate, variable_name)
            except Exception:
                log.debug("Failed to extract variable values from predicate", exc_info=True)
                valid_variable_values = None

        if valid_variable_values is not None:
            internal_on = [v for v in on_cols if v in valid_variable_values]
        else:
            internal_on = list(on_cols)

        if not internal_on:
            empty = pl.DataFrame(schema=declared_schema)
            if with_columns is not None:
                empty = empty.select(with_columns)
            yield empty
            return

        wc_set = set(with_columns) if with_columns is not None else set()
        need_value = (value_name in pred_root_names) or (value_name in wc_set) or (with_columns is None)
        need_variable = (variable_name in pred_root_names) or (variable_name in wc_set) or (with_columns is None)

        internal_cols: list[str] = []
        for c in index_cols:
            if c in pred_root_names or c in wc_set or with_columns is None:
                internal_cols.append(c)
        if need_value:
            for v in internal_on:
                if v not in internal_cols:
                    internal_cols.append(v)
        for c in source_cols:
            if c in internal_cols:
                continue
            if c in pred_root_names:
                internal_cols.append(c)

        if not internal_cols:
            internal_cols = [index_cols[0]]

        subset = source
        if pred_index_safe is not None:
            subset = subset.filter(pred_index_safe)
        subset = subset.select(internal_cols)

        if need_value:
            result_lf = subset.unpivot(
                index=[c for c in index_cols if c in internal_cols],
                on=internal_on,
                variable_name=variable_name,
                value_name=value_name,
            )
            if result_lf.collect_schema()[value_name] != declared_value_dtype:
                result_lf = result_lf.with_columns(pl.col(value_name).cast(declared_value_dtype))
        else:
            base = subset.collect()
            if need_variable:
                parts = [base.with_columns(pl.lit(v, dtype=declared_schema[variable_name]).alias(variable_name)) for v in internal_on]
            else:
                parts = [base] * len(internal_on)  # N references; pl.concat materialises N identical copies
            result_lf = pl.concat(parts, how="vertical").lazy() if parts else pl.DataFrame(schema=declared_schema).lazy()

        if predicate is not None:
            result_lf = result_lf.filter(predicate)

        if with_columns is not None:
            result_lf = result_lf.select(with_columns)

        if n_rows is not None:
            result_lf = result_lf.head(n_rows)

        result_df = result_lf.collect()
        if batch_size is None:
            yield result_df
        else:
            yield from result_df.iter_slices(n_rows=batch_size)

    return register_io_source_with_is_pure(
        source_generator,
        schema=lambda: _resolve()["declared_schema"],
        is_pure=is_pure,
    )
