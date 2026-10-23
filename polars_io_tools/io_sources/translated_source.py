from __future__ import annotations

import json
import logging
from functools import reduce
from typing import Literal

import polars as pl
from pydantic import BaseModel

from polars_io_tools.io_sources.base import (
    AliasNode,
    BaseExprNode,
    BinaryExprNode,
    CastNode,
    ExprVisitor,
    FunctionNode,
    LiteralNode,
)
from polars_io_tools.io_sources.enum import (
    BooleanFunctionType,
    DataType,
    OperatorType,
    TimeUnit,
)

from .base import get_parsed_expr

log = logging.getLogger(__name__)

__all__ = ("TranslatedPredicateVisitor", "translate_polars_predicate")


_BINARY_OPS = {
    OperatorType.AND: pl.Expr.and_,
    OperatorType.OR: pl.Expr.or_,
    OperatorType.EQ: pl.Expr.eq,
    OperatorType.EQ_VALIDITY: pl.Expr.eq_missing,
    OperatorType.NOT_EQ: pl.Expr.ne,
    OperatorType.NOT_EQ_VALIDITY: pl.Expr.ne_missing,
    OperatorType.GT: pl.Expr.gt,
    OperatorType.GT_EQ: pl.Expr.ge,
    OperatorType.LT: pl.Expr.lt,
    OperatorType.LT_EQ: pl.Expr.le,
}

_UNARY_FUNCTIONS = {
    BooleanFunctionType.IS_NULL: pl.Expr.is_null,
    BooleanFunctionType.IS_NOT_NULL: pl.Expr.is_not_null,
    BooleanFunctionType.NOT: pl.Expr.not_,
}


class TranslatedPredicateVisitor(ExprVisitor[pl.Expr | None]):
    """Visitor that rewrites predicates to operate on underlying storage types.

    Produces a new pl.Expr (or None if cannot be rewritten) by walking the AST.
    """

    def __init__(self, mapping: dict[str, pl.DataType]):
        self.mapping = mapping
        self._result: pl.Expr | None = None

    # Utilities
    def _eval(self, node: BaseExprNode) -> pl.Expr | None:
        node.accept(self)
        return self._result

    def process_results(self) -> pl.Expr | None:
        return self._result

    def default_visit(self, node: BaseExprNode) -> None:
        # An untranslated expression is only valid on the stored frame if it reads no mapped column.
        self._result = None if set(node.expr.meta.root_names()) & self.mapping.keys() else node.expr

    # Leaf nodes
    def visit_literal(self, node: LiteralNode) -> None:
        # Keep the typed literal: converting through Python values truncates sub-microsecond precision.
        self._result = node.expr

    def visit_alias(self, node: AliasNode) -> None:
        self._result = self._eval(node.input)

    def visit_column(self, node: BaseExprNode) -> None:  # ColumnNode
        # type: ignore[override]
        # Cast mapped columns to their target logical dtype inside predicate
        name = getattr(node, "name", None)
        if name is None:
            self._result = node.expr
            return
        m = self.mapping.get(name)
        # Same cast the reader applies to mapped columns; user casts are layered on top in visit_cast.
        self._result = pl.col(name) if m is None else pl.col(name).cast(m, strict=False)

    def visit_cast(self, node: CastNode) -> None:
        # Re-apply user casts on top of mapped base columns without undoing
        # the underlying mapping. If the input resolves to a column with a
        # mapping, first cast that column to the target logical dtype, then
        # apply the user cast on top.
        base = self._eval(node.input)
        # Unknown options may hide wrap_numerical=True, which changes values.
        if base is None or node.options is None:
            self._result = None
            return
        # If input is a simple column with mapping, _eval already applied
        # the mapped cast. So we simply cast the resulting expression to
        # the requested dtype, preserving the layering.
        try:
            self._result = base.cast(node.dtype, strict=False, wrap_numerical=node.options == "Overflowing")
        except Exception as e:  # noqa: BLE001 -- intentional broad catch (defensive fallback)
            log.warning(f"Recevied error {e!s} when applying a cast on {base!s}")
            self._result = None

    # Binary expressions
    def visit_binary_expr(self, node: BinaryExprNode) -> None:
        # Casts on literals are kept, as they can change the value.
        left_expr = self._eval(node.left)
        right_expr = self._eval(node.right)
        log.debug(f"Resolved binary operands left={left_expr} right={right_expr}")
        if left_expr is None or right_expr is None:
            self._result = None
            return
        op = _BINARY_OPS.get(node.op)
        if op is None:
            self.default_visit(node)
            return
        self._result = op(left_expr, right_expr)

    # Functions
    def visit_function(self, node: FunctionNode) -> None:
        ft = node.function_type
        if ft in _UNARY_FUNCTIONS and len(node.inputs) == 1:
            inner = self._eval(node.inputs[0])
            self._result = None if inner is None else _UNARY_FUNCTIONS[ft](inner)
            return
        if ft == BooleanFunctionType.IS_IN and len(node.inputs) == 2:
            base_expr, values = self._eval(node.inputs[0]), self._eval(node.inputs[1])
            if base_expr is None or values is None:
                self._result = None
            else:
                # Polars 1.8 has no ``nulls_equal`` keyword.
                self._result = base_expr.is_in(values, nulls_equal=True) if node.options.get("nulls_equal") else base_expr.is_in(values)
            return
        if ft == BooleanFunctionType.IS_BETWEEN and len(node.inputs) == 3:
            base_expr, lower, upper = (self._eval(n) for n in node.inputs)
            closed_str = str(node.options.get("closed", "both")).lower()
            # Validate closed parameter
            closed_val: Literal["left", "right", "both", "none"] = closed_str if closed_str in ("left", "right", "both", "none") else "both"  # type: ignore[assignment]
            self._result = None if base_expr is None or lower is None or upper is None else base_expr.is_between(lower, upper, closed=closed_val)
            return
        self.default_visit(node)


def translate_polars_predicate(predicate: pl.Expr | None, mapping: dict[str, pl.DataType]) -> pl.Expr | None:
    if predicate is None:
        return None
    node = get_parsed_expr(predicate)
    if node is None:
        return None
    visitor = TranslatedPredicateVisitor(mapping)
    # The original predicate is re-applied after the scan, so an untranslatable top-level conjunct can be dropped.
    conjuncts, parts = [node], []
    while conjuncts:
        cur = conjuncts.pop()
        if isinstance(cur, BinaryExprNode) and cur.op == OperatorType.AND:
            conjuncts += [cur.left, cur.right]
        elif (expr := visitor._eval(cur)) is not None:
            parts.append(expr)
    return reduce(pl.Expr.and_, parts) if parts else None


LOGICAL_MAPPING_META_KEY = b"cpl.logical_mapping.v1"


class LogicalSpec(BaseModel):
    dtype: DataType
    unit: TimeUnit | None = None
    time_zone: str | None = None


class LogicalMappingMetadata(BaseModel):
    version: int = 1
    columns: dict[str, LogicalSpec]


def mapping_to_metadata(mapping: dict[str, pl.DataType]) -> bytes:
    cols: dict[str, LogicalSpec] = {}
    for name, dt in mapping.items():
        ed = DataType.from_polars_dtype(dt)
        if ed == DataType.UNKNOWN:
            raise ValueError(f"Unsupported dtype {dt} for column {name!r} in logical mapping metadata")
        unit = TimeUnit.from_string(getattr(dt, "time_unit", None))
        tz = getattr(dt, "time_zone", None)
        cols[name] = LogicalSpec(dtype=ed, unit=unit, time_zone=tz)
    model = LogicalMappingMetadata(version=1, columns=cols)
    return json.dumps(model.model_dump(mode="json")).encode()


def metadata_to_mapping(meta_bytes: bytes) -> dict[str, pl.DataType]:
    data = json.loads(meta_bytes)
    model = LogicalMappingMetadata.model_validate(data)
    cols: dict[str, pl.DataType] = {}
    for name, spec in model.columns.items():
        base_cls = spec.dtype.get_class()
        if spec.dtype == DataType.DATETIME:
            unit_str = spec.unit.to_datetime_conversion() if spec.unit is not None else None
            kwargs = {}
            if unit_str:
                kwargs["time_unit"] = unit_str
            if spec.time_zone:
                kwargs["time_zone"] = spec.time_zone
            dt = pl.Datetime(**kwargs)
        elif spec.dtype == DataType.DURATION:
            unit_str = spec.unit.to_datetime_conversion() if spec.unit is not None else None
            if unit_str and unit_str in ("ns", "us", "ms"):
                dt = pl.Duration(unit_str)  # type: ignore[arg-type]
            else:
                dt = pl.Duration()
        elif spec.dtype == DataType.TIME:
            dt = pl.Time()
        else:
            dt = base_cls
        cols[name] = dt
    return cols
