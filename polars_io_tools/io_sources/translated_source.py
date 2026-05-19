from __future__ import annotations

import logging
from typing import Dict, Literal, Optional

import orjson
import polars as pl
from pydantic import BaseModel

from polars_io_tools.io_sources.base import (
    BaseExprNode,
    BinaryExprNode,
    CastNode,
    ExprVisitor,
    FunctionNode,
    LiteralNode,
    extract_column_name,
)
from polars_io_tools.io_sources.enum import (
    BooleanFunctionType,
    DataType,
    OperatorType,
    TimeUnit,
)

from .base import get_parsed_expr

log = logging.getLogger(__name__)

__all__ = ["TranslatedPredicateVisitor", "translate_polars_predicate"]


def _to_target_expr(col: pl.Expr, m: Optional[pl.DataType]) -> pl.Expr:
    if m is None:
        return col
    # Cast mapped columns to their target logical dtype; layering of any
    # additional user casts is handled in visit_cast by re-applying the
    # original cast on top of the mapped base.
    if isinstance(m, pl.Datetime):
        tu = m.time_unit
        tz = m.time_zone
        target_dt = pl.Datetime(time_unit=tu, time_zone=tz)
        return col.cast(target_dt, strict=False)
    if isinstance(m, pl.Duration):
        target_dt = pl.Duration(time_unit=m.time_unit)
        return col.cast(target_dt, strict=False)
    if (m == pl.Time) or isinstance(m, pl.Time):
        return col.cast(pl.Time, strict=False)
    return col


class TranslatedPredicateVisitor(ExprVisitor[Optional[pl.Expr]]):
    """Visitor that rewrites predicates to operate on underlying storage types.

    Produces a new pl.Expr (or None if cannot be rewritten) by walking the AST.
    """

    def __init__(self, mapping: dict[str, pl.DataType]):
        self.mapping = mapping
        self._result: Optional[pl.Expr] = None

    # Utilities
    def _eval(self, node: BaseExprNode) -> Optional[pl.Expr]:
        node.accept(self)
        return self._result

    def process_results(self) -> Optional[pl.Expr]:
        return self._result

    def default_visit(self, node: BaseExprNode) -> None:
        # Fallback to the original expression when we cannot or do not rewrite
        self._result = node.expr

    def _strip_literal_casts(self, node: BaseExprNode) -> BaseExprNode:
        """
        Remove cast/strict_cast wrappers that sit on top of literal nodes.

        Why: Polars can automatically apply implicit casts to align the types
        of both sides of a comparison. However, when a literal is explicitly
        wrapped in a cast (e.g. `2025-01-01.strict_cast(Datetime('us'))`), that
        cast may prevent the optimizer from pushing the predicate down into a
        custom IO source. By unwrapping casts that are on top of literals only,
        we minimize the surface area of this rewrite and let Polars handle the
        type alignment. We do NOT touch casts that sit on top of columns or any
        non-literal expressions.

        This function peels off a chain of CastNode(s) as long as the child is
        an extractable-literal node. If the cast wraps a column or anything that
        is not a literal, it is left intact.
        """
        cur = node
        # Only peel when the child subtree can provide a literal value
        while isinstance(cur, CastNode) and getattr(cur.input, "can_extract_literal", False):
            cur = cur.input
        return cur

    # Leaf nodes
    def visit_literal(self, node: LiteralNode) -> None:
        self._result = pl.lit(node.value)

    def visit_column(self, node: "BaseExprNode") -> None:  # ColumnNode
        # type: ignore[override]
        # Cast mapped columns to their target logical dtype inside predicate
        name = getattr(node, "name", None)
        if name is None:
            self._result = node.expr
            return
        m = self.mapping.get(name)
        if m is not None:
            self._result = _to_target_expr(pl.col(name), m)
            return
        self._result = pl.col(name)

    def visit_cast(self, node: CastNode) -> None:
        # Re-apply user casts on top of mapped base columns without undoing
        # the underlying mapping. If the input resolves to a column with a
        # mapping, first cast that column to the target logical dtype, then
        # apply the user cast on top.
        base = self._eval(node.input)
        if base is None:
            self._result = node.expr
            return
        # If input is a simple column with mapping, _eval already applied
        # _to_target_expr. So we simply cast the resulting expression to
        # the requested dtype, preserving the layering.
        try:
            self._result = base.cast(node.dtype, strict=False)
        except Exception as e:
            log.warning(f"Recevied error {str(e)} when applying a cast on {str(base)}")
            self._result = node.expr

    # Binary expressions
    def visit_binary_expr(self, node: BinaryExprNode) -> None:
        # Short-circuit boolean connectors by visiting children
        if node.op in (OperatorType.AND, OperatorType.OR):
            left_expr = self._eval(node.left)
            right_expr = self._eval(node.right)
            log.debug(f"Resolved AND/OR left={left_expr} right={right_expr}")
            if left_expr is None or right_expr is None:
                self._result = None
                return
            self._result = (left_expr & right_expr) if node.op == OperatorType.AND else (left_expr | right_expr)
            return

        # Strip casts that sit on top of literals so that Polars can do
        # implicit casting and we can maximize pushdown opportunities.
        left_node = self._strip_literal_casts(node.left)
        right_node = self._strip_literal_casts(node.right)

        # Note: we intentionally avoid integer-lowering of temporal literals.
        # Casts on columns are kept, and casts on literals are stripped above.

        # Fallback: evaluate both sides with casting applied to mapped columns
        left_expr = self._eval(left_node)
        right_expr = self._eval(right_node)
        log.debug(f"Resolved comparison left={left_expr} right={right_expr}")
        if left_expr is None or right_expr is None:
            self._result = None
            return
        # Ensure literals are cast to the mapped dtype when comparing against a mapped column
        left_col = extract_column_name(left_node)
        right_col = extract_column_name(right_node)
        mapped_dtype = None
        col_side = None
        if left_col and left_col in self.mapping:
            mapped_dtype = self.mapping.get(left_col)
            col_side = "left"
        elif right_col and right_col in self.mapping:
            mapped_dtype = self.mapping.get(right_col)
            col_side = "right"
        if mapped_dtype is not None:
            # If the opposite side is a literal, cast it to the mapped dtype
            if left_node.can_extract_literal and col_side == "right":
                right_expr = _to_target_expr(pl.col(right_col), mapped_dtype)  # ensure column cast first
                lv = pl.lit(left_node.value).cast(mapped_dtype, strict=False)
                left_expr = lv
            elif right_node.can_extract_literal and col_side == "left":
                left_expr = _to_target_expr(pl.col(left_col), mapped_dtype)
                rv = pl.lit(right_node.value).cast(mapped_dtype, strict=False)
                right_expr = rv
        if node.op in (OperatorType.EQ, OperatorType.EQ_VALIDITY):
            self._result = left_expr == right_expr
            return
        if node.op in (OperatorType.NOT_EQ, OperatorType.NOT_EQ_VALIDITY):
            self._result = left_expr != right_expr
            return
        if node.op == OperatorType.GT:
            self._result = left_expr > right_expr
            return
        if node.op == OperatorType.GT_EQ:
            self._result = left_expr >= right_expr
            return
        if node.op == OperatorType.LT:
            self._result = left_expr < right_expr
            return
        if node.op == OperatorType.LT_EQ:
            self._result = left_expr <= right_expr
            return
        self._result = node.expr

    # Functions
    def visit_function(self, node: FunctionNode) -> None:
        ft = node.function_type
        if isinstance(ft, BooleanFunctionType):
            # is_null / is_not_null
            if ft == BooleanFunctionType.IS_NULL:
                col_name = extract_column_name(node.inputs[0])
                self._result = pl.col(col_name).is_null() if col_name else node.expr
                return
            if ft == BooleanFunctionType.IS_NOT_NULL:
                col_name = extract_column_name(node.inputs[0])
                self._result = pl.col(col_name).is_not_null() if col_name else node.expr
                return
            if ft == BooleanFunctionType.NOT and len(node.inputs) >= 1:
                # Negation: rewrite inner expression, then negate
                inner = self._eval(node.inputs[0])
                self._result = (~inner) if inner is not None else node.expr
                return
            if ft == BooleanFunctionType.IS_IN and len(node.inputs) >= 2:
                # Evaluate the column/input side with mapping-applied casts,
                # and explicitly align RHS literal list to the same logical dtype.
                base_expr = self._eval(node.inputs[0])
                col_name = extract_column_name(node.inputs[0])
                vals = node.inputs[1].value if getattr(node.inputs[1], "can_extract_literal", False) else None
                if base_expr is not None and isinstance(vals, (list, tuple, set)):
                    self._result = base_expr.is_in(list(vals))
                    return
                self._result = node.expr
                return
            if ft == BooleanFunctionType.IS_BETWEEN and len(node.inputs) >= 3:
                # Use mapped base column expression and let Polars do type alignment
                base_expr = self._eval(node.inputs[0])
                lower = node.inputs[1].value if getattr(node.inputs[1], "can_extract_literal", False) else None
                upper = node.inputs[2].value if getattr(node.inputs[2], "can_extract_literal", False) else None
                closed_str = str(node.options.get("closed", "both")).lower()
                # Validate closed parameter
                closed_val: Literal["left", "right", "both", "none"] = closed_str if closed_str in ("left", "right", "both", "none") else "both"  # type: ignore[assignment]
                if base_expr is not None and lower is not None and upper is not None:
                    self._result = base_expr.is_between(lower, upper, closed=closed_val)
                    return
                self._result = node.expr
                return

        # default: keep original
        self._result = node.expr


def translate_polars_predicate(predicate: Optional[pl.Expr], mapping: dict[str, pl.DataType]) -> Optional[pl.Expr]:
    if predicate is None:
        return None
    node = get_parsed_expr(predicate)
    if node is None:
        return None
    visitor = TranslatedPredicateVisitor(mapping)
    node.accept(visitor)
    return visitor.process_results()


LOGICAL_MAPPING_META_KEY = b"cpl.logical_mapping.v1"


class LogicalSpec(BaseModel):
    dtype: DataType
    unit: Optional[TimeUnit] = None
    time_zone: Optional[str] = None


class LogicalMappingMetadata(BaseModel):
    version: int = 1
    columns: Dict[str, LogicalSpec]


def mapping_to_metadata(mapping: dict[str, pl.DataType]) -> bytes:
    cols: Dict[str, LogicalSpec] = {}
    for name, dt in mapping.items():
        ed = DataType.from_polars_dtype(dt)
        unit = TimeUnit.from_string(getattr(dt, "time_unit", None))
        tz = getattr(dt, "time_zone", None)
        cols[name] = LogicalSpec(dtype=ed, unit=unit, time_zone=tz)
    model = LogicalMappingMetadata(version=1, columns=cols)
    return orjson.dumps(model.model_dump())


def metadata_to_mapping(meta_bytes: bytes) -> dict[str, pl.DataType]:
    data = orjson.loads(meta_bytes)
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
