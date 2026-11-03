from __future__ import annotations

import logging
from collections.abc import Collection, Iterator
from typing import TYPE_CHECKING, Any, cast, overload

import polars as pl

from .base import (
    AliasNode,
    BaseExprNode,
    BinaryExprNode,
    CastNode,
    ColumnNode,
    ExprVisitor,
    FunctionNode,
    LiteralNode,
    get_parsed_expr,
    has_exact_value,
    is_value_order_preserving_cast,
)
from .enum import BooleanFunctionType, OperatorType
from .util import collect_lf_in_io_source, optional_deps_error, register_io_source_with_is_pure

try:
    import narwhals as nw
    from narwhals.typing import FrameT
except ImportError as exc:
    raise optional_deps_error("narwhals support") from exc

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ("from_narwhals",)

log = logging.getLogger(__name__)


def get_polars_to_narwhals_type_map() -> dict[pl.DataType, nw.dtypes.DType]:
    type_map = {
        # Integer Types
        pl.Int8: nw.dtypes.Int8,
        pl.Int16: nw.dtypes.Int16,
        pl.Int32: nw.dtypes.Int32,
        pl.Int64: nw.dtypes.Int64,
        # Unsigned Integer Types
        pl.UInt8: nw.dtypes.UInt8,
        pl.UInt16: nw.dtypes.UInt16,
        pl.UInt32: nw.dtypes.UInt32,
        pl.UInt64: nw.dtypes.UInt64,
        # Float Types
        pl.Float32: nw.dtypes.Float32,
        pl.Float64: nw.dtypes.Float64,
        # String Type (Note: Polars uses Utf8, Narwhals uses String)
        pl.Utf8: nw.dtypes.String,
        # Temporal Types
        pl.Date: nw.dtypes.Date,
        pl.Datetime: nw.dtypes.Datetime,
        pl.Duration: nw.dtypes.Duration,
        # Other Types
        pl.Categorical: nw.dtypes.Categorical,
        pl.Decimal: nw.dtypes.Decimal,
        pl.Object: nw.dtypes.Object,
        pl.Struct: nw.dtypes.Struct,
        pl.List: nw.dtypes.List,
        pl.Unknown: nw.dtypes.Unknown,  # Handle Unknown type explicitly
    }
    # Types not supported from V1 stable API
    if hasattr(nw.dtypes, "Time"):
        type_map[pl.Time] = nw.dtypes.Time
    if hasattr(nw.dtypes, "Binary"):
        type_map[pl.Binary] = nw.dtypes.Binary

    return type_map


class _NWBuilder(ExprVisitor[Any | None]):
    """
    Walk the parsed node tree and build a Narwhals expression.

    We *return* the constructed object from `process_results`.
    """

    def __init__(self, schema: pl.Schema | dict[str, pl.DataType] | None = None, object_columns: Collection[str] = ()):
        self.expr: Any | None = None
        self.schema = schema or {}
        self.object_columns = object_columns

    def visit(self, node: BaseExprNode) -> None:
        # Reset so that node types without a visit method translate to None instead of a stale child result.
        self.expr = None
        super().visit(node)

    def default_result(self):
        return self.expr

    def _dtype(self, node: BaseExprNode) -> pl.DataType | None:
        if isinstance(node, ColumnNode):
            return self.schema.get(node.name)
        if isinstance(node, CastNode):
            return node.dtype
        dtype = pl.select(node.expr).to_series().dtype
        return dtype.inner if isinstance(dtype, pl.List) else dtype

    def visit_alias(self, node: AliasNode):
        self.visit(node.input)
        self.expr = self.default_result()

    def visit_literal(self, node: LiteralNode):
        self.expr = nw.lit(node.value) if has_exact_value(node) else None

    def visit_column(self, node: ColumnNode):
        # Backends apply Python semantics to object values (pandas evaluates None | True as False), so object columns are not translated.
        self.expr = None if node.name in self.object_columns else nw.col(node.name)

    def visit_binary_expr(self, node: BinaryExprNode):
        self.visit(node.left)
        left = self.expr
        self.visit(node.right)
        right = self.expr

        self.expr = None
        if left is None or right is None:
            return

        op = node.op
        nan_checks = []
        if op not in (OperatorType.AND, OperatorType.LOGICAL_AND, OperatorType.OR, OperatorType.LOGICAL_OR):
            if not (_is_null_safe(node.left) and _is_null_safe(node.right)):
                return
            left_dtype, right_dtype = self._dtype(node.left), self._dtype(node.right)
            ordering = op in (OperatorType.GT, OperatorType.GT_EQ, OperatorType.LT, OperatorType.LT_EQ)
            if not _backend_compares_like_polars(left_dtype, right_dtype, ordering=ordering):
                return
            # Literals cannot hold NaN here: visit_literal declines them.
            for operand, operand_expr, dtype in ((node.left, left, left_dtype), (node.right, right, right_dtype)):
                if not isinstance(operand, LiteralNode) and dtype.is_float():
                    nan_checks.append(operand_expr.is_nan())

        try:
            if op == OperatorType.EQ:
                self.expr = left == right
            elif op == OperatorType.NOT_EQ:
                self.expr = left != right
            elif op in (OperatorType.EQ_VALIDITY, OperatorType.NOT_EQ_VALIDITY):
                equal_with_nulls = (left == right).fill_null(False) | (left.is_null() & right.is_null())
                self.expr = equal_with_nulls if op == OperatorType.EQ_VALIDITY else ~equal_with_nulls
            elif op == OperatorType.GT:
                self.expr = left > right
            elif op == OperatorType.GT_EQ:
                self.expr = left >= right
            elif op == OperatorType.LT:
                self.expr = left < right
            elif op == OperatorType.LT_EQ:
                self.expr = left <= right
            elif op in (OperatorType.AND, OperatorType.LOGICAL_AND):
                self.expr = left & right
            elif op in (OperatorType.OR, OperatorType.LOGICAL_OR):
                self.expr = left | right
            # Polars sorts NaN above every number and treats NaN == NaN as true, unlike Arrow-backed data, so keep NaN rows.
            if self.expr is not None and op not in (OperatorType.NOT_EQ, OperatorType.NOT_EQ_VALIDITY):
                for nan_check in nan_checks:
                    self.expr = self.expr | nan_check
        except Exception:  # We do this because we want to keep the pushdown safe  # noqa: BLE001 -- intentional broad catch (defensive fallback)
            self.expr = None

    def visit_cast(self, node: CastNode):
        # Backends differ on lossy, non-strict and parameterised casts, so only lossless casts of a column are translated.
        source_dtype = self.schema.get(node.input.name) if isinstance(node.input, ColumnNode) else None
        if source_dtype is None or not is_value_order_preserving_cast(source_dtype, node.dtype):
            return
        self.visit(node.input)
        if not source_dtype.is_(node.dtype):
            target_class = get_polars_to_narwhals_type_map().get(type(node.dtype)) if node.dtype.is_integer() else None
            self.expr = None if target_class is None else self.expr.cast(target_class)

    def visit_function(self, node: FunctionNode):
        ft = node.function_type
        valid_function_types = {BooleanFunctionType.IS_NULL, BooleanFunctionType.IS_NOT_NULL, BooleanFunctionType.IS_IN}

        if ft in valid_function_types:
            # We only care about the first argument (the column)
            arg_node = node.inputs[0]
            if not _is_null_safe(arg_node):
                return
            self.visit(arg_node)
            col_expr = self.expr

            if col_expr is None:
                return

            if ft == BooleanFunctionType.IS_NULL:
                self.expr = col_expr.is_null()
            elif ft == BooleanFunctionType.IS_NOT_NULL:
                self.expr = ~col_expr.is_null()
            elif ft == BooleanFunctionType.IS_IN and len(node.inputs) >= 2:
                val_node = node.inputs[1]
                arg_dtype = self._dtype(arg_node)
                if has_exact_value(val_node, collection=True) and (
                    arg_dtype is None or _backend_compares_like_polars(arg_dtype, self._dtype(val_node), ordering=False)
                ):
                    values = val_node.value
                    if node.options.get("nulls_equal", False):
                        # Polars currently accepts scalar RHS values for is_in, but that usage is deprecated.
                        members = values if isinstance(values, (list, tuple, set, frozenset)) else [values]
                        non_null = [value for value in members if value is not None]
                        membership = col_expr.is_in(non_null).fill_null(False)
                        self.expr = membership | col_expr.is_null() if len(non_null) != len(members) else membership
                    else:
                        self.expr = col_expr.is_in(values)
                else:
                    self.expr = None


def _is_null_safe(node: BaseExprNode) -> bool:
    """Whether the translated node yields null and NaN exactly where Polars does, so testing or comparing it keeps the same rows."""
    return isinstance(node, (ColumnNode, LiteralNode, CastNode))


def _backend_compares_like_polars(left: pl.DataType | None, right: pl.DataType | None, *, ordering: bool) -> bool:
    """Whether non-Polars backends compare values of these dtypes with the same result as Polars."""
    if left is None or right is None:
        return False
    for a, b in ((left, right), (right, left)):
        # Pandas compares temporal values with numbers via their integer representation.
        if a.is_temporal() and b.is_numeric():
            return False
        # Backends may compare unsigned 64-bit values through a signed type, or reject the mix outright.
        if a == pl.UInt64:
            return False
        if isinstance(a, pl.Datetime) and isinstance(b, pl.Datetime) and (a.time_zone is None) != (b.time_zone is None):
            return False
        # Polars orders Enum by category position and Categorical lexically; backends may differ.
        if ordering and isinstance(a, (pl.Enum, pl.Categorical)):
            return False
    return True


def polars_to_nw(pred: pl.Expr, schema: pl.Schema | dict[str, pl.DataType] | None = None, *, object_columns: Collection[str] = ()) -> nw.Expr | None:
    """
    Try to translate `pred` to a Narwhals expression; return None on failure.

    Comparisons and casts involving a column are only translated when ``schema`` gives the column's dtype. Expressions
    that use a column in ``object_columns`` are not translated.
    """
    node: BaseExprNode = get_parsed_expr(pred)
    builder = _NWBuilder(schema, object_columns)
    builder.visit(node)
    return builder.process_results()


def _to_polars(nw_df: nw.DataFrame[Any]) -> pl.DataFrame:
    if nw_df.implementation in (nw.Implementation.PANDAS, nw.Implementation.MODIN, nw.Implementation.CUDF):
        # Filtered pandas-like frames may have a non-RangeIndex; converting
        # through Arrow would turn that index into an extra data column.
        return nw_df.to_polars()
    # The docs specify that the PyCapsule interface is faster for 1-time calls
    try:
        return nw.from_arrow(nw_df, backend="polars").to_native()  # type: ignore[return-value]
    except Exception:
        log.exception("PyCapsule interface failed; falling back to Arrow conversion")
        # pl.from_arrow on a Table always returns DataFrame
        return cast(pl.DataFrame, pl.from_arrow(nw_df.to_arrow()))


_DTYPES_CONVERTED_AS_DECLARED = (
    nw.Boolean,
    nw.Int8,
    nw.Int16,
    nw.Int32,
    nw.Int64,
    nw.UInt8,
    nw.UInt16,
    nw.UInt32,
    nw.UInt64,
    nw.Float32,
    nw.Float64,
    nw.Decimal,
    nw.String,
    nw.Binary,
    nw.Date,
    nw.Time,
)


def _converts_as_declared(dtype: nw.dtypes.DType) -> bool:
    """Whether data of this Narwhals dtype converts to the dtype's Polars equivalent.

    Unknown, Object, Enum, Categorical, 128-bit integers and second time units (which Polars lacks) do not.
    """
    if isinstance(dtype, (nw.List, nw.Array)):
        return _converts_as_declared(dtype.inner)
    if isinstance(dtype, nw.Struct):
        return all(_converts_as_declared(field.dtype) for field in dtype.fields)
    if isinstance(dtype, (nw.Datetime, nw.Duration)):
        return dtype.time_unit != "s"
    return dtype.base_type() in _DTYPES_CONVERTED_AS_DECLARED


def scan_narwhals(obj: Any, fetch_size: int, description: str | None = None) -> pl.LazyFrame:
    """
    Turn an arbitrary Narwhals frame/series/lazyframe into a **Polars LazyFrame**
    so that the rest of the Polars optimisation pipeline can work unchanged.

    Args:
        obj: Anything accepted by `nw.from_native` (pandas, Polars, Arrow, …).
        fetch_size (int): Number of rows to fetch at a time. This is a default needed by the
            source generator function that scan_narwhals wraps (because it is required
            by the Polars IO plugins API). This value will only be used if Polars
            does not pass a value for batch size; if it does, that will be used instead.
            There is no default value for this parameter, because scan_narwhals isn't
            supposed to be called directly.
        description: Optional free-form description of this source instance, attached to its OpenTelemetry span (``explain_detail``).
        **kwargs: Additional arguments for the database connector
    Returns:
        pl.LazyFrame
    """

    # Wrap the incoming object into a Narwhals LazyFrame
    nw_frame: FrameT = nw.from_native(obj).lazy()

    # Schema (deferred so the underlying Narwhals frame's schema is only
    # resolved when Polars actually needs it, at collect time).
    def schema() -> pl.Schema:
        nw_schema = nw_frame.collect_schema()
        from_data = [name for name, dtype in nw_schema.items() if not _converts_as_declared(dtype)]
        declared = nw.Schema({name: dtype for name, dtype in nw_schema.items() if name not in from_data}).to_polars()
        object_columns = [name for name in from_data if isinstance(nw_schema[name], nw.Object)]
        sampled = [name for name in from_data if name not in object_columns]
        resolved = dict(_to_polars(nw_frame.select(sampled).head(1).collect()).schema) if sampled else {}
        # An object column's dtype depends on all of its values. One that does not convert, such as [1, "a"], keeps its first
        # row's dtype so that queries not reading it still run.
        for name in object_columns:
            column = nw_frame.select(name)
            try:
                resolved[name] = _to_polars(column.collect()).schema[name]
            except Exception:  # noqa: BLE001 -- reading the column fails the same way
                resolved[name] = _to_polars(column.head(1).collect()).schema[name]
        return pl.Schema({name: resolved[name] if name in resolved else declared[name] for name in nw_schema})

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        # Start from the original Narwhals lazy frame
        nf = nw_frame

        # If the backend is Polars, delegate to Polars LazyFrame and stream via collect_batches
        if obj.implementation == nw.Implementation.POLARS:
            pl_lf = nf.to_native()  # Polars LazyFrame
            if with_columns is not None:
                pl_lf = pl_lf.select(with_columns)
            if predicate is not None:
                pl_lf = pl_lf.filter(predicate)
            if n_rows is not None:
                pl_lf = pl_lf.head(n_rows)

            bs = batch_size or fetch_size
            try:
                yield from collect_lf_in_io_source(pl_lf, bs)
            except Exception as e:
                err_msg = (
                    f"Failed during collection in Narwhals Polars backend path.\nPolars plan:\n{pl_lf.explain()}\nError: {e.__class__.__name__}:{e}"
                )
                raise RuntimeError(err_msg) from e
            return

        # Non-Polars backend: use Narwhals and convert to Polars DataFrame
        # projection push-down
        if with_columns is not None:
            nf = nf.select(with_columns)

        object_columns = [name for name, dtype in nf.collect_schema().items() if isinstance(dtype, nw.Object)]
        pl_schema = schema() if predicate is not None or object_columns else None

        # predicate push-down
        nw_pred = None
        try:
            nw_pred = None if predicate is None else polars_to_nw(predicate, pl_schema, object_columns=object_columns)
        except Exception:
            log.exception("Failed to translate predicate.")

        def collect(pred: nw.Expr | None) -> pl.DataFrame:
            frame = nf if pred is None else nf.filter(pred)
            if n_rows is not None:  # early stopping, if possible
                frame = frame.head(n_rows)
            # A subset of an object column can convert to a narrower dtype than the whole column, e.g. String when all null.
            df = _to_polars(frame.collect())
            return df.cast({name: pl_schema[name] for name in object_columns if df.schema[name] != pl_schema[name]})

        pl_df = None
        if nw_pred is not None:
            try:
                # Conversion of a filtered frame can infer different dtypes (e.g. empty object columns), so the predicate is
                # reapplied inside the fallback boundary.
                pl_df = collect(nw_pred).filter(predicate)
            except Exception:
                log.exception(f"Failed to apply predicate {nw_pred}; collecting without it.")
        if pl_df is None:
            try:
                pl_df = collect(None)
            except Exception as e:
                err_msg = "Failed to collect lazy narwhals frame."
                if with_columns:
                    err_msg += f"\nWith columns: {with_columns}"
                err_msg += f"\n\nWhile running the above, received error: {e.__class__.__name__}:{e}"
                raise RuntimeError(err_msg) from e
            if predicate is not None:
                pl_df = pl_df.filter(predicate)

        bs = batch_size or fetch_size
        if bs is None:
            yield pl_df
        else:
            yield from pl_df.iter_slices(n_rows=bs)

    return register_io_source_with_is_pure(io_source=source_generator, schema=schema, explain_detail=description)


# from_narwhals takes either a NW lazyframe or a NW dataframe
# if it's a dataframe, just call to_polars and call it a day.
# if lazy, then use scan_narwhals to push things down
@overload
def from_narwhals(obj: nw.DataFrame[Any], fetch_size: int = 10_000, description: str | None = None) -> pl.DataFrame: ...


@overload
def from_narwhals(obj: nw.LazyFrame[Any], fetch_size: int = 10_000, description: str | None = None) -> pl.LazyFrame: ...


def from_narwhals(obj: FrameT, fetch_size: int = 10_000, description: str | None = None) -> pl.DataFrame | pl.LazyFrame:
    """
    Accept either a Narwhals `DataFrame` or `LazyFrame` and hands
    back the equivalent Polars object.

    Args:
        obj: Narwhals DataFrame **or** Narwhals LazyFrame.
        fetch_size (int, default 10_000): Passed through to `scan_narwhals` when `obj` is lazy.
        description: Optional free-form description of this source instance, attached to its OpenTelemetry span (``explain_detail``). Only applies when `obj` is lazy and non-Polars (i.e. registered as an IO source).

    Returns:
        polars.DataFrame        if `obj` is a Narwhals DataFrame
        polars.LazyFrame        if `obj` is a Narwhals LazyFrame
    """
    if not isinstance(obj, (nw.DataFrame, nw.LazyFrame)):
        raise TypeError(f"Expected a Narwhals DataFrame or LazyFrame, got {type(obj)}")

    if obj.implementation == nw.Implementation.POLARS:
        # If the object is already a Polars object, just return it
        # to_native() on a POLARS implementation returns the underlying pl.DataFrame or pl.LazyFrame
        return cast(pl.DataFrame | pl.LazyFrame, obj.to_native())

    if callable(getattr(obj, "collect", None)):
        return scan_narwhals(obj, fetch_size=fetch_size, description=description)
    try:
        return obj.to_polars()  # type: ignore[return-value]
    except AttributeError:
        # Fallback: round-trip via Arrow
        arrow_tbl: pa.Table = obj.to_arrow()
        result = pl.DataFrame(arrow_tbl)
        return result
