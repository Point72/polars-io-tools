from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Union, cast, overload

import narwhals as nw
import polars as pl
from narwhals.typing import FrameT

if TYPE_CHECKING:
    import pyarrow as pa

from .base import AliasNode, BaseExprNode, BinaryExprNode, CastNode, ColumnNode, ExprVisitor, FunctionNode, LiteralNode, get_parsed_expr
from .enum import BooleanFunctionType, OperatorType
from .util import collect_lf_in_io_source, register_io_source_with_is_pure

__all__ = ["from_narwhals"]

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


class _NWBuilder(ExprVisitor[Optional[Any]]):
    """
    Walk the parsed node tree and build a Narwhals expression.

    We *return* the constructed object from `process_results`.
    """

    def __init__(self):
        self.expr: Optional[Any] = None

    def default_result(self):
        return self.expr

    def visit_alias(self, node: AliasNode):
        self.visit(node.input)
        self.expr = self.default_result()

    def visit_literal(self, node: LiteralNode):
        self.expr = nw.lit(node.value)

    def visit_column(self, node: ColumnNode):
        self.expr = nw.col(node.name)

    def visit_binary_expr(self, node: BinaryExprNode):
        self.visit(node.left)
        left = self.expr
        self.visit(node.right)
        right = self.expr

        if left is None or right is None:
            self.expr = None
            return

        op = node.op
        try:
            if op in (OperatorType.EQ, OperatorType.EQ_VALIDITY):
                self.expr = left == right
            elif op in (OperatorType.NOT_EQ, OperatorType.NOT_EQ_VALIDITY):
                self.expr = left != right
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
            else:
                self.expr = None
        except Exception:  # We do this because we want to keep the pushdown safe
            self.expr = None

    def visit_cast(self, node: CastNode):
        # we visit internally first
        self.visit(node.input)
        if node.dtype != pl.Boolean and self.expr is not None:
            type_map = get_polars_to_narwhals_type_map()
            target_class = type_map.get(type(node.dtype), nw.dtypes.String)
            log.debug(f"Mapping {node.dtype.__class__} to {target_class} for Cast")
            # TODO: Remove default string cast (and also in the lazy SQL reader)
            self.expr = self.expr.cast(type_map.get(type(node.dtype), nw.dtypes.String))

    def visit_function(self, node: FunctionNode):
        ft = node.function_type
        valid_function_types = set([BooleanFunctionType.IS_NULL, BooleanFunctionType.IS_NOT_NULL, BooleanFunctionType.IS_IN])

        if ft in valid_function_types:
            # We only care about the first argument (the column)
            arg_node = node.inputs[0]
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
                if val_node.can_extract_literal:
                    self.expr = col_expr.is_in(val_node.value)
                else:
                    self.expr = None
        else:
            self.expr = None


def polars_to_nw(pred: pl.Expr) -> Optional[nw.Expr]:
    """
    Try to translate `pred` to a Narwhals expression; return None on failure.
    """
    node: BaseExprNode = get_parsed_expr(pred)
    builder = _NWBuilder()
    builder.visit(node)
    return builder.process_results()


def scan_narwhals(obj: Any, fetch_size: int) -> pl.LazyFrame:
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
        **kwargs: Additional arguments for the database connector
    Returns:
        pl.LazyFrame
    """

    # Wrap the incoming object into a Narwhals LazyFrame
    nw_frame: FrameT = nw.from_native(obj).lazy()

    # Schema (deferred so the underlying Narwhals frame's schema is only
    # resolved when Polars actually needs it, at collect time).
    def schema() -> pl.Schema:
        return nw_frame.collect_schema().to_polars()

    def source_generator(
        with_columns: Optional[List[str]],
        predicate: Optional[pl.Expr],
        n_rows: Optional[int],
        batch_size: Optional[int],
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

        # predicate push-down
        try:
            nw_pred = None if predicate is None else polars_to_nw(predicate)
            nf = nf if nw_pred is None else nf.filter(nw_pred)
        except Exception:
            log.exception("Failed to translate predicate.")

        if n_rows is not None:  # early stopping, if possible
            nf = nf.head(n_rows)

        try:
            nw_df = nf.collect()
        except Exception as e:
            err_msg = "Failed to collect lazy narwhals frame."
            if nw_pred is not None:
                err_msg += f"\nWith predicate: {nw_pred}"
            if with_columns:
                err_msg += f"\nWith columns: {with_columns}"
            err_msg += f"\n\nWhile running the above, received error: {e.__class__.__name__}:{e}"
            raise RuntimeError(err_msg) from e

        # The docs specify that the PyCapsule interface is faster for 1-time calls
        pl_df: pl.DataFrame
        try:
            pl_df = nw.from_arrow(nw_df, backend="polars").to_native()  # type: ignore[assignment]
        except Exception:
            log.exception("PyCapsule interface failed; falling back to Arrow conversion")
            result = pl.from_arrow(nw_df.to_arrow())
            # pl.from_arrow on a Table always returns DataFrame
            pl_df = cast(pl.DataFrame, result)

        if predicate is not None:
            pl_df = pl_df.filter(predicate)

        bs = batch_size or fetch_size
        if bs is None:
            yield pl_df
        else:
            yield from pl_df.iter_slices(n_rows=bs)

    return register_io_source_with_is_pure(io_source=source_generator, schema=schema)


# from_narwhals takes either a NW lazyframe or a NW dataframe
# if it's a dataframe, just call to_polars and call it a day.
# if lazy, then use scan_narwhals to push things down
@overload
def from_narwhals(obj: nw.DataFrame[Any], fetch_size: int = 10_000) -> pl.DataFrame: ...


@overload
def from_narwhals(obj: nw.LazyFrame[Any], fetch_size: int = 10_000) -> pl.LazyFrame: ...


def from_narwhals(obj: FrameT, fetch_size: int = 10_000) -> Union[pl.DataFrame, pl.LazyFrame]:
    """
    Accept either a Narwhals `DataFrame` or `LazyFrame` and hands
    back the equivalent Polars object.

    Args:
        obj: Narwhals DataFrame **or** Narwhals LazyFrame.
        fetch_size (int, default 10_000): Passed through to `scan_narwhals` when `obj` is lazy.

    Returns:
        polars.DataFrame        if `obj` is a Narwhals DataFrame
        polars.LazyFrame        if `obj` is a Narwhals LazyFrame
    """
    if not isinstance(obj, (nw.DataFrame, nw.LazyFrame)):
        raise TypeError(f"Expected a Narwhals DataFrame or LazyFrame, got {type(obj)}")

    if obj.implementation == nw.Implementation.POLARS:
        # If the object is already a Polars object, just return it
        # to_native() on a POLARS implementation returns the underlying pl.DataFrame or pl.LazyFrame
        return cast(Union[pl.DataFrame, pl.LazyFrame], obj.to_native())

    if callable(getattr(obj, "collect", None)):
        return scan_narwhals(obj, fetch_size=fetch_size)
    try:
        return obj.to_polars()  # type: ignore[return-value]
    except AttributeError:
        # Fallback: round-trip via Arrow
        arrow_tbl: pa.Table = obj.to_arrow()
        result = pl.DataFrame(arrow_tbl)
        return result
