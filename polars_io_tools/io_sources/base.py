import datetime
import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import polars as pl
from packaging import version

try:
    from polars._plr import _expr_nodes as en  # polars >= 1.32
except ImportError:
    from polars.polars import _expr_nodes as en  # polars < 1.32
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field, model_validator

from .enum import (  # noqa: F401 -- DataType, Expr and TimeUnit stay importable from this module
    DataType,
    Expr,
    FunctionType,
    GenericFunctionType,
    OperatorType,
    TemporalFunctionType,
    TimeUnit,
    get_function_enum,
)

# Configure logging
log = logging.getLogger(__name__)

# Generic type for visitor results
T = TypeVar("T")

# Sentinel value for failed literal extraction
FAILED_LITERAL_RESULT = object()

# Define public exports
__all__ = (
    "convert_datetime_to_polars",
    "extract_column_name",
    "extract_value_preserving_column_name",
    "has_exact_value",
    "is_value_order_preserving_cast",
)


if version.parse(pl.__version__) < version.parse("1.28.0"):
    warnings.warn(
        "Polars versions < 1.28.0 do not push down casts to custom io sources, including implicit casts. This means that, for example, filters that are comparisons of a datetime column with nanosecond precision with a python datetime object (which has microsecond precision) will not get pushed down."
    )


class BaseExprNode(BaseModel, ABC):
    """Base model for all expression nodes."""

    expr: pl.Expr
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allow pl.Expr type
    )
    can_extract_literal: bool = False

    @property
    def value(self) -> Any:
        """Get the literal value of this node. Only valid if can_extract_literal is True.

        Subclasses that support literal extraction should override this.
        """
        return None

    @abstractmethod
    def accept(self, visitor: "ExprVisitor[T]") -> None:
        """Accept a visitor by dispatching to the appropriate visit method."""

    @abstractmethod
    def get_children(self) -> list["BaseExprNode"]:
        """Get all child nodes."""

    def visit_children(self, visitor: "ExprVisitor[T]") -> None:
        """Visit all child nodes with the given visitor."""
        for node in self.get_children():
            node.accept(visitor)


class ExtractableLiteralNode(BaseExprNode):
    """Base model for nodes that we might be able to extract a literal value from. `value` only has a meaning if `can_extract_literal` is True."""

    _value: Any = PrivateAttr(None)
    _has_set_value: bool = PrivateAttr(False)

    @model_validator(mode="after")
    def validate_can_extact_literal(self):
        # If we can extract a literal from this node
        # all([]) == True
        if all(x.can_extract_literal for x in self.get_children()):
            self.can_extract_literal = True
        return self

    # We add the ignore below for mypy compatibility
    @computed_field  # type: ignore[misc]
    @property
    def value(self) -> Any:
        if not self.can_extract_literal:
            return None
        if self._has_set_value:
            return self._value
        else:
            self._value = get_literal_value(self.expr)
            self._has_set_value = True
            return self._value


class AnonymousFunctionNode(ExtractableLiteralNode):
    input: BaseExprNode

    # function   since it's a udf, we dont encode the raw function
    def accept(self, visitor: "ExprVisitor[T]") -> None:
        visitor.visit_anon_function(self)

    def get_children(self) -> list[BaseExprNode]:
        return [self.input]


class AliasNode(ExtractableLiteralNode):
    """Node representing an alias expression."""

    input: BaseExprNode
    name: str | None = None

    def accept(self, visitor: "ExprVisitor[T]") -> None:
        visitor.visit_alias(self)

    def get_children(self) -> list[BaseExprNode]:
        """Get the input node."""
        return [self.input]


class BinaryExprNode(BaseExprNode):
    """Node representing a binary expression."""

    left: BaseExprNode
    op: OperatorType
    right: BaseExprNode

    def accept(self, visitor: "ExprVisitor[T]") -> None:
        visitor.visit_binary_expr(self)

    def get_children(self) -> list[BaseExprNode]:
        """Get the left and right nodes."""
        return [self.left, self.right]


class CastNode(ExtractableLiteralNode):
    """Node representing a cast operation."""

    input: BaseExprNode
    dtype: pl.DataType
    options: str | None = None
    """One of ``"Strict"``, ``"NonStrict"`` or ``"Overflowing"`` (``wrap_numerical=True``)."""

    def accept(self, visitor: "ExprVisitor[T]") -> None:
        visitor.visit_cast(self)

    def get_children(self) -> list[BaseExprNode]:
        """Get the input node."""
        return [self.input]


class ColumnNode(BaseExprNode):
    """Node representing a column reference."""

    name: str

    def accept(self, visitor: "ExprVisitor[T]") -> None:
        visitor.visit_column(self)

    def get_children(self) -> list[BaseExprNode]:
        """No children to get."""
        return []


class ColumnsNode(BaseExprNode):
    """Node representing multiple column references."""

    names: list[str]

    def accept(self, visitor: "ExprVisitor[T]") -> None:
        visitor.visit_columns(self)

    def get_children(self) -> list[BaseExprNode]:
        """No children to get."""
        return []


class ErrorNode(BaseExprNode):
    """Node representing a parsing error."""

    error: str

    def accept(self, visitor: "ExprVisitor[T]") -> None:
        visitor.visit_error(self)

    def get_children(self) -> list[BaseExprNode]:
        """No children to get."""
        return []


class ExcludeNode(BaseExprNode):
    """Node representing an exclude operation."""

    input: BaseExprNode
    exclude: list[str | pl.DataType]

    def accept(self, visitor: "ExprVisitor[T]") -> None:
        visitor.visit_exclude(self)

    def get_children(self) -> list[BaseExprNode]:
        """Get the input node."""
        return [self.input]


class ExplodeNode(BaseExprNode):
    """Node representing an explode operation."""

    input: BaseExprNode

    def accept(self, visitor: "ExprVisitor[T]") -> None:
        visitor.visit_explode(self)

    def get_children(self) -> list[BaseExprNode]:
        """Get the input node."""
        return [self.input]


class FieldNode(BaseExprNode):
    """Node representing field access in a struct."""

    fields: list[str]

    def accept(self, visitor: "ExprVisitor[T]") -> None:
        visitor.visit_field(self)

    def get_children(self) -> list[BaseExprNode]:
        """No children to get."""
        return []


class FilterNode(BaseExprNode):
    """Node representing a filter operation."""

    input: BaseExprNode
    by: BaseExprNode

    def accept(self, visitor: "ExprVisitor[T]") -> None:
        visitor.visit_filter(self)

    def get_children(self) -> list[BaseExprNode]:
        """Get the input and by nodes."""
        return [self.input, self.by]


class FunctionNode(ExtractableLiteralNode):
    """Node representing a general function call."""

    inputs: list[BaseExprNode]
    function_type: FunctionType
    options: dict[str, Any] = Field(default_factory=dict)

    def accept(self, visitor: "ExprVisitor[T]") -> None:
        visitor.visit_function(self)

    def get_children(self) -> list[BaseExprNode]:
        """Get all input nodes."""
        return self.inputs


class GatherNode(BaseExprNode):
    """Node representing a gather operation."""

    input: BaseExprNode
    idx: BaseExprNode
    returns_scalar: bool = False

    def accept(self, visitor: "ExprVisitor[T]") -> None:
        visitor.visit_gather(self)

    def get_children(self) -> list[BaseExprNode]:
        """Get the input and idx nodes."""
        return [self.input, self.idx]


class KeepNameNode(BaseExprNode):
    """Node representing a keep_name operation."""

    input: BaseExprNode

    def accept(self, visitor: "ExprVisitor[T]") -> None:
        visitor.visit_keep_name(self)

    def get_children(self) -> list[BaseExprNode]:
        """Get the input node."""
        return [self.input]


class LenNode(BaseExprNode):
    """Node representing the len operation."""

    def accept(self, visitor: "ExprVisitor[T]") -> None:
        visitor.visit_len(self)

    def get_children(self) -> list[BaseExprNode]:
        """No children to get."""
        return []


class LiteralNode(ExtractableLiteralNode):
    """Node representing a literal value."""

    can_extract_literal: bool = True  # we can always extract a literal value from this node

    def accept(self, visitor: "ExprVisitor[T]") -> None:
        visitor.visit_literal(self)

    def get_children(self) -> list[BaseExprNode]:
        """No children to get."""
        return []


class NthNode(BaseExprNode):
    """Node representing an nth selection."""

    index: int

    def accept(self, visitor: "ExprVisitor[T]") -> None:
        visitor.visit_nth(self)

    def get_children(self) -> list[BaseExprNode]:
        """No children to get."""
        return []


class SelectorNode(BaseExprNode):
    """Node representing a Selector"""

    # The definition of the different values Selector takes are here:
    # https://github.com/pola-rs/polars/blob/main/crates/polars-plan/src/dsl/selector.rs
    strict: bool | None = None
    names: list[str] = []
    indices: list[int] = []

    def accept(self, visitor: "ExprVisitor[T]") -> None:
        visitor.visit_selector(self)

    def get_children(self) -> list[BaseExprNode]:
        """No children to get."""
        return []


class SliceNode(BaseExprNode):
    """Node representing a slice operation."""

    input: BaseExprNode
    offset: BaseExprNode
    length: BaseExprNode

    def accept(self, visitor: "ExprVisitor[T]") -> None:
        visitor.visit_slice(self)

    def get_children(self) -> list[BaseExprNode]:
        """Get the input, offset, and length nodes."""
        return [self.input, self.offset, self.length]


class SortByNode(BaseExprNode):
    """Node representing a sort_by operation."""

    input: BaseExprNode
    by: list[BaseExprNode]
    options: dict[str, Any] = Field(default_factory=dict)

    def accept(self, visitor: "ExprVisitor[T]") -> None:
        visitor.visit_sort_by(self)

    def get_children(self) -> list[BaseExprNode]:
        """Get the input and by nodes."""
        return [self.input] + self.by


class SortNode(BaseExprNode):
    """Node representing a sort operation."""

    input: BaseExprNode
    options: dict[str, Any] = Field(default_factory=dict)

    def accept(self, visitor: "ExprVisitor[T]") -> None:
        visitor.visit_sort(self)

    def get_children(self) -> list[BaseExprNode]:
        """Get the input node."""
        return [self.input]


class TernaryNode(BaseExprNode):
    """Node representing a ternary condition (if-then-else)."""

    predicate: BaseExprNode
    truthy: BaseExprNode
    falsy: BaseExprNode

    def accept(self, visitor: "ExprVisitor[T]") -> None:
        visitor.visit_ternary(self)

    def get_children(self) -> list[BaseExprNode]:
        """Get the predicate, truthy, and falsy nodes."""
        return [self.predicate, self.truthy, self.falsy]


class UnknownNode(BaseExprNode):
    """Node representing an unrecognized expression type."""

    data: dict[str, Any]

    def accept(self, visitor: "ExprVisitor[T]") -> None:
        visitor.visit_unknown(self)

    def get_children(self) -> list[BaseExprNode]:
        """No children to get."""
        return []


class WildcardNode(BaseExprNode):
    """Node representing a wildcard selector."""

    def accept(self, visitor: "ExprVisitor[T]") -> None:
        visitor.visit_wildcard(self)

    def get_children(self) -> list[BaseExprNode]:
        """No children to get."""
        return []


class ExprVisitor(Generic[T]):
    """
    Base implementation of the visitor interface with default methods.
    All visit methods delegate to default_visit by default.
    """

    def visit(self, node: BaseExprNode) -> None:
        """Entry point for the visitor pattern."""
        node.accept(self)

    def process_results(self) -> T:
        """Process and return the final result."""
        return self.default_result()

    def visit_and_process_results(self, node: BaseExprNode) -> T:
        """Utility function to visit a node and process the results."""
        self.visit(node)
        return self.process_results()

    def default_result(self) -> T:
        """Default result for unhandled node types."""
        return None  # type: ignore

    def default_visit(self, node: BaseExprNode) -> None:
        """Default visit method."""

    def visit_alias(self, node: AliasNode) -> None:
        self.default_visit(node)

    def visit_column(self, node: ColumnNode) -> None:
        self.default_visit(node)

    def visit_columns(self, node: ColumnsNode) -> None:
        self.default_visit(node)

    # def visit_dtype_column(self, node: DtypeColumnNode) -> None:
    #     self.default_visit(node)

    # def visit_index_column(self, node: IndexColumnNode) -> None:
    #     self.default_visit(node)

    def visit_literal(self, node: LiteralNode) -> None:
        self.default_visit(node)

    def visit_binary_expr(self, node: BinaryExprNode) -> None:
        self.default_visit(node)

    def visit_cast(self, node: CastNode) -> None:
        self.default_visit(node)

    def visit_sort(self, node: SortNode) -> None:
        self.default_visit(node)

    def visit_gather(self, node: GatherNode) -> None:
        self.default_visit(node)

    def visit_sort_by(self, node: SortByNode) -> None:
        self.default_visit(node)

    def visit_ternary(self, node: TernaryNode) -> None:
        self.default_visit(node)

    def visit_function(self, node: FunctionNode) -> None:
        self.default_visit(node)

    def visit_explode(self, node: ExplodeNode) -> None:
        self.default_visit(node)

    def visit_filter(self, node: FilterNode) -> None:
        self.default_visit(node)

    def visit_slice(self, node: SliceNode) -> None:
        self.default_visit(node)

    def visit_exclude(self, node: ExcludeNode) -> None:
        self.default_visit(node)

    def visit_keep_name(self, node: KeepNameNode) -> None:
        self.default_visit(node)

    def visit_len(self, node: LenNode) -> None:
        self.default_visit(node)

    def visit_nth(self, node: NthNode) -> None:
        self.default_visit(node)

    def visit_field(self, node: FieldNode) -> None:
        self.default_visit(node)

    def visit_wildcard(self, node: WildcardNode) -> None:
        self.default_visit(node)

    def visit_error(self, node: ErrorNode) -> None:
        self.default_visit(node)

    def visit_unknown(self, node: UnknownNode) -> None:
        self.default_visit(node)

    def visit_anon_function(self, node: AnonymousFunctionNode) -> None:
        self.default_visit(node)

    def visit_selector(self, node: SelectorNode) -> None:
        self.default_visit(node)


# ---------------------------------------------------------------------------
# NodeTraverser-based parser
#
# Built on polars' private ``NodeTraverser`` API (``LazyFrame._ldf.visit()`` plus ``add_expressions`` and ``view_expression``). The traverser exposes
# a typed view of expression nodes after polars' DSL→IR conversion, which is exactly the shape that arrives at a custom IO source plugin via
# ``polars.io.plugins.register_io_source``. Structural traversal is driven by ``expr.meta.pop()`` so each sub-node receives an accurate ``pl.Expr``
# (downstream visitors rely on ``node.expr.meta.root_names()`` and ``~node.expr``); the traverser is used purely for typed dispatch and for direct
# access to ``Operator``, function-category enums, ``Literal.value``/``dtype`` and ``Cast.dtype``.
# ---------------------------------------------------------------------------

_OPERATOR_NAME_MAP: dict[str, OperatorType] = {op.value: op for op in OperatorType}

# (polars typed enum class, category key consumed by `get_function_enum`). Order does not matter; lookup is by isinstance on the head of
# `Function.function_data`. Some classes (e.g. `StructFunction`, added in polars 1.31) may be absent on older supported polars versions;
# `getattr` keeps the table compatible across the supported range and the missing categories fall through to `UNKNOWN`.
_FUNCTION_CATEGORY_MAP: list[tuple[type, str]] = [
    (cls, key)
    for cls, key in [
        (getattr(en, "BooleanFunction", None), "Boolean"),
        (getattr(en, "StringFunction", None), "StringExpr"),
        (getattr(en, "TemporalFunction", None), "TemporalExpr"),
        (getattr(en, "StructFunction", None), "StructExpr"),
    ]
    if cls is not None
]

# NodeTraverser returns lowercase `closed` strings; downstream visitors (range_visitor, dnf_visitor, sql_utils, translated_source) expect
# capitalized variants matching the legacy JSON parser's output.
_CLOSED_MAP = {"both": "Both", "left": "Left", "right": "Right", "none": "None"}
_CAST_OPTIONS_MAP = {0: "Strict", 1: "NonStrict", 2: "Overflowing"}


class NodeTraverserParser:
    """Parse a :class:`pl.Expr` into a :class:`BaseExprNode` tree using polars' ``NodeTraverser`` for typed dispatch and ``expr.meta.pop()``
    for sub-expression attribution.
    """

    def __init__(self) -> None:
        # Dispatch on the polars typed-view class.
        self._parser_map: dict[type, Any] = {
            en.Column: self._parse_column,
            en.Literal: self._parse_literal,
            en.Len: self._parse_len,
            en.Cast: self._parse_cast,
            en.Alias: self._parse_alias,
            en.Sort: self._parse_sort,
            en.BinaryExpr: self._parse_binary_expr,
            en.Filter: self._parse_filter,
            en.Gather: self._parse_gather,
            en.SortBy: self._parse_sort_by,
            en.Ternary: self._parse_ternary,
            en.Slice: self._parse_slice,
            en.Function: self._parse_function_expr,
        }

    def parse(self, expr: pl.Expr) -> BaseExprNode:
        """Parse a Polars expression into structured node types via the NodeTraverser API."""
        try:
            visitor = self._build_visitor(expr)
            return self._parse(expr, visitor)
        except Exception as e:
            log.warning("NodeTraverserParser unexpected failure: %s", e, exc_info=True)
            return ErrorNode(expr=expr, error=str(e))

    def _build_visitor(self, expr: pl.Expr) -> Any:
        schema = {name: pl.Null for name in expr.meta.root_names()}
        return pl.LazyFrame(schema=schema)._ldf.visit()

    def _view(self, visitor: Any, expr: pl.Expr) -> Any | None:
        """Add ``expr`` to the traverser and return its typed view. Returns ``None`` if polars refuses or hasn't implemented the node — caller
        falls back to :class:`UnknownNode`.
        """
        try:
            [node_id], _ = visitor.add_expressions([expr._pyexpr])
        except Exception:  # noqa: BLE001 -- intentional broad catch (defensive fallback)
            return None
        try:
            return visitor.view_expression(node_id)
        except (NotImplementedError, pl.exceptions.PanicException):
            return None

    def _parse(self, expr: pl.Expr, visitor: Any) -> BaseExprNode:
        """Parse a single expression node by dispatching on the polars typed view's class.

        Per-subtree errors are isolated here: a failure in one ``_parse_<kind>`` method collapses that subtree to an :class:`ErrorNode` rather than the whole top-level expression.
        """
        # The typed view skips naming wrappers (alias, name.keep, ...), so skip them here too, or the children taken from
        # ``meta.pop()`` would belong to the wrapper.
        while len(children := expr.meta.pop()) == 1 and children[0].meta.undo_aliases().meta.eq(expr.meta.undo_aliases()):
            expr = children[0]
        obj = self._view(visitor, expr)
        if obj is None:
            # polars' NodeTraverser has no typed view for a few IR nodes that can appear in a pushed-down
            # predicate -- notably list functions (``col.list.*``) and array functions (``col.arr.*``), which
            # raise ``NotImplementedError`` from ``view_expression``. Those subtrees degrade to ``UnknownNode``
            # here, so predicates containing them are not pushed down. Recovering them requires polars to
            # expose ``ListFunction`` / ``ArrayFunction`` typed views *and* a matching entry in
            # ``_FUNCTION_CATEGORY_MAP`` here; it will not resolve automatically from the polars change alone.
            return UnknownNode(expr=expr, data={"reason": "node_traverser_refused"})
        handler = self._parser_map.get(type(obj))
        if handler is None:
            return UnknownNode(expr=expr, data={"node_type": type(obj).__name__})
        try:
            return handler(expr, obj, visitor)
        except Exception as e:  # noqa: BLE001 -- intentional broad catch (defensive fallback)
            return ErrorNode(expr=expr, error=str(e))

    # Leaves
    def _parse_column(self, expr: pl.Expr, obj: Any, visitor: Any) -> BaseExprNode:
        return ColumnNode(expr=expr, name=obj.name)

    def _parse_literal(self, expr: pl.Expr, obj: Any, visitor: Any) -> BaseExprNode:
        return LiteralNode(expr=expr)

    def _parse_len(self, expr: pl.Expr, obj: Any, visitor: Any) -> BaseExprNode:
        return LenNode(expr=expr)

    # Single-input wrappers
    def _parse_cast(self, expr: pl.Expr, obj: Any, visitor: Any) -> BaseExprNode:
        (input_expr,) = expr.meta.pop()
        return CastNode(expr=expr, input=self._parse(input_expr, visitor), dtype=obj.dtype, options=_CAST_OPTIONS_MAP.get(obj.options))

    def _parse_alias(self, expr: pl.Expr, obj: Any, visitor: Any) -> BaseExprNode:
        (input_expr,) = expr.meta.pop()
        return AliasNode(expr=expr, input=self._parse(input_expr, visitor), name=obj.name)

    def _parse_sort(self, expr: pl.Expr, obj: Any, visitor: Any) -> BaseExprNode:
        (input_expr,) = expr.meta.pop()
        return SortNode(expr=expr, input=self._parse(input_expr, visitor), options={"options": obj.options})

    # Two-input
    def _parse_binary_expr(self, expr: pl.Expr, obj: Any, visitor: Any) -> BaseExprNode:
        children = list(reversed(expr.meta.pop()))
        if len(children) != 2:
            return ErrorNode(expr=expr, error=f"BinaryExpr expected 2 children, got {len(children)}")
        op_name = repr(obj.op).rsplit(".", 1)[-1]
        op = _OPERATOR_NAME_MAP.get(op_name)
        if op is None:
            return UnknownNode(expr=expr, data={"unknown_op": op_name})
        return BinaryExprNode(expr=expr, left=self._parse(children[0], visitor), op=op, right=self._parse(children[1], visitor))

    def _parse_filter(self, expr: pl.Expr, obj: Any, visitor: Any) -> BaseExprNode:
        children = list(expr.meta.pop())
        if len(children) < 2:
            return ErrorNode(expr=expr, error="Filter expected 2 children")
        return FilterNode(expr=expr, input=self._parse(children[0], visitor), by=self._parse(children[1], visitor))

    def _parse_gather(self, expr: pl.Expr, obj: Any, visitor: Any) -> BaseExprNode:
        children = list(expr.meta.pop())
        if len(children) < 2:
            return ErrorNode(expr=expr, error="Gather expected 2 children")
        return GatherNode(
            expr=expr,
            input=self._parse(children[0], visitor),
            idx=self._parse(children[1], visitor),
            returns_scalar=bool(obj.scalar),
        )

    def _parse_sort_by(self, expr: pl.Expr, obj: Any, visitor: Any) -> BaseExprNode:
        children = list(expr.meta.pop())
        if len(children) < 2:
            return ErrorNode(expr=expr, error="SortBy expected >=2 children")
        return SortByNode(
            expr=expr,
            input=self._parse(children[0], visitor),
            by=[self._parse(c, visitor) for c in children[1:]],
            options={"sort_options": obj.sort_options},
        )

    # Three-input
    def _parse_ternary(self, expr: pl.Expr, obj: Any, visitor: Any) -> BaseExprNode:
        children = list(expr.meta.pop())
        if len(children) != 3:
            return ErrorNode(expr=expr, error="Ternary expected 3 children")
        predicate_expr, falsy_expr, truthy_expr = children
        return TernaryNode(
            expr=expr,
            predicate=self._parse(predicate_expr, visitor),
            truthy=self._parse(truthy_expr, visitor),
            falsy=self._parse(falsy_expr, visitor),
        )

    def _parse_slice(self, expr: pl.Expr, obj: Any, visitor: Any) -> BaseExprNode:
        children = list(expr.meta.pop())
        if len(children) != 3:
            return ErrorNode(expr=expr, error="Slice expected 3 children")
        return SliceNode(
            expr=expr,
            input=self._parse(children[0], visitor),
            offset=self._parse(children[1], visitor),
            length=self._parse(children[2], visitor),
        )

    # Variadic
    def _parse_function_expr(self, expr: pl.Expr, obj: Any, visitor: Any) -> BaseExprNode:
        child_exprs = list(reversed(expr.meta.pop()))
        fn_enum, options = self._function_type_and_options(obj.function_data)
        return FunctionNode(
            expr=expr,
            function_type=fn_enum,
            inputs=[self._parse(c, visitor) for c in child_exprs],
            options=options,
        )

    def _function_type_and_options(self, function_data: tuple[Any, ...]) -> tuple[FunctionType, dict[str, Any]]:
        """Resolve a ``Function.function_data`` tuple to ``(our function-type enum, options dict)``.

        Only the option keys that downstream consumers actually inspect are populated. In production these are ``closed`` (range/dnf/sql)
        and ``literal`` (sql); ``nulls_equal`` and ``strict`` are emitted for parity with one unmarked test assertion. Everything else
        collapses to ``{}`` — the legacy parser's broader option dict is dead code on the predicate-pushdown path.
        """
        if not function_data:
            return GenericFunctionType.UNKNOWN, {}
        head, args = function_data[0], function_data[1:]
        for polars_cls, category in _FUNCTION_CATEGORY_MAP:
            if isinstance(head, polars_cls):
                name = repr(head).rsplit(".", 1)[-1]
                fn_enum = get_function_enum(category, name) or GenericFunctionType.UNKNOWN
                return fn_enum, _named_function_options(category, name, args)
        if isinstance(head, str):
            name_pascal = "".join(part.capitalize() for part in head.split("_"))
            try:
                return GenericFunctionType(name_pascal), {}
            except ValueError:
                return GenericFunctionType.UNKNOWN, {}
        return GenericFunctionType.UNKNOWN, {}


def _named_function_options(category: str, name: str, args: tuple[Any, ...]) -> dict[str, Any]:
    if not args:
        return {}
    if category == "Boolean" and name == "IsBetween":
        closed = args[0]
        if isinstance(closed, str):
            closed = _CLOSED_MAP.get(closed, closed)
        return {"closed": closed}
    if category == "Boolean" and name == "IsIn":
        return {"nulls_equal": args[0]}
    if category == "StringExpr" and name == "Contains":
        out: dict[str, Any] = {"literal": args[0]}
        if len(args) >= 2:
            out["strict"] = args[1]
        return out
    return {}


def get_literal_value(expr: pl.Expr) -> Any:
    """Extract the value from a Polars expression as a python object. This does not neccessarily have to be a polars literal expression itself.

    Examples:

    >>> get_literal_value(pl.lit([1, 2, 3]))
    [1, 2, 3]

    >>> get_literal_value(pl.lit(12).cast(pl.Int32).alias("foo"))
    12
    """
    try:
        df_empty = pl.DataFrame()
        res = df_empty.select(expr).to_series().to_list()
        if len(res) != 1:
            return res
        return res[0]
    except (Exception, pl.exceptions.PanicException) as e:  # noqa: BLE001 -- intentional broad catch (defensive fallback)
        log.warning(f"Error getting literal value: {e}")
        return FAILED_LITERAL_RESULT  # Use a sentinel to indicate failure


def has_exact_value(node: BaseExprNode, *, collection: bool = False) -> bool:
    """Whether ``node`` is a literal whose ``value`` holds it exactly.

    With ``collection=False`` the literal must be a single scalar; with ``collection=True`` it may be a list or Series of values.
    Literals containing NaN are rejected, as Python NaNs never compare equal. Python temporal values hold microseconds, so a
    nanosecond literal with a sub-microsecond part is not held exactly.
    """
    return _exact_literal(node, collection=collection) is not None


def _exact_literal(node: BaseExprNode, *, collection: bool = False) -> pl.Series | None:
    """The values of ``node`` as a Series if :func:`has_exact_value` holds, else None."""
    if not node.can_extract_literal or node.value is FAILED_LITERAL_RESULT:
        return None
    series = pl.select(node.expr).to_series()
    if isinstance(series.dtype, pl.List):
        if not collection or len(series) != 1:
            return None
        series = series[0]
    elif not collection and len(series) != 1:
        return None
    if series.dtype.is_float() and series.is_nan().any():
        return None
    if series.dtype.is_temporal() and not series.equals(pl.Series(series.to_list(), dtype=series.dtype)):
        return None
    return series


def extract_column_name(node: BaseExprNode) -> str | None:
    """Extract column name from a node, handling aliases and casts.

    This is useful for visitors that need to identify column references
    whether they are direct ColumnNodes or wrapped in AliasNodes/CastNodes.
    Handles nested combinations like cast->alias->alias recursively.

    Args:
        node: The expression node to extract column name from

    Returns:
        Column name if node represents a single column (possibly wrapped),
        None otherwise
    """
    # Check that this expression only references one column
    cur = node
    while True:
        col_names = cur.expr.meta.root_names()
        if len(col_names) != 1:
            return None

        if isinstance(cur, ColumnNode):
            return col_names[0]
        elif isinstance(cur, (AliasNode, CastNode)):
            cur = cur.input
        elif isinstance(cur, FunctionNode) and cur.function_type == TemporalFunctionType.DATE and len(cur.inputs) == 1:
            # ``col.dt.date()`` floors a Datetime to day granularity: order- and domain-preserving, hence equivalent
            # to ``col.cast(pl.Date)`` (unwrapped above). Treat it identically so it resolves wherever a cast would.
            cur = cur.inputs[0]
        else:
            return None


def is_value_order_preserving_cast(src: pl.DataType, tgt: pl.DataType) -> bool:
    """Whether casting ``src`` to ``tgt`` keeps every value distinct and in the same order.

    True for an identity cast, an integer cast whose target range contains the source range, and a decimal cast that
    keeps the scale and does not reduce precision. A predicate on ``col.cast(tgt)`` can only be applied to ``col`` itself
    when this holds.
    """
    # ``is_`` rather than ``==``: an unparameterized dtype such as ``pl.Datetime`` compares equal to any time unit.
    if src.is_(tgt):
        return True
    if src.is_integer() and tgt.is_integer():
        (src_min, src_max), (tgt_min, tgt_max) = (pl.select(lo=d.min(), hi=d.max()).row(0) for d in (src, tgt))
        return tgt_min <= src_min and src_max <= tgt_max
    if isinstance(src, pl.Decimal) and isinstance(tgt, pl.Decimal):
        if (src.scale or 0) != (tgt.scale or 0):
            return False
        return tgt.precision is None or (src.precision is not None and tgt.precision >= src.precision)
    return False


def extract_value_preserving_column_name(node: BaseExprNode, schema: pl.Schema | dict[str, pl.DataType] | None = None) -> str | None:
    """Extract a column name through aliases and value- and order-preserving casts.

    A cast can change values (``cast(pl.Int8, wrap_numerical=True)`` maps 257 to 1), so a predicate on a cast column
    does not in general hold for the raw column. Casts are looked through only when ``schema`` gives the column's dtype
    and every cast in the chain passes :func:`is_value_order_preserving_cast`, as with the supertype cast Polars inserts
    when comparing a ``Decimal(10, 2)`` column with a literal.

    Args:
        node: The expression node to extract a column name from.
        schema: Optional column-name to dtype mapping, required for any cast to be looked through.

    Returns:
        The column name, or ``None`` if the node is not a single column under such wrappers.
    """
    cur = node
    cast_targets: list[pl.DataType] = []
    while True:
        col_names = cur.expr.meta.root_names()
        if len(col_names) != 1:
            return None

        if isinstance(cur, ColumnNode):
            column = col_names[0]
            if not cast_targets:
                return column
            prev = (schema or {}).get(column)
            # Check the cast chain from the column's dtype outward (innermost cast first).
            for target in reversed(cast_targets):
                if prev is None or not is_value_order_preserving_cast(prev, target):
                    return None
                prev = target
            return column
        elif isinstance(cur, AliasNode):
            cur = cur.input
        elif isinstance(cur, CastNode):
            cast_targets.append(cur.dtype)
            cur = cur.input
        else:
            return None


def get_parsed_expr(expr_or_node: pl.Expr | BaseExprNode) -> BaseExprNode | None:
    """Parse a Polars expression or return the node if it's already parsed. This is designed to be run after polars has applied its optimizations, such as the predicate passed to a custom io plugin. Thus, we avoid handling some Polars expressions that cannot be passed as such, for example, like polars window functions or aggregations."""
    if isinstance(expr_or_node, BaseExprNode):
        return expr_or_node
    return NodeTraverserParser().parse(expr_or_node)


def convert_datetime_to_polars(dt_val: datetime.datetime, schema_cast: Any | None = None) -> pl.Expr:
    """Helper function to convert a datetime value to a Polars expression that is eligible for predicate pushdown. This is only needed for polars<1.28"""
    if version.parse(pl.__version__) > version.parse("1.28"):
        new_expr = pl.lit(dt_val, dtype=pl.Datetime(time_unit="ns"))
    else:
        # Related to polars issue: https://github.com/pola-rs/polars/issues/21790
        import polars.polars as inner_pl  # type: ignore[import-not-found]
        from polars._utils.wrap import wrap_expr

        if dt_val.tzinfo is not None:
            raise ValueError("Datetime values with timezone info are not supported yet.")

        # For some reason, we have to cast as 'ns' first...
        new_expr = wrap_expr(inner_pl.lit(dt_val, is_scalar=True, allow_object=False)).cast(dtype=pl.Datetime(time_unit="ns"))
    if schema_cast is not None:
        new_expr = new_expr.cast(dtype=schema_cast)
    return new_expr
