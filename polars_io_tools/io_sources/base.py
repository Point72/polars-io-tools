import datetime
import logging
import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import orjson
import polars as pl
from packaging import version

try:
    from polars._plr import _expr_nodes as en  # polars >= 1.32
except ImportError:
    from polars.polars import _expr_nodes as en  # polars < 1.32
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field, model_validator

from .enum import (
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
__all__ = [
    "convert_datetime_to_polars",
    "extract_column_name",
]


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
    # options: Dict[str, Any] = Field(default_factory=dict)

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


class ExprParser:
    """
    Advanced parser that converts Polars expressions to a structured node hierarchy.
    This parser handles a comprehensive range of expression types based on the Rust implementation. It is designed to be called after polars optimizations have been applied, such as when registering a custom io plugin. Thus, it only covers a subset of the full expression set from polars.
    """

    def __init__(self):
        # Define the parser map using Expr enum values as keys
        self._parser_map = {
            Expr.BinaryExpr: self._parse_binary_expr,
            Expr.Column: self._parse_column,
            Expr.Columns: self._parse_columns,
            Expr.Literal: self._parse_literal,
            Expr.Alias: self._parse_alias,
            Expr.Function: self._parse_function_expr,
            Expr.Cast: self._parse_cast,
            Expr.Sort: self._parse_sort,
            Expr.Gather: self._parse_gather,
            Expr.SortBy: self._parse_sort_by,
            Expr.Ternary: self._parse_ternary,
            Expr.Explode: self._parse_explode,
            Expr.Filter: self._parse_filter,
            Expr.Slice: self._parse_slice,
            Expr.Exclude: self._parse_exclude,
            Expr.KeepName: self._parse_keep_name,
            Expr.Len: self._parse_len,
            Expr.Nth: self._parse_nth,
            Expr.Field: self._parse_field,
            Expr.Wildcard: self._parse_wildcard,
            Expr.AnonymousFunction: self._parse_anon_function,
            Expr.Selector: self._parse_selector,
        }

    def parse(self, expr: pl.Expr) -> BaseExprNode:
        """Parse a Polars expression into structured node types using a map-based approach."""
        try:
            # Get the serialized representation
            expr_json = expr.meta.serialize(format="json")
            expr_dict = orjson.loads(expr_json)

            # Check if expr_dict contains exactly one key
            if len(expr_dict) == 1:
                expr_type_str = next(iter(expr_dict))
                try:
                    # Convert string to Expr enum value
                    expr_type = Expr(expr_type_str)
                    if expr_type in self._parser_map:
                        # Call the appropriate parser function
                        return self._parser_map[expr_type](expr, expr_dict)
                except ValueError:
                    # Invalid expression type (not in Expr enum)
                    pass

            # If we reach here, either the dict doesn't have exactly one key
            # or the expression type isn't in our parser map
            return UnknownNode(expr=expr, data=expr_dict)

        except Exception as e:
            log.warning(f"Error parsing expression: {e}", exc_info=True)
            return ErrorNode(expr=expr, error=str(e))

    # Simple parser methods
    def _parse_column(self, expr: pl.Expr, expr_dict: dict) -> BaseExprNode:
        """Parse a column expression."""
        return ColumnNode(expr=expr, name=expr_dict["Column"])

    def _parse_columns(self, expr: pl.Expr, expr_dict: dict) -> BaseExprNode:
        """Parse a columns expression."""
        return ColumnsNode(expr=expr, names=expr_dict["Columns"])

    def _parse_literal(self, expr: pl.Expr, expr_dict: dict) -> BaseExprNode:
        """Parse a literal expression."""
        return LiteralNode(expr=expr)

    def _parse_len(self, expr: pl.Expr, expr_dict: dict) -> BaseExprNode:
        """Parse a len expression."""
        return LenNode(expr=expr)

    def _parse_wildcard(self, expr: pl.Expr, expr_dict: dict) -> BaseExprNode:
        """Parse a wildcard expression."""
        return WildcardNode(expr=expr)

    def _parse_selector(self, expr: pl.Expr, expr_dict: dict) -> BaseExprNode:
        core_dict = expr_dict.pop(Expr.Selector.name)
        if "ByName" in core_dict:
            vals = core_dict["ByName"]
            names = vals["names"]
            strict = vals["strict"]
            return SelectorNode(expr=expr, names=names, strict=strict)
        elif "ByIndex" in core_dict:
            vals = core_dict["ByName"]
            indices = vals["indices"]
            strict = vals["strict"]
            return SelectorNode(expr=expr, indices=indices, strict=strict)
        else:
            raise ValueError(f"Unhandled selector {expr = } with serialization {expr_dict}")

    def _parse_binary_expr(self, expr: pl.Expr, expr_dict: dict) -> BaseExprNode:
        """Parse binary expressions (comparisons, arithmetic, and logical operators)."""
        try:
            binary_expr = expr_dict.get("BinaryExpr", {})
            op_str = binary_expr.get("op") if isinstance(binary_expr, dict) else binary_expr

            # Handle case where op is None or not a string
            if not op_str or not isinstance(op_str, str):
                return UnknownNode(expr=expr, data={"error": "Invalid operation type"})

            # Convert the string to our enum type
            try:
                op = OperatorType(op_str)
            except ValueError:
                log.warning(f"Unknown binary operator: {op_str}")
                return UnknownNode(expr=expr, data={"unknown_op": op_str})

            # Get operands (reverse to preserve original order)
            inputs = list(reversed(expr.meta.pop()))
            if len(inputs) != 2:
                return ErrorNode(expr=expr, error=f"Expected 2 inputs for binary expression, got {len(inputs)}")

            left_expr, right_expr = inputs
            left_node = self.parse(left_expr)
            right_node = self.parse(right_expr)

            # Create a proper BinaryExprNode
            return BinaryExprNode(expr=expr, left=left_node, op=op, right=right_node)

        except Exception as e:  # noqa: BLE001 -- intentional broad catch (defensive fallback)
            log.warning(f"Error parsing binary expression: {e}")
            return ErrorNode(expr=expr, error=f"Binary expression parsing error: {e!s}")

    def _parse_alias(self, expr: pl.Expr, expr_dict: dict) -> BaseExprNode:
        """Parse an alias expression."""
        # Extract the alias name
        alias_info = expr_dict.get("Alias", {})
        alias_name = None
        if len(alias_info) > 1:
            alias_name = alias_info[1]  # the name is the second argument

        # Get the inner expression
        inputs = list(expr.meta.pop())
        if not inputs:
            # In polars 1.31.0-beta.1 .meta.pop does not work properly for alias expressions.
            # When that is fixed, we should remove this check.
            root_names = expr.meta.root_names()
            if len(root_names) != 1:
                return ErrorNode(expr=expr, error="No input expression found for alias")
            inner_expr = pl.col(root_names[0])  # Use the root name as the inner expression
        else:
            inner_expr = inputs[0]
        inner_node = self.parse(inner_expr)

        return AliasNode(expr=expr, input=inner_node, name=alias_name)

    def _parse_function_expr(self, expr: pl.Expr, expr_dict: dict) -> BaseExprNode:
        """Parse function expressions using a simplified approach with a single function node type."""
        try:
            func_dict = expr_dict.get("Function", {})

            # Get function inputs (reverse to preserve original order)
            inputs = list(reversed(expr.meta.pop()))
            input_nodes = [self.parse(input_expr) for input_expr in inputs]

            # Check for typed functions
            if "function" in func_dict:
                function_info = func_dict["function"]

                # Handle direct function names (like FillNull) and categorized functions
                if isinstance(function_info, str):
                    # Direct function name (e.g., "FillNull")
                    function_name = function_info
                    options = {}

                    # Check if it's a known generic function
                    try:
                        function_enum = GenericFunctionType(function_name)
                    except ValueError:
                        function_enum = GenericFunctionType.UNKNOWN

                    return FunctionNode(expr=expr, function_type=function_enum, inputs=input_nodes, options=options)

                elif isinstance(function_info, dict):
                    # Categorized function (e.g., {"StringExpr": "Contains"})
                    category = next(iter(function_info.keys()), None)

                    if category is not None:
                        # Get the function data associated with this category
                        func_data = function_info[category]

                        # Extract function name and options
                        if isinstance(func_data, dict):
                            # The function data is a dict with one key (function name) and value (options)
                            function_name = next(iter(func_data.keys()), "")
                            if isinstance(func_data[function_name], dict):
                                options = func_data[function_name]
                            else:
                                # The "option" is just a single value
                                # We include it to preserve the information
                                # even though the structure is a bit altered now
                                # (we provide the key ourselves)
                                options = func_data
                        else:
                            # The function data is just a string (the function name)
                            function_name = str(func_data)
                            options = {}

                        # Use our helper to get the corresponding enum
                        function_enum = get_function_enum(category, function_name)
                        if function_enum is None:
                            function_enum = GenericFunctionType.UNKNOWN

                        # Create a unified function node
                        return FunctionNode(expr=expr, function_type=function_enum, inputs=input_nodes, options=options)

                # Fallback for unknown cases
                return FunctionNode(expr=expr, function_type=GenericFunctionType.UNKNOWN, inputs=input_nodes)

            # Fallback if "function" key not in func_dict
            return FunctionNode(expr=expr, function_type=GenericFunctionType.UNKNOWN, inputs=input_nodes)

        except Exception as e:
            log.warning(f"Error parsing function expression: {e}", exc_info=True)
            return ErrorNode(expr=expr, error=str(e))

    def _parse_cast(self, expr: pl.Expr, expr_dict: dict) -> BaseExprNode:
        """Parse a cast expression."""
        try:
            cast_info = expr_dict.get("Cast", {})

            # Get the input expression
            inputs = list(expr.meta.pop())
            if not inputs:
                return ErrorNode(expr=expr, error="No input found for cast operation")

            input_expr = inputs[0]
            input_node = self.parse(input_expr)

            # Extract dtype and options
            dtype = cast_info.get("dtype") if isinstance(cast_info, dict) else cast_info
            # Convert dtype string to polars DataType if needed
            # HERE We handle the case with DataTypeExpr
            # https://github.com/pola-rs/polars/blob/383f1b37a0a313815fa94b5ede5450e4d7a73f33/crates/polars-plan/src/dsl/datatype_expr.rs#L14
            if isinstance(dtype, dict):
                if "Literal" in dtype:
                    # If it's a literal, we can extract the type directly
                    dtype = dtype["Literal"]
                elif "OfExpr" in dtype:
                    return ErrorNode(expr=expr, error="DataTypeExpr not handled yet.")

            if isinstance(dtype, str):
                pos_args = None
            elif isinstance(dtype, dict):
                dtype, pos_args = next(iter(dtype.items()))
            else:
                return ErrorNode(expr=expr, error="Invalid dtype format")

            dtype = DataType(dtype).get_class()
            if dtype is pl.Datetime and pos_args:
                # TODO: This is a hack to handle datetime parsing
                pos_args[0] = TimeUnit(pos_args[0]).to_datetime_conversion()
            # pl.Unknown cannot be called with arguments
            if pos_args and dtype != pl.Unknown:
                dtype = dtype(*pos_args)
            else:
                dtype = dtype()

            return CastNode(
                expr=expr,
                input=input_node,
                dtype=dtype,
            )

        except Exception as e:
            log.warning(f"Error parsing Cast expression: {e}", exc_info=True)
            return ErrorNode(expr=expr, error=f"Cast parsing error: {e!s}")

    def _parse_sort(self, expr: pl.Expr, expr_dict: dict) -> BaseExprNode:
        """Parse a sort expression."""
        try:
            sort_info = expr_dict.get("Sort", {})

            # Get the input expression
            inputs = list(expr.meta.pop())
            if not inputs:
                return ErrorNode(expr=expr, error="No input found for sort operation")

            input_expr = inputs[0]
            input_node = self.parse(input_expr)

            # Extract options
            options = {}
            if isinstance(sort_info, dict):
                options = sort_info.get("options", {})

            return SortNode(expr=expr, input=input_node, options=options)

        except Exception as e:
            log.warning(f"Error parsing Sort expression: {e}", exc_info=True)
            return ErrorNode(expr=expr, error=f"Sort parsing error: {e!s}")

    def _parse_gather(self, expr: pl.Expr, expr_dict: dict) -> BaseExprNode:
        """Parse a gather expression."""
        try:
            gather_info = expr_dict.get("Gather", {})

            # Get the input expressions
            inputs = list(expr.meta.pop())
            if len(inputs) < 2:
                return ErrorNode(expr=expr, error="Not enough inputs for gather operation")

            input_expr = inputs[0]
            idx_expr = inputs[1]

            input_node = self.parse(input_expr)
            idx_node = self.parse(idx_expr)

            # Extract returns_scalar option
            returns_scalar = False
            if isinstance(gather_info, dict):
                returns_scalar = gather_info.get("returns_scalar", False)

            return GatherNode(expr=expr, input=input_node, idx=idx_node, returns_scalar=returns_scalar)

        except Exception as e:
            log.warning(f"Error parsing Gather expression: {e}", exc_info=True)
            return ErrorNode(expr=expr, error=f"Gather parsing error: {e!s}")

    def _parse_sort_by(self, expr: pl.Expr, expr_dict: dict) -> BaseExprNode:
        """Parse a sort_by expression."""
        try:
            sort_by_info = expr_dict.get("SortBy", {})

            # Get all inputs
            inputs = list(expr.meta.pop())
            if len(inputs) < 2:  # Need at least one expression and one sort key
                return ErrorNode(expr=expr, error="Not enough inputs for sort_by operation")

            # First input is the expression to sort
            input_expr = inputs[0]
            input_node = self.parse(input_expr)

            # Rest are the sort keys
            by_nodes = [self.parse(key_expr) for key_expr in inputs[1:]]

            # Extract options
            options = {}
            if isinstance(sort_by_info, dict):
                options = sort_by_info.get("sort_options", {})

            return SortByNode(expr=expr, input=input_node, by=by_nodes, options=options)

        except Exception as e:
            log.warning(f"Error parsing SortBy expression: {e}", exc_info=True)
            return ErrorNode(expr=expr, error=f"SortBy parsing error: {e!s}")

    def _parse_ternary(self, expr: pl.Expr, expr_dict: dict) -> BaseExprNode:
        """Parse a ternary expression."""
        try:
            # Get all three inputs: predicate, truthy, falsy
            # We don't reverse here
            inputs = list(expr.meta.pop())
            if len(inputs) != 3:
                return ErrorNode(expr=expr, error="Expected 3 inputs for ternary expression")

            # Note the order!
            predicate_expr, falsy_expr, truthy_expr = inputs

            predicate_node = self.parse(predicate_expr)
            truthy_node = self.parse(truthy_expr)
            falsy_node = self.parse(falsy_expr)

            return TernaryNode(expr=expr, predicate=predicate_node, truthy=truthy_node, falsy=falsy_node)

        except Exception as e:  # noqa: BLE001 -- intentional broad catch (defensive fallback)
            log.warning(f"Error parsing Ternary expression: {e}")
            return ErrorNode(expr=expr, error=f"Ternary parsing error: {e!s}")

    def _parse_explode(self, expr: pl.Expr, expr_dict: dict) -> BaseExprNode:
        """Parse an explode expression."""
        try:
            # Get the input expression
            inputs = list(expr.meta.pop())
            if not inputs:
                return ErrorNode(expr=expr, error="No input found for explode operation")

            input_expr = inputs[0]
            input_node = self.parse(input_expr)

            return ExplodeNode(expr=expr, input=input_node)

        except Exception as e:
            log.warning(f"Error parsing Explode expression: {e}", exc_info=True)
            return ErrorNode(expr=expr, error=f"Explode parsing error: {e!s}")

    def _parse_filter(self, expr: pl.Expr, expr_dict: dict) -> BaseExprNode:
        """Parse a filter expression."""
        try:
            # Get the input expressions
            inputs = list(expr.meta.pop())
            if len(inputs) < 2:
                return ErrorNode(expr=expr, error="Not enough inputs for filter operation")

            input_expr = inputs[0]
            by_expr = inputs[1]

            input_node = self.parse(input_expr)
            by_node = self.parse(by_expr)

            return FilterNode(expr=expr, input=input_node, by=by_node)

        except Exception as e:
            log.warning(f"Error parsing Filter expression: {e}", exc_info=True)
            return ErrorNode(expr=expr, error=f"Filter parsing error: {e!s}")

    def _parse_slice(self, expr: pl.Expr, expr_dict: dict) -> BaseExprNode:
        """Parse a slice expression."""
        try:
            # Get the input expressions
            inputs = list(expr.meta.pop())
            if len(inputs) < 3:  # Need input, offset, and length
                return ErrorNode(expr=expr, error="Not enough inputs for slice operation")

            input_expr = inputs[0]
            offset_expr = inputs[1]
            length_expr = inputs[2]

            input_node = self.parse(input_expr)
            offset_node = self.parse(offset_expr)
            length_node = self.parse(length_expr)

            return SliceNode(expr=expr, input=input_node, offset=offset_node, length=length_node)

        except Exception as e:
            log.warning(f"Error parsing Slice expression: {e}", exc_info=True)
            return ErrorNode(expr=expr, error=f"Slice parsing error: {e!s}")

    def _parse_exclude(self, expr: pl.Expr, expr_dict: dict) -> BaseExprNode:
        """Parse an exclude expression."""
        try:
            # Get the input expression
            inputs = list(expr.meta.pop())
            if not inputs or len(inputs) < 2:  # Need at least one input and one exclude parameter
                return ErrorNode(expr=expr, error="Not enough inputs for exclude operation")

            input_expr = inputs[0]
            input_node = self.parse(input_expr)

            # Rest are exclusions (typically column names)
            exclude_items = []
            for i in range(1, len(inputs)):
                # Try to get literal values or other identifiers
                try:
                    exclude_val = get_literal_value(inputs[i])
                    exclude_items.append(exclude_val)
                except Exception:  # noqa: BLE001 -- intentional broad catch (defensive fallback)
                    # If we can't get a literal, try parsing as a expression
                    exclude_node = self.parse(inputs[i])
                    if isinstance(exclude_node, LiteralNode):
                        exclude_items.append(exclude_node.value)
                    else:
                        # For now, we'll just use a string representation
                        exclude_items.append(str(inputs[i]))

            return ExcludeNode(expr=expr, input=input_node, exclude=exclude_items)

        except Exception as e:
            log.warning(f"Error parsing Exclude expression: {e}", exc_info=True)
            return ErrorNode(expr=expr, error=f"Exclude parsing error: {e!s}")

    def _parse_keep_name(self, expr: pl.Expr, expr_dict: dict) -> BaseExprNode:
        """Parse a keep_name expression."""
        try:
            # Get the input expression
            inputs = list(expr.meta.pop())
            if not inputs:
                return ErrorNode(expr=expr, error="No input found for keep_name operation")

            input_expr = inputs[0]
            input_node = self.parse(input_expr)

            return KeepNameNode(expr=expr, input=input_node)

        except Exception as e:
            log.warning(f"Error parsing KeepName expression: {e}", exc_info=True)
            return ErrorNode(expr=expr, error=f"KeepName parsing error: {e!s}")

    def _parse_nth(self, expr: pl.Expr, expr_dict: dict) -> BaseExprNode:
        """Parse an nth expression."""
        try:
            nth_info = expr_dict.get("Nth")

            if not isinstance(nth_info, int):
                return ErrorNode(expr=expr, error="Invalid Nth index value")

            return NthNode(expr=expr, index=nth_info)

        except Exception as e:
            log.warning(f"Error parsing Nth expression: {e}", exc_info=True)
            return ErrorNode(expr=expr, error=f"Nth parsing error: {e!s}")

    def _parse_field(self, expr: pl.Expr, expr_dict: dict) -> BaseExprNode:
        """Parse a field expression."""
        try:
            field_info = expr_dict.get("Field")

            if not isinstance(field_info, list):
                field_info = [field_info] if field_info else []

            return FieldNode(expr=expr, fields=field_info)

        except Exception as e:
            log.warning(f"Error parsing Field expression: {e}", exc_info=True)
            return ErrorNode(expr=expr, error=f"Field parsing error: {e!s}")

    def _parse_anon_function(self, expr: pl.Expr, expr_dict: dict) -> BaseExprNode:
        input_node = self.parse(expr.meta.pop()[0])
        return AnonymousFunctionNode(expr=expr, input=input_node)


# ---------------------------------------------------------------------------
# NodeTraverser-based parser
#
# Built on polars' private ``NodeTraverser`` API (``LazyFrame._ldf.visit()`` plus ``add_expressions`` and ``view_expression``). The traverser exposes
# a typed view of expression nodes after polars' DSL→IR conversion, which is exactly the shape that arrives at a custom IO source plugin via
# ``polars.io.plugins.register_io_source``. Structural traversal is driven by ``expr.meta.pop()`` so each sub-node receives an accurate ``pl.Expr``
# (downstream visitors rely on ``node.expr.meta.root_names()`` and ``~node.expr``); the traverser is used purely for typed dispatch and for direct
# access to ``Operator``, function-category enums, ``Literal.value``/``dtype`` and ``Cast.dtype``. This is the default parser; the legacy
# JSON-serialization based :class:`ExprParser` above remains reachable for one release via the ``POLARS_IO_TOOLS_USE_LEGACY_EXPR_PARSER`` env var.
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


class NodeTraverserParser:
    """Parse a :class:`pl.Expr` into a :class:`BaseExprNode` tree using polars' ``NodeTraverser`` for typed dispatch and ``expr.meta.pop()``
    for sub-expression attribution.

    Mirrors the structure of :class:`ExprParser` above (dispatch via a class-keyed parser map, one ``_parse_<kind>`` method per node type) so
    that the eventual Phase B swap is a near-mechanical replacement.
    """

    def __init__(self) -> None:
        # Dispatch on the polars typed-view class, mirroring ExprParser._parser_map.
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
        except NotImplementedError:
            return None

    def _parse(self, expr: pl.Expr, visitor: Any) -> BaseExprNode:
        """Parse a single expression node by dispatching on the polars typed view's class.

        Per-subtree errors are isolated here (matching the legacy ``ExprParser``'s per-method try/except behavior): a failure in one
        ``_parse_<kind>`` method collapses that subtree to an :class:`ErrorNode` rather than the whole top-level expression.
        """
        obj = self._view(visitor, expr)
        if obj is None:
            # polars' NodeTraverser has no typed view for a few IR nodes that can appear in a pushed-down
            # predicate -- notably list functions (``col.list.*``) and array functions (``col.arr.*``), which
            # raise ``NotImplementedError`` from ``view_expression``. Those subtrees degrade to ``UnknownNode``
            # here, so predicates containing them are not pushed down under this parser (the legacy parser, via
            # ``POLARS_IO_TOOLS_USE_LEGACY_EXPR_PARSER=1``, still handles them). Recovering them requires polars to
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
        return CastNode(expr=expr, input=self._parse(input_expr, visitor), dtype=obj.dtype)

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
    except Exception as e:  # noqa: BLE001 -- intentional broad catch (defensive fallback)
        log.warning(f"Error getting literal value: {e}")
        return FAILED_LITERAL_RESULT  # Use a sentinel to indicate failure


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


def get_parsed_expr(expr_or_node: pl.Expr | BaseExprNode) -> BaseExprNode | None:
    """Parse a Polars expression or return the node if it's already parsed. This is designed to be run after polars has applied its optimizations, such as the predicate passed to a custom io plugin. Thus, we avoid handling some Polars expressions that cannot be passed as such, for example, like polars window functions or aggregations.

    Set ``POLARS_IO_TOOLS_USE_LEGACY_EXPR_PARSER=1`` to fall back to the legacy parser if you hit a problem with the default.
    """
    if isinstance(expr_or_node, BaseExprNode):
        return expr_or_node

    if os.environ.get("POLARS_IO_TOOLS_USE_LEGACY_EXPR_PARSER") == "1":
        return ExprParser().parse(expr_or_node)
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
