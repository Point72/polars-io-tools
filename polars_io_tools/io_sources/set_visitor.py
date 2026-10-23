import logging
from typing import Any

import polars as pl

from .base import (
    AliasNode,
    BaseExprNode,
    BinaryExprNode,
    CastNode,
    ExprVisitor,
    FunctionNode,
    extract_value_preserving_column_name,
    get_parsed_expr,
)
from .dnf_visitor import _has_comparable_value
from .enum import BooleanFunctionType, OperatorType

# Configure logging
log = logging.getLogger(__name__)

__all__ = ("convert_expr_to_valid_values",)


class SetVisitor(ExprVisitor[set[Any] | None]):
    """
    Visitor that extracts valid values for a specific column.
    Each method updates internal state rather than returning values.
    """

    def __init__(self, target_column: str, schema: pl.Schema | dict[str, pl.DataType] | None = None):
        self.target_column = target_column
        self.schema = schema
        # Track inclusions and exclusions as state
        self.inclusions = None  # None means "any value possible"
        self.exclusions = set()  # Empty set means "no exclusions"
        # Whether every row whose value is in ``process_results()`` satisfies the predicate, so NOT can exclude them.
        self.exact = False

    def visit_binary_expr(self, node: BinaryExprNode) -> None:
        """Handle binary expressions (comparisons and logical operations)"""
        if node.op.is_comparison():
            # Handle comparison operators (=, !=, etc.)
            column = extract_value_preserving_column_name(node.left, self.schema)
            if column == self.target_column and _has_comparable_value(node.right, column, self.schema):
                # Try to extract value from right side
                value = node.right.value

                if node.op in [OperatorType.EQ, OperatorType.EQ_VALIDITY]:
                    # Equality: add to inclusions (create new set for consistency)
                    if self.inclusions is None:
                        self.inclusions = {value}
                    else:
                        self.inclusions = self.inclusions.union({value})
                    # ``x == None`` is null on every row, unlike ``eq_missing(None)``.
                    self.exact = node.op == OperatorType.EQ_VALIDITY or value is not None
                elif node.op in [OperatorType.NOT_EQ, OperatorType.NOT_EQ_VALIDITY]:
                    # Inequality: add to exclusions
                    self.exclusions.add(value)

        elif node.op.is_bitwise_or_logical():
            # Handle logical operations (AND, OR)
            if node.op in [OperatorType.AND, OperatorType.LOGICAL_AND]:
                # For AND, we need to intersect the results of both sides
                left_visitor = SetVisitor(self.target_column, self.schema)
                left_visitor.visit(node.left)

                right_visitor = SetVisitor(self.target_column, self.schema)
                right_visitor.visit(node.right)

                # Update inclusions - intersection for AND
                if left_visitor.inclusions is not None and right_visitor.inclusions is not None:
                    self.inclusions = left_visitor.inclusions.intersection(right_visitor.inclusions)
                elif left_visitor.inclusions is not None:
                    self.inclusions = left_visitor.inclusions
                elif right_visitor.inclusions is not None:
                    self.inclusions = right_visitor.inclusions

                # Update exclusions - union for AND
                self.exclusions = self.exclusions.union(left_visitor.exclusions).union(right_visitor.exclusions)
                self.exact = _both_exact(left_visitor, right_visitor)

            elif node.op in [OperatorType.OR, OperatorType.LOGICAL_OR]:
                # For OR, we need to union the results of both sides
                left_visitor = SetVisitor(self.target_column, self.schema)
                left_visitor.visit(node.left)

                right_visitor = SetVisitor(self.target_column, self.schema)
                right_visitor.visit(node.right)

                # Update inclusions - union for OR
                if left_visitor.inclusions is not None and right_visitor.inclusions is not None:
                    self.inclusions = left_visitor.inclusions.union(right_visitor.inclusions)
                else:
                    # If either side allows any value, the result allows any value
                    self.inclusions = None

                # Update exclusions - intersection for OR
                # Only values excluded by both sides should be excluded
                self.exclusions = left_visitor.exclusions.intersection(right_visitor.exclusions)
                self.exact = _both_exact(left_visitor, right_visitor)

    def visit_function(self, node: FunctionNode) -> None:
        """Handle function expressions (IS_NULL, IS_IN, etc.)"""
        if not isinstance(node.function_type, BooleanFunctionType):
            return

        # Handle IS_NULL function
        if node.function_type == BooleanFunctionType.IS_NULL:
            if len(node.inputs) > 0:
                column = extract_value_preserving_column_name(node.inputs[0], self.schema)
                if column == self.target_column:
                    # Add NULL to inclusions (create new set for consistency)
                    if self.inclusions is None:
                        self.inclusions = {None}
                    else:
                        self.inclusions = self.inclusions.union({None})
                    self.exact = True

        # Handle IS_NOT_NULL function
        elif node.function_type == BooleanFunctionType.IS_NOT_NULL:
            if len(node.inputs) > 0:
                column = extract_value_preserving_column_name(node.inputs[0], self.schema)
                if column == self.target_column:
                    # Add NULL to exclusions
                    self.exclusions.add(None)

        # Handle IS_IN function
        elif node.function_type == BooleanFunctionType.IS_IN:
            if len(node.inputs) >= 2:
                column = extract_value_preserving_column_name(node.inputs[0], self.schema)
                if column == self.target_column and _has_comparable_value(node.inputs[1], column, self.schema, collection=True):
                    try:
                        # Get the values
                        values = node.inputs[1].value
                        if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
                            values = {values}
                        else:
                            values = set(values)

                        # Update state
                        if self.inclusions is None:
                            self.inclusions = values.copy()
                        else:
                            self.inclusions = self.inclusions.intersection(values)
                        self.exact = bool(node.options.get("nulls_equal", False)) or None not in values
                    except (TypeError, ValueError):
                        # For unhashable types
                        pass

        # Handle NOT function
        elif node.function_type == BooleanFunctionType.NOT and len(node.inputs) > 0:
            # Create a sub-visitor for the inner expression
            inner_visitor = SetVisitor(self.target_column, self.schema)
            inner_visitor.visit(node.inputs[0])

            # Invert the results, unless the inner inclusions are a relaxed superset
            inner_values = inner_visitor.process_results()
            if inner_values is not None and inner_visitor.exact:
                self.exclusions.update(inner_values)

                # If we have our own inclusions, remove these exclusions
                if self.inclusions is not None:
                    self.inclusions = self.inclusions - inner_values

            # For exclusions, we might be able to add what was excluded
            if self.inclusions is not None:
                self.inclusions = self.inclusions.union(inner_visitor.exclusions)

    def visit_cast(self, node: CastNode) -> None:
        """Handle cast expressions."""
        if node.dtype == pl.Boolean:
            # If casting to boolean, process the underlying expression.
            # The inclusion/exclusion logic will be handled by the visitor
            # when visiting the input node.
            self.visit(node.input)

    def visit_alias(self, node: AliasNode) -> None:
        """Handle alias expressions by visiting the underlying input."""
        self.visit(node.input)

    def process_results(self) -> set[Any] | None:
        """Get the final set of valid values based on tracked state."""
        # If we have inclusions, apply exclusions
        if self.inclusions is not None:
            return self.inclusions - self.exclusions
        # Otherwise, we can't determine valid values without knowing universe
        return None


def _both_exact(left: SetVisitor, right: SetVisitor) -> bool:
    return left.exact and right.exact and left.inclusions is not None and right.inclusions is not None


def convert_expr_to_valid_values(
    expr_or_node: pl.Expr | BaseExprNode, column: str, schema: pl.Schema | dict[str, pl.DataType] | None = None
) -> set[Any] | None:
    """
    Extract valid values for a column from an expression or DNF.

    Args:
        expr_or_node: Either a Polars expression or an already-parsed expression node.
        column: The column to extract values for.
        schema: Optional column -> dtype mapping. Casts on the column are only looked through when the schema shows
            them to be value- and order-preserving.

    Returns:
        Set of valid values, or None if the expression does not restrict the column to a known set.
    """
    try:
        node = get_parsed_expr(expr_or_node)
        visitor = SetVisitor(column, schema)
        # build up state
        visitor.visit(node)
        return visitor.process_results()
    except Exception:
        log.exception("Error extracting valid values from expression failed.")
        return None  # Conversion failed
