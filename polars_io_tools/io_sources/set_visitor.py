import logging
from typing import Any, Optional, Set, Union

import polars as pl

from .base import AliasNode, BaseExprNode, BinaryExprNode, CastNode, ExprVisitor, FunctionNode, extract_column_name, get_parsed_expr
from .enum import BooleanFunctionType, OperatorType

# Configure logging
log = logging.getLogger(__name__)

__all__ = ("convert_expr_to_valid_values",)


class SetVisitor(ExprVisitor[Optional[Set[Any]]]):
    """
    Visitor that extracts valid values for a specific column.
    Each method updates internal state rather than returning values.
    """

    def __init__(self, target_column: str):
        self.target_column = target_column
        # Track inclusions and exclusions as state
        self.inclusions = None  # None means "any value possible"
        self.exclusions = set()  # Empty set means "no exclusions"

    def visit_binary_expr(self, node: BinaryExprNode) -> None:
        """Handle binary expressions (comparisons and logical operations)"""
        if node.op.is_comparison():
            # Handle comparison operators (=, !=, etc.)
            column = extract_column_name(node.left)
            if column == self.target_column:
                # Try to extract value from right side
                if node.right.can_extract_literal:
                    value = node.right.value

                    if node.op in [OperatorType.EQ, OperatorType.EQ_VALIDITY]:
                        # Equality: add to inclusions (create new set for consistency)
                        if self.inclusions is None:
                            self.inclusions = {value}
                        else:
                            self.inclusions = self.inclusions.union({value})
                    elif node.op in [OperatorType.NOT_EQ, OperatorType.NOT_EQ_VALIDITY]:
                        # Inequality: add to exclusions
                        self.exclusions.add(value)

        elif node.op.is_bitwise_or_logical():
            # Handle logical operations (AND, OR)
            if node.op in [OperatorType.AND, OperatorType.LOGICAL_AND]:
                # For AND, we need to intersect the results of both sides
                left_visitor = SetVisitor(self.target_column)
                left_visitor.visit(node.left)

                right_visitor = SetVisitor(self.target_column)
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

            elif node.op in [OperatorType.OR, OperatorType.LOGICAL_OR]:
                # For OR, we need to union the results of both sides
                left_visitor = SetVisitor(self.target_column)
                left_visitor.visit(node.left)

                right_visitor = SetVisitor(self.target_column)
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

    def visit_function(self, node: FunctionNode) -> None:
        """Handle function expressions (IS_NULL, IS_IN, etc.)"""
        if not isinstance(node.function_type, BooleanFunctionType):
            return

        # Handle IS_NULL function
        if node.function_type == BooleanFunctionType.IS_NULL:
            if len(node.inputs) > 0:
                column = extract_column_name(node.inputs[0])
                if column == self.target_column:
                    # Add NULL to inclusions (create new set for consistency)
                    if self.inclusions is None:
                        self.inclusions = {None}
                    else:
                        self.inclusions = self.inclusions.union({None})

        # Handle IS_NOT_NULL function
        elif node.function_type == BooleanFunctionType.IS_NOT_NULL:
            if len(node.inputs) > 0:
                column = extract_column_name(node.inputs[0])
                if column == self.target_column:
                    # Add NULL to exclusions
                    self.exclusions.add(None)

        # Handle IS_IN function
        elif node.function_type == BooleanFunctionType.IS_IN:
            if len(node.inputs) >= 2 and node.inputs[1].can_extract_literal:
                column = extract_column_name(node.inputs[0])
                if column == self.target_column:
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
                    except (TypeError, ValueError):
                        # For unhashable types
                        pass

        # Handle NOT function
        elif node.function_type == BooleanFunctionType.NOT:
            if len(node.inputs) > 0:
                # Create a sub-visitor for the inner expression
                inner_visitor = SetVisitor(self.target_column)
                inner_visitor.visit(node.inputs[0])

                # Invert the results
                if inner_visitor.inclusions is not None:
                    for val in inner_visitor.inclusions:
                        self.exclusions.add(val)

                    # If we have our own inclusions, remove these exclusions
                    if self.inclusions is not None:
                        self.inclusions = self.inclusions - inner_visitor.inclusions

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

    def process_results(self) -> Optional[Set[Any]]:
        """Get the final set of valid values based on tracked state."""
        # If we have inclusions, apply exclusions
        if self.inclusions is not None:
            return self.inclusions - self.exclusions
        # Otherwise, we can't determine valid values without knowing universe
        return None


def convert_expr_to_valid_values(expr_or_node: Union[pl.Expr, BaseExprNode], column: str) -> Optional[Set[Any]]:
    """
    Extract valid values for a column from an expression or DNF.

    Args:
        expr_or_dnf: Either a Polars expression or DNF format filters
        column: The column to extract values for

    Returns:
        Set of valid values or None if any value is
    """
    try:
        node = get_parsed_expr(expr_or_node)
        visitor = SetVisitor(column)
        # build up state
        visitor.visit(node)
        return visitor.process_results()
    except Exception:
        log.exception("Error extracting valid values from expression failed.")
        return None  # Conversion failed
