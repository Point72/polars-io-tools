import logging
from collections.abc import Iterable
from typing import AbstractSet, Optional, Union

import polars as pl

from .base import (
    BaseExprNode,
    BinaryExprNode,
    CastNode,
    ExprVisitor,
    FunctionNode,
    TernaryNode,
    get_parsed_expr,
)
from .enum import (
    BooleanFunctionType,
    OperatorType,
)

# Configure logging
log = logging.getLogger(__name__)

__all__ = ("restrict_expr_to_columns",)


def _is_relevant(expr: pl.Expr, columns: AbstractSet[str]) -> bool:
    """
    Check if the expression is relevant to the specified columns.
    It checks if all root names of the expression are in the set of columns.
    """
    return set(expr.meta.root_names()).issubset(columns)


def _combine_and(expr1: Optional[pl.Expr], expr2: Optional[pl.Expr]) -> Optional[pl.Expr]:
    """
    Combine two expressions with AND logic.
    If either expression is None, return the other expression.
    If both are None, return None.
    """
    if expr1 is None:
        return expr2
    if expr2 is None:
        return expr1
    return expr1 & expr2


def _combine_or(expr1: Optional[pl.Expr], expr2: Optional[pl.Expr]) -> Optional[pl.Expr]:
    """
    Combine two expressions with OR logic.
    Both must be not None to create a valid expression.
    """
    if expr1 is None or expr2 is None:
        return None
    return expr1 | expr2


def _combine_all(exprs: list[pl.Expr]) -> Optional[pl.Expr]:
    """Combine list of expressions with AND (all_horizontal). Returns None if empty."""
    if not exprs:
        return None
    if len(exprs) == 1:
        return exprs[0]
    return pl.all_horizontal(*exprs)


def _combine_any(exprs: list[pl.Expr]) -> Optional[pl.Expr]:
    """Combine list of expressions with OR (any_horizontal). Returns None if empty."""
    if not exprs:
        return None
    if len(exprs) == 1:
        return exprs[0]
    return pl.any_horizontal(*exprs)


class RestrictPredicateVisitor(ExprVisitor[Optional[pl.Expr]]):
    """
    Visitor that restricts a predicate to only reference a specific set of columns.
    It tries to create a less restrictive predicate that only involves the specified columns,
    returning None if no valid restriction can be created.
    """

    def __init__(self, columns: AbstractSet[str], is_negated: bool = False):
        """
        Initialize with the set of columns to restrict the predicate to.

        Args:
            columns: Set of column names to keep in the predicate
            is_negated: Whether we should be negating the predicates. This is to pass down DeMorgan's law
        """
        # We use frozenset to ensure we never alter the columns.
        if isinstance(columns, frozenset):
            self.columns: AbstractSet[str] = columns  # this is ok to share since we will not modify it
        else:
            self.columns = frozenset(columns)
        self.result = None  # Will store the final restricted expression
        self.negated = is_negated

    def default_visit(self, node: BaseExprNode):
        if _is_relevant(node.expr, self.columns):
            self.result = ~node.expr if self.negated else node.expr
        else:
            self.result = None

    def visit_binary_expr(self, node: BinaryExprNode) -> None:
        # If the entire expression only references our columns, handle it directly
        self.default_visit(node)
        if self.result is not None:
            return

        # For AND/OR operators, apply De Morgan's laws when negated
        if node.op in [OperatorType.AND, OperatorType.OR, OperatorType.LOGICAL_AND, OperatorType.LOGICAL_OR]:
            is_and_op = node.op in [OperatorType.AND, OperatorType.LOGICAL_AND]

            # Visit both sides
            left_visitor = RestrictPredicateVisitor(self.columns, self.negated)
            left_visitor.visit(node.left)
            left_result = left_visitor.process_results()

            right_visitor = RestrictPredicateVisitor(self.columns, self.negated)
            right_visitor.visit(node.right)
            right_result = right_visitor.process_results()

            # Apply the appropriate combination based on op and negation
            if is_and_op:
                if not self.negated:
                    # Regular AND: both sides must be true
                    self.result = _combine_and(left_result, right_result)
                else:
                    # Negated AND: NOT(A AND B) = NOT(A) OR NOT(B)
                    # But our visitors already handled the negation, so we use OR
                    self.result = _combine_or(left_result, right_result)
            else:  # OR operation
                if not self.negated:
                    # Regular OR: either side can be true
                    self.result = _combine_or(left_result, right_result)
                else:
                    # Negated OR: NOT(A OR B) = NOT(A) AND NOT(B)
                    # But our visitors already handled the negation, so we use AND
                    self.result = _combine_and(left_result, right_result)

    def visit_function(self, node: FunctionNode) -> None:
        self.default_visit(node)
        if self.result is not None:
            return
        if node.function_type == BooleanFunctionType.NOT:
            inner_visitor = RestrictPredicateVisitor(self.columns, not self.negated)
            inner_visitor.visit(node.inputs[0])
            inner_result = inner_visitor.process_results()
            self.result = None if inner_result is None else inner_result
        elif node.function_type == BooleanFunctionType.ALL_HORIZONTAL:
            results = []
            for input_node in node.inputs:
                inner_visitor = RestrictPredicateVisitor(self.columns, self.negated)
                inner_visitor.visit(input_node)
                inner_result = inner_visitor.process_results()
                if inner_result is not None:
                    results.append(inner_result)
            if not self.negated:
                self.result = _combine_all(results)
            else:
                self.result = _combine_any(results)
        elif node.function_type == BooleanFunctionType.ANY_HORIZONTAL:
            results = []
            for input_node in node.inputs:
                inner_visitor = RestrictPredicateVisitor(self.columns, self.negated)
                inner_visitor.visit(input_node)
                inner_result = inner_visitor.process_results()
                if inner_result is not None:
                    results.append(inner_result)
            if not self.negated:
                self.result = _combine_any(results) if len(results) == len(node.inputs) else None
            else:
                self.result = _combine_all(results) if len(results) == len(node.inputs) else None

    def visit_ternary(self, node: TernaryNode) -> None:
        # pl.when(A).then(B).otherwise(C)
        # same as
        # (A & B) | (~A & C)

        # ~A is tricky because of needing to apply DeMorgan's laws
        # but we should do that.
        # Handle ternary operations with respect to negation
        self.default_visit(node)
        if self.result is not None:
            return

        # For a ternary op: if(pred, true_expr, false_expr)
        # When negated: if(pred, ~true_expr, ~false_expr)
        pred_visitor = RestrictPredicateVisitor(self.columns, self.negated)
        pred_visitor.visit(node.predicate)
        pred_result = pred_visitor.process_results()

        opp_pred_visitor = RestrictPredicateVisitor(self.columns, not self.negated)
        opp_pred_visitor.visit(node.predicate)
        opp_pred_result = opp_pred_visitor.process_results()

        true_visitor = RestrictPredicateVisitor(self.columns, self.negated)
        true_visitor.visit(node.truthy)
        true_result = true_visitor.process_results()

        false_visitor = RestrictPredicateVisitor(self.columns, self.negated)
        false_visitor.visit(node.falsy)
        false_result = false_visitor.process_results()

        if true_result is None and false_result is None:
            # This means we need to do an OR of the predicate
            # and it's negation. This is trivially true, so it is not
            # a real restriction, and we set our result to None
            self.result = None
            return

        # Construct the ternary expression with the restricted parts
        # It will be correct, but might not be simplified.
        self.result = _combine_or(
            _combine_and(pred_result, true_result),  # predicate & truthy
            _combine_and(opp_pred_result, false_result),  # falsy
        )

    def visit_cast(self, node: CastNode) -> None:
        """Handle cast expressions."""
        self.default_visit(node)
        if self.result is not None:
            return
        if node.dtype == pl.Boolean:
            # If casting to boolean, process the underlying expression for restriction.
            # The result of visiting node.input will be stored in self.result.
            self.visit(node.input)
        else:
            # If not casting to boolean, we cannot restrict the expression to the current columns.
            self.result = None

    def process_results(self) -> Optional[pl.Expr]:
        """Get the final restricted predicate or None if not possible"""
        return self.result


def restrict_expr_to_columns(expr_or_node: Union[pl.Expr, BaseExprNode], columns: Iterable[str]) -> Optional[pl.Expr]:
    """
    Given a predicate expression involving many columns, try to find a less restrictive
    predicate that only involves the provided columns.

    This function is useful for simplifying complex filter expressions to ones that only
    reference a subset of columns while maintaining logical consistency.

    Examples:
        Basic AND operation with multiple columns:
            >>> expr = (pl.col("foo") > 0) & (pl.col("bar") < 0)
            >>> simplified = restrict_expr_to_columns(expr, {"foo"})
            >>> simplified is not None
            True

        OR operation cannot be simplified when columns appear in different operands:
            >>> expr = (pl.col("foo") > 0) | (pl.col("bar") < 0)
            >>> restrict_expr_to_columns(expr, {"foo"}) is None
            True

        Complex expressions with multiple AND/OR combinations:
            >>> expr = ((pl.col("symbol") == "US") & (pl.col("price") > 100)) | ((pl.col("symbol") == "EU") & (pl.col("volume") > 1000))
            >>> simplified = restrict_expr_to_columns(expr, {"symbol"})
            >>> simplified is not None
            True

    Args:
        expr_or_node: A Polars expression or parsed expression node
        columns: Iterable of column names to restrict the predicate to

    Returns:
        A simplified predicate that only references the specified columns, or None if
        such simplification is not possible without changing the logical meaning.
    """

    # Convert columns to a frozenset to ensure efficient lookups
    columns_set: AbstractSet[str] = frozenset(columns)

    try:
        # If input is a Polars expression, check if restriction is needed
        if isinstance(expr_or_node, pl.Expr):
            expr = expr_or_node
        else:
            expr = expr_or_node.expr

        if isinstance(expr, pl.Expr):
            # If expression already only involves the specified columns, return it as is
            if _is_relevant(expr, columns_set):
                return expr

        # Parse the expression if needed
        node = get_parsed_expr(expr_or_node)
        if node is None:
            return None

        # Create and use the visitor
        visitor = RestrictPredicateVisitor(columns_set)
        visitor.visit(node)
        return visitor.process_results()
    except Exception as e:
        log.exception(f"Failed to restrict predicate: {e}")
        return None
