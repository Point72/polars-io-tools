import logging
from collections.abc import Iterable, Set as AbstractSet

import polars as pl

from .base import (
    BaseExprNode,
    BinaryExprNode,
    CastNode,
    ExprVisitor,
    FunctionNode,
    LiteralNode,
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

_LOGICAL_OPS = {OperatorType.AND, OperatorType.OR, OperatorType.LOGICAL_AND, OperatorType.LOGICAL_OR}
_LOGICAL_FUNCTIONS = {BooleanFunctionType.NOT, BooleanFunctionType.ALL_HORIZONTAL, BooleanFunctionType.ANY_HORIZONTAL}


def _is_relevant(expr: pl.Expr, columns: AbstractSet[str]) -> bool:
    """
    Check if the expression is relevant to the specified columns.
    It checks if all root names of the expression are in the set of columns.
    Expressions without root names or with selectors (``pl.all()``, regex, multiple columns) are not relevant.
    """
    root_names = expr.meta.root_names()
    return bool(root_names) and not expr.meta.has_multiple_outputs() and set(root_names).issubset(columns)


def _is_boolean(node: BaseExprNode) -> bool:
    """Check if the node is known to produce booleans, so its AND/OR/NOT are logical rather than bitwise."""
    if isinstance(node, BinaryExprNode):
        if node.op.is_comparison():
            return True
        return node.op in _LOGICAL_OPS and _is_boolean(node.left) and _is_boolean(node.right)
    if isinstance(node, FunctionNode) and isinstance(node.function_type, BooleanFunctionType):
        return node.function_type not in _LOGICAL_FUNCTIONS or all(_is_boolean(input_node) for input_node in node.inputs)
    return isinstance(node, CastNode) and node.dtype == pl.Boolean


def _as_boolean(result: pl.Expr | None, node: BaseExprNode) -> pl.Expr | None:
    """Coerce a restricted operand of a horizontal AND/OR to Boolean, as the original operation would."""
    if result is None or _is_boolean(node):
        return result
    return pl.all_horizontal(result)


def _combine_and(expr1: pl.Expr | None, expr2: pl.Expr | None) -> pl.Expr | None:
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


def _combine_or(expr1: pl.Expr | None, expr2: pl.Expr | None) -> pl.Expr | None:
    """
    Combine two expressions with OR logic.
    Both must be not None to create a valid expression.
    """
    if expr1 is None or expr2 is None:
        return None
    return expr1 | expr2


def _combine_all(exprs: list[pl.Expr]) -> pl.Expr | None:
    """Combine list of expressions with AND (all_horizontal). Returns None if empty."""
    if not exprs:
        return None
    if len(exprs) == 1:
        return exprs[0]
    return pl.all_horizontal(*exprs)


def _combine_any(exprs: list[pl.Expr]) -> pl.Expr | None:
    """Combine list of expressions with OR (any_horizontal). Returns None if empty."""
    if not exprs:
        return None
    if len(exprs) == 1:
        return exprs[0]
    return pl.any_horizontal(*exprs)


class RestrictPredicateVisitor(ExprVisitor[pl.Expr | None]):
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

    def visit_literal(self, node: LiteralNode) -> None:
        # A constant boolean (or null) holds on any frame, so it is kept: ``false`` keeps no rows anywhere.
        if node.can_extract_literal and (node.value is None or isinstance(node.value, bool)):
            self.result = ~node.expr if self.negated else node.expr
        else:
            self.default_visit(node)

    def _restrict(self, node: BaseExprNode, negated: bool) -> pl.Expr | None:
        visitor = RestrictPredicateVisitor(self.columns, negated)
        visitor.visit(node)
        return visitor.process_results()

    def visit_binary_expr(self, node: BinaryExprNode) -> None:
        # If the entire expression only references our columns, handle it directly
        self.default_visit(node)
        if self.result is not None:
            return

        if node.op in _LOGICAL_OPS:
            left_result = self._restrict(node.left, self.negated)
            right_result = self._restrict(node.right, self.negated)
            if node.op in (OperatorType.LOGICAL_AND, OperatorType.LOGICAL_OR):
                left_result = _as_boolean(left_result, node.left)
                right_result = _as_boolean(right_result, node.right)
            # The operands already carry the negation, so by De Morgan a negated AND combines as OR and vice versa.
            is_and_op = node.op in (OperatorType.AND, OperatorType.LOGICAL_AND)
            self.result = _combine_and(left_result, right_result) if is_and_op != self.negated else _combine_or(left_result, right_result)

    def visit_function(self, node: FunctionNode) -> None:
        self.default_visit(node)
        if self.result is not None:
            return
        if node.function_type == BooleanFunctionType.NOT:
            self.result = self._restrict(node.inputs[0], not self.negated)
        elif node.function_type in (BooleanFunctionType.ALL_HORIZONTAL, BooleanFunctionType.ANY_HORIZONTAL):
            results = [r for n in node.inputs if (r := _as_boolean(self._restrict(n, self.negated), n)) is not None]
            is_all = node.function_type == BooleanFunctionType.ALL_HORIZONTAL
            # Only a plain conjunction may drop operands that cannot be restricted.
            if len(results) == len(node.inputs) or (is_all and not self.negated):
                self.result = _combine_all(results) if is_all != self.negated else _combine_any(results)

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
        # When negated: if(pred, ~true_expr, ~false_expr), so the predicate itself is never negated
        pred_result = self._restrict(node.predicate, negated=False)
        opp_pred_result = self._restrict(node.predicate, negated=True)
        if opp_pred_result is not None:
            # A null predicate selects the falsy branch, so those rows must be kept as well
            opp_pred_result = opp_pred_result.fill_null(True)

        true_result = self._restrict(node.truthy, self.negated)
        false_result = self._restrict(node.falsy, self.negated)

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
        if node.dtype == pl.Boolean and _is_boolean(node.input):
            # If casting a boolean expression to boolean, process the underlying expression for restriction.
            # The result of visiting node.input will be stored in self.result.
            self.visit(node.input)
        else:
            # If not casting to boolean, we cannot restrict the expression to the current columns.
            self.result = None

    def process_results(self) -> pl.Expr | None:
        """Get the final restricted predicate or None if not possible"""
        return self.result


def restrict_expr_to_columns(expr_or_node: pl.Expr | BaseExprNode, columns: Iterable[str]) -> pl.Expr | None:
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

        # If expression already only involves the specified columns, return it as is
        if isinstance(expr, pl.Expr) and _is_relevant(expr, columns_set):
            return expr

        # Parse the expression if needed
        node = get_parsed_expr(expr_or_node)
        if node is None:
            return None

        # Create and use the visitor
        visitor = RestrictPredicateVisitor(columns_set)
        visitor.visit(node)
        return visitor.process_results()
    except Exception:
        log.exception("Failed to restrict predicate")
        return None
