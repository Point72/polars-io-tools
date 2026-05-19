import itertools
import logging
import re
from datetime import timedelta
from typing import Any, List, Literal, Optional, Set, Tuple, Union

import polars as pl
import portion as P
from portion import Bound, Interval

from .base import (
    AliasNode,
    BaseExprNode,
    BinaryExprNode,
    CastNode,
    ExprVisitor,
    FunctionNode,
    TernaryNode,
    extract_column_name,
    get_parsed_expr,
)
from .enum import (
    BooleanFunctionType,
    OperatorType,
    StringFunctionType,
)

log = logging.getLogger(__name__)


DNFOperator = Literal[
    "=",
    ">",
    "<",
    "<=",
    ">=",
    "~",
    "is",
    "in",
    "!=",
    "!>",
    "!<",
    "!<=",
    "!>=",
    "!~",
    "!is",
    "!in",
    "==",
    "<>",
    "=~",
    "not in",
    "is not",
]

DNFTuple = Tuple[str, DNFOperator, Any]  # (column, op, value)
DNFClause = List[DNFTuple]  # A single conjunction
DNF = List[DNFClause]  # Full DNF expression


__all__ = (
    "DNFVisitor",
    "DNF",
    "DNFClause",
    "DNFTuple",
    "convert_expr_to_dnf",
    "is_contradiction",  # Add new function to __all__
)


class ColumnConstraintAnalyzer:
    def __init__(self, column_name: str):
        self.column_name = column_name
        # Core constraints
        self.min_bound: Optional[Tuple[Any, bool]] = None  # (value, is_inclusive)
        self.max_bound: Optional[Tuple[Any, bool]] = None  # (value, is_inclusive)
        self.exact_values: Set[Any] = set()
        self.inclusion_set: Optional[Set[Any]] = None  # From IN operators
        self.exclusion_values: Set[Any] = set()  # From != and NOT IN
        self.is_null: Optional[bool] = None  # True=IS NULL, False=IS NOT NULL
        self.contradiction_found = False

    def update_from_predicate(self, op: str, value: Any) -> None:
        """Update constraints based on operator and value."""
        if self.contradiction_found:
            return

        # Skip non-constraining NULL comparisons for bounds/regex
        if value is None and op in [">", ">=", "<", "<=", "~", "!~", "=~"]:
            return

        # Equal operators
        if op in ["=", "=="]:
            self.exact_values.add(value)
            # Check NULL contradictions
            if (value is None and self.is_null is False) or (value is not None and self.is_null is True):
                self.contradiction_found = True

        # Not equal operators
        elif op in ["!=", "<>"]:
            self.exclusion_values.add(value)

        # Greater than operators
        elif op == ">" and value is not None:
            if self.min_bound is None or value >= self.min_bound[0]:
                self.min_bound = (value, False)  # Not inclusive

        # Greater than or equal operators
        elif op == ">=" and value is not None:
            if (
                self.min_bound is None or value > self.min_bound[0]
            ):  # if they are equal, we don't update since we don't want to overwrite the inclusive status
                self.min_bound = (value, True)  # Inclusive

        # Less than operators
        elif op == "<" and value is not None:
            if self.max_bound is None or value <= self.max_bound[0]:
                self.max_bound = (value, False)  # Not inclusive

        # Less than or equal operators
        elif op == "<=" and value is not None:
            if (
                self.max_bound is None or value < self.max_bound[0]
            ):  # if they are equal, we don't update since we don't want to overwrite the inclusive status
                self.max_bound = (value, True)  # Inclusive

        # IN operator
        elif op == "in":
            if not isinstance(value, (list, set, tuple)):
                value = [value]
            current_set = set(value)

            if self.inclusion_set is None:
                self.inclusion_set = current_set
            else:
                self.inclusion_set.intersection_update(current_set)

            # Check for contradictions
            if not self.inclusion_set:  # Empty set means contradiction
                self.contradiction_found = True

        # NOT IN operator
        elif op in ["!in", "not in"]:
            if not isinstance(value, (list, set, tuple)):
                value = [value]
            self.exclusion_values.update(value)

        # IS operator (NULL, TRUE, FALSE)
        elif op == "is":
            is_null_value = value is None
            if self.is_null is not None and self.is_null != is_null_value:
                self.contradiction_found = True
            self.is_null = is_null_value

        # IS NOT operator
        elif op in ["!is", "is not"] and value is None:  # IS NOT NULL
            if self.is_null is True:
                self.contradiction_found = True
            self.is_null = False

            # Check for contradictions
            if None in self.exact_values:
                self.contradiction_found = True

        # TODO: Support regex operators
        # Regex operators

        # Check range contradictions
        if self.min_bound and self.max_bound:
            min_val, min_inclusive = self.min_bound
            max_val, max_inclusive = self.max_bound
            try:
                if min_val > max_val:
                    self.contradiction_found = True
                elif min_val == max_val and not (min_inclusive and max_inclusive):
                    self.contradiction_found = True
            except TypeError:
                pass  # Skip type incompatibility check

    def is_value_valid(self, value: Any) -> bool:
        """Check if a value satisfies all constraints."""
        # Check NULL constraint
        if self.is_null is True:
            return value is None
        elif self.is_null is False and value is None:
            return False

        # Check exact values, inclusion, exclusion
        if self.exact_values and value not in self.exact_values:
            return False
        if self.inclusion_set is not None and value not in self.inclusion_set:
            return False
        if value in self.exclusion_values:
            return False

        # Skip remaining checks for NULL values
        if value is None:
            return True

        # Check bounds
        if self.min_bound:
            min_val, min_inclusive = self.min_bound
            try:
                if value < min_val or (value == min_val and not min_inclusive):
                    return False
            except TypeError:
                return False

        if self.max_bound:
            max_val, max_inclusive = self.max_bound
            try:
                if value > max_val or (value == max_val and not max_inclusive):
                    return False
            except TypeError:
                return False

        return True

    def has_contradiction(self, schema: Optional[pl.Schema] = None) -> bool:
        """Check if the constraints are contradictory."""
        # Check if contradiction was found during updates
        if self.contradiction_found:
            return True

        # Check for multiple exact values
        if len(self.exact_values) > 1:
            return True

        # Check if exact value is valid under all constraints
        if self.exact_values:
            exact_value = next(iter(self.exact_values))
            if not self.is_value_valid(exact_value):
                return True

        # Check if inclusion set has any valid value
        if self.inclusion_set is not None:
            if not self.inclusion_set:  # FIXED: Check for empty inclusion set
                return True
            # If none of the values in the inclusion set are valid, we have a contradiction.
            if not any(self.is_value_valid(v) for v in self.inclusion_set):
                return True

        # NULL can't satisfy bounds
        if self.is_null is True and (self.min_bound is not None or self.max_bound is not None):
            return True

        # If we have not errored yet, then we can check the schema, and see if
        # we can iterate over the values in the bounds.
        if schema is not None and self.min_bound and self.max_bound:
            col_typ = schema.get(self.column_name)
            if col_typ is None:
                return False
            if col_typ.is_integer():
                jump = 1
            elif col_typ == pl.Date:
                jump = timedelta(days=1)
            else:
                # We cannot iterate over the values in the bounds, so we cannot say if we have a contradiction at this point.
                return False
            min_val, min_inclusive = self.min_bound
            max_val, max_inclusive = self.max_bound

            left_bound = Bound.CLOSED if min_inclusive else Bound.OPEN
            right_bound = Bound.CLOSED if max_inclusive else Bound.OPEN

            interval = Interval.from_atomic(left_bound, min_val, max_val, right_bound)
            for val in P.iterate(interval, step=jump):
                # Check if the value is valid under all constraints
                if self.is_value_valid(val):
                    return False
            # If we reached here, everything in the interval is invalid, so we have a contradiction
            return True

        return False


def is_contradiction(clause: DNFClause, schema: Optional[pl.Schema] = None) -> bool:
    """Check if a DNF clause contains a contradiction.

    Args:
        clause: A DNF clause to check for contradictions.
        schema: The schema of the DataFrame the DNF clause is based on, this provides type information.

    """
    # Group predicates by column
    column_analyzers = {}

    for col, op, val in clause:
        if col not in column_analyzers:
            column_analyzers[col] = ColumnConstraintAnalyzer(col)

        try:
            # Update column constraints and check for contradictions during update
            column_analyzers[col].update_from_predicate(op, val)
            if column_analyzers[col].contradiction_found:
                return True
        except Exception:
            log.exception("Error processing predicate: %s %s %s", col, op, val)
            # FIXED: Handle unexpected exceptions more robustly
            # If we can't process a predicate, we should treat it conservatively
            continue

    for col, analyzer in column_analyzers.items():
        if analyzer.has_contradiction(schema=schema):
            return True

    return False


class DNFVisitor(ExprVisitor[Optional[DNF]]):
    """
    Visitor that converts expressions to DNF format.
    """

    def __init__(self):
        # Initialize state
        self.result_dnf: Optional[DNF] = None

    def default_result(self) -> Optional[DNF]:
        """Default result is None."""
        return None

    def visit_binary_expr(self, node: BinaryExprNode) -> None:
        """Handle binary expressions (comparisons and logical operations)"""
        if node.op.is_comparison():
            # Handle comparison operators (=, !=, etc.)
            column = extract_column_name(node.left)
            if column is not None and node.right.can_extract_literal:
                value = node.right.value

                # Map OperatorType to string operator for DNF
                op_map = {
                    OperatorType.EQ: "=",
                    OperatorType.EQ_VALIDITY: "=",
                    OperatorType.NOT_EQ: "!=",
                    OperatorType.NOT_EQ_VALIDITY: "!=",
                    OperatorType.GT: ">",
                    OperatorType.GT_EQ: ">=",
                    OperatorType.LT: "<",
                    OperatorType.LT_EQ: "<=",
                }

                if node.op in op_map:
                    op_str: DNFOperator = op_map[node.op]  # type: ignore[assignment]
                    self.result_dnf = [[(column, op_str, value)]]

        elif node.op.is_bitwise_or_logical():
            # Handle logical operations (AND, OR)
            # Create sub-visitors for each branch
            left_dnf = DNFVisitor().visit_and_process_results(node.left)
            right_dnf = DNFVisitor().visit_and_process_results(node.right)

            if node.op in [OperatorType.AND, OperatorType.LOGICAL_AND]:
                self.result_dnf = combine_and_dnf(left_dnf, right_dnf)
            elif node.op in [OperatorType.OR, OperatorType.LOGICAL_OR]:
                self.result_dnf = combine_or_dnf(left_dnf, right_dnf)

    def visit_function(self, node: FunctionNode) -> None:
        """Handle function expressions (IS_NULL, IS_IN, etc.)"""
        # Handle IS_NULL function
        if node.function_type == BooleanFunctionType.IS_NULL:
            if len(node.inputs) > 0:
                column = extract_column_name(node.inputs[0])
                if column is not None:
                    self.result_dnf = [[(column, "is", None)]]

        # Handle IS_NOT_NULL function
        elif node.function_type == BooleanFunctionType.IS_NOT_NULL:
            if len(node.inputs) > 0:
                column = extract_column_name(node.inputs[0])
                if column is not None:
                    self.result_dnf = [[(column, "is not", None)]]

        # Handle IS_IN function
        elif node.function_type == BooleanFunctionType.IS_IN:
            if len(node.inputs) >= 2 and node.inputs[1].can_extract_literal:
                column = extract_column_name(node.inputs[0])
                if column is not None:
                    values = node.inputs[1].value
                    if not isinstance(values, list):
                        # we do this for the case where the input is a single value
                        # and extracting that literal gives us a single value
                        values = [values]
                    self.result_dnf = [[(column, "in", values)]]

        # Handle IS_BETWEEN function
        elif node.function_type == BooleanFunctionType.IS_BETWEEN:
            if len(node.inputs) >= 3 and node.inputs[1].can_extract_literal and node.inputs[2].can_extract_literal:
                column = extract_column_name(node.inputs[0])
                if column is not None:
                    lower = node.inputs[1].value
                    upper = node.inputs[2].value

                    # Get closed parameter from options or default to "both"
                    closed = node.options["closed"]

                    # Map closed parameter to operators
                    lower_op = ">=" if closed in ["Both", "Left"] else ">"
                    upper_op = "<=" if closed in ["Both", "Right"] else "<"

                    # Set DNF with two predicates
                    self.result_dnf = [[(column, lower_op, lower), (column, upper_op, upper)]]

        # Handle NOT function
        elif node.function_type == BooleanFunctionType.NOT:
            if len(node.inputs) > 0:
                inner_visitor = DNFVisitor()
                inner_visitor.visit(node.inputs[0])
                inner_dnf = inner_visitor.process_results()

                if inner_dnf is not None:
                    self.result_dnf = negate_dnf(inner_dnf)

        # Handle ALL_HORIZONTAL function (ANDing all inputs)
        elif node.function_type == BooleanFunctionType.ALL_HORIZONTAL:
            for input_node in node.inputs:
                inner_visitor = DNFVisitor()
                inner_visitor.visit(input_node)
                inner_dnf = inner_visitor.process_results()
                if inner_dnf is not None:
                    self.result_dnf = combine_and_dnf(self.result_dnf, inner_dnf)

        # Handle ANY_HORIZONTAL function (ORing all inputs)
        elif node.function_type == BooleanFunctionType.ANY_HORIZONTAL:
            combined_dnf: Optional[DNF] = None
            for input_node in node.inputs:
                inner_visitor = DNFVisitor()
                inner_visitor.visit(input_node)
                inner_dnf = inner_visitor.process_results()
                if inner_dnf is not None:
                    combined_dnf = combine_or_dnf(combined_dnf, inner_dnf) if combined_dnf is not None else inner_dnf
            self.result_dnf = combined_dnf

        # Handle string functions
        elif isinstance(node.function_type, StringFunctionType):
            if len(node.inputs) >= 2 and node.inputs[1].can_extract_literal:
                column = extract_column_name(node.inputs[0])
                if column is not None:
                    pattern = node.inputs[1].value

                    if node.function_type == StringFunctionType.STARTS_WITH:
                        regex_pattern = f"^{re.escape(pattern)}"
                        self.result_dnf = [[(column, "~", regex_pattern)]]
                    elif node.function_type == StringFunctionType.ENDS_WITH:
                        regex_pattern = f"{re.escape(pattern)}$"
                        self.result_dnf = [[(column, "~", regex_pattern)]]
                    elif node.function_type == StringFunctionType.CONTAINS:
                        regex_pattern = f".*{re.escape(pattern)}.*"
                        self.result_dnf = [[(column, "~", regex_pattern)]]

    def visit_ternary(self, node: TernaryNode) -> None:
        """Visit ternary node"""
        # pl.when(A).then(B).otherwise(C)
        # same as (A & B) | (!A & C)
        predicate_dnf = DNFVisitor().visit_and_process_results(node.predicate)
        # If we can extract the literal, it will be a boolean
        # (polars will complain otherwise.)
        if node.truthy.can_extract_literal:
            truthy_dnf_val = node.truthy.value
            # we use empty list for False
            predicate_and_truthy = predicate_dnf if truthy_dnf_val else []
        else:
            truthy_dnf = DNFVisitor().visit_and_process_results(node.truthy)
            predicate_and_truthy = combine_and_dnf(predicate_dnf, truthy_dnf)

        if node.falsy.can_extract_literal:
            falsy_dnf_val = node.falsy.value
            # we use empty list for False
            not_predicate_and_falsy = predicate_dnf if falsy_dnf_val else []
        else:
            falsy_dnf = DNFVisitor().visit_and_process_results(node.falsy)
            # negate_dnf requires non-None DNF; if predicate_dnf is None, we cannot negate
            if predicate_dnf is not None:
                not_predicate_and_falsy = combine_and_dnf(negate_dnf(predicate_dnf), falsy_dnf)
            else:
                not_predicate_and_falsy = falsy_dnf
        self.result_dnf = combine_or_dnf(predicate_and_truthy, not_predicate_and_falsy)

    def visit_cast(self, node: CastNode) -> None:
        """Handle cast expressions."""
        if node.dtype == pl.Boolean:
            # If casting to boolean, process the underlying expression,
            # since it might be a duplicative expression.
            self.visit(node.input)

    def visit_alias(self, node: AliasNode) -> None:
        """Handle alias expressions by visiting the underlying input."""
        self.visit(node.input)

    def process_results(self) -> Optional[DNF]:
        """Return the accumulated DNF result."""
        return self.result_dnf


def combine_and_dnf(left_dnf: Optional[DNF], right_dnf: Optional[DNF]) -> Optional[DNF]:
    """Combine two DNFs with AND logic."""
    if left_dnf is None and right_dnf is None:
        return None
    elif left_dnf is None:
        return right_dnf
    elif right_dnf is None:
        return left_dnf

    result = []
    for left_clause in left_dnf:
        for right_clause in right_dnf:
            result.append(left_clause + right_clause)
    return result


def combine_or_dnf(left_dnf: Optional[DNF], right_dnf: Optional[DNF]) -> Optional[DNF]:
    """Combine two DNFs with OR logic."""
    if left_dnf is None or right_dnf is None:
        return None
    return left_dnf + right_dnf


def negate_operator(op: str) -> str:
    """Return the negated version of an operator."""
    if op.startswith("!"):
        return op[1:]

    operator_map = {
        "=": "!=",
        "==": "!=",
        "<>": "=",
        ">": "<=",
        "<": ">=",
        ">=": "<",
        "<=": ">",
        "~": "!~",
        "in": "!in",
        "not in": "in",
        "is": "is not",
        "is not": "is",
    }
    return operator_map.get(op, f"!{op}")


def negate_dnf(dnf: DNF) -> DNF:
    """Negate a DNF expression."""
    if not dnf:
        # Empty DNF means FALSE, negation is TRUE
        # Represent TRUE as an empty conjunction
        return [[]]
    cnf_result = []

    # Convert to CNF (NOT of DNF)
    for clause in dnf:
        negated_clause = []
        for col, op, val in clause:
            negated_clause.append((col, negate_operator(op), val))
        cnf_result.append(negated_clause)

    # Convert CNF to DNF using cartesian product
    result = []
    for combo in itertools.product(*cnf_result):
        result.append(list(combo))

    return result


## public API
def convert_expr_to_dnf(expr_or_node: Union[pl.Expr, BaseExprNode]) -> Optional[DNF]:
    """
    Convert a Polars expression tree to DNF format.
    Returns the expression in DNF format, or None if conversion fails.
    """
    try:
        node = get_parsed_expr(expr_or_node)
        visitor = DNFVisitor()
        visitor.visit(node)
        return visitor.process_results()
    except Exception:
        log.exception("Error converting expression to DNF")
        return None  # Conversion failed


def _is_contradiction(expr: pl.Expr, schema: Optional[pl.Schema] = None) -> bool:
    """Evaluates whether a  Polars expression is a contradiction.
    We perform this ourselves manually, as polars doesn't short-circuit
    if a set of filters is impossible. A full scan is performed.
    When this issue is resolved we can remove this code:
    https://github.com/pola-rs/polars/issues/21862
    """
    parsed_expr = get_parsed_expr(expr)
    # We special-case for the case where the expression is a literal False
    # to avoid the DNF conversion.
    if parsed_expr.can_extract_literal and parsed_expr.value is False:
        return True
    dnf_clause = convert_expr_to_dnf(parsed_expr)

    if not dnf_clause:
        # No dnf clause, so we have no contradiction
        return False
    for dnf in dnf_clause:
        if not is_contradiction(dnf, schema=schema):
            # Possible, not a contradiction
            log.debug("No contradiction in dnf %s, we must query %s", str(dnf), str(expr))
            return False
    return True
