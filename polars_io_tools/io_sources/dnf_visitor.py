import functools
import itertools
import logging
import re
from datetime import timedelta
from typing import Any, Literal

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
    extract_value_preserving_column_name,
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

DNFTuple = tuple[str, DNFOperator, Any]  # (column, op, value)
DNFClause = list[DNFTuple]  # A single conjunction
DNF = list[DNFClause]  # Full DNF expression


# Rewrites ``lit(1) < col`` as ``col > 1``; operators not listed are symmetric.
_FLIPPED_COMPARISON = {
    OperatorType.GT: OperatorType.LT,
    OperatorType.LT: OperatorType.GT,
    OperatorType.GT_EQ: OperatorType.LT_EQ,
    OperatorType.LT_EQ: OperatorType.GT_EQ,
}


__all__ = (
    "DNF",
    "DNFClause",
    "DNFTuple",
    "DNFVisitor",
    "convert_expr_to_dnf",
    "is_contradiction",  # Add new function to __all__
)


class ColumnConstraintAnalyzer:
    def __init__(self, column_name: str):
        self.column_name = column_name
        # Core constraints
        self.min_bound: tuple[Any, bool] | None = None  # (value, is_inclusive)
        self.max_bound: tuple[Any, bool] | None = None  # (value, is_inclusive)
        self.exact_values: set[Any] = set()
        self.inclusion_set: set[Any] | None = None  # From IN operators
        self.exclusion_values: set[Any] = set()  # From != and NOT IN
        self.is_null: bool | None = None  # True=IS NULL, False=IS NOT NULL
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
                if min_val > max_val or min_val == max_val and not (min_inclusive and max_inclusive):
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

    def has_contradiction(self, schema: pl.Schema | None = None) -> bool:
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


def is_contradiction(clause: DNFClause, schema: pl.Schema | None = None) -> bool:
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


class DNFVisitor(ExprVisitor[DNF | None]):
    """
    Visitor that converts expressions to DNF format.
    """

    def __init__(self, schema: pl.Schema | dict[str, pl.DataType] | None = None):
        # Initialize state
        self.result_dnf: DNF | None = None
        self.schema = schema
        # A converted DNF may keep more rows than the predicate (e.g. an AND with an unconvertible side dropped).
        # Negating such a superset gives a subset that drops rows, so NOT only negates results marked safe.
        self.negation_safe: bool = False

    def _subconvert(self, node: BaseExprNode) -> tuple[DNF | None, bool]:
        """Convert a child node, returning its DNF and whether that DNF is safe to negate."""
        sub = DNFVisitor(self.schema)
        sub.visit(node)
        return sub.result_dnf, sub.negation_safe

    def _combine(self, nodes: list[BaseExprNode], is_and: bool) -> None:
        """Combine the DNFs of ``nodes`` with AND or OR."""
        if not nodes:
            return
        results = [self._subconvert(n) for n in nodes]
        self.result_dnf = functools.reduce(combine_and_dnf if is_and else combine_or_dnf, (dnf for dnf, _ in results))
        self.negation_safe = all(dnf is not None and safe for dnf, safe in results)

    def default_result(self) -> DNF | None:
        """Default result is None."""
        return None

    def _set_comparison(self, column: str, op: OperatorType, value: Any) -> None:
        """Record the DNF for a ``column <op> value`` comparison (value already extracted)."""
        if op == OperatorType.EQ_VALIDITY:
            # ``~eq_missing(v)`` keeps null rows but ``!=`` drops them, so ``=`` is not safe to negate.
            if value is None:
                self.result_dnf = [[(column, "is", None)]]
                self.negation_safe = True
            else:
                self.result_dnf = [[(column, "=", value)]]
            return

        if op == OperatorType.NOT_EQ_VALIDITY:
            # ``ne_missing(v)`` keeps null rows, which ``!=`` alone would drop.
            if value is None:
                self.result_dnf = [[(column, "is not", None)]]
            else:
                self.result_dnf = [[(column, "!=", value)], [(column, "is", None)]]
            self.negation_safe = True
            return

        op_map = {
            OperatorType.EQ: "=",
            OperatorType.NOT_EQ: "!=",
            OperatorType.GT: ">",
            OperatorType.GT_EQ: ">=",
            OperatorType.LT: "<",
            OperatorType.LT_EQ: "<=",
        }
        if op in op_map:
            op_str: DNFOperator = op_map[op]  # type: ignore[assignment]
            self.result_dnf = [[(column, op_str, value)]]
            # A comparison with a null literal is null on every row, so it has no well-defined negation.
            self.negation_safe = value is not None

    def visit_binary_expr(self, node: BinaryExprNode) -> None:
        """Handle binary expressions (comparisons and logical operations)"""
        if node.op.is_comparison():
            # Handle comparison operators (=, !=, etc.), with the column on either side
            column = extract_value_preserving_column_name(node.left, self.schema)
            if column is not None and node.right.can_extract_literal:
                self._set_comparison(column, node.op, node.right.value)
            else:
                column = extract_value_preserving_column_name(node.right, self.schema)
                if column is not None and node.left.can_extract_literal:
                    self._set_comparison(column, _FLIPPED_COMPARISON.get(node.op, node.op), node.left.value)

        elif node.op.is_bitwise_or_logical():
            # Handle logical operations (AND, OR)
            if node.op in [OperatorType.AND, OperatorType.LOGICAL_AND]:
                self._combine([node.left, node.right], is_and=True)
            elif node.op in [OperatorType.OR, OperatorType.LOGICAL_OR]:
                self._combine([node.left, node.right], is_and=False)

    def visit_function(self, node: FunctionNode) -> None:
        """Handle function expressions (IS_NULL, IS_IN, etc.)"""
        # Handle IS_NULL function
        if node.function_type == BooleanFunctionType.IS_NULL:
            if len(node.inputs) > 0:
                column = extract_value_preserving_column_name(node.inputs[0], self.schema)
                if column is not None:
                    self.result_dnf = [[(column, "is", None)]]
                    self.negation_safe = True

        # Handle IS_NOT_NULL function
        elif node.function_type == BooleanFunctionType.IS_NOT_NULL:
            if len(node.inputs) > 0:
                column = extract_value_preserving_column_name(node.inputs[0], self.schema)
                if column is not None:
                    self.result_dnf = [[(column, "is not", None)]]
                    self.negation_safe = True

        # Handle IS_IN function
        elif node.function_type == BooleanFunctionType.IS_IN:
            if len(node.inputs) >= 2 and node.inputs[1].can_extract_literal:
                column = extract_value_preserving_column_name(node.inputs[0], self.schema)
                if column is not None:
                    values = node.inputs[1].value
                    if not isinstance(values, list):
                        # we do this for the case where the input is a single value
                        # and extracting that literal gives us a single value
                        values = [values]
                    # With ``nulls_equal=True``, ``is_in`` and its negation can match null rows, which ``in``/``!in`` drop.
                    nulls_equal = bool(node.options.get("nulls_equal", False))
                    non_null = [v for v in values if v is not None]
                    if nulls_equal and len(non_null) < len(values):
                        self.result_dnf = ([[(column, "in", non_null)]] if non_null else []) + [[(column, "is", None)]]
                    else:
                        self.result_dnf = [[(column, "in", values)]]
                    self.negation_safe = not nulls_equal

        # Handle IS_BETWEEN function
        elif node.function_type == BooleanFunctionType.IS_BETWEEN:
            if len(node.inputs) >= 3 and node.inputs[1].can_extract_literal and node.inputs[2].can_extract_literal:
                column = extract_value_preserving_column_name(node.inputs[0], self.schema)
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
                    self.negation_safe = True

        # Handle NOT function
        elif node.function_type == BooleanFunctionType.NOT:
            if len(node.inputs) > 0:
                inner_dnf, inner_safe = self._subconvert(node.inputs[0])
                if inner_dnf is not None and inner_safe:
                    self.result_dnf = negate_dnf(inner_dnf)
                    self.negation_safe = True

        # Handle ALL_HORIZONTAL function (ANDing all inputs)
        elif node.function_type == BooleanFunctionType.ALL_HORIZONTAL:
            self._combine(node.inputs, is_and=True)

        # Handle ANY_HORIZONTAL function (ORing all inputs)
        elif node.function_type == BooleanFunctionType.ANY_HORIZONTAL:
            self._combine(node.inputs, is_and=False)

        # Handle string functions
        elif isinstance(node.function_type, StringFunctionType) and len(node.inputs) >= 2 and node.inputs[1].can_extract_literal:
            column = extract_value_preserving_column_name(node.inputs[0], self.schema)
            if column is not None:
                pattern = node.inputs[1].value
                regex_pattern = None
                if node.function_type == StringFunctionType.STARTS_WITH:
                    regex_pattern = f"^{re.escape(pattern)}"
                elif node.function_type == StringFunctionType.ENDS_WITH:
                    regex_pattern = f"{re.escape(pattern)}$"
                elif node.function_type == StringFunctionType.CONTAINS:
                    regex_pattern = f".*{re.escape(pattern) if node.options.get('literal') else f'(?:{pattern})'}.*"
                if regex_pattern is not None:
                    self.result_dnf = [[(column, "~", regex_pattern)]]
                    self.negation_safe = True

    def visit_ternary(self, node: TernaryNode) -> None:
        """Visit ternary node"""
        # pl.when(P).then(T).otherwise(F) is the same as (P & T) | (~P & F).
        predicate_dnf, predicate_safe = self._subconvert(node.predicate)

        # A literal arm is a boolean; the empty DNF is False.
        if node.truthy.can_extract_literal:
            predicate_and_truthy = predicate_dnf if node.truthy.value else []
        else:
            truthy_dnf, _ = self._subconvert(node.truthy)
            predicate_and_truthy = combine_and_dnf(predicate_dnf, truthy_dnf)

        # Without a safe ~P, combine_and_dnf falls back to F alone, a superset of ~P & F.
        negated_predicate = None
        if predicate_dnf is not None and predicate_safe:
            # Polars also takes the otherwise branch where P is null, which happens only where a column in P is null.
            columns = dict.fromkeys(c for clause in predicate_dnf for c, _, _ in clause)
            negated_predicate = negate_dnf(predicate_dnf) + [[(c, "is", None)] for c in columns]
        if node.falsy.can_extract_literal:
            not_predicate_and_falsy = negated_predicate if node.falsy.value else []
        else:
            falsy_dnf, _ = self._subconvert(node.falsy)
            not_predicate_and_falsy = combine_and_dnf(negated_predicate, falsy_dnf)

        result = combine_or_dnf(predicate_and_truthy, not_predicate_and_falsy)
        self.result_dnf = None if result is None else [clause for clause in result if not is_contradiction(clause)]

    def visit_cast(self, node: CastNode) -> None:
        """Handle cast expressions."""
        if node.dtype == pl.Boolean:
            # If casting to boolean, process the underlying expression,
            # since it might be a duplicative expression.
            self.visit(node.input)

    def visit_alias(self, node: AliasNode) -> None:
        """Handle alias expressions by visiting the underlying input."""
        self.visit(node.input)

    def process_results(self) -> DNF | None:
        """Return the accumulated DNF result."""
        return self.result_dnf


def combine_and_dnf(left_dnf: DNF | None, right_dnf: DNF | None) -> DNF | None:
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


def combine_or_dnf(left_dnf: DNF | None, right_dnf: DNF | None) -> DNF | None:
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
def convert_expr_to_dnf(expr_or_node: pl.Expr | BaseExprNode, schema: pl.Schema | dict[str, pl.DataType] | None = None) -> DNF | None:
    """
    Convert a Polars expression tree to DNF format.
    Returns the expression in DNF format, or None if conversion fails.

    Args:
        expr_or_node: The Polars expression (or already-parsed node) to convert.
        schema: Optional column -> dtype mapping. Casts on a column are only looked through when the schema shows
            them to be value- and order-preserving.
    """
    try:
        node = get_parsed_expr(expr_or_node)
        visitor = DNFVisitor(schema)
        visitor.visit(node)
        return visitor.process_results()
    except Exception:
        log.exception("Error converting expression to DNF")
        return None  # Conversion failed


def _is_contradiction(expr: pl.Expr, schema: pl.Schema | None = None) -> bool:
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
