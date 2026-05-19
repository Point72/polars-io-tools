import logging
import re
from typing import Any, List, Optional, Union, cast

import polars as pl
import sqlglot
from sqlglot.dialects.dialect import Dialect, Dialects

from .base import AliasNode, BinaryExprNode, CastNode, ColumnNode, ExprVisitor, FunctionNode, LiteralNode, TernaryNode, get_parsed_expr
from .enum import ArrayFunctionType, BooleanFunctionType, ListFunctionType, OperatorType, StringFunctionType, TemporalFunctionType
from .sql_dialects import MSSQL

__all__ = [
    "fix_three_part_identifiers",
    "convert_predicate_to_sql",
    "apply_polars_io_source_exprs",
    "create_sqlglot_literal",
    "SQLExpressionVisitor",
]


# Configure logging
log = logging.getLogger(__name__)


def create_sqlglot_literal(value: Any) -> sqlglot.exp.Expression:
    """Create a sqlglot literal from a raw value.

    - None -> NULL
    - bool -> SQL boolean (TRUE / FALSE)
    - Numeric -> unquoted literal
    - Other types (str, date, datetime, time, etc.) -> quoted string literal
    """
    if value is None:
        return sqlglot.exp.Null()

    if isinstance(value, bool):
        return sqlglot.exp.Boolean(this=value)

    is_plain_numeric = isinstance(value, (int, float))
    return sqlglot.exp.Literal(
        this=str(value) if not isinstance(value, str) else value,
        is_string=not is_plain_numeric,  # quote when not numeric; among other things, this allows use of both datetime.datetime and datetime.time
    )


def fix_three_part_identifiers(node):
    """
    Recursively fix malformed three-part identifiers in sqlglot AST.

    Reference:
    https://sqlglot.com/sqlglot.html#build-and-modify-sql
    """

    # Handle Table nodes specifically
    if isinstance(node, sqlglot.exp.Table):
        table_id = node.this

        # Check for malformed three-part identifier pattern
        if isinstance(table_id, sqlglot.exp.Identifier) and _is_malformed_three_part_identifier(table_id):
            # Extract the parts
            db, schema, table = table_id.this.split(".")

            # Reconstruct the table with proper structure
            node.set("this", sqlglot.exp.Identifier(this=table, quoted=True))
            node.set("db", sqlglot.exp.Identifier(this=schema, quoted=True))
            node.set("catalog", sqlglot.exp.Identifier(this=db, quoted=True))

    return node


def _is_malformed_three_part_identifier(identifier):
    """Check if identifier contains malformed three-part name"""
    return (
        "." in identifier.this
        and identifier.args.get("quoted")
        and identifier.this.count(".") == 2
        and
        # Additional validation: ensure it looks like db.schema.table
        all(part.strip() for part in identifier.this.split("."))
    )


class SQLExpressionVisitor(ExprVisitor[Optional[sqlglot.exp.Expression]]):
    """
    Visitor that converts Polars expressions to SQLGlot expressions.
    """

    def __init__(self, dialect: Union[str, Dialects, type[Dialect], None] = Dialects.TSQL):
        # Normalize to ``Dialects`` enum so internal checks use
        # ``self.dialect == Dialects.TSQL`` instead of raw strings.
        if dialect is None or dialect is MSSQL:
            self.dialect: Dialects = Dialects.TSQL
        elif isinstance(dialect, Dialects):
            self.dialect = dialect
        elif isinstance(dialect, str):
            # Dialects is a str enum, so Dialects("tsql") == Dialects.TSQL
            try:
                self.dialect = Dialects(dialect)
            except ValueError:
                self.dialect = Dialects.DIALECT  # unknown → generic
        else:
            self.dialect = Dialects.TSQL
        self.result: Optional[sqlglot.exp.Expression] = None

    def default_result(self) -> Optional[sqlglot.exp.Expression]:
        """Default result is None."""
        return self.result

    def visit_column(self, node: ColumnNode) -> None:
        """Convert column reference to SQL column."""
        self.result = sqlglot.exp.Column(this=node.name)

    def visit_literal(self, node: LiteralNode) -> None:
        """Convert literal value to SQL literal."""

        value = node.value

        # Handle different literal types via central helper
        self.result = create_sqlglot_literal(value)

    def visit_binary_expr(self, node: BinaryExprNode) -> None:
        """Convert binary expression to SQL expression."""
        # Visit left and right nodes
        left_visitor = SQLExpressionVisitor(self.dialect)
        left_visitor.visit(node.left)
        left_expr = left_visitor.process_results()

        right_visitor = SQLExpressionVisitor(self.dialect)
        right_visitor.visit(node.right)
        right_expr = right_visitor.process_results()

        if node.op == OperatorType.AND:
            if left_expr is None and right_expr is not None:
                self.result = right_expr
                return
            elif right_expr is None and left_expr is not None:
                self.result = left_expr
                return
            elif left_expr is None and right_expr is None:
                self.result = None
                return
        else:
            # For other operators, if either expression is None, result should be None
            # This is the same as before (when the following conditional block was
            # the only thing here)
            if left_expr is None or right_expr is None:
                self.result = None
                return

        # Map operators to SQLGlot expressions
        op_map = {
            OperatorType.EQ: sqlglot.exp.EQ,
            OperatorType.EQ_VALIDITY: sqlglot.exp.EQ,
            OperatorType.NOT_EQ: sqlglot.exp.NEQ,
            OperatorType.NOT_EQ_VALIDITY: sqlglot.exp.NEQ,
            OperatorType.GT: sqlglot.exp.GT,
            OperatorType.GT_EQ: sqlglot.exp.GTE,
            OperatorType.LT: sqlglot.exp.LT,
            OperatorType.LT_EQ: sqlglot.exp.LTE,
            OperatorType.AND: sqlglot.exp.And,
            OperatorType.OR: sqlglot.exp.Or,
            OperatorType.LOGICAL_AND: sqlglot.exp.And,
            OperatorType.LOGICAL_OR: sqlglot.exp.Or,
            OperatorType.PLUS: sqlglot.exp.Add,
            OperatorType.MINUS: sqlglot.exp.Sub,
            OperatorType.MULTIPLY: sqlglot.exp.Mul,
            OperatorType.DIVIDE: sqlglot.exp.Div,
            OperatorType.MODULUS: sqlglot.exp.Mod,
            OperatorType.FLOOR_DIVIDE: lambda x, y: sqlglot.exp.Floor(this=sqlglot.exp.Div(this=x, expression=y)),
        }

        if node.op in op_map:
            binary_op_result = op_map[node.op](this=left_expr, expression=right_expr)  # type: ignore[call-arg]
            # We add parenthesis in order to enforce operator precedence
            self.result = sqlglot.exp.Paren(this=binary_op_result)
        else:
            # For unsupported operators, log and return None
            log.warning(f"Unsupported operator in SQL conversion: {node.op}")
            self.result = None

    def visit_function(self, node: FunctionNode) -> None:
        """Convert function calls to SQL functions."""
        # Process function inputs
        input_exprs = []
        for input_node in node.inputs:
            input_visitor = SQLExpressionVisitor(self.dialect)
            input_visitor.visit(input_node)
            input_expr = input_visitor.process_results()
            if input_expr is None:
                self.result = None
                return
            input_exprs.append(input_expr)

        # Handle different function types
        if isinstance(node.function_type, BooleanFunctionType):
            self._handle_boolean_function(node, input_exprs)
        elif isinstance(node.function_type, StringFunctionType):
            self._handle_string_function(node, input_exprs)
        elif isinstance(node.function_type, TemporalFunctionType):
            self._handle_temporal_function(node, input_exprs)
        elif isinstance(node.function_type, ListFunctionType) or isinstance(node.function_type, ArrayFunctionType):
            self._handle_list_function(node, input_exprs)
        else:
            log.warning(f"Unsupported function type in SQL conversion: {node.function_type}")
            self.result = None

    def _handle_boolean_function(self, node: FunctionNode, input_exprs: List[sqlglot.exp.Expression]) -> None:
        """Handle boolean functions."""
        if not input_exprs:
            self.result = None
            return

        if node.function_type == BooleanFunctionType.IS_NULL:
            self.result = sqlglot.exp.Is(this=input_exprs[0], expression=sqlglot.exp.Null())
        elif node.function_type == BooleanFunctionType.IS_NOT_NULL:
            self.result = sqlglot.exp.Not(this=sqlglot.exp.Is(this=input_exprs[0], expression=sqlglot.exp.Null()))
        elif node.function_type == BooleanFunctionType.IS_IN and len(input_exprs) >= 2:
            if isinstance(node.inputs[1].value, (list, tuple, set)):
                values = [create_sqlglot_literal(v) for v in node.inputs[1].value]
                self.result = sqlglot.exp.In(this=input_exprs[0], expressions=values)
            else:
                self.result = sqlglot.exp.In(this=input_exprs[0], expressions=[input_exprs[1]])
        elif node.function_type == BooleanFunctionType.NOT:
            self.result = sqlglot.exp.Not(this=input_exprs[0])
        elif node.function_type == BooleanFunctionType.IS_BETWEEN and len(input_exprs) >= 3:
            # Get closed parameter from options
            closed = node.options.get("closed", "Both")

            if closed == "Both":
                lower = sqlglot.exp.GTE(this=input_exprs[0], expression=input_exprs[1])
                upper = sqlglot.exp.LTE(this=input_exprs[0], expression=input_exprs[2])
            elif closed == "Left":
                lower = sqlglot.exp.GTE(this=input_exprs[0], expression=input_exprs[1])
                upper = sqlglot.exp.LT(this=input_exprs[0], expression=input_exprs[2])
            elif closed == "Right":
                lower = sqlglot.exp.GT(this=input_exprs[0], expression=input_exprs[1])
                upper = sqlglot.exp.LTE(this=input_exprs[0], expression=input_exprs[2])
            else:  # "Neither"
                lower = sqlglot.exp.GT(this=input_exprs[0], expression=input_exprs[1])
                upper = sqlglot.exp.LT(this=input_exprs[0], expression=input_exprs[2])

            self.result = sqlglot.exp.And(this=lower, expression=upper)
        else:
            log.warning(f"Unsupported boolean function: {node.function_type}")
            self.result = None

    def _dialect_supports_regexp(self) -> bool:
        """Check if the current dialect supports RegexpLike.

        Returns True for ClickHouse (match()), PostgreSQL (~), DuckDB (REGEXP_MATCHES), etc.
        Returns False for TSQL and SQLite which have no native regex support.
        """
        gen_class = type(sqlglot.Dialect.get_or_raise(self.dialect).generator())
        return sqlglot.exp.RegexpLike in gen_class.TRANSFORMS

    def _handle_string_function(self, node: FunctionNode, input_exprs: List[sqlglot.exp.Expression]) -> None:
        """Handle string functions."""
        if not input_exprs:
            self.result = None
            return

        if node.function_type == StringFunctionType.CONTAINS and len(input_exprs) >= 2:
            if isinstance(node.inputs[1].value, str):
                pattern_str = node.inputs[1].value
                is_literal = node.options.get("literal", False)
                # If regex mode and pattern has regex metacharacters, `LIKE` can't represent it.
                # Use RegexpLike for dialects that support it. Otherwise, skip pushdown - the database returns all rows and Polars applies the regex filter in-memory.
                if not is_literal and re.escape(pattern_str) != pattern_str:
                    if self._dialect_supports_regexp():
                        pattern_expr = create_sqlglot_literal(pattern_str)
                        self.result = sqlglot.exp.RegexpLike(this=input_exprs[0], expression=pattern_expr)
                    else:
                        self.result = None
                    return
                like_pattern = f"%{pattern_str}%"
                literal_expr = create_sqlglot_literal(like_pattern)
                self.result = sqlglot.exp.Like(this=input_exprs[0], expression=literal_expr)
            else:
                self.result = None
        elif node.function_type == StringFunctionType.STARTS_WITH and len(input_exprs) >= 2:
            if isinstance(node.inputs[1].value, str):
                pattern = f"{node.inputs[1].value}%"
                literal_expr = create_sqlglot_literal(pattern)
                self.result = sqlglot.exp.Like(this=input_exprs[0], expression=literal_expr)
            else:
                self.result = None
        elif node.function_type == StringFunctionType.ENDS_WITH and len(input_exprs) >= 2:
            if isinstance(node.inputs[1].value, str):
                pattern = f"%{node.inputs[1].value}"
                literal_expr = create_sqlglot_literal(pattern)
                self.result = sqlglot.exp.Like(this=input_exprs[0], expression=literal_expr)
            else:
                self.result = None
        elif node.function_type == StringFunctionType.UPPERCASE:
            self.result = sqlglot.exp.Upper(this=input_exprs[0])
        elif node.function_type == StringFunctionType.LOWERCASE:
            self.result = sqlglot.exp.Lower(this=input_exprs[0])
        else:
            log.warning(f"Unsupported string function: {node.function_type}")
            self.result = None

    def _handle_temporal_function(self, node: FunctionNode, input_exprs: List[sqlglot.exp.Expression]) -> None:
        """Handle temporal functions."""
        # Map of temporal functions to SQL functions
        # This is dialect-specific and may need adjustments

        # We have a fast-path for `TemporalFunctionType` objects
        total_mapping: dict[TemporalFunctionType, sqlglot.exp.Literal] = {
            TemporalFunctionType.TOTAL_DAYS: sqlglot.exp.Literal.string("DAY"),
            TemporalFunctionType.TOTAL_HOURS: sqlglot.exp.Literal.string("HOUR"),
            TemporalFunctionType.TOTAL_MINUTES: sqlglot.exp.Literal.string("MINUTE"),
            TemporalFunctionType.TOTAL_SECONDS: sqlglot.exp.Literal.string("SECOND"),
        }

        if node.function_type in total_mapping and self.dialect == Dialects.TSQL:
            if len(input_exprs) == 2:
                # SqlGlot DateDiff prints as DATEDIFF(unit, this, expression)
                # Cast is safe because we checked node.function_type in total_mapping (which only has TemporalFunctionType keys)
                temporal_type = cast(TemporalFunctionType, node.function_type)
                self.result = sqlglot.exp.DateDiff(
                    unit=total_mapping[temporal_type],
                    this=input_exprs[1],  # end-date
                    expression=input_exprs[0],  # start-date
                )
            else:
                log.warning(
                    "%s requires exactly two date inputs, got %d",
                    node.function_type,
                    len(input_exprs),
                )
                self.result = None
            return
        if self.dialect == Dialects.TSQL:  # SQL Server
            func_map = {
                TemporalFunctionType.YEAR: lambda x: sqlglot.exp.Extract(this="YEAR", expression=x),
                TemporalFunctionType.MONTH: lambda x: sqlglot.exp.Extract(this="MONTH", expression=x),
                TemporalFunctionType.DAY: lambda x: sqlglot.exp.Extract(this="DAY", expression=x),
                TemporalFunctionType.HOUR: lambda x: sqlglot.exp.Extract(this="HOUR", expression=x),
                TemporalFunctionType.MINUTE: lambda x: sqlglot.exp.Extract(this="MINUTE", expression=x),
                TemporalFunctionType.SECOND: lambda x: sqlglot.exp.Extract(this="SECOND", expression=x),
            }
        else:  # Generic SQL
            func_map = {
                TemporalFunctionType.YEAR: lambda x: sqlglot.exp.Extract(this="YEAR", expression=x),
                TemporalFunctionType.MONTH: lambda x: sqlglot.exp.Extract(this="MONTH", expression=x),
                TemporalFunctionType.DAY: lambda x: sqlglot.exp.Extract(this="DAY", expression=x),
                TemporalFunctionType.HOUR: lambda x: sqlglot.exp.Extract(this="HOUR", expression=x),
                TemporalFunctionType.MINUTE: lambda x: sqlglot.exp.Extract(this="MINUTE", expression=x),
                TemporalFunctionType.SECOND: lambda x: sqlglot.exp.Extract(this="SECOND", expression=x),
            }

        if node.function_type in func_map and input_exprs:
            self.result = func_map[node.function_type](input_exprs[0])
        else:
            log.warning(f"Unsupported temporal function: {node.function_type}")
            self.result = None

    def _handle_list_function(self, node: FunctionNode, input_exprs: List[sqlglot.exp.Expression]) -> None:
        """Handle list/array functions."""
        # Most list functions don't have direct SQL equivalents
        # We'll implement a few common ones
        if isinstance(node.function_type, ListFunctionType):
            if node.function_type == ListFunctionType.LENGTH and input_exprs:
                self.result = sqlglot.exp.ArraySize(this=input_exprs[0])
            elif node.function_type == ListFunctionType.CONTAINS and len(input_exprs) >= 2:
                # This is highly dialect-specific
                if self.dialect == Dialects.POSTGRES:
                    # PostgreSQL array contains
                    self.result = sqlglot.exp.Contains(this=input_exprs[0], expression=input_exprs[1])
                else:
                    log.warning(f"Array CONTAINS not supported in dialect: {self.dialect}")
                    self.result = None
            else:
                log.warning(f"Unsupported list function: {node.function_type}")
                self.result = None
        else:
            log.warning(f"Unsupported array function: {node.function_type}")
            self.result = None

    def visit_cast(self, node: CastNode) -> None:
        """Convert cast operations to SQL CAST."""
        # Visit the input node
        input_visitor = SQLExpressionVisitor(self.dialect)
        input_visitor.visit(node.input)
        input_expr = input_visitor.process_results()

        if input_expr is None:
            self.result = None
            return

        # This is a fast-path for boolean casts
        # Narwhals always wraps casts to boolean
        # We ignore if the cast is to `Unknown`, and
        # let the underlying SQL engine handle it
        if node.dtype in (pl.Boolean, pl.Unknown):
            self.result = input_expr
            return

        # Map Polars types to SQL types
        type_map = {
            pl.Int8: sqlglot.exp.DataType(this=sqlglot.exp.DataType.Type.TINYINT),
            pl.Int16: sqlglot.exp.DataType(this=sqlglot.exp.DataType.Type.SMALLINT),
            pl.Int32: sqlglot.exp.DataType(this=sqlglot.exp.DataType.Type.INT),
            pl.Int64: sqlglot.exp.DataType(this=sqlglot.exp.DataType.Type.BIGINT),
            pl.UInt8: sqlglot.exp.DataType(this=sqlglot.exp.DataType.Type.UTINYINT)
            if hasattr(sqlglot.exp.DataType.Type, "UTINYINT")
            else sqlglot.exp.DataType(this=sqlglot.exp.DataType.Type.TINYINT),
            pl.UInt16: sqlglot.exp.DataType(this=sqlglot.exp.DataType.Type.USMALLINT)
            if hasattr(sqlglot.exp.DataType.Type, "USMALLINT")
            else sqlglot.exp.DataType(this=sqlglot.exp.DataType.Type.SMALLINT),
            pl.UInt32: sqlglot.exp.DataType(this=sqlglot.exp.DataType.Type.UINT)
            if hasattr(sqlglot.exp.DataType.Type, "UINT")
            else sqlglot.exp.DataType(this=sqlglot.exp.DataType.Type.INT),
            pl.UInt64: sqlglot.exp.DataType(this=sqlglot.exp.DataType.Type.UBIGINT)
            if hasattr(sqlglot.exp.DataType.Type, "UBIGINT")
            else sqlglot.exp.DataType(this=sqlglot.exp.DataType.Type.BIGINT),
            pl.Float32: sqlglot.exp.DataType(this=sqlglot.exp.DataType.Type.FLOAT),
            pl.Float64: sqlglot.exp.DataType(this=sqlglot.exp.DataType.Type.DOUBLE),
            pl.Utf8: sqlglot.exp.DataType(this=sqlglot.exp.DataType.Type.VARCHAR),
            pl.Date: sqlglot.exp.DataType(this=sqlglot.exp.DataType.Type.DATE),
            pl.Datetime: sqlglot.exp.DataType(this=sqlglot.exp.DataType.Type.TIMESTAMP),
            pl.Time: sqlglot.exp.DataType(this=sqlglot.exp.DataType.Type.TIME),
            pl.Decimal: sqlglot.exp.DataType(this=sqlglot.exp.DataType.Type.DECIMAL),
        }

        # Get the SQL type
        sql_type = type_map.get(type(node.dtype), "VARCHAR")

        # Create the CAST expression
        self.result = sqlglot.exp.Cast(this=input_expr, to=sql_type)

    def visit_alias(self, node: AliasNode) -> None:
        """Convert alias expressions to SQL aliases."""
        # We DON'T add the alias to the input expression since that
        # can create incorrect SQL. we need to refer to the original column name.
        # For example, inside a WHERE, we can't use an alias expression, we need
        # the original name.
        self.visit(node.input)

    def visit_ternary(self, node: TernaryNode) -> None:
        """Convert ternary expressions to SQL CASE expressions."""
        # Visit the predicate, truthy, and falsy nodes
        pred_visitor = SQLExpressionVisitor(self.dialect)
        pred_visitor.visit(node.predicate)
        pred_expr = pred_visitor.process_results()

        true_visitor = SQLExpressionVisitor(self.dialect)
        true_visitor.visit(node.truthy)
        true_expr = true_visitor.process_results()

        false_visitor = SQLExpressionVisitor(self.dialect)
        false_visitor.visit(node.falsy)
        false_expr = false_visitor.process_results()

        if pred_expr is None:
            self.result = None
            return

        # We need to effectively handle Polars ternary syntax
        # Observe that pl.when(A).then(B).otherwise(C) is the
        # same as (A AND B) OR (NOT A AND C)

        # When both true and false branches are available, combine them.
        # If only the true branch is available --> enforce the condition "A AND B"
        # If only the false branch is available --> enforce the condition "NOT A AND C"
        # Return `None` if both branches are uknown
        if true_expr is not None and false_expr is not None:
            self.result = sqlglot.exp.Or(
                this=sqlglot.exp.And(this=pred_expr, expression=true_expr),
                expression=sqlglot.exp.And(this=sqlglot.exp.Not(this=pred_expr), expression=false_expr),
            )
        elif true_expr is not None:
            self.result = sqlglot.exp.And(this=pred_expr, expression=true_expr)
        elif false_expr is not None:
            self.result = sqlglot.exp.And(this=sqlglot.exp.Not(this=pred_expr), expression=false_expr)
        else:
            self.result = None


def convert_predicate_to_sql(predicate: pl.Expr, dialect: Union[str, type[Dialect], None] = "tsql") -> Optional[sqlglot.exp.Expression]:
    """
    Convert a Polars predicate expression to a SQLGlot expression.

    Args:
        predicate (pl.Expr): The Polars predicate expression
        dialect (str): SQL dialect to use

    Returns:
        Optional[sqlglot.exp.Expression]: SQLGlot expression or None if conversion failed
    """
    try:
        node = get_parsed_expr(predicate)

        # Use the visitor to convert to SQL
        visitor = SQLExpressionVisitor(dialect)
        visitor.visit(node)
        return visitor.process_results()
    except Exception as e:
        log.exception(f"Error converting predicate to SQL: {e}")
        return None


def _strip_table_qualifier(node: sqlglot.exp.Expression) -> sqlglot.exp.Expression:
    """Strip table qualifiers from column references for use in an outer query.

    When ORDER BY is hoisted from the inner subquery to the outer query, column
    references like ``cb.EventDate`` must become just ``EventDate`` because the
    table alias ``cb`` only exists inside the subquery.  The outer query sees
    flat column names from the subquery's result set.

    Known limitation: ``ORDER BY <ordinal>`` (e.g. ``ORDER BY 1``) is hoisted
    as-is.  If the outer SELECT reorders or drops columns, the ordinal may point
    to a different column — a silent semantic change.  Ordinal ORDER BY is rare
    in practice, but this is a real correctness gap.
    """
    if isinstance(node, sqlglot.exp.Column) and node.args.get("table"):
        return sqlglot.exp.Column(this=node.this.copy())
    return node


def _is_mssql_dialect(dialect: Optional[Union[str, type[Dialect]]]) -> bool:
    """Check whether the dialect is MSSQL."""
    return dialect is MSSQL


def _get_query_output_columns(query: sqlglot.exp.Expression) -> Optional[list[str]]:
    """Extract unqualified output column names from a SELECT's projection list.

    Returns None if names cannot be determined statically (e.g. SELECT *,
    expressions without aliases, non-Select nodes).
    """
    if not isinstance(query, sqlglot.exp.Select):
        return None
    names = []
    for expr in query.selects:
        if isinstance(expr, sqlglot.exp.Star):
            return None
        if isinstance(expr, sqlglot.exp.Alias):
            names.append(expr.alias)
        elif isinstance(expr, sqlglot.exp.Column):
            names.append(expr.name)
        else:
            return None
    return names


def _prepare_inner_for_subquery(
    parsed_query: sqlglot.exp.Expression,
    dialect: Optional[Union[str, type[Dialect]]] = None,
) -> tuple[sqlglot.exp.Expression, Optional[sqlglot.exp.Order], list[sqlglot.exp.Expression]]:
    """Prepare a query for subquery wrapping by extracting clauses that must be hoisted.

    Hoisting is **MSSQL-specific**.  Other dialects silently ignore ORDER BY in
    subqueries and don't have OPTION hints, so no hoisting is needed.  When
    *dialect* is not MSSQL the returned ``order_to_hoist`` and
    ``options_to_hoist`` will always be empty and the inner query is unchanged
    (apart from being copied).

    **MSSQL behaviour:**

    - ORDER BY in derived tables / subqueries is a hard error (error 1033) unless
      TOP or OFFSET is also present.  When neither is present we *move* the ORDER BY
      to the outer query so it can be re-applied there.  When TOP or OFFSET IS
      present the ORDER BY is meaningful (it determines which rows are kept) and
      stays in the inner query.

    - OPTION hints (e.g. ``OPTION (HASH JOIN)``) must appear at statement level,
      not inside a subquery — we always move them out.

    Returns ``(inner_query, order_to_hoist, options_to_hoist)`` where *inner_query*
    is a copy of the original with the hoisted clauses removed.
    """
    inner = parsed_query.copy()
    mssql = _is_mssql_dialect(dialect)

    # ORDER BY
    # MSSQL error 1033: ORDER BY is illegal in a subquery without TOP/OFFSET.
    # We move it to the outer query and strip table qualifiers (see
    # _strip_table_qualifier for details and known limitations).
    #
    # Non-MSSQL dialects: ORDER BY is harmlessly ignored in subqueries, so we
    # leave it in place — this avoids the ordinal-shift and column-not-in-SELECT
    # edge cases entirely.
    order_to_hoist: Optional[sqlglot.exp.Order] = None
    # OPTION hints
    # MSSQL requires OPTION(...) at statement level, not inside subqueries.
    # Other dialects don't use OPTION hints, so this is a no-op for them.
    options_to_hoist: list[sqlglot.exp.Expression] = []
    if mssql:
        order_node = inner.args.get("order")
        if order_node is not None:
            has_limit = inner.args.get("limit") is not None
            has_offset = inner.args.get("offset") is not None
            if not has_limit and not has_offset:
                order_to_hoist = order_node.copy()
                order_to_hoist = order_to_hoist.transform(_strip_table_qualifier)
                order_node.pop()
        for hint in list(inner.find_all(sqlglot.exp.QueryOption)):
            options_to_hoist.append(hint.copy())
            hint.pop()

    return inner, order_to_hoist, options_to_hoist


def apply_polars_io_source_exprs(
    query: sqlglot.exp.Expression,
    dialect: Optional[Union[str, type[Dialect]]],
    with_columns: Optional[List[str]],
    predicate: Optional[pl.Expr],
    n_rows: Optional[int],
    batch_size: Optional[int],
) -> sqlglot.exp.Expression:
    """Apply Polars IO source expressions using subquery wrapping.

    Wraps the original query as a subquery and applies column selection,
    predicates, and row limits on the outer query.
    """
    if with_columns is not None or predicate is not None or n_rows is not None:
        # When the only pushdown is a column selection that exactly matches the
        # query's own projection (a "redundant" select), skip subquery wrapping
        # entirely — return the original query unchanged.
        if with_columns is not None and predicate is None and n_rows is None:
            output_cols = _get_query_output_columns(query)
            if output_cols is not None and set(with_columns) == set(output_cols):
                return query.copy().transform(fix_three_part_identifiers)

        inner, order_to_hoist, options_to_hoist = _prepare_inner_for_subquery(query, dialect=dialect)

        subquery = sqlglot.exp.Subquery(
            this=inner,
            alias=sqlglot.exp.TableAlias(this=sqlglot.exp.Identifier(this="__cpl_subq")),
        )
        outer = sqlglot.exp.Select().from_(subquery, dialect=dialect)

        # Column selection — flat names from the subquery output
        if with_columns is not None:
            outer = outer.select(
                *[sqlglot.exp.Column(this=name) for name in with_columns],
                append=False,
                dialect=dialect,
            )
        else:
            outer = outer.select(sqlglot.exp.Star(), append=False, dialect=dialect)

        # Predicate — alias names resolve to real columns in the subquery
        # output so no rewriting is needed
        if predicate is not None:
            sql_predicate = convert_predicate_to_sql(predicate, dialect)
            if sql_predicate is not None:
                outer = outer.where(sql_predicate, dialect=dialect)

        # Row limit
        if n_rows is not None:
            outer = outer.limit(n_rows, dialect=dialect)

        # Re-apply hoisted clauses to the outer query
        if order_to_hoist is not None:
            outer.set("order", order_to_hoist)
        for opt in options_to_hoist:
            outer.args.setdefault("options", []).append(opt)

        modified_query = outer
    else:
        modified_query = query.copy()

    # Handle identifiers in the parsed tree
    query_fixed = modified_query.transform(fix_three_part_identifiers)
    return query_fixed
