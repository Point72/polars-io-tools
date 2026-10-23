import logging
import math
import re
from datetime import UTC, date, datetime, timedelta
from typing import Any, cast

import polars as pl
import sqlglot
from sqlglot.dialects.dialect import Dialect, Dialects

from .base import (
    AliasNode,
    BaseExprNode,
    BinaryExprNode,
    CastNode,
    ColumnNode,
    ExprVisitor,
    FunctionNode,
    LiteralNode,
    TernaryNode,
    get_parsed_expr,
    has_exact_value,
)
from .enum import ArrayFunctionType, BooleanFunctionType, ListFunctionType, OperatorType, StringFunctionType, TemporalFunctionType
from .sql_dialects import MSSQL

__all__ = (
    "DEFAULT_MAX_IN_PREDICATE_SIZE",
    "SQLExpressionVisitor",
    "apply_polars_io_source_exprs",
    "convert_predicate_to_sql",
    "create_sqlglot_literal",
    "fix_three_part_identifiers",
    "polars_dtype_to_sqlglot_type",
)


# Configure logging
log = logging.getLogger(__name__)


# Upper bound on the number of literals an ``is_in`` predicate may contain before we
# stop translating it into a SQL ``IN (...)`` clause. Very large lists (e.g. an entire
# key universe pushed down by a join) can exceed a backend's expression-complexity
# limits — SQL Server raises error 8632 ("An expression services limit has been
# reached"). Skipping pushdown for such a predicate is always correct: the polars IO
# source re-applies the full predicate to the fetched rows, so only the pre-filter
# narrowing (an optimization) is lost.
DEFAULT_MAX_IN_PREDICATE_SIZE = 4096


def polars_dtype_to_sqlglot_type(dtype: pl.DataType | type[pl.DataType], *, strict: bool = False) -> sqlglot.exp.DataType:
    """
    Convert a Polars dtype (class or instance) to a SQLGlot ``DataType``.

    Decimal precision and scale are preserved so a decimal cast does not silently
    truncate the fractional part.

    Args:
        dtype (pl.DataType): The Polars dtype to translate.
        strict (bool): When True, raise for a dtype with no dedicated SQL mapping
            instead of falling back to ``VARCHAR``.

    Returns:
        sqlglot.exp.DataType: The SQL type to cast to.

    Raises:
        ValueError: If ``strict`` is set and ``dtype`` has no dedicated SQL mapping.
    """
    type_map = {
        pl.Int8: sqlglot.exp.DataType.Type.TINYINT,
        pl.Int16: sqlglot.exp.DataType.Type.SMALLINT,
        pl.Int32: sqlglot.exp.DataType.Type.INT,
        pl.Int64: sqlglot.exp.DataType.Type.BIGINT,
        pl.UInt8: getattr(sqlglot.exp.DataType.Type, "UTINYINT", sqlglot.exp.DataType.Type.TINYINT),
        pl.UInt16: getattr(sqlglot.exp.DataType.Type, "USMALLINT", sqlglot.exp.DataType.Type.SMALLINT),
        pl.UInt32: getattr(sqlglot.exp.DataType.Type, "UINT", sqlglot.exp.DataType.Type.INT),
        pl.UInt64: getattr(sqlglot.exp.DataType.Type, "UBIGINT", sqlglot.exp.DataType.Type.BIGINT),
        pl.Float32: sqlglot.exp.DataType.Type.FLOAT,
        pl.Float64: sqlglot.exp.DataType.Type.DOUBLE,
        pl.Utf8: sqlglot.exp.DataType.Type.VARCHAR,
        pl.Date: sqlglot.exp.DataType.Type.DATE,
        pl.Datetime: sqlglot.exp.DataType.Type.TIMESTAMP,
        pl.Time: sqlglot.exp.DataType.Type.TIME,
        pl.Decimal: sqlglot.exp.DataType.Type.DECIMAL,
    }
    key = dtype if isinstance(dtype, type) else type(dtype)
    base = type_map.get(key)
    if base is None:
        if strict:
            supported = ", ".join(sorted(t.__name__ for t in type_map))
            raise ValueError(f"No SQL type mapping for Polars dtype {dtype!r}. Supported dtypes: {supported}.")
        return sqlglot.exp.DataType(this=sqlglot.exp.DataType.Type.VARCHAR)

    # TIMESTAMP has no time zone, so labelling the local wall-clock times it returns with one would shift them.
    if strict and key is pl.Datetime and not isinstance(dtype, type) and dtype.time_zone is not None:
        raise ValueError(f"No SQL type mapping for time-zone-aware {dtype!r}. Cast to a naive Datetime instead.")

    # Preserve decimal precision/scale so the cast keeps the fractional part.
    if key is pl.Decimal and not isinstance(dtype, type):
        precision = getattr(dtype, "precision", None)
        if precision is not None:
            scale = getattr(dtype, "scale", None) or 0
            return sqlglot.exp.DataType.build(f"DECIMAL({precision}, {scale})")

    return sqlglot.exp.DataType(this=base)


def _to_datetime64(text: str, precision: int, *, tz_aware: bool) -> sqlglot.exp.Expression:
    """Build a ClickHouse ``toDateTime64(text, precision[, 'UTC'])`` instant literal."""
    args: list[sqlglot.exp.Expression] = [sqlglot.exp.Literal.string(text), sqlglot.exp.Literal.number(precision)]
    if tz_aware:
        args.append(sqlglot.exp.Literal.string("UTC"))
    return sqlglot.exp.func("toDateTime64", *args)


def _clickhouse_datetime_literal(expr: pl.Expr) -> sqlglot.exp.Expression | None:
    """Render a datetime literal as a full-precision ClickHouse ``toDateTime64`` instant.

    ``LiteralNode.value`` is a microsecond-max Python ``datetime``, so a ``Datetime('ns')``
    literal -- and the bounds of the nanosecond ``is_between`` Polars builds for a nanosecond
    column -- would truncate and drop matching rows. The physical value keeps full precision;
    render the UTC instant (tz-aware) or wall-clock instant (naive) at the literal's own
    resolution. Returns ``None`` for anything that is not a single, representable Datetime value,
    so the caller skips pushdown and leaves the exact client-side filter in place.
    """
    try:
        series = pl.DataFrame().select(expr).to_series()
    except Exception:  # noqa: BLE001 -- defensive: skip pushdown if the literal can't be evaluated
        return None
    dtype = series.dtype
    if not isinstance(dtype, pl.Datetime) or len(series) != 1:
        return None
    ticks = series.to_physical().to_list()[0]
    if ticks is None:
        return None
    ticks_per_second, precision = {"ms": (1_000, 3), "us": (1_000_000, 6), "ns": (1_000_000_000, 9)}[dtype.time_unit]
    seconds, fraction = divmod(ticks, ticks_per_second)
    try:
        instant = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)
    except (OverflowError, OSError):
        return None
    text = instant.strftime("%Y-%m-%d %H:%M:%S") + (f".{fraction:0{precision}d}" if fraction else "")
    return _to_datetime64(text, precision, tz_aware=dtype.time_zone is not None)


def _is_pushable_value(value: Any) -> bool:
    """Return False for literal values with no portable SQL spelling (non-finite floats, durations, binary, strings with NUL)."""
    if isinstance(value, str):
        return "\x00" not in value
    return not isinstance(value, (timedelta, bytes)) and not (isinstance(value, float) and not math.isfinite(value))


_PREDICATE_TYPES = (sqlglot.exp.Predicate, sqlglot.exp.Connector, sqlglot.exp.Not)


def _is_boolean_sql(expr: sqlglot.exp.Expression) -> bool:
    """Return True if ``expr`` is known to produce a boolean in SQL."""
    return isinstance(expr.unnest(), (*_PREDICATE_TYPES, sqlglot.exp.Boolean))


def _like(column: sqlglot.exp.Expression, text: str, prefix: str = "", suffix: str = "") -> sqlglot.exp.Expression | None:
    """Build ``column LIKE '<prefix><text><suffix>'``, or None if ``text`` has characters LIKE would treat specially."""
    # Escaping these is dialect-specific (backslash, ESCAPE clause, T-SQL brackets), so skip pushdown instead.
    if any(char in text for char in "%_\\["):
        return None
    return sqlglot.exp.Like(this=column, expression=create_sqlglot_literal(f"{prefix}{text}{suffix}"))


def create_sqlglot_literal(value: Any, dialect: str | Dialects | type[Dialect] | None = None) -> sqlglot.exp.Expression:
    """Create a sqlglot literal from a raw value.

    - None -> NULL
    - bool -> SQL boolean (TRUE / FALSE)
    - Numeric -> unquoted literal
    - Timezone-aware datetime -> the equivalent UTC instant. ClickHouse renders it as an
      explicit ``toDateTime64(..., 'UTC')`` because it rejects an offset-bearing string
      against a ``DateTime64`` column; other dialects keep the offset string unchanged.
    - Other types (str, date, naive datetime, time, etc.) -> quoted string literal

    ``dialect`` selects backend-specific rendering; it currently only affects
    timezone-aware datetimes.
    """
    if value is None:
        return sqlglot.exp.Null()

    if isinstance(value, bool):
        return sqlglot.exp.Boolean(this=value)

    if isinstance(value, date) and not isinstance(value, datetime) and dialect == Dialects.ORACLE:
        return sqlglot.exp.DateStrToDate(this=sqlglot.exp.Literal.string(str(value)))

    if isinstance(value, datetime) and value.utcoffset() is not None and dialect == Dialects.CLICKHOUSE:
        # A tz-aware datetime stringifies with a UTC offset (e.g. "...+00:00") that ClickHouse
        # rejects against a DateTime64 column. Emit an explicit-UTC instant so the comparison is
        # correct regardless of the column's own timezone (a bare naive string would be parsed in
        # the column's tz and could shift the bounds). This microsecond path serves ``is_in``
        # values; scalar comparisons preserve full precision via ``visit_literal``.
        utc_naive = value.astimezone(UTC).replace(tzinfo=None)
        return _to_datetime64(str(utc_naive), 6, tz_aware=True)

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


# Dialects whose LIKE is case-sensitive by default.
_CASE_SENSITIVE_LIKE_DIALECTS = (Dialects.DUCKDB, Dialects.POSTGRES, Dialects.CLICKHOUSE)


class SQLExpressionVisitor(ExprVisitor[sqlglot.exp.Expression | None]):
    """
    Visitor that converts Polars expressions to SQLGlot expressions.
    """

    def __init__(
        self,
        dialect: str | Dialects | type[Dialect] | None = Dialects.TSQL,
        max_in_predicate_size: int = DEFAULT_MAX_IN_PREDICATE_SIZE,
    ):
        # Maximum ``is_in`` list length to translate into a SQL ``IN (...)`` clause;
        # larger lists skip pushdown (see ``DEFAULT_MAX_IN_PREDICATE_SIZE``). Propagated
        # to child visitors so the guard applies at any depth of the expression tree.
        self.max_in_predicate_size = max_in_predicate_size
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
        elif isinstance(dialect, type) and issubclass(dialect, Dialect):
            # Map a SQLGlot dialect class (e.g. ClickHouse) to its Dialects enum by name, so
            # dialect-specific handling is not lost; an unknown class falls back to generic.
            try:
                self.dialect = Dialects(dialect.__name__.lower())
            except ValueError:
                self.dialect = Dialects.DIALECT
        else:
            self.dialect = Dialects.TSQL
        self.result: sqlglot.exp.Expression | None = None
        # True while this node only feeds AND/OR up to the root of the predicate, so dropping part of it
        # can only widen the SQL filter. Under NOT, comparisons, CASE, etc. it must be translated exactly.
        self._monotone = True

    def _visit_child(self, node: BaseExprNode, monotone: bool = False) -> sqlglot.exp.Expression | None:
        """Translate a sub-expression, parenthesising boolean results so they nest with the right precedence."""
        visitor = SQLExpressionVisitor(self.dialect, self.max_in_predicate_size)
        visitor._monotone = monotone
        visitor.visit(node)
        result = visitor.process_results()
        if isinstance(result, _PREDICATE_TYPES):
            result = sqlglot.exp.Paren(this=result)
        return result

    def default_result(self) -> sqlglot.exp.Expression | None:
        """Default result is None."""
        return self.result

    def visit_column(self, node: ColumnNode) -> None:
        """Convert column reference to SQL column."""
        self.result = sqlglot.exp.Column(this=node.name)

    def visit_literal(self, node: LiteralNode) -> None:
        """Convert literal value to SQL literal."""

        value = node.value
        if not _is_pushable_value(value):
            self.result = None
            return

        # ClickHouse DateTime64 keeps up to nanoseconds, but ``node.value`` is microsecond-max;
        # render the literal from its source expression so nanosecond-column filters stay exact.
        # ``None`` (unrepresentable) skips pushdown, leaving the exact client-side filter.
        if self.dialect == Dialects.CLICKHOUSE and isinstance(value, datetime):
            self.result = _clickhouse_datetime_literal(node.expr)
            return
        if not has_exact_value(node):
            self.result = None
            return

        # Handle different literal types via central helper
        self.result = create_sqlglot_literal(value, self.dialect)

    def visit_binary_expr(self, node: BinaryExprNode) -> None:
        """Convert binary expression to SQL expression."""
        monotone = self._monotone and node.op in (OperatorType.AND, OperatorType.OR, OperatorType.LOGICAL_AND, OperatorType.LOGICAL_OR)
        left_expr = self._visit_child(node.left, monotone)
        right_expr = self._visit_child(node.right, monotone)

        if node.op == OperatorType.AND and self._monotone and (left_expr is None) != (right_expr is None):
            # Dropping an untranslatable side of an AND only widens the filter; Polars re-applies the full predicate.
            self.result = left_expr if left_expr is not None else right_expr
            return
        if left_expr is None or right_expr is None:
            self.result = None
            return
        if node.op in (OperatorType.AND, OperatorType.OR) and not self._monotone and not (_is_boolean_sql(left_expr) and _is_boolean_sql(right_expr)):
            # ``&`` and ``|`` are bitwise on integers; outside the filter's top level the operands may not be booleans.
            self.result = None
            return

        # Arithmetic is not translated: Polars integers wrap on overflow and ``/`` is always true division.
        op_map = {
            OperatorType.EQ: sqlglot.exp.EQ,
            OperatorType.NOT_EQ: sqlglot.exp.NEQ,
            OperatorType.GT: sqlglot.exp.GT,
            OperatorType.GT_EQ: sqlglot.exp.GTE,
            OperatorType.LT: sqlglot.exp.LT,
            OperatorType.LT_EQ: sqlglot.exp.LTE,
            OperatorType.AND: sqlglot.exp.And,
            OperatorType.OR: sqlglot.exp.Or,
            OperatorType.LOGICAL_AND: sqlglot.exp.And,
            OperatorType.LOGICAL_OR: sqlglot.exp.Or,
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
        # The ``is_in`` collection is rendered from its literal value, not visited as a scalar.
        inputs = node.inputs[:1] if node.function_type == BooleanFunctionType.IS_IN else node.inputs
        for input_node in inputs:
            input_expr = self._visit_child(input_node)
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
        elif isinstance(node.function_type, (ListFunctionType, ArrayFunctionType)):
            self._handle_list_function(node, input_exprs)
        else:
            log.warning(f"Unsupported function type in SQL conversion: {node.function_type}")
            self.result = None

    def _handle_boolean_function(self, node: FunctionNode, input_exprs: list[sqlglot.exp.Expression]) -> None:
        """Handle boolean functions."""
        if not input_exprs:
            self.result = None
            return

        if node.function_type == BooleanFunctionType.IS_NULL:
            self.result = sqlglot.exp.Is(this=input_exprs[0], expression=sqlglot.exp.Null())
        elif node.function_type == BooleanFunctionType.IS_NOT_NULL:
            self.result = sqlglot.exp.Not(this=sqlglot.exp.Is(this=input_exprs[0], expression=sqlglot.exp.Null()))
        elif node.function_type == BooleanFunctionType.IS_IN and len(node.inputs) >= 2:
            self.result = self._is_in(node, input_exprs[0])
        elif node.function_type == BooleanFunctionType.NOT:
            # ``~`` is bitwise on integers; at the top level of a filter the operand must be boolean.
            self.result = sqlglot.exp.Not(this=input_exprs[0]) if self._monotone or _is_boolean_sql(input_exprs[0]) else None
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

    def _is_in(self, node: FunctionNode, column: sqlglot.exp.Expression) -> sqlglot.exp.Expression | None:
        """Translate ``is_in`` against a literal collection."""
        nulls_equal = node.options.get("nulls_equal", False)
        if nulls_equal and not self._monotone:
            # Polars returns False on null rows where SQL ``IN`` returns null, so NOT would drop them.
            return None
        in_values = node.inputs[1].value
        if not isinstance(in_values, (list, tuple, set)):
            return None
        if len(in_values) > self.max_in_predicate_size:
            # Oversized ``IN (...)`` lists can exceed a backend's expression
            # limits (e.g. SQL Server error 8632). Skip pushdown; the IO source
            # re-applies the full predicate to the fetched rows for correctness.
            log.debug(
                f"Skipping IN-list pushdown for column with {len(in_values)} values (exceeds max_in_predicate_size={self.max_in_predicate_size})"
            )
            return None
        # A NULL in a SQL ``IN`` list turns non-matches into null, so drop it and match null rows explicitly for ``nulls_equal``.
        non_null = [v for v in in_values if v is not None]
        if not non_null or not all(_is_pushable_value(v) for v in non_null) or not has_exact_value(node.inputs[1], collection=True):
            return None
        floats = [v for v in non_null if isinstance(v, float)]
        if floats and not self._monotone and pl.Series(floats).cast(pl.Float32).cast(pl.Float64).to_list() != floats:
            # Polars matches a Float32 column against these as Float64, but a database may round them to FLOAT.
            return None
        result = sqlglot.exp.In(this=column, expressions=[create_sqlglot_literal(v, self.dialect) for v in non_null])
        if nulls_equal and len(non_null) < len(in_values):
            return sqlglot.exp.Or(this=sqlglot.exp.Paren(this=result), expression=sqlglot.exp.Is(this=column.copy(), expression=sqlglot.exp.Null()))
        return result

    def _handle_string_function(self, node: FunctionNode, input_exprs: list[sqlglot.exp.Expression]) -> None:
        """Handle string functions."""
        like_affixes = {StringFunctionType.CONTAINS: ("%", "%"), StringFunctionType.STARTS_WITH: ("", "%"), StringFunctionType.ENDS_WITH: ("%", "")}
        if node.function_type not in like_affixes or len(input_exprs) < 2:
            # Not translated: strip_chars (Polars strips all whitespace or given characters, SQL TRIM only spaces),
            # concat_str (separator, and null handling of SQL CONCAT varies by dialect) and to_uppercase/to_lowercase
            # (Unicode case mapping, e.g. "ß" -> "SS", varies by database), among others.
            log.warning(f"Unsupported string function: {node.function_type}")
            self.result = None
            return

        text = node.inputs[1].value
        if not self._monotone and self.dialect not in _CASE_SENSITIVE_LIKE_DIALECTS:
            # LIKE may ignore case (SQLite, case-insensitive collations), which is only safe where extra rows are allowed.
            self.result = None
        elif not isinstance(text, str) or (
            # Regex syntax and semantics (e.g. Unicode ``\d``) differ between Polars and databases, so only plain text is pushed.
            node.function_type == StringFunctionType.CONTAINS and not node.options.get("literal", False) and re.escape(text) != text
        ):
            self.result = None
        else:
            self.result = _like(input_exprs[0], text, *like_affixes[node.function_type])

    def _handle_temporal_function(self, node: FunctionNode, input_exprs: list[sqlglot.exp.Expression]) -> None:
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

        def _date_floor(x: sqlglot.exp.Expression) -> sqlglot.exp.Expression:
            """Floor a timestamp to its calendar date, matching Polars ``.dt.date()``."""
            if self.dialect == Dialects.CLICKHOUSE:
                # ClickHouse ``Date`` spans only 1970-2149; ``Date32`` (1900-2299) covers the
                # boundaries where ``Date`` diverges from Polars.
                return sqlglot.exp.Cast(this=x, to=sqlglot.exp.DataType(this=sqlglot.exp.DataType.Type.DATE32))
            if self.dialect == Dialects.ORACLE:
                # Oracle ``DATE`` retains the time component, so a plain cast keeps the clock
                # time and a noon timestamp would not match a midnight date literal; ``TRUNC``
                # floors to the calendar day.
                return sqlglot.exp.func("TRUNC", x)
            if self.dialect == Dialects.SQLITE:
                # SQLite has no date type; ``CAST(... AS DATE)`` coerces to a numeric (e.g. 2024),
                # so use the ``DATE()`` function to extract the calendar date.
                return sqlglot.exp.Date(this=x)
            return sqlglot.exp.Cast(this=x, to=sqlglot.exp.DataType(this=sqlglot.exp.DataType.Type.DATE))

        def _extract(part: str):
            # SQLite has no ``EXTRACT``.
            return lambda x: None if self.dialect == Dialects.SQLITE else sqlglot.exp.Extract(this=sqlglot.exp.Var(this=part), expression=x)

        # SECOND is left out: some dialects (e.g. PostgreSQL) return fractional seconds where Polars returns an integer.
        func_map = {
            TemporalFunctionType.YEAR: _extract("YEAR"),
            TemporalFunctionType.MONTH: _extract("MONTH"),
            TemporalFunctionType.DAY: _extract("DAY"),
            TemporalFunctionType.HOUR: _extract("HOUR"),
            TemporalFunctionType.MINUTE: _extract("MINUTE"),
            TemporalFunctionType.DATE: _date_floor,
        }

        if node.function_type in func_map and input_exprs:
            self.result = func_map[node.function_type](input_exprs[0])
        else:
            log.warning(f"Unsupported temporal function: {node.function_type}")
            self.result = None

    def _handle_list_function(self, node: FunctionNode, input_exprs: list[sqlglot.exp.Expression]) -> None:
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
        # Boolean and Unknown casts pass the input through unchanged (Narwhals wraps predicates in a boolean cast).
        if node.dtype == pl.Unknown:
            self.result = self._visit_child(node.input, monotone=self._monotone)
            return
        if node.dtype == pl.Boolean:
            # A boolean cast of a number tests for non-zero, so it is only dropped when the input is exactly a boolean.
            result = self._visit_child(node.input)
            self.result = result if result is not None and _is_boolean_sql(result) else None
            return

        # Only strict float casts are translated. Others differ from SQL: non-strict casts give null on failure, integer and
        # Decimal targets round differently, string formatting varies by database, and Polars reads integers as epoch
        # offsets when casting to temporal types.
        if node.options != "Strict" or not node.dtype.is_float():
            self.result = None
            return
        try:
            sql_type = polars_dtype_to_sqlglot_type(node.dtype, strict=True)
        except ValueError:
            self.result = None
            return

        input_expr = self._visit_child(node.input)
        self.result = None if input_expr is None else sqlglot.exp.Cast(this=input_expr, to=sql_type)

    def visit_alias(self, node: AliasNode) -> None:
        """Convert alias expressions to SQL aliases."""
        # We DON'T add the alias to the input expression since that
        # can create incorrect SQL. we need to refer to the original column name.
        # For example, inside a WHERE, we can't use an alias expression, we need
        # the original name.
        self.visit(node.input)

    def visit_ternary(self, node: TernaryNode) -> None:
        """Convert ternary expressions to SQL CASE expressions."""
        # T-SQL and Oracle have no boolean values, so a CASE cannot return a predicate there.
        if self.dialect in (Dialects.TSQL, Dialects.ORACLE):
            self.result = None
            return
        parts = [self._visit_child(child) for child in (node.predicate, node.truthy, node.falsy)]
        if any(part is None for part in parts):
            self.result = None
            return
        pred_expr, true_expr, false_expr = parts
        # A null condition falls through to ELSE, matching Polars ``otherwise``.
        self.result = sqlglot.exp.Case(ifs=[sqlglot.exp.If(this=pred_expr, true=true_expr)], default=false_expr)


def convert_predicate_to_sql(
    predicate: pl.Expr,
    dialect: str | type[Dialect] | None = "tsql",
    max_in_predicate_size: int = DEFAULT_MAX_IN_PREDICATE_SIZE,
) -> sqlglot.exp.Expression | None:
    """
    Convert a Polars predicate expression to a SQLGlot expression.

    Args:
        predicate (pl.Expr): The Polars predicate expression
        dialect (str): SQL dialect to use
        max_in_predicate_size (int): Maximum ``is_in`` list length translated to a SQL
            ``IN (...)`` clause; larger lists skip pushdown (see
            ``DEFAULT_MAX_IN_PREDICATE_SIZE``).

    Returns:
        Optional[sqlglot.exp.Expression]: SQLGlot expression or None if conversion failed
    """
    try:
        node = get_parsed_expr(predicate)

        # Use the visitor to convert to SQL
        visitor = SQLExpressionVisitor(dialect, max_in_predicate_size)
        visitor.visit(node)
        return visitor.process_results()
    except (Exception, pl.exceptions.PanicException):
        log.exception("Error converting predicate to SQL")
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


def _is_mssql_dialect(dialect: str | type[Dialect] | None) -> bool:
    """Check whether the dialect is MSSQL."""
    return dialect is MSSQL


def _is_known_dialect(dialect: str | type[Dialect] | None) -> bool:
    """Check whether SQLGlot recognizes the provided dialect."""
    if dialect is None:
        return False
    try:
        Dialect.get_or_raise(dialect)
    except ValueError:
        return False
    return True


def _quote_column_leaf_identifier(node: sqlglot.exp.Expression) -> sqlglot.exp.Expression:
    """Quote only the leaf identifier of a column expression."""
    if isinstance(node, sqlglot.exp.Column) and not node.is_star:
        identifier = node.this
        if isinstance(identifier, sqlglot.exp.Identifier):
            identifier.set("quoted", True)
        elif isinstance(identifier, str):
            node.set("this", sqlglot.exp.Identifier(this=identifier, quoted=True))
    return node


def _leftmost_output_select(query: sqlglot.exp.Expression) -> sqlglot.exp.Select | None:
    """Follow output-defining left branches to the leftmost SELECT."""
    current = query
    while True:
        if isinstance(current, sqlglot.exp.Select):
            return current
        if isinstance(current, sqlglot.exp.SetOperation):
            current = current.this
            continue
        if isinstance(current, (sqlglot.exp.Paren, sqlglot.exp.Subquery)):
            current = current.this
            continue
        return None


def _stabilize_mssql_projection_aliases(query: sqlglot.exp.Expression) -> None:
    """Quote explicit aliases and name simple columns in the output-defining SELECT."""
    output_select = _leftmost_output_select(query)
    if output_select is None:
        return

    projections = []
    for projection in output_select.selects:
        if projection.is_star:
            projections.append(projection)
            continue
        if isinstance(projection, sqlglot.exp.Alias):
            alias = projection.args.get("alias")
            if isinstance(alias, sqlglot.exp.Identifier):
                alias.set("quoted", True)
            projections.append(projection)
            continue
        if isinstance(projection, sqlglot.exp.Column) and projection.output_name:
            projections.append(
                sqlglot.exp.Alias(
                    this=projection,
                    alias=sqlglot.exp.Identifier(this=projection.output_name, quoted=True),
                )
            )
            continue
        projections.append(projection)

    output_select.set("expressions", projections)


def _get_query_output_columns(query: sqlglot.exp.Expression) -> list[str] | None:
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
    dialect: str | type[Dialect] | None = None,
) -> tuple[sqlglot.exp.Expression, sqlglot.exp.Order | None, list[sqlglot.exp.Expression]]:
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
    order_to_hoist: sqlglot.exp.Order | None = None
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


def wrap_query_with_casts(
    query: sqlglot.exp.Expression,
    dialect: str | type[Dialect] | None,
    ordered_columns: list[str],
    cast_map: dict[str, pl.DataType | type[pl.DataType]],
) -> sqlglot.exp.Expression:
    """Wrap ``query`` in a projecting subquery that casts selected columns server-side.

    Every column in ``ordered_columns`` is re-projected (preserving order); columns
    present in ``cast_map`` are wrapped in a SQL ``CAST`` to the mapped type, the rest
    pass through untouched. The original query is left intact as the inner relation,
    so ``select *`` keeps flowing new upstream columns.

    The narrowing lands one subquery level *below* the predicate/projection pushdown
    added later by :func:`apply_polars_io_source_exprs`, so a filter on a cast column
    (e.g. ``datetime`` narrowed to ``date``) resolves against the already-cast output
    column and is pushed to the database rather than evaluated client-side.

    Clauses that are illegal inside a derived table (an MSSQL top-level ``ORDER BY``
    without ``TOP``/``OFFSET``, or ``OPTION`` hints) are hoisted onto the cast
    ``SELECT`` so the nested subquery stays valid; the later wrapping in
    :func:`apply_polars_io_source_exprs` hoists them again to statement level.
    """
    quote = _is_known_dialect(dialect)

    def _ident(name: str) -> sqlglot.exp.Identifier:
        return sqlglot.exp.Identifier(this=name, quoted=quote)

    inner_query, order_to_hoist, options_to_hoist = _prepare_inner_for_subquery(query, dialect=dialect)

    projections: list[sqlglot.exp.Expression] = []
    for name in ordered_columns:
        col = sqlglot.exp.Column(this=_ident(name))
        if name in cast_map:
            sql_type = polars_dtype_to_sqlglot_type(cast_map[name], strict=True)
            projections.append(sqlglot.exp.Cast(this=col, to=sql_type).as_(name, quoted=quote))
        else:
            projections.append(col)

    inner = sqlglot.exp.Subquery(
        this=inner_query,
        alias=sqlglot.exp.TableAlias(this=sqlglot.exp.Identifier(this="__cpl_cast")),
    )
    cast_select = sqlglot.exp.Select().from_(inner, dialect=dialect).select(*projections, append=False, dialect=dialect)

    # Re-apply the clauses hoisted out of the (now nested) original query.
    if order_to_hoist is not None:
        cast_select.set("order", order_to_hoist)
    for opt in options_to_hoist:
        cast_select.args.setdefault("options", []).append(opt)

    return cast_select


def apply_polars_io_source_exprs(
    query: sqlglot.exp.Expression,
    dialect: str | type[Dialect] | None,
    with_columns: list[str] | None,
    predicate: pl.Expr | None,
    n_rows: int | None,
    batch_size: int | None,
    max_in_predicate_size: int = DEFAULT_MAX_IN_PREDICATE_SIZE,
) -> sqlglot.exp.Expression:
    """Apply Polars IO source expressions using subquery wrapping.

    Wraps the original query as a subquery and applies column selection,
    predicates, and row limits on the outer query.

    ``max_in_predicate_size`` bounds how large an ``is_in`` list may be before its
    pushdown into a SQL ``IN (...)`` clause is skipped (see
    ``DEFAULT_MAX_IN_PREDICATE_SIZE``).
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
        if _is_mssql_dialect(dialect):
            _stabilize_mssql_projection_aliases(inner)

        subquery = sqlglot.exp.Subquery(
            this=inner,
            alias=sqlglot.exp.TableAlias(this=sqlglot.exp.Identifier(this="__cpl_subq")),
        )
        outer = sqlglot.exp.Select().from_(subquery, dialect=dialect)

        # Column selection — flat names from the subquery output
        if with_columns is not None:
            quote_outer_identifiers = _is_known_dialect(dialect)
            outer = outer.select(
                *[
                    sqlglot.exp.Column(
                        this=sqlglot.exp.Identifier(this=name, quoted=True) if quote_outer_identifiers else name,
                    )
                    for name in with_columns
                ],
                append=False,
                dialect=dialect,
            )
        else:
            outer = outer.select(sqlglot.exp.Star(), append=False, dialect=dialect)

        # Predicate — alias names resolve to real columns in the subquery
        # output so no rewriting is needed
        if predicate is not None:
            sql_predicate = convert_predicate_to_sql(predicate, dialect, max_in_predicate_size)
            if sql_predicate is not None:
                if _is_known_dialect(dialect):
                    sql_predicate = sql_predicate.transform(_quote_column_leaf_identifier)
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
