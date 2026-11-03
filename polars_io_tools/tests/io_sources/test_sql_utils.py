"""Translating Polars filters to SQL (``convert_predicate_to_sql`` and ``SQLExpressionVisitor``), and the helpers in
``io_sources.sql_utils`` that rewrite a reader's query before it is sent.

Most tests are tables: filters (or parsed nodes, or Python values) and the SQL text each becomes in a dialect; queries
and how an MSSQL ORDER BY or OPTION hint moves out of a subquery; and filters with the rows their SQL WHERE clause
keeps, checked on DuckDB or SQLite.
"""

import sqlite3
from dataclasses import KW_ONLY, dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal

import duckdb
import numpy as np
import polars as pl
import pytest
from sqlglot import exp, parse_one
from sqlglot.dialects import ClickHouse
from sqlglot.dialects.dialect import Dialect

from polars_io_tools.io_sources.base import BaseExprNode, BinaryExprNode, ColumnNode, FunctionNode, LiteralNode, get_parsed_expr
from polars_io_tools.io_sources.enum import BooleanFunctionType, OperatorType, TemporalFunctionType
from polars_io_tools.io_sources.sql_dialects import MSSQL
from polars_io_tools.io_sources.sql_utils import (
    DEFAULT_MAX_IN_PREDICATE_SIZE,
    SQLExpressionVisitor,
    _prepare_inner_for_subquery,
    _stabilize_mssql_projection_aliases,
    apply_polars_io_source_exprs,
    convert_predicate_to_sql,
    create_sqlglot_literal,
)
from polars_io_tools.tests.helpers.assertions import assert_translated_text, sql_text
from polars_io_tools.tests.helpers.cases import as_params, epoch_ns


@pytest.fixture
def duckdb_connection():
    conn = duckdb.connect()
    yield conn
    conn.close()


def _visit(predicate: pl.Expr, dialect) -> exp.Expression | None:
    """Translate without the exception guard of ``convert_predicate_to_sql`` so failures surface."""
    visitor = SQLExpressionVisitor(dialect)
    visitor.visit(get_parsed_expr(predicate))
    return visitor.process_results()


_EASTERN_STANDARD_TIME = timezone(timedelta(hours=-5))


_CLICKHOUSE_JAN_15_2026_UTC = "toDateTime64('2026-01-15 00:00:00', 6, 'UTC')"


_IDS_OVER_IN_LIMIT = [f"ID{i}" for i in range(DEFAULT_MAX_IN_PREDICATE_SIZE + 1)]


_TS_DATE_AND_SYMBOL = (pl.col("ts").dt.date() == date(2026, 1, 15)) & pl.col("symbol").is_in(["A", "B"])


_REGEXES = ["^(CL|GC|SI|HO)", ".*abc", "foo|bar", "[A-Z]+", r"CL\d+"]


@dataclass(frozen=True)
class SqlTranslationCase:
    """A Polars filter, and the SQL text it translates to for a dialect."""

    id: str
    _: KW_ONLY
    predicate: pl.Expr
    dialect: str | type[Dialect] | None
    # what the SQL text must hold, from ``sql_text``; None when nothing may be translated
    expected_text: dict | None
    # render the SQL in sqlglot's default dialect instead of ``dialect``
    render_in_default_dialect: bool = False
    # skip the error handling of ``convert_predicate_to_sql``, so failures show
    call_visitor_directly: bool = False
    max_in_predicate_size: int = DEFAULT_MAX_IN_PREDICATE_SIZE


TRANSLATIONS = [
    SqlTranslationCase(
        "wrap_binary_expr", predicate=(pl.col("A") > 5) | pl.col("B").eq(10), dialect="tsql", expected_text=sql_text(equals="((A > 5) OR (B = 10))")
    ),
    # tz-aware datetimes on ClickHouse are explicit-UTC instants; elsewhere offset strings (issue #52)
    SqlTranslationCase(
        "tz_aware_datetime_clickhouse_uses_explicit_utc",
        predicate=(pl.col("ts") >= datetime(2026, 1, 15, tzinfo=UTC))
        & (pl.col("ts") < datetime(2026, 1, 15, 9, 0, 0, 123456, tzinfo=_EASTERN_STANDARD_TIME)),
        dialect="clickhouse",
        expected_text=sql_text(_CLICKHOUSE_JAN_15_2026_UTC, "toDateTime64('2026-01-15 14:00:00.123456', 6, 'UTC')", omits="+00:00"),
    ),
    SqlTranslationCase(
        "tz_aware_datetime_clickhouse_dialect_class-str",
        predicate=pl.col("ts") >= datetime(2026, 1, 15, tzinfo=UTC),
        dialect="clickhouse",
        expected_text=sql_text(_CLICKHOUSE_JAN_15_2026_UTC, omits="+00:00"),
    ),
    SqlTranslationCase(
        "tz_aware_datetime_clickhouse_dialect_class-class",
        predicate=pl.col("ts") >= datetime(2026, 1, 15, tzinfo=UTC),
        dialect=ClickHouse,
        expected_text=sql_text(_CLICKHOUSE_JAN_15_2026_UTC, omits="+00:00"),
    ),
    SqlTranslationCase(
        "tz_aware_datetime_is_in_clickhouse_uses_explicit_utc",
        predicate=pl.col("ts").is_in([datetime(2026, 1, 15, tzinfo=UTC), datetime(2026, 1, 16, tzinfo=UTC)]),
        dialect="clickhouse",
        expected_text=sql_text(_CLICKHOUSE_JAN_15_2026_UTC, "toDateTime64('2026-01-16 00:00:00', 6, 'UTC')", omits="+00:00"),
    ),
    *[
        SqlTranslationCase(
            f"tz_aware_datetime_non_clickhouse_unchanged-{d}",
            predicate=pl.col("ts") >= datetime(2026, 1, 15, tzinfo=UTC),
            dialect=d,
            expected_text=sql_text("2026-01-15 00:00:00+00:00"),
        )
        for d in ["tsql", "postgres", "duckdb"]
    ],
    SqlTranslationCase(
        "naive_datetime_clickhouse_uses_todatetime64_without_tz",
        predicate=pl.col("ts") >= datetime(2024, 1, 15, 12, 30),
        dialect="clickhouse",
        expected_text=sql_text("toDateTime64('2024-01-15 12:30:00', 6)", omits="'UTC'"),
    ),
    # the MSSQL class and no dialect behave like "tsql"
    *[
        SqlTranslationCase(
            f"convert_predicate_to_sql_with_mssql_class-{n}",
            predicate=pl.col("x") > 5,
            dialect=d,
            expected_text=sql_text(equals="(x > 5)"),
            render_in_default_dialect=True,
        )
        for n, d in [("tsql", "tsql"), ("MSSQL", MSSQL), ("None", None)]
    ],
    # dt.date() becomes a calendar-date expression per dialect (issue #51)
    *[
        SqlTranslationCase(
            f"dt_date_predicate_is_pushed_down-{d}", predicate=_TS_DATE_AND_SYMBOL, dialect=d, expected_text=sql_text(want, "symbol IN ('A', 'B')")
        )
        for d, want in [
            ("clickhouse", "CAST(ts AS Nullable(Date32)) = '2026-01-15'"),
            ("tsql", "CAST(ts AS DATE) = '2026-01-15'"),
            ("postgres", "CAST(ts AS DATE) = '2026-01-15'"),
            ("duckdb", "CAST(ts AS DATE) = '2026-01-15'"),
            # SQLite has no DATE type and Oracle DATE keeps the time, so both need calendar-date functions.
            ("sqlite", "DATE(ts) = '2026-01-15'"),
            ("oracle", "TRUNC(ts) = TO_DATE('2026-01-15', 'YYYY-MM-DD')"),
        ]
    ],
    SqlTranslationCase(
        "dt_date_pushdown_oracle_uses_trunc",
        predicate=pl.col("ts").dt.date() == date(2024, 5, 6),
        dialect="oracle",
        expected_text=sql_text("TRUNC(ts) = TO_DATE('2024-05-06', 'YYYY-MM-DD')", omits="CAST(ts AS DATE)"),
    ),
    SqlTranslationCase(
        "date_is_in_pushdown_oracle_uses_typed_literals",
        predicate=pl.col("day").is_in([date(2024, 5, 6), date(2024, 5, 7)]),
        dialect="oracle",
        expected_text=sql_text("day IN (TO_DATE('2024-05-06', 'YYYY-MM-DD'), TO_DATE('2024-05-07', 'YYYY-MM-DD'))"),
    ),
    # an is_in list over the size limit is not pushed; an AND keeps its other side
    SqlTranslationCase(
        "oversized_in_predicate_is_not_pushed_down-small",
        predicate=pl.col("id").is_in([f"ID{i}" for i in range(5)]),
        dialect="tsql",
        expected_text=sql_text("IN"),
    ),
    SqlTranslationCase(
        "oversized_in_predicate_is_not_pushed_down-big", predicate=pl.col("id").is_in(_IDS_OVER_IN_LIMIT), dialect="tsql", expected_text=None
    ),
    SqlTranslationCase(
        "oversized_in_predicate_is_not_pushed_down-and",
        predicate=pl.col("id").is_in(_IDS_OVER_IN_LIMIT) & (pl.col("year") == 2025),
        dialect="tsql",
        expected_text=sql_text("year", omits="IN"),
    ),
    SqlTranslationCase(
        "oversized_in_predicate_is_not_pushed_down-raised_limit",
        predicate=pl.col("id").is_in(_IDS_OVER_IN_LIMIT),
        dialect="tsql",
        expected_text=sql_text("IN"),
        max_in_predicate_size=len(_IDS_OVER_IN_LIMIT),
    ),
    # regex semantics differ between databases and Polars
    *[
        SqlTranslationCase(f"str_contains_regex_skips_pushdown-{d}-{r}", predicate=pl.col("item_name").str.contains(r), dialect=d, expected_text=None)
        for d in ["tsql", "postgres", "clickhouse", "duckdb"]
        for r in _REGEXES
    ],
    SqlTranslationCase(
        "str_contains_regex_with_and_partial_pushdown",
        predicate=pl.col("item_name").str.contains("^test") & (pl.col("year") > 2020),
        dialect="tsql",
        expected_text=sql_text("year", omits="LIKE"),
        render_in_default_dialect=True,
    ),
    *[
        SqlTranslationCase(
            f"predicate_without_exact_sql_translation_is_not_pushed_down-{p}", predicate=p, dialect=d, expected_text=None, call_visitor_directly=True
        )
        for p, d in [
            (pl.col("ts").dt.second() == 0, "postgres"),
            (pl.col("a").cast(pl.UInt8) > 3, "postgres"),
            (pl.col("a").cast(pl.Int128) > 3, "duckdb"),
            (pl.col("f").cast(pl.String) == "1.0", "duckdb"),
            (pl.col("s").str.starts_with("a\\"), "postgres"),
            (pl.col("s").str.contains("a_b", literal=True), "tsql"),
            (pl.col("a") // 2 == 1, "duckdb"),
        ]
    ],
    # Polars panics while exposing this literal; the panic must not escape conversion
    SqlTranslationCase(
        "nanosecond_numpy_literal_in_is_in_is_not_pushed_down",
        predicate=pl.col("ts").is_in([np.datetime64("2024-01-01T00:00:00.000000001", "ns")]),
        dialect="duckdb",
        expected_text=None,
    ),
    *[
        SqlTranslationCase(
            f"dt_year_renders_in_dialect-{d}",
            predicate=pl.col("ts").dt.year() == 2024,
            dialect=d,
            expected_text=sql_text("2024"),
            call_visitor_directly=True,
        )
        for d in ["tsql", "snowflake", "duckdb"]
    ],
    SqlTranslationCase(
        "head_with_non_sql_filter_returns_correct_count-udf",
        predicate=pl.col("score").map_elements(lambda v: v > 80, return_dtype=pl.Boolean),
        dialect="tsql",
        expected_text=None,
    ),
]


@pytest.mark.parametrize("case", as_params(TRANSLATIONS))
def test_translation(case: SqlTranslationCase):
    if case.call_visitor_directly:
        result = _visit(case.predicate, case.dialect)
    else:
        result = convert_predicate_to_sql(case.predicate, case.dialect, max_in_predicate_size=case.max_in_predicate_size)
    render_dialect = None if case.render_in_default_dialect else case.dialect
    if case.expected_text is None:
        assert result is None, result.sql(render_dialect)
    else:
        assert_translated_text(None if result is None else result.sql(render_dialect), **case.expected_text)


def _comparison_node(name, op, value):
    return BinaryExprNode(
        expr=pl.col(name) > value, left=ColumnNode(expr=pl.col(name), name=name), op=op, right=LiteralNode(expr=pl.lit(value), value=value)
    )


def _logical_node(op):
    return BinaryExprNode(expr=pl.lit(True), left=_comparison_node("A", OperatorType.GT, 5), op=op, right=_comparison_node("B", OperatorType.GT, 10))


def _start_to_end_date_node(function_type):
    nodes = [get_parsed_expr(pl.col("StartDate")), get_parsed_expr(pl.col("EndDate"))]
    return FunctionNode(expr=pl.col("StartDate"), inputs=nodes, function_type=function_type)


_MY_COLUMN_NODE = ColumnNode(expr=pl.col("my_column"), name="my_column")


_DATEDIFF_DAYS = "DATEDIFF(DAY, StartDate, EndDate)"


@dataclass(frozen=True)
class VisitNodeCase:
    """A hand-built parsed node, and the sqlglot type and SQL text the visitor makes of it."""

    id: str
    _: KW_ONLY
    node: BaseExprNode
    # None: the type is not checked
    expected_type: type[exp.Expression] | None
    # what the SQL text must hold, from ``sql_text``
    expected_text: dict
    dialect: str | type[Dialect] | None


NODES = [
    # LOGICAL_OR / LOGICAL_AND come from Polars' optimizer; unmapped they fell back to client-side filtering
    VisitNodeCase(
        "logical_or_operator",
        node=_logical_node(OperatorType.LOGICAL_OR),
        expected_type=None,
        expected_text=sql_text("OR", "A > 5", "B > 10"),
        dialect="tsql",
    ),
    VisitNodeCase(
        "logical_and_operator",
        node=_logical_node(OperatorType.LOGICAL_AND),
        expected_type=None,
        expected_text=sql_text("AND", "A > 5", "B > 10"),
        dialect="tsql",
    ),
    VisitNodeCase("visit_column", node=_MY_COLUMN_NODE, expected_type=exp.Column, expected_text=sql_text(equals="my_column"), dialect="tsql"),
    VisitNodeCase(
        "visit_literal-number",
        node=LiteralNode(expr=pl.lit(42), value=42),
        expected_type=exp.Literal,
        expected_text=sql_text(equals="42"),
        dialect="tsql",
    ),
    VisitNodeCase(
        "visit_literal-null",
        node=LiteralNode(expr=pl.lit(None), value=None),
        expected_type=exp.Null,
        expected_text=sql_text(equals="NULL"),
        dialect="tsql",
    ),
    VisitNodeCase(
        "visit_function_is_null",
        node=FunctionNode(expr=pl.col("my_column").is_null(), inputs=[_MY_COLUMN_NODE], function_type=BooleanFunctionType.IS_NULL),
        expected_type=exp.Is,
        expected_text=sql_text(equals="my_column IS NULL"),
        dialect="tsql",
    ),
    VisitNodeCase(
        "total_days_to_datediff-days",
        node=_start_to_end_date_node(TemporalFunctionType.TOTAL_DAYS),
        expected_type=exp.DateDiff,
        expected_text=sql_text(equals=_DATEDIFF_DAYS),
        dialect="tsql",
    ),
    VisitNodeCase(
        "total_days_to_datediff-hours",
        node=_start_to_end_date_node(TemporalFunctionType.TOTAL_HOURS),
        expected_type=exp.DateDiff,
        expected_text=sql_text(equals="DATEDIFF(HOUR, StartDate, EndDate)"),
        dialect="tsql",
    ),
    # the MSSQL class and None normalize to "tsql", so the TSQL-only DATEDIFF still applies
    *[
        VisitNodeCase(
            f"visitor_dialect_normalization-{n}",
            node=_start_to_end_date_node(TemporalFunctionType.TOTAL_DAYS),
            expected_type=exp.DateDiff,
            expected_text=sql_text(equals=_DATEDIFF_DAYS),
            dialect=d,
        )
        for n, d in [("tsql", "tsql"), ("MSSQL", MSSQL), ("None", None)]
    ],
]


@pytest.mark.parametrize("case", as_params(NODES))
def test_visit_node(case: VisitNodeCase):
    visitor = SQLExpressionVisitor(dialect=case.dialect)
    visitor.visit(case.node)
    result = visitor.process_results()
    assert case.expected_type is None or isinstance(result, case.expected_type), type(result)
    assert_translated_text(result.sql(dialect="tsql"), **case.expected_text)


@dataclass(frozen=True)
class SqlLiteralCase:
    """A Python value, and the sqlglot type and SQL it becomes."""

    id: str
    _: KW_ONLY
    value: object
    expected_type: type[exp.Expression]
    expected_sql: str


LITERALS = [
    SqlLiteralCase("create_sqlglot_literal_various_types-none", value=None, expected_type=exp.Null, expected_sql="NULL"),
    SqlLiteralCase("create_sqlglot_literal_various_types-int", value=123, expected_type=exp.Literal, expected_sql="123"),
    SqlLiteralCase("create_sqlglot_literal_various_types-float", value=12.34, expected_type=exp.Literal, expected_sql="12.34"),
    SqlLiteralCase("create_sqlglot_literal_various_types-str", value="abc", expected_type=exp.Literal, expected_sql="'abc'"),
    SqlLiteralCase("create_sqlglot_literal_various_types-date", value=date(2025, 1, 2), expected_type=exp.Literal, expected_sql="'2025-01-02'"),
    # booleans are SQL booleans, not the strings 'True'/'False'
    SqlLiteralCase("create_sqlglot_literal_various_types-true", value=True, expected_type=exp.Boolean, expected_sql="TRUE"),
    SqlLiteralCase("create_sqlglot_literal_various_types-false", value=False, expected_type=exp.Boolean, expected_sql="FALSE"),
    SqlLiteralCase(
        "naive_datetime_literal_unchanged", value=datetime(2026, 1, 15, 12, 30), expected_type=exp.Literal, expected_sql="'2026-01-15 12:30:00'"
    ),
]


@pytest.mark.parametrize("case", as_params(LITERALS))
def test_create_sqlglot_literal(case: SqlLiteralCase):
    result = create_sqlglot_literal(case.value)
    assert isinstance(result, case.expected_type)
    assert result.sql() == case.expected_sql


def test_create_sqlglot_literal_bool_via_duckdb(duckdb_connection):
    """A quoted 'True' would fail in DuckDB with a type mismatch against a boolean column."""
    duckdb_connection.execute("CREATE TEMPORARY TABLE IF NOT EXISTS bool_test (flag BOOLEAN, val INTEGER)")
    duckdb_connection.execute("INSERT INTO bool_test VALUES (TRUE, 1), (FALSE, 2), (TRUE, 3)")
    for value, want in [(True, [1, 3]), (False, [2])]:
        literal = create_sqlglot_literal(value).sql(dialect="duckdb")
        assert literal == str(value).upper()
        assert sorted(r[0] for r in duckdb_connection.execute(f"SELECT val FROM bool_test WHERE flag = {literal}").fetchall()) == want


def test_clickhouse_datetime_literal_returns_none_on_overflow():
    """A datetime beyond Python's year range has no representable form, so pushdown is skipped, not truncated."""
    from polars_io_tools.io_sources.sql_utils import _clickhouse_datetime_literal

    assert _clickhouse_datetime_literal(pl.lit(pl.Series([300_000_000_000_000], dtype=pl.Datetime("ms")))) is None


def test_visit_literal_skips_pushdown_when_precision_unrecoverable(monkeypatch):
    """Real out-of-range datetimes raise earlier, so patch the renderer to reach the guard."""
    import polars_io_tools.io_sources.sql_utils as su

    monkeypatch.setattr(su, "_clickhouse_datetime_literal", lambda expr: None)
    assert convert_predicate_to_sql(pl.col("ts") >= datetime(2026, 1, 15, tzinfo=UTC), "clickhouse") is None


def test_nanosecond_datetime_pushdown_clickhouse_preserves_precision():
    """Equality on a ``Datetime(ns)`` column reaches the source as an ``is_between`` whose bounds share a microsecond.

    Both bounds must survive as distinct 9-digit ``toDateTime64`` values, or ClickHouse drops matching rows.
    """
    from polars.io.plugins import register_io_source

    captured: dict[str, pl.Expr] = {}

    def source(with_columns, predicate, n_rows, batch_size):
        captured["predicate"] = predicate
        yield pl.DataFrame({"ts": pl.Series("ts", [epoch_ns(datetime(2024, 1, 15)) + 123_456_789], dtype=pl.Int64).cast(pl.Datetime("ns", "UTC"))})

    register_io_source(source, schema={"ts": pl.Datetime("ns", "UTC")}).filter(
        pl.col("ts") == datetime(2024, 1, 15, 0, 0, 0, 123456, tzinfo=UTC)
    ).collect()
    assert_translated_text(
        convert_predicate_to_sql(captured["predicate"], "clickhouse").sql(dialect="clickhouse"),
        ["toDateTime64('2024-01-15 00:00:00.123456000', 9, 'UTC')", "toDateTime64('2024-01-15 00:00:00.123456999', 9, 'UTC')"],
    )


@dataclass(frozen=True)
class SubqueryHoistingCase:
    """A query, and what ``_prepare_inner_for_subquery`` moves from it to the outer query.

    MSSQL rejects ORDER BY in a derived table without TOP/OFFSET (error 1033), so it and OPTION hints move to the outer
    query.
    """

    id: str
    _: KW_ONLY
    query: str
    dialect: str | type[Dialect] | None
    inner_keeps_order_by: bool
    # what the ORDER BY moved out must hold, from ``sql_text``; None when none is moved
    moved_order_by: dict | None
    # the text of each OPTION hint moved out
    moved_options: list[str]


HOISTING = [
    SubqueryHoistingCase(
        "mssql_order_by_without_top_is_moved_to_outer",
        query="SELECT region, amount FROM Sales ORDER BY amount DESC",
        inner_keeps_order_by=False,
        moved_order_by=sql_text("amount"),
        moved_options=[],
        dialect=MSSQL,
    ),
    SubqueryHoistingCase(
        "mssql_order_by_with_limit_stays_in_inner",
        query="SELECT TOP 2 region, amount FROM Sales ORDER BY amount DESC",
        inner_keeps_order_by=True,
        moved_order_by=None,
        moved_options=[],
        dialect=MSSQL,
    ),
    # the table alias only exists inside the subquery
    SubqueryHoistingCase(
        "mssql_order_by_hoisted_with_table_qualifiers_stripped",
        query="SELECT cb.region, cb.amount FROM Sales cb ORDER BY cb.amount DESC",
        inner_keeps_order_by=False,
        moved_order_by=sql_text("amount", omits="cb."),
        moved_options=[],
        dialect=MSSQL,
    ),
    SubqueryHoistingCase(
        "mssql_order_by_with_offset_stays_in_inner",
        query="SELECT region, amount FROM Sales ORDER BY amount OFFSET 1 ROWS FETCH NEXT 2 ROWS ONLY",
        inner_keeps_order_by=True,
        moved_order_by=None,
        moved_options=[],
        dialect=MSSQL,
    ),
    SubqueryHoistingCase(
        "mssql_option_hints_hoisted",
        query="SELECT region FROM Sales OPTION (HASH JOIN)",
        inner_keeps_order_by=False,
        moved_order_by=None,
        moved_options=["HASH JOIN"],
        dialect=MSSQL,
    ),
    SubqueryHoistingCase(
        "non_mssql_no_hoisting",
        query="SELECT region, amount FROM Sales ORDER BY amount DESC",
        inner_keeps_order_by=True,
        moved_order_by=None,
        moved_options=[],
        dialect="duckdb",
    ),
]


@pytest.mark.parametrize("case", as_params(HOISTING))
def test_prepare_inner_for_subquery(case: SubqueryHoistingCase):
    inner, order, hoisted_options = _prepare_inner_for_subquery(parse_one(case.query, dialect=MSSQL), dialect=case.dialect)
    assert (inner.args.get("order") is not None) == case.inner_keeps_order_by
    if case.moved_order_by is None:
        assert order is None
    else:
        assert_translated_text(order.sql(), ignore_case=True, **case.moved_order_by)
    assert len(hoisted_options) == len(case.moved_options)
    for got, want in zip(hoisted_options, case.moved_options):
        assert_translated_text(got.sql(dialect=MSSQL), want, ignore_case=True)
    assert not (case.moved_options and list(inner.find_all(exp.QueryOption)))


# Filters whose SQL WHERE clause must run and return every row ``df.filter`` keeps (Polars filters again afterwards, so
# extra rows are fine).
_MIXED_VALUES_WITH_NULLS = pl.DataFrame(
    {
        "id": range(7),
        "a": [3, None, -7, 4, 1, 2, 5],
        "b": [30, 1, 200, 0, 5, None, 150],
        "s": ["abc", "\tabc", "a\\b", "x_b", "xab", None, "a%c"],
        "f": [2.7, 1.0, None, -0.5, 2.0, 1.5, 3.2],
        "ts": [
            datetime(2024, 1, 1, 0, 0, 0, 500),
            datetime(2023, 6, 1),
            None,
            datetime(2024, 3, 1),
            datetime(2024, 1, 1),
            None,
            datetime(2022, 1, 1),
        ],
    }
)


# A condition with no SQL translation.
_S_HAS_THREE_CHARACTERS = pl.col("s").str.len_chars() == 3


@dataclass(frozen=True)
class WhereKeepsRowsCase:
    """A filter, data, and a dialect whose SQL WHERE clause must keep every row the filter keeps."""

    id: str
    _: KW_ONLY
    predicate: pl.Expr
    data: pl.DataFrame
    dialect: str


KEEPS = [
    WhereKeepsRowsCase(f"pushed_predicate_keeps_all_matching_rows-{p}", predicate=p, data=_MIXED_VALUES_WITH_NULLS, dialect="duckdb")
    for p in [
        ~pl.when(pl.col("a") > 2).then(pl.col("b") > 100).otherwise(pl.col("b") > 0),
        pl.col("a").is_between(2, 4).is_null(),
        pl.col("a").is_between(2, 4) == False,
        ((pl.col("a") > 1) & _S_HAS_THREE_CHARACTERS) == False,
        ((pl.col("a") > 1) & _S_HAS_THREE_CHARACTERS).is_null(),
        ~((pl.col("a") > 1) & _S_HAS_THREE_CHARACTERS),
        pl.when(pl.col("a") > 2).then(pl.col("b") > 100).otherwise(_S_HAS_THREE_CHARACTERS),
        pl.when(pl.col("a") > 2).then(1).otherwise(0) == 0,
        pl.col("a").ne_missing(2),
        ~pl.col("a").eq_missing(2),
        pl.col("a").is_in([1, None], nulls_equal=True),
        ~pl.col("a").is_in([1, None]),
        pl.col("a") % 2 == 1,
        pl.col("f").cast(pl.Int64) == 2,
        (pl.col("a") & 3) == 0,
        pl.col("s").str.strip_chars() == "abc",
        pl.col("s").str.strip_chars("x") == "ab",
        pl.concat_str([pl.col("s"), pl.col("s")], separator="-") == "abc-abc",
        ~pl.col("s").str.ends_with("_b"),
        pl.col("ts").dt.year() == 2024,
        pl.col("ts").cast(pl.Datetime("ms")) == datetime(2024, 1, 1),
        pl.col("f") < float("inf"),
        pl.col("f").is_in([float("nan")]),
        pl.col("ts") < pl.col("ts") + timedelta(days=1),
    ]
] + [
    # where Polars and SQL semantics differ
    WhereKeepsRowsCase(f"pushed_predicate_keeps_polars_semantics-{p}", predicate=p, data=pl.DataFrame(data), dialect=d)
    for data, p, d in [
        ({"x": [2]}, pl.col("x").cast(pl.Boolean) == True, "duckdb"),
        ({"x": [1]}, (~pl.col("x")).cast(pl.Boolean), "duckdb"),
        ({"x": [1]}, (~pl.col("x")) == -2, "duckdb"),
        ({"x": [2**63 - 1]}, (pl.col("x") + 1) < 0, "duckdb"),
        ({"x": [b"ab"]}, pl.col("x") == b"ab", "duckdb"),
        ({"x": [b"ab"]}, pl.col("x").is_in([b"ab"]), "duckdb"),
        ({"x": ["straße"]}, pl.col("x").str.to_uppercase() == "STRASSE", "duckdb"),
        ({"x": ["A"]}, ~pl.col("x").str.starts_with("a"), "sqlite"),
        ({"x": ["A"]}, ~pl.col("x").str.contains("a", literal=True), "sqlite"),
        ({"x": ["\u0663"]}, pl.col("x").str.contains(r"^\d$"), "duckdb"),
        ({"x": ["a"]}, pl.col("x").str.contains(r"\u{61}"), "duckdb"),
        ({"x": ["bad"]}, pl.col("x").cast(pl.Float64, strict=False).is_null(), "duckdb"),
        ({"x": [0]}, pl.col("x").cast(pl.Datetime("us")).dt.year() == 1970, "duckdb"),
        ({"x": pl.Series([datetime(2024, 1, 1, 23)]).dt.replace_time_zone("UTC")}, pl.col("x").cast(pl.Datetime("us")).dt.hour() == 23, "duckdb"),
        ({"x": [1], "ys": [[1]]}, pl.col("x").is_in(pl.col("ys")), "duckdb"),
        ({"x": ["a\x00b"]}, pl.col("x") == "a\x00b", "duckdb"),
        ({"x": ["a", "b"]}, pl.col("x") == pl.lit(pl.Series(["a", "b"])), "duckdb"),
        ({"x": [1.25]}, pl.col("x").cast(pl.Decimal(4, 1)) == Decimal("1.2"), "duckdb"),
        ({"x": pl.Series([0.1], dtype=pl.Float32)}, ~pl.col("x").is_in([0.1, 0.2]), "duckdb"),
    ]
]


def _where_rows(df: pl.DataFrame, where: str, dialect: str) -> set[int]:
    """Row numbers of ``df`` that ``SELECT ... WHERE where`` returns."""
    sql = f"SELECT __row FROM t WHERE {where}"
    if dialect == "sqlite":
        con = sqlite3.connect(":memory:")
        con.execute(f"CREATE TABLE t ({', '.join(df.columns)})")
        con.executemany(f"INSERT INTO t VALUES ({', '.join('?' * df.width)})", df.rows())
        return {r[0] for r in con.execute(sql).fetchall()}
    con = duckdb.connect()
    con.execute("SET TimeZone = 'America/New_York'")
    con.register("t", df.to_arrow())
    return {r[0] for r in con.sql(sql).fetchall()}


@pytest.mark.parametrize("case", as_params(KEEPS))
def test_where_keeps_matching_rows(case: WhereKeepsRowsCase):
    where = _visit(case.predicate, case.dialect)
    if where is None:
        return
    df = case.data.with_row_index("__row")
    kept = set(df.filter(case.predicate)["__row"])
    assert kept <= _where_rows(df, where.sql(case.dialect), case.dialect), where.sql(case.dialect)


@dataclass(frozen=True)
class SqliteTextTimestampCase:
    """A filter on a SQLite text timestamp column, and the rows its WHERE clause returns."""

    id: str
    _: KW_ONLY
    pred: pl.Expr
    want: list[tuple]
    # the rows are exactly ``want``, not only a superset
    exact: bool


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            # CAST(... AS DATE) coerces to an integer in SQLite and would drop the non-midnight row
            SqliteTextTimestampCase(
                "dt_date_pushdown_sqlite_selects_non_midnight_row",
                pred=pl.col("ts").dt.date() == date(2024, 5, 6),
                want=[("2024-05-06 12:34:56", 1)],
                exact=True,
            ),
            SqliteTextTimestampCase(
                "dt_year_pushdown_sqlite_runs",
                pred=(pl.col("ts").dt.year() == 2024) & (pl.col("x") > 0),
                want=[("2024-05-06 12:34:56", 1)],
                exact=False,
            ),
        ]
    ),
)
def test_sqlite_where_on_text_timestamps(case: SqliteTextTimestampCase):
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE t (ts TEXT, x INTEGER)")
    con.execute("INSERT INTO t VALUES ('2024-05-06 12:34:56', 1), ('2024-05-07 00:00:00', 2), ('2023-05-07 00:00:00', 3)")
    rows = con.execute(f"SELECT ts, x FROM t WHERE {convert_predicate_to_sql(case.pred, 'sqlite').sql(dialect='sqlite')}").fetchall()
    assert rows == case.want if case.exact else set(case.want) <= set(rows)


# apply_polars_io_source_exprs: identifier quoting and MSSQL projection-alias stabilization on the parsed query
@dataclass(frozen=True)
class DialectQuotingCase:
    """A dialect and a column name from source metadata, and the quoted name in the SQL."""

    id: str
    _: KW_ONLY
    dialect: str
    effective_name: str
    expected_sql: str


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            DialectQuotingCase('postgres-group-"group"', dialect="postgres", effective_name="group", expected_sql='"group"'),
            DialectQuotingCase('oracle-GROUP-"GROUP"', dialect="oracle", effective_name="GROUP", expected_sql='"GROUP"'),
            DialectQuotingCase('snowflake-GROUP-"GROUP"', dialect="snowflake", effective_name="GROUP", expected_sql='"GROUP"'),
        ]
    ),
)
def test_outer_metadata_columns_use_dialect_quoting(case: DialectQuotingCase):
    """Known dialects quote the effective names supplied by source metadata."""
    query = parse_one("SELECT source_name FROM source_table", dialect=case.dialect)

    result = apply_polars_io_source_exprs(
        query,
        dialect=case.dialect,
        with_columns=[case.effective_name],
        predicate=None,
        n_rows=None,
        batch_size=None,
    )

    assert result.selects[0].sql(dialect=case.dialect) == case.expected_sql


def test_outer_identifiers_preserve_generic_dialect_behavior():
    """A missing dialect leaves metadata and predicate identifiers unquoted."""
    result = apply_polars_io_source_exprs(
        parse_one("SELECT source_name FROM source_table"),
        dialect=None,
        with_columns=["CamelCase"],
        predicate=pl.col("CamelCase") > 1,
        n_rows=None,
        batch_size=None,
    )

    selected_identifier = result.selects[0].this
    predicate_column = next(result.args["where"].find_all(exp.Column))
    assert selected_identifier == "CamelCase"
    assert predicate_column.this == "CamelCase"


def test_predicate_quoting_is_limited_to_outer_column_leaves():
    """Predicate quoting does not rewrite source SQL or non-column AST nodes."""
    query = parse_one(
        """
        SELECT src.value AS result, LOWER(src.label) AS normalized
        FROM catalog.schema.source_table AS src
        WHERE EXISTS (
            SELECT 1
            FROM related_table AS rel
            WHERE rel.source_id = src.id
        )
        ORDER BY src.value
        """,
        dialect="postgres",
    )

    result = apply_polars_io_source_exprs(
        query,
        dialect="postgres",
        with_columns=None,
        predicate=(pl.col("result") > 1) & (pl.col("normalized") == "literal"),
        n_rows=None,
        batch_size=None,
    )

    outer_columns = list(result.args["where"].find_all(exp.Column))
    assert [column.name for column in outer_columns] == ["result", "normalized"]
    assert all(column.this.quoted for column in outer_columns)

    inner = result.args["from_"].this.this
    column_identifiers = [column.this for column in inner.find_all(exp.Column)]
    tables = list(inner.find_all(exp.Table))
    qualified_table_parts = [table.parts for table in tables]
    projection_aliases = [projection.args["alias"] for projection in inner.selects]
    table_aliases = [table.args["alias"].this for table in tables]

    assert [identifier.name for identifier in column_identifiers] == ["value", "label", "value", "source_id", "id"]
    assert [[identifier.name for identifier in parts] for parts in qualified_table_parts] == [
        ["catalog", "schema", "source_table"],
        ["related_table"],
    ]
    assert [identifier.name for identifier in projection_aliases] == ["result", "normalized"]
    assert [identifier.name for identifier in table_aliases] == ["src", "rel"]

    assert all(not identifier.quoted for identifier in column_identifiers)
    assert all(not identifier.quoted for parts in qualified_table_parts for identifier in parts)
    assert all(not identifier.quoted for identifier in projection_aliases)
    assert all(not identifier.quoted for identifier in table_aliases)
    assert isinstance(next(inner.find_all(exp.Lower)), exp.Lower)
    assert [(literal.this, literal.is_string) for literal in result.args["where"].find_all(exp.Literal)] == [
        ("1", False),
        ("literal", True),
    ]
    assert inner.args["order"] is not None


def test_reordered_full_schema_selection_preserves_shortcut():
    """A reordered redundant selection still returns the original query shape."""
    query = parse_one("SELECT first_name, second_name FROM source_table", dialect="postgres")

    result = apply_polars_io_source_exprs(
        query,
        dialect="postgres",
        with_columns=["second_name", "first_name"],
        predicate=None,
        n_rows=None,
        batch_size=None,
    )

    assert result.sql(dialect="postgres") == query.sql(dialect="postgres")
    assert [column.name for column in result.selects] == ["first_name", "second_name"]


def test_mssql_direct_select_preserves_projection_order_and_stabilizes_supported_aliases():
    query = parse_one(
        "SELECT first_col, *, second_col, t.*, COUNT(*) AS total, COUNT(*), third_col FROM source_table AS t",
        dialect=MSSQL,
    )

    result = apply_polars_io_source_exprs(
        query,
        dialect=MSSQL,
        with_columns=None,
        predicate=pl.col("first_col") > 0,
        n_rows=None,
        batch_size=None,
    )

    inner = result.args["from_"].this.this
    assert [projection.sql(dialect=MSSQL) for projection in inner.selects] == [
        "[first_col] AS [first_col]",
        "*",
        "[second_col] AS [second_col]",
        "[t].*",
        "COUNT(*) AS [total]",
        "COUNT(*)",
        "[third_col] AS [third_col]",
    ]


def test_mssql_qualified_mixed_case_column_is_quoted_in_outer_pushdowns():
    query = parse_one("SELECT src.DataDate FROM source_table AS src", dialect=MSSQL)

    result = apply_polars_io_source_exprs(
        query,
        dialect=MSSQL,
        with_columns=["DataDate"],
        predicate=pl.col("DataDate") > 1,
        n_rows=None,
        batch_size=None,
    )

    inner = result.args["from_"].this.this
    assert isinstance(inner, exp.Select)
    assert len(inner.selects) == 1
    inner_projection = inner.selects[0]
    assert isinstance(inner_projection, exp.Alias)
    assert inner_projection.sql(dialect=MSSQL) == "[src].[DataDate] AS [DataDate]"
    assert inner_projection.args["alias"].name == "DataDate"
    assert inner_projection.args["alias"].quoted

    assert len(result.selects) == 1
    outer_projection = result.selects[0]
    assert isinstance(outer_projection, exp.Column)
    assert outer_projection.name == "DataDate"
    assert outer_projection.this.quoted

    predicate_columns = list(result.args["where"].find_all(exp.Column))
    assert [column.name for column in predicate_columns] == ["DataDate"]
    assert predicate_columns[0].this.quoted


def test_mssql_projection_alias_stabilization_is_idempotent():
    query = parse_one("SELECT first_col, COUNT(*) AS total, COUNT(*) FROM source_table", dialect=MSSQL)

    _stabilize_mssql_projection_aliases(query)
    first_sql = query.sql(dialect=MSSQL)
    _stabilize_mssql_projection_aliases(query)

    assert query.sql(dialect=MSSQL) == first_sql
    assert [projection.sql(dialect=MSSQL) for projection in query.selects] == [
        "[first_col] AS [first_col]",
        "COUNT(*) AS [total]",
        "COUNT(*)",
    ]


def test_mssql_cte_projection_stabilization_does_not_rewrite_cte_body():
    root = parse_one(
        "WITH cte AS (SELECT DataDate FROM source_table) SELECT DataDate FROM cte",
        dialect=MSSQL,
    )

    assert isinstance(root, exp.Select)
    _stabilize_mssql_projection_aliases(root)

    assert len(root.selects) == 1
    output_projection = root.selects[0]
    assert isinstance(output_projection, exp.Alias)
    assert output_projection.this.name == "DataDate"
    assert output_projection.args["alias"].name == "DataDate"
    assert output_projection.args["alias"].quoted

    cte_body = root.args["with_"].expressions[0].this
    assert isinstance(cte_body, exp.Select)
    assert len(cte_body.selects) == 1
    cte_projection = cte_body.selects[0]
    assert isinstance(cte_projection, exp.Column)
    assert cte_projection.name == "DataDate"
    assert not cte_projection.this.quoted


def test_mssql_parenthesized_set_operations_only_stabilize_leftmost_output_branch():
    query = parse_one(
        """
        (((SELECT a, COUNT(*) AS left_count FROM table_a)
          INTERSECT
          (SELECT c, COUNT(*) AS intersect_count FROM table_c))
         UNION
         (SELECT e, COUNT(*) AS union_count FROM table_e))
        EXCEPT
        SELECT g, COUNT(*) AS except_count FROM table_g
        """,
        dialect=MSSQL,
    )

    result = apply_polars_io_source_exprs(
        query,
        dialect=MSSQL,
        with_columns=["a"],
        predicate=None,
        n_rows=None,
        batch_size=None,
    )

    except_root = result.args["from_"].this.this
    union_root = except_root.this.this
    intersect_root = union_root.this.this
    leftmost_select = intersect_root.this.this
    intersect_right = intersect_root.expression.this
    union_right = union_root.expression.this
    except_right = except_root.expression

    assert [projection.sql(dialect=MSSQL) for projection in leftmost_select.selects] == [
        "[a] AS [a]",
        "COUNT(*) AS [left_count]",
    ]
    assert [projection.sql(dialect=MSSQL) for projection in intersect_right.selects] == [
        "[c]",
        "COUNT(*) AS intersect_count",
    ]
    assert [projection.sql(dialect=MSSQL) for projection in union_right.selects] == [
        "[e]",
        "COUNT(*) AS union_count",
    ]
    assert [projection.sql(dialect=MSSQL) for projection in except_right.selects] == [
        "[g]",
        "COUNT(*) AS except_count",
    ]


def test_cast_map_hoists_mssql_order_by_out_of_derived_table():
    """A top-level MSSQL ORDER BY must end up after the cast derived table, not inside it."""
    from polars_io_tools.io_sources.sql_utils import wrap_query_with_casts

    wrapped = wrap_query_with_casts(parse_one("SELECT * FROM dbo.t ORDER BY x", dialect=MSSQL), MSSQL, ["x", "y"], {"x": pl.Date})
    sql = apply_polars_io_source_exprs(wrapped.copy(), MSSQL, None, pl.col("x") == date(2025, 5, 30), None, None).sql(dialect=MSSQL)
    assert_translated_text(sql, ["AS __cpl_cast) AS __cpl_subq", "ORDER BY [x]"], omits="ORDER BY [x]) AS __cpl_cast")
    assert sql.rstrip().endswith("ORDER BY [x]")


def test_cast_map_preserves_decimal_scale():
    from polars_io_tools.io_sources.sql_utils import polars_dtype_to_sqlglot_type

    assert polars_dtype_to_sqlglot_type(pl.Decimal(10, 2)).sql(dialect=MSSQL) == "NUMERIC(10, 2)"
