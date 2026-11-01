"""scan_db and its SQL translation, against an in-memory DuckDB behind a fake ``arrow_odbc`` (see ``helpers.sql_backends``).

Most tests are tables: a predicate or node and the SQL text it becomes per dialect, a predicate and the rows its WHERE
clause keeps, and a reader query whose result must equal what the database returns for the same query, filtered and
selected by Polars. The query table also runs ``scan_clickhouse`` where the two readers share a test.
"""

import concurrent.futures
import logging
import sqlite3
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import duckdb
import numpy as np
import polars as pl
import pyarrow as pa
import pytest
from polars.testing import assert_frame_equal
from sqlglot import exp, parse_one
from sqlglot.dialects import ClickHouse

import polars_io_tools as cpl
from polars_io_tools.io_sources import lazy_sql_reader
from polars_io_tools.io_sources.base import BinaryExprNode, ColumnNode, FunctionNode, LiteralNode, get_parsed_expr
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
from polars_io_tools.tests.helpers.assertions import assert_matches_native, assert_text
from polars_io_tools.tests.helpers.cases import Case, params
from polars_io_tools.tests.helpers.sql_backends import ENTRY_POINTS, Backend, cc_bond_df, connect, employee_df, install

ELECTION_CSV = (
    "Country,ElectFor,EventDate,Status,KnowledgeDate,SourceFile,Isbackfill,DBTimestamp\n"
    "Chile,Referendum,2022-09-04T00:00:00.000,Held,2022-09-13T16:51:26.000,\\\\cubistdata\\cds\\Prod\\ElectionCalendar\\20220913\\20220913_1651Past.csv,0,2022-09-13T17:03:07.000\n"
    "Nicaragua,Nicaraguan Presidency,2021-11-07T00:00:00.000,Held,2023-04-18T16:38:30.000,\\\\cubistdata\\cds\\Prod\\ElectionCalendar\\20230418\\20230418_1638Past.csv,0,2023-04-18T16:49:12.000\n"
    "Jordan,Jordanian House of Deputies,2010-11-09T00:00:00.000,Snap,2017-04-17T02:32:24.000,E:\\SourceData\\ElectionCalendar\\20170417\\20170417_02.csv,1,2010-10-10T00:00:00.000\n"
    "Kiribati,Kiribati Presidency,2024-06-30T00:00:00.000,Date not confirmed,2024-04-01T09:34:51.000,\\\\cubistdata\\cds\\Prod\\ElectionCalendar\\20240401\\20240401_0934Upcoming.csv,0,2024-04-01T09:46:00.000\n"
    "Liberia,President,2017-10-10T00:00:00.000,Held,2018-04-03T00:07:41.000,E:\\SourceData\\ElectionCalendar\\20180403\\20180403_00.csv,1,2018-04-03T00:07:44.000\n"
    "Honduras,President,2017-11-26T00:00:00.000,Confirmed,2017-04-17T02:32:24.000,E:\\SourceData\\ElectionCalendar\\20170417\\20170417_02.csv,1,2017-04-17T02:32:27.000\n"
    "Moldova,Moldovan Parliament,1998-03-22T00:00:00.000,Held,2017-04-17T02:32:25.000,E:\\SourceData\\ElectionCalendar\\20170417\\20170417_02.csv,1,1998-02-20T00:00:00.000\n"
    "Singapore,Singapore Parliament,2020-07-10T00:00:00.000,Confirmed,2020-07-09T17:08:37.000,//sacrshfs03/SourceData/ElectionCalendar\\20200709\\20200709_17.csv,0,2020-07-09T17:10:22.000\n"
    "India,Indian People's Assembly,2024-04-19T00:00:00.000,Held,2024-06-17T16:48:06.000,\\\\cubistdata\\cds\\Prod\\ElectionCalendar\\20240617\\20240617_1648Past.csv,0,2024-06-17T16:59:07.000\n"
    "Guatemala,Guatemalan Presidency,2023-06-25T00:00:00.000,Held,2023-07-11T10:43:16.000,\\\\cubistdata\\cds\\Prod\\ElectionCalendar\\20230711\\20230711_1043Past.csv,0,2023-07-11T10:54:04.000\n"
)


@pytest.fixture(scope="module")
def duckdb_connection():
    import io

    conn = connect(CC_Bond=cc_bond_df(), ElectionCalendar=pl.read_csv(io.StringIO(ELECTION_CSV), try_parse_dates=True), EmployeeTbl=employee_df())
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def db(monkeypatch, duckdb_connection):
    """The backend both readers query, inside a transaction rolled back after the test."""
    backend = Backend(duckdb_connection)
    install(monkeypatch, backend)
    duckdb_connection.execute("BEGIN TRANSACTION")
    yield backend
    duckdb_connection.execute("ROLLBACK")


@pytest.fixture(autouse=True)
def clear_piot_cache():
    """Clear the global cache: equal serialized LazyFrames in different tests would share cached data."""
    from polars_io_tools.io_sources import lazy_cache

    lazy_cache._CACHE.clear()
    yield
    lazy_cache._CACHE.clear()


def scan_db(query: str, **kw) -> pl.LazyFrame:
    return ENTRY_POINTS["scan_db"](query, **kw)


def _union(*rows: str) -> str:
    return " UNION ALL ".join(f"SELECT {r}" for r in rows)


# Tables a query row needs, created inside the test's transaction.
TABLES = {
    "A": "CREATE TABLE A AS "
    + _union("1 AS id, 10 AS shared, 100 AS aval", "2 AS id, 20 AS shared, 200 AS aval", "3 AS id, 30 AS shared, 300 AS aval"),
    "B": "CREATE TABLE B AS " + _union("10 AS shared, 1000 AS bval", "30 AS shared, 3000 AS bval", "40 AS shared, 4000 AS bval"),
    "CreditRef": "CREATE TABLE CreditRef AS "
    + _union(
        "1 AS ID, 100 AS ID_BB_COMPANY, 'US' AS CNTRY, DATE '2024-01-15' AS SOME_DATE",
        "2 AS ID, 200 AS ID_BB_COMPANY, 'GB' AS CNTRY, DATE '2024-02-20' AS SOME_DATE",
        "3 AS ID, 100 AS ID_BB_COMPANY, 'JP' AS CNTRY, DATE '2024-03-10' AS SOME_DATE",
    ),
    "BBCredit": "CREATE TABLE BBCredit AS " + _union("100 AS ID_BB_COMPANY, 'USA' AS CNTRY_OF_RISK", "200 AS ID_BB_COMPANY, 'GBR' AS CNTRY_OF_RISK"),
    "PointID": "ALTER TABLE CreditRef ADD COLUMN PointID BIGINT; UPDATE CreditRef SET PointID = ID * 100000",
    "StarA": "CREATE TABLE StarA AS " + _union("1 AS aid, 10 AS joinkey", "2 AS aid, 20 AS joinkey"),
    "StarB": "CREATE TABLE StarB AS " + _union("10 AS bkey, 'x' AS bval", "20 AS bkey, 'y' AS bval"),
    "JoinA": "CREATE TABLE JoinA AS " + _union("1 AS id, 10 AS shared, 100 AS aval", "2 AS id, 20 AS shared, 200 AS aval"),
    "JoinB": "CREATE TABLE JoinB AS " + _union("10 AS shared, 1000 AS bval", "20 AS shared, 2000 AS bval"),
    "Orders": "CREATE TABLE Orders AS "
    + _union(
        "1 AS order_id, 10 AS customer_id, DATE '2024-01-15' AS order_date",
        "2 AS order_id, 20 AS customer_id, DATE '2024-02-20' AS order_date",
        "3 AS order_id, 10 AS customer_id, DATE '2024-03-10' AS order_date",
    ),
    "Products": "CREATE TABLE Products AS "
    + _union(
        "1 AS product_id, 'Widget' AS product_name", "2 AS product_id, 'Gadget' AS product_name", "3 AS product_id, 'Doohickey' AS product_name"
    ),
    "Customers": "CREATE TABLE Customers AS " + _union("10 AS customer_id, 'Alice' AS cust_name", "20 AS customer_id, 'Bob' AS cust_name"),
    "CommaA": "CREATE TABLE CommaA AS " + _union("1 AS aid, 10 AS shared_key", "2 AS aid, 20 AS shared_key"),
    "CommaB": "CREATE TABLE CommaB AS " + _union("10 AS shared_key, 'x' AS bval", "20 AS shared_key, 'y' AS bval"),
    "Sales": "CREATE TABLE Sales AS "
    + _union(
        "'East' AS region, 100 AS amount, DATE '2024-01-15' AS sale_date",
        "'East' AS region, 200 AS amount, DATE '2024-02-20' AS sale_date",
        "'West' AS region, 150 AS amount, DATE '2024-01-10' AS sale_date",
        "'West' AS region, 300 AS amount, DATE '2024-03-05' AS sale_date",
    ),
    "SalesQ1": "CREATE TABLE SalesQ1 AS " + _union("'East' AS region, 100 AS amount", "'West' AS region, 200 AS amount"),
    "SalesQ2": "CREATE TABLE SalesQ2 AS " + _union("'East' AS region, 300 AS amount", "'North' AS region, 400 AS amount"),
    "UnionA": "CREATE TABLE UnionA AS " + _union("1 AS id, 'a' AS name, 10 AS val", "2 AS id, 'b' AS name, 20 AS val"),
    "UnionB": "CREATE TABLE UnionB AS " + _union("3 AS id, 'c' AS name, 30 AS val"),
    "OrderQualA": "CREATE TABLE OrderQualA AS " + _union("1 AS id, DATE '2024-01-01' AS event_date", "2 AS id, DATE '2024-03-01' AS event_date"),
    # RecordDate is a TIMESTAMP holding dates: the mis-declared type cast_map narrows.
    "RecordTbl": "CREATE TABLE RecordTbl AS "
    + _union(
        "CAST('2025-05-30 09:15:00' AS TIMESTAMP) AS RecordDate, 1 AS PointID, 1.5 AS Rate",
        "CAST('2025-05-30 17:45:00' AS TIMESTAMP) AS RecordDate, 2 AS PointID, 2.5 AS Rate",
        "CAST('2025-05-31 00:00:00' AS TIMESTAMP) AS RecordDate, 3 AS PointID, 3.5 AS Rate",
    ),
    "QuotedMetadataColumns": 'CREATE TABLE QuotedMetadataColumns AS SELECT 1 AS "group", 2 AS "value with space", 3 AS ordinary',
    "ns_tbl": "CREATE TABLE ns_tbl AS SELECT * FROM (VALUES (1, TIMESTAMP_NS '2024-01-01 00:00:00'), (2, TIMESTAMP_NS '2024-01-01 00:00:00.000000500')) t(id, ts)",
}


# Predicate -> SQL text. ``expect`` is None when nothing may be pushed, else assert_text arguments.
def _visit(predicate: pl.Expr, dialect) -> exp.Expression | None:
    """Translate without the exception guard of ``convert_predicate_to_sql`` so failures surface."""
    visitor = SQLExpressionVisitor(dialect)
    visitor.visit(get_parsed_expr(predicate))
    return visitor.process_results()


def T(id, pred, dialect, expect, render="same", strict=False, **opts):
    return Case(id, pred=pred, dialect=dialect, render=dialect if render == "same" else render, strict=strict, opts=opts, expect=expect)


def X(*contains, omits=(), equals=None):
    return {"contains": contains, "omits": omits, "equals": equals}


ts, a, s, f, x = pl.col("ts"), pl.col("a"), pl.col("s"), pl.col("f"), pl.col("x")
_EST = timezone(timedelta(hours=-5))
_CH_UTC = "toDateTime64('2026-01-15 00:00:00', 6, 'UTC')"
_BIG_IN = [f"ID{i}" for i in range(DEFAULT_MAX_IN_PREDICATE_SIZE + 1)]
_DT_DATE = (ts.dt.date() == date(2026, 1, 15)) & pl.col("symbol").is_in(["A", "B"])
_REGEXES = ["^(CL|GC|SI|HO)", ".*abc", "foo|bar", "[A-Z]+", r"CL\d+"]

TRANSLATIONS = [
    T("wrap_binary_expr", (pl.col("A") > 5) | pl.col("B").eq(10), "tsql", X(equals="((A > 5) OR (B = 10))")),
    # tz-aware datetimes on ClickHouse are explicit-UTC instants; elsewhere offset strings (issue #52)
    T(
        "tz_aware_datetime_clickhouse_uses_explicit_utc",
        (ts >= datetime(2026, 1, 15, tzinfo=UTC)) & (ts < datetime(2026, 1, 15, 9, 0, 0, 123456, tzinfo=_EST)),
        "clickhouse",
        X(_CH_UTC, "toDateTime64('2026-01-15 14:00:00.123456', 6, 'UTC')", omits="+00:00"),
    ),
    T("tz_aware_datetime_clickhouse_dialect_class-str", ts >= datetime(2026, 1, 15, tzinfo=UTC), "clickhouse", X(_CH_UTC, omits="+00:00")),
    T(
        "tz_aware_datetime_clickhouse_dialect_class-class",
        ts >= datetime(2026, 1, 15, tzinfo=UTC),
        ClickHouse,
        X(_CH_UTC, omits="+00:00"),
        render="clickhouse",
    ),
    T(
        "tz_aware_datetime_is_in_clickhouse_uses_explicit_utc",
        ts.is_in([datetime(2026, 1, 15, tzinfo=UTC), datetime(2026, 1, 16, tzinfo=UTC)]),
        "clickhouse",
        X(_CH_UTC, "toDateTime64('2026-01-16 00:00:00', 6, 'UTC')", omits="+00:00"),
    ),
    *[
        T(f"tz_aware_datetime_non_clickhouse_unchanged-{d}", ts >= datetime(2026, 1, 15, tzinfo=UTC), d, X("2026-01-15 00:00:00+00:00"))
        for d in ["tsql", "postgres", "duckdb"]
    ],
    T(
        "naive_datetime_clickhouse_uses_todatetime64_without_tz",
        ts >= datetime(2024, 1, 15, 12, 30),
        "clickhouse",
        X("toDateTime64('2024-01-15 12:30:00', 6)", omits="'UTC'"),
    ),
    # the MSSQL class and no dialect behave like "tsql"
    *[
        T(f"convert_predicate_to_sql_with_mssql_class-{n}", x > 5, d, X(equals="(x > 5)"), render=None)
        for n, d in [("tsql", "tsql"), ("MSSQL", MSSQL), ("None", None)]
    ],
    # dt.date() becomes a calendar-date expression per dialect (issue #51)
    *[
        T(f"dt_date_predicate_is_pushed_down-{d}", _DT_DATE, d, X(want, "symbol IN ('A', 'B')"))
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
    T(
        "dt_date_pushdown_oracle_uses_trunc",
        ts.dt.date() == date(2024, 5, 6),
        "oracle",
        X("TRUNC(ts) = TO_DATE('2024-05-06', 'YYYY-MM-DD')", omits="CAST(ts AS DATE)"),
    ),
    T(
        "date_is_in_pushdown_oracle_uses_typed_literals",
        pl.col("day").is_in([date(2024, 5, 6), date(2024, 5, 7)]),
        "oracle",
        X("day IN (TO_DATE('2024-05-06', 'YYYY-MM-DD'), TO_DATE('2024-05-07', 'YYYY-MM-DD'))"),
    ),
    # an is_in list over the size limit is not pushed; an AND keeps its other side
    T("oversized_in_predicate_is_not_pushed_down-small", pl.col("id").is_in([f"ID{i}" for i in range(5)]), "tsql", X("IN")),
    T("oversized_in_predicate_is_not_pushed_down-big", pl.col("id").is_in(_BIG_IN), "tsql", None),
    T("oversized_in_predicate_is_not_pushed_down-and", pl.col("id").is_in(_BIG_IN) & (pl.col("year") == 2025), "tsql", X("year", omits="IN")),
    T("oversized_in_predicate_is_not_pushed_down-raised_limit", pl.col("id").is_in(_BIG_IN), "tsql", X("IN"), max_in_predicate_size=len(_BIG_IN)),
    # regex semantics differ between databases and Polars
    *[
        T(f"str_contains_regex_skips_pushdown-{d}-{r}", pl.col("Centre").str.contains(r), d, None)
        for d in ["tsql", "postgres", "clickhouse", "duckdb"]
        for r in _REGEXES
    ],
    T(
        "str_contains_regex_with_and_partial_pushdown",
        pl.col("Centre").str.contains("^test") & (pl.col("EventYear") > 2020),
        "tsql",
        X("EventYear", omits="LIKE"),
        render=None,
    ),
    *[
        T(f"predicate_without_exact_sql_translation_is_not_pushed_down-{p}", p, d, None, strict=True)
        for p, d in [
            (ts.dt.second() == 0, "postgres"),
            (a.cast(pl.UInt8) > 3, "postgres"),
            (a.cast(pl.Int128) > 3, "duckdb"),
            (f.cast(pl.String) == "1.0", "duckdb"),
            (s.str.starts_with("a\\"), "postgres"),
            (s.str.contains("a_b", literal=True), "tsql"),
            (a // 2 == 1, "duckdb"),
        ]
    ],
    # Polars panics while exposing this literal; the panic must not escape conversion
    T("nanosecond_numpy_literal_in_is_in_is_not_pushed_down", ts.is_in([np.datetime64("2024-01-01T00:00:00.000000001", "ns")]), "duckdb", None),
    *[T(f"dt_year_renders_in_dialect-{d}", ts.dt.year() == 2024, d, X("2024"), strict=True) for d in ["tsql", "snowflake", "duckdb"]],
    T("head_with_non_sql_filter_returns_correct_count-udf", pl.col("score").map_elements(lambda v: v > 80, return_dtype=pl.Boolean), "tsql", None),
]


@pytest.mark.parametrize(
    ("pred", "dialect", "render", "strict", "opts", "expect"), params(TRANSLATIONS, "pred", "dialect", "render", "strict", "opts", "expect")
)
def test_translation(pred, dialect, render, strict, opts, expect):
    result = _visit(pred, dialect) if strict else convert_predicate_to_sql(pred, dialect, **opts)
    if expect is None:
        assert result is None, result.sql(render)
    else:
        assert_text(None if result is None else result.sql(render), **expect)


# A hand-built node -> the sqlglot type and SQL text the visitor produces.
def _cmp(name, op, value):
    return BinaryExprNode(
        expr=pl.col(name) > value, left=ColumnNode(expr=pl.col(name), name=name), op=op, right=LiteralNode(expr=pl.lit(value), value=value)
    )


def _logical(op):
    return BinaryExprNode(expr=pl.lit(True), left=_cmp("A", OperatorType.GT, 5), op=op, right=_cmp("B", OperatorType.GT, 10))


def _span(function_type):
    nodes = [get_parsed_expr(pl.col("StartDate")), get_parsed_expr(pl.col("EndDate"))]
    return FunctionNode(expr=pl.col("StartDate"), inputs=nodes, function_type=function_type)


_my_col = ColumnNode(expr=pl.col("my_column"), name="my_column")
_DAYS = "DATEDIFF(DAY, StartDate, EndDate)"

NODES = [
    # LOGICAL_OR / LOGICAL_AND come from Polars' optimizer; unmapped they fell back to client-side filtering
    Case("logical_or_operator", node=_logical(OperatorType.LOGICAL_OR), expect=(None, X("OR", "A > 5", "B > 10")), dialect="tsql"),
    Case("logical_and_operator", node=_logical(OperatorType.LOGICAL_AND), expect=(None, X("AND", "A > 5", "B > 10")), dialect="tsql"),
    Case("visit_column", node=_my_col, expect=(exp.Column, X(equals="my_column")), dialect="tsql"),
    Case("visit_literal-number", node=LiteralNode(expr=pl.lit(42), value=42), expect=(exp.Literal, X(equals="42")), dialect="tsql"),
    Case("visit_literal-null", node=LiteralNode(expr=pl.lit(None), value=None), expect=(exp.Null, X(equals="NULL")), dialect="tsql"),
    Case(
        "visit_function_is_null",
        node=FunctionNode(expr=pl.col("my_column").is_null(), inputs=[_my_col], function_type=BooleanFunctionType.IS_NULL),
        expect=(exp.Is, X(equals="my_column IS NULL")),
        dialect="tsql",
    ),
    Case("total_days_to_datediff-days", node=_span(TemporalFunctionType.TOTAL_DAYS), expect=(exp.DateDiff, X(equals=_DAYS)), dialect="tsql"),
    Case(
        "total_days_to_datediff-hours",
        node=_span(TemporalFunctionType.TOTAL_HOURS),
        expect=(exp.DateDiff, X(equals="DATEDIFF(HOUR, StartDate, EndDate)")),
        dialect="tsql",
    ),
    # the MSSQL class and None normalize to "tsql", so the TSQL-only DATEDIFF still applies
    *[
        Case(f"visitor_dialect_normalization-{n}", node=_span(TemporalFunctionType.TOTAL_DAYS), expect=(exp.DateDiff, X(equals=_DAYS)), dialect=d)
        for n, d in [("tsql", "tsql"), ("MSSQL", MSSQL), ("None", None)]
    ],
]


@pytest.mark.parametrize(("node", "dialect", "expect"), params(NODES, "node", "dialect", "expect"))
def test_visit_node(node, dialect, expect):
    visitor = SQLExpressionVisitor(dialect=dialect)
    visitor.visit(node)
    result = visitor.process_results()
    typ, text = expect
    assert typ is None or isinstance(result, typ), type(result)
    assert_text(result.sql(dialect="tsql"), **text)


def test_total_days_parses_to_function_node():
    node = get_parsed_expr((pl.col("end_dt") - pl.col("start_dt")).dt.total_days())
    assert isinstance(node, FunctionNode) and node.function_type == TemporalFunctionType.TOTAL_DAYS


LITERALS = [
    Case("create_sqlglot_literal_various_types-none", value=None, expect=(exp.Null, "NULL")),
    Case("create_sqlglot_literal_various_types-int", value=123, expect=(exp.Literal, "123")),
    Case("create_sqlglot_literal_various_types-float", value=12.34, expect=(exp.Literal, "12.34")),
    Case("create_sqlglot_literal_various_types-str", value="abc", expect=(exp.Literal, "'abc'")),
    Case("create_sqlglot_literal_various_types-date", value=date(2025, 1, 2), expect=(exp.Literal, "'2025-01-02'")),
    # booleans are SQL booleans, not the strings 'True'/'False'
    Case("create_sqlglot_literal_various_types-true", value=True, expect=(exp.Boolean, "TRUE")),
    Case("create_sqlglot_literal_various_types-false", value=False, expect=(exp.Boolean, "FALSE")),
    Case("naive_datetime_literal_unchanged", value=datetime(2026, 1, 15, 12, 30), expect=(exp.Literal, "'2026-01-15 12:30:00'")),
]


@pytest.mark.parametrize(("value", "expect"), params(LITERALS, "value", "expect"))
def test_create_sqlglot_literal(value, expect):
    result = create_sqlglot_literal(value)
    assert isinstance(result, expect[0])
    assert result.sql() == expect[1]


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
    assert convert_predicate_to_sql(ts >= datetime(2026, 1, 15, tzinfo=UTC), "clickhouse") is None


def test_nanosecond_datetime_pushdown_clickhouse_preserves_precision():
    """Equality on a ``Datetime(ns)`` column reaches the source as an ``is_between`` whose bounds share a microsecond.

    Both bounds must survive as distinct 9-digit ``toDateTime64`` values, or ClickHouse drops matching rows.
    """
    from polars.io.plugins import register_io_source

    captured: dict[str, pl.Expr] = {}

    def source(with_columns, predicate, n_rows, batch_size):
        captured["predicate"] = predicate
        yield pl.DataFrame({"ts": pl.Series("ts", [1705276800123456789], dtype=pl.Int64).cast(pl.Datetime("ns", "UTC"))})

    register_io_source(source, schema={"ts": pl.Datetime("ns", "UTC")}).filter(ts == datetime(2024, 1, 15, 0, 0, 0, 123456, tzinfo=UTC)).collect()
    assert_text(
        convert_predicate_to_sql(captured["predicate"], "clickhouse").sql(dialect="clickhouse"),
        ["toDateTime64('2024-01-15 00:00:00.123456000', 9, 'UTC')", "toDateTime64('2024-01-15 00:00:00.123456999', 9, 'UTC')"],
    )


# MSSQL rendering: bracket quoting, qualified stars, and case kept for anonymous nodes (table-valued functions).
MSSQL_RENDER = [
    Case("mssql_bracket_quoting-plain", source=exp.Column(this="CenterID"), expect=X(equals="[CenterID]")),
    Case("mssql_bracket_quoting-reserved", source=exp.Column(this="GROUP"), expect=X(equals="[GROUP]")),
    Case("mssql_bracket_quoting-bracket", source=exp.Column(this="Some ] Column"), expect=X(equals="[Some ]] Column]")),
    Case("mssql_bracket_quoting-qualified", source=exp.Column(this="Amount", table="Sales"), expect=X(equals="[Sales].[Amount]")),
    Case(
        "qualified_star_in_join",
        source="SELECT a.*, b.SomeCol FROM ServerA.dbo.vTableOne a LEFT JOIN ServerB.dbo.vTableTwo b"
        " ON a.JoinKey = b.JoinKey AND a.AsOfDate BETWEEN b.StartDate AND b.EndDate",
        expect=X("[a].*", omits="[*]"),
    ),
    Case("qualified_star_without_join", source="SELECT t.* FROM SomeDB.dbo.SomeTable t", expect=X("[t].*", omits="[*]")),
    Case("unqualified_star", source="SELECT * FROM SomeTable", expect=X("SELECT *", omits="[*]")),
    Case(
        "tvf_with_literal_args_preserves_case",
        source="SELECT * FROM [FuturesData].[dbo].f_GetFuturesMaxVolAsOfDate('2026-02-06')",
        expect=X("f_GetFuturesMaxVolAsOfDate", omits="F_GETFUTURESMAXVOLASOFDATE"),
    ),
    Case("table_with_nolock_hint_preserves_case", source="SELECT * FROM MyTable (nolock)", expect=X("MyTable", omits="MYTABLE")),
    Case(
        "tvf_with_multiple_args_preserves_case",
        source="SELECT * FROM [db].[dbo].fn_GetData('2026-01-01', 42)",
        expect=X("fn_GetData", omits="FN_GETDATA"),
    ),
    Case(
        "tvf_roundtrip_preserves_full_query",
        source="SELECT * FROM [FuturesData].[dbo].f_GetFuturesMaxVolAsOfDate('2026-02-06')",
        expect=X("[FuturesData]", "[dbo]", "f_GetFuturesMaxVolAsOfDate('2026-02-06')"),
    ),
]


@pytest.mark.parametrize(("source", "expect"), params(MSSQL_RENDER, "source", "expect"))
def test_mssql_render(source, expect):
    node = parse_one(source, dialect=MSSQL) if isinstance(source, str) else source
    assert_text(node.sql(dialect=MSSQL), **expect)


# _prepare_inner_for_subquery: MSSQL rejects ORDER BY in a derived table without TOP/OFFSET (error 1033), so it and
# OPTION hints move to the outer query. Expect: (inner keeps ORDER BY, hoisted ORDER BY text or None, hoisted OPTIONs).
HOISTING = [
    Case(
        "mssql_order_by_without_top_is_moved_to_outer",
        query="SELECT region, amount FROM Sales ORDER BY amount DESC",
        expect=(False, X("amount"), []),
        dialect=MSSQL,
    ),
    Case(
        "mssql_order_by_with_limit_stays_in_inner",
        query="SELECT TOP 2 region, amount FROM Sales ORDER BY amount DESC",
        expect=(True, None, []),
        dialect=MSSQL,
    ),
    # the table alias only exists inside the subquery
    Case(
        "mssql_order_by_hoisted_with_table_qualifiers_stripped",
        query="SELECT cb.region, cb.amount FROM Sales cb ORDER BY cb.amount DESC",
        expect=(False, X("amount", omits="cb."), []),
        dialect=MSSQL,
    ),
    Case(
        "mssql_order_by_with_offset_stays_in_inner",
        query="SELECT region, amount FROM Sales ORDER BY amount OFFSET 1 ROWS FETCH NEXT 2 ROWS ONLY",
        expect=(True, None, []),
        dialect=MSSQL,
    ),
    Case("mssql_option_hints_hoisted", query="SELECT region FROM Sales OPTION (HASH JOIN)", expect=(False, None, ["HASH JOIN"]), dialect=MSSQL),
    Case("non_mssql_no_hoisting", query="SELECT region, amount FROM Sales ORDER BY amount DESC", expect=(True, None, []), dialect="duckdb"),
]


@pytest.mark.parametrize(("query", "dialect", "expect"), params(HOISTING, "query", "dialect", "expect"))
def test_prepare_inner_for_subquery(query, dialect, expect):
    inner_order, hoisted, options = expect
    inner, order, hoisted_options = _prepare_inner_for_subquery(parse_one(query, dialect=MSSQL), dialect=dialect)
    assert (inner.args.get("order") is not None) == inner_order
    if hoisted is None:
        assert order is None
    else:
        assert_text(order.sql(), ignore_case=True, **hoisted)
    assert len(hoisted_options) == len(options)
    for got, want in zip(hoisted_options, options):
        assert_text(got.sql(dialect=MSSQL), want, ignore_case=True)
    assert not (options and list(inner.find_all(exp.QueryOption)))


# Predicate -> the WHERE clause runs and returns every row ``df.filter`` keeps (Polars re-filters, so extra rows are fine).
_PUSHDOWN_DF = pl.DataFrame(
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
_UNTRANSLATABLE = s.str.len_chars() == 3
b = pl.col("b")

KEEPS = [
    Case(f"pushed_predicate_keeps_all_matching_rows-{p}", pred=p, data=_PUSHDOWN_DF, dialect="duckdb")
    for p in [
        ~pl.when(a > 2).then(b > 100).otherwise(b > 0),
        a.is_between(2, 4).is_null(),
        a.is_between(2, 4) == False,
        ((a > 1) & _UNTRANSLATABLE) == False,
        ((a > 1) & _UNTRANSLATABLE).is_null(),
        ~((a > 1) & _UNTRANSLATABLE),
        pl.when(a > 2).then(b > 100).otherwise(_UNTRANSLATABLE),
        pl.when(a > 2).then(1).otherwise(0) == 0,
        a.ne_missing(2),
        ~a.eq_missing(2),
        a.is_in([1, None], nulls_equal=True),
        ~a.is_in([1, None]),
        a % 2 == 1,
        f.cast(pl.Int64) == 2,
        (a & 3) == 0,
        s.str.strip_chars() == "abc",
        s.str.strip_chars("x") == "ab",
        pl.concat_str([s, s], separator="-") == "abc-abc",
        ~s.str.ends_with("_b"),
        ts.dt.year() == 2024,
        ts.cast(pl.Datetime("ms")) == datetime(2024, 1, 1),
        f < float("inf"),
        f.is_in([float("nan")]),
        ts < ts + timedelta(days=1),
    ]
] + [
    # where Polars and SQL semantics differ
    Case(f"pushed_predicate_keeps_polars_semantics-{p}", pred=p, data=pl.DataFrame(data), dialect=d)
    for data, p, d in [
        ({"x": [2]}, x.cast(pl.Boolean) == True, "duckdb"),
        ({"x": [1]}, (~x).cast(pl.Boolean), "duckdb"),
        ({"x": [1]}, (~x) == -2, "duckdb"),
        ({"x": [2**63 - 1]}, (x + 1) < 0, "duckdb"),
        ({"x": [b"ab"]}, x == b"ab", "duckdb"),
        ({"x": [b"ab"]}, x.is_in([b"ab"]), "duckdb"),
        ({"x": ["straße"]}, x.str.to_uppercase() == "STRASSE", "duckdb"),
        ({"x": ["A"]}, ~x.str.starts_with("a"), "sqlite"),
        ({"x": ["A"]}, ~x.str.contains("a", literal=True), "sqlite"),
        ({"x": ["\u0663"]}, x.str.contains(r"^\d$"), "duckdb"),
        ({"x": ["a"]}, x.str.contains(r"\u{61}"), "duckdb"),
        ({"x": ["bad"]}, x.cast(pl.Float64, strict=False).is_null(), "duckdb"),
        ({"x": [0]}, x.cast(pl.Datetime("us")).dt.year() == 1970, "duckdb"),
        ({"x": pl.Series([datetime(2024, 1, 1, 23)]).dt.replace_time_zone("UTC")}, x.cast(pl.Datetime("us")).dt.hour() == 23, "duckdb"),
        ({"x": [1], "ys": [[1]]}, x.is_in(pl.col("ys")), "duckdb"),
        ({"x": ["a\x00b"]}, x == "a\x00b", "duckdb"),
        ({"x": ["a", "b"]}, x == pl.lit(pl.Series(["a", "b"])), "duckdb"),
        ({"x": [1.25]}, x.cast(pl.Decimal(4, 1)) == Decimal("1.2"), "duckdb"),
        ({"x": pl.Series([0.1], dtype=pl.Float32)}, ~x.is_in([0.1, 0.2]), "duckdb"),
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


@pytest.mark.parametrize(("pred", "data", "dialect"), params(KEEPS, "pred", "data", "dialect"))
def test_where_keeps_matching_rows(pred, data, dialect):
    where = _visit(pred, dialect)
    if where is None:
        return
    df = data.with_row_index("__row")
    kept = set(df.filter(pred)["__row"])
    assert kept <= _where_rows(df, where.sql(dialect), dialect), where.sql(dialect)


@pytest.mark.parametrize(
    ("pred", "want", "exact"),
    [
        # CAST(... AS DATE) coerces to an integer in SQLite and would drop the non-midnight row
        pytest.param(ts.dt.date() == date(2024, 5, 6), [("2024-05-06 12:34:56", 1)], True, id="dt_date_pushdown_sqlite_selects_non_midnight_row"),
        pytest.param((ts.dt.year() == 2024) & (x > 0), [("2024-05-06 12:34:56", 1)], False, id="dt_year_pushdown_sqlite_runs"),
    ],
)
def test_sqlite_where_on_text_timestamps(pred, want, exact):
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE t (ts TEXT, x INTEGER)")
    con.execute("INSERT INTO t VALUES ('2024-05-06 12:34:56', 1), ('2024-05-07 00:00:00', 2), ('2023-05-07 00:00:00', 3)")
    rows = con.execute(f"SELECT ts, x FROM t WHERE {convert_predicate_to_sql(pred, 'sqlite').sql(dialect='sqlite')}").fetchall()
    assert rows == want if exact else set(want) <= set(rows)


def _identity(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf


@dataclass(frozen=True)
class QueryCase:
    """A reader query and the Polars steps ``op`` run on it; the result equals the database's answer put through ``op``.

    The ``expected_sql*`` fields check the one data query the reader sends, or its part before (inner) or after (outer)
    the ``__cpl_subq`` subquery wrapper, case-insensitively. A tuple item of a ``contains`` field is satisfied by any of
    its alternatives.
    """

    id: str
    query: str
    op: Callable[[pl.LazyFrame], pl.LazyFrame] = _identity
    also_clickhouse: bool = False
    # names in TABLES to create first
    tables: tuple[str, ...] = ()
    # reader keyword arguments; a callable value is called for each test
    scan_kwargs: dict = field(default_factory=dict)
    # applied to the database's answer before ``op``
    native_transform: Callable[[pl.DataFrame], pl.DataFrame] | None = None
    check_row_order: bool = False
    expected_height: int | None = None
    expected_query_count: int | None = None
    expected_log: str | None = None
    log_level: int = logging.WARNING
    # the query with this WHERE clause, then selecting ``expected_where_columns``, gives the same frame
    expected_where: str | None = None
    expected_where_columns: list[str] | None = None
    expected_sql_contains: tuple = ()
    expected_sql_omits: tuple = ()
    expected_sql_startswith: str | None = None
    expected_inner_sql_contains: tuple = ()
    expected_outer_sql_contains: tuple = ()
    expected_outer_sql_omits: tuple = ()

    @property
    def checks_sql(self) -> bool:
        return bool(
            self.expected_sql_contains
            or self.expected_sql_omits
            or self.expected_sql_startswith
            or self.expected_inner_sql_contains
            or self.expected_outer_sql_contains
            or self.expected_outer_sql_omits
        )


def _query_params(cases: list[QueryCase]) -> list:
    """One ``(entry, tables, case)`` param per reader the case runs on."""
    return [
        pytest.param(entry, case.tables, case, id=f"{case.id}-{entry}" if case.also_clickhouse else case.id)
        for case in cases
        for entry in (["scan_db", "scan_clickhouse"] if case.also_clickhouse else ["scan_db"])
    ]


def _filter(*preds, select=None, after=None):
    def op(lf):
        lf = lf.select(select) if select else lf
        lf = lf.filter(*preds)
        return lf.select(after) if after else lf

    return op


def _select(cols):
    return lambda lf: lf.select(cols)


def _pickled(op=lambda lf: lf):
    import cloudpickle

    return lambda lf: cloudpickle.loads(cloudpickle.dumps(op(lf)))


_CB_ALL = ["CenterID", "CentreCode", "ISOCountryCode", "Centre", "EventYear", "EventDate", "EventDayOfWeek", "EventName", "FileType"]
_DAY = pl.col("EventName").str.contains("Day")
_COMPLEX = pl.col("EventYear").is_between(2015, 2050) & ((pl.col("EventDayOfWeek") == "Fri") | (pl.col("EventDayOfWeek") == "Sat")) & _DAY
_UDF = pl.col("score").map_elements(lambda v: v > 80, return_dtype=pl.Boolean)
_JOIN_STAR = "SELECT cr.*, br.CNTRY_OF_RISK FROM CreditRef cr LEFT JOIN BBCredit br ON cr.ID_BB_COMPANY = br.ID_BB_COMPANY"
_JOIN_CAST = "SELECT cr.*, CAST(cr.SOME_DATE AS DATE) AS bus_date, br.CNTRY_OF_RISK FROM CreditRef cr LEFT JOIN BBCredit br ON cr.ID_BB_COMPANY = br.ID_BB_COMPANY"
_MULTI_STAR = "SELECT o.*, p.*, c.cust_name FROM Orders o CROSS JOIN Products p LEFT JOIN Customers c ON o.customer_id = c.customer_id"
_STAR_TABLES = ("CreditRef", "BBCredit")
_MULTI_TABLES = ("Orders", "Products", "Customers")
_DATE_FILTER = (pl.col("EventDate") >= datetime(2013, 1, 1)) & (pl.col("EventDate") < datetime(2017, 1, 1))
_COUNTRIES = pl.col("ISOCountryCode").is_in(["GT", "BD", "RO"])
_RECORD = {"cast_map": {"RecordDate": pl.Date}}


def _record_dates(df):
    return df.with_columns(pl.col("RecordDate").cast(pl.Date))


def _parts(*cs):
    return lambda: (cpl.ReadPartition(pl.col("ISOCountryCode") == c, key=c) for c in cs)


_NESTED_STAR = "SELECT * FROM (SELECT CAST(EventDate AS DATE) AS EventDateAsDate, * FROM CC_Bond) AS t"

QUERIES = [
    # filters with an equivalent SQL WHERE, shared by scan_db and scan_clickhouse
    QueryCase(
        "basic_query",
        "SELECT CenterID, Centre, EventYear FROM CC_Bond",
        _filter(pl.col("EventYear") > 2015),
        also_clickhouse=True,
        expected_where="EventYear > 2015",
        check_row_order=True,
    ),
    QueryCase(
        "column_selection",
        "SELECT * FROM CC_Bond",
        _filter(pl.col("Centre").str.contains("Da"), select=["CenterID", "Centre", "EventDate"]),
        also_clickhouse=True,
        expected_where="Centre LIKE '%Da%'",
        expected_where_columns=["CenterID", "Centre", "EventDate"],
    ),
    QueryCase(
        "complex_query",
        f"SELECT {', '.join(_CB_ALL)} FROM CC_Bond",
        _filter(_COMPLEX, select=_CB_ALL),
        also_clickhouse=True,
        expected_where="EventYear BETWEEN 2015 AND 2050 AND (EventDayOfWeek = 'Fri' OR EventDayOfWeek = 'Sat') AND EventName LIKE '%Day%'",
        expected_where_columns=_CB_ALL,
    ),
    QueryCase(
        "not_null",
        "SELECT CenterID, Centre, EventDate FROM CC_Bond",
        _filter(pl.col("EventDate").is_not_null()),
        also_clickhouse=True,
        expected_where="EventDate IS NOT NULL",
        check_row_order=True,
    ),
    # filter on a column the selection drops
    QueryCase(
        "empty_df",
        "SELECT CenterID, Centre, EventDate, EventYear FROM CC_Bond",
        _filter(pl.col("EventYear") < 1900, after=["CenterID", "Centre", "EventDate"]),
        also_clickhouse=True,
        expected_where="EventYear < 1900",
        expected_where_columns=["CenterID", "Centre", "EventDate"],
        check_row_order=True,
        expected_height=0,
    ),
    QueryCase(
        "in_operator",
        "SELECT CenterID, Centre, EventYear, EventDate, EventDayOfWeek FROM CC_Bond",
        _filter(pl.col("EventDayOfWeek").is_in(["Fri", "Sat", "Sun"])),
        also_clickhouse=True,
        expected_where="EventDayOfWeek IN ('Fri', 'Sat', 'Sun')",
    ),
    QueryCase(
        "not_like",
        "SELECT CenterID, Centre, EventYear, EventName FROM CC_Bond",
        _filter(~_DAY),
        also_clickhouse=True,
        expected_where="EventName NOT LIKE '%Day%'",
    ),
    QueryCase(
        "implicit_cast_query",
        "SELECT CenterID, Centre, EventYear FROM CC_Bond",
        _filter(pl.col("EventYear") > 2015.1),
        expected_where="EventYear > 2015",
        check_row_order=True,
    ),
    QueryCase(
        "join",
        "SELECT cb.CenterID, cb.Centre, ec.Country, ec.ElectFor FROM CC_Bond AS cb, ElectionCalendar AS ec",
        _filter(pl.col("Country").str.contains("a")),
        expected_where="ec.Country LIKE '%a%'",
    ),
    *[
        QueryCase(f"str_contains-{i}", "SELECT * FROM CC_Bond", _filter(p))
        for i, p in enumerate(
            [
                pl.col("Centre").str.contains("^(Dh|Ph)"),
                pl.col("Centre").str.contains(".*mala"),
                pl.col("Centre").str.contains("Da"),
                pl.col("Centre").str.contains("Da", literal=True),
                pl.col("Centre").str.contains("^(Dh|Ph)") & (pl.col("EventYear") > 2020),
                ~pl.col("Centre").str.contains("Da"),
            ]
        )
    ],
    # dt.date() pushes down as a DATE cast (issue #51); Polars re-filters, so only the SQL shows it
    QueryCase(
        "dt_date_filter_pushdown-eq",
        "SELECT CenterID, Centre, EventDate FROM CC_Bond",
        _filter(pl.col("EventDate").dt.date() == date(2013, 10, 24)),
        expected_sql_contains=('CAST("EventDate" AS DATE)',),
    ),
    QueryCase(
        "dt_date_filter_pushdown-ge",
        "SELECT CenterID, Centre, EventDate FROM CC_Bond",
        _filter(pl.col("EventDate").dt.date() >= date(2013, 10, 24)),
        expected_sql_contains=('CAST("EventDate" AS DATE)',),
    ),
    # equality on a nanosecond column is an is_between whose upper bound keeps sub-microsecond digits
    QueryCase(
        "nanosecond_equality_is_not_truncated",
        "SELECT id, ts FROM ns_tbl",
        _filter(ts == datetime(2024, 1, 1)),
        tables=("ns_tbl",),
        expected_height=2,
    ),
    # aliases and casts in the query
    QueryCase(
        "rename_with_filter_integration",
        "SELECT CenterID, EventYear FROM CC_Bond",
        lambda lf: lf.select(pl.col("CenterID").alias("center_id"), pl.col("EventYear").alias("event_year")).filter(
            pl.col("event_year").is_not_null()
        ),
    ),
    QueryCase(
        "select_alias_predicate_pushdown_simple",
        "SELECT EventYear AS event_year, CenterID AS center_id FROM CC_Bond",
        _filter(pl.col("event_year") > 2016, after=["event_year", "center_id"]),
    ),
    QueryCase("select_alias_cast_datetime_to_date_types", "SELECT CAST(EventDate AS DATE) AS BusDate, CenterID FROM CC_Bond"),
    QueryCase("select_original_and_cast_same_column", "SELECT EventDate, CAST(EventDate AS DATE) AS BusDate, CenterID FROM CC_Bond"),
    QueryCase(
        "nested_select_star_with_cast_alias",
        _NESTED_STAR,
        check_row_order=True,
        log_level=logging.DEBUG,
        expected_log="Executing SQL with pushdown: " + parse_one(_NESTED_STAR, dialect="duckdb").sql(dialect="duckdb"),
    ),
    # cloudpickle round trip (needed by Ray)
    QueryCase("scan_db_pickle_basic", "SELECT * FROM CC_Bond", _pickled(), check_row_order=True),
    QueryCase("scan_db_pickle_with_filter", "SELECT * FROM CC_Bond", _pickled(_filter(pl.col("EventYear") > 2020)), check_row_order=True),
    QueryCase("scan_db_pickle_with_select", "SELECT CenterID, Centre, EventYear FROM CC_Bond", _pickled(), check_row_order=True),
    # head: Polars pushes n_rows only without a predicate; head(n).filter(p) reads n rows before filtering
    QueryCase(
        "polars_does_not_push_n_rows_with_predicate",
        "SELECT * FROM EmployeeTbl",
        lambda lf: lf.filter(pl.col("score") > 80).head(5),
        expected_sql_contains=("WHERE",),
        expected_sql_omits=("LIMIT",),
        check_row_order=True,
    ),
    QueryCase(
        "polars_pushes_n_rows_without_predicate",
        "SELECT * FROM EmployeeTbl",
        lambda lf: lf.head(5),
        also_clickhouse=True,
        expected_sql_contains=("LIMIT",),
        check_row_order=True,
    ),
    QueryCase(
        "head_with_non_sql_filter_returns_correct_count", "SELECT * FROM EmployeeTbl", lambda lf: lf.filter(_UDF).head(10), check_row_order=True
    ),
    QueryCase("head_then_filter-sql", "SELECT * FROM EmployeeTbl", lambda lf: lf.head(10).filter(pl.col("score") > 80), check_row_order=True),
    QueryCase("head_then_filter-udf", "SELECT * FROM EmployeeTbl", lambda lf: lf.head(10).filter(_UDF), check_row_order=True),
    QueryCase(
        "workaround_collect_then_head", "SELECT * FROM EmployeeTbl", lambda lf: lf.filter(_UDF).collect().head(10).lazy(), check_row_order=True
    ),
    QueryCase(
        "head_zero_skips_query",
        "SELECT CenterID, Centre, EventYear FROM CC_Bond",
        lambda lf: lf.head(0),
        also_clickhouse=True,
        expected_query_count=0,
    ),
    QueryCase(
        "head_zero_with_column_selection",
        "SELECT CenterID, Centre, EventYear FROM CC_Bond",
        lambda lf: lf.select(["CenterID", "Centre"]).head(0),
        also_clickhouse=True,
        expected_query_count=0,
    ),
    # joins: rebuilt projections keep the table qualifier, or the query is wrapped as a subquery
    QueryCase(
        "with_columns_preserves_qualification_on_join",
        "SELECT a.shared, a.aval, b.bval FROM A AS a LEFT JOIN B AS b ON a.shared = b.shared",
        _select(["shared", "aval"]),
        tables=("A", "B"),
        expected_sql_contains=(("SELECT a.shared, a.aval", 'SELECT "a"."shared", "a"."aval"', "SELECT [a].[shared], [a].[aval]"),),
    ),
    QueryCase(
        "select_shared_column_is_qualified",
        _JOIN_STAR,
        _select(["ID_BB_COMPANY", "CNTRY_OF_RISK"]),
        tables=_STAR_TABLES,
        expected_sql_contains=("__cpl_subq",),
    ),
    QueryCase(
        "select_non_shared_column_is_qualified", _JOIN_STAR, _select(["ID", "CNTRY"]), tables=_STAR_TABLES, expected_sql_contains=("__cpl_subq",)
    ),
    QueryCase(
        "select_all_star_expanded_columns",
        _JOIN_STAR,
        _select(["ID", "ID_BB_COMPANY", "CNTRY", "SOME_DATE"]),
        tables=_STAR_TABLES,
        expected_sql_contains=("__cpl_subq",),
    ),
    QueryCase(
        "predicate_pushdown_works_with_star_join", _JOIN_STAR, _filter(pl.col("CNTRY") == "US"), tables=_STAR_TABLES, expected_sql_contains=("WHERE",)
    ),
    # columns only on the cr side resolve without a qualifier
    QueryCase(
        "predicate_with_multiple_filters",
        _JOIN_STAR,
        _filter((pl.col("SOME_DATE") == date(2024, 1, 15)) & (pl.col("PointID") == 100000)),
        tables=(*_STAR_TABLES, "PointID"),
        expected_sql_contains=("WHERE",),
        expected_height=1,
    ),
    QueryCase(
        "combined_select_qualified_and_predicate_unqualified",
        _JOIN_STAR,
        _filter(pl.col("CNTRY") == "US", after=["ID_BB_COMPANY", "CNTRY_OF_RISK"]),
        tables=_STAR_TABLES,
        expected_sql_contains=(
            "__cpl_subq",
            "WHERE",
        ),
    ),
    QueryCase(
        "alias_predicate_rewrites_to_underlying_expression",
        _JOIN_CAST,
        _filter(pl.col("bus_date") >= date(2024, 2, 1)),
        tables=_STAR_TABLES,
        expected_sql_contains=(
            "WHERE",
            "CAST",
        ),
    ),
    QueryCase(
        "select_alias_column_alongside_star_expanded",
        _JOIN_CAST,
        _select(["bus_date", "ID_BB_COMPANY"]),
        tables=_STAR_TABLES,
        expected_sql_contains=(
            "__cpl_subq",
            "CAST",
            "bus_date",
        ),
    ),
    QueryCase(
        "bare_star_join_does_not_qualify",
        "SELECT * FROM StarA a LEFT JOIN StarB b ON a.joinkey = b.bkey",
        _select(["aid", "bval"]),
        tables=("StarA", "StarB"),
        expected_sql_omits=("a.aid", '"a"."aid"'),
    ),
    # a redundant full-schema select leaves the SQL unchanged
    QueryCase(
        "explicit_columns_with_join_uses_subquery",
        "SELECT a.shared, b.bval FROM JoinA a LEFT JOIN JoinB b ON a.shared = b.shared",
        _select(["shared", "bval"]),
        tables=("JoinA", "JoinB"),
        expected_sql_contains=(
            ("a.shared", '"a"."shared"', "[a].[shared]"),
            ("b.bval", '"b"."bval"', "[b].[bval]"),
        ),
        expected_sql_omits=("__cpl_subq",),
    ),
    QueryCase(
        "no_expansion_when_no_joins",
        "SELECT * FROM CreditRef",
        _select(["ID", "CNTRY"]),
        tables=_STAR_TABLES,
        expected_sql_omits=("cr.ID", '"cr"."ID"'),
    ),
    QueryCase("no_modification_without_pushdown", _JOIN_STAR, tables=_STAR_TABLES, expected_sql_contains=(("cr.*", '"cr".*'),)),
    QueryCase(
        "multi_star_select_shared_uses_subquery",
        _MULTI_STAR,
        _select(["customer_id", "cust_name"]),
        tables=_MULTI_TABLES,
        expected_sql_contains=("__cpl_subq",),
    ),
    QueryCase(
        "multi_star_select_non_shared_uses_subquery",
        _MULTI_STAR,
        _select(["order_id", "product_name"]),
        tables=_MULTI_TABLES,
        expected_sql_contains=("__cpl_subq",),
    ),
    QueryCase(
        "multi_star_filter_with_subquery",
        _MULTI_STAR,
        _filter(pl.col("cust_name") == "Alice", after=["order_id", "cust_name", "product_name"]),
        tables=_MULTI_TABLES,
        expected_sql_contains=(
            "__cpl_subq",
            "WHERE",
        ),
        expected_height=6,
    ),
    QueryCase("multi_star_no_subquery_without_pushdown", _MULTI_STAR, tables=_MULTI_TABLES, expected_sql_omits=("__cpl_subq",)),
    # comma-separated tables are implicit cross joins
    QueryCase(
        "comma_join_with_qualified_star_uses_subquery",
        "SELECT a.*, b.bval FROM CommaA a, CommaB b WHERE a.shared_key = b.shared_key",
        _select(["shared_key", "bval"]),
        tables=("CommaA", "CommaB"),
        expected_sql_contains=("__cpl_subq",),
    ),
    # subquery wrapping: ORDER BY without TOP/OFFSET would move out (MSSQL only); others stay inside
    QueryCase(
        "order_by_stays_in_inner_for_non_mssql",
        "SELECT region, amount FROM Sales ORDER BY amount DESC",
        _select(["region", "amount"]),
        tables=("Sales",),
        expected_sql_contains=("ORDER BY",),
        expected_sql_omits=("__cpl_subq",),
        check_row_order=True,
    ),
    QueryCase(
        "order_by_with_limit_stays_in_inner",
        "SELECT region, amount FROM Sales ORDER BY amount DESC LIMIT 2",
        _select(["region"]),
        tables=("Sales",),
        expected_sql_contains=("__cpl_subq",),
        expected_inner_sql_contains=("ORDER BY",),
        check_row_order=True,
    ),
    QueryCase(
        "no_wrapping_without_pushdown",
        "SELECT region, amount FROM Sales ORDER BY amount",
        tables=("Sales",),
        expected_sql_contains=("ORDER BY",),
        expected_sql_omits=("__cpl_subq",),
        check_row_order=True,
    ),
    QueryCase(
        "alias_predicate_works_without_rewriting",
        "SELECT region, CAST(sale_date AS DATE) AS sale_date FROM Sales",
        _filter(pl.col("sale_date") >= date(2024, 2, 1)),
        tables=("Sales",),
        expected_sql_contains=("__cpl_subq",),
        expected_outer_sql_contains=("WHERE",),
        expected_outer_sql_omits=("CAST",),
    ),
    # a predicate on a UNION filters every branch
    QueryCase(
        "union_predicate_filters_all_branches",
        "SELECT region, amount FROM SalesQ1 UNION ALL SELECT region, amount FROM SalesQ2",
        _filter(pl.col("region") == "East"),
        tables=("SalesQ1", "SalesQ2"),
        expected_sql_contains=("__cpl_subq",),
        expected_height=2,
    ),
    QueryCase(
        "union_column_selection",
        "SELECT id, name, val FROM UnionA UNION ALL SELECT id, name, val FROM UnionB",
        _select(["id", "name"]),
        tables=("UnionA", "UnionB"),
        expected_sql_contains=("__cpl_subq",),
    ),
    QueryCase(
        "group_by_with_predicate_pushdown",
        "SELECT region, SUM(amount) AS total FROM Sales GROUP BY region",
        _filter(pl.col("total") >= 250),
        tables=("Sales",),
        expected_sql_contains=("__cpl_subq",),
        expected_inner_sql_contains=("GROUP BY",),
    ),
    QueryCase(
        "cte_query_with_column_selection",
        "WITH cte AS (SELECT region, amount FROM Sales) SELECT region, amount FROM cte",
        _select(["region"]),
        tables=("Sales",),
        expected_sql_contains=("__cpl_subq",),
    ),
    QueryCase(
        "order_by_with_offset_stays_in_inner",
        "SELECT region, amount FROM Sales ORDER BY amount LIMIT 2 OFFSET 1",
        _select(["region"]),
        tables=("Sales",),
        expected_sql_contains=("__cpl_subq",),
        expected_inner_sql_contains=("ORDER BY",),
        check_row_order=True,
    ),
    QueryCase(
        "order_by_qualified_stays_in_inner_for_non_mssql",
        "SELECT a.id, a.event_date FROM OrderQualA a ORDER BY a.event_date",
        _select(["id", "event_date"]),
        tables=("OrderQualA",),
        expected_sql_contains=("ORDER BY",),
        expected_sql_omits=("__cpl_subq",),
        check_row_order=True,
    ),
    QueryCase(
        "predicate_only_pushdown",
        "SELECT region, amount, sale_date FROM Sales",
        _filter(pl.col("region") == "East"),
        tables=("Sales",),
        expected_sql_contains=("__cpl_subq",),
        expected_outer_sql_contains=("WHERE",),
    ),
    QueryCase(
        "nested_subquery_wrapping",
        "SELECT region, amount FROM (SELECT region, amount FROM Sales) AS inner_q",
        _select(["region"]),
        tables=("Sales",),
        expected_sql_contains=("__cpl_subq",),
    ),
    # metadata-derived reserved and special-character names stay selectable
    QueryCase(
        "column_selection_quotes_metadata_names",
        "SELECT * FROM QuotedMetadataColumns",
        _select(["group", "value with space"]),
        tables=("QuotedMetadataColumns",),
        scan_kwargs={"krb5": False},
    ),
    # cast_map narrows a column server side
    QueryCase(
        "cast_map_narrows_column_server_side",
        "SELECT * FROM RecordTbl",
        lambda lf: lf.sort("PointID"),
        tables=("RecordTbl",),
        scan_kwargs=_RECORD,
        native_transform=_record_dates,
        check_row_order=True,
    ),
    QueryCase(
        "cast_map_pushes_date_predicate_to_sql",
        "SELECT * FROM RecordTbl",
        lambda lf: lf.filter(pl.col("RecordDate") == date(2025, 5, 30)).sort("PointID"),
        tables=("RecordTbl",),
        scan_kwargs=_RECORD,
        native_transform=_record_dates,
        check_row_order=True,
        expected_sql_contains=(
            "CAST",
            "DATE",
            "RecordDate",
            "WHERE",
        ),
        expected_height=2,
    ),
    # the inner cast layer lists every column; the outer select prunes
    QueryCase(
        "cast_map_projection_pushdown_still_prunes",
        "SELECT * FROM RecordTbl",
        lambda lf: lf.select(["RecordDate", "PointID"]).sort("PointID"),
        tables=("RecordTbl",),
        scan_kwargs=_RECORD,
        native_transform=_record_dates,
        check_row_order=True,
        expected_sql_startswith='SELECT "RECORDDATE", "POINTID" FROM',
    ),
    # partitioned reads return the unpartitioned rows
    *[
        QueryCase(
            f"by_time_matches_unpartitioned-{every}",
            "SELECT * FROM CC_Bond",
            _filter(_DATE_FILTER),
            scan_kwargs={"partitions": lambda every=every: cpl.by_time("EventDate", every)},
        )
        for every in ["1y", "1mo", "6mo", 90, "1q"]
    ],
    # [2013-01-01, 2017-01-01) split yearly is four windows (exclusive upper bound)
    QueryCase(
        "by_time_partition_count",
        "SELECT * FROM CC_Bond",
        _filter(_DATE_FILTER),
        scan_kwargs={"partitions": lambda: cpl.by_time("EventDate", "1y")},
        log_level=logging.DEBUG,
        expected_log="running 4 partition(s)",
    ),
    QueryCase(
        "by_value_derived_from_predicate",
        "SELECT * FROM CC_Bond",
        _filter(_COUNTRIES),
        scan_kwargs={"partitions": lambda: cpl.by_value("ISOCountryCode")},
    ),
    QueryCase(
        "by_value_explicit_groups",
        "SELECT * FROM CC_Bond",
        _filter(_COUNTRIES),
        scan_kwargs={"partitions": lambda: cpl.by_value("ISOCountryCode", [["GT", "BD"], "RO"])},
    ),
    QueryCase(
        "by_range_numeric",
        "SELECT * FROM CC_Bond",
        _filter((pl.col("EventYear") >= 2013) & (pl.col("EventYear") < 2055)),
        scan_kwargs={"partitions": lambda: cpl.by_range("EventYear", every=10)},
    ),
    # every CC_Bond row has FileType 'G'
    QueryCase(
        "explicit_read_partition_list",
        "SELECT * FROM CC_Bond",
        scan_kwargs={"partitions": lambda: [cpl.ReadPartition(pl.col("FileType") == "G", key="G")]},
    ),
    # a one-shot generator of slices, and a projection without the partition column the client-side filter needs
    QueryCase(
        "generator_partitions_retain_columns_under_projection",
        "SELECT * FROM CC_Bond",
        _filter(_COUNTRIES, after="CentreCode"),
        scan_kwargs={"partitions": _parts("GT", "BD", "RO")},
    ),
    QueryCase(
        "max_concurrency_matches_unpartitioned",
        "SELECT * FROM CC_Bond",
        _filter(_DATE_FILTER),
        scan_kwargs={"partitions": lambda: cpl.by_time("EventDate", "1mo"), "max_concurrency": 1},
    ),
    # a slice predicate SQL cannot express is applied client side
    QueryCase(
        "untranslatable_predicate_filters_client_side",
        "SELECT * FROM CC_Bond",
        scan_kwargs={
            "partitions": lambda: [cpl.ReadPartition(pl.col("EventName").map_elements(lambda v: "Day" in v, return_dtype=pl.Boolean), key=0)]
        },
        native_transform=lambda df: df.filter(_DAY),
    ),
    QueryCase(
        "unbounded_predicate_falls_back_to_single_query",
        "SELECT * FROM CC_Bond",
        scan_kwargs={"partitions": lambda: cpl.by_time("EventDate")},
        log_level=logging.DEBUG,
        expected_log="running 1 partition(s)",
    ),
    # a top-level ORDER BY only loses global order under partitioning: warn, don't raise
    QueryCase(
        "partitioned_query_with_order_by_warns_not_raises",
        "SELECT * FROM CC_Bond ORDER BY EventDate DESC",
        _filter(_DATE_FILTER),
        scan_kwargs={"partitions": lambda: cpl.by_time("EventDate", "1y")},
        expected_log="ORDER BY",
    ),
]


@pytest.fixture
def tables(request, db):
    """Creates the TABLES entries named by the parameter inside the test's transaction."""
    for name in request.param:
        db.conn.execute(TABLES[name])


def _native(db, case: QueryCase) -> pl.LazyFrame:
    df = db.native(case.query)
    return case.op((case.native_transform(df) if case.native_transform else df).lazy())


@pytest.mark.parametrize(("entry", "tables", "case"), _query_params(QUERIES), indirect=["tables"])
def test_scan_matches_native(db, tables, caplog, entry, case):
    lf = ENTRY_POINTS[entry](case.query, **{k: v() if callable(v) else v for k, v in case.scan_kwargs.items()})
    start = len(db.calls)
    with caplog.at_level(case.log_level):
        got = assert_matches_native(case.op(lf), _native(db, case), None, check_row_order=case.check_row_order)
    sent = db.queries(start)
    calls = len(db.calls) - start

    assert got.columns == _native(db, case).collect_schema().names()
    assert case.op(lf).collect_schema() == got.schema
    if case.expected_height is not None:
        assert got.height == case.expected_height
    if case.expected_where is not None:
        filtered = ENTRY_POINTS[entry](f"{case.query} WHERE {case.expected_where}")
        if case.expected_where_columns is not None:
            filtered = filtered.select(case.expected_where_columns)
        assert_frame_equal(filtered.collect(), got, check_row_order=case.check_row_order)

    if case.expected_log is not None:
        assert any(case.expected_log in r.message for r in caplog.records), [r.message for r in caplog.records]
    if case.expected_query_count is not None:
        assert calls == case.expected_query_count, db.calls[start:]
    if case.checks_sql:
        (sql,) = sent
        inner, outer = sql.split("__cpl_subq")[0], sql.split("__cpl_subq")[-1]
        assert_text(sql, case.expected_sql_contains, case.expected_sql_omits, startswith=case.expected_sql_startswith, ignore_case=True)
        assert_text(inner, case.expected_inner_sql_contains, ignore_case=True)
        assert_text(outer, case.expected_outer_sql_contains, case.expected_outer_sql_omits, ignore_case=True)


def _identity_schema(schema):
    return schema


_PA_SCHEMA = pa.schema([("CenterID", pa.int64()), ("Centre", pa.string())])
_PAIR = "SELECT CenterID, Centre FROM CC_Bond"


@pytest.mark.parametrize(
    ("query", "kwargs"),
    [
        pytest.param("SELECT * FROM CC_Bond", {"query_timeout_sec": 30}, id="query_timeout_sec_passed_to_schema_query"),
        pytest.param(
            "SELECT * FROM CC_Bond",
            {"query_timeout_sec": 30, "login_timeout_sec": 10, "max_text_size": 4096},
            id="multiple_kwargs_passed_to_both_queries",
        ),
        pytest.param(_PAIR, {"query_timeout_sec": 30, "schema": _PA_SCHEMA}, id="schema_kwarg_passed_to_both_queries"),
        pytest.param(_PAIR, {"query_timeout_sec": 30, "map_schema": _identity_schema}, id="map_schema_kwarg_passed_to_both_queries"),
        pytest.param(
            _PAIR,
            {"query_timeout_sec": 30, "login_timeout_sec": 10, "schema": _PA_SCHEMA, "map_schema": _identity_schema},
            id="all_kwargs_passed_to_both_queries",
        ),
    ],
)
def test_arrow_odbc_kwargs_reach_schema_and_data_queries(db, query, kwargs):
    lf = scan_db(query, krb5=False, **kwargs)
    lf.collect()
    for data in (False, True):
        (call,) = [c for c in db.calls if ("LIMIT 0" not in c["query"]) == data]
        assert {k: call["kwargs"].get(k) for k in kwargs} == kwargs, call["kwargs"]


_CC = "SELECT * FROM CC_Bond"
_ERR = (ValueError, pl.exceptions.ComputeError)

RAISES = [
    Case(
        "max_partitions_guardrail_raises",
        run=lambda: scan_db(_CC, partitions=cpl.by_time("EventDate", "1d"), max_partitions=5).filter(_DATE_FILTER).collect(),
        expect=(pl.exceptions.ComputeError, "exceeds max_partitions"),
    ),
    Case("unknown_partition_column_raises", run=lambda: scan_db(_CC, partitions=cpl.by_time("not_a_column")), expect=(ValueError, "not_a_column")),
    Case(
        "explicit_partitions_over_max_raises",
        run=lambda: scan_db(_CC, partitions=list(_parts("GT", "BD", "RO")()), max_partitions=2).collect(),
        expect=(pl.exceptions.ComputeError, "exceeds max_partitions"),
    ),
    *[
        Case(
            f"partitioned_query_with_unsafe_clause_raises-{match}",
            run=lambda q=q: scan_db(q, partitions=cpl.by_time("EventDate", "1y")).filter(_DATE_FILTER).collect(),
            expect=(_ERR, match),
        )
        for q, match in [
            ("SELECT * FROM CC_Bond ORDER BY EventDate LIMIT 10", "LIMIT"),
            ("SELECT * FROM CC_Bond OFFSET 5 ROWS", "OFFSET"),
            ("SELECT *, ROW_NUMBER() OVER (ORDER BY EventDate) AS rn FROM CC_Bond", "window"),
        ]
    ],
    Case(
        "cast_map_unknown_column_raises",
        run=lambda: scan_db("SELECT * FROM RecordTbl", cast_map={"NotAColumn": pl.Date}),
        expect=(ValueError, "cast_map references column"),
    ),
    # Boolean has no server-side narrowing; reporting it while emitting VARCHAR would misrepresent the data
    Case(
        "cast_map_rejects_unsupported_dtype",
        run=lambda: scan_db("SELECT * FROM RecordTbl", cast_map={"RecordDate": pl.Boolean}),
        expect=(ValueError, "No SQL type mapping"),
    ),
    Case(
        "cast_map_rejects_time_zone_aware_datetime",
        run=lambda: scan_db("SELECT * FROM RecordTbl", cast_map={"RecordDate": pl.Datetime("us", "UTC")}),
        expect=(ValueError, "time-zone-aware"),
    ),
]


@pytest.mark.parametrize(("run", "expect"), params(RAISES, "run", "expect"))
def test_scan_db_raises(db, run, expect):
    db.conn.execute(TABLES["RecordTbl"])
    with pytest.raises(expect[0], match=expect[1]):
        run()


@pytest.mark.parametrize(
    ("cast_map", "dtype"),
    [
        pytest.param({"RecordDate": pl.Date}, pl.Date, id="cast_map_reports_target_dtype_in_schema"),
        pytest.param(None, pl.Datetime("us"), id="cast_map_none_is_unchanged"),
    ],
)
def test_cast_map_schema(db, cast_map, dtype):
    db.conn.execute(TABLES["RecordTbl"])
    schema = scan_db("SELECT * FROM RecordTbl", cast_map=cast_map).collect_schema()
    assert schema["RecordDate"] == dtype
    assert set(schema.names()) == {"RecordDate", "PointID", "Rate"}


def test_cast_map_hoists_mssql_order_by_out_of_derived_table():
    """A top-level MSSQL ORDER BY must end up after the cast derived table, not inside it."""
    from polars_io_tools.io_sources.sql_utils import wrap_query_with_casts

    wrapped = wrap_query_with_casts(parse_one("SELECT * FROM dbo.t ORDER BY x", dialect=MSSQL), MSSQL, ["x", "y"], {"x": pl.Date})
    sql = apply_polars_io_source_exprs(wrapped.copy(), MSSQL, None, x == date(2025, 5, 30), None, None).sql(dialect=MSSQL)
    assert_text(sql, ["AS __cpl_cast) AS __cpl_subq", "ORDER BY [x]"], omits="ORDER BY [x]) AS __cpl_cast")
    assert sql.rstrip().endswith("ORDER BY [x]")


def test_cast_map_preserves_decimal_scale():
    from polars_io_tools.io_sources.sql_utils import polars_dtype_to_sqlglot_type

    assert polars_dtype_to_sqlglot_type(pl.Decimal(10, 2)).sql(dialect=MSSQL) == "NUMERIC(10, 2)"


# Row limits: the exact rows depend on completion order, so only the count and values are checked.
@pytest.mark.parametrize(
    ("query", "scan", "op", "height", "columns", "values"),
    [
        pytest.param(
            _CC,
            lambda: cpl.by_time("EventDate", "1y"),
            lambda lf: lf.filter(_DATE_FILTER).select("CenterID", "EventDate").head(3),
            3,
            ["CenterID", "EventDate"],
            None,
            id="projection_and_row_limit",
        ),
        # no outer predicate, so n_rows reaches the source and the fan-out must honor it
        pytest.param(
            _CC,
            lambda: cpl.by_value("ISOCountryCode", [["GT"], ["BD"], ["RO"]]),
            lambda lf: lf.head(3),
            3,
            None,
            ("ISOCountryCode", {"GT", "BD", "RO"}),
            id="unordered_row_limit",
        ),
        # the guard only applies to a partitioned read
        pytest.param("SELECT * FROM CC_Bond LIMIT 5", None, lambda lf: lf, 5, None, None, id="unpartitioned_query_with_limit_is_allowed"),
    ],
)
def test_row_limit(query, scan, op, height, columns, values):
    got = op(scan_db(query, **({"partitions": scan()} if scan else {}))).collect()
    assert got.height == height
    assert columns is None or got.columns == columns
    assert values is None or set(got[values[0]]) <= values[1]


def test_generator_partitions_are_reusable():
    """A generator of slices is materialized once, so collecting twice gives the same rows."""
    lf = scan_db(_CC, partitions=_parts("GT", "BD", "RO")()).filter(_COUNTRIES)
    first = lf.collect()
    assert first.height > 0
    assert_frame_equal(lf.collect(), first, check_row_order=False)


# scan_db fans partitions across a thread pool; these tests install a 3-worker pool and, where needed, their own reader.
@pytest.fixture
def pool(monkeypatch):
    workers = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    monkeypatch.setattr(lazy_sql_reader, "_get_sql_executor", lambda: (workers, 3))
    yield workers
    workers.shutdown(wait=True)


def _install_reader(monkeypatch, read):
    monkeypatch.setitem(sys.modules, "arrow_odbc", SimpleNamespace(read_arrow_batches_from_odbc=read))


_BY_COUNTRY = [["GT"], ["BD"], ["RO"]]


@pytest.mark.usefixtures("pool")
def test_partitioned_head_does_not_push_limit_into_slices(db):
    """A slice predicate may be enforced client side, so a LIMIT in a slice query could drop rows."""
    got = scan_db(_CC, partitions=cpl.by_value("ISOCountryCode", _BY_COUNTRY)).head(2).collect()
    assert got.height == 2
    slices = [q for q in db.queries() if "IN (" in q]
    assert slices
    for q in slices:
        assert_text(q, omits=["LIMIT", " TOP "], ignore_case=True)


@pytest.mark.usefixtures("pool")
@pytest.mark.parametrize(
    ("fail_on", "match"),
    [
        pytest.param("'GT'", "partition slice", id="partition_worker_error_surfaces_with_slice_sql"),
        # completion order: an errored worker posts no done marker but must still surface with its slice SQL
        pytest.param("'RO'", "'RO'", id="partition_worker_error_surfaces"),
    ],
)
def test_partition_worker_error(db, monkeypatch, fail_on, match):
    def read(query, *args, **kwargs):
        if fail_on in query:
            raise RuntimeError("odbc boom")
        return db.odbc(query, *args, **kwargs)

    _install_reader(monkeypatch, read)
    with pytest.raises((RuntimeError, pl.exceptions.ComputeError), match=match):
        scan_db(_CC, partitions=cpl.by_value("ISOCountryCode", _BY_COUNTRY)).collect()


def _replicated(db, query):
    """The query's rows 50 times over, so a slice streams many one-row batches."""
    table = db.table(query)
    return pa.concat_tables([table] * 50) if table.num_rows else table


@pytest.mark.usefixtures("pool")
def test_streaming_is_bounded_not_whole_slice(db, monkeypatch):
    """head(1) reads a few batches; buffering whole slices would read nearly all of them."""
    produced = [0]

    class CountingReader:
        def __init__(self, table):
            self.schema = table.schema
            self._it = iter(table.to_batches(max_chunksize=1))

        def __iter__(self):
            return self

        def __next__(self):
            batch = next(self._it)
            with db.lock:
                produced[0] += 1
            return batch

    _install_reader(monkeypatch, lambda query, *a, **k: CountingReader(_replicated(db, query)))
    # enumerated slices push no predicate, so head(1) pushes n_rows and exercises early stop
    lf = scan_db(_CC, partitions=cpl.by_value("EventYear", [[2013], [2021], [2053], [2054]]))
    full = lf.collect()
    full_batches, produced[0] = produced[0], 0
    assert full.height > 9 and full_batches > 20
    assert lf.head(1).collect().height == 1
    # the shared queue holds at most k items and each of the k workers parks after its next blocked put
    assert produced[0] <= 3 * 3, f"head(1) read {produced[0]} batches"
    assert produced[0] < full_batches


@pytest.mark.usefixtures("pool")
def test_early_stop_joins_running_workers(db, monkeypatch):
    """On early stop every worker that started streaming has exited when collect returns."""
    import time

    entered, exited = [0], [0]

    class SlowReader:
        def __init__(self, table):
            self.schema = table.schema
            self._table = table

        def __iter__(self):
            with db.lock:
                entered[0] += 1
            try:
                for batch in self._table.to_batches(max_chunksize=1):
                    time.sleep(0.02)
                    yield batch
            finally:
                with db.lock:
                    exited[0] += 1

    _install_reader(monkeypatch, lambda query, *a, **k: SlowReader(_replicated(db, query)))
    assert scan_db(_CC, partitions=cpl.by_value("EventYear", [[2013], [2021], [2053]])).head(1).collect().height == 1
    assert entered[0] >= 1
    assert entered[0] == exited[0], f"{entered[0] - exited[0]} worker(s) still running"


def test_partial_submission_failure_does_not_leak_workers(db, monkeypatch):
    """When a submission fails partway through the fan-out, the workers already started are told to stop."""
    real_pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)

    class FlakyExecutor:
        n = 0

        def submit(self, *args, **kwargs):
            self.n += 1
            if self.n == 2:
                raise RuntimeError("submit boom")
            return real_pool.submit(*args, **kwargs)

    monkeypatch.setattr(lazy_sql_reader, "_get_sql_executor", lambda: (FlakyExecutor(), 3))
    try:
        with pytest.raises((RuntimeError, pl.exceptions.ComputeError), match="submit boom"):
            scan_db(_CC, partitions=cpl.by_value("ISOCountryCode", _BY_COUNTRY)).collect()
        real_pool.shutdown(wait=True)  # hangs if a worker is stranded on a full queue
    finally:
        real_pool.shutdown(wait=True)


# apply_polars_io_source_exprs: identifier quoting and MSSQL projection-alias stabilization on the parsed query
@pytest.mark.parametrize(
    ("dialect", "effective_name", "expected_sql"),
    [
        ("postgres", "group", '"group"'),
        ("oracle", "GROUP", '"GROUP"'),
        ("snowflake", "GROUP", '"GROUP"'),
    ],
)
def test_outer_metadata_columns_use_dialect_quoting(dialect, effective_name, expected_sql):
    """Known dialects quote the effective names supplied by source metadata."""
    query = parse_one("SELECT source_name FROM source_table", dialect=dialect)

    result = apply_polars_io_source_exprs(
        query,
        dialect=dialect,
        with_columns=[effective_name],
        predicate=None,
        n_rows=None,
        batch_size=None,
    )

    assert result.selects[0].sql(dialect=dialect) == expected_sql


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


@pytest.mark.parametrize("range_dates", [True, False])
def test_piot_cache_integration(db, range_dates):
    """A cached scan_db queries the database once, then answers covered filters from the cache."""
    emp_lf = scan_db("SELECT * FROM EmployeeTbl").piot.cache(order_by="id", partition_cols=["test_date", "name"])

    def apply_filters(lf, names=("Alice", "Bob"), start=date(2022, 1, 1), end=date(2022, 1, 2)):
        if range_dates:
            lf = lf.filter(pl.col.test_date.is_between(start, end))
        else:
            lf = lf.filter(pl.col.test_date.is_in(pl.date_range(start, end, closed="both", eager=True).to_list()))
        return lf.filter(pl.col.name.is_in(list(names)))

    # (names, database queries): the first read misses, repeats and subsets hit. Bartholomew has no rows in the
    # cached dates, so no partition filter excludes Alice and the cache cannot tell the read is covered.
    for names, queries in [(("Alice", "Bob"), 1), (("Alice", "Bob"), 0), (("Alice", "Bob"), 0), (("Bartholomew", "Alice"), 1)]:
        start = len(db.calls)
        res = emp_lf.pipe(apply_filters, names).collect()
        assert len(db.calls) - start == queries, names
        assert_frame_equal(res, employee_df().pipe(apply_filters, names))


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1mo", "1mo"), ("2w", "2w"), ("5d", "5d"), ("1q", "1q"), ("3y", "3y"), ("mo", "1mo"), ("month", "1mo"), ("quarter", "1q"), (7, "7d")],
)
def test_interval_str_valid(value, expected):
    from polars_io_tools.io_sources.partitions import _interval_str

    assert _interval_str(value) == expected


@pytest.mark.parametrize("value", ["", "1", "3h", "-2d", 0, -5, "1mth", True])
def test_interval_str_invalid(value):
    from polars_io_tools.io_sources.partitions import _interval_str

    with pytest.raises((ValueError, TypeError)):
        _interval_str(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    # blank is unset; malformed values raise naming the variable, at use and not at import
    [("", None), (" ", None), ("  ", None), (" 4 ", 4), ("auto", ValueError), ("8.5", ValueError), ("0", ValueError), ("-1", ValueError)],
    ids=lambda v: repr(v) if isinstance(v, str) else None,
)
def test_connection_budget_env_parse(monkeypatch, value, expected):
    monkeypatch.setenv("POLARS_IO_TOOLS_MAX_SQL_CONNECTIONS", value)
    if expected is ValueError:
        with pytest.raises(ValueError, match="POLARS_IO_TOOLS_MAX_SQL_CONNECTIONS"):
            lazy_sql_reader._resolve_sql_connection_budget()
    else:
        assert lazy_sql_reader._resolve_sql_connection_budget() == (expected or min(pl.thread_pool_size(), 8))


@pytest.mark.parametrize("dialect_query", ["duckdb", "tsql"])
@pytest.mark.parametrize(
    ("query", "unsafe", "order"),
    [
        ("SELECT * FROM t", None, False),
        ("SELECT * FROM t ORDER BY id", None, True),
        ("SELECT * FROM t LIMIT 5", "LIMIT/TOP", False),
        ("SELECT * FROM (SELECT * FROM t LIMIT 5) s", None, False),  # nested LIMIT is the user's own semantics
        ("SELECT TOP 10 * FROM t", "LIMIT/TOP", False),
        ("SELECT * FROM a UNION ALL SELECT * FROM b ORDER BY 1", None, True),
        ("SELECT * FROM a UNION ALL SELECT * FROM b LIMIT 10", "LIMIT/TOP", False),
        ("SELECT x FROM t QUALIFY ROW_NUMBER() OVER (PARTITION BY g ORDER BY id) = 1", "a window function", False),
    ],
)
def test_partition_guard_ast_detection(dialect_query, query, unsafe, order):
    if "TOP" in query and dialect_query != "tsql":
        pytest.skip("TOP is MSSQL-only syntax")
    parsed = parse_one(query, dialect=dialect_query)
    assert lazy_sql_reader._partition_unsafe_clause(parsed) == unsafe
    assert lazy_sql_reader._has_top_level_order(parsed) == order
