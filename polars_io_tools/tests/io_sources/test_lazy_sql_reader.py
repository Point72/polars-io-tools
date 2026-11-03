"""scan_db, against an in-memory DuckDB behind a fake ``arrow_odbc`` (see ``helpers.sql_backends``).

The main table holds reader queries whose result must equal what the database returns for the same query, filtered and
selected by Polars. It also runs the queries ``scan_clickhouse`` shares with ``scan_db`` (``helpers.sql_query_cases``).
The translation of filters to SQL is tested in ``test_sql_utils.py``.
"""

import concurrent.futures
import logging
import sys
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from datetime import date, datetime
from types import SimpleNamespace

import polars as pl
import pyarrow as pa
import pytest
from polars.testing import assert_frame_equal
from sqlglot import parse_one

import polars_io_tools as cpl
from polars_io_tools.io_sources import lazy_sql_reader
from polars_io_tools.tests.helpers.assertions import assert_translated_text
from polars_io_tools.tests.helpers.cases import as_params
from polars_io_tools.tests.helpers.sql_backends import (
    SQL_READERS,
    FakeDatabase,
    bond_holidays_df,
    duckdb_with_tables,
    employee_test_scores_df,
    route_readers_to,
)
from polars_io_tools.tests.helpers.sql_query_cases import (
    EVENT_NAME_HAS_DAY,
    QueryCase,
    SentSql,
    check_scan_matches_database,
    filtered,
    shared_query_params,
)

SAMPLE_B_CSV = (
    "country,label,event_date,status,knowledge_date,source_file,is_backfill,db_timestamp\n"
    "Chile,Referendum,2022-09-04T00:00:00.000,Held,2022-09-13T16:51:26.000,data/sample_b/20220913_1651Past.csv,0,2022-09-13T17:03:07.000\n"
    "Nicaragua,Nicaraguan Presidency,2021-11-07T00:00:00.000,Held,2023-04-18T16:38:30.000,data/sample_b/20230418_1638Past.csv,0,2023-04-18T16:49:12.000\n"
    "Jordan,Jordanian House of Deputies,2010-11-09T00:00:00.000,Snap,2017-04-17T02:32:24.000,data/sample_b/20170417_02.csv,1,2010-10-10T00:00:00.000\n"
    "Kiribati,Kiribati Presidency,2024-06-30T00:00:00.000,Date not confirmed,2024-04-01T09:34:51.000,data/sample_b/20240401_0934Upcoming.csv,0,2024-04-01T09:46:00.000\n"
    "Liberia,President,2017-10-10T00:00:00.000,Held,2018-04-03T00:07:41.000,data/sample_b/20180403_00.csv,1,2018-04-03T00:07:44.000\n"
    "Honduras,President,2017-11-26T00:00:00.000,Confirmed,2017-04-17T02:32:24.000,data/sample_b/20170417_02.csv,1,2017-04-17T02:32:27.000\n"
    "Moldova,Moldovan Parliament,1998-03-22T00:00:00.000,Held,2017-04-17T02:32:25.000,data/sample_b/20170417_02.csv,1,1998-02-20T00:00:00.000\n"
    "Singapore,Singapore Parliament,2020-07-10T00:00:00.000,Confirmed,2020-07-09T17:08:37.000,data/sample_b/20200709_17.csv,0,2020-07-09T17:10:22.000\n"
    "India,Indian People's Assembly,2024-04-19T00:00:00.000,Held,2024-06-17T16:48:06.000,data/sample_b/20240617_1648Past.csv,0,2024-06-17T16:59:07.000\n"
    "Guatemala,Guatemalan Presidency,2023-06-25T00:00:00.000,Held,2023-07-11T10:43:16.000,data/sample_b/20230711_1043Past.csv,0,2023-07-11T10:54:04.000\n"
)


@pytest.fixture(scope="module")
def duckdb_connection():
    import io

    conn = duckdb_with_tables(
        SampleA=bond_holidays_df(),
        ElectionCalendar=pl.read_csv(io.StringIO(SAMPLE_B_CSV), try_parse_dates=True),
        EmployeeTbl=employee_test_scores_df(),
    )
    yield conn
    conn.close()


@pytest.fixture
def fake_database(monkeypatch, duckdb_connection):
    """The fake database both readers query, inside a transaction rolled back after the test."""
    backend = FakeDatabase(duckdb_connection)
    route_readers_to(monkeypatch, backend)
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
    return SQL_READERS["scan_db"](query, **kw)


def _union(*rows: str) -> str:
    return " UNION ALL ".join(f"SELECT {r}" for r in rows)


# Tables a query row needs, created inside the test's transaction.
TABLES = {
    "A": "CREATE TABLE A AS "
    + _union("1 AS id, 10 AS shared, 100 AS aval", "2 AS id, 20 AS shared, 200 AS aval", "3 AS id, 30 AS shared, 300 AS aval"),
    "B": "CREATE TABLE B AS " + _union("10 AS shared, 1000 AS bval", "30 AS shared, 3000 AS bval", "40 AS shared, 4000 AS bval"),
    "LeftTbl": "CREATE TABLE LeftTbl AS "
    + _union(
        "1 AS ID, 100 AS JOIN_KEY, 'x' AS LEFT_ATTR, DATE '2024-01-15' AS SOME_DATE",
        "2 AS ID, 200 AS JOIN_KEY, 'y' AS LEFT_ATTR, DATE '2024-02-20' AS SOME_DATE",
        "3 AS ID, 100 AS JOIN_KEY, 'z' AS LEFT_ATTR, DATE '2024-03-10' AS SOME_DATE",
    ),
    "RightTbl": "CREATE TABLE RightTbl AS " + _union("100 AS JOIN_KEY, 'r1' AS RIGHT_ATTR", "200 AS JOIN_KEY, 'r2' AS RIGHT_ATTR"),
    "PointID": "ALTER TABLE LeftTbl ADD COLUMN PointID BIGINT; UPDATE LeftTbl SET PointID = ID * 100000",
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


def _selected(cols):
    return lambda lf: lf.select(cols)


def _after_pickle_round_trip(op=lambda lf: lf):
    import cloudpickle

    return lambda lf: cloudpickle.loads(cloudpickle.dumps(op(lf)))


_PYTHON_FUNCTION_FILTER = pl.col("score").map_elements(lambda v: v > 80, return_dtype=pl.Boolean)
_JOIN_STAR = "SELECT cr.*, br.RIGHT_ATTR FROM LeftTbl cr LEFT JOIN RightTbl br ON cr.JOIN_KEY = br.JOIN_KEY"
_JOIN_CAST = "SELECT cr.*, CAST(cr.SOME_DATE AS DATE) AS bus_date, br.RIGHT_ATTR FROM LeftTbl cr LEFT JOIN RightTbl br ON cr.JOIN_KEY = br.JOIN_KEY"
_MULTI_STAR = "SELECT o.*, p.*, c.cust_name FROM Orders o CROSS JOIN Products p LEFT JOIN Customers c ON o.customer_id = c.customer_id"
_STAR_TABLES = ("LeftTbl", "RightTbl")
_MULTI_TABLES = ("Orders", "Products", "Customers")
_EVENT_DATE_2013_TO_2016 = (pl.col("event_date") >= datetime(2013, 1, 1)) & (pl.col("event_date") < datetime(2017, 1, 1))
_GT_BD_OR_RO = pl.col("country_code").is_in(["GT", "BD", "RO"])
_RECORD_DATE_AS_DATE = {"cast_map": {"RecordDate": pl.Date}}


def _record_dates(df):
    return df.with_columns(pl.col("RecordDate").cast(pl.Date))


def _one_partition_per_country(*codes):
    return lambda: (cpl.ReadPartition(pl.col("country_code") == code, key=code) for code in codes)


_NESTED_STAR = "SELECT * FROM (SELECT CAST(event_date AS DATE) AS EventDateAsDate, * FROM SampleA) AS t"

# Reader queries and Polars steps, which must give what the database gives. The queries scan_clickhouse also runs are
# in ``helpers.sql_query_cases``.
QUERIES = [
    QueryCase(
        "implicit_cast_query",
        query="SELECT item_id, item_name, year FROM SampleA",
        steps=filtered(pl.col("year") > 2015.1),
        expected_where="year > 2015",
        check_row_order=True,
    ),
    QueryCase(
        "join",
        query="SELECT cb.item_id, cb.item_name, ec.country, ec.label FROM SampleA AS cb, ElectionCalendar AS ec",
        steps=filtered(pl.col("country").str.contains("a")),
        expected_where="ec.country LIKE '%a%'",
    ),
    *[
        QueryCase(f"str_contains-{name}", query="SELECT * FROM SampleA", steps=filtered(p))
        for name, p in [
            ("regex_alternation", pl.col("item_name").str.contains("^(Dh|Ph)")),
            ("regex_wildcard", pl.col("item_name").str.contains(".*mala")),
            ("substring", pl.col("item_name").str.contains("Da")),
            ("literal_substring", pl.col("item_name").str.contains("Da", literal=True)),
            ("regex_and_year_filter", pl.col("item_name").str.contains("^(Dh|Ph)") & (pl.col("year") > 2020)),
            ("not_substring", ~pl.col("item_name").str.contains("Da")),
        ]
    ],
    # dt.date() pushes down as a DATE cast (issue #51); Polars re-filters, so only the SQL shows it
    QueryCase(
        "dt_date_filter_pushdown-eq",
        query="SELECT item_id, item_name, event_date FROM SampleA",
        steps=filtered(pl.col("event_date").dt.date() == date(2013, 10, 24)),
        sent_sql=SentSql(contains=('CAST("event_date" AS DATE)',)),
    ),
    QueryCase(
        "dt_date_filter_pushdown-ge",
        query="SELECT item_id, item_name, event_date FROM SampleA",
        steps=filtered(pl.col("event_date").dt.date() >= date(2013, 10, 24)),
        sent_sql=SentSql(contains=('CAST("event_date" AS DATE)',)),
    ),
    # equality on a nanosecond column is an is_between whose upper bound keeps sub-microsecond digits
    QueryCase(
        "nanosecond_equality_is_not_truncated",
        query="SELECT id, ts FROM ns_tbl",
        steps=filtered(pl.col("ts") == datetime(2024, 1, 1)),
        tables=("ns_tbl",),
        expected_height=2,
    ),
    # aliases and casts in the query
    QueryCase(
        "rename_with_filter_integration",
        query="SELECT item_id, year FROM SampleA",
        steps=lambda lf: lf.select(pl.col("item_id").alias("center_id"), pl.col("year").alias("event_year")).filter(
            pl.col("event_year").is_not_null()
        ),
    ),
    QueryCase(
        "select_alias_predicate_pushdown_simple",
        query="SELECT year AS event_year, item_id AS center_id FROM SampleA",
        steps=filtered(pl.col("event_year") > 2016, after=["event_year", "center_id"]),
    ),
    QueryCase("select_alias_cast_datetime_to_date_types", query="SELECT CAST(event_date AS DATE) AS BusDate, item_id FROM SampleA"),
    QueryCase("select_original_and_cast_same_column", query="SELECT event_date, CAST(event_date AS DATE) AS BusDate, item_id FROM SampleA"),
    QueryCase(
        "nested_select_star_with_cast_alias",
        query=_NESTED_STAR,
        check_row_order=True,
        log_level=logging.DEBUG,
        expected_log="Executing SQL with pushdown: " + parse_one(_NESTED_STAR, dialect="duckdb").sql(dialect="duckdb"),
    ),
    # cloudpickle round trip (needed by Ray)
    QueryCase("scan_db_pickle_basic", query="SELECT * FROM SampleA", steps=_after_pickle_round_trip(), check_row_order=True),
    QueryCase(
        "scan_db_pickle_with_filter",
        query="SELECT * FROM SampleA",
        steps=_after_pickle_round_trip(filtered(pl.col("year") > 2020)),
        check_row_order=True,
    ),
    QueryCase(
        "scan_db_pickle_with_select", query="SELECT item_id, item_name, year FROM SampleA", steps=_after_pickle_round_trip(), check_row_order=True
    ),
    # head: Polars pushes n_rows only without a predicate; head(n).filter(p) reads n rows before filtering
    QueryCase(
        "polars_does_not_push_n_rows_with_predicate",
        query="SELECT * FROM EmployeeTbl",
        steps=lambda lf: lf.filter(pl.col("score") > 80).head(5),
        sent_sql=SentSql(contains=("WHERE",), omits=("LIMIT",)),
        check_row_order=True,
    ),
    QueryCase(
        "head_with_non_sql_filter_returns_correct_count",
        query="SELECT * FROM EmployeeTbl",
        steps=lambda lf: lf.filter(_PYTHON_FUNCTION_FILTER).head(10),
        check_row_order=True,
    ),
    QueryCase(
        "head_then_filter-sql", query="SELECT * FROM EmployeeTbl", steps=lambda lf: lf.head(10).filter(pl.col("score") > 80), check_row_order=True
    ),
    QueryCase(
        "head_then_filter-udf", query="SELECT * FROM EmployeeTbl", steps=lambda lf: lf.head(10).filter(_PYTHON_FUNCTION_FILTER), check_row_order=True
    ),
    QueryCase(
        "workaround_collect_then_head",
        query="SELECT * FROM EmployeeTbl",
        steps=lambda lf: lf.filter(_PYTHON_FUNCTION_FILTER).collect().head(10).lazy(),
        check_row_order=True,
    ),
    # joins: rebuilt projections keep the table qualifier, or the query is wrapped as a subquery
    QueryCase(
        "with_columns_preserves_qualification_on_join",
        query="SELECT a.shared, a.aval, b.bval FROM A AS a LEFT JOIN B AS b ON a.shared = b.shared",
        steps=_selected(["shared", "aval"]),
        tables=("A", "B"),
        sent_sql=SentSql(contains=(("SELECT a.shared, a.aval", 'SELECT "a"."shared", "a"."aval"', "SELECT [a].[shared], [a].[aval]"),)),
    ),
    QueryCase(
        "select_shared_column_is_qualified",
        query=_JOIN_STAR,
        steps=_selected(["JOIN_KEY", "RIGHT_ATTR"]),
        tables=_STAR_TABLES,
        sent_sql=SentSql(contains=("__cpl_subq",)),
    ),
    QueryCase(
        "select_non_shared_column_is_qualified",
        query=_JOIN_STAR,
        steps=_selected(["ID", "LEFT_ATTR"]),
        tables=_STAR_TABLES,
        sent_sql=SentSql(contains=("__cpl_subq",)),
    ),
    QueryCase(
        "select_all_star_expanded_columns",
        query=_JOIN_STAR,
        steps=_selected(["ID", "JOIN_KEY", "LEFT_ATTR", "SOME_DATE"]),
        tables=_STAR_TABLES,
        sent_sql=SentSql(contains=("__cpl_subq",)),
    ),
    QueryCase(
        "predicate_pushdown_works_with_star_join",
        query=_JOIN_STAR,
        steps=filtered(pl.col("LEFT_ATTR") == "x"),
        tables=_STAR_TABLES,
        sent_sql=SentSql(contains=("WHERE",)),
    ),
    # columns only on the cr side resolve without a qualifier
    QueryCase(
        "predicate_with_multiple_filters",
        query=_JOIN_STAR,
        steps=filtered((pl.col("SOME_DATE") == date(2024, 1, 15)) & (pl.col("PointID") == 100000)),
        tables=(*_STAR_TABLES, "PointID"),
        sent_sql=SentSql(contains=("WHERE",)),
        expected_height=1,
    ),
    QueryCase(
        "combined_select_qualified_and_predicate_unqualified",
        query=_JOIN_STAR,
        steps=filtered(pl.col("LEFT_ATTR") == "x", after=["JOIN_KEY", "RIGHT_ATTR"]),
        tables=_STAR_TABLES,
        sent_sql=SentSql(contains=("__cpl_subq", "WHERE")),
    ),
    QueryCase(
        "alias_predicate_rewrites_to_underlying_expression",
        query=_JOIN_CAST,
        steps=filtered(pl.col("bus_date") >= date(2024, 2, 1)),
        tables=_STAR_TABLES,
        sent_sql=SentSql(contains=("WHERE", "CAST")),
    ),
    QueryCase(
        "select_alias_column_alongside_star_expanded",
        query=_JOIN_CAST,
        steps=_selected(["bus_date", "JOIN_KEY"]),
        tables=_STAR_TABLES,
        sent_sql=SentSql(contains=("__cpl_subq", "CAST", "bus_date")),
    ),
    QueryCase(
        "bare_star_join_does_not_qualify",
        query="SELECT * FROM StarA a LEFT JOIN StarB b ON a.joinkey = b.bkey",
        steps=_selected(["aid", "bval"]),
        tables=("StarA", "StarB"),
        sent_sql=SentSql(omits=("a.aid", '"a"."aid"')),
    ),
    # a redundant full-schema select leaves the SQL unchanged
    QueryCase(
        "explicit_columns_with_join_uses_subquery",
        query="SELECT a.shared, b.bval FROM JoinA a LEFT JOIN JoinB b ON a.shared = b.shared",
        steps=_selected(["shared", "bval"]),
        tables=("JoinA", "JoinB"),
        sent_sql=SentSql(contains=(("a.shared", '"a"."shared"', "[a].[shared]"), ("b.bval", '"b"."bval"', "[b].[bval]")), omits=("__cpl_subq",)),
    ),
    QueryCase(
        "no_expansion_when_no_joins",
        query="SELECT * FROM LeftTbl",
        steps=_selected(["ID", "LEFT_ATTR"]),
        tables=_STAR_TABLES,
        sent_sql=SentSql(omits=("cr.ID", '"cr"."ID"')),
    ),
    QueryCase("no_modification_without_pushdown", query=_JOIN_STAR, tables=_STAR_TABLES, sent_sql=SentSql(contains=(("cr.*", '"cr".*'),))),
    QueryCase(
        "multi_star_select_shared_uses_subquery",
        query=_MULTI_STAR,
        steps=_selected(["customer_id", "cust_name"]),
        tables=_MULTI_TABLES,
        sent_sql=SentSql(contains=("__cpl_subq",)),
    ),
    QueryCase(
        "multi_star_select_non_shared_uses_subquery",
        query=_MULTI_STAR,
        steps=_selected(["order_id", "product_name"]),
        tables=_MULTI_TABLES,
        sent_sql=SentSql(contains=("__cpl_subq",)),
    ),
    QueryCase(
        "multi_star_filter_with_subquery",
        query=_MULTI_STAR,
        steps=filtered(pl.col("cust_name") == "Alice", after=["order_id", "cust_name", "product_name"]),
        tables=_MULTI_TABLES,
        sent_sql=SentSql(contains=("__cpl_subq", "WHERE")),
        expected_height=6,
    ),
    QueryCase("multi_star_no_subquery_without_pushdown", query=_MULTI_STAR, tables=_MULTI_TABLES, sent_sql=SentSql(omits=("__cpl_subq",))),
    # comma-separated tables are implicit cross joins
    QueryCase(
        "comma_join_with_qualified_star_uses_subquery",
        query="SELECT a.*, b.bval FROM CommaA a, CommaB b WHERE a.shared_key = b.shared_key",
        steps=_selected(["shared_key", "bval"]),
        tables=("CommaA", "CommaB"),
        sent_sql=SentSql(contains=("__cpl_subq",)),
    ),
    # subquery wrapping: ORDER BY without TOP/OFFSET would move out (MSSQL only); others stay inside
    QueryCase(
        "order_by_stays_in_inner_for_non_mssql",
        query="SELECT region, amount FROM Sales ORDER BY amount DESC",
        steps=_selected(["region", "amount"]),
        tables=("Sales",),
        sent_sql=SentSql(contains=("ORDER BY",), omits=("__cpl_subq",)),
        check_row_order=True,
    ),
    QueryCase(
        "order_by_with_limit_stays_in_inner",
        query="SELECT region, amount FROM Sales ORDER BY amount DESC LIMIT 2",
        steps=_selected(["region"]),
        tables=("Sales",),
        sent_sql=SentSql(contains=("__cpl_subq",), inner_contains=("ORDER BY",)),
        check_row_order=True,
    ),
    QueryCase(
        "no_wrapping_without_pushdown",
        query="SELECT region, amount FROM Sales ORDER BY amount",
        tables=("Sales",),
        sent_sql=SentSql(contains=("ORDER BY",), omits=("__cpl_subq",)),
        check_row_order=True,
    ),
    QueryCase(
        "alias_predicate_works_without_rewriting",
        query="SELECT region, CAST(sale_date AS DATE) AS sale_date FROM Sales",
        steps=filtered(pl.col("sale_date") >= date(2024, 2, 1)),
        tables=("Sales",),
        sent_sql=SentSql(contains=("__cpl_subq",), outer_contains=("WHERE",), outer_omits=("CAST",)),
    ),
    # a predicate on a UNION filters every branch
    QueryCase(
        "union_predicate_filters_all_branches",
        query="SELECT region, amount FROM SalesQ1 UNION ALL SELECT region, amount FROM SalesQ2",
        steps=filtered(pl.col("region") == "East"),
        tables=("SalesQ1", "SalesQ2"),
        sent_sql=SentSql(contains=("__cpl_subq",)),
        expected_height=2,
    ),
    QueryCase(
        "union_column_selection",
        query="SELECT id, name, val FROM UnionA UNION ALL SELECT id, name, val FROM UnionB",
        steps=_selected(["id", "name"]),
        tables=("UnionA", "UnionB"),
        sent_sql=SentSql(contains=("__cpl_subq",)),
    ),
    QueryCase(
        "group_by_with_predicate_pushdown",
        query="SELECT region, SUM(amount) AS total FROM Sales GROUP BY region",
        steps=filtered(pl.col("total") >= 250),
        tables=("Sales",),
        sent_sql=SentSql(contains=("__cpl_subq",), inner_contains=("GROUP BY",)),
    ),
    QueryCase(
        "cte_query_with_column_selection",
        query="WITH cte AS (SELECT region, amount FROM Sales) SELECT region, amount FROM cte",
        steps=_selected(["region"]),
        tables=("Sales",),
        sent_sql=SentSql(contains=("__cpl_subq",)),
    ),
    QueryCase(
        "order_by_with_offset_stays_in_inner",
        query="SELECT region, amount FROM Sales ORDER BY amount LIMIT 2 OFFSET 1",
        steps=_selected(["region"]),
        tables=("Sales",),
        sent_sql=SentSql(contains=("__cpl_subq",), inner_contains=("ORDER BY",)),
        check_row_order=True,
    ),
    QueryCase(
        "order_by_qualified_stays_in_inner_for_non_mssql",
        query="SELECT a.id, a.event_date FROM OrderQualA a ORDER BY a.event_date",
        steps=_selected(["id", "event_date"]),
        tables=("OrderQualA",),
        sent_sql=SentSql(contains=("ORDER BY",), omits=("__cpl_subq",)),
        check_row_order=True,
    ),
    QueryCase(
        "predicate_only_pushdown",
        query="SELECT region, amount, sale_date FROM Sales",
        steps=filtered(pl.col("region") == "East"),
        tables=("Sales",),
        sent_sql=SentSql(contains=("__cpl_subq",), outer_contains=("WHERE",)),
    ),
    QueryCase(
        "nested_subquery_wrapping",
        query="SELECT region, amount FROM (SELECT region, amount FROM Sales) AS inner_q",
        steps=_selected(["region"]),
        tables=("Sales",),
        sent_sql=SentSql(contains=("__cpl_subq",)),
    ),
    # metadata-derived reserved and special-character names stay selectable
    QueryCase(
        "column_selection_quotes_metadata_names",
        query="SELECT * FROM QuotedMetadataColumns",
        steps=_selected(["group", "value with space"]),
        tables=("QuotedMetadataColumns",),
        scan_kwargs={"krb5": False},
    ),
    # cast_map narrows a column server side
    QueryCase(
        "cast_map_narrows_column_server_side",
        query="SELECT * FROM RecordTbl",
        steps=lambda lf: lf.sort("PointID"),
        tables=("RecordTbl",),
        scan_kwargs=_RECORD_DATE_AS_DATE,
        native_transform=_record_dates,
        check_row_order=True,
    ),
    QueryCase(
        "cast_map_pushes_date_predicate_to_sql",
        query="SELECT * FROM RecordTbl",
        steps=lambda lf: lf.filter(pl.col("RecordDate") == date(2025, 5, 30)).sort("PointID"),
        tables=("RecordTbl",),
        scan_kwargs=_RECORD_DATE_AS_DATE,
        native_transform=_record_dates,
        check_row_order=True,
        sent_sql=SentSql(contains=("CAST", "DATE", "RecordDate", "WHERE")),
        expected_height=2,
    ),
    # the inner cast layer lists every column; the outer select prunes
    QueryCase(
        "cast_map_projection_pushdown_still_prunes",
        query="SELECT * FROM RecordTbl",
        steps=lambda lf: lf.select(["RecordDate", "PointID"]).sort("PointID"),
        tables=("RecordTbl",),
        scan_kwargs=_RECORD_DATE_AS_DATE,
        native_transform=_record_dates,
        check_row_order=True,
        sent_sql=SentSql(startswith='SELECT "RECORDDATE", "POINTID" FROM'),
    ),
    # partitioned reads return the unpartitioned rows
    *[
        QueryCase(
            f"by_time_matches_unpartitioned-{every}",
            query="SELECT * FROM SampleA",
            steps=filtered(_EVENT_DATE_2013_TO_2016),
            scan_kwargs={"partitions": lambda every=every: cpl.by_time("event_date", every)},
        )
        for every in ["1y", "1mo", "6mo", 90, "1q"]
    ],
    # [2013-01-01, 2017-01-01) split yearly is four windows (exclusive upper bound)
    QueryCase(
        "by_time_partition_count",
        query="SELECT * FROM SampleA",
        steps=filtered(_EVENT_DATE_2013_TO_2016),
        scan_kwargs={"partitions": lambda: cpl.by_time("event_date", "1y")},
        log_level=logging.DEBUG,
        expected_log="running 4 partition(s)",
    ),
    QueryCase(
        "by_value_derived_from_predicate",
        query="SELECT * FROM SampleA",
        steps=filtered(_GT_BD_OR_RO),
        scan_kwargs={"partitions": lambda: cpl.by_value("country_code")},
    ),
    QueryCase(
        "by_value_explicit_groups",
        query="SELECT * FROM SampleA",
        steps=filtered(_GT_BD_OR_RO),
        scan_kwargs={"partitions": lambda: cpl.by_value("country_code", [["GT", "BD"], "RO"])},
    ),
    QueryCase(
        "by_range_numeric",
        query="SELECT * FROM SampleA",
        steps=filtered((pl.col("year") >= 2013) & (pl.col("year") < 2055)),
        scan_kwargs={"partitions": lambda: cpl.by_range("year", every=10)},
    ),
    # every SampleA row has file_type 'G'
    QueryCase(
        "explicit_read_partition_list",
        query="SELECT * FROM SampleA",
        scan_kwargs={"partitions": lambda: [cpl.ReadPartition(pl.col("file_type") == "G", key="G")]},
    ),
    # a one-shot generator of slices, and a projection without the partition column the client-side filter needs
    QueryCase(
        "generator_partitions_retain_columns_under_projection",
        query="SELECT * FROM SampleA",
        steps=filtered(_GT_BD_OR_RO, after="item_code"),
        scan_kwargs={"partitions": _one_partition_per_country("GT", "BD", "RO")},
    ),
    QueryCase(
        "max_concurrency_matches_unpartitioned",
        query="SELECT * FROM SampleA",
        steps=filtered(_EVENT_DATE_2013_TO_2016),
        scan_kwargs={"partitions": lambda: cpl.by_time("event_date", "1mo"), "max_concurrency": 1},
    ),
    # a slice predicate SQL cannot express is applied client side
    QueryCase(
        "untranslatable_predicate_filters_client_side",
        query="SELECT * FROM SampleA",
        scan_kwargs={
            "partitions": lambda: [cpl.ReadPartition(pl.col("event_name").map_elements(lambda v: "Day" in v, return_dtype=pl.Boolean), key=0)]
        },
        native_transform=lambda df: df.filter(EVENT_NAME_HAS_DAY),
    ),
    QueryCase(
        "unbounded_predicate_falls_back_to_single_query",
        query="SELECT * FROM SampleA",
        scan_kwargs={"partitions": lambda: cpl.by_time("event_date")},
        log_level=logging.DEBUG,
        expected_log="running 1 partition(s)",
    ),
    # a top-level ORDER BY only loses global order under partitioning: warn, don't raise
    QueryCase(
        "partitioned_query_with_order_by_warns_not_raises",
        query="SELECT * FROM SampleA ORDER BY event_date DESC",
        steps=filtered(_EVENT_DATE_2013_TO_2016),
        scan_kwargs={"partitions": lambda: cpl.by_time("event_date", "1y")},
        expected_log="ORDER BY",
    ),
]


@pytest.fixture
def record_tbl(fake_database):
    """``fake_database`` holding RecordTbl inside the test's transaction."""
    fake_database.conn.execute(TABLES["RecordTbl"])
    return fake_database


@pytest.mark.parametrize(("entry", "case"), [*(pytest.param("scan_db", case, id=case.id) for case in QUERIES), *shared_query_params("scan_db")])
def test_scan_matches_native(fake_database, caplog, entry, case):
    for name in case.tables:
        fake_database.conn.execute(TABLES[name])
    check_scan_matches_database(fake_database, caplog, entry, case)


def test_where_check_rejects_a_column_only_the_where_read_returns(fake_database, caplog, monkeypatch):
    """Without ``expected_where_columns`` the WHERE read is compared as read, so an extra column in it fails."""
    case = QueryCase(
        "where",
        query="SELECT item_id, item_name, year FROM SampleA",
        steps=filtered(pl.col("year") > 2015),
        expected_where="year > 2015",
    )
    scan_db = SQL_READERS["scan_db"]
    monkeypatch.setitem(
        SQL_READERS, "scan_db", lambda query, **kw: scan_db(query, **kw).with_columns(extra=pl.lit(1)) if " WHERE " in query else scan_db(query, **kw)
    )
    with pytest.raises(AssertionError, match="column"):
        check_scan_matches_database(fake_database, caplog, "scan_db", case)


def _identity_schema(schema):
    return schema


_PA_SCHEMA = pa.schema([("item_id", pa.int64()), ("item_name", pa.string())])
_CENTER_ID_AND_CENTRE = "SELECT item_id, item_name FROM SampleA"


@dataclass(frozen=True)
class OdbcKwargsCase:
    """A query and arrow-odbc keyword arguments, which both the schema query and the data query must receive."""

    id: str
    _: KW_ONLY
    query: str
    kwargs: dict


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            OdbcKwargsCase("query_timeout_sec_passed_to_schema_query", query="SELECT * FROM SampleA", kwargs={"query_timeout_sec": 30}),
            OdbcKwargsCase(
                "multiple_kwargs_passed_to_both_queries",
                query="SELECT * FROM SampleA",
                kwargs={"query_timeout_sec": 30, "login_timeout_sec": 10, "max_text_size": 4096},
            ),
            OdbcKwargsCase(
                "schema_kwarg_passed_to_both_queries", query=_CENTER_ID_AND_CENTRE, kwargs={"query_timeout_sec": 30, "schema": _PA_SCHEMA}
            ),
            OdbcKwargsCase(
                "map_schema_kwarg_passed_to_both_queries",
                query=_CENTER_ID_AND_CENTRE,
                kwargs={"query_timeout_sec": 30, "map_schema": _identity_schema},
            ),
            OdbcKwargsCase(
                "all_kwargs_passed_to_both_queries",
                query=_CENTER_ID_AND_CENTRE,
                kwargs={"query_timeout_sec": 30, "login_timeout_sec": 10, "schema": _PA_SCHEMA, "map_schema": _identity_schema},
            ),
        ]
    ),
)
def test_arrow_odbc_kwargs_reach_schema_and_data_queries(fake_database, case: OdbcKwargsCase):
    lf = scan_db(case.query, krb5=False, **case.kwargs)
    lf.collect()
    for data in (False, True):
        (call,) = [c for c in fake_database.calls if ("LIMIT 0" not in c["query"]) == data]
        assert {k: call["kwargs"].get(k) for k in case.kwargs} == case.kwargs, call["kwargs"]


_ALL_BOND_HOLIDAYS = "SELECT * FROM SampleA"
_READER_ERRORS = (ValueError, pl.exceptions.ComputeError)


@dataclass(frozen=True)
class ReaderErrorCase:
    """A reader call that must raise, and the error it raises."""

    id: str
    _: KW_ONLY
    run: Callable[[], object]
    # the exception type, or a tuple of types
    error: type[Exception] | tuple[type[Exception], ...]
    # a pattern the error message must match
    match: str


RAISES = [
    ReaderErrorCase(
        "max_partitions_guardrail_raises",
        run=lambda: (
            scan_db(_ALL_BOND_HOLIDAYS, partitions=cpl.by_time("event_date", "1d"), max_partitions=5).filter(_EVENT_DATE_2013_TO_2016).collect()
        ),
        error=pl.exceptions.ComputeError,
        match="exceeds max_partitions",
    ),
    ReaderErrorCase(
        "unknown_partition_column_raises",
        run=lambda: scan_db(_ALL_BOND_HOLIDAYS, partitions=cpl.by_time("not_a_column")),
        error=ValueError,
        match="not_a_column",
    ),
    ReaderErrorCase(
        "explicit_partitions_over_max_raises",
        run=lambda: scan_db(_ALL_BOND_HOLIDAYS, partitions=list(_one_partition_per_country("GT", "BD", "RO")()), max_partitions=2).collect(),
        error=pl.exceptions.ComputeError,
        match="exceeds max_partitions",
    ),
    *[
        ReaderErrorCase(
            f"partitioned_query_with_unsafe_clause_raises-{match}",
            run=lambda q=q: scan_db(q, partitions=cpl.by_time("event_date", "1y")).filter(_EVENT_DATE_2013_TO_2016).collect(),
            error=_READER_ERRORS,
            match=match,
        )
        for q, match in [
            ("SELECT * FROM SampleA ORDER BY event_date LIMIT 10", "LIMIT"),
            ("SELECT * FROM SampleA OFFSET 5 ROWS", "OFFSET"),
            ("SELECT *, ROW_NUMBER() OVER (ORDER BY event_date) AS rn FROM SampleA", "window"),
        ]
    ],
    ReaderErrorCase(
        "cast_map_unknown_column_raises",
        run=lambda: scan_db("SELECT * FROM RecordTbl", cast_map={"NotAColumn": pl.Date}),
        error=ValueError,
        match="cast_map references column",
    ),
    # Boolean has no server-side narrowing; reporting it while emitting VARCHAR would misrepresent the data
    ReaderErrorCase(
        "cast_map_rejects_unsupported_dtype",
        run=lambda: scan_db("SELECT * FROM RecordTbl", cast_map={"RecordDate": pl.Boolean}),
        error=ValueError,
        match="No SQL type mapping",
    ),
    ReaderErrorCase(
        "cast_map_rejects_time_zone_aware_datetime",
        run=lambda: scan_db("SELECT * FROM RecordTbl", cast_map={"RecordDate": pl.Datetime("us", "UTC")}),
        error=ValueError,
        match="time-zone-aware",
    ),
]


@pytest.mark.parametrize("case", as_params(RAISES))
def test_scan_db_raises(record_tbl, case: ReaderErrorCase):
    with pytest.raises(case.error, match=case.match):
        case.run()


@dataclass(frozen=True)
class CastMapSchemaCase:
    """A ``cast_map``, and the dtype the schema reports for ``RecordDate``."""

    id: str
    _: KW_ONLY
    cast_map: dict | None
    dtype: pl.DataType | type[pl.DataType]


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            CastMapSchemaCase("cast_map_reports_target_dtype_in_schema", cast_map={"RecordDate": pl.Date}, dtype=pl.Date),
            CastMapSchemaCase("cast_map_none_is_unchanged", cast_map=None, dtype=pl.Datetime("us")),
        ]
    ),
)
def test_cast_map_schema(record_tbl, case: CastMapSchemaCase):
    schema = scan_db("SELECT * FROM RecordTbl", cast_map=case.cast_map).collect_schema()
    assert schema["RecordDate"] == case.dtype
    assert set(schema.names()) == {"RecordDate", "PointID", "Rate"}


# Row limits: the exact rows depend on completion order, so only the count and values are checked.
@dataclass(frozen=True)
class RowLimitCase:
    """A query, its partitions and a Polars query on the result, and what comes back."""

    id: str
    _: KW_ONLY
    query: str
    # makes the partitions; None: the read is not partitioned
    scan: Callable | None
    op: Callable[[pl.LazyFrame], pl.LazyFrame]
    height: int
    # None: not checked
    columns: list[str] | None
    # (column, the values it may hold); None: not checked
    values: tuple | None


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            RowLimitCase(
                "projection_and_row_limit",
                query=_ALL_BOND_HOLIDAYS,
                scan=lambda: cpl.by_time("event_date", "1y"),
                op=lambda lf: lf.filter(_EVENT_DATE_2013_TO_2016).select("item_id", "event_date").head(3),
                height=3,
                columns=["item_id", "event_date"],
                values=None,
            ),
            # no outer predicate, so n_rows reaches the source and the fan-out must honor it
            RowLimitCase(
                "unordered_row_limit",
                query=_ALL_BOND_HOLIDAYS,
                scan=lambda: cpl.by_value("country_code", [["GT"], ["BD"], ["RO"]]),
                op=lambda lf: lf.head(3),
                height=3,
                columns=None,
                values=("country_code", {"GT", "BD", "RO"}),
            ),
            # the guard only applies to a partitioned read
            RowLimitCase(
                "unpartitioned_query_with_limit_is_allowed",
                query="SELECT * FROM SampleA LIMIT 5",
                scan=None,
                op=lambda lf: lf,
                height=5,
                columns=None,
                values=None,
            ),
        ]
    ),
)
@pytest.mark.usefixtures("fake_database")
def test_row_limit(case: RowLimitCase):
    got = case.op(scan_db(case.query, **({"partitions": case.scan()} if case.scan else {}))).collect()
    assert got.height == case.height
    assert case.columns is None or got.columns == case.columns
    assert case.values is None or set(got[case.values[0]]) <= case.values[1]


@pytest.mark.usefixtures("fake_database")
def test_generator_partitions_are_reusable():
    """A generator of slices is materialized once, so collecting twice gives the same rows."""
    lf = scan_db(_ALL_BOND_HOLIDAYS, partitions=_one_partition_per_country("GT", "BD", "RO")()).filter(_GT_BD_OR_RO)
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


_ONE_COUNTRY_PER_PARTITION = [["GT"], ["BD"], ["RO"]]


@pytest.mark.usefixtures("pool")
def test_partitioned_head_does_not_push_limit_into_slices(fake_database):
    """A slice predicate may be enforced client side, so a LIMIT in a slice query could drop rows."""
    got = scan_db(_ALL_BOND_HOLIDAYS, partitions=cpl.by_value("country_code", _ONE_COUNTRY_PER_PARTITION)).head(2).collect()
    assert got.height == 2
    slices = [q for q in fake_database.queries() if "IN (" in q]
    assert slices
    for q in slices:
        assert_translated_text(q, omits=["LIMIT", " TOP "], ignore_case=True)


@dataclass(frozen=True)
class PartitionWorkerErrorCase:
    """A partition read that fails, and the error message pattern."""

    id: str
    _: KW_ONLY
    # a read whose query holds this text fails
    fail_on: str
    match: str


@pytest.mark.usefixtures("pool")
@pytest.mark.parametrize(
    "case",
    as_params(
        [
            PartitionWorkerErrorCase("partition_worker_error_surfaces_with_slice_sql", fail_on="'GT'", match="partition slice"),
            # completion order: an errored worker posts no done marker but must still surface with its slice SQL
            PartitionWorkerErrorCase("partition_worker_error_surfaces", fail_on="'RO'", match="'RO'"),
        ]
    ),
)
def test_partition_worker_error(fake_database, monkeypatch, case: PartitionWorkerErrorCase):
    def read(query, *args, **kwargs):
        if case.fail_on in query:
            raise RuntimeError("odbc boom")
        return fake_database.read_through_odbc(query, *args, **kwargs)

    _install_reader(monkeypatch, read)
    with pytest.raises((RuntimeError, pl.exceptions.ComputeError), match=case.match):
        scan_db(_ALL_BOND_HOLIDAYS, partitions=cpl.by_value("country_code", _ONE_COUNTRY_PER_PARTITION)).collect()


def _replicated(fake_database, query):
    """The query's rows 50 times over, so a slice streams many one-row batches."""
    table = fake_database.table(query)
    return pa.concat_tables([table] * 50) if table.num_rows else table


@pytest.mark.usefixtures("pool")
def test_streaming_is_bounded_not_whole_slice(fake_database, monkeypatch):
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
            with fake_database.lock:
                produced[0] += 1
            return batch

    _install_reader(monkeypatch, lambda query, *a, **k: CountingReader(_replicated(fake_database, query)))
    # enumerated slices push no predicate, so head(1) pushes n_rows and exercises early stop
    lf = scan_db(_ALL_BOND_HOLIDAYS, partitions=cpl.by_value("year", [[2013], [2021], [2053], [2054]]))
    full = lf.collect()
    full_batches, produced[0] = produced[0], 0
    assert full.height > 9 and full_batches > 20
    assert lf.head(1).collect().height == 1
    # the shared queue holds at most k items and each of the k workers parks after its next blocked put
    assert produced[0] <= 3 * 3, f"head(1) read {produced[0]} batches"
    assert produced[0] < full_batches


@pytest.mark.usefixtures("pool")
def test_early_stop_joins_running_workers(fake_database, monkeypatch):
    """On early stop every worker that started streaming has exited when collect returns."""
    import time

    entered, exited = [0], [0]

    class SlowReader:
        def __init__(self, table):
            self.schema = table.schema
            self._table = table

        def __iter__(self):
            with fake_database.lock:
                entered[0] += 1
            try:
                for batch in self._table.to_batches(max_chunksize=1):
                    time.sleep(0.02)
                    yield batch
            finally:
                with fake_database.lock:
                    exited[0] += 1

    _install_reader(monkeypatch, lambda query, *a, **k: SlowReader(_replicated(fake_database, query)))
    assert scan_db(_ALL_BOND_HOLIDAYS, partitions=cpl.by_value("year", [[2013], [2021], [2053]])).head(1).collect().height == 1
    assert entered[0] >= 1
    assert entered[0] == exited[0], f"{entered[0] - exited[0]} worker(s) still running"


def test_partial_submission_failure_does_not_leak_workers(fake_database, monkeypatch):
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
            scan_db(_ALL_BOND_HOLIDAYS, partitions=cpl.by_value("country_code", _ONE_COUNTRY_PER_PARTITION)).collect()
        real_pool.shutdown(wait=True)  # hangs if a worker is stranded on a full queue
    finally:
        real_pool.shutdown(wait=True)


@pytest.mark.parametrize("range_dates", [True, False], ids=["is_between", "is_in_date_list"])
def test_piot_cache_integration(fake_database, range_dates):
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
        start = len(fake_database.calls)
        res = emp_lf.pipe(apply_filters, names).collect()
        assert len(fake_database.calls) - start == queries, names
        assert_frame_equal(res, employee_test_scores_df().pipe(apply_filters, names))


@dataclass(frozen=True)
class ConnectionBudgetCase:
    """A ``POLARS_IO_TOOLS_MAX_SQL_CONNECTIONS`` value, and the connection budget read from it."""

    id: str
    _: KW_ONLY
    value: str
    # None: the default budget; ValueError: the value is rejected
    expected: int | None | type[ValueError]


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            # blank is unset; malformed values raise naming the variable, at use and not at import
            ConnectionBudgetCase("''-None", value="", expected=None),
            ConnectionBudgetCase("' '-None", value=" ", expected=None),
            ConnectionBudgetCase("'  '-None", value="  ", expected=None),
            ConnectionBudgetCase("' 4 '-4", value=" 4 ", expected=4),
            ConnectionBudgetCase("'auto'-ValueError", value="auto", expected=ValueError),
            ConnectionBudgetCase("'8.5'-ValueError", value="8.5", expected=ValueError),
            ConnectionBudgetCase("'0'-ValueError", value="0", expected=ValueError),
            ConnectionBudgetCase("'-1'-ValueError", value="-1", expected=ValueError),
        ]
    ),
)
def test_connection_budget_env_parse(monkeypatch, case: ConnectionBudgetCase):
    monkeypatch.setenv("POLARS_IO_TOOLS_MAX_SQL_CONNECTIONS", case.value)
    if case.expected is ValueError:
        with pytest.raises(ValueError, match="POLARS_IO_TOOLS_MAX_SQL_CONNECTIONS"):
            lazy_sql_reader._resolve_sql_connection_budget()
    else:
        assert lazy_sql_reader._resolve_sql_connection_budget() == (case.expected or min(pl.thread_pool_size(), 8))


@dataclass(frozen=True)
class PartitionGuardCase:
    """A query, the clause that makes it unsafe to partition, and whether it has a top-level ORDER BY."""

    id: str
    _: KW_ONLY
    query: str
    unsafe: str | None
    order: bool


@pytest.mark.parametrize("dialect_query", ["duckdb", "tsql"])
@pytest.mark.parametrize(
    "case",
    as_params(
        [
            PartitionGuardCase("plain_select", query="SELECT * FROM t", unsafe=None, order=False),
            PartitionGuardCase("order_by", query="SELECT * FROM t ORDER BY id", unsafe=None, order=True),
            PartitionGuardCase("limit", query="SELECT * FROM t LIMIT 5", unsafe="LIMIT/TOP", order=False),
            PartitionGuardCase("limit_in_subquery", query="SELECT * FROM (SELECT * FROM t LIMIT 5) s", unsafe=None, order=False),
            PartitionGuardCase("top", query="SELECT TOP 10 * FROM t", unsafe="LIMIT/TOP", order=False),
            PartitionGuardCase(
                "union_order_by",
                query="SELECT * FROM a UNION ALL SELECT * FROM b ORDER BY 1",
                unsafe=None,
                order=True,
            ),
            PartitionGuardCase(
                "union_limit",
                query="SELECT * FROM a UNION ALL SELECT * FROM b LIMIT 10",
                unsafe="LIMIT/TOP",
                order=False,
            ),
            PartitionGuardCase(
                "qualify_window_function",
                query="SELECT x FROM t QUALIFY ROW_NUMBER() OVER (PARTITION BY g ORDER BY id) = 1",
                unsafe="a window function",
                order=False,
            ),
        ]
    ),
)
def test_partition_guard_ast_detection(dialect_query, case: PartitionGuardCase):
    if "TOP" in case.query and dialect_query != "tsql":
        pytest.skip("TOP is MSSQL-only syntax")
    parsed = parse_one(case.query, dialect=dialect_query)
    assert lazy_sql_reader._partition_unsafe_clause(parsed) == case.unsafe
    assert lazy_sql_reader._has_top_level_order(parsed) == case.order
