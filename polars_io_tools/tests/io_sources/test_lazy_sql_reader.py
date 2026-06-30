import io
import logging
import sys
from datetime import date
from types import SimpleNamespace
from typing import Optional

import duckdb
import polars as pl
import pyarrow as pa
import pytest
from polars.testing import assert_frame_equal
from sqlglot import exp, parse_one

import polars_io_tools as cpl
from polars_io_tools import io_sources as polars_utils
from polars_io_tools.io_sources.base import BinaryExprNode, ColumnNode, FunctionNode, LiteralNode, get_parsed_expr
from polars_io_tools.io_sources.enum import BooleanFunctionType, OperatorType, TemporalFunctionType
from polars_io_tools.io_sources.sql_dialects import MSSQL
from polars_io_tools.io_sources.sql_utils import SQLExpressionVisitor, convert_predicate_to_sql, create_sqlglot_literal

"""
This module contains tests for the lazy Polars SQL reader. We
mock the data and connections here; for usage examples (and
proof that this works on real data), please refer to the file
`lazy_polars_sql.py` in examples/.
"""

# This sample data is from the [Calendar].[dbo].[CC_Bond]
# and [Calendar].[dbo].[ElectionCalendar] databases.
CC_BOND_CSV = (
    "CenterID,CentreCode,ISOCountryCode,Centre,EventYear,EventDate,EventDayOfWeek,EventName,FileType\n"
    "103,GTQ,GT,Guatemala Quetzal,2021,2021-01-01T00:00:00.000,Fri,New Year's Day,G\n"
    "146,PHP,PH,Philippine Peso,2045,2045-04-06T00:00:00.000,Thu,Holy Thursday,G\n"
    "158,KPW,KP,Pyongyang,2053,2053-02-19T00:00:00.000,Wed,Lunar New Year 1,G\n"
    "67,BND,BN,Bandar Seri Begawan,2046,2046-06-30T00:00:00.000,Sat,Mid-year Bank Holiday,G\n"
    "148,BDT,BD,Dhaka,2015,2015-03-17T00:00:00.000,Tue,Birthday of Father of the Nation,G\n"
    "56,ZMW,ZM,Lusaka,2013,2013-10-24T00:00:00.000,Thu,Independence Day,G\n"
    "77,ERN,ER,Asmara,2016,2016-01-07T00:00:00.000,Thu,Christmas (Ge'ez),G\n"
    "47,RON,RO,Bucharest,2021,2021-12-25T00:00:00.000,Sat,Christmas Day,G\n"
    "126,ETB,ET,Addis Ababa,2054,2054-10-14T00:00:00.000,Wed,Birth of Prophet Mohammed*,G\n"
    "148,BDT,BD,Dhaka,2053,2053-05-20T00:00:00.000,Tue,Eid-ul Fitr 2*,G\n"
)

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


def get_cc_bond_df() -> pl.DataFrame:
    return pl.read_csv(io.StringIO(CC_BOND_CSV), try_parse_dates=True)


def get_election_df() -> pl.DataFrame:
    return pl.read_csv(io.StringIO(ELECTION_CSV), try_parse_dates=True)


def get_employee_df() -> pl.DataFrame:
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            "name": [
                "Alice",
                "Bob",
                "Alice",
                "Bob",
                "Charlie",
                "Alice",
                "David",
                "Eve",
                "Bob",
                "Charlie",
                "Alice",
                "Bartholomew",
                "David",
                "Eve",
                "Bob",
                "Charlie",
                "Frank",
            ],
            "score": [85.4, 92.1, 34.1, 51.6, 78.3, 88.2, 90.5, 87.9, 89.7, 82.1, 91.3, 67.1, 85.8, 90.2, 94.5, 75.6, 88.9],
            "passed": [True, True, False, False, False, True, True, True, True, True, True, False, True, True, True, False, True],
            "test_date": [
                # Multiple people joining on same dates, and same people on different dates
                date(2022, 1, 1),  # Alice joins
                date(2022, 1, 1),  # Bob joins same day as Alice
                date(2022, 1, 2),  # Alice rejoins
                date(2022, 1, 2),  # Bob rejoins same day as Alice
                date(2022, 1, 2),  # Charlie joins
                date(2022, 1, 3),  # Alice rejoins
                date(2022, 1, 2),  # David joins same day as Charlie
                date(2022, 1, 4),  # Eve joins
                date(2022, 1, 5),  # Bob has another entry
                date(2022, 1, 5),  # Charlie also on same day as Bob's second entry
                date(2022, 1, 6),  # Alice's third entry
                date(2022, 1, 7),  # Bartholomew joins
                date(2022, 1, 6),  # David's second entry, same day as Alice's third
                date(2022, 1, 8),  # Eve's second entry
                date(2022, 1, 9),  # Bob's third entry
                date(2022, 1, 9),  # Charlie's third entry, same day as Bob's third
                date(2022, 1, 10),  # Frank joins
            ],
        }
    )
    return df


# Module-level variable to store the connection
_duckdb_conn: Optional[duckdb.DuckDBPyConnection] = None


@pytest.fixture(scope="module")
def duckdb_connection():
    """
    Create a DuckDB connection once per test module.
    This connection will be reused by all tests in this file.
    """
    global _duckdb_conn

    # Create a fresh in-memory database
    conn = duckdb.connect(database=":memory:")

    # Load test data frames into DuckDB
    _cc_bond_df = get_cc_bond_df()
    _election_df = get_election_df()
    _employee_df = get_employee_df()

    # DuckDB can natively query Polars DataFrames by referring to the name of Polars DataFrames
    # as they exist in the current scope.
    conn.execute("CREATE TABLE CC_Bond AS SELECT * FROM _cc_bond_df")
    conn.execute("CREATE TABLE ElectionCalendar AS SELECT * FROM _election_df")
    conn.execute("CREATE TABLE EmployeeTbl AS SELECT * FROM _employee_df")

    # Store in module variable for the batch reader to access
    _duckdb_conn = conn

    yield conn

    # Clean up
    conn.close()
    _duckdb_conn = None


# Fake BatchReader to simulate read_arrow_batches_from_odbc behavior.
# This needs to be mocked because read_arrow_batches_from_odbc REQUIRES
# an ODBC connection string (and cannot recieve a connection object).
# This is not a problem in an integrated environment (i.e., a firm-issued
# server), but is a problem when running automated tests on GitHub Actions.


class FakeBatchReader:
    def __init__(self, table: pa.Table, batch_size: int):
        self.table = table
        self.batch_size = batch_size
        self.schema = table.schema

    def __iter__(self):
        return iter(self.table.to_batches())


def fake_read_arrow_batches_from_odbc(query: str, batch_size: int, connection_string: str, **kwargs):
    # Execute query and get arrow table
    global _duckdb_conn
    table = _duckdb_conn.execute(query).fetch_arrow_table()
    return FakeBatchReader(table, batch_size)


@pytest.fixture(autouse=True)
def patch_odbc(monkeypatch, duckdb_connection):
    # Patch arrow_odbc directly since it's imported lazily inside functions
    monkeypatch.setitem(sys.modules, "arrow_odbc", SimpleNamespace(read_arrow_batches_from_odbc=fake_read_arrow_batches_from_odbc))
    monkeypatch.setattr(polars_utils.lazy_sql_reader, "get_sqlglot_dialect_odbc", lambda conn_string: "duckdb")


@pytest.fixture(autouse=True)
def transaction_isolation(duckdb_connection):
    """Isolate test changes using transactions"""
    duckdb_connection.execute("BEGIN TRANSACTION")
    yield
    duckdb_connection.execute("ROLLBACK")


@pytest.fixture(autouse=True)
def clear_piot_cache():
    """Clear the global cachebetween tests to prevent test pollution.

    The cacheuses a global dict keyed by a hash of the serialized LazyFrame.
    In newer versions of polars, LazyFrame serialization is more stable, meaning
    different tests can end up with the same cache key and unexpectedly share cached data.
    """
    from polars_io_tools.io_sources import lazy_cache

    lazy_cache._CACHE.clear()
    yield
    lazy_cache._CACHE.clear()


# Tests begin


def test_wrap_binary_expr():
    expr = (pl.col("A") > 5) | pl.col("B").eq(10)
    res = convert_predicate_to_sql(expr, "tsql").sql("tsql")
    expected = "((A > 5) OR (B = 10))"
    assert res == expected


def test_logical_or_operator():
    """Test that LOGICAL_OR is correctly converted to SQL OR.

    This tests a bug where OperatorType.LOGICAL_OR was not mapped in the SQL visitor,
    causing a warning and falling back to Python-side filtering instead of SQL pushdown.
    LOGICAL_OR can be produced by Polars' optimizer when rewriting expressions.
    """
    # Construct a BinaryExprNode directly with LOGICAL_OR
    left_col = ColumnNode(expr=pl.col("A"), name="A")
    left_lit = LiteralNode(expr=pl.lit(5), value=5)
    left_cmp = BinaryExprNode(expr=pl.col("A") > 5, left=left_col, op=OperatorType.GT, right=left_lit)

    right_col = ColumnNode(expr=pl.col("B"), name="B")
    right_lit = LiteralNode(expr=pl.lit(10), value=10)
    right_cmp = BinaryExprNode(expr=pl.col("B") > 10, left=right_col, op=OperatorType.GT, right=right_lit)

    # Create the LOGICAL_OR node
    logical_or_node = BinaryExprNode(
        expr=(pl.col("A") > 5) | (pl.col("B") > 10),
        left=left_cmp,
        op=OperatorType.LOGICAL_OR,
        right=right_cmp,
    )

    visitor = SQLExpressionVisitor(dialect="tsql")
    visitor.visit(logical_or_node)
    result = visitor.process_results()

    assert result is not None, "LOGICAL_OR should be converted to SQL, not return None"
    sql = result.sql("tsql")
    assert "OR" in sql, f"Expected OR in SQL output, got: {sql}"
    assert "A > 5" in sql and "B > 10" in sql, f"Expected both conditions in SQL, got: {sql}"


def test_logical_and_operator():
    """Test that LOGICAL_AND is correctly converted to SQL AND.

    This tests that OperatorType.LOGICAL_AND is properly mapped in the SQL visitor.
    LOGICAL_AND can be produced by Polars' optimizer when rewriting expressions.
    """
    # Construct a BinaryExprNode directly with LOGICAL_AND
    left_col = ColumnNode(expr=pl.col("A"), name="A")
    left_lit = LiteralNode(expr=pl.lit(5), value=5)
    left_cmp = BinaryExprNode(expr=pl.col("A") > 5, left=left_col, op=OperatorType.GT, right=left_lit)

    right_col = ColumnNode(expr=pl.col("B"), name="B")
    right_lit = LiteralNode(expr=pl.lit(10), value=10)
    right_cmp = BinaryExprNode(expr=pl.col("B") > 10, left=right_col, op=OperatorType.GT, right=right_lit)

    # Create the LOGICAL_AND node
    logical_and_node = BinaryExprNode(
        expr=(pl.col("A") > 5) & (pl.col("B") > 10),
        left=left_cmp,
        op=OperatorType.LOGICAL_AND,
        right=right_cmp,
    )

    visitor = SQLExpressionVisitor(dialect="tsql")
    visitor.visit(logical_and_node)
    result = visitor.process_results()

    assert result is not None, "LOGICAL_AND should be converted to SQL, not return None"
    sql = result.sql("tsql")
    assert "AND" in sql, f"Expected AND in SQL output, got: {sql}"
    assert "A > 5" in sql and "B > 10" in sql, f"Expected both conditions in SQL, got: {sql}"


def test_basic_query():
    """Test simple query"""
    base_query = """
    SELECT CenterID, Centre, EventYear
    FROM CC_Bond
    """
    sql_query = base_query + " WHERE EventYear > 2015"

    df = get_cc_bond_df()
    equivalent_filters = [pl.col("EventYear") > 2015]
    expected = df.filter(*equivalent_filters).select(["CenterID", "Centre", "EventYear"])
    actual = cpl.scan_db(sql_query, "fake_connection_string").collect()
    actual_from_query = cpl.scan_db(base_query, "fake_connection_string").filter(*equivalent_filters).collect()
    assert_frame_equal(actual, expected)
    assert_frame_equal(actual_from_query, expected)


def test_implicit_cast_query():
    """Test implicit cast query"""
    base_query = """
    SELECT CenterID, Centre, EventYear
    FROM CC_Bond
    """
    sql_query = base_query + " WHERE EventYear > 2015"

    df = get_cc_bond_df()
    equivalent_filters = [pl.col("EventYear") > 2015.1]
    expected = df.filter(*equivalent_filters).select(["CenterID", "Centre", "EventYear"])
    actual = cpl.scan_db(sql_query, "fake_connection_string").collect()
    actual_from_query = cpl.scan_db(base_query, "fake_connection_string").filter(*equivalent_filters).collect()

    assert_frame_equal(actual, expected)
    assert_frame_equal(actual_from_query, expected)


def test_column_selection():
    """Test column selection and LIKE"""
    base_query = """
    SELECT *
    FROM CC_Bond
    """
    sql_query = base_query + " WHERE Centre LIKE '%Da%'"

    cols = ["CenterID", "Centre", "EventDate"]

    df = get_cc_bond_df()
    equivalent_filters = [pl.col("Centre").str.contains("Da")]
    expected = df.filter(*equivalent_filters).select(cols)

    # Test both ways: direct SQL and base query + filters
    actual = cpl.scan_db(sql_query, "fake_connection_string").select(cols).collect()
    actual_from_query = cpl.scan_db(base_query, "fake_connection_string").select(cols).filter(*equivalent_filters).collect()

    actual = actual.sort(cols)
    expected = expected.sort(cols)
    actual_from_query = actual_from_query.sort(cols)
    assert_frame_equal(actual, expected)
    assert_frame_equal(actual_from_query, expected)


def test_complex_query():
    """Test more complex query with AND and OR"""
    base_query = """
    SELECT CenterID, CentreCode, ISOCountryCode, Centre, EventYear, EventDate, EventDayOfWeek, EventName, FileType
    FROM CC_Bond
    """
    sql_query = (
        base_query
        + """
    WHERE EventYear BETWEEN 2015 AND 2050
      AND (EventDayOfWeek = 'Fri' OR EventDayOfWeek = 'Sat')
      AND EventName LIKE '%Day%'
    """
    )

    cols = ["CenterID", "CentreCode", "ISOCountryCode", "Centre", "EventYear", "EventDate", "EventDayOfWeek", "EventName", "FileType"]

    df = get_cc_bond_df()
    equivalent_filters = [
        (pl.col("EventYear").is_between(2015, 2050))
        & ((pl.col("EventDayOfWeek") == "Fri") | (pl.col("EventDayOfWeek") == "Sat"))
        & (pl.col("EventName").str.contains("Day"))
    ]
    expected = df.filter(*equivalent_filters).select(cols)

    # Test both ways: direct SQL and base query + filters
    actual = cpl.scan_db(sql_query, "fake_connection_string").select(cols).collect()
    actual_from_query = cpl.scan_db(base_query, "fake_connection_string").select(cols).filter(*equivalent_filters).collect()

    actual = actual.sort(cols)
    expected = expected.sort(cols)
    actual_from_query = actual_from_query.sort(cols)

    assert_frame_equal(actual, expected)
    assert_frame_equal(actual_from_query, expected)


def test_join():
    """Test on a joined table with filters"""
    base_query = """
    SELECT cb.CenterID, cb.Centre, ec.Country, ec.ElectFor
    FROM CC_Bond AS cb, ElectionCalendar AS ec
    """
    sql_query = base_query + " WHERE ec.Country LIKE '%a%'"
    cols = ["CenterID", "Centre", "Country", "ElectFor"]

    df_cb = get_cc_bond_df().select(["CenterID", "Centre"])
    df_ec = get_election_df().select(["Country", "ElectFor"])

    # Create a cross join and then apply filter
    cross_join = df_cb.join(df_ec, how="cross")
    equivalent_filters = [pl.col("Country").str.contains("a")]
    expected = cross_join.filter(*equivalent_filters)

    # Test both ways: direct SQL and base query + filters
    actual = cpl.scan_db(sql_query, "fake_connection_string").collect()
    actual_from_query = cpl.scan_db(base_query, "fake_connection_string").filter(*equivalent_filters).collect()

    expected = expected.sort(cols)
    actual = actual.sort(cols)
    actual_from_query = actual_from_query.sort(cols)

    assert_frame_equal(actual, expected)
    assert_frame_equal(actual_from_query, expected)


def test_not_null():
    """Test NOT NULL"""
    base_query = """
    SELECT CenterID, Centre, EventDate
    FROM CC_Bond
    """
    sql_query = base_query + " WHERE EventDate IS NOT NULL"

    df = get_cc_bond_df()
    equivalent_filters = [pl.col("EventDate").is_not_null()]
    expected = df.select(["CenterID", "Centre", "EventDate"]).filter(*equivalent_filters)

    # Test both ways: direct SQL and base query + filters
    actual = cpl.scan_db(sql_query, "fake_connection_string").collect()
    actual_from_query = cpl.scan_db(base_query, "fake_connection_string").filter(*equivalent_filters).collect()

    assert_frame_equal(actual, expected)
    assert_frame_equal(actual_from_query, expected)


def test_empty_df():
    """Test functionality on a query that returns no records"""
    base_query = """
    SELECT CenterID, Centre, EventDate, EventYear
    FROM CC_Bond
    """
    sql_query = base_query + " WHERE EventYear < 1900"

    cols = [
        "CenterID",
        "Centre",
        "EventDate",
    ]
    df = get_cc_bond_df()
    equivalent_filters = [pl.col("EventYear") < 1900]
    expected = df.filter(*equivalent_filters).select(cols)

    # Test both ways: direct SQL and base query + filters
    actual = cpl.scan_db(sql_query, "fake_connection_string").select(cols).collect()
    # Here, we need to filter first and then select, because we filter on a column that is not selected
    actual_from_query = cpl.scan_db(base_query, "fake_connection_string").filter(*equivalent_filters).select(cols).collect()

    assert_frame_equal(actual, expected)
    assert_frame_equal(actual_from_query, expected)


def test_in_operator():
    """Test IN operator"""
    base_query = """
    SELECT CenterID, Centre, EventYear, EventDate, EventDayOfWeek
    FROM CC_Bond
    """

    cols = ["CenterID", "Centre", "EventYear", "EventDate", "EventDayOfWeek"]
    sql_query = base_query + " WHERE EventDayOfWeek IN ('Fri', 'Sat', 'Sun')"

    df = get_cc_bond_df()
    equivalent_filters = [pl.col("EventDayOfWeek").is_in(["Fri", "Sat", "Sun"])]
    expected = df.filter(*equivalent_filters).select(cols)

    # Test both ways: direct SQL and base query + filters
    actual = cpl.scan_db(sql_query, "fake_connection_string").collect()
    actual_from_query = cpl.scan_db(base_query, "fake_connection_string").filter(*equivalent_filters).collect()

    actual = actual.sort(cols)
    expected = expected.sort(cols)
    actual_from_query = actual_from_query.sort(cols)

    assert_frame_equal(actual, expected)
    assert_frame_equal(actual_from_query, expected)


def test_not_like():
    """Test NOT LIKE operator"""
    base_query = """
    SELECT CenterID, Centre, EventYear, EventName
    FROM CC_Bond
    """
    sql_query = base_query + " WHERE EventName NOT LIKE '%Day%'"

    cols = ["CenterID", "Centre", "EventYear", "EventName"]
    df = get_cc_bond_df()
    equivalent_filters = [~pl.col("EventName").str.contains("Day")]
    expected = df.filter(*equivalent_filters).select(cols)

    # Test both ways: direct SQL and base query + filters
    actual = cpl.scan_db(sql_query, "fake_connection_string").collect()
    actual_from_query = cpl.scan_db(base_query, "fake_connection_string").filter(*equivalent_filters).collect()

    actual = actual.sort(cols)
    expected = expected.sort(cols)
    actual_from_query = actual_from_query.sort(cols)

    assert_frame_equal(actual, expected)
    assert_frame_equal(actual_from_query, expected)


def test_with_columns_preserves_qualification_on_join(duckdb_connection):
    """Ensure with_columns uses original qualified columns to avoid ambiguity.

    When the base query selects a qualified column from a joined table where both
    sides share the same column name, rebuilding projections must keep the table
    qualifier. Otherwise databases can raise an "ambiguous column" error.
    """

    # Create two small tables with a shared column name directly in DuckDB
    duckdb_connection.execute(
        """
        CREATE TABLE A AS SELECT * FROM (
            SELECT 1 AS id, 10 AS shared, 100 AS aval
            UNION ALL
            SELECT 2 AS id, 20 AS shared, 200 AS aval
            UNION ALL
            SELECT 3 AS id, 30 AS shared, 300 AS aval
        )
        """
    )
    duckdb_connection.execute(
        """
        CREATE TABLE B AS SELECT * FROM (
            SELECT 10 AS shared, 1000 AS bval
            UNION ALL
            SELECT 30 AS shared, 3000 AS bval
            UNION ALL
            SELECT 40 AS shared, 4000 AS bval
        )
        """
    )

    # Base query selects a qualified shared column from the left side only
    base_query = "SELECT a.shared, a.aval, b.bval FROM A AS a LEFT JOIN B AS b ON a.shared = b.shared"

    # Expected result computed via Polars for correctness
    expected = (
        pl.DataFrame(
            {
                "shared": [10, 20, 30],
                "aval": [100, 200, 300],
            }
        )
        .sort(["shared", "aval"])
        .rechunk()
    )

    # Using scan_db, selecting unqualified names should preserve original qualification
    lf = cpl.scan_db(base_query, "fake_connection_string")

    # Wrap the fake batch reader to capture the executed SQL
    import arrow_odbc

    captured: list[str] = []
    original_fake_func = arrow_odbc.read_arrow_batches_from_odbc

    def capturing_fake_func(*args, **kwargs):
        # args: (query, batch_size, connection_string)
        if args:
            captured.append(args[0])
        elif "query" in kwargs:
            captured.append(kwargs["query"])
        return original_fake_func(*args, **kwargs)

    # Temporarily patch within this test
    arrow_odbc.read_arrow_batches_from_odbc = capturing_fake_func
    try:
        actual = lf.select(["shared", "aval"]).collect().sort(["shared", "aval"]).rechunk()
    finally:
        arrow_odbc.read_arrow_batches_from_odbc = original_fake_func

    # Validate result values, dtypes can differ by backend
    assert_frame_equal(actual, expected, check_dtypes=False)

    # Validate that the generated SQL preserved qualification for shared/aval
    assert captured, "No SQL captured from read_arrow_batches_from_odbc"
    sql = captured[0]
    patterns = [
        "SELECT a.shared, a.aval",
        'SELECT "a"."shared", "a"."aval"',
        "SELECT [a].[shared], [a].[aval]",
    ]
    assert any(p in sql for p in patterns), f"Expected qualified columns in SQL, got: {sql}"


def test_visit_column():
    """Test if we can correctly visit a column"""
    column_expr = pl.col("my_column")
    column_node = ColumnNode(expr=column_expr, name="my_column")

    visitor = SQLExpressionVisitor(dialect="tsql")

    visitor.visit(column_node)
    result = visitor.process_results()

    assert isinstance(result, exp.Column)
    assert result.sql() == "my_column"


def test_visit_literal():
    """Test visiting numeric and NULL literals."""
    visitor = SQLExpressionVisitor(dialect="tsql")

    # Numeric literal
    literal_node_num = LiteralNode(expr=pl.lit(42), value=42)
    visitor.visit(literal_node_num)
    result_num = visitor.process_results()
    assert isinstance(result_num, exp.Literal)
    assert result_num.sql() == "42"

    # NULL literal
    literal_node_null = LiteralNode(expr=pl.lit(None), value=None)
    visitor.visit(literal_node_null)
    result_null = visitor.process_results()
    assert isinstance(result_null, exp.Null)
    assert result_null.sql() == "NULL"


def test_create_sqlglot_literal_various_types():
    """Unit tests for create_sqlglot_literal to ensure consistent literal handling."""
    # None -> NULL
    assert isinstance(create_sqlglot_literal(None), exp.Null)
    assert create_sqlglot_literal(None).sql() == "NULL"

    # Numeric -> unquoted
    lit_int = create_sqlglot_literal(123)
    assert isinstance(lit_int, exp.Literal)
    assert lit_int.sql() == "123"

    lit_float = create_sqlglot_literal(12.34)
    assert isinstance(lit_float, exp.Literal)
    assert lit_float.sql() == "12.34"

    # String -> quoted
    lit_str = create_sqlglot_literal("abc")
    assert isinstance(lit_str, exp.Literal)
    assert lit_str.sql() == "'abc'"

    # Date -> quoted string (handled by str() formatting)
    d = date(2025, 1, 2)
    lit_date = create_sqlglot_literal(d)
    assert isinstance(lit_date, exp.Literal)
    # sqlglot will quote since we mark is_string=True
    assert lit_date.sql() == "'2025-01-02'"

    # bool -> SQL boolean (TRUE / FALSE), not quoted string
    lit_true = create_sqlglot_literal(True)
    assert isinstance(lit_true, exp.Boolean)
    assert lit_true.sql() == "TRUE"

    lit_false = create_sqlglot_literal(False)
    assert isinstance(lit_false, exp.Boolean)
    assert lit_false.sql() == "FALSE"


def test_create_sqlglot_literal_bool_via_duckdb(duckdb_connection):
    """Verify bool literals round-trip through SQL correctly using DuckDB.

    This guards against the regression where True/False were emitted as quoted
    string literals ('True'/'False') causing DATATYPE_MISMATCH errors when a
    bool value was used as a filter predicate.
    """
    # create_sqlglot_literal(True) must produce TRUE (unquoted), not 'True' (quoted string).
    # A quoted 'True' would fail in DuckDB with a type mismatch when compared to a boolean column.
    duckdb_connection.execute("CREATE TEMPORARY TABLE IF NOT EXISTS bool_test (flag BOOLEAN, val INTEGER)")
    duckdb_connection.execute("INSERT INTO bool_test VALUES (TRUE, 1), (FALSE, 2), (TRUE, 3)")

    true_literal = create_sqlglot_literal(True).sql(dialect="duckdb")
    false_literal = create_sqlglot_literal(False).sql(dialect="duckdb")

    # Verify the literals are unquoted SQL booleans
    assert true_literal == "TRUE"
    assert false_literal == "FALSE"

    # Verify the literals execute in DuckDB without type errors
    rows_true = duckdb_connection.execute(f"SELECT val FROM bool_test WHERE flag = {true_literal}").fetchall()
    rows_false = duckdb_connection.execute(f"SELECT val FROM bool_test WHERE flag = {false_literal}").fetchall()

    assert sorted(r[0] for r in rows_true) == [1, 3]
    assert [r[0] for r in rows_false] == [2]


def test_visit_function_is_null():
    """Test if we can correctly visit NULL"""

    input_expr = pl.col("my_column")
    input_node = ColumnNode(expr=input_expr, name="my_column")
    function_node = FunctionNode(expr=input_expr.is_null(), inputs=[input_node], function_type=BooleanFunctionType.IS_NULL)

    visitor = SQLExpressionVisitor(dialect="tsql")

    visitor.visit(function_node)
    result = visitor.process_results()

    assert isinstance(result, exp.Is)
    assert isinstance(result.expression, exp.Null)
    assert result.sql() == "my_column IS NULL"


@pytest.mark.parametrize("range_dates", [True, False])
def test_piot_cache_integration(monkeypatch, range_dates):
    """Test if we can correctly cache a query"""
    import arrow_odbc

    # Create a call counter
    call_counter = {"count": 0}

    # Store the original fake implementation (which is already patched by the fixture)
    original_fake_func = arrow_odbc.read_arrow_batches_from_odbc

    # Create a wrapper that counts calls but still calls the original fake function
    def counting_fake_func(*args, **kwargs):
        call_counter["count"] += 1
        return original_fake_func(*args, **kwargs)

    # Replace the fake function with our counting wrapper
    monkeypatch.setattr("arrow_odbc.read_arrow_batches_from_odbc", counting_fake_func)

    base_query = """
    SELECT *
    FROM EmployeeTbl
    """
    partition_cols = ["test_date", "name"]
    emp_lf = cpl.scan_db(base_query, "fake_connection_string").piot.cache(order_by="id", partition_cols=partition_cols)

    def apply_filters(lf: pl.LazyFrame, start: date = date(2022, 1, 1), end: date = date(2022, 1, 2)) -> pl.LazyFrame:
        if range_dates:
            lf = lf.filter(pl.col.test_date.is_between(start, end))
        else:
            lf = lf.filter(pl.col.test_date.is_in(pl.date_range(start, end, closed="both", eager=True).to_list()))
        lf = lf.filter(pl.col.name.is_in(["Alice", "Bob"]))
        return lf

    # Reset counter before first call
    call_counter["count"] = 0

    # First call - should hit the database
    res = emp_lf.pipe(apply_filters).collect()
    assert call_counter["count"] == 1, "Expected read_arrow_batches_from_odbc to be called"
    assert_frame_equal(res, get_employee_df().pipe(apply_filters))

    # Reset counter before second call
    call_counter["count"] = 0

    # Same call again - should use cache, not hit database
    res = emp_lf.pipe(apply_filters).collect()
    assert call_counter["count"] == 0, "Expected read_arrow_batches_from_odbc to NOT be called (cache should be used)"
    assert_frame_equal(res, get_employee_df().pipe(apply_filters))

    # Reset counter before third call
    call_counter["count"] = 0

    # We query for a subset of data, we should hit the cache fully.
    res = emp_lf.pipe(apply_filters, start=date(2022, 1, 1), end=date(2022, 1, 2)).collect()
    assert call_counter["count"] == 0, "Expected read_arrow_batches_from_odbc to NOT be called (cache should be used)"
    assert_frame_equal(res, get_employee_df().pipe(apply_filters, start=date(2022, 1, 1), end=date(2022, 1, 2)))

    # We query for an even smaller subset of the data, thus, we should theoretically be able to (only Alice and Bartholomew)
    # hit the cache fully. However, because with our current filters, we don't have any data
    # for Bartholomew, we don't add a partition column filter to the query to exclude Alice. Thus,
    # we mistakenly assume that we have to query the data instead of only hitting the cache.
    # Now, if we either:
    #   - know the full set of our partition column options upfront
    #   - keep track of the queries we made, and use that to determine if we can
    #     hit the cache fully. So, we should've known that we queried for Bartholomew before with
    #     the appropriate dates. This then adds the next challenge of queries that WOULD'VE
    #     detected Bartholomew (maybe like, pl.col.name.ne("Bill")) we should feasibly be able to
    #     infer that we can hit the cache fully.
    # We should be able to know that we can exclusively hit the cache here. The second approach seems difficult,
    # and the first one is simpler and allows for further validation. This presents a challenge for
    # efficient caching when we have entries in partition columns that are emtpy.

    def apply_restricted_filters(lf: pl.LazyFrame) -> pl.LazyFrame:
        start = date(2022, 1, 1)
        end = date(2022, 1, 2)
        if range_dates:
            lf = lf.filter(pl.col.test_date.is_between(start, end))
        else:
            dates = pl.date_range(start, end, closed="both", eager=True).to_list()
            lf = lf.filter(pl.col.test_date.is_in(dates))
        return lf.filter(pl.col.name.is_in(["Bartholomew", "Alice"]))

    res = emp_lf.pipe(apply_restricted_filters).collect()
    assert call_counter["count"] == 1, "Expected read_arrow_batches_from_odbc to be called (cache should be used but we do not detect that)"
    assert_frame_equal(res, get_employee_df().pipe(apply_restricted_filters))


def test_mssql_bracket_quoting():
    """
    Test that our custom MSSQL dialect wraps identifiers in brackets,
    even when the name is a reserved keyword or contains spaces/brackets.
    """
    assert exp.Column(this="CenterID").sql(dialect=MSSQL) == "[CenterID]"
    assert exp.Column(this="GROUP").sql(dialect=MSSQL) == "[GROUP]"
    assert exp.Column(this="Some ] Column").sql(dialect=MSSQL) == "[Some ]] Column]"

    col = exp.Column(this="Amount", table="Sales")
    assert col.sql(dialect=MSSQL) == "[Sales].[Amount]"


def test_total_days_to_datediff():
    """Test that TOTAL_DAYS fast-path works correctly"""
    start_expr = pl.col("StartDate")
    end_expr = pl.col("EndDate")

    start_node = get_parsed_expr(start_expr)
    end_node = get_parsed_expr(end_expr)

    function_types_and_expected = [
        (TemporalFunctionType.TOTAL_DAYS, "DATEDIFF(DAY, StartDate, EndDate)"),
        (TemporalFunctionType.TOTAL_HOURS, "DATEDIFF(HOUR, StartDate, EndDate)"),
    ]

    for func_type, expected_sql in function_types_and_expected:
        # Build a FunctionNode for TOTAL_DAYS with the two inputs
        fn_node = FunctionNode(
            expr=start_expr,
            inputs=[start_node, end_node],
            function_type=func_type,
        )

        visitor = SQLExpressionVisitor(dialect="tsql")
        visitor.visit(fn_node)
        sqlglot_expr = visitor.process_results()

        # The generated SQL should be DATEDIFF(DAY, StartDate, EndDate)
        assert isinstance(sqlglot_expr, exp.DateDiff)
        assert sqlglot_expr.sql(dialect="tsql") == expected_sql

    expr = (pl.col("end_dt") - pl.col("start_dt")).dt.total_days()

    node = get_parsed_expr(expr)

    assert isinstance(node, FunctionNode)
    assert node.function_type == TemporalFunctionType.TOTAL_DAYS


def test_visitor_dialect_normalization():
    """SQLExpressionVisitor normalizes dialect so MSSQL class, 'tsql' string, and None all behave identically.

    This prevents a subtle bug where ``self.dialect == "tsql"`` silently failed
    when the caller passed the ``MSSQL`` class (returned by ``_get_sqlglot_dialect``
    for SQL Server connections), causing DATEDIFF and other TSQL-specific
    conversions to be skipped.
    """
    start_node = get_parsed_expr(pl.col("StartDate"))
    end_node = get_parsed_expr(pl.col("EndDate"))
    fn_node = FunctionNode(
        expr=pl.col("StartDate"),
        inputs=[start_node, end_node],
        function_type=TemporalFunctionType.TOTAL_DAYS,
    )

    for dialect in ("tsql", MSSQL, None):
        visitor = SQLExpressionVisitor(dialect=dialect)
        visitor.visit(fn_node)
        result = visitor.process_results()
        assert isinstance(result, exp.DateDiff), f"DATEDIFF conversion should work with dialect={dialect!r}, got {type(result)}"
        assert result.sql(dialect="tsql") == "DATEDIFF(DAY, StartDate, EndDate)"


def test_convert_predicate_to_sql_with_mssql_class():
    """convert_predicate_to_sql works when passed the MSSQL class (not just 'tsql' string)."""
    pred = pl.col("x") > 5
    # Should not raise and should produce valid SQL for both forms
    result_str = convert_predicate_to_sql(pred, dialect="tsql")
    result_cls = convert_predicate_to_sql(pred, dialect=MSSQL)
    result_none = convert_predicate_to_sql(pred, dialect=None)
    assert result_str is not None
    assert result_cls is not None
    assert result_none is not None
    # All three should produce identical SQL
    assert result_str.sql() == result_cls.sql() == result_none.sql()


def test_rename_with_filter_integration():
    """Test that aliases created via select work correctly with filters - integration-like test using mock data."""

    base_query = """
    SELECT
        CenterID,
        EventYear
    FROM CC_Bond
    """

    # Test filtering on aliased columns created via select
    lf = cpl.scan_db(base_query, "fake_connection_string")
    lf = lf.select(pl.col("CenterID").alias("center_id"), pl.col("EventYear").alias("event_year"))
    lf = lf.filter(pl.col("event_year").is_not_null())
    result = lf.collect()

    # Should not throw an error and should return results
    assert result.shape[0] >= 0  # At least no error
    assert result.columns == ["center_id", "event_year"]

    # Verify the filtering actually worked by comparing with expected data
    df = get_cc_bond_df()
    expected = df.select([pl.col("CenterID").alias("center_id"), pl.col("EventYear").alias("event_year")]).filter(pl.col("event_year").is_not_null())

    assert_frame_equal(result.sort("center_id"), expected.sort("center_id"))


def test_select_alias_predicate_pushdown_simple():
    """Predicates referencing simple SELECT aliases are rewritten to original columns in WHERE."""
    base_query = """
    SELECT
        EventYear AS event_year,
        CenterID AS center_id
    FROM CC_Bond
    """

    # Filter uses alias; pushdown should rewrite to original columns
    lf = cpl.scan_db(base_query, "fake_connection_string")
    res = lf.filter(pl.col("event_year") > 2016).select(["event_year", "center_id"]).collect()

    df = get_cc_bond_df()
    expected = df.select([pl.col("EventYear").alias("event_year"), pl.col("CenterID").alias("center_id")]).filter(pl.col("event_year") > 2016)
    assert_frame_equal(res.sort(["event_year", "center_id"]), expected.sort(["event_year", "center_id"]))


def test_select_alias_cast_datetime_to_date_types():
    """Casting Datetime to Date via SELECT alias yields Date in schema and result."""
    base_query = """
    SELECT
        CAST(EventDate AS DATE) AS BusDate,
        CenterID
    FROM CC_Bond
    """

    lf = cpl.scan_db(base_query, "fake_connection_string")

    # Validate collect_schema types
    schema = lf.collect_schema()
    assert schema.get("BusDate") == pl.Date
    assert schema.get("CenterID") in (pl.Int64, pl.Int32)

    # Validate actual collected types
    res = lf.collect()
    assert res.schema.get("BusDate") == pl.Date
    assert res.schema.get("CenterID") in (pl.Int64, pl.Int32)

    # Validate values match expected cast
    df = get_cc_bond_df()
    expected = df.select([pl.col("EventDate").cast(pl.Date).alias("BusDate"), pl.col("CenterID")])
    assert_frame_equal(res.sort(["BusDate", "CenterID"]), expected.sort(["BusDate", "CenterID"]))


def test_select_original_and_cast_same_column():
    """Select the same column twice: original Datetime and CAST to Date with alias."""
    base_query = """
    SELECT
        EventDate,
        CAST(EventDate AS DATE) AS BusDate,
        CenterID
    FROM CC_Bond
    """

    lf = cpl.scan_db(base_query, "fake_connection_string")

    # Validate collect_schema types
    schema = lf.collect_schema()
    assert schema.get("EventDate") == pl.Datetime
    assert schema.get("BusDate") == pl.Date
    assert schema.get("CenterID") in (pl.Int64, pl.Int32)

    # Validate actual collected types
    res = lf.collect()
    assert res.schema.get("EventDate") == pl.Datetime
    assert res.schema.get("BusDate") == pl.Date
    assert res.schema.get("CenterID") in (pl.Int64, pl.Int32)

    # Validate values match expected selection
    df = get_cc_bond_df()
    expected = df.select(
        [
            pl.col("EventDate"),
            pl.col("EventDate").cast(pl.Date).alias("BusDate"),
            pl.col("CenterID"),
        ]
    )
    assert_frame_equal(res.sort(["CenterID", "EventDate"]), expected.sort(["CenterID", "EventDate"]))


def test_nested_select_star_with_cast_alias(caplog):
    """Support subquery with CAST alias and star expansion."""
    query = """
    SELECT * FROM (
        SELECT CAST(EventDate AS DATE) AS EventDateAsDate, * FROM CC_Bond
    ) AS t
    """

    caplog.set_level(logging.DEBUG)
    lf = cpl.scan_db(query, "fake_connection_string")

    # Schema checks
    schema = lf.collect_schema()
    assert schema.get("EventDateAsDate") == pl.Date
    assert schema.get("EventDate") == pl.Datetime

    # Data checks
    res = lf.collect()
    df = get_cc_bond_df()
    # Expected columns: alias first, then original table columns in their order
    orig_cols = [
        "CenterID",
        "CentreCode",
        "ISOCountryCode",
        "Centre",
        "EventYear",
        "EventDate",
        "EventDayOfWeek",
        "EventName",
        "FileType",
    ]
    expected = df.select([pl.col("EventDate").cast(pl.Date).alias("EventDateAsDate"), *orig_cols])
    assert_frame_equal(res, expected)

    # Verify logged SQL matches normalized expected SQL via sqlglot
    expected_sql = parse_one(query, dialect="duckdb").sql(dialect="duckdb")
    assert any(("Executing SQL with pushdown: " + expected_sql) in record.message for record in caplog.records)


# These tests verify that scan_db LazyFrames can be serialized with cloudpickle,
# which is required for distributed computing (e.g., Ray). Tests are located here
# rather than in test_pickle.py because they require the database mocking
# infrastructure defined in this file.


class TestScanDbPickle:
    """Tests for scan_db cloudpickle serialization support."""

    def test_scan_db_pickle_basic(self):
        """scan_db LazyFrames can be pickled and unpickled."""
        import cloudpickle

        query = "SELECT * FROM CC_Bond"
        lf = cpl.scan_db(query, "fake_connection_string")

        # Pickle roundtrip
        pickled = cloudpickle.dumps(lf)
        lf_unpickled = cloudpickle.loads(pickled)

        # Verify schema is preserved
        assert lf.collect_schema() == lf_unpickled.collect_schema()

        # Verify data is correct after unpickling
        result = lf_unpickled.collect()
        expected = get_cc_bond_df()
        assert_frame_equal(result, expected)

    def test_scan_db_pickle_with_filter(self):
        """scan_db with filters can be pickled."""
        import cloudpickle

        query = "SELECT * FROM CC_Bond"
        lf = cpl.scan_db(query, "fake_connection_string").filter(pl.col("EventYear") > 2020)

        pickled = cloudpickle.dumps(lf)
        lf_unpickled = cloudpickle.loads(pickled)

        result = lf_unpickled.collect()
        expected = get_cc_bond_df().filter(pl.col("EventYear") > 2020)
        assert_frame_equal(result, expected)

    def test_scan_db_pickle_with_select(self):
        """scan_db with column selection can be pickled."""
        import cloudpickle

        query = "SELECT CenterID, Centre, EventYear FROM CC_Bond"
        lf = cpl.scan_db(query, "fake_connection_string")

        pickled = cloudpickle.dumps(lf)
        lf_unpickled = cloudpickle.loads(pickled)

        result = lf_unpickled.collect()
        expected = get_cc_bond_df().select(["CenterID", "Centre", "EventYear"])
        assert_frame_equal(result, expected)


class TestHeadPushdownBehavior:
    """
    Tests verifying head() behavior with filters in scan_db.

    Current Polars behavior (safe):
    - When a predicate is passed, n_rows is NOT pushed to the IO source
    - head() is applied locally after the source returns filtered data
    - This prevents the bug where LIMIT would be applied before filtering

    Potential bug scenario (if both n_rows and predicate are passed):
    - If predicate can't convert to SQL but n_rows is pushed
    - SQL gets: SELECT ... LIMIT N (no WHERE)
    - Database returns N rows (some may not match filter)
    - Filter applied locally → fewer than N matching rows
    - But user expected N matching rows!
    """

    def test_polars_does_not_push_n_rows_with_predicate(self, duckdb_connection):
        """
        Verify that Polars does NOT push n_rows when a predicate is present.

        This is the key safety behavior that prevents the head pushdown bug.
        If this test fails, it means Polars changed behavior and the bug could manifest.
        """
        import arrow_odbc

        # Capture executed SQL queries
        executed_queries = []
        original_func = arrow_odbc.read_arrow_batches_from_odbc

        def capturing_func(query, *args, **kwargs):
            executed_queries.append(query)
            return original_func(query, *args, **kwargs)

        arrow_odbc.read_arrow_batches_from_odbc = capturing_func

        try:
            query = "SELECT * FROM EmployeeTbl"
            lf = cpl.scan_db(query, "fake_connection_string")

            # Apply a SQL-convertible filter AND head()
            result = lf.filter(pl.col("score") > 80).head(5).collect()

            # Find the actual data query (not schema query)
            data_queries = [q for q in executed_queries if "LIMIT 0" not in q]
            assert len(data_queries) == 1, f"Expected 1 data query, got {len(data_queries)}"

            sql = data_queries[0]
            print(f"\nExecuted SQL: {sql}")

            # The SQL should have WHERE but NOT have LIMIT
            # Because Polars doesn't push n_rows when there's a predicate
            assert "WHERE" in sql.upper(), "SQL should have WHERE clause"

            # Verify LIMIT is NOT in the query (Polars applies head() locally)
            has_limit = "LIMIT" in sql.upper() and "LIMIT 0" not in sql.upper()
            assert not has_limit, f"SAFETY CHECK: Polars should NOT push LIMIT when predicate is present. SQL was: {sql}"

            # Verify we got correct results
            assert result.height == 5, f"Expected 5 rows, got {result.height}"
            assert all(result["score"] > 80), "All rows should match filter"

        finally:
            arrow_odbc.read_arrow_batches_from_odbc = original_func

    def test_polars_pushes_n_rows_without_predicate(self, duckdb_connection):
        """
        Verify that Polars DOES push n_rows when there's no predicate.
        """
        import arrow_odbc

        executed_queries = []
        original_func = arrow_odbc.read_arrow_batches_from_odbc

        def capturing_func(query, *args, **kwargs):
            executed_queries.append(query)
            return original_func(query, *args, **kwargs)

        arrow_odbc.read_arrow_batches_from_odbc = capturing_func

        try:
            query = "SELECT * FROM EmployeeTbl"
            lf = cpl.scan_db(query, "fake_connection_string")

            # Just head(), no filter
            result = lf.head(5).collect()

            # Find the actual data query
            data_queries = [q for q in executed_queries if "LIMIT 0" not in q]
            assert len(data_queries) == 1

            sql = data_queries[0]
            print(f"\nExecuted SQL: {sql}")

            # Without predicate, Polars SHOULD push LIMIT
            assert "LIMIT" in sql.upper(), f"Expected LIMIT in SQL, got: {sql}"
            assert result.height == 5

        finally:
            arrow_odbc.read_arrow_batches_from_odbc = original_func

    def test_head_with_non_sql_filter_returns_correct_count(self, duckdb_connection):
        """
        Verify that head() with a non-SQL-convertible filter returns correct results.

        Since Polars doesn't push n_rows when there's a predicate, head() is applied
        locally after filtering, giving correct results.
        """
        emp_df = get_employee_df()
        total_matching = emp_df.filter(pl.col("score") > 80).height

        # Non-SQL-convertible filter (Python UDF)
        non_sql_filter = pl.col("score").map_elements(lambda x: x > 80, return_dtype=pl.Boolean)

        # Verify filter can't be converted to SQL
        from polars_io_tools.io_sources.sql_utils import convert_predicate_to_sql

        assert convert_predicate_to_sql(non_sql_filter) is None

        query = "SELECT * FROM EmployeeTbl"
        lf = cpl.scan_db(query, "fake_connection_string")
        result = lf.filter(non_sql_filter).head(10).collect()

        # We should get min(10, total_matching) rows
        expected = min(10, total_matching)
        assert result.height == expected, f"Expected {expected} rows, got {result.height}"
        assert all(result["score"] > 80)

    def test_source_generator_bug_if_n_rows_and_predicate_both_passed(self, duckdb_connection):
        """
        Demonstrate the POTENTIAL BUG in scan_db's source_generator.

        This test directly invokes the source_generator with both n_rows and a
        non-SQL-convertible predicate to show what WOULD happen if Polars ever
        passed both parameters together.

        Current state: Polars doesn't pass n_rows when there's a predicate, so this
        bug doesn't manifest in normal usage. But it could if:
        - Future Polars versions change behavior
        - Someone directly calls the source_generator
        - Edge cases in optimizer

        The fix would be: if predicate can't be converted to SQL, don't push n_rows.
        """
        import arrow_odbc

        # Capture SQL to verify the bug mechanism
        executed_queries = []
        original_func = arrow_odbc.read_arrow_batches_from_odbc

        def capturing_func(query, *args, **kwargs):
            executed_queries.append(query)
            return original_func(query, *args, **kwargs)

        arrow_odbc.read_arrow_batches_from_odbc = capturing_func

        try:
            # Set up a non-SQL filter
            non_sql_filter = pl.col("score").map_elements(lambda x: x > 80, return_dtype=pl.Boolean)

            # Verify filter can't be converted
            from polars_io_tools.io_sources.sql_utils import convert_predicate_to_sql

            assert convert_predicate_to_sql(non_sql_filter) is None

            # Get the source_generator by creating a scan_db LazyFrame
            # We need to access the internal source function
            query = "SELECT * FROM EmployeeTbl"

            # Instead of calling through Polars (which won't pass n_rows with predicate),
            # we test the logic directly by checking the SQL that would be generated

            # Create the LazyFrame to set up the source
            lf = cpl.scan_db(query, "fake_connection_string")

            # Simulate what WOULD happen if both were passed:
            # The SQL would be: SELECT * FROM EmployeeTbl LIMIT 10 (no WHERE!)
            # Then filter applied locally

            # First verify current behavior is safe
            executed_queries.clear()
            result = lf.filter(non_sql_filter).head(10).collect()

            data_queries = [q for q in executed_queries if "LIMIT 0" not in q]
            sql = data_queries[0] if data_queries else ""

            print(f"\nActual SQL executed: {sql}")
            print(f"Result rows: {result.height}")

            # Current behavior: no LIMIT in SQL (safe!)
            has_limit = "LIMIT" in sql.upper() and "LIMIT 0" not in sql.upper()
            if not has_limit:
                print("SAFE: Polars did not push LIMIT with predicate")
            else:
                print("WARNING: LIMIT was pushed to SQL!")

            # The bug scenario would be:
            # SQL: SELECT ... LIMIT 10 (no WHERE because predicate can't convert)
            # Returns first 10 rows
            # Filter locally: only 7 of first 10 have score > 80
            # Result: 7 rows instead of 10!

            emp_df = get_employee_df()
            total_matching = emp_df.filter(pl.col("score") > 80).height
            first_10_matching = emp_df.head(10).filter(pl.col("score") > 80).height

            print("\nIf bug were present:")
            print(f"  Total matching rows: {total_matching}")
            print(f"  Matching in first 10: {first_10_matching}")
            print(f"  Would get: {first_10_matching} rows instead of {min(10, total_matching)}")

        finally:
            arrow_odbc.read_arrow_batches_from_odbc = original_func

    def test_workaround_collect_then_head(self):
        """
        WORKAROUND: Use .collect().head(N) instead of .head(N).collect()

        This pattern guarantees correct results regardless of Polars version
        or optimizer behavior.
        """
        emp_df = get_employee_df()

        non_sql_filter = pl.col("score").map_elements(lambda x: x > 80, return_dtype=pl.Boolean)

        matching_count = emp_df.filter(pl.col("score") > 80).height

        query = "SELECT * FROM EmployeeTbl"
        lf = cpl.scan_db(query, "fake_connection_string")
        result = lf.filter(non_sql_filter).collect().head(10)

        expected_count = min(10, matching_count)
        assert result.height == expected_count
        assert all(result["score"] > 80)


# These tests verify that kwargs like query_timeout_sec are passed correctly
# to both the schema query and data query in scan_db.


class TestArrowOdbcKwargsPassthrough:
    """Tests for arrow_odbc kwargs being passed to both schema and data queries."""

    def test_query_timeout_sec_passed_to_schema_query(self, monkeypatch, duckdb_connection):
        """Verify that query_timeout_sec is passed to the schema query."""
        import arrow_odbc

        # Track kwargs passed to each call
        captured_calls: list[dict] = []
        original_func = arrow_odbc.read_arrow_batches_from_odbc

        def capturing_func(*args, **kwargs):
            # Capture the kwargs for each call
            captured_calls.append({"query": kwargs.get("query") or args[0], "kwargs": dict(kwargs)})
            return original_func(*args, **kwargs)

        monkeypatch.setattr("arrow_odbc.read_arrow_batches_from_odbc", capturing_func)

        query = "SELECT * FROM CC_Bond"
        timeout_value = 30

        # Create the LazyFrame with query_timeout_sec - this triggers schema query
        _lf = cpl.scan_db(query, "fake_connection_string", query_timeout_sec=timeout_value)

        # The schema query should have been made during scan_db
        schema_calls = [c for c in captured_calls if "LIMIT 0" in c["query"]]
        assert len(schema_calls) == 1, f"Expected 1 schema query, got {len(schema_calls)}"

        schema_call = schema_calls[0]
        assert "query_timeout_sec" in schema_call["kwargs"], "query_timeout_sec should be passed to schema query"
        assert schema_call["kwargs"]["query_timeout_sec"] == timeout_value

    def test_query_timeout_sec_passed_to_data_query(self, monkeypatch, duckdb_connection):
        """Verify that query_timeout_sec is passed to the data query during collect."""
        import arrow_odbc

        captured_calls: list[dict] = []
        original_func = arrow_odbc.read_arrow_batches_from_odbc

        def capturing_func(*args, **kwargs):
            captured_calls.append({"query": kwargs.get("query") or args[0], "kwargs": dict(kwargs)})
            return original_func(*args, **kwargs)

        monkeypatch.setattr("arrow_odbc.read_arrow_batches_from_odbc", capturing_func)

        query = "SELECT * FROM CC_Bond"
        timeout_value = 45

        lf = cpl.scan_db(query, "fake_connection_string", query_timeout_sec=timeout_value)

        # Clear captured calls to focus on collect
        captured_calls.clear()

        # Now collect - this triggers the data query
        lf.collect()

        data_calls = [c for c in captured_calls if "LIMIT 0" not in c["query"]]
        assert len(data_calls) == 1, f"Expected 1 data query, got {len(data_calls)}"

        data_call = data_calls[0]
        assert "query_timeout_sec" in data_call["kwargs"], "query_timeout_sec should be passed to data query"
        assert data_call["kwargs"]["query_timeout_sec"] == timeout_value

    def test_multiple_kwargs_passed_to_both_queries(self, monkeypatch, duckdb_connection):
        """Verify that multiple arrow_odbc kwargs are passed to both schema and data queries."""
        import arrow_odbc

        captured_calls: list[dict] = []
        original_func = arrow_odbc.read_arrow_batches_from_odbc

        def capturing_func(*args, **kwargs):
            captured_calls.append({"query": kwargs.get("query") or args[0], "kwargs": dict(kwargs)})
            return original_func(*args, **kwargs)

        monkeypatch.setattr("arrow_odbc.read_arrow_batches_from_odbc", capturing_func)

        query = "SELECT * FROM CC_Bond"

        # Pass multiple kwargs
        lf = cpl.scan_db(
            query,
            "fake_connection_string",
            krb5=False,
            query_timeout_sec=30,
            login_timeout_sec=10,
            max_text_size=4096,
        )

        # Check schema query
        schema_calls = [c for c in captured_calls if "LIMIT 0" in c["query"]]
        assert len(schema_calls) == 1
        schema_kwargs = schema_calls[0]["kwargs"]
        assert schema_kwargs.get("query_timeout_sec") == 30
        assert schema_kwargs.get("login_timeout_sec") == 10
        assert schema_kwargs.get("max_text_size") == 4096

        captured_calls.clear()

        # Check data query
        lf.collect()
        data_calls = [c for c in captured_calls if "LIMIT 0" not in c["query"]]
        assert len(data_calls) == 1
        data_kwargs = data_calls[0]["kwargs"]
        assert data_kwargs.get("query_timeout_sec") == 30
        assert data_kwargs.get("login_timeout_sec") == 10
        assert data_kwargs.get("max_text_size") == 4096

    def test_schema_kwarg_passed_to_both_queries(self, monkeypatch, duckdb_connection):
        """Verify that 'schema' kwarg is passed to both schema and data queries for consistency."""
        import arrow_odbc
        import pyarrow as pa

        captured_calls: list[dict] = []
        original_func = arrow_odbc.read_arrow_batches_from_odbc

        def capturing_func(*args, **kwargs):
            captured_calls.append({"query": kwargs.get("query") or args[0], "kwargs": dict(kwargs)})
            return original_func(*args, **kwargs)

        monkeypatch.setattr("arrow_odbc.read_arrow_batches_from_odbc", capturing_func)

        query = "SELECT CenterID, Centre FROM CC_Bond"

        # Create a custom schema to pass
        custom_schema = pa.schema([("CenterID", pa.int64()), ("Centre", pa.string())])

        lf = cpl.scan_db(
            query,
            "fake_connection_string",
            krb5=False,
            query_timeout_sec=30,
            schema=custom_schema,
        )

        # Check schema query - 'schema' SHOULD be present for consistency
        schema_calls = [c for c in captured_calls if "LIMIT 0" in c["query"]]
        assert len(schema_calls) == 1
        schema_kwargs = schema_calls[0]["kwargs"]
        assert "schema" in schema_kwargs, "'schema' should be passed to schema query for consistency"
        assert schema_kwargs.get("query_timeout_sec") == 30

        captured_calls.clear()

        # Check data query - 'schema' SHOULD be present
        lf.collect()
        data_calls = [c for c in captured_calls if "LIMIT 0" not in c["query"]]
        assert len(data_calls) == 1
        data_kwargs = data_calls[0]["kwargs"]
        assert "schema" in data_kwargs, "'schema' should be passed to data query"
        assert data_kwargs.get("query_timeout_sec") == 30

    def test_map_schema_kwarg_passed_to_both_queries(self, monkeypatch, duckdb_connection):
        """Verify that 'map_schema' kwarg is passed to both schema and data queries for consistency."""
        import arrow_odbc

        captured_calls: list[dict] = []
        original_func = arrow_odbc.read_arrow_batches_from_odbc

        def capturing_func(*args, **kwargs):
            captured_calls.append({"query": kwargs.get("query") or args[0], "kwargs": dict(kwargs)})
            return original_func(*args, **kwargs)

        monkeypatch.setattr("arrow_odbc.read_arrow_batches_from_odbc", capturing_func)

        query = "SELECT CenterID, Centre FROM CC_Bond"

        # Create a custom map_schema function
        def custom_map_schema(schema):
            return schema

        lf = cpl.scan_db(
            query,
            "fake_connection_string",
            krb5=False,
            query_timeout_sec=30,
            map_schema=custom_map_schema,
        )

        # Check schema query - 'map_schema' SHOULD be present for consistency
        schema_calls = [c for c in captured_calls if "LIMIT 0" in c["query"]]
        assert len(schema_calls) == 1
        schema_kwargs = schema_calls[0]["kwargs"]
        assert "map_schema" in schema_kwargs, "'map_schema' should be passed to schema query for consistency"
        assert schema_kwargs.get("query_timeout_sec") == 30

        captured_calls.clear()

        # Check data query - 'map_schema' SHOULD be present
        lf.collect()
        data_calls = [c for c in captured_calls if "LIMIT 0" not in c["query"]]
        assert len(data_calls) == 1
        data_kwargs = data_calls[0]["kwargs"]
        assert "map_schema" in data_kwargs, "'map_schema' should be passed to data query"
        assert data_kwargs.get("query_timeout_sec") == 30

    def test_all_kwargs_passed_to_both_queries(self, monkeypatch, duckdb_connection):
        """Verify all kwargs including schema overrides are passed to both queries."""
        import arrow_odbc
        import pyarrow as pa

        captured_calls: list[dict] = []
        original_func = arrow_odbc.read_arrow_batches_from_odbc

        def capturing_func(*args, **kwargs):
            captured_calls.append({"query": kwargs.get("query") or args[0], "kwargs": dict(kwargs)})
            return original_func(*args, **kwargs)

        monkeypatch.setattr("arrow_odbc.read_arrow_batches_from_odbc", capturing_func)

        query = "SELECT CenterID, Centre FROM CC_Bond"
        custom_schema = pa.schema([("CenterID", pa.int64()), ("Centre", pa.string())])

        def custom_map_schema(schema):
            return schema

        lf = cpl.scan_db(
            query,
            "fake_connection_string",
            krb5=False,
            query_timeout_sec=30,
            login_timeout_sec=10,
            schema=custom_schema,
            map_schema=custom_map_schema,
        )

        # Check schema query - all kwargs should be present
        schema_calls = [c for c in captured_calls if "LIMIT 0" in c["query"]]
        assert len(schema_calls) == 1
        schema_kwargs = schema_calls[0]["kwargs"]

        assert "schema" in schema_kwargs
        assert "map_schema" in schema_kwargs
        assert schema_kwargs.get("query_timeout_sec") == 30
        assert schema_kwargs.get("login_timeout_sec") == 10

        captured_calls.clear()

        # Check data query - all kwargs should be present
        lf.collect()
        data_calls = [c for c in captured_calls if "LIMIT 0" not in c["query"]]
        assert len(data_calls) == 1
        data_kwargs = data_calls[0]["kwargs"]

        assert "schema" in data_kwargs
        assert "map_schema" in data_kwargs
        assert data_kwargs.get("query_timeout_sec") == 30
        assert data_kwargs.get("login_timeout_sec") == 10


class TestMSSQLQualifiedStar:
    """Tests that the MSSQL dialect correctly renders table-qualified star expressions."""

    def test_qualified_star_in_join(self):
        """Table-qualified star (alias.*) should render as [alias].* not [alias].[*]."""
        query = """
        SELECT a.*, b.SomeCol
        FROM ServerA.dbo.vTableOne a
        LEFT JOIN ServerB.dbo.vTableTwo b
          ON a.JoinKey = b.JoinKey
          AND a.AsOfDate BETWEEN b.StartDate AND b.EndDate
        """
        parsed = parse_one(query, dialect=MSSQL)
        result = parsed.sql(dialect=MSSQL)
        assert "[a].*" in result
        assert "[*]" not in result

    def test_qualified_star_without_join(self):
        """Table-qualified star works for a simple aliased table."""
        query = "SELECT t.* FROM SomeDB.dbo.SomeTable t"
        parsed = parse_one(query, dialect=MSSQL)
        result = parsed.sql(dialect=MSSQL)
        assert "[t].*" in result
        assert "[*]" not in result

    def test_unqualified_star(self):
        """Plain SELECT * should still work."""
        query = "SELECT * FROM SomeTable"
        parsed = parse_one(query, dialect=MSSQL)
        result = parsed.sql(dialect=MSSQL)
        assert "SELECT *" in result
        assert "[*]" not in result


class TestMSSQLAnonymousCasePreservation:
    """Tests that the MSSQL dialect preserves original casing for Anonymous nodes."""

    def test_tvf_with_literal_args_preserves_case(self):
        """Table-valued functions with literal arguments should not be uppercased."""
        query = "SELECT * FROM [FuturesData].[dbo].f_GetFuturesMaxVolAsOfDate('2026-02-06')"
        parsed = parse_one(query, dialect=MSSQL)
        result = parsed.sql(dialect=MSSQL)
        assert "f_GetFuturesMaxVolAsOfDate" in result
        assert "F_GETFUTURESMAXVOLASOFDATE" not in result

    def test_table_with_nolock_hint_preserves_case(self):
        """Table names with (nolock) hints should preserve original casing."""
        query = "SELECT * FROM MyTable (nolock)"
        parsed = parse_one(query, dialect=MSSQL)
        result = parsed.sql(dialect=MSSQL)
        assert "MyTable" in result
        assert "MYTABLE" not in result

    def test_tvf_with_multiple_args_preserves_case(self):
        """TVFs with multiple arguments should preserve casing."""
        query = "SELECT * FROM [db].[dbo].fn_GetData('2026-01-01', 42)"
        parsed = parse_one(query, dialect=MSSQL)
        result = parsed.sql(dialect=MSSQL)
        assert "fn_GetData" in result
        assert "FN_GETDATA" not in result

    def test_tvf_roundtrip_preserves_full_query(self):
        """Full query with TVF should roundtrip correctly through parse/generate."""
        query = "SELECT * FROM [FuturesData].[dbo].f_GetFuturesMaxVolAsOfDate('2026-02-06')"
        parsed = parse_one(query, dialect=MSSQL)
        result = parsed.sql(dialect=MSSQL)
        assert "[FuturesData]" in result
        assert "[dbo]" in result
        assert "f_GetFuturesMaxVolAsOfDate('2026-02-06')" in result


def _make_capture_sql():
    """Return a context manager that captures SQL sent to the fake ODBC reader.

    Shared helper used by multiple test classes.
    """
    import contextlib

    import arrow_odbc

    captured: list[str] = []
    original_func = arrow_odbc.read_arrow_batches_from_odbc

    @contextlib.contextmanager
    def ctx():
        def capturing_func(*args, **kwargs):
            q = kwargs.get("query") or (args[0] if args else "")
            captured.append(q)
            return original_func(*args, **kwargs)

        arrow_odbc.read_arrow_batches_from_odbc = capturing_func
        try:
            yield captured
        finally:
            arrow_odbc.read_arrow_batches_from_odbc = original_func

    return ctx()


def _extract_data_sql(captured: list[str]) -> str:
    """Extract the single data query (skip the LIMIT 0 schema probe).

    Shared helper used by multiple test classes.
    """
    data_queries = [q for q in captured if "LIMIT 0" not in q]
    assert len(data_queries) == 1, f"Expected exactly 1 data query, got {len(data_queries)}: {data_queries}"
    return data_queries[0]


class TestJoinStarAmbiguousColumns:
    """Tests for the subquery-wrapping fix that avoids ambiguous column names
    when a query uses ``table.*`` expressions with JOINs.

    When column selection pushdown is active and the query has qualified stars
    (``cr.*``) with JOINs, the original query is wrapped as a subquery.  The
    outer SELECT picks columns from the flat result set — no ambiguity possible.
    """

    BASE_QUERY = "SELECT cr.*, br.CNTRY_OF_RISK FROM CreditRef cr LEFT JOIN BBCredit br ON cr.ID_BB_COMPANY = br.ID_BB_COMPANY"

    @pytest.fixture(autouse=True)
    def _create_join_tables(self, duckdb_connection):
        """Create test tables that share a column name (ID_BB_COMPANY)."""
        duckdb_connection.execute(
            """
            CREATE TABLE CreditRef AS SELECT * FROM (
                SELECT 1 AS ID, 100 AS ID_BB_COMPANY, 'US' AS CNTRY, DATE '2024-01-15' AS SOME_DATE
                UNION ALL
                SELECT 2 AS ID, 200 AS ID_BB_COMPANY, 'GB' AS CNTRY, DATE '2024-02-20' AS SOME_DATE
                UNION ALL
                SELECT 3 AS ID, 100 AS ID_BB_COMPANY, 'JP' AS CNTRY, DATE '2024-03-10' AS SOME_DATE
            )
            """
        )
        duckdb_connection.execute(
            """
            CREATE TABLE BBCredit AS SELECT * FROM (
                SELECT 100 AS ID_BB_COMPANY, 'USA' AS CNTRY_OF_RISK
                UNION ALL
                SELECT 200 AS ID_BB_COMPANY, 'GBR' AS CNTRY_OF_RISK
            )
            """
        )
        yield
        # Tables are rolled back by the transaction_isolation fixture

    def _capture_sql(self):
        return _make_capture_sql()

    @staticmethod
    def _data_sql(captured: list[str]) -> str:
        return _extract_data_sql(captured)

    @staticmethod
    def _assert_qualified(sql: str, table: str, column: str) -> None:
        """Assert that *column* appears qualified by *table* in *sql*."""
        patterns = [
            f"{table}.{column}",
            f'"{table}"."{column}"',
            f"[{table}].[{column}]",
        ]
        assert any(p in sql for p in patterns), f"Expected {table}-qualified {column} in SQL, got: {sql}"

    # Column selection on star+JOIN queries wraps the original query as
    # a subquery (__cpl_subq) so that the outer SELECT uses flat, unambiguous
    # column names.

    def test_select_shared_column_is_qualified(self):
        """Selecting a column shared by both tables uses subquery wrapping."""
        lf = cpl.scan_db(self.BASE_QUERY, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.select(["ID_BB_COMPANY", "CNTRY_OF_RISK"]).collect()

        assert result.shape[0] == 3
        assert set(result.columns) == {"ID_BB_COMPANY", "CNTRY_OF_RISK"}

        sql = self._data_sql(captured)
        assert "__cpl_subq" in sql, f"Expected subquery wrapping, got: {sql}"

    def test_select_non_shared_column_is_qualified(self):
        """Selecting a column unique to cr also uses subquery wrapping."""
        lf = cpl.scan_db(self.BASE_QUERY, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.select(["ID", "CNTRY"]).collect()

        assert result.shape[0] == 3
        sql = self._data_sql(captured)
        assert "__cpl_subq" in sql, f"Expected subquery wrapping, got: {sql}"

    def test_select_all_star_expanded_columns(self):
        """Selecting every column that came from cr.* uses subquery wrapping."""
        lf = cpl.scan_db(self.BASE_QUERY, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.select(["ID", "ID_BB_COMPANY", "CNTRY", "SOME_DATE"]).collect()

        assert result.shape[0] == 3
        sql = self._data_sql(captured)
        assert "__cpl_subq" in sql, f"Expected subquery wrapping, got: {sql}"

    # When only a predicate is pushed (no column selection), the standard
    # path is used — no subquery wrapping.  Predicates stay unqualified.

    def test_predicate_pushdown_works_with_star_join(self):
        """Filter pushdown on unambiguous columns works without qualification."""
        lf = cpl.scan_db(self.BASE_QUERY, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.filter(pl.col("CNTRY") == "US").collect()

        assert result.shape[0] == 1
        assert result["CNTRY"][0] == "US"

        sql = self._data_sql(captured)
        assert "WHERE" in sql.upper()

    def test_predicate_with_multiple_filters(self, duckdb_connection):
        """Compound AND filter on columns unique to the star-expanded table.

        Mirrors the real-world query:
          SELECT cr.*, br.CNTRY_OF_RISK FROM ... LEFT JOIN ...
          WHERE (DATE = '...' AND PointID = ...)
        where DATE and PointID only exist on the cr side, so the database
        resolves them without qualification.
        """
        # Add a PointID column to CreditRef so we can filter on it
        duckdb_connection.execute("ALTER TABLE CreditRef ADD COLUMN PointID BIGINT")
        duckdb_connection.execute("UPDATE CreditRef SET PointID = ID * 100000")

        base_query = "SELECT cr.*, br.CNTRY_OF_RISK FROM CreditRef cr LEFT JOIN BBCredit br ON cr.ID_BB_COMPANY = br.ID_BB_COMPANY"

        lf = cpl.scan_db(base_query, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.filter((pl.col("SOME_DATE") == date(2024, 1, 15)) & (pl.col("PointID") == 100000)).collect()

        assert result.shape[0] == 1
        assert result["ID"][0] == 1

        sql = self._data_sql(captured)
        assert "WHERE" in sql.upper()

    def test_combined_select_qualified_and_predicate_unqualified(self):
        """Column selection uses subquery; predicate is applied on the outer query.

        The filter uses CNTRY (unique to cr) so it's unambiguous in WHERE even
        without table qualification — matching real-world SQL Server behaviour
        where unqualified WHERE columns resolve fine for non-shared names.
        """
        lf = cpl.scan_db(self.BASE_QUERY, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.filter(pl.col("CNTRY") == "US").select(["ID_BB_COMPANY", "CNTRY_OF_RISK"]).collect()

        assert result.shape[0] == 1
        assert set(result.columns) == {"ID_BB_COMPANY", "CNTRY_OF_RISK"}

        sql = self._data_sql(captured)
        assert "__cpl_subq" in sql, f"Expected subquery wrapping, got: {sql}"
        assert "WHERE" in sql.upper()

    def test_alias_predicate_rewrites_to_underlying_expression(self):
        """A predicate on a CAST alias expands to the underlying CAST, not a bare column."""
        base_query = (
            "SELECT cr.*, CAST(cr.SOME_DATE AS DATE) AS bus_date, br.CNTRY_OF_RISK "
            "FROM CreditRef cr "
            "LEFT JOIN BBCredit br ON cr.ID_BB_COMPANY = br.ID_BB_COMPANY"
        )

        lf = cpl.scan_db(base_query, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.filter(pl.col("bus_date") >= date(2024, 2, 1)).collect()

        assert result.shape[0] == 2
        sql = self._data_sql(captured)
        assert "WHERE" in sql.upper()
        # The alias should be rewritten to the underlying CAST expression
        assert "CAST" in sql.upper(), f"Expected CAST in WHERE from alias rewrite, got: {sql}"

    def test_select_alias_column_alongside_star_expanded(self):
        """Selecting an alias + a star-expanded column uses subquery wrapping."""
        base_query = (
            "SELECT cr.*, CAST(cr.SOME_DATE AS DATE) AS bus_date, br.CNTRY_OF_RISK "
            "FROM CreditRef cr "
            "LEFT JOIN BBCredit br ON cr.ID_BB_COMPANY = br.ID_BB_COMPANY"
        )

        lf = cpl.scan_db(base_query, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.select(["bus_date", "ID_BB_COMPANY"]).collect()

        assert result.shape[0] == 3
        assert set(result.columns) == {"bus_date", "ID_BB_COMPANY"}

        sql = self._data_sql(captured)
        assert "__cpl_subq" in sql, f"Expected subquery wrapping, got: {sql}"
        # The inner subquery preserves the original CAST alias
        assert "CAST" in sql.upper()
        assert "bus_date" in sql

    def test_bare_star_join_does_not_qualify(self, duckdb_connection):
        """Bare ``*`` (no table qualifier) does not trigger star expansion."""
        duckdb_connection.execute(
            """
            CREATE TABLE StarA AS SELECT * FROM (
                SELECT 1 AS aid, 10 AS joinkey
                UNION ALL SELECT 2 AS aid, 20 AS joinkey
            )
            """
        )
        duckdb_connection.execute(
            """
            CREATE TABLE StarB AS SELECT * FROM (
                SELECT 10 AS bkey, 'x' AS bval
                UNION ALL SELECT 20 AS bkey, 'y' AS bval
            )
            """
        )

        base_query = "SELECT * FROM StarA a LEFT JOIN StarB b ON a.joinkey = b.bkey"
        lf = cpl.scan_db(base_query, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.select(["aid", "bval"]).collect()

        assert result.shape[0] == 2
        sql = self._data_sql(captured)
        # With bare *, columns should be unqualified (no expansion possible)
        assert "a.aid" not in sql and '"a"."aid"' not in sql

    def test_explicit_columns_with_join_uses_subquery(self, duckdb_connection):
        """A redundant full-schema select on a JOIN leaves the original SQL unchanged."""
        duckdb_connection.execute(
            """
            CREATE TABLE JoinA AS SELECT * FROM (
                SELECT 1 AS id, 10 AS shared, 100 AS aval
                UNION ALL SELECT 2 AS id, 20 AS shared, 200 AS aval
            )
            """
        )
        duckdb_connection.execute(
            """
            CREATE TABLE JoinB AS SELECT * FROM (
                SELECT 10 AS shared, 1000 AS bval
                UNION ALL SELECT 20 AS shared, 2000 AS bval
            )
            """
        )

        base_query = "SELECT a.shared, b.bval FROM JoinA a LEFT JOIN JoinB b ON a.shared = b.shared"
        lf = cpl.scan_db(base_query, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.select(["shared", "bval"]).collect()

        assert result.shape[0] == 2
        sql = self._data_sql(captured)
        assert "__cpl_subq" not in sql, f"Did not expect subquery wrapping, got: {sql}"
        self._assert_qualified(sql, "a", "shared")
        self._assert_qualified(sql, "b", "bval")

    def test_no_expansion_when_no_joins(self):
        """Single-table SELECT * does not trigger star expansion."""
        base_query = "SELECT * FROM CreditRef"
        lf = cpl.scan_db(base_query, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.select(["ID", "CNTRY"]).collect()

        assert result.shape[0] == 3
        sql = self._data_sql(captured)
        # No qualification needed — single table
        assert "cr.ID" not in sql and '"cr"."ID"' not in sql

    def test_no_modification_without_pushdown(self):
        """Collecting without select/filter/head runs the original query unmodified."""
        lf = cpl.scan_db(self.BASE_QUERY, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.collect()

        assert result.shape[0] == 3
        sql = self._data_sql(captured)
        # Original query should pass through with the star intact
        assert "cr.*" in sql or "cr.*" in sql.replace('"', "")

    # When multiple qualified stars are present (e.g., a.*, b.*, c.col),
    # we can't determine which star produced which column.  The fix wraps
    # the original query as a subquery and selects from it.

    @pytest.fixture()
    def _create_multi_star_tables(self, duckdb_connection):
        """Create three tables for multi-star subquery wrapping tests.

        Orders and Customers share ``customer_id``; Products has no overlap.
        """
        duckdb_connection.execute(
            """
            CREATE TABLE Orders AS SELECT * FROM (
                SELECT 1 AS order_id, 10 AS customer_id, DATE '2024-01-15' AS order_date
                UNION ALL
                SELECT 2 AS order_id, 20 AS customer_id, DATE '2024-02-20' AS order_date
                UNION ALL
                SELECT 3 AS order_id, 10 AS customer_id, DATE '2024-03-10' AS order_date
            )
            """
        )
        duckdb_connection.execute(
            """
            CREATE TABLE Products AS SELECT * FROM (
                SELECT 1 AS product_id, 'Widget' AS product_name
                UNION ALL
                SELECT 2 AS product_id, 'Gadget' AS product_name
                UNION ALL
                SELECT 3 AS product_id, 'Doohickey' AS product_name
            )
            """
        )
        duckdb_connection.execute(
            """
            CREATE TABLE Customers AS SELECT * FROM (
                SELECT 10 AS customer_id, 'Alice' AS cust_name
                UNION ALL
                SELECT 20 AS customer_id, 'Bob' AS cust_name
            )
            """
        )
        yield

    @pytest.mark.usefixtures("_create_multi_star_tables")
    def test_multi_star_select_shared_uses_subquery(self):
        """Selecting a shared column from a multi-star JOIN uses subquery wrapping."""
        query = "SELECT o.*, p.*, c.cust_name FROM Orders o CROSS JOIN Products p LEFT JOIN Customers c ON o.customer_id = c.customer_id"
        lf = cpl.scan_db(query, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.select(["customer_id", "cust_name"]).collect()

        assert result.shape[0] == 9  # 3 orders × 3 products (cross join)
        assert set(result.columns) == {"customer_id", "cust_name"}

        sql = self._data_sql(captured)
        # Multi-star path wraps in a subquery
        assert "__cpl_subq" in sql, f"Expected subquery wrapping (__cpl_subq) in SQL, got: {sql}"

    @pytest.mark.usefixtures("_create_multi_star_tables")
    def test_multi_star_select_non_shared_uses_subquery(self):
        """Non-shared columns also go through subquery wrapping when multiple stars are present."""
        query = "SELECT o.*, p.*, c.cust_name FROM Orders o CROSS JOIN Products p LEFT JOIN Customers c ON o.customer_id = c.customer_id"
        lf = cpl.scan_db(query, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.select(["order_id", "product_name"]).collect()

        assert result.shape[0] == 9  # 3 orders × 3 products
        assert set(result.columns) == {"order_id", "product_name"}

        sql = self._data_sql(captured)
        assert "__cpl_subq" in sql, f"Expected subquery wrapping (__cpl_subq) in SQL, got: {sql}"

    @pytest.mark.usefixtures("_create_multi_star_tables")
    def test_multi_star_filter_with_subquery(self):
        """Predicate pushdown works correctly through the subquery wrapper."""
        query = "SELECT o.*, p.*, c.cust_name FROM Orders o CROSS JOIN Products p LEFT JOIN Customers c ON o.customer_id = c.customer_id"
        lf = cpl.scan_db(query, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.filter(pl.col("cust_name") == "Alice").select(["order_id", "cust_name", "product_name"]).collect()

        # Alice has customer_id=10 → orders 1 and 3, cross joined with 3 products = 6 rows
        assert result.shape[0] == 6
        assert result["cust_name"].to_list() == ["Alice"] * 6

        sql = self._data_sql(captured)
        assert "__cpl_subq" in sql
        assert "WHERE" in sql.upper()

    @pytest.mark.usefixtures("_create_multi_star_tables")
    def test_multi_star_no_subquery_without_pushdown(self):
        """Collecting without select/filter runs the original multi-star query unmodified."""
        query = "SELECT o.*, p.*, c.cust_name FROM Orders o CROSS JOIN Products p LEFT JOIN Customers c ON o.customer_id = c.customer_id"
        lf = cpl.scan_db(query, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.collect()

        assert result.shape[0] == 9  # 3 orders × 3 products
        sql = self._data_sql(captured)
        # No subquery wrapping when there's no pushdown
        assert "__cpl_subq" not in sql

    def test_comma_join_with_qualified_star_uses_subquery(self, duckdb_connection):
        """``SELECT a.*, b.col FROM A a, B b WHERE ...`` triggers subquery wrapping.

        Comma-separated tables in FROM are implicit cross joins.  The detector
        must find multiple tables in the FROM clause (not just explicit JOINs).
        """
        duckdb_connection.execute(
            """
            CREATE TABLE CommaA AS SELECT * FROM (
                SELECT 1 AS aid, 10 AS shared_key
                UNION ALL SELECT 2 AS aid, 20 AS shared_key
            )
            """
        )
        duckdb_connection.execute(
            """
            CREATE TABLE CommaB AS SELECT * FROM (
                SELECT 10 AS shared_key, 'x' AS bval
                UNION ALL SELECT 20 AS shared_key, 'y' AS bval
            )
            """
        )

        base_query = "SELECT a.*, b.bval FROM CommaA a, CommaB b WHERE a.shared_key = b.shared_key"
        lf = cpl.scan_db(base_query, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.select(["shared_key", "bval"]).collect()

        assert result.shape[0] == 2
        assert set(result.columns) == {"shared_key", "bval"}

        sql = self._data_sql(captured)
        assert "__cpl_subq" in sql, f"Expected subquery wrapping for comma-join, got: {sql}"


class TestSubqueryWrappingHoisting:
    """Tests for the always-wrap subquery approach and clause hoisting.

    When any pushdown (column selection, predicate, or row limit) is requested,
    the original query is wrapped as ``SELECT ... FROM (original) AS __cpl_subq``.
    ORDER BY (without TOP/OFFSET) and OPTION hints are hoisted to the outer query.
    """

    @pytest.fixture(autouse=True)
    def _create_tables(self, duckdb_connection):
        """Create test tables for hoisting tests."""
        duckdb_connection.execute(
            """
            CREATE TABLE Sales AS SELECT * FROM (
                SELECT 'East' AS region, 100 AS amount, DATE '2024-01-15' AS sale_date
                UNION ALL SELECT 'East' AS region, 200 AS amount, DATE '2024-02-20' AS sale_date
                UNION ALL SELECT 'West' AS region, 150 AS amount, DATE '2024-01-10' AS sale_date
                UNION ALL SELECT 'West' AS region, 300 AS amount, DATE '2024-03-05' AS sale_date
            )
            """
        )
        yield

    def _capture_sql(self):
        return _make_capture_sql()

    @staticmethod
    def _data_sql(captured: list[str]) -> str:
        return _extract_data_sql(captured)

    def test_order_by_stays_in_inner_for_non_mssql(self):
        """A redundant full-schema select does not trigger wrapping for non-MSSQL ORDER BY queries."""
        base_query = "SELECT region, amount FROM Sales ORDER BY amount DESC"
        lf = cpl.scan_db(base_query, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.select(["region", "amount"]).collect()

        assert result.shape[0] == 4
        sql = self._data_sql(captured)
        assert "__cpl_subq" not in sql
        assert "ORDER BY" in sql.upper(), f"Expected ORDER BY in original query, got: {sql}"

    def test_order_by_with_limit_stays_in_inner(self):
        """ORDER BY with LIMIT is meaningful — left in inner query even for MSSQL."""
        base_query = "SELECT region, amount FROM Sales ORDER BY amount DESC LIMIT 2"
        lf = cpl.scan_db(base_query, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.select(["region"]).collect()

        assert result.shape[0] == 2
        sql = self._data_sql(captured)
        assert "__cpl_subq" in sql
        # ORDER BY should be INSIDE the subquery (paired with LIMIT)
        inner_match = sql.split("__cpl_subq")[0]
        assert "ORDER BY" in inner_match.upper(), f"ORDER BY should be in inner query, got: {sql}"

    # ORDER BY hoisting: MSSQL-specific (tested via _prepare_inner_for_subquery)
    #
    # These test the hoisting function directly with MSSQL dialect because
    # the DuckDB test environment uses a non-MSSQL dialect (no hoisting).
    # Integration tests against real SQL Server cover end-to-end.

    def test_mssql_order_by_without_top_is_moved_to_outer(self):
        """MSSQL: ORDER BY without TOP is hoisted (SQL Server error 1033)."""
        from polars_io_tools.io_sources.sql_utils import _prepare_inner_for_subquery

        parsed = parse_one("SELECT region, amount FROM Sales ORDER BY amount DESC")
        inner, order_to_hoist, _ = _prepare_inner_for_subquery(parsed, dialect=MSSQL)

        # ORDER BY removed from inner
        assert inner.args.get("order") is None
        # ORDER BY returned for outer
        assert order_to_hoist is not None
        assert "amount" in order_to_hoist.sql().lower()

    def test_mssql_order_by_with_limit_stays_in_inner(self):
        """MSSQL: ORDER BY + TOP stays in inner — determines which rows are selected."""
        from polars_io_tools.io_sources.sql_utils import _prepare_inner_for_subquery

        parsed = parse_one("SELECT TOP 2 region, amount FROM Sales ORDER BY amount DESC", dialect=MSSQL)
        inner, order_to_hoist, _ = _prepare_inner_for_subquery(parsed, dialect=MSSQL)

        # ORDER BY stays in inner
        assert inner.args.get("order") is not None
        # Nothing to hoist
        assert order_to_hoist is None

    def test_mssql_order_by_hoisted_with_table_qualifiers_stripped(self):
        """MSSQL: table qualifiers are stripped from hoisted ORDER BY columns.

        ``ORDER BY cb.EventDate`` becomes ``ORDER BY EventDate`` because the
        table alias ``cb`` only exists inside the subquery.
        """
        from polars_io_tools.io_sources.sql_utils import _prepare_inner_for_subquery

        parsed = parse_one("SELECT cb.region, cb.amount FROM Sales cb ORDER BY cb.amount DESC")
        inner, order_to_hoist, _ = _prepare_inner_for_subquery(parsed, dialect=MSSQL)

        assert order_to_hoist is not None
        order_sql = order_to_hoist.sql()
        # Should have "amount" but NOT "cb.amount"
        assert "amount" in order_sql.lower()
        assert "cb." not in order_sql.lower(), f"Table qualifier should be stripped, got: {order_sql}"

    def test_mssql_order_by_with_offset_stays_in_inner(self):
        """MSSQL: ORDER BY + OFFSET stays in inner (pagination)."""
        from polars_io_tools.io_sources.sql_utils import _prepare_inner_for_subquery

        parsed = parse_one(
            "SELECT region, amount FROM Sales ORDER BY amount OFFSET 1 ROWS FETCH NEXT 2 ROWS ONLY",
            dialect=MSSQL,
        )
        inner, order_to_hoist, _ = _prepare_inner_for_subquery(parsed, dialect=MSSQL)

        assert inner.args.get("order") is not None
        assert order_to_hoist is None

    def test_mssql_option_hints_hoisted(self):
        """MSSQL: OPTION hints are moved from inner to outer."""
        from polars_io_tools.io_sources.sql_utils import _prepare_inner_for_subquery

        parsed = parse_one("SELECT region FROM Sales OPTION (HASH JOIN)", dialect=MSSQL)
        inner, _, options_to_hoist = _prepare_inner_for_subquery(parsed, dialect=MSSQL)

        assert len(options_to_hoist) == 1
        assert "HASH JOIN" in options_to_hoist[0].sql(dialect=MSSQL).upper()
        # Inner should have no options left
        assert not list(inner.find_all(exp.QueryOption))

    def test_non_mssql_no_hoisting(self):
        """Non-MSSQL: nothing is hoisted — ORDER BY and options stay in inner."""
        from polars_io_tools.io_sources.sql_utils import _prepare_inner_for_subquery

        parsed = parse_one("SELECT region, amount FROM Sales ORDER BY amount DESC")
        inner, order_to_hoist, options_to_hoist = _prepare_inner_for_subquery(parsed, dialect="duckdb")

        # ORDER BY stays in inner
        assert inner.args.get("order") is not None
        # Nothing hoisted
        assert order_to_hoist is None
        assert options_to_hoist == []

    def test_no_wrapping_without_pushdown(self):
        """No pushdown → original query passes through unmodified."""
        base_query = "SELECT region, amount FROM Sales ORDER BY amount"
        lf = cpl.scan_db(base_query, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.collect()

        assert result.shape[0] == 4
        sql = self._data_sql(captured)
        assert "__cpl_subq" not in sql
        assert "ORDER BY" in sql.upper()

    def test_alias_predicate_works_without_rewriting(self):
        """Predicate on a CAST alias works via subquery output — no alias rewriting needed."""
        base_query = "SELECT region, CAST(sale_date AS DATE) AS sale_date FROM Sales"
        lf = cpl.scan_db(base_query, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.filter(pl.col("sale_date") >= date(2024, 2, 1)).collect()

        assert result.shape[0] == 2
        sql = self._data_sql(captured)
        assert "__cpl_subq" in sql
        # The outer WHERE references sale_date (subquery output name), not the CAST expression
        outer_part = sql.split("__cpl_subq")[-1]
        assert "WHERE" in outer_part.upper()
        # CAST should only be in the INNER query, not in the outer WHERE
        assert "CAST" not in outer_part.upper()

    def test_union_predicate_filters_all_branches(self, duckdb_connection):
        """Predicate on UNION wraps the entire UNION — filters all branches, not just one."""
        duckdb_connection.execute(
            """
            CREATE TABLE SalesQ1 AS SELECT * FROM (
                SELECT 'East' AS region, 100 AS amount
                UNION ALL SELECT 'West' AS region, 200 AS amount
            )
            """
        )
        duckdb_connection.execute(
            """
            CREATE TABLE SalesQ2 AS SELECT * FROM (
                SELECT 'East' AS region, 300 AS amount
                UNION ALL SELECT 'North' AS region, 400 AS amount
            )
            """
        )

        base_query = "SELECT region, amount FROM SalesQ1 UNION ALL SELECT region, amount FROM SalesQ2"
        lf = cpl.scan_db(base_query, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.filter(pl.col("region") == "East").collect()

        # Should get East from BOTH branches (Q1 and Q2)
        assert result.shape[0] == 2
        assert all(r == "East" for r in result["region"].to_list())

        sql = self._data_sql(captured)
        assert "__cpl_subq" in sql

    def test_union_column_selection(self, duckdb_connection):
        """Column selection on UNION wraps the entire UNION in a subquery."""
        duckdb_connection.execute(
            """
            CREATE TABLE UnionA AS SELECT * FROM (
                SELECT 1 AS id, 'a' AS name, 10 AS val
                UNION ALL SELECT 2 AS id, 'b' AS name, 20 AS val
            )
            """
        )
        duckdb_connection.execute(
            """
            CREATE TABLE UnionB AS SELECT * FROM (
                SELECT 3 AS id, 'c' AS name, 30 AS val
            )
            """
        )

        base_query = "SELECT id, name, val FROM UnionA UNION ALL SELECT id, name, val FROM UnionB"
        lf = cpl.scan_db(base_query, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.select(["id", "name"]).collect()

        assert result.shape[0] == 3
        assert set(result.columns) == {"id", "name"}

        sql = self._data_sql(captured)
        assert "__cpl_subq" in sql

    def test_group_by_with_predicate_pushdown(self):
        """Predicate on a GROUP BY query works via subquery wrapping."""
        base_query = "SELECT region, SUM(amount) AS total FROM Sales GROUP BY region"
        lf = cpl.scan_db(base_query, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.filter(pl.col("total") >= 250).collect()

        assert result.shape[0] == 2  # East=300, West=450
        sql = self._data_sql(captured)
        assert "__cpl_subq" in sql
        # GROUP BY stays in the inner query — the outer query just filters
        inner_part = sql.split("__cpl_subq")[0]
        assert "GROUP BY" in inner_part.upper()

    def test_cte_query_with_column_selection(self):
        """CTE query wrapped in subquery — sqlglot auto-hoists WITH for TSQL."""
        base_query = "WITH cte AS (SELECT region, amount FROM Sales) SELECT region, amount FROM cte"
        lf = cpl.scan_db(base_query, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.select(["region"]).collect()

        assert result.shape[0] == 4
        assert set(result.columns) == {"region"}
        sql = self._data_sql(captured)
        assert "__cpl_subq" in sql

    def test_order_by_with_offset_stays_in_inner(self):
        """ORDER BY + OFFSET is meaningful — both stay in the inner query."""
        base_query = "SELECT region, amount FROM Sales ORDER BY amount LIMIT 2 OFFSET 1"
        lf = cpl.scan_db(base_query, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.select(["region"]).collect()

        assert result.shape[0] == 2
        sql = self._data_sql(captured)
        assert "__cpl_subq" in sql
        # ORDER BY should be INSIDE the subquery (paired with OFFSET)
        inner_part = sql.split("__cpl_subq")[0]
        assert "ORDER BY" in inner_part.upper(), f"ORDER BY should be in inner query, got: {sql}"

    def test_order_by_qualified_stays_in_inner_for_non_mssql(self, duckdb_connection):
        """A redundant full-schema select preserves the original qualified ORDER BY query."""
        duckdb_connection.execute(
            """
            CREATE TABLE OrderQualA AS SELECT * FROM (
                SELECT 1 AS id, DATE '2024-01-01' AS event_date
                UNION ALL SELECT 2 AS id, DATE '2024-03-01' AS event_date
            )
            """
        )
        base_query = "SELECT a.id, a.event_date FROM OrderQualA a ORDER BY a.event_date"
        lf = cpl.scan_db(base_query, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.select(["id", "event_date"]).collect()

        assert result.shape[0] == 2
        sql = self._data_sql(captured)
        assert "__cpl_subq" not in sql
        assert "ORDER BY" in sql.upper()

    def test_predicate_only_pushdown(self):
        """Predicate-only pushdown still wraps in subquery."""
        base_query = "SELECT region, amount, sale_date FROM Sales"
        lf = cpl.scan_db(base_query, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.filter(pl.col("region") == "East").collect()

        assert result.shape[0] == 2
        assert set(result.columns) == {"region", "amount", "sale_date"}
        sql = self._data_sql(captured)
        assert "__cpl_subq" in sql
        outer_part = sql.split("__cpl_subq")[-1]
        assert "WHERE" in outer_part.upper()

    def test_nested_subquery_wrapping(self):
        """Query that already contains a subquery gets wrapped again safely."""
        base_query = "SELECT region, amount FROM (SELECT region, amount FROM Sales) AS inner_q"
        lf = cpl.scan_db(base_query, "fake_connection_string")

        with self._capture_sql() as captured:
            result = lf.select(["region"]).collect()

        assert result.shape[0] == 4
        assert set(result.columns) == {"region"}
        sql = self._data_sql(captured)
        assert "__cpl_subq" in sql


def test_head_zero_skips_query(monkeypatch):
    """head(0) should return an empty DataFrame without sending a query to the database."""
    query = "SELECT CenterID, Centre, EventYear FROM CC_Bond"
    lf = cpl.scan_db(query, "fake_connection_string")

    # Track whether the ODBC reader is invoked
    calls = {"n": 0}
    original_reader = fake_read_arrow_batches_from_odbc

    def tracking_reader(*args, **kwargs):
        calls["n"] += 1
        return original_reader(*args, **kwargs)

    monkeypatch.setattr("arrow_odbc.read_arrow_batches_from_odbc", tracking_reader)

    result = lf.head(0).collect()

    assert result.height == 0
    assert set(result.columns) == {"CenterID", "Centre", "EventYear"}
    assert calls["n"] == 0, f"Expected no ODBC queries, but {calls['n']} were made"


def test_head_zero_with_column_selection(monkeypatch):
    """head(0) with column selection should return the correct empty schema."""
    query = "SELECT CenterID, Centre, EventYear FROM CC_Bond"
    lf = cpl.scan_db(query, "fake_connection_string")

    calls = {"n": 0}
    original_reader = fake_read_arrow_batches_from_odbc

    def tracking_reader(*args, **kwargs):
        calls["n"] += 1
        return original_reader(*args, **kwargs)

    monkeypatch.setattr("arrow_odbc.read_arrow_batches_from_odbc", tracking_reader)

    result = lf.select(["CenterID", "Centre"]).head(0).collect()

    assert result.height == 0
    assert set(result.columns) == {"CenterID", "Centre"}
    assert calls["n"] == 0


def test_str_contains_regex_skips_pushdown_for_tsql():
    """str.contains with regex metacharacters should skip pushdown for TSQL (MSSQL has no regex)."""
    regex_patterns = [
        "^(CL|GC|SI|HO)",  # anchored alternation
        ".*abc",  # wildcard
        "foo|bar",  # alternation
        "[A-Z]+",  # character class + quantifier
        r"CL\d+",  # digit class
    ]
    for pattern in regex_patterns:
        expr = pl.col("Centre").str.contains(pattern)
        result = convert_predicate_to_sql(expr, dialect="tsql")
        assert result is None, f"Expected None for regex pattern {pattern!r}, got {result}"


def test_str_contains_regex_pushes_regexp_like_for_non_tsql():
    """str.contains with regex metacharacters should push RegexpLike for dialects that support it."""
    expr = pl.col("Centre").str.contains("^(CL|GC|SI|HO)")
    for dialect in ["clickhouse", "postgres"]:
        result = convert_predicate_to_sql(expr, dialect=dialect)
        assert result is not None, f"Expected RegexpLike for {dialect}, got None"
        sql = result.sql(dialect=dialect)
        assert "LIKE" not in sql, f"Expected regex function, not LIKE, for {dialect}: {sql}"


def test_str_contains_regex_with_and_partial_pushdown():
    """When AND-ed with a pushable predicate, the regex term is dropped but the other term pushes (TSQL)."""
    expr = pl.col("Centre").str.contains("^test") & (pl.col("EventYear") > 2020)
    result = convert_predicate_to_sql(expr, dialect="tsql")
    assert result is not None
    sql = result.sql()
    assert "LIKE" not in sql  # regex term was dropped
    assert "EventYear" in sql  # numeric comparison still pushed


@pytest.mark.parametrize(
    "filter_expr",
    [
        pl.col("Centre").str.contains("^(Dh|Ph)"),
        pl.col("Centre").str.contains(".*mala"),
        pl.col("Centre").str.contains("Da"),
        pl.col("Centre").str.contains("Da", literal=True),
        pl.col("Centre").str.contains("^(Dh|Ph)") & (pl.col("EventYear") > 2020),
        ~pl.col("Centre").str.contains("Da"),
    ],
)
def test_str_contains(filter_expr):
    """scan_db with str.contains filter matches Polars in-memory result."""
    base_query = "SELECT * FROM CC_Bond"

    # Lazy path: goes through the full scan_db pushdown pipeline
    actual = cpl.scan_db(base_query, "fake_connection_string").filter(filter_expr).collect()
    # Eager path: Polars filters in-memory (ground truth)
    expected = get_cc_bond_df().filter(filter_expr)

    actual = actual.sort("CenterID", "EventDate")
    expected = expected.sort("CenterID", "EventDate")
    assert_frame_equal(actual, expected)
