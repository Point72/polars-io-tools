import gzip
import io
import zlib
from datetime import date

import duckdb
import polars as pl
import pyarrow as pa
import pytest
import requests
from polars.testing import assert_frame_equal
from urllib3.response import HTTPResponse

import polars_io_tools as cpl
from polars_io_tools import io_sources as polars_utils
from polars_io_tools._compat import POLARS_HAS_COLLECT_BATCHES
from polars_io_tools.io_sources.lazy_clickhouse_reader import get_batch_reader_http

"""
This module contains tests for the lazy ClickHouse reader (scan_clickhouse).
Query tests mock get_batch_reader_http to execute queries against an in-memory DuckDB
database, returning results as Arrow IPC streams (the same format ClickHouse
uses with FORMAT ArrowStream).
HTTP tests mock requests.post instead, exercising the real batch reader.
"""

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


def get_cc_bond_df() -> pl.DataFrame:
    return pl.read_csv(io.StringIO(CC_BOND_CSV), try_parse_dates=True)


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
                date(2022, 1, 1),
                date(2022, 1, 1),
                date(2022, 1, 2),
                date(2022, 1, 2),
                date(2022, 1, 2),
                date(2022, 1, 3),
                date(2022, 1, 2),
                date(2022, 1, 4),
                date(2022, 1, 5),
                date(2022, 1, 5),
                date(2022, 1, 6),
                date(2022, 1, 7),
                date(2022, 1, 6),
                date(2022, 1, 8),
                date(2022, 1, 9),
                date(2022, 1, 9),
                date(2022, 1, 10),
            ],
        }
    )
    return df


# Module-level variable to store the connection
_duckdb_conn: duckdb.DuckDBPyConnection | None = None


@pytest.fixture(scope="module")
def duckdb_connection():
    """Create a DuckDB connection once per test module."""
    global _duckdb_conn

    conn = duckdb.connect(database=":memory:")

    _cc_bond_df = get_cc_bond_df()
    _employee_df = get_employee_df()

    conn.execute("CREATE TABLE CC_Bond AS SELECT * FROM _cc_bond_df")
    conn.execute("CREATE TABLE EmployeeTbl AS SELECT * FROM _employee_df")

    _duckdb_conn = conn

    yield conn

    conn.close()
    _duckdb_conn = None


# Fake get_batch_reader_http that returns an Arrow IPC stream reader backed
# by DuckDB. This exercises the real read_next_batch() / StopIteration /
# reader.close() code path in scan_clickhouse.


def fake_get_batch_reader_http(query: str, url: str, params: dict, **kwargs):
    global _duckdb_conn
    table = _duckdb_conn.execute(query).fetch_arrow_table()
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, table.schema)
    for batch in table.to_batches():
        writer.write_batch(batch)
    writer.close()
    buf = sink.getvalue()
    return pa.ipc.open_stream(buf)


@pytest.fixture(autouse=True)
def patch_clickhouse(monkeypatch, duckdb_connection):
    monkeypatch.setattr(polars_utils.lazy_clickhouse_reader, "get_batch_reader_http", fake_get_batch_reader_http)


@pytest.fixture(autouse=True)
def transaction_isolation(duckdb_connection):
    """Isolate test changes using transactions."""
    duckdb_connection.execute("BEGIN TRANSACTION")
    yield
    duckdb_connection.execute("ROLLBACK")


# Tests begin

FAKE_URL = "http://localhost:8123"
FAKE_PARAMS = {}


@pytest.mark.parametrize("content_encoding", [None, "gzip", "deflate"])
@pytest.mark.parametrize("empty", [False, True])
def test_scan_clickhouse_decodes_http_response(monkeypatch, content_encoding, empty):
    """Test HTTP decoding for schema and data reads, including empty results."""
    ch_mod = polars_utils.lazy_clickhouse_reader

    expected = pl.DataFrame({"id": [1, 2, 3, 4], "label": ["alpha", "β", None, ""]})
    if empty:
        expected = expected.head(0)
    queries = []
    responses = []

    def fake_post(url, *, params, stream, timeout=None):
        assert stream is True
        query = params["query"]
        queries.append(query)
        table = expected.head(0).to_arrow() if "LIMIT 0" in query else expected.to_arrow()
        sink = io.BytesIO()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table, max_chunksize=2)
        payload = sink.getvalue()
        headers = {}
        if content_encoding == "gzip":
            payload = gzip.compress(payload)
        elif content_encoding == "deflate":
            payload = zlib.compress(payload)
        if content_encoding is not None:
            headers["Content-Encoding"] = content_encoding

        response = requests.Response()
        response.status_code = 200
        response.headers.update(headers)
        # Match Requests' streaming transport: raw reads do not decode by default.
        response.raw = HTTPResponse(body=io.BytesIO(payload), headers=headers, preload_content=False, decode_content=False)
        responses.append(response)
        return response

    # Restore the real reader replaced by patch_clickhouse; only mock the HTTP request.
    monkeypatch.setattr(ch_mod, "get_batch_reader_http", get_batch_reader_http)
    monkeypatch.setattr(ch_mod.requests, "post", fake_post)
    try:
        actual = cpl.scan_clickhouse("SELECT id, label FROM events", FAKE_URL, FAKE_PARAMS).collect()
        assert_frame_equal(actual, expected)
        assert len(queries) == 2
        assert "LIMIT 0" in queries[0]
        assert all(query.endswith("FORMAT ArrowStream") for query in queries)
    finally:
        for response in responses:
            response.close()


def _capture_reader_calls(monkeypatch):
    calls = []

    def capturing_reader(query, url, params, **kwargs):
        calls.append({"query": query, "params": dict(params), "kwargs": dict(kwargs)})
        return fake_get_batch_reader_http(query, url, params, **kwargs)

    monkeypatch.setattr(polars_utils.lazy_clickhouse_reader, "get_batch_reader_http", capturing_reader)
    return calls


def test_scan_clickhouse_forwards_timeout_to_schema_and_data(monkeypatch):
    calls = _capture_reader_calls(monkeypatch)
    timeout = (2, 30)

    cpl.scan_clickhouse("SELECT CenterID FROM CC_Bond", FAKE_URL, FAKE_PARAMS, timeout=timeout).collect()

    assert len(calls) == 2
    assert all(call["kwargs"]["timeout"] == timeout for call in calls)


def test_scan_clickhouse_uses_fetch_size_as_block_size(monkeypatch):
    calls = _capture_reader_calls(monkeypatch)

    cpl.scan_clickhouse("SELECT CenterID FROM CC_Bond", FAKE_URL, FAKE_PARAMS, fetch_size=321).collect()

    data_calls = [call for call in calls if "LIMIT 0" not in call["query"]]
    assert [call["params"]["max_block_size"] for call in data_calls] == [321]


@pytest.mark.skipif(not POLARS_HAS_COLLECT_BATCHES, reason="collect_batches requires Polars >= 1.34.0")
def test_scan_clickhouse_runtime_batch_size_overrides_fetch_size(monkeypatch):
    ch_mod = polars_utils.lazy_clickhouse_reader
    original_register = ch_mod.register_io_source_with_is_pure
    runtime_batch_sizes = []

    def capturing_register(source, *args, **kwargs):
        def capturing_source(with_columns, predicate, n_rows, batch_size):
            runtime_batch_sizes.append(batch_size)
            yield from source(with_columns, predicate, n_rows, batch_size)

        return original_register(capturing_source, *args, **kwargs)

    monkeypatch.setattr(ch_mod, "register_io_source_with_is_pure", capturing_register)
    calls = _capture_reader_calls(monkeypatch)
    lf = cpl.scan_clickhouse("SELECT CenterID FROM CC_Bond", FAKE_URL, FAKE_PARAMS, fetch_size=200000)

    list(lf.collect_batches(chunk_size=2))

    data_calls = [call for call in calls if "LIMIT 0" not in call["query"]]
    assert runtime_batch_sizes
    assert [call["params"]["max_block_size"] for call in data_calls] == runtime_batch_sizes


def test_scan_clickhouse_preserves_explicit_max_block_size(monkeypatch):
    calls = _capture_reader_calls(monkeypatch)
    params = {"max_block_size": 777}

    cpl.scan_clickhouse("SELECT CenterID FROM CC_Bond", FAKE_URL, params, fetch_size=321).collect()

    assert all(call["params"]["max_block_size"] == 777 for call in calls)
    assert params == {"max_block_size": 777}


def test_basic_query():
    """Test simple query with WHERE clause."""
    base_query = """
    SELECT CenterID, Centre, EventYear
    FROM CC_Bond
    """
    sql_query = base_query + " WHERE EventYear > 2015"

    df = get_cc_bond_df()
    equivalent_filters = [pl.col("EventYear") > 2015]
    expected = df.filter(*equivalent_filters).select(["CenterID", "Centre", "EventYear"])
    actual = cpl.scan_clickhouse(sql_query, FAKE_URL, FAKE_PARAMS).collect()
    actual_from_query = cpl.scan_clickhouse(base_query, FAKE_URL, FAKE_PARAMS).filter(*equivalent_filters).collect()
    assert_frame_equal(actual, expected)
    assert_frame_equal(actual_from_query, expected)


def test_column_selection():
    """Test column selection and LIKE filter."""
    base_query = """
    SELECT *
    FROM CC_Bond
    """
    sql_query = base_query + " WHERE Centre LIKE '%Da%'"

    cols = ["CenterID", "Centre", "EventDate"]

    df = get_cc_bond_df()
    equivalent_filters = [pl.col("Centre").str.contains("Da")]
    expected = df.filter(*equivalent_filters).select(cols)

    actual = cpl.scan_clickhouse(sql_query, FAKE_URL, FAKE_PARAMS).select(cols).collect()
    actual_from_query = cpl.scan_clickhouse(base_query, FAKE_URL, FAKE_PARAMS).select(cols).filter(*equivalent_filters).collect()

    actual = actual.sort(cols)
    expected = expected.sort(cols)
    actual_from_query = actual_from_query.sort(cols)
    assert_frame_equal(actual, expected)
    assert_frame_equal(actual_from_query, expected)


def test_complex_query():
    """Test complex query with AND and OR."""
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

    actual = cpl.scan_clickhouse(sql_query, FAKE_URL, FAKE_PARAMS).select(cols).collect()
    actual_from_query = cpl.scan_clickhouse(base_query, FAKE_URL, FAKE_PARAMS).select(cols).filter(*equivalent_filters).collect()

    actual = actual.sort(cols)
    expected = expected.sort(cols)
    actual_from_query = actual_from_query.sort(cols)

    assert_frame_equal(actual, expected)
    assert_frame_equal(actual_from_query, expected)


def test_not_null():
    """Test NOT NULL filter."""
    base_query = """
    SELECT CenterID, Centre, EventDate
    FROM CC_Bond
    """
    sql_query = base_query + " WHERE EventDate IS NOT NULL"

    df = get_cc_bond_df()
    equivalent_filters = [pl.col("EventDate").is_not_null()]
    expected = df.select(["CenterID", "Centre", "EventDate"]).filter(*equivalent_filters)

    actual = cpl.scan_clickhouse(sql_query, FAKE_URL, FAKE_PARAMS).collect()
    actual_from_query = cpl.scan_clickhouse(base_query, FAKE_URL, FAKE_PARAMS).filter(*equivalent_filters).collect()

    assert_frame_equal(actual, expected)
    assert_frame_equal(actual_from_query, expected)


def test_empty_df():
    """Test functionality on a query that returns no records."""
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

    actual = cpl.scan_clickhouse(sql_query, FAKE_URL, FAKE_PARAMS).select(cols).collect()
    actual_from_query = cpl.scan_clickhouse(base_query, FAKE_URL, FAKE_PARAMS).filter(*equivalent_filters).select(cols).collect()

    assert_frame_equal(actual, expected)
    assert_frame_equal(actual_from_query, expected)


def test_in_operator():
    """Test IN operator."""
    base_query = """
    SELECT CenterID, Centre, EventYear, EventDate, EventDayOfWeek
    FROM CC_Bond
    """

    cols = ["CenterID", "Centre", "EventYear", "EventDate", "EventDayOfWeek"]
    sql_query = base_query + " WHERE EventDayOfWeek IN ('Fri', 'Sat', 'Sun')"

    df = get_cc_bond_df()
    equivalent_filters = [pl.col("EventDayOfWeek").is_in(["Fri", "Sat", "Sun"])]
    expected = df.filter(*equivalent_filters).select(cols)

    actual = cpl.scan_clickhouse(sql_query, FAKE_URL, FAKE_PARAMS).collect()
    actual_from_query = cpl.scan_clickhouse(base_query, FAKE_URL, FAKE_PARAMS).filter(*equivalent_filters).collect()

    actual = actual.sort(cols)
    expected = expected.sort(cols)
    actual_from_query = actual_from_query.sort(cols)

    assert_frame_equal(actual, expected)
    assert_frame_equal(actual_from_query, expected)


def test_not_like():
    """Test NOT LIKE operator."""
    base_query = """
    SELECT CenterID, Centre, EventYear, EventName
    FROM CC_Bond
    """
    sql_query = base_query + " WHERE EventName NOT LIKE '%Day%'"

    cols = ["CenterID", "Centre", "EventYear", "EventName"]
    df = get_cc_bond_df()
    equivalent_filters = [~pl.col("EventName").str.contains("Day")]
    expected = df.filter(*equivalent_filters).select(cols)

    actual = cpl.scan_clickhouse(sql_query, FAKE_URL, FAKE_PARAMS).collect()
    actual_from_query = cpl.scan_clickhouse(base_query, FAKE_URL, FAKE_PARAMS).filter(*equivalent_filters).collect()

    actual = actual.sort(cols)
    expected = expected.sort(cols)
    actual_from_query = actual_from_query.sort(cols)

    assert_frame_equal(actual, expected)
    assert_frame_equal(actual_from_query, expected)


def test_head_pushdown_without_predicate():
    """Verify that head() pushes LIMIT when there's no predicate."""
    captured_queries = []

    original_fake = fake_get_batch_reader_http

    def capturing_fake(query, url, params, **kwargs):
        captured_queries.append(query)
        return original_fake(query, url, params)

    # Temporarily replace the already-patched function
    import polars_io_tools.io_sources.lazy_clickhouse_reader as ch_mod

    prev = ch_mod.get_batch_reader_http
    ch_mod.get_batch_reader_http = capturing_fake
    try:
        query = "SELECT * FROM EmployeeTbl"
        lf = cpl.scan_clickhouse(query, FAKE_URL, FAKE_PARAMS)

        result = lf.head(5).collect()

        data_queries = [q for q in captured_queries if "LIMIT 0" not in q]
        assert len(data_queries) == 1

        sql = data_queries[0]
        assert "LIMIT" in sql.upper(), f"Expected LIMIT in SQL, got: {sql}"
        assert result.height == 5
    finally:
        ch_mod.get_batch_reader_http = prev


def test_dt_date_filter_pushdown():
    """`.dt.date()` pushes down as a ``Date32`` cast rather than being dropped (issue #51)."""
    ch_mod = polars_utils.lazy_clickhouse_reader

    captured_queries: list[str] = []
    prev = ch_mod.get_batch_reader_http

    def capturing_fake(query, url, params, **kwargs):
        captured_queries.append(query)
        if "LIMIT 0" in query:
            return fake_get_batch_reader_http(query, url, params)
        # Data query uses ClickHouse-only Date32 syntax DuckDB can't parse; return an empty
        # stream with the full schema so the generator completes and we can inspect the SQL.
        return fake_get_batch_reader_http("SELECT * FROM CC_Bond LIMIT 0", url, params)

    ch_mod.get_batch_reader_http = capturing_fake
    try:
        lf = cpl.scan_clickhouse("SELECT * FROM CC_Bond", FAKE_URL, FAKE_PARAMS)
        captured_queries.clear()

        lf.filter(pl.col("EventDate").dt.date() == date(2016, 1, 1)).collect()

        data_queries = [q for q in captured_queries if "LIMIT 0" not in q]
        assert len(data_queries) == 1
        sql = data_queries[0]
        assert "Date32" in sql, f"Expected a Date32 cast in pushed-down SQL, got: {sql}"
        assert "EventDate" in sql
        assert "2016-01-01" in sql
    finally:
        ch_mod.get_batch_reader_http = prev


def test_head_zero_skips_query():
    """head(0) should return an empty DataFrame without sending a query to ClickHouse."""
    ch_mod = polars_utils.lazy_clickhouse_reader

    captured_queries: list[str] = []
    prev = ch_mod.get_batch_reader_http

    def capturing_fake(query, url, params, **kwargs):
        captured_queries.append(query)
        return fake_get_batch_reader_http(query, url, params)

    ch_mod.get_batch_reader_http = capturing_fake
    try:
        query = "SELECT * FROM EmployeeTbl"
        lf = cpl.scan_clickhouse(query, FAKE_URL, FAKE_PARAMS)

        # Reset captured queries after schema fetch
        captured_queries.clear()

        result = lf.head(0).collect()

        assert result.height == 0
        assert len(captured_queries) == 0, f"Expected no queries after schema fetch, but {len(captured_queries)} were made"
    finally:
        ch_mod.get_batch_reader_http = prev


def test_head_zero_with_column_selection():
    """head(0) with column selection should return the correct empty schema."""
    ch_mod = polars_utils.lazy_clickhouse_reader

    captured_queries: list[str] = []
    prev = ch_mod.get_batch_reader_http

    def capturing_fake(query, url, params, **kwargs):
        captured_queries.append(query)
        return fake_get_batch_reader_http(query, url, params)

    ch_mod.get_batch_reader_http = capturing_fake
    try:
        query = "SELECT * FROM EmployeeTbl"
        lf = cpl.scan_clickhouse(query, FAKE_URL, FAKE_PARAMS)

        captured_queries.clear()

        result = lf.select(["id", "name"]).head(0).collect()

        assert result.height == 0
        assert set(result.columns) == {"id", "name"}
        assert len(captured_queries) == 0
    finally:
        ch_mod.get_batch_reader_http = prev
