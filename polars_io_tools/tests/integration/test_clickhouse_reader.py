"""
Integration tests for the lazy Polars ClickHouse reader.

This module contains integration tests that require an actual ClickHouse connection.
These tests verify that scan_clickhouse works correctly against real databases.

The tests are self-contained: they seed their own database and tables with a small
deterministic dataset and drop them afterwards, so no pre-existing data is required.

Note: These tests are excluded from the regular test suite by default and must be
run explicitly when a ClickHouse endpoint is available via:
    pytest --clickhouse-url=<url> --clickhouse-user=<user> --clickhouse-password=<password>
"""

import logging
from datetime import UTC, datetime

import polars as pl
import pytest
import requests
from polars.testing import assert_frame_equal

import polars_io_tools as cpl

pytestmark = pytest.mark.clickhouse_required

TEST_DB = "polars_io_tools_test"
QUOTE_BAR_TABLE = f"{TEST_DB}.quote_bar_10m"
TRADE_BAR_TABLE = f'"{TEST_DB}"."trade_bar_5m"'
# Raw tick quotes carry nanosecond event timestamps (bars are coarse aggregates, so they don't).
RAW_QUOTES_TABLE = f"{TEST_DB}.raw_quotes"


def _clickhouse_command(sql: str, url: str, params: dict) -> str:
    """Execute a SQL statement against ClickHouse via HTTP POST."""
    r = requests.post(url, params=params, data=sql.encode("utf-8"))
    r.raise_for_status()
    return r.text.strip()


_PUSHDOWN_LOG_PREFIX = "Executing SQL with pushdown:"


def _pushed_down_sql(caplog) -> list[str]:
    """Return the SQL statements the reader logged as pushed down to ClickHouse."""
    return [record.message.split(_PUSHDOWN_LOG_PREFIX, 1)[1].strip() for record in caplog.records if _PUSHDOWN_LOG_PREFIX in record.message]


@pytest.fixture(scope="module")
def clickhouse_url_and_params(clickhouse_url, clickhouse_user, clickhouse_password):
    """Return (url, params) tuple for scan_clickhouse calls."""
    return clickhouse_url, {"user": clickhouse_user, "password": clickhouse_password}


@pytest.fixture(scope="module", autouse=True)
def _seed_tables(clickhouse_url_and_params):
    """Create and populate quote_bar_10m and trade_bar_5m, then drop the database.

    Everything lives in a dedicated database so the tests never touch tables in the
    caller's default database. The database is recreated fresh for idempotent reruns.
    """
    url, params = clickhouse_url_and_params
    _clickhouse_command(f"DROP DATABASE IF EXISTS {TEST_DB}", url, params)
    _clickhouse_command(f"CREATE DATABASE {TEST_DB}", url, params)

    _clickhouse_command(
        f"""
        CREATE TABLE {QUOTE_BAR_TABLE} (
            active Bool,
            instrument String,
            run_session_id Int64,
            bar_start_time DateTime64(6),
            avg_spr Float64,
            avg_mid Float64,
            avg_ask_size Float64,
            close_mid Float64,
            count_mid_changes Int64
        ) ENGINE = MergeTree() ORDER BY (instrument, run_session_id)
        """,
        url,
        params,
    )
    _clickhouse_command(
        f"""
        INSERT INTO {QUOTE_BAR_TABLE}
            (active, instrument, run_session_id, bar_start_time, avg_spr, avg_mid, avg_ask_size, close_mid, count_mid_changes)
        VALUES
            (1, 'AAA', 1, '2024-01-01 00:00:00.000000', 0.10, 10.0, 5.0, 10.1, 5),
            (1, 'BBB', 1, '2024-01-01 00:01:00.000000', 0.20, 20.0, 6.0, 20.1, 3),
            (0, 'CCC', 2, '2024-01-02 00:00:00.000000', 0.30, 30.0, 7.0, 30.1, 0),
            (1, 'AAA', 2, '2024-01-02 00:01:00.000000', 0.15, 15.0, 8.0, 15.1, 2),
            (0, 'BBB', 3, '2024-01-03 00:00:00.000000', 0.00, 0.0, 9.0, 0.0, 1),
            (1, 'CCC', 3, '2024-01-03 00:01:00.000000', 0.25, 25.0, 10.0, 25.1, 4)
        """,
        url,
        params,
    )

    _clickhouse_command(
        f"""
        CREATE TABLE {TEST_DB}.trade_bar_5m (
            instrument String,
            run_session_id Int64,
            avg_price Float64,
            volume Int64,
            vwap Float64
        ) ENGINE = MergeTree() ORDER BY (instrument, run_session_id)
        """,
        url,
        params,
    )
    _clickhouse_command(
        f"""
        INSERT INTO {TEST_DB}.trade_bar_5m (instrument, run_session_id, avg_price, volume, vwap)
        VALUES
            ('AAA', 1, 10.5, 100, 10.4),
            ('BBB', 1, 20.5, 200, 20.4),
            ('CCC', 2, 30.5, 300, 30.4),
            ('AAA', 2, 15.5, 150, 15.4)
        """,
        url,
        params,
    )

    # Raw quotes: tick-level rows with a nanosecond, tz-aware event timestamp. The two AAA rows
    # share the same microsecond but differ in the nanosecond digits, which exercises datetime
    # pushdown precision (a microsecond-truncating pushdown would drop the sub-microsecond row).
    _clickhouse_command(
        f"""
        CREATE TABLE {RAW_QUOTES_TABLE} (
            instrument String,
            event_time DateTime64(9, 'UTC'),
            bid Float64,
            ask Float64
        ) ENGINE = MergeTree() ORDER BY (instrument, event_time)
        """,
        url,
        params,
    )
    _clickhouse_command(
        f"""
        INSERT INTO {RAW_QUOTES_TABLE} (instrument, event_time, bid, ask)
        VALUES
            ('AAA', '2024-01-15 00:00:00.123456000', 10.00, 10.01),
            ('AAA', '2024-01-15 00:00:00.123456789', 10.02, 10.03),
            ('BBB', '2024-01-16 12:30:00.000000000', 20.00, 20.01)
        """,
        url,
        params,
    )

    yield

    try:
        _clickhouse_command(f"DROP DATABASE IF EXISTS {TEST_DB}", url, params)
    except Exception:
        pass


def test_basic_query(clickhouse_url_and_params):
    """Test that a basic query returns a non-empty DataFrame with expected columns."""
    url, params = clickhouse_url_and_params
    sql_query = f"""
    SELECT active, instrument, run_session_id, bar_start_time, avg_mid
    FROM {QUOTE_BAR_TABLE}
    LIMIT 100
    """
    result = cpl.scan_clickhouse(sql_query, url, params).collect()

    assert result.shape[0] > 0
    assert result.columns == ["active", "instrument", "run_session_id", "bar_start_time", "avg_mid"]


def test_column_selection(clickhouse_url_and_params):
    """Test that selecting a subset of columns via Polars .select() works."""
    url, params = clickhouse_url_and_params
    sql_query = f"""
    SELECT *
    FROM {QUOTE_BAR_TABLE}
    LIMIT 100
    """
    cols = ["instrument", "avg_mid", "avg_spr"]
    result = cpl.scan_clickhouse(sql_query, url, params).select(cols).collect()

    assert result.columns == cols
    assert result.shape[0] > 0


def test_filter_equality(clickhouse_url_and_params):
    """Test that SQL WHERE and Polars filter on equality produce equivalent results."""
    url, params = clickhouse_url_and_params

    # Approach 1: Filter in ClickHouse SQL
    sql_with_where = f"""
    SELECT instrument, run_session_id, avg_mid
    FROM {QUOTE_BAR_TABLE}
    WHERE active = true
    ORDER BY instrument, run_session_id, avg_mid
    """
    result_sql = cpl.scan_clickhouse(sql_with_where, url, params).collect()

    # Approach 2: Filter via Polars lazy filter (predicate pushdown expected)
    sql_base = f"""
    SELECT instrument, run_session_id, avg_mid, active
    FROM {QUOTE_BAR_TABLE}
    ORDER BY instrument, run_session_id, avg_mid
    """
    result_polars = (
        cpl.scan_clickhouse(sql_base, url, params).filter(pl.col("active") == True).select(["instrument", "run_session_id", "avg_mid"]).collect()
    )

    # Verify each approach individually
    assert result_sql.shape[0] > 0
    assert result_polars.shape[0] > 0

    # Compare results from both approaches
    sort_cols = ["instrument", "run_session_id", "avg_mid"]
    assert_frame_equal(result_sql.sort(sort_cols), result_polars.sort(sort_cols))


def test_filter_comparison(clickhouse_url_and_params):
    """Test that SQL WHERE and Polars filter on comparison produce equivalent results."""
    url, params = clickhouse_url_and_params

    # Approach 1: Filter in ClickHouse SQL
    sql_with_where = f"""
    SELECT instrument, avg_mid, avg_spr
    FROM {QUOTE_BAR_TABLE}
    WHERE avg_mid > 0
    ORDER BY instrument, avg_mid, avg_spr
    """
    result_sql = cpl.scan_clickhouse(sql_with_where, url, params).collect()

    # Approach 2: Filter via Polars lazy filter (predicate pushdown expected)
    sql_base = f"""
    SELECT instrument, avg_mid, avg_spr
    FROM {QUOTE_BAR_TABLE}
    ORDER BY instrument, avg_mid, avg_spr
    """
    result_polars = cpl.scan_clickhouse(sql_base, url, params).filter(pl.col("avg_mid") > 0).collect()

    # Verify each approach individually
    assert result_sql.shape[0] > 0
    assert result_polars.shape[0] > 0
    assert (result_sql["avg_mid"] > 0).all()
    assert (result_polars["avg_mid"] > 0).all()

    # Compare results from both approaches
    sort_cols = ["instrument", "avg_mid", "avg_spr"]
    assert_frame_equal(result_sql.sort(sort_cols), result_polars.sort(sort_cols))


def test_filter_and(clickhouse_url_and_params):
    """Test that SQL WHERE with AND and Polars filter produce equivalent results."""
    url, params = clickhouse_url_and_params
    select_cols = ["instrument", "run_session_id", "avg_mid"]

    # Approach 1: Filter in ClickHouse SQL
    sql_with_where = f"""
    SELECT instrument, run_session_id, avg_mid
    FROM {QUOTE_BAR_TABLE}
    WHERE active = true AND avg_mid > 0
    ORDER BY instrument, run_session_id, avg_mid
    """
    result_sql = cpl.scan_clickhouse(sql_with_where, url, params).collect()

    # Approach 2: Filter via Polars lazy filter (predicate pushdown expected)
    sql_base = f"""
    SELECT instrument, run_session_id, avg_mid, active
    FROM {QUOTE_BAR_TABLE}
    ORDER BY instrument, run_session_id, avg_mid
    """
    result_polars = (
        cpl.scan_clickhouse(sql_base, url, params).filter((pl.col("active") == True) & (pl.col("avg_mid") > 0)).select(select_cols).collect()
    )

    # Verify each approach individually
    assert result_sql.shape[0] > 0
    assert result_polars.shape[0] > 0
    assert (result_sql["avg_mid"] > 0).all()
    assert (result_polars["avg_mid"] > 0).all()

    # Compare results from both approaches
    assert_frame_equal(result_sql.sort(select_cols), result_polars.sort(select_cols))


def test_filter_in_operator(clickhouse_url_and_params):
    """Test that SQL WHERE IN and Polars is_in filter produce equivalent results."""
    url, params = clickhouse_url_and_params

    # First get some instrument values to use in IN clause
    sql_distinct = f"""
    SELECT DISTINCT instrument
    FROM {QUOTE_BAR_TABLE}
    LIMIT 5
    """
    instruments_df = cpl.scan_clickhouse(sql_distinct, url, params).collect()
    instruments = instruments_df["instrument"].to_list()[:2]

    if len(instruments) < 2:
        pytest.skip("Not enough distinct instruments for IN test")

    select_cols = ["instrument", "avg_mid", "bar_start_time"]
    in_list = ", ".join(f"'{v}'" for v in instruments)

    # Approach 1: Filter in ClickHouse SQL
    sql_with_where = f"""
    SELECT instrument, avg_mid, bar_start_time
    FROM {QUOTE_BAR_TABLE}
    WHERE instrument IN ({in_list})
    ORDER BY instrument, avg_mid, bar_start_time
    """
    result_sql = cpl.scan_clickhouse(sql_with_where, url, params).collect()

    # Approach 2: Filter via Polars lazy filter (predicate pushdown expected)
    sql_base = f"""
    SELECT instrument, avg_mid, bar_start_time
    FROM {QUOTE_BAR_TABLE}
    ORDER BY instrument, avg_mid, bar_start_time
    """
    result_polars = cpl.scan_clickhouse(sql_base, url, params).filter(pl.col("instrument").is_in(instruments)).collect()

    # Verify each approach individually
    assert result_sql.shape[0] > 0
    assert result_polars.shape[0] > 0
    assert set(result_sql["instrument"].unique().to_list()).issubset(set(instruments))
    assert set(result_polars["instrument"].unique().to_list()).issubset(set(instruments))

    # Compare results from both approaches
    assert_frame_equal(result_sql.sort(select_cols), result_polars.sort(select_cols))


def test_filter_not_null(clickhouse_url_and_params):
    """Test IS NOT NULL filter."""
    url, params = clickhouse_url_and_params
    sql_query = f"""
    SELECT instrument, avg_mid, avg_spr
    FROM {QUOTE_BAR_TABLE}
    LIMIT 100
    """
    result = cpl.scan_clickhouse(sql_query, url, params).filter(pl.col("avg_mid").is_not_null()).collect()

    assert result.shape[0] > 0
    assert result["avg_mid"].null_count() == 0


def test_complex_filter(clickhouse_url_and_params):
    """Test that complex SQL WHERE and Polars filter produce equivalent results."""
    url, params = clickhouse_url_and_params
    select_cols = ["instrument", "run_session_id", "avg_mid", "avg_spr", "count_mid_changes"]

    # Approach 1: Filter in ClickHouse SQL
    sql_with_where = f"""
    SELECT instrument, run_session_id, avg_mid, avg_spr, count_mid_changes
    FROM {QUOTE_BAR_TABLE}
    WHERE active = true
      AND avg_mid > 0
      AND count_mid_changes >= 0
    ORDER BY instrument, run_session_id, avg_mid, avg_spr, count_mid_changes
    """
    result_sql = cpl.scan_clickhouse(sql_with_where, url, params).collect()

    # Approach 2: Filter via Polars lazy filter (predicate pushdown expected)
    sql_base = f"""
    SELECT instrument, run_session_id, avg_mid, avg_spr, count_mid_changes, active
    FROM {QUOTE_BAR_TABLE}
    ORDER BY instrument, run_session_id, avg_mid, avg_spr, count_mid_changes
    """
    result_polars = (
        cpl.scan_clickhouse(sql_base, url, params)
        .filter((pl.col("active") == True) & (pl.col("avg_mid") > 0) & (pl.col("count_mid_changes") >= 0))
        .select(select_cols)
        .collect()
    )

    # Verify each approach individually
    assert result_sql.columns == select_cols
    assert result_polars.columns == select_cols
    assert (result_sql["avg_mid"] > 0).all()
    assert (result_sql["count_mid_changes"] >= 0).all()
    assert (result_polars["avg_mid"] > 0).all()
    assert (result_polars["count_mid_changes"] >= 0).all()

    # Compare results from both approaches
    assert_frame_equal(result_sql.sort(select_cols), result_polars.sort(select_cols))


def test_empty_result(clickhouse_url_and_params):
    """Test that a query returning no rows produces an empty DataFrame with correct schema."""
    url, params = clickhouse_url_and_params
    # Use a filter that should match nothing
    sql_query = f"""
    SELECT instrument, avg_mid, bar_start_time
    FROM {QUOTE_BAR_TABLE}
    WHERE run_session_id = -999999
    """
    result = cpl.scan_clickhouse(sql_query, url, params).collect()

    assert result.shape[0] == 0
    assert result.columns == ["instrument", "avg_mid", "bar_start_time"]


def test_head_pushdown(clickhouse_url_and_params):
    """Test that .head() pushes down a LIMIT to ClickHouse."""
    url, params = clickhouse_url_and_params
    sql_query = f"""
    SELECT instrument, avg_mid, bar_start_time
    FROM {QUOTE_BAR_TABLE}
    """
    result = cpl.scan_clickhouse(sql_query, url, params).head(3).collect()

    assert result.shape[0] == 3


def test_head_pushdown_with_log(clickhouse_url_and_params, caplog):
    """Test that head pushdown is reflected in the executed SQL."""
    url, params = clickhouse_url_and_params
    caplog.set_level(logging.DEBUG)
    sql_query = f"""
    SELECT instrument, avg_mid
    FROM {QUOTE_BAR_TABLE}
    """
    result = cpl.scan_clickhouse(sql_query, url, params).head(3).collect()

    assert result.shape[0] == 3
    assert any("Executing SQL with pushdown" in record.message and "LIMIT" in record.message for record in caplog.records)


def test_select_star_quote_bar(clickhouse_url_and_params):
    """Test SELECT * returns all columns for quote_bar_10m."""
    url, params = clickhouse_url_and_params
    sql_query = f"""
    SELECT *
    FROM {QUOTE_BAR_TABLE}
    LIMIT 10
    """
    result = cpl.scan_clickhouse(sql_query, url, params).collect()

    expected_cols = [
        "active",
        "instrument",
        "run_session_id",
        "bar_start_time",
        "avg_spr",
        "avg_mid",
        "avg_ask_size",
        "close_mid",
        "count_mid_changes",
    ]
    assert result.columns == expected_cols
    assert result.shape[0] > 0


def test_nonexistent_column(clickhouse_url_and_params):
    """Test that querying a nonexistent column raises an error."""
    url, params = clickhouse_url_and_params
    sql_query = f"""
    SELECT does_not_exist
    FROM {QUOTE_BAR_TABLE}
    LIMIT 1
    """
    with pytest.raises(Exception):
        cpl.scan_clickhouse(sql_query, url, params).collect()


def test_alias_with_filter(clickhouse_url_and_params):
    """Test that column aliases work correctly with filters."""
    url, params = clickhouse_url_and_params
    sql_query = f"""
    SELECT
        instrument AS inst,
        avg_mid AS mid_price,
        run_session_id AS session_id
    FROM {QUOTE_BAR_TABLE}
    LIMIT 100
    """
    lf = cpl.scan_clickhouse(sql_query, url, params)
    result = lf.filter(pl.col("mid_price") > 0).collect()

    assert result.columns == ["inst", "mid_price", "session_id"]
    assert result.shape[0] > 0
    assert (result["mid_price"] > 0).all()


def test_join(clickhouse_url_and_params):
    """Test a SQL JOIN between quote_bar_10m and trade_bar_5m."""
    url, params = clickhouse_url_and_params
    sql_query = f"""
    SELECT
        q.instrument AS q_instrument,
        q.run_session_id AS q_session_id,
        q.bar_start_time AS q_bar_start_time,
        q.avg_mid,
        q.avg_spr,
        t.avg_price,
        t.volume,
        t.vwap
    FROM {QUOTE_BAR_TABLE} AS q
    INNER JOIN {TRADE_BAR_TABLE} AS t
        ON q.instrument = t.instrument
        AND q.run_session_id = t.run_session_id
    LIMIT 100
    """
    result = cpl.scan_clickhouse(sql_query, url, params).collect()

    assert result.shape[0] > 0
    expected_cols = [
        "q_instrument",
        "q_session_id",
        "q_bar_start_time",
        "avg_mid",
        "avg_spr",
        "avg_price",
        "volume",
        "vwap",
    ]
    assert result.columns == expected_cols


def test_join_with_filter(clickhouse_url_and_params):
    """Test that SQL WHERE and Polars filter on a JOIN produce equivalent results."""
    url, params = clickhouse_url_and_params
    select_cols = ["instrument", "run_session_id", "avg_mid", "avg_price", "volume"]

    # Approach 1: Filter in ClickHouse SQL
    sql_with_where = f"""
    SELECT
        q.instrument,
        q.run_session_id,
        q.avg_mid,
        t.avg_price,
        t.volume
    FROM {QUOTE_BAR_TABLE} AS q
    INNER JOIN {TRADE_BAR_TABLE} AS t
        ON q.instrument = t.instrument
        AND q.run_session_id = t.run_session_id
    WHERE q.avg_mid > 0
    ORDER BY q.instrument, q.run_session_id, q.avg_mid, t.avg_price, t.volume
    """
    result_sql = cpl.scan_clickhouse(sql_with_where, url, params).collect()

    # Approach 2: Filter via Polars lazy filter (predicate pushdown expected)
    sql_base = f"""
    SELECT
        q.instrument,
        q.run_session_id,
        q.avg_mid,
        t.avg_price,
        t.volume
    FROM {QUOTE_BAR_TABLE} AS q
    INNER JOIN {TRADE_BAR_TABLE} AS t
        ON q.instrument = t.instrument
        AND q.run_session_id = t.run_session_id
    ORDER BY q.instrument, q.run_session_id, q.avg_mid, t.avg_price, t.volume
    """
    result_polars = cpl.scan_clickhouse(sql_base, url, params).filter(pl.col("avg_mid") > 0).collect()

    # Verify each approach individually
    assert result_sql.columns == select_cols
    assert result_polars.columns == select_cols
    assert result_sql.shape[0] > 0
    assert result_polars.shape[0] > 0
    assert (result_sql["avg_mid"] > 0).all()
    assert (result_polars["avg_mid"] > 0).all()

    # Compare results from both approaches
    assert_frame_equal(result_sql.sort(select_cols), result_polars.sort(select_cols))


def test_join_select_columns(clickhouse_url_and_params):
    """Test a JOIN query with Polars-side column selection."""
    url, params = clickhouse_url_and_params
    sql_query = f"""
    SELECT
        q.instrument AS q_instrument,
        q.avg_mid,
        q.avg_spr,
        t.avg_price,
        t.volume,
        t.vwap
    FROM {QUOTE_BAR_TABLE} AS q
    INNER JOIN {TRADE_BAR_TABLE} AS t
        ON q.instrument = t.instrument
        AND q.run_session_id = t.run_session_id
    LIMIT 100
    """
    cols = ["q_instrument", "avg_mid", "avg_price"]
    result = cpl.scan_clickhouse(sql_query, url, params).select(cols).collect()

    assert result.columns == cols


def test_column_pushdown_logged(clickhouse_url_and_params, caplog):
    """Verify that column selection pushdown appears in debug logs."""
    url, params = clickhouse_url_and_params
    caplog.set_level(logging.DEBUG)

    sql_query = f"""
    SELECT *
    FROM {QUOTE_BAR_TABLE}
    LIMIT 50
    """
    cpl.scan_clickhouse(sql_query, url, params).select(["instrument", "avg_mid"]).collect()

    assert any(
        "Executing SQL with pushdown" in record.message and "instrument" in record.message and "avg_mid" in record.message
        for record in caplog.records
    )


def test_predicate_pushdown_logged(clickhouse_url_and_params, caplog):
    """Verify that predicate pushdown appears in debug logs."""
    url, params = clickhouse_url_and_params
    caplog.set_level(logging.DEBUG)

    sql_query = f"""
    SELECT instrument, avg_mid, active
    FROM {QUOTE_BAR_TABLE}
    LIMIT 100
    """
    cpl.scan_clickhouse(sql_query, url, params).filter(pl.col("active") == True).collect()

    # The base query has no WHERE clause, so a pushed-down statement containing a WHERE
    # predicate on `active` proves the predicate reached ClickHouse rather than being
    # applied by Polars after the scan.
    pushed = _pushed_down_sql(caplog)
    assert pushed, "expected the reader to log a pushed-down SQL statement"
    assert any("WHERE" in sql.upper() and "active" in sql.lower() for sql in pushed)


def test_filter_timezone_aware_datetime(clickhouse_url_and_params):
    """A timezone-aware datetime filter on a DateTime64(_, 'UTC') column pushes down and works.

    The reader previously emitted an offset-bearing string that ClickHouse rejects (HTTP 400);
    it must now round-trip and return the expected rows.
    """
    url, params = clickhouse_url_and_params
    base = f"SELECT instrument, event_time FROM {RAW_QUOTES_TABLE}"
    result = cpl.scan_clickhouse(base, url, params).filter(pl.col("event_time") >= datetime(2024, 1, 16, tzinfo=UTC)).collect()

    assert result["instrument"].to_list() == ["BBB"]


def test_filter_nanosecond_precision_not_truncated(clickhouse_url_and_params, caplog):
    """A microsecond-precision filter on a DateTime64(9) column keeps sub-microsecond rows.

    Equality against a nanosecond column matches the whole microsecond bucket, so both the
    ``.123456000`` and ``.123456789`` rows qualify. If pushdown truncated the bound to
    microseconds, ClickHouse would drop the ``.123456789`` row before the client-side filter,
    so returning both rows proves the nanosecond bounds survive pushdown.
    """
    url, params = clickhouse_url_and_params
    caplog.set_level(logging.DEBUG)

    base = f"SELECT instrument, event_time, bid FROM {RAW_QUOTES_TABLE}"
    result = cpl.scan_clickhouse(base, url, params).filter(pl.col("event_time") == datetime(2024, 1, 15, 0, 0, 0, 123456, tzinfo=UTC)).collect()

    # Both AAA rows share microsecond .123456 but differ in nanoseconds; both must be returned.
    assert result.height == 2
    assert set(result["instrument"].to_list()) == {"AAA"}

    # The pushed SQL must carry the full nanosecond upper bound, not a collapsed microsecond one.
    pushed = _pushed_down_sql(caplog)
    assert any("toDateTime64('2024-01-15 00:00:00.123456999', 9, 'UTC')" in sql for sql in pushed), pushed
