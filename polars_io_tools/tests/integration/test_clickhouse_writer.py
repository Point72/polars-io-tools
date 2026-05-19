"""
Integration tests for the lazy Polars ClickHouse writer (sink_clickhouse).

This module contains integration tests that require an actual ClickHouse connection.
These tests verify that sink_clickhouse works correctly against real databases,
including roundtrip tests that write data and read it back with scan_clickhouse.

Prerequisites:
- ClickHouse HTTP endpoint URL, username, and password
- Write permission to the target database

Note: These tests are excluded from the regular test suite by default and must be
run explicitly when database access is available via:
    pytest --clickhouse-url=<url> --clickhouse-user=<user> --clickhouse-password=<password>
"""

import uuid
from datetime import date, datetime

import polars as pl
import pytest
import requests
from polars.testing import assert_frame_equal

import polars_io_tools as cpl

pytestmark = pytest.mark.clickhouse_required


def _clickhouse_command(sql: str, url: str, params: dict) -> str:
    """Execute a SQL command against ClickHouse via HTTP POST."""
    r = requests.post(url, params=params, data=sql.encode("utf-8"))
    r.raise_for_status()
    return r.text.strip()


def _cast_to_schema(result: pl.DataFrame, expected: pl.DataFrame) -> pl.DataFrame:
    """Cast result columns to match expected dtypes.

    ClickHouse ArrowStream serializes Date as UInt16 (days since epoch) and
    DateTime64 with varying precision. This helper casts the read-back result
    to the original DataFrame's schema so roundtrip comparisons work.
    """
    return result.cast({col: expected.schema[col] for col in expected.columns})


@pytest.fixture(scope="module")
def ch_params(clickhouse_url, clickhouse_user, clickhouse_password):
    """Return (url, params) tuple for sink/scan_clickhouse calls."""
    return clickhouse_url, {"user": clickhouse_user, "password": clickhouse_password}


@pytest.fixture
def ch_table(ch_params):
    """Yield a unique table name and drop it after the test."""
    url, params = ch_params
    table_name = f"__cpl_test_{uuid.uuid4().hex[:12]}"
    yield table_name
    try:
        _clickhouse_command(f"DROP TABLE IF EXISTS {table_name}", url, params)
    except Exception:
        pass


def _create_basic_table(table_name: str, url: str, params: dict) -> None:
    """Create a table matching _get_basic_df schema."""
    _clickhouse_command(
        f"""
        CREATE TABLE {table_name} (
            id Int64,
            name String,
            score Float64,
            passed Bool,
            test_date Date
        ) ENGINE = MergeTree() ORDER BY id
        """,
        url,
        params,
    )


def _create_all_types_table(table_name: str, url: str, params: dict) -> None:
    """Create a table matching _get_all_types_df schema."""
    _clickhouse_command(
        f"""
        CREATE TABLE {table_name} (
            col_int Int64,
            col_float Float64,
            col_str String,
            col_bool Bool,
            col_date Date,
            col_datetime DateTime64(6)
        ) ENGINE = MergeTree() ORDER BY col_int
        """,
        url,
        params,
    )


def _get_basic_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "score": [85.4, 92.1, 78.3, 90.5, 87.9],
            "passed": [True, True, False, True, True],
            "test_date": [
                date(2022, 1, 1),
                date(2022, 1, 2),
                date(2022, 1, 3),
                date(2022, 1, 4),
                date(2022, 1, 5),
            ],
        }
    )


def _get_all_types_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col_int": pl.Series([1, 2, 3], dtype=pl.Int64),
            "col_float": pl.Series([1.1, 2.2, 3.3], dtype=pl.Float64),
            "col_str": pl.Series(["a", "b", "c"], dtype=pl.String),
            "col_bool": pl.Series([True, False, True], dtype=pl.Boolean),
            "col_date": pl.Series(
                [date(2024, 1, 1), date(2024, 6, 15), date(2024, 12, 31)],
                dtype=pl.Date,
            ),
            "col_datetime": pl.Series(
                [
                    datetime(2024, 1, 1, 12, 0, 0),
                    datetime(2024, 6, 15, 8, 30, 0),
                    datetime(2024, 12, 31, 23, 59, 59),
                ],
                dtype=pl.Datetime("us"),
            ),
        }
    )


def test_basic_roundtrip(ch_params, ch_table):
    """Write data, read it back, verify match."""
    url, params = ch_params
    df = _get_basic_df()
    _create_basic_table(ch_table, url, params)

    cpl.sink_clickhouse(df.lazy(), ch_table, url, params)

    result = cpl.scan_clickhouse(f"SELECT * FROM {ch_table}", url, params).collect()
    assert_frame_equal(_cast_to_schema(result, df).sort("id"), df.sort("id"))


def test_roundtrip_all_types(ch_params, ch_table):
    """Roundtrip with Int, Float, String, Bool, Date, Datetime. Verify values."""
    url, params = ch_params
    df = _get_all_types_df()
    _create_all_types_table(ch_table, url, params)

    cpl.sink_clickhouse(df.lazy(), ch_table, url, params)

    result = cpl.scan_clickhouse(f"SELECT * FROM {ch_table}", url, params).collect()
    assert_frame_equal(
        _cast_to_schema(result, df).sort("col_int"),
        df.sort("col_int"),
    )


def test_batched_write_roundtrip(ch_params, ch_table):
    """Write with chunk_size, read back, verify all rows present."""
    url, params = ch_params
    df = _get_basic_df()
    _create_basic_table(ch_table, url, params)

    cpl.sink_clickhouse(df.lazy(), ch_table, url, params, chunk_size=2)

    result = cpl.scan_clickhouse(f"SELECT * FROM {ch_table}", url, params).collect()
    assert_frame_equal(_cast_to_schema(result, df).sort("id"), df.sort("id"))


def test_comprehensive_type_roundtrip(ch_params, ch_table):
    """Write all supported Polars types to ClickHouse and verify roundtrip values.

    Tests passthrough types (written as-is) and cast types (Duration/Time/Categorical
    are cast before writing). For cast types, we verify the transformed values roundtrip
    correctly through ClickHouse.
    """
    from datetime import time

    url, params = ch_params

    # Build a DataFrame covering all type categories
    df = pl.DataFrame(
        {
            # Integer types
            "col_i8": pl.Series([1, 2, 3], dtype=pl.Int8),
            "col_i16": pl.Series([100, 200, 300], dtype=pl.Int16),
            "col_i32": pl.Series([1000, 2000, 3000], dtype=pl.Int32),
            "col_i64": pl.Series([10_000, 20_000, 30_000], dtype=pl.Int64),
            "col_u8": pl.Series([1, 2, 3], dtype=pl.UInt8),
            "col_u16": pl.Series([100, 200, 300], dtype=pl.UInt16),
            "col_u32": pl.Series([1000, 2000, 3000], dtype=pl.UInt32),
            "col_u64": pl.Series([10_000, 20_000, 30_000], dtype=pl.UInt64),
            # Float types
            "col_f32": pl.Series([1.5, 2.5, 3.5], dtype=pl.Float32),
            "col_f64": pl.Series([1.123456789, 2.987654321, 3.141592653], dtype=pl.Float64),
            # String and boolean
            "col_str": pl.Series(["alpha", "beta", "gamma"], dtype=pl.String),
            "col_bool": pl.Series([True, False, True], dtype=pl.Boolean),
            # Date and Datetime variants
            "col_date": pl.Series(
                [date(2024, 1, 1), date(2024, 6, 15), date(2024, 12, 31)],
                dtype=pl.Date,
            ),
            "col_dt_us": pl.Series(
                [datetime(2024, 1, 1, 12, 0), datetime(2024, 6, 15, 8, 30), datetime(2024, 12, 31, 23, 59)],
                dtype=pl.Datetime("us"),
            ),
            "col_dt_ms": pl.Series(
                [datetime(2024, 1, 1, 12, 0), datetime(2024, 6, 15, 8, 30), datetime(2024, 12, 31, 23, 59)],
                dtype=pl.Datetime("ms"),
            ),
            # Cast types: Duration -> Int64
            "col_dur_us": pl.Series([1_000_000, 2_000_000, 3_000_000], dtype=pl.Duration("us")),
            # Cast types: Time -> Int64
            "col_time": pl.Series([time(12, 0, 0), time(8, 30, 0), time(23, 59, 59)], dtype=pl.Time),
            # Cast types: Categorical -> String
            "col_cat": pl.Series(["x", "y", "z"], dtype=pl.Categorical),
        }
    )

    # Create ClickHouse table with matching schema
    # Passthrough types map directly; cast types need their target ClickHouse type.
    _clickhouse_command(
        f"""
        CREATE TABLE {ch_table} (
            col_i8 Int8,
            col_i16 Int16,
            col_i32 Int32,
            col_i64 Int64,
            col_u8 UInt8,
            col_u16 UInt16,
            col_u32 UInt32,
            col_u64 UInt64,
            col_f32 Float32,
            col_f64 Float64,
            col_str String,
            col_bool Bool,
            col_date Date,
            col_dt_us DateTime64(6),
            col_dt_ms DateTime64(3),
            col_dur_us Int64,
            col_time Int64,
            col_cat String
        ) ENGINE = MergeTree() ORDER BY col_i64
        """,
        url,
        params,
    )

    # Write
    cpl.sink_clickhouse(df.lazy(), ch_table, url, params)

    # Read back
    result = cpl.scan_clickhouse(f"SELECT * FROM {ch_table}", url, params).collect()
    assert result.height == 3
    assert set(result.columns) == set(df.columns)

    # Build the expected DataFrame after casting
    # _prepare_for_clickhouse casts Duration->Int64, Time->Int64, Categorical->String
    expected = df.with_columns(
        pl.col("col_dur_us").cast(pl.Int64),
        pl.col("col_time").cast(pl.Int64),
        pl.col("col_cat").cast(pl.String),
    )

    # Cast the read-back result to match expected dtypes (ClickHouse ArrowStream
    # serializes Date as UInt16, DateTime64 with varying precision, etc.)
    result_cast = _cast_to_schema(result, expected)
    assert_frame_equal(result_cast.sort("col_i64"), expected.sort("col_i64"))


def test_append(ch_params, ch_table):
    """Write twice, verify combined row count."""
    url, params = ch_params
    df = _get_basic_df()
    _create_basic_table(ch_table, url, params)

    cpl.sink_clickhouse(df.lazy(), ch_table, url, params)
    cpl.sink_clickhouse(df.lazy(), ch_table, url, params)

    result = cpl.scan_clickhouse(f"SELECT * FROM {ch_table}", url, params).collect()
    assert result.height == 10
