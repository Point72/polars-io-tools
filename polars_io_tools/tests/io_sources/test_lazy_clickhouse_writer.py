from datetime import date
from typing import Optional

import duckdb
import polars as pl
import pyarrow as pa
import pytest
from polars.testing import assert_frame_equal

import polars_io_tools as cpl
from polars_io_tools import io_sources as polars_utils

"""
This module contains tests for the lazy ClickHouse writer (sink_clickhouse).
We mock _write_arrow_to_clickhouse and _clickhouse_command to execute writes
against an in-memory DuckDB database, similar to the reader tests.
"""


def get_employee_df() -> pl.DataFrame:
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


_duckdb_conn: Optional[duckdb.DuckDBPyConnection] = None


@pytest.fixture(scope="module")
def duckdb_connection():
    """Create a DuckDB connection once per test module."""
    global _duckdb_conn
    conn = duckdb.connect(database=":memory:")
    _duckdb_conn = conn
    yield conn
    conn.close()
    _duckdb_conn = None


@pytest.fixture(autouse=True)
def reset_state(duckdb_connection):
    """Isolate each test via DuckDB transactions."""
    duckdb_connection.execute("BEGIN TRANSACTION")
    yield
    duckdb_connection.execute("ROLLBACK")


def fake_write_arrow_to_clickhouse(table: str, arrow_table: pa.Table, url: str, params: dict) -> None:
    """Mock _write_arrow_to_clickhouse: insert Arrow data into DuckDB."""
    global _duckdb_conn

    # Create DuckDB table from Arrow if it doesn't exist yet
    try:
        _duckdb_conn.execute(f"SELECT 1 FROM {table} LIMIT 0")
    except duckdb.CatalogException:
        _duckdb_conn.execute(f"CREATE TABLE {table} AS SELECT * FROM arrow_table WHERE 1=0")

    _duckdb_conn.execute(f"INSERT INTO {table} SELECT * FROM arrow_table")


@pytest.fixture(autouse=True)
def patch_writer(monkeypatch, duckdb_connection):
    """Monkeypatch the writer's HTTP functions to use DuckDB."""
    monkeypatch.setattr(
        polars_utils.lazy_clickhouse_writer,
        "_write_arrow_to_clickhouse",
        fake_write_arrow_to_clickhouse,
    )


def read_duckdb_table(table: str) -> pl.DataFrame:
    global _duckdb_conn
    arrow_table = _duckdb_conn.execute(f"SELECT * FROM {table}").fetch_arrow_table()
    return pl.DataFrame(arrow_table)


FAKE_URL = "http://localhost:8123"
FAKE_PARAMS = {}


class TestBasicWrite:
    def test_basic_write(self):
        """Write a LazyFrame and verify the data lands in DuckDB."""
        df = get_employee_df()

        cpl.sink_clickhouse(df.lazy(), "test_basic", FAKE_URL, FAKE_PARAMS)

        result = read_duckdb_table("test_basic")
        assert_frame_equal(result, df)

    def test_write_empty_dataframe(self):
        """Writing zero rows should succeed without error."""
        df = get_employee_df().head(0)

        cpl.sink_clickhouse(df.lazy(), "test_empty", FAKE_URL, FAKE_PARAMS)

        result = read_duckdb_table("test_empty")
        assert result.height == 0

    def test_write_various_dtypes(self):
        """Write a DataFrame with Int64, Float64, String, Boolean, Date columns."""
        df = get_employee_df()

        cpl.sink_clickhouse(df.lazy(), "test_dtypes", FAKE_URL, FAKE_PARAMS)

        result = read_duckdb_table("test_dtypes")
        assert result.height == 5
        assert set(result.columns) == {"id", "name", "score", "passed", "test_date"}

    def test_write_twice_appends(self):
        """Writing twice to the same table appends rows."""
        df = get_employee_df()

        cpl.sink_clickhouse(df.lazy(), "test_append", FAKE_URL, FAKE_PARAMS)
        cpl.sink_clickhouse(df.lazy(), "test_append", FAKE_URL, FAKE_PARAMS)

        result = read_duckdb_table("test_append")
        assert result.height == 10


class TestBatching:
    def test_write_with_batching(self):
        """Write with chunk_size=2 and verify all rows arrive."""
        df = get_employee_df()

        cpl.sink_clickhouse(
            df.lazy(),
            "test_batched",
            FAKE_URL,
            FAKE_PARAMS,
            chunk_size=2,
        )

        result = read_duckdb_table("test_batched")
        assert_frame_equal(result.sort("id"), df.sort("id"))


class TestTypeCasting:
    """Verify that _prepare_for_clickhouse casts types correctly before writing."""

    def test_passthrough_types(self):
        """Types that ClickHouse supports natively should pass through unchanged."""
        df = pl.DataFrame(
            {
                "col_bool": pl.Series([True, False], dtype=pl.Boolean),
                "col_i8": pl.Series([1, 2], dtype=pl.Int8),
                "col_i16": pl.Series([1, 2], dtype=pl.Int16),
                "col_i32": pl.Series([1, 2], dtype=pl.Int32),
                "col_i64": pl.Series([1, 2], dtype=pl.Int64),
                "col_u8": pl.Series([1, 2], dtype=pl.UInt8),
                "col_u16": pl.Series([1, 2], dtype=pl.UInt16),
                "col_u32": pl.Series([1, 2], dtype=pl.UInt32),
                "col_u64": pl.Series([1, 2], dtype=pl.UInt64),
                "col_f32": pl.Series([1.0, 2.0], dtype=pl.Float32),
                "col_f64": pl.Series([1.0, 2.0], dtype=pl.Float64),
                "col_str": pl.Series(["a", "b"], dtype=pl.String),
                "col_date": pl.Series([date(2024, 1, 1), date(2024, 6, 15)], dtype=pl.Date),
                "col_dt_us": pl.Series([1_000_000, 2_000_000], dtype=pl.Datetime("us")),
                "col_dt_ns": pl.Series([1_000_000, 2_000_000], dtype=pl.Datetime("ns")),
                "col_dt_ms": pl.Series([1_000, 2_000], dtype=pl.Datetime("ms")),
            }
        )
        cpl.sink_clickhouse(df.lazy(), "test_passthrough", FAKE_URL, FAKE_PARAMS)

        result = read_duckdb_table("test_passthrough")
        assert result.height == 2
        # All passthrough types should keep their original schema
        for col in df.columns:
            assert result[col].dtype == df[col].dtype, f"Column {col}: expected {df[col].dtype}, got {result[col].dtype}"

    def test_duration_cast_to_int64(self):
        """Duration columns should be cast to Int64 (raw tick count)."""
        df = pl.DataFrame(
            {
                "id": [1, 2],
                "dur_us": pl.Series([1_000_000, 2_000_000], dtype=pl.Duration("us")),
                "dur_ns": pl.Series([1_000_000_000, 2_000_000_000], dtype=pl.Duration("ns")),
                "dur_ms": pl.Series([1_000, 2_000], dtype=pl.Duration("ms")),
            }
        )
        cpl.sink_clickhouse(df.lazy(), "test_duration", FAKE_URL, FAKE_PARAMS)

        result = read_duckdb_table("test_duration")
        assert result["dur_us"].dtype == pl.Int64
        assert result["dur_ns"].dtype == pl.Int64
        assert result["dur_ms"].dtype == pl.Int64
        # Values should be the raw tick counts
        assert result["dur_us"].to_list() == [1_000_000, 2_000_000]
        assert result["dur_ns"].to_list() == [1_000_000_000, 2_000_000_000]
        assert result["dur_ms"].to_list() == [1_000, 2_000]

    def test_time_cast_to_int64(self):
        """Time columns should be cast to Int64 (nanoseconds since midnight)."""
        from datetime import time

        df = pl.DataFrame(
            {
                "id": [1, 2],
                "t": pl.Series([time(12, 0, 0), time(23, 59, 59)], dtype=pl.Time),
            }
        )
        cpl.sink_clickhouse(df.lazy(), "test_time", FAKE_URL, FAKE_PARAMS)

        result = read_duckdb_table("test_time")
        assert result["t"].dtype == pl.Int64
        # 12:00:00 = 12 * 3600 * 1e9 ns
        assert result["t"][0] == 12 * 3600 * 1_000_000_000

    def test_categorical_cast_to_string(self):
        """Categorical columns should be cast to String."""
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "cat": pl.Series(["a", "b", "a"], dtype=pl.Categorical),
            }
        )
        cpl.sink_clickhouse(df.lazy(), "test_categorical", FAKE_URL, FAKE_PARAMS)

        result = read_duckdb_table("test_categorical")
        assert result["cat"].dtype == pl.String
        assert result["cat"].to_list() == ["a", "b", "a"]

    def test_enum_cast_to_string(self):
        """Enum columns should be cast to String."""
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "color": pl.Series(
                    ["red", "green", "blue"],
                    dtype=pl.Enum(["red", "green", "blue"]),
                ),
            }
        )
        cpl.sink_clickhouse(df.lazy(), "test_enum", FAKE_URL, FAKE_PARAMS)

        result = read_duckdb_table("test_enum")
        assert result["color"].dtype == pl.String
        assert result["color"].to_list() == ["red", "green", "blue"]

    def test_all_cast_types_together(self):
        """Verify a DataFrame mixing passthrough and cast types writes correctly."""
        from datetime import time

        df = pl.DataFrame(
            {
                "id": pl.Series([1, 2], dtype=pl.Int64),
                "name": pl.Series(["x", "y"], dtype=pl.String),
                "dur": pl.Series([100, 200], dtype=pl.Duration("us")),
                "t": pl.Series([time(1, 0), time(2, 0)], dtype=pl.Time),
                "cat": pl.Series(["a", "b"], dtype=pl.Categorical),
            }
        )
        cpl.sink_clickhouse(df.lazy(), "test_mixed", FAKE_URL, FAKE_PARAMS)

        result = read_duckdb_table("test_mixed")
        assert result.height == 2
        # Passthrough columns keep dtype
        assert result["id"].dtype == pl.Int64
        assert result["name"].dtype == pl.String
        # Cast columns become Int64 or String
        assert result["dur"].dtype == pl.Int64
        assert result["t"].dtype == pl.Int64
        assert result["cat"].dtype == pl.String
        # Values preserved
        assert result["id"].to_list() == [1, 2]
        assert result["dur"].to_list() == [100, 200]
        assert result["cat"].to_list() == ["a", "b"]


class TestNamespaceMethod:
    def test_piot_sink_clickhouse(self):
        """lf.piot.sink_clickhouse(...) should work."""
        df = get_employee_df()

        df.lazy().piot.sink_clickhouse("test_namespace", FAKE_URL, FAKE_PARAMS)

        result = read_duckdb_table("test_namespace")
        assert_frame_equal(result, df)
