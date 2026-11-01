"""sink_clickhouse, with the HTTP insert replaced by an insert into an in-memory DuckDB.

Two tables: a write -> the table's contents; a column dtype -> the dtype and values ClickHouse receives.
"""

from datetime import date, time

import duckdb
import polars as pl
import pyarrow as pa
import pytest
from polars.testing import assert_frame_equal

import polars_io_tools as cpl
from polars_io_tools.io_sources import lazy_clickhouse_writer
from polars_io_tools.tests.helpers.cases import ENUM, Case, params

URL = "http://localhost:8123"

EMPLOYEES = pl.DataFrame(
    {
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "score": [85.4, 92.1, 78.3, 90.5, 87.9],
        "passed": [True, True, False, True, True],
        "test_date": [date(2022, 1, d) for d in range(1, 6)],
    }
)


@pytest.fixture(scope="module")
def duckdb_connection():
    conn = duckdb.connect(database=":memory:")
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def read_table(monkeypatch, duckdb_connection):
    """Route writes to DuckDB inside a transaction; return a reader of a written table."""

    def write(table: str, arrow_table: pa.Table, url: str, params: dict) -> None:
        duckdb_connection.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM arrow_table WHERE 1=0")
        duckdb_connection.execute(f"INSERT INTO {table} SELECT * FROM arrow_table")

    monkeypatch.setattr(lazy_clickhouse_writer, "_write_arrow_to_clickhouse", write)
    duckdb_connection.execute("BEGIN TRANSACTION")
    yield lambda table: pl.DataFrame(duckdb_connection.sql(f"SELECT * FROM {table}").to_arrow_table())
    duckdb_connection.execute("ROLLBACK")


WRITES = [
    Case("basic_write", write=lambda lf: cpl.sink_clickhouse(lf, "t", URL, {}), expect=EMPLOYEES),
    Case("write_empty_dataframe", write=lambda lf: cpl.sink_clickhouse(lf.head(0), "t", URL, {}), expect=EMPLOYEES.head(0)),
    Case("write_twice_appends", write=lambda lf: [cpl.sink_clickhouse(lf, "t", URL, {}) for _ in range(2)], expect=pl.concat([EMPLOYEES, EMPLOYEES])),
    Case("write_with_batching", write=lambda lf: cpl.sink_clickhouse(lf, "t", URL, {}, chunk_size=2), expect=EMPLOYEES),
    Case("piot_sink_clickhouse", write=lambda lf: lf.piot.sink_clickhouse("t", URL, {}), expect=EMPLOYEES),
]


@pytest.mark.parametrize(("write", "expect"), params(WRITES, "write", "expect"))
def test_write(read_table, write, expect):
    write(EMPLOYEES.lazy())
    assert_frame_equal(read_table("t"), expect, check_row_order=False)


def _s(values, dtype):
    return pl.Series(values, dtype=dtype)


# dtype -> what reaches ClickHouse: dtypes it supports pass through; durations and times become their Int64 ticks
# (time: nanoseconds since midnight); categoricals and enums become strings.
DTYPES = [
    *[
        Case(f"passthrough_types-{dtype}", series=_s(values, dtype), expect=None)
        for dtype, values in [
            (pl.Boolean, [True, False]),
            *[(t, [1, 2]) for t in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)],
            (pl.Float32, [1.0, 2.0]),
            (pl.Float64, [1.0, 2.0]),
            (pl.String, ["a", "b"]),
            (pl.Date, [date(2024, 1, 1), date(2024, 6, 15)]),
            (pl.Datetime("us"), [1_000_000, 2_000_000]),
            (pl.Datetime("ns"), [1_000_000, 2_000_000]),
            (pl.Datetime("ms"), [1_000, 2_000]),
        ]
    ],
    Case("duration_cast_to_int64-us", series=_s([1_000_000, 2_000_000], pl.Duration("us")), expect=_s([1_000_000, 2_000_000], pl.Int64)),
    Case(
        "duration_cast_to_int64-ns", series=_s([1_000_000_000, 2_000_000_000], pl.Duration("ns")), expect=_s([1_000_000_000, 2_000_000_000], pl.Int64)
    ),
    Case("duration_cast_to_int64-ms", series=_s([1_000, 2_000], pl.Duration("ms")), expect=_s([1_000, 2_000], pl.Int64)),
    Case(
        "time_cast_to_int64",
        series=_s([time(12, 0, 0), time(23, 59, 59)], pl.Time),
        expect=_s([12 * 3600 * 10**9, (23 * 3600 + 59 * 60 + 59) * 10**9], pl.Int64),
    ),
    Case("categorical_cast_to_string", series=_s(["a", "b", "a"], pl.Categorical), expect=_s(["a", "b", "a"], pl.String)),
    Case("enum_cast_to_string", series=_s(["c", "a", "b"], ENUM), expect=_s(["c", "a", "b"], pl.String)),
]


@pytest.mark.parametrize(("series", "expect"), params(DTYPES, "series", "expect"))
def test_column_dtype(read_table, series, expect):
    cpl.sink_clickhouse(pl.DataFrame({"col": series}).lazy(), "t", URL, {})
    assert_frame_equal(read_table("t"), pl.DataFrame({"col": series if expect is None else expect}))


def test_all_cast_types_together(read_table):
    """Every column of the dtype table in one frame."""
    n = 2
    df = pl.DataFrame({c.id: c.series[:n] for c in DTYPES})
    cpl.sink_clickhouse(df.lazy(), "t", URL, {})
    assert_frame_equal(read_table("t"), pl.DataFrame({c.id: (c.series if c.expect is None else c.expect)[:n] for c in DTYPES}))
