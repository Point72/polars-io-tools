"""sink_clickhouse, with the HTTP insert replaced by an insert into an in-memory DuckDB.

Two tables: a write -> the table's contents; a column dtype -> the dtype and values ClickHouse receives.
"""

from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from datetime import date, time

import duckdb
import polars as pl
import pyarrow as pa
import pytest
from polars.testing import assert_frame_equal

import polars_io_tools as cpl
from polars_io_tools.io_sources import lazy_clickhouse_writer
from polars_io_tools.tests.helpers.cases import UNSORTED_ENUM, as_params

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


@dataclass(frozen=True)
class ClickhouseWriteCase:
    """A way to write the employee frame, and what the table holds afterwards."""

    id: str
    _: KW_ONLY
    # receives the employee frame and writes it
    write: Callable[[pl.LazyFrame], object]
    expected: pl.DataFrame


WRITES = [
    ClickhouseWriteCase("basic_write", write=lambda lf: cpl.sink_clickhouse(lf, "t", URL, {}), expected=EMPLOYEES),
    ClickhouseWriteCase("write_empty_dataframe", write=lambda lf: cpl.sink_clickhouse(lf.head(0), "t", URL, {}), expected=EMPLOYEES.head(0)),
    ClickhouseWriteCase(
        "write_twice_appends", write=lambda lf: [cpl.sink_clickhouse(lf, "t", URL, {}) for _ in range(2)], expected=pl.concat([EMPLOYEES, EMPLOYEES])
    ),
    ClickhouseWriteCase("write_with_batching", write=lambda lf: cpl.sink_clickhouse(lf, "t", URL, {}, chunk_size=2), expected=EMPLOYEES),
    ClickhouseWriteCase("piot_sink_clickhouse", write=lambda lf: lf.piot.sink_clickhouse("t", URL, {}), expected=EMPLOYEES),
]


@pytest.mark.parametrize("case", as_params(WRITES))
def test_write(read_table, case: ClickhouseWriteCase):
    case.write(EMPLOYEES.lazy())
    assert_frame_equal(read_table("t"), case.expected, check_row_order=False)


def _series(values, dtype):
    return pl.Series(values, dtype=dtype)


@dataclass(frozen=True)
class ClickhouseDtypeCase:
    """A column, and the column ClickHouse receives when it is written."""

    id: str
    _: KW_ONLY
    series: pl.Series
    # None: the column is received unchanged
    expected: pl.Series | None


# dtype -> what reaches ClickHouse: dtypes it supports pass through; durations and times become their Int64 ticks
# (time: nanoseconds since midnight); categoricals and enums become strings.
DTYPES = [
    *[
        ClickhouseDtypeCase(f"passthrough_types-{dtype}", series=_series(values, dtype), expected=None)
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
    ClickhouseDtypeCase(
        "duration_cast_to_int64-us", series=_series([1_000_000, 2_000_000], pl.Duration("us")), expected=_series([1_000_000, 2_000_000], pl.Int64)
    ),
    ClickhouseDtypeCase(
        "duration_cast_to_int64-ns",
        series=_series([1_000_000_000, 2_000_000_000], pl.Duration("ns")),
        expected=_series([1_000_000_000, 2_000_000_000], pl.Int64),
    ),
    ClickhouseDtypeCase("duration_cast_to_int64-ms", series=_series([1_000, 2_000], pl.Duration("ms")), expected=_series([1_000, 2_000], pl.Int64)),
    ClickhouseDtypeCase(
        "time_cast_to_int64",
        series=_series([time(12, 0, 0), time(23, 59, 59)], pl.Time),
        expected=_series([12 * 3600 * 10**9, (23 * 3600 + 59 * 60 + 59) * 10**9], pl.Int64),
    ),
    ClickhouseDtypeCase("categorical_cast_to_string", series=_series(["a", "b", "a"], pl.Categorical), expected=_series(["a", "b", "a"], pl.String)),
    ClickhouseDtypeCase("enum_cast_to_string", series=_series(["c", "a", "b"], UNSORTED_ENUM), expected=_series(["c", "a", "b"], pl.String)),
]


@pytest.mark.parametrize("case", as_params(DTYPES))
def test_column_dtype(read_table, case: ClickhouseDtypeCase):
    cpl.sink_clickhouse(pl.DataFrame({"col": case.series}).lazy(), "t", URL, {})
    assert_frame_equal(read_table("t"), pl.DataFrame({"col": case.series if case.expected is None else case.expected}))


def test_all_cast_types_together(read_table):
    """Every column of the dtype table in one frame."""
    n = 2
    df = pl.DataFrame({c.id: c.series[:n] for c in DTYPES})
    cpl.sink_clickhouse(df.lazy(), "t", URL, {})
    assert_frame_equal(read_table("t"), pl.DataFrame({c.id: (c.series if c.expected is None else c.expected)[:n] for c in DTYPES}))
