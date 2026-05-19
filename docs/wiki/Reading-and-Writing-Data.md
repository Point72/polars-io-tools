# Reading and writing data

These recipes show how to read from and write to external systems while keeping
predicate and projection pushdown. Each `scan_*` function returns a `LazyFrame`; filters
and column selections you chain onto it are translated into the source's own query
before any data is fetched. The connection details below are illustrative — point them
at your own servers.

All examples assume the package is imported once so the `piot` namespace is registered:

```python
import polars as pl
import polars_io_tools  # registers the .piot namespace
```

## Read from a SQL database

`scan_db` runs a SQL query over any [arrow-odbc](https://pypi.org/project/arrow-odbc/)
connection (SQL Server, PostgreSQL, Oracle, MySQL, Snowflake, and others). Filters and
column selections are translated back into SQL `WHERE` and `SELECT` clauses and appended
to your query, so the database does the filtering.

```python
from polars_io_tools import scan_db

lf = scan_db(
    "SELECT id, ts, price FROM trades",
    connection="Driver={PostgreSQL};Server=db.example.com;Database=mkt;Uid=reader;Pwd=...",
)

# Only `id` and `price` for rows after the cutoff are pulled from the database:
# the predicate becomes a SQL WHERE clause and the projection becomes a narrower SELECT.
result = (
    lf.filter(pl.col("ts") >= pl.datetime(2024, 1, 1))
    .select("id", "price")
    .collect()
)
```

The dialect is detected from the ODBC connection, so the generated SQL matches your
database. Pass `fetch_size=` to control the batch size used when Polars does not
request one.

## Read from ClickHouse

`scan_clickhouse` streams query results over ClickHouse's HTTP interface as Arrow IPC.
Predicates and projections are folded into the SQL query you provide.

```python
from polars_io_tools import scan_clickhouse

lf = scan_clickhouse(
    "SELECT * FROM metrics.cpu",
    url="https://clickhouse.example.com:8443",
    params={"user": "default", "password": "...", "database": "metrics"},
)

result = lf.filter(pl.col("date") >= pl.date(2024, 1, 1)).collect()
```

## Read Datadog metrics

`scan_datadog` queries the Datadog metrics API. A filter on the `timestamp` column is
required and defines the time range that is requested; large ranges are split into
chunks (one day by default) to respect API limits. Columns missing from a response are
filled with nulls so the schema stays stable.

```python
from polars_io_tools import scan_datadog

lf = scan_datadog(
    "avg:system.cpu.user{*}",
    api_key="...",
    app_key="...",
)

result = lf.filter(
    pl.col("timestamp") >= pl.datetime(2025, 1, 1)
).collect()
```

If you expect fields beyond the default schema, pass them via `additional_schema=`.
Without a bounded `timestamp` predicate the function raises, because it cannot determine
the time range to query.

## Read a Delta Lake table

`scan_delta` wraps Polars' `pl.scan_delta` and adds two things: partition pruning via the
`deltalake` library, and transparent recovery of logical types (`Datetime[ns/ms]`,
`Duration`, `Time`) that Delta cannot store natively but that `sink_delta` records in the
table metadata.

```python
from polars_io_tools import scan_delta

lf = scan_delta("s3://bucket/path/to/table")

# Partition predicates skip irrelevant Parquet files before any are read.
result = lf.filter(pl.col("date") == pl.date(2024, 6, 1)).collect()
```

Partition pushdown is on by default (`pushdown_predicate_deltalake=True`); set it to
`False` to fall back to the standard `pl.scan_delta` path. Use `aws_profile=` to select
S3 credentials when you are not passing an explicit `credential_provider`.

## Bridge from another dataframe library

`from_narwhals` accepts a [Narwhals](https://narwhals-dev.github.io/narwhals/) `DataFrame`
or `LazyFrame` — which can wrap pandas, Dask, DuckDB, PyArrow, and more — and returns the
equivalent Polars object. A Narwhals `LazyFrame` is backed by a custom Polars source so
filters bridge across to the underlying engine.

```python
import narwhals as nw
from polars_io_tools import from_narwhals

pl_frame = from_narwhals(some_narwhals_frame)
```

## Write a LazyFrame to Delta Lake

Polars only offers `DataFrame.write_delta` for eager frames. `sink_delta` writes a
`LazyFrame` and handles the same logical types `scan_delta` recovers, converting them to
integers for storage and embedding a mapping in the table metadata.

```python
lf = pl.LazyFrame({"id": [1, 2], "ts": [pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 2)]})

lf.piot.sink_delta("s3://bucket/path/to/table", mode="overwrite")
```

`mode` accepts `"error"` (default), `"append"`, `"overwrite"`, `"ignore"`, and `"merge"`.
For large frames, pass `chunk_size=` to write in streaming batches (for `append`,
`error`, and `ignore` modes).

## Write a LazyFrame to ClickHouse

`sink_clickhouse` writes a `LazyFrame` to an **existing** ClickHouse table over HTTP Arrow
IPC. The target table must already exist — table creation depends on engine, ordering,
and partitioning choices that cannot be inferred from a schema. Types ClickHouse lacks
(`Duration`, `Time`, `Categorical`/`Enum`) are cast automatically before writing.

```python
lf = pl.LazyFrame({"id": [1, 2, 3], "value": [10.0, 20.0, 30.0]})

lf.piot.sink_clickhouse(
    table="metrics.my_table",
    url="https://clickhouse.example.com:8443",
    params={"user": "default", "password": "...", "database": "metrics"},
)
```

Pass `chunk_size=` (requires Polars >= 1.34.0) to POST in batches rather than
materializing the whole frame; note that batched writes are not transactional.

## See also

- [Query Optimization](Query-Optimization) — combine these sources with joins,
  multi-source composition, and caching.
- [API Reference](API-Reference) — full signatures for every function above.
- [Concepts](Concepts) — how a filter on a `LazyFrame` becomes a query against a source.
