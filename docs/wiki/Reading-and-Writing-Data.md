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

If a column is stored as a different type than it is used as — for example a `datetime`
column that is logically a `date`, or a numeric id delivered as `float` that should be an
integer — cast it server-side with `cast_map` so filters on it still push down:

```python
lf = scan_db(
    "SELECT * FROM trades",
    connection="Driver={PostgreSQL};Server=db.example.com;Database=mkt;Uid=reader;******",
    cast_map={"ts": pl.Date},
)

# `ts` is narrowed to a date in the query, so this filter becomes a SQL WHERE clause
# instead of being applied after a full scan.
result = lf.filter(pl.col("ts") == pl.date(2024, 1, 1)).collect()
```

`cast_map` wraps your query in a projecting subquery that casts the named columns and
passes the rest through, so `select *` keeps flowing every column.

A predicate on a cast column is pushed down as `CAST(col AS ...) <op> value`. Wrapping the
column in a function can stop the database from using an index on it, so the effect on the
query plan is **backend dependent** — some engines optimize particular conversions (for
example SQL Server can still seek on `CAST(datetime AS date)`) while others fall back to a
full scan. When a filtered column is indexed and on a hot path, prefer filtering the
physical column directly over its cast form.

## Speed up a large SQL read by partitioning it

When a scan-like extract is bounded by an indexed column, a single ODBC cursor is often the
bottleneck. Pass `partitions=` and `scan_db` splits the read into independent slices and pulls
them over parallel connections, concatenating the results in order. Build the slices from the
filter you push down with `by_time`, `by_value`, or `by_range`:

```python
from polars_io_tools import scan_db, by_time

lf = scan_db(
    "SELECT * FROM daily_prices",
    connection="Driver={PostgreSQL};Server=db.example.com;Database=mkt;Uid=reader;******",
    partitions=by_time("price_date", every="1mo"),
)

# One slice per month, taken from the pushed-down date range:
result = lf.filter(
    (pl.col("price_date") >= pl.date(2025, 1, 1)) & (pl.col("price_date") < pl.date(2025, 7, 1))
).collect()
```

- `by_time(column, every=)` — calendar windows; `every` is an interval string (`"1mo"`, `"2w"`,
  `"5d"`, `"1q"`, `"1y"`) or an integer number of days.
- `by_value(column, values=None)` — one slice per discrete value; with `values=None` the values
  are read from the `IN` filter you push down.
- `by_range(column, every=)` — fixed-width numeric buckets over the pushed-down range.

How many slices run at once is capped by a process-wide SQL connection budget
(`POLARS_IO_TOOLS_MAX_SQL_CONNECTIONS`; default `min(pl.thread_pool_size(), 8)` — a modest 8 on a
normal machine, self-throttling to 1 in fan-out clusters that pin `POLARS_MAX_THREADS=1`, since
these reads are IO-bound and the cap is a connection budget, not the CPU thread pool); pass
`max_concurrency=` to throttle below it on a busy server. Partitioning is fully opt-in — with no
`partitions=`, or when a partitioner cannot derive a bounded split, `scan_db` runs the query over
a single connection. Prefer a handful of medium slices over many tiny ones: each slice is a
separate query with its own planning and round-trip cost.

For hand-built slices, pass an iterable of `ReadPartition(predicate, key)`. Predicates that
translate to SQL are pushed to the database; any part that cannot (for example an arbitrary
Python UDF) is still enforced client-side, so each slice stays exact.

Rows are yielded in completion order (whichever slice's batches arrive first), which is not
deterministic; if you need a specific order, sort in Polars with `.sort()` after the read. If your
query has a top-level `ORDER BY`, partitioning can't re-establish it across slices, so `scan_db`
logs a warning and ignores it (the rows are still correct) — again, `.sort()` after the read.
Clauses that would change the *result* under partitioning — a top-level `LIMIT`/`OFFSET`/`TOP`/`FETCH`,
`QUALIFY`, or a window function — raise instead; apply them in Polars after the read (`.head()`, etc.),
or read without `partitions=`.

Each slice is streamed batch-by-batch over its own connection, so peak memory stays proportional to
the connection budget times the arrow batch size — a large slice is never fully buffered in memory.

Each slice opens its own ODBC connection, so a read split into many slices pays that many connects.
If connect overhead dominates (many small slices, or TLS/Kerberos on every connect), prefer a few
larger slices, and/or let the ODBC driver manager recycle physical connections by calling
[`arrow_odbc.enable_odbc_connection_pooling()`](https://arrow-odbc.readthedocs.io/) once before your
first read — pooled connects skip the handshake. This is a process-global arrow-odbc / driver-manager
setting (it affects all ODBC use in your process and reuses physical connections, so session-scoped
state such as temp tables can persist across them), so it is left to the caller rather than toggled
by `scan_db`.

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
