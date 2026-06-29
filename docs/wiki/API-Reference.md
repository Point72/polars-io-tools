# API reference

The public surface of `polars-io-tools`. Importing the package (`import polars_io_tools`) registers the `piot` namespace and re-exports the functions below at
the top level (`from polars_io_tools import scan_db, multi_source, ...`).

Most operations are available two ways:

- as a method on the `piot` LazyFrame namespace — `lf.piot.cache(...)` — where the
  LazyFrame is the implicit first argument, and
- as a top-level function — `cache(lf, ...)`.

The signatures below show the namespace form where one exists.

## The `piot` LazyFrame namespace

### `cache`

```python
lf.piot.cache(cache=None, *, order_by, partition_cols=(), cache_mode="cache", validate=True, log_explain=False, **kwargs)
```

Maintain an intermediate, per-column cache of the LazyFrame, optionally partitioned by
`partition_cols`. Predicates on partition columns restrict which partitions are cached.
`cache` defaults to a global in-memory dict; pass a custom mapping (such as
`diskcache.Cache`) to persist across sessions. `cache_mode` is `"cache"`, `"rebuild"`, or
`"ignore"`. `order_by` is required and must uniquely identify each row (within each
partition when `partition_cols` is used): columns are cached sorted by it so that
independently cached columns stay aligned regardless of source ordering. Uniqueness is
verified unless `validate=False`.

### `cache_parquet`

```python
lf.piot.cache_parquet(cache_path, date_column=None, *, time_unit="monthly",
                      partition_format=None, cache_mode=CacheMode.CACHE, aws_profile=None,
                      write_kwargs=None, read_kwargs=None, extra_partition_cols=None,
                      schema=None, write_bounding_columns=None)
```

Cache the LazyFrame to date-partitioned Parquet files on local disk or S3. Queries read
only the partitions their predicate needs; missing partitions are fetched upstream and
written back. `time_unit` is `"daily"`, `"monthly"`, or `"yearly"`. Returns a LazyFrame
reading from the cache.

### `debug`

```python
lf.piot.debug(log_level=None)
```

A pass-through source that logs (or prints, when `log_level` is `None`) the column
projection, predicate, row limit, and optimized plan that Polars passes to a source.
Useful for understanding what gets pushed down.

### `filtered_join`

```python
lf.piot.filtered_join(lf2, on=None, how="inner", *, left_on=None, right_on=None,
                      nulls_equal=False, log_explain=False, **join_kwargs)
```

Inner or left join that materializes the left frame and pushes its join keys to `lf2` as
an `is_in(...)` filter before joining. Equivalent results to `LazyFrame.join`, but `lf2`
only produces matching rows.

### `filtered_join_asof`

```python
lf.piot.filtered_join_asof(lf2, *, left_on=None, right_on=None, on=None, by=None,
                           by_left=None, by_right=None, strategy="backward",
                           tolerance=None, log_explain=True, **join_kwargs)
```

Asof join with the same right-side filter pushdown as `filtered_join`, expanding the
pushed temporal range to respect `tolerance` and `strategy`. Only `timedelta` tolerances
are currently supported.

### `ts_with_columns`

```python
lf.piot.ts_with_columns(*exprs, index_col=None, linked_cols=None, lookback=None,
                        lookahead=None, log_explain=False)
```

Apply window, cumulative, or forward-fill expressions while preserving time-based filter
pushdown. Filters on `linked_cols` are converted to filters on `index_col`, expanded back
by `lookback` and forward by `lookahead`, applied before the expressions, and the
original filter is re-applied afterward.

### `with_columns_topo`

```python
lf.piot.with_columns_topo(exprs)
```

Add columns in topological (dependency) order, batching independent expressions into the
same `with_columns` call to encourage parallel execution. Supports single-output
expressions only (no selectors).

### `filter_no_pushdown`

```python
lf.piot.filter_no_pushdown(expressions)
```

Apply one or more filter expressions that the optimizer will not push down. Useful for
predicates that cannot be translated to a source, or where pushdown would interfere with
common sub-plan elimination.

### `execute_on_ray`

```python
lf.piot.execute_on_ray(*, date_column, time_unit, return_as="arrow",
                       remote_options=None, max_concurrency=100)
```

Split the LazyFrame into calendar periods and execute each on an already-initialised Ray
cluster. `time_unit` is `"daily"`, `"monthly"`, or `"yearly"`. Requires `ray.init()` to
have been called and a bounded predicate on `date_column`.

### `sink_delta`

```python
lf.piot.sink_delta(target, *, mode="error", overwrite_schema=None, storage_options=None,
                   credential_provider="auto", delta_write_options=None,
                   delta_merge_options=None, translate_logical_types=True,
                   chunk_size=None, aws_profile=None)
```

Write the LazyFrame to a Delta Lake table. `mode` is `"error"`, `"append"`, `"overwrite"`,
`"ignore"`, or `"merge"`. With `translate_logical_types=True`, types Delta cannot store
natively (`Datetime[ns/ms]`, `Duration`, `Time`) are converted to integers and recorded in
table metadata for recovery by `scan_delta`.

### `sink_clickhouse`

```python
lf.piot.sink_clickhouse(table, url, params, *, chunk_size=None)
```

Write the LazyFrame to an existing ClickHouse table over HTTP Arrow IPC. `Duration`,
`Time`, `Categorical`, and `Enum` columns are cast automatically. `chunk_size` enables
batched (non-transactional) writes and requires Polars >= 1.34.0.

### `iter_rows`

```python
lf.piot.iter_rows(*, named=False, buffer_size=512, maintain_order=True)
```

Iterate over rows by collecting in batches, yielding tuples (or dicts when `named=True`)
without materializing the whole frame.

## Reading sources

### `scan_db`

```python
scan_db(query, connection, fetch_size=10000, **kwargs) -> pl.LazyFrame
```

Run a SQL query over an ODBC `connection` string with predicate and projection pushdown.
The SQL dialect is detected from the connection; filters become `WHERE` clauses and
selections narrow the `SELECT`.

### `scan_clickhouse`

```python
scan_clickhouse(query, url, params, fetch_size=10000)
```

Stream a ClickHouse query result over HTTP as Arrow IPC, folding predicates and
projections into the SQL. `params` carries `user`, `password`, and `database`.

### `scan_datadog`

```python
scan_datadog(query, api_key, app_key, max_chunk_duration_seconds=86400, dd_interval=None,
             additional_schema={}, overwrite_schema=False) -> pl.LazyFrame
```

Query the Datadog metrics API. A bounded predicate on the `timestamp` column is required
and defines the requested range, which is split into chunks of at most
`max_chunk_duration_seconds`. Missing fields are null-filled to keep the schema stable.

### `scan_delta`

```python
scan_delta(source, *, version=None, storage_options=None, credential_provider="auto",
           delta_table_options=None, use_pyarrow=False, pyarrow_options=None, rechunk=None,
           aws_profile=None, pushdown_predicate_deltalake=True) -> pl.LazyFrame
```

Lazily read a Delta Lake table. Wraps `pl.scan_delta`, adds partition pruning
(`pushdown_predicate_deltalake=True`), and restores logical types recorded by `sink_delta`.

### `from_narwhals`

```python
from_narwhals(obj, fetch_size=10_000) -> pl.DataFrame | pl.LazyFrame
```

Convert a Narwhals `DataFrame` or `LazyFrame` (wrapping pandas, Dask, DuckDB, PyArrow,
etc.) into the equivalent Polars object; lazy inputs are backed by a custom source so
filters bridge to the underlying engine.

## Writing sinks

### `sink_delta` (function)

```python
sink_delta(lf, target, *, mode="error", ...)
```

Top-level form of `lf.piot.sink_delta`; see the namespace entry above.

### `sink_clickhouse` (function)

```python
sink_clickhouse(lf, table, url, params, *, chunk_size=None) -> None
```

Top-level form of `lf.piot.sink_clickhouse`; see the namespace entry above.

## Composing frames

### `multi_source`

```python
multi_source(sources, combine, *, combine_kwargs=None, sources_as_kwargs=False,
             log_explain=False) -> pl.LazyFrame
```

Build a LazyFrame from named sources, each paired with a `dict[str, FilterSpec]`, plus a
`combine` function. Filters on the output are transformed per source by their
`FilterSpec`, applied before `combine`, and the original filter is re-applied after.

### `FilterSpec`

```python
FilterSpec(source_col=None, lookback=timedelta(), lookahead=timedelta(), value_mapping=None)
```

Describes how a filter on an output column maps to a source: rename to `source_col`,
expand temporal ranges by `lookback`/`lookahead`, and remap values with `value_mapping` (a
dict or callable).

### `concat_named`

```python
concat_named(lf_dict, identifier_cols, *, log_explain=False, **kwargs) -> pl.LazyFrame
```

Vertically concatenate frames keyed by identifier tuples, adding `identifier_cols`. A
filter on an identifier column only materializes the matching frames.

### `join_between`

```python
join_between(left, right, left_on, right_on_start, right_on_end, by=None, how="left")
    -> pl.LazyFrame
```

Match each `left_on` value to the right-side row whose `[right_on_start, right_on_end]`
interval contains it, optionally with an equi-join on `by`. Returns at most one match per
left row (non-overlapping intervals).

## Caching

### `CacheMode`

```python
CacheMode.CACHE | CacheMode.IGNORE | CacheMode.REBUILD
```

Caching behavior passed to `cache_parquet`: read-and-fill, bypass, or refresh in-scope
partitions.

## Utilities

### `disable_optimizations`

```python
with disable_optimizations():
    ...
```

Context manager that replaces `piot` optimizations with their plain-Polars equivalents,
so `LazyFrame.explain()` produces a readable plan and results can be compared with and
without optimization.
