# API reference

The public surface of `polars-io-tools`. Importing the package (`import polars_io_tools`) registers the `piot` namespace and re-exports the functions below at
the top level (`from polars_io_tools import scan_db, pushdown_combine, ...`).

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

### `cache_memory`

```python
lf.piot.cache_memory(*, schema)
```

Collect the LazyFrame once into an in-memory buffer and replay it on every subsequent
reference or collect, so a frame referenced from several branches executes upstream once
rather than once per reference. Unlike `cache`, the key is not derived by serializing the
plan — the buffer lives in the returned frame's own closure — so it also works with
`register_io_source` plugins that close over non-serializable state (a connection, a lock,
an open iterator). `schema` is the declared output schema, or a zero-argument callable
returning it; a callable lets `collect_schema()` resolve without materializing the buffer.
The collected frame is reconciled against `schema` once (a missing declared column or dtype
mismatch raises; extra columns are dropped), and predicates and projections are applied to
the buffer after materialization. The build runs at most once: a failure is recorded and
re-raised on later collects, and the buffer is released when the returned frame is dropped.
As a top-level function, `cache_memory(build_or_lf, *, schema)` additionally accepts a
zero-argument builder callable, executed on first row demand, in place of a LazyFrame.

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
lf.piot.execute_on_ray(partitions, *, return_as="arrow",
                       remote_options=None, max_concurrency=100,
                       preserve_partition_order=None)
```

Distribute the LazyFrame across an already-initialised Ray cluster, running one task per
partition. `partitions` is either a partitioner or an explicit iterable of
`ReadPartition(predicate, key)` (a `RayPartition` additionally carries per-task
`remote_options`). Build partitions with:

- `by_time(column, every)` — calendar windows derived from the pushed-down date range
  (`every` is `"1mo"`/`"2w"`/`"5d"`/`"1q"`/`"1y"` or an integer number of days).
- `by_value(column, values=None)` — one task per discrete value; derived from the pushed-down
  `IN` filter when `values` is omitted.
- `by_range(column, every)` — fixed-width numeric buckets over the pushed-down range.
- `by_key(partitions, by, *, partition_remote_options=None)` — equality on caller-enumerated
  keys; `by` is a column name, list, selector, or a `pl.Expr` (e.g. `pl.col("id").hash() % N`).
  `partition_remote_options` sets per-partition Ray options from a struct column or `{key: dict}`.
- `discrete_partitions` / `cartesian_partitions` — explicit `col.is_in(...)` member lists and
  `date_window × bucket` products.

A partitioner requires a bounded predicate on its column. Requires `ray.init()` to have been
called. As a legacy shortcut, `execute_on_ray(date_column=..., time_unit="daily"|"monthly"|"yearly")`
is equivalent to `partitions=by_time(date_column, ...)`.

Chaining multiple `execute_on_ray` calls relies on predicate pushdown surviving intervening
operations — partition once at the outermost boundary. For multi-stage distributed pipelines,
prefer [Polars Cloud](https://docs.cloud.pola.rs/polars-cloud/).

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
scan_db(query, connection, fetch_size=10000, cast_map=None, **kwargs) -> pl.LazyFrame
```

Run a SQL query over an ODBC `connection` string with predicate and projection pushdown.
The SQL dialect is detected from the connection; filters become `WHERE` clauses and
selections narrow the `SELECT`. Pass `cast_map={column: dtype}` to cast columns
server-side (a SQL `CAST`, keeping `select *`) and report the narrowed dtype in the
schema, so a filter on a cast column still pushes down — for example narrowing a
`datetime` column that is logically a `date`, or a `float` id that should be an integer.
A predicate on a cast column pushes down over its `CAST(...)`, which can affect index use;
the impact on the query plan is backend dependent (see Reading and Writing Data).

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

### `scan_synthetic_regression`

```python
scan_synthetic_regression(*, n_samples, n_features, n_responses=1, use_weights=False,
                          weights_low=0.5, weights_high=1.5, betas=None, epsilon_loc=0.0,
                          epsilon_scale=1.0, chunk_key=None, n_chunks=None, seed=None,
                          fetch_size=10_000) -> pl.LazyFrame
```

Lazy source of synthetic linear-regression data `Y = X @ B + E` with Gaussian noise.
Emits `x0..x{n_features-1}` and `y0..y{n_responses-1}`; with `use_weights=True` a `weight`
column is added and noise is scaled so a WLS fit recovers `betas`. Row values are
batch-independent for a fixed `seed`, and predicate/projection/`head` pushdowns apply.

### `scan_synthetic_panel`

```python
scan_synthetic_panel(*, start_date, end_date, freq="1D", n_symbols=1, n_features,
                     n_responses=1, betas=None, use_weights=False, weights_low=0.5,
                     weights_high=1.5, categories=None, group_by=None, epsilon_loc=0.0,
                     epsilon_scale=1.0, seed=None, fetch_size=10_000) -> pl.LazyFrame
```

Lazy source of synthetic panel data on a `(date, symbol)` grid, one row per pair over the
business days in `[start_date, end_date]`. Same regression model as
`scan_synthetic_regression`, plus optional decorative `categories` and per-group
coefficients via `group_by=(name, values)`. Rows are yielded date-by-date so
`.set_sorted("date").group_by("date")` streams cleanly.

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

### `pushdown_combine`

```python
pushdown_combine(sources, combine, *, combine_kwargs=None, sources_as_kwargs=False,
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

### `IntervalFilterSpec`

```python
IntervalFilterSpec(start_col, end_col, closed="both", value_mapping=None)
```

Maps a filter on a request-date column onto a validity-interval source whose rows are valid
over `[start_col, end_col]`. A range `[lo, hi]` pushes the overlap `start_col <= hi AND end_col >= lo` (adjusted for `closed`), remapping bounds with `value_mapping` first.

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

## Reshaping

### `pushdown_pivot`

```python
pushdown_pivot(source, on, on_columns, *, index, values, aggregate_function=None,
               maintain_order=False, separator="_", column_naming="auto",
               is_pure=True, dense_on=False) -> pl.LazyFrame
```

Pushdown-friendly `LazyFrame.pivot`. The output is identical to
`source.pivot(on, on_columns=on_columns, ...)`, but downstream projections and predicates
are pushed into the wrapper: selecting a subset of pivoted columns becomes an upstream
`on`-value row filter, and index/pivoted-column predicates flow through. `on_columns` pins
the output schema. `dense_on=True` skips the index-recovery scan (roughly 2x faster) when
every `index` value is known to have a row for every `on` value; it silently drops rows on
sparse data. `aggregate_function` is not yet supported.

### `pushdown_unpivot`

```python
pushdown_unpivot(source, *, index, on=None, variable_name="variable",
                 value_name="value", is_pure=True) -> pl.LazyFrame
```

Pushdown-friendly `LazyFrame.unpivot`. The output is identical to
`source.unpivot(index=index, on=on, ...)`, but a filter on `variable` (for example
`variable == "A"` or `variable.is_in([...])`) is rewritten as an upstream column
projection, and filters on `index` flow through. `on` defaults to all non-`index` columns.

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
