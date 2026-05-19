# Query optimization

These recipes cover the operations `polars-io-tools` adds for building lazy queries that
keep predicate pushdown alive — joins, multi-source composition, time-series windows,
and caching. They work the same on in-memory frames and on remote `scan_*` sources; the
in-memory examples below run as-is.

```python
import polars as pl
import polars_io_tools  # registers the .piot namespace
```

## Pre-filter the right side of a join

A plain inner join reads the entire right frame and only then discards non-matching
rows. `filtered_join` materializes the left frame, extracts its join keys, and pushes
them to the right frame as an `is_in(...)` filter before the join.

```python
left = pl.LazyFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
right = pl.LazyFrame({"x": [-1, -2, 3], "z": [7, 8, 9]})

left.piot.filtered_join(right, on="x").collect()
# shape: (1, 3)
# ┌─────┬─────┬─────┐
# │ x   ┆ y   ┆ z   │
# ╞═════╪═════╪═════╡
# │ 3   ┆ 6   ┆ 9   │
# └─────┴─────┴─────┘
```

Supports `how="inner"` (default) and `how="left"`, single- or multi-column `on` (or
`left_on`/`right_on`). The optimization pays off most when the right frame is a large or
remote source and the left frame is comparatively small.

## Join time-series frames with temporal pushdown

`filtered_join_asof` is the asof (ordered/temporal) counterpart of `filtered_join`: it
applies the same right-side filter pushdown, expanding the pushed range to respect the
asof tolerance so the nearest match is still found.

```python
quotes = pl.LazyFrame({"t": [1, 2, 3], "bid": [10, 11, 12]})
trades = pl.LazyFrame({"t": [2, 3], "px": [10.5, 11.5]})

trades.piot.filtered_join_asof(quotes, on="t").collect()
```

## Join a point to the interval that contains it

`join_between` matches each left-side value to the right-side row whose
`[start, end]` interval contains it — useful for resolving a date to the record that was
effective on it. It implements a range join with a single backward asof plus a validation
step, returning at most one match per left row.

```python
from polars_io_tools import join_between
from datetime import date

observations = pl.LazyFrame({
    "symbol": ["ESH4", "ESH4"],
    "obs_date": [date(2024, 1, 15), date(2024, 3, 10)],
})
contracts = pl.LazyFrame({
    "symbol": ["ESH4", "ESH4"],
    "start": [date(2024, 1, 1), date(2024, 3, 1)],
    "end": [date(2024, 1, 31), date(2024, 3, 31)],
})

join_between(
    observations, contracts,
    left_on="obs_date", right_on_start="start", right_on_end="end",
    by="symbol",
).collect()
```

For overlapping intervals where each match should produce a row, use Polars'
`LazyFrame.join_where` instead.

## Combine sources with coordinated filter pushdown

`multi_source` builds a single `LazyFrame` from several named sources plus a `combine`
function. When the result is filtered, each source receives a *transformed* version of
the filter described by its `FilterSpec` — a renamed column, an expanded date range, or
a remapped value — and the original filter is re-applied after `combine` runs.

```python
from datetime import timedelta
from polars_io_tools import multi_source, FilterSpec

lf = multi_source(
    sources={
        "prices": (prices_lf, {"date": FilterSpec(), "id": FilterSpec()}),
        "signals": (signals_lf, {
            "date": FilterSpec(lookback=timedelta(days=5)),   # fetch 5 extra days
            "id": FilterSpec(source_col="identifier"),         # different column name
        }),
    },
    combine=lambda s: s["prices"].join(s["signals"], on=["date", "id"]),
)

result = lf.filter(pl.col("date") > pl.date(2024, 1, 1)).collect()
```

Here the `signals` source is asked for five extra days of history (so rolling
calculations in `combine` are correct) and filtered on its `identifier` column, while the
final output is trimmed back to exactly the requested dates. `FilterSpec` also accepts a
`lookahead` and a `value_mapping` (a dict or callable) for translating filter values.

## Concatenate frames and prune by identifier

`concat_named` vertically concatenates a dictionary of frames, adding identifier columns
derived from the keys. Unlike `pl.concat`, a filter on an identifier column only
materializes the frames that match — a workaround for
[pola-rs/polars#24782](https://github.com/pola-rs/polars/issues/24782).

```python
from polars_io_tools import concat_named

lf1 = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
lf2 = pl.LazyFrame({"a": [7, 8, 9], "b": [10, 11, 12]})

# Only lf1 is materialized; lf2 is pruned by the filter.
concat_named({("foo",): lf1, ("bar",): lf2}, ["source"]).filter(
    pl.col("source") == "foo"
).collect()
```

## Apply rolling windows without losing pushdown

`ts_with_columns` lets window, cumulative, and forward-fill expressions keep filter
pushdown by widening time-column filters with a `lookback`/`lookahead`, computing the
expressions, then re-applying the original filter. See
[Getting Started](Getting-Started#step-3--keep-pushdown-through-a-rolling-window) for a
worked example.

## Cache results across iterations

`cache` keeps an in-memory, per-column (optionally per-partition) cache so repeated
collects reuse already-computed columns:

```python
df = (
    pl.LazyFrame({"x": [1, 2, 3]})
    .with_columns((pl.col("x") * 2).alias("slow"))
    .piot.cache()
)

df.select(pl.col("slow").max()).collect()   # computes and caches "slow"
df.select(pl.col("slow").min()).collect()   # reuses the cached column
```

Pass a custom mapping (for example `diskcache.Cache(...)`) to persist across sessions,
and `partition_cols=` so that filters on partition columns restrict which partitions are
cached. `cache` relies on the source producing consistent row ordering across collects.

## Cache to partitioned Parquet on disk or S3

`cache_parquet` materializes a `LazyFrame` to date-partitioned Parquet files. Subsequent
queries read only the partitions their predicate needs, and missing partitions are
fetched from upstream and written back.

```python
lf = prices_lf.piot.cache_parquet(
    "s3://bucket/cache/prices",
    date_column="date",
    time_unit="monthly",
)

# Reads only the matching monthly partitions, fetching any that are missing.
lf.filter(pl.col("date").is_between(pl.date(2024, 1, 1), pl.date(2024, 3, 31))).collect()
```

Use `cache_mode=CacheMode.REBUILD` to refresh partitions in the query scope while leaving
others intact, and `aws_profile=` to select S3 credentials.

## Control pushdown explicitly

Two helpers give you manual control when the optimizer's defaults are not what you want:

- `filter_no_pushdown` applies a filter that the optimizer will **not** push down —
  useful when a predicate cannot be translated to an external source, or when pushdown
  would interfere with common sub-plan elimination (for example in a self-join against a
  filtered copy of the frame).
- `with_columns_topo` adds columns in dependency order, batching independent expressions
  into the same `with_columns` call so they can run in parallel even when later columns
  reference earlier ones.

```python
# Keep this predicate from being pushed into a source.
lf.piot.filter_no_pushdown(pl.col("score") > 0.9)

# Add columns where later ones depend on earlier ones.
lf.piot.with_columns_topo([
    (pl.col("x") + 1).alias("b"),
    (pl.col("b") * 2).alias("c"),   # depends on b
])
```

## See also

- [Distributed Execution](Distributed-Execution) — scale these queries across Ray.
- [API Reference](API-Reference) — full signatures.
- [Concepts](Concepts) — why these operations preserve pushdown.
