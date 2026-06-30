# Getting started

This tutorial walks you through the core idea behind `polars-io-tools`: writing ordinary
Polars lazy queries while filters and column selections are pushed down so that less
data is ever read or moved. You will build up a small pipeline step by step, watching
the pushdown happen on in-memory frames so that every snippet runs as-is — no database
or cluster required.

By the end you will be able to:

- register and use the `piot` LazyFrame namespace,
- see exactly what Polars pushes into a source,
- run a join that pre-filters the right-hand frame,
- apply a rolling time-series window without losing pushdown, and
- cache intermediate results for fast iteration.

## Before you begin

Install the package (see the [Installation](Installation) guide for `conda` and source
builds):

```bash
pip install polars-io-tools
```

Importing the package registers a `piot` namespace on every Polars `LazyFrame`. That is
the entry point for everything in this tutorial:

```python
import polars as pl
import polars_io_tools  # registers the .piot namespace
```

## Step 1 — see what gets pushed down

The `debug` source is a pass-through that prints the column projection, predicate, and
row limit that Polars hands to a source at collection time. It is the quickest way to
build intuition for pushdown.

```python
lf = pl.LazyFrame({"a": [1, 2, 3], "b": [10, 20, 30], "c": ["x", "y", "z"]})

(
    lf.piot.debug()
    .filter(pl.col("a") > 1)
    .select("a", "b")
    .collect()
)
```

When you collect, `debug` prints the request it received. You will see that Polars asked
only for columns `a` and `b` (the `select`) and passed the predicate `a > 1` down to the
source, rather than reading column `c` or every row. That projection-and-predicate
bundle is exactly what the real `scan_*` sources translate into SQL, Datadog time
ranges, or Delta partition filters.

## Step 2 — join with pushdown to the right frame

A normal inner join cannot pre-filter the right frame with the keys found on the left —
Polars does not turn the join into a filter for you, so the entire right frame is read.
`filtered_join` materializes the left frame first, then pushes the matching keys down to
the right frame as an `is_in(...)` filter before the join runs.

```python
left = pl.LazyFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
right = pl.LazyFrame({"x": [-1, -2, 3], "z": [7, 8, 9]})

left.piot.filtered_join(right, on="x").collect()
# shape: (1, 3)
# ┌─────┬─────┬─────┐
# │ x   ┆ y   ┆ z   │
# │ --- ┆ --- ┆ --- │
# │ i64 ┆ i64 ┆ i64 │
# ╞═════╪═════╪═════╡
# │ 3   ┆ 6   ┆ 9   │
# └─────┴─────┴─────┘
```

The result is identical to a plain `join`, but the right frame only ever produced the
rows whose `x` matched the left frame. When `right` is a remote `scan_*` source, that
pushed-down `is_in` filter becomes part of the query against the source.

## Step 3 — keep pushdown through a rolling window

Rolling, cumulative, and forward-fill expressions normally block filter pushdown,
because Polars cannot know which rows a window touches. `ts_with_columns` solves this:
it extracts filters on a time-like column, expands them by a `lookback` (and optional
`lookahead`), applies that widened filter *first*, then computes the window, and finally
re-applies your original filter to trim the result.

```python
from datetime import date, timedelta

df = pl.LazyFrame({
    "Date": [date(2025, 1, i) for i in range(1, 6)],
    "EventDate": [date(2025, 1, i) for i in range(2, 7)],
    "Value": [10, 20, 30, 40, 50],
})

(
    df.piot.ts_with_columns(
        pl.col("Value").cum_sum().alias("CumValue"),
        index_col="Date",
        lookback=timedelta(days=3),
        linked_cols=["EventDate"],
    )
    .filter(pl.col("EventDate") >= date(2025, 1, 5))
    .collect()
)
# shape: (2, 4)
# ┌────────────┬────────────┬───────┬──────────┐
# │ Date       ┆ EventDate  ┆ Value ┆ CumValue │
# │ ---        ┆ ---        ┆ ---   ┆ ---      │
# │ date       ┆ date       ┆ i64   ┆ i64      │
# ╞════════════╪════════════╪═══════╪══════════╡
# │ 2025-01-04 ┆ 2025-01-05 ┆ 40    ┆ 90       │
# │ 2025-01-05 ┆ 2025-01-06 ┆ 50    ┆ 140      │
# └────────────┴────────────┴───────┴──────────┘
```

The filter on `EventDate` was converted to a filter on `Date` expanded three days back,
so the cumulative sum sees the historical rows it needs, yet only the requested
`EventDate` rows are returned.

## Step 4 — cache results while you iterate

When you are exploring data and re-running queries, `cache` stores already-computed
columns (optionally per partition) so later collects reuse them instead of recomputing.

```python
df = (
    pl.LazyFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    .with_columns((pl.col("x") * 2).alias("slow"))
    .piot.cache(order_by="x")
)

# First collect computes "slow" and stores it in the cache.
df.select(pl.col("slow").max()).collect()

# This collect reuses the cached "slow" column instead of recomputing it.
df.select(pl.col("slow").min()).collect()
```

By default the cache is an in-memory dictionary; pass your own mapping (for example a
`diskcache.Cache`) to persist results across sessions. `order_by` is a column (or columns)
that uniquely identifies each row, so columns cached in separate collects stay aligned.

## What you built

You wrote ordinary-looking Polars queries — a projection, a join, a windowed
calculation, and a cached pipeline — and in every case `polars-io-tools` arranged for
filters and column selections to be applied as early as possible. On in-memory frames
the payoff is invisible; against a SQL warehouse, ClickHouse, Datadog, or a partitioned
Delta table, the same code reads dramatically less data.

## Next steps

- Point these techniques at real systems in [Reading and Writing Data](Reading-and-Writing-Data).
- Go deeper on joins, multi-source composition, and caching in [Query Optimization](Query-Optimization).
- Understand the machinery in [Concepts](Concepts).
- Look up exact signatures in the [API Reference](API-Reference).
