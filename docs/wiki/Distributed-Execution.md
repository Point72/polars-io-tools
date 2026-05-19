# Distributed execution

`execute_on_ray` runs a `LazyFrame` across an already-initialised
[Ray](https://www.ray.io/) cluster by splitting the work into calendar periods. Each
period becomes a Ray task that filters the source to its own time window and collects
independently; the results are streamed back and concatenated. Because the split is
driven by a predicate on the date column, each task only reads the data for its slice.

```python
import polars as pl
import ray
import polars_io_tools  # registers the .piot namespace

ray.init()  # connect to or start a Ray cluster first

lf = scan_db(
    "SELECT date, symbol, price FROM trades",
    connection="...",
)

result = (
    lf.filter(pl.col("date").is_between(pl.date(2024, 1, 1), pl.date(2024, 3, 31)))
    .piot.execute_on_ray(date_column="date", time_unit="monthly")
    .collect()
)
```

## How the work is split

- `date_column` names the `Date`/`Datetime` column that defines the partitioning axis.
- `time_unit` chooses the granularity of each task: `"daily"`, `"monthly"`, or
  `"yearly"`.
- The query above splits into three monthly tasks (January, February, March 2024), each
  fetching only its month.

## Requirements

- **Call `ray.init()` first.** `execute_on_ray` raises if Ray is not initialised.
- **Provide a bounded predicate on `date_column`.** The function derives each task's time
  range from the pushed-down filter, so a two-sided bound (such as `is_between(...)`, or
  a pair of `>=` / `<=` filters) is required. Without it the call raises.

## Tuning

- `return_as` controls how each worker ships its result: `"arrow"` (zero-copy Arrow
  buffers, the default), `"ipc"`, or `"parquet"`.
- `remote_options` is passed through to each Ray task's `.options(...)` — use it to
  request `num_cpus`, `num_gpus`, or a runtime environment (for example to set
  `POLARS_MAX_THREADS`).
- `max_concurrency` caps how many tasks run at once (default 100), following Ray's
  [limit-pending-tasks](https://docs.ray.io/en/latest/ray-core/patterns/limit-pending-tasks.html)
  pattern.

```python
lf.piot.execute_on_ray(
    date_column="date",
    time_unit="daily",
    remote_options={"num_cpus": 4},
    max_concurrency=50,
).collect()
```

## See also

- [Query Optimization](Query-Optimization) — build the `LazyFrame` you distribute.
- [API Reference](API-Reference#execute_on_ray) — full signature.
