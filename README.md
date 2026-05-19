# polars io tools

Custom parsing extensions for lazy polars

[![Build Status](https://github.com/Point72/polars-io-tools/actions/workflows/build.yaml/badge.svg?branch=main&event=push)](https://github.com/Point72/polars-io-tools/actions/workflows/build.yaml)
[![codecov](https://codecov.io/gh/Point72/polars-io-tools/branch/main/graph/badge.svg)](https://codecov.io/gh/Point72/polars-io-tools)
[![License](https://img.shields.io/github/license/Point72/polars-io-tools)](https://github.com/Point72/polars-io-tools)
[![PyPI](https://img.shields.io/pypi/v/polars-io-tools.svg)](https://pypi.python.org/pypi/polars-io-tools)

## Overview

`polars-io-tools` extends [Polars](https://pola.rs/) lazy execution with custom I/O
sources that push filters and column projections all the way down into the systems
that hold your data — SQL databases, ClickHouse, Datadog, and Delta Lake — instead of
loading everything and filtering in memory. It also adds lazy-friendly operations
(joins, multi-source composition, time-series windows, caching, distributed execution)
that keep predicate pushdown working where vanilla Polars would otherwise give up and
materialize the whole frame.

Everything is exposed through the `piot` LazyFrame namespace and a handful of
top-level `scan_*` / `sink_*` functions, so it composes naturally with the Polars
API you already use.

### Who is this for

Reach for `polars-io-tools` when you want Polars' lazy API over data that lives in an
external store, and you care about *not* fetching rows or columns you will immediately
throw away. It is most valuable for large, partitioned, or remote datasets where a
filter on a date or key column should translate into a smaller query against the
source. If your data already fits comfortably in memory or lives in local Parquet/CSV,
plain Polars is the simpler choice.

## Installation

```bash
pip install polars-io-tools
```

`polars-io-tools` requires Python 3.11 or newer. See the
[Installation guide](https://github.com/Point72/polars-io-tools/wiki/Installation)
for `conda` and source builds.

## Quickstart

Importing the package registers the `piot` namespace on every Polars `LazyFrame`:

```python
import polars as pl
import polars_io_tools  # registers the .piot namespace

left = pl.LazyFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
right = pl.LazyFrame({"x": [-1, -2, 3], "z": [7, 8, 9]})

# An inner join where the keys present on the left are pushed down as a
# filter on the right frame *before* the join runs.
result = left.piot.filtered_join(right, on="x").collect()
print(result)
# shape: (1, 3)
# ┌─────┬─────┬─────┐
# │ x   ┆ y   ┆ z   │
# │ --- ┆ --- ┆ --- │
# │ i64 ┆ i64 ┆ i64 │
# ╞═════╪═════╪═════╡
# │ 3   ┆ 6   ┆ 9   │
# └─────┴─────┴─────┘
```

For a guided walkthrough, start with the
[Getting Started tutorial](https://github.com/Point72/polars-io-tools/wiki/Getting-Started).

## What's included

- **Lazy I/O sources with predicate & projection pushdown** — `scan_db` (any
  ODBC database), `scan_clickhouse`, `scan_datadog`, `scan_delta`, and `from_narwhals`.
  Filters on the resulting `LazyFrame` are translated into the source's own query
  language (SQL `WHERE`, Datadog time ranges, Delta partition pruning) so only the
  matching rows and columns are fetched.
- **Lazy writers** — `sink_delta` and `sink_clickhouse` write a `LazyFrame` directly to
  Delta Lake or ClickHouse, including streaming/chunked writes and transparent handling
  of types the target store cannot represent natively.
- **Pushdown-preserving query building** — `filtered_join`, `filtered_join_asof`,
  `join_between`, `multi_source`, `concat_named`, and `ts_with_columns` express joins,
  multi-source composition, and rolling/lookback time-series logic without blocking the
  filter pushdown that those operations normally defeat.
- **Caching** — `cache` keeps an in-memory, column- and partition-level cache for
  iterative work; `cache_parquet` materializes date-partitioned Parquet on local disk
  or S3, fetching only the partitions a query needs.
- **Distributed execution** — `execute_on_ray` splits a `LazyFrame` by calendar period
  and runs the partitions across an existing Ray cluster.
- **Ergonomics** — `iter_rows` for memory-efficient row iteration, `debug` to inspect
  what Polars pushes into a source, and `disable_optimizations` to compare against plain
  Polars.

## Documentation

Full documentation lives in the
[project wiki](https://github.com/Point72/polars-io-tools/wiki):

- [Getting Started](https://github.com/Point72/polars-io-tools/wiki/Getting-Started) — a guided tutorial.
- [Reading and Writing Data](https://github.com/Point72/polars-io-tools/wiki/Reading-and-Writing-Data) — recipes for each I/O source and sink.
- [Query Optimization](https://github.com/Point72/polars-io-tools/wiki/Query-Optimization) — joins, composition, time-series, and caching.
- [Distributed Execution](https://github.com/Point72/polars-io-tools/wiki/Distributed-Execution) — running on Ray.
- [API Reference](https://github.com/Point72/polars-io-tools/wiki/API-Reference) — the public functions and `piot` namespace.
- [Concepts](https://github.com/Point72/polars-io-tools/wiki/Concepts) — how predicate pushdown into custom sources works.

## Contributing

Contributions are welcome. See the
[Contributing guide](https://github.com/Point72/polars-io-tools/wiki/Contribute)
and [Local Development Setup](https://github.com/Point72/polars-io-tools/wiki/Local-Development-Setup)
to get started.

## License

`polars-io-tools` is licensed under the [Apache 2.0 license](LICENSE).
