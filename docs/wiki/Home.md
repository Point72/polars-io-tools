# polars-io-tools

`polars-io-tools` extends [Polars](https://pola.rs/) lazy execution with custom I/O
sources that push filters and column projections down into the systems that hold your
data — SQL databases, ClickHouse, Datadog, and Delta Lake — instead of loading
everything and filtering in memory. It also adds lazy-friendly operations (joins,
multi-source composition, time-series windows, caching, distributed execution) that
keep predicate pushdown working where vanilla Polars would otherwise materialize the
whole frame.

Everything is exposed through the `piot` LazyFrame namespace and a few top-level
`scan_*` / `sink_*` functions, so it composes with the Polars API you already use.

## Where to go next

This documentation follows the [Diátaxis](https://diataxis.fr/) structure — pick the
page that matches what you are trying to do.

- **[Getting Started](Getting-Started)** — a guided, hands-on tutorial that builds a
  small pushdown-aware pipeline from scratch. Start here if you are new.
- **[Reading and Writing Data](Reading-and-Writing-Data)** — task recipes for each I/O
  source (`scan_db`, `scan_clickhouse`, `scan_datadog`, `scan_delta`, `from_narwhals`)
  and sink (`sink_delta`, `sink_clickhouse`).
- **[Query Optimization](Query-Optimization)** — recipes for joins, multi-source
  composition, time-series lookback windows, and caching while preserving pushdown.
- **[Distributed Execution](Distributed-Execution)** — running a `LazyFrame` across a
  Ray cluster with `execute_on_ray`.
- **[API Reference](API-Reference)** — the public functions and `piot` namespace
  methods, with signatures.
- **[Concepts](Concepts)** — how Polars I/O sources work and why these tools can push
  filters into external systems.

## Getting set up

- **[Installation](Installation)** — install with `pip`, `conda`, or from source.
- **[Contributing](Contribute)** — how to report issues and propose changes.
- **[Local Development Setup](Local-Development-Setup)** — set up a development
  environment and commit signing.
- **[Build from Source](Build-from-Source)** — build the package locally.
