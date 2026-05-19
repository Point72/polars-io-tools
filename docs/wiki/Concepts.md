# Concepts

This page explains how `polars-io-tools` works and why it can push filters into systems
that Polars otherwise treats as opaque. It is background reading — for step-by-step
instructions see [Reading and Writing Data](Reading-and-Writing-Data) and
[Query Optimization](Query-Optimization).

## Polars I/O sources

Polars lets you register a Python callable as a lazy source. When a query that scans such
a source is collected, the optimizer calls back into your function with the work it has
already pushed as far down the plan as it can:

- the **column projection** — the subset of columns the rest of the query actually needs,
- the **predicate** — a single `pl.Expr` combining the filters that apply to this scan,
- a **row limit** and a **batch size**.

Your function is then free to use those hints however it likes before yielding
DataFrames. A naive source ignores them and returns everything; a smart source reads only
the requested columns and rows. Every `scan_*` function in this library is such a source:
it receives the projection and predicate and turns them into a narrower query against the
real backend — a SQL `WHERE` clause for `scan_db` and `scan_clickhouse`, a time range for
`scan_datadog`, a set of partition files for `scan_delta`. That is what "pushdown" means
here: the filter runs at the source, not in memory after the fact.

You can watch this happen with `lf.piot.debug()`, which prints exactly the projection,
predicate, and limits Polars hands to a source.

## Translating a predicate into a backend query

The predicate Polars passes down is a Polars expression tree, not SQL or a date range. To
push it into an external system the library has to *understand* it. It parses the
expression into a small abstract syntax tree and walks it with purpose-built visitors,
each of which answers a different question about the filter:

- **Disjunctive normal form** — flatten the predicate into a set of clauses, each a
  conjunction of `(column, operator, value)` constraints, so it can be analyzed column by
  column.
- **Range extraction** — reduce the constraints on a date/datetime column to a single
  interval, which becomes a Datadog time range, a Ray partition boundary, or a
  `BETWEEN` in SQL.
- **Valid-value extraction** — collect the allowed and excluded values for a column,
  which becomes an `IN (...)` list.
- **Column restriction** — rewrite a predicate so it only references a subset of columns,
  producing a weaker but still-correct filter to push to one side of a join or one source
  of a composition.

Because a filter is only pushed down when it can be translated soundly, the worst case is
simply that some filtering happens in memory instead of at the source — never that the
wrong rows come back. These visitors are part of the public API for authors writing their
own sources; most users never touch them directly.

## Why ordinary operations block pushdown — and how the tools restore it

Polars is conservative about pushing filters past operations whose result depends on rows
it cannot see in advance. Three common cases motivate most of this library:

- **Joins.** An inner join discards right-side rows that have no left-side match, but
  Polars does not turn that into a filter on the right source, so the whole right side is
  read. [`filtered_join`](Query-Optimization#pre-filter-the-right-side-of-a-join)
  materializes the left frame, extracts its keys, and pushes them to the right side as an
  `is_in(...)` filter first.

- **Rolling and cumulative windows.** A `cum_sum` or rolling mean reads neighbouring rows,
  so Polars stops pushing any time filter past it — otherwise the window would be computed
  over the wrong rows.
  [`ts_with_columns`](Query-Optimization#apply-rolling-windows-without-losing-pushdown)
  widens the time filter by a `lookback`/`lookahead`, applies it *before* the window so
  the source still reads less data, computes the window over the widened set, and trims
  back to the original request afterward.

- **Concatenation over a constant column.** Filtering a `pl.concat` on a literal-valued
  identifier column does not prune the branches that cannot match, so every branch runs.
  [`concat_named`](Query-Optimization#concatenate-frames-and-prune-by-identifier)
  intercepts the predicate and only materializes the matching frames.

In each case the trick is the same: intercept the predicate at a custom source, transform
it into something that is safe to apply earlier, and re-apply the exact original filter at
the end so the result is identical to the naive version. [`multi_source`](Query-Optimization#combine-sources-with-coordinated-filter-pushdown)
generalises this to arbitrary compositions, with a per-source `FilterSpec` describing how
each output filter maps onto each source.

## Logical types the backend cannot store

Some Polars logical types have no native representation in a target store. Delta Lake, for
example, supports `Datetime[us]` but not `Datetime[ns]`, `Datetime[ms]`, `Duration`, or
`Time`; ClickHouse has no `Duration`, `Time`, or categorical type. Rather than refuse to
write such columns, `sink_delta` and `sink_clickhouse` cast them to integers on write.
`sink_delta` additionally records the original-to-stored type mapping in the table
metadata, and `scan_delta` reads that mapping back to cast the columns to their logical
types on read — and to translate filters on those logical columns into filters on the
stored integer representation, preserving pushdown even across the type change.

## A note on correctness

The optimizations here change *when and where* filtering happens, not *what* the answer
is. Every transformed filter is paired with the original predicate so the final output
matches plain Polars. The main assumption to be aware of is row-order stability: `cache`
stores columns independently and reassembles them by position, so a source that returns
rows in a different order on each collect can misalign cached columns. Use
`disable_optimizations()` to run a query through plain-Polars equivalents when you want to
compare results directly.

## See also

- [API Reference](API-Reference) — the public functions and `piot` namespace.
- [Query Optimization](Query-Optimization) — applying these ideas.
