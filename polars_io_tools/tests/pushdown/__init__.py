"""Tests that record what plain Polars hands to a data source (filters, column selections and row counts) for each
kind of query.

There is one file per Polars operation. Most hold a table whose rows build a query over sources that record what they
receive (``PredicateTracker``) and state, per source, whether a filter or a column selection reached it. Pivot and unpivot
rows are in ``io_sources/test_pushdown_pivot.py`` and ``io_sources/test_pushdown_unpivot.py``, next to the wrappers
they are compared with.

Rows marked ``pytest.mark.gap`` record a filter or selection Polars could safely hand to the source but does not; if a
new Polars version starts doing it, the row fails and should be updated. Each such row's query is self-contained, so it
can be copied into a Polars GitHub issue as a short example.
"""
