"""Pushdown characterization suite for plain polars LazyFrame pipelines.

Lock-in tests that record cases where polars' optimizer fails to push down
filters or projections that are logically sound. Each ``test_*_NOT_pushed``
test is intentionally self-contained so its body can be lifted directly
into a polars GitHub issue as a minimal reproducer.

Validated against polars 1.40.1.
"""
