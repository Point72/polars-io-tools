"""Pushdown characterization suite for plain polars LazyFrame pipelines.

Lock-in rows record cases where polars' optimizer fails to push down filters or projections that are logically sound.
Each ``gap`` row's pipeline is self-contained so it can be lifted into a polars GitHub issue as a minimal reproducer.

Validated against polars 1.40.1.
"""
