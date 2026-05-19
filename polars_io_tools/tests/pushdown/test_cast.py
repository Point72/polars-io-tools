"""Pushdown matrix for cast-before-filter rewrites.

A widening cast (e.g. ``Int32 -> Int64``) followed by a filter on the cast
output could be rewritten as a filter on the source column with the
constant likewise re-typed: the cast preserves all source values exactly,
so any predicate that holds on the cast value holds on the source value
too.

Soundness caveat: this rewrite is **only** safe for widening casts. A
narrowing cast (e.g. ``Int64 -> Int32``) can wrap or overflow, so the
predicate evaluated on the cast value differs from the predicate evaluated
on the source value. Tests in this file use a widening cast only.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_io_tools.testing import PredicateTracker


@pytest.fixture
def tracker() -> PredicateTracker:
    return PredicateTracker(pl.DataFrame({"x": pl.Series([1, 2, 3, 4, 5], dtype=pl.Int32)}))


@pytest.mark.gap
def test_widening_cast_then_filter_NOT_pushed(tracker: PredicateTracker):
    """Lock-in: ``with_columns(col("x").cast(Int64)).filter(col("x") > 2)``
    where source ``x`` is ``Int32`` is not pushed. The cast is widening and
    lossless, so the filter is equivalent to ``col("x") > 2`` on the source.

    Related: https://github.com/pola-rs/polars/issues/23369 (general
    optimizer pass for filter-before-cast rewrites; widening pushdown is
    described as the first sub-case there).
    """
    tracker.lazy_frame.with_columns(pl.col("x").cast(pl.Int64)).filter(pl.col("x") > 2).collect()
    assert tracker.last_predicate is None, f"unexpectedly pushed: {tracker.last_predicate!r}"
