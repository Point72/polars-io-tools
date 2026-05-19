"""Pushdown matrix for window expressions (``.over(...)``).

All cases here are baselines (regression coverage). Window expressions
preserve row count, so a filter on the *partition* column is sound to push
to the source: dropping partitions that the downstream doesn't use cannot
change the values within retained partitions.

Filters on *unrelated* source columns are NOT included as lock-ins because
they are NOT sound to push past a ``with_columns(... .over(k))``: removing
rows from a partition changes the partition's window output (e.g. for
``shift``, ``cumsum``, ``rank``).
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_io_tools.testing import PredicateTracker


@pytest.fixture
def tracker() -> PredicateTracker:
    return PredicateTracker(pl.DataFrame({"k": [1, 1, 2, 2, 3], "v": [10, None, 30, None, 50]}))


def test_filter_on_partition_col_of_shift_over_pushed(tracker: PredicateTracker):
    """Baseline: ``filter(k == 1)`` after ``shift().over("k")`` is pushed —
    other partitions are independent and dropping them doesn't affect the
    retained partition's shift output."""
    tracker.lazy_frame.with_columns(pl.col("v").shift(1).over("k").alias("vs")).filter(pl.col("k") == 1).collect()
    assert tracker.last_predicate is not None


def test_filter_on_partition_col_of_forward_fill_over_pushed(tracker: PredicateTracker):
    """Baseline: same for ``forward_fill().over("k")``."""
    tracker.lazy_frame.with_columns(pl.col("v").forward_fill().over("k").alias("vs")).filter(pl.col("k") == 1).collect()
    assert tracker.last_predicate is not None
