"""Pushdown matrix for regular ``pl.LazyFrame.join``.

Asof joins are in ``test_join_asof.py``; the lookup-join lock-in is in
``test_join_lookup.py``.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_io_tools.testing import PredicateTracker


@pytest.fixture
def tracker_a() -> PredicateTracker:
    return PredicateTracker(pl.DataFrame({"k": [1, 2, 3], "a": [10, 20, 30]}))


@pytest.fixture
def tracker_b() -> PredicateTracker:
    return PredicateTracker(pl.DataFrame({"k": [1, 2, 3], "b": [100, 200, 300], "c": [1, 2, 3]}))


def test_inner_join_filter_on_key_pushed(tracker_a: PredicateTracker, tracker_b: PredicateTracker):
    """Baseline: an inner join filter on the join key is pushed to both sides."""
    tracker_a.lazy_frame.join(tracker_b.lazy_frame, on="k").filter(pl.col("k") == 2).collect()
    assert tracker_a.last_predicate is not None
    assert tracker_b.last_predicate is not None


def test_right_join_filter_on_key_pushed(tracker_a: PredicateTracker, tracker_b: PredicateTracker):
    """Baseline: right joins also push the join-key filter to both sides."""
    tracker_a.lazy_frame.join(tracker_b.lazy_frame, on="k", how="right").filter(pl.col("k") == 2).collect()
    assert tracker_a.last_predicate is not None
    assert tracker_b.last_predicate is not None


def test_inner_join_projection_prunes_unused_right_cols(tracker_a: PredicateTracker, tracker_b: PredicateTracker):
    """Baseline: selecting only ``[k, a, b]`` from the join lets polars drop
    ``c`` from the right-side scan."""
    tracker_a.lazy_frame.join(tracker_b.lazy_frame, on="k").select(["k", "a", "b"]).collect()
    assert tracker_b.last_with_columns is not None
    assert "c" not in tracker_b.last_with_columns
