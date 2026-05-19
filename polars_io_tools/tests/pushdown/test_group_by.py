"""Pushdown matrix for ``pl.LazyFrame.group_by`` + aggregation.

Window/``.over(...)`` cases live in ``test_window.py``.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_io_tools.testing import PredicateTracker


@pytest.fixture
def tracker() -> PredicateTracker:
    return PredicateTracker(
        pl.DataFrame(
            {
                "g": ["a", "a", "b", "b"],
                "h": [1, 1, 2, 2],
                "k": [1, 10, 2, 20],
                "v": [10, 20, 30, 40],
                "w": [1, 2, 3, 4],
            }
        )
    )


def test_filter_on_group_key_pushed(tracker: PredicateTracker):
    """Baseline: filter on the grouping key is pushed."""
    tracker.lazy_frame.group_by("g").agg(pl.col("v").sum()).filter(pl.col("g") == "a").collect()
    assert tracker.last_predicate is not None


def test_filter_on_secondary_group_key_pushed(tracker: PredicateTracker):
    """Baseline: filter on a secondary grouping key is pushed."""
    tracker.lazy_frame.group_by(["g", "h"]).agg(pl.col("v").sum()).filter(pl.col("h") == 1).collect()
    assert tracker.last_predicate is not None


def test_select_subset_of_aggs_prunes_unused_source_cols(tracker: PredicateTracker):
    """Baseline: when downstream selects only some of the agg outputs,
    polars drops the source columns that fed the unused aggs."""
    tracker.lazy_frame.group_by("g").agg(
        pl.col("v").sum().alias("vs"),
        pl.col("w").sum().alias("ws"),
    ).select(["g", "vs"]).collect()
    assert tracker.last_with_columns is not None
    assert "w" not in tracker.last_with_columns


def test_group_by_literal_key_pushes_projection(tracker: PredicateTracker):
    """Regression coverage for https://github.com/pola-rs/polars/issues/22623:
    ``group_by(pl.lit(1))`` should still allow projection pushdown of the
    source columns that feed the aggs."""
    tracker.lazy_frame.group_by(pl.lit(1)).agg(pl.col("v").sum()).collect()
    assert tracker.last_with_columns is not None
    assert "v" in tracker.last_with_columns


@pytest.mark.gap
def test_expr_filter_inside_agg_NOT_pushed(tracker: PredicateTracker):
    """Lock-in: ``col.filter(predicate).sum()`` inside an agg could push the
    predicate as a source-side filter (since rows where the predicate is
    false contribute nothing to the agg). Polars does not perform this
    rewrite.

    See https://github.com/pola-rs/polars/issues/27207.
    """
    tracker.lazy_frame.group_by("g").agg(pl.col("v").filter(pl.col("k") > 5).sum()).collect()
    assert tracker.last_predicate is None, f"unexpectedly pushed: {tracker.last_predicate!r}"
