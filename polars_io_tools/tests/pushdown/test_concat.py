"""Pushdown matrix for ``pl.concat`` (vertical and horizontal)."""

from __future__ import annotations

import polars as pl
import pytest

from polars_io_tools.testing import PredicateTracker


@pytest.fixture
def tracker_left() -> PredicateTracker:
    return PredicateTracker(pl.DataFrame({"k": [1, 2, 3], "a": [10, 20, 30]}))


@pytest.fixture
def tracker_right() -> PredicateTracker:
    return PredicateTracker(pl.DataFrame({"k": [1, 2, 3], "b": [100, 200, 300]}))


def test_vertical_concat_filter_on_shared_col_pushed(tracker_left: PredicateTracker, tracker_right: PredicateTracker):
    """Baseline: vertical concat pushes a filter on a shared column to both sides."""
    a = tracker_left.lazy_frame.select(["k", "a"])
    b = tracker_right.lazy_frame.select(["k", pl.col("b").alias("a")])
    pl.concat([a, b]).filter(pl.col("k") == 2).collect()
    assert tracker_left.last_predicate is not None, "left side predicate not pushed"
    assert tracker_right.last_predicate is not None, "right side predicate not pushed"


@pytest.mark.gap
def test_horizontal_concat_filter_on_left_col_NOT_pushed(tracker_left: PredicateTracker, tracker_right: PredicateTracker):
    """Lock-in: ``pl.concat(how="horizontal")`` does not push a filter on a
    column from one of its inputs down to that input. Because horizontal
    concat is row-positional and requires equal lengths, a sound rewrite
    must apply the same row-position mask to every input — not push the
    column predicate to one side. Polars does neither today: the predicate
    sits above the concat and both inputs are scanned in full.

    See https://github.com/pola-rs/polars/issues/26552.
    """
    out = pl.concat(
        [tracker_left.lazy_frame, tracker_right.lazy_frame.select("b")],
        how="horizontal",
    ).filter(pl.col("k") > 1)
    out.collect()
    assert tracker_left.last_predicate is None, f"unexpectedly pushed: {tracker_left.last_predicate!r}"


@pytest.mark.gap
def test_vertical_concat_literal_side_NOT_pruned():
    """Lock-in: when one side of a vertical concat tags rows with a literal
    label and the filter selects the other label, that side could be pruned
    entirely (constant folding + dead-code elimination).

    Concrete case::

        a = scan_a.with_columns(pl.lit("real").alias("kind"))
        b = scan_b.with_columns(pl.lit("synthetic").alias("kind"))
        pl.concat([a, b]).filter(pl.col("kind") == "real")

    Sound rewrite: drop ``b`` from the plan entirely. Polars currently scans
    both sides (call_count==1 on each tracker).

    No exact upstream issue tracks this case as of the validation date.
    Umbrella: https://github.com/pola-rs/polars/issues/23345.
    """
    ta = PredicateTracker(pl.DataFrame({"k": [1, 2, 3], "v": [10, 20, 30]}))
    tb = PredicateTracker(pl.DataFrame({"k": [4, 5], "v": [40, 50]}))
    a = ta.lazy_frame.with_columns(pl.lit("real").alias("kind"))
    b = tb.lazy_frame.with_columns(pl.lit("synthetic").alias("kind"))
    pl.concat([a, b]).filter(pl.col("kind") == "real").collect()
    assert ta.call_count == 1
    assert tb.call_count == 1, f"synthetic side was scanned {tb.call_count} time(s); should be pruned (0)"
