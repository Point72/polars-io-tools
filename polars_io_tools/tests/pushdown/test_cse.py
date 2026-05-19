"""CSE-vs-pushdown priority lock-in.

When the same LazyFrame is filtered with two different predicates and the
results concatenated::

    pl.concat([lf.filter(p1), lf.filter(p2)])

polars currently honors pushdown — pushing ``p1`` into one scan of ``lf``
and ``p2`` into another — at the cost of scanning ``lf`` twice. CSE could
collapse this to a single scan with the predicate ``p1 | p2`` pushed once
plus a residual split downstream, paying for one source read instead of two.
Downstream code can work around this by explicitly suppressing the pushdown
so CSE can collapse the scans.

Tracked upstream: https://github.com/pola-rs/polars/issues/26502
("Common subplan elimination is pessimized by filters").
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_io_tools.testing import PredicateTracker


@pytest.fixture
def tracker() -> PredicateTracker:
    return PredicateTracker(pl.DataFrame({"k": [1, 2, 3, 4, 5], "v": [10, 20, 30, 40, 50]}))


@pytest.mark.gap
def test_concat_of_two_filters_scans_source_twice(tracker: PredicateTracker):
    """Lock-in: ``pl.concat([lf.filter(p1), lf.filter(p2)])`` scans the
    source twice (call_count == 2). A CSE-aware optimizer could collapse
    the two filtered branches into a single source scan with predicate
    ``p1 | p2`` pushed once."""
    pl.concat(
        [
            tracker.lazy_frame.filter(pl.col("k") < 3),
            tracker.lazy_frame.filter(pl.col("k") > 3),
        ]
    ).collect()
    assert tracker.call_count == 2, f"expected 2 source scans (CSE not applied); got {tracker.call_count}"
