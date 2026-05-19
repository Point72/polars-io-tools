"""Pushdown matrix for ``pl.LazyFrame.join_asof``.

Notable cases that are deliberately *NOT* tested as lock-ins because they
are not unconditionally sound:

* Pushing a filter on the ``on=`` key to the right side is unsound for
  backward asof: a right row with ``t=2`` can still be the valid backward
  match for a left row with ``t=5``. Filtering ``t >= 3`` on the right
  drops valid match candidates.
* Pushing a filter on a right-only column to the right side changes the
  set of match candidates and is not equivalent to filtering after the
  join.
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from polars_io_tools.testing import PredicateTracker


@pytest.fixture
def tracker_a() -> PredicateTracker:
    df = pl.DataFrame({"t": [1, 2, 3, 4], "g": ["a", "a", "b", "b"], "va": [10, 20, 30, 40]}).sort("t")
    return PredicateTracker(df)


@pytest.fixture
def tracker_b() -> PredicateTracker:
    df = pl.DataFrame({"t": [1, 2, 3, 4], "g": ["a", "a", "b", "b"], "vb": [100, 200, 300, 400]}).sort("t")
    return PredicateTracker(df)


def test_filter_on_by_column_pushed(tracker_a: PredicateTracker, tracker_b: PredicateTracker):
    """Baseline: filter on the ``by=`` column is pushed to both sides — a
    ``by`` value either matches in both sides or yields a null in the join
    output, so restricting it pre-join is equivalent."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tracker_a.lazy_frame.join_asof(tracker_b.lazy_frame, on="t", by="g").filter(pl.col("g") == "a").collect()
    assert tracker_a.last_predicate is not None
    assert tracker_b.last_predicate is not None


@pytest.mark.gap
def test_post_join_elementwise_expr_on_right_col_NOT_moved_into_right_scan(tracker_a: PredicateTracker, tracker_b: PredicateTracker):
    """Lock-in: a post-join column-wise pure expression on a right-only
    column (e.g. ``(col("vb") + 1).alias(...)``) could be moved into the
    right-side scan as a ``with_columns`` — the asof join only inspects
    the ``on=`` and ``by=`` columns, and the expression is null-preserving
    and row-local. Polars does not perform this rewrite.

    See https://github.com/pola-rs/polars/issues/25867. Soundness caveat:
    only applies to null-preserving column-wise pure expressions.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tracker_a.lazy_frame.join_asof(tracker_b.lazy_frame, on="t", by="g").with_columns((pl.col("vb") + 1).alias("vb1")).collect()
    assert tracker_b.last_with_columns is None or "vb1" not in (tracker_b.last_with_columns or []), (
        f"unexpectedly moved into right scan: {tracker_b.last_with_columns!r}"
    )
