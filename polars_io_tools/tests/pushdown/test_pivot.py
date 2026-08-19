"""
Pushdown matrix for ``pl.LazyFrame.pivot`` (currently unstable in polars).

Lazy ``pivot`` is implemented internally as a ``group_by`` plus a set of ``filter().item()`` aggregations — one per value
listed in ``on_columns``. Pushdown therefore behaves like a group-by aggregation:

* Filters on the ``index`` column flow through to the source.
* If the downstream selects only the ``index`` column, polars drops all the pivot aggregations and the source is asked for
  just the index column.
* For all other cases (filter on a pivoted output column, selecting a subset of pivoted output columns) polars does not
  synthesize an upstream rewrite — and for the cases below this is *correct*, not a missed optimization. See the per-test
  docstrings for the soundness rationale.

These tests assert pushdown behavior only. End-to-end result correctness of ``pivot`` itself is exercised by polars' own test
suite; the polars-io-tools ``pushdown_pivot`` wrapper exposes the soundness preconditions explicitly (``dense_on=True``) and is
verified against bare polars in ``polars_io_tools/tests/io_sources/test_pushdown_pivot.py``.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_io_tools.testing import PredicateTracker
from polars_io_tools.tests.pushdown.utils import _check


@pytest.fixture
def tracker() -> PredicateTracker:
    df = pl.DataFrame(
        {
            "name": ["A", "A", "A", "B", "B", "B"],
            "subject": ["m", "p", "x", "m", "p", "x"],
            "score": [1, 2, 3, 4, 5, 6],
        }
    )
    return PredicateTracker(df)


def _pivoted(tracker: PredicateTracker) -> pl.LazyFrame:
    return tracker.lazy_frame.pivot(
        "subject",
        on_columns=["m", "p", "x"],
        index="name",
        values="score",
    )


def test_filter_on_index_is_pushed(tracker: PredicateTracker):
    """Filter on the `index` column passes through the implicit group_by."""
    lf = _pivoted(tracker).filter(pl.col("name") == "A")
    _check(tracker, lf, predicate_pushed=True)


def test_filter_on_pivoted_output_NOT_pushed(tracker: PredicateTracker):
    """Filter on a pivoted output column (`m`) is post-aggregation; not pushed.

    With the default ``sum`` aggregation, ``col("m") > 2`` is a HAVING clause on ``sum(score where subject == "m")`` —
    pushing ``score > 2`` upstream would discard rows whose individual scores are below the threshold but whose group-sum
    exceeds it. Polars' choice not to push is correct, not a missed optimization.
    """
    lf = _pivoted(tracker).filter(pl.col("m") > 2)
    _check(tracker, lf, predicate_pushed=False)


def test_filter_on_pivoted_output_is_not_null_NOT_pushed(tracker: PredicateTracker):
    """`is_not_null` on a pivoted column is also a correct non-push.

    Rewriting ``filter(col("m").is_not_null())`` as the upstream filter ``subject == "m"`` would survive only those source
    rows, collapsing the values of the other pivoted columns (``p``, ``x``) to all-null in the output. The user-visible
    result differs from bare pivot. Polars correctly leaves this filter post-pivot.
    """
    lf = _pivoted(tracker).filter(pl.col("m").is_not_null())
    _check(tracker, lf, predicate_pushed=False)


def test_select_index_only_pushes_projection(tracker: PredicateTracker):
    """Selecting only the index makes polars drop the pivot aggregations and ask the source for just the index column.
    (Already optimized.)"""
    lf = _pivoted(tracker).select("name")
    _check(tracker, lf, projection_pushed=True, pushed_columns={"name"})


def test_select_subset_of_pivoted_output_NOT_pushed_as_filter(tracker: PredicateTracker):
    """Selecting `[name, m]` is a correct non-push for bare pivot.

    The candidate rewrite — source filter ``subject == "m"`` plus projection ``[name, subject, score]`` — is unsound when
    the source is sparse over ``on``: any index value that has no row with ``subject == "m"`` would silently disappear from
    the output, whereas bare pivot keeps it with ``m = null``. The bare optimizer cannot prove the density precondition, so
    it correctly does not push. ``pushdown_pivot`` exposes the precondition as ``dense_on=True``.
    """
    lf = _pivoted(tracker).select(["name", "m"])
    _check(tracker, lf, projection_pushed=False)


def test_select_two_pivoted_outputs_NOT_pushed_as_filter(tracker: PredicateTracker):
    """Same soundness rationale as the single-column case, with the candidate upstream filter
    ``subject.is_in(["m", "p"])``."""
    lf = _pivoted(tracker).select(["name", "m", "p"])
    _check(tracker, lf, projection_pushed=False)
