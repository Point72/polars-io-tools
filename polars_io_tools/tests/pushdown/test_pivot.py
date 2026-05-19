"""
Pushdown matrix for ``pl.LazyFrame.pivot`` (currently unstable in polars).

Lazy ``pivot`` is implemented internally as a ``group_by`` plus a set of
``filter().item()`` aggregations — one per value listed in ``on_columns``.
Pushdown therefore behaves like a group-by aggregation:

* Filters on the ``index`` column flow through to the source.
* If the downstream selects only the ``index`` column, polars drops all the
  pivot aggregations and the source is asked for just the index column.
* For all other cases (selecting a subset of pivoted output columns, filtering
  by an output column), polars currently performs no rewrite — the symmetry
  ``column-projection ↔ row-filter on the `on` column`` is missed.

These tests assert pushdown behavior only. End-to-end result correctness of
``pivot`` itself is exercised by polars' own test suite; correctness of the
polars-io-tools wrappers (Phase 2) is verified in the ``mode="wrapped"``
parametrize axis added on top of this matrix in a follow-up PR.

NOTE (PR #259 review): Several lock-ins below are misframed or depend on a
silent sparse-``on`` precondition; the rewrites they describe are not
unconditionally sound. They will be reworked when PR #259
(``pit/pivot-unpivot-pushdown-wrappers``) resolves. Until then they are
preserved verbatim from their original location to keep history continuous.
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


@pytest.mark.gap
def test_filter_on_pivoted_output_NOT_pushed(tracker: PredicateTracker):
    """Filter on a pivoted output column (`m`) is post-aggregation; not pushed."""
    lf = _pivoted(tracker).filter(pl.col("m") > 2)
    _check(tracker, lf, predicate_pushed=False)


@pytest.mark.gap
def test_filter_on_pivoted_output_is_not_null_NOT_pushed(tracker: PredicateTracker):
    """Lock-in: `is_not_null` on a pivoted column could become a row filter
    `subject == "m"` upstream; polars does not perform this rewrite."""
    lf = _pivoted(tracker).filter(pl.col("m").is_not_null())
    _check(tracker, lf, predicate_pushed=False)


def test_select_index_only_pushes_projection(tracker: PredicateTracker):
    """Selecting only the index makes polars drop the pivot aggregations and
    ask the source for just the index column. (Already optimized.)"""
    lf = _pivoted(tracker).select("name")
    _check(tracker, lf, projection_pushed=True, pushed_columns={"name"})


@pytest.mark.gap
def test_select_subset_of_pivoted_output_NOT_pushed_as_filter(tracker: PredicateTracker):
    """Lock-in: selecting `[name, m]` could be rewritten as the source row
    filter `subject == "m"` plus a `[name, subject, score]` projection.
    Polars does not — it asks the source for all rows and all columns."""
    lf = _pivoted(tracker).select(["name", "m"])
    _check(tracker, lf, projection_pushed=False)


@pytest.mark.gap
def test_select_two_pivoted_outputs_NOT_pushed_as_filter(tracker: PredicateTracker):
    """Lock-in: same as above with `subject.is_in(["m","p"])` as the missed
    upstream filter."""
    lf = _pivoted(tracker).select(["name", "m", "p"])
    _check(tracker, lf, projection_pushed=False)
