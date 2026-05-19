"""
Pushdown matrix for ``pl.LazyFrame.unpivot``.

Each test characterizes how polars' optimizer routes a downstream filter or
projection through ``unpivot`` to the underlying source. Tests where polars
*could* push but doesn't are framed as lock-in assertions; they will start
failing — and need to be flipped — when polars improves or when a
polars-io-tools wrapper takes over.

The lock-ins below assert ``predicate_pushed=False`` / ``projection_pushed=False``
without prescribing the exact rewrite shape. Sound rewrites do exist
(``filter(variable == "X")`` → source projection ``[*index, X]``;
``filter(value > X)`` → partial source prefilter ``(A>X) | (B>X) | ...``
plus a retained residual filter), but polars does neither today.

These tests assert pushdown behavior only. End-to-end result correctness of
``unpivot`` itself is exercised by polars' own test suite; correctness of
the polars-io-tools wrappers (Phase 2) is verified in the ``mode="wrapped"``
parametrize axis added on top of this matrix in a follow-up PR.
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
            "id": [1, 2, 3],
            "A": [10, 20, 30],
            "B": [100, 200, 300],
            "C": [1000, 2000, 3000],
        }
    )
    return PredicateTracker(df)


def test_filter_on_id_is_pushed(tracker: PredicateTracker):
    """Filter on an id_var passes through unpivot unchanged."""
    lf = tracker.lazy_frame.unpivot(index=["id"]).filter(pl.col("id") > 1)
    _check(tracker, lf, predicate_pushed=True)


@pytest.mark.gap
def test_filter_on_variable_eq_NOT_pushed(tracker: PredicateTracker):
    """Lock-in: ``filter(variable == "A")`` could become source-side selection
    of ``[id, A]``, but polars does not perform this rewrite."""
    lf = tracker.lazy_frame.unpivot(index=["id"]).filter(pl.col("variable") == "A")
    _check(tracker, lf, predicate_pushed=False)


@pytest.mark.gap
def test_filter_on_variable_isin_NOT_pushed(tracker: PredicateTracker):
    """Lock-in: same as above for an is_in filter on `variable`."""
    lf = tracker.lazy_frame.unpivot(index=["id"]).filter(pl.col("variable").is_in(["A", "B"]))
    _check(tracker, lf, predicate_pushed=False)


@pytest.mark.gap
def test_filter_on_value_NOT_pushed(tracker: PredicateTracker):
    """Filter on `value` is not generally pushable; polars correctly refuses."""
    lf = tracker.lazy_frame.unpivot(index=["id"]).filter(pl.col("value") > 50)
    _check(tracker, lf, predicate_pushed=False)


@pytest.mark.gap
def test_select_id_only_NOT_pushed(tracker: PredicateTracker):
    """Lock-in: `select("id")` could prune value columns at the source
    (only their count matters for the row multiplication)."""
    lf = tracker.lazy_frame.unpivot(index=["id"]).select("id")
    _check(tracker, lf, projection_pushed=False)


@pytest.mark.gap
def test_select_variable_only_NOT_pushed(tracker: PredicateTracker):
    """Lock-in: `select("variable")` only needs source value-column names."""
    lf = tracker.lazy_frame.unpivot(index=["id"]).select("variable")
    _check(tracker, lf, projection_pushed=False)


@pytest.mark.gap
def test_select_value_only_NOT_pushed(tracker: PredicateTracker):
    """Lock-in: `select("value")` could push the full set of value columns."""
    lf = tracker.lazy_frame.unpivot(index=["id"]).select("value")
    _check(tracker, lf, projection_pushed=False)


@pytest.mark.gap
def test_filter_id_and_select_value_partial_pushdown(tracker: PredicateTracker):
    """Filter on id is pushed; projection is not. The two axes are independent
    under the current polars optimizer."""
    lf = tracker.lazy_frame.unpivot(index=["id"]).select(["id", "value"]).filter(pl.col("id") > 1)
    _check(tracker, lf, predicate_pushed=True, projection_pushed=False)
