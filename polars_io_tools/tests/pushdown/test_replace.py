"""Pushdown matrix for ``pl.Expr.replace_strict`` (native value mapping).

A filter on the column produced by ``replace_strict`` could be rewritten
as ``source_col.is_in([keys mapping to the filtered output])`` on the
source scan: ``replace_strict`` is a deterministic value-by-value mapping
expressed as a literal dict, and (unlike non-strict ``replace``) every
source value is guaranteed to be a mapping key — no pass-through case
complicates the inverse derivation.

Polars 1.40.1 does not perform this rewrite. No exact upstream issue
tracks this case as of the validation date; it is one shape of the
broader "derive a source-side filter from a downstream filter on a
deterministically-computed column" idea, umbrella-tracked in
https://github.com/pola-rs/polars/issues/23345.

The non-strict ``replace`` variant is intentionally NOT a lock-in here:
the naive inverse ``source.is_in(mapped_keys)`` is unsound when a source
value happens to equal one of the mapping's *output* values, because that
value passes through ``replace`` unchanged and would satisfy the filter
without appearing in ``mapped_keys``. A correct rewrite for non-strict
``replace`` requires a more complex predicate.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_io_tools.testing import PredicateTracker

_MAPPING = {"A": "X", "B": "X", "C": "Y", "D": "Y"}


@pytest.fixture
def tracker() -> PredicateTracker:
    return PredicateTracker(pl.DataFrame({"model": ["A", "B", "C", "D"], "val": [1, 2, 3, 4]}))


@pytest.mark.gap
def test_replace_strict_filter_on_derived_col_NOT_pushed(tracker: PredicateTracker):
    """Lock-in: ``with_columns(col.replace_strict({...})).filter(derived == "X")``
    is not rewritten as ``model.is_in(["A", "B"])`` on the source."""
    tracker.lazy_frame.with_columns(pl.col("model").replace_strict(_MAPPING).alias("family")).filter(pl.col("family") == "X").collect()
    assert tracker.last_predicate is None, f"unexpectedly pushed: {tracker.last_predicate!r}"


def test_filter_on_source_col_after_replace_strict_pushed(tracker: PredicateTracker):
    """Baseline: a filter on the *source* column (passed straight through
    ``replace_strict`` without going via the derived column) is pushed."""
    tracker.lazy_frame.with_columns(pl.col("model").replace_strict(_MAPPING).alias("family")).filter(pl.col("model") == "B").collect()
    assert tracker.last_predicate is not None
