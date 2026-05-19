"""Lookup-join filter pushdown ("dynamic filter pushdown").

When a LazyFrame is joined to a small lookup table and then filtered on a
column from the lookup, polars could:

1. Materialize the lookup at plan time (or push it as a semi-join filter),
2. Compute the set of source-side keys that satisfy the downstream filter,
3. Rewrite the downstream filter as ``source_key.is_in([eligible keys])``
   on the source scan.

This is the "semi-join filter pushdown" / "dynamic filter pushdown"
optimization. Polars 1.40.1 does not implement it. The manual ``is_in``
form *does* push (see baseline test below), confirming the rewrite target
is achievable.

This is the natural alternative to user-side workarounds like cubist's
``multi_source(value_mapping=...)`` for many-to-one mappings expressed as
a small dataframe.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_io_tools.testing import PredicateTracker


@pytest.fixture
def tracker() -> PredicateTracker:
    return PredicateTracker(pl.DataFrame({"model": ["A", "B", "C", "D"], "val": [1, 2, 3, 4]}))


@pytest.mark.gap
def test_lookup_join_filter_on_derived_col_NOT_pushed_as_isin(
    tracker: PredicateTracker,
):
    """Lock-in: ``join(small_lookup).filter(col_from_lookup == X)`` is not
    rewritten as ``model.is_in([keys mapping to X])`` on the source.

    Setup mirrors a typical model→family mapping: lookup has 4 models split
    across 2 families; downstream selects family X.

    No exact upstream issue tracks this case as of the validation date.
    Closest related: https://github.com/pola-rs/polars/issues/21710
    ("transform multi-attribute equality joins into filter + single-attribute
    join") — a different shape of the same "derive a small filter from a
    join's right side" idea. Umbrella:
    https://github.com/pola-rs/polars/issues/23345.
    """
    lookup = pl.DataFrame({"model": ["A", "B", "C", "D"], "family": ["X", "X", "Y", "Y"]}).lazy()
    tracker.lazy_frame.join(lookup, on="model").filter(pl.col("family") == "X").collect()
    assert tracker.last_predicate is None, f"unexpectedly pushed: {tracker.last_predicate!r}"


def test_manual_isin_filter_pushed(tracker: PredicateTracker):
    """Baseline: the rewrite target — a manual ``model.is_in([...])`` filter —
    *does* push. This shows the optimization is achievable; only the
    automatic derivation from a lookup join is missing."""
    tracker.lazy_frame.filter(pl.col("model").is_in(["A", "B"])).collect()
    assert tracker.last_predicate is not None
