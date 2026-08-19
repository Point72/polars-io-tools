"""
Result-equivalence and pushdown matrix for ``polars_io_tools.pushdown_pivot``.

These tests complement the bare characterization suite in ``tests/pushdown/test_pivot.py``. Each scenario is checked against bare
``pl.LazyFrame.pivot`` to verify the wrapper preserves semantics, and against the :class:`PredicateTracker` to verify the
wrapper actually drives pushdown.

Two ``dense_on`` modes are exercised:

* ``dense_on=False`` (default, *smart*): a second narrow scan recovers index values that the synthesized ``on ∈ S`` filter
  would have dropped. Always matches bare; works on sparse data.
* ``dense_on=True`` (*fast*): the recovery scan is skipped. Faster, but silently drops rows on sparse data (the precondition
  the caller asserted has been violated).
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_io_tools import pushdown_pivot
from polars_io_tools.testing import PredicateTracker

DENSE_DF = pl.DataFrame(
    {
        "name": ["A", "A", "A", "B", "B", "B"],
        "subject": ["m", "p", "x", "m", "p", "x"],
        "score": [1, 2, 3, 4, 5, 6],
    }
)

SPARSE_DF = pl.DataFrame(
    {
        # A is missing subject "m"; B is missing subject "x"; C only has "p".
        "name": ["A", "A", "B", "B", "C"],
        "subject": ["p", "x", "m", "p", "p"],
        "score": [1, 2, 3, 4, 5],
    }
)

KW = {"on": "subject", "on_columns": ["m", "p", "x"], "index": "name", "values": "score"}


def _bare(df: pl.DataFrame, *, transform):
    return transform(df.lazy().pivot(**KW)).sort("name").collect()


def _wrapped(df: pl.DataFrame, *, transform, dense_on: bool):
    tracker = PredicateTracker(df)
    out = transform(pushdown_pivot(tracker.lazy_frame, dense_on=dense_on, **KW)).sort("name").collect()
    return out, tracker


@pytest.mark.parametrize("dense_on", [False, True])
def test_full_select_matches_bare_dense(dense_on):
    """No projection pruning → no synthesized on-filter; both modes match bare."""
    bare = _bare(DENSE_DF, transform=lambda lf: lf)
    out, _ = _wrapped(DENSE_DF, transform=lambda lf: lf, dense_on=dense_on)
    assert_frame_equal(out, bare, check_dtypes=True, check_column_order=False)


@pytest.mark.parametrize("dense_on", [False, True])
def test_select_index_only_matches_bare(dense_on):
    bare = _bare(DENSE_DF, transform=lambda lf: lf.select("name"))
    out, tracker = _wrapped(DENSE_DF, transform=lambda lf: lf.select("name"), dense_on=dense_on)
    assert_frame_equal(out, bare, check_dtypes=True, check_column_order=False)
    assert tracker.last_with_columns is not None, "selecting only the index should push a projection"


@pytest.mark.parametrize("dense_on", [False, True])
def test_filter_on_index_matches_bare_and_pushes(dense_on):
    bare = _bare(DENSE_DF, transform=lambda lf: lf.filter(pl.col("name") == "A"))
    out, tracker = _wrapped(DENSE_DF, transform=lambda lf: lf.filter(pl.col("name") == "A"), dense_on=dense_on)
    assert_frame_equal(out, bare, check_dtypes=True, check_column_order=False)
    assert tracker.last_predicate is not None, "filter on index should be pushed in either mode"


def test_select_subset_dense_on_True_pushes_synthesized_filter():
    """`select([name, m])` with `dense_on=True` synthesizes upstream `subject == "m"`.

    (No projection is pushed because the source's only columns are exactly those internally needed —
    ``[name, subject, score]`` — so polars treats the projection as a no-op.)
    """
    bare = _bare(DENSE_DF, transform=lambda lf: lf.select(["name", "m"]))
    out, tracker = _wrapped(DENSE_DF, transform=lambda lf: lf.select(["name", "m"]), dense_on=True)
    assert_frame_equal(out, bare, check_dtypes=True, check_column_order=False)
    assert tracker.last_predicate is not None, "expected synthesized on-filter to be pushed"


def test_select_subset_dense_on_False_matches_bare_via_recovery():
    """Default smart mode: prefiltered pivot + index recovery, matches bare."""
    bare = _bare(DENSE_DF, transform=lambda lf: lf.select(["name", "m"]))
    out, _ = _wrapped(DENSE_DF, transform=lambda lf: lf.select(["name", "m"]), dense_on=False)
    assert_frame_equal(out, bare, check_dtypes=True, check_column_order=False)


def test_select_subset_dense_on_True_pushes_projection_on_wide_source():
    """With extra source columns, the synthesized projection narrows the scan to just `[index, on, values]` (and the
    synthesized filter is still pushed)."""
    df = DENSE_DF.with_columns(
        pl.lit(0.0).alias("extra1"),
        pl.lit(0.0).alias("extra2"),
    )
    tracker = PredicateTracker(df)
    out = (
        pushdown_pivot(tracker.lazy_frame, on="subject", on_columns=["m", "p", "x"], index="name", values="score", dense_on=True)
        .select(["name", "m"])
        .collect()
    )
    bare = df.lazy().pivot("subject", on_columns=["m", "p", "x"], index="name", values="score").select(["name", "m"]).sort("name").collect()
    assert_frame_equal(out.sort("name"), bare, check_dtypes=True, check_column_order=False)
    assert tracker.last_predicate is not None
    assert set(tracker.last_with_columns) == {"name", "subject", "score"}

    bare = _bare(DENSE_DF, transform=lambda lf: lf.select(["name", "m", "p"]))
    out, tracker = _wrapped(DENSE_DF, transform=lambda lf: lf.select(["name", "m", "p"]), dense_on=True)
    assert_frame_equal(out, bare, check_dtypes=True, check_column_order=False)
    assert tracker.last_predicate is not None


def test_sparse_data_dense_on_False_matches_bare():
    """Headline correctness regression: smart mode preserves dropped index rows."""
    bare = _bare(SPARSE_DF, transform=lambda lf: lf.select(["name", "m"]))
    out, _ = _wrapped(SPARSE_DF, transform=lambda lf: lf.select(["name", "m"]), dense_on=False)
    assert_frame_equal(out, bare, check_dtypes=True, check_column_order=False)
    # Sanity: A and C both appear with null `m` (they had no `subject == "m"` row).
    rows = {r["name"]: r["m"] for r in out.to_dicts()}
    assert rows == {"A": None, "B": 3, "C": None}


def test_sparse_data_dense_on_True_drops_rows_as_documented():
    """Precondition-violation case: opting into the fast path on sparse data is documented to drop rows. Test pins that
    behavior so a future silent change is caught."""
    out, _ = _wrapped(SPARSE_DF, transform=lambda lf: lf.select(["name", "m"]), dense_on=True)
    rows = {r["name"]: r["m"] for r in out.to_dicts()}
    assert rows == {"B": 3}, "dense_on=True on sparse data should drop A and C"


def test_maintain_order_suppresses_on_filter():
    """With ``maintain_order=True`` the wrapper does not synthesize the on-filter (it could change first-appearance
    ordering); the result still matches bare in either dense_on mode."""
    df = pl.DataFrame(
        {
            "name": ["B", "A", "A", "B"],
            "subject": ["m", "p", "m", "p"],
            "score": [1, 2, 3, 4],
        }
    )
    bare = df.lazy().pivot("subject", on_columns=["m", "p"], index="name", values="score", maintain_order=True).select(["name", "m"]).collect()
    for dense_on in (False, True):
        tracker = PredicateTracker(df)
        out = (
            pushdown_pivot(
                tracker.lazy_frame,
                on="subject",
                on_columns=["m", "p"],
                index="name",
                values="score",
                maintain_order=True,
                dense_on=dense_on,
            )
            .select(["name", "m"])
            .collect()
        )
        assert tracker.last_predicate is None, f"on-filter must not be pushed when maintain_order=True (dense_on={dense_on})"
        assert_frame_equal(out, bare, check_dtypes=True)


def test_predicate_on_dropped_pivoted_column_dense_on_True():
    """Filter on `p` plus `select("name")` materializes `p` internally, pushes the synthesized on-filter, and matches bare."""
    bare = DENSE_DF.lazy().pivot(**KW).filter(pl.col("p") > 3).select("name").collect()
    tracker = PredicateTracker(DENSE_DF)
    wrapped = pushdown_pivot(tracker.lazy_frame, dense_on=True, **KW).filter(pl.col("p") > 3).select("name").collect()
    assert tracker.last_predicate is not None
    assert_frame_equal(wrapped, bare, check_dtypes=True, check_row_order=False)


def test_aggregate_function_not_implemented():
    with pytest.raises(NotImplementedError):
        pushdown_pivot(
            DENSE_DF.lazy(),
            on="subject",
            on_columns=["m"],
            index="name",
            values="score",
            aggregate_function="sum",
        )
