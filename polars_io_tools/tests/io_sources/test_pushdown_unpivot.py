"""
Result-equivalence and pushdown matrix for ``polars_io_tools.pushdown_unpivot``.

Each scenario is checked against bare ``pl.LazyFrame.unpivot`` to verify the wrapper preserves semantics, and against the
:class:`PredicateTracker` to verify the wrapper actually drives pushdown the bare implementation misses (filter-on-
``variable`` ↔ source-side column projection).
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_io_tools import pushdown_unpivot
from polars_io_tools.testing import PredicateTracker

MODES = ["bare", "wrapped"]


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


def _unpivot(tracker: PredicateTracker, mode: str, **kwargs) -> pl.LazyFrame:
    if mode == "bare":
        return tracker.lazy_frame.unpivot(**kwargs)
    return pushdown_unpivot(tracker.lazy_frame, **kwargs)


def _check(
    tracker: PredicateTracker,
    lf: pl.LazyFrame,
    *,
    expected_df: pl.DataFrame,
    predicate_pushed: bool | None = None,
    projection_pushed: bool | None = None,
) -> None:
    tracker.reset()
    got = lf.collect()
    assert_frame_equal(
        got.sort(got.columns),
        expected_df.sort(expected_df.columns),
        check_dtypes=True,
        check_column_order=False,
    )
    if predicate_pushed is True:
        assert tracker.last_predicate is not None, "predicate was not pushed"
    elif predicate_pushed is False:
        assert tracker.last_predicate is None, f"predicate was unexpectedly pushed: {tracker.last_predicate!r}"
    if projection_pushed is True:
        assert tracker.last_with_columns is not None, "projection was not pushed"
    elif projection_pushed is False:
        assert tracker.last_with_columns is None, f"projection was unexpectedly pushed: {tracker.last_with_columns!r}"


def _expected(tracker: PredicateTracker, build) -> pl.DataFrame:
    return build(tracker.lazy_frame.unpivot(index=["id"])).collect()


@pytest.mark.parametrize("mode", MODES)
def test_filter_on_id_is_pushed(tracker, mode):
    lf = _unpivot(tracker, mode, index=["id"]).filter(pl.col("id") > 1)
    expected = _expected(tracker, lambda lf_: lf_.filter(pl.col("id") > 1))
    _check(tracker, lf, expected_df=expected, predicate_pushed=True)


@pytest.mark.parametrize("mode,proj", [("bare", False), ("wrapped", True)])
def test_filter_on_variable_eq(tracker, mode, proj):
    """``filter(variable == "A")`` becomes upstream projection ``[id, A]`` under the wrapper."""
    lf = _unpivot(tracker, mode, index=["id"]).filter(pl.col("variable") == "A")
    expected = _expected(tracker, lambda lf_: lf_.filter(pl.col("variable") == "A"))
    _check(tracker, lf, expected_df=expected, projection_pushed=proj)


@pytest.mark.parametrize("mode,proj", [("bare", False), ("wrapped", True)])
def test_filter_on_variable_isin(tracker, mode, proj):
    lf = _unpivot(tracker, mode, index=["id"]).filter(pl.col("variable").is_in(["A", "B"]))
    expected = _expected(tracker, lambda lf_: lf_.filter(pl.col("variable").is_in(["A", "B"])))
    _check(tracker, lf, expected_df=expected, projection_pushed=proj)


@pytest.mark.parametrize("mode", MODES)
def test_filter_on_value_NOT_pushed(tracker, mode):
    lf = _unpivot(tracker, mode, index=["id"]).filter(pl.col("value") > 50)
    expected = _expected(tracker, lambda lf_: lf_.filter(pl.col("value") > 50))
    _check(tracker, lf, expected_df=expected, predicate_pushed=False)


@pytest.mark.parametrize("mode,proj", [("bare", False), ("wrapped", True)])
def test_select_id_only(tracker, mode, proj):
    lf = _unpivot(tracker, mode, index=["id"]).select("id")
    expected = _expected(tracker, lambda lf_: lf_.select("id"))
    _check(tracker, lf, expected_df=expected, projection_pushed=proj)


@pytest.mark.parametrize("mode,proj", [("bare", False), ("wrapped", True)])
def test_select_variable_only(tracker, mode, proj):
    lf = _unpivot(tracker, mode, index=["id"]).select("variable")
    expected = _expected(tracker, lambda lf_: lf_.select("variable"))
    _check(tracker, lf, expected_df=expected, projection_pushed=proj)


@pytest.mark.parametrize("mode,proj", [("bare", False), ("wrapped", True)])
def test_select_value_only(tracker, mode, proj):
    lf = _unpivot(tracker, mode, index=["id"]).select("value")
    expected = _expected(tracker, lambda lf_: lf_.select("value"))
    _check(tracker, lf, expected_df=expected, projection_pushed=proj)


@pytest.mark.parametrize("mode", MODES)
def test_filter_id_and_select_value(tracker, mode):
    lf = _unpivot(tracker, mode, index=["id"]).select(["id", "value"]).filter(pl.col("id") > 1)
    expected = _expected(tracker, lambda lf_: lf_.select(["id", "value"]).filter(pl.col("id") > 1))
    _check(tracker, lf, expected_df=expected, predicate_pushed=True)


@pytest.mark.parametrize("mode", MODES)
def test_or_predicate_spanning_columns(tracker, mode):
    """OR across id and variable: must not split into pushable parts and produce wrong results."""
    pred = (pl.col("id") > 2) | (pl.col("variable") == "A")
    lf = _unpivot(tracker, mode, index=["id"]).filter(pred)
    expected = _expected(tracker, lambda lf_: lf_.filter(pred))
    _check(tracker, lf, expected_df=expected)


@pytest.mark.parametrize("mode", MODES)
def test_predicate_references_value_and_variable(tracker, mode):
    pred = (pl.col("variable") == "B") & (pl.col("value") > 100)
    lf = _unpivot(tracker, mode, index=["id"]).filter(pred)
    expected = _expected(tracker, lambda lf_: lf_.filter(pred))
    _check(tracker, lf, expected_df=expected)


@pytest.mark.parametrize("mode", MODES)
def test_empty_variable_set(tracker, mode):
    """``variable.is_in([])`` produces a zero-row frame with the declared schema."""
    lf = _unpivot(tracker, mode, index=["id"]).filter(pl.col("variable").is_in([]))
    got = lf.collect()
    assert got.height == 0
    assert got.columns == ["id", "variable", "value"]


def test_mixed_dtype_value_unification_wrapped():
    """When ``value`` columns have mixed dtypes, bare unpivot promotes to a common supertype (Float64). The wrapper must emit
    the same dtype even when restricted to a single Int column upstream by a variable filter."""
    df = pl.DataFrame({"id": [1, 2], "i": [10, 20], "f": [1.5, 2.5]})
    t = PredicateTracker(df)
    bare = t.lazy_frame.unpivot(index=["id"]).filter(pl.col("variable") == "i").collect()
    wrapped = pushdown_unpivot(t.lazy_frame, index=["id"]).filter(pl.col("variable") == "i").collect()
    assert wrapped.schema["value"] == bare.schema["value"]
    assert_frame_equal(wrapped.sort(wrapped.columns), bare.sort(bare.columns), check_dtypes=True)


def test_value_filter_remains_in_wrapped_plan():
    """``filter(value > X)`` is *not* a replacement for the synthesized upstream column projection — the residual filter must
    still apply after unpivot. Inspect the plan to lock this in."""
    df = pl.DataFrame({"id": [1, 2], "A": [10, 200], "B": [20, 300]})
    t = PredicateTracker(df)
    lf = pushdown_unpivot(t.lazy_frame, index=["id"]).filter(pl.col("value") > 50)
    plan = lf.explain()
    assert "value" in plan and ("> 50" in plan or "(50)" in plan), f"expected residual filter on value in plan:\n{plan}"
    bare = t.lazy_frame.unpivot(index=["id"]).filter(pl.col("value") > 50).collect()
    assert_frame_equal(lf.collect().sort(lf.collect().columns), bare.sort(bare.columns), check_dtypes=True)


def test_select_id_only_row_count_invariant_wrapped():
    """``select("id")`` after unpivot must still produce ``len(rows) * len(on)`` rows."""
    df = pl.DataFrame({"id": [1, 2, 3], "A": [10, 20, 30], "B": [100, 200, 300]})
    t = PredicateTracker(df)
    out = pushdown_unpivot(t.lazy_frame, index=["id"]).select("id").collect()
    assert out.height == 3 * 2
