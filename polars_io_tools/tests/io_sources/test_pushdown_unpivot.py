"""pushdown_unpivot against bare ``LazyFrame.unpivot``, and what each pushes to the source.

Every row runs bare and wrapped. Both must return what bare unpivot returns, and each mode names whether a predicate or a
projection reached the source. Bare ``lock_in`` rows are marked ``gap``: Polars could push there but does not.
``filter(variable == "A")`` could become the source projection ``[id, A]``, and ``filter(value > X)`` a partial source
filter ``(A > X) | (B > X) | ...`` with the filter kept after the unpivot. The wrapper performs the first.
"""

import polars as pl
import pytest

from polars_io_tools import pushdown_unpivot
from polars_io_tools.testing import PredicateTracker
from polars_io_tools.tests.helpers.catalogue import Case
from polars_io_tools.tests.helpers.oracles import assert_matches_native, assert_pushdown

c = pl.col
gap = pytest.mark.gap
WIDE = pl.DataFrame({"id": [1, 2, 3], "A": [10, 20, 30], "B": [100, 200, 300], "C": [1000, 2000, 3000]})
NOT_PUSHED = {"predicate": False, "projection": False}


def U(id, build, bare=None, wrapped=None, lock_in=False, **kw):
    """A row: ``build`` maps the unpivoted frame to the query; ``bare``/``wrapped`` are ``assert_pushdown`` arguments."""
    return Case(id, build, {"bare": bare or {}, "wrapped": wrapped or {}}, {"lock_in": lock_in, **kw})


ROWS = [
    U("filter_on_id_is_pushed", lambda lf: lf.filter(c("id") > 1), {"predicate": True}, {"predicate": True}),
    U("filter_on_variable_eq", lambda lf: lf.filter(c("variable") == "A"), NOT_PUSHED, {"projection": True}, lock_in=True),
    U("filter_on_variable_isin", lambda lf: lf.filter(c("variable").is_in(["A", "B"])), NOT_PUSHED, {"projection": True}, lock_in=True),
    # Not pushable in general; the wrapper keeps the filter after the unpivot.
    U("filter_on_value_NOT_pushed", lambda lf: lf.filter(c("value") > 50), {"predicate": False}, {"predicate": False}, lock_in=True),
    # Only the count of value columns matters for select("id"); select("variable") needs only their names.
    U("select_id_only", lambda lf: lf.select("id"), {"projection": False}, {"projection": True}, lock_in=True),
    U("select_variable_only", lambda lf: lf.select("variable"), {"projection": False}, {"projection": True}, lock_in=True),
    U("select_value_only", lambda lf: lf.select("value"), {"projection": False}, {"projection": True}, lock_in=True),
    U(
        "filter_id_and_select_value",
        lambda lf: lf.select("id", "value").filter(c("id") > 1),
        {"predicate": True, "projection": False},
        {"predicate": True},
        lock_in=True,
    ),
    # An OR across id and variable must not be split into pushable parts.
    U("or_predicate_spanning_columns", lambda lf: lf.filter((c("id") > 2) | (c("variable") == "A"))),
    U("predicate_references_value_and_variable", lambda lf: lf.filter((c("variable") == "B") & (c("value") > 100))),
    U("empty_variable_set", lambda lf: lf.filter(c("variable").is_in([])), columns=["id", "variable", "value"]),
]


def _params():
    for row in ROWS:
        for mode in ("bare", "wrapped"):
            marks = gap if mode == "bare" and row.kw["lock_in"] else ()
            yield pytest.param(mode, row.pred, row.expect[mode], row.kw.get("columns"), id=f"{mode}-{row.id}", marks=marks)


@pytest.mark.parametrize(("mode", "build", "observed", "columns"), list(_params()))
def test_unpivot(mode, build, observed, columns):
    tracker = PredicateTracker(WIDE)
    lf = tracker.lazy_frame.unpivot(index=["id"]) if mode == "bare" else pushdown_unpivot(tracker.lazy_frame, index=["id"])
    reference = build(tracker.lazy_frame.unpivot(index=["id"])).collect()
    tracker.reset()
    got = assert_matches_native(build(lf), reference.lazy(), None)
    assert_pushdown(tracker, **observed)
    if columns:
        assert got.columns == columns


# (frame, index, query) the wrapper must answer as bare unpivot does.
MATCHES_BARE = [
    # Mixed value dtypes promote to a common supertype even when a variable filter keeps only the Int column.
    Case(
        "mixed_dtype_value_unification_wrapped",
        pl.DataFrame({"id": [1, 2], "i": [10, 20], "f": [1.5, 2.5]}),
        lambda lf: lf.filter(c("variable") == "i"),
        {"index": ["id"]},
    ),
    Case("select_id_only_row_count_invariant_wrapped", WIDE.drop("C"), lambda lf: lf.select("id"), {"index": ["id"]}),
    Case(
        "restricted_unpivot_matches_bare-no_index",
        pl.DataFrame({"A": [10, 20], "B": [30, 40]}),
        lambda lf: lf.filter(c("variable") == "A").select("variable"),
        {"index": []},
    ),
    # The subset (i, f) has supertype Float64, but the full unpivot casts i straight to String.
    Case(
        "restricted_unpivot_matches_bare-string_supertype",
        pl.DataFrame({"id": [1], "i": [10], "f": [1.5], "s": ["abc"]}),
        lambda lf: lf.filter(c("variable").is_in(["i", "f"]) & (c("value") == "10")),
        {"index": "id"},
    ),
]


@pytest.mark.parametrize("case", MATCHES_BARE, ids=lambda case: case.id)
def test_wrapped_matches_bare(case):
    df, query, index = case.pred, case.expect, case.kw["index"]
    assert_matches_native(query(pushdown_unpivot(df.lazy(), index=index)), query(df.lazy().unpivot(index=index)), None)


def test_value_filter_remains_in_wrapped_plan():
    """``filter(value > X)`` cannot replace the synthesized source projection, so it stays after the unpivot."""
    lf = pl.LazyFrame({"id": [1, 2], "A": [10, 200], "B": [20, 300]})
    wrapped = pushdown_unpivot(lf, index=["id"]).filter(c("value") > 50)
    plan = wrapped.explain()
    assert "value" in plan and ("> 50" in plan or "(50)" in plan), plan
    assert_matches_native(wrapped, lf.unpivot(index=["id"]).filter(c("value") > 50), None)
