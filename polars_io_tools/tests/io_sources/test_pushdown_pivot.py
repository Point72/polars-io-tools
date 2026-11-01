"""pushdown_pivot against bare ``LazyFrame.pivot``, and what each pushes to the source.

Lazy pivot is a group_by with one ``filter().item()`` aggregation per ``on_columns`` value. Bare, a filter on the index
reaches the source and selecting only the index prunes the source to it; nothing else is pushed, which is correct:

* a filter on a pivoted column is a HAVING on an aggregate, so pushing ``score > 2`` would drop rows a group needs;
* ``is_not_null`` on a pivoted column rewritten as ``subject == "m"`` would null the other pivoted columns;
* selecting ``[name, m]`` rewritten as the source filter ``subject == "m"`` drops index values with no "m" row when the
  source is sparse over ``on``, which bare pivot keeps with ``m = null``.

The wrapper pushes that filter when the caller asserts density (``dense_on=True``, mode ``dense``); by default (mode
``smart``) a second narrow scan recovers the index values the filter dropped. Rows run in the modes they name, and
every wrapped mode except ``dense`` on sparse data must return what bare pivot returns.
"""

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from polars_io_tools import pushdown_pivot
from polars_io_tools.testing import PredicateTracker
from polars_io_tools.tests.helpers.assertions import assert_matches_native, assert_pushdown
from polars_io_tools.tests.helpers.cases import Case

c = pl.col
DENSE = pl.DataFrame({"name": ["A", "A", "A", "B", "B", "B"], "subject": ["m", "p", "x", "m", "p", "x"], "score": [1, 2, 3, 4, 5, 6]})
# A has no "m", B no "x", and C only "p".
SPARSE = pl.DataFrame({"name": ["A", "A", "B", "B", "C"], "subject": ["p", "x", "m", "p", "p"], "score": [1, 2, 3, 4, 5]})
WIDE = DENSE.with_columns(pl.lit(0.0).alias("extra1"), pl.lit(0.0).alias("extra2"))
ORDERED = pl.DataFrame({"name": ["B", "A", "A", "B"], "subject": ["m", "p", "m", "p"], "score": [1, 2, 3, 4]})
KW = {"on": "subject", "on_columns": ["m", "p", "x"], "index": "name", "values": "score"}
DENSE_ON = {"smart": False, "dense": True}


def V(id, query, modes, df=DENSE, **kw):
    """A row: ``query`` maps the pivoted frame to the query; ``modes`` maps each mode to ``assert_pushdown`` arguments."""
    return Case(id, query=query, modes=modes, kw={"df": df, "pivot": KW} | kw)


def _rows_named(expected: dict):
    def check(got):
        assert dict(zip(got["name"], got["m"])) == expected

    return check


ROWS = [
    V("filter_on_index_is_pushed", lambda lf: lf.filter(c("name") == "A"), {m: {"predicate": True} for m in ("bare", "smart", "dense")}),
    V("filter_on_pivoted_output_NOT_pushed", lambda lf: lf.filter(c("m") > 2), {"bare": {"predicate": False}}),
    V("filter_on_pivoted_output_is_not_null_NOT_pushed", lambda lf: lf.filter(c("m").is_not_null()), {"bare": {"predicate": False}}),
    V(
        "select_index_only",
        lambda lf: lf.select("name"),
        {"bare": {"projection": {"name"}}, "smart": {"projection": True}, "dense": {"projection": True}},
    ),
    V("full_select_matches_bare_dense", lambda lf: lf, {"smart": {}, "dense": {}}),
    # The source holds exactly the columns the pivot needs, so no projection is pushed.
    V("select_subset", lambda lf: lf.select("name", "m"), {"bare": {"projection": False}, "smart": {}, "dense": {"predicate": True}}),
    V("select_two_pivoted_outputs", lambda lf: lf.select("name", "m", "p"), {"bare": {"projection": False}, "dense": {"predicate": True}}),
    V(
        "select_subset_dense_on_True_pushes_projection_on_wide_source",
        lambda lf: lf.select("name", "m"),
        {"dense": {"predicate": True, "projection": {"name", "subject", "score"}}},
        df=WIDE,
    ),
    V(
        "sparse_data_dense_on_False_matches_bare",
        lambda lf: lf.select("name", "m"),
        {"smart": {}},
        df=SPARSE,
        check=_rows_named({"A": None, "B": 3, "C": None}),
    ),
    # Opting into the fast path on sparse data drops A and C, as documented.
    V(
        "sparse_data_dense_on_True_drops_rows_as_documented",
        lambda lf: lf.select("name", "m"),
        {"dense": {}},
        df=SPARSE,
        native=False,
        check=_rows_named({"B": 3}),
    ),
    # The on-filter could change the order of first appearance, so it is not pushed.
    V(
        "maintain_order_suppresses_on_filter",
        lambda lf: lf.select("name", "m"),
        {"smart": {"predicate": False}, "dense": {"predicate": False}},
        df=ORDERED,
        pivot={"on": "subject", "on_columns": ["m", "p"], "index": "name", "values": "score", "maintain_order": True},
    ),
    # A filter on p with select("name") still materializes p and pushes the on-filter.
    V("predicate_on_dropped_pivoted_column_dense_on_True", lambda lf: lf.filter(c("p") > 3).select("name"), {"dense": {"predicate": True}}),
]


def _params():
    for row in ROWS:
        for mode, observed in row.modes.items():
            yield pytest.param(mode, row.query, observed, row.kw, id=f"{mode}-{row.id}")


@pytest.mark.parametrize(("mode", "query", "observed", "kw"), list(_params()))
def test_pivot(mode, query, observed, kw):
    tracker = PredicateTracker(kw["df"])
    pivot = kw["pivot"]
    lf = tracker.lazy_frame.pivot(**pivot) if mode == "bare" else pushdown_pivot(tracker.lazy_frame, dense_on=DENSE_ON[mode], **pivot)
    if mode != "bare" and kw.get("native", True):
        reference = query(tracker.lazy_frame.pivot(**pivot)).collect()
        tracker.reset()
        got = assert_matches_native(query(lf), reference.lazy(), None, check_row_order=pivot.get("maintain_order", False))
    else:
        got = query(lf).collect()
    assert_pushdown(tracker, **observed)
    if "check" in kw:
        kw["check"](got)


def test_aggregate_function_not_implemented():
    with pytest.raises(NotImplementedError):
        pushdown_pivot(DENSE.lazy(), on="subject", on_columns=["m"], index="name", values="score", aggregate_function="sum")


def _first_pivoted(name, df, on_columns):
    kw = {"on": "kind", "on_columns": on_columns, "index": "id", "values": "v"}
    col = df.lazy().pivot(**kw).collect_schema().names()[1]
    return Case(f"filter_on_first_pivoted_column_matches_bare-{name}", df=df, query=lambda lf: lf.filter(c(col) < 15).select("id", col), kw=kw)


def _decimal_on(name, extra):
    df = pl.DataFrame({"id": [1, 1, 1], "kind": pl.Series([1, 2, 3], dtype=pl.Decimal(3, 0)), "v": [10, 20, 30]})
    cols = ["id", "1", "2", *(["1000"] if extra else [])]
    kw = {"on": "kind", "on_columns": [Decimal(1), Decimal(2), Decimal(3), *extra], "index": "id", "values": "v"}
    return Case(f"decimal_on_values_filter_matches_bare-{name}", df=df, query=lambda lf: lf.filter(c("1") < c("2")).select(cols), kw=kw)


def _series_on(name, days):
    df = pl.DataFrame({"id": [1, 1, 1], "kind": days, "v": [0, 1, 2]})
    kw = {"on": "kind", "on_columns": days, "index": "id", "values": "v"}
    cols = df.lazy().pivot(**kw).collect_schema().names()[:3]
    return Case(f"series_on_columns_are_named_in_their_unit-{name}", df=df, query=lambda lf: lf.select(cols), kw=kw)


_FOLD_EARLY = datetime(2024, 11, 3, 1, 30, tzinfo=ZoneInfo("America/New_York"))
_DAYS = pl.Series([datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)])
_NULL_KIND = pl.DataFrame({"id": [1, 1], "kind": [None, "B"], "v": [10, 20]})
# (frame, query, pivot arguments) the default wrapper must answer as bare pivot does.
MATCHES_BARE = [
    # A null index value survives the recovery join.
    _first_pivoted("null_index", pl.DataFrame({"id": [None, None, 1, 1], "kind": ["A", "B", "A", "B"], "v": [10, 20, 30, 40]}), ["A", "B"]),
    # A null ``on`` category survives the synthesized ``on`` filter.
    _first_pivoted("null_on", _NULL_KIND, [None, "B"]),
    _first_pivoted("null_on_extra_column", _NULL_KIND, [None, "B", "C"]),
    # DST-fold datetimes compare equal in Python but are distinct categories.
    _first_pivoted(
        "dst_fold",
        pl.DataFrame(
            {"id": [1, 1], "kind": pl.Series([_FOLD_EARLY, _FOLD_EARLY.replace(fold=1)]).cast(pl.Datetime("us", "America/New_York")), "v": [1, 2]}
        ),
        [_FOLD_EARLY, _FOLD_EARLY.replace(fold=1)],
    ),
    _decimal_on("in_range", []),
    _decimal_on("out_of_range", [Decimal(1000)]),
    _series_on("ms", _DAYS.cast(pl.Datetime("ms"))),
    _series_on("ns", _DAYS.cast(pl.Datetime("ns"))),
    _series_on("ns_sub_us", pl.Series([1000, 1001, 1002]).cast(pl.Datetime("ns"))),
]


@pytest.mark.parametrize("case", MATCHES_BARE, ids=lambda case: case.id)
def test_wrapped_matches_bare(case):
    df, query, kw = case.df, case.query, case.kw
    assert_matches_native(query(pushdown_pivot(df.lazy(), **kw)), query(df.lazy().pivot(**kw)), None)
