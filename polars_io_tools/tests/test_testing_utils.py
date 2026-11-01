"""polars_io_tools.testing: PredicateTracker, PredicateAnalyzer and io_source_assert.

Tables: a filtered tracker -> what the analyzer finds on each column; a tracker assertion call -> passes or raises;
an io_source_assert call -> the predicate its assertion saw and the result.
"""

from datetime import date, timedelta

import polars as pl
import pytest

from polars_io_tools.io_sources.base import BinaryExprNode
from polars_io_tools.io_sources.pushdown_combine import FilterSpec, pushdown_combine
from polars_io_tools.testing import PredicateAnalyzer, PredicateTracker, io_source_assert
from polars_io_tools.tests.helpers.assertions import assert_matches_native
from polars_io_tools.tests.helpers.cases import Case, params


def _d(day: int) -> date:
    return date(2024, 1, day)


def _days(n: int, **more) -> pl.DataFrame:
    return pl.DataFrame({"date": [_d(i) for i in range(1, n + 1)], **more})


def _lookback(lf: pl.LazyFrame) -> pl.LazyFrame:
    return pushdown_combine(sources={"data": (lf, {"date": FilterSpec(lookback=timedelta(days=3))})}, combine=lambda s: s["data"])


def _combined(col: str):
    return lambda lf: pushdown_combine(sources={"data": (lf, {col: FilterSpec()})}, combine=lambda s: s["data"])


def A(id, df, pred, temporal=None, discrete=None, binary_nodes=1, wrap=None):
    """``temporal``/``discrete``: column -> the bounds / value sets of every filter the analyzer finds on it.
    ``binary_nodes``: the least number of binary nodes in the pushed predicate."""
    return Case(id, pred=pred, df=df, wrap=wrap, temporal=temporal or {}, discrete=discrete or {}, binary_nodes=binary_nodes)


date_, cat = pl.col("date"), pl.col("category")
ANALYZER = [
    A("tracks_predicate", _days(5, val=[1, 2, 3, 4, 5]), date_ >= _d(3), {"date": [(_d(3), None)]}),
    A("find_temporal_filter_between", _days(10), date_.is_between(_d(3), _d(7)), {"date": [(_d(3), _d(7))]}, binary_nodes=0),
    A("find_temporal_filters_returns_list", _days(10), (date_ >= _d(3)) & (date_ <= _d(7)), {"date": [(_d(3), None), (None, _d(7))]}),
    # the lookback pushes the expanded filter (Jan 2) and the original (Jan 5)
    A(
        "find_temporal_filters_with_lookback",
        _days(10, val=list(range(10))),
        date_ >= _d(5),
        {"date": [(_d(2), None), (_d(5), None)]},
        wrap=_lookback,
    ),
    A("find_discrete_filter_eq", pl.DataFrame({"category": ["A", "B", "C", "A", "B"]}), cat == "A", discrete={"category": [{"A"}]}),
    A(
        "find_discrete_filter_is_in",
        pl.DataFrame({"category": ["A", "B", "C", "D", "E"]}),
        cat.is_in(["A", "C", "E"]),
        discrete={"category": [{"A", "C", "E"}]},
        binary_nodes=0,
    ),
    A(
        "find_discrete_filters_returns_list",
        pl.DataFrame({"category": ["A", "B", "C"]}),
        (cat == "A") | (cat == "B"),
        discrete={"category": [{"A"}, {"B"}]},
    ),
    A(
        "tracker_with_discrete_filter",
        pl.DataFrame({"category": ["A", "B", "C", "A", "B"], "val": [1, 2, 3, 4, 5]}),
        cat == "A",
        discrete={"category": [{"A"}]},
        wrap=_combined("category"),
    ),
    # no filter on b, nor on a column that does not exist
    A(
        "has_filter_on_column",
        pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
        pl.col("a") > 1,
        {"b": [], "nonexistent": []},
        {"b": [], "nonexistent": []},
    ),
    A("find_all_nodes_by_predicate", pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), (pl.col("a") > 1) & (pl.col("b") < 6), binary_nodes=3),
]


def _sorted_bounds(bounds):
    return sorted(bounds, key=lambda b: tuple((x is None, x or date.min) for x in b))


@pytest.mark.parametrize(
    ("pred", "df", "wrap", "temporal", "discrete", "binary_nodes"), params(ANALYZER, "pred", "df", "wrap", "temporal", "discrete", "binary_nodes")
)
def test_analyzer(pred, df, wrap, temporal, discrete, binary_nodes):
    tracker = PredicateTracker(df)
    lf = tracker.lazy_frame if wrap is None else wrap(tracker.lazy_frame)
    assert_matches_native(lf, df.lazy(), pred, check_row_order=True)
    assert tracker.call_count == 1
    analyzer = tracker.get_analyzer()
    assert isinstance(analyzer, PredicateAnalyzer)
    for col in {*temporal, *discrete}:
        filters = analyzer.find_temporal_filters(col)
        assert _sorted_bounds(analyzer.extract_temporal_bounds(f) for f in filters) == _sorted_bounds(temporal.get(col, []))
        assert analyzer.find_temporal_filter(col) is (filters[0] if filters else None)
        values = analyzer.find_discrete_filters(col)
        assert sorted(map(sorted, (analyzer.extract_discrete_values(f) for f in values))) == sorted(map(sorted, discrete.get(col, [])))
        assert analyzer.find_discrete_filter(col) is (values[0] if values else None)
        assert analyzer.count_filters_on_column(col) == len(filters) + len(values)
        assert analyzer.has_filter_on_column(col) == bool(filters or values)
    nodes = analyzer.find_all_nodes_by_predicate(lambda n: isinstance(n, BinaryExprNode))
    assert len(nodes) >= binary_nodes
    first = analyzer.find_node_by_predicate(lambda n: isinstance(n, BinaryExprNode))
    assert (first is None) == (not nodes) and (first is None or isinstance(first, BinaryExprNode))


def test_tracks_with_columns():
    tracker = PredicateTracker(pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}))
    assert tracker.lazy_frame.select(["a", "b"]).collect().columns == ["a", "b"]
    assert tracker.last_with_columns == ["a", "b"]


def test_multiple_collects_then_reset():
    """The tracker counts every collect and keeps the last predicate until reset."""
    tracker = PredicateTracker(pl.DataFrame({"val": [1, 2, 3, 4, 5]}))
    with pytest.raises(ValueError, match="No predicate has been pushed"):
        tracker.get_analyzer()
    tracker.lazy_frame.filter(pl.col("val") > 2).collect()
    tracker.lazy_frame.filter(pl.col("val") < 4).collect()
    assert tracker.call_count == 2
    assert tracker.last_predicate is not None
    tracker.reset()
    assert (tracker.last_predicate, tracker.last_with_columns, tracker.call_count) == (None, None, 0)


def test_direct_and_source_filter():
    """direct_filter filters the frame; source_filter resets, then filters through the source."""
    tracker = PredicateTracker(pl.DataFrame({"val": [1, 2, 3, 4, 5]}))
    assert tracker.direct_filter(pl.col("val") > 2)["val"].to_list() == [3, 4, 5]
    for pred, want in [(pl.col("val") > 2, [3, 4, 5]), (pl.col("val") < 4, [1, 2, 3])]:
        assert tracker.source_filter(pred)["val"].to_list() == want
        assert tracker.call_count == 1
        assert tracker.last_predicate is not None


_ABC = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
_VALS = pl.DataFrame({"val": [1, 2, 3, 4, 5]})
_FLOATS = pl.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
# window expressions are not pushed down
_WINDOW = pl.col("val") > pl.col("val").mean()

# tracker assertion call -> passes (None) or raises (exception, match)
ASSERTIONS = [
    Case("assert_results_match_passing", df=_VALS, expect=None, call=lambda t: t.assert_results_match(pl.col("val") > 2)),
    Case("assert_predicate_pushed_down_passing", df=_VALS, expect=None, call=lambda t: t.assert_predicate_pushed_down(pl.col("val") > 2)),
    Case(
        "assert_predicate_pushed_down_expected_not_pushed",
        df=_FLOATS,
        expect=None,
        call=lambda t: t.assert_predicate_pushed_down(_WINDOW, expected_pushed_down=False),
    ),
    Case(
        "assert_predicate_pushed_down_fails_when_not_pushed",
        df=_FLOATS,
        expect=(AssertionError, "not pushed down"),
        call=lambda t: t.assert_predicate_pushed_down(_WINDOW, expected_pushed_down=True),
    ),
    Case(
        "assert_predicate_pushed_down_fails_when_unexpectedly_pushed",
        df=_VALS,
        expect=(AssertionError, "pushed down when it shouldn't"),
        call=lambda t: t.assert_predicate_pushed_down(pl.col("val") > 2, expected_pushed_down=False),
    ),
    Case(
        "assert_projection_pushed_down_wrong_expected_columns",
        df=_ABC,
        expect=(AssertionError, "does not match"),
        call=lambda t: t.assert_projection_pushed_down(["a"], expected_columns={"a", "b"}),
    ),
    Case(
        "assert_projection_pushed_down_unexpected_pushdown",
        df=_ABC,
        expect=(AssertionError, "when it shouldn't"),
        call=lambda t: t.assert_projection_pushed_down(["a"], expected_pushed_down=False),
    ),
    Case("assert_pushed_down_requires_at_least_one", df=_ABC, expect=(ValueError, "At least one"), call=lambda t: t.assert_pushed_down()),
    Case(
        "assert_pushed_down_predicate_not_pushed_branch",
        df=_FLOATS,
        expect=None,
        call=lambda t: t.assert_pushed_down(predicate=_WINDOW, expected_predicate_pushed=False),
    ),
    # the predicate may use a column the projection drops: the filter runs before the select
    Case(
        "assert_pushed_down_combined_axes",
        df=_ABC,
        expect=None,
        call=lambda t: t.assert_pushed_down(
            predicate=pl.col("a") > 1, projection=["b"], expected_predicate_pushed=True, expected_projection_pushed=True
        ),
    ),
    # a generator projection must not be exhausted before the reference is computed
    Case(
        "assert_pushed_down_normalizes_iterable_projection",
        df=_ABC,
        expect=None,
        call=lambda t: t.assert_pushed_down(projection=(c for c in ["a", "b"]), expected_projection_pushed=True, expected_columns={"a", "b"}),
    ),
]


@pytest.mark.parametrize(("df", "call", "expect"), params(ASSERTIONS, "df", "call", "expect"))
def test_tracker_assertion(df, call, expect):
    tracker = PredicateTracker(df)
    if expect is None:
        call(tracker)
    else:
        with pytest.raises(expect[0], match=expect[1]):
            call(tracker)


def test_assert_predicate_pushed_down_with_custom_assertion():
    calls = []
    PredicateTracker(_VALS).assert_predicate_pushed_down(pl.col("val") > 2, lambda original, pushed: calls.append(pushed))
    assert len(calls) == 1 and calls[0] is not None


def _bounds_from_jan3(pred):
    analyzer = PredicateAnalyzer(pred)
    assert analyzer.extract_temporal_bounds(analyzer.find_temporal_filter("date"))[0] == _d(3)


def _fail(pred):
    raise AssertionError("Intentional failure")


# (frame, filter, assertion on the pushed predicate) -> the result equals Polars', or the assertion's error surfaces
IO_SOURCE_ASSERT = [
    Case("assertion_called_with_predicate", pred=pl.col("val") > 2, expect=None, df=_VALS, check=None),
    Case("assertion_passes", pred=pl.col("val") > 2, expect=None, df=_VALS, check=lambda p: None),
    Case(
        "assertion_fails", pred=pl.col("val") > 2, expect=((pl.exceptions.ComputeError, AssertionError), "Intentional failure"), df=_VALS, check=_fail
    ),
    Case("no_predicate_when_no_filter", pred=None, expect=None, df=pl.DataFrame({"val": [1, 2, 3]}), check=None),
    Case(
        "result_is_filtered_correctly",
        pred=pl.col("a") > 2,
        expect=None,
        df=pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"]}),
        check=None,
    ),
    Case("with_predicate_analyzer", pred=date_ >= _d(3), expect=None, df=_days(5, val=[1, 2, 3, 4, 5]), check=_bounds_from_jan3),
]


@pytest.mark.parametrize(("pred", "df", "check", "expect"), params(IO_SOURCE_ASSERT, "pred", "df", "check", "expect"))
def test_io_source_assert(pred, df, check, expect):
    seen = []

    def assertion(pushed):
        seen.append(pushed)
        if check is not None:
            check(pushed)

    lf = io_source_assert(df, assertion)
    if expect is not None:
        with pytest.raises(expect[0], match=expect[1]):
            lf.filter(pred).collect()
        return
    assert_matches_native(lf, df.lazy(), pred, check_row_order=True)
    assert len(seen) == 1
    assert (seen[0] is None) == (pred is None)
