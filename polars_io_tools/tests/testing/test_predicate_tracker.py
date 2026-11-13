"""polars_io_tools.testing: PredicateTracker, PredicateAnalyzer and io_source_assert.

One table per tool: filters and what the analyzer finds on each column; tracker assertion calls and whether each
passes or raises; and io_source_assert calls with the filter its callback receives and the result.
"""

from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field
from datetime import date, timedelta

import polars as pl
import pytest

from polars_io_tools.io_sources.base import BinaryExprNode
from polars_io_tools.io_sources.pushdown_combine import FilterSpec, pushdown_combine
from polars_io_tools.testing import PredicateAnalyzer, PredicateTracker, io_source_assert
from polars_io_tools.tests.helpers.assertions import assert_matches_plain_polars
from polars_io_tools.tests.helpers.cases import as_params


def _january_2024(day: int) -> date:
    return date(2024, 1, day)


def _daily_frame(n: int, **more_columns) -> pl.DataFrame:
    return pl.DataFrame({"date": [_january_2024(i) for i in range(1, n + 1)], **more_columns})


def _combined_with_3_day_lookback(lf: pl.LazyFrame) -> pl.LazyFrame:
    return pushdown_combine(sources={"data": (lf, {"date": FilterSpec(lookback=timedelta(days=3))})}, combine=lambda s: s["data"])


def _combined_with_filter_on(col: str):
    return lambda lf: pushdown_combine(sources={"data": (lf, {col: FilterSpec()})}, combine=lambda s: s["data"])


@dataclass(frozen=True)
class AnalyzerCase:
    """A filter on a tracked source, and what the analyzer finds in the filter the source received."""

    id: str
    _: KW_ONLY
    frame: pl.DataFrame
    predicate: pl.Expr
    # column -> the (lower, upper) bounds of every date filter on it
    date_bounds: dict[str, list[tuple]] = field(default_factory=dict)
    # column -> the allowed values of every equality or ``is_in`` filter on it
    value_sets: dict[str, list[set]] = field(default_factory=dict)
    # the least number of binary operations (comparisons, ``&``, ``|``) in the received filter
    min_binary_nodes: int = 1
    # applied to the tracked source before the filter
    wrap_source: Callable[[pl.LazyFrame], pl.LazyFrame] | None = None


ANALYZER = [
    AnalyzerCase(
        "tracks_predicate",
        frame=_daily_frame(5, val=[1, 2, 3, 4, 5]),
        predicate=pl.col("date") >= _january_2024(3),
        date_bounds={"date": [(_january_2024(3), None)]},
    ),
    AnalyzerCase(
        "find_temporal_filter_between",
        frame=_daily_frame(10),
        predicate=pl.col("date").is_between(_january_2024(3), _january_2024(7)),
        date_bounds={"date": [(_january_2024(3), _january_2024(7))]},
        min_binary_nodes=0,
    ),
    AnalyzerCase(
        "find_temporal_filters_returns_list",
        frame=_daily_frame(10),
        predicate=(pl.col("date") >= _january_2024(3)) & (pl.col("date") <= _january_2024(7)),
        date_bounds={"date": [(_january_2024(3), None), (None, _january_2024(7))]},
    ),
    # the lookback pushes the expanded filter (Jan 2) and the original (Jan 5)
    AnalyzerCase(
        "find_temporal_filters_with_lookback",
        frame=_daily_frame(10, val=list(range(10))),
        predicate=pl.col("date") >= _january_2024(5),
        date_bounds={"date": [(_january_2024(2), None), (_january_2024(5), None)]},
        wrap_source=_combined_with_3_day_lookback,
    ),
    AnalyzerCase(
        "find_discrete_filter_eq",
        frame=pl.DataFrame({"category": ["A", "B", "C", "A", "B"]}),
        predicate=pl.col("category") == "A",
        value_sets={"category": [{"A"}]},
    ),
    AnalyzerCase(
        "find_discrete_filter_is_in",
        frame=pl.DataFrame({"category": ["A", "B", "C", "D", "E"]}),
        predicate=pl.col("category").is_in(["A", "C", "E"]),
        value_sets={"category": [{"A", "C", "E"}]},
        min_binary_nodes=0,
    ),
    AnalyzerCase(
        "find_discrete_filters_returns_list",
        frame=pl.DataFrame({"category": ["A", "B", "C"]}),
        predicate=(pl.col("category") == "A") | (pl.col("category") == "B"),
        value_sets={"category": [{"A"}, {"B"}]},
    ),
    AnalyzerCase(
        "tracker_with_discrete_filter",
        frame=pl.DataFrame({"category": ["A", "B", "C", "A", "B"], "val": [1, 2, 3, 4, 5]}),
        predicate=pl.col("category") == "A",
        value_sets={"category": [{"A"}]},
        wrap_source=_combined_with_filter_on("category"),
    ),
    # no filter on b, nor on a column that does not exist
    AnalyzerCase(
        "has_filter_on_column",
        frame=pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
        predicate=pl.col("a") > 1,
        date_bounds={"b": [], "nonexistent": []},
        value_sets={"b": [], "nonexistent": []},
    ),
    AnalyzerCase(
        "find_all_nodes_by_predicate",
        frame=pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
        predicate=(pl.col("a") > 1) & (pl.col("b") < 6),
        min_binary_nodes=3,
    ),
]


def _sorted_bounds(bounds):
    return sorted(bounds, key=lambda b: tuple((x is None, x or date.min) for x in b))


@pytest.mark.parametrize("case", as_params(ANALYZER))
def test_analyzer(case: AnalyzerCase):
    tracker = PredicateTracker(case.frame)
    lf = tracker.lazy_frame if case.wrap_source is None else case.wrap_source(tracker.lazy_frame)
    assert_matches_plain_polars(lf, case.frame.lazy(), case.predicate, check_row_order=True)
    assert tracker.call_count == 1
    analyzer = tracker.get_analyzer()
    assert isinstance(analyzer, PredicateAnalyzer)
    for col in {*case.date_bounds, *case.value_sets}:
        filters = analyzer.find_temporal_filters(col)
        assert _sorted_bounds(analyzer.extract_temporal_bounds(f) for f in filters) == _sorted_bounds(case.date_bounds.get(col, []))
        assert analyzer.find_temporal_filter(col) is (filters[0] if filters else None)
        values = analyzer.find_discrete_filters(col)
        assert sorted(map(sorted, (analyzer.extract_discrete_values(f) for f in values))) == sorted(map(sorted, case.value_sets.get(col, [])))
        assert analyzer.find_discrete_filter(col) is (values[0] if values else None)
        assert analyzer.count_filters_on_column(col) == len(filters) + len(values)
        assert analyzer.has_filter_on_column(col) == bool(filters or values)
    nodes = analyzer.find_all_nodes_by_predicate(lambda n: isinstance(n, BinaryExprNode))
    assert len(nodes) >= case.min_binary_nodes
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


_THREE_COLUMNS = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
_FIVE_INTS = pl.DataFrame({"val": [1, 2, 3, 4, 5]})
_FIVE_FLOATS = pl.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
# A filter comparing with the column mean, which Polars cannot hand to the source.
_ABOVE_MEAN = pl.col("val") > pl.col("val").mean()


@dataclass(frozen=True)
class TrackerAssertionCase:
    """A tracker assertion call, and whether it passes or raises."""

    id: str
    _: KW_ONLY
    frame: pl.DataFrame
    assertion_call: Callable[[PredicateTracker], None]
    # (exception type, message pattern) it raises; None when it passes
    expected_error: tuple | None


ASSERTIONS = [
    TrackerAssertionCase(
        "assert_results_match_passing", frame=_FIVE_INTS, expected_error=None, assertion_call=lambda t: t.assert_results_match(pl.col("val") > 2)
    ),
    TrackerAssertionCase(
        "assert_predicate_pushed_down_passing",
        frame=_FIVE_INTS,
        expected_error=None,
        assertion_call=lambda t: t.assert_predicate_pushed_down(pl.col("val") > 2),
    ),
    TrackerAssertionCase(
        "assert_predicate_pushed_down_expected_not_pushed",
        frame=_FIVE_FLOATS,
        expected_error=None,
        assertion_call=lambda t: t.assert_predicate_pushed_down(_ABOVE_MEAN, expected_pushed_down=False),
    ),
    TrackerAssertionCase(
        "assert_predicate_pushed_down_fails_when_not_pushed",
        frame=_FIVE_FLOATS,
        expected_error=(AssertionError, "not pushed down"),
        assertion_call=lambda t: t.assert_predicate_pushed_down(_ABOVE_MEAN, expected_pushed_down=True),
    ),
    TrackerAssertionCase(
        "assert_predicate_pushed_down_fails_when_unexpectedly_pushed",
        frame=_FIVE_INTS,
        expected_error=(AssertionError, "pushed down when it shouldn't"),
        assertion_call=lambda t: t.assert_predicate_pushed_down(pl.col("val") > 2, expected_pushed_down=False),
    ),
    TrackerAssertionCase(
        "assert_projection_pushed_down_wrong_expected_columns",
        frame=_THREE_COLUMNS,
        expected_error=(AssertionError, "does not match"),
        assertion_call=lambda t: t.assert_projection_pushed_down(["a"], expected_columns={"a", "b"}),
    ),
    TrackerAssertionCase(
        "assert_projection_pushed_down_unexpected_pushdown",
        frame=_THREE_COLUMNS,
        expected_error=(AssertionError, "when it shouldn't"),
        assertion_call=lambda t: t.assert_projection_pushed_down(["a"], expected_pushed_down=False),
    ),
    TrackerAssertionCase(
        "assert_pushed_down_requires_at_least_one",
        frame=_THREE_COLUMNS,
        expected_error=(ValueError, "At least one"),
        assertion_call=lambda t: t.assert_pushed_down(),
    ),
    TrackerAssertionCase(
        "assert_pushed_down_predicate_not_pushed_branch",
        frame=_FIVE_FLOATS,
        expected_error=None,
        assertion_call=lambda t: t.assert_pushed_down(predicate=_ABOVE_MEAN, expected_predicate_pushed=False),
    ),
    # the predicate may use a column the projection drops: the filter runs before the select
    TrackerAssertionCase(
        "assert_pushed_down_combined_axes",
        frame=_THREE_COLUMNS,
        expected_error=None,
        assertion_call=lambda t: t.assert_pushed_down(
            predicate=pl.col("a") > 1, projection=["b"], expected_predicate_pushed=True, expected_projection_pushed=True
        ),
    ),
    # a generator projection must not be exhausted before the reference is computed
    TrackerAssertionCase(
        "assert_pushed_down_normalizes_iterable_projection",
        frame=_THREE_COLUMNS,
        expected_error=None,
        assertion_call=lambda t: t.assert_pushed_down(
            projection=(c for c in ["a", "b"]), expected_projection_pushed=True, expected_columns={"a", "b"}
        ),
    ),
]


@pytest.mark.parametrize("case", as_params(ASSERTIONS))
def test_tracker_assertion(case: TrackerAssertionCase):
    tracker = PredicateTracker(case.frame)
    if case.expected_error is None:
        case.assertion_call(tracker)
    else:
        with pytest.raises(case.expected_error[0], match=case.expected_error[1]):
            case.assertion_call(tracker)


def test_assert_predicate_pushed_down_with_custom_assertion():
    calls = []
    PredicateTracker(_FIVE_INTS).assert_predicate_pushed_down(pl.col("val") > 2, lambda original, pushed: calls.append(pushed))
    assert len(calls) == 1 and calls[0] is not None


def _lower_date_bound_is_jan_3(pred):
    analyzer = PredicateAnalyzer(pred)
    assert analyzer.extract_temporal_bounds(analyzer.find_temporal_filter("date"))[0] == _january_2024(3)


def _always_fail(pred):
    raise AssertionError("Intentional failure")


@dataclass(frozen=True)
class IoSourceAssertCase:
    """A frame, a filter and a check on the filter the source receives; the result must equal plain Polars', or the
    check's error must come out of collect."""

    id: str
    _: KW_ONLY
    predicate: pl.Expr | None
    # (exception type, message pattern) that collecting raises; None when it does not raise
    expected_error: tuple | None
    frame: pl.DataFrame
    check_pushed: Callable[[pl.Expr | None], None] | None


IO_SOURCE_ASSERT = [
    IoSourceAssertCase("assertion_called_with_predicate", predicate=pl.col("val") > 2, expected_error=None, frame=_FIVE_INTS, check_pushed=None),
    IoSourceAssertCase("assertion_passes", predicate=pl.col("val") > 2, expected_error=None, frame=_FIVE_INTS, check_pushed=lambda p: None),
    IoSourceAssertCase(
        "assertion_fails",
        predicate=pl.col("val") > 2,
        expected_error=((pl.exceptions.ComputeError, AssertionError), "Intentional failure"),
        frame=_FIVE_INTS,
        check_pushed=_always_fail,
    ),
    IoSourceAssertCase("no_predicate_when_no_filter", predicate=None, expected_error=None, frame=pl.DataFrame({"val": [1, 2, 3]}), check_pushed=None),
    IoSourceAssertCase(
        "result_is_filtered_correctly",
        predicate=pl.col("a") > 2,
        expected_error=None,
        frame=pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"]}),
        check_pushed=None,
    ),
    IoSourceAssertCase(
        "with_predicate_analyzer",
        predicate=pl.col("date") >= _january_2024(3),
        expected_error=None,
        frame=_daily_frame(5, val=[1, 2, 3, 4, 5]),
        check_pushed=_lower_date_bound_is_jan_3,
    ),
]


@pytest.mark.parametrize("case", as_params(IO_SOURCE_ASSERT))
def test_io_source_assert(case: IoSourceAssertCase):
    seen = []

    def assertion(pushed):
        seen.append(pushed)
        if case.check_pushed is not None:
            case.check_pushed(pushed)

    lf = io_source_assert(case.frame, assertion)
    if case.expected_error is not None:
        with pytest.raises(case.expected_error[0], match=case.expected_error[1]):
            lf.filter(case.predicate).collect()
        return
    assert_matches_plain_polars(lf, case.frame.lazy(), case.predicate, check_row_order=True)
    assert len(seen) == 1
    assert (seen[0] is None) == (case.predicate is None)
