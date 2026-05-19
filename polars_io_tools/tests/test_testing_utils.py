"""
Tests for the polars_io_tools.testing module.

These tests verify that the predicate tracking and analysis utilities work correctly.
"""

from datetime import date, timedelta

import polars as pl
import pytest

from polars_io_tools.io_sources.multi_source import FilterSpec, multi_source
from polars_io_tools.testing import PredicateAnalyzer, PredicateTracker, io_source_assert


class TestPredicateTracker:
    """Tests for the PredicateTracker class."""

    def test_tracks_predicate(self):
        """PredicateTracker captures pushed predicates."""
        df = pl.DataFrame({"date": [date(2024, 1, i) for i in range(1, 6)], "val": [1, 2, 3, 4, 5]})
        tracker = PredicateTracker(df)

        lf = tracker.lazy_frame
        result = lf.filter(pl.col("date") >= date(2024, 1, 3)).collect()

        assert tracker.last_predicate is not None
        assert len(result) == 3
        assert tracker.call_count == 1

    def test_tracks_with_columns(self):
        """PredicateTracker captures column projection."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        tracker = PredicateTracker(df)

        result = tracker.lazy_frame.select(["a", "b"]).collect()

        assert tracker.last_with_columns == ["a", "b"]
        assert set(result.columns) == {"a", "b"}

    def test_reset(self):
        """PredicateTracker.reset() clears tracked state."""
        df = pl.DataFrame({"val": [1, 2, 3]})
        tracker = PredicateTracker(df)

        tracker.lazy_frame.filter(pl.col("val") > 1).collect()
        assert tracker.last_predicate is not None
        assert tracker.call_count == 1

        tracker.reset()
        assert tracker.last_predicate is None
        assert tracker.last_with_columns is None
        assert tracker.call_count == 0

    def test_get_analyzer(self):
        """PredicateTracker.get_analyzer() returns a PredicateAnalyzer."""
        df = pl.DataFrame({"date": [date(2024, 1, i) for i in range(1, 6)], "val": [1, 2, 3, 4, 5]})
        tracker = PredicateTracker(df)

        tracker.lazy_frame.filter(pl.col("date") >= date(2024, 1, 3)).collect()

        analyzer = tracker.get_analyzer()
        assert isinstance(analyzer, PredicateAnalyzer)

    def test_get_analyzer_raises_without_predicate(self):
        """PredicateTracker.get_analyzer() raises if no predicate pushed."""
        df = pl.DataFrame({"val": [1, 2, 3]})
        tracker = PredicateTracker(df)

        with pytest.raises(ValueError, match="No predicate has been pushed"):
            tracker.get_analyzer()

    def test_multiple_collects(self):
        """PredicateTracker tracks the last predicate across multiple collects."""
        df = pl.DataFrame({"val": [1, 2, 3, 4, 5]})
        tracker = PredicateTracker(df)

        tracker.lazy_frame.filter(pl.col("val") > 2).collect()
        assert tracker.call_count == 1

        tracker.lazy_frame.filter(pl.col("val") < 4).collect()
        assert tracker.call_count == 2


class TestPredicateAnalyzer:
    """Tests for the PredicateAnalyzer class."""

    def test_find_temporal_filter_gte(self):
        """PredicateAnalyzer finds >= temporal filters."""
        df = pl.DataFrame({"date": [date(2024, 1, i) for i in range(1, 6)]})
        tracker = PredicateTracker(df)

        tracker.lazy_frame.filter(pl.col("date") >= date(2024, 1, 3)).collect()

        analyzer = tracker.get_analyzer()
        temporal = analyzer.find_temporal_filter("date")
        assert temporal is not None

        lower, upper = analyzer.extract_temporal_bounds(temporal)
        assert lower == date(2024, 1, 3)
        assert upper is None

    def test_find_temporal_filter_between(self):
        """PredicateAnalyzer finds is_between temporal filters."""
        df = pl.DataFrame({"date": [date(2024, 1, i) for i in range(1, 11)]})
        tracker = PredicateTracker(df)

        tracker.lazy_frame.filter(pl.col("date").is_between(date(2024, 1, 3), date(2024, 1, 7))).collect()

        analyzer = tracker.get_analyzer()
        temporal = analyzer.find_temporal_filter("date")
        assert temporal is not None

        lower, upper = analyzer.extract_temporal_bounds(temporal)
        assert lower == date(2024, 1, 3)
        assert upper == date(2024, 1, 7)

    def test_find_discrete_filter_eq(self):
        """PredicateAnalyzer finds equality filters."""
        df = pl.DataFrame({"category": ["A", "B", "C", "A", "B"]})
        tracker = PredicateTracker(df)

        tracker.lazy_frame.filter(pl.col("category") == "A").collect()

        analyzer = tracker.get_analyzer()
        discrete = analyzer.find_discrete_filter("category")
        assert discrete is not None

        values = analyzer.extract_discrete_values(discrete)
        assert values == {"A"}

    def test_find_discrete_filter_is_in(self):
        """PredicateAnalyzer finds is_in filters."""
        df = pl.DataFrame({"category": ["A", "B", "C", "D", "E"]})
        tracker = PredicateTracker(df)

        tracker.lazy_frame.filter(pl.col("category").is_in(["A", "C", "E"])).collect()

        analyzer = tracker.get_analyzer()
        discrete = analyzer.find_discrete_filter("category")
        assert discrete is not None

        values = analyzer.extract_discrete_values(discrete)
        assert values == {"A", "C", "E"}

    def test_has_filter_on_column(self):
        """PredicateAnalyzer.has_filter_on_column() works correctly."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        tracker = PredicateTracker(df)

        tracker.lazy_frame.filter(pl.col("a") > 1).collect()

        analyzer = tracker.get_analyzer()
        assert analyzer.has_filter_on_column("a")
        assert not analyzer.has_filter_on_column("b")

    def test_find_node_by_predicate(self):
        """PredicateAnalyzer.find_node_by_predicate() works with custom predicates."""
        df = pl.DataFrame({"val": [1, 2, 3, 4, 5]})
        tracker = PredicateTracker(df)

        tracker.lazy_frame.filter(pl.col("val") > 2).collect()

        analyzer = tracker.get_analyzer()

        # Custom predicate to find any node
        from polars_io_tools.io_sources.base import BinaryExprNode

        node = analyzer.find_node_by_predicate(lambda n: isinstance(n, BinaryExprNode))
        assert node is not None


class TestPredicateAnalyzerPluralMethods:
    """Tests for the plural find methods (find_temporal_filters, find_discrete_filters, etc.)."""

    def test_find_temporal_filters_returns_list(self):
        """find_temporal_filters returns a list of all temporal filters."""
        df = pl.DataFrame({"date": [date(2024, 1, i) for i in range(1, 11)]})
        tracker = PredicateTracker(df)

        # Two filters ANDed together
        tracker.lazy_frame.filter((pl.col("date") >= date(2024, 1, 3)) & (pl.col("date") <= date(2024, 1, 7))).collect()

        analyzer = tracker.get_analyzer()
        filters = analyzer.find_temporal_filters("date")

        assert isinstance(filters, list)
        assert len(filters) == 2

    def test_find_temporal_filters_empty_when_no_match(self):
        """find_temporal_filters returns empty list when no filters found."""
        df = pl.DataFrame({"val": [1, 2, 3]})
        tracker = PredicateTracker(df)

        tracker.lazy_frame.filter(pl.col("val") > 1).collect()

        analyzer = tracker.get_analyzer()
        filters = analyzer.find_temporal_filters("nonexistent")

        assert filters == []

    def test_find_temporal_filters_with_lookback(self):
        """find_temporal_filters finds both expanded and original filters."""
        df = pl.DataFrame({"date": [date(2024, 1, i) for i in range(1, 11)], "val": list(range(10))})
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec(lookback=timedelta(days=3))})},
            combine=lambda s: s["data"],
        )

        lf.filter(pl.col("date") >= date(2024, 1, 5)).collect()

        analyzer = tracker.get_analyzer()
        filters = analyzer.find_temporal_filters("date")

        # Should have 2 filters: expanded (Jan 2) and original (Jan 5)
        assert len(filters) == 2

        lower_bounds = {analyzer.extract_temporal_bounds(f)[0] for f in filters}
        assert date(2024, 1, 2) in lower_bounds
        assert date(2024, 1, 5) in lower_bounds

    def test_find_discrete_filters_returns_list(self):
        """find_discrete_filters returns a list of all discrete filters."""
        df = pl.DataFrame({"category": ["A", "B", "C"]})
        tracker = PredicateTracker(df)

        # Two EQ filters ANDed (impossible but tests the structure)
        tracker.lazy_frame.filter((pl.col("category") == "A") | (pl.col("category") == "B")).collect()

        analyzer = tracker.get_analyzer()
        filters = analyzer.find_discrete_filters("category")

        assert isinstance(filters, list)
        assert len(filters) == 2

    def test_find_discrete_filters_empty_when_no_match(self):
        """find_discrete_filters returns empty list when no filters found."""
        df = pl.DataFrame({"val": [1, 2, 3]})
        tracker = PredicateTracker(df)

        tracker.lazy_frame.filter(pl.col("val") > 1).collect()

        analyzer = tracker.get_analyzer()
        filters = analyzer.find_discrete_filters("nonexistent")

        assert filters == []

    def test_count_filters_on_column(self):
        """count_filters_on_column returns total filter count."""
        df = pl.DataFrame({"date": [date(2024, 1, i) for i in range(1, 11)], "val": list(range(10))})
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec(lookback=timedelta(days=3))})},
            combine=lambda s: s["data"],
        )

        lf.filter(pl.col("date") >= date(2024, 1, 5)).collect()

        analyzer = tracker.get_analyzer()

        assert analyzer.count_filters_on_column("date") == 2
        assert analyzer.count_filters_on_column("nonexistent") == 0

    def test_find_all_nodes_by_predicate(self):
        """find_all_nodes_by_predicate returns all matching nodes."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        tracker = PredicateTracker(df)

        tracker.lazy_frame.filter((pl.col("a") > 1) & (pl.col("b") < 6)).collect()

        analyzer = tracker.get_analyzer()

        from polars_io_tools.io_sources.base import BinaryExprNode

        all_binary = analyzer.find_all_nodes_by_predicate(lambda n: isinstance(n, BinaryExprNode))

        # Should find at least the AND node and the two comparison nodes
        assert len(all_binary) >= 3

    def test_singular_methods_return_first(self):
        """Singular methods (find_temporal_filter) return first match."""
        df = pl.DataFrame({"date": [date(2024, 1, i) for i in range(1, 11)], "val": list(range(10))})
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec(lookback=timedelta(days=3))})},
            combine=lambda s: s["data"],
        )

        lf.filter(pl.col("date") >= date(2024, 1, 5)).collect()

        analyzer = tracker.get_analyzer()

        # Singular method should return first match
        single = analyzer.find_temporal_filter("date")
        assert single is not None

        # Should be same as first element of plural method
        plural = analyzer.find_temporal_filters("date")
        assert single is plural[0]


class TestIntegrationWithMultiSource:
    """Integration tests with multi_source."""

    def test_tracker_with_multi_source(self):
        """PredicateTracker works with multi_source."""
        df = pl.DataFrame({"date": [date(2024, 1, i) for i in range(1, 11)], "val": list(range(10))})
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec(lookback=timedelta(days=3))})},
            combine=lambda s: s["data"],
        )

        lf.filter(pl.col("date") >= date(2024, 1, 5)).collect()

        # Verify both filters were pushed: expanded and original
        analyzer = tracker.get_analyzer()
        filters = analyzer.find_temporal_filters("date")
        assert len(filters) == 2

        # Extract all lower bounds
        lower_bounds = {analyzer.extract_temporal_bounds(f)[0] for f in filters}
        assert date(2024, 1, 2) in lower_bounds  # Expanded (Jan 5 - 3 days)
        assert date(2024, 1, 5) in lower_bounds  # Original filter

    def test_tracker_with_discrete_filter(self):
        """PredicateTracker works with discrete filters in multi_source."""
        df = pl.DataFrame({"category": ["A", "B", "C", "A", "B"], "val": [1, 2, 3, 4, 5]})
        tracker = PredicateTracker(df)

        result_lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"category": FilterSpec()})},
            combine=lambda s: s["data"],
        )

        result = result_lf.filter(pl.col("category") == "A").collect()

        assert len(result) == 2
        assert tracker.last_predicate is not None


class TestPredicateTrackerAssertions:
    """Tests for the assertion methods on PredicateTracker."""

    def test_direct_filter(self):
        """direct_filter applies filter directly to DataFrame."""
        df = pl.DataFrame({"val": [1, 2, 3, 4, 5]})
        tracker = PredicateTracker(df)

        result = tracker.direct_filter(pl.col("val") > 2)

        assert len(result) == 3
        assert result["val"].to_list() == [3, 4, 5]

    def test_source_filter(self):
        """source_filter applies filter through IO source and tracks predicate."""
        df = pl.DataFrame({"val": [1, 2, 3, 4, 5]})
        tracker = PredicateTracker(df)

        result = tracker.source_filter(pl.col("val") > 2)

        assert len(result) == 3
        assert tracker.last_predicate is not None
        assert tracker.call_count == 1

    def test_source_filter_resets_tracker(self):
        """source_filter resets the tracker before filtering."""
        df = pl.DataFrame({"val": [1, 2, 3, 4, 5]})
        tracker = PredicateTracker(df)

        # First filter
        tracker.source_filter(pl.col("val") > 2)
        assert tracker.call_count == 1

        # Second filter should reset call_count
        tracker.source_filter(pl.col("val") < 4)
        assert tracker.call_count == 1  # Reset, then incremented once

    def test_assert_results_match_passing(self):
        """assert_results_match passes when results are identical."""
        df = pl.DataFrame({"val": [1, 2, 3, 4, 5]})
        tracker = PredicateTracker(df)

        # This should not raise
        tracker.assert_results_match(pl.col("val") > 2)

    def test_assert_predicate_pushed_down_passing(self):
        """assert_predicate_pushed_down passes when predicate is pushed."""
        df = pl.DataFrame({"val": [1, 2, 3, 4, 5]})
        tracker = PredicateTracker(df)

        # Simple predicates should be pushed down
        tracker.assert_predicate_pushed_down(pl.col("val") > 2)

    def test_assert_predicate_pushed_down_with_custom_assertion(self):
        """assert_predicate_pushed_down calls custom assertion function."""
        df = pl.DataFrame({"val": [1, 2, 3, 4, 5]})
        tracker = PredicateTracker(df)

        custom_called = []

        def custom_assertion(original_expr, pushed_expr):
            custom_called.append((original_expr, pushed_expr))
            assert pushed_expr is not None

        tracker.assert_predicate_pushed_down(pl.col("val") > 2, custom_assertion)

        assert len(custom_called) == 1
        assert custom_called[0][1] is not None

    def test_assert_predicate_pushed_down_expected_not_pushed(self):
        """assert_predicate_pushed_down works when expecting no pushdown."""
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        tracker = PredicateTracker(df)

        # Window functions don't get pushed down
        expr = pl.col("val") > pl.col("val").mean()
        tracker.assert_predicate_pushed_down(expr, expected_pushed_down=False)

    def test_assert_predicate_pushed_down_fails_when_not_pushed(self):
        """assert_predicate_pushed_down fails when expected pushdown doesn't happen."""
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        tracker = PredicateTracker(df)

        # Window functions don't get pushed down
        expr = pl.col("val") > pl.col("val").mean()

        with pytest.raises(AssertionError, match="not pushed down"):
            tracker.assert_predicate_pushed_down(expr, expected_pushed_down=True)

    def test_assert_predicate_pushed_down_fails_when_unexpectedly_pushed(self):
        """assert_predicate_pushed_down fails when unexpected pushdown happens."""
        df = pl.DataFrame({"val": [1, 2, 3, 4, 5]})
        tracker = PredicateTracker(df)

        # Simple predicates are pushed down
        expr = pl.col("val") > 2

        with pytest.raises(AssertionError, match="pushed down when it shouldn't"):
            tracker.assert_predicate_pushed_down(expr, expected_pushed_down=False)


class TestProjectionAndCombinedAssertions:
    """Edge-case tests for the new projection / combined pushdown assertions.

    Happy-path coverage comes from the pivot/unpivot matrix tests that consume
    these helpers; here we only exercise the failure branches and validation.
    """

    def _df(self) -> pl.DataFrame:
        return pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

    def test_assert_projection_pushed_down_wrong_expected_columns(self):
        tracker = PredicateTracker(self._df())
        with pytest.raises(AssertionError, match="does not match"):
            tracker.assert_projection_pushed_down(["a"], expected_columns={"a", "b"})

    def test_assert_projection_pushed_down_unexpected_pushdown(self):
        tracker = PredicateTracker(self._df())
        with pytest.raises(AssertionError, match="when it shouldn't"):
            tracker.assert_projection_pushed_down(["a"], expected_pushed_down=False)

    def test_assert_pushed_down_requires_at_least_one(self):
        tracker = PredicateTracker(self._df())
        with pytest.raises(ValueError, match="At least one"):
            tracker.assert_pushed_down()

    def test_assert_pushed_down_predicate_not_pushed_branch(self):
        # Window expressions don't get pushed; verify the not-pushed branch works.
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0]})
        tracker = PredicateTracker(df)
        tracker.assert_pushed_down(
            predicate=pl.col("val") > pl.col("val").mean(),
            expected_predicate_pushed=False,
        )

    def test_assert_pushed_down_combined_axes(self):
        """Both predicate and projection in one call: the predicate may
        reference a column that the projection drops (filter runs before
        select), and both axes are independently asserted."""
        tracker = PredicateTracker(self._df())
        tracker.assert_pushed_down(
            predicate=pl.col("a") > 1,
            projection=["b"],
            expected_predicate_pushed=True,
            expected_projection_pushed=True,
        )

    def test_assert_pushed_down_normalizes_iterable_projection(self):
        """A generator-based ``projection`` must not be exhausted between the
        source pipeline and the direct reference computation."""
        tracker = PredicateTracker(self._df())
        tracker.assert_pushed_down(
            projection=(c for c in ["a", "b"]),
            expected_projection_pushed=True,
            expected_columns={"a", "b"},
        )


class TestIoSourceAssert:
    """Tests for the io_source_assert function."""

    def test_assertion_called_with_predicate(self):
        """io_source_assert calls assertion function with pushed predicate."""
        df = pl.DataFrame({"val": [1, 2, 3, 4, 5]})
        captured = []

        def capture_predicate(pred):
            captured.append(pred)

        lf = io_source_assert(df, capture_predicate)
        lf.filter(pl.col("val") > 2).collect()

        assert len(captured) == 1
        assert captured[0] is not None

    def test_assertion_passes(self):
        """io_source_assert allows collection when assertion passes."""
        df = pl.DataFrame({"val": [1, 2, 3, 4, 5]})

        def assert_predicate_exists(pred):
            assert pred is not None, "Expected predicate to be pushed down"

        lf = io_source_assert(df, assert_predicate_exists)
        result = lf.filter(pl.col("val") > 2).collect()

        assert len(result) == 3
        assert result["val"].to_list() == [3, 4, 5]

    def test_assertion_fails(self):
        """io_source_assert raises when assertion fails."""
        df = pl.DataFrame({"val": [1, 2, 3, 4, 5]})

        def always_fail(pred):
            raise AssertionError("Intentional failure")

        lf = io_source_assert(df, always_fail)

        # Polars wraps exceptions from IO sources in ComputeError
        with pytest.raises(pl.exceptions.ComputeError, match="Intentional failure"):
            lf.filter(pl.col("val") > 2).collect()

    def test_no_predicate_when_no_filter(self):
        """io_source_assert receives None when no filter is applied."""
        df = pl.DataFrame({"val": [1, 2, 3]})
        captured = []

        def capture_predicate(pred):
            captured.append(pred)

        lf = io_source_assert(df, capture_predicate)
        lf.collect()

        assert len(captured) == 1
        assert captured[0] is None

    def test_result_is_filtered_correctly(self):
        """io_source_assert returns correctly filtered data."""
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"]})

        lf = io_source_assert(df, lambda pred: None)  # No-op assertion
        result = lf.filter(pl.col("a") > 2).collect()

        expected = pl.DataFrame({"a": [3, 4, 5], "b": ["z", "x", "y"]})
        assert result.equals(expected)

    def test_with_predicate_analyzer(self):
        """io_source_assert can use PredicateAnalyzer in assertion."""
        df = pl.DataFrame({"date": [date(2024, 1, i) for i in range(1, 6)], "val": [1, 2, 3, 4, 5]})

        def check_temporal_filter(pred):
            assert pred is not None
            analyzer = PredicateAnalyzer(pred)
            temporal = analyzer.find_temporal_filter("date")
            assert temporal is not None
            lower, upper = analyzer.extract_temporal_bounds(temporal)
            assert lower == date(2024, 1, 3)

        lf = io_source_assert(df, check_temporal_filter)
        result = lf.filter(pl.col("date") >= date(2024, 1, 3)).collect()

        assert len(result) == 3
