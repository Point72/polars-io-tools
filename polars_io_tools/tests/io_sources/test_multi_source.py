"""
Tests for the multi_source function and FilterSpec class.

This module tests the coordinated filter pushdown capabilities of multi_source,
including:
- Basic filter propagation
- Lookback/lookahead temporal expansion
- Value mapping (dict and callable)
- Column name remapping
- Multiple sources with different specs
- Edge cases (empty filters, empty results, schema inference)
"""

from datetime import date, datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_io_tools.io_sources.base import BinaryExprNode, FunctionNode
from polars_io_tools.io_sources.enum import BooleanFunctionType, OperatorType
from polars_io_tools.io_sources.multi_source import (
    FilterSpec,
    _apply_value_mapping,
    _compute_output_schema,
    _get_source_col,
    multi_source,
)
from polars_io_tools.testing import PredicateAnalyzer, PredicateTracker


class TestFilterSpecDefaults:
    """Test FilterSpec default values."""

    def test_defaults(self):
        """FilterSpec should have sensible defaults."""
        spec = FilterSpec()
        assert spec.source_col is None
        assert spec.lookback == timedelta()
        assert spec.lookahead == timedelta()
        assert spec.value_mapping is None

    def test_source_col_override(self):
        """source_col can be explicitly set."""
        spec = FilterSpec(source_col="DataDate")
        assert spec.source_col == "DataDate"

    def test_lookback_override(self):
        """lookback can be explicitly set."""
        spec = FilterSpec(lookback=timedelta(days=5))
        assert spec.lookback == timedelta(days=5)

    def test_lookahead_override(self):
        """lookahead can be explicitly set."""
        spec = FilterSpec(lookahead=timedelta(days=3))
        assert spec.lookahead == timedelta(days=3)

    def test_value_mapping_dict(self):
        """value_mapping can be a dict."""
        mapping = {"A": "a", "B": "b"}
        spec = FilterSpec(value_mapping=mapping)
        assert spec.value_mapping == mapping

    def test_value_mapping_callable(self):
        """value_mapping can be a callable."""
        spec = FilterSpec(value_mapping=str.upper)
        assert spec.value_mapping == str.upper

    def test_combined_options(self):
        """All options can be set together."""
        spec = FilterSpec(
            source_col="col",
            lookback=timedelta(days=5),
            lookahead=timedelta(days=3),
            value_mapping={"x": "y"},
        )
        assert spec.source_col == "col"
        assert spec.lookback == timedelta(days=5)
        assert spec.lookahead == timedelta(days=3)
        assert spec.value_mapping == {"x": "y"}


class TestApplyValueMapping:
    """Test the _apply_value_mapping helper function."""

    def test_none_mapping(self):
        """None mapping returns values unchanged with empty unmapped set."""
        values = {"a", "b", "c"}
        mapped, unmapped = _apply_value_mapping(values, None)
        assert mapped == values
        assert unmapped == set()

    def test_dict_mapping_all_keys_present(self):
        """Dict mapping transforms all values when all keys are present."""
        values = {"USE4S", "CNE5S"}
        mapping = {"USE4S": "US4S", "CNE5S": "CN5S"}
        mapped, unmapped = _apply_value_mapping(values, mapping)
        assert mapped == {"US4S", "CN5S"}
        assert unmapped == set()

    def test_dict_mapping_partial_keys(self):
        """Dict mapping returns unmapped values separately (not pushed down)."""
        values = {"USE4S", "UNKNOWN"}
        mapping = {"USE4S": "US4S"}
        mapped, unmapped = _apply_value_mapping(values, mapping)
        assert mapped == {"US4S"}
        assert unmapped == {"UNKNOWN"}

    def test_dict_mapping_no_keys_present(self):
        """Dict mapping with no matching keys returns all values as unmapped."""
        values = {"UNKNOWN1", "UNKNOWN2"}
        mapping = {"USE4S": "US4S"}
        mapped, unmapped = _apply_value_mapping(values, mapping)
        assert mapped == set()
        assert unmapped == {"UNKNOWN1", "UNKNOWN2"}

    def test_callable_mapping(self):
        """Callable mapping applies function to each value."""
        values = {"foo", "bar"}
        mapped, unmapped = _apply_value_mapping(values, str.upper)
        assert mapped == {"FOO", "BAR"}
        assert unmapped == set()

    def test_callable_mapping_custom_function(self):
        """Custom callable mapping works correctly."""
        values = {1, 2, 3}
        mapped, unmapped = _apply_value_mapping(values, lambda x: x * 2)
        assert mapped == {2, 4, 6}
        assert unmapped == set()

    def test_empty_values(self):
        """Empty input returns empty output."""
        mapped, unmapped = _apply_value_mapping(set(), {"a": "b"})
        assert mapped == set()
        assert unmapped == set()


class TestGetSourceCol:
    """Test the _get_source_col helper function."""

    def test_none_source_col_returns_output_col(self):
        """When source_col is None, returns the output column name."""
        spec = FilterSpec()
        assert _get_source_col("date", spec) == "date"

    def test_explicit_source_col(self):
        """When source_col is set, returns it."""
        spec = FilterSpec(source_col="DataDate")
        assert _get_source_col("date", spec) == "DataDate"


class TestComputeOutputSchema:
    """Test the _compute_output_schema helper function."""

    def test_simple_join_schema(self):
        """Schema is computed correctly for a simple join."""
        left = pl.LazyFrame({"a": [1], "b": [2]})
        right = pl.LazyFrame({"a": [1], "c": [3]})
        sources = {"left": (left, {}), "right": (right, {})}

        def combine(s):
            return s["left"].join(s["right"], on="a")

        schema = _compute_output_schema(sources, combine)
        assert set(schema.keys()) == {"a", "b", "c"}

    def test_with_columns_schema(self):
        """Schema includes columns added via with_columns."""
        left = pl.LazyFrame({"a": [1]})
        sources = {"left": (left, {})}

        def combine(s):
            return s["left"].with_columns(pl.lit(42).alias("new_col"))

        schema = _compute_output_schema(sources, combine)
        assert "new_col" in schema
        assert schema["new_col"] == pl.Int32

    def test_select_schema(self):
        """Schema reflects column selection."""
        left = pl.LazyFrame({"a": [1], "b": [2], "c": [3]})
        sources = {"left": (left, {})}

        def combine(s):
            return s["left"].select(["a", "b"])

        schema = _compute_output_schema(sources, combine)
        assert set(schema.keys()) == {"a", "b"}


class TestMultiSourceBasic:
    """Test basic multi_source functionality."""

    def test_no_filters(self):
        """multi_source works without any filters applied."""
        left = pl.LazyFrame({"id": [1, 2], "val": [10, 20]})
        right = pl.LazyFrame({"id": [1, 2], "other": [100, 200]})

        lf = multi_source(
            sources={
                "left": (left, {}),
                "right": (right, {}),
            },
            combine=lambda s: s["left"].join(s["right"], on="id"),
        )

        result = lf.collect()
        expected = pl.DataFrame({"id": [1, 2], "val": [10, 20], "other": [100, 200]})
        assert_frame_equal(result, expected)

    def test_simple_date_filter(self):
        """Simple date filter propagates to both sources."""
        dates = [date(2024, 1, i) for i in range(1, 11)]
        left = pl.LazyFrame({"date": dates, "val": list(range(10))})
        right = pl.LazyFrame({"date": dates, "other": list(range(10, 20))})

        lf = multi_source(
            sources={
                "left": (left, {"date": FilterSpec()}),
                "right": (right, {"date": FilterSpec()}),
            },
            combine=lambda s: s["left"].join(s["right"], on="date", maintain_order="left"),
        )

        result = lf.filter(pl.col("date") >= date(2024, 1, 5)).collect()
        expected = pl.DataFrame(
            {
                "date": [date(2024, 1, i) for i in range(5, 11)],
                "val": [4, 5, 6, 7, 8, 9],
                "other": [14, 15, 16, 17, 18, 19],
            }
        )
        # With maintain_order="left", row order should be preserved
        assert_frame_equal(result, expected)

    def test_dt_date_filter_prunes_datetime_source(self):
        """A ``.dt.date()`` window on a Datetime source prunes upstream (no full scan) and stays correct.

        ``.dt.date()`` floors to day granularity (order-preserving, like ``cast(pl.Date)``), so the temporal range
        must still be extracted and pushed down rather than collapsing to the universe interval.
        """
        ts = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(90)]
        df = pl.DataFrame({"ts": ts, "val": list(range(90))})
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"ts": FilterSpec()})},
            combine=lambda s: s["data"],
        )

        start, end = date(2024, 1, 1), date(2024, 1, 31)
        result = lf.filter((pl.col("ts").dt.date() >= start) & (pl.col("ts").dt.date() <= end)).collect()

        # Range was extracted, so the source received a constraining predicate (not a silent full scan).
        assert tracker.last_predicate is not None
        assert result.height == 31
        assert result["ts"].max() == datetime(2024, 1, 31)

    def test_simple_equality_filter(self):
        """Equality filter propagates correctly."""
        left = pl.LazyFrame({"id": ["A", "B", "C"], "val": [1, 2, 3]})
        right = pl.LazyFrame({"id": ["A", "B", "C"], "other": [10, 20, 30]})

        lf = multi_source(
            sources={
                "left": (left, {"id": FilterSpec()}),
                "right": (right, {"id": FilterSpec()}),
            },
            combine=lambda s: s["left"].join(s["right"], on="id"),
        )

        result = lf.filter(pl.col("id") == "B").collect()
        expected = pl.DataFrame({"id": ["B"], "val": [2], "other": [20]})
        assert_frame_equal(result, expected)

    def test_is_in_filter(self):
        """is_in filter propagates correctly."""
        left = pl.LazyFrame({"id": ["A", "B", "C", "D"], "val": [1, 2, 3, 4]})
        right = pl.LazyFrame({"id": ["A", "B", "C", "D"], "other": [10, 20, 30, 40]})

        lf = multi_source(
            sources={
                "left": (left, {"id": FilterSpec()}),
                "right": (right, {"id": FilterSpec()}),
            },
            combine=lambda s: s["left"].join(s["right"], on="id"),
        )

        result = lf.filter(pl.col("id").is_in(["A", "C"])).collect()
        expected = pl.DataFrame({"id": ["A", "C"], "val": [1, 3], "other": [10, 30]})
        assert_frame_equal(result, expected, check_row_order=False)


class TestMultiSourceLookback:
    """Test lookback functionality in multi_source."""

    def test_lookback_with_date_equality(self):
        """Lookback works correctly with date equality filter (date == specific_date)."""
        dates = [date(2024, 1, i) for i in range(1, 15)]
        values = list(range(1, 15))

        df = pl.DataFrame({"date": dates, "val": values})
        tracker = PredicateTracker(df)

        # Combine function that computes a lagged value (needs lookback)
        def combine_with_lag(s):
            return s["data"].with_columns(pl.col("val").shift(3).alias("val_lag3"))

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec(lookback=timedelta(days=4))})},
            combine=combine_with_lag,
        )

        # Filter for exactly Jan 5
        result = lf.filter(pl.col("date") == date(2024, 1, 5)).collect()

        # The lagged value should be available (val on Jan 5 is 5, lag3 should be 2 from Jan 2)
        expected = pl.DataFrame({"date": [date(2024, 1, 5)], "val": [5], "val_lag3": [2]})
        assert_frame_equal(result, expected)

    def test_lookahead_with_date_equality(self):
        """Lookahead works correctly with date equality filter (date == specific_date)."""
        dates = [date(2024, 1, i) for i in range(1, 15)]
        values = list(range(1, 15))

        df = pl.DataFrame({"date": dates, "val": values})
        tracker = PredicateTracker(df)

        # Combine function that computes a lead value (needs lookahead)
        def combine_with_lead(s):
            return s["data"].with_columns(pl.col("val").shift(-3).alias("val_lead3"))

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec(lookahead=timedelta(days=4))})},
            combine=combine_with_lead,
        )

        # Filter for exactly Jan 5
        result = lf.filter(pl.col("date") == date(2024, 1, 5)).collect()

        # The lead value should be available (val on Jan 5 is 5, lead3 should be 8 from Jan 8)
        expected = pl.DataFrame({"date": [date(2024, 1, 5)], "val": [5], "val_lead3": [8]})
        assert_frame_equal(result, expected)

    def test_lookback_and_lookahead_with_date_equality(self):
        """Combined lookback and lookahead work correctly with date equality filter."""
        dates = [date(2024, 1, i) for i in range(1, 15)]
        values = list(range(1, 15))

        df = pl.DataFrame({"date": dates, "val": values})
        tracker = PredicateTracker(df)

        # Combine function that computes both lag and lead
        def combine_with_lag_and_lead(s):
            return s["data"].with_columns(
                pl.col("val").shift(2).alias("val_lag2"),
                pl.col("val").shift(-2).alias("val_lead2"),
            )

        lf = multi_source(
            sources={
                "data": (
                    tracker.lazy_frame,
                    {"date": FilterSpec(lookback=timedelta(days=3), lookahead=timedelta(days=3))},
                )
            },
            combine=combine_with_lag_and_lead,
        )

        # Filter for exactly Jan 7
        result = lf.filter(pl.col("date") == date(2024, 1, 7)).collect()

        # Both lag and lead should be available
        expected = pl.DataFrame({"date": [date(2024, 1, 7)], "val": [7], "val_lag2": [5], "val_lead2": [9]})
        assert_frame_equal(result, expected)

    def test_lookback_expands_date_range(self):
        """Lookback expands the date range for the source."""
        dates = [date(2024, 1, i) for i in range(1, 11)]
        values = list(range(100, 110))

        df = pl.DataFrame({"date": dates, "val": values})
        left_tracker = PredicateTracker(df)

        right = pl.LazyFrame({"date": dates, "other": list(range(10))})

        lf = multi_source(
            sources={
                "left": (left_tracker.lazy_frame, {"date": FilterSpec(lookback=timedelta(days=3))}),
                "right": (right, {"date": FilterSpec()}),
            },
            combine=lambda s: s["left"].join(s["right"], on="date", maintain_order="left"),
        )

        # Filter for date >= Jan 5, but left source should get date >= Jan 2 (3 days lookback)
        result = lf.filter(pl.col("date") >= date(2024, 1, 5)).collect()

        # The final result should respect the original filter
        expected = pl.DataFrame(
            {
                "date": [date(2024, 1, i) for i in range(5, 11)],
                "val": [104, 105, 106, 107, 108, 109],
                "other": [4, 5, 6, 7, 8, 9],
            }
        )
        # With maintain_order="left", row order should be preserved
        assert_frame_equal(result, expected)

        # The left source should have received an expanded filter
        pushed_predicate = left_tracker.last_predicate
        assert pushed_predicate is not None

    def test_lookback_for_rolling_computation(self):
        """Lookback enables correct rolling computations."""
        dates = [date(2024, 1, i) for i in range(1, 11)]
        values = [float(i) for i in range(1, 11)]

        data = pl.LazyFrame({"date": dates, "val": values})

        def combine_with_rolling(s: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
            return s["data"].sort("date").with_columns(pl.col("val").rolling_sum(window_size=3, min_samples=1).alias("rolling_sum"))

        lf = multi_source(
            sources={
                "data": (data, {"date": FilterSpec(lookback=timedelta(days=3))}),
            },
            combine=combine_with_rolling,
        )

        result = lf.filter(pl.col("date") >= date(2024, 1, 5)).collect()

        expected = pl.DataFrame(
            {
                "date": [date(2024, 1, i) for i in range(5, 11)],
                "val": [float(i) for i in range(5, 11)],
                "rolling_sum": [12.0, 15.0, 18.0, 21.0, 24.0, 27.0],
            }
        )
        assert_frame_equal(result, expected)


class TestMultiSourceLookahead:
    """Test lookahead functionality in multi_source."""

    def test_lookahead_expands_date_range(self):
        """Lookahead expands the upper date range for the source."""
        dates = [date(2024, 1, i) for i in range(1, 11)]
        values = list(range(100, 110))

        df = pl.DataFrame({"date": dates, "val": values})
        left_tracker = PredicateTracker(df)

        lf = multi_source(
            sources={
                "left": (left_tracker.lazy_frame, {"date": FilterSpec(lookahead=timedelta(days=2))}),
            },
            combine=lambda s: s["left"],
        )

        # Filter for date <= Jan 5, left source should get date <= Jan 7 (2 days lookahead)
        result = lf.filter(pl.col("date") <= date(2024, 1, 5)).collect()

        # The final result should respect the original filter
        expected = pl.DataFrame(
            {
                "date": [date(2024, 1, i) for i in range(1, 6)],
                "val": [100, 101, 102, 103, 104],
            }
        )
        assert_frame_equal(result, expected)

    def test_combined_lookback_lookahead(self):
        """Lookback and lookahead can be used together."""
        dates = [date(2024, 1, i) for i in range(1, 11)]
        values = list(range(100, 110))

        df = pl.DataFrame({"date": dates, "val": values})
        left_tracker = PredicateTracker(df)

        lf = multi_source(
            sources={
                "left": (
                    left_tracker.lazy_frame,
                    {"date": FilterSpec(lookback=timedelta(days=2), lookahead=timedelta(days=2))},
                ),
            },
            combine=lambda s: s["left"],
        )

        # Filter for date between Jan 4 and Jan 7
        # Source should get Jan 2 to Jan 9
        result = lf.filter(pl.col("date").is_between(date(2024, 1, 4), date(2024, 1, 7))).collect()

        # Final result respects original filter
        expected = pl.DataFrame(
            {
                "date": [date(2024, 1, i) for i in range(4, 8)],
                "val": [103, 104, 105, 106],
            }
        )
        assert_frame_equal(result, expected)


class TestMultiSourceValueMapping:
    """Test value mapping functionality in multi_source."""

    def test_dict_value_mapping(self):
        """Dict value mapping transforms filter values."""
        # Source uses "region_code" with values like "NA"
        # Output uses "region" with values like "NORTH_AMERICA"
        # The combine function must map the source values back to output values
        source_df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 1)],
                "region_code": ["NA", "EU"],
                "val": [100, 200],
            }
        )
        tracker = PredicateTracker(source_df)

        REGION_TO_CODE = {"NORTH_AMERICA": "NA", "EUROPE": "EU"}
        CODE_TO_REGION = {"NA": "NORTH_AMERICA", "EU": "EUROPE"}  # Reverse mapping for combine

        lf = multi_source(
            sources={
                "data": (
                    tracker.lazy_frame,
                    {
                        "region": FilterSpec(source_col="region_code", value_mapping=REGION_TO_CODE),
                    },
                ),
            },
            # The combine function maps region_code back to region
            combine=lambda s: s["data"].with_columns(pl.col("region_code").replace(CODE_TO_REGION).alias("region")).drop("region_code"),
        )

        result = lf.filter(pl.col("region") == "NORTH_AMERICA").collect()
        expected = pl.DataFrame({"date": [date(2024, 1, 1)], "val": [100], "region": ["NORTH_AMERICA"]})
        assert_frame_equal(result, expected)

    def test_callable_value_mapping(self):
        """Callable value mapping transforms filter values."""
        source_df = pl.DataFrame(
            {
                "name": ["foo", "bar", "baz"],
                "val": [1, 2, 3],
            }
        )
        tracker = PredicateTracker(source_df)

        # Output uses uppercase names, source uses lowercase
        lf = multi_source(
            sources={
                "data": (
                    tracker.lazy_frame,
                    {"name": FilterSpec(value_mapping=str.lower)},
                ),
            },
            combine=lambda s: s["data"].with_columns(pl.col("name").str.to_uppercase().alias("name")),
        )

        # Filter for uppercase "FOO" should become lowercase "foo" for source
        result = lf.filter(pl.col("name") == "FOO").collect()
        expected = pl.DataFrame({"name": ["FOO"], "val": [1]})
        assert_frame_equal(result, expected)

    def test_is_in_with_value_mapping(self):
        """Value mapping works with is_in filters."""
        source_df = pl.DataFrame(
            {
                "region_code": ["NA", "EU", "APAC"],
                "val": [1, 2, 3],
            }
        )
        tracker = PredicateTracker(source_df)

        REGION_TO_CODE = {"NORTH_AMERICA": "NA", "EUROPE": "EU", "ASIA_PACIFIC": "APAC"}
        CODE_TO_REGION = {"NA": "NORTH_AMERICA", "EU": "EUROPE", "APAC": "ASIA_PACIFIC"}

        lf = multi_source(
            sources={
                "data": (
                    tracker.lazy_frame,
                    {"region": FilterSpec(source_col="region_code", value_mapping=REGION_TO_CODE)},
                ),
            },
            # Map region_code back to region for the output
            combine=lambda s: s["data"].with_columns(pl.col("region_code").replace(CODE_TO_REGION).alias("region")).drop("region_code"),
        )

        result = lf.filter(pl.col("region").is_in(["NORTH_AMERICA", "ASIA_PACIFIC"])).collect()
        expected = pl.DataFrame({"val": [1, 3], "region": ["NORTH_AMERICA", "ASIA_PACIFIC"]})
        assert_frame_equal(result, expected, check_row_order=False)

    def test_unmapped_value_not_pushed_down(self):
        """Values not in the mapping dict are NOT pushed down to avoid incorrect filtering.

        This tests the safe behavior: if a user filters on a value that's not in the
        value_mapping dict, we don't push that filter down. Instead, we let the
        combine() run on unfiltered data and the final predicate handles the filtering.
        """
        # Source uses region codes; we only have partial mapping
        source_df = pl.DataFrame(
            {
                "region_code": ["NA", "EU", "APAC", "LATAM"],
                "val": [1, 2, 3, 4],
            }
        )
        tracker = PredicateTracker(source_df)

        # Partial mapping - LATAM is not included
        REGION_TO_CODE = {"NORTH_AMERICA": "NA", "EUROPE": "EU"}
        CODE_TO_REGION = {"NA": "NORTH_AMERICA", "EU": "EUROPE", "APAC": "ASIA_PACIFIC", "LATAM": "LATIN_AMERICA"}

        lf = multi_source(
            sources={
                "data": (
                    tracker.lazy_frame,
                    {"region": FilterSpec(source_col="region_code", value_mapping=REGION_TO_CODE)},
                ),
            },
            # combine() creates region column from region_code
            combine=lambda s: s["data"].with_columns(pl.col("region_code").replace(CODE_TO_REGION).alias("region")).drop("region_code"),
        )

        # Filter for "LATIN_AMERICA" which is NOT in the value_mapping
        # The old (buggy) behavior would push down region_code == "LATIN_AMERICA" which would
        # return no rows. The new (safe) behavior doesn't push down the filter.
        result = lf.filter(pl.col("region") == "LATIN_AMERICA").collect()

        # Should get the LATAM row because the filter is applied AFTER combine()
        expected = pl.DataFrame({"val": [4], "region": ["LATIN_AMERICA"]})
        assert_frame_equal(result, expected)

    def test_mixed_mapped_and_unmapped_values_in_is_in(self):
        """When is_in includes both mapped and unmapped values, only mapped values are pushed."""
        source_df = pl.DataFrame(
            {
                "region_code": ["NA", "EU", "APAC", "LATAM"],
                "val": [1, 2, 3, 4],
            }
        )
        tracker = PredicateTracker(source_df)

        # Partial mapping - only NA and EU are mapped
        REGION_TO_CODE = {"NORTH_AMERICA": "NA", "EUROPE": "EU"}
        CODE_TO_REGION = {"NA": "NORTH_AMERICA", "EU": "EUROPE", "APAC": "ASIA_PACIFIC", "LATAM": "LATIN_AMERICA"}

        lf = multi_source(
            sources={
                "data": (
                    tracker.lazy_frame,
                    {"region": FilterSpec(source_col="region_code", value_mapping=REGION_TO_CODE)},
                ),
            },
            combine=lambda s: s["data"].with_columns(pl.col("region_code").replace(CODE_TO_REGION).alias("region")).drop("region_code"),
        )

        # Filter for both mapped (NORTH_AMERICA) and unmapped (LATIN_AMERICA) values
        # Only NORTH_AMERICA->NA should be pushed down
        # LATIN_AMERICA is unmapped, so no filter for it is pushed
        result = lf.filter(pl.col("region").is_in(["NORTH_AMERICA", "LATIN_AMERICA"])).collect()

        # Should get both rows
        expected = pl.DataFrame({"val": [1, 4], "region": ["NORTH_AMERICA", "LATIN_AMERICA"]})
        assert_frame_equal(result, expected, check_row_order=False)


class TestMultiSourceColumnRemapping:
    """Test source column name remapping."""

    def test_different_source_column_name(self):
        """Source can use different column name than output."""
        source_df = pl.DataFrame(
            {
                "DataDate": [date(2024, 1, 1), date(2024, 1, 2)],
                "val": [100, 200],
            }
        )
        tracker = PredicateTracker(source_df)

        lf = multi_source(
            sources={
                "data": (
                    tracker.lazy_frame,
                    {"date": FilterSpec(source_col="DataDate")},
                ),
            },
            combine=lambda s: s["data"].rename({"DataDate": "date"}),
        )

        result = lf.filter(pl.col("date") == date(2024, 1, 1)).collect()
        expected = pl.DataFrame({"date": [date(2024, 1, 1)], "val": [100]})
        assert_frame_equal(result, expected)


class TestMultiSourceMultipleSources:
    """Test multi_source with multiple sources having different specs."""

    def test_different_lookback_per_source(self):
        """Different sources can have different lookback values."""
        dates = [date(2024, 1, i) for i in range(1, 11)]

        df1 = pl.DataFrame({"date": dates, "val1": list(range(10))})
        df2 = pl.DataFrame({"date": dates, "val2": list(range(10, 20))})

        source1_tracker = PredicateTracker(df1)
        source2_tracker = PredicateTracker(df2)

        lf = multi_source(
            sources={
                "source1": (source1_tracker.lazy_frame, {"date": FilterSpec(lookback=timedelta(days=5))}),
                "source2": (source2_tracker.lazy_frame, {"date": FilterSpec(lookback=timedelta(days=2))}),
            },
            combine=lambda s: s["source1"].join(s["source2"], on="date", maintain_order="right"),
        )

        result = lf.filter(pl.col("date") >= date(2024, 1, 7)).collect()

        # Final result respects original filter, ordered by right (source2)
        expected = pl.DataFrame(
            {
                "date": [date(2024, 1, i) for i in range(7, 11)],
                "val1": [6, 7, 8, 9],
                "val2": [16, 17, 18, 19],
            }
        )
        # With maintain_order="right", row order should follow source2
        assert_frame_equal(result, expected)

    def test_mixed_filter_types_per_source(self):
        """Different sources can specify specs for different columns."""
        dates = [date(2024, 1, i) for i in range(1, 6)]

        df1 = pl.DataFrame(
            {
                "date": dates,
                "group": ["A", "B", "A", "B", "A"],
                "val1": [1, 2, 3, 4, 5],
            }
        )
        df2 = pl.DataFrame(
            {
                "date": dates,
                "group": ["A", "B", "A", "B", "A"],  # Same values as source1
                "val2": [10, 20, 30, 40, 50],
            }
        )

        source1_tracker = PredicateTracker(df1)
        source2_tracker = PredicateTracker(df2)

        lf = multi_source(
            sources={
                "source1": (
                    source1_tracker.lazy_frame,
                    {
                        "date": FilterSpec(lookback=timedelta(days=2)),
                        "group": FilterSpec(),
                    },
                ),
                "source2": (
                    source2_tracker.lazy_frame,
                    {
                        "date": FilterSpec(),
                        "group": FilterSpec(),
                    },
                ),
            },
            combine=lambda s: s["source1"].join(
                s["source2"],
                on=["date", "group"],
                maintain_order="left",
            ),
        )

        result = lf.filter((pl.col("date") >= date(2024, 1, 3)) & (pl.col("group") == "A")).collect()

        expected = pl.DataFrame(
            {
                "date": [date(2024, 1, 3), date(2024, 1, 5)],
                "group": ["A", "A"],
                "val1": [3, 5],
                "val2": [30, 50],
            }
        )
        # With maintain_order="left", row order follows source1
        assert_frame_equal(result, expected)


class TestMultiSourceEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_result(self):
        """multi_source handles filters that result in empty output."""
        left = pl.LazyFrame({"date": [date(2024, 1, 1)], "val": [1]})

        lf = multi_source(
            sources={"left": (left, {"date": FilterSpec()})},
            combine=lambda s: s["left"],
        )

        # Filter for a date not in the data
        result = lf.filter(pl.col("date") == date(2024, 12, 31)).collect()
        assert len(result) == 0

    def test_no_filter_specs(self):
        """multi_source works when no FilterSpecs are provided."""
        left = pl.LazyFrame({"id": [1, 2], "val": [10, 20]})

        lf = multi_source(
            sources={"left": (left, {})},
            combine=lambda s: s["left"],
        )

        result = lf.filter(pl.col("id") == 1).collect()
        assert len(result) == 1

    def test_source_col_not_in_schema(self):
        """Gracefully handle when source_col doesn't exist in source schema."""
        left = pl.LazyFrame({"id": [1, 2], "val": [10, 20]})

        lf = multi_source(
            sources={
                "left": (
                    left,
                    {"nonexistent": FilterSpec(source_col="also_nonexistent")},
                )
            },
            combine=lambda s: s["left"],
        )

        # Should work without errors, just no filter pushdown for that spec
        result = lf.collect()
        assert len(result) == 2

    def test_multiple_filters_combined(self):
        """Multiple filter conditions work together."""
        dates = [date(2024, 1, i) for i in range(1, 11)]
        left = pl.LazyFrame(
            {
                "date": dates,
                "group": ["A", "B"] * 5,
                "val": list(range(10)),
            }
        )

        lf = multi_source(
            sources={
                "left": (
                    left,
                    {
                        "date": FilterSpec(),
                        "group": FilterSpec(),
                    },
                )
            },
            combine=lambda s: s["left"],
        )

        result = lf.filter((pl.col("date") >= date(2024, 1, 5)) & (pl.col("group") == "A")).collect()

        assert len(result) == 3
        assert all(r == "A" for r in result["group"].to_list())
        assert result["date"].min() == date(2024, 1, 5)

    def test_datetime_column(self):
        """Works with datetime columns, not just date."""
        datetimes = [datetime(2024, 1, 1, i) for i in range(24)]
        left = pl.LazyFrame({"ts": datetimes, "val": list(range(24))})

        lf = multi_source(
            sources={"left": (left, {"ts": FilterSpec(lookback=timedelta(hours=3))})},
            combine=lambda s: s["left"],
        )

        result = lf.filter(pl.col("ts") >= datetime(2024, 1, 1, 12)).collect()
        assert len(result) == 12
        assert result["ts"].min() == datetime(2024, 1, 1, 12)

    def test_column_selection_with_columns(self):
        """with_columns parameter is respected."""
        left = pl.LazyFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "a": [1, 2],
                "b": [10, 20],
                "c": [100, 200],
            }
        )

        lf = multi_source(
            sources={"left": (left, {"date": FilterSpec()})},
            combine=lambda s: s["left"],
        )

        # Select only specific columns
        result = lf.select(["date", "a"]).collect()
        assert set(result.columns) == {"date", "a"}

    def test_row_limit(self):
        """n_rows limit is respected."""
        dates = [date(2024, 1, i) for i in range(1, 11)]
        left = pl.LazyFrame({"date": dates, "val": list(range(10))})

        lf = multi_source(
            sources={"left": (left, {"date": FilterSpec()})},
            combine=lambda s: s["left"],
        )

        result = lf.head(3).collect()
        assert len(result) == 3


class TestMultiSourceComplexJoinUseCase:
    """Test multi_source with a complex multi-source join scenario."""

    def test_multi_source_join_with_mapping_and_lookback(self):
        """
        Test a complex join pattern where:
        - Multiple sources are joined
        - Some sources need lookback for calculations
        - Category values need mapping (category -> category_code)
        """
        dates = [date(2024, 1, i) for i in range(1, 11)]

        # Primary data: uses "category" directly
        primary_data = pl.LazyFrame(
            {
                "date": dates,
                "category": ["CAT_A"] * 10,
                "item_id": list(range(10)),
            }
        )

        # Reference data: uses "category_code", needs 4-day lookback
        # Create dates from Dec 27 through Jan 14
        ref_dates = [date(2023, 12, 27) + timedelta(days=i) for i in range(19)]
        reference_data = pl.LazyFrame(
            {
                "date": ref_dates,
                "category_code": ["A"] * 19,
                "value": [0.05 + i * 0.001 for i in range(19)],
            }
        )

        # Category mapping table
        category_mapping = pl.LazyFrame(
            {
                "category": ["CAT_A", "CAT_B"],
                "category_code": ["A", "B"],
            }
        )

        CATEGORY_TO_CODE = {"CAT_A": "A", "CAT_B": "B"}

        def combine(s: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
            # Join primary_data with category_mapping
            df_primary = s["primary"].join(s["mapping"], on="category", how="left")

            # Join with reference data
            df = df_primary.join(s["reference"], on=["date", "category_code"], how="left")

            return df

        lf = multi_source(
            sources={
                "primary": (
                    primary_data,
                    {
                        "date": FilterSpec(),
                        "category": FilterSpec(),
                    },
                ),
                "reference": (
                    reference_data,
                    {
                        "date": FilterSpec(lookback=timedelta(days=4)),
                        "category": FilterSpec(source_col="category_code", value_mapping=CATEGORY_TO_CODE),
                    },
                ),
                "mapping": (
                    category_mapping,
                    {
                        "category": FilterSpec(),
                    },
                ),
            },
            combine=combine,
        )

        # Filter for a specific date range and category
        result = lf.filter((pl.col("date").is_between(date(2024, 1, 5), date(2024, 1, 7))) & (pl.col("category") == "CAT_A")).collect()

        assert len(result) == 3
        assert result["date"].min() == date(2024, 1, 5)
        assert result["date"].max() == date(2024, 1, 7)
        assert result["category"].to_list() == ["CAT_A"] * 3
        # Reference values should be joined correctly
        assert "value" in result.columns
        assert all(r is not None for r in result["value"].to_list())


class TestFilterPushdownVerification:
    """Tests that verify filters are actually being pushed down to sources with correct structure."""

    def test_date_gte_filter_structure(self):
        """Verify date >= filter is pushed with correct structure."""
        dates = [date(2024, 1, i) for i in range(1, 11)]
        df = pl.DataFrame({"date": dates, "val": list(range(10))})
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec()})},
            combine=lambda s: s["data"],
        )

        result = lf.filter(pl.col("date") >= date(2024, 1, 5)).collect()

        # Verify result is correct
        assert len(result) == 6
        assert result["date"].min() == date(2024, 1, 5)

        # Verify predicate structure - find the temporal filter on 'date'
        pushed = tracker.last_predicate
        assert pushed is not None

        analyzer = PredicateAnalyzer(pushed)
        temporal_filter = analyzer.find_temporal_filter("date")
        assert temporal_filter is not None, "Should find a temporal filter on 'date'"

        lower, upper = analyzer.extract_temporal_bounds(temporal_filter)
        assert lower == date(2024, 1, 5), f"Lower bound should be Jan 5, got {lower}"

    def test_date_between_filter_structure(self):
        """Verify date is_between filter is pushed with correct structure."""
        dates = [date(2024, 1, i) for i in range(1, 11)]
        df = pl.DataFrame({"date": dates, "val": list(range(10))})
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec()})},
            combine=lambda s: s["data"],
        )

        result = lf.filter(pl.col("date").is_between(date(2024, 1, 3), date(2024, 1, 7))).collect()

        # Verify result is correct
        assert len(result) == 5
        assert result["date"].min() == date(2024, 1, 3)
        assert result["date"].max() == date(2024, 1, 7)

        # Verify predicate structure
        pushed = tracker.last_predicate
        assert pushed is not None

        analyzer = PredicateAnalyzer(pushed)
        temporal_filter = analyzer.find_temporal_filter("date")
        assert temporal_filter is not None

        lower, upper = analyzer.extract_temporal_bounds(temporal_filter)
        assert lower == date(2024, 1, 3)
        assert upper == date(2024, 1, 7)

    def test_lookback_expands_date_range_correctly(self):
        """Verify lookback correctly expands the pushed date range."""
        dates = [date(2024, 1, i) for i in range(1, 11)]
        df = pl.DataFrame({"date": dates, "val": list(range(10))})
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec(lookback=timedelta(days=3))})},
            combine=lambda s: s["data"],
        )

        # Filter for >= Jan 5, but source should get >= Jan 2 (3-day lookback)
        result = lf.filter(pl.col("date") >= date(2024, 1, 5)).collect()

        # Final result respects original filter
        assert len(result) == 6
        assert result["date"].min() == date(2024, 1, 5)

        # Verify the pushed predicate has expanded range
        pushed = tracker.last_predicate
        assert pushed is not None

        # Find the temporal filter - should have lookback applied
        analyzer = PredicateAnalyzer(pushed)
        temporal_filter = analyzer.find_temporal_filter("date")
        assert temporal_filter is not None

        lower, upper = analyzer.extract_temporal_bounds(temporal_filter)
        # The pushed value should be Jan 2 (3-day lookback from Jan 5)
        assert lower == date(2024, 1, 2), f"Expected Jan 2 (with lookback), got {lower}"

    def test_lookback_with_between_allows_rolling_computation(self):
        """Verify lookback allows accessing historical data for rolling computations."""
        dates = [date(2024, 1, i) for i in range(1, 15)]
        # Values: 1, 2, 3, ... 14
        df = pl.DataFrame({"date": dates, "val": list(range(1, 15))})
        tracker = PredicateTracker(df)

        # Combine function that computes a lagged value (needs lookback)
        def combine_with_lag(s):
            # Add a column that shows the value from 3 days ago
            return s["data"].with_columns(pl.col("val").shift(3).alias("val_lag3"))

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec(lookback=timedelta(days=4))})},
            combine=combine_with_lag,
        )

        # Filter for between Jan 5 and Jan 10
        result = lf.filter(pl.col("date").is_between(date(2024, 1, 5), date(2024, 1, 10))).collect()

        # Final result respects original filter
        assert len(result) == 6
        assert result["date"].min() == date(2024, 1, 5)
        assert result["date"].max() == date(2024, 1, 10)

        # Verify the lagged value is available (requires lookback to work)
        # On Jan 5 (val=5), val_lag3 should be 2 (from Jan 2)
        jan5_row = result.filter(pl.col("date") == date(2024, 1, 5))
        assert jan5_row["val_lag3"][0] == 2, "Lookback should make historical data available for lag"

        # On Jan 10 (val=10), val_lag3 should be 7 (from Jan 7)
        jan10_row = result.filter(pl.col("date") == date(2024, 1, 10))
        assert jan10_row["val_lag3"][0] == 7

    def test_lookback_with_is_between_and_complex_combine(self):
        """Verify lookback expands is_between bounds when combine uses historical data.

        This test documents an important behavior: Polars' query optimizer may combine
        filters when the combine function is simple (e.g., identity). When this happens,
        only the tighter filter (the original) is pushed to the source.

        However, when the combine function contains operations that prevent filter
        pushdown optimization (like .shift(), .rolling(), etc.), the expanded filter
        IS pushed to the source, allowing lookback/lookahead to work correctly.

        This is correct behavior - if the combine function doesn't use the extra rows
        fetched by lookback, there's no point in fetching them.
        """
        dates = [date(2024, 1, i) for i in range(1, 15)]
        df = pl.DataFrame({"date": dates, "val": list(range(1, 15))})
        tracker = PredicateTracker(df)

        # Combine function with shift - prevents Polars from optimizing through it
        def combine_with_lag(s):
            return s["data"].with_columns(pl.col("val").shift(3).alias("val_lag3"))

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec(lookback=timedelta(days=4))})},
            combine=combine_with_lag,
        )

        # Filter with is_between - lookback should expand the lower bound
        result = lf.filter(pl.col("date").is_between(date(2024, 1, 5), date(2024, 1, 10))).collect()

        # Verify result is correct
        assert len(result) == 6
        assert result["date"].min() == date(2024, 1, 5)
        assert result["date"].max() == date(2024, 1, 10)

        # Verify lag values are correct (proves lookback worked)
        jan5_row = result.filter(pl.col("date") == date(2024, 1, 5))
        assert jan5_row["val_lag3"][0] == 2, "Jan 5 lag should be 2 (from Jan 2)"

        # Verify the pushed predicate has expanded bounds
        pushed = tracker.last_predicate
        assert pushed is not None

        analyzer = PredicateAnalyzer(pushed)
        temporal_filter = analyzer.find_temporal_filter("date")
        assert temporal_filter is not None

        lower, upper = analyzer.extract_temporal_bounds(temporal_filter)
        # Lower should be Jan 1 (4-day lookback from Jan 5)
        assert lower == date(2024, 1, 1), f"Expected Jan 1 (with lookback), got {lower}"
        # Upper should remain Jan 10 (no lookahead)
        assert upper == date(2024, 1, 10), f"Expected Jan 10, got {upper}"

    def test_lookback_optimization_with_identity_combine(self):
        """Document that Polars may optimize away lookback when combine is identity.

        When the combine function is simple (like identity: lambda s: s["data"]),
        Polars' query optimizer can see that the expanded filter and final filter
        can be combined. In this case, only the tighter filter is pushed to the source.

        This is NOT a bug - it's correct optimization behavior. The lookback is only
        useful when the combine function actually uses historical data.
        """
        dates = [date(2024, 1, i) for i in range(1, 15)]
        df = pl.DataFrame({"date": dates, "val": list(range(1, 15))})
        tracker = PredicateTracker(df)

        # Identity combine - Polars can optimize through this
        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec(lookback=timedelta(days=4))})},
            combine=lambda s: s["data"],
        )

        result = lf.filter(pl.col("date").is_between(date(2024, 1, 5), date(2024, 1, 10))).collect()

        # Result is still correct
        assert len(result) == 6
        assert result["date"].min() == date(2024, 1, 5)

        # The pushed predicate may be the original (tighter) filter due to optimization
        # This is expected behavior - Polars optimizes redundant lookback fetches
        pushed = tracker.last_predicate
        assert pushed is not None

        analyzer = PredicateAnalyzer(pushed)
        temporal_filter = analyzer.find_temporal_filter("date")
        assert temporal_filter is not None

        lower, upper = analyzer.extract_temporal_bounds(temporal_filter)
        # Note: Due to Polars optimization, lower may be Jan 5 (original) not Jan 1 (expanded)
        # Both are valid - the key is that results are correct
        assert lower in (date(2024, 1, 1), date(2024, 1, 5)), f"Lower should be Jan 1 or Jan 5, got {lower}"
        assert upper == date(2024, 1, 10)

    def test_lookahead_expands_upper_bound(self):
        """Verify lookahead expands upper bound of filter."""
        dates = [date(2024, 1, i) for i in range(1, 15)]
        df = pl.DataFrame({"date": dates, "val": list(range(14))})
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec(lookahead=timedelta(days=3))})},
            combine=lambda s: s["data"],
        )

        # Filter for <= Jan 7, source should get <= Jan 10 (3-day lookahead)
        result = lf.filter(pl.col("date") <= date(2024, 1, 7)).collect()

        # Final result respects original filter
        assert len(result) == 7
        assert result["date"].max() == date(2024, 1, 7)

        # Verify the pushed predicate has expanded upper bound
        pushed = tracker.last_predicate
        assert pushed is not None

        analyzer = PredicateAnalyzer(pushed)
        temporal_filter = analyzer.find_temporal_filter("date")
        assert temporal_filter is not None

        lower, upper = analyzer.extract_temporal_bounds(temporal_filter)
        # The pushed value should be Jan 10 (3-day lookahead from Jan 7)
        assert upper == date(2024, 1, 10), f"Expected Jan 10 (with lookahead), got {upper}"

    def test_discrete_equality_filter_structure(self):
        """Verify discrete equality filter is pushed with correct structure."""
        df = pl.DataFrame({"group": ["A", "B", "C"], "val": [1, 2, 3]})
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"group": FilterSpec()})},
            combine=lambda s: s["data"],
        )

        result = lf.filter(pl.col("group") == "B").collect()

        # Verify result
        assert len(result) == 1
        assert result["group"][0] == "B"

        # Verify predicate structure
        pushed = tracker.last_predicate
        assert pushed is not None

        # Find the discrete filter on 'group'
        analyzer = PredicateAnalyzer(pushed)
        discrete_filter = analyzer.find_discrete_filter("group")
        assert discrete_filter is not None, "Should find a discrete filter on 'group'"
        assert isinstance(discrete_filter, BinaryExprNode)
        assert discrete_filter.op == OperatorType.EQ
        assert discrete_filter.right.value == "B"

    def test_discrete_is_in_filter_structure(self):
        """Verify discrete is_in filter is pushed with correct structure."""
        df = pl.DataFrame({"group": ["A", "B", "C", "D"], "val": [1, 2, 3, 4]})
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"group": FilterSpec()})},
            combine=lambda s: s["data"],
        )

        result = lf.filter(pl.col("group").is_in(["A", "C"])).collect()

        # Verify result
        assert len(result) == 2
        assert set(result["group"].to_list()) == {"A", "C"}

        # Verify predicate structure
        pushed = tracker.last_predicate
        assert pushed is not None

        # Find the discrete filter
        analyzer = PredicateAnalyzer(pushed)
        discrete_filter = analyzer.find_discrete_filter("group")
        assert discrete_filter is not None
        assert isinstance(discrete_filter, FunctionNode)
        assert discrete_filter.function_type == BooleanFunctionType.IS_IN
        assert set(discrete_filter.inputs[1].value) == {"A", "C"}

    def test_value_mapping_transforms_pushed_value(self):
        """Verify value mapping transforms the value in the pushed filter."""
        df = pl.DataFrame(
            {
                "region_code": ["NA", "EU"],
                "val": [1, 2],
            }
        )
        tracker = PredicateTracker(df)

        CODE_TO_REGION = {"NA": "NORTH_AMERICA", "EU": "EUROPE"}

        lf = multi_source(
            sources={
                "data": (
                    tracker.lazy_frame,
                    {
                        "region": FilterSpec(
                            source_col="region_code",
                            value_mapping={"NORTH_AMERICA": "NA", "EUROPE": "EU"},
                        )
                    },
                )
            },
            combine=lambda s: s["data"].with_columns(pl.col("region_code").replace(CODE_TO_REGION).alias("region")).drop("region_code"),
        )

        result = lf.filter(pl.col("region") == "NORTH_AMERICA").collect()

        # Verify result
        assert len(result) == 1
        assert result["region"][0] == "NORTH_AMERICA"

        # Verify the pushed predicate uses the MAPPED value and SOURCE column
        pushed = tracker.last_predicate
        assert pushed is not None

        # Find the discrete filter on 'region_code' (source column)
        analyzer = PredicateAnalyzer(pushed)
        discrete_filter = analyzer.find_discrete_filter("region_code")
        assert discrete_filter is not None, "Should find a filter on 'region_code'"
        assert isinstance(discrete_filter, BinaryExprNode)
        assert discrete_filter.op == OperatorType.EQ
        # Should use mapped value "NA", not original "NORTH_AMERICA"
        assert discrete_filter.right.value == "NA"

    def test_value_mapping_with_is_in(self):
        """Verify value mapping works correctly with is_in filters."""
        df = pl.DataFrame(
            {
                "region_code": ["NA", "EU", "APAC"],
                "val": [1, 2, 3],
            }
        )
        tracker = PredicateTracker(df)

        CODE_TO_REGION = {"NA": "NORTH_AMERICA", "EU": "EUROPE", "APAC": "ASIA_PACIFIC"}
        REGION_TO_CODE = {"NORTH_AMERICA": "NA", "EUROPE": "EU", "ASIA_PACIFIC": "APAC"}

        lf = multi_source(
            sources={
                "data": (
                    tracker.lazy_frame,
                    {
                        "region": FilterSpec(
                            source_col="region_code",
                            value_mapping=REGION_TO_CODE,
                        )
                    },
                )
            },
            combine=lambda s: s["data"].with_columns(pl.col("region_code").replace(CODE_TO_REGION).alias("region")).drop("region_code"),
        )

        result = lf.filter(pl.col("region").is_in(["NORTH_AMERICA", "ASIA_PACIFIC"])).collect()

        # Verify result
        assert len(result) == 2
        assert set(result["region"].to_list()) == {"NORTH_AMERICA", "ASIA_PACIFIC"}

        # Verify the pushed predicate uses mapped values
        pushed = tracker.last_predicate
        assert pushed is not None

        # Find the discrete filter on source column
        analyzer = PredicateAnalyzer(pushed)
        discrete_filter = analyzer.find_discrete_filter("region_code")
        assert discrete_filter is not None
        assert isinstance(discrete_filter, FunctionNode)
        assert discrete_filter.function_type == BooleanFunctionType.IS_IN
        # Values should be mapped: NORTH_AMERICA->NA, ASIA_PACIFIC->APAC
        assert set(discrete_filter.inputs[1].value) == {"NA", "APAC"}

    def test_column_remapping_in_pushed_filter(self):
        """Verify column remapping works in pushed filter."""
        df = pl.DataFrame(
            {
                "DataDate": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "val": [1, 2, 3],
            }
        )
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={
                "data": (
                    tracker.lazy_frame,
                    {"date": FilterSpec(source_col="DataDate")},
                )
            },
            combine=lambda s: s["data"].rename({"DataDate": "date"}),
        )

        result = lf.filter(pl.col("date") == date(2024, 1, 2)).collect()

        # Verify result
        assert len(result) == 1
        assert result["date"][0] == date(2024, 1, 2)

        # Verify the pushed predicate uses source column name 'DataDate'
        pushed = tracker.last_predicate
        assert pushed is not None

        # For equality on a date column, it's extracted as a discrete filter (EQ)
        analyzer = PredicateAnalyzer(pushed)
        discrete_filter = analyzer.find_discrete_filter("DataDate")
        assert discrete_filter is not None, "Should find an equality filter on 'DataDate'"
        assert isinstance(discrete_filter, BinaryExprNode)
        assert discrete_filter.op == OperatorType.EQ
        assert discrete_filter.right.value == date(2024, 1, 2)

    def test_multiple_sources_get_different_predicates(self):
        """Verify different sources receive different predicates based on their specs."""
        dates = [date(2024, 1, i) for i in range(1, 11)]

        df1 = pl.DataFrame({"date": dates, "val1": list(range(10))})
        df2 = pl.DataFrame({"date": dates, "val2": list(range(10, 20))})

        source1_tracker = PredicateTracker(df1)
        source2_tracker = PredicateTracker(df2)

        lf = multi_source(
            sources={
                "source1": (source1_tracker.lazy_frame, {"date": FilterSpec(lookback=timedelta(days=5))}),
                "source2": (source2_tracker.lazy_frame, {"date": FilterSpec()}),  # No lookback
            },
            combine=lambda s: s["source1"].join(s["source2"], on="date"),
        )

        result = lf.filter(pl.col("date") >= date(2024, 1, 7)).collect()

        # Verify result
        assert len(result) == 4
        assert result["date"].min() == date(2024, 1, 7)

        # Source1 should have lookback (date >= Jan 2)
        pushed1 = source1_tracker.last_predicate
        assert pushed1 is not None
        analyzer1 = PredicateAnalyzer(pushed1)
        temporal1 = analyzer1.find_temporal_filter("date")
        assert temporal1 is not None
        lower1, _ = analyzer1.extract_temporal_bounds(temporal1)
        assert lower1 == date(2024, 1, 2), f"Source1 should have lookback to Jan 2, got {lower1}"

        # Source2 should have no lookback (date >= Jan 7)
        pushed2 = source2_tracker.last_predicate
        assert pushed2 is not None
        analyzer2 = PredicateAnalyzer(pushed2)
        temporal2 = analyzer2.find_temporal_filter("date")
        assert temporal2 is not None
        lower2, _ = analyzer2.extract_temporal_bounds(temporal2)
        assert lower2 == date(2024, 1, 7), f"Source2 should have no lookback (Jan 7), got {lower2}"


class TestMultiSourceRobustness:
    """Additional tests for edge cases and robustness."""

    def test_filter_on_column_without_spec(self):
        """Filtering on a column without a FilterSpec still works (no pushdown optimization)."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, i) for i in range(1, 6)],
                "unspecified_col": ["A", "B", "C", "D", "E"],
                "val": [1, 2, 3, 4, 5],
            }
        )
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec()})},  # Only date has FilterSpec
            combine=lambda s: s["data"],
        )

        # Filter on a column that has no FilterSpec
        result = lf.filter(pl.col("unspecified_col") == "C").collect()

        # Should still work, just without optimization
        assert len(result) == 1
        assert result["val"][0] == 3

    def test_complex_predicate_with_or(self):
        """Complex predicates with OR are handled correctly."""
        dates = [date(2024, 1, i) for i in range(1, 11)]
        df = pl.DataFrame({"date": dates, "val": list(range(10))})
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec()})},
            combine=lambda s: s["data"],
        )

        # Complex OR filter
        result = lf.filter((pl.col("date") == date(2024, 1, 3)) | (pl.col("date") == date(2024, 1, 7))).collect()

        assert len(result) == 2
        assert set(result["date"].to_list()) == {date(2024, 1, 3), date(2024, 1, 7)}

    def test_nested_filters_with_and(self):
        """Nested AND filters work correctly."""
        dates = [date(2024, 1, i) for i in range(1, 11)]
        groups = ["A", "B"] * 5
        df = pl.DataFrame({"date": dates, "group": groups, "val": list(range(10))})
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={
                "data": (
                    tracker.lazy_frame,
                    {
                        "date": FilterSpec(lookback=timedelta(days=2)),
                        "group": FilterSpec(),
                    },
                )
            },
            combine=lambda s: s["data"],
        )

        # Nested AND filter
        result = lf.filter((pl.col("date") >= date(2024, 1, 5)) & (pl.col("group") == "A")).collect()

        assert len(result) == 3
        assert all(m == "A" for m in result["group"].to_list())
        assert result["date"].min() == date(2024, 1, 5)

    def test_filter_with_null_values(self):
        """Handles data with null values correctly."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2), None, date(2024, 1, 4)],
                "val": [1, 2, 3, 4],
            }
        )
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec()})},
            combine=lambda s: s["data"],
        )

        result = lf.filter(pl.col("date") >= date(2024, 1, 2)).collect()

        # Should filter out the null and dates before Jan 2
        assert len(result) == 2
        assert date(2024, 1, 2) in result["date"].to_list()
        assert date(2024, 1, 4) in result["date"].to_list()

    def test_multiple_collects(self):
        """LazyFrame can be collected multiple times with different filters."""
        dates = [date(2024, 1, i) for i in range(1, 11)]
        df = pl.DataFrame({"date": dates, "val": list(range(10))})
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec()})},
            combine=lambda s: s["data"],
        )

        # First collect
        result1 = lf.filter(pl.col("date") >= date(2024, 1, 5)).collect()
        assert len(result1) == 6

        # Second collect with different filter
        result2 = lf.filter(pl.col("date") <= date(2024, 1, 3)).collect()
        assert len(result2) == 3

        # Both should work correctly
        assert result1["date"].min() == date(2024, 1, 5)
        assert result2["date"].max() == date(2024, 1, 3)

    def test_empty_source(self):
        """Handles empty source data correctly."""
        df = pl.DataFrame({"date": [], "val": []}, schema={"date": pl.Date, "val": pl.Int64})
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec()})},
            combine=lambda s: s["data"],
        )

        result = lf.filter(pl.col("date") >= date(2024, 1, 1)).collect()
        assert len(result) == 0

    def test_very_large_lookback(self):
        """Large lookback values work correctly."""
        dates = [date(2024, 1, i) for i in range(1, 11)]
        df = pl.DataFrame({"date": dates, "val": list(range(10))})
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={
                "data": (tracker.lazy_frame, {"date": FilterSpec(lookback=timedelta(days=365))})  # 1 year
            },
            combine=lambda s: s["data"],
        )

        result = lf.filter(pl.col("date") >= date(2024, 1, 5)).collect()

        # Should still work - lookback expands to before our data
        assert len(result) == 6
        assert result["date"].min() == date(2024, 1, 5)

    def test_combine_with_additional_columns(self):
        """Combine function can add computed columns."""
        dates = [date(2024, 1, i) for i in range(1, 6)]
        df = pl.DataFrame({"date": dates, "val": [10, 20, 30, 40, 50]})
        tracker = PredicateTracker(df)

        def combine_with_computed(s):
            return s["data"].with_columns(
                (pl.col("val") * 2).alias("val_doubled"),
                pl.lit("constant").alias("const_col"),
            )

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec()})},
            combine=combine_with_computed,
        )

        result = lf.filter(pl.col("date") >= date(2024, 1, 3)).collect()

        assert len(result) == 3
        assert "val_doubled" in result.columns
        assert "const_col" in result.columns
        assert result.filter(pl.col("date") == date(2024, 1, 3))["val_doubled"][0] == 60

    def test_combine_drops_columns(self):
        """Combine function can drop columns."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, i) for i in range(1, 6)],
                "keep_col": [1, 2, 3, 4, 5],
                "drop_col": ["a", "b", "c", "d", "e"],
            }
        )
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec()})},
            combine=lambda s: s["data"].drop("drop_col"),
        )

        result = lf.filter(pl.col("date") >= date(2024, 1, 3)).collect()

        assert len(result) == 3
        assert "keep_col" in result.columns
        assert "drop_col" not in result.columns


class TestCombineKwargs:
    """Tests for the combine_kwargs parameter."""

    def test_combine_kwargs_basic(self):
        """combine_kwargs are passed to the combine function."""
        dates = [date(2024, 1, i) for i in range(1, 6)]
        df = pl.DataFrame({"date": dates, "val": [1, 2, 3, 4, 5]})
        tracker = PredicateTracker(df)

        def combine_with_multiplier(sources, multiplier):
            return sources["data"].with_columns((pl.col("val") * multiplier).alias("scaled_val"))

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec()})},
            combine=combine_with_multiplier,
            combine_kwargs={"multiplier": 10},
        )

        result = lf.filter(pl.col("date") >= date(2024, 1, 3)).collect()

        assert len(result) == 3
        assert "scaled_val" in result.columns
        # val=3 on Jan 3 -> scaled_val=30
        assert result.filter(pl.col("date") == date(2024, 1, 3))["scaled_val"][0] == 30

    def test_combine_kwargs_with_mapping_dict(self):
        """combine_kwargs can include mapping dictionaries."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "code": ["US", "CN"],
                "val": [100, 200],
            }
        )
        tracker = PredicateTracker(df)

        CODE_TO_REGION = {"US": "North America", "CN": "Asia"}

        def combine_with_mapping(sources, code_to_region):
            return sources["data"].with_columns(pl.col("code").replace(code_to_region).alias("region"))

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec()})},
            combine=combine_with_mapping,
            combine_kwargs={"code_to_region": CODE_TO_REGION},
        )

        result = lf.collect()

        assert len(result) == 2
        assert "region" in result.columns
        assert result.filter(pl.col("code") == "US")["region"][0] == "North America"
        assert result.filter(pl.col("code") == "CN")["region"][0] == "Asia"

    def test_combine_kwargs_multiple_args(self):
        """combine_kwargs can include multiple arguments."""
        dates = [date(2024, 1, i) for i in range(1, 6)]
        df = pl.DataFrame({"date": dates, "val": [1, 2, 3, 4, 5]})
        tracker = PredicateTracker(df)

        def combine_with_multiple_args(sources, multiplier, suffix, constant):
            return sources["data"].with_columns(
                (pl.col("val") * multiplier).alias(f"scaled_{suffix}"),
                pl.lit(constant).alias("const_col"),
            )

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec()})},
            combine=combine_with_multiple_args,
            combine_kwargs={"multiplier": 2, "suffix": "doubled", "constant": "hello"},
        )

        result = lf.collect()

        assert "scaled_doubled" in result.columns
        assert "const_col" in result.columns
        assert result["scaled_doubled"].to_list() == [2, 4, 6, 8, 10]
        assert all(v == "hello" for v in result["const_col"].to_list())

    def test_combine_kwargs_with_none(self):
        """combine_kwargs=None works correctly (default behavior)."""
        dates = [date(2024, 1, i) for i in range(1, 6)]
        df = pl.DataFrame({"date": dates, "val": [1, 2, 3, 4, 5]})
        tracker = PredicateTracker(df)

        # Lambda that takes only sources dict (no kwargs)
        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec()})},
            combine=lambda s: s["data"],
            combine_kwargs=None,  # Explicit None
        )

        result = lf.collect()
        assert len(result) == 5

    def test_combine_kwargs_empty_dict(self):
        """combine_kwargs={} works correctly."""
        dates = [date(2024, 1, i) for i in range(1, 6)]
        df = pl.DataFrame({"date": dates, "val": [1, 2, 3, 4, 5]})
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec()})},
            combine=lambda s: s["data"],
            combine_kwargs={},  # Empty dict
        )

        result = lf.collect()
        assert len(result) == 5

    def test_combine_kwargs_with_join_parameters(self):
        """combine_kwargs can parameterize join logic."""
        dates = [date(2024, 1, i) for i in range(1, 6)]

        left_df = pl.DataFrame({"date": dates, "key": ["A", "B", "C", "D", "E"], "left_val": [1, 2, 3, 4, 5]})
        right_df = pl.DataFrame({"key": ["A", "C", "E"], "right_val": [100, 300, 500]})

        left_tracker = PredicateTracker(left_df)
        right_tracker = PredicateTracker(right_df)

        def combine_with_join_type(sources, join_how):
            return sources["left"].join(sources["right"], on="key", how=join_how)

        # Test with inner join
        lf_inner = multi_source(
            sources={
                "left": (left_tracker.lazy_frame, {"date": FilterSpec()}),
                "right": (right_tracker.lazy_frame, {}),
            },
            combine=combine_with_join_type,
            combine_kwargs={"join_how": "inner"},
        )

        result_inner = lf_inner.collect()
        assert len(result_inner) == 3  # Only A, C, E match

        # Test with left join
        lf_left = multi_source(
            sources={
                "left": (left_tracker.lazy_frame, {"date": FilterSpec()}),
                "right": (right_tracker.lazy_frame, {}),
            },
            combine=combine_with_join_type,
            combine_kwargs={"join_how": "left"},
        )

        result_left = lf_left.collect()
        assert len(result_left) == 5  # All left rows preserved

    def test_combine_kwargs_complex_join_pattern(self):
        """Complex pattern: parameterized category mapping with lookback."""
        dates = [date(2024, 1, i) for i in range(1, 6)]

        # Primary data uses full category names
        primary_data = pl.DataFrame(
            {
                "date": dates,
                "category": ["CATEGORY_A"] * 5,
                "item_id": list(range(5)),
            }
        )

        # Reference data uses category codes
        reference_data = pl.DataFrame(
            {
                "date": dates,
                "category_code": ["A"] * 5,
                "value": [0.05 + i * 0.001 for i in range(5)],
            }
        )

        primary_tracker = PredicateTracker(primary_data)
        reference_tracker = PredicateTracker(reference_data)

        # The mapping is passed via kwargs, not hardcoded in combine
        CATEGORY_TO_CODE = {"CATEGORY_A": "A", "CATEGORY_B": "B"}

        def combine_primary_and_reference(sources, category_to_code):
            # Add category_code column to primary based on category mapping
            primary_with_code = sources["primary"].with_columns(pl.col("category").replace(category_to_code).alias("category_code"))

            # Join with reference on date and category_code
            return primary_with_code.join(sources["reference"], on=["date", "category_code"], how="left")

        lf = multi_source(
            sources={
                "primary": (
                    primary_tracker.lazy_frame,
                    {
                        "date": FilterSpec(),
                        "category": FilterSpec(),
                    },
                ),
                "reference": (
                    reference_tracker.lazy_frame,
                    {
                        "date": FilterSpec(lookback=timedelta(days=2)),
                        "category": FilterSpec(source_col="category_code", value_mapping=CATEGORY_TO_CODE),
                    },
                ),
            },
            combine=combine_primary_and_reference,
            combine_kwargs={"category_to_code": CATEGORY_TO_CODE},
        )

        result = lf.filter((pl.col("date") >= date(2024, 1, 2)) & (pl.col("category") == "CATEGORY_A")).collect()

        assert len(result) == 4
        assert "value" in result.columns
        # Verify reference data joined correctly
        assert all(r is not None for r in result["value"].to_list())


class TestSourcesAsKwargs:
    """Tests for the sources_as_kwargs parameter."""

    def test_sources_as_kwargs_basic(self):
        """sources_as_kwargs=True passes sources as keyword arguments."""
        dates = [date(2024, 1, i) for i in range(1, 6)]

        left_df = pl.DataFrame({"date": dates, "left_val": [1, 2, 3, 4, 5]})
        right_df = pl.DataFrame({"date": dates, "right_val": [10, 20, 30, 40, 50]})

        left_tracker = PredicateTracker(left_df)
        right_tracker = PredicateTracker(right_df)

        # Combine function takes sources as keyword arguments
        def combine(left, right):
            return left.join(right, on="date")

        lf = multi_source(
            sources={
                "left": (left_tracker.lazy_frame, {"date": FilterSpec()}),
                "right": (right_tracker.lazy_frame, {"date": FilterSpec()}),
            },
            combine=combine,
            sources_as_kwargs=True,
        )

        result = lf.filter(pl.col("date") >= date(2024, 1, 3)).collect()

        assert len(result) == 3
        assert "left_val" in result.columns
        assert "right_val" in result.columns

    def test_sources_as_kwargs_with_combine_kwargs(self):
        """sources_as_kwargs=True works with combine_kwargs."""
        dates = [date(2024, 1, i) for i in range(1, 6)]
        df = pl.DataFrame({"date": dates, "val": [1, 2, 3, 4, 5]})
        tracker = PredicateTracker(df)

        # Combine function takes source as kwarg plus additional kwargs
        def combine(data, multiplier, suffix):
            return data.with_columns((pl.col("val") * multiplier).alias(f"scaled_{suffix}"))

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec()})},
            combine=combine,
            combine_kwargs={"multiplier": 10, "suffix": "x10"},
            sources_as_kwargs=True,
        )

        result = lf.collect()

        assert len(result) == 5
        assert "scaled_x10" in result.columns
        assert result["scaled_x10"].to_list() == [10, 20, 30, 40, 50]

    def test_sources_as_kwargs_with_lookback(self):
        """sources_as_kwargs=True works with lookback."""
        dates = [date(2024, 1, i) for i in range(1, 15)]
        values = list(range(1, 15))
        df = pl.DataFrame({"date": dates, "val": values})
        tracker = PredicateTracker(df)

        def combine(data):
            return data.with_columns(pl.col("val").shift(3).alias("val_lag3"))

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec(lookback=timedelta(days=4))})},
            combine=combine,
            sources_as_kwargs=True,
        )

        result = lf.filter(pl.col("date") == date(2024, 1, 5)).collect()

        assert len(result) == 1
        assert result["val_lag3"][0] == 2  # Lag should work due to lookback

    def test_sources_as_kwargs_multiple_sources(self):
        """sources_as_kwargs=True works with multiple sources."""
        dates = [date(2024, 1, i) for i in range(1, 6)]

        prices_df = pl.DataFrame({"date": dates, "price": [100.0, 101.0, 102.0, 103.0, 104.0]})
        volumes_df = pl.DataFrame({"date": dates, "volume": [1000, 1100, 1200, 1300, 1400]})
        metadata_df = pl.DataFrame({"date": dates, "category": ["A", "A", "B", "B", "A"]})

        prices_tracker = PredicateTracker(prices_df)
        volumes_tracker = PredicateTracker(volumes_df)
        metadata_tracker = PredicateTracker(metadata_df)

        def combine(prices, volumes, metadata):
            return prices.join(volumes, on="date").join(metadata, on="date")

        lf = multi_source(
            sources={
                "prices": (prices_tracker.lazy_frame, {"date": FilterSpec()}),
                "volumes": (volumes_tracker.lazy_frame, {"date": FilterSpec()}),
                "metadata": (metadata_tracker.lazy_frame, {"date": FilterSpec()}),
            },
            combine=combine,
            sources_as_kwargs=True,
        )

        result = lf.filter(pl.col("date") >= date(2024, 1, 3)).collect()

        assert len(result) == 3
        assert set(result.columns) == {"date", "price", "volume", "category"}

    def test_sources_as_kwargs_false_is_default(self):
        """Verify sources_as_kwargs=False is the default (dict style)."""
        dates = [date(2024, 1, i) for i in range(1, 6)]
        df = pl.DataFrame({"date": dates, "val": [1, 2, 3, 4, 5]})
        tracker = PredicateTracker(df)

        # This combine function expects a dict (default behavior)
        def combine(sources):
            return sources["data"]

        lf = multi_source(
            sources={"data": (tracker.lazy_frame, {"date": FilterSpec()})},
            combine=combine,
            # sources_as_kwargs not specified, should default to False
        )

        result = lf.collect()
        assert len(result) == 5

    def test_sources_as_kwargs_with_value_mapping(self):
        """sources_as_kwargs=True works with value mapping."""
        df = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "region_code": ["NA", "EU"],
                "val": [100, 200],
            }
        )
        tracker = PredicateTracker(df)

        REGION_TO_CODE = {"NORTH_AMERICA": "NA", "EUROPE": "EU"}
        CODE_TO_REGION = {"NA": "NORTH_AMERICA", "EU": "EUROPE"}

        def combine(data):
            return data.with_columns(pl.col("region_code").replace(CODE_TO_REGION).alias("region")).drop("region_code")

        lf = multi_source(
            sources={
                "data": (
                    tracker.lazy_frame,
                    {"region": FilterSpec(source_col="region_code", value_mapping=REGION_TO_CODE)},
                )
            },
            combine=combine,
            sources_as_kwargs=True,
        )

        result = lf.filter(pl.col("region") == "NORTH_AMERICA").collect()

        assert len(result) == 1
        assert result["region"][0] == "NORTH_AMERICA"

    def test_sources_as_kwargs_complex_join_pattern(self):
        """sources_as_kwargs=True with complex join and kwargs."""
        dates = [date(2024, 1, i) for i in range(1, 6)]

        primary_df = pl.DataFrame({"date": dates, "category": ["A"] * 5, "item_id": list(range(5))})
        reference_df = pl.DataFrame({"date": dates, "category_code": ["A"] * 5, "value": [0.1, 0.2, 0.3, 0.4, 0.5]})

        primary_tracker = PredicateTracker(primary_df)
        reference_tracker = PredicateTracker(reference_df)

        CATEGORY_TO_CODE = {"A": "A", "B": "B"}

        def combine(primary, reference, category_mapping):
            # Add category_code to primary using mapping
            primary_with_code = primary.with_columns(pl.col("category").replace(category_mapping).alias("category_code"))
            return primary_with_code.join(reference, on=["date", "category_code"], how="left")

        lf = multi_source(
            sources={
                "primary": (primary_tracker.lazy_frame, {"date": FilterSpec(), "category": FilterSpec()}),
                "reference": (
                    reference_tracker.lazy_frame,
                    {
                        "date": FilterSpec(lookback=timedelta(days=2)),
                        "category": FilterSpec(source_col="category_code", value_mapping=CATEGORY_TO_CODE),
                    },
                ),
            },
            combine=combine,
            combine_kwargs={"category_mapping": CATEGORY_TO_CODE},
            sources_as_kwargs=True,
        )

        result = lf.filter((pl.col("date") >= date(2024, 1, 2)) & (pl.col("category") == "A")).collect()

        assert len(result) == 4
        assert "value" in result.columns


class TestDateFilterOnDatetimeSource:
    """Regression tests for the bug where a Date-typed user filter linked to a
    Datetime source column silently dropped intraday rows.

    Without the fix, the upper bound of the pushed predicate collapses to
    midnight of the bound day (because Polars promotes a Date literal to a
    midnight Datetime when comparing against a Datetime column), excluding
    every intraday row on that day.
    """

    @staticmethod
    def _make_intraday_df():
        return pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 10),
                    datetime(2024, 1, 1, 14),
                    datetime(2024, 1, 2, 10),
                    datetime(2024, 1, 2, 14),
                    datetime(2024, 1, 3, 10),
                    datetime(2024, 1, 3, 14),
                ],
                "value": [1, 2, 3, 4, 5, 6],
            }
        )

    @staticmethod
    def _combine(s):
        return s["main"].with_columns(pl.col("timestamp").cast(pl.Date).alias("data_date"))

    def test_eq_covers_full_day(self):
        """`data_date == d` against a Datetime source must include all intraday rows on day d."""
        df = self._make_intraday_df()
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={"main": (tracker.lazy_frame, {"data_date": FilterSpec(source_col="timestamp")})},
            combine=self._combine,
        )
        result = lf.filter(pl.col("data_date") == date(2024, 1, 2)).collect()
        assert len(result) == 2
        assert sorted(result["value"].to_list()) == [3, 4]

    def test_eq_with_lookback_covers_full_day(self):
        """`data_date == d` plus lookback against a Datetime source still includes all intraday rows on day d."""
        df = self._make_intraday_df()
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={
                "main": (
                    tracker.lazy_frame,
                    {"data_date": FilterSpec(source_col="timestamp", lookback=timedelta(days=1))},
                )
            },
            combine=self._combine,
        )
        result = lf.filter(pl.col("data_date") == date(2024, 1, 2)).collect()
        assert len(result) == 2
        assert sorted(result["value"].to_list()) == [3, 4]

    def test_eq_with_subday_lookback_fetches_prior_evening(self):
        """Sub-day ``lookback`` must shift bounds across day boundaries even when
        the user filter literal is a Date (not a Datetime).

        Regression for: ``date - timedelta(hours=18)`` returns ``date`` unchanged
        in Python (truncates the timedelta to whole days), so if extension to
        full-day datetimes happens *after* lookback the sub-day shift is
        silently dropped and the prior-evening rows are not pulled.
        """
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 20),  # prior evening — should be fetched by 18h lookback
                    datetime(2024, 1, 2, 4),
                    datetime(2024, 1, 2, 10),
                    datetime(2024, 1, 3, 4),
                ],
                "value": [10, 20, 30, 40],
            }
        )
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={
                "main": (
                    tracker.lazy_frame,
                    {"data_date": FilterSpec(source_col="timestamp", lookback=timedelta(hours=18))},
                )
            },
            combine=self._combine,
        )
        # Filter the *combined* frame to day 2; the source must have pulled
        # the prior evening so the post-combine output (which keeps timestamps
        # in the prior evening that match data_date == d - 1) has the right rows.
        # The user-visible contract is that the underlying source predicate
        # spans [2024-01-01 06:00, 2024-01-03 00:00) — i.e., sub-day lookback
        # crosses the day boundary AND the upper bound still covers the full
        # day so intraday rows on day d are kept.
        result = lf.filter(pl.col("data_date") == date(2024, 1, 2)).collect()
        pushed = str(tracker.last_predicate)
        assert "2024-01-01 06:00:00" in pushed, f"sub-day lookback was truncated; pushed predicate: {pushed}"
        assert "2024-01-03 00:00:00" in pushed, f"upper bound did not cover full day; pushed predicate: {pushed}"
        # And the actual intraday rows on day 2 must survive (full-day widening).
        assert sorted(result["value"].to_list()) == [20, 30]

    def test_le_covers_full_day(self):
        """`data_date <= d` against a Datetime source must include all intraday rows on day d."""
        df = self._make_intraday_df()
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={"main": (tracker.lazy_frame, {"data_date": FilterSpec(source_col="timestamp")})},
            combine=self._combine,
        )
        result = lf.filter(pl.col("data_date") <= date(2024, 1, 2)).collect()
        assert len(result) == 4
        assert sorted(result["value"].to_list()) == [1, 2, 3, 4]

    def test_ge_starts_at_midnight(self):
        """`data_date >= d` against a Datetime source starts at midnight of d (inclusive)."""
        df = self._make_intraday_df()
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={"main": (tracker.lazy_frame, {"data_date": FilterSpec(source_col="timestamp")})},
            combine=self._combine,
        )
        result = lf.filter(pl.col("data_date") >= date(2024, 1, 2)).collect()
        assert len(result) == 4
        assert sorted(result["value"].to_list()) == [3, 4, 5, 6]

    def test_lt_excludes_full_day(self):
        """`data_date < d` against a Datetime source excludes all intraday rows on day d."""
        df = self._make_intraday_df()
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={"main": (tracker.lazy_frame, {"data_date": FilterSpec(source_col="timestamp")})},
            combine=self._combine,
        )
        result = lf.filter(pl.col("data_date") < date(2024, 1, 2)).collect()
        assert len(result) == 2
        assert sorted(result["value"].to_list()) == [1, 2]

    def test_gt_starts_next_day(self):
        """`data_date > d` against a Datetime source includes only rows strictly after day d."""
        df = self._make_intraday_df()
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={"main": (tracker.lazy_frame, {"data_date": FilterSpec(source_col="timestamp")})},
            combine=self._combine,
        )
        result = lf.filter(pl.col("data_date") > date(2024, 1, 2)).collect()
        assert len(result) == 2
        assert sorted(result["value"].to_list()) == [5, 6]

    def test_is_in_covers_full_days(self):
        """`data_date in [d1, d2]` against a Datetime source covers all intraday rows on both days."""
        df = self._make_intraday_df()
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={"main": (tracker.lazy_frame, {"data_date": FilterSpec(source_col="timestamp")})},
            combine=self._combine,
        )
        result = lf.filter(pl.col("data_date").is_in([date(2024, 1, 2), date(2024, 1, 3)])).collect()
        assert len(result) == 4
        assert sorted(result["value"].to_list()) == [3, 4, 5, 6]

    def test_datetime_filter_on_datetime_source_unchanged(self):
        """A Datetime filter on a Datetime source preserves sub-day precision."""
        df = self._make_intraday_df()
        tracker = PredicateTracker(df)

        lf = multi_source(
            sources={"main": (tracker.lazy_frame, {"timestamp": FilterSpec()})},
            combine=lambda s: s["main"],
        )
        result = lf.filter(pl.col("timestamp") >= datetime(2024, 1, 2, 12, 0)).collect()
        assert len(result) == 3
        assert result["value"].to_list() == [4, 5, 6]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
