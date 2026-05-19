from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import polars as pl
import portion
import pytest
from portion import Bound, Interval

from polars_io_tools.io_sources.range_visitor import (
    IntervalVisitor,
    _convert_atomic_interval_to_polars_expr,
    _convert_interval_to_polars_expr,
    _extend_to_full_dates,
    _lookahead_interval,
    _lookback_interval,
    convert_expr_to_datetime_range,
    convert_expr_to_range,
)


class TestRangeExtractor:
    """Test suite for the range extraction functionality using numeric values."""

    def test_simple_equality_range(self):
        """Test range extraction with a single equality."""
        # Test with integer
        test_value = 42
        expr = pl.col("value_col") == test_value
        result = convert_expr_to_range(expr, "value_col")

        # Should be a singleton interval
        assert result.atomic  # Single interval
        assert result.lower == test_value
        assert result.upper == test_value
        assert result.left is Bound.CLOSED
        assert result.right is Bound.CLOSED

        # Test with custom point creation function
        expr = pl.col("value_col") == test_value
        result = convert_expr_to_range(expr, "value_col", create_point_func=lambda x: Interval.from_atomic(Bound.CLOSED, x, x + 1, Bound.OPEN))

        # Should create a small range from value to value+1
        assert result.atomic
        assert result.lower == test_value
        assert result.upper == test_value + 1
        assert result.left is Bound.CLOSED
        assert result.right is Bound.OPEN

    def test_numeric_range(self):
        """Test extracting a simple numeric range."""
        start_value = 10
        end_value = 100

        expr = (pl.col("value_col") >= start_value) & (pl.col("value_col") < end_value)
        result = convert_expr_to_range(expr, "value_col")

        assert result.atomic
        assert result.lower == start_value
        assert result.upper == end_value
        assert result.left is Bound.CLOSED
        assert result.right is Bound.OPEN

    def test_mixed_int_float_range(self):
        """Test range extraction with mixed int and float types."""
        int_val = 10
        float_val = 20.5

        # Range from int to float
        expr = (pl.col("mixed_col") >= int_val) & (pl.col("mixed_col") < float_val)
        result = convert_expr_to_range(expr, "mixed_col")

        assert result.atomic
        assert result.lower == int_val
        assert result.upper == float_val
        assert result.left is Bound.CLOSED
        assert result.right is Bound.OPEN

    def test_or_int_float_range(self):
        """Test range with OR conditions mixing int and float."""
        int_val = 15
        float_val = 20.5

        # OR between int equality and float equality
        expr = (pl.col("mixed_col") == int_val) | (pl.col("mixed_col") == float_val)
        result = convert_expr_to_range(expr, "mixed_col")

        # Should result in two distinct points
        assert not result.atomic
        # Check if int_val is in the resulting interval
        assert int_val in result
        # Check if float_val is in the resulting interval
        assert float_val in result

        # Test with different validation function
        expr = (pl.col("mixed_col") == int_val) | (pl.col("mixed_col") == float_val)
        result = convert_expr_to_range(
            expr,
            "mixed_col",
            validate_value_func=lambda x: int(x) if isinstance(x, float) else x,
            create_point_func=lambda x: portion.singleton(int(x)),
        )

        # Check if both values are in the resulting interval
        # The float will be converted to an integer
        assert int_val in result
        assert int(float_val) in result
        # The original float should not be in the result
        assert float_val not in result

    def test_in_operator_range(self):
        """Test 'is_in' operator with a list of numbers."""
        num_list = [10, 20, 30]
        # Using is_in instead of 'in' operator
        expr = pl.col("value_col").is_in(num_list)

        # Test with standard point creation
        result = convert_expr_to_range(expr, "value_col")

        # Should contain exactly the numbers in the list
        assert not result.atomic  # Multiple intervals
        for n in num_list:
            assert n in result
            # Shouldn't include values in between
            if n < num_list[-1]:
                mid_point = n + 0.5  # midpoint between n and n+1
                assert mid_point not in result

        # Test with custom point creation that expands each point to a range
        result = convert_expr_to_range(expr, "value_col", create_point_func=lambda x: Interval.from_atomic(Bound.CLOSED, x, x + 1, Bound.OPEN))

        # Should include numbers and their expansions
        assert not result.atomic  # Multiple intervals
        for n in num_list:
            assert n in result
            assert n + 0.5 in result  # Should include values within expanded range

    def test_complex_nested_expression(self):
        """Test with complex nested expressions mixing int and float."""
        int_val = 15
        float_val1 = 30.5
        float_val2 = 50.5

        # Complex nested expression with mixed types
        expr = (pl.col("mixed_col") >= int_val) & ((pl.col("mixed_col") < float_val1) | (pl.col("mixed_col") == float_val2))
        result = convert_expr_to_range(expr, "mixed_col")

        # Should contain two disjoint intervals
        assert not result.atomic
        for i in range(15, 31):
            assert i in result

        assert 31 not in result
        assert 50 not in result
        assert 50.5 in result

    def test_unbounded_ranges(self):
        """Test unbounded ranges with mixed int/float types."""
        int_val = 15
        float_val = 30.5

        # Unbounded upper range with int
        expr1 = pl.col("mixed_col") >= int_val
        result1 = convert_expr_to_range(expr1, "mixed_col")

        assert result1.atomic
        assert result1.lower == int_val
        assert result1.upper == portion.inf
        assert result1.left is Bound.CLOSED

        # Unbounded lower range with float
        expr2 = pl.col("mixed_col") < float_val
        result2 = convert_expr_to_range(expr2, "mixed_col")

        assert result2.atomic
        assert result2.lower == -portion.inf
        assert result2.upper == float_val
        assert result2.right is Bound.OPEN

    def test_value_validation(self):
        """Test with a custom validation function."""
        float_val = 42.7
        expr = pl.col("value_col") >= float_val

        # Custom validation that rounds numbers
        def round_value(value):
            if isinstance(value, (int, float)):
                return round(value)
            return value

        result = convert_expr_to_range(expr, "value_col", validate_value_func=round_value)

        # Should use the rounded value (43)
        assert result.atomic
        assert result.lower == 43  # 42.7 rounded to 43
        assert result.upper == portion.inf
        assert result.left is Bound.CLOSED


class TestAdvancedMixedDateTimeRanges:
    """Advanced tests for ranges with mixed date/datetime types."""

    def test_contradictory_mixed_types(self):
        """Test contradictory expressions with mixed types."""
        date_val = date(2023, 5, 15)
        datetime_val = datetime(2023, 5, 10, 12, 0)  # Earlier than date_val

        # This is impossible: mixed_col >= date_val AND mixed_col < datetime_val
        # since datetime_val is earlier than date_val
        expr = (pl.col("mixed_col") >= date_val) & (pl.col("mixed_col") < datetime_val)
        result = convert_expr_to_datetime_range(expr, "mixed_col")

        # Should be None since the condition is impossible
        assert result.empty

    def test_is_between_mixed_types(self):
        """Test is_between function with mixed types."""
        date_val = date(2023, 5, 15)
        datetime_val = datetime(2023, 5, 16, 12, 0)

        # Using is_between with mixed types
        expr = pl.col("mixed_col").is_between(date_val, datetime_val)
        result = convert_expr_to_datetime_range(expr, "mixed_col", get_enclosure=False)

        assert result.atomic
        assert result.lower == datetime.combine(date_val, time())
        assert result.upper == datetime_val
        assert result.left is Bound.CLOSED  # is_between is inclusive by default
        assert result.right is Bound.CLOSED

        # Test with different bound types and closed="left"
        expr2 = pl.col("mixed_col").is_between(date_val, datetime_val, closed="left")
        result2 = convert_expr_to_datetime_range(expr2, "mixed_col", get_enclosure=False)

        assert result2.atomic
        assert result2.lower == datetime.combine(date_val, time())
        assert result2.upper == datetime_val
        assert result2.left is Bound.CLOSED
        assert result2.right is Bound.OPEN

    def test_complex_or_with_mixed_types(self):
        """Test complex OR expressions with mixed date and datetime types."""
        expr1 = (pl.col("mixed_col") >= date(2023, 1, 1)) & (pl.col("mixed_col") < datetime(2023, 3, 1, 0, 0))
        expr2 = (pl.col("mixed_col") >= datetime(2023, 5, 1, 12, 0)) & (pl.col("mixed_col") < date(2023, 6, 1))
        expr = expr1 | expr2

        result = convert_expr_to_datetime_range(expr, "mixed_col", get_enclosure=False)

        # Should have two separate ranges
        assert not result.atomic

        # Check both ranges are in result
        for test_date in [datetime(2023, 1, 15), datetime(2023, 2, 15), datetime(2023, 5, 15, 0, 0)]:
            assert test_date in result

        # Check dates not in either range
        assert datetime.combine(date(2023, 4, 1), time()) not in result
        assert datetime.combine(date(2023, 7, 1), time()) not in result

    def test_different_timezone_ranges(self):
        """Test ranges with datetimes in different timezones."""
        utc_dt = datetime(2023, 5, 15, 12, 0, tzinfo=timezone.utc)
        est_dt = datetime(2023, 5, 15, 8, 0, tzinfo=timezone(timedelta(hours=-4)))  # EST, same time as UTC 12:00

        # Test with equivalent times in different zones
        expr = (pl.col("timestamp") >= utc_dt) & (pl.col("timestamp") < est_dt)
        result = convert_expr_to_range(
            expr, "timestamp", validate_value_func=lambda dt: dt if dt.tzinfo is None else dt.astimezone(timezone.utc).replace(tzinfo=None)
        )

        # The result should be an empty interval as they represent the same time when normalized
        assert result.empty or utc_dt.replace(tzinfo=None) in result

    def test_complex_overlapping_ranges(self):
        """Test complex overlapping ranges with mixed types."""
        # Create multiple overlapping ranges
        expr1 = (pl.col("mixed_col") >= date(2023, 1, 1)) & (pl.col("mixed_col") < datetime(2023, 3, 1, 0, 0))
        expr2 = (pl.col("mixed_col") >= date(2023, 2, 15)) & (pl.col("mixed_col") < datetime(2023, 4, 1, 0, 0))
        expr3 = (pl.col("mixed_col") >= date(2023, 6, 1)) & (pl.col("mixed_col") < datetime(2023, 7, 1, 0, 0))

        expr = expr1 | expr2 | expr3
        result = convert_expr_to_datetime_range(expr, "mixed_col", get_enclosure=False)

        # Should contain two disjoint intervals
        assert not result.atomic

        # Under the hood, everything gets converted to a datetime

        # First interval should be Jan 1 to Apr 1
        assert datetime(2023, 1, 1) in result
        assert datetime(2023, 3, 31) in result
        assert datetime(2023, 3, 31, 23, 59, 59) in result

        # Second interval should be Jun 1 to Jul 1
        assert datetime(2023, 6, 1) in result
        assert datetime(2023, 6, 30) in result

        # And no dates in between
        assert datetime(2023, 5, 1) not in result

    def test_in_with_mixed_types_combined_with_range(self):
        """Test combining is_in with range conditions using mixed types."""
        date_list = [date(2023, 1, 15), date(2023, 2, 15), date(2023, 3, 15)]
        min_date = date(2023, 2, 1)

        # Combine is_in with a range condition (AND)
        expr1 = pl.col("mixed_col").is_in(date_list) & (pl.col("mixed_col") >= min_date)
        result1 = convert_expr_to_range(expr1, "mixed_col")

        # Should filter out dates before min_date
        assert not result1.atomic
        assert date(2023, 1, 15) not in result1  # This date is before min_date
        assert date(2023, 2, 15) in result1
        assert date(2023, 3, 15) in result1


class TestTernaryExpressionRanges:
    """Test suite for range extraction with ternary expressions."""

    def test_simple_ternary_with_date_ranges(self):
        """Test range extraction with simple ternary expression involving date ranges."""
        # we currently do not factor in the predicate, this test confirms that behavior
        expr1 = (
            pl.when(pl.col("is_us_market"))
            .then((pl.col("date") >= date(2023, 1, 1)) & (pl.col("date") < date(2023, 3, 1)))
            .otherwise((pl.col("date") >= date(2023, 6, 1)) & (pl.col("date") < date(2023, 9, 1)))
        )
        expr2 = (
            pl.when(pl.col("is_us_market") & pl.col("date") >= date(2023, 2, 1))
            .then((pl.col("date") >= date(2023, 1, 1)) & (pl.col("date") < date(2023, 3, 1)))
            .otherwise((pl.col("date") >= date(2023, 6, 1)) & (pl.col("date") < date(2023, 9, 1)))
        )

        for expr in [expr1, expr2]:
            # This should extract the union of both ranges
            result = convert_expr_to_range(expr, "date")

            # Should have two separate ranges
            assert not result.atomic

            # The interval should include both date ranges
            assert date(2023, 1, 15) in result  # In first range
            assert date(2023, 2, 15) in result  # In first range
            assert date(2023, 6, 15) in result  # In second range
            assert date(2023, 8, 15) in result  # In second range

            # But not dates in between
            assert date(2023, 4, 15) not in result
            assert date(2023, 5, 15) not in result

            # This should extract the union of both ranges
            result = convert_expr_to_range(expr, "date")

            # Should have two separate ranges
            assert not result.atomic

            # The interval should include both date ranges
            assert date(2023, 1, 15) in result  # In first range
            assert date(2023, 2, 15) in result  # In first range
            assert date(2023, 6, 15) in result  # In second range
            assert date(2023, 8, 15) in result  # In second range

            # But not dates in between
            assert date(2023, 4, 15) not in result
            assert date(2023, 5, 15) not in result

    def test_ternary_one_branch_with_date_range(self):
        """Test range extraction with ternary where only one branch has date ranges."""
        # When is_weekend is True, use date range (weekends in Q1)
        # Otherwise, no date constraint
        expr = (
            pl.when(pl.col("is_weekend"))
            .then((pl.col("date") >= date(2023, 1, 1)) & (pl.col("date") < date(2023, 4, 1)))
            .otherwise(
                pl.lit(True)  # No date constraint
            )
        )

        result = convert_expr_to_range(expr, "date")

        # Should cover all dates since one branch has no constraints
        assert result.atomic
        assert result.lower == -portion.inf
        assert result.upper == portion.inf

    def test_ternary_combined_with_other_conditions(self):
        """Test ternary expressions combined with other filter conditions."""
        # Ternary combined with additional date range filter
        expr = pl.when(pl.col("region") == "US").then((pl.col("date") >= date(2023, 1, 1)) & (pl.col("date") < date(2023, 4, 1))).otherwise(
            (pl.col("date") >= date(2023, 6, 1)) & (pl.col("date") < date(2023, 9, 1))
        ) & (pl.col("date") < date(2023, 12, 31))

        # This should extract both ranges, each within the overall constraint
        result = convert_expr_to_range(expr, "date")

        # Should have two separate ranges
        assert not result.atomic

        # First range: Jan 1 to Apr 1
        assert date(2023, 1, 15) in result
        assert date(2023, 3, 15) in result

        # Second range: Jun 1 to Sep 1
        assert date(2023, 6, 15) in result
        assert date(2023, 8, 15) in result

        # But not dates outside either range or after Dec 31
        assert date(2023, 5, 1) not in result
        assert date(2023, 10, 1) not in result


class TestDatetimeRangeExtractor:
    """Test suite for the datetime range extraction functionality."""

    def test_single_equality(self):
        """Test filter with a single equality."""
        test_date = date(2023, 5, 15)
        # Simple equality expression
        expr = pl.col("date_col") == test_date
        result = convert_expr_to_datetime_range(expr, "date_col", coerce_date_to_datetime=False)
        # we include everything on the date
        assert result.atomic
        assert result.lower == datetime.combine(test_date, time())
        assert result.upper == datetime.combine(test_date + timedelta(1), time())
        assert result.left is Bound.CLOSED
        assert result.right is Bound.OPEN

        result = convert_expr_to_datetime_range(expr, "date_col", coerce_date_to_datetime=True)
        # we include only the specific date
        assert result.atomic
        assert result.lower == datetime.combine(test_date, time())
        assert result.upper == datetime.combine(test_date, time())
        assert result.left is Bound.CLOSED
        assert result.right is Bound.CLOSED

    def test_date_range(self):
        """Test simple date range with >= and <."""
        start_date = date(2023, 1, 1)
        end_date = date(2023, 12, 31)
        # Combine two conditions with AND to create a range
        expr = (pl.col("date_col") >= start_date) & (pl.col("date_col") < end_date)
        result = convert_expr_to_datetime_range(expr, "date_col")
        assert result.atomic
        assert result.lower == datetime.combine(start_date, time())
        assert result.upper == datetime.combine(end_date, time())
        assert result.left is Bound.CLOSED
        assert result.right is Bound.OPEN

    def test_datetime_range(self):
        """Test datetime range with time components."""
        start_time = datetime(2023, 5, 15, 9, 0)
        end_time = datetime(2023, 5, 15, 17, 0)
        # Using > and <= operators to create a datetime range
        expr = (pl.col("timestamp") > start_time) & (pl.col("timestamp") <= end_time)
        result = convert_expr_to_datetime_range(expr, "timestamp")
        assert result.atomic
        assert result.lower == start_time
        assert result.upper == end_time
        assert result.left is Bound.OPEN
        assert result.right is Bound.CLOSED

    @pytest.mark.parametrize("strict", [True, False])
    def test_cast_date_and_datetime(self, strict):
        """Test casting date to datetime in a range."""
        start_date = date(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        # Using >= and < operators to create a range
        expr = (pl.col("date_col") >= start_date) & (pl.col("date_col").cast(pl.Datetime, strict=strict) < end_date)
        result = convert_expr_to_datetime_range(expr, "date_col", coerce_date_to_datetime=True)
        assert result.atomic
        assert result.lower == datetime.combine(start_date, time())
        assert result.upper == datetime.combine(end_date, time())
        assert result.left is Bound.CLOSED
        assert result.right is Bound.OPEN

    def test_cast_datetime_time_unit(self):
        """Test casting date to datetime in a range."""
        end_date = date(2023, 12, 31)
        # Using >= and < operators to create a range
        expr = pl.col("datetime_col").cast(pl.Datetime(time_unit="ns")) == end_date
        result = convert_expr_to_datetime_range(expr, "datetime_col", coerce_date_to_datetime=True)
        assert result.atomic
        assert result.lower == datetime.combine(end_date, time())
        assert result.upper == datetime.combine(end_date, time())
        assert result.left is Bound.CLOSED
        assert result.right is Bound.CLOSED

        result = convert_expr_to_datetime_range(expr, "datetime_col", coerce_date_to_datetime=False)
        assert result.atomic
        assert result.lower == datetime.combine(end_date, time())
        assert result.upper == datetime.combine((date(2023, 12, 31) + timedelta(1)), time())
        assert result.left is Bound.CLOSED
        assert result.right is Bound.OPEN

    def test_datetime_cast_time(self):
        """Test casting datetime to time"""
        end = datetime(2023, 1, 1, 17, 0)
        expr = pl.col("datetime_col").is_between(end - timedelta(hours=1), end + timedelta(hours=1))
        # The time cast can't get converted to a datetime, so we ignore it
        expr2 = pl.col("datetime_col").cast(pl.Time()).is_between(time(12, 15), time(19, 0))
        result = convert_expr_to_datetime_range(expr & expr2, "datetime_col", coerce_date_to_datetime=True)
        assert result.atomic
        assert result.lower == (end - timedelta(hours=1))
        assert result.upper == (end + timedelta(hours=1))
        assert result.left is Bound.CLOSED
        assert result.right is Bound.CLOSED

    def test_cast_datetime_time_zone(self):
        """Test casting date to datetime in a range."""
        end = datetime(2023, 1, 1, 17, 0, tzinfo=ZoneInfo("America/New_York"))
        expr = (
            pl.col("datetime_col")
            .cast(pl.Datetime(time_unit="ns", time_zone="America/New_York"))
            .is_between(end - timedelta(hours=1), end + timedelta(hours=1))
        )
        # We convert to UTC
        result = convert_expr_to_datetime_range(expr, "datetime_col", coerce_date_to_datetime=True)
        assert result.atomic
        assert result.lower == (end - timedelta(hours=1)).astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
        assert result.upper == (end + timedelta(hours=1)).astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
        assert result.left is Bound.CLOSED
        assert result.right is Bound.CLOSED

    def test_or_conditions(self):
        """Test multiple conditions with OR logic."""
        date1 = date(2023, 1, 15)
        date2 = date(2023, 2, 15)
        # Using OR to combine two equalities
        expr = (pl.col("date_col") == date1) | (pl.col("date_col") == date2)
        result = convert_expr_to_datetime_range(expr, "date_col")
        # Should include both dates in the enclosure
        assert result is not None
        assert result.atomic
        assert result.lower <= datetime.combine(date1, time())  # Min date is at most date1
        assert result.upper >= datetime.combine(date2, time())  # Max date is at least date2

    def test_unbounded_min(self):
        """Test range with only an upper bound."""
        end_date = date(2023, 12, 31)
        # Only upper bound specified
        expr = pl.col("date_col") <= end_date
        result = convert_expr_to_datetime_range(expr, "date_col")
        assert result.atomic
        assert result.lower == -portion.inf
        assert result.upper == datetime.combine(end_date, time())
        assert result.right is Bound.CLOSED

    def test_unbounded_max(self):
        """Test range with only a lower bound."""
        start_date = date(2023, 1, 1)
        # Only lower bound specified
        expr = pl.col("date_col") >= start_date
        result = convert_expr_to_datetime_range(expr, "date_col")
        assert result.atomic
        assert result.lower == datetime.combine(start_date, time())
        assert result.upper == portion.inf
        assert result.left is Bound.CLOSED

    def test_in_operator(self):
        """Test 'is_in' operator with a list of dates."""
        date_list = [date(2023, 1, 15), date(2023, 2, 15), date(2023, 3, 15)]
        # Using is_in instead of 'in' operator
        expr = pl.col("date_col").is_in(date_list)
        result = convert_expr_to_datetime_range(expr, "date_col", coerce_date_to_datetime=False)
        # Should include min and max from the list (enclosure)
        assert result.atomic
        assert result.lower == datetime.combine(min(date_list), time())
        assert result.upper == datetime.combine(max(date_list) + timedelta(1), time())
        assert result.left is Bound.CLOSED
        assert result.right is Bound.OPEN

        result = convert_expr_to_datetime_range(expr, "date_col", coerce_date_to_datetime=True)
        assert result.atomic
        assert result.lower == datetime.combine(min(date_list), time())
        assert result.upper == datetime.combine(max(date_list), time())
        assert result.left is Bound.CLOSED
        assert result.right is Bound.CLOSED

    def test_in_with_other_conditions(self):
        """Test 'is_in' operator combined with range conditions."""
        date_list = [date(2023, 1, 15), date(2023, 2, 15), date(2023, 3, 15)]
        min_date = date(2023, 2, 1)
        # Combine is_in with a range condition
        expr = pl.col("date_col").is_in(date_list) & (pl.col("date_col") >= min_date)
        # Because we have an AND value, both conditions must be True
        result = convert_expr_to_datetime_range(expr, "date_col", coerce_date_to_datetime=False)
        assert result.atomic
        assert result.lower == datetime.combine(date(2023, 2, 15), time())
        assert result.upper == datetime.combine(date(2023, 3, 15) + timedelta(1), time())
        assert result.left is Bound.CLOSED
        assert result.right is Bound.OPEN

        expr = pl.col("date_col").is_in(date_list) | (pl.col("date_col") >= min_date)
        # Because we have an OR, either condition is True
        result = convert_expr_to_datetime_range(expr, "date_col", coerce_date_to_datetime=False)
        assert result.atomic
        assert result.lower == datetime.combine(date(2023, 1, 15), time())
        assert result.upper == portion.inf
        assert result.left is Bound.CLOSED

    def test_not_equal(self):
        """Test != operator in a range."""
        test_date = date(2023, 5, 15)
        other_date = date(2023, 5, 10)
        # Combine range conditions with not equal
        expr = (pl.col("date_col") >= other_date) & (pl.col("date_col") <= date(2023, 5, 20)) & (pl.col("date_col") != test_date)
        result = convert_expr_to_datetime_range(expr, "date_col")
        # Range should still be valid, we just can't express the excluded value
        assert result is not None
        assert datetime.combine(other_date, time()) <= result.lower or result.lower == -portion.inf
        assert datetime(2023, 5, 20) >= result.upper or result.upper == portion.inf

    def test_contradiction_equal_not_equal(self):
        """Test contradiction with = and != on the same value."""
        test_date = date(2023, 5, 15)
        expr = (pl.col("date_col") == test_date) & (pl.col("date_col") != test_date)
        result = convert_expr_to_datetime_range(expr, "date_col", coerce_date_to_datetime=False)
        # we ignore != for datetime
        assert result.atomic
        assert result.lower == datetime.combine(test_date, time())
        assert result.upper == datetime.combine(test_date + timedelta(1), time())
        assert result.left is Bound.CLOSED
        assert result.right is Bound.OPEN

    def test_contradiction_range(self):
        """Test contradiction with conflicting ranges."""
        # This is impossible: date >= 5/15 AND date < 5/10
        expr_list = [
            pl.col("time_col").is_between(datetime(2024, 2, 2), datetime(2023, 2, 3), closed="left"),
            (pl.col("time_col") >= date(2023, 5, 15)) & (pl.col("time_col") < date(2023, 5, 10)),
        ]
        for expr in expr_list:
            result = convert_expr_to_datetime_range(expr, "time_col")
            assert result.empty

    def test_multiple_equalities_consistent(self):
        """Test multiple consistent equality constraints."""
        test_date = date(2023, 5, 15)
        # Same value with redundant condition
        expr = (pl.col("date_col") == test_date) & (pl.col("date_col") == test_date)
        result = convert_expr_to_datetime_range(expr, "date_col", coerce_date_to_datetime=False)
        assert result.atomic
        assert result.lower == datetime.combine(test_date, time())
        assert result.upper == datetime.combine(test_date + timedelta(1), time())
        assert result.left is Bound.CLOSED
        assert result.right is Bound.OPEN

    def test_multiple_equalities_contradictory(self):
        """Test multiple contradictory equality constraints."""
        # Different equality values - contradiction
        expr = (pl.col("date_col") == date(2023, 5, 15)) & (pl.col("date_col") == date(2023, 5, 16))
        result = convert_expr_to_datetime_range(expr, "date_col")
        assert result.empty
        result = convert_expr_to_datetime_range(expr, "date_col", coerce_date_to_datetime=False)
        assert result.empty

    def test_column_not_in_expression(self):
        """Test expression that doesn't mention the column at all."""
        # No mention of date_col
        expr = (pl.col("other_col1") == "value1") | (pl.col("other_col2") == "value2")
        result = convert_expr_to_datetime_range(expr, "date_col")
        assert result.atomic
        assert result.lower == -portion.inf
        assert result.upper == portion.inf

    def test_column_with_other_columns(self):
        """Test where the column appears alongside others."""
        # date_col appears with other columns
        expr = (pl.col("other_col") == "some_value") & (pl.col("date_col") >= date(2023, 5, 15))
        result = convert_expr_to_datetime_range(expr, "date_col")
        assert result.atomic
        assert result.lower == datetime.combine(date(2023, 5, 15), time())
        assert result.upper == portion.inf
        assert result.left is Bound.CLOSED

    def test_is_null(self):
        """Test 'is_null' operator for NULL checks."""
        expr = pl.col("date_col").is_null()
        result = convert_expr_to_datetime_range(expr, "date_col")
        assert result.atomic
        assert result.lower == -portion.inf
        assert result.upper == portion.inf

    def test_datetime_comparisons_with_timezone(self):
        """Test comparisons with timezone-aware datetimes."""
        # Create timezone-aware datetimes
        dt1 = datetime(2023, 5, 15, 9, 0, tzinfo=timezone.utc)
        dt2 = datetime(2023, 5, 15, 17, 0, tzinfo=timezone.utc)

        expr = (pl.col("timestamp") >= dt1) & (pl.col("timestamp") <= dt2)
        result = convert_expr_to_datetime_range(expr, "timestamp")
        assert result.atomic
        assert result.lower == dt1.replace(tzinfo=None)
        assert result.upper == dt2.replace(tzinfo=None)
        assert result.left is Bound.CLOSED
        assert result.right is Bound.CLOSED

    def test_complex_or_conditions(self):
        """Test complex OR conditions with overlapping ranges."""
        # Two partially overlapping date ranges
        expr1 = (pl.col("date_col") >= date(2023, 1, 1)) & (pl.col("date_col") < date(2023, 6, 1))
        expr2 = (pl.col("date_col") >= date(2023, 5, 1)) & (pl.col("date_col") < date(2023, 12, 31))
        expr = expr1 | expr2
        result = convert_expr_to_datetime_range(expr, "date_col")
        # Combined range should be from Jan 1 to Dec 31
        assert result.atomic
        assert result.lower == datetime.combine(date(2023, 1, 1), time())
        assert result.upper == datetime.combine(date(2023, 12, 31), time())
        assert result.left is Bound.CLOSED
        assert result.right is Bound.OPEN

    def test_disjoint_ranges(self):
        """Test OR conditions with disjoint ranges."""
        # Two non-overlapping date ranges
        expr1 = (pl.col("date_col") >= date(2023, 1, 1)) & (pl.col("date_col") < date(2023, 3, 1))
        expr2 = (pl.col("date_col") >= date(2023, 6, 1)) & (pl.col("date_col") < date(2023, 9, 1))
        expr = expr1 | expr2
        result = convert_expr_to_datetime_range(expr, "date_col")
        # Combined range should span both (enclosure)
        assert result.atomic
        assert result.lower == datetime.combine(date(2023, 1, 1), time())
        assert result.upper == datetime.combine(date(2023, 9, 1), time())
        assert result.left is Bound.CLOSED
        assert result.right is Bound.OPEN

    def test_datetime_vs_date(self):
        """Test mixing datetime and date objects."""
        date_only = date(2023, 5, 15)
        datetime_val = datetime(2023, 5, 15, 12, 0)

        expr = (pl.col("mixed_col") >= date_only) & (pl.col("mixed_col") < datetime_val)
        result = convert_expr_to_datetime_range(expr, "mixed_col")
        # Should work fine even with mixed types
        assert result.atomic
        assert result.lower == datetime.combine(date_only, time())
        assert result.upper == datetime_val
        assert result.left is Bound.CLOSED
        assert result.right is Bound.OPEN

    def test_datetime_with_multiple_expr(self):
        datetime_val = datetime(2023, 5, 15, 12, 0)
        expr = pl.col("Date").is_between(datetime_val, datetime_val + timedelta(1)) & pl.col("symol").is_in(["A", "B"])
        result = convert_expr_to_datetime_range(expr, "Date")
        assert result.atomic
        assert result.lower == datetime_val
        assert result.upper == datetime_val + timedelta(1)
        assert result.left is Bound.CLOSED
        assert result.right is Bound.CLOSED


class TestDatetimeRangeEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_datetime_microseconds(self):
        """Test handling of microsecond precision."""
        dt1 = datetime(2023, 5, 15, 9, 0, 0, 0)
        dt2 = datetime(2023, 5, 15, 9, 0, 0, 1)  # 1 microsecond later
        # Convert tuple filter to polars expression
        expr = (pl.col("timestamp") > dt1) & (pl.col("timestamp") <= dt2)
        result = convert_expr_to_datetime_range(expr, "timestamp")
        assert result.atomic
        assert result.lower == dt1
        assert result.upper == dt2
        assert result.left is Bound.OPEN
        assert result.right is Bound.CLOSED

    def test_adjacent_date_ranges(self):
        """Test union of adjacent but non-overlapping ranges."""
        # Convert two distinct filter conditions to OR of expressions
        expr1 = (pl.col("date_col") >= date(2023, 1, 1)) & (pl.col("date_col") < date(2023, 2, 1))
        expr2 = (pl.col("date_col") >= date(2023, 2, 1)) & (pl.col("date_col") < date(2023, 3, 1))
        expr = expr1 | expr2

        result = convert_expr_to_datetime_range(expr, "date_col")
        # Should create a continuous range (enclosure)
        assert result.atomic
        assert result.lower == datetime.combine(date(2023, 1, 1), time())
        assert result.upper == datetime.combine(date(2023, 3, 1), time())
        assert result.left is Bound.CLOSED
        assert result.right is Bound.OPEN

    def test_contradiction_in_one_conjunction(self):
        """Test case where one part has a contradiction."""
        # First part is a contradiction, second part is valid
        expr_contradiction = (pl.col("date_col") >= date(2023, 5, 15)) & (pl.col("date_col") < date(2023, 5, 10))
        expr_valid = pl.col("date_col") == date(2023, 6, 1)
        expr = expr_contradiction | expr_valid

        result = convert_expr_to_datetime_range(expr, "date_col")
        # Should ignore the contradictory expression
        assert result.atomic
        assert result.lower == datetime.combine(date(2023, 6, 1), time())
        assert result.upper == datetime.combine(date(2023, 6, 1), time())
        assert result.left is Bound.CLOSED
        assert result.right is Bound.CLOSED

        result = convert_expr_to_datetime_range(expr, "date_col", coerce_date_to_datetime=False)
        # Should ignore the contradictory expression
        assert result.atomic
        assert result.lower == datetime.combine(date(2023, 6, 1), time())
        assert result.upper == datetime.combine(date(2023, 6, 1) + timedelta(1), time())
        assert result.left is Bound.CLOSED
        assert result.right is Bound.OPEN

    def test_in_empty_list(self):
        """Test 'is_in' operator with an empty list."""
        # Convert to is_in with empty list
        expr = pl.col("date_col").is_in([])
        result = convert_expr_to_datetime_range(expr, "date_col")
        assert result.empty

    def test_not_in_combined_with_range(self):
        """Test 'not in' combined with range conditions."""
        # Use ~ operator for NOT with is_in
        expr = (pl.col("date_col") >= date(2023, 1, 1)) & (pl.col("date_col") <= date(2023, 12, 31)) & ~pl.col("date_col").is_in([date(2023, 7, 4)])
        result = convert_expr_to_datetime_range(expr, "date_col")
        # Should return the full range since we can't represent the exclusion
        assert result.atomic
        assert result.lower == datetime.combine(date(2023, 1, 1), time())
        assert result.upper == datetime.combine(date(2023, 12, 31), time())
        assert result.left is Bound.CLOSED
        assert result.right is Bound.CLOSED

    def test_datetime_range_with_complex_expr(self):
        """Test extracting date range from a complex Polars expression."""
        # Create a complex expression with multiple date conditions
        expr = ((pl.col("date") >= date(2023, 1, 1)) & (pl.col("date") < date(2023, 6, 1))) | (
            (pl.col("date") >= date(2023, 8, 1)) & (pl.col("date") < date(2023, 12, 31))
        )

        # Extract the date range using our visitor
        result = convert_expr_to_datetime_range(expr, "date")

        # Verify the result spans both ranges (enclosure)
        assert result.atomic
        assert result.lower == datetime.combine(date(2023, 1, 1), time())
        assert result.upper == datetime.combine(date(2023, 12, 31), time())
        assert result.left is Bound.CLOSED
        assert result.right is Bound.OPEN

    def test_multiple_in_expressions(self):
        """Test multiple 'is_in' expressions combined."""
        # Using multiple is_in expressions with OR
        dates_set1 = [date(2023, 1, 15), date(2023, 2, 15)]
        dates_set2 = [date(2023, 3, 15), date(2023, 4, 15)]

        expr = pl.col("date_col").is_in(dates_set1) | pl.col("date_col").is_in(dates_set2)
        result = convert_expr_to_datetime_range(expr, "date_col", coerce_date_to_datetime=False)

        # Should cover the full range from min to max of both sets (enclosure)
        assert result.atomic
        assert result.lower == datetime.combine(min(dates_set1 + dates_set2), time())
        assert result.upper == datetime.combine(max(dates_set1 + dates_set2) + timedelta(1), time())
        assert result.left is Bound.CLOSED
        assert result.right is Bound.OPEN

    def test_nested_expressions(self):
        """Test with deeply nested expressions."""
        # Complex nested expression
        expr = (pl.col("date_col") >= date(2023, 1, 1)) & (
            (pl.col("date_col") < date(2023, 3, 1)) | ((pl.col("date_col") >= date(2023, 6, 1)) & (pl.col("date_col") < date(2023, 9, 1)))
        )

        # This should extract either:
        # - dates from Jan 1 to Mar 1 (exclusive), or
        # - dates from Jun 1 to Sep 1 (exclusive)
        result = convert_expr_to_datetime_range(expr, "date_col")

        # The overall range covers from Jan 1 to Sep 1 (enclosure)
        assert result.atomic
        assert result.lower == datetime.combine(date(2023, 1, 1), time())
        assert result.upper == datetime.combine(date(2023, 9, 1), time())
        assert result.left is Bound.CLOSED
        assert result.right is Bound.OPEN


class TestRangeVisitor:
    """Test suite for the RangeVisitor class."""

    def test_range_visitor_copy(self):
        """Test the RangeVisitor class with a simple expression."""
        # Create a simple expression
        constraints_list = [None, portion.empty(), Interval.from_atomic(Bound.CLOSED, date(2023, 1, 1), date(2023, 1, 2), Bound.OPEN)]

        for constraints in constraints_list:
            visitor = IntervalVisitor(
                "date_col",
                None,
                None,
            )
            if constraints is not None:
                visitor.constraints = constraints
            new_visitor = visitor.copy()
            assert new_visitor is not visitor
            assert new_visitor.target_column == visitor.target_column
            assert new_visitor.constraints == visitor.constraints


class TestCastNodeVisitor:
    """Test suite for the visit_cast method in IntervalVisitor."""

    def test_boolean_cast_processes_input_node(self):
        """Test that Boolean casts process their input node."""
        # Create a comparison and cast to Boolean
        expr = (pl.col("value_col") >= 10).cast(pl.Boolean)
        result = convert_expr_to_range(expr, "value_col")

        # Should extract constraint from input node
        assert result.atomic
        assert result.lower == 10
        assert result.upper == portion.inf
        assert result.left is Bound.CLOSED

    def test_non_boolean_cast_ignores_input_node(self):
        """Test that non-Boolean casts ignore their input node."""
        # Create a comparison and cast to non-Boolean
        expr = (pl.col("value_col") >= 10).cast(pl.Utf8)
        result = convert_expr_to_range(expr, "value_col")

        # Should not extract constraint
        assert result.atomic
        assert result.lower == -portion.inf
        assert result.upper == portion.inf

    def test_target_column_check_in_cast(self):
        """Test the target column check in visit_cast."""
        # Cast involving non-target column
        expr = (pl.col("other_col") >= 10).cast(pl.Boolean)
        result = convert_expr_to_range(expr, "value_col")

        # Should not extract constraint for different column
        assert result.atomic
        assert result.lower == -portion.inf
        assert result.upper == portion.inf

    def test_nested_boolean_casts_process_all_levels(self):
        """Test that nested Boolean casts process all levels."""
        # Multiple levels of Boolean casts
        expr = ((pl.col("value_col") >= 10).cast(pl.Boolean)).cast(pl.Boolean)
        result = convert_expr_to_range(expr, "value_col")

        # Should extract constraint from innermost level
        assert result.atomic
        assert result.lower == 10
        assert result.upper == portion.inf

    def test_boolean_cast_with_complex_input(self):
        """Test Boolean cast with complex input expression."""
        # Complex input with AND condition
        expr = ((pl.col("value_col") >= 10) & (pl.col("value_col") < 100)).cast(pl.Boolean)
        result = convert_expr_to_range(expr, "value_col")

        # Should extract full range from complex input
        assert result.atomic
        assert result.lower == 10
        assert result.upper == 100
        assert result.left is Bound.CLOSED
        assert result.right is Bound.OPEN

    def test_boolean_cast_in_conjunction(self):
        """Test Boolean cast as part of a conjunction."""
        # Boolean cast in conjunction with another expression
        expr = ((pl.col("value_col") >= 10).cast(pl.Boolean)) & (pl.col("value_col") < 100)
        result = convert_expr_to_range(expr, "value_col")

        # Should combine constraints from both parts
        assert result.atomic
        assert result.lower == 10
        assert result.upper == 100
        assert result.left is Bound.CLOSED
        assert result.right is Bound.OPEN

    def test_cast_with_function(self):
        """Test Boolean cast with function expressions."""
        # Function result cast to Boolean
        expr = (pl.col("value_col").is_in([10, 20, 30])).cast(pl.Boolean)
        result = convert_expr_to_range(expr, "value_col")

        # Should extract values from function
        assert not result.atomic  # Multiple discrete values
        assert 10 in result
        assert 20 in result
        assert 30 in result
        assert 15 not in result  # Point between values should not be included

    def test_cast_with_ternary(self):
        """Test Boolean cast with ternary expression."""
        # Ternary expression cast to Boolean
        expr = (
            pl.when(pl.col("condition"))
            .then((pl.col("value_col") >= 10) & (pl.col("value_col") < 50))
            .otherwise((pl.col("value_col") >= 100) & (pl.col("value_col") < 150))
        ).cast(pl.Boolean)

        result = convert_expr_to_range(expr, "value_col")

        # Should extract constraints from both branches
        # Note: The actual behavior here depends on how the visitor handles ternaries
        # It will likely be a joined interval covering both ranges
        assert 20 in result  # In first range
        assert 120 in result  # In second range
        # Don't check if value between ranges is included since that depends on implementation

    def test_boolean_cast_with_datetime(self):
        """Test Boolean cast with datetime expressions."""
        # Datetime comparison cast to Boolean
        start_date = date(2023, 1, 1)
        end_date = date(2023, 12, 31)
        expr = ((pl.col("date_col") >= start_date) & (pl.col("date_col") < end_date)).cast(pl.Boolean)
        result = convert_expr_to_datetime_range(expr, "date_col")

        # Should extract the date range
        assert result.atomic
        assert result.lower == datetime.combine(start_date, time())
        assert result.upper == datetime.combine(end_date, time())
        assert result.left is Bound.CLOSED
        assert result.right is Bound.OPEN

    def test_mixed_boolean_casts_in_expr(self):
        """Test mixing Boolean and non-Boolean casts."""
        # Mix of Boolean and non-Boolean casts in expression
        expr = ((pl.col("value_col") >= 10).cast(pl.Boolean)) & ((pl.col("value_col") < 100).cast(pl.Utf8))
        result = convert_expr_to_range(expr, "value_col")

        # Should extract constraint from the Boolean cast only
        assert result.atomic
        assert result.lower == 10
        assert result.upper == portion.inf
        assert result.left is Bound.CLOSED

    def test_boolean_cast_column_directly(self):
        """Test casting the column itself to Boolean."""
        # Direct cast of column to Boolean
        expr = pl.col("value_col").cast(pl.Boolean)
        result = convert_expr_to_range(expr, "value_col")

        # No constraints should be added since there's no comparison
        assert result.atomic
        assert result.lower == -portion.inf
        assert result.upper == portion.inf


class TestCastWithDatetimeRanges:
    """Test cast node with datetime range extraction."""

    @pytest.mark.parametrize("strict", [True, False])
    def test_cast_date_to_datetime_with_comparison(self, strict):
        """Test casting a date column to datetime before comparison."""
        start_date = date(2023, 1, 1)
        end_datetime = datetime(2023, 12, 31, 12, 0)

        # Cast date column to datetime, then compare
        expr = (pl.col("date_col") >= start_date) & (pl.col("date_col").cast(pl.Datetime, strict=strict) <= end_datetime)
        result = convert_expr_to_datetime_range(expr, "date_col")

        assert result.atomic
        assert result.lower == datetime.combine(start_date, time())
        assert result.upper == end_datetime
        assert result.left is Bound.CLOSED
        assert result.right is Bound.CLOSED

    def test_cast_with_time_unit_and_timezone(self):
        """Test casting with time unit and timezone specifications."""
        # Cast with specific time unit and timezone
        expr = pl.col("date_col").cast(pl.Datetime(time_unit="ms", time_zone="UTC")) >= datetime(2023, 1, 1, tzinfo=timezone.utc)
        result = convert_expr_to_datetime_range(expr, "date_col")
        assert result.atomic
        assert result.lower == datetime(2023, 1, 1)
        assert result.upper == portion.inf

    def test_boolean_cast_of_datetime_comparison(self):
        """Test Boolean cast of a datetime comparison."""
        dt = datetime(2023, 5, 15, 12, 0)

        # Cast a datetime comparison to Boolean
        expr = (pl.col("dt_col") == dt).cast(pl.Boolean)
        result = convert_expr_to_datetime_range(expr, "dt_col", coerce_date_to_datetime=True)

        # Should extract the equality constraint
        assert result.atomic
        assert result.lower == dt
        assert result.upper == dt
        assert result.left is Bound.CLOSED
        assert result.right is Bound.CLOSED

    def test_cast_boolean_to_datetime(self):
        """Test casting a Boolean to datetime (not meaningful but should be handled)."""
        expr = pl.col("bool_col").cast(pl.Datetime)
        result = convert_expr_to_datetime_range(expr, "bool_col")

        # No meaningful range can be extracted
        assert result.atomic
        assert result.lower == -portion.inf
        assert result.upper == portion.inf

    def test_nested_cast_to_boolean(self):
        expr = (
            ((pl.col("a").is_in([1, 5, 2])) & (pl.col("a").is_in([5, 2]).not_()))
            .cast(pl.Boolean)
            .and_(((pl.col("date") >= date(2023, 10, 1)) & (pl.col("date") <= date(2023, 10, 3))).cast(pl.Boolean))
        )
        result = convert_expr_to_datetime_range(expr, "date", get_enclosure=False)
        expected = portion.closed(datetime(2023, 10, 1), datetime(2023, 10, 3))
        assert result == expected


class TestRangeExtractorAliasHandling:
    """Test suite for alias handling in range extraction."""

    def test_alias_simple_equality(self):
        """Test range extraction with aliased column equality."""
        test_value = 42
        # Create an expression with alias - the key bug scenario
        expr = pl.col("original_col").alias("renamed_col") == test_value

        # This should work by looking through the alias to find the original column
        result = convert_expr_to_range(expr, "original_col")

        # Should be a singleton interval
        assert result.atomic
        assert result.lower == test_value
        assert result.upper == test_value
        assert result.left is Bound.CLOSED
        assert result.right is Bound.CLOSED

    def test_alias_with_range_constraints(self):
        """Test range extraction with aliased column range constraints."""
        start_value = 10
        end_value = 100

        # Expression with aliased column in range constraints
        expr = (pl.col("value_col").alias("renamed_value") >= start_value) & (pl.col("value_col").alias("renamed_value") < end_value)
        result = convert_expr_to_range(expr, "value_col")

        # Should extract the proper range
        assert result.atomic
        assert result.lower == start_value
        assert result.upper == end_value
        assert result.left is Bound.CLOSED
        assert result.right is Bound.OPEN

    def test_alias_with_in_function(self):
        """Test range extraction with aliased column in IS_IN function."""
        values = [10, 20, 30]

        # Expression with aliased column in IS_IN
        expr = pl.col("test_col").alias("test_alias").is_in(values)
        result = convert_expr_to_range(expr, "test_col")

        # Should create union of point intervals
        expected_intervals = [portion.singleton(v) for v in values]
        expected = expected_intervals[0]
        for interval in expected_intervals[1:]:
            expected = expected | interval

        assert result == expected

    def test_alias_with_between_function(self):
        """Test range extraction with aliased column in IS_BETWEEN function."""
        lower = 5
        upper = 15

        # Expression with aliased column in IS_BETWEEN
        expr = pl.col("range_col").alias("range_alias").is_between(lower, upper)
        result = convert_expr_to_range(expr, "range_col")

        # Should be a closed interval
        assert result.atomic
        assert result.lower == lower
        assert result.upper == upper
        assert result.left is Bound.CLOSED
        assert result.right is Bound.CLOSED

    def test_alias_datetime_range_extraction(self):
        """Test datetime range extraction with aliased columns."""
        start_date = date(2023, 1, 1)
        end_date = date(2023, 12, 31)

        # Expression with aliased datetime column
        expr = (pl.col("date_col").alias("date_alias") >= start_date) & (pl.col("date_col").alias("date_alias") <= end_date)
        result = convert_expr_to_datetime_range(expr, "date_col")

        # Should extract proper datetime range
        assert result.atomic
        assert result.lower == datetime.combine(start_date, time())
        assert result.upper == datetime.combine(end_date, time())
        assert result.left is Bound.CLOSED
        assert result.right is Bound.CLOSED

    def test_complex_nested_alias_cast_combinations(self):
        """Test complex combinations of nested aliases and casts."""
        test_value = 42

        # Test cast -> alias -> alias
        expr1 = pl.col("original").cast(pl.Int64).alias("first").alias("second") == test_value
        result1 = convert_expr_to_range(expr1, "original")
        assert result1.atomic
        assert result1.lower == test_value
        assert result1.upper == test_value

        # Test alias -> cast -> alias
        expr2 = pl.col("original").alias("first").cast(pl.Int64).alias("second") == test_value
        result2 = convert_expr_to_range(expr2, "original")
        assert result2.atomic
        assert result2.lower == test_value
        assert result2.upper == test_value

        # Test with IS_BETWEEN and complex nesting
        start_val, end_val = 10, 20
        expr3 = pl.col("value").cast(pl.Int64).alias("v1").alias("v2").is_between(start_val, end_val)
        result3 = convert_expr_to_range(expr3, "value")
        assert result3.atomic
        assert result3.lower == start_val
        assert result3.upper == end_val


def _test_expression_with_df(expr, index_col, test_dates, expected_indices):
    """Helper function to test polars expressions against a test dataframe.

    Args:
        expr: The polars expression to test
        index_col: The name of the date column
        test_dates: List of dates to test
        expected_indices: Indices of test_dates that should pass the filter
    """
    test_df = pl.DataFrame({index_col: test_dates})
    if expr is None:
        # If expression is None, we should get all rows
        filtered_df = test_df
    else:
        # Apply the expression
        filtered_df = test_df.filter(expr)

    # Get the filtered dates and compare with expected
    filtered_dates = filtered_df[index_col].to_list()
    expected_dates = [test_dates[i] for i in expected_indices]

    assert set(filtered_dates) == set(expected_dates)


class TestLookbackInterval:
    def test_simple_closed_interval(self):
        """Test lookback on a basic closed interval."""
        start = datetime(2023, 1, 5)
        end = datetime(2023, 1, 10)
        interval = portion.closed(start, end)
        lookback = timedelta(days=1)

        result = _lookback_interval(interval, lookback)
        expected = portion.closed(datetime(2023, 1, 4), end)
        assert result == expected

    def test_open_interval(self):
        """Test lookback on an open interval."""
        start = datetime(2023, 1, 5)
        end = datetime(2023, 1, 10)
        interval = portion.open(start, end)
        lookback = timedelta(days=2)

        result = _lookback_interval(interval, lookback)
        expected = portion.open(lower=datetime(2023, 1, 3), upper=end)
        assert result == expected

    def test_infinite_lower_bound(self):
        """Test lookback on interval with infinite lower bound."""
        end = datetime(2023, 1, 10)
        interval = portion.openclosed(lower=-portion.inf, upper=end)
        lookback = timedelta(days=1)

        result = _lookback_interval(interval, lookback)
        assert result == interval

    def test_compound_interval(self):
        """Test lookback on a compound interval (union of intervals)."""
        interval1 = portion.closed(datetime(2023, 1, 1), datetime(2023, 1, 3))
        interval2 = portion.closed(datetime(2023, 1, 5), datetime(2023, 1, 7))
        compound_interval = interval1 | interval2
        lookback = timedelta(days=1)

        result = _lookback_interval(compound_interval, lookback)
        expected = portion.closed(datetime(2022, 12, 31), datetime(2023, 1, 3)) | portion.closed(datetime(2023, 1, 4), datetime(2023, 1, 7))
        assert result == expected

    def test_date_bounds(self):
        """Test lookback on interval with date (not datetime) bounds."""
        start = date(2023, 1, 5)
        end = date(2023, 1, 10)
        interval = portion.closed(start, end)
        lookback = timedelta(days=3)

        result = _lookback_interval(interval, lookback)
        expected = portion.closed(date(2023, 1, 2), end)
        assert result == expected


class TestLookaheadInterval:
    def test_simple_closed_interval(self):
        """Test lookahead on a basic closed interval."""
        start = datetime(2023, 1, 5)
        end = datetime(2023, 1, 10)
        interval = portion.closed(start, end)
        lookahead = timedelta(days=1)

        result = _lookahead_interval(interval, lookahead)
        expected = portion.closed(start, datetime(2023, 1, 11))
        assert result == expected

    def test_open_interval(self):
        """Test lookahead on an open interval."""
        start = datetime(2023, 1, 5)
        end = datetime(2023, 1, 10)
        interval = portion.open(start, end)
        lookahead = timedelta(days=2)

        result = _lookahead_interval(interval, lookahead)
        expected = portion.open(lower=start, upper=datetime(2023, 1, 12))
        assert result == expected

    def test_infinite_upper_bound(self):
        """Test lookahead on interval with infinite upper bound."""
        start = datetime(2023, 1, 10)
        interval = portion.closedopen(lower=start, upper=portion.inf)
        lookahead = timedelta(days=1)

        result = _lookahead_interval(interval, lookahead)
        assert result == interval

    def test_compound_interval(self):
        """Test lookahead on a compound interval (union of intervals)."""
        interval1 = portion.closed(datetime(2023, 1, 1), datetime(2023, 1, 3))
        interval2 = portion.closed(datetime(2023, 1, 5), datetime(2023, 1, 7))
        compound_interval = interval1 | interval2
        lookahead = timedelta(days=1)

        result = _lookahead_interval(compound_interval, lookahead)
        expected = portion.closed(datetime(2023, 1, 1), datetime(2023, 1, 4)) | portion.closed(datetime(2023, 1, 5), datetime(2023, 1, 8))
        assert result == expected

    def test_date_bounds(self):
        """Test lookahead on interval with date (not datetime) bounds."""
        start = date(2023, 1, 5)
        end = date(2023, 1, 10)
        interval = portion.closed(start, end)
        lookahead = timedelta(days=3)

        result = _lookahead_interval(interval, lookahead)
        expected = portion.closed(start, date(2023, 1, 13))
        assert result == expected


class TestExtendToFullDates:
    def test_datetime_bounds(self):
        """Test extending intervals with datetime bounds to full dates."""
        start = datetime(2023, 1, 5, 12, 30, 45)
        end = datetime(2023, 1, 10, 14, 20, 15)
        interval = portion.closed(start, end)

        result = _extend_to_full_dates(interval)
        expected = portion.closed(date(2023, 1, 5), date(2023, 1, 10))
        assert result == expected

    def test_date_bounds(self):
        """Test extending intervals that already have date bounds."""
        start = date(2023, 1, 5)
        end = date(2023, 1, 10)
        interval = portion.closed(start, end)

        result = _extend_to_full_dates(interval)
        assert result == interval

    def test_infinite_bounds(self):
        """Test extending intervals with infinite bounds."""
        interval = portion.open(-portion.inf, portion.inf)

        result = _extend_to_full_dates(interval)
        assert result == interval

    def test_compound_interval(self):
        """Test extending compound intervals to full dates."""
        interval1 = portion.closed(datetime(2023, 1, 1, 8, 30), datetime(2023, 1, 3, 16, 45))
        interval2 = portion.closed(datetime(2023, 1, 5, 9, 15), datetime(2023, 1, 7, 17, 30))
        compound_interval = interval1 | interval2

        result = _extend_to_full_dates(compound_interval)
        expected = portion.closed(date(2023, 1, 1), date(2023, 1, 3)) | portion.closed(date(2023, 1, 5), date(2023, 1, 7))
        assert result == expected


class TestConvertAtomicIntervalToPolarsExpr:
    def test_fully_open_interval(self):
        """Test converting open interval to polars expression."""
        start = datetime(2023, 1, 5)
        end = datetime(2023, 1, 10)
        interval = portion.open(start, end)
        index_col = "date"

        result = _convert_atomic_interval_to_polars_expr(interval, index_col)
        test_dates = [
            datetime(2023, 1, 4),
            datetime(2023, 1, 5),
            datetime(2023, 1, 7),
            datetime(2023, 1, 10),
            datetime(2023, 1, 11),
        ]
        _test_expression_with_df(result, index_col, test_dates, [2])

    def test_fully_closed_interval(self):
        """Test converting closed interval to polars expression."""
        start = datetime(2023, 1, 5)
        end = datetime(2023, 1, 10)
        interval = portion.closed(start, end)
        index_col = "date"

        result = _convert_atomic_interval_to_polars_expr(interval, index_col)
        test_dates = [
            datetime(2023, 1, 4),
            datetime(2023, 1, 5),
            datetime(2023, 1, 7),
            datetime(2023, 1, 10),
            datetime(2023, 1, 11),
        ]
        _test_expression_with_df(result, index_col, test_dates, [1, 2, 3])

    def test_left_open_right_closed_interval(self):
        """Test converting left-open, right-closed interval to polars expression."""
        start = datetime(2023, 1, 5)
        end = datetime(2023, 1, 10)
        interval = portion.openclosed(start, end)
        index_col = "date"

        result = _convert_atomic_interval_to_polars_expr(interval, index_col)
        test_dates = [
            datetime(2023, 1, 4),
            datetime(2023, 1, 5),
            datetime(2023, 1, 7),
            datetime(2023, 1, 10),
            datetime(2023, 1, 11),
        ]
        _test_expression_with_df(result, index_col, test_dates, [2, 3])

    def test_left_closed_right_open_interval(self):
        """Test converting left-closed, right-open interval to polars expression."""
        start = datetime(2023, 1, 5)
        end = datetime(2023, 1, 10)
        interval = portion.closedopen(start, end)
        index_col = "date"

        result = _convert_atomic_interval_to_polars_expr(interval, index_col)
        test_dates = [
            datetime(2023, 1, 4),
            datetime(2023, 1, 5),
            datetime(2023, 1, 7),
            datetime(2023, 1, 10),
            datetime(2023, 1, 11),
        ]
        _test_expression_with_df(result, index_col, test_dates, [1, 2])

    def test_infinite_lower_bound(self):
        """Test converting interval with infinite lower bound to polars expression."""
        end = datetime(2023, 1, 10)
        interval = portion.openclosed(lower=-portion.inf, upper=end)
        index_col = "date"

        result = _convert_atomic_interval_to_polars_expr(interval, index_col)
        test_dates = [
            datetime(2022, 1, 1),
            datetime(2023, 1, 4),
            datetime(2023, 1, 10),
            datetime(2023, 1, 11),
        ]
        _test_expression_with_df(result, index_col, test_dates, [0, 1, 2])

    def test_infinite_upper_bound(self):
        """Test converting interval with infinite upper bound to polars expression."""
        start = datetime(2023, 1, 5)
        interval = portion.closedopen(lower=start, upper=portion.inf)
        index_col = "date"

        result = _convert_atomic_interval_to_polars_expr(interval, index_col)
        test_dates = [
            datetime(2023, 1, 4),
            datetime(2023, 1, 5),
            datetime(2023, 1, 10),
            datetime(2024, 1, 1),
        ]
        _test_expression_with_df(result, index_col, test_dates, [1, 2, 3])

    def test_full_infinite_interval(self):
        """Test converting fully infinite interval to polars expression."""
        interval = portion.open(-portion.inf, portion.inf)
        index_col = "date"

        result = _convert_atomic_interval_to_polars_expr(interval, index_col)
        test_dates = [datetime(2022, 1, 1), datetime(2023, 1, 5), datetime(2023, 1, 10), datetime(2024, 1, 1)]

        assert result is None
        _test_expression_with_df(result, index_col, test_dates, [0, 1, 2, 3])

    def test_singleton_interval(self):
        """Test converting singleton interval to polars expression."""
        value = datetime(2023, 1, 5)
        interval = portion.singleton(value)
        index_col = "date"

        result = _convert_atomic_interval_to_polars_expr(interval, index_col)
        test_dates = [
            datetime(2023, 1, 4),
            datetime(2023, 1, 5),
            datetime(2023, 1, 6),
        ]
        _test_expression_with_df(result, index_col, test_dates, [1])


class TestConvertIntervalToPolarsExpr:
    def test_simple_atomic_interval(self):
        """Test converting simple atomic interval to polars expression."""
        start = datetime(2023, 1, 5)
        end = datetime(2023, 1, 10)
        interval = portion.closed(start, end)
        index_col = "date"

        result = _convert_interval_to_polars_expr(interval, index_col)
        test_dates = [
            datetime(2023, 1, 4),
            datetime(2023, 1, 5),
            datetime(2023, 1, 7),
            datetime(2023, 1, 10),
            datetime(2023, 1, 11),
        ]
        _test_expression_with_df(result, index_col, test_dates, [1, 2, 3])

    def test_compound_interval(self):
        """Test converting compound interval to polars expression."""
        interval1 = portion.closed(datetime(2023, 1, 1), datetime(2023, 1, 3))
        interval2 = portion.closed(datetime(2023, 1, 5), datetime(2023, 1, 7))
        compound_interval = interval1 | interval2
        index_col = "date"

        result = _convert_interval_to_polars_expr(compound_interval, index_col)
        test_dates = [
            datetime(2022, 12, 31),
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
            datetime(2023, 1, 3),
            datetime(2023, 1, 4),
            datetime(2023, 1, 5),
            datetime(2023, 1, 6),
            datetime(2023, 1, 7),
            datetime(2023, 1, 8),
        ]
        _test_expression_with_df(result, index_col, test_dates, [1, 2, 3, 5, 6, 7])

    def test_empty_interval(self):
        """Test converting empty interval returns False (not None)."""
        interval = portion.empty()
        index_col = "date"

        result = _convert_interval_to_polars_expr(interval, index_col)
        assert result is False, "Empty interval should return False, not None"

    def test_universe_interval(self):
        """Test that universe interval returns None (no restriction)."""
        universe = portion.closed(-portion.inf, portion.inf)
        assert _convert_interval_to_polars_expr(universe, "date") is None

    def test_mixed_bounds_compound(self):
        """Test converting compound interval with mixed bound types to polars expression."""
        interval1 = portion.openclosed(datetime(2023, 1, 1), datetime(2023, 1, 3))
        interval2 = portion.closedopen(datetime(2023, 1, 5), datetime(2023, 1, 7))
        compound_interval = interval1 | interval2
        index_col = "date"

        result = _convert_interval_to_polars_expr(compound_interval, index_col)
        test_dates = [
            datetime(2022, 12, 31),
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
            datetime(2023, 1, 3),
            datetime(2023, 1, 4),
            datetime(2023, 1, 5),
            datetime(2023, 1, 6),
            datetime(2023, 1, 7),
            datetime(2023, 1, 8),
        ]
        _test_expression_with_df(result, index_col, test_dates, [2, 3, 5, 6])


class TestPartitionPruningLogic:
    """
    Regression tests for the closed="left" bug: verify that partition specs
    generated by _partition_specs are correctly trimmed by intersecting
    with the extracted temporal interval.

    These tests exercise the exact code path used by execute_on_ray's
    source_generator without needing Ray or the Polars optimizer.
    """

    def test_closed_left_removes_upper_bound_partition(self):
        """
        closed="left" produces [Jan 2, Jan 10).  _partition_specs includes
        a partition for Jan 10, but [Jan 10, Jan 11) ∩ [Jan 2, Jan 10) = ∅
        so it must be removed entirely.
        """
        from polars_io_tools.io_sources.lazy_ray import _partition_specs, _trim_partition_specs

        start = date(2024, 1, 2)
        end = date(2024, 1, 10)

        predicate = pl.col("ts").is_between(start, end, closed="left")
        date_interval = convert_expr_to_datetime_range(predicate, "ts", get_enclosure=False)
        assert not date_interval.empty

        specs = _partition_specs(date_interval.lower, date_interval.upper, "daily")
        assert len(specs) == 9  # Jan 2..10 inclusive before trimming

        trimmed = _trim_partition_specs(specs, date_interval, pl.Date)

        assert len(trimmed) == 8, f"Expected 8 partitions (Jan 2-9), got {len(trimmed)}. The Jan 10 partition should have been removed."
        trimmed_starts = {s for s, _ in trimmed}
        assert datetime(2024, 1, 10) not in trimmed_starts

    def test_closed_both_keeps_all_partitions(self):
        """closed='both' (default) must NOT remove any partitions."""
        from polars_io_tools.io_sources.lazy_ray import _partition_specs, _trim_partition_specs

        start = date(2024, 1, 2)
        end = date(2024, 1, 10)

        predicate = pl.col("ts").is_between(start, end)
        date_interval = convert_expr_to_datetime_range(predicate, "ts", get_enclosure=False)
        specs = _partition_specs(date_interval.lower, date_interval.upper, "daily")

        trimmed = _trim_partition_specs(specs, date_interval, pl.Date)

        assert len(trimmed) == len(specs), f"closed='both' should not remove any partitions, but {len(specs) - len(trimmed)} were removed"

    def test_monthly_both_bounds_trimmed(self):
        """
        Both start and end are trimmed to the intersection with the user's
        interval.  Closed upper bounds are extended via _extend_interval so
        that _execute_partition's `col < end` still includes the boundary value.
        """
        from polars_io_tools.io_sources.lazy_ray import _partition_specs, _trim_partition_specs

        # User wants Jan 15 through Mar 10 (Datetime column, us precision)
        start = datetime(2024, 1, 15)
        end = datetime(2024, 3, 10)
        col_type = pl.Datetime("us")

        predicate = pl.col("ts").is_between(start, end)
        date_interval = convert_expr_to_datetime_range(predicate, "ts", get_enclosure=False)
        specs = _partition_specs(date_interval.lower, date_interval.upper, "monthly")

        # Raw specs: [Jan 1, Feb 1), [Feb 1, Mar 1), [Mar 1, Apr 1)
        assert len(specs) == 3

        trimmed = _trim_partition_specs(specs, date_interval, col_type)

        assert len(trimmed) == 3  # all three months have overlap
        # First partition start trimmed: Jan 15, not Jan 1
        assert trimmed[0][0] == datetime(2024, 1, 15)
        # First partition end: Feb 1 (intersection is [Jan 15, Feb 1), open upper → kept)
        assert trimmed[0][1] == datetime(2024, 2, 1)
        # Middle partition unchanged (intersection is [Feb 1, Mar 1), open upper)
        assert trimmed[1] == (datetime(2024, 2, 1), datetime(2024, 3, 1))
        # Last partition: intersection is [Mar 1, Mar 10] (closed upper from user's filter)
        # → end extended by _extend_interval: +1µs for Datetime("us")
        assert trimmed[2][0] == datetime(2024, 3, 1)
        assert trimmed[2][1] == datetime(2024, 3, 10, 0, 0, 0, 1)
