import datetime
import os
import shutil
import time
from pathlib import PureWindowsPath

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest
from polars.testing import assert_frame_equal

import polars_io_tools as cpl  # noqa
from polars_io_tools.io_sources.lazy_cache_parquet import CacheMode, _get_expected_partitions_df, cache_parquet
from polars_io_tools.tests.helpers.cache_parquet_shared import exercise_daily_cache_parquet


class TestGetExpectedPartitionsDFBehavior:
    """Comprehensive behavioral tests for _get_expected_partitions_df function."""

    @pytest.fixture
    def comprehensive_schema(self):
        """Schema with various data types for realistic testing."""
        return pl.Schema(
            {
                "date": pl.Date,
                "timestamp": pl.Datetime("us"),
                "region": pl.String,
                "country": pl.String,
                "category": pl.String,
                "value": pl.Float64,
                "count": pl.Int64,
                "active": pl.Boolean,
            }
        )

    def test_none_predicate_behavior(self, comprehensive_schema):
        """Test that None predicate always returns empty DataFrame with correct partition schema."""
        # Test with just date column
        result = _get_expected_partitions_df(pred=None, date_column="date", extra_cols=[], time_unit="monthly", schema=comprehensive_schema)

        expected = pl.DataFrame({"date": [None]}, schema={"date": pl.Date})
        # We fill in null values for unrestricted partitions
        assert_frame_equal(result, expected)

        # Test with extra columns
        result_with_extras = _get_expected_partitions_df(
            pred=None, date_column="date", extra_cols=["region", "country"], time_unit="daily", schema=comprehensive_schema
        )

        expected_schema = {"region": pl.String, "country": pl.String, "date": pl.Date}
        expected = pl.DataFrame({"date": [None], "region": [None], "country": [None]}, schema=expected_schema)
        assert_frame_equal(result_with_extras, expected)

    def test_simple_date_equality_monthly(self, comprehensive_schema):
        """Test simple date equality creates single monthly partition."""
        test_date = datetime.date(2024, 3, 15)
        pred = pl.col("date") == test_date

        result = _get_expected_partitions_df(pred=pred, date_column="date", extra_cols=[], time_unit="monthly", schema=comprehensive_schema)

        # Should return one partition for March 2024
        # The exact behavior depends on implementation but should include the target date
        assert result.height >= 1  # At least one partition
        assert "date" in result.columns

        # All dates should be within the same month as test_date
        dates = result["date"].to_list()
        assert dates == [datetime.date(2024, 3, 1)]

    def test_simple_date_equality_daily(self, comprehensive_schema):
        """Test simple date equality creates single daily partition."""
        test_date = datetime.date(2024, 5, 20)
        pred = pl.col("date") == test_date

        result = _get_expected_partitions_df(pred=pred, date_column="date", extra_cols=[], time_unit="daily", schema=comprehensive_schema)

        # Should return exactly one partition for the specific day
        assert result.height >= 1
        dates = result["date"].to_list()
        # Should include the exact date or dates that encompass it
        assert any(d == test_date for d in dates if d is not None)

    def test_date_range_creates_multiple_monthly_partitions(self, comprehensive_schema):
        """Test date range spanning multiple months creates exactly the expected monthly partitions."""
        start_date = datetime.date(2024, 1, 15)
        end_date = datetime.date(2024, 4, 10)

        # Test with is_between - this spans from mid-January to mid-April
        pred = pl.col("date").is_between(start_date, end_date)

        result = _get_expected_partitions_df(pred=pred, date_column="date", extra_cols=[], time_unit="monthly", schema=comprehensive_schema)

        # Should create partitions for January, February, March, and April 2024
        # For monthly partitioning, we expect the first day of each month that contains dates in the range
        expected = pl.DataFrame(
            {
                "date": [
                    datetime.date(2024, 1, 1),  # January 2024 (contains Jan 15)
                    datetime.date(2024, 2, 1),  # February 2024 (entire month in range)
                    datetime.date(2024, 3, 1),  # March 2024 (entire month in range)
                    datetime.date(2024, 4, 1),  # April 2024 (contains Apr 1-10)
                ]
            },
            schema={"date": pl.Date},
        )

        # Sort both to ensure consistent ordering for comparison
        assert_frame_equal(result.sort("date"), expected.sort("date"))

    def test_extra_columns_with_simple_predicates(self, comprehensive_schema):
        """Test extra partition columns with simple predicates."""
        test_date = datetime.date(2024, 7, 4)

        # Test with one extra column
        pred = (pl.col("date") == test_date) & (pl.col("region") == "US")

        result = _get_expected_partitions_df(pred=pred, date_column="date", extra_cols=["region"], time_unit="daily", schema=comprehensive_schema)

        assert "date" in result.columns
        assert "region" in result.columns
        expected = pl.DataFrame(
            {"region": ["US"], "date": [test_date]},
        )
        assert_frame_equal(result, expected)

    def test_multiple_extra_columns_complex_predicate(self, comprehensive_schema):
        """Test multiple extra partition columns with is_in predicates create cartesian product of all combinations."""
        test_date = datetime.date(2024, 8, 12)

        # Fixed HTML encoding: using & instead of &amp;
        pred = (pl.col("date") == test_date) & (pl.col("region").is_in(["US", "EU", "APAC"])) & (pl.col("category").is_in(["A", "B"]))

        result = _get_expected_partitions_df(
            pred=pred, date_column="date", extra_cols=["region", "category"], time_unit="daily", schema=comprehensive_schema
        )

        # Should create cartesian product: 3 regions × 2 categories = 6 combinations
        # All with the same date since it's a single date equality
        expected = pl.DataFrame(
            {
                "region": ["US", "US", "EU", "EU", "APAC", "APAC"],
                "category": ["A", "B", "A", "B", "A", "B"],
                "date": [test_date, test_date, test_date, test_date, test_date, test_date],
            },
            schema={"region": pl.String, "category": pl.String, "date": pl.Date},
        )

        # Sort both for consistent comparison since order might vary
        assert_frame_equal(result.sort(["region", "category"]), expected.sort(["region", "category"]))

    def test_or_conditions_create_multiple_partitions(self, comprehensive_schema):
        """Test OR conditions create multiple separate partitions."""
        date1 = datetime.date(2024, 9, 5)
        date2 = datetime.date(2024, 10, 15)

        # Two separate dates with OR
        pred = (pl.col("date") == date1) | (pl.col("date") == date2)

        result = _get_expected_partitions_df(pred=pred, date_column="date", extra_cols=[], time_unit="monthly", schema=comprehensive_schema)

        dates = set(result["date"].to_list())
        assert dates == set([datetime.date(2024, 9, 1), datetime.date(2024, 10, 1)])

    def test_or_with_extra_columns_creates_all_combinations(self, comprehensive_schema):
        """Test OR conditions with extra columns create exact cartesian product of all valid combinations."""
        test_date = datetime.date(2024, 11, 8)

        # Fixed HTML encoding: using & instead of &amp;
        # This predicate means: date==test_date AND (region==US OR region==EU) AND (category==X OR category==Y)
        pred = (
            (pl.col("date") == test_date)
            & ((pl.col("region") == "US") | (pl.col("region") == "EU"))
            & ((pl.col("category") == "X") | (pl.col("category") == "Y"))
        )

        result = _get_expected_partitions_df(
            pred=pred, date_column="date", extra_cols=["region", "category"], time_unit="daily", schema=comprehensive_schema
        )

        # Should create exactly 4 combinations: cartesian product of 2 regions × 2 categories
        # All with the same date since it's a single date equality
        expected = pl.DataFrame(
            {
                "region": ["US", "US", "EU", "EU"],
                "category": ["X", "Y", "X", "Y"],
                "date": [test_date, test_date, test_date, test_date],
            },
            schema={"region": pl.String, "category": pl.String, "date": pl.Date},
        )

        # Sort both for consistent comparison since order might vary
        assert_frame_equal(result.sort(["region", "category"]), expected.sort(["region", "category"]))

    def test_contradictory_predicates(self, comprehensive_schema):
        """Test contradictory predicates return empty results."""
        # Impossible condition: date can't be two different values simultaneously
        pred = (pl.col("date") == datetime.date(2024, 1, 1)) & (pl.col("date") == datetime.date(2024, 2, 1))

        result = _get_expected_partitions_df(pred=pred, date_column="date", extra_cols=[], time_unit="monthly", schema=comprehensive_schema)

        # Should return empty DataFrame for impossible conditions
        assert result.height == 0
        assert "date" in result.columns

    def test_different_time_units_same_predicate(self, comprehensive_schema):
        """Test same predicate with different time units produces different granularity."""
        date_range_start = datetime.date(2024, 3, 10)
        date_range_end = datetime.date(2024, 5, 20)
        pred = pl.col("date").is_between(date_range_start, date_range_end)

        # Test daily partitioning
        result_daily = _get_expected_partitions_df(pred=pred, date_column="date", extra_cols=[], time_unit="daily", schema=comprehensive_schema)

        # Test monthly partitioning
        result_monthly = _get_expected_partitions_df(pred=pred, date_column="date", extra_cols=[], time_unit="monthly", schema=comprehensive_schema)

        # Test yearly partitioning
        result_yearly = _get_expected_partitions_df(pred=pred, date_column="date", extra_cols=[], time_unit="yearly", schema=comprehensive_schema)

        # Daily should generally have more partitions than monthly
        # Monthly should have more partitions than yearly (unless range is within one year)
        # Daily partitioning should create more granular partitions
        daily_dates = set(result_daily["date"].to_list())
        monthly_dates = set(result_monthly["date"].to_list())

        assert set(pl.date_range(date_range_start, date_range_end, interval="1d", eager=True)) == set(daily_dates)
        assert set(pl.date_range(date_range_start.replace(day=1), date_range_end, interval="1mo", eager=True)) == set(monthly_dates)
        yearly_dates = result_yearly["date"].to_list()
        assert set(pl.date_range(date_range_start.replace(month=1, day=1), date_range_end, interval="1y", eager=True)) == set(yearly_dates)

    def test_open_ended_lower_bound_only(self, comprehensive_schema):
        """Lower-bound-only predicate should yield no finite expected partitions."""
        start_date = datetime.date(2024, 6, 5)
        pred = pl.col("date") >= start_date

        result = _get_expected_partitions_df(pred=pred, date_column="date", extra_cols=[], time_unit="daily", schema=comprehensive_schema)

        assert result.is_empty()
        assert result.columns == ["date"]

    def test_open_ended_upper_bound_only(self, comprehensive_schema):
        """Upper-bound-only predicate should yield no finite expected partitions."""
        end_date = datetime.date(2024, 6, 2)
        pred = pl.col("date") <= end_date

        result = _get_expected_partitions_df(pred=pred, date_column="date", extra_cols=[], time_unit="daily", schema=comprehensive_schema)

        assert result.is_empty()
        assert result.columns == ["date"]

    def test_predicate_with_non_partition_columns_ignored(self, comprehensive_schema):
        """Test that predicates on non-partition columns are handled gracefully."""
        # Predicate that includes both partition and non-partition columns
        pred = (
            (pl.col("date") == datetime.date(2024, 12, 25)) & (pl.col("region") == "US") & (pl.col("value") > 100.0)  # Non-partition column
        )

        result = _get_expected_partitions_df(pred=pred, date_column="date", extra_cols=["region"], time_unit="daily", schema=comprehensive_schema)
        expected = pl.DataFrame({"region": ["US"], "date": [datetime.date(2024, 12, 25)]}, schema={"region": pl.String, "date": pl.Date})
        assert_frame_equal(result, expected)

    def test_complex_nested_or_and_conditions(self, comprehensive_schema):
        """Test complex nested OR and AND conditions create union of partitions from each branch."""
        # Complex predicate: (date1 AND region1) OR (date2 AND region2)
        date1 = datetime.date(2024, 6, 1)
        date2 = datetime.date(2024, 7, 1)

        # Fixed HTML encoding: using & instead of &amp;
        pred = ((pl.col("date") == date1) & (pl.col("region") == "US")) | ((pl.col("date") == date2) & (pl.col("region") == "EU"))

        result = _get_expected_partitions_df(pred=pred, date_column="date", extra_cols=["region"], time_unit="monthly", schema=comprehensive_schema)

        # Should create exactly 2 partitions: one for each OR branch
        # For monthly partitioning, dates become first day of their respective months
        expected = pl.DataFrame(
            {
                "region": ["US", "EU"],
                "date": [datetime.date(2024, 6, 1), datetime.date(2024, 7, 1)],
            },
            schema={"region": pl.String, "date": pl.Date},
        )

        assert_frame_equal(result.sort(["date", "region"]), expected.sort(["date", "region"]))

    def test_timestamp_column_as_date_column(self, comprehensive_schema):
        """Test using timestamp column as the date partition column extracts date portion."""
        test_timestamp = datetime.datetime(2024, 4, 15, 14, 30, 45)
        pred = pl.col("timestamp") == test_timestamp

        result = _get_expected_partitions_df(
            pred=pred,
            date_column="timestamp",  # Using timestamp as date column
            extra_cols=[],
            time_unit="daily",
            schema=comprehensive_schema,
        )

        # Should return the date portion of the timestamp for daily partitioning
        # The timestamp gets converted to just the date part
        expected = pl.DataFrame({"timestamp": [datetime.date(2024, 4, 15)]}, schema={"timestamp": pl.Datetime("us")})

        assert_frame_equal(result, expected)

    def test_large_date_ranges_yearly_partitioning(self, comprehensive_schema):
        """Test function with large date ranges using yearly partitioning creates exact yearly partitions."""
        # Large range spanning multiple years
        start_date = datetime.date(2020, 1, 1)
        end_date = datetime.date(2024, 12, 31)

        # Fixed HTML encoding: using & instead of &amp;
        pred = pl.col("date").is_between(start_date, end_date) & pl.col("region").is_in(["US", "EU"])

        result = _get_expected_partitions_df(
            pred=pred,
            date_column="date",
            extra_cols=["region"],
            time_unit="yearly",  # Use yearly to keep partition count manageable
            schema=comprehensive_schema,
        )

        # Should create partitions for each year (2020-2024) × each region (US, EU) = 10 partitions
        expected_years = [2020, 2021, 2022, 2023, 2024]
        expected_regions = ["US", "EU"]

        expected_data = []
        for year in expected_years:
            for region in expected_regions:
                expected_data.append(
                    {
                        "region": region,
                        "date": datetime.date(year, 1, 1),  # First day of year for yearly partitioning
                    }
                )

        expected = pl.DataFrame(expected_data, schema={"region": pl.String, "date": pl.Date})
        assert_frame_equal(result.sort(["date", "region"]), expected.sort(["date", "region"]))

    def test_is_in_predicate_with_multiple_dates(self, comprehensive_schema):
        """Test is_in predicate with multiple dates creates exact monthly partitions."""
        target_dates = [datetime.date(2024, 2, 14), datetime.date(2024, 3, 17), datetime.date(2024, 4, 22)]

        pred = pl.col("date").is_in(target_dates)

        result = _get_expected_partitions_df(pred=pred, date_column="date", extra_cols=[], time_unit="monthly", schema=comprehensive_schema)

        # Should create partitions for February, March, and April 2024
        # For monthly partitioning, use first day of each month
        expected = pl.DataFrame(
            {
                "date": [
                    datetime.date(2024, 2, 1),  # February 2024
                    datetime.date(2024, 3, 1),  # March 2024
                    datetime.date(2024, 4, 1),  # April 2024
                ]
            },
            schema={"date": pl.Date},
        )

        assert_frame_equal(result.sort("date"), expected.sort("date"))


def test_simple(tmp_path):
    df = pl.DataFrame({"date": [datetime.date(2023, 1, 15)], "value": [42]}).lazy()
    result = df.piot.cache_parquet(cache_path=tmp_path, date_column="date")
    assert isinstance(result, pl.LazyFrame)
    result.collect()

    parquet_file = tmp_path / "monthly" / "2023-01.parquet"
    assert parquet_file.exists()

    # read back metadata embedded in parquet
    meta = pq.read_metadata(parquet_file).metadata
    meta = {k.decode(): v.decode() for k, v in meta.items()}
    assert meta["__piot__time_unit"] == "monthly"
    assert meta["__piot__partition_format"] == "$year-$month"

    read_df = result.collect()
    assert read_df.shape == (1, 2)
    assert read_df["value"].to_list() == [42]


def test_callable_not_invoked_when_cache_satisfies(tmp_path):
    """If the cache fully satisfies reads, a callable source should not be invoked."""
    # First, create cache with known data
    initial_df = pl.DataFrame(
        {
            "date": [datetime.date(2023, 1, 1), datetime.date(2023, 1, 2)],
            "region": ["US", "US"],
            "value": [10, 20],
        }
    ).lazy()

    cached = initial_df.piot.cache_parquet(cache_path=tmp_path, date_column="date", time_unit="daily", extra_partition_cols=["region"])
    cached.collect()

    # Define a callable that should NOT be called if cache suffices
    called = {"count": 0}

    def make_source():
        called["count"] += 1
        return pl.DataFrame(
            {
                "date": [datetime.date(2023, 1, 1), datetime.date(2023, 1, 2)],
                "region": ["US", "US"],
                "value": [10, 20],
            }
        ).lazy()

    # Query that is fully satisfied by the cache (no write required)
    lf = cache_parquet(
        make_source,
        cache_path=tmp_path,
        date_column="date",
        time_unit="daily",
        extra_partition_cols=["region"],
    ).filter(pl.col("region") == "US", pl.col("date").is_between(datetime.date(2023, 1, 1), datetime.date(2023, 1, 2)))

    out = lf.collect()

    # Ensure callable was never invoked
    assert called["count"] == 0
    # Ensure data was read from cache and matches initial values
    assert out.shape == (2, 3)
    assert set(out["value"].to_list()) == {10, 20}

    lf = cache_parquet(
        make_source,
        cache_path=tmp_path,
        date_column="date",
        time_unit="daily",
        extra_partition_cols=["region"],
    ).filter(
        pl.col("region") == "US",
    )

    out = lf.collect()

    # No date restriction, so lazyframe IS invoked
    assert called["count"] == 1
    # Ensure data was read from cache and matches initial values
    assert out.shape == (2, 3)
    assert set(out["value"].to_list()) == {10, 20}


def test_complex(tmp_path):
    lf = pl.LazyFrame(
        {
            "date": (
                x := pl.datetime_range(
                    # note the use of datetime columns
                    datetime.datetime(2023, 1, 1),
                    datetime.datetime(2024, 12, 31),
                    interval="1d",
                    eager=True,
                )
            ),
            "feature": np.random.rand(x.shape[0]),
            "quantity": range(len(x)),
            "price": 100 + np.arange(x.shape[0]),
        }
    )

    result = (
        lf.piot.cache_parquet(cache_path=str(tmp_path), date_column="date")
        .filter((pl.col("date") < datetime.datetime(2024, 12, 15)) & (pl.col("date") > datetime.datetime(2024, 10, 15)))
        .filter(pl.col("quantity") % 2 != 0)
        .select(["date", "feature", "price"])
    )

    read_df = result.collect(engine="streaming")
    assert read_df.shape == (30, 3)

    assert len(os.listdir(tmp_path / "monthly")) == 3

    from_cache = [pl.read_parquet(f"{str(tmp_path)}/monthly/2024-{month}.parquet") for month in ["10", "11", "12"]]
    assert from_cache[0].shape == (31, 4)
    assert from_cache[1].shape == (30, 4)
    assert from_cache[2].shape == (31, 4)


def test_custom_partition_format(tmp_path):
    daily_df = pl.DataFrame({"date": [datetime.date(2023, 1, 15)], "value": [1]}).lazy()
    monthly_df = pl.DataFrame({"date": [datetime.date(2023, 2, 15)], "value": [2]}).lazy()
    yearly_df = pl.DataFrame({"date": [datetime.date(2023, 3, 15)], "value": [3]}).lazy()

    daily_df.piot.cache_parquet(
        cache_path=tmp_path / "daily_test",
        date_column="date",
        time_unit="daily",
        partition_format="y=$year/m=$month/d=$day",
    ).collect()
    monthly_df.piot.cache_parquet(
        cache_path=tmp_path / "monthly_test",
        date_column="date",
        time_unit="monthly",
        partition_format="year=$year/month=$month",
    ).collect()
    yearly_df.piot.cache_parquet(
        cache_path=tmp_path / "yearly_test",
        date_column="date",
        time_unit="yearly",
        partition_format="year=$year",
    ).collect()

    assert (tmp_path / "daily_test" / "daily" / "y=2023/m=01/d=15.parquet").exists()
    assert (tmp_path / "monthly_test" / "monthly" / "year=2023/month=02.parquet").exists()
    assert (tmp_path / "yearly_test" / "yearly" / "year=2023.parquet").exists()

    daily_data = pl.scan_parquet(tmp_path / "daily_test" / "daily" / "y=2023/m=01/d=15.parquet").collect()
    monthly_data = pl.scan_parquet(tmp_path / "monthly_test" / "monthly" / "year=2023/month=02.parquet").collect()
    yearly_data = pl.scan_parquet(tmp_path / "yearly_test" / "yearly" / "year=2023.parquet").collect()

    assert daily_data["value"].to_list() == [1]
    assert monthly_data["value"].to_list() == [2]
    assert yearly_data["value"].to_list() == [3]


def test_bad_partition_format(tmp_path):
    df = pl.DataFrame({"date": [datetime.date(2023, 1, 15)], "value": [42]}).lazy()

    with pytest.raises(ValueError, match="missing"):
        df.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="monthly",
            partition_format="$year",  # missing $month
        )

    with pytest.raises(ValueError, match="unexpected"):
        df.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="monthly",
            partition_format="$year-$month-$day",
        )

    with pytest.raises(ValueError):
        df.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="monthly",
            partition_format="$year-${month",
        )

    with pytest.raises(ValueError, match="Invalid time unit"):
        df.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="occasionally",
        )


def test_custom_partition_format_cache_readback(tmp_path):
    """Test that custom partition formats work correctly with cache read-back functionality."""
    # Test data spanning multiple partitions
    df1 = pl.DataFrame({"date": [datetime.date(2024, 6, 1), datetime.date(2024, 6, 2), datetime.date(2024, 6, 3)], "value": [10, 20, 30]}).lazy()

    # Write initial data with custom partition format
    cache_path = tmp_path / "custom_format_test"
    result1 = df1.piot.cache_parquet(
        cache_path=cache_path,
        date_column="date",
        time_unit="daily",
        partition_format="theYear=$year/theMonth=$month/theDay=$day",
    ).collect()

    # Verify initial data
    assert result1.sort("date")["value"].to_list() == [10, 20, 30]

    # Verify file structure exists
    assert (cache_path / "daily" / "theYear=2024/theMonth=06/theDay=01.parquet").exists()
    assert (cache_path / "daily" / "theYear=2024/theMonth=06/theDay=02.parquet").exists()
    assert (cache_path / "daily" / "theYear=2024/theMonth=06/theDay=03.parquet").exists()

    # Now try to append new data - this should trigger the existing partition parsing
    df2 = pl.DataFrame({"date": [datetime.date(2024, 6, 4), datetime.date(2024, 6, 5)], "value": [40, 50]}).lazy()

    result2 = df2.piot.cache_parquet(
        cache_path=cache_path,
        date_column="date",
        time_unit="daily",
        partition_format="theYear=$year/theMonth=$month/theDay=$day",
    ).collect()

    # Should return all data (existing + new)
    expected_values = [10, 20, 30, 40, 50]
    assert sorted(result2["value"].to_list()) == expected_values

    # Verify new files were created
    assert (cache_path / "daily" / "theYear=2024/theMonth=06/theDay=04.parquet").exists()
    assert (cache_path / "daily" / "theYear=2024/theMonth=06/theDay=05.parquet").exists()


def test_overwrite(tmp_path):
    df1 = pl.DataFrame({"date": [datetime.date(2023, 1, 15)], "value": [42]}).lazy()
    df1.piot.cache_parquet(cache_path=tmp_path, date_column="date").collect()

    partition_path = tmp_path / "monthly" / "2023-01.parquet"
    first_mtime = os.path.getmtime(partition_path)
    time.sleep(0.01)

    df2 = pl.DataFrame({"date": [datetime.date(2023, 1, 15)], "value": [99]}).lazy()
    df2.piot.cache_parquet(cache_path=tmp_path, date_column="date", cache_mode=CacheMode.REBUILD).collect()

    new_data = pl.scan_parquet(partition_path).collect()
    assert new_data["value"].to_list() == [99]
    assert first_mtime < os.path.getmtime(partition_path)


def test_multiple_partitions(tmp_path):
    df = pl.DataFrame(
        {
            "date": [
                datetime.date(2023, 1, 15),
                datetime.date(2023, 2, 20),
                datetime.date(2023, 3, 25),
            ],
            "value": [1, 2, 3],
        }
    ).lazy()

    df.piot.cache_parquet(cache_path=tmp_path, date_column="date").collect()

    base = tmp_path / "monthly"
    for m in ("01", "02", "03"):
        assert (base / f"2023-{m}.parquet").exists()

    jan = pl.scan_parquet(base / "2023-01.parquet").collect()
    feb = pl.scan_parquet(base / "2023-02.parquet").collect()
    mar = pl.scan_parquet(base / "2023-03.parquet").collect()
    assert jan["value"].to_list() == [1]
    assert feb["value"].to_list() == [2]
    assert mar["value"].to_list() == [3]

    # also works for other time units
    df.piot.cache_parquet(tmp_path / "daily_test", "date", time_unit="daily").collect()
    df.piot.cache_parquet(tmp_path / "yearly_test", "date", time_unit="yearly").collect()

    assert (tmp_path / "daily_test" / "daily" / "2023-01-15.parquet").exists()
    assert (tmp_path / "yearly_test" / "yearly" / "2023.parquet").exists()


def test_missing_written(tmp_path):
    """
    Ensure that partitions for which *no* data exists are still written as
    empty parquet files.
    """
    # only one row in January
    df = pl.DataFrame({"date": [datetime.date(2023, 1, 15)], "value": [1]}).lazy()

    (
        df.piot.cache_parquet(cache_path=tmp_path, date_column="date")
        .filter((pl.col("date") >= datetime.date(2023, 1, 1)) & (pl.col("date") <= datetime.date(2023, 3, 31)))
        .collect()
    )

    base = tmp_path / "monthly"
    # existing data
    assert (base / "2023-01.parquet").exists()

    # empty partitions must also exist
    for month in ("02", "03"):
        part_file = base / f"2023-{month}.parquet"
        assert part_file.exists(), f"missing empty parquet for 2023-{month}"
        assert pl.read_parquet(part_file).height == 0


def test_contradiction(monkeypatch, tmp_path):
    """
    Test that contradiction detection prevents writing twice.
    """

    dates = pl.date_range(
        datetime.date(2024, 1, 1),
        datetime.date(2024, 1, 31),
        interval="1d",
        eager=True,
    )
    base = pl.LazyFrame({"date": dates, "val": pl.arange(0, len(dates), eager=True)})

    # Predicate that selects a subset (10th .. 20th)
    pred = (pl.col("date") >= datetime.date(2024, 1, 10)) & (pl.col("date") <= datetime.date(2024, 1, 20))

    call_counter = {"n": 0}
    original_sink = pl.LazyFrame.sink_parquet

    def _counting_sink(self, *args, **kwargs):
        call_counter["n"] += 1
        return original_sink(self, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", _counting_sink, raising=True)

    cached = base.piot.cache_parquet(
        cache_path=tmp_path,
        date_column="date",
        time_unit="monthly",
    )

    cached.filter(pred).collect()

    cached.filter(pred).collect()

    assert call_counter["n"] == 1, f"sink_parquet was called {call_counter['n']} times; contradiction logic failed"


def test_extra_partition_columns(tmp_path):
    """Test that extra partition columns are correctly handled."""
    df = pl.DataFrame(
        {
            "date": [datetime.date(2024, 1, 3), datetime.date(2024, 1, 4)] * 2,
            "country": ["US", "CA"] * 2,
            "val": [1, 2, 3, 4],
        }
    ).lazy()

    (
        df.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="monthly",
            extra_partition_cols=["country"],
        ).collect()
    )

    assert (tmp_path / "monthly" / "US" / "2024-01.parquet").is_file()
    assert (tmp_path / "monthly" / "CA" / "2024-01.parquet").is_file()

    read_back = pl.scan_parquet(f"{tmp_path}/monthly/**/*.parquet").collect()
    assert set(read_back["country"]) == {"US", "CA"}
    assert len(read_back) == 4


def test_between_pred_uses_existing_partitions(tmp_path, monkeypatch):
    """
    Test that contradiction detection works on an `is_between`
    call preceded by two individual daily partitions.
    """
    import datetime as datetime

    import polars as pl

    df = pl.DataFrame(
        {
            "date": [datetime.date(2021, 3, 1), datetime.date(2021, 3, 2)],
            "val": [10, 20],
        }
    ).lazy()

    # 1 ─ write 2021-03-01
    df.piot.cache_parquet(tmp_path, "date", time_unit="daily").filter(pl.col("date") == datetime.date(2021, 3, 1)).collect()

    # 2 ─ write 2021-03-02
    df.piot.cache_parquet(tmp_path, "date", time_unit="daily").filter(pl.col("date") == datetime.date(2021, 3, 2)).collect()

    # patch sink_parquet; any extra write will increment the counter
    cnt = {"n": 0}
    orig_sink = pl.LazyFrame.sink_parquet

    def _count(*a, **k):
        cnt["n"] += 1
        return orig_sink(*a, **k)

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", _count, raising=True)

    between_pred = pl.col("date").is_between(datetime.date(2021, 3, 1), datetime.date(2021, 3, 2), closed="both")

    # 3 ─ query the BETWEEN predicate; should be served from cache
    out = df.piot.cache_parquet(tmp_path, "date", time_unit="daily").filter(between_pred).collect()

    assert cnt["n"] == 0, "sink_parquet ran again; expected cache hit"
    assert out["val"].to_list() == [10, 20]


def test_empty_partitions_with_extra_cols(tmp_path):
    """
    Tests that we correctly write empty partitions for
    `extra_partition_cols` when a queried combination has no data.
    """
    df = pl.DataFrame(
        {
            "date": [datetime.date(2024, 5, 10), datetime.date(2024, 5, 10)],
            "region": ["NA", "EU"],
            "value": [100, 200],
        }
    ).lazy()

    # Query for NA, EU, and APAC. APAC has no data in the source frame.
    predicate = pl.col("region").is_in(["NA", "EU", "APAC"]) & (pl.col("date") == datetime.date(2024, 5, 10))

    (
        df.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="daily",
            extra_partition_cols="region",
        )
        .filter(predicate)
        .collect()
    )

    # Check that partitions with data were written correctly
    na_path = tmp_path / "daily" / "NA" / "2024-05-10.parquet"
    eu_path = tmp_path / "daily" / "EU" / "2024-05-10.parquet"
    assert na_path.exists()
    assert eu_path.exists()
    assert pl.read_parquet(na_path)["value"][0] == 100
    assert pl.read_parquet(eu_path)["value"][0] == 200

    # Crucially, check that the empty partition for APAC was also created
    apac_path = tmp_path / "daily" / "APAC" / "2024-05-10.parquet"
    assert apac_path.exists(), "Empty partition for APAC was not created"
    assert pl.read_parquet(apac_path).height == 0


def test_predicate_pushdown(tmp_path, capsys):
    """
    Tests that a simple filter applied AFTER .cache_parquet() is correctly
    pushed down to the upstream source during the initial discovery phase.
    """

    df = pl.DataFrame({"d": [datetime.date(2020, 1, 1), datetime.date(2020, 1, 2)], "val": [1, 2]}).lazy().piot.debug()

    result_df = df.piot.cache_parquet(tmp_path, date_column="d", time_unit="daily").filter(pl.col("d") == datetime.date(2020, 1, 1)).collect()

    captured = capsys.readouterr()
    stdout = captured.out

    expected_predicate_str = 'predicate=[(col("d")) == (2020-01-01)]'
    other_possibility = 'predicate=col("d").is_between([2020-01-01, 2020-01-02])'
    assert any(s in stdout for s in (expected_predicate_str, other_possibility)), "The simple predicate was not pushed down to the upstream source."
    assert result_df.shape == (1, 2)
    assert result_df["d"][0] == datetime.date(2020, 1, 1)


def test_write_uses_widened_predicate(tmp_path, capsys):
    """
    Tests that the write operation uses a "widened" predicate to fetch
    the full partition, not just the slice requested by the user.
    """
    from datetime import date

    # Arrange: Source data spans multiple days in a month
    df = pl.DataFrame(
        {"date": [date(2024, 5, 1), date(2024, 5, 15), date(2024, 5, 30)], "val": [1, 15, 30], "partition": ["foo", "bar", "baz"]}
    ).lazy()

    # Act: Chain the debug source, then cache with a filter for a *single day*
    (df.piot.debug().piot.cache_parquet(tmp_path, date_column="date", time_unit="monthly").filter(pl.col("date") == date(2024, 5, 15)).collect())

    # Assert: Check the output from the debug source
    captured = capsys.readouterr().out

    # The second call is for writing. It should use the WIDENED predicate
    # to fetch the entire month of May.
    # The predicate should be something like: is_between(2024-05-01, 2024-06-01)
    assert "is_between" in captured
    assert "2024-05-01" in captured
    assert "2024-06-01" in captured

    # Assert that the written file contains the full partition's data
    written_file = tmp_path / "monthly" / "2024-05.parquet"
    assert written_file.exists()
    data_in_cache = pl.read_parquet(written_file)
    assert data_in_cache.shape == (3, 3), "The cached file should contain all 3 rows for the month"
    assert data_in_cache["val"].to_list() == [1, 15, 30]


def test_model_predicate_is_pushed_down(tmp_path, capsys):
    """
    Test that pushdown happens for a model predicate
    even when the filter is applied after the `cache_parquet` call.
    """
    df = (
        pl.DataFrame(
            {
                "date": [
                    datetime.date(2025, 5, 1),
                    datetime.date(2025, 5, 2),
                    datetime.date(2025, 5, 3),
                ],
                "model": ["USE4S", "OTHER", "USE4S"],
                "val": [10, 20, 30],
            }
        )
        .lazy()
        .piot.debug()
    )

    (
        df.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="monthly",
            extra_partition_cols="model",
        )
        .filter(pl.col("model") == "USE4S")
        .collect()
    )

    out = capsys.readouterr().out
    assert 'col("model")' in out and '"USE4S"' in out, "model predicate was not pushed down; helper-column filter still blocks predicate push-down"


def test_empty_result_from_cache(tmp_path):
    """
    Test that no data doesn't yield an error.
    """
    # Upstream frame has only 2024-02-01
    src = pl.DataFrame({"date": [datetime.date(2024, 2, 1)], "val": [1]}).lazy()

    # Go through cache_parquet, then ask for a date that isn't there
    lf = src.piot.cache_parquet(
        cache_path=tmp_path,
        date_column="date",
        time_unit="monthly",
    ).filter(pl.col("date") == datetime.date(2025, 1, 1))

    out = lf.collect()

    assert out.is_empty()
    assert out.columns == ["date", "val"]


def test_sequential_cache_parquet(tmp_path):
    """Test that sequential `cache_parquet` calls work correctly."""
    base = pl.LazyFrame({"date": [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2), datetime.date(2024, 2, 1)], "x": [1, 3, 5]})

    sequential = base.piot.cache_parquet(tmp_path, "date", time_unit="monthly").piot.cache_parquet(tmp_path, "date", time_unit="monthly")

    out = sequential.collect()
    assert out.shape == (3, 2)


def test_missing_written_daily_with_extra_cols(tmp_path):
    """
    Test that we correctly write empty partitions for daily partitions when
    extra partition columns are specified.
    """
    df = pl.LazyFrame(
        {
            "date": [datetime.date(2024, 5, 10), datetime.date(2024, 5, 10)],
            "region": ["NA", "EU"],
            "value": [100, 200],
        }
    )

    # Predicate selects TWO days and THREE regions (APAC has no rows).
    pred = (
        pl.col("region").is_in(["NA", "EU", "APAC"]) & (pl.col("date") >= datetime.date(2024, 5, 10)) & (pl.col("date") <= datetime.date(2024, 5, 11))
    )

    (
        df.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="daily",
            extra_partition_cols="region",
        )
        .filter(pred)
        .collect()
    )

    base = tmp_path / "daily"
    # Real data rows exist only for (NA, 2024-05-10) and (EU, 2024-05-10).
    # The test checks that the *four* missing partitions are still written.
    expected_empty = [
        ("NA", "2024-05-11"),
        ("EU", "2024-05-11"),
        ("APAC", "2024-05-10"),
        ("APAC", "2024-05-11"),
    ]
    for region, day in expected_empty:
        f = base / region / f"{day}.parquet"
        assert f.is_file(), f"missing empty parquet for {region}/{day}"
        assert pl.read_parquet(f).height == 0


def test_missing_written_daily_with_extra_cols_gap(tmp_path):
    """
    Test that we correctly write empty partitions for daily partitions when
    extra partition columns are specified, and there is a gap.
    """
    df = pl.LazyFrame(
        {
            "date": [datetime.date(2024, 5, 10), datetime.date(2024, 5, 11)],
            "region": ["NA", "EU"],
            "value": [100, 200],
        }
    )

    # Predicate selects TWO days and THREE regions (APAC has no rows).
    pred = (
        pl.col("region").is_in(["NA", "EU", "APAC"])
        & (pl.col("date") >= datetime.date(2024, 5, 10))
        & (pl.col("date") <= datetime.date(2024, 5, 14))
        & (pl.col("date") != datetime.date(2024, 5, 11))
    )

    (
        df.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="daily",
            extra_partition_cols="region",
        )
        .filter(pred)
        .collect()
    )

    base = tmp_path / "daily"
    for region in ["NA", "EU", "APAC"]:
        # Check that the partition for 2024-05-11 exists
        f = base / region / "2024-05-11.parquet"
        assert not f.exists(), f"should not generate parquet for {region}/2024-05-11"


def test_polars_private_api_still_exists():
    """Assert the Polars credential provider API contains a method we expect to use."""
    cred = pl.CredentialProviderAWS(profile_name=None, _storage_options_has_endpoint_url=False)
    assert hasattr(cred, "_storage_update_options"), "_storage_update_options() removed upstream"
    out = cred._storage_update_options()
    assert isinstance(out, dict), "unexpected return type"


@pytest.mark.parametrize("run_first", ["collect", "no_collect"])
def test_intermediate_collect(tmp_path, run_first):
    """
    Test an edge case that we discovered: an intermediate collect
    should not break the caching mechanism.
    """

    def run(base, do_collect: bool):
        if base.exists():
            shutil.rmtree(base)
        src = pl.DataFrame({"a": [1, 2, 3], "date": [datetime.date(2025, 7, 31)] * 3}).lazy()

        ldf = src.piot.cache_parquet(base / "foo_test", "date", time_unit="daily")
        if do_collect:
            ldf.collect()

        ldf1 = ldf.with_columns(pl.lit("foo").alias("foo")).piot.cache_parquet(base / "foo_test1", "date", time_unit="daily")
        ldf2 = ldf.with_columns(pl.lit("bar").alias("bar")).piot.cache_parquet(base / "foo_test2", "date", time_unit="daily")

        res = ldf1.join(ldf2, on="a").collect()

        cache = {}
        for p in base.rglob("*.parquet"):
            key = p.relative_to(base).as_posix()
            cache[key] = pl.read_parquet(p)
        return res, cache

    if run_first == "no_collect":
        # Run without intermediate collect
        res_no_collect, cache_no_collect = run(tmp_path / "scenario_no_collect", False)
        res_collect, cache_collect = run(tmp_path / "scenario_with_collect", True)
    else:
        res_collect, cache_collect = run(tmp_path / "scenario_with_collect", True)
        res_no_collect, cache_no_collect = run(tmp_path / "scenario_no_collect", False)

    assert_frame_equal(res_no_collect, res_collect)
    assert set(cache_no_collect) == set(cache_collect)
    for k in cache_no_collect:
        assert_frame_equal(cache_no_collect[k], cache_collect[k])


def test_missing_written_daily_with_extra_cols_gap2(tmp_path):
    """
    Test that we correctly write empty partitions for daily partitions when
    extra partition columns are specified, and there is a gap.
    """
    df = pl.LazyFrame(
        {
            "date": [datetime.date(2024, 5, 10), datetime.date(2024, 5, 11)],
            "region": ["NA", "EU"],
            "value": [100, 200],
        }
    )
    # We include "NA" in the is_in, but exclude it explicitly
    pred = (pl.col.region.is_in(["EU", "NA"]) & pl.col.region.ne("NA")) & (pl.col.date.eq(datetime.date(2024, 5, 10)))
    res = (
        df.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="daily",
            extra_partition_cols="region",
        )
        .filter(pred)
        .collect()
    )
    assert res.is_empty()
    base = tmp_path / "daily"
    f = base / "NA" / "2024-05-10.parquet"
    if f.exists():
        # Should not be empty, we have data for "NA" on "2024-05-10"
        assert not pl.read_parquet(f).is_empty()


def test_push_and_parse_exclusions_to_custom_io_source(tmp_path):
    from polars.io.plugins import register_io_source

    from polars_io_tools.io_sources.range_visitor import convert_expr_to_datetime_range
    from polars_io_tools.io_sources.set_visitor import convert_expr_to_valid_values

    predicate_record = {}

    def my_io(self: pl.LazyFrame) -> pl.LazyFrame:
        schema = self.collect_schema()

        def source_generator(with_columns, predicate, n_rows, batch_size):
            # print("Running my_io\n")
            ldf = self.clone()
            if predicate is not None:
                print("PREDICATE IS NOT NONE\n")
                predicate_record["predicate"] = predicate
                ldf = ldf.filter(predicate)
            else:
                print("PREDICATE IS NONE\n")
            if with_columns is not None:
                ldf = ldf.select(with_columns)
            if n_rows is not None:
                ldf = ldf.head(n_rows)
            print(ldf.explain())
            yield ldf.collect()

        return register_io_source(source_generator, schema=schema, validate_schema=True)

    df = pl.LazyFrame(
        {
            "date": [datetime.date(2024, 5, 10), datetime.date(2024, 5, 11)],
            "region": ["NA", "EU"],
            "value": [100, 200],
        }
    )
    lf = df.clone()
    pred = pl.col.region.eq("NA") & (pl.col.date.eq(datetime.date(2024, 5, 10)))
    res = (
        my_io(df)
        .piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="daily",
            extra_partition_cols="region",
        )
        .filter(pred)
        .collect()
    )
    assert_frame_equal(res, lf.filter(pred).collect())
    base = tmp_path / "daily"
    region, day, exp_df = ("NA", "2024-05-10", df.filter(pred).collect())
    f = base / region / f"{day}.parquet"
    assert f.is_file(), f"missing empty parquet for {region}/{day}"
    assert_frame_equal(pl.read_parquet(f), exp_df)
    # reset predicate holder
    predicate_record = {}
    # call again, wider predicate
    pred2 = pl.col.region.is_in(["NA", "EU"]) & (pl.col.date.eq(datetime.date(2024, 5, 10)))
    res = (
        my_io(df)
        .piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="daily",
            extra_partition_cols="region",
        )
        .filter(pred2)
        .collect()
    )
    assert_frame_equal(res, lf.filter(pred2).collect())
    pred_recorded = predicate_record.get("predicate")
    assert pred_recorded is not None, "predicate was not recorded"
    pred_og_inputs = pred2.meta.pop()
    new_pred_inputs = pred_recorded.meta.pop()
    pred_seri = set([e.meta.serialize(format="json") for e in pred_og_inputs])
    pred_recorded_seri = set([e.meta.serialize(format="json") for e in new_pred_inputs])
    assert pred_seri != pred_recorded_seri, "recorded predicate matches original predicate; expected exclusion"
    raw_date_info = convert_expr_to_datetime_range(pred_recorded, "date", get_enclosure=False)
    valid_values = convert_expr_to_valid_values(pred_recorded, "region")

    assert raw_date_info.lower == datetime.datetime(2024, 5, 10, 0, 0)
    assert valid_values == set(["EU"]), (
        f"Received {valid_values = } expected just to have EU since we cached NA for 2024-05-10"
    )  # we have data for "NA" cached, so shouldnt query


def test_large_partition_query(tmp_path):
    """
    Test that we correctly write empty partitions for daily partitions when
    extra partition columns are specified, and there is a gap.
    """
    df = pl.LazyFrame(
        {
            "date": pl.date_range(datetime.date(2020, 1, 1), datetime.date(2026, 1, 1), eager=True),
        }
    )
    df = df.with_columns(
        region=pl.col.date.dt.day().replace_strict({1: "NA", 2: "EU", 3: "CA", 4: "NA", 6: "NA"}, default="US", return_dtype=pl.Utf8),
        value=pl.lit(100),
    )
    small_pred = pl.col.region.eq("US") & pl.col.date.is_between(datetime.date(2020, 1, 1), datetime.date(2020, 6, 1))
    _res = (
        df.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="daily",
            extra_partition_cols="region",
        )
        .filter(small_pred)
        .collect()
    )
    big_pred = pl.col.region.is_in(["US", "NA"]) & pl.col.date.is_between(datetime.date(2019, 12, 15), datetime.date(2020, 9, 14))
    _res2 = (
        df.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="daily",
            extra_partition_cols="region",
        )
        .filter(big_pred)
        .collect()
    )
    expected = df.filter(big_pred).collect()
    missing_expected = expected.join(
        _res2,
        on=["date", "region"],
        how="anti",
    )
    assert missing_expected.is_empty(), "missing expected rows in the result"


def test_huge_partition_query(tmp_path, caplog):
    """
    Test a large partition query that spans multiple years. We have partial data in the cache, and if we submitted each requested partition as a combination of OR's, we would end up with a very large query that could segfault.
    """
    df = pl.LazyFrame(
        {
            "date": pl.date_range(datetime.date(2020, 1, 1), datetime.date(2026, 1, 1), eager=True),
        }
    )
    df = df.with_columns(
        region=pl.col.date.dt.day().replace_strict({1: "NA", 2: "EU", 3: "CA", 4: "NA", 6: "NA"}, default="US", return_dtype=pl.Utf8),
        value=pl.lit(100),
    )
    small_pred = pl.col.region.eq("US") & pl.col.date.is_between(datetime.date(2020, 1, 1), datetime.date(2021, 2, 1))
    res = (
        df.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="daily",
            extra_partition_cols="region",
        )
        .filter(small_pred)
        .collect()
    )
    expected = df.filter(small_pred).collect()
    assert_frame_equal(res.sort("date", "region"), expected.sort("date", "region"))

    big_pred = pl.col.region.is_in(["US", "NA"]) & pl.col.date.is_between(datetime.date(2020, 1, 1), datetime.date(2024, 2, 1))

    import logging

    caplog.set_level(logging.DEBUG, logger="polars_io_tools.io_sources.lazy_cache_parquet")
    res2 = (
        df.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="daily",
            extra_partition_cols="region",
        )
        .filter(big_pred)
        .collect()
    )
    expected = df.filter(big_pred).collect()
    missing_expected = expected.join(
        res2,
        on=["date", "region"],
        how="anti",
    )
    assert missing_expected.is_empty(), "missing expected rows in the result"
    assert_frame_equal(res2.sort("date", "region"), expected.sort("date", "region"))
    # Huge bounded queries avoid very large explicit path lists on Windows.
    logs = caplog.text
    assert "Writing partitions sequentially" in logs
    assert "exceeding limit" in logs
    assert "Cache scan uses glob" in logs


def test_get_fs_path_directory_info_strips_trailing_slash_s3():
    from polars_io_tools.io_sources.lazy_cache_parquet import _get_fs_path_directory_info

    fs_prefix, time_dir = _get_fs_path_directory_info("s3://mybucket/my/prefix/", "daily")
    assert fs_prefix == "mybucket/my/prefix/daily"
    assert time_dir.startswith("s3://mybucket/my/prefix/daily")


def test_get_fs_path_directory_info_preserves_double_leading_slash_s3():
    from polars_io_tools.io_sources.lazy_cache_parquet import _get_fs_path_directory_info

    fs_prefix, time_dir = _get_fs_path_directory_info("s3://mybucket//double/leading/", "monthly")
    assert fs_prefix == "mybucket//double/leading/monthly"
    assert time_dir.startswith("s3://mybucket//double/leading/monthly")


def test_get_fs_path_directory_info_local_path_gets_file_prefix(tmp_path):
    """Test that plain local paths get file:// prefix for atomic writes via object-store."""
    from polars_io_tools.io_sources.lazy_cache_parquet import _get_fs_path_directory_info

    fs_prefix, time_dir = _get_fs_path_directory_info(str(tmp_path), "daily")
    assert fs_prefix == str(tmp_path / "daily")
    assert time_dir == (tmp_path / "daily").as_uri()


def test_get_fs_path_directory_info_file_uri_not_double_prefixed(tmp_path):
    """Test that file:// URIs don't get double-prefixed."""
    from polars_io_tools.io_sources.lazy_cache_parquet import _get_fs_path_directory_info

    file_uri = tmp_path.as_uri()
    fs_prefix, time_dir = _get_fs_path_directory_info(file_uri, "daily")
    assert fs_prefix == str(tmp_path / "daily")
    assert time_dir == (tmp_path / "daily").as_uri()
    # Ensure no double file:// prefix
    assert not time_dir.startswith("file://file://")


def test_get_fs_path_directory_info_local_and_file_uri_equivalent(tmp_path):
    """Test that plain path and file:// URI produce identical results."""
    from polars_io_tools.io_sources.lazy_cache_parquet import _get_fs_path_directory_info

    plain_prefix, plain_dir = _get_fs_path_directory_info(str(tmp_path), "monthly")
    file_prefix, file_dir = _get_fs_path_directory_info(tmp_path.as_uri(), "monthly")

    assert plain_prefix == file_prefix
    assert plain_dir == file_dir


def test_path_as_file_uri_formats_windows_drive_path():
    from polars_io_tools.io_sources.lazy_cache_parquet import _path_as_file_uri

    uri = _path_as_file_uri(PureWindowsPath("C:/Users/runneradmin/AppData/Local/Temp/cache/daily"))
    assert uri == "file:///C:/Users/runneradmin/AppData/Local/Temp/cache/daily"


class TestOptionalDateColumn:
    """Test suite for optional date column functionality."""

    def test_cache_parquet_no_date_single_extra_col(self, tmp_path):
        """Test cache_parquet with no date column and single extra partition column."""
        df = pl.DataFrame(
            {"region": ["US", "US", "EU", "EU", "APAC"], "value": [100, 200, 150, 250, 180], "category": ["A", "B", "A", "B", "A"]}
        ).lazy()

        # Cache with region partitioning only
        cached_df = df.piot.cache_parquet(
            cache_path=tmp_path,
            date_column=None,
            extra_partition_cols="region",  # Test string input
        )

        result = cached_df.collect()
        assert result.shape == (5, 3)
        assert set(result.columns) == {"region", "value", "category"}

        # Verify cache structure - should create null/US.parquet, null/EU.parquet, null/APAC.parquet
        cache_files = list(tmp_path.rglob("*.parquet"))
        assert len(cache_files) == 3  # One for each region

        # Verify all files are in null directory
        for f in cache_files:
            assert "null" in str(f)

        # Test reading from cache again
        cached_df2 = df.piot.cache_parquet(cache_path=tmp_path, date_column=None, extra_partition_cols="region")
        result2 = cached_df2.collect()
        assert result.shape == result2.shape
        assert result.equals(result2)

    def test_cache_parquet_no_date_multiple_extra_cols(self, tmp_path):
        """Test cache_parquet with no date column and multiple extra partition columns."""
        df = pl.DataFrame(
            {
                "region": ["US", "US", "EU", "EU", "APAC", "APAC"],
                "product": ["A", "B", "A", "B", "A", "B"],
                "value": [100, 200, 150, 250, 180, 220],
                "score": [1.1, 2.2, 1.5, 2.5, 1.8, 2.8],
            }
        ).lazy()

        # Cache with both region and product partitioning
        cached_df = df.piot.cache_parquet(cache_path=tmp_path, date_column=None, extra_partition_cols=["region", "product"])

        result = cached_df.collect()
        assert result.shape == (6, 4)

        # Verify all combinations are cached - should be 3 regions × 2 products = 6 files
        cache_files = list(tmp_path.rglob("*.parquet"))
        assert len(cache_files) == 6

        # Verify directory structure: null/region/product.parquet
        file_paths = [f.relative_to(tmp_path).as_posix() for f in cache_files]
        expected_patterns = [
            "null/US/A.parquet",
            "null/US/B.parquet",
            "null/EU/A.parquet",
            "null/EU/B.parquet",
            "null/APAC/A.parquet",
            "null/APAC/B.parquet",
        ]
        for pattern in expected_patterns:
            assert any(pattern in path for path in file_paths), f"Missing pattern: {pattern}"

    def test_cache_parquet_no_date_no_extra_cols(self, tmp_path):
        """Test cache_parquet with no date column and no extra partition columns (single table)."""
        df = pl.DataFrame({"value": [100, 200, 150], "name": ["Alice", "Bob", "Charlie"], "active": [True, False, True]}).lazy()

        # Cache with no partitioning - should create single file
        cached_df = df.piot.cache_parquet(cache_path=tmp_path, date_column=None, extra_partition_cols=None)

        result = cached_df.collect()
        assert result.shape == (3, 3)
        assert set(result.columns) == {"value", "name", "active"}

        # Verify single cache file
        cache_files = list(tmp_path.rglob("*.parquet"))
        assert len(cache_files) == 1

        # Should be in null/data.parquet
        cache_file = cache_files[0]
        assert "null" in str(cache_file)
        assert "data.parquet" in str(cache_file)

        # Test reading from cache again
        cached_df2 = df.piot.cache_parquet(cache_path=tmp_path, date_column=None, extra_partition_cols=None)
        result2 = cached_df2.collect()
        assert result.equals(result2)

    def test_cache_parquet_no_date_with_predicate(self, tmp_path):
        """Test that predicates work with no date column."""
        df = pl.DataFrame(
            {"region": ["US", "US", "EU", "EU", "APAC"], "category": ["X", "Y", "X", "Y", "X"], "value": [100, 200, 150, 250, 180]}
        ).lazy()

        # Cache all data first
        df.piot.cache_parquet(cache_path=tmp_path, date_column=None, extra_partition_cols=["region", "category"]).collect()

        # Now read with a predicate
        filtered_result = (
            df.filter(pl.col("region") == "US")
            .piot.cache_parquet(cache_path=tmp_path, date_column=None, extra_partition_cols=["region", "category"])
            .collect()
        )

        # Should return all data (cache scans all files), but we can verify the original filter works
        original_filtered = df.filter(pl.col("region") == "US").collect()
        assert filtered_result.shape == (5, 3)  # Cache returns all data
        assert original_filtered.shape == (2, 3)  # Original filter works

    def test_cache_parquet_no_date_different_cache_modes(self, tmp_path):
        """Test different cache modes work with no date column."""
        df = pl.DataFrame({"category": ["A", "B", "A", "B"], "value": [100, 200, 150, 250]}).lazy()

        # Test IGNORE mode
        result_ignore = df.piot.cache_parquet(
            cache_path=tmp_path / "ignore", date_column=None, extra_partition_cols=["category"], cache_mode=CacheMode.IGNORE
        ).collect()

        # Should return original data without caching
        assert result_ignore.shape == (4, 2)
        cache_files = list((tmp_path / "ignore").rglob("*.parquet"))
        assert len(cache_files) == 0

        # Test CACHE mode (normal)
        result_cache = df.piot.cache_parquet(
            cache_path=tmp_path / "cache", date_column=None, extra_partition_cols=["category"], cache_mode=CacheMode.CACHE
        ).collect()

        assert result_cache.shape == (4, 2)
        cache_files = list((tmp_path / "cache").rglob("*.parquet"))
        assert len(cache_files) == 2  # A and B partitions

        # Test REBUILD mode
        result_rebuild = df.piot.cache_parquet(
            cache_path=tmp_path / "cache",  # Same path
            date_column=None,
            extra_partition_cols=["category"],
            cache_mode=CacheMode.REBUILD,
        ).collect()

        assert result_rebuild.shape == (4, 2)
        assert result_cache.equals(result_rebuild)

    def test_cache_parquet_no_date_metadata(self, tmp_path):
        """Test that metadata is correctly set for no date column cases."""
        df = pl.DataFrame({"region": ["US", "EU"], "value": [100, 200]}).lazy()

        # Cache with extra columns
        df.piot.cache_parquet(cache_path=tmp_path / "with_extra", date_column=None, extra_partition_cols=["region"]).collect()

        # Check metadata in the parquet files
        cache_files = list((tmp_path / "with_extra").rglob("*.parquet"))
        assert len(cache_files) == 2

        # Read metadata from one file
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(cache_files[0])
        metadata = parquet_file.metadata.metadata

        # Should have null time_unit and partitioned_by info
        assert metadata[b"__piot__time_unit"] == b"null"
        assert metadata[b"__piot__partitioned_by"] == b"region"

        # Test single partition case
        df_single = pl.DataFrame({"value": [1, 2, 3]}).lazy()
        df_single.piot.cache_parquet(cache_path=tmp_path / "single", date_column=None, extra_partition_cols=None).collect()

        cache_files_single = list((tmp_path / "single").rglob("*.parquet"))
        assert len(cache_files_single) == 1

        parquet_file_single = pq.ParquetFile(cache_files_single[0])
        metadata_single = parquet_file_single.metadata.metadata

        assert metadata_single[b"__piot__time_unit"] == b"null"
        assert metadata_single[b"__piot__partitioned_by"] == b""

    def test_cache_parquet_no_date_backward_compatibility(self, tmp_path):
        """Test that existing functionality still works alongside new features."""
        import datetime

        # Test with date column (existing functionality)
        df_with_date = pl.DataFrame(
            {"date": [datetime.date(2024, 1, 1), datetime.date(2024, 2, 1)], "region": ["US", "EU"], "value": [100, 200]}
        ).lazy()

        result_with_date = df_with_date.piot.cache_parquet(
            cache_path=tmp_path / "with_date", date_column="date", time_unit="monthly", extra_partition_cols=["region"]
        ).collect()

        assert result_with_date.shape == (2, 3)
        cache_files_date = list((tmp_path / "with_date").rglob("*.parquet"))
        assert len(cache_files_date) == 2  # One per month-region combination

        # Verify these are in monthly directory, not null
        for f in cache_files_date:
            assert "monthly" in str(f)
            assert "null" not in str(f)

        # Test without date column (new functionality)
        df_no_date = pl.DataFrame({"region": ["US", "EU", "APAC"], "value": [100, 200, 300]}).lazy()

        result_no_date = df_no_date.piot.cache_parquet(cache_path=tmp_path / "no_date", date_column=None, extra_partition_cols=["region"]).collect()

        assert result_no_date.shape == (3, 2)
        cache_files_no_date = list((tmp_path / "no_date").rglob("*.parquet"))
        assert len(cache_files_no_date) == 3

        # Verify these are in null directory
        for f in cache_files_no_date:
            assert "null" in str(f)
            assert "monthly" not in str(f)


@pytest.mark.parametrize(
    "partition_format",
    [
        None,  # use default for daily
        "theYear=$year/theMonth=$month/theDay=$day",  # custom format
    ],
)
def test_open_ended_filters_query_expected_partitions(tmp_path, partition_format):
    import datetime

    import polars as pl

    from polars_io_tools.io_sources.util import _storage_options_for

    aws_profile = None

    # Local filesystem path (unit test, no S3 dependencies)
    cache_root = str(tmp_path / "open_ended_cache")

    # Build a small LazyFrame with a date column
    lf = pl.DataFrame(
        {
            "date": pl.date_range(datetime.datetime(2024, 6, 1), datetime.datetime(2024, 6, 3), interval="1d", eager=True),
            "value": [11, 22, 33],
        }
    ).lazy()

    # First write with custom partition template; returned df should match input slice
    collected = lf.piot.cache_parquet(
        cache_path=cache_root,
        date_column="date",
        time_unit="daily",
        # partition_format="theYear=$year/theMonth=$month/theDay=$day",
        aws_profile=aws_profile,
        cache_mode=cpl.CacheMode.REBUILD,
    ).collect()
    assert collected.shape == (3, 2)
    assert collected["date"].to_list() == [datetime.date(2024, 6, 1), datetime.date(2024, 6, 2), datetime.date(2024, 6, 3)]
    assert collected["value"].to_list() == [11, 22, 33]

    # Append a row
    appended = (
        pl.DataFrame({"date": [datetime.datetime(2024, 6, 5)], "value": [44]})
        .with_columns(pl.col("date").cast(pl.Date))
        .lazy()
        .piot.cache_parquet(
            cache_path=cache_root,
            date_column="date",
            time_unit="daily",
            # partition_format="theYear=$year/theMonth=$month/theDay=$day",
            aws_profile=aws_profile,
        )
        .filter(pl.col("date") >= datetime.date(2024, 6, 4))
        .collect()
    )
    assert appended.shape == (1, 2)
    assert appended["date"].to_list() == [datetime.date(2024, 6, 5)]
    assert appended["value"].to_list() == [44]

    # Verify via scan_parquet
    polars_opts = _storage_options_for(cache_root, aws_profile=aws_profile).polars
    pl_cache_root = cache_root + "/**/*.parquet"
    out = pl.scan_parquet(pl_cache_root, storage_options=polars_opts).sort("date").collect()

    assert out.shape[0] == 4
    assert out["date"].min() == datetime.date(2024, 6, 1)
    assert out["date"].max() == datetime.date(2024, 6, 5)

    # Verify filters are handled properly (predicate applied to cached data)
    filtered = (
        lf.piot.cache_parquet(
            cache_path=cache_root,
            date_column="date",
            time_unit="daily",
            # partition_format="theYear=$year/theMonth=$month/theDay=$day",
            aws_profile=aws_profile,
        )
        .filter(pl.col("date") >= datetime.date(2024, 6, 2))
        .sort("date")
        .collect()
    )
    # One-sided lower bound should read cached partitions and include appended row
    assert filtered["date"].to_list() == [datetime.date(2024, 6, 2), datetime.date(2024, 6, 3), datetime.date(2024, 6, 5)]
    assert filtered["value"].to_list() == [22, 33, 44]


def test_one_sided_with_extra_cols_no_empty_writes(tmp_path):
    """
    One-sided lower bound on date with extra partition filters should:
    - Read upstream for matching data
    - Write only the actual partitions with rows
    - NOT create empty parquet files for partitions with no rows
    """
    df = pl.LazyFrame(
        {
            "date": [datetime.date(2024, 6, 1), datetime.date(2024, 6, 2), datetime.date(2024, 6, 5)],
            "region": ["US", "US", "EU"],
            "value": [10, 20, 50],
        }
    )

    # Seed cache
    base = tmp_path / "one_sided_extra"
    df.piot.cache_parquet(cache_path=base, date_column="date", time_unit="daily", extra_partition_cols="region").collect()

    # One-sided lower bound with extra column filter
    lf = df.piot.cache_parquet(cache_path=base, date_column="date", time_unit="daily", extra_partition_cols="region").filter(
        (pl.col("date") >= datetime.date(2024, 6, 2)) & (pl.col("region").is_in(["US", "EU", "APAC"]))
    )
    out = lf.sort(["date", "region"]).collect()

    # Verify only matching rows returned
    assert out.shape == (2, 3)
    assert out["date"].to_list() == [datetime.date(2024, 6, 2), datetime.date(2024, 6, 5)]
    assert out["region"].to_list() == ["US", "EU"]

    # Verify cache contents: only files with rows are present; no empty APAC files
    base_daily = base / "daily"
    assert (base_daily / "US" / "2024-06-02.parquet").exists()
    assert (base_daily / "EU" / "2024-06-05.parquet").exists()
    assert not (base_daily / "APAC" / "2024-06-02.parquet").exists()
    assert not (base_daily / "APAC" / "2024-06-05.parquet").exists()


def test_one_sided_upper_bound_with_extra_cols(tmp_path):
    """
    Upper-bound-only on date with extra partition filters should:
    - Read upstream for matching data
    - Write only the actual partitions with rows
    - NOT create empty parquet files for partitions with no rows
    """
    df = pl.LazyFrame(
        {
            "date": [datetime.date(2024, 6, 1), datetime.date(2024, 6, 2), datetime.date(2024, 6, 5)],
            "region": ["US", "US", "EU"],
            "value": [10, 20, 50],
        }
    )

    # Seed cache
    base = tmp_path / "one_sided_extra_upper"
    df.piot.cache_parquet(cache_path=base, date_column="date", time_unit="daily", extra_partition_cols="region").collect()

    # One-sided upper bound with extra column filter
    lf = df.piot.cache_parquet(cache_path=base, date_column="date", time_unit="daily", extra_partition_cols="region").filter(
        (pl.col("date") <= datetime.date(2024, 6, 2)) & (pl.col("region").is_in(["US", "EU", "APAC"]))
    )
    out = lf.sort(["date", "region"]).collect()

    # Verify only matching rows returned
    assert out.shape == (2, 3)
    assert out["date"].to_list() == [datetime.date(2024, 6, 1), datetime.date(2024, 6, 2)]
    assert out["region"].to_list() == ["US", "US"]

    # Verify cache contents: only files with rows are present; no empty APAC files
    base_daily = base / "daily"
    assert (base_daily / "US" / "2024-06-01.parquet").exists()
    assert (base_daily / "US" / "2024-06-02.parquet").exists()
    assert not (base_daily / "APAC" / "2024-06-01.parquet").exists()
    assert not (base_daily / "APAC" / "2024-06-02.parquet").exists()


def test_one_sided_lower_bound_reads_cache_and_upstream(tmp_path):
    """
    For one-sided lower bound queries, ensure we read cached partitions and only
    fetch upstream for partitions not already in cache.
    """
    # Seed cache with early dates
    df = pl.LazyFrame({"date": [datetime.date(2024, 6, 1), datetime.date(2024, 6, 2)], "val": [1, 2]})
    base = tmp_path / "one_sided_lower"
    df.piot.cache_parquet(base, "date", time_unit="daily").collect()

    # Upstream has a later date
    df2 = pl.LazyFrame({"date": [datetime.date(2024, 6, 5)], "val": [5]})

    # Query lower bound >= 2024-06-2 against combined source
    # Build combined source by concatenation
    out = (
        pl.concat([df2, df])
        .piot.cache_parquet(base, "date", time_unit="daily")
        .filter(pl.col("date") >= datetime.date(2024, 6, 2))
        .sort("date")
        .collect()
    )

    assert out["date"].to_list() == [datetime.date(2024, 6, 2), datetime.date(2024, 6, 5)]
    assert out["val"].to_list() == [2, 5]


def test_no_date_with_extra_cols_unconstrained_queries_upstream(tmp_path, capsys):
    """
    When no date_column but extra_partition_cols are provided, UNCONSTRAINED queries
    should always query upstream (new partition values might exist), filtering out
    already-cached partitions. Data is still correctly returned from cache.
    """
    df = pl.DataFrame({"region": ["US", "EU", "US"], "value": [1, 2, 3]}).lazy().piot.debug()

    # Cache without a date column but with extra partition cols
    cached = df.piot.cache_parquet(cache_path=tmp_path, date_column=None, extra_partition_cols="region")
    # First collect with a lower-bound-like filter on value
    out1 = cached.filter(pl.col("value") >= 2).sort(["region", "value"]).collect()
    assert out1.to_dict(as_series=False) == {"region": ["EU", "US"], "value": [2, 3]}

    # Capture debug output to ensure upstream was called
    first = capsys.readouterr().out
    assert "predicate=" in first, "expected upstream debug output on first run"

    # Second run: with partition columns, UNCONSTRAINED always queries upstream
    # (but filters out existing partitions, so no new data is written)
    out2 = cached.filter(pl.col("value") >= 2).sort(["region", "value"]).collect()
    assert_frame_equal(out2, out1)

    second = capsys.readouterr().out
    # With partition columns, upstream IS queried (with not_existing filter)
    assert "predicate=" in second, "UNCONSTRAINED with partition cols should query upstream"


def test_strftime_from_template_and_build_scan_paths(tmp_path):
    import polars as pl

    from polars_io_tools.io_sources.lazy_cache_parquet import _build_scan_paths, _strftime_from_template

    tmpl = "theYear=$year/theMonth=$month/theDay=$day"
    fmt = _strftime_from_template(tmpl)
    assert fmt == "theYear=%Y/theMonth=%m/theDay=%d"

    parts = pl.DataFrame({"region": ["US", "EU"], "date": [datetime.date(2024, 6, 1), datetime.date(2024, 6, 2)]})
    time_dir = str(tmp_path / "daily")
    # Custom template should be honored
    paths = _build_scan_paths(parts, time_dir, tmpl, "daily", "date", ["region"])
    assert paths == [
        f"{time_dir}/EU/theYear=2024/theMonth=06/theDay=02.parquet",
        f"{time_dir}/US/theYear=2024/theMonth=06/theDay=01.parquet",
    ]

    # Default template should fall back to YYYY-MM-DD
    paths_default = _build_scan_paths(parts, time_dir, None, "daily", "date", ["region"])
    assert paths_default == [
        f"{time_dir}/EU/2024-06-02.parquet",
        f"{time_dir}/US/2024-06-01.parquet",
    ]


def test_no_date_no_extra_first_writes_second_reads_cache(tmp_path, capsys):
    """No date, no extra partition columns: first run writes; second run reads cache only."""
    df = pl.DataFrame({"val": [1, 2, 3]}).lazy().piot.debug()

    cached = df.piot.cache_parquet(cache_path=tmp_path, date_column=None)

    out1 = cached.collect()
    assert out1.shape == (3, 1)

    first = capsys.readouterr().out
    assert "debug called" in first, "expected upstream debug on first run"

    out2 = cached.collect()
    assert_frame_equal(out2, out1)

    second = capsys.readouterr().out
    assert second.count("debug called") == 0, "second run should read from cache only"

    # Verify cache structure - should create a single file under null
    cache_files = list(tmp_path.rglob("*.parquet"))
    assert len(cache_files) == 1
    for f in cache_files:
        assert "null" in str(f)


def test_one_sided_lower_bound_upstream_predicate_shape(tmp_path, capsys):
    from datetime import date

    # Seed cache with early dates
    base_src = pl.DataFrame({"date": [date(2024, 6, 1), date(2024, 6, 2), date(2024, 6, 3)], "val": [1, 2, 3]}).lazy()
    cache_root = tmp_path / "one_sided_shape"
    base_src.piot.cache_parquet(cache_root, "date", time_unit="daily").collect()

    # Upstream with a later date; add debug to inspect upstream predicate
    upstream = pl.DataFrame({"date": [date(2024, 6, 5)], "val": [5]}).lazy().piot.debug()

    (upstream.piot.cache_parquet(cache_root, "date", time_unit="daily").filter(pl.col("date") >= date(2024, 6, 2)).collect())

    out = capsys.readouterr().out
    # Current behavior: one-sided read restricts upstream by original predicate only (>= lower bound)
    assert ">= (2024-06-02)" in out or ">= 2024-06-02" in out
    # And not a bounded is_between for the upstream path
    assert "is_between" not in out


def test_one_sided_upper_bound_upstream_predicate_shape(tmp_path, capsys):
    from datetime import date

    # Seed cache with later dates
    base_src = pl.DataFrame({"date": [date(2024, 6, 3), date(2024, 6, 4), date(2024, 6, 5)], "val": [3, 4, 5]}).lazy()
    cache_root = tmp_path / "one_sided_upper_shape"
    base_src.piot.cache_parquet(cache_root, "date", time_unit="daily").collect()

    # Upstream with an earlier date; add debug to inspect upstream predicate
    upstream = pl.DataFrame({"date": [date(2024, 6, 2)], "val": [2]}).lazy().piot.debug()

    (upstream.piot.cache_parquet(cache_root, "date", time_unit="daily").filter(pl.col("date") <= date(2024, 6, 4)).collect())

    out = capsys.readouterr().out
    # Current behavior: one-sided read restricts upstream by original predicate only (<= upper bound)
    assert "<= (2024-06-04)" in out or "<= 2024-06-04" in out
    assert "is_between" not in out


def test_one_sided_lower_bound_upstream_contains_not_existing_predicate(tmp_path, capsys):
    from datetime import date

    # Seed cache with early dates
    cache_root = tmp_path / "one_sided_not_pred"
    base_src = pl.DataFrame({"date": [date(2024, 6, 1), date(2024, 6, 2)], "val": [1, 2]}).lazy()
    base_src.piot.cache_parquet(cache_root, "date", time_unit="daily").collect()

    # Upstream has a later date; wrap with debug to inspect upstream plan
    upstream = pl.DataFrame({"date": [date(2024, 6, 5)], "val": [5]}).lazy().piot.debug()

    (upstream.piot.cache_parquet(cache_root, "date", time_unit="daily").filter(pl.col("date") >= date(2024, 6, 2)).collect())

    out = capsys.readouterr().out
    # Assert the upstream predicate contains the lower bound and a NOT over existing partitions
    assert ">= (2024-06-02)" in out or ">= 2024-06-02" in out
    assert "~(" in out or "!" in out, "expected NOT predicate to exclude existing partitions"
    assert 'col("date")' in out, "expected date column in NOT predicate"
    # Should reference at least one cached date in the NOT predicate
    assert "2024-06-01" in out or "2024-06-02" in out


def test_no_date_with_extra_unconstrained_queries_upstream_for_new_partitions(tmp_path):
    """No date, extra partition columns, unconstrained: queries upstream for new partitions.

    With partition columns, UNCONSTRAINED always queries upstream (new partition values
    might exist), filtering out already-cached partitions.
    """
    df_cache = pl.DataFrame({"region": ["US"], "val": [1]}).lazy()
    root = tmp_path / "no_date_extra_unconstrained"
    df_cache.piot.cache_parquet(root, date_column=None, extra_partition_cols="region").collect()

    # Query with a different upstream - since we have partition cols, it queries upstream
    # filtering out existing partitions (US). EU is new, so it gets written.
    df_up = pl.DataFrame({"region": ["EU"], "val": [2]}).lazy()
    out_first = df_up.piot.cache_parquet(root, date_column=None, extra_partition_cols="region").collect()
    # Both US (from cache) and EU (newly written from upstream) are returned
    assert sorted(out_first["val"].to_list()) == [1, 2]

    # Third call with same upstream - EU already cached, so no new writes
    out_second = df_up.piot.cache_parquet(root, date_column=None, extra_partition_cols="region").collect()
    assert sorted(out_second["val"].to_list()) == [1, 2]


def test_local_cache_parquet_daily_shared(tmp_path, caplog):
    cache_root = str(tmp_path)
    import logging

    caplog.set_level(logging.DEBUG, logger="polars_io_tools.io_sources.lazy_cache_parquet")
    collected, appended, filtered = exercise_daily_cache_parquet(
        cache_root=cache_root,
        aws_profile=None,
        partition_format="theYear=$year/theMonth=$month/theDay=$day",
    )
    assert collected.shape == (3, 2)
    assert collected["date"].to_list() == [datetime.date(2024, 6, 1), datetime.date(2024, 6, 2), datetime.date(2024, 6, 3)]
    assert collected["value"].to_list() == [11, 22, 33]

    assert appended.shape == (1, 2)
    assert appended["date"].to_list() == [datetime.date(2024, 6, 5)]
    assert appended["value"].to_list() == [44]

    assert filtered["date"].to_list() == [datetime.date(2024, 6, 2), datetime.date(2024, 6, 3)]
    assert filtered["value"].to_list() == [22, 33]
    # The appended query uses a bounded predicate; expect enumerated path selection
    logs = caplog.text
    assert "Cache scan uses" in logs and "enumerated" in logs


class TestHelperFunctions:
    """Tests for helper functions extracted during refactoring."""

    def test_compute_partitions_to_write_empty_existing(self):
        from polars_io_tools.io_sources.lazy_cache_parquet import Enumerability, PartitionInfo, _compute_partitions_to_write

        expected = pl.DataFrame({"date": [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)]})
        existing = pl.DataFrame(schema={"date": pl.Date})
        part_info = PartitionInfo(
            enumerability=Enumerability.FINITE,
            expected_parts_df=expected,
            existing_parts_df=existing,
            join_cols=["date"],
        )
        result = _compute_partitions_to_write(part_info, existing)
        assert_frame_equal(result, expected)

    def test_compute_partitions_to_write_some_existing(self):
        from polars_io_tools.io_sources.lazy_cache_parquet import Enumerability, PartitionInfo, _compute_partitions_to_write

        expected = pl.DataFrame({"date": [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2), datetime.date(2024, 1, 3)]})
        existing = pl.DataFrame({"date": [datetime.date(2024, 1, 1), datetime.date(2024, 1, 3)]})
        part_info = PartitionInfo(
            enumerability=Enumerability.FINITE,
            expected_parts_df=expected,
            existing_parts_df=existing,
            join_cols=["date"],
        )
        result = _compute_partitions_to_write(part_info, existing)
        assert result["date"].to_list() == [datetime.date(2024, 1, 2)]

    def test_compute_partitions_to_write_all_existing(self):
        from polars_io_tools.io_sources.lazy_cache_parquet import Enumerability, PartitionInfo, _compute_partitions_to_write

        expected = pl.DataFrame({"date": [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)]})
        existing = pl.DataFrame({"date": [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)]})
        part_info = PartitionInfo(
            enumerability=Enumerability.FINITE,
            expected_parts_df=expected,
            existing_parts_df=existing,
            join_cols=["date"],
        )
        result = _compute_partitions_to_write(part_info, existing)
        assert result.is_empty()

    def test_compute_partitions_to_write_no_join_cols(self):
        from polars_io_tools.io_sources.lazy_cache_parquet import Enumerability, PartitionInfo, _compute_partitions_to_write

        expected = pl.DataFrame({"val": [1]})
        existing = pl.DataFrame({"val": [1]})
        part_info = PartitionInfo(
            enumerability=Enumerability.FINITE,
            expected_parts_df=expected,
            existing_parts_df=existing,
            join_cols=[],
        )
        result = _compute_partitions_to_write(part_info, existing)
        assert_frame_equal(result, expected)

    def test_write_empty_parquet_files_sequentially(self, tmp_path):
        from polars_io_tools.io_sources.lazy_cache_parquet import _write_empty_parquet_files_sequentially

        paths = [str(tmp_path / "one.parquet"), str(tmp_path / "nested" / "two.parquet")]
        schema = pl.Schema({"value": pl.Int64})

        _write_empty_parquet_files_sequentially(
            paths,
            schema=schema,
            metadata={},
            storage_options={},
            credential_provider=None,
            write_kwargs={},
        )

        for path in paths:
            assert pl.read_parquet(path).schema == schema
            assert pl.read_parquet(path).height == 0

    def test_build_read_plan_falls_back_to_glob_for_many_paths(self):
        from polars_io_tools.io_sources.lazy_cache_parquet import (
            _MAX_ENUMERATED_SCAN_PATHS,
            Enumerability,
            PartitionInfo,
            _build_read_plan,
        )

        dates = [datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(_MAX_ENUMERATED_SCAN_PATHS + 1)]
        part_info = PartitionInfo(
            enumerability=Enumerability.FINITE,
            expected_parts_df=pl.DataFrame({"date": dates}),
            existing_parts_df=pl.DataFrame({"date": dates}),
            join_cols=["date"],
        )

        read_plan = _build_read_plan(
            partition_info=part_info,
            date_column="date",
            predicate=pl.col("date") >= dates[0],
            time_unit_dir="file:///tmp/cache/daily",
            template_for_metadata=None,
            effective_time_unit="daily",
        )

        assert read_plan.use_paths is None

    def test_build_final_scan_cache_only(self):
        from polars_io_tools.io_sources.lazy_cache_parquet import _build_final_scan

        cache_df = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]}).lazy()
        result = _build_final_scan(cache_df, None, None, None).collect()
        assert result.shape == (3, 2)

    def test_build_final_scan_with_predicate(self):
        from polars_io_tools.io_sources.lazy_cache_parquet import _build_final_scan

        cache_df = pl.DataFrame({"x": [1, 2, 3]}).lazy()
        result = _build_final_scan(cache_df, pl.col("x") > 1, None, None).collect()
        assert result["x"].to_list() == [2, 3]

    def test_build_final_scan_with_columns(self):
        from polars_io_tools.io_sources.lazy_cache_parquet import _build_final_scan

        cache_df = pl.DataFrame({"x": [1, 2], "y": ["a", "b"], "z": [10, 20]}).lazy()
        result = _build_final_scan(cache_df, None, ["x", "y"], None).collect()
        assert result.columns == ["x", "y"]

    def test_build_final_scan_with_limit(self):
        from polars_io_tools.io_sources.lazy_cache_parquet import _build_final_scan

        cache_df = pl.DataFrame({"x": [1, 2, 3, 4, 5]}).lazy()
        result = _build_final_scan(cache_df, None, None, 2).collect()
        assert len(result) == 2


class TestComputeClippedExpectedPartitions:
    """Tests for _compute_clipped_expected_partitions function."""

    def test_returns_none_for_empty_interval(self):
        import portion

        from polars_io_tools.io_sources.lazy_cache_parquet import _compute_clipped_expected_partitions

        existing = pl.DataFrame({"date": [datetime.date(2024, 6, 1)]})
        result = _compute_clipped_expected_partitions(
            predicate_interval=portion.empty(),
            existing_parts_df=existing,
            date_column="date",
            extra_cols=[],
            time_unit="daily",
            schema={"date": pl.Date},
        )
        assert result is None

    def test_returns_none_for_fully_bounded_interval(self):
        import portion

        from polars_io_tools.io_sources.lazy_cache_parquet import _compute_clipped_expected_partitions

        existing = pl.DataFrame({"date": [datetime.date(2024, 6, 1), datetime.date(2024, 6, 5)]})
        # Fully bounded: [2024-06-01, 2024-06-05]
        result = _compute_clipped_expected_partitions(
            predicate_interval=portion.closed(datetime.date(2024, 6, 1), datetime.date(2024, 6, 5)),
            existing_parts_df=existing,
            date_column="date",
            extra_cols=[],
            time_unit="daily",
            schema={"date": pl.Date},
        )
        assert result is None

    def test_returns_none_for_fully_unbounded_interval(self):
        import portion

        from polars_io_tools.io_sources.lazy_cache_parquet import _compute_clipped_expected_partitions

        existing = pl.DataFrame({"date": [datetime.date(2024, 6, 1)]})
        # Fully unbounded: (-inf, +inf)
        result = _compute_clipped_expected_partitions(
            predicate_interval=portion.open(-portion.inf, portion.inf),
            existing_parts_df=existing,
            date_column="date",
            extra_cols=[],
            time_unit="daily",
            schema={"date": pl.Date},
        )
        assert result is None

    def test_returns_none_for_empty_existing_partitions(self):
        import portion

        from polars_io_tools.io_sources.lazy_cache_parquet import _compute_clipped_expected_partitions

        existing = pl.DataFrame(schema={"date": pl.Date})
        result = _compute_clipped_expected_partitions(
            predicate_interval=portion.closedopen(datetime.date(2024, 6, 1), portion.inf),
            existing_parts_df=existing,
            date_column="date",
            extra_cols=[],
            time_unit="daily",
            schema={"date": pl.Date},
        )
        assert result is None

    def test_returns_none_for_missing_date_column(self):
        import portion

        from polars_io_tools.io_sources.lazy_cache_parquet import _compute_clipped_expected_partitions

        existing = pl.DataFrame({"other_col": [1, 2, 3]})
        result = _compute_clipped_expected_partitions(
            predicate_interval=portion.closedopen(datetime.date(2024, 6, 1), portion.inf),
            existing_parts_df=existing,
            date_column="date",
            extra_cols=[],
            time_unit="daily",
            schema={"date": pl.Date},
        )
        assert result is None

    def test_returns_none_for_invalid_clipped_range(self):
        import portion

        from polars_io_tools.io_sources.lazy_cache_parquet import _compute_clipped_expected_partitions

        # existing_max is 2024-06-01, but predicate_min is 2024-06-10
        # clipped_min (2024-06-10) > clipped_max (2024-06-01) -> invalid
        existing = pl.DataFrame({"date": [datetime.date(2024, 6, 1)]})
        result = _compute_clipped_expected_partitions(
            predicate_interval=portion.closedopen(datetime.date(2024, 6, 10), portion.inf),
            existing_parts_df=existing,
            date_column="date",
            extra_cols=[],
            time_unit="daily",
            schema={"date": pl.Date},
        )
        assert result is None

    def test_unbounded_above_clips_to_existing_max(self):
        import portion

        from polars_io_tools.io_sources.lazy_cache_parquet import _compute_clipped_expected_partitions

        # date >= 2024-06-02, existing max is 2024-06-05
        # Should enumerate 2024-06-02, 03, 04, 05
        existing = pl.DataFrame({"date": [datetime.date(2024, 6, 1), datetime.date(2024, 6, 5)]})
        result = _compute_clipped_expected_partitions(
            predicate_interval=portion.closedopen(datetime.date(2024, 6, 2), portion.inf),
            existing_parts_df=existing,
            date_column="date",
            extra_cols=[],
            time_unit="daily",
            schema={"date": pl.Date},
        )
        assert result is not None
        dates = sorted(result["date"].to_list())
        assert dates == [
            datetime.date(2024, 6, 2),
            datetime.date(2024, 6, 3),
            datetime.date(2024, 6, 4),
            datetime.date(2024, 6, 5),
        ]

    def test_unbounded_below_clips_to_existing_min(self):
        import portion

        from polars_io_tools.io_sources.lazy_cache_parquet import _compute_clipped_expected_partitions

        # date <= 2024-06-04, existing min is 2024-06-02
        # Should enumerate 2024-06-02, 03, 04
        existing = pl.DataFrame({"date": [datetime.date(2024, 6, 2), datetime.date(2024, 6, 5)]})
        result = _compute_clipped_expected_partitions(
            predicate_interval=portion.openclosed(-portion.inf, datetime.date(2024, 6, 4)),
            existing_parts_df=existing,
            date_column="date",
            extra_cols=[],
            time_unit="daily",
            schema={"date": pl.Date},
        )
        assert result is not None
        dates = sorted(result["date"].to_list())
        assert dates == [
            datetime.date(2024, 6, 2),
            datetime.date(2024, 6, 3),
            datetime.date(2024, 6, 4),
        ]

    def test_with_extra_cols_cross_product(self):
        import portion

        from polars_io_tools.io_sources.lazy_cache_parquet import _compute_clipped_expected_partitions

        # date >= 2024-06-01 with extra col "region" having values US, EU
        # existing max is 2024-06-02
        existing = pl.DataFrame(
            {
                "date": [datetime.date(2024, 6, 1), datetime.date(2024, 6, 2)],
                "region": ["US", "EU"],
            }
        )
        result = _compute_clipped_expected_partitions(
            predicate_interval=portion.closedopen(datetime.date(2024, 6, 1), portion.inf),
            existing_parts_df=existing,
            date_column="date",
            extra_cols=["region"],
            time_unit="daily",
            schema={"date": pl.Date, "region": pl.String},
        )
        assert result is not None
        # Should have 2 dates x 2 regions = 4 rows
        assert len(result) == 4
        assert set(result["region"].to_list()) == {"US", "EU"}
        assert set(result["date"].to_list()) == {datetime.date(2024, 6, 1), datetime.date(2024, 6, 2)}

    def test_with_datetime_in_interval(self):
        import portion

        from polars_io_tools.io_sources.lazy_cache_parquet import _compute_clipped_expected_partitions

        # Interval uses datetime, should convert to date
        existing = pl.DataFrame({"date": [datetime.date(2024, 6, 1), datetime.date(2024, 6, 3)]})
        result = _compute_clipped_expected_partitions(
            predicate_interval=portion.closedopen(datetime.datetime(2024, 6, 2, 10, 30), portion.inf),
            existing_parts_df=existing,
            date_column="date",
            extra_cols=[],
            time_unit="daily",
            schema={"date": pl.Date},
        )
        assert result is not None
        dates = sorted(result["date"].to_list())
        assert dates == [datetime.date(2024, 6, 2), datetime.date(2024, 6, 3)]

    def test_monthly_time_unit(self):
        import portion

        from polars_io_tools.io_sources.lazy_cache_parquet import _compute_clipped_expected_partitions

        # date >= 2024-02-15, existing max is 2024-04-10
        # With monthly widening: 2024-02-01, 2024-03-01, 2024-04-01
        existing = pl.DataFrame({"date": [datetime.date(2024, 1, 15), datetime.date(2024, 4, 10)]})
        result = _compute_clipped_expected_partitions(
            predicate_interval=portion.closedopen(datetime.date(2024, 2, 15), portion.inf),
            existing_parts_df=existing,
            date_column="date",
            extra_cols=[],
            time_unit="monthly",
            schema={"date": pl.Date},
        )
        assert result is not None
        dates = sorted(result["date"].to_list())
        assert dates == [
            datetime.date(2024, 2, 1),
            datetime.date(2024, 3, 1),
            datetime.date(2024, 4, 1),
        ]

    def test_yearly_time_unit(self):
        import portion

        from polars_io_tools.io_sources.lazy_cache_parquet import _compute_clipped_expected_partitions

        # date >= 2022-06-15, existing max is 2024-03-10
        # With yearly widening: 2022-01-01, 2023-01-01, 2024-01-01
        existing = pl.DataFrame({"date": [datetime.date(2021, 1, 1), datetime.date(2024, 3, 10)]})
        result = _compute_clipped_expected_partitions(
            predicate_interval=portion.closedopen(datetime.date(2022, 6, 15), portion.inf),
            existing_parts_df=existing,
            date_column="date",
            extra_cols=[],
            time_unit="yearly",
            schema={"date": pl.Date},
        )
        assert result is not None
        dates = sorted(result["date"].to_list())
        assert dates == [
            datetime.date(2022, 1, 1),
            datetime.date(2023, 1, 1),
            datetime.date(2024, 1, 1),
        ]

    def test_single_date_existing(self):
        import portion

        from polars_io_tools.io_sources.lazy_cache_parquet import _compute_clipped_expected_partitions

        # Only one existing date, predicate starts at same date
        existing = pl.DataFrame({"date": [datetime.date(2024, 6, 1)]})
        result = _compute_clipped_expected_partitions(
            predicate_interval=portion.closedopen(datetime.date(2024, 6, 1), portion.inf),
            existing_parts_df=existing,
            date_column="date",
            extra_cols=[],
            time_unit="daily",
            schema={"date": pl.Date},
        )
        assert result is not None
        assert result["date"].to_list() == [datetime.date(2024, 6, 1)]


class TestUnboundedRangeGapFilling:
    """Tests for the new behavior: filling gaps within existing bounds for unbounded ranges."""

    def test_unbounded_above_fills_gaps_up_to_existing_max(self, tmp_path):
        """
        For unbounded above (date >= X): Fill empty partitions from X to max(existing).
        Beyond max(existing), don't create empty partitions.
        """
        # Seed cache with data at dates 1, 3, 5 (gap at 2 and 4)
        df = pl.LazyFrame(
            {
                "date": [datetime.date(2024, 6, 1), datetime.date(2024, 6, 3), datetime.date(2024, 6, 5)],
                "val": [1, 3, 5],
            }
        )
        base = tmp_path / "unbounded_above_gaps"
        df.piot.cache_parquet(base, "date", time_unit="daily").collect()

        # Now query with unbounded above: date >= 2024-06-02
        # This should fill gaps at 2024-06-02 and 2024-06-04 (within existing range)
        # But NOT at 2024-06-06 or beyond (outside existing range)
        df.piot.cache_parquet(base, "date", time_unit="daily").filter(pl.col("date") >= datetime.date(2024, 6, 2)).collect()

        base_daily = base / "daily"

        # Original data partitions should exist
        assert (base_daily / "2024-06-01.parquet").exists()
        assert (base_daily / "2024-06-03.parquet").exists()
        assert (base_daily / "2024-06-05.parquet").exists()

        # Gaps WITHIN existing range (2024-06-01 to 2024-06-05) should be filled with empty files
        assert (base_daily / "2024-06-02.parquet").exists()
        assert (base_daily / "2024-06-04.parquet").exists()
        # Verify they are empty
        assert pl.read_parquet(base_daily / "2024-06-02.parquet").is_empty()
        assert pl.read_parquet(base_daily / "2024-06-04.parquet").is_empty()

        # Partitions BEYOND existing max should NOT be created
        assert not (base_daily / "2024-06-06.parquet").exists()
        assert not (base_daily / "2024-06-07.parquet").exists()

    def test_unbounded_below_fills_gaps_down_to_existing_min(self, tmp_path):
        """
        For unbounded below (date <= Y): Fill empty partitions from min(existing) to Y.
        Before min(existing), don't create empty partitions.
        """
        # Seed cache with data at dates 3, 5, 7 (gap at 4 and 6)
        df = pl.LazyFrame(
            {
                "date": [datetime.date(2024, 6, 3), datetime.date(2024, 6, 5), datetime.date(2024, 6, 7)],
                "val": [3, 5, 7],
            }
        )
        base = tmp_path / "unbounded_below_gaps"
        df.piot.cache_parquet(base, "date", time_unit="daily").collect()

        # Now query with unbounded below: date <= 2024-06-06
        # This should fill gaps at 2024-06-04 (within existing range)
        # But NOT at 2024-06-01, 2024-06-02 (before existing min)
        df.piot.cache_parquet(base, "date", time_unit="daily").filter(pl.col("date") <= datetime.date(2024, 6, 6)).collect()

        base_daily = base / "daily"

        # Original data partitions should exist
        assert (base_daily / "2024-06-03.parquet").exists()
        assert (base_daily / "2024-06-05.parquet").exists()
        assert (base_daily / "2024-06-07.parquet").exists()

        # Gaps WITHIN the clipped range (2024-06-03 to 2024-06-06) should be filled
        assert (base_daily / "2024-06-04.parquet").exists()
        assert (base_daily / "2024-06-06.parquet").exists()
        # Verify they are empty
        assert pl.read_parquet(base_daily / "2024-06-04.parquet").is_empty()
        assert pl.read_parquet(base_daily / "2024-06-06.parquet").is_empty()

        # Partitions BEFORE existing min should NOT be created
        assert not (base_daily / "2024-06-01.parquet").exists()
        assert not (base_daily / "2024-06-02.parquet").exists()

    def test_unbounded_with_extra_cols_fills_gaps(self, tmp_path):
        """
        Unbounded queries with extra partition columns should also fill gaps within existing bounds.
        """
        # Seed cache with data at dates 1, 3 for region US only
        df = pl.LazyFrame(
            {
                "date": [datetime.date(2024, 6, 1), datetime.date(2024, 6, 3)],
                "region": ["US", "US"],
                "val": [1, 3],
            }
        )
        base = tmp_path / "unbounded_extra_cols"
        df.piot.cache_parquet(base, "date", time_unit="daily", extra_partition_cols="region").collect()

        # Query with unbounded above: date >= 2024-06-01
        df.piot.cache_parquet(base, "date", time_unit="daily", extra_partition_cols="region").filter(
            pl.col("date") >= datetime.date(2024, 6, 1)
        ).collect()

        base_daily = base / "daily"

        # Original partitions should exist
        assert (base_daily / "US" / "2024-06-01.parquet").exists()
        assert (base_daily / "US" / "2024-06-03.parquet").exists()

        # Gap within existing range should be filled
        assert (base_daily / "US" / "2024-06-02.parquet").exists()
        assert pl.read_parquet(base_daily / "US" / "2024-06-02.parquet").is_empty()

        # Beyond existing max should NOT be created
        assert not (base_daily / "US" / "2024-06-04.parquet").exists()


class TestRebuildBehavior:
    """Tests for CacheMode.REBUILD behavior.

    REBUILD should:
    - Ignore cache on reads (query upstream for all data matching predicate)
    - Write fresh data to cache (overwriting existing partition files)
    - Preserve partitions outside the current query scope (no cache deletion)
    """

    def test_rebuild_preserves_other_partitions(self, tmp_path):
        """REBUILD should NOT delete partitions outside the queried range."""
        # First, create cache with data for multiple months
        df_multi = pl.DataFrame(
            {
                "date": [
                    datetime.date(2024, 1, 15),
                    datetime.date(2024, 2, 15),
                    datetime.date(2024, 3, 15),
                ],
                "value": [100, 200, 300],
            }
        ).lazy()
        df_multi.piot.cache_parquet(cache_path=tmp_path, date_column="date", time_unit="monthly").collect()

        base = tmp_path / "monthly"
        assert (base / "2024-01.parquet").exists()
        assert (base / "2024-02.parquet").exists()
        assert (base / "2024-03.parquet").exists()

        # Now REBUILD with different data, but only for February
        df_rebuild = pl.DataFrame({"date": [datetime.date(2024, 2, 20)], "value": [999]}).lazy()
        df_rebuild.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="monthly",
            cache_mode=CacheMode.REBUILD,
        ).filter(pl.col("date").is_between(datetime.date(2024, 2, 1), datetime.date(2024, 2, 28))).collect()

        # January and March partitions should still exist with original data
        jan_data = pl.read_parquet(base / "2024-01.parquet")
        mar_data = pl.read_parquet(base / "2024-03.parquet")
        assert jan_data["value"].to_list() == [100]
        assert mar_data["value"].to_list() == [300]

        # February should have the new data
        feb_data = pl.read_parquet(base / "2024-02.parquet")
        assert feb_data["value"].to_list() == [999]

    def test_rebuild_overwrites_existing_partition(self, tmp_path):
        """REBUILD should overwrite existing partition files with fresh upstream data."""
        # First cache
        df1 = pl.DataFrame({"date": [datetime.date(2024, 5, 10)], "region": ["US"], "value": [100]}).lazy()
        df1.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="daily",
            extra_partition_cols="region",
        ).collect()

        base_daily = tmp_path / "daily"
        assert pl.read_parquet(base_daily / "US" / "2024-05-10.parquet")["value"].to_list() == [100]

        # REBUILD with different data for same partition
        df2 = pl.DataFrame({"date": [datetime.date(2024, 5, 10)], "region": ["US"], "value": [999]}).lazy()
        df2.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="daily",
            extra_partition_cols="region",
            cache_mode=CacheMode.REBUILD,
        ).filter((pl.col("date") == datetime.date(2024, 5, 10)) & (pl.col("region") == "US")).collect()

        # Should have the new value
        assert pl.read_parquet(base_daily / "US" / "2024-05-10.parquet")["value"].to_list() == [999]

    def test_rebuild_writes_empty_files_for_missing_data(self, tmp_path):
        """REBUILD should write empty files for expected partitions with no upstream data."""
        # Source has data only for some dates
        df = pl.DataFrame({"date": [datetime.date(2024, 6, 1), datetime.date(2024, 6, 3)], "value": [10, 30]}).lazy()

        # REBUILD with a query that spans more dates than have data
        df.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="daily",
            cache_mode=CacheMode.REBUILD,
        ).filter(pl.col("date").is_between(datetime.date(2024, 6, 1), datetime.date(2024, 6, 3))).collect()

        base_daily = tmp_path / "daily"

        # Partitions with data
        assert not pl.read_parquet(base_daily / "2024-06-01.parquet").is_empty()
        assert not pl.read_parquet(base_daily / "2024-06-03.parquet").is_empty()

        # Empty partition for date with no data
        assert (base_daily / "2024-06-02.parquet").exists()
        assert pl.read_parquet(base_daily / "2024-06-02.parquet").is_empty()

    def test_rebuild_overwrites_previously_empty_partition(self, tmp_path):
        """REBUILD should overwrite a previously empty partition if data now exists."""
        # First, create cache where 2024-06-02 is empty
        df1 = pl.DataFrame({"date": [datetime.date(2024, 6, 1), datetime.date(2024, 6, 3)], "value": [10, 30]}).lazy()
        df1.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="daily",
        ).filter(pl.col("date").is_between(datetime.date(2024, 6, 1), datetime.date(2024, 6, 3))).collect()

        base_daily = tmp_path / "daily"
        assert pl.read_parquet(base_daily / "2024-06-02.parquet").is_empty()

        # REBUILD with data that now includes 2024-06-02
        df2 = pl.DataFrame(
            {
                "date": [datetime.date(2024, 6, 1), datetime.date(2024, 6, 2), datetime.date(2024, 6, 3)],
                "value": [10, 20, 30],
            }
        ).lazy()
        df2.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="daily",
            cache_mode=CacheMode.REBUILD,
        ).filter(pl.col("date").is_between(datetime.date(2024, 6, 1), datetime.date(2024, 6, 3))).collect()

        # 2024-06-02 should now have data
        assert pl.read_parquet(base_daily / "2024-06-02.parquet")["value"].to_list() == [20]

    def test_rebuild_queries_upstream_not_cache(self, tmp_path, capsys):
        """REBUILD should always query upstream, not read from cache."""
        # Create initial cache
        df_cached = pl.DataFrame({"date": [datetime.date(2024, 7, 1)], "value": [100]}).lazy()
        df_cached.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="daily",
        ).collect()

        # REBUILD with debug to verify upstream is called
        df_upstream = pl.DataFrame({"date": [datetime.date(2024, 7, 1)], "value": [999]}).lazy().piot.debug()

        result = (
            df_upstream.piot.cache_parquet(
                cache_path=tmp_path,
                date_column="date",
                time_unit="daily",
                cache_mode=CacheMode.REBUILD,
            )
            .filter(pl.col("date") == datetime.date(2024, 7, 1))
            .collect()
        )

        captured = capsys.readouterr().out
        # Debug should have been called, indicating upstream was queried
        assert "debug called" in captured
        # Result should have the new value from upstream
        assert result["value"].to_list() == [999]

    def test_rebuild_with_extra_cols_preserves_other_regions(self, tmp_path):
        """REBUILD with extra partition cols should preserve partitions for other column values."""
        # Create cache with multiple regions
        df_multi = pl.DataFrame(
            {
                "date": [datetime.date(2024, 8, 1), datetime.date(2024, 8, 1)],
                "region": ["US", "EU"],
                "value": [100, 200],
            }
        ).lazy()
        df_multi.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="daily",
            extra_partition_cols="region",
        ).collect()

        base_daily = tmp_path / "daily"
        assert pl.read_parquet(base_daily / "US" / "2024-08-01.parquet")["value"].to_list() == [100]
        assert pl.read_parquet(base_daily / "EU" / "2024-08-01.parquet")["value"].to_list() == [200]

        # REBUILD only US region
        df_rebuild = pl.DataFrame({"date": [datetime.date(2024, 8, 1)], "region": ["US"], "value": [999]}).lazy()
        df_rebuild.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="daily",
            extra_partition_cols="region",
            cache_mode=CacheMode.REBUILD,
        ).filter((pl.col("date") == datetime.date(2024, 8, 1)) & (pl.col("region") == "US")).collect()

        # US should have new value, EU should be unchanged
        assert pl.read_parquet(base_daily / "US" / "2024-08-01.parquet")["value"].to_list() == [999]
        assert pl.read_parquet(base_daily / "EU" / "2024-08-01.parquet")["value"].to_list() == [200]

    def test_rebuild_no_date_column(self, tmp_path):
        """REBUILD should work correctly without a date column."""
        # Create initial cache
        df1 = pl.DataFrame({"region": ["US", "EU"], "value": [100, 200]}).lazy()
        df1.piot.cache_parquet(
            cache_path=tmp_path,
            date_column=None,
            extra_partition_cols="region",
        ).collect()

        base_null = tmp_path / "null"
        assert pl.read_parquet(base_null / "US.parquet")["value"].to_list() == [100]
        assert pl.read_parquet(base_null / "EU.parquet")["value"].to_list() == [200]

        # REBUILD US only
        df2 = pl.DataFrame({"region": ["US"], "value": [999]}).lazy()
        df2.piot.cache_parquet(
            cache_path=tmp_path,
            date_column=None,
            extra_partition_cols="region",
            cache_mode=CacheMode.REBUILD,
        ).filter(pl.col("region") == "US").collect()

        # US should have new value
        assert pl.read_parquet(base_null / "US.parquet")["value"].to_list() == [999]
        # EU should be unchanged
        assert pl.read_parquet(base_null / "EU.parquet")["value"].to_list() == [200]

    def test_rebuild_empty_to_data(self, tmp_path):
        """REBUILD should correctly overwrite empty partitions when data becomes available."""
        # First query creates empty partition
        df1 = pl.DataFrame({"date": [], "value": []}, schema={"date": pl.Date, "value": pl.Int64}).lazy()
        df1.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="daily",
            cache_mode=CacheMode.REBUILD,
        ).filter(pl.col("date") == datetime.date(2024, 9, 1)).collect()

        base_daily = tmp_path / "daily"
        assert (base_daily / "2024-09-01.parquet").exists()
        assert pl.read_parquet(base_daily / "2024-09-01.parquet").is_empty()

        # REBUILD with actual data
        df2 = pl.DataFrame({"date": [datetime.date(2024, 9, 1)], "value": [42]}).lazy()
        df2.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="daily",
            cache_mode=CacheMode.REBUILD,
        ).filter(pl.col("date") == datetime.date(2024, 9, 1)).collect()

        assert pl.read_parquet(base_daily / "2024-09-01.parquet")["value"].to_list() == [42]

    def test_rebuild_data_to_empty(self, tmp_path):
        """REBUILD should correctly overwrite non-empty partitions when data disappears."""
        # First create with data
        df1 = pl.DataFrame({"date": [datetime.date(2024, 10, 1)], "value": [42]}).lazy()
        df1.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="daily",
        ).collect()

        base_daily = tmp_path / "daily"
        assert pl.read_parquet(base_daily / "2024-10-01.parquet")["value"].to_list() == [42]

        # REBUILD with empty source for that partition
        df2 = pl.DataFrame({"date": [], "value": []}, schema={"date": pl.Date, "value": pl.Int64}).lazy()
        df2.piot.cache_parquet(
            cache_path=tmp_path,
            date_column="date",
            time_unit="daily",
            cache_mode=CacheMode.REBUILD,
        ).filter(pl.col("date") == datetime.date(2024, 10, 1)).collect()

        # Partition should now be empty
        assert pl.read_parquet(base_daily / "2024-10-01.parquet").is_empty()


class TestCacheModePydantic:
    """Test CacheMode enum pydantic integration for ergonomic model field usage."""

    def test_cachemode_validate_from_string(self):
        """CacheMode.validate should accept string enum names."""
        assert CacheMode.validate("CACHE") == CacheMode.CACHE
        assert CacheMode.validate("IGNORE") == CacheMode.IGNORE
        assert CacheMode.validate("REBUILD") == CacheMode.REBUILD

    def test_cachemode_validate_from_int(self):
        """CacheMode.validate should accept integer values."""
        assert CacheMode.validate(1) == CacheMode.CACHE
        assert CacheMode.validate(2) == CacheMode.IGNORE
        assert CacheMode.validate(3) == CacheMode.REBUILD

    def test_cachemode_validate_from_enum(self):
        """CacheMode.validate should accept CacheMode instances."""
        assert CacheMode.validate(CacheMode.CACHE) == CacheMode.CACHE
        assert CacheMode.validate(CacheMode.REBUILD) == CacheMode.REBUILD

    def test_cachemode_validate_invalid_string(self):
        """CacheMode.validate should raise KeyError for invalid string names."""
        with pytest.raises(KeyError):
            CacheMode.validate("INVALID")

    def test_cachemode_validate_invalid_int(self):
        """CacheMode.validate should raise ValueError for invalid integer values."""
        with pytest.raises(ValueError):
            CacheMode.validate(999)

    def test_cachemode_validate_invalid_type(self):
        """CacheMode.validate should raise ValueError for invalid types."""
        with pytest.raises(ValueError, match="Cannot convert value to CacheMode"):
            CacheMode.validate(3.14)

    def test_cachemode_in_pydantic_model_from_string(self):
        """CacheMode should work in a pydantic model when passed as a string."""
        from pydantic import BaseModel

        class MyModel(BaseModel):
            cache_mode: CacheMode

        # Should accept string
        model = MyModel(cache_mode="REBUILD")
        assert model.cache_mode == CacheMode.REBUILD

        model2 = MyModel(cache_mode="CACHE")
        assert model2.cache_mode == CacheMode.CACHE

    def test_cachemode_in_pydantic_model_from_int(self):
        """CacheMode should work in a pydantic model when passed as an int."""
        from pydantic import BaseModel

        class MyModel(BaseModel):
            cache_mode: CacheMode

        # Should accept int
        model = MyModel(cache_mode=1)
        assert model.cache_mode == CacheMode.CACHE

        model2 = MyModel(cache_mode=3)
        assert model2.cache_mode == CacheMode.REBUILD

    def test_cachemode_in_pydantic_model_from_enum(self):
        """CacheMode should work in a pydantic model when passed as an enum."""
        from pydantic import BaseModel

        class MyModel(BaseModel):
            cache_mode: CacheMode

        model = MyModel(cache_mode=CacheMode.IGNORE)
        assert model.cache_mode == CacheMode.IGNORE

    def test_cachemode_pydantic_json_serialization(self):
        """CacheMode should serialize to string name in JSON."""
        from pydantic import BaseModel

        class MyModel(BaseModel):
            cache_mode: CacheMode

        model = MyModel(cache_mode=CacheMode.REBUILD)
        json_output = model.model_dump_json()
        assert '"cache_mode":"REBUILD"' in json_output

    def test_cachemode_pydantic_model_dump(self):
        """CacheMode should dump correctly in python and json mode."""
        from pydantic import BaseModel

        class MyModel(BaseModel):
            cache_mode: CacheMode

        model = MyModel(cache_mode="CACHE")

        # Python mode should return the enum
        python_dump = model.model_dump(mode="python")
        assert python_dump["cache_mode"] == CacheMode.CACHE

        # JSON mode should return the string name
        json_dump = model.model_dump(mode="json")
        assert json_dump["cache_mode"] == "CACHE"

    def test_cachemode_pydantic_json_schema(self):
        """CacheMode should generate proper JSON schema with string enum values."""
        from pydantic import BaseModel

        class MyModel(BaseModel):
            cache_mode: CacheMode

        schema = MyModel.model_json_schema()
        # The schema should show CacheMode as a string enum with the member names
        assert "CacheMode" in str(schema) or "cache_mode" in str(schema)
        # Check that the enum values are present in schema
        cache_mode_props = schema.get("properties", {}).get("cache_mode", {})
        assert "enum" in cache_mode_props or "$ref" in cache_mode_props

    def test_cachemode_with_default_value(self):
        """CacheMode should work with default values in pydantic models."""
        from pydantic import BaseModel

        class MyModel(BaseModel):
            cache_mode: CacheMode = CacheMode.CACHE

        # Default value
        model = MyModel()
        assert model.cache_mode == CacheMode.CACHE

        # Override with string
        model2 = MyModel(cache_mode="REBUILD")
        assert model2.cache_mode == CacheMode.REBUILD


class TestWriteBoundingColumns:
    """Opt-in ``write_bounding_columns`` kwarg on ``cache_parquet``.

    Small SYNTHETIC LazyFrames (no real data sources). Validates default-off identity,
    bounded-write restriction, read-back correctness, that the kwarg is permitted in any
    cache mode, degrade behavior when the predicate cannot be restricted, and the pinned
    empty-subset case.
    """

    @staticmethod
    def _make_source():
        """Five rows, all in 2023-01 (one monthly partition). ``x`` and ``y`` are NON-partition columns."""
        return pl.DataFrame(
            {
                "date": [datetime.date(2023, 1, 10)] * 5,
                "x": [1, 2, 3, 4, 5],
                "y": [100, 200, 300, 400, 500],
                "value": [10, 20, 30, 40, 50],
            }
        ).lazy()

    @staticmethod
    def _read_partition(root):
        return pl.read_parquet(os.path.join(root, "monthly", "2023-01.parquet")).sort("x")

    @staticmethod
    def _walk_partition_frames(root):
        """Map of relative parquet path -> sorted frame, for content comparison."""
        out = {}
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if f.endswith(".parquet"):
                    p = os.path.join(dirpath, f)
                    out[os.path.relpath(p, root)] = pl.read_parquet(p).sort("x")
        return out

    def test_default_off_is_identical_to_not_passing(self, tmp_path):
        """write_bounding_columns=None must produce the same files/rows as not passing the kwarg at all."""
        d_omitted = tmp_path / "omitted"
        d_none = tmp_path / "none"

        cache_parquet(self._make_source, cache_path=d_omitted, date_column="date").collect()
        cache_parquet(self._make_source, cache_path=d_none, date_column="date", write_bounding_columns=None).collect()

        frames_omitted = self._walk_partition_frames(d_omitted)
        frames_none = self._walk_partition_frames(d_none)

        assert set(frames_omitted) == set(frames_none)
        assert frames_omitted  # sanity: something was written
        for rel, frame in frames_omitted.items():
            assert frame.equals(frames_none[rel]), f"mismatch in {rel}"

    def test_bounded_write_restricts_on_disk_rows(self, tmp_path):
        """With write_bounding_columns=['x'] and a pushed is_in([1,2]), only the subset is written to disk."""
        subset = [1, 2]
        d = tmp_path / "bounded"
        cache_parquet(
            self._make_source,
            cache_path=d,
            date_column="date",
            cache_mode=CacheMode.REBUILD,
            write_bounding_columns=["x"],
        ).filter(pl.col("x").is_in(subset)).collect()

        on_disk = self._read_partition(d)
        assert on_disk["x"].to_list() == subset

    def test_unbounded_write_keeps_all_rows(self, tmp_path):
        """Control: without write_bounding_columns, the same query writes the FULL partition to disk."""
        d = tmp_path / "unbounded"
        cache_parquet(
            self._make_source,
            cache_path=d,
            date_column="date",
            cache_mode=CacheMode.REBUILD,
        ).filter(pl.col("x").is_in([1, 2])).collect()

        on_disk = self._read_partition(d)
        assert on_disk["x"].to_list() == [1, 2, 3, 4, 5]

    def test_bounded_and_unbounded_readback_identity(self, tmp_path):
        """Reading either cache back UNDER the same predicate yields identical rows (correctness preserved)."""
        pred = pl.col("x").is_in([2, 4])

        d_bounded = tmp_path / "b"
        d_unbounded = tmp_path / "u"

        bounded = (
            cache_parquet(
                self._make_source,
                cache_path=d_bounded,
                date_column="date",
                cache_mode=CacheMode.REBUILD,
                write_bounding_columns=["x"],
            )
            .filter(pred)
            .collect()
            .sort("x")
        )
        unbounded = (
            cache_parquet(
                self._make_source,
                cache_path=d_unbounded,
                date_column="date",
                cache_mode=CacheMode.REBUILD,
            )
            .filter(pred)
            .collect()
            .sort("x")
        )

        assert bounded.equals(unbounded)
        assert bounded["x"].to_list() == [2, 4]

    def test_cache_mode_allowed_with_write_bounding_columns(self, tmp_path):
        """write_bounding_columns is opt-in and permitted in any cache mode.

        Under CacheMode.CACHE the bounded write still applies on a fresh partition; the
        predicate-scoped semantics are the caller's responsibility (see docstring).
        """
        d = tmp_path / "cache_bounded"
        out = (
            cache_parquet(
                self._make_source,
                cache_path=d,
                date_column="date",
                cache_mode=CacheMode.CACHE,
                write_bounding_columns=["x"],
            )
            .filter(pl.col("x").is_in([1, 2]))
            .collect()
        )
        assert out["x"].to_list() == [1, 2]
        # The freshly written partition is bounded to the predicate.
        assert self._read_partition(d)["x"].to_list() == [1, 2]

    def test_rebuild_mode_allowed(self, tmp_path):
        """REBUILD with write_bounding_columns writes the predicate-bounded subset."""
        out = (
            cache_parquet(
                self._make_source,
                cache_path=tmp_path / "rebuild_ok",
                date_column="date",
                cache_mode=CacheMode.REBUILD,
                write_bounding_columns=["x"],
            )
            .filter(pl.col("x").is_in([1]))
            .collect()
        )
        assert out["x"].to_list() == [1]

    def test_ignore_mode_passthrough(self, tmp_path):
        """IGNORE bypasses the cache entirely; write_bounding_columns is a no-op (no raise, upstream returned)."""
        out = (
            cache_parquet(
                self._make_source,
                cache_path=tmp_path / "ignore",
                date_column="date",
                cache_mode=CacheMode.IGNORE,
                write_bounding_columns=["x"],
            )
            .filter(pl.col("x").is_in([1, 2]))
            .collect()
            .sort("x")
        )
        # IGNORE returns upstream filtered by the read-side predicate; no cache files written.
        assert out["x"].to_list() == [1, 2]
        assert not os.path.exists(os.path.join(tmp_path / "ignore", "monthly"))

    def test_predicate_not_restrictable_degrades_to_unbounded(self, tmp_path):
        """write_bounding_columns=['x'] but predicate references only 'y' => restrict returns None => write is NOT bounded."""
        d = tmp_path / "degrade"
        cache_parquet(
            self._make_source,
            cache_path=d,
            date_column="date",
            cache_mode=CacheMode.REBUILD,
            write_bounding_columns=["x"],
        ).filter(pl.col("y").is_in([100, 200])).collect()

        on_disk = self._read_partition(d)
        # No bounding column in the predicate -> full partition written (no crash, degrade-safe).
        assert on_disk["x"].to_list() == [1, 2, 3, 4, 5]

    def test_empty_subset_pins_current_behavior(self, tmp_path):
        """Degenerate empty subset is_in([]).

        FINDING: a bounded write with an empty subset produces ZERO partition files, so the
        subsequent read raises ComputeError ('expanded paths were empty'). The UNBOUNDED
        equivalent returns an empty frame cleanly. This test PINS the current (divergent)
        behavior so any future fix is visible.
        """
        d = tmp_path / "empty"
        with pytest.raises(pl.exceptions.ComputeError):
            cache_parquet(
                self._make_source,
                cache_path=d,
                date_column="date",
                cache_mode=CacheMode.REBUILD,
                write_bounding_columns=["x"],
            ).filter(pl.col("x").is_in([])).collect()

    def test_empty_subset_unbounded_is_clean(self, tmp_path):
        """Contrast control: the UNBOUNDED empty-subset read returns an empty frame with no error."""
        d = tmp_path / "empty_unbounded"
        out = (
            cache_parquet(
                self._make_source,
                cache_path=d,
                date_column="date",
                cache_mode=CacheMode.REBUILD,
            )
            .filter(pl.col("x").is_in([]))
            .collect()
        )
        assert out.height == 0
        assert set(out.columns) == {"date", "x", "y", "value"}
