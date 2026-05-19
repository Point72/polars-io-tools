import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_io_tools as cpl

from .conftest import io_source_assert


def test_concat_named_basic():
    """Test basic functionality of concat_named."""
    df1 = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = pl.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]})

    # Track whether each dataframe is queried
    queried = {"df1": False, "df2": False}

    def assert_func1(predicate):
        queried["df1"] = True
        assert predicate is None  # No predicate should be applied

    def assert_func2(predicate):
        queried["df2"] = True
        assert predicate is None  # No predicate should be applied

    lf1 = io_source_assert(df1, assert_func1)
    lf2 = io_source_assert(df2, assert_func2)

    result = cpl.concat_named({("foo",): lf1, ("bar",): lf2}, ["source"]).collect()

    expected = pl.DataFrame({"a": [1, 2, 3, 7, 8, 9], "b": [4, 5, 6, 10, 11, 12], "source": ["foo", "foo", "foo", "bar", "bar", "bar"]})

    assert_frame_equal(result, expected)
    assert queried["df1"]  # Both dataframes should be queried
    assert queried["df2"]


def test_concat_named_filter_pushdown():
    """Test that concat_named correctly pushes down filters on unique columns."""
    df1 = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = pl.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]})

    # Track which dataframes are queried
    queried = {"df1": False, "df2": False}

    def assert_func1(predicate):
        queried["df1"] = True
        # df1 should be queried because its key ("foo") matches the filter
        assert predicate is None

    def assert_func2(predicate):
        queried["df2"] = True
        # df2 should not be queried since we filter for source == "foo"
        assert False, "df2 should not be queried when filtering for 'foo'"

    lf1 = io_source_assert(df1, assert_func1)
    lf2 = io_source_assert(df2, assert_func2)

    result = cpl.concat_named({("foo",): lf1, ("bar",): lf2}, ["source"]).filter(pl.col("source") == "foo").collect()

    expected = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "source": ["foo", "foo", "foo"]})

    assert_frame_equal(result, expected)
    assert queried["df1"]  # Only df1 should be queried
    assert not queried["df2"]  # df2 should not be queried


def test_concat_named_multiple_unique_columns():
    """Test concat_named with multiple unique columns."""
    df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pl.DataFrame({"a": [5, 6], "b": [7, 8]})
    df3 = pl.DataFrame({"a": [9, 10], "b": [11, 12]})
    df4 = pl.DataFrame({"a": [13, 14], "b": [15, 16]})

    # Track which dataframes are queried
    queried = {f"df{i}": False for i in range(1, 5)}

    def create_assert_func(df_name, should_query):
        def assert_func(predicate):
            queried[df_name] = True
            if not should_query:
                assert False, f"{df_name} should not be queried"
            assert predicate is None

        return assert_func

    lf1 = io_source_assert(df1, create_assert_func("df1", True))  # (east, A) - should query
    lf2 = io_source_assert(df2, create_assert_func("df2", False))  # (east, B) - should not query
    lf3 = io_source_assert(df3, create_assert_func("df3", False))  # (west, A) - should not query
    lf4 = io_source_assert(df4, create_assert_func("df4", False))  # (west, B) - should not query

    result = (
        cpl.concat_named({("east", "A"): lf1, ("east", "B"): lf2, ("west", "A"): lf3, ("west", "B"): lf4}, ["region", "type"])
        .filter((pl.col("region") == "east") & (pl.col("type") == "A"))
        .collect()
    )

    expected = pl.DataFrame({"a": [1, 2], "b": [3, 4], "region": ["east", "east"], "type": ["A", "A"]})

    assert_frame_equal(result, expected)
    # Verify only the correct dataframe was queried
    assert queried["df1"]
    assert not queried["df2"]
    assert not queried["df3"]
    assert not queried["df4"]


def test_concat_named_different_data_types():
    """Test concat_named with different data types for the unique columns."""
    df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pl.DataFrame({"a": [5, 6], "b": [7, 8]})

    date1 = datetime.date(2023, 1, 1)
    date2 = datetime.date(2023, 2, 1)

    # Track which dataframes are queried
    queried = {"df1": False, "df2": False}

    def assert_func1(predicate):
        queried["df1"] = True
        # df1 should be queried with date1
        assert predicate is None

    def assert_func2(predicate):
        queried["df2"] = True
        # df2 should not be queried when filtering for date1
        assert False, "df2 should not be queried"

    lf1 = io_source_assert(df1, assert_func1)
    lf2 = io_source_assert(df2, assert_func2)

    result = cpl.concat_named({(date1, 100): lf1, (date2, 200): lf2}, ["date", ("id", pl.Float64)]).filter(pl.col("date") == date1).collect()

    expected = pl.DataFrame({"a": [1, 2], "b": [3, 4], "date": [date1, date1], "id": [100.0, 100.0]})
    assert_frame_equal(result, expected)
    assert queried["df1"]
    assert not queried["df2"]


def test_concat_named_empty_dict():
    """Test concat_named with an empty dictionary."""
    # The function should raise a ValueError for empty dictionaries
    with pytest.raises(ValueError):
        cpl.concat_named({}, ["source"]).collect()


def test_concat_named_or_condition():
    """Test concat_named with an OR condition on unique columns."""
    df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pl.DataFrame({"a": [5, 6], "b": [7, 8]})
    df3 = pl.DataFrame({"a": [9, 10], "b": [11, 12]})

    # Track which dataframes are queried
    queried = {"df1": False, "df2": False, "df3": False}

    def assert_func1(predicate):
        queried["df1"] = True
        # df1 should be queried (source = "foo")
        assert predicate is not None

    def assert_func2(predicate):
        queried["df2"] = True
        # df2 should be queried (source = "bar")
        assert predicate is not None

    def assert_func3(predicate):
        queried["df3"] = True
        # df3 should not be queried (source = "baz")
        assert False, "df3 should not be queried"

    lf1 = io_source_assert(df1, assert_func1)
    lf2 = io_source_assert(df2, assert_func2)
    lf3 = io_source_assert(df3, assert_func3)

    result = (
        cpl.concat_named({("foo",): lf1, ("bar",): lf2, ("baz",): lf3}, ["source"])
        .filter((pl.col("source") == "foo") | (pl.col("source") == "bar"), pl.col("a") > 1)
        .collect()
    )

    expected = pl.DataFrame({"a": [2, 5, 6], "b": [4, 7, 8], "source": ["foo", "bar", "bar"]})

    assert_frame_equal(result, expected)
    assert queried["df1"]
    assert queried["df2"]
    assert not queried["df3"]


def test_concat_named_is_in_filter():
    """Test concat_named with an is_in filter on unique columns."""
    df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pl.DataFrame({"a": [5, 6], "b": [7, 8]})
    df3 = pl.DataFrame({"a": [9, 10], "b": [11, 12]})

    # Track which dataframes are queried
    queried = {"df1": False, "df2": False, "df3": False}

    def assert_func1(predicate):
        queried["df1"] = True
        assert predicate is None

    def assert_func2(predicate):
        queried["df2"] = True
        assert predicate is None

    def assert_func3(predicate):
        queried["df3"] = True
        assert False, "df3 should not be queried"

    lf1 = io_source_assert(df1, assert_func1)
    lf2 = io_source_assert(df2, assert_func2)
    lf3 = io_source_assert(df3, assert_func3)

    result = cpl.concat_named({("foo",): lf1, ("bar",): lf2, ("baz",): lf3}, ["source"]).filter(pl.col("source").is_in(["foo", "bar"])).collect()

    expected = pl.DataFrame({"a": [1, 2, 5, 6], "b": [3, 4, 7, 8], "source": ["foo", "foo", "bar", "bar"]})

    assert_frame_equal(result, expected)
    assert queried["df1"]
    assert queried["df2"]
    assert not queried["df3"]


def test_concat_named_filter_non_unique_column():
    """Test concat_named with filtering on non-unique columns."""
    df1 = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = pl.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]})

    # Track which dataframes are queried
    queried = {"df1": False, "df2": False}

    def assert_func1(predicate):
        queried["df1"] = True

    def assert_func2(predicate):
        queried["df2"] = True

    lf1 = io_source_assert(df1, assert_func1)
    lf2 = io_source_assert(df2, assert_func2)

    result = cpl.concat_named({("foo",): lf1, ("bar",): lf2}, ["source"]).filter(pl.col("a") == 1).collect()

    expected = pl.DataFrame({"a": [1], "b": [4], "source": ["foo"]})

    assert_frame_equal(result, expected)
    # Both dataframes should be queried since filter is on a non-unique column
    assert queried["df1"]
    assert queried["df2"]


def test_concat_named_complex_predicate():
    """Test concat_named with complex predicates combining unique and non-unique columns."""
    df1 = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = pl.DataFrame({"a": [1, 8, 9], "b": [10, 11, 12]})

    # Track which dataframes are queried
    queried = {"df1": False, "df2": False}

    def assert_func1(predicate):
        queried["df1"] = True

    def assert_func2(predicate):
        queried["df2"] = True
        # df2 should not be queried (source = "bar")
        assert False, "df2 should not be queried"

    lf1 = io_source_assert(df1, assert_func1)
    lf2 = io_source_assert(df2, assert_func2)

    # Filtering on both source and 'a'
    result = cpl.concat_named({("foo",): lf1, ("bar",): lf2}, ["source"]).filter((pl.col("source") == "foo") & (pl.col("a") >= 2)).collect()

    expected = pl.DataFrame({"a": [2, 3], "b": [5, 6], "source": ["foo", "foo"]})

    assert_frame_equal(result, expected)
    assert queried["df1"]  # Should query df1 because source = "foo"
    assert not queried["df2"]  # Should not query df2


def test_concat_named_mismatched_columns_keys():
    """Test concat_named with mismatched numbers of unique columns and keys."""
    df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pl.DataFrame({"a": [5, 6], "b": [7, 8]})

    # The keys are tuples of length 1, but we provide 2 unique columns
    with pytest.raises(ValueError, match="Number of unique columns 2 does not match number of keys 1"):
        cpl.concat_named({("foo",): df1.lazy(), ("bar",): df2.lazy()}, ["source", "type"]).collect()


def test_concat_named_with_column_selection():
    """Test concat_named with column selection."""
    df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    df2 = pl.DataFrame({"a": [7, 8], "b": [9, 10], "c": [11, 12]})

    # Track which dataframes are queried
    queried = {"df1": False, "df2": False}

    def assert_func1(predicate):
        queried["df1"] = True
        assert predicate is None

    def assert_func2(predicate):
        queried["df2"] = True
        assert False, "df2 should not be queried"

    lf1 = io_source_assert(df1, assert_func1)
    lf2 = io_source_assert(df2, assert_func2)

    result = cpl.concat_named({("foo",): lf1, ("bar",): lf2}, ["source"]).filter(pl.col("source") == "foo").select(["a", "c"]).collect()

    expected = pl.DataFrame(
        {
            "a": [1, 2],
            "c": [5, 6],
        }
    )

    assert_frame_equal(result, expected)
    assert queried["df1"]
    assert not queried["df2"]


def test_concat_named_filter_with_inequality():
    """Test concat_named using a date filter with inequality."""
    df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pl.DataFrame({"a": [5, 6], "b": [7, 8]})

    # Track which dataframes are queried
    queried = {"df1": False, "df2": False}

    def assert_func1(predicate):
        queried["df1"] = True

    def assert_func2(predicate):
        queried["df2"] = True
        assert False, "df2 should not be queried"

    lf1 = io_source_assert(df1, assert_func1)
    lf2 = io_source_assert(df2, assert_func2)

    result = (
        cpl.concat_named({(datetime.date(2025, 5, 1),): lf1, (datetime.date(2025, 5, 2),): lf2}, ["source"])
        .filter(pl.col("source") < datetime.date(2025, 5, 2))
        .collect()
    )

    expected = pl.DataFrame({"a": [1, 2], "b": [3, 4], "source": [datetime.date(2025, 5, 1), datetime.date(2025, 5, 1)]})

    assert_frame_equal(result, expected)
    assert queried["df1"]
    assert not queried["df2"]
