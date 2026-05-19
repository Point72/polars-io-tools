from datetime import date, datetime, timedelta

import polars as pl
import pytest
from packaging import version
from polars.testing import assert_frame_equal

from polars_io_tools.io_sources.base import BinaryExprNode, get_literal_value, get_parsed_expr
from polars_io_tools.io_sources.enum import BooleanFunctionType, OperatorType

from .conftest import io_source_assert


def test_filtered_join_basic():
    """Test basic functionality of filtered_join with single join column."""
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6.0, 7.0, 8.0],
            "ham": ["a", "b", "c"],
        }
    )
    other_df = pl.DataFrame({"apple": ["x", "y", "z"], "ham": ["a", "b", "d"], "bar": ["a", "b", "c"], "foo2": [1, 2, 3]})

    def assert_func(predicate):
        parsed_expr = get_parsed_expr(predicate)
        assert parsed_expr.function_type == BooleanFunctionType.IS_IN
        assert parsed_expr.inputs[0].expr.meta == pl.col("ham")
        assert sorted(get_literal_value(parsed_expr.inputs[1].expr)) == ["a", "b", "c"]

    other_lf = io_source_assert(other_df, assert_func)
    res = df.lazy().piot.filtered_join(other_lf, on="ham", maintain_order="left").collect()

    expected = pl.DataFrame(
        {
            "foo": [1, 2],
            "bar": [6.0, 7.0],
            "ham": ["a", "b"],
            "apple": ["x", "y"],
            "bar_right": ["a", "b"],
            "foo2": [1, 2],
        }
    )
    assert_frame_equal(res, expected)


@pytest.mark.parametrize("how", ["inner", "left"])
def test_filtered_join_basic_no_results_left_side(how):
    """Test basic functionality of filtered_join with single join column when no results are found on the left side."""
    if version.parse(pl.__version__) > version.parse("1.31.0"):
        # Here, polars pushes down filters for us, so they will be pushed down.
        pytest.skip("Test not applicable for polars > 1.31.0 due to automatic filter pushdown")
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6.0, 7.0, 8.0],
            "ham": ["a", "b", "c"],
        }
    )
    other_df = pl.DataFrame({"apple": ["x", "y", "z"], "ham": ["a", "b", "d"], "bar": ["a", "b", "d"], "foo2": [1, 2, 3]})

    def assert_func(predicate):
        assert False, "This should not be called since the left side is empty"

    other_lf = io_source_assert(other_df, assert_func)
    res = df.lazy().piot.filtered_join(other_lf, on="ham", how=how, maintain_order="left").filter(pl.col.ham == "d").collect()

    expected = pl.DataFrame({}, schema=df.join(other_df, on="ham", maintain_order="left").schema)
    assert_frame_equal(res, expected)


def test_filtered_join_single_filter_value():
    """Test basic functionality of filtered_join with single join column."""
    df = pl.DataFrame(
        {
            "foo": [
                1,
            ],
            "bar": [
                6.0,
            ],
            "ham": [
                "a",
            ],
        }
    )
    other_df = pl.DataFrame({"apple": ["x", "y", "z"], "ham": ["a", "b", "d"], "bar": ["a", "b", "c"], "foo2": [1, 2, 3]})

    def assert_func(predicate):
        parsed_expr = get_parsed_expr(predicate)
        assert parsed_expr.op == OperatorType.EQ
        assert parsed_expr.left.expr.meta == pl.col("ham")
        assert parsed_expr.right.value == "a"

    other_lf = io_source_assert(other_df, assert_func)
    res = df.lazy().piot.filtered_join(other_lf, on="ham", maintain_order="left").collect()

    expected = pl.DataFrame(
        {
            "foo": [1],
            "bar": [6.0],
            "ham": ["a"],
            "apple": ["x"],
            "bar_right": ["a"],
            "foo2": [1],
        }
    )
    assert_frame_equal(res, expected)


def test_filtered_join_column_selection():
    """Test filtered_join with projection pushdown."""
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6.0, 7.0, 8.0],
            "ham": ["a", "b", "c"],
        }
    )
    other_df = pl.DataFrame({"apple": ["x", "y", "z"], "ham": ["a", "b", "d"], "bar": ["a", "b", "c"], "foo2": [1, 2, 3]})

    def assert_func(predicate):
        parsed_expr = get_parsed_expr(predicate)
        assert parsed_expr.function_type == BooleanFunctionType.IS_IN
        assert parsed_expr.inputs[0].expr.meta == pl.col("ham")
        assert sorted(get_literal_value(parsed_expr.inputs[1].expr)) == ["a", "b", "c"]

    res = (
        df.lazy()
        .piot.filtered_join(
            io_source_assert(other_df, assert_func),
            on="ham",
            maintain_order="left",
        )
        .select("foo", "bar")
        .collect()
    )
    expected = pl.DataFrame(
        {
            "foo": [1, 2],
            "bar": [6.0, 7.0],
        }
    )
    assert_frame_equal(res, expected)


def test_filtered_join_multiple_columns():
    """Test filtered_join with multiple join columns."""
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6.0, 7.0, 8.0],
            "ham": ["a", "b", "c"],
        }
    )
    other_df = pl.DataFrame({"apple": ["x", "y", "z"], "ham": ["a", "b", "d"], "bar": ["a", "b", "c"], "foo2": [1, 2, 3]})

    def assert_func(predicate):
        parsed_expr = get_parsed_expr(predicate)
        assert parsed_expr.op == OperatorType.AND

        for input_expr in [parsed_expr.left, parsed_expr.right]:
            assert input_expr.function_type == BooleanFunctionType.IS_IN
            col_expr = input_expr.inputs[0].expr
            if col_expr.meta == pl.col("ham"):
                assert sorted(get_literal_value(input_expr.inputs[1].expr)) == ["a", "b", "c"]
            elif col_expr.meta == pl.col("foo2"):
                assert sorted(get_literal_value(input_expr.inputs[1].expr)) == [1, 2, 3]
            else:
                assert False, f"Unexpected column: {str(col_expr)}"

    res = (
        df.lazy()
        .piot.filtered_join(
            io_source_assert(other_df, assert_func),
            left_on=["ham", "foo"],
            right_on=["ham", "foo2"],
            maintain_order="left",
        )
        .collect()
    )
    expected = pl.DataFrame(
        {
            "foo": [1, 2],
            "bar": [6.0, 7.0],
            "ham": ["a", "b"],
            "apple": ["x", "y"],
            "bar_right": ["a", "b"],
        }
    )
    assert_frame_equal(res, expected)


def test_filtered_join_multiple_columns_with_extra_filter():
    """Test filtered_join with multiple join columns and an extra filter condition."""
    # Create test dataframes with columns for joining and filtering
    df = pl.DataFrame({"id": [1, 2, 3, 4], "category": ["A", "B", "A", "C"], "value": [10, 20, 30, 40]})

    other_df = pl.DataFrame({"id": [1, 2, 3, 5], "category": ["A", "B", "A", "D"], "score": [15, 25, 35, 55], "active": [True, False, True, False]})

    # This is the key part - we need to verify that both join conditions AND
    # the filter condition are pushed down to the source
    def assert_func(predicate):
        parsed_expr = get_parsed_expr(predicate)

        # For a complex expression tree with multiple AND conditions,
        # we need to collect all leaf conditions
        conditions = []

        def collect_conditions(expr):
            # Recursively traverse the expression tree to find all conditions
            if hasattr(expr, "op") and expr.op == OperatorType.AND:
                collect_conditions(expr.left)
                collect_conditions(expr.right)
            else:
                conditions.append(expr)

        collect_conditions(parsed_expr)

        # Track which conditions we've found
        found = {"id_is_in": False, "category_is_in": False, "active_eq": False}

        # Check each condition
        for cond in conditions:
            if hasattr(cond, "function_type") and cond.function_type == BooleanFunctionType.IS_IN:
                col_expr = cond.inputs[0].expr
                if col_expr.meta == pl.col("id"):
                    # Verify the id column contains all expected values from left df
                    assert sorted(get_literal_value(cond.inputs[1].expr)) == [1, 2, 3, 4]
                    found["id_is_in"] = True
                elif col_expr.meta == pl.col("category"):
                    # Verify the category column contains all expected values from left df
                    assert sorted(get_literal_value(cond.inputs[1].expr)) == ["A", "B", "C"]
                    found["category_is_in"] = True
            elif hasattr(cond, "op") and cond.op == OperatorType.EQ:
                # Check for the extra filter condition (active == True)
                if cond.left.expr.meta == pl.col("active"):
                    assert get_literal_value(cond.right.expr) is True
                    found["active_eq"] = True

        # Ensure we found all expected conditions
        for condition_name, was_found in found.items():
            assert was_found, f"Missing condition: {condition_name}"

    # Join with multiple columns using the filtered right dataframe
    res = (
        df.lazy()
        .piot.filtered_join(
            io_source_assert(other_df, assert_func), left_on=["id", "category"], right_on=["id", "category"], maintain_order="left_right"
        )
        .filter(pl.col("active") == pl.lit(True))
        .collect()
    )

    # Expected result - only rows that match BOTH join conditions AND the filter
    expected = pl.DataFrame({"id": [1, 3], "category": ["A", "A"], "value": [10, 30], "score": [15, 35], "active": [True, True]})
    assert_frame_equal(res, expected)


def test_filtered_join_different_join_types():
    """Test filtered_join with different join types (inner, left, right, outer)."""
    df = pl.DataFrame({"id": [1, 2, 3, 4], "value": ["a", "b", "c", "d"]})
    other_df = pl.DataFrame({"id": [1, 2, 5, 6], "desc": ["x", "y", "p", "q"]})

    def assert_func(predicate):
        parsed_expr = get_parsed_expr(predicate)
        assert parsed_expr.function_type == BooleanFunctionType.IS_IN
        assert parsed_expr.inputs[0].expr.meta == pl.col("id")
        assert sorted(get_literal_value(parsed_expr.inputs[1].expr)) == [1, 2, 3, 4]

    # Inner join (default)
    other_lf = io_source_assert(other_df, assert_func)
    res = df.lazy().piot.filtered_join(other_lf, on="id", how="inner", maintain_order="left").collect()
    expected = pl.DataFrame({"id": [1, 2], "value": ["a", "b"], "desc": ["x", "y"]})
    assert_frame_equal(res, expected)


def test_filtered_join_with_nulls():
    """Test filtered_join handling of null values."""
    df = pl.DataFrame({"id": [1, 2, None, 4], "value": ["a", "b", "c", "d"]})
    other_df = pl.DataFrame({"id": [1, None, 3, 5], "desc": ["x", "y", "z", "p"]})

    def assert_func(predicate):
        parsed_expr = get_parsed_expr(predicate)
        assert parsed_expr.function_type == BooleanFunctionType.IS_IN
        assert parsed_expr.inputs[0].expr.meta == pl.col("id")
        # Should only include non-null values in the filter
        assert sorted(get_literal_value(parsed_expr.inputs[1].expr)) == [1, 2, 4]

    res = df.lazy().piot.filtered_join(io_source_assert(other_df, assert_func), on="id", maintain_order="left").collect()

    # Only the row with id=1 should match
    expected = pl.DataFrame({"id": [1], "value": ["a"], "desc": ["x"]})
    assert_frame_equal(res, expected)


def test_filtered_join_with_various_data_types():
    """Test filtered_join with various data types including dates and booleans."""
    df = pl.DataFrame({"id": [1, 2, 3], "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)], "flag": [True, False, True]})
    other_df = pl.DataFrame({"id": [1, 2, 4], "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 4)], "desc": ["x", "y", "z"]})

    def assert_func(predicate):
        parsed_expr = get_parsed_expr(predicate)
        assert parsed_expr.op == OperatorType.AND

        for input_expr in [parsed_expr.left, parsed_expr.right]:
            assert input_expr.function_type == BooleanFunctionType.IS_IN
            col_expr = input_expr.inputs[0].expr
            if col_expr.meta == pl.col("id"):
                assert sorted(get_literal_value(input_expr.inputs[1].expr)) == [1, 2, 3]
            elif col_expr.meta == pl.col("date"):
                dates = get_literal_value(input_expr.inputs[1].expr)
                assert len(dates) == 3
                assert all(isinstance(d, date) for d in dates)
            else:
                assert False, f"Unexpected column: {str(col_expr)}"

    res = (
        df.lazy()
        .piot.filtered_join(
            io_source_assert(other_df, assert_func),
            left_on=["id", "date"],
            right_on=["id", "date"],
            maintain_order="left",
        )
        .collect()
    )

    expected = pl.DataFrame({"id": [1, 2], "date": [date(2023, 1, 1), date(2023, 1, 2)], "flag": [True, False], "desc": ["x", "y"]})
    assert_frame_equal(res, expected)


def test_filtered_join_empty_dataframe():
    """Test filtered_join behavior with empty DataFrames."""
    # Empty left DataFrame
    df_empty = pl.DataFrame({"id": [], "value": []}, schema={"id": pl.Int64, "value": pl.Float64})
    other_df = pl.DataFrame({"id": [1, 2, 3], "desc": ["x", "y", "z"]})

    def assert_func_empty(predicate):
        # Since left DataFrame is empty, the filter should contain an empty list
        parsed_expr = get_parsed_expr(predicate)
        assert parsed_expr.function_type == BooleanFunctionType.IS_IN
        assert parsed_expr.inputs[0].expr.meta == pl.col("id")
        assert get_literal_value(parsed_expr.inputs[1].expr) == []

    res = df_empty.lazy().piot.filtered_join(io_source_assert(other_df, assert_func_empty), on="id", maintain_order="left").collect()

    # Result should be empty with correct schema
    assert res.is_empty()
    assert "id" in res.columns and "value" in res.columns and "desc" in res.columns


def test_filtered_join_suffix():
    """Test filtered_join with custom suffix for duplicate column names."""
    df = pl.DataFrame({"id": [1, 2, 3], "common": ["a", "b", "c"]})
    other_df = pl.DataFrame({"id": [1, 2, 4], "common": ["x", "y", "z"]})

    def assert_func(predicate):
        parsed_expr = get_parsed_expr(predicate)
        assert parsed_expr.function_type == BooleanFunctionType.IS_IN
        assert parsed_expr.inputs[0].expr.meta == pl.col("id")
        assert sorted(get_literal_value(parsed_expr.inputs[1].expr)) == [1, 2, 3]

    # With default suffix
    res = df.lazy().piot.filtered_join(io_source_assert(other_df, assert_func), on="id", maintain_order="left").collect()

    expected = pl.DataFrame({"id": [1, 2], "common": ["a", "b"], "common_right": ["x", "y"]})
    assert_frame_equal(res, expected)

    # With custom suffix
    res = df.lazy().piot.filtered_join(io_source_assert(other_df, assert_func), on="id", suffix="_custom", maintain_order="left").collect()

    expected_custom = pl.DataFrame({"id": [1, 2], "common": ["a", "b"], "common_custom": ["x", "y"]})
    assert_frame_equal(res, expected_custom)


def test_filtered_join_suffixed_column_in_derived_expression():
    """Test filtered_join when suffixed columns are used in derived expressions but not directly selected.

    This tests a subtle bug where projection pushdown could break suffix handling:
    - If with_columns includes "quantity_previous" (suffixed name from right side)
    - But the base column "quantity" exists in both left and right schemas
    - We must keep "quantity" in the left side to create a name conflict
    - Otherwise "quantity" from the right side won't get the suffix applied

    The symptom was: ColumnNotFoundError for "quantity_previous" because
    without the conflict, the right side's "quantity" stayed as "quantity".
    """
    # Left dataframe with a date column for joining and columns that will conflict
    left_df = pl.DataFrame(
        {
            "data_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            "data_date_previous": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "id": [1, 2, 3],
            "quantity": [100, 200, 300],
            "price": [10.0, 20.0, 30.0],
        }
    )

    # Right dataframe - same structure, will be joined to get "previous" values
    right_df = pl.DataFrame(
        {
            "data_date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "id": [1, 2, 3],
            "quantity": [90, 190, 290],
            "price": [9.0, 19.0, 29.0],
        }
    )

    # Perform the filtered join - this creates quantity_previous and price_previous
    result = (
        left_df.lazy()
        .piot.filtered_join(
            right_df.lazy(),
            left_on=["id", "data_date_previous"],
            right_on=["id", "data_date"],
            how="left",
            suffix="_previous",
            maintain_order="left",
        )
        # Use suffixed columns in a derived expression
        .with_columns(((pl.col("price") - pl.col("price_previous")) * pl.col("quantity_previous")).alias("value_change"))
        # Only select derived column and identifiers - NOT the suffixed columns directly
        # This triggers projection pushdown that requests suffixed columns
        .select("data_date", "id", "value_change")
        .collect()
    )

    # Expected: the value_change should be computed correctly
    # Row 1: (10.0 - 9.0) * 90 = 90.0
    # Row 2: (20.0 - 19.0) * 190 = 190.0
    # Row 3: (30.0 - 29.0) * 290 = 290.0
    expected = pl.DataFrame(
        {
            "data_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            "id": [1, 2, 3],
            "value_change": [90.0, 190.0, 290.0],
        }
    )

    assert_frame_equal(result, expected)


def test_filtered_join_suffixed_column_multiple_conflicts():
    """Test that multiple suffixed columns are handled correctly with projection pushdown."""
    left_df = pl.DataFrame(
        {
            "key": [1, 2],
            "a": [10, 20],
            "b": [100, 200],
            "c": [1000, 2000],
        }
    )

    right_df = pl.DataFrame(
        {
            "key": [1, 2],
            "a": [11, 21],
            "b": [110, 210],
            "c": [1100, 2100],
        }
    )

    # Use all three suffixed columns in a computation, but don't select them directly
    result = (
        left_df.lazy()
        .piot.filtered_join(right_df.lazy(), on="key", suffix="_right", maintain_order="left")
        .with_columns((pl.col("a_right") + pl.col("b_right") + pl.col("c_right")).alias("sum_right"))
        .select("key", "sum_right")
        .collect()
    )

    # Row 1: 11 + 110 + 1100 = 1221
    # Row 2: 21 + 210 + 2100 = 2331
    expected = pl.DataFrame({"key": [1, 2], "sum_right": [1221, 2331]})

    assert_frame_equal(result, expected)


@pytest.mark.parametrize("how", ["inner", "left"])
def test_filtered_join_complex_workflow(how):
    """Test filtered_join in a more complex data workflow with multiple operations."""
    df = pl.DataFrame({"id": [1, 2, 3, 4], "value": [10, 20, 30, 40]})
    other_df = pl.DataFrame({"id": [1, 2, 3, 5], "multiplier": [2, 3, 4, 5]})

    def assert_func(predicate):
        def assert_is_in(expr):
            assert expr.function_type == BooleanFunctionType.IS_IN
            assert expr.inputs[0].expr.meta == pl.col("id")
            assert sorted(get_literal_value(expr.inputs[1].expr)) == [1, 2, 3, 4]
            # only the filter from the join gets pushed down

        parsed_expr = get_parsed_expr(predicate)
        # In polars >1.31.0, joins can be rewritten, and here the "left" join
        # becomes an inner join.
        if how == "left" and version.parse(pl.__version__) <= version.parse("1.31.0"):
            # We only pass the filter generated from the join
            assert_is_in(parsed_expr)
            pytest.skip("Test behavior differs for left joins in polars <= 1.31.0")
        assert parsed_expr.op == OperatorType.AND
        for input_expr in [parsed_expr.left, parsed_expr.right]:
            if getattr(input_expr, "function_type", None) == BooleanFunctionType.IS_IN:
                assert_is_in(input_expr)
            elif input_expr.op == OperatorType.LT:
                assert input_expr.left.expr.meta == pl.col("multiplier")
                assert get_literal_value(input_expr.right.expr) == 4
            else:
                assert False, f"Unexpected expr: {str(predicate)}"

    # Complex workflow: join, calculate, filter, aggregate
    res = (
        df.lazy()
        .piot.filtered_join(io_source_assert(other_df, assert_func), on="id", how=how, maintain_order="left")
        .with_columns((pl.col("value") * pl.col("multiplier")).alias("product"))
        .filter(pl.col("product") > 50)
        .group_by("multiplier")
        .agg(pl.sum("product").alias("sum_product"))
        .sort("multiplier")
        .filter(pl.col("multiplier") < 4)
        .collect()
    )

    expected = pl.DataFrame({"multiplier": [3], "sum_product": [60]})
    assert_frame_equal(res, expected)


def test_filtered_join_sink_parquet(tmp_path):
    """Test filtered_join with projection pushdown."""
    # Create a temporary directory for the parquet file
    tmp_path.mkdir(exist_ok=True)
    file_path = tmp_path / "join_data.parquet"

    df = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6.0, 7.0, 8.0],
            "ham": ["a", "b", "c"],
        }
    )
    other_df = pl.DataFrame({"apple": ["x", "y", "z"], "ham": ["a", "b", "d"], "bar": ["a", "b", "c"], "foo2": [1, 2, 3]})

    def assert_func(predicate):
        parsed_expr = get_parsed_expr(predicate)
        assert parsed_expr.function_type == BooleanFunctionType.IS_IN
        assert parsed_expr.inputs[0].expr.meta == pl.col("ham")
        assert sorted(get_literal_value(parsed_expr.inputs[1].expr)) == ["a", "b", "c"]

    df.lazy().piot.filtered_join(
        io_source_assert(other_df, assert_func),
        on="ham",
        maintain_order="left",
    ).select("bar", "foo").sink_parquet(file_path)

    result = pl.read_parquet(file_path)
    expected = pl.DataFrame(
        {
            "bar": [6.0, 7.0],
            "foo": [1, 2],
        }
    )
    assert_frame_equal(result, expected)


########### Filtered Join AsOf Tests ###########
@pytest.mark.parametrize("include_id", [True, False])
@pytest.mark.parametrize(
    "filters",
    [
        None,
        pl.col("timestamp") > datetime(2023, 1, 1, 9, 7, 0),
        ((pl.col("timestamp") <= datetime(2023, 1, 1, 9, 3, 0)) & (pl.col("trade_id") != 2)),
    ],
)
@pytest.mark.parametrize("on_same_column", [True, False])
@pytest.mark.parametrize("strategy", ["backward", "forward", "nearest"])
@pytest.mark.parametrize("select_columns,coalesce", [[True, False], [False, True]])
def test_filtered_join_asof_basic_backward(include_id, filters, on_same_column, strategy, select_columns, coalesce):
    """Test basic functionality of filtered_join_asof with backward strategy."""
    # Left dataframe with timestamps - represents trades
    on, left_on, right_on = "timestamp", None, None
    if on_same_column:
        on, left_on, right_on = "timestamp", None, None
    else:
        on, left_on, right_on = None, "timestamp", "timestamp2"
    df_left = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 1, 9, 0, 0),
                datetime(2023, 1, 1, 9, 5, 0),
                datetime(2023, 1, 1, 9, 10, 0),
            ],
            "trade_id": [1, 2, 2],
            "price": [100.0, 105.0, 102.0],
            "shared_col": [1, 2, 3],
        }
    ).with_columns(pl.col("timestamp").set_sorted())

    # Right dataframe with quotes - represents market data
    df_right = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 1, 8, 58, 0),
                datetime(2023, 1, 1, 9, 2, 0),
                datetime(2023, 1, 1, 9, 7, 0),
                datetime(2023, 1, 1, 9, 12, 0),
            ],
            "timestamp2": [
                datetime(2023, 1, 1, 8, 58, 1),
                datetime(2023, 1, 1, 9, 2, 1),
                datetime(2023, 1, 1, 9, 7, 1),
                datetime(2023, 1, 1, 9, 12, 1),
            ],
            "quote_id": [1, 2, 2, 4],
            "bid": [99.0, 103.0, 104.0, 101.0],
            "shared_col": [-1, -2, -3, -4],
        }
    ).with_columns(pl.col("timestamp").set_sorted())

    tolerance = timedelta(minutes=3)

    def assert_func(predicate):
        """Validate that filters are pushed down correctly to the right dataframe."""
        # We make sure we do not add predicates unnecessarily
        if filters is None:
            assert predicate is None
            return
        parsed_expr = get_parsed_expr(predicate)
        parsed_filter = get_parsed_expr(filters)

        assert isinstance(parsed_expr, BinaryExprNode)
        assert isinstance(parsed_filter, BinaryExprNode)

        filter_right_binary = isinstance(parsed_filter.right, BinaryExprNode)
        if filter_right_binary:
            filter_left_value = parsed_filter.left.right.value
            filter_right_value = parsed_filter.right.right.value
            filter_values = set([filter_left_value, filter_right_value])
        else:
            filter_values = set([parsed_filter.right.value])

        if include_id and filter_right_binary:
            expr_left_value = parsed_expr.left.right.value
            expr_right_value = parsed_expr.right.right.value
            if strategy in ["forward", "nearest"]:
                # In this case, the upper bound is affected
                # We subtract the tolerance from it to move backwards.
                if isinstance(expr_left_value, (datetime, date)):
                    expr_left_value -= tolerance
                else:
                    expr_right_value -= tolerance

            filter_left_value = parsed_filter.left.right.value
            filter_right_value = parsed_filter.right.right.value

            # Order does not matter, so we use set to compare
            assert set([filter_left_value, filter_right_value]) == set([expr_left_value, expr_right_value])
            return
        # If we only have 1 filter, we should have 1 value pushed down.
        parsed_value = parsed_expr.right.value
        if filter_right_binary:
            if strategy in ["forward", "nearest"]:
                # then we shift the upper bound in the 2 filters case
                parsed_value -= tolerance
        else:
            if strategy in ["backward", "nearest"]:
                # We shift the lower bound in the 1 filter case
                parsed_value += tolerance

        assert parsed_value in filter_values
        target_col = right_on if right_on is not None else on
        assert parsed_expr.left.name == target_col

    right_lf = io_source_assert(df_right, assert_func)

    # Perform filtered as-of join with 3-minute tolerance
    kwargs = dict(
        on=on,
        left_on=left_on,
        right_on=right_on,
        strategy=strategy,
        tolerance=tolerance,
        coalesce=coalesce,
    )

    if include_id:
        kwargs.update(
            dict(
                by_left="trade_id",
                by_right="quote_id",
                check_sortedness=False,
            )
        )

    result = df_left.lazy().clone().piot.filtered_join_asof(right_lf, **kwargs)
    expected = df_left.lazy().clone().join_asof(df_right.lazy(), **kwargs)
    if filters is not None:
        result = result.filter(filters)
        expected = expected.filter(filters)

    if select_columns:
        result = result.select(["timestamp", "price", "shared_col"])
        expected = expected.select(["timestamp", "price", "shared_col"])
    else:
        # The orders of these might be different because we manually set the order of the columns
        # in the expected dataframe, so we sort them to compare. Once the Polars release after 1.30.0
        # comes out, we should be able to remove this and assert the orders are the same, we won't
        # need to adjust to the order of the schema (was causing issues sinking to parquet)
        result = result.select(sorted(result.collect_schema().keys()))
        expected = expected.select(sorted(expected.collect_schema().keys()))
    expected = expected.collect()
    result = result.collect()
    assert_frame_equal(result, expected)


def test_filtered_join_asof_empty_left_dataframe():
    """Test filtered_join_asof with empty left DataFrame."""
    df_left_empty = pl.DataFrame({"timestamp": [], "value": []}, schema={"timestamp": pl.Datetime, "value": pl.Float64}).with_columns(
        pl.col("timestamp").set_sorted()
    )

    df_right = pl.DataFrame(
        {
            "timestamp": [datetime(2023, 1, 1, 9, 0, 0), datetime(2023, 1, 1, 9, 5, 0)],
            "data": ["a", "b"],
        }
    ).with_columns(pl.col("timestamp").set_sorted())

    # Test both should produce empty results with same schema
    result = df_left_empty.lazy().piot.filtered_join_asof(df_right.lazy(), on="timestamp", strategy="backward").collect()

    expected = df_left_empty.lazy().join_asof(df_right.lazy(), on="timestamp", strategy="backward").collect()

    assert_frame_equal(result, expected)
    assert result.is_empty()


def test_filtered_join_asof_empty_right_dataframe():
    """Test filtered_join_asof with empty right DataFrame."""
    df_left = pl.DataFrame(
        {
            "timestamp": [datetime(2023, 1, 1, 9, 0, 0)],
            "value": [1],
        }
    ).with_columns(pl.col("timestamp").set_sorted())

    df_right_empty = pl.LazyFrame({"timestamp": [], "data": []}, schema={"timestamp": pl.Datetime, "data": pl.Utf8}).with_columns(
        pl.col("timestamp").set_sorted()
    )

    result = df_left.lazy().piot.filtered_join_asof(df_right_empty, on="timestamp", strategy="backward").collect()

    expected = df_left.lazy().join_asof(df_right_empty, on="timestamp", strategy="backward").collect()

    assert_frame_equal(result, expected)
    # Should have left row but no right match (null values)
    assert len(result) == 1
    assert result["data"][0] is None


@pytest.mark.parametrize("strategy", ["backward", "forward", "nearest"])
def test_filtered_join_asof_no_tolerance(strategy):
    """Test filtered_join_asof without tolerance parameter."""
    df_left = pl.DataFrame(
        {
            "timestamp": [datetime(2023, 1, 1, 9, 5, 0)],
            "id": [1],
        }
    ).with_columns(pl.col("timestamp").set_sorted())

    df_right = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 1, 9, 0, 0),
                datetime(2023, 1, 1, 9, 3, 0),
                datetime(2023, 1, 1, 9, 8, 0),
            ],
            "value": [10, 20, 30],
        }
    ).with_columns(pl.col("timestamp").set_sorted())

    filters = [pl.col("timestamp") > datetime(2023, 1, 1, 9, 5, 0)]

    def assert_func(predicate):
        # Without tolerance, we do not push down on the right side.
        assert predicate is None

    right_lf = io_source_assert(df_right, assert_func)

    result = df_left.lazy().piot.filtered_join_asof(
        right_lf,
        on="timestamp",
        strategy=strategy,
        # No tolerance parameter
    )

    expected = df_left.lazy().join_asof(df_right.lazy(), on="timestamp", strategy=strategy)

    expected = expected.filter(filters).collect()
    result = result.filter(filters).collect()

    assert_frame_equal(result, expected)


class TestJoinBetween:
    """Test join_between function for range joins with equi-join support."""

    def test_basic_range_join_no_by(self):
        """Test basic range join without by columns."""
        from polars_io_tools import join_between

        left = pl.DataFrame(
            {
                "trade_id": [1, 2, 3],
                "trade_date": [date(2024, 1, 15), date(2024, 2, 15), date(2024, 3, 15)],
                "amount": [100.0, 200.0, 300.0],
            }
        ).lazy()

        right = pl.DataFrame(
            {
                "valid_from": [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)],
                "valid_to": [date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 31)],
                "price": [10.0, 20.0, 30.0],
            }
        ).lazy()

        result = join_between(left, right, left_on="trade_date", right_on_start="valid_from", right_on_end="valid_to").collect()

        assert len(result) == 3
        assert result.filter(pl.col("trade_id") == 1)["price"][0] == 10.0
        assert result.filter(pl.col("trade_id") == 2)["price"][0] == 20.0
        assert result.filter(pl.col("trade_id") == 3)["price"][0] == 30.0

    def test_equi_join_with_by_column(self):
        """Test join with by column for equi-join on common column."""
        from polars_io_tools import join_between

        left = pl.DataFrame(
            {
                "trade_id": [1, 2, 3, 4],
                "security_id": ["AAPL", "AAPL", "GOOG", "GOOG"],
                "trade_date": [date(2024, 1, 15), date(2024, 2, 15), date(2024, 1, 15), date(2024, 2, 15)],
            }
        ).lazy()

        right = pl.DataFrame(
            {
                "security_id": ["AAPL", "AAPL", "GOOG", "GOOG"],
                "valid_from": [date(2024, 1, 1), date(2024, 2, 1), date(2024, 1, 1), date(2024, 2, 1)],
                "valid_to": [date(2024, 1, 31), date(2024, 2, 29), date(2024, 1, 31), date(2024, 2, 29)],
                "price": [150.0, 155.0, 140.0, 145.0],
            }
        ).lazy()

        result = join_between(left, right, left_on="trade_date", right_on_start="valid_from", right_on_end="valid_to", by="security_id").collect()

        assert len(result) == 4
        assert result.filter(pl.col("trade_id") == 1)["price"][0] == 150.0
        assert result.filter(pl.col("trade_id") == 2)["price"][0] == 155.0
        assert result.filter(pl.col("trade_id") == 3)["price"][0] == 140.0
        assert result.filter(pl.col("trade_id") == 4)["price"][0] == 145.0

    def test_left_join_preserves_rows_when_by_not_in_right(self):
        """Test that left join preserves rows when by value doesn't exist in right table."""
        from polars_io_tools import join_between

        left = pl.DataFrame(
            {
                "trade_id": [1, 2, 3],
                "security_id": ["AAPL", "MSFT", "GOOG"],
                "trade_date": [date(2024, 1, 15), date(2024, 1, 15), date(2024, 1, 15)],
            }
        ).lazy()

        right = pl.DataFrame(
            {
                "security_id": ["AAPL", "GOOG"],
                "valid_from": [date(2024, 1, 1), date(2024, 1, 1)],
                "valid_to": [date(2024, 1, 31), date(2024, 1, 31)],
                "price": [150.0, 140.0],
            }
        ).lazy()

        result = join_between(
            left, right, left_on="trade_date", right_on_start="valid_from", right_on_end="valid_to", by="security_id", how="left"
        ).collect()

        assert len(result) == 3
        assert result.filter(pl.col("trade_id") == 1)["price"][0] == 150.0
        assert result.filter(pl.col("trade_id") == 2)["price"][0] is None
        assert result.filter(pl.col("trade_id") == 3)["price"][0] == 140.0

    def test_inner_join_drops_rows_when_by_not_in_right(self):
        """Test that inner join drops rows when by value doesn't exist in right table."""
        from polars_io_tools import join_between

        left = pl.DataFrame(
            {
                "trade_id": [1, 2, 3],
                "security_id": ["AAPL", "MSFT", "GOOG"],
                "trade_date": [date(2024, 1, 15), date(2024, 1, 15), date(2024, 1, 15)],
            }
        ).lazy()

        right = pl.DataFrame(
            {
                "security_id": ["AAPL", "GOOG"],
                "valid_from": [date(2024, 1, 1), date(2024, 1, 1)],
                "valid_to": [date(2024, 1, 31), date(2024, 1, 31)],
                "price": [150.0, 140.0],
            }
        ).lazy()

        result = join_between(
            left, right, left_on="trade_date", right_on_start="valid_from", right_on_end="valid_to", by="security_id", how="inner"
        ).collect()

        assert len(result) == 2
        trade_ids = result["trade_id"].to_list()
        assert 1 in trade_ids
        assert 3 in trade_ids
        assert 2 not in trade_ids

    def test_equi_join_with_composite_key(self):
        """Test join between with multiple by columns (composite key)."""
        from polars_io_tools import join_between

        left = pl.DataFrame(
            {
                "trade_id": [1, 2],
                "security_id": ["ABC", "ABC"],
                "source": ["Bloomberg", "Reuters"],
                "trade_date": [date(2024, 1, 15), date(2024, 1, 15)],
            }
        ).lazy()

        right = pl.DataFrame(
            {
                "security_id": ["ABC", "ABC"],
                "source": ["Bloomberg", "Reuters"],
                "valid_from": [date(2024, 1, 1), date(2024, 1, 1)],
                "valid_to": [date(2024, 1, 31), date(2024, 1, 31)],
                "price": [100.0, 101.0],
            }
        ).lazy()

        result = join_between(
            left, right, left_on="trade_date", right_on_start="valid_from", right_on_end="valid_to", by=["security_id", "source"]
        ).collect()

        assert len(result) == 2
        assert result.filter(pl.col("trade_id") == 1)["price"][0] == 100.0
        assert result.filter(pl.col("trade_id") == 2)["price"][0] == 101.0

    def test_left_join_preserves_unmatched_range(self):
        """Test that left join preserves rows with no matching range."""
        from polars_io_tools import join_between

        left = pl.DataFrame(
            {
                "trade_id": [1, 2, 3],
                "security_id": ["AAPL", "AAPL", "AAPL"],
                "trade_date": [date(2024, 1, 15), date(2024, 6, 15), date(2024, 3, 15)],
            }
        ).lazy()

        right = pl.DataFrame(
            {
                "security_id": ["AAPL", "AAPL"],
                "valid_from": [date(2024, 1, 1), date(2024, 3, 1)],
                "valid_to": [date(2024, 1, 31), date(2024, 3, 31)],
                "price": [10.0, 30.0],
            }
        ).lazy()

        result = join_between(
            left, right, left_on="trade_date", right_on_start="valid_from", right_on_end="valid_to", by="security_id", how="left"
        ).collect()

        assert len(result) == 3
        assert result.filter(pl.col("trade_id") == 2)["price"][0] is None
        assert result.filter(pl.col("trade_id") == 1)["price"][0] is not None
        assert result.filter(pl.col("trade_id") == 3)["price"][0] is not None

    def test_inner_join_excludes_unmatched_range(self):
        """Test that inner join excludes rows with no matching range."""
        from polars_io_tools import join_between

        left = pl.DataFrame(
            {
                "trade_id": [1, 2, 3],
                "security_id": ["AAPL", "AAPL", "AAPL"],
                "trade_date": [date(2024, 1, 15), date(2024, 6, 15), date(2024, 3, 15)],
            }
        ).lazy()

        right = pl.DataFrame(
            {
                "security_id": ["AAPL", "AAPL"],
                "valid_from": [date(2024, 1, 1), date(2024, 3, 1)],
                "valid_to": [date(2024, 1, 31), date(2024, 3, 31)],
                "price": [10.0, 30.0],
            }
        ).lazy()

        result = join_between(
            left, right, left_on="trade_date", right_on_start="valid_from", right_on_end="valid_to", by="security_id", how="inner"
        ).collect()

        assert len(result) == 2
        trade_ids = result["trade_id"].to_list()
        assert 1 in trade_ids
        assert 3 in trade_ids
        assert 2 not in trade_ids

    def test_gap_between_ranges_returns_null(self):
        """Test that dates falling in gaps between ranges get null values."""
        from polars_io_tools import join_between

        left = pl.DataFrame({"trade_id": [1], "security_id": ["AAPL"], "trade_date": [date(2024, 2, 15)]}).lazy()

        right = pl.DataFrame(
            {
                "security_id": ["AAPL", "AAPL"],
                "valid_from": [date(2024, 1, 1), date(2024, 3, 1)],
                "valid_to": [date(2024, 1, 31), date(2024, 3, 31)],
                "price": [10.0, 30.0],
            }
        ).lazy()

        result = join_between(
            left, right, left_on="trade_date", right_on_start="valid_from", right_on_end="valid_to", by="security_id", how="left"
        ).collect()

        assert len(result) == 1
        assert result["price"][0] is None

    def test_date_before_all_ranges(self):
        """Test that dates before all ranges get null values."""
        from polars_io_tools import join_between

        left = pl.DataFrame({"trade_id": [1], "security_id": ["AAPL"], "trade_date": [date(2023, 12, 15)]}).lazy()

        right = pl.DataFrame(
            {
                "security_id": ["AAPL"],
                "valid_from": [date(2024, 1, 1)],
                "valid_to": [date(2024, 1, 31)],
                "price": [10.0],
            }
        ).lazy()

        result = join_between(
            left, right, left_on="trade_date", right_on_start="valid_from", right_on_end="valid_to", by="security_id", how="left"
        ).collect()

        assert len(result) == 1
        assert result["price"][0] is None

    def test_date_after_all_ranges(self):
        """Test that dates after all ranges get null values."""
        from polars_io_tools import join_between

        left = pl.DataFrame({"trade_id": [1], "security_id": ["AAPL"], "trade_date": [date(2024, 12, 15)]}).lazy()

        right = pl.DataFrame(
            {
                "security_id": ["AAPL"],
                "valid_from": [date(2024, 1, 1)],
                "valid_to": [date(2024, 1, 31)],
                "price": [10.0],
            }
        ).lazy()

        result = join_between(
            left, right, left_on="trade_date", right_on_start="valid_from", right_on_end="valid_to", by="security_id", how="left"
        ).collect()

        assert len(result) == 1
        assert result["price"][0] is None

    def test_boundary_dates_inclusive(self):
        """Test that boundary dates (exactly on start or end) are included."""
        from polars_io_tools import join_between

        left = pl.DataFrame(
            {
                "trade_id": [1, 2, 3],
                "security_id": ["AAPL", "AAPL", "AAPL"],
                "trade_date": [date(2024, 1, 1), date(2024, 1, 15), date(2024, 1, 31)],
            }
        ).lazy()

        right = pl.DataFrame({"security_id": ["AAPL"], "valid_from": [date(2024, 1, 1)], "valid_to": [date(2024, 1, 31)], "price": [10.0]}).lazy()

        result = join_between(left, right, left_on="trade_date", right_on_start="valid_from", right_on_end="valid_to", by="security_id").collect()

        assert len(result) == 3
        for i in range(3):
            assert result["price"][i] == 10.0

    def test_with_datetime_columns(self):
        """Test join_between works with datetime columns, not just dates."""
        from polars_io_tools import join_between

        left = pl.DataFrame(
            {
                "event_id": [1, 2, 3],
                "session_type": ["trading", "trading", "trading"],
                "event_time": [
                    datetime(2024, 1, 15, 10, 30),
                    datetime(2024, 1, 15, 14, 30),
                    datetime(2024, 1, 15, 18, 30),
                ],
            }
        ).lazy()

        right = pl.DataFrame(
            {
                "session_type": ["trading", "trading"],
                "session_start": [datetime(2024, 1, 15, 9, 0), datetime(2024, 1, 15, 13, 0)],
                "session_end": [datetime(2024, 1, 15, 12, 0), datetime(2024, 1, 15, 17, 0)],
                "session_name": ["Morning", "Afternoon"],
            }
        ).lazy()

        result = join_between(
            left, right, left_on="event_time", right_on_start="session_start", right_on_end="session_end", by="session_type", how="left"
        ).collect()

        assert len(result) == 3
        assert result.filter(pl.col("event_id") == 1)["session_name"][0] == "Morning"
        assert result.filter(pl.col("event_id") == 2)["session_name"][0] == "Afternoon"
        assert result.filter(pl.col("event_id") == 3)["session_name"][0] is None

    def test_column_name_collision_nulls_right_not_left(self):
        """Regression: when left and right share a column name, only the right (suffixed) column should be nulled."""
        from polars_io_tools import join_between

        left = pl.DataFrame({"ts": [5], "value": ["left"]}).lazy()
        right = pl.DataFrame({"start": [1], "end": [3], "value": ["right"]}).lazy()

        result = join_between(left, right, left_on="ts", right_on_start="start", right_on_end="end", how="left").collect()

        assert len(result) == 1
        # Left column must be preserved
        assert result["value"][0] == "left"
        # Right column (suffixed) must be nulled — no range match
        assert result["value_right"][0] is None

    def test_right_on_end_name_collision_validation(self):
        """Regression: when right_on_end collides with a left column name, validation must use the suffixed name."""
        from polars_io_tools import join_between

        # Window B ends at 10:00:04, reading is at 10:00:05 — should NOT match
        windows = pl.DataFrame(
            {
                "event_id": ["A", "B"],
                "window_start": [datetime(2023, 1, 1, 10, 0, 0), datetime(2023, 1, 1, 10, 0, 3)],
                "timestamp": [datetime(2023, 1, 1, 10, 0, 10), datetime(2023, 1, 1, 10, 0, 4)],
            }
        ).lazy()
        readings = pl.DataFrame({"timestamp": [datetime(2023, 1, 1, 10, 0, 5)], "value": [99]}).lazy()

        result = join_between(readings, windows, left_on="timestamp", right_on_start="window_start", right_on_end="timestamp", how="inner").collect()

        # B.timestamp (10:00:04) < reading.timestamp (10:00:05) — inner join should exclude it
        assert len(result) == 0

    def test_right_on_start_equal_left_on_preserves_left_key(self):
        """Regression: when right_on_start == left_on, invalid left joins must not null the preserved left key."""
        from polars_io_tools import join_between

        left = pl.DataFrame({"x": [10]}).lazy()
        right = pl.DataFrame({"x": [1], "end": [5], "payload": [99]}).lazy()

        result = join_between(left, right, left_on="x", right_on_start="x", right_on_end="end", how="left").collect()

        assert len(result) == 1
        assert result["x"][0] == 10
        assert result["end"][0] is None
        assert result["payload"][0] is None

    def test_overlapping_intervals_returns_single_match(self):
        """Document: join_between returns at most one match per left row (nearest start via asof)."""
        from polars_io_tools import join_between

        left = pl.DataFrame({"timestamp": [datetime(2023, 1, 1, 10, 0, 4)], "value": [99]}).lazy()
        right = pl.DataFrame(
            {
                "event_id": ["A", "B"],
                "window_start": [datetime(2023, 1, 1, 10, 0, 0), datetime(2023, 1, 1, 10, 0, 3)],
                "window_end": [datetime(2023, 1, 1, 10, 0, 5), datetime(2023, 1, 1, 10, 0, 8)],
            }
        ).lazy()

        result = join_between(left, right, left_on="timestamp", right_on_start="window_start", right_on_end="window_end", how="inner").collect()

        # Both A and B contain 10:00:04, but only B (nearest start) is returned
        assert len(result) == 1
        assert result["event_id"][0] == "B"
