import re
from datetime import date, datetime
from typing import List, Optional

import narwhals as nw
import narwhals.stable.v1 as nw_stable
import polars as pl
import portion
import pyarrow as pa
import pytest
from polars.io.plugins import register_io_source
from polars.testing import assert_frame_equal

import polars_io_tools as cpl
from polars_io_tools.io_sources.lazy_narwhals_reader import polars_to_nw
from polars_io_tools.io_sources.range_visitor import convert_expr_to_datetime_range
from polars_io_tools.io_sources.set_visitor import convert_expr_to_valid_values
from polars_io_tools.io_sources.sql_utils import convert_predicate_to_sql

pd = pytest.importorskip("pandas", exc_type=ImportError)

# This module intentionally does not test different frame types
# (e.g., Polars DataFrame, Arrow Table, etc.) because we delegate
# the conversion (and all requisite error handling) to Narwhals.

GLOBAL_DATA_DICT = {
    "a": [1, 2, 3, 4, 5, 6],
    "b": [10, 20, 30, 40, 50, 60],
    "cat": ["x", "y", "x", "y", "x", "y"],
}


def apply_transformation(df):
    return df.filter(pl.col("a") > 3).select("cat", "b").group_by("cat").agg(pl.col("b").sum().alias("b_sum")).sort("cat")


POLARS_ANSWER = apply_transformation(pl.DataFrame(GLOBAL_DATA_DICT))

_EXPRESSION_CONVERSIONS = [
    (pl.col("a").is_null(), nw.col("a").is_null()),
    (pl.col("a").is_not_null(), ~nw.col("a").is_null()),
    (pl.col("a").is_in([1, 2, 3]), nw.col("a").is_in([1, 2, 3])),
]


@pytest.mark.parametrize("polars_expr, narwhals_expr", _EXPRESSION_CONVERSIONS)
def test_narwhals_expression_creation(polars_expr: pl.Expr, narwhals_expr: nw.Expr):
    assert str(polars_to_nw(polars_expr)) == str(narwhals_expr)


def test_non_narwhals_input():
    """Test that `from_narwhals` raises an error when given a non-Narwhals input."""
    with pytest.raises(TypeError, match="Expected a Narwhals DataFrame or LazyFrame"):
        cpl.from_narwhals(pl.DataFrame(GLOBAL_DATA_DICT))


def test_polars_shortcut():
    """Test that `from_narwhals` shortcuts when provided a Polars frame."""
    native_pl_df = pl.DataFrame(GLOBAL_DATA_DICT)
    nw_df = nw.from_native(native_pl_df)
    assert nw_df.implementation == nw.Implementation.POLARS
    returned_pl_df = cpl.from_narwhals(nw_df)

    assert isinstance(returned_pl_df, pl.DataFrame)
    assert_frame_equal(returned_pl_df, native_pl_df)
    assert not isinstance(returned_pl_df, pl.LazyFrame)

    # We repeat the same thing as above, but with a LazyFrame
    native_pl_lf = pl.LazyFrame(GLOBAL_DATA_DICT)
    nw_lf = nw.from_native(native_pl_lf)
    assert nw_lf.implementation == nw.Implementation.POLARS
    returned_pl_lf = cpl.from_narwhals(nw_lf)

    assert isinstance(returned_pl_lf, pl.LazyFrame)
    assert_frame_equal(returned_pl_lf, native_pl_lf)
    assert not isinstance(returned_pl_lf, pl.DataFrame)


def test_from_narwhals_dataframe():
    """Test that `from_narwhals` correctly handles a Narwhals DataFrame."""
    pd_df = pd.DataFrame(GLOBAL_DATA_DICT)
    nw_df = nw.from_native(pd_df)

    pl_df = cpl.from_narwhals(nw_df)

    assert isinstance(pl_df, pl.DataFrame)

    result = apply_transformation(pl_df.lazy()).collect()
    assert_frame_equal(result, POLARS_ANSWER)


def test_from_narwhals_lazyframe():
    """Test that `from_narwhals` correctly handles a Narwhals LazyFrame."""
    pd_df = pd.DataFrame(GLOBAL_DATA_DICT)
    nw_lazy = nw.from_native(pd_df).lazy()

    pl_lazy = cpl.from_narwhals(nw_lazy)

    assert isinstance(pl_lazy, pl.LazyFrame)

    result = apply_transformation(pl_lazy).collect()
    assert_frame_equal(result, POLARS_ANSWER)


def test_from_narwhals_stable_lazyframe():
    """Test that `from_narwhals` correctly handles a Narwhals LazyFrame."""
    pd_df = pd.DataFrame(GLOBAL_DATA_DICT)
    nw_lazy = nw_stable.from_native(pd_df).lazy()

    pl_lazy = cpl.from_narwhals(nw_lazy)

    assert isinstance(pl_lazy, pl.LazyFrame)

    result = apply_transformation(pl_lazy).collect()
    assert_frame_equal(result, POLARS_ANSWER)

    # Repeated call to ensure stability
    result2 = apply_transformation(pl_lazy).collect()
    assert_frame_equal(result2, POLARS_ANSWER)


def test_from_narwhals_with_predicate_pushdown():
    """
    Test that `from_narwhals` correctly pushes down predicates when using LazyFrames.
    This is already test implicitly above through the use of `apply_transformation`, but
    this is an explicit test
    """
    pd_df = pd.DataFrame(GLOBAL_DATA_DICT)
    nw_lazy = nw.from_native(pd_df).lazy()

    pl_lazy = cpl.from_narwhals(nw_lazy)

    filtered = pl_lazy.filter(pl.col("a") > 3)
    result = filtered.select("cat", "b").group_by("cat").agg(pl.col("b").sum().alias("b_sum")).sort("cat").collect()

    assert_frame_equal(result, POLARS_ANSWER)


def test_from_narwhals_custom_batch_size():
    """Test that `from_narwhals` correctly handles custom batch size parameter"""

    large_data = {"a": pa.array(list(range(1000))), "b": pa.array(list(range(1000, 2000))), "cat": pa.array(["x", "y"] * 500)}
    arrow_table = pa.Table.from_pydict(large_data)

    nw_lazy = nw.from_native(arrow_table).lazy()

    custom_batch_size = 10
    pl_lazy = cpl.from_narwhals(nw_lazy, fetch_size=custom_batch_size)

    result = pl_lazy.filter(pl.col("a") < 10).collect()
    expected = pl.DataFrame({"a": list(range(10)), "b": list(range(1000, 1010)), "cat": ["x", "y"] * 5})

    assert_frame_equal(result, expected)


def test_narwhals_wrapping_polars():
    def my_scan(df: pl.LazyFrame) -> pl.LazyFrame:
        schema = df.collect_schema()
        expected_set = set([1])

        expected_lower = portion.closed(lower=-portion.inf, upper=datetime(2023, 10, 1))
        expected_upper = portion.openclosed(lower=datetime(2023, 10, 8), upper=portion.inf)
        expected_interval = expected_lower | expected_upper

        def source_generator(
            with_columns: Optional[List[str]],
            predicate: Optional[pl.Expr],
            n_rows: Optional[int],
            batch_size: Optional[int],
        ):
            assert convert_expr_to_valid_values(predicate, "a") == expected_set
            assert convert_expr_to_datetime_range(predicate, "date", get_enclosure=False) == expected_interval

            if predicate is not None:
                df2 = df.filter(predicate)
            if with_columns is not None:
                df2 = df2.select(with_columns)
            yield df2.collect()

        return register_io_source(io_source=source_generator, schema=schema)

    true_df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [True, False, None],
            "d": ["a", "b", "c"],
            "date": [date(2023, 10, 1), date(2023, 10, 2), date(2023, 10, 3)],
        }
    )
    nw_lf = nw.from_native(my_scan(true_df.lazy()))
    res = (
        nw_lf.filter((nw.col("a").is_in([1, 5, 2])) & (~nw.col("a").is_in([5, 2])))
        .filter(~(nw.col("a").is_in([4, 5]) | nw.col("date").is_between(date(2023, 10, 1), date(2023, 10, 8), closed="right")))
        .select("a", "c", "date")
        .collect()
        .to_polars()
    )
    assert_frame_equal(
        res,
        pl.DataFrame(
            {
                "a": [
                    1,
                ],
                "c": [
                    True,
                ],
                "date": [
                    date(2023, 10, 1),
                ],
            }
        ),
    )


def test_from_narwhals_polars_predicate_pushdown():
    """
    Test that cpl.from_narwhals properly supports polars-specific predicates
    (is_null, is_not_null, is_in) and that they work correctly after conversion.
    """

    # Create test data with nulls and various categories for comprehensive testing
    test_data = {
        "id": [1, 2, 3, 4, 5, 6, 7, 8],
        "name": ["Alice", None, "Charlie", "David", None, "Frank", "Grace", None],
        "category": ["A", "B", "A", "C", "B", "C", "A", "B"],
        "score": [85.5, 92.0, None, 78.0, 88.5, None, 95.0, 82.0],
        "active": [True, False, None, True, False, True, None, False],
    }

    # Test with pandas as the source (most common non-polars case)
    pd_df = pd.DataFrame(test_data)
    nw_df = nw.from_native(pd_df)

    # Convert to polars using cpl.from_narwhals
    pl_df = cpl.from_narwhals(nw_df)
    assert isinstance(pl_df, pl.DataFrame)

    # Test 1: is_null predicate pushdown and correctness
    result_null_names = pl_df.lazy().filter(pl.col("name").is_null()).collect()
    expected_null_names = pl.DataFrame(
        {"id": [2, 5, 8], "name": [None, None, None], "category": ["B", "B", "B"], "score": [92.0, 88.5, 82.0], "active": [False, False, False]},
        schema_overrides={"name": pl.String},
    )
    assert_frame_equal(result_null_names, expected_null_names)

    # Test 2: is_not_null predicate pushdown and correctness
    result_not_null_scores = pl_df.lazy().filter(pl.col("score").is_not_null()).collect()
    expected_not_null_scores = pl.DataFrame(
        {
            "id": [1, 2, 4, 5, 7, 8],
            "name": ["Alice", None, "David", None, "Grace", None],
            "category": ["A", "B", "C", "B", "A", "B"],
            "score": [85.5, 92.0, 78.0, 88.5, 95.0, 82.0],
            "active": [True, False, True, False, None, False],
        }
    )
    assert_frame_equal(result_not_null_scores, expected_not_null_scores)

    # Test 3: is_in predicate pushdown and correctness
    result_categories_in = pl_df.lazy().filter(pl.col("category").is_in(["A", "C"])).collect()
    expected_categories_in = pl.DataFrame(
        {
            "id": [1, 3, 4, 6, 7],
            "name": ["Alice", "Charlie", "David", "Frank", "Grace"],
            "category": ["A", "A", "C", "C", "A"],
            "score": [85.5, None, 78.0, None, 95.0],
            "active": [True, None, True, True, None],
        }
    )
    assert_frame_equal(result_categories_in, expected_categories_in)

    # Test 4: Complex combined predicates - this tests that multiple polars predicates
    # work together correctly after narwhals conversion
    result_combined = (
        pl_df.lazy()
        .filter(
            pl.col("name").is_not_null()  # Not null names
            & pl.col("category").is_in(["A", "C"])  # Categories A or C
            & (pl.col("score").is_null() | (pl.col("score") > 80.0))  # Null scores OR score > 80
        )
        .collect()
    )
    expected_combined = pl.DataFrame(
        {
            "id": [1, 3, 6, 7],
            "name": ["Alice", "Charlie", "Frank", "Grace"],
            "category": ["A", "A", "C", "A"],
            "score": [85.5, None, None, 95.0],
            "active": [True, None, True, None],
        }
    )
    assert_frame_equal(result_combined, expected_combined)

    # Test 5: Test is_in with single value
    result_single_in = pl_df.lazy().filter(pl.col("category").is_in(["C"])).collect()
    expected_single_in = pl.DataFrame(
        {"id": [4, 6], "name": ["David", "Frank"], "category": ["C", "C"], "score": [78.0, None], "active": [True, True]}
    )
    assert_frame_equal(result_single_in, expected_single_in)


def _strip_brackets(sql: str) -> str:
    """
    This is a convenience function for the test below.
    It removes brackets from SQL strings (which is
    necessary when dealing with MSSQL/T-SQL).
    """
    return re.sub(r"\[([^\]]+)]", r"\1", sql)


def test_alias_filter_predicate_pushdown_to_original_column():
    """
    Test that when we:
    1. Have a base Polars custom IO source
    2. Convert to Narwhals
    3. Alias a column
    4. Select the aliased column
    5. Filter on the aliased column name

    The predicate pushed down to the original IO source correctly references
    the ORIGINAL column name, not the alias.
    """
    captured_predicates = []

    def my_scan(df: pl.LazyFrame) -> pl.LazyFrame:
        schema = df.collect_schema()

        def source_generator(
            with_columns: Optional[List[str]],
            predicate: Optional[pl.Expr],
            n_rows: Optional[int],
            batch_size: Optional[int],
        ):
            # Capture the predicate that was pushed down
            captured_predicates.append(predicate)

            df2 = df
            if predicate is not None:
                df2 = df2.filter(predicate)
            if with_columns is not None:
                df2 = df2.select(with_columns)
            yield df2.collect()

        return register_io_source(io_source=source_generator, schema=schema)

    # Create source data with original column name "original_col"
    source_df = pl.DataFrame(
        {
            "original_col": [1, 2, 3, 4, 5],
            "other_col": ["a", "b", "c", "d", "e"],
        }
    )

    # Create custom IO source and convert to narwhals
    custom_lf = my_scan(source_df.lazy())
    nw_lf = nw.from_native(custom_lf)

    # Alias the column, select it, and filter on the aliased name
    result = (
        nw_lf.with_columns(nw.col("original_col").alias("renamed_col"))
        .select("renamed_col", "other_col")
        .filter(nw.col("renamed_col") > 2)
        .collect()
        .to_polars()
    )

    # Verify the result is correct
    expected = pl.DataFrame(
        {
            "renamed_col": [3, 4, 5],
            "other_col": ["c", "d", "e"],
        }
    )
    assert_frame_equal(result, expected)

    # Verify a predicate was pushed down
    assert len(captured_predicates) == 1
    pushed_predicate = captured_predicates[0]
    assert pushed_predicate is not None

    # The pushed predicate should reference "original_col", not "renamed_col"
    predicate_str = str(pushed_predicate)
    assert "original_col" in predicate_str, f"Expected predicate to reference 'original_col' but got: {predicate_str}"
    assert "renamed_col" not in predicate_str, f"Predicate should NOT reference 'renamed_col' but got: {predicate_str}"


def test_boolean_cast_is_removed_from_sql():
    """
    Test that the hot-paths for booleans work correctly
    (which is necessary for the Narwhals integration).
    """
    base = pl.col("data_date") >= date(2025, 7, 7)

    with_cast = base.cast(pl.Boolean)

    sql_plain = convert_predicate_to_sql(base, dialect="tsql")
    sql_cast = convert_predicate_to_sql(with_cast, dialect="tsql")

    # Both conversions must succeed and the resulting
    # WHERE-clause must be identical (ignoring quoting)
    assert sql_plain is not None and sql_cast is not None

    assert _strip_brackets(sql_plain.sql(dialect="tsql")) == _strip_brackets(sql_cast.sql(dialect="tsql"))

    # Another similar case
    pred = pl.col("TradeDate").is_between(date(2025, 7, 7), date(2025, 7, 10)).cast(pl.Boolean)

    res = convert_predicate_to_sql(pred, dialect="tsql").sql(dialect="tsql")
    assert "CAST" not in res


def test_sql_lazy_frame():
    def my_scan(df: pl.LazyFrame) -> pl.LazyFrame:
        schema = df.collect_schema()

        def source_generator(
            with_columns: Optional[List[str]],
            predicate: Optional[pl.Expr],
            n_rows: Optional[int],
            batch_size: Optional[int],
        ):
            assert predicate is not None
            print(str(predicate))
            assert convert_predicate_to_sql(predicate, dialect="mssql") is not None
            if predicate is not None:
                df2 = df.filter(predicate)
            if with_columns is not None:
                df2 = df2.select(with_columns)
            yield df2.collect()

        return register_io_source(io_source=source_generator, schema=schema)

    raw_lf = pl.LazyFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [True, False, None],
            "d": ["a", "b", "c"],
            "e": [1.1, 2.2, 3.3],
            "date": [date(2023, 10, 1), date(2023, 10, 2), date(2023, 10, 3)],
        }
    )

    nw_frame = nw.from_native(my_scan(raw_lf))
    lf_frame = cpl.from_narwhals(nw_frame)

    my_scan(raw_lf).filter(pl.col("e").cast(pl.Int64, strict=False) > 1).collect()
    res = lf_frame.filter(pl.col("e").cast(pl.Int64, strict=False) > 1).collect()

    assert_frame_equal(res, my_scan(raw_lf).filter(pl.col("e").cast(pl.Int64, strict=False) > 1).collect())


# These tests verify that from_narwhals LazyFrames can be serialized with cloudpickle,
# which is required for distributed computing (e.g., Ray). Tests are located here
# rather than in test_pickle.py because they require narwhals as an external dependency.


class TestFromNarwhalsPickle:
    """Tests for from_narwhals cloudpickle serialization support."""

    def test_from_narwhals_pickle_pandas_backed(self):
        """from_narwhals with pandas-backed DataFrame can be pickled."""
        import cloudpickle

        # Create a pandas-backed narwhals LazyFrame
        pd_df = pd.DataFrame(GLOBAL_DATA_DICT)
        nw_lazy = nw.from_native(pd_df).lazy()

        # Convert to Polars LazyFrame via from_narwhals
        pl_lazy = cpl.from_narwhals(nw_lazy)
        assert isinstance(pl_lazy, pl.LazyFrame)

        # Pickle roundtrip
        pickled = cloudpickle.dumps(pl_lazy)
        lf_unpickled = cloudpickle.loads(pickled)

        # Verify results match
        expected = apply_transformation(pl_lazy).collect()
        result = apply_transformation(lf_unpickled).collect()
        assert_frame_equal(result, expected)

    def test_from_narwhals_pickle_pandas_dataframe(self):
        """from_narwhals with pandas DataFrame (not lazy) can be pickled."""
        import cloudpickle

        # Create a pandas-backed narwhals DataFrame (eager)
        pd_df = pd.DataFrame(GLOBAL_DATA_DICT)
        nw_df = nw.from_native(pd_df)

        # Convert to Polars DataFrame via from_narwhals
        pl_df = cpl.from_narwhals(nw_df)
        assert isinstance(pl_df, pl.DataFrame)

        # Make it lazy and pickle
        pl_lazy = pl_df.lazy()
        pickled = cloudpickle.dumps(pl_lazy)
        lf_unpickled = cloudpickle.loads(pickled)

        # Verify results match
        expected = apply_transformation(pl_lazy).collect()
        result = apply_transformation(lf_unpickled).collect()
        assert_frame_equal(result, expected)

    def test_from_narwhals_pickle_pyarrow_backed(self):
        """from_narwhals with PyArrow-backed frame can be pickled."""
        import cloudpickle

        # Create a PyArrow table and wrap in narwhals
        arrow_table = pa.table(GLOBAL_DATA_DICT)
        nw_df = nw.from_native(arrow_table)

        # Convert to Polars via from_narwhals
        pl_df = cpl.from_narwhals(nw_df)
        assert isinstance(pl_df, pl.DataFrame)

        # Make it lazy and pickle
        pl_lazy = pl_df.lazy()
        pickled = cloudpickle.dumps(pl_lazy)
        lf_unpickled = cloudpickle.loads(pickled)

        # Verify results match
        expected = apply_transformation(pl_lazy).collect()
        result = apply_transformation(lf_unpickled).collect()
        assert_frame_equal(result, expected)

    def test_from_narwhals_pickle_with_filter(self):
        """from_narwhals LazyFrame with filters can be pickled."""
        import cloudpickle

        pd_df = pd.DataFrame(GLOBAL_DATA_DICT)
        nw_lazy = nw.from_native(pd_df).lazy()
        pl_lazy = cpl.from_narwhals(nw_lazy)

        # Add a filter
        lf_filtered = pl_lazy.filter(pl.col("a") > 3)

        pickled = cloudpickle.dumps(lf_filtered)
        lf_unpickled = cloudpickle.loads(pickled)

        expected = pl_lazy.filter(pl.col("a") > 3).collect()
        result = lf_unpickled.collect()
        assert_frame_equal(result, expected)
