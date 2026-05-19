import threading
from collections.abc import Iterator
from datetime import date, datetime, timedelta, timezone
from graphlib import CycleError

import polars as pl
import pytest
from packaging import version
from polars.exceptions import ColumnNotFoundError, SchemaError
from polars.testing import assert_frame_equal

from polars_io_tools.io_sources.util import (
    _ONPREM_ENDPOINTS_TO_RESOLVE,
    _resolve_endpoint_hostname,
    _storage_options_for,
    register_io_source_with_is_pure,
    with_columns_topo,
    wrap_io_source_with_error_catching,
)

from .conftest import io_source_assert


def _sample_lf() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": ["x", "y", "x", "y", "x"],
        }
    )


def _complex_sample_lf() -> pl.LazyFrame:
    """Sample LazyFrame with various data types for comprehensive testing."""
    # Create timezone-aware datetimes
    utc = timezone.utc
    est = timezone(timedelta(hours=-5))

    return pl.LazyFrame(
        {
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
            "str_col": ["a", "b", "c", "d", "e"],
            "bool_col": [True, False, True, False, True],
            "date_col": [
                date(2023, 1, 1),
                date(2023, 2, 1),
                date(2023, 3, 1),
                date(2023, 4, 1),
                date(2023, 5, 1),
            ],
            "datetime_col": [
                datetime(2023, 1, 1, 10, 0),
                datetime(2023, 2, 1, 11, 0),
                datetime(2023, 3, 1, 12, 0),
                datetime(2023, 4, 1, 13, 0),
                datetime(2023, 5, 1, 14, 0),
            ],
            # Timezone-aware datetimes
            "datetime_utc": [
                datetime(2023, 1, 1, 10, 0, tzinfo=utc),
                datetime(2023, 2, 1, 11, 0, tzinfo=utc),
                datetime(2023, 3, 1, 12, 0, tzinfo=utc),
                datetime(2023, 4, 1, 13, 0, tzinfo=utc),
                datetime(2023, 5, 1, 14, 0, tzinfo=utc),
            ],
            "datetime_est": [
                datetime(2023, 1, 1, 10, 0, tzinfo=est),
                datetime(2023, 2, 1, 11, 0, tzinfo=est),
                datetime(2023, 3, 1, 12, 0, tzinfo=est),
                datetime(2023, 4, 1, 13, 0, tzinfo=est),
                datetime(2023, 5, 1, 14, 0, tzinfo=est),
            ],
            "nullable_int": [1, None, 3, None, 5],
            # Varying length lists for more complex testing
            "list_col": [[1, 2], [3, 4, 5], [6], [7, 8, 9, 10], []],
        }
    )


def _verify_filter_no_pushdown(lf: pl.LazyFrame, expr: list[pl.Expr]) -> None:
    """Helper to verify filter_no_pushdown behavior matches normal filter."""
    normal_filter = lf.clone().filter(expr)
    no_pushdown_filter = lf.clone().piot.filter_no_pushdown(expr)
    no_pushdown_filter_disabled = lf.clone().piot.filter_no_pushdown(expr, _disable_optimizations=True)

    # Verify plan differences - no_pushdown should use FILTER, not SELECTION
    no_pushdown_explain = no_pushdown_filter.explain()
    assert "FILTER" in no_pushdown_explain

    # Verify results are identical
    normal_df = normal_filter.collect()
    no_pushdown_df = no_pushdown_filter.collect()
    disabled_df = no_pushdown_filter_disabled.collect()

    assert_frame_equal(normal_df, no_pushdown_df)
    assert_frame_equal(normal_df, disabled_df)


_EXPRS = [[pl.col("c") == "x", pl.col("a") > 2], [pl.col("c") == "x"]]


@pytest.mark.parametrize("expr", _EXPRS)
def test_filter_no_pushdown_via_pipe_print_explain(expr):
    lf = _sample_lf()
    lf = io_source_assert(lf.collect(), lambda _: True)
    normal_filter = lf.filter(expr)
    no_pushdown_filter = lf.piot.filter_no_pushdown(expr)
    no_pushdown_filter_disabled = lf.piot.filter_no_pushdown(expr, _disable_optimizations=True)

    # Print the plan so we can visually confirm no pushdown
    expected_pushed = [normal_filter, no_pushdown_filter_disabled]
    for lf in expected_pushed:
        explain = lf.explain()
        assert "SELECTION: " in explain
        assert "FILTER" not in explain
    no_pushdown_filter_explain = no_pushdown_filter.explain()
    assert "SELECTION" not in no_pushdown_filter_explain
    assert "FILTER" in no_pushdown_filter_explain

    # Basic sanity: only rows with c == 'x'
    normal_filter_df = normal_filter.collect()
    no_pushdown_filter_df = no_pushdown_filter.collect()
    no_pushdown_filter_disabled_df = no_pushdown_filter_disabled.collect()

    assert_frame_equal(normal_filter_df, no_pushdown_filter_df)
    assert_frame_equal(normal_filter_df, no_pushdown_filter_disabled_df)


# Focused test expressions covering key scenarios
_COMPREHENSIVE_EXPRS = [
    # Basic data type tests
    pl.col("int_col").cast(pl.Float64) > 2.5,
    pl.col("str_col").alias("renamed_str") == "c",
    pl.col("nullable_int").is_not_null(),
    # Timezone-aware datetime operations
    pl.col("datetime_utc").dt.hour().alias("utc_hour") >= 12,
    pl.col("datetime_utc").dt.convert_time_zone("US/Eastern").alias("utc_to_est").dt.hour() >= 5,
    # Complex ternary/conditional expressions
    pl.when(pl.col("int_col") > 3).then(pl.col("str_col")).otherwise(pl.lit("default")).alias("conditional") == "d",
    pl.when(pl.col("list_col").list.len() == 0)
    .then(pl.lit("empty"))
    .when(pl.col("list_col").list.len() == 1)
    .then(pl.lit("single"))
    .otherwise(pl.lit("multiple"))
    .alias("list_category")
    == "multiple",
    # Variable length list operations
    pl.col("list_col").list.len() > 2,
    pl.col("list_col").list.len() == 0,  # Test empty lists
    pl.col("list_col").list.sum().alias("list_sum").fill_null(0) > 10,
    # Multiple expressions (tests the AND combination optimization)
    [pl.col("int_col") > 2, pl.col("str_col") != "b"],
    [pl.col("bool_col"), pl.col("nullable_int").is_not_null(), pl.col("date_col").dt.year() == 2023],
    [
        pl.col("bool_col") | pl.col("nullable_int").is_not_null(),
        pl.col("date_col").dt.year() == 2023,
        pl.any_horizontal(pl.col("int_col") > 3, pl.col("float_col") < 5.0),
    ],
]


@pytest.mark.parametrize("expr", _COMPREHENSIVE_EXPRS)
def test_filter_no_pushdown_comprehensive(expr):
    """Test filter_no_pushdown with focused scenarios:
    - Basic casts and aliases
    - Timezone-aware datetime operations
    - Complex ternary/conditional expressions (pl.when chains)
    - Variable length list operations
    - Multiple expressions (tests single AND'ed temporary column optimization)
    """
    lf = _complex_sample_lf()
    lf = io_source_assert(lf.collect(), lambda _: True)
    _verify_filter_no_pushdown(lf, expr)


def test_filter_no_pushdown_edge_cases():
    """Test edge cases and complex multi-type expressions."""
    lf = _complex_sample_lf()
    lf = io_source_assert(lf.collect(), lambda _: True)

    # Empty filter list
    result = lf.piot.filter_no_pushdown([])
    assert_frame_equal(result.collect(), lf.collect())

    # Complex expression combining multiple data types, casts, aliases, and timezone operations
    complex_expr = [
        (pl.col("int_col").cast(pl.Float64) > 2.0)
        & (pl.col("str_col").alias("renamed") != "x")
        & (pl.col("nullable_int").is_not_null())
        & (pl.col("date_col").dt.year() == 2023)
        & (pl.col("datetime_utc").dt.hour().alias("utc_hour") >= 10)
    ]

    _verify_filter_no_pushdown(lf, complex_expr)


class TestRegisterIoSourceWithIsPure:
    """Test the register_io_source_with_is_pure utility function."""

    def test_cse_optimization_with_is_pure(self):
        """Test that is_pure=True enables CSE optimization in self-joins."""
        # Shared counter and lock to track how many times the io_source is called
        counter = 0
        lock = threading.Lock()

        def counting_io_source(
            with_columns: list[str] | None,
            predicate: pl.Expr | None,
            n_rows: int | None,
            batch_size: int | None,
        ) -> Iterator[pl.DataFrame]:
            nonlocal counter
            with lock:
                counter += 1

            # Return a simple DataFrame
            df = pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})

            # Apply any filters/selections that were pushed down
            if predicate is not None:
                df = df.filter(predicate)
            if with_columns is not None:
                df = df.select(with_columns)
            if n_rows is not None:
                df = df.head(n_rows)

            yield df

        schema = {"id": pl.Int64, "value": pl.Int64}

        # Create LazyFrame using our helper function
        lf = register_io_source_with_is_pure(counting_io_source, schema=schema)

        # Perform a self-join - this should trigger CSE if is_pure=True is working
        result = lf.join(lf, on="id", how="inner").collect()

        # Verify the result is correct
        expected = pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30], "value_right": [10, 20, 30]})
        assert result.equals(expected)

        # Check if CSE is working based on Polars version
        if version.parse(pl.__version__) >= version.parse("1.33.1"):
            # With is_pure=True, CSE should work and counter should be 1
            assert counter == 1, f"Expected counter=1 (CSE working), but got counter={counter}"

    def test_kwargs_passthrough(self):
        """Test that additional kwargs are passed through correctly."""

        def dummy_io_source(with_columns, predicate, n_rows, batch_size):
            # incorrect schema
            yield pl.DataFrame({"y": [1, 2, 3]})

        schema = {"x": pl.Int64}

        # Test with additional kwargs
        result = register_io_source_with_is_pure(dummy_io_source, schema=schema, validate_schema=True)

        assert isinstance(result, pl.LazyFrame)

        # This should error if validate_schema=True was passed correctly
        with pytest.raises(SchemaError):
            result.collect()


class TestWrapIoSourceWithErrorCatching:
    """Test the wrap_io_source_with_error_catching utility function."""

    def test_error_catching_preserves_function_metadata(self):
        """Test that functools.wraps preserves original function metadata."""

        def original_io_source(with_columns, predicate, n_rows, batch_size):
            """Original docstring for testing."""
            yield pl.DataFrame({"x": [1, 2, 3]})

        # Add some custom attributes for testing
        original_io_source.custom_attr = "test_value"

        wrapped = wrap_io_source_with_error_catching(original_io_source, identifier="test")

        # Check that metadata is preserved
        assert wrapped.__name__ == "original_io_source"
        assert wrapped.__doc__ == "Original docstring for testing."
        assert hasattr(wrapped, "__wrapped__")
        assert wrapped.__wrapped__ is original_io_source

    def test_error_catching_normal_operation(self):
        """Test that error catching doesn't interfere with normal operation."""

        def working_io_source(with_columns, predicate, n_rows, batch_size):
            """A working IO source."""
            df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})

            # Apply filters/selections
            if predicate is not None:
                df = df.filter(predicate)
            if with_columns is not None:
                df = df.select(with_columns)
            if n_rows is not None:
                df = df.head(n_rows)

            yield df

        wrapped = wrap_io_source_with_error_catching(working_io_source, identifier="test")

        # Test normal operation
        results = list(wrapped(with_columns=["x"], predicate=pl.col("x") > 1, n_rows=2, batch_size=None))

        assert len(results) == 1
        df = results[0]
        expected = pl.DataFrame({"x": [2, 3]})
        assert_frame_equal(df, expected)

    def test_error_catching_catches_exceptions(self, caplog):
        """Test that error catching re-raises with detailed context, without logging."""

        class CustomError(Exception):
            pass

        def failing_io_source(with_columns, predicate, n_rows, batch_size):
            """An IO source that always fails."""
            raise CustomError("Something went wrong!")
            yield  # This line is never reached

        wrapped = wrap_io_source_with_error_catching(failing_io_source, identifier="test_failing")

        # Test that the exception is caught and re-raised with context
        with pytest.raises(RuntimeError) as exc_info:
            list(wrapped(with_columns=None, predicate=None, n_rows=None, batch_size=None))

        # Check the exception message contains our context
        assert "IO Source 'failing_io_source' failed" in str(exc_info.value)
        assert "Something went wrong!" in str(exc_info.value)

        # Check that the original exception is chained
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, CustomError)

        # Check that detailed error information is included in the exception message
        msg = str(exc_info.value)
        assert "=== IO SOURCE ERROR ===" in msg
        assert "Function: failing_io_source" in msg
        assert "Identifier: test_failing" in msg
        assert "Error Type: CustomError" in msg
        assert "Error Message: Something went wrong!" in msg

    def test_error_catching_with_generator_exception(self, caplog):
        """Test error catching when exception occurs during generator execution."""

        def failing_generator_io_source(with_columns, predicate, n_rows, batch_size):
            """An IO source that fails during iteration."""
            yield pl.DataFrame({"x": [1]})  # First yield succeeds
            raise ValueError("Generator failed on second iteration")
            yield pl.DataFrame({"x": [2]})  # This is never reached

        wrapped = wrap_io_source_with_error_catching(failing_generator_io_source, identifier="generator_test")

        generator = wrapped(with_columns=None, predicate=None, n_rows=None, batch_size=None)

        # First iteration should work
        first_result = next(generator)
        expected_first = pl.DataFrame({"x": [1]})
        assert_frame_equal(first_result, expected_first)

        # Second iteration should raise our wrapped exception
        with pytest.raises(RuntimeError) as exc_info:
            next(generator)

        assert "IO Source 'failing_generator_io_source' failed" in str(exc_info.value)
        assert "Generator failed on second iteration" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_error_catching_preserves_arguments_in_error(self, caplog):
        """Test that error messages include the arguments passed to the function (no logging)."""

        def failing_io_source(with_columns, predicate, n_rows, batch_size):
            raise Exception("Test error")
            yield

        wrapped = wrap_io_source_with_error_catching(failing_io_source, identifier="args_test")

        test_predicate = (pl.col("x") > 5) & (pl.col("y") < 11)
        test_columns = ["x", "y"]

        with pytest.raises(RuntimeError) as exc_info:
            list(wrapped(with_columns=test_columns, predicate=test_predicate, n_rows=10, batch_size=100))

        # Check that arguments are included in the exception message (not logs)
        msg = str(exc_info.value)
        assert "Call Arguments:" in msg
        assert "with_columns" in msg
        assert "predicate" in msg
        assert "n_rows" in msg
        assert "batch_size" in msg
        # Check full string representation of predicate is included
        assert str(test_predicate) in msg

    def test_error_catching_converts_expr_in_args_to_string(self, caplog):
        """Test that pl.Expr instances in positional args are converted to full string representation."""

        def failing_io_source_with_args(arg1, arg2, arg3, arg4):
            """IO source that takes positional args."""
            raise Exception("Test error with args")
            yield

        wrapped = wrap_io_source_with_error_catching(failing_io_source_with_args, identifier="args_expr_test")

        # Create a complex expression that would be truncated in repr but not in str
        complex_expr = (
            pl.col("data_date").is_between(pl.lit("2024-01-01"), pl.lit("2024-12-31")) & (pl.col("region") == "US") & (pl.col("value") > 100)
        )

        with pytest.raises(RuntimeError) as exc_info:
            list(wrapped(None, complex_expr, None, 100000))

        msg = str(exc_info.value)
        # The full expression should be in the error message as a string,
        # not truncated like the repr would be
        assert "data_date" in msg
        assert "is_between" in msg
        assert "region" in msg
        assert "value" in msg
        # Should NOT have the truncated repr format like "<Expr [...] at 0x...>"
        assert "<Expr [" not in msg

    def test_error_catching_handles_mixed_args_and_kwargs(self, caplog):
        """Test that both args and kwargs with pl.Expr are properly converted."""

        def failing_io_source_mixed(arg1, arg2, **kwargs):
            """IO source with mixed positional and keyword args."""
            raise Exception("Mixed args error")
            yield

        wrapped = wrap_io_source_with_error_catching(failing_io_source_mixed, identifier="mixed_test")

        expr_in_args = pl.col("x").cast(pl.Float64) + pl.col("y")
        expr_in_kwargs = pl.col("z").is_in([1, 2, 3])

        with pytest.raises(RuntimeError) as exc_info:
            list(wrapped(expr_in_args, "normal_arg", predicate=expr_in_kwargs, other="value"))

        msg = str(exc_info.value)
        # Both expressions should be fully represented
        assert "cast" in msg or "Float64" in msg  # from args
        assert "is_in" in msg  # from kwargs
        # Should NOT have truncated repr
        assert "<Expr [" not in msg

    def test_catching_then_debug_logging_no_error(self, caplog):
        """Catching the error and logging debug with exc_info should emit no error logs,
        and debug logs should contain the full enhanced error info."""

        import logging

        class Boom(Exception):
            pass

        def failing_io_source(with_columns, predicate, n_rows, batch_size):
            raise Boom("kaboom!")
            yield  # unreachable

        wrapped = wrap_io_source_with_error_catching(failing_io_source, identifier="debug_logging_test")

        caplog.clear()
        caplog.set_level(logging.DEBUG)

        # Trigger and catch the error, then log at debug with exc_info
        try:
            list(wrapped(with_columns=None, predicate=None, n_rows=None, batch_size=None))
        except RuntimeError:
            logging.debug("Captured wrapped IO error", exc_info=True)

        # Ensure no error-level log was emitted by our wrapper
        assert all(record.levelno < logging.ERROR for record in caplog.records), "Unexpected error logs present"

        # Ensure our debug log contains the enhanced error info (from exception formatting)
        text = caplog.text
        assert "=== IO SOURCE ERROR ===" in text
        assert "Function: failing_io_source" in text
        assert "Identifier: debug_logging_test" in text
        assert "Error Type: Boom" in text
        assert "Error Message: kaboom!" in text


class TestWithColumnsTopo:
    """Tests for with_columns_topo covering simple and complex dependencies."""

    def test_simple_dependency_two_columns(self):
        lf = pl.LazyFrame({"b": [1, 2]})
        out = with_columns_topo(
            lf,
            [
                (pl.col("b") + 1).name.suffix("_2"),
                (pl.col("b_2") + 1).name.suffix("_3"),
            ],
        )
        df = out.select(["b_2", "b_2_3"]).collect()
        assert_frame_equal(df, pl.DataFrame({"b_2": [2, 3], "b_2_3": [3, 4]}))

    def test_chain_dependencies(self):
        # a depends on b, b depends on c
        lf = pl.LazyFrame({"c": [1, 2, 3]})
        exprs = [
            (pl.col("c") * 2).alias("b"),
            (pl.col("b") + 1).alias("a"),
        ]
        out = with_columns_topo(lf, exprs)
        df = out.select(["c", "b", "a"]).collect()
        assert_frame_equal(df, pl.DataFrame({"c": [1, 2, 3], "b": [2, 4, 6], "a": [3, 5, 7]}))

    def test_diamond_dependencies(self):
        # b and c both depend on base x, and d depends on both b and c
        lf = pl.LazyFrame({"x": [1, 2, 3]})
        exprs = [
            (pl.col("b") + pl.col("c")).alias("d"),
            (pl.col("x") + 1).alias("b"),
            (pl.col("x") * 2).alias("c"),
        ]
        out = with_columns_topo(lf, exprs)
        df = out.select(["b", "c", "d"]).collect()
        assert_frame_equal(df, pl.DataFrame({"b": [2, 3, 4], "c": [2, 4, 6], "d": [4, 7, 10]}))

    def test_diamond_dependencies_multiple_input(self):
        # b and c both depend on base x, and d depends on both b and c
        lf = pl.LazyFrame({"x": [1, 2, 3]})
        exprs = [
            (pl.col("b") + pl.col("c")).alias("e"),
            (pl.col("x") + 1).alias("b"),
            (pl.col("x") * 2).alias("c"),
            pl.sum_horizontal(pl.col("b"), pl.col("c")).alias("d"),
        ]
        out = with_columns_topo(lf, exprs)
        df = out.select(["b", "c", "e", "d"]).collect()
        assert_frame_equal(
            df,
            pl.DataFrame({"b": [2, 3, 4], "c": [2, 4, 6], "e": [4, 7, 10], "d": [4, 7, 10]}),
        )

    def test_diamond_dependencies_multiple_input_boolean(self):
        # b and c both depend on base x; d_sum depends on both via sum_horizontal;
        # d_any depends on boolean conditions over both via any_horizontal
        lf = pl.LazyFrame({"x": [1, 2, 3]})
        exprs = [
            (pl.col("x") + 1).alias("b"),
            (pl.col("x") * 2).alias("c"),
            pl.sum_horizontal(pl.col("b"), pl.col("c")).alias("d_sum"),
            pl.any_horizontal(pl.col("b") > 2, pl.col("c") > 3).alias("d_any"),
        ]
        out = with_columns_topo(lf, exprs)
        df = out.select(["b", "c", "d_sum", "d_any"]).collect()
        assert_frame_equal(
            df,
            pl.DataFrame(
                {
                    "b": [2, 3, 4],
                    "c": [2, 4, 6],
                    "d_sum": [4, 7, 10],
                    "d_any": [False, True, True],
                }
            ),
        )

    def test_independent_batched(self):
        # Ensure independent expressions are batched in one with_columns call
        lf = pl.LazyFrame({"x": [1, 2, 3]})
        exprs = [
            (pl.col("x") + 1).alias("a"),
            (pl.col("x") * 2).alias("b"),
            (pl.col("x") - 3).alias("c"),
        ]
        out = with_columns_topo(lf, exprs)
        plan = out.explain()
        # Ensure independent expressions were batched in a single WITH_COLUMNS stage
        assert plan.count("WITH_COLUMNS") == 1
        assert 'alias("a")' in plan
        assert 'alias("b")' in plan
        assert 'alias("c")' in plan
        df = out.select(["a", "b", "c"]).collect()
        assert_frame_equal(df, pl.DataFrame({"a": [2, 3, 4], "b": [2, 4, 6], "c": [-2, -1, 0]}))

    def test_mixed_interleaved_dependencies(self):
        # Complex graph: a,b independent; d depends on a; e depends on b and d
        lf = pl.LazyFrame({"x": [1, 2]})
        exprs = [
            (pl.col("x") + 1).alias("a"),
            (pl.col("x") * 3).alias("b"),
            (pl.col("a") * 2).alias("d"),
            (pl.col("b") + pl.col("d")).alias("e"),
        ]
        out = lf.piot.with_columns_topo(exprs)
        df = out.select(["a", "b", "d", "e"]).collect()
        assert_frame_equal(df, pl.DataFrame({"a": [2, 3], "b": [3, 6], "d": [4, 6], "e": [7, 12]}))

    def test_double_alias_chaining(self):
        # Double alias chaining on a single expression path
        lf = pl.LazyFrame({"b": [1, 2]})
        exprs = [
            (pl.col("b") + 1).alias("b2").alias("b3"),
        ]
        out = lf.piot.with_columns_topo(exprs)
        df = out.select(["b3"]).collect()
        assert_frame_equal(df, pl.DataFrame({"b3": [2, 3]}))

    def test_double_alias_chain_dependency_on_final(self):
        # A dependent column referencing the final alias should work
        lf = pl.LazyFrame({"b": [1, 2]})
        exprs = [
            (pl.col("b") + 1).alias("b2").alias("b3"),
            (pl.col("b3") + 1).alias("b4"),
        ]
        out = lf.piot.with_columns_topo(exprs)
        df = out.select(["b3", "b4"]).collect()
        assert_frame_equal(df, pl.DataFrame({"b3": [2, 3], "b4": [3, 4]}))

    def test_double_alias_chain_dependency_on_intermediate_errors(self):
        # Referencing the intermediate alias (b2) should error because only the final alias (b3) exists
        lf = pl.LazyFrame({"b": [1, 2]})
        exprs = [
            (pl.col("b") + 1).alias("b2").alias("b3"),
            (pl.col("b2") + 1).alias("b4"),
        ]
        out = lf.piot.with_columns_topo(exprs)
        with pytest.raises(ColumnNotFoundError):
            out.collect()

    def test_cycle_detection(self):
        lf = pl.LazyFrame({"x": [1, 2]})
        exprs = [
            (pl.col("b") + 1).alias("a"),
            (pl.col("a") + 1).alias("b"),
        ]
        with pytest.raises(CycleError):
            lf.piot.with_columns_topo(exprs)

    def test_empty_exprs_returns_original(self):
        # Empty expressions list should return the original LF unchanged
        lf = _sample_lf()
        out = lf.piot.with_columns_topo([])
        assert_frame_equal(out.collect(), lf.collect())

    def test_large_topo_sort_scale(self):
        # Massive DAG: ~10k columns, max depth 15, varying sizes per layer
        rows = 3
        lf = pl.LazyFrame({"x": list(range(rows))})

        total_cols = 10_000
        num_layers = 15
        base = total_cols // num_layers
        rem = total_cols % num_layers
        sizes = [base + (1 if i < rem else 0) for i in range(num_layers)]

        exprs: list[pl.Expr] = []
        # Layer 0: depend on base column 'x'
        for k in range(sizes[0]):
            name = f"c0_{k}"
            exprs.append((pl.col("x") + pl.lit(k)).alias(name))

        # Subsequent layers: each column depends on same-index column from previous layer
        for layer in range(1, num_layers):
            prev_size = sizes[layer - 1]
            for k in range(sizes[layer]):
                upstream = f"c{layer - 1}_{k % prev_size}"
                name = f"c{layer}_{k}"
                # Add layer index to propagate deterministic values
                exprs.append((pl.col(upstream) + pl.lit(layer)).alias(name))

        out = lf.piot.with_columns_topo(exprs)

        # Select a representative subset across layers to keep memory modest
        sample_cols = [
            "c0_0",
            f"c1_{sizes[1] - 1}",
            f"c5_{sizes[5] // 2}",
            f"c14_{(sizes[14] - 1)}",
        ]

        df = out.select(sample_cols).collect()

        # Expected values: c0_k = x + k; c_l_k = x + (k' from prev) + sum_{t=1..l} t
        def expected_for_layer(layer: int, k: int) -> list[int]:
            # Resolve k back to initial index through modulo chain; since we map k -> k % prev_size,
            # the base index effectively becomes k modulo sizes[l0] iteratively. For our construction,
            # taking modulo only once at each step preserves the original k for same-index mapping.
            base_k = k
            s = 0
            for t in range(1, layer + 1):
                s += t
            return [x + base_k + s for x in range(rows)]

        expected = pl.DataFrame(
            {
                sample_cols[0]: [x + 0 for x in range(rows)],
                sample_cols[1]: expected_for_layer(1, sizes[1] - 1),
                sample_cols[2]: expected_for_layer(5, sizes[5] // 2),
                sample_cols[3]: expected_for_layer(14, sizes[14] - 1),
            }
        )

        assert_frame_equal(df, expected)

    @pytest.mark.xfail(reason="Selectors expansion not yet supported by with_columns_topo")
    def test_with_columns_topo_with_selector(self):
        import polars.selectors as cs

        lf = pl.LazyFrame(
            {
                "a": [1.123, 2.456, 3.789],
                "b": [4.987, 5.654, 6.321],
                "c": ["foo", "bar", "baz"],
            }
        )
        exprs = [cs.numeric().as_expr().round(2)]
        out = with_columns_topo(lf, exprs)
        df = out.collect()
        expected = pl.DataFrame(
            {
                "a": [1.12, 2.46, 3.79],
                "b": [4.99, 5.65, 6.32],
                "c": ["foo", "bar", "baz"],
            }
        )
        assert_frame_equal(df, expected)

    @pytest.mark.xfail(reason="Implicit selectors expansion not yet supported by with_columns_topo")
    def test_with_columns_topo_with_implicit_selector_and_dependency(self):
        # Implicit selector via multi-column expr and .name.suffix; plus dependent expr
        lf = pl.LazyFrame(
            {
                "a": [1, 2, 3],
                "b": [10, 20, 30],
            }
        )

        exprs = [
            (pl.col("a", "b") + 1).name.suffix("_2"),  # creates a_2, b_2 implicitly
            (pl.col("a_2") + pl.col("b_2")).alias("sum2"),  # depends on outputs above
        ]

        out = with_columns_topo(lf, exprs)
        df = out.collect()

        expected = pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [10, 20, 30],
                "a_2": [2, 3, 4],
                "b_2": [11, 21, 31],
                "sum2": [13, 24, 35],
            }
        )

        # Order may differ; compare by selecting expected columns
        assert_frame_equal(df.select(expected.columns), expected)


class TestStorageOptionsFor:
    """Tests for `_storage_options_for` with thorough boto3/session mocking."""

    def test_non_s3_uri_returns_empty(self):
        opts = _storage_options_for("file:///tmp/cache")
        assert opts.pyarrow == {}
        assert opts.polars == {}

        opts2 = _storage_options_for("/local/path/cache")
        assert opts2.pyarrow == {}
        assert opts2.polars == {}

        opts3 = _storage_options_for("http://example.com/cache")
        assert opts3.pyarrow == {}
        assert opts3.polars == {}

    def test_s3_with_credentials_and_token(self, monkeypatch):
        class FakeCreds:
            def __init__(self, access_key: str, secret_key: str, token: str | None):
                self.access_key = access_key
                self.secret_key = secret_key
                self.token = token

        class FakeSession:
            def __init__(self, profile_name: str | None = None):
                # capture profile_name for later assertions if needed
                self.profile_name = profile_name
                self.region_name = "us-west-2"
                self._creds = FakeCreds("AKIA_TEST", "SECRET_TEST", "TOKEN_TEST")

            def get_credentials(self):
                return self._creds

        # Patch boto3.Session directly (boto3 is imported inside _storage_options_for)
        monkeypatch.setattr("boto3.Session", FakeSession)

        # Explicit endpoint via query should win over env
        monkeypatch.setenv("AWS_ENDPOINT_URL", "https://env-endpoint")
        monkeypatch.setenv("AWS_S3_ENDPOINT", "https://env-s3-endpoint")

        uri = "s3://bucket/path?endpoint_override=https://query-endpoint"
        opts = _storage_options_for(uri, aws_profile="test-profile")

        # pyarrow-style keys
        assert opts.pyarrow["access_key"] == "AKIA_TEST"
        assert opts.pyarrow["secret_key"] == "SECRET_TEST"
        assert opts.pyarrow["session_token"] == "TOKEN_TEST"
        assert opts.pyarrow["region"] == "us-west-2"
        assert opts.pyarrow["endpoint_override"] == "https://query-endpoint"

        # polars/deltalake-style keys
        assert opts.polars["aws_access_key_id"] == "AKIA_TEST"
        assert opts.polars["aws_secret_access_key"] == "SECRET_TEST"
        assert opts.polars["aws_session_token"] == "TOKEN_TEST"
        assert opts.polars["endpoint_url"] == "https://query-endpoint"

    def test_s3_with_credentials_no_token(self, monkeypatch):
        class FakeCreds:
            def __init__(self, access_key: str, secret_key: str, token: str | None):
                self.access_key = access_key
                self.secret_key = secret_key
                self.token = token

        class FakeSession:
            def __init__(self, profile_name: str | None = None):
                self.profile_name = profile_name
                self.region_name = "eu-central-1"
                self._creds = FakeCreds("AKIA_NO_TOKEN", "SECRET_NO_TOKEN", None)

            def get_credentials(self):
                return self._creds

        monkeypatch.setattr("boto3.Session", FakeSession)

        # No endpoint in query; env should set endpoint
        monkeypatch.setenv("AWS_ENDPOINT_URL", "https://env-only-endpoint")
        monkeypatch.delenv("AWS_S3_ENDPOINT", raising=False)

        opts = _storage_options_for("s3://bucket/path")

        # Token keys should not be present
        assert "session_token" not in opts.pyarrow
        assert "aws_session_token" not in opts.polars

        # Region is present only in pyarrow opts
        assert opts.pyarrow["region"] == "eu-central-1"
        assert "region" not in opts.polars

        # Endpoint derived from env
        assert opts.pyarrow["endpoint_override"] == "https://env-only-endpoint"
        assert opts.polars["endpoint_url"] == "https://env-only-endpoint"

    def test_s3a_scheme_supported(self, monkeypatch):
        class FakeCreds:
            def __init__(self, access_key: str, secret_key: str, token: str | None):
                self.access_key = access_key
                self.secret_key = secret_key
                self.token = token

        class FakeSession:
            def __init__(self, profile_name: str | None = None):
                self.profile_name = profile_name
                self.region_name = None
                self._creds = FakeCreds("AKIA_S3A", "SECRET_S3A", None)

            def get_credentials(self):
                return self._creds

        class FakeCredentialProvider:
            def __init__(self, **kwargs):
                pass

            def _storage_update_options(self):
                return {}

        monkeypatch.setattr("boto3.Session", FakeSession)
        monkeypatch.setattr("polars_io_tools.io_sources.util.pl.CredentialProviderAWS", FakeCredentialProvider, raising=False)
        # Ensure neither env var is set
        monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("AWS_S3_ENDPOINT", raising=False)

        opts = _storage_options_for("s3a://bucket/path")

        assert opts.pyarrow["access_key"] == "AKIA_S3A"
        assert opts.pyarrow["secret_key"] == "SECRET_S3A"
        assert "session_token" not in opts.pyarrow
        # No region key when region_name is None
        assert "region" not in opts.pyarrow
        # No endpoint keys when none provided
        assert "endpoint_override" not in opts.pyarrow
        assert "endpoint_url" not in opts.polars
        assert opts.polars["aws_access_key_id"] == "AKIA_S3A"
        assert opts.polars["aws_secret_access_key"] == "SECRET_S3A"
        assert "aws_session_token" not in opts.polars

    def test_endpoint_env_precedence(self, monkeypatch):
        class FakeCreds:
            def __init__(self, access_key: str, secret_key: str, token: str | None):
                self.access_key = access_key
                self.secret_key = secret_key
                self.token = token

        class FakeSession:
            def __init__(self, profile_name: str | None = None):
                self.profile_name = profile_name
                self.region_name = "ap-southeast-1"
                self._creds = FakeCreds("AKIA_ENV", "SECRET_ENV", None)

            def get_credentials(self):
                return self._creds

        monkeypatch.setattr("boto3.Session", FakeSession)

        # Both env vars set; AWS_ENDPOINT_URL should win when no query param
        monkeypatch.setenv("AWS_ENDPOINT_URL", "https://endpoint-url")
        monkeypatch.setenv("AWS_S3_ENDPOINT", "https://s3-endpoint")

        opts = _storage_options_for("s3://bucket/path")
        assert opts.pyarrow["endpoint_override"] == "https://endpoint-url"
        assert opts.polars["endpoint_url"] == "https://endpoint-url"

        # If query param provided, it should override env
        opts2 = _storage_options_for("s3://bucket/path?endpoint_override=https://query")
        assert opts2.pyarrow["endpoint_override"] == "https://query"
        assert opts2.polars["endpoint_url"] == "https://query"

    def test_no_credentials_only_region_and_endpoint(self, monkeypatch):
        class FakeSession:
            def __init__(self, profile_name: str | None = None):
                self.profile_name = profile_name
                self.region_name = "us-east-1"

            def get_credentials(self):
                return None

        monkeypatch.setattr("boto3.Session", FakeSession)
        monkeypatch.setenv("AWS_S3_ENDPOINT", "https://only-env-endpoint")
        monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)

        opts = _storage_options_for("s3://bucket/path")

        # No access keys when creds is None
        assert "access_key" not in opts.pyarrow
        assert "secret_key" not in opts.pyarrow
        assert "session_token" not in opts.pyarrow
        assert "aws_access_key_id" not in opts.polars
        assert "aws_secret_access_key" not in opts.polars
        assert "aws_session_token" not in opts.polars

        # Region still present for pyarrow
        assert opts.pyarrow["region"] == "us-east-1"
        # Endpoint present in both
        assert opts.pyarrow["endpoint_override"] == "https://only-env-endpoint"
        assert opts.polars["endpoint_url"] == "https://only-env-endpoint"

    def test_profile_name_forwarded_to_boto3_session(self, monkeypatch):
        captured = {"profile": None}

        class FakeSession:
            def __init__(self, profile_name: str | None = None):
                captured["profile"] = profile_name
                self.region_name = None

            def get_credentials(self):
                class C:
                    access_key = "AKIA_PROFILE"
                    secret_key = "SECRET_PROFILE"
                    token = None

                return C()

        class FakeCredentialProvider:
            def __init__(self, **kwargs):
                pass

            def _storage_update_options(self):
                return {}

        monkeypatch.setattr("boto3.Session", FakeSession)
        monkeypatch.setattr("polars_io_tools.io_sources.util.pl.CredentialProviderAWS", FakeCredentialProvider, raising=False)

        _ = _storage_options_for("s3://bucket/path", aws_profile="dev-profile")

        assert captured["profile"] == "dev-profile"

    def test_tuple_decomposition_non_s3(self):
        pa_opts, pl_opts, cred_provider = _storage_options_for("file:///tmp/cache")
        assert pa_opts == {}
        assert pl_opts == {}
        assert cred_provider is None

    def test_tuple_decomposition_s3_with_token(self, monkeypatch):
        class FakeCreds:
            def __init__(self, access_key: str, secret_key: str, token: str | None):
                self.access_key = access_key
                self.secret_key = secret_key
                self.token = token

        class FakeSession:
            def __init__(self, profile_name: str | None = None):
                self.profile_name = profile_name
                self.region_name = "us-west-1"
                self._creds = FakeCreds("AKIA_TUPLE", "SECRET_TUPLE", "TOKEN_TUPLE")

            def get_credentials(self):
                return self._creds

        class FakeCredentialProvider:
            def __init__(self, **kwargs):
                pass

            def _storage_update_options(self):
                return {}

        monkeypatch.setattr("boto3.Session", FakeSession)
        monkeypatch.setattr("polars_io_tools.io_sources.util.pl.CredentialProviderAWS", FakeCredentialProvider, raising=False)
        monkeypatch.setenv("AWS_ENDPOINT_URL", "https://tuple-endpoint")

        uri = "s3://bucket/path?endpoint_override=https://query-endpoint"
        pa_opts, pl_opts, cred_provider = _storage_options_for(uri, aws_profile="tuple-profile")

        assert pa_opts["access_key"] == "AKIA_TUPLE"
        assert pa_opts["secret_key"] == "SECRET_TUPLE"
        assert pa_opts["session_token"] == "TOKEN_TUPLE"
        assert pa_opts["region"] == "us-west-1"
        assert pa_opts["endpoint_override"] == "https://query-endpoint"

        assert pl_opts["aws_access_key_id"] == "AKIA_TUPLE"
        assert pl_opts["aws_secret_access_key"] == "SECRET_TUPLE"
        assert pl_opts["aws_session_token"] == "TOKEN_TUPLE"
        assert pl_opts["endpoint_url"] == "https://query-endpoint"
        assert cred_provider is not None

    def test_tuple_decomposition_s3_no_token(self, monkeypatch):
        class FakeCreds:
            def __init__(self, access_key: str, secret_key: str, token: str | None):
                self.access_key = access_key
                self.secret_key = secret_key
                self.token = token

        class FakeSession:
            def __init__(self, profile_name: str | None = None):
                self.profile_name = profile_name
                self.region_name = "eu-west-3"
                self._creds = FakeCreds("AKIA_TUPLE_NO", "SECRET_TUPLE_NO", None)

            def get_credentials(self):
                return self._creds

        class FakeCredentialProvider:
            def __init__(self, **kwargs):
                pass

            def _storage_update_options(self):
                return {}

        monkeypatch.setattr("boto3.Session", FakeSession)
        monkeypatch.setattr("polars_io_tools.io_sources.util.pl.CredentialProviderAWS", FakeCredentialProvider, raising=False)
        monkeypatch.setenv("AWS_S3_ENDPOINT", "https://tuple-env-endpoint")
        monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)

        pa_opts, pl_opts, cred_provider = _storage_options_for("s3://bucket/path")

        assert "session_token" not in pa_opts
        assert "aws_session_token" not in pl_opts
        assert pa_opts["region"] == "eu-west-3"
        assert pa_opts["endpoint_override"] == "https://tuple-env-endpoint"
        assert pl_opts["endpoint_url"] == "https://tuple-env-endpoint"
        assert cred_provider is not None


class TestResolveEndpointHostname:
    """Tests for `_resolve_endpoint_hostname` function."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear the LRU cache before each test to ensure isolation."""
        _resolve_endpoint_hostname.cache_clear()

    def test_resolve_hostname_with_port(self, monkeypatch):
        """Test resolving hostname to IP when endpoint has a port."""
        monkeypatch.setattr("socket.gethostbyname", lambda hostname: "10.1.2.3")

        result = _resolve_endpoint_hostname("http://gridprodobs:9020")

        assert result == "http://10.1.2.3:9020"

    def test_resolve_hostname_with_fqdn_and_port(self, monkeypatch):
        """Test resolving fully qualified domain name to IP."""
        monkeypatch.setattr("socket.gethostbyname", lambda hostname: "10.4.5.6")

        result = _resolve_endpoint_hostname("http://gridprodobs.saccap.int.:9020")

        assert result == "http://10.4.5.6:9020"

    def test_resolve_hostname_without_port(self, monkeypatch):
        """Test resolving hostname to IP when endpoint has no port."""
        monkeypatch.setattr("socket.gethostbyname", lambda hostname: "192.168.1.100")

        result = _resolve_endpoint_hostname("http://example.com")

        assert result == "http://192.168.1.100"

    def test_resolve_https_endpoint(self, monkeypatch):
        """Test resolving HTTPS endpoints."""
        monkeypatch.setattr("socket.gethostbyname", lambda hostname: "10.0.0.1")

        result = _resolve_endpoint_hostname("https://secure.example.com:443")

        assert result == "https://10.0.0.1:443"

    def test_resolve_hostname_with_path(self, monkeypatch):
        """Test that paths are preserved in the resolved URL."""
        monkeypatch.setattr("socket.gethostbyname", lambda hostname: "10.1.2.3")

        result = _resolve_endpoint_hostname("http://gridprodobs:9020/some/path")

        assert result == "http://10.1.2.3:9020/some/path"

    def test_resolve_failure_returns_original(self, monkeypatch, caplog):
        """Test that DNS resolution failure returns the original endpoint and logs a warning."""
        import socket as socket_module

        def failing_gethostbyname(hostname):
            raise socket_module.gaierror("Name or service not known")

        monkeypatch.setattr("socket.gethostbyname", failing_gethostbyname)

        result = _resolve_endpoint_hostname("http://nonexistent.host:9020")

        assert result == "http://nonexistent.host:9020"
        assert "Failed to resolve hostname" in caplog.text

    def test_resolve_no_hostname_returns_original(self):
        """Test that endpoints with no hostname are returned unchanged."""
        # URL with no hostname (unusual but possible)
        result = _resolve_endpoint_hostname("file:///path/to/file")

        assert result == "file:///path/to/file"

    def test_gethostbyname_called_with_correct_hostname(self, monkeypatch):
        """Verify that socket.gethostbyname is called with the correct hostname."""
        captured_hostname = {}

        def capture_gethostbyname(hostname):
            captured_hostname["value"] = hostname
            return "10.1.2.3"

        monkeypatch.setattr("socket.gethostbyname", capture_gethostbyname)

        _resolve_endpoint_hostname("http://gridprodobs.saccap.int.:9020")

        assert captured_hostname["value"] == "gridprodobs.saccap.int."

    def test_gethostbyname_called_with_short_hostname(self, monkeypatch):
        """Verify that socket.gethostbyname is called with short hostname."""
        captured_hostname = {}

        def capture_gethostbyname(hostname):
            captured_hostname["value"] = hostname
            return "10.7.8.9"

        monkeypatch.setattr("socket.gethostbyname", capture_gethostbyname)

        _resolve_endpoint_hostname("http://gridprodobs:9020")

        assert captured_hostname["value"] == "gridprodobs"


class TestOnpremEndpointResolutionIntegration:
    """Tests for on-prem endpoint hostname resolution integration in `_storage_options_for`."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear the LRU cache before each test to ensure isolation."""
        _resolve_endpoint_hostname.cache_clear()

    def test_onprem_endpoint_constants_exist(self):
        """Verify the on-prem endpoints set contains the expected values."""
        assert "http://gridprodobs.saccap.int.:9020" in _ONPREM_ENDPOINTS_TO_RESOLVE
        assert "http://gridprodobs:9020" in _ONPREM_ENDPOINTS_TO_RESOLVE

    def test_storage_options_resolves_onprem_endpoint_fqdn(self, monkeypatch):
        """Test that _storage_options_for resolves the FQDN on-prem endpoint to IP."""

        class FakeCreds:
            access_key = "AKIA_TEST"
            secret_key = "SECRET_TEST"
            token = None

        class FakeSession:
            def __init__(self, profile_name=None):
                self.region_name = "us-east-1"

            def get_credentials(self):
                return FakeCreds()

        class FakeCredentialProvider:
            def __init__(self, **kwargs):
                pass

            def _storage_update_options(self):
                return {}

        monkeypatch.setattr("boto3.Session", FakeSession)
        monkeypatch.setattr("polars_io_tools.io_sources.util.pl.CredentialProviderAWS", FakeCredentialProvider, raising=False)
        monkeypatch.setattr("socket.gethostbyname", lambda hostname: "10.100.200.1")
        monkeypatch.setenv("AWS_ENDPOINT_URL", "http://gridprodobs.saccap.int.:9020")

        opts = _storage_options_for("s3://bucket/path")

        # Endpoint should be resolved to IP
        assert opts.pyarrow["endpoint_override"] == "http://10.100.200.1:9020"
        assert opts.polars["endpoint_url"] == "http://10.100.200.1:9020"
        assert opts.polars["allow_http"] == "true"

    def test_storage_options_resolves_onprem_endpoint_short(self, monkeypatch):
        """Test that _storage_options_for resolves the short on-prem endpoint to IP."""

        class FakeCreds:
            access_key = "AKIA_TEST"
            secret_key = "SECRET_TEST"
            token = None

        class FakeSession:
            def __init__(self, profile_name=None):
                self.region_name = "us-east-1"

            def get_credentials(self):
                return FakeCreds()

        class FakeCredentialProvider:
            def __init__(self, **kwargs):
                pass

            def _storage_update_options(self):
                return {}

        monkeypatch.setattr("boto3.Session", FakeSession)
        monkeypatch.setattr("polars_io_tools.io_sources.util.pl.CredentialProviderAWS", FakeCredentialProvider, raising=False)
        monkeypatch.setattr("socket.gethostbyname", lambda hostname: "10.50.60.70")
        monkeypatch.setenv("AWS_ENDPOINT_URL", "http://gridprodobs:9020")

        opts = _storage_options_for("s3://bucket/path")

        # Endpoint should be resolved to IP
        assert opts.pyarrow["endpoint_override"] == "http://10.50.60.70:9020"
        assert opts.polars["endpoint_url"] == "http://10.50.60.70:9020"

    def test_storage_options_does_not_resolve_non_onprem_endpoint(self, monkeypatch):
        """Test that _storage_options_for does NOT resolve non-on-prem endpoints."""

        class FakeCreds:
            access_key = "AKIA_TEST"
            secret_key = "SECRET_TEST"
            token = None

        class FakeSession:
            def __init__(self, profile_name=None):
                self.region_name = "us-east-1"

            def get_credentials(self):
                return FakeCreds()

        class FakeCredentialProvider:
            def __init__(self, **kwargs):
                pass

            def _storage_update_options(self):
                return {}

        gethostbyname_called = {"called": False}

        def track_gethostbyname(hostname):
            gethostbyname_called["called"] = True
            return "10.0.0.1"

        monkeypatch.setattr("boto3.Session", FakeSession)
        monkeypatch.setattr("polars_io_tools.io_sources.util.pl.CredentialProviderAWS", FakeCredentialProvider, raising=False)
        monkeypatch.setattr("socket.gethostbyname", track_gethostbyname)
        monkeypatch.setenv("AWS_ENDPOINT_URL", "https://s3.amazonaws.com")

        opts = _storage_options_for("s3://bucket/path")

        # Endpoint should NOT be resolved - should stay as original
        assert opts.pyarrow["endpoint_override"] == "https://s3.amazonaws.com"
        assert opts.polars["endpoint_url"] == "https://s3.amazonaws.com"
        # gethostbyname should not have been called
        assert not gethostbyname_called["called"]

    def test_storage_options_resolution_failure_keeps_original(self, monkeypatch, caplog):
        """Test that DNS resolution failure in _storage_options_for keeps the original endpoint."""
        import socket as socket_module

        class FakeCreds:
            access_key = "AKIA_TEST"
            secret_key = "SECRET_TEST"
            token = None

        class FakeSession:
            def __init__(self, profile_name=None):
                self.region_name = "us-east-1"

            def get_credentials(self):
                return FakeCreds()

        class FakeCredentialProvider:
            def __init__(self, **kwargs):
                pass

            def _storage_update_options(self):
                return {}

        def failing_gethostbyname(hostname):
            raise socket_module.gaierror("Name or service not known")

        monkeypatch.setattr("boto3.Session", FakeSession)
        monkeypatch.setattr("polars_io_tools.io_sources.util.pl.CredentialProviderAWS", FakeCredentialProvider, raising=False)
        monkeypatch.setattr("socket.gethostbyname", failing_gethostbyname)
        monkeypatch.setenv("AWS_ENDPOINT_URL", "http://gridprodobs:9020")

        opts = _storage_options_for("s3://bucket/path")

        # Endpoint should be kept as original due to resolution failure
        assert opts.pyarrow["endpoint_override"] == "http://gridprodobs:9020"
        assert opts.polars["endpoint_url"] == "http://gridprodobs:9020"
        assert "Failed to resolve hostname" in caplog.text

    def test_storage_options_via_query_param_resolves_onprem(self, monkeypatch):
        """Test that on-prem endpoint specified via query param is also resolved."""

        class FakeCreds:
            access_key = "AKIA_TEST"
            secret_key = "SECRET_TEST"
            token = None

        class FakeSession:
            def __init__(self, profile_name=None):
                self.region_name = "us-east-1"

            def get_credentials(self):
                return FakeCreds()

        class FakeCredentialProvider:
            def __init__(self, **kwargs):
                pass

            def _storage_update_options(self):
                return {}

        monkeypatch.setattr("boto3.Session", FakeSession)
        monkeypatch.setattr("polars_io_tools.io_sources.util.pl.CredentialProviderAWS", FakeCredentialProvider, raising=False)
        monkeypatch.setattr("socket.gethostbyname", lambda hostname: "10.11.12.13")
        monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("AWS_S3_ENDPOINT", raising=False)

        uri = "s3://bucket/path?endpoint_override=http://gridprodobs.saccap.int.:9020"
        opts = _storage_options_for(uri)

        # Endpoint should be resolved to IP
        assert opts.pyarrow["endpoint_override"] == "http://10.11.12.13:9020"
        assert opts.polars["endpoint_url"] == "http://10.11.12.13:9020"
