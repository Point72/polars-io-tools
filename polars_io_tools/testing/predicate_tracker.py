"""
Predicate tracking and analysis utilities for testing filter pushdown.

This module provides tools for:
1. Creating IO sources that track pushed-down predicates
2. Analyzing the structure of pushed predicates
3. Extracting filter bounds and values from predicate trees
4. Asserting that predicates are pushed down correctly
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Iterable, Iterator, Optional, Union

import polars as pl
from polars.io.plugins import register_io_source
from polars.testing import assert_frame_equal

from polars_io_tools.io_sources.base import BinaryExprNode, FunctionNode, get_parsed_expr
from polars_io_tools.io_sources.enum import BooleanFunctionType, OperatorType

__all__ = ("PredicateTracker", "PredicateAnalyzer", "io_source_assert")


def _materialize_columns(
    columns: Union[str, Iterable[str], pl.Expr, Iterable[pl.Expr]],
) -> Union[str, pl.Expr, list]:
    """Normalize a ``columns`` argument so it can be iterated more than once.

    A single string or ``pl.Expr`` is returned as-is; any other iterable is
    materialized to a list. This avoids generator-exhaustion bugs in helpers
    that consume ``columns`` for both the source pipeline and a direct
    reference computation.
    """
    if isinstance(columns, (str, pl.Expr)):
        return columns
    return list(columns)


@dataclass
class PredicateTracker:
    """
    A utility class that creates an IO source which tracks pushed-down predicates.

    This is useful for testing that filters are correctly pushed down to sources
    in multi_source or other IO source implementations.

    Args:
        df (pl.DataFrame): The DataFrame to use as the underlying data source.

    Attributes:
        lazy_frame (pl.LazyFrame): The LazyFrame that can be used in multi_source or other operations.
        last_predicate (pl.Expr | None): The last predicate that was pushed down during collection.
        last_with_columns (list[str] | None): The last column projection that was pushed down.
        call_count (int): Number of times the source has been called.

    Example:
        ::

            tracker = PredicateTracker(df)
            lf = multi_source(
                sources={"data": (tracker.lazy_frame, {"date": FilterSpec()})},
                combine=lambda s: s["data"],
            )
            result = lf.filter(pl.col("date") >= date(2024, 1, 5)).collect()

            # Check what was pushed down
            assert tracker.last_predicate is not None
            analyzer = tracker.get_analyzer()
    """

    df: pl.DataFrame
    last_predicate: Optional[pl.Expr] = field(default=None, init=False)
    last_with_columns: Optional[list[str]] = field(default=None, init=False)
    call_count: int = field(default=0, init=False)
    _lazy_frame: Optional[pl.LazyFrame] = field(default=None, init=False, repr=False)

    @property
    def lazy_frame(self) -> pl.LazyFrame:
        """Get the LazyFrame that tracks predicates."""
        if self._lazy_frame is None:
            self._lazy_frame = self._create_tracking_source()
        return self._lazy_frame

    def _create_tracking_source(self) -> pl.LazyFrame:
        """Create the IO source that tracks predicates."""
        tracker = self  # Capture self for the closure

        def source_generator(
            with_columns: Optional[list[str]],
            predicate: Optional[pl.Expr],
            n_rows: Optional[int],
            batch_size: Optional[int],
        ) -> Iterator[pl.DataFrame]:
            tracker.last_predicate = predicate
            tracker.last_with_columns = with_columns
            tracker.call_count += 1

            result = tracker.df.clone()

            if with_columns is not None:
                result = result.select(with_columns)
            if predicate is not None:
                result = result.filter(predicate)
            if n_rows is not None:
                result = result.head(n_rows)

            yield result

        return register_io_source(io_source=source_generator, schema=self.df.schema, is_pure=True)

    def reset(self) -> None:
        """Reset the tracked state."""
        self.last_predicate = None
        self.last_with_columns = None
        self.call_count = 0

    def get_analyzer(self) -> PredicateAnalyzer:
        """
        Get an analyzer for the last pushed predicate.

        Returns:
            PredicateAnalyzer: An analyzer for inspecting the predicate structure.

        Raises:
            ValueError: If no predicate has been pushed yet.
        """
        if self.last_predicate is None:
            raise ValueError("No predicate has been pushed yet. Call collect() first.")
        return PredicateAnalyzer(self.last_predicate)

    def direct_filter(self, expr: pl.Expr) -> pl.DataFrame:
        """
        Apply the filter expression directly to the underlying DataFrame.

        This provides a baseline for comparison with the IO source results.

        Args:
            expr (pl.Expr): The filter expression to apply.

        Returns:
            pl.DataFrame: The filtered DataFrame.
        """
        return self.df.clone().filter(expr)

    def source_filter(self, expr: pl.Expr) -> pl.DataFrame:
        """
        Apply the filter expression through the IO source.

        This triggers predicate pushdown and captures the pushed predicate.
        The tracker is reset before filtering.

        Args:
            expr (pl.Expr): The filter expression to apply.

        Returns:
            pl.DataFrame: The filtered DataFrame from the IO source.
        """
        self.reset()
        return self.lazy_frame.filter(expr).collect()

    def assert_results_match(self, expr: pl.Expr) -> None:
        """
        Assert that filtering through the IO source gives the same result
        as filtering the DataFrame directly.

        This verifies correctness regardless of whether the predicate was pushed down.

        Args:
            expr (pl.Expr): The filter expression to test.

        Raises:
            AssertionError: If the results don't match.
        """
        direct_result = self.direct_filter(expr)
        source_result = self.source_filter(expr)
        assert_frame_equal(direct_result, source_result)

    def assert_predicate_pushed_down(
        self,
        expr: pl.Expr,
        assert_expr_predicate: Optional[Callable[[pl.Expr, Optional[pl.Expr]], None]] = None,
        expected_pushed_down: bool = True,
    ) -> None:
        """
        Assert that the predicate is pushed down and produces correct results.

        This method verifies:
        1. The predicate was (or wasn't) pushed down as expected
        2. The pushed predicate produces identical results to the original expression
        3. The source result matches the direct filter result

        Args:
            expr (pl.Expr): The filter expression to test.
            assert_expr_predicate (Callable[[pl.Expr, pl.Expr | None], None], optional):
                A custom assertion function that receives (original_expr, pushed_expr).
                Use this to verify specific properties of the pushed predicate.
            expected_pushed_down (bool, default True): Whether the predicate is expected to be pushed down.
                Set to False when testing expressions that should NOT be pushed down.

        Raises:
            AssertionError: If the predicate pushdown behavior doesn't match expectations,
                or if results don't match.

        Example:
            ::

                tracker = PredicateTracker(df)

                # Basic assertion - predicate should be pushed and results should match
                tracker.lazy_frame.filter(pl.col("date") >= date(2024, 1, 5)).collect()
                tracker.assert_predicate_pushed_down(pl.col("date") >= date(2024, 1, 5))

                # With custom assertion on the pushed predicate
                def check_predicate(original, pushed):
                    analyzer = PredicateAnalyzer(pushed)
                    lower, upper = analyzer.extract_temporal_bounds(
                        analyzer.find_temporal_filter("date")
                    )
                    assert lower == date(2024, 1, 2)  # Lookback applied

                tracker.assert_predicate_pushed_down(expr, check_predicate)
        """
        source_result = self.source_filter(expr)

        # Run custom assertion if provided
        if assert_expr_predicate is not None:
            assert_expr_predicate(expr, self.last_predicate)

        # Verify pushdown behavior
        if expected_pushed_down:
            assert self.last_predicate is not None, "Predicate was not pushed down as expected"
        else:
            assert self.last_predicate is None, "Predicate was pushed down when it shouldn't have been"

        # Source result must match a direct application of the original predicate
        # regardless of whether pushdown happened.
        original_expr_result = self.direct_filter(expr)
        assert_frame_equal(
            source_result,
            original_expr_result,
            check_row_order=False,
        )

        # When the predicate was pushed, additionally verify the pushed predicate
        # is semantically equivalent to the original.
        if expected_pushed_down:
            pushed_predicate_result = self.direct_filter(self.last_predicate)
            assert_frame_equal(
                pushed_predicate_result,
                original_expr_result,
                check_row_order=False,
            )

    def source_select(self, columns: Union[str, Iterable[str], pl.Expr, Iterable[pl.Expr]]) -> pl.DataFrame:
        """
        Apply a column selection through the IO source.

        This triggers projection pushdown and captures `last_with_columns`.
        The tracker is reset before selecting.

        Args:
            columns (str | Iterable[str] | pl.Expr | Iterable[pl.Expr]): The columns / select expressions to apply via ``LazyFrame.select``.

        Returns:
            pl.DataFrame: The selected DataFrame from the IO source.
        """
        self.reset()
        return self.lazy_frame.select(_materialize_columns(columns)).collect()

    def assert_projection_pushed_down(
        self,
        columns: Union[str, Iterable[str], pl.Expr, Iterable[pl.Expr]],
        expected_columns: Optional[Iterable[str]] = None,
        expected_pushed_down: bool = True,
    ) -> None:
        """
        Assert that a column selection is pushed down to the source.

        Verifies:
        1. The projection was (or wasn't) pushed down as expected.
        2. If pushed and ``expected_columns`` is given, ``last_with_columns`` matches
           that set (order-insensitive).
        3. The source result matches the direct ``df.select(columns)`` result.

        Args:
            columns (str | Iterable[str] | pl.Expr | Iterable[pl.Expr]): The column selection to apply via ``LazyFrame.select``.
            expected_columns (Iterable[str], optional): If provided, asserts that the pushed ``with_columns`` (as a set) equals
                this set. Useful for verifying that only a subset of source columns was
                requested. Ignored when ``expected_pushed_down`` is ``False``.
            expected_pushed_down (bool, default True): Whether the projection is expected to be pushed down. Set to ``False``
                to lock in cases where polars does not currently push.

        Raises:
            AssertionError: If the projection pushdown behavior or the resulting frame does not
                match expectations.
        """
        columns = _materialize_columns(columns)
        source_result = self.source_select(columns)

        if expected_pushed_down:
            assert self.last_with_columns is not None, "Projection was not pushed down as expected"
            if expected_columns is not None:
                assert set(self.last_with_columns) == set(expected_columns), (
                    f"Pushed projection {self.last_with_columns!r} does not match expected {list(expected_columns)!r}"
                )
        else:
            assert self.last_with_columns is None, f"Projection was pushed down ({self.last_with_columns!r}) when it shouldn't have been"

        direct_result = self.df.clone().select(columns)
        assert_frame_equal(source_result, direct_result, check_row_order=False)

    def assert_pushed_down(
        self,
        *,
        predicate: Optional[pl.Expr] = None,
        projection: Optional[Union[str, Iterable[str], pl.Expr, Iterable[pl.Expr]]] = None,
        expected_predicate_pushed: Optional[bool] = None,
        expected_projection_pushed: Optional[bool] = None,
        expected_columns: Optional[Iterable[str]] = None,
        assert_expr_predicate: Optional[Callable[[pl.Expr, Optional[pl.Expr]], None]] = None,
    ) -> None:
        """
        Assert combined predicate and projection pushdown behavior.

        Runs ``self.lazy_frame.filter(predicate).select(projection).collect()`` (omitting
        either step if its argument is ``None``) and asserts pushdown expectations on
        both axes plus result equivalence to the direct DataFrame computation.

        ``filter`` is applied before ``select`` so that the predicate may reference
        columns that the projection drops.

        At least one of ``predicate`` or ``projection`` must be provided.

        For tests that need to insert reshape operations (e.g. ``unpivot`` / ``pivot``)
        between the source and the select/filter steps, just collect the LazyFrame
        you built and assert on ``tracker.last_predicate`` / ``tracker.last_with_columns``
        directly — this helper is intentionally limited to the simple source → filter
        → select shape.

        Args:
            predicate (pl.Expr, optional): Filter expression to apply via ``LazyFrame.filter``. If omitted, no
                filter step is added.
            projection (str | Iterable[str] | pl.Expr | Iterable[pl.Expr], optional):
                Column selection to apply via ``LazyFrame.select``. If omitted, no
                select step is added.
            expected_predicate_pushed (bool, optional): Whether ``predicate`` is expected to be pushed down. ``None`` (default)
                means do not check this axis. Ignored when ``predicate`` is ``None``.
            expected_projection_pushed (bool, optional): Whether ``projection`` is expected to be pushed down. ``None`` (default)
                means do not check this axis. Ignored when ``projection`` is ``None``.
            expected_columns (Iterable[str], optional): If provided and the projection is pushed, asserts that ``last_with_columns``
                (as a set) equals this set.
            assert_expr_predicate (Callable[[pl.Expr, pl.Expr | None], None], optional):
                Custom assertion called as ``assert_expr_predicate(predicate, last_predicate)``.
                Useful for inspecting the structure of the pushed predicate.
        """
        if predicate is None and projection is None:
            raise ValueError("At least one of `predicate` or `projection` must be provided")

        if projection is not None:
            projection = _materialize_columns(projection)

        self.reset()
        lf: pl.LazyFrame = self.lazy_frame
        if predicate is not None:
            lf = lf.filter(predicate)
        if projection is not None:
            lf = lf.select(projection)
        source_result = lf.collect()

        if predicate is not None:
            if assert_expr_predicate is not None:
                assert_expr_predicate(predicate, self.last_predicate)
            if expected_predicate_pushed is True:
                assert self.last_predicate is not None, "Predicate was not pushed down as expected"
            elif expected_predicate_pushed is False:
                assert self.last_predicate is None, "Predicate was pushed down when it shouldn't have been"

        if projection is not None:
            if expected_projection_pushed is True:
                assert self.last_with_columns is not None, "Projection was not pushed down as expected"
                if expected_columns is not None:
                    assert set(self.last_with_columns) == set(expected_columns), (
                        f"Pushed projection {self.last_with_columns!r} does not match expected {list(expected_columns)!r}"
                    )
            elif expected_projection_pushed is False:
                assert self.last_with_columns is None, f"Projection was pushed down ({self.last_with_columns!r}) when it shouldn't have been"

        direct: pl.DataFrame = self.df.clone()
        if predicate is not None:
            direct = direct.filter(predicate)
        if projection is not None:
            direct = direct.select(projection)
        assert_frame_equal(source_result, direct, check_row_order=False)


@dataclass
class PredicateAnalyzer:
    """
    Utility class for analyzing the structure of pushed-down predicates.

    This class provides methods to find and extract information from filter
    predicates, useful for verifying that filter pushdown is working correctly.

    Args:
        predicate (pl.Expr): The predicate expression to analyze.

    Example:
        ::

            analyzer = PredicateAnalyzer(pushed_predicate)

            # Find temporal filters
            temporal = analyzer.find_temporal_filter("date")
            lower, upper = analyzer.extract_temporal_bounds(temporal)

            # Find discrete filters
            discrete = analyzer.find_discrete_filter("category")
            values = analyzer.extract_discrete_values(discrete)
    """

    predicate: pl.Expr
    _parsed: Any = field(default=None, init=False, repr=False)

    @property
    def parsed(self) -> Any:
        """Get the parsed expression tree."""
        if self._parsed is None:
            self._parsed = get_parsed_expr(self.predicate)
        return self._parsed

    def find_node_by_predicate(self, predicate_fn: Callable[[Any], bool]) -> Any | None:
        """
        Recursively find a node in the expression tree that matches the predicate function.

        Args:
            predicate_fn (Callable[[Any], bool]): A function that returns True for the node(s) you're looking for.

        Returns:
            Any | None: The first matching node, or None if not found.

        See Also:
            find_all_nodes_by_predicate : Returns all matching nodes instead of just the first.
        """
        results = self.find_all_nodes_by_predicate(predicate_fn)
        return results[0] if results else None

    def find_all_nodes_by_predicate(self, predicate_fn: Callable[[Any], bool]) -> list[Any]:
        """
        Recursively find all nodes in the expression tree that match the predicate function.

        Nodes are returned in depth-first traversal order (left subtree before right).

        Args:
            predicate_fn (Callable[[Any], bool]): A function that returns True for the node(s) you're looking for.

        Returns:
            list[Any]: All matching nodes in traversal order, or empty list if none found.

        Example:
            ::

                # Find all equality comparisons
                def is_eq(node):
                    return isinstance(node, BinaryExprNode) and node.op == OperatorType.EQ
                all_eq_nodes = analyzer.find_all_nodes_by_predicate(is_eq)
        """
        results: list[Any] = []
        self._find_all_nodes_recursive(self.parsed, predicate_fn, results)
        return results

    def _find_all_nodes_recursive(self, node: Any, predicate_fn: Callable[[Any], bool], results: list[Any]) -> None:
        """Recursively search for all matching nodes."""
        if predicate_fn(node):
            results.append(node)
        if isinstance(node, BinaryExprNode):
            self._find_all_nodes_recursive(node.left, predicate_fn, results)
            self._find_all_nodes_recursive(node.right, predicate_fn, results)
        if isinstance(node, FunctionNode):
            for inp in node.inputs:
                self._find_all_nodes_recursive(inp, predicate_fn, results)

    def _make_temporal_predicates(self, col_name: str) -> tuple[Callable[[Any], bool], Callable[[Any], bool]]:
        """Create predicate functions for matching temporal filters on a column."""

        def is_matching_binary(node: Any) -> bool:
            if not isinstance(node, BinaryExprNode):
                return False
            if node.op not in (OperatorType.GT_EQ, OperatorType.LT_EQ, OperatorType.GT, OperatorType.LT):
                return False
            # Check if left is a column or cast of column
            left = node.left
            while hasattr(left, "input"):  # Unwrap CastNode
                left = left.input
            return hasattr(left, "name") and left.name == col_name

        def is_matching_between(node: Any) -> bool:
            if not isinstance(node, FunctionNode):
                return False
            if node.function_type != BooleanFunctionType.IS_BETWEEN:
                return False
            # Check if first input is the column
            first_input = node.inputs[0]
            while hasattr(first_input, "input"):  # Unwrap CastNode
                first_input = first_input.input
            return hasattr(first_input, "name") and first_input.name == col_name

        return is_matching_binary, is_matching_between

    def find_temporal_filter(self, col_name: str) -> Any | None:
        """
        Find a temporal filter (GT_EQ, LT_EQ, GT, LT, IS_BETWEEN) on the given column.

        Returns only the first matching filter. Use `find_temporal_filters` to get all.

        Args:
            col_name (str): The column name to search for.

        Returns:
            Any | None: The first filter node found, or None if not found.

        See Also:
            find_temporal_filters : Returns all temporal filters on the column.
        """
        filters = self.find_temporal_filters(col_name)
        return filters[0] if filters else None

    def find_temporal_filters(self, col_name: str) -> list[Any]:
        """
        Find all temporal filters (GT_EQ, LT_EQ, GT, LT, IS_BETWEEN) on the given column.

        Filters are returned in the order they appear in the predicate tree
        (depth-first traversal).

        Args:
            col_name (str): The column name to search for.

        Returns:
            list[Any]: All filter nodes found, in traversal order. Empty list if none found.

        Example:
            ::

                # With lookback, you'll often see two filters: expanded and original
                filters = analyzer.find_temporal_filters("date")
                for f in filters:
                    lower, upper = analyzer.extract_temporal_bounds(f)
                    print(f"Filter: lower={lower}, upper={upper}")
        """
        is_matching_binary, is_matching_between = self._make_temporal_predicates(col_name)

        # Find all IS_BETWEEN filters
        between_filters = self.find_all_nodes_by_predicate(is_matching_between)

        # Find all binary comparison filters
        binary_filters = self.find_all_nodes_by_predicate(is_matching_binary)

        # Return IS_BETWEEN first, then binary comparisons
        return between_filters + binary_filters

    def _make_discrete_predicates(self, col_name: str) -> tuple[Callable[[Any], bool], Callable[[Any], bool]]:
        """Create predicate functions for matching discrete filters on a column."""

        def is_matching_eq(node: Any) -> bool:
            if not isinstance(node, BinaryExprNode):
                return False
            if node.op != OperatorType.EQ:
                return False
            return hasattr(node.left, "name") and node.left.name == col_name

        def is_matching_is_in(node: Any) -> bool:
            if not isinstance(node, FunctionNode):
                return False
            if node.function_type != BooleanFunctionType.IS_IN:
                return False
            return hasattr(node.inputs[0], "name") and node.inputs[0].name == col_name

        return is_matching_eq, is_matching_is_in

    def find_discrete_filter(self, col_name: str) -> Any | None:
        """
        Find a discrete filter (EQ or IS_IN) on the given column.

        Returns only the first matching filter. Use `find_discrete_filters` to get all.

        Args:
            col_name (str): The column name to search for.

        Returns:
            Any | None: The first filter node found, or None if not found.

        See Also:
            find_discrete_filters : Returns all discrete filters on the column.
        """
        filters = self.find_discrete_filters(col_name)
        return filters[0] if filters else None

    def find_discrete_filters(self, col_name: str) -> list[Any]:
        """
        Find all discrete filters (EQ or IS_IN) on the given column.

        Filters are returned in the order they appear in the predicate tree
        (depth-first traversal).

        Args:
            col_name (str): The column name to search for.

        Returns:
            list[Any]: All filter nodes found, in traversal order. Empty list if none found.

        Example:
            ::

                # With value mapping, you might see multiple EQ filters
                filters = analyzer.find_discrete_filters("region_code")
                for f in filters:
                    values = analyzer.extract_discrete_values(f)
                    print(f"Filter values: {values}")
        """
        is_matching_eq, is_matching_is_in = self._make_discrete_predicates(col_name)

        # Find all IS_IN filters
        is_in_filters = self.find_all_nodes_by_predicate(is_matching_is_in)

        # Find all EQ filters
        eq_filters = self.find_all_nodes_by_predicate(is_matching_eq)

        # Return IS_IN first, then EQ comparisons
        return is_in_filters + eq_filters

    def extract_temporal_bounds(self, node: Any) -> tuple[date | datetime | None, date | datetime | None]:
        """
        Extract the temporal bounds from a filter node.

        Args:
            node (Any): A temporal filter node (from find_temporal_filter).

        Returns:
            tuple[date | datetime | None, date | datetime | None]: (lower_bound, upper_bound) where either may be None for one-sided filters.
        """
        if node is None:
            return (None, None)

        if isinstance(node, FunctionNode) and node.function_type == BooleanFunctionType.IS_BETWEEN:
            lower = node.inputs[1].value
            upper = node.inputs[2].value
            # Convert datetime to date if needed
            if hasattr(lower, "date") and callable(lower.date):
                lower = lower.date()
            if hasattr(upper, "date") and callable(upper.date):
                upper = upper.date()
            return (lower, upper)

        if isinstance(node, BinaryExprNode):
            val = node.right.value
            # Convert datetime to date if needed
            if hasattr(val, "date") and callable(val.date):
                val = val.date()
            if node.op in (OperatorType.GT_EQ, OperatorType.GT):
                return (val, None)
            if node.op in (OperatorType.LT_EQ, OperatorType.LT):
                return (None, val)

        return (None, None)

    def extract_discrete_values(self, node: Any) -> set[Any] | None:
        """
        Extract the discrete values from a filter node.

        Args:
            node (Any): A discrete filter node (from find_discrete_filter).

        Returns:
            set[Any] | None: The set of values in the filter, or None if the node is invalid.
        """
        if node is None:
            return None

        if isinstance(node, BinaryExprNode) and node.op == OperatorType.EQ:
            return {node.right.value}

        if isinstance(node, FunctionNode) and node.function_type == BooleanFunctionType.IS_IN:
            return set(node.inputs[1].value)

        return None

    def has_filter_on_column(self, col_name: str) -> bool:
        """
        Check if there's any filter on the given column.

        Args:
            col_name (str): The column name to check.

        Returns:
            bool: True if there's a filter on this column.
        """
        return len(self.find_temporal_filters(col_name)) > 0 or len(self.find_discrete_filters(col_name)) > 0

    def count_filters_on_column(self, col_name: str) -> int:
        """
        Count the total number of filters on the given column.

        Args:
            col_name (str): The column name to check.

        Returns:
            int: Total number of temporal and discrete filters on this column.

        Example:
            ::

                # With lookback, expect 2 filters: expanded + original
                assert analyzer.count_filters_on_column("date") == 2
        """
        return len(self.find_temporal_filters(col_name)) + len(self.find_discrete_filters(col_name))


def io_source_assert(df: pl.DataFrame, assert_func: Callable[[Optional[pl.Expr]], None]) -> pl.LazyFrame:
    """
    Create a LazyFrame that runs assertions on predicates pushed down to the source.

    This is useful for inline testing of predicate pushdown behavior. The assertion
    function is called with the pushed predicate each time the source is collected.

    Args:
        df (pl.DataFrame): The DataFrame to use as the underlying data source.
        assert_func (Callable[[pl.Expr | None], None]): A function that receives the pushed predicate and should raise AssertionError
            if the predicate doesn't meet expectations.

    Returns:
        pl.LazyFrame: A LazyFrame that will call assert_func with the pushed predicate on collection.

    Example:
        ::

            # Assert that a predicate is pushed down
            lf = io_source_assert(df, lambda pred: assert pred is not None)
            lf.filter(pl.col("date") >= date(2024, 1, 5)).collect()

            # Assert specific predicate structure
            def check_predicate(pred):
                assert pred is not None
                analyzer = PredicateAnalyzer(pred)
                lower, _ = analyzer.extract_temporal_bounds(
                    analyzer.find_temporal_filter("date")
                )
                assert lower == date(2024, 1, 2)

            lf = io_source_assert(df, check_predicate)
            lf.filter(pl.col("date") >= date(2024, 1, 5)).collect()
    """

    def source_generator(
        with_columns: Optional[list[str]],
        predicate: Optional[pl.Expr],
        n_rows: Optional[int],
        batch_size: Optional[int],
    ) -> Iterator[pl.DataFrame]:
        # Run the assertion on the pushed predicate
        assert_func(predicate)

        result = df.clone()

        if predicate is not None:
            result = result.filter(predicate)
        if with_columns is not None:
            result = result.select(with_columns)
        if n_rows is not None:
            result = result.head(n_rows)

        yield result

    return register_io_source(io_source=source_generator, schema=df.schema, is_pure=True)
