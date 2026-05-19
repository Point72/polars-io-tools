"""
Testing utilities for polars-io-tools.

This module provides utilities for testing filter pushdown behavior in IO sources.
It can be used by other libraries that build on polars-io-tools to verify their
filter pushdown implementations.

Example usage::

    from polars_io_tools.testing import PredicateTracker, PredicateAnalyzer

    # Create a tracking source
    df = pl.DataFrame({"date": dates, "val": values})
    tracker = PredicateTracker(df)

    # Use the LazyFrame in your multi_source or IO source
    lf = multi_source(
        sources={"data": (tracker.lazy_frame, {"date": FilterSpec()})},
        combine=lambda s: s["data"],
    )

    # Collect with a filter
    result = lf.filter(pl.col("date") >= date(2024, 1, 5)).collect()

    # Analyze what was pushed down
    analyzer = tracker.get_analyzer()
    temporal_filter = analyzer.find_temporal_filter("date")
    lower, upper = analyzer.extract_temporal_bounds(temporal_filter)

    # Or use the assertion methods for testing
    tracker.assert_predicate_pushed_down(pl.col("date") >= date(2024, 1, 5))
    tracker.assert_results_match(pl.col("date") >= date(2024, 1, 5))

    # Projection pushdown:
    tracker.assert_projection_pushed_down(["date"], expected_columns={"date"})

    # Combined predicate + projection:
    tracker.assert_pushed_down(
        predicate=pl.col("date") >= date(2024, 1, 5),
        projection=["date", "val"],
        expected_columns={"date", "val"},
    )
"""

from .predicate_tracker import (
    PredicateAnalyzer,
    PredicateTracker,
    io_source_assert,
)

__all__ = ("PredicateTracker", "PredicateAnalyzer", "io_source_assert")
