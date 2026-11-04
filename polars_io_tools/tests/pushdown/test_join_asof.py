"""Filters and expressions after ``join_asof``.

A ``by`` value either matches on both sides or is null after the join, so a filter on it reaches both sources. A
filter on the ``on`` column reaches the left source only: removing right rows by it could change which right row a
kept left row matches. A filter on a right-only column reaches neither source, for the same reason. An expression on
a right-only column computed after the join stays after the join, although the right source could compute it. These
hold on every Polars version the tests run on.
"""

import polars as pl
import pytest

from polars_io_tools.tests.helpers.assertions import ExpectedPushdown
from polars_io_tools.tests.helpers.cases import as_params
from polars_io_tools.tests.pushdown.helpers import PushdownCase, check_pushdown

ASOF = {
    "left": pl.DataFrame({"time": [1, 2, 3, 4], "group": ["a", "a", "b", "b"], "left_value": [10, 20, 30, 40]}),
    "right": pl.DataFrame({"time": [1, 2, 3, 4], "group": ["a", "a", "b", "b"], "right_value": [100, 200, 300, 400]}),
}


def _join_asof_by_group(frames):
    return frames["left"].join_asof(frames["right"], on="time", by="group")


# With ``by`` groups Polars cannot check that ``on`` is sorted and warns; the data here is sorted, so the warning is noise.
ignore_unsorted_columns_warning = pytest.mark.filterwarnings("ignore:Sortedness of columns cannot be checked")

ROWS = [
    PushdownCase(
        "join_asof-filter_on_by_column_pushed",
        sources=ASOF,
        pipeline=lambda frames: _join_asof_by_group(frames).filter(pl.col("group") == "a"),
        expected_pushdown={"left": ExpectedPushdown(predicate=True), "right": ExpectedPushdown(predicate=True)},
        marks=(ignore_unsorted_columns_warning,),
    ),
    PushdownCase(
        "join_asof-filter_on_on_column_reaches_left_only",
        sources=ASOF,
        pipeline=lambda frames: _join_asof_by_group(frames).filter(pl.col("time") >= 3),
        expected_pushdown={"left": ExpectedPushdown(predicate=True), "right": ExpectedPushdown(predicate=False)},
        marks=(ignore_unsorted_columns_warning,),
    ),
    PushdownCase(
        "join_asof-filter_on_right_only_column_not_pushed",
        sources=ASOF,
        pipeline=lambda frames: _join_asof_by_group(frames).filter(pl.col("right_value") > 150),
        expected_pushdown={"left": ExpectedPushdown(predicate=False), "right": ExpectedPushdown(predicate=False)},
        marks=(ignore_unsorted_columns_warning,),
    ),
    # ``right_value + 1`` works row by row and keeps nulls, so the right source could compute it (polars#25867).
    PushdownCase(
        "join_asof-post_join_elementwise_expr_on_right_col_NOT_moved_into_right_scan",
        sources=ASOF,
        pipeline=lambda frames: _join_asof_by_group(frames).with_columns((pl.col("right_value") + 1).alias("right_value_plus_one")),
        result_check=lambda _, trackers: "right_value_plus_one" not in (trackers["right"].last_with_columns or []),
        marks=(pytest.mark.gap, ignore_unsorted_columns_warning),
    ),
]


@pytest.mark.parametrize("case", as_params(ROWS))
def test_pushdown(case: PushdownCase):
    check_pushdown(case)
