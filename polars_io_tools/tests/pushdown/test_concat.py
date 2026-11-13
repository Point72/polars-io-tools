"""Filters after ``pl.concat``, vertical and horizontal."""

import polars as pl
import pytest

from polars_io_tools.tests.helpers.assertions import ExpectedPushdown
from polars_io_tools.tests.helpers.cases import as_params
from polars_io_tools.tests.pushdown.helpers import PushdownCase, check_pushdown

KEYED_VALUES = pl.DataFrame({"key": [1, 2, 3], "value": [10, 20, 30]})
CONCAT = {"left": KEYED_VALUES, "right": pl.DataFrame({"key": [1, 2, 3], "other_value": [100, 200, 300]})}
LABELLED = {"first": pl.DataFrame({"key": [1, 2, 3], "value": [10, 20, 30]}), "second": pl.DataFrame({"key": [4, 5], "value": [40, 50]})}

ROWS = [
    PushdownCase(
        "concat-vertical_concat_filter_on_shared_col_pushed",
        sources=CONCAT,
        pipeline=lambda frames: pl.concat(
            [frames["left"].select("key", "value"), frames["right"].select("key", pl.col("other_value").alias("value"))]
        ).filter(pl.col("key") == 2),
        expected_pushdown={"left": ExpectedPushdown(predicate=True), "right": ExpectedPushdown(predicate=True)},
    ),
    # Horizontal concat pairs rows by position, so a filter could only reach the sources if it kept the same rows of
    # every input. Polars hands nothing to the sources (polars#26552).
    PushdownCase(
        "concat-horizontal_concat_filter_on_left_col_NOT_pushed",
        sources=CONCAT,
        pipeline=lambda frames: pl.concat([frames["left"], frames["right"].select("other_value")], how="horizontal").filter(pl.col("key") > 1),
        expected_pushdown={"left": ExpectedPushdown(predicate=False)},
        marks=(pytest.mark.gap,),
    ),
    # The filter keeps only the rows labelled "real", so the second source need not be read; both are (polars#23345).
    PushdownCase(
        "concat-vertical_concat_literal_side_NOT_pruned",
        sources=LABELLED,
        pipeline=lambda frames: pl.concat(
            [frames["first"].with_columns(pl.lit("real").alias("kind")), frames["second"].with_columns(pl.lit("synthetic").alias("kind"))]
        ).filter(pl.col("kind") == "real"),
        expected_pushdown={"first": ExpectedPushdown(calls=1), "second": ExpectedPushdown(calls=1)},
        marks=(pytest.mark.gap,),
    ),
]


@pytest.mark.parametrize("case", as_params(ROWS))
def test_pushdown(case: PushdownCase):
    check_pushdown(case)
