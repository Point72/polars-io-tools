"""Filters and column selections after an equi-join of two sources."""

import polars as pl
import pytest

from polars_io_tools.tests.helpers.assertions import ExpectedPushdown
from polars_io_tools.tests.helpers.cases import as_params
from polars_io_tools.tests.pushdown.helpers import PushdownCase, check_pushdown

KEYED_VALUES = pl.DataFrame({"key": [1, 2, 3], "value": [10, 20, 30]})
JOINED = {"left": KEYED_VALUES, "right": pl.DataFrame({"key": [1, 2, 3], "other_value": [100, 200, 300], "unused": [1, 2, 3]})}

ROWS = [
    PushdownCase(
        "join-inner_join_filter_on_key_pushed",
        sources=JOINED,
        pipeline=lambda frames: frames["left"].join(frames["right"], on="key").filter(pl.col("key") == 2),
        expected_pushdown={"left": ExpectedPushdown(predicate=True), "right": ExpectedPushdown(predicate=True)},
    ),
    PushdownCase(
        "join-right_join_filter_on_key_pushed",
        sources=JOINED,
        pipeline=lambda frames: frames["left"].join(frames["right"], on="key", how="right").filter(pl.col("key") == 2),
        expected_pushdown={"left": ExpectedPushdown(predicate=True), "right": ExpectedPushdown(predicate=True)},
    ),
    PushdownCase(
        "join-inner_join_projection_prunes_unused_right_cols",
        sources=JOINED,
        pipeline=lambda frames: frames["left"].join(frames["right"], on="key").select("key", "value", "other_value"),
        expected_pushdown={"right": ExpectedPushdown(projection={"key", "other_value"})},
    ),
]


@pytest.mark.parametrize("case", as_params(ROWS))
def test_pushdown(case: PushdownCase):
    check_pushdown(case)
