"""Filters after a window expression (``over``).

Dropping other partitions leaves a partition's window values unchanged, so a filter on the ``over`` key reaches the
source. A filter on another column would change those values, so it must not be handed on.
"""

import polars as pl
import pytest

from polars_io_tools.tests.helpers.assertions import ExpectedPushdown
from polars_io_tools.tests.helpers.cases import as_params
from polars_io_tools.tests.pushdown.helpers import PushdownCase, check_pushdown

WINDOWED = {"table": pl.DataFrame({"key": [1, 1, 2, 2, 3], "value": [10, None, 30, None, 50]})}

ROWS = [
    PushdownCase(
        "window-filter_on_partition_col_of_shift_over_pushed",
        sources=WINDOWED,
        pipeline=lambda frames: frames["table"].with_columns(pl.col("value").shift(1).over("key").alias("previous_value")).filter(pl.col("key") == 1),
        expected_pushdown={"table": ExpectedPushdown(predicate=True)},
    ),
    PushdownCase(
        "window-filter_on_partition_col_of_forward_fill_over_pushed",
        sources=WINDOWED,
        pipeline=lambda frames: (
            frames["table"].with_columns(pl.col("value").forward_fill().over("key").alias("filled_value")).filter(pl.col("key") == 1)
        ),
        expected_pushdown={"table": ExpectedPushdown(predicate=True)},
    ),
]


@pytest.mark.parametrize("case", as_params(ROWS))
def test_pushdown(case: PushdownCase):
    check_pushdown(case)
