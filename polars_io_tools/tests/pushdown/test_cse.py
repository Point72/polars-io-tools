"""Common subplan elimination: one source read by two branches of a query.

Two filters of one frame scan it twice; one scan with ``first | second`` would do (polars#26502).
"""

import polars as pl
import pytest

from polars_io_tools.tests.helpers.assertions import ExpectedPushdown
from polars_io_tools.tests.helpers.cases import as_params
from polars_io_tools.tests.pushdown.helpers import PushdownCase, check_pushdown

FIVE_KEYS = {"table": pl.DataFrame({"key": [1, 2, 3, 4, 5], "value": [10, 20, 30, 40, 50]})}

ROWS = [
    PushdownCase(
        "cse-concat_of_two_filters_scans_source_twice",
        sources=FIVE_KEYS,
        pipeline=lambda frames: pl.concat([frames["table"].filter(pl.col("key") < 3), frames["table"].filter(pl.col("key") > 3)]),
        expected_pushdown={"table": ExpectedPushdown(calls=2)},
        marks=(pytest.mark.gap,),
    ),
]


@pytest.mark.parametrize("case", as_params(ROWS))
def test_pushdown(case: PushdownCase):
    check_pushdown(case)
