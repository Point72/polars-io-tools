"""A filter after a cast.

Casting to a wider type keeps every value, so a filter after the cast could run on the source (polars#23369). Polars
older than 2.1 keeps the filter after the cast. A cast to a narrower type can change values, so it is not tested.
"""

import polars as pl
import pytest

from polars_io_tools.tests.helpers.assertions import ExpectedPushdown
from polars_io_tools.tests.helpers.cases import as_params
from polars_io_tools.tests.helpers.polars_versions import polars_has
from polars_io_tools.tests.pushdown.helpers import PushdownCase, check_pushdown

INT32_VALUES = {"table": pl.DataFrame({"value": pl.Series([1, 2, 3, 4, 5], dtype=pl.Int32)})}

ROWS = [
    PushdownCase(
        "cast-widening_cast_then_filter_pushdown",
        sources=INT32_VALUES,
        pipeline=lambda frames: frames["table"].with_columns(pl.col("value").cast(pl.Int64)).filter(pl.col("value") > 2),
        expected_pushdown={"table": ExpectedPushdown(predicate=polars_has("pushes_widening_cast"))},
        result_check=lambda result, _: result["value"].to_list() == [3, 4, 5],
        marks=(pytest.mark.gap,),
    ),
]


@pytest.mark.parametrize("case", as_params(ROWS))
def test_pushdown(case: PushdownCase):
    check_pushdown(case)
