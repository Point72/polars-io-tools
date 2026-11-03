"""A filter on a column added by a join with a small lookup table.

The filter could become an ``is_in`` on the source's join key (polars#21710, polars#23345); Polars does not do this.
The same filter written by hand as an ``is_in`` reaches the source.
"""

import polars as pl
import pytest

from polars_io_tools.tests.helpers.assertions import ExpectedPushdown
from polars_io_tools.tests.helpers.cases import as_params
from polars_io_tools.tests.pushdown.helpers import PushdownCase, check_pushdown

MODELS = {"table": pl.DataFrame({"model": ["A", "B", "C", "D"], "value": [1, 2, 3, 4]})}
MODEL_FAMILIES = pl.DataFrame({"model": ["A", "B", "C", "D"], "family": ["X", "X", "Y", "Y"]}).lazy()

ROWS = [
    PushdownCase(
        "join_lookup-lookup_join_filter_on_derived_col_NOT_pushed_as_isin",
        sources=MODELS,
        pipeline=lambda frames: frames["table"].join(MODEL_FAMILIES, on="model").filter(pl.col("family") == "X"),
        expected_pushdown={"table": ExpectedPushdown(predicate=False)},
        marks=(pytest.mark.gap,),
    ),
    PushdownCase(
        "join_lookup-manual_isin_filter_pushed",
        sources=MODELS,
        pipeline=lambda frames: frames["table"].filter(pl.col("model").is_in(["A", "B"])),
        expected_pushdown={"table": ExpectedPushdown(predicate=True)},
    ),
]


@pytest.mark.parametrize("case", as_params(ROWS))
def test_pushdown(case: PushdownCase):
    check_pushdown(case)
