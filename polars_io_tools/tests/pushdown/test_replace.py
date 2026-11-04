"""Filters after ``replace_strict``, on the column it makes and on the column it reads.

``replace_strict`` maps every source value through a literal dict, so a filter on its output could become an ``is_in``
on the source (polars#23345). Non-strict ``replace`` passes unmapped values through and has no such simple inverse, so
it is not tested.
"""

import polars as pl
import pytest

from polars_io_tools.tests.helpers.assertions import ExpectedPushdown
from polars_io_tools.tests.helpers.cases import as_params
from polars_io_tools.tests.pushdown.helpers import PushdownCase, check_pushdown

MODELS = {"table": pl.DataFrame({"model": ["A", "B", "C", "D"], "value": [1, 2, 3, 4]})}
FAMILY_OF_MODEL = {"A": "X", "B": "X", "C": "Y", "D": "Y"}

ROWS = [
    PushdownCase(
        "replace-replace_strict_filter_on_derived_col_NOT_pushed",
        sources=MODELS,
        pipeline=lambda frames: (
            frames["table"].with_columns(pl.col("model").replace_strict(FAMILY_OF_MODEL).alias("family")).filter(pl.col("family") == "X")
        ),
        expected_pushdown={"table": ExpectedPushdown(predicate=False)},
        marks=(pytest.mark.gap,),
    ),
    PushdownCase(
        "replace-filter_on_source_col_after_replace_strict_pushed",
        sources=MODELS,
        pipeline=lambda frames: (
            frames["table"].with_columns(pl.col("model").replace_strict(FAMILY_OF_MODEL).alias("family")).filter(pl.col("model") == "B")
        ),
        expected_pushdown={"table": ExpectedPushdown(predicate=True)},
    ),
]


@pytest.mark.parametrize("case", as_params(ROWS))
def test_pushdown(case: PushdownCase):
    check_pushdown(case)
