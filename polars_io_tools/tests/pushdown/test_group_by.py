"""Filters and column selections after ``group_by().agg()``."""

import polars as pl
import pytest

from polars_io_tools.tests.helpers.assertions import ExpectedPushdown
from polars_io_tools.tests.helpers.cases import as_params
from polars_io_tools.tests.pushdown.helpers import PushdownCase, check_pushdown

GROUPED = {
    "table": pl.DataFrame(
        {"group": ["a", "a", "b", "b"], "subgroup": [1, 1, 2, 2], "key": [1, 10, 2, 20], "value": [10, 20, 30, 40], "weight": [1, 2, 3, 4]}
    )
}

ROWS = [
    PushdownCase(
        "group_by-filter_on_group_key_pushed",
        sources=GROUPED,
        pipeline=lambda frames: frames["table"].group_by("group").agg(pl.col("value").sum()).filter(pl.col("group") == "a"),
        expected_pushdown={"table": ExpectedPushdown(predicate=True)},
    ),
    PushdownCase(
        "group_by-filter_on_secondary_group_key_pushed",
        sources=GROUPED,
        pipeline=lambda frames: frames["table"].group_by("group", "subgroup").agg(pl.col("value").sum()).filter(pl.col("subgroup") == 1),
        expected_pushdown={"table": ExpectedPushdown(predicate=True)},
    ),
    PushdownCase(
        "group_by-select_subset_of_aggs_prunes_unused_source_cols",
        sources=GROUPED,
        pipeline=lambda frames: (
            frames["table"]
            .group_by("group")
            .agg(pl.col("value").sum().alias("value_sum"), pl.col("weight").sum().alias("weight_sum"))
            .select("group", "value_sum")
        ),
        expected_pushdown={"table": ExpectedPushdown(projection={"group", "value"})},
    ),
    # polars#22623: grouping by a literal still prunes the source columns the aggregations do not use.
    PushdownCase(
        "group_by-group_by_literal_key_pushes_projection",
        sources=GROUPED,
        pipeline=lambda frames: frames["table"].group_by(pl.lit(1)).agg(pl.col("value").sum()),
        expected_pushdown={"table": ExpectedPushdown(projection={"value"})},
    ),
    # Rows the filter drops add nothing to the sum, so the filter could reach the source (polars#27207).
    PushdownCase(
        "group_by-expr_filter_inside_agg_NOT_pushed",
        sources=GROUPED,
        pipeline=lambda frames: frames["table"].group_by("group").agg(pl.col("value").filter(pl.col("key") > 5).sum()),
        expected_pushdown={"table": ExpectedPushdown(predicate=False)},
        marks=(pytest.mark.gap,),
    ),
]


@pytest.mark.parametrize("case", as_params(ROWS))
def test_pushdown(case: PushdownCase):
    check_pushdown(case)
