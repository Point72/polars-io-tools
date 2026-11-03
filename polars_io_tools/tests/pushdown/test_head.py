"""``head`` and ``limit``: Polars hands the row count to a source only when no filter comes before it."""

import polars as pl
import pytest
from polars.io.plugins import register_io_source


@pytest.mark.parametrize("limit_method", ["head", "limit"])
@pytest.mark.parametrize("with_filter", [True, False], ids=["limit_not_pushed_when_filter_present", "limit_pushed_when_no_filter"])
def test_limit_pushdown(with_filter, limit_method):
    """Polars passes ``n_rows`` to a source only without a filter; with one, a source that cannot apply the whole
    filter would otherwise return too few rows."""
    received = {}

    def tracking_source(with_columns, predicate, n_rows, batch_size):
        received.update(predicate=predicate, n_rows=n_rows)
        yield pl.DataFrame({"x": [1, 2, 3, 4, 5]})

    lf = register_io_source(tracking_source, schema={"x": pl.Int64})
    if with_filter:
        lf = lf.filter(pl.col("x") > 2)
    getattr(lf, limit_method)(3).collect()
    assert (received["predicate"] is not None) is with_filter
    assert received["n_rows"] == (None if with_filter else 3)
