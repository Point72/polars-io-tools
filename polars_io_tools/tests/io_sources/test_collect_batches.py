import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_io_tools as cpl
from polars_io_tools._compat import POLARS_HAS_COLLECT_BATCHES


@pytest.mark.skipif(not POLARS_HAS_COLLECT_BATCHES, reason="collect_batches requires Polars >= 1.34.0")
def test_collect_batches_concat_named_yields_equal():
    lf1 = pl.LazyFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    lf2 = pl.LazyFrame({"a": [4, 5, 6], "b": [40, 50, 60]})

    lf = cpl.concat_named({("foo",): lf1, ("bar",): lf2}, ["source"]).select(["a", "b", "source"]).sort(["source", "a"])  # stable compare

    chunks = list(lf.collect_batches(chunk_size=2))
    combined = pl.concat(chunks, how="vertical").sort(["source", "a"])  # ensure identical ordering
    expected = lf.collect()

    assert_frame_equal(combined, expected)


@pytest.mark.skipif(not POLARS_HAS_COLLECT_BATCHES, reason="collect_batches requires Polars >= 1.34.0")
def test_collect_batches_ts_with_columns_yields_equal():
    dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2), datetime.date(2024, 1, 3), datetime.date(2024, 1, 4)]
    vals = [1, 2, 3, 4]
    base = pl.LazyFrame({"Date": dates, "val": vals})

    lf = base.piot.ts_with_columns([pl.col("val") * 2], index_col="Date")
    chunks = list(lf.collect_batches(chunk_size=2))
    combined = pl.concat(chunks, how="vertical").sort("Date")
    expected = lf.collect().sort("Date")

    assert_frame_equal(combined, expected)


@pytest.mark.skipif(not POLARS_HAS_COLLECT_BATCHES, reason="collect_batches requires Polars >= 1.34.0")
def test_collect_batches_filtered_join_yields_equal():
    left = pl.LazyFrame({"id": [1, 2, 3, 4], "x": [10, 20, 30, 40]})
    right = pl.LazyFrame({"id": [1, 2, 3, 4], "y": [100, 200, 300, 400]})

    lf = left.piot.filtered_join(right, on="id", maintain_order="left")
    chunks = list(lf.collect_batches(chunk_size=2))
    combined = pl.concat(chunks, how="vertical").sort("id")
    expected = lf.collect().sort("id")

    assert_frame_equal(combined, expected)


@pytest.mark.skipif(not POLARS_HAS_COLLECT_BATCHES, reason="collect_batches requires Polars >= 1.34.0")
def test_collect_batches_lazy_debug_yields_equal():
    base = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
    lf = base.piot.debug()
    chunks = list(lf.collect_batches(chunk_size=2))
    combined = pl.concat(chunks, how="vertical").sort("a")
    expected = lf.collect().sort("a")

    assert_frame_equal(combined, expected)
