"""LazyFrame.piot.iter_rows: every row of the collected frame, in order, as tuples or dicts, whatever the buffer size."""

import datetime

import polars as pl
import pytest

from polars_io_tools._compat import POLARS_HAS_COLLECT_BATCHES
from polars_io_tools.tests.helpers.catalogue import Case, params


def _ids(n, **more):
    return pl.LazyFrame({"id": range(n), **more})


def R(id, lf, marks=(), **kw):
    return Case(id, lf, kw=kw, marks=marks)


_AB = pl.LazyFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
_DATES = [datetime.date(2024, 1, d) for d in (1, 2, 3)]

ROWS = [
    R("iter_rows_returns_all_rows", pl.LazyFrame({"a": range(100), "b": range(100, 200)}), named=False),
    R("iter_rows_named_true_returns_dicts", _AB, named=True),
    R("iter_rows_named_false_returns_tuples", _AB, named=False),
    R("iter_rows_default_named_is_false", pl.LazyFrame({"a": [1, 2], "b": [10, 20]})),
    R("iter_rows_with_small_buffer", _ids(10), buffer_size=3, named=True),
    R("iter_rows_buffer_larger_than_total", _ids(10, value=range(100, 110)), buffer_size=1000, named=True),
    R("iter_rows_buffer_equals_total", _ids(50), buffer_size=50, named=True),
    # batches of 1000, 1000 and 125
    R("iter_rows_not_evenly_divisible", _ids(2125), buffer_size=1000, named=True),
    *[
        R(f"iter_rows_various_buffer_sizes-{b}", _ids(1000, value=range(1000, 2000)), buffer_size=b, named=True)
        for b in [1, 10, 100, 333, 500, 999, 1000, 2000]
    ],
    R("iter_rows_empty_lazyframe", pl.LazyFrame(schema={"a": pl.Int64, "b": pl.Int64}), named=True),
    R("iter_rows_single_row", pl.LazyFrame({"a": [42], "b": [100]}), named=True),
    R("iter_rows_single_column", pl.LazyFrame({"value": [1, 2, 3, 4, 5]}), named=True),
    R("iter_rows_many_columns", pl.LazyFrame({f"col_{i}": range(i * 100, i * 100 + 100) for i in range(20)}), buffer_size=25, named=True),
    R("iter_rows_no_missing_rows", _ids(2500), buffer_size=512, named=True),
    R("iter_rows_no_duplicates", _ids(1000), buffer_size=333, named=True),
    R("iter_rows_sequential_ids", _ids(5000), buffer_size=777, named=True),
    R("iter_rows_batch_boundaries", _ids(1005), buffer_size=100, named=True),
    R("iter_rows_maintain_order_true", _ids(1000, value=range(1000, 2000)), buffer_size=250, maintain_order=True, named=True),
    R(
        "iter_rows_maintain_order_false",
        _ids(500),
        marks=pytest.mark.skipif(not POLARS_HAS_COLLECT_BATCHES, reason="maintain_order requires Polars >= 1.34.0"),
        buffer_size=100,
        maintain_order=False,
        named=True,
    ),
    R("iter_rows_with_filter", _ids(100, value=range(100)).filter(pl.col("value") % 2 == 0), buffer_size=20, named=True),
    R(
        "iter_rows_with_select",
        pl.LazyFrame({"a": range(50), "b": range(50, 100), "c": range(100, 150)}).select(["a", "c"]),
        buffer_size=15,
        named=True,
    ),
    R("iter_rows_with_sort", pl.LazyFrame({"id": [5, 2, 8, 1, 9, 3]}).sort("id"), buffer_size=2, named=True),
    R("iter_rows_with_with_columns", pl.LazyFrame({"a": range(20)}).with_columns((pl.col("a") * 2).alias("doubled")), buffer_size=7, named=True),
    R("iter_rows_with_dates", pl.LazyFrame({"date": _DATES, "value": [1, 2, 3]}), named=True),
    R("iter_rows_with_strings", pl.LazyFrame({"name": ["Alice", "Bob", "Charlie"], "age": [30, 25, 35]}), named=True),
    R("iter_rows_with_nulls", pl.LazyFrame({"a": [1, None, 3], "b": [None, 2, 3]}), named=True),
    R(
        "iter_rows_with_mixed_types",
        pl.LazyFrame(
            {"int_col": [1, 2, 3], "float_col": [1.1, 2.2, 3.3], "str_col": ["a", "b", "c"], "bool_col": [True, False, True], "date_col": _DATES}
        ),
        buffer_size=2,
        named=True,
    ),
    R("iter_rows_matches_collect", pl.LazyFrame({"a": range(1000), "b": range(1000, 2000)}), buffer_size=250, named=True),
    R("iter_rows_large_dataset", _ids(50_000, value=range(50_000, 100_000)), buffer_size=5000, named=True),
]


@pytest.mark.parametrize(("lf", "kw"), params(ROWS, "pred", "kw"))
def test_iter_rows(lf, kw):
    got = list(lf.piot.iter_rows(**kw))
    want = lf.collect().rows(named=kw.get("named", False))
    if kw.get("maintain_order", True):
        assert got == want
    else:
        assert sorted(got, key=lambda r: r["id"]) == want
