"""LazyFrame.piot.iter_rows: every row of the collected frame, in order, as tuples or dicts, whatever the buffer size."""

import datetime
from dataclasses import KW_ONLY, dataclass, field

import polars as pl
import pytest

from polars_io_tools._compat import POLARS_HAS_COLLECT_BATCHES
from polars_io_tools.tests.helpers.cases import as_params


def _numbered_rows(n, **more_columns):
    return pl.LazyFrame({"id": range(n), **more_columns})


@dataclass(frozen=True)
class IterRowsCase:
    """A frame and options for ``iter_rows``, which must give the rows of the collected frame."""

    id: str
    _: KW_ONLY
    lazy_frame: pl.LazyFrame
    # keyword arguments for ``iter_rows``
    iter_rows_options: dict = field(default_factory=dict)
    marks: tuple = ()


_THREE_ROWS = pl.LazyFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
_JAN_1_2_3_2024 = [datetime.date(2024, 1, d) for d in (1, 2, 3)]

# Frames of different shapes and buffer sizes.
ROWS = [
    IterRowsCase("iter_rows_returns_all_rows", lazy_frame=pl.LazyFrame({"a": range(100), "b": range(100, 200)}), iter_rows_options={"named": False}),
    IterRowsCase("iter_rows_named_true_returns_dicts", lazy_frame=_THREE_ROWS, iter_rows_options={"named": True}),
    IterRowsCase("iter_rows_named_false_returns_tuples", lazy_frame=_THREE_ROWS, iter_rows_options={"named": False}),
    IterRowsCase("iter_rows_default_named_is_false", lazy_frame=pl.LazyFrame({"a": [1, 2], "b": [10, 20]})),
    IterRowsCase("iter_rows_with_small_buffer", lazy_frame=_numbered_rows(10), iter_rows_options={"buffer_size": 3, "named": True}),
    IterRowsCase(
        "iter_rows_buffer_larger_than_total",
        lazy_frame=_numbered_rows(10, value=range(100, 110)),
        iter_rows_options={"buffer_size": 1000, "named": True},
    ),
    IterRowsCase("iter_rows_buffer_equals_total", lazy_frame=_numbered_rows(50), iter_rows_options={"buffer_size": 50, "named": True}),
    # batches of 1000, 1000 and 125
    IterRowsCase("iter_rows_not_evenly_divisible", lazy_frame=_numbered_rows(2125), iter_rows_options={"buffer_size": 1000, "named": True}),
    *[
        IterRowsCase(
            f"iter_rows_various_buffer_sizes-{b}",
            lazy_frame=_numbered_rows(1000, value=range(1000, 2000)),
            iter_rows_options={"buffer_size": b, "named": True},
        )
        for b in [1, 10, 100, 333, 500, 999, 1000, 2000]
    ],
    IterRowsCase("iter_rows_empty_lazyframe", lazy_frame=pl.LazyFrame(schema={"a": pl.Int64, "b": pl.Int64}), iter_rows_options={"named": True}),
    IterRowsCase("iter_rows_single_row", lazy_frame=pl.LazyFrame({"a": [42], "b": [100]}), iter_rows_options={"named": True}),
    IterRowsCase("iter_rows_single_column", lazy_frame=pl.LazyFrame({"value": [1, 2, 3, 4, 5]}), iter_rows_options={"named": True}),
    IterRowsCase(
        "iter_rows_many_columns",
        lazy_frame=pl.LazyFrame({f"col_{i}": range(i * 100, i * 100 + 100) for i in range(20)}),
        iter_rows_options={"buffer_size": 25, "named": True},
    ),
    IterRowsCase("iter_rows_no_missing_rows", lazy_frame=_numbered_rows(2500), iter_rows_options={"buffer_size": 512, "named": True}),
    IterRowsCase("iter_rows_no_duplicates", lazy_frame=_numbered_rows(1000), iter_rows_options={"buffer_size": 333, "named": True}),
    IterRowsCase("iter_rows_sequential_ids", lazy_frame=_numbered_rows(5000), iter_rows_options={"buffer_size": 777, "named": True}),
    IterRowsCase("iter_rows_batch_boundaries", lazy_frame=_numbered_rows(1005), iter_rows_options={"buffer_size": 100, "named": True}),
    IterRowsCase(
        "iter_rows_maintain_order_true",
        lazy_frame=_numbered_rows(1000, value=range(1000, 2000)),
        iter_rows_options={"buffer_size": 250, "maintain_order": True, "named": True},
    ),
    IterRowsCase(
        "iter_rows_maintain_order_false",
        lazy_frame=_numbered_rows(500),
        marks=(pytest.mark.skipif(not POLARS_HAS_COLLECT_BATCHES, reason="maintain_order requires Polars >= 1.34.0"),),
        iter_rows_options={"buffer_size": 100, "maintain_order": False, "named": True},
    ),
    IterRowsCase(
        "iter_rows_with_filter",
        lazy_frame=_numbered_rows(100, value=range(100)).filter(pl.col("value") % 2 == 0),
        iter_rows_options={"buffer_size": 20, "named": True},
    ),
    IterRowsCase(
        "iter_rows_with_select",
        lazy_frame=pl.LazyFrame({"a": range(50), "b": range(50, 100), "c": range(100, 150)}).select(["a", "c"]),
        iter_rows_options={"buffer_size": 15, "named": True},
    ),
    IterRowsCase(
        "iter_rows_with_sort", lazy_frame=pl.LazyFrame({"id": [5, 2, 8, 1, 9, 3]}).sort("id"), iter_rows_options={"buffer_size": 2, "named": True}
    ),
    IterRowsCase(
        "iter_rows_with_with_columns",
        lazy_frame=pl.LazyFrame({"a": range(20)}).with_columns((pl.col("a") * 2).alias("doubled")),
        iter_rows_options={"buffer_size": 7, "named": True},
    ),
    IterRowsCase("iter_rows_with_dates", lazy_frame=pl.LazyFrame({"date": _JAN_1_2_3_2024, "value": [1, 2, 3]}), iter_rows_options={"named": True}),
    IterRowsCase(
        "iter_rows_with_strings",
        lazy_frame=pl.LazyFrame({"name": ["Alice", "Bob", "Charlie"], "age": [30, 25, 35]}),
        iter_rows_options={"named": True},
    ),
    IterRowsCase("iter_rows_with_nulls", lazy_frame=pl.LazyFrame({"a": [1, None, 3], "b": [None, 2, 3]}), iter_rows_options={"named": True}),
    IterRowsCase(
        "iter_rows_with_mixed_types",
        lazy_frame=pl.LazyFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
                "date_col": _JAN_1_2_3_2024,
            }
        ),
        iter_rows_options={"buffer_size": 2, "named": True},
    ),
    IterRowsCase(
        "iter_rows_matches_collect",
        lazy_frame=pl.LazyFrame({"a": range(1000), "b": range(1000, 2000)}),
        iter_rows_options={"buffer_size": 250, "named": True},
    ),
    IterRowsCase(
        "iter_rows_large_dataset",
        lazy_frame=_numbered_rows(50_000, value=range(50_000, 100_000)),
        iter_rows_options={"buffer_size": 5000, "named": True},
    ),
]


@pytest.mark.parametrize("case", as_params(ROWS))
def test_iter_rows(case: IterRowsCase):
    got = list(case.lazy_frame.piot.iter_rows(**case.iter_rows_options))
    want = case.lazy_frame.collect().rows(named=case.iter_rows_options.get("named", False))
    if case.iter_rows_options.get("maintain_order", True):
        assert got == want
    else:
        assert sorted(got, key=lambda r: r["id"]) == want
