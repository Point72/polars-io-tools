"""Tests for ``cache_parquet(sort_by=...)``: per-file physical sort and row-group pruning.

``sort_by`` is a write-side layout option. It physically orders the rows of each cache file so that
(1) per-row-group statistics on the first key become non-overlapping, letting ``scan_parquet`` skip row
groups, and (2) grouped temporal operations that require a sorted index run on the cached read without a
manual sort. It intentionally does **not** mark the returned frame as globally sorted, because the key is
not monotonic across partitions.
"""

import datetime
import glob
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from polars_io_tools.io_sources.lazy_cache_parquet import CacheMode, cache_parquet

_DT = pl.Datetime("us")


def _cache_files(cache_path) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(Path(cache_path) / "**" / "*.parquet"), recursive=True))


def _intraday_frame() -> pl.DataFrame:
    """Two symbols, one month, deliberately shuffled timestamps."""
    base = datetime.datetime(2024, 1, 10, 9, 30)
    rows = []
    for sym in ("b", "a"):
        for ts in (3, 1, 4, 2):
            rows.append({"sym": sym, "ts": ts, "date": base})
    return pl.DataFrame(rows, schema_overrides={"date": _DT})


def test_each_partition_file_is_physically_sorted(tmp_path):
    df = _intraday_frame()
    cache_parquet(df.lazy(), tmp_path, date_column="date", time_unit="monthly", extra_partition_cols="sym", sort_by="ts").collect()

    files = _cache_files(tmp_path)
    assert len(files) == 2  # one file per symbol
    for f in files:
        ts = pl.read_parquet(f)["ts"]
        assert ts.is_sorted(), f"{f.name} not sorted: {ts.to_list()}"
        assert ts.to_list() == [1, 2, 3, 4]


def test_descending_sort(tmp_path):
    df = _intraday_frame()
    cache_parquet(df.lazy(), tmp_path, date_column="date", time_unit="monthly", extra_partition_cols="sym", sort_by="ts", descending=True).collect()
    for f in _cache_files(tmp_path):
        assert pl.read_parquet(f)["ts"].to_list() == [4, 3, 2, 1]


def test_single_partition_no_date_no_extra_cols(tmp_path):
    """The synthetic single-partition key must not break the sort (regression for the 'data' literal)."""
    df = pl.DataFrame({"ts": [3, 1, 2], "v": [9.0, 8.0, 7.0]})
    cache_parquet(df.lazy(), tmp_path, date_column=None, sort_by="ts").collect()
    (f,) = _cache_files(tmp_path)
    assert pl.read_parquet(f)["ts"].to_list() == [1, 2, 3]


def test_group_by_dynamic_runs_without_manual_sort(tmp_path):
    df = _intraday_frame()
    lf = cache_parquet(df.lazy(), tmp_path, date_column="date", time_unit="monthly", extra_partition_cols="sym", sort_by="ts")
    # No manual .sort(); the physical write-side order satisfies the sorted-index requirement.
    out = lf.group_by_dynamic("ts", every="1i", group_by="sym").agg(pl.len().alias("n")).collect()
    assert out.height == 8  # 4 one-unit windows x 2 symbols


def test_rolling_runs_without_manual_sort(tmp_path):
    df = _intraday_frame()
    lf = cache_parquet(df.lazy(), tmp_path, date_column="date", time_unit="monthly", extra_partition_cols="sym", sort_by="ts")
    out = lf.rolling(index_column="ts", period="2i", group_by="sym").agg(pl.len().alias("n")).collect()
    assert out.height == df.height


def test_unsorted_cache_rejects_group_by_dynamic(tmp_path):
    """Without sort_by the index is not physically ordered, so the temporal op fails loudly (never silently wrong)."""
    df = _intraday_frame()
    lf = cache_parquet(df.lazy(), tmp_path, date_column="date", time_unit="monthly", extra_partition_cols="sym")
    with pytest.raises(pl.exceptions.PolarsError, match="not sorted"):
        lf.group_by_dynamic("ts", every="1i", group_by="sym").agg(pl.len()).collect()


def test_returned_frame_is_not_marked_globally_sorted(tmp_path):
    """sort_by must not assert a global sorted flag: across symbols the key is not monotonic."""
    df = _intraday_frame()
    lf = cache_parquet(df.lazy(), tmp_path, date_column="date", time_unit="monthly", extra_partition_cols="sym", sort_by="ts")
    assert lf.collect()["ts"].flags["SORTED_ASC"] is False


def test_row_groups_are_pruned_on_range_predicate(tmp_path):
    import numpy as np

    rng = np.random.default_rng(0)
    n = 30_000
    df = pl.DataFrame({"k": rng.integers(0, 100_000, n), "v": rng.random(n)})

    def row_groups_touched(cache_path, lo, hi):
        (f,) = _cache_files(cache_path)
        md = pq.read_metadata(f)
        col = md.schema.names.index("k")
        total = md.num_row_groups
        touched = sum(not (md.row_group(i).column(col).statistics.max < lo or md.row_group(i).column(col).statistics.min > hi) for i in range(total))
        return total, touched

    unsorted = tmp_path / "unsorted"
    sorted_ = tmp_path / "sorted"
    cache_parquet(df.lazy(), unsorted, date_column=None, write_kwargs={"row_group_size": 5_000}).collect()
    cache_parquet(df.lazy(), sorted_, date_column=None, sort_by="k", write_kwargs={"row_group_size": 5_000}).collect()

    total_u, touched_u = row_groups_touched(unsorted, 100, 200)
    total_s, touched_s = row_groups_touched(sorted_, 100, 200)
    assert total_u > 1 and total_s > 1
    assert touched_u == total_u  # unsorted: every row group spans the full range, nothing can be skipped
    assert touched_s < touched_u  # sorted: the range lands in only a few row groups


def test_sorted_read_returns_same_rows_as_unsorted(tmp_path):
    """Sorting changes layout, not contents."""
    df = _intraday_frame()
    plain = cache_parquet(df.lazy(), tmp_path / "plain", date_column="date", time_unit="monthly", extra_partition_cols="sym").collect()
    srt = cache_parquet(df.lazy(), tmp_path / "srt", date_column="date", time_unit="monthly", extra_partition_cols="sym", sort_by="ts").collect()
    pl.testing.assert_frame_equal(plain.sort("sym", "ts"), srt.sort("sym", "ts"))


def test_rebuild_resorts_existing_partition(tmp_path):
    """CACHE does not rewrite existing files; REBUILD applies a newly configured sort_by."""
    df = _intraday_frame()
    cache_parquet(df.lazy(), tmp_path, date_column="date", time_unit="monthly", extra_partition_cols="sym").collect()
    cache_parquet(
        df.lazy(), tmp_path, date_column="date", time_unit="monthly", extra_partition_cols="sym", sort_by="ts", cache_mode=CacheMode.REBUILD
    ).collect()
    for f in _cache_files(tmp_path):
        assert pl.read_parquet(f)["ts"].to_list() == [1, 2, 3, 4]


def test_multi_key_sort_with_mixed_direction(tmp_path):
    """Secondary key ordering and per-key descending, verified against an independent reference."""
    df = pl.DataFrame(
        {
            "k1": [2, 1, 2, 1, 1, 2],
            "k2": [10, 30, 20, 10, 20, 30],
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    cache_parquet(df.lazy(), tmp_path, date_column=None, sort_by=["k1", "k2"], descending=[False, True]).collect()
    (f,) = _cache_files(tmp_path)
    got = pl.read_parquet(f).select("k1", "k2")
    expected = df.sort(["k1", "k2"], descending=[False, True]).select("k1", "k2")
    pl.testing.assert_frame_equal(got, expected)


def test_within_file_order_is_pure_sort_by_for_subdaily_bucket(tmp_path):
    """Daily rows land in one monthly file; the file must be ordered purely by sort_by, not by raw date.

    Regression against prefixing the sort with the raw date column (finer than the month bucket) instead
    of the month bucket -- that would order rows by day then sort_by, leaving the file not sorted by sort_by.
    """
    rows = []
    for day in (3, 1, 2):  # three different days, same month
        for ts in (30, 10, 20):
            rows.append({"ts": ts, "date": datetime.datetime(2024, 1, 1 + day, 9, 30)})
    df = pl.DataFrame(rows, schema_overrides={"date": _DT})
    assert df["date"].n_unique() == 3  # genuinely multiple days in the one monthly partition

    cache_parquet(df.lazy(), tmp_path, date_column="date", time_unit="monthly", sort_by="ts").collect()
    (f,) = _cache_files(tmp_path)
    ts = pl.read_parquet(f)["ts"]
    assert ts.is_sorted(), f"file ordered by raw date, not sort_by: {ts.to_list()}"
    assert ts.to_list() == sorted(ts.to_list())


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"sort_by": "missing"}, "not present"),
        ({"sort_by": ["ts"], "descending": [True, False]}, "descending must"),
    ],
)
def test_validation_errors(tmp_path, kwargs, match):
    df = pl.DataFrame({"ts": [3, 1, 2]})
    with pytest.raises(ValueError, match=match):
        cache_parquet(df.lazy(), tmp_path, date_column=None, **kwargs).collect()
