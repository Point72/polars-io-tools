"""Helpers for the ``cache_parquet`` and ``cache_memory`` tests.

They compare a cached read with the same read in plain Polars, count and record the reads that reach the data behind a
cache, and list the parquet files a cache wrote.
"""

import datetime
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_io_tools as cpl
from polars_io_tools.io_sources.util import _storage_options_for, register_io_source_with_is_pure
from polars_io_tools.tests.helpers.assertions import failure_note


def write_append_and_filter_daily_cache(
    cache_root: str,
    aws_profile: str | None = None,
    partition_format: str | None = "theYear=$year/theMonth=$month/theDay=$day",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Fill a daily ``cache_parquet`` cache at ``cache_root`` with 1-3 June 2024, add 5 June, then read 2-3 June back.

    Returns the frame each of the three reads gave: the first fill, the read of 4-7 June after adding 5 June, and the
    filtered read of 2-3 June.
    """
    lf = pl.DataFrame(
        {
            "date": pl.date_range(datetime.datetime(2024, 6, 1), datetime.datetime(2024, 6, 3), interval="1d", eager=True),
            "value": [11, 22, 33],
        }
    ).lazy()

    collected = lf.piot.cache_parquet(
        cache_path=cache_root,
        date_column="date",
        time_unit="daily",
        partition_format=partition_format,
        aws_profile=aws_profile,
        cache_mode=cpl.CacheMode.REBUILD,
    ).collect()

    appended = (
        pl.DataFrame({"date": [datetime.datetime(2024, 6, 5)], "value": [44]})
        .with_columns(pl.col("date").cast(pl.Date))
        .lazy()
        .piot.cache_parquet(
            cache_path=cache_root,
            date_column="date",
            time_unit="daily",
            partition_format=partition_format,
            aws_profile=aws_profile,
        )
        .filter(pl.col("date").is_between(datetime.date(2024, 6, 4), datetime.date(2024, 6, 7)))
        .collect()
    )

    # The cache files must be readable by plain Polars with the cache's storage options.
    polars_options = _storage_options_for(cache_root, aws_profile=aws_profile).polars
    pl.scan_parquet(cache_root + "/**/*.parquet", storage_options=polars_options).sort("date").collect()

    filtered = (
        lf.piot.cache_parquet(
            cache_path=cache_root,
            date_column="date",
            time_unit="daily",
            partition_format=partition_format,
            aws_profile=aws_profile,
        )
        .filter(pl.col("date").is_between(datetime.date(2024, 6, 2), datetime.date(2024, 6, 3)))
        .sort("date")
        .collect()
    )

    return collected, appended, filtered


def _sorted(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(df.columns, nulls_last=True)


def collect_filtered(lf: pl.LazyFrame, pred: pl.Expr | None) -> pl.DataFrame:
    """Collect ``lf`` filtered by ``pred``; ``None`` keeps every row."""
    return (lf if pred is None else lf.filter(pred)).collect()


def assert_cache_matches_plain_polars(source: pl.LazyFrame, cache_path, pred: pl.Expr | None, *, runs: int = 2, **kwargs) -> pl.DataFrame:
    """Read ``cache_parquet(source, cache_path, **kwargs)`` filtered by ``pred`` ``runs`` times; each read must give the
    rows ``source`` filtered by ``pred`` gives, in any order.

    The first read fills the cache unless it was filled before; later reads come from the cache. If Polars rejects the
    query on ``source``, every read must raise the same exception type. Returns the last read.
    """
    from polars_io_tools.io_sources.lazy_cache_parquet import cache_parquet

    try:
        want = _sorted(collect_filtered(source, pred))
    except pl.exceptions.PolarsError as exc:
        for _ in range(runs):
            with pytest.raises(type(exc)):
                collect_filtered(cache_parquet(source, cache_path, **kwargs), pred)
        return None
    for run in range(runs):
        got = collect_filtered(cache_parquet(source, cache_path, **kwargs), pred)
        with failure_note(f"read {run + 1} of {runs}, filtered by {pred}"):
            assert_frame_equal(_sorted(got), want)
    return got


class RecordingSource:
    """A source serving ``df`` that records the filter it receives on every read; ``lazy`` is its lazy frame.

    ``predicates`` holds one entry per read (``None`` when no filter reached it), so a test can count the reads that
    got past a cache and see what each one asked for.
    """

    def __init__(self, df: pl.DataFrame, *, is_pure: bool = False):
        self.df = df
        self.predicates: list[pl.Expr | None] = []

        def read(with_columns, predicate, n_rows, batch_size):
            self.predicates.append(predicate)
            out = df if predicate is None else df.filter(predicate)
            yield out if with_columns is None else out.select(with_columns)

        self.lazy = register_io_source_with_is_pure(read, schema=df.schema, is_pure=is_pure)

    @property
    def calls(self) -> int:
        return len(self.predicates)

    def rows_kept_by_read(self, rows: pl.DataFrame, read: int = -1) -> pl.DataFrame:
        """The rows of ``rows`` that the filter of read number ``read`` (the last one by default) keeps."""
        pred = self.predicates[read]
        return rows if pred is None else rows.filter(pred)


def read_cache_files(root) -> dict[str, pl.DataFrame]:
    """Read every parquet file under ``root``, keyed by its path relative to ``root`` with ``/`` separators."""
    root = Path(root)
    return {p.relative_to(root).as_posix(): pl.read_parquet(p) for p in sorted(root.rglob("*.parquet"))}


def assert_cache_files(root, expected: dict[str, list | None], column: str = "value", *, exact: bool = False) -> None:
    """Each path in ``expected`` (relative to ``root``) is a parquet file whose ``column`` holds the listed values in any
    order; ``[]`` is a file with no rows and ``None`` means the file must not exist. With ``exact``, no other file exists.
    """
    files = read_cache_files(root)
    for path, values in expected.items():
        if values is None:
            assert path not in files, f"unexpected file {path}"
        else:
            assert path in files, f"missing file {path}; have {sorted(files)}"
            assert sorted(files[path][column].to_list()) == sorted(values), (path, files[path])
    if exact:
        assert sorted(files) == sorted(p for p, v in expected.items() if v is not None), sorted(files)
