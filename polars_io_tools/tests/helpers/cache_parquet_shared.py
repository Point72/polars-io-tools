"""Shared cache test logic and the cache assertions: cached reads match native Polars, upstream calls are counted."""

import datetime
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_io_tools as cpl
from polars_io_tools.io_sources.util import _storage_options_for, register_io_source_with_is_pure


def exercise_daily_cache_parquet(
    cache_root: str,
    aws_profile: str | None = None,
    partition_format: str | None = "theYear=$year/theMonth=$month/theDay=$day",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Shared test logic for daily cache_parquet behavior.

    Returns a tuple of (collected_initial, appended, filtered) DataFrames.
    """
    # Build a small LazyFrame with a date column
    lf = pl.DataFrame(
        {
            "date": pl.date_range(datetime.datetime(2024, 6, 1), datetime.datetime(2024, 6, 3), interval="1d", eager=True),
            "value": [11, 22, 33],
        }
    ).lazy()

    # First write with custom partition template; returned df should match input slice
    collected = lf.piot.cache_parquet(
        cache_path=cache_root,
        date_column="date",
        time_unit="daily",
        partition_format=partition_format,
        aws_profile=aws_profile,
        cache_mode=cpl.CacheMode.REBUILD,
    ).collect()

    # Append a row
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

    # Verify via scan_parquet
    polars_opts = _storage_options_for(cache_root, aws_profile=aws_profile).polars
    pl_cache_root = cache_root + "/**/*.parquet"
    # Read back all cached rows via wildcard for sanity; caller may assert on this separately.
    _ = pl.scan_parquet(pl_cache_root, storage_options=polars_opts).sort("date").collect()

    # Verify filters are handled properly (predicate applied to cached data)
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
    """``lf`` filtered by ``pred`` (``None`` keeps every row), collected."""
    return (lf if pred is None else lf.filter(pred)).collect()


def assert_cached_matches_native(source: pl.LazyFrame, cache_path, pred: pl.Expr | None, *, runs: int = 2, **kwargs) -> pl.DataFrame:
    """``cache_parquet(source, cache_path, **kwargs)`` filtered by ``pred`` equals ``source`` filtered by ``pred`` on each of ``runs`` reads.

    The first read is cold unless the cache was filled before; later reads are warm. Rows are compared in any order.
    If Polars rejects the query on ``source``, every read must raise the same exception type. Returns the last result.
    """
    from polars_io_tools.io_sources.lazy_cache_parquet import cache_parquet

    try:
        want = _sorted(collect_filtered(source, pred))
    except pl.exceptions.PolarsError as exc:
        for _ in range(runs):
            with pytest.raises(type(exc)):
                collect_filtered(cache_parquet(source, cache_path, **kwargs), pred)
        return None
    for _ in range(runs):
        got = collect_filtered(cache_parquet(source, cache_path, **kwargs), pred)
        assert_frame_equal(_sorted(got), want)
    return got


class UpstreamSpy:
    """An IO source over ``df`` that records the predicate of every call; ``lazy`` is the frame to cache."""

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

    def kept(self, probe: pl.DataFrame, call: int = -1) -> pl.DataFrame:
        """The rows of ``probe`` that the predicate of upstream call ``call`` keeps."""
        pred = self.predicates[call]
        return probe if pred is None else probe.filter(pred)


def cache_files(root) -> dict[str, pl.DataFrame]:
    """Every parquet file under ``root`` by POSIX path relative to ``root``."""
    root = Path(root)
    return {p.relative_to(root).as_posix(): pl.read_parquet(p) for p in sorted(root.rglob("*.parquet"))}


def assert_cache_files(root, expected: dict[str, list | None], column: str = "value", *, exact: bool = False) -> None:
    """Each listed file holds the listed ``column`` values in any order (``[]`` is an empty file); ``None`` means no file.

    With ``exact``, no other file exists.
    """
    files = cache_files(root)
    for path, values in expected.items():
        if values is None:
            assert path not in files, f"unexpected file {path}"
        else:
            assert path in files, f"missing file {path}; have {sorted(files)}"
            assert sorted(files[path][column].to_list()) == sorted(values), (path, files[path])
    if exact:
        assert sorted(files) == sorted(p for p, v in expected.items() if v is not None), sorted(files)
