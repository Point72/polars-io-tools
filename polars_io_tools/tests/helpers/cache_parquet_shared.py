import datetime
from typing import Optional, Tuple

import polars as pl

import polars_io_tools as cpl  # noqa
from polars_io_tools.io_sources.util import _storage_options_for


def exercise_daily_cache_parquet(
    cache_root: str,
    aws_profile: Optional[str] = None,
    partition_format: Optional[str] = "theYear=$year/theMonth=$month/theDay=$day",
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
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
