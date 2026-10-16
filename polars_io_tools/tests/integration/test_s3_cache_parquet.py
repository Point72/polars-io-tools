"""
Integration test for cache_parquet using S3.

Requires passing --aws-profile and optionally --bucket via pytest CLI.
Bucket defaults to "polars-io-tools-tests". Root path derives from file name.
"""

import datetime

import pytest

import polars_io_tools as cpl  # noqa
from polars_io_tools.tests.helpers.cache_parquet_shared import exercise_daily_cache_parquet


@pytest.mark.integration
@pytest.mark.aws_profile_required
def test_s3_cache_parquet_daily(s3_root, aws_profile):
    cache_root = s3_root
    collected, appended, filtered = exercise_daily_cache_parquet(
        cache_root=cache_root,
        aws_profile=aws_profile,
        partition_format="theYear=$year/theMonth=$month/theDay=$day",
    )
    assert collected.shape == (3, 2)
    assert collected["date"].to_list() == [datetime.date(2024, 6, 1), datetime.date(2024, 6, 2), datetime.date(2024, 6, 3)]
    assert collected["value"].to_list() == [11, 22, 33]
    assert appended.shape == (1, 2)
    assert appended["date"].to_list() == [datetime.date(2024, 6, 5)]
    assert appended["value"].to_list() == [44]
    assert filtered["date"].to_list() == [datetime.date(2024, 6, 2), datetime.date(2024, 6, 3)]
    assert filtered["value"].to_list() == [22, 33]


@pytest.mark.integration
@pytest.mark.aws_profile_required
def test_s3_cache_parquet_no_write_modes(s3_root, aws_profile):
    import polars as pl
    import pyarrow.fs as pa_fs

    from polars_io_tools.io_sources.lazy_cache_parquet import CacheMode, cache_parquet
    from polars_io_tools.io_sources.util import _storage_options_for

    fs = pa_fs.S3FileSystem(**_storage_options_for(s3_root, aws_profile=aws_profile).pyarrow)
    root = s3_root.replace("s3://", "", 1)

    def snapshot():
        infos = fs.get_file_info(pa_fs.FileSelector(root, recursive=True, allow_not_found=True))
        return {(i.path, i.type, i.size, i.mtime_ns) for i in infos}

    jan, feb = datetime.date(2024, 1, 15), datetime.date(2024, 2, 15)
    jan_feb = pl.col("date").is_between(datetime.date(2024, 1, 1), datetime.date(2024, 2, 29))
    cache_path = f"{s3_root}/no_write_modes"
    missing_path = f"{s3_root}/no_write_modes_missing"

    cache_parquet(pl.LazyFrame({"date": [jan], "value": [1]}), cache_path, "date", aws_profile=aws_profile).filter(
        pl.col("date").is_between(datetime.date(2024, 1, 1), datetime.date(2024, 1, 31))
    ).collect()
    before = snapshot()

    def fail():
        raise AssertionError("upstream must not be called")

    upstream = pl.LazyFrame({"date": [jan, feb], "value": [91, 92]})
    read_only = cache_parquet(upstream, cache_path, "date", aws_profile=aws_profile, cache_mode=CacheMode.READ_ONLY)
    assert sorted(read_only.filter(jan_feb).collect()["value"].to_list()) == [1, 92]
    cache_only = cache_parquet(fail, cache_path, "date", aws_profile=aws_profile, cache_mode=CacheMode.CACHE_ONLY)
    assert cache_only.filter(jan_feb).collect()["value"].to_list() == [1]

    missing = cache_parquet(upstream, missing_path, "date", aws_profile=aws_profile, cache_mode=CacheMode.READ_ONLY)
    assert sorted(missing.collect()["value"].to_list()) == [91, 92]
    schema = upstream.collect_schema()
    missing = cache_parquet(fail, missing_path, "date", aws_profile=aws_profile, cache_mode=CacheMode.CACHE_ONLY, schema=schema)
    assert missing.collect().is_empty()

    assert snapshot() == before
