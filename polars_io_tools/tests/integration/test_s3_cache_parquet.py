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
