"""Tests for lazy_cache_parquet.py (``cache_parquet``).

Results are compared with plain Polars by ``assert_cache_matches_plain_polars`` and the files left behind are checked
by ``assert_cache_files``. Most sections are tables of cases; tests that patch writers, clocks or locks are plain
functions. The sections, in order (search for ``# --- Section:`` to move between them):

- Planning helpers: which partitions to read and write
- CacheMode as a pydantic field
- Layout: the files a write leaves and the metadata in them
- Sequences of reads on one cache (``STEPS``)
- Results match plain Polars in every mode and dtype
- Calls to the upstream source
- write_bounding_columns
- Filters that list no partitions, wildcards and REBUILD
- Expiry (ttl and final_after) and refresh
- Open-sink cap, failed writes, write batches and listing
- More expiry and refresh: other cache modes, a callable ttl, null dates
- Retries of failed cache reads
- Write lock
- File visitor and write batches
- Time partitions, and a cache over another cache
- Partition values no file can be named, and files outside the cache
- Property test, and the bugs it found: random filters on a cache, cold then warm, each compared with a plain filter
- Threads and pickling
- Regressions: the ``REGRESSIONS`` table. Each row fills a cache with seed reads, then runs a list of reads, each of
  which must return what plain Polars returns, on an empty cache and again on a filled one. The rows of
  ``NO_REWRITE_REGRESSIONS`` also require that the seed's files are not rewritten. Bugs that need a patched platform,
  pre-written legacy files or a changing upstream are plain functions after the tables.
- Partition keys: parsing, dtypes, spellings, and exclusion predicates
- REBUILD scope: what a rebuild may overwrite or remove
- Signed zero keys: -0.0 and 0.0 are one partition
- Modes that never write: READ_ONLY and CACHE_ONLY
- sort_by: a physical sort of each cache file
"""

import contextlib
import datetime
import glob
import importlib
import os
import pickle
import shutil
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field
from datetime import timedelta
from decimal import Decimal
from pathlib import Path, PureWindowsPath
from types import EllipsisType, SimpleNamespace
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

import cloudpickle
import numpy as np
import polars as pl
import portion
import pyarrow.parquet as pq
import pydantic
import pytest
from hypothesis import example, given, strategies as st
from polars.exceptions import ComputeError, PanicException
from polars.testing import assert_frame_equal
from pydantic import BaseModel

from polars_io_tools._compat import POLARS_GE_2, POLARS_HAS_PARTITION_BY
from polars_io_tools.io_sources import lazy_cache_parquet
from polars_io_tools.io_sources.lazy_cache_parquet import (
    CacheMode,
    Enumerability,
    PartitionInfo,
    _build_final_scan,
    _compute_clipped_expected_partitions,
    _compute_partitions_to_write,
    _construct_partitions_df_from_existing,
    _default_expired,
    _extract_existing_files,
    _get_expected_partitions_df,
    _get_fs_path_directory_info,
    _rebuild_partition_scope,
    cache_parquet,
)
from polars_io_tools.io_sources.util import register_io_source_with_is_pure
from polars_io_tools.tests.helpers.assertions import failure_note
from polars_io_tools.tests.helpers.cache_helpers import (
    RecordingSource,
    assert_cache_files,
    assert_cache_matches_plain_polars,
    collect_filtered,
    read_cache_files,
    write_append_and_filter_daily_cache,
)
from polars_io_tools.tests.helpers.cases import DATE_AND_DATETIME_DTYPES, DATE_PREDICATES, DatePredicateCase, as_params, daily_frame
from polars_io_tools.tests.helpers.pickling import pickle_round_trip
from polars_io_tools.tests.helpers.polars_versions import polars_has, xfail_unless_polars_has
from polars_io_tools.tests.helpers.property_checks import SETTINGS, check_cache_keeps_filtered_rows
from polars_io_tools.tests.helpers.strategies import PARTITION_KEY_POOLS, partitioned_frames_and_predicates
from polars_io_tools.tests.helpers.subprocesses import SUBPROCESS_TIMEOUT_S, run_in_subprocess

UTC = datetime.UTC
MODES = [CacheMode.CACHE, CacheMode.READ_ONLY, CacheMode.CACHE_ONLY, CacheMode.REBUILD, CacheMode.IGNORE]


def _lazy_frame(**cols) -> pl.LazyFrame:
    return pl.LazyFrame(cols)


# --- Section: Planning helpers: which partitions to read and write ---

# _get_expected_partitions_df: the partitions a filter covers, given the date column, extra partition columns and time
# unit. Expected frames hold the extra columns, then the date column; a column the filter does not limit is null.

SCHEMA = pl.Schema(
    {
        "date": pl.Date,
        "timestamp": pl.Datetime("us"),
        "region": pl.String,
        "country": pl.String,
        "category": pl.String,
        "value": pl.Float64,
        "count": pl.Int64,
        "active": pl.Boolean,
    }
)


def expected_partitions(dates, date_column="date", **extra_columns) -> pl.DataFrame:
    """A partition frame: ``extra_columns`` (a value repeated, or a list) followed by the date column ``dates``."""
    n = len(dates)
    columns = {k: v if isinstance(v, list) else [v] * n for k, v in extra_columns.items()}
    return pl.DataFrame({**columns, date_column: dates}, schema={**{k: pl.String for k in extra_columns}, date_column: SCHEMA[date_column]})


@dataclass(frozen=True)
class ExpectedPartitionsCase:
    """A filter on a cache partitioned by a date column and maybe other columns, and the partitions it covers."""

    id: str
    _: KW_ONLY
    predicate: pl.Expr
    expected: pl.DataFrame
    time_unit: str = "monthly"
    # the other partition columns
    extra_columns: tuple[str, ...] | list[str] = ()
    date_column: str = "date"


def _first_of_months(*months, year=2024):
    return [datetime.date(year, m, 1) for m in months]


_MAR_10_TO_MAY_20 = pl.col("date").is_between(datetime.date(2024, 3, 10), datetime.date(2024, 5, 20))
EXPECTED_PARTITIONS = [
    ExpectedPartitionsCase("none_predicate_behavior-date", predicate=None, expected=expected_partitions([None])),
    ExpectedPartitionsCase(
        "none_predicate_behavior-extra_cols",
        predicate=None,
        expected=expected_partitions([None], region=None, country=None),
        time_unit="daily",
        extra_columns=["region", "country"],
    ),
    ExpectedPartitionsCase(
        "simple_date_equality_monthly", predicate=pl.col("date") == datetime.date(2024, 3, 15), expected=expected_partitions(_first_of_months(3))
    ),
    ExpectedPartitionsCase(
        "simple_date_equality_daily",
        predicate=pl.col("date") == datetime.date(2024, 5, 20),
        expected=expected_partitions([datetime.date(2024, 5, 20)]),
        time_unit="daily",
    ),
    ExpectedPartitionsCase(
        "date_range_creates_multiple_monthly_partitions",
        predicate=pl.col("date").is_between(datetime.date(2024, 1, 15), datetime.date(2024, 4, 10)),
        expected=expected_partitions(_first_of_months(1, 2, 3, 4)),
    ),
    ExpectedPartitionsCase(
        "extra_columns_with_simple_predicates",
        predicate=(pl.col("date") == datetime.date(2024, 7, 4)) & (pl.col("region") == "US"),
        expected=expected_partitions([datetime.date(2024, 7, 4)], region="US"),
        time_unit="daily",
        extra_columns=["region"],
    ),
    ExpectedPartitionsCase(
        "multiple_extra_columns_complex_predicate",
        predicate=(pl.col("date") == datetime.date(2024, 8, 12))
        & pl.col("region").is_in(["US", "EU", "APAC"])
        & pl.col("category").is_in(["A", "B"]),
        expected=expected_partitions([datetime.date(2024, 8, 12)] * 6, region=["US", "US", "EU", "EU", "APAC", "APAC"], category=["A", "B"] * 3),
        time_unit="daily",
        extra_columns=["region", "category"],
    ),
    ExpectedPartitionsCase(
        "or_conditions_create_multiple_partitions",
        predicate=(pl.col("date") == datetime.date(2024, 9, 5)) | (pl.col("date") == datetime.date(2024, 10, 15)),
        expected=expected_partitions(_first_of_months(9, 10)),
    ),
    ExpectedPartitionsCase(
        "or_with_extra_columns_creates_all_combinations",
        predicate=(pl.col("date") == datetime.date(2024, 11, 8))
        & ((pl.col("region") == "US") | (pl.col("region") == "EU"))
        & ((pl.col("category") == "X") | (pl.col("category") == "Y")),
        expected=expected_partitions([datetime.date(2024, 11, 8)] * 4, region=["US", "US", "EU", "EU"], category=["X", "Y"] * 2),
        time_unit="daily",
        extra_columns=["region", "category"],
    ),
    ExpectedPartitionsCase(
        "contradictory_predicates",
        predicate=(pl.col("date") == datetime.date(2024, 1, 1)) & (pl.col("date") == datetime.date(2024, 2, 1)),
        expected=expected_partitions([]),
    ),
    ExpectedPartitionsCase(
        "different_time_units_same_predicate-daily",
        predicate=_MAR_10_TO_MAY_20,
        expected=expected_partitions(pl.date_range(datetime.date(2024, 3, 10), datetime.date(2024, 5, 20), eager=True).to_list()),
        time_unit="daily",
    ),
    ExpectedPartitionsCase(
        "different_time_units_same_predicate-monthly", predicate=_MAR_10_TO_MAY_20, expected=expected_partitions(_first_of_months(3, 4, 5))
    ),
    ExpectedPartitionsCase(
        "different_time_units_same_predicate-yearly",
        predicate=_MAR_10_TO_MAY_20,
        expected=expected_partitions(_first_of_months(1)),
        time_unit="yearly",
    ),
    ExpectedPartitionsCase(
        "open_ended_lower_bound_only", predicate=pl.col("date") >= datetime.date(2024, 6, 5), expected=expected_partitions([]), time_unit="daily"
    ),
    ExpectedPartitionsCase(
        "open_ended_upper_bound_only", predicate=pl.col("date") <= datetime.date(2024, 6, 2), expected=expected_partitions([]), time_unit="daily"
    ),
    ExpectedPartitionsCase(
        "predicate_with_non_partition_columns_ignored",
        predicate=(pl.col("date") == datetime.date(2024, 12, 25)) & (pl.col("region") == "US") & (pl.col("value") > 100.0),
        expected=expected_partitions([datetime.date(2024, 12, 25)], region="US"),
        time_unit="daily",
        extra_columns=["region"],
    ),
    ExpectedPartitionsCase(
        "complex_nested_or_and_conditions",
        predicate=((pl.col("date") == datetime.date(2024, 6, 1)) & (pl.col("region") == "US"))
        | ((pl.col("date") == datetime.date(2024, 7, 1)) & (pl.col("region") == "EU")),
        expected=expected_partitions(_first_of_months(6, 7), region=["US", "EU"]),
        extra_columns=["region"],
    ),
    ExpectedPartitionsCase(
        "timestamp_column_as_date_column",
        predicate=pl.col("timestamp") == datetime.datetime(2024, 4, 15, 14, 30, 45),
        expected=expected_partitions([datetime.datetime(2024, 4, 15)], date_column="timestamp"),
        time_unit="daily",
        date_column="timestamp",
    ),
    ExpectedPartitionsCase(
        "large_date_ranges_yearly_partitioning",
        predicate=pl.col("date").is_between(datetime.date(2020, 1, 1), datetime.date(2024, 12, 31)) & pl.col("region").is_in(["US", "EU"]),
        expected=expected_partitions([datetime.date(y, 1, 1) for y in range(2020, 2025) for _ in "ab"], region=["US", "EU"] * 5),
        time_unit="yearly",
        extra_columns=["region"],
    ),
    ExpectedPartitionsCase(
        "is_in_predicate_with_multiple_dates",
        predicate=pl.col("date").is_in([datetime.date(2024, 2, 14), datetime.date(2024, 3, 17), datetime.date(2024, 4, 22)]),
        expected=expected_partitions(_first_of_months(2, 3, 4)),
    ),
]


@pytest.mark.parametrize("case", as_params(EXPECTED_PARTITIONS))
def test_get_expected_partitions_df(case: ExpectedPartitionsCase):
    result = _get_expected_partitions_df(
        pred=case.predicate, date_column=case.date_column, extra_cols=list(case.extra_columns), time_unit=case.time_unit, schema=SCHEMA
    )
    assert_frame_equal(result, case.expected, check_row_order=False)


# _compute_clipped_expected_partitions: a one-sided interval is clipped to the cached date range, or None.


def _june(*days: int) -> list[datetime.date]:
    return [datetime.date(2024, 6, d) for d in days]


@dataclass(frozen=True)
class NotClippedCase:
    """A date interval, and cached partitions it cannot be clipped to."""

    id: str
    _: KW_ONLY
    interval: portion.Interval
    existing: pl.DataFrame


def _cached_dates(dates: list[datetime.date]) -> pl.DataFrame:
    return pl.DataFrame({"date": dates}, schema={"date": pl.Date})


NOT_CLIPPED = [
    NotClippedCase("empty_interval", interval=portion.empty(), existing=_cached_dates(_june(1))),
    NotClippedCase(
        "fully_bounded_interval", interval=portion.closed(datetime.date(2024, 6, 1), datetime.date(2024, 6, 5)), existing=_cached_dates(_june(1, 5))
    ),
    NotClippedCase("fully_unbounded_interval", interval=portion.open(-portion.inf, portion.inf), existing=_cached_dates(_june(1))),
    NotClippedCase("empty_existing_partitions", interval=portion.closedopen(datetime.date(2024, 6, 1), portion.inf), existing=_cached_dates([])),
    NotClippedCase(
        "missing_date_column", interval=portion.closedopen(datetime.date(2024, 6, 1), portion.inf), existing=pl.DataFrame({"other_col": [1, 2, 3]})
    ),
    NotClippedCase("invalid_clipped_range", interval=portion.closedopen(datetime.date(2024, 6, 10), portion.inf), existing=_cached_dates(_june(1))),
]


@pytest.mark.parametrize("case", as_params(NOT_CLIPPED))
def test_compute_clipped_expected_partitions_returns_none(case: NotClippedCase):
    result = _compute_clipped_expected_partitions(
        predicate_interval=case.interval,
        existing_parts_df=case.existing,
        date_column="date",
        extra_cols=[],
        time_unit="daily",
        schema={"date": pl.Date},
    )
    assert result is None


@dataclass(frozen=True)
class ClippedPartitionsCase:
    """A one-sided date interval, the cached dates, and the partition dates the interval is clipped to."""

    id: str
    _: KW_ONLY
    interval: portion.Interval
    existing_dates: list[datetime.date]
    expected: list[datetime.date]
    time_unit: str = "daily"
    # also partition the cache by these regions, one per cached date
    regions: list[str] | None = None


CLIPPED = [
    ClippedPartitionsCase(
        "unbounded_above_clips_to_existing_max",
        interval=portion.closedopen(datetime.date(2024, 6, 2), portion.inf),
        existing_dates=_june(1, 5),
        expected=_june(2, 3, 4, 5),
    ),
    ClippedPartitionsCase(
        "unbounded_below_clips_to_existing_min",
        interval=portion.openclosed(-portion.inf, datetime.date(2024, 6, 4)),
        existing_dates=_june(2, 5),
        expected=_june(2, 3, 4),
    ),
    ClippedPartitionsCase(
        "with_extra_cols_cross_product",
        interval=portion.closedopen(datetime.date(2024, 6, 1), portion.inf),
        existing_dates=_june(1, 2),
        expected=_june(1, 2),
        regions=["US", "EU"],
    ),
    ClippedPartitionsCase(
        "with_datetime_in_interval",
        interval=portion.closedopen(datetime.datetime(2024, 6, 2, 10, 30), portion.inf),
        existing_dates=_june(1, 3),
        expected=_june(2, 3),
    ),
    ClippedPartitionsCase(
        "monthly_time_unit",
        interval=portion.closedopen(datetime.date(2024, 2, 15), portion.inf),
        existing_dates=[datetime.date(2024, 1, 15), datetime.date(2024, 4, 10)],
        expected=_first_of_months(2, 3, 4),
        time_unit="monthly",
    ),
    ClippedPartitionsCase(
        "yearly_time_unit",
        interval=portion.closedopen(datetime.date(2022, 6, 15), portion.inf),
        existing_dates=[datetime.date(2021, 1, 1), datetime.date(2024, 3, 10)],
        expected=[datetime.date(y, 1, 1) for y in (2022, 2023, 2024)],
        time_unit="yearly",
    ),
    ClippedPartitionsCase(
        "single_date_existing", interval=portion.closedopen(datetime.date(2024, 6, 1), portion.inf), existing_dates=_june(1), expected=_june(1)
    ),
]


@pytest.mark.parametrize("case", as_params(CLIPPED))
def test_compute_clipped_expected_partitions(case: ClippedPartitionsCase):
    existing_df = _cached_dates(case.existing_dates)
    if case.regions:
        existing_df = existing_df.with_columns(region=pl.Series(case.regions))
    extra = ["region"] if case.regions else []
    schema = {"date": pl.Date, **({"region": pl.String} if case.regions else {})}
    result = _compute_clipped_expected_partitions(
        predicate_interval=case.interval, existing_parts_df=existing_df, date_column="date", extra_cols=extra, time_unit=case.time_unit, schema=schema
    )
    want = _cached_dates(case.expected)
    if case.regions:
        # Partition frames hold the extra columns before the date.
        want = pl.DataFrame({"region": case.regions}).join(want, how="cross")
    assert_frame_equal(result, want, check_row_order=False)


# Small helpers


@dataclass(frozen=True)
class PartitionsToWriteCase:
    """Expected and existing partitions, as days of January 2024, and the days left to write."""

    id: str
    _: KW_ONLY
    expected_days: list[int]
    existing_days: list[int]
    remaining_days: list[int]


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            PartitionsToWriteCase("empty_existing", expected_days=[1, 2], existing_days=[], remaining_days=[1, 2]),
            PartitionsToWriteCase("some_existing", expected_days=[1, 2, 3], existing_days=[1, 3], remaining_days=[2]),
            PartitionsToWriteCase("all_existing", expected_days=[1, 2], existing_days=[1, 2], remaining_days=[]),
        ]
    ),
)
def test_compute_partitions_to_write(case: PartitionsToWriteCase):
    def days(ds):
        return pl.DataFrame({"date": [datetime.date(2024, 1, d) for d in ds]}, schema={"date": pl.Date})

    info = PartitionInfo(
        enumerability=Enumerability.FINITE, expected_parts_df=days(case.expected_days), existing_parts_df=days(case.existing_days), join_cols=["date"]
    )
    assert_frame_equal(_compute_partitions_to_write(info), days(case.remaining_days))


def test_compute_partitions_to_write_no_join_cols():
    expected = pl.DataFrame({"val": [1]})
    info = PartitionInfo(enumerability=Enumerability.FINITE, expected_parts_df=expected, existing_parts_df=expected, join_cols=[])
    assert_frame_equal(_compute_partitions_to_write(info), expected)


def test_write_empty_parquet_file(tmp_path):
    from polars_io_tools.io_sources.lazy_cache_parquet import _write_empty_parquet_file

    path = str(tmp_path / "nested" / "one.parquet")
    schema = pl.Schema({"value": pl.Int64})
    _write_empty_parquet_file(path, schema=schema, metadata={}, storage_options={}, credential_provider=None, write_kwargs={})
    assert_frame_equal(pl.read_parquet(path), pl.DataFrame(schema=schema))


def test_build_read_plan_falls_back_to_glob_for_many_paths():
    from polars_io_tools.io_sources.lazy_cache_parquet import _MAX_ENUMERATED_SCAN_PATHS, _build_read_plan

    dates = [datetime.date(2024, 1, 1) + timedelta(days=i) for i in range(_MAX_ENUMERATED_SCAN_PATHS + 1)]
    info = PartitionInfo(
        enumerability=Enumerability.FINITE,
        expected_parts_df=pl.DataFrame({"date": dates}),
        existing_parts_df=pl.DataFrame({"date": dates, "file": [str(d) for d in dates]}),
        join_cols=["date"],
    )
    plan = _build_read_plan(
        partition_info=info,
        predicate=pl.col("date") >= dates[0],
        time_unit_dir="file:///tmp/cache/daily",
        has_cached_files=True,
        file_key_column="file",
    )
    assert plan.use_paths is None


_SCAN = pl.DataFrame({"x": [1, 2, 3, 4, 5], "y": list("abcde")})


@dataclass(frozen=True)
class FinalScanCase:
    """A filter, columns and row limit for ``_build_final_scan``, and the frame it returns."""

    id: str
    _: KW_ONLY
    pred: pl.Expr | None
    cols: list[str] | None
    n_rows: int | None
    expected: pl.DataFrame


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            FinalScanCase("cache_only", pred=None, cols=None, n_rows=None, expected=_SCAN),
            FinalScanCase("with_predicate", pred=pl.col("x") > 1, cols=None, n_rows=None, expected=_SCAN.filter(pl.col("x") > 1)),
            FinalScanCase("with_columns", pred=None, cols=["y"], n_rows=None, expected=_SCAN.select("y")),
            FinalScanCase("with_limit", pred=None, cols=None, n_rows=2, expected=_SCAN.head(2)),
        ]
    ),
)
def test_build_final_scan(case: FinalScanCase):
    assert_frame_equal(_build_final_scan(_SCAN.lazy(), case.pred, case.cols, case.n_rows).collect(), case.expected)


@dataclass(frozen=True)
class DirectoryInfoCase:
    """A cache root and time unit, and the path prefix and time directory ``_get_fs_path_directory_info`` returns."""

    id: str
    _: KW_ONLY
    # ``{tmp}`` and ``{uri}`` stand for the test's directory as a path and as a URI
    root: str
    time_unit: str
    prefix: str
    time_dir: str


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            DirectoryInfoCase(
                "strips_trailing_slash_s3",
                root="s3://mybucket/my/prefix/",
                time_unit="daily",
                prefix="mybucket/my/prefix/daily",
                time_dir="s3://mybucket/my/prefix/daily",
            ),
            DirectoryInfoCase(
                "preserves_double_leading_slash_s3",
                root="s3://mybucket//double/leading/",
                time_unit="monthly",
                prefix="mybucket//double/leading/monthly",
                time_dir="s3://mybucket//double/leading/monthly",
            ),
            DirectoryInfoCase("local_path_gets_file_prefix", root="{tmp}", time_unit="daily", prefix="{tmp}/daily", time_dir="{uri}/daily"),
            DirectoryInfoCase("file_uri_not_double_prefixed", root="{uri}", time_unit="daily", prefix="{tmp}/daily", time_dir="{uri}/daily"),
            DirectoryInfoCase("local_and_file_uri_equivalent", root="{uri}", time_unit="monthly", prefix="{tmp}/monthly", time_dir="{uri}/monthly"),
        ]
    ),
)
def test_get_fs_path_directory_info(tmp_path, case: DirectoryInfoCase):
    def fill(s):
        return s.format(tmp=str(tmp_path), uri=tmp_path.as_uri())

    assert _get_fs_path_directory_info(fill(case.root), case.time_unit) == (
        str(Path(fill(case.prefix))) if "{tmp}" in case.prefix else case.prefix,
        fill(case.time_dir),
    )


def test_path_as_file_uri_formats_windows_drive_path():
    from polars_io_tools.io_sources.lazy_cache_parquet import _path_as_file_uri

    uri = _path_as_file_uri(PureWindowsPath("C:/Users/runneradmin/AppData/Local/Temp/cache/daily"))
    assert uri == "file:///C:/Users/runneradmin/AppData/Local/Temp/cache/daily"


def test_strftime_from_template():
    from polars_io_tools.io_sources.lazy_cache_parquet import _strftime_from_template

    assert _strftime_from_template("theYear=$year/theMonth=$month/theDay=$day") == "theYear=%Y/theMonth=%m/theDay=%d"


def test_polars_private_api_still_exists():
    """The Polars credential provider still has the private method the cache uses."""
    cred = pl.CredentialProviderAWS(profile_name=None, **({} if POLARS_GE_2 else {"_storage_options_has_endpoint_url": False}))
    assert isinstance(cred._storage_update_options(), dict)


# --- Section: CacheMode as a pydantic field ---


class _Model(BaseModel):
    cache_mode: CacheMode = CacheMode.CACHE


@dataclass(frozen=True)
class CacheModeValidateCase:
    """A value, and the CacheMode ``CacheMode.validate`` makes of it or the exception it raises."""

    id: str
    _: KW_ONLY
    value: object
    expected: CacheMode | type[Exception]


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            CacheModeValidateCase("validate_from_string-CACHE", value="CACHE", expected=CacheMode.CACHE),
            CacheModeValidateCase("validate_from_string-IGNORE", value="IGNORE", expected=CacheMode.IGNORE),
            CacheModeValidateCase("validate_from_string-REBUILD", value="REBUILD", expected=CacheMode.REBUILD),
            CacheModeValidateCase("validate_from_int-1", value=1, expected=CacheMode.CACHE),
            CacheModeValidateCase("validate_from_int-2", value=2, expected=CacheMode.IGNORE),
            CacheModeValidateCase("validate_from_int-3", value=3, expected=CacheMode.REBUILD),
            CacheModeValidateCase("validate_from_enum-CACHE", value=CacheMode.CACHE, expected=CacheMode.CACHE),
            CacheModeValidateCase("validate_from_enum-REBUILD", value=CacheMode.REBUILD, expected=CacheMode.REBUILD),
            CacheModeValidateCase("validate_invalid_string", value="INVALID", expected=KeyError),
            CacheModeValidateCase("validate_invalid_int", value=999, expected=ValueError),
            CacheModeValidateCase("validate_invalid_type", value=3.14, expected=ValueError),
        ]
    ),
)
def test_cachemode_validate(case: CacheModeValidateCase):
    if isinstance(case.expected, type):
        with pytest.raises(case.expected, match="Cannot convert value to CacheMode" if isinstance(case.value, float) else None):
            CacheMode.validate(case.value)
    else:
        assert CacheMode.validate(case.value) == case.expected


@dataclass(frozen=True)
class CacheModeFieldCase:
    """A value given to a pydantic CacheMode field, and the CacheMode the field holds."""

    id: str
    _: KW_ONLY
    value: object
    expected: CacheMode


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            CacheModeFieldCase("from_string-REBUILD", value="REBUILD", expected=CacheMode.REBUILD),
            CacheModeFieldCase("from_string-CACHE", value="CACHE", expected=CacheMode.CACHE),
            CacheModeFieldCase("from_int-1", value=1, expected=CacheMode.CACHE),
            CacheModeFieldCase("from_int-3", value=3, expected=CacheMode.REBUILD),
            CacheModeFieldCase("from_enum", value=CacheMode.IGNORE, expected=CacheMode.IGNORE),
        ]
    ),
)
def test_cachemode_in_pydantic_model(case: CacheModeFieldCase):
    model = _Model(cache_mode=case.value)
    assert model.cache_mode == case.expected
    assert model.model_dump(mode="python") == {"cache_mode": case.expected}
    assert model.model_dump(mode="json") == {"cache_mode": case.expected.name}
    assert model.model_dump_json() == f'{{"cache_mode":"{case.expected.name}"}}'


def test_cachemode_pydantic_json_schema_and_default():
    props = _Model.model_json_schema()["properties"]["cache_mode"]
    assert "enum" in props or "$ref" in props
    assert _Model().cache_mode == CacheMode.CACHE


# --- Section: Layout: the files a write leaves and the metadata in them ---


@dataclass(frozen=True)
class LayoutCase:
    """Rows written with a time unit and a partition format, and the files the cache holds afterwards."""

    id: str
    _: KW_ONLY
    time_unit: str
    # None: the default format of ``time_unit``
    partition_format: str | None
    dates: list[datetime.date]
    expected_files: list[str]


_JAN_FEB_MAR_2023 = [datetime.date(2023, 1, 15), datetime.date(2023, 2, 20), datetime.date(2023, 3, 25)]
LAYOUTS = [
    LayoutCase("simple", time_unit="monthly", partition_format=None, dates=[datetime.date(2023, 1, 15)], expected_files=["monthly/2023-01.parquet"]),
    LayoutCase(
        "custom_partition_format-daily",
        time_unit="daily",
        partition_format="y=$year/m=$month/d=$day",
        dates=[datetime.date(2023, 1, 15)],
        expected_files=["daily/y=2023/m=01/d=15.parquet"],
    ),
    LayoutCase(
        "custom_partition_format-monthly",
        time_unit="monthly",
        partition_format="year=$year/month=$month",
        dates=[datetime.date(2023, 2, 15)],
        expected_files=["monthly/year=2023/month=02.parquet"],
    ),
    LayoutCase(
        "custom_partition_format-yearly",
        time_unit="yearly",
        partition_format="year=$year",
        dates=[datetime.date(2023, 3, 15)],
        expected_files=["yearly/year=2023.parquet"],
    ),
    LayoutCase(
        "multiple_partitions-monthly",
        time_unit="monthly",
        partition_format=None,
        dates=_JAN_FEB_MAR_2023,
        expected_files=["monthly/2023-01.parquet", "monthly/2023-02.parquet", "monthly/2023-03.parquet"],
    ),
    LayoutCase(
        "multiple_partitions-daily",
        time_unit="daily",
        partition_format=None,
        dates=_JAN_FEB_MAR_2023,
        expected_files=["daily/2023-01-15.parquet", "daily/2023-02-20.parquet", "daily/2023-03-25.parquet"],
    ),
    LayoutCase(
        "multiple_partitions-yearly", time_unit="yearly", partition_format=None, dates=_JAN_FEB_MAR_2023, expected_files=["yearly/2023.parquet"]
    ),
]
_DEFAULT_FORMAT = {"daily": "$year-$month-$day", "monthly": "$year-$month", "yearly": "$year"}


@pytest.mark.parametrize("case", as_params(LAYOUTS))
def test_layout(tmp_path, case: LayoutCase):
    source = _lazy_frame(date=case.dates, value=list(range(1, len(case.dates) + 1)))
    result = cache_parquet(source, tmp_path, "date", time_unit=case.time_unit, partition_format=case.partition_format)
    assert isinstance(result, pl.LazyFrame)
    assert_frame_equal(result.collect(), source.collect(), check_row_order=False)
    written = read_cache_files(tmp_path)
    assert sorted(written) == case.expected_files
    assert_frame_equal(pl.concat(written.values()), source.collect(), check_row_order=False)
    for path in case.expected_files:
        meta = {k.decode(): v.decode() for k, v in pq.read_metadata(tmp_path / path).metadata.items()}
        assert meta["__piot__time_unit"] == case.time_unit
        assert meta["__piot__partition_format"] == (case.partition_format or _DEFAULT_FORMAT[case.time_unit])


@dataclass(frozen=True)
class BadPartitionFormatCase:
    """Cache options that ``cache_parquet`` rejects, and the error message pattern."""

    id: str
    _: KW_ONLY
    kwargs: dict
    match: str | None


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            BadPartitionFormatCase("missing", kwargs={"partition_format": "$year"}, match="missing"),
            BadPartitionFormatCase("unexpected", kwargs={"partition_format": "$year-$month-$day"}, match="unexpected"),
            BadPartitionFormatCase("malformed", kwargs={"partition_format": "$year-${month"}, match=None),
            BadPartitionFormatCase("time_unit", kwargs={"time_unit": "occasionally"}, match="Invalid time unit"),
        ]
    ),
)
def test_bad_partition_format(tmp_path, case: BadPartitionFormatCase):
    with pytest.raises(ValueError, match=case.match):
        _lazy_frame(date=[datetime.date(2023, 1, 15)], value=[42]).piot.cache_parquet(
            cache_path=tmp_path, date_column="date", **({"time_unit": "monthly"} | case.kwargs)
        )


def test_streaming_query_caches_only_the_months_it_reads(tmp_path):
    x = pl.datetime_range(datetime.datetime(2023, 1, 1), datetime.datetime(2024, 12, 31), interval="1d", eager=True)
    lf = pl.LazyFrame({"date": x, "feature": np.random.rand(x.shape[0]), "quantity": range(len(x)), "price": 100 + np.arange(x.shape[0])})
    pred = (pl.col("date") < datetime.datetime(2024, 12, 15)) & (pl.col("date") > datetime.datetime(2024, 10, 15))
    query = lf.piot.cache_parquet(cache_path=str(tmp_path), date_column="date").filter(pred).filter(pl.col("quantity") % 2 != 0)
    assert query.select(["date", "feature", "price"]).collect(engine="streaming").shape == (30, 3)
    assert {p: f.shape for p, f in read_cache_files(tmp_path).items()} == {
        f"monthly/2024-{m}.parquet": (n, 4) for m, n in [("10", 31), ("11", 30), ("12", 31)]
    }


# --- Section: Sequences of reads on one cache (STEPS) ---

# Steps: a cache is driven by a sequence of reads; then the files it holds are checked.


@dataclass(frozen=True)
class Step:
    """One read: ``source`` through the cache in ``mode``, filtered by ``predicate``.

    With ``compare_with_polars`` the result equals plain Polars on the same source and predicate; otherwise its
    ``value`` column holds ``expected_values`` in any order.
    """

    source: Any
    _: KW_ONLY
    predicate: pl.Expr | None = None
    mode: CacheMode = CacheMode.CACHE
    compare_with_polars: bool = True
    expected_values: list | None = None

    def __post_init__(self):
        assert self.compare_with_polars == (self.expected_values is None), "set compare_with_polars=False with expected_values"


@dataclass(frozen=True)
class CacheFilesCase:
    """Reads on one cache in turn, and the files the cache holds afterwards (see ``assert_cache_files``)."""

    id: str
    _: KW_ONLY
    steps: list[Step]
    expected_files: dict[str, list | None]
    # no file exists beyond those listed with values
    only_expected_files: bool = False
    cache_kwargs: dict = field(default_factory=dict)


def run_steps(root, steps: list[Step], **kwargs) -> None:
    for i, step in enumerate(steps):
        context = f"step {i}: predicate {step.predicate}, mode {step.mode.name}"
        got = collect_filtered(cache_parquet(step.source, root, cache_mode=step.mode, **kwargs), step.predicate)
        if step.compare_with_polars:
            try:
                assert_frame_equal(got, collect_filtered(step.source, step.predicate), check_row_order=False)
            except AssertionError as exc:
                raise AssertionError(f"{context}: {exc}") from exc
        else:
            assert sorted(got["value"].to_list()) == sorted(step.expected_values), context


def _daily_rows(days, values=None, month=6, **extra):
    """One row per day of ``days`` in 2024 month ``month``; ``value`` counts up from 1 unless ``values`` is given."""
    days = [datetime.date(2024, month, d) for d in days]
    return _lazy_frame(date=days, value=values if values is not None else list(range(1, len(days) + 1)), **extra)


_DAILY_BY_REGION = {"time_unit": "daily", "extra_partition_cols": "region"}
_MAY_10_NA_AND_EU = _lazy_frame(date=[datetime.date(2024, 5, 10)] * 2, region=["NA", "EU"], value=[100, 200])
_MAY_10_NA_MAY_11_EU = _lazy_frame(date=[datetime.date(2024, 5, 10), datetime.date(2024, 5, 11)], region=["NA", "EU"], value=[100, 200])
_JUNE_1_2_US_JUNE_5_EU = _lazy_frame(
    date=[datetime.date(2024, 6, 1), datetime.date(2024, 6, 2), datetime.date(2024, 6, 5)], region=["US", "US", "EU"], value=[10, 20, 50]
)
_US_EU_OR_APAC = pl.col("region").is_in(["US", "EU", "APAC"])
_JAN_FEB_MAR_2024 = _lazy_frame(date=[datetime.date(2024, 1, 15), datetime.date(2024, 2, 15), datetime.date(2024, 3, 15)], value=[100, 200, 300])
_NO_ROWS = pl.LazyFrame(schema={"date": pl.Date, "value": pl.Int64})

STEPS = [
    # empty files mark queried partitions without rows
    CacheFilesCase(
        "missing_written",
        steps=[
            Step(
                _lazy_frame(date=[datetime.date(2023, 1, 15)], value=[1]),
                predicate=pl.col("date").is_between(datetime.date(2023, 1, 1), datetime.date(2023, 3, 31)),
            )
        ],
        expected_files={"monthly/2023-01.parquet": [1], "monthly/2023-02.parquet": [], "monthly/2023-03.parquet": []},
        only_expected_files=True,
    ),
    CacheFilesCase(
        "empty_partitions_with_extra_cols",
        steps=[Step(_MAY_10_NA_AND_EU, predicate=pl.col("region").is_in(["NA", "EU", "APAC"]) & (pl.col("date") == datetime.date(2024, 5, 10)))],
        expected_files={"daily/NA/2024-05-10.parquet": [100], "daily/EU/2024-05-10.parquet": [200], "daily/APAC/2024-05-10.parquet": []},
        only_expected_files=True,
        cache_kwargs=_DAILY_BY_REGION,
    ),
    CacheFilesCase(
        "missing_written_daily_with_extra_cols",
        steps=[
            Step(
                _MAY_10_NA_AND_EU,
                predicate=pl.col("region").is_in(["NA", "EU", "APAC"])
                & pl.col("date").is_between(datetime.date(2024, 5, 10), datetime.date(2024, 5, 11)),
            )
        ],
        expected_files={
            f"daily/{r}/2024-05-{d}.parquet": v
            for r, d, v in [("NA", 10, [100]), ("EU", 10, [200]), ("NA", 11, []), ("EU", 11, []), ("APAC", 10, []), ("APAC", 11, [])]
        },
        only_expected_files=True,
        cache_kwargs=_DAILY_BY_REGION,
    ),
    CacheFilesCase(
        "missing_written_daily_with_extra_cols_gap",
        steps=[
            Step(
                _MAY_10_NA_MAY_11_EU,
                predicate=pl.col("region").is_in(["NA", "EU", "APAC"])
                & pl.col("date").is_between(datetime.date(2024, 5, 10), datetime.date(2024, 5, 14))
                & (pl.col("date") != datetime.date(2024, 5, 11)),
            )
        ],
        expected_files={f"daily/{r}/2024-05-11.parquet": None for r in ["NA", "EU", "APAC"]}
        | {"daily/NA/2024-05-10.parquet": [100], "daily/APAC/2024-05-14.parquet": []},
        cache_kwargs=_DAILY_BY_REGION,
    ),
    CacheFilesCase(
        "missing_written_daily_with_extra_cols_gap2",
        steps=[
            Step(
                _MAY_10_NA_MAY_11_EU,
                predicate=pl.col("region").is_in(["EU", "NA"]) & pl.col("region").ne("NA") & pl.col("date").eq(datetime.date(2024, 5, 10)),
            )
        ],
        expected_files={"daily/NA/2024-05-10.parquet": None, "daily/EU/2024-05-10.parquet": []},
        only_expected_files=True,
        cache_kwargs=_DAILY_BY_REGION,
    ),
    # one-sided reads write whole partitions that have rows, and no empty files beyond the cached range
    CacheFilesCase(
        "one_sided_with_extra_cols_no_empty_writes",
        steps=[Step(_JUNE_1_2_US_JUNE_5_EU), Step(_JUNE_1_2_US_JUNE_5_EU, predicate=(pl.col("date") >= datetime.date(2024, 6, 2)) & _US_EU_OR_APAC)],
        expected_files={
            "daily/US/2024-06-02.parquet": [20],
            "daily/EU/2024-06-05.parquet": [50],
            "daily/APAC/2024-06-02.parquet": None,
            "daily/APAC/2024-06-05.parquet": None,
        },
        cache_kwargs=_DAILY_BY_REGION,
    ),
    CacheFilesCase(
        "one_sided_upper_bound_with_extra_cols",
        steps=[Step(_JUNE_1_2_US_JUNE_5_EU), Step(_JUNE_1_2_US_JUNE_5_EU, predicate=(pl.col("date") <= datetime.date(2024, 6, 2)) & _US_EU_OR_APAC)],
        expected_files={
            "daily/US/2024-06-01.parquet": [10],
            "daily/US/2024-06-02.parquet": [20],
            "daily/APAC/2024-06-01.parquet": None,
            "daily/APAC/2024-06-02.parquet": None,
        },
        cache_kwargs=_DAILY_BY_REGION,
    ),
    CacheFilesCase(
        "one_sided_lower_bound_reads_cache_and_upstream",
        steps=[
            Step(_daily_rows([1, 2])),
            Step(pl.concat([_daily_rows([5], [5]), _daily_rows([1, 2])]), predicate=pl.col("date") >= datetime.date(2024, 6, 2)),
        ],
        expected_files={"daily/2024-06-01.parquet": [1], "daily/2024-06-02.parquet": [2], "daily/2024-06-05.parquet": [5]},
        cache_kwargs={"time_unit": "daily"},
    ),
    CacheFilesCase(
        "unbounded_above_fills_gaps_up_to_existing_max",
        steps=[
            Step(_daily_rows([1, 3, 5], [1, 3, 5])),
            Step(_daily_rows([1, 3, 5], [1, 3, 5]), predicate=pl.col("date") >= datetime.date(2024, 6, 2)),
        ],
        expected_files={f"daily/2024-06-0{d}.parquet": v for d, v in [(1, [1]), (2, []), (3, [3]), (4, []), (5, [5]), (6, None), (7, None)]},
        only_expected_files=True,
        cache_kwargs={"time_unit": "daily"},
    ),
    CacheFilesCase(
        "unbounded_below_fills_gaps_down_to_existing_min",
        steps=[
            Step(_daily_rows([3, 5, 7], [3, 5, 7])),
            Step(_daily_rows([3, 5, 7], [3, 5, 7]), predicate=pl.col("date") <= datetime.date(2024, 6, 6)),
        ],
        expected_files={f"daily/2024-06-0{d}.parquet": v for d, v in [(1, None), (2, None), (3, [3]), (4, []), (5, [5]), (6, []), (7, [7])]},
        only_expected_files=True,
        cache_kwargs={"time_unit": "daily"},
    ),
    CacheFilesCase(
        "unbounded_with_extra_cols_fills_gaps",
        steps=[
            Step(_daily_rows([1, 3], [1, 3], region=["US", "US"])),
            Step(_daily_rows([1, 3], [1, 3], region=["US", "US"]), predicate=pl.col("date") >= datetime.date(2024, 6, 1)),
        ],
        expected_files={
            "daily/US/2024-06-01.parquet": [1],
            "daily/US/2024-06-02.parquet": [],
            "daily/US/2024-06-03.parquet": [3],
            "daily/US/2024-06-04.parquet": None,
        },
        only_expected_files=True,
        cache_kwargs=_DAILY_BY_REGION,
    ),
    # REBUILD reads upstream, overwrites the partitions in scope, and keeps the others
    CacheFilesCase(
        "overwrite",
        steps=[
            Step(_lazy_frame(date=[datetime.date(2023, 1, 15)], value=[42])),
            Step(_lazy_frame(date=[datetime.date(2023, 1, 15)], value=[99]), mode=CacheMode.REBUILD),
        ],
        expected_files={"monthly/2023-01.parquet": [99]},
        only_expected_files=True,
    ),
    CacheFilesCase(
        "rebuild_preserves_other_partitions",
        steps=[
            Step(_JAN_FEB_MAR_2024),
            Step(
                _lazy_frame(date=[datetime.date(2024, 2, 20)], value=[999]),
                predicate=pl.col("date").is_between(datetime.date(2024, 2, 1), datetime.date(2024, 2, 28)),
                mode=CacheMode.REBUILD,
            ),
        ],
        expected_files={"monthly/2024-01.parquet": [100], "monthly/2024-02.parquet": [999], "monthly/2024-03.parquet": [300]},
        only_expected_files=True,
        cache_kwargs={"time_unit": "monthly"},
    ),
    CacheFilesCase(
        "rebuild_overwrites_existing_partition",
        steps=[
            Step(_lazy_frame(date=[datetime.date(2024, 5, 10)], region=["US"], value=[100])),
            Step(
                _lazy_frame(date=[datetime.date(2024, 5, 10)], region=["US"], value=[999]),
                predicate=(pl.col("date") == datetime.date(2024, 5, 10)) & (pl.col("region") == "US"),
                mode=CacheMode.REBUILD,
            ),
        ],
        expected_files={"daily/US/2024-05-10.parquet": [999]},
        only_expected_files=True,
        cache_kwargs=_DAILY_BY_REGION,
    ),
    CacheFilesCase(
        "rebuild_writes_empty_files_for_missing_data",
        steps=[
            Step(
                _daily_rows([1, 3], [10, 30]),
                predicate=pl.col("date").is_between(datetime.date(2024, 6, 1), datetime.date(2024, 6, 3)),
                mode=CacheMode.REBUILD,
            )
        ],
        expected_files={"daily/2024-06-01.parquet": [10], "daily/2024-06-02.parquet": [], "daily/2024-06-03.parquet": [30]},
        only_expected_files=True,
        cache_kwargs={"time_unit": "daily"},
    ),
    CacheFilesCase(
        "rebuild_overwrites_previously_empty_partition",
        steps=[
            Step(_daily_rows([1, 3], [10, 30]), predicate=pl.col("date").is_between(datetime.date(2024, 6, 1), datetime.date(2024, 6, 3))),
            Step(
                _daily_rows([1, 2, 3], [10, 20, 30]),
                predicate=pl.col("date").is_between(datetime.date(2024, 6, 1), datetime.date(2024, 6, 3)),
                mode=CacheMode.REBUILD,
            ),
        ],
        expected_files={"daily/2024-06-01.parquet": [10], "daily/2024-06-02.parquet": [20], "daily/2024-06-03.parquet": [30]},
        only_expected_files=True,
        cache_kwargs={"time_unit": "daily"},
    ),
    CacheFilesCase(
        "rebuild_with_extra_cols_preserves_other_regions",
        steps=[
            Step(_lazy_frame(date=[datetime.date(2024, 8, 1)] * 2, region=["US", "EU"], value=[100, 200])),
            Step(
                _lazy_frame(date=[datetime.date(2024, 8, 1)], region=["US"], value=[999]),
                predicate=(pl.col("date") == datetime.date(2024, 8, 1)) & (pl.col("region") == "US"),
                mode=CacheMode.REBUILD,
            ),
        ],
        expected_files={"daily/US/2024-08-01.parquet": [999], "daily/EU/2024-08-01.parquet": [200]},
        only_expected_files=True,
        cache_kwargs=_DAILY_BY_REGION,
    ),
    CacheFilesCase(
        "rebuild_no_date_column",
        steps=[
            Step(_lazy_frame(region=["US", "EU"], value=[100, 200])),
            Step(_lazy_frame(region=["US"], value=[999]), predicate=pl.col("region") == "US", mode=CacheMode.REBUILD),
        ],
        expected_files={"null/US.parquet": [999], "null/EU.parquet": [200]},
        only_expected_files=True,
        cache_kwargs={"date_column": None, "extra_partition_cols": "region"},
    ),
    CacheFilesCase(
        "rebuild_empty_to_data",
        steps=[
            Step(_NO_ROWS, predicate=pl.col("date") == datetime.date(2024, 9, 1), mode=CacheMode.REBUILD),
            Step(
                _lazy_frame(date=[datetime.date(2024, 9, 1)], value=[42]),
                predicate=pl.col("date") == datetime.date(2024, 9, 1),
                mode=CacheMode.REBUILD,
            ),
        ],
        expected_files={"daily/2024-09-01.parquet": [42]},
        only_expected_files=True,
        cache_kwargs={"time_unit": "daily"},
    ),
    CacheFilesCase(
        "rebuild_data_to_empty",
        steps=[
            Step(_lazy_frame(date=[datetime.date(2024, 10, 1)], value=[42])),
            Step(_NO_ROWS, predicate=pl.col("date") == datetime.date(2024, 10, 1), mode=CacheMode.REBUILD),
        ],
        expected_files={"daily/2024-10-01.parquet": []},
        only_expected_files=True,
        cache_kwargs={"time_unit": "daily"},
    ),
    # reads served from the cache
    CacheFilesCase(
        "between_pred_uses_existing_partitions",
        steps=[
            Step(
                _lazy_frame(date=[datetime.date(2021, 3, 1), datetime.date(2021, 3, 2)], value=[10, 20]),
                predicate=pl.col("date") == datetime.date(2021, 3, 1),
            ),
            Step(
                _lazy_frame(date=[datetime.date(2021, 3, 1), datetime.date(2021, 3, 2)], value=[10, 20]),
                predicate=pl.col("date") == datetime.date(2021, 3, 2),
            ),
            Step(
                _lazy_frame(date=[datetime.date(2021, 3, 1), datetime.date(2021, 3, 2)], value=[0, 0]),
                predicate=pl.col("date").is_between(datetime.date(2021, 3, 1), datetime.date(2021, 3, 2)),
                compare_with_polars=False,
                expected_values=[10, 20],
            ),
        ],
        expected_files={"daily/2021-03-01.parquet": [10], "daily/2021-03-02.parquet": [20]},
        only_expected_files=True,
        cache_kwargs={"time_unit": "daily"},
    ),
    CacheFilesCase(
        "custom_partition_format_cache_readback",
        steps=[
            Step(_daily_rows([1, 2, 3], [10, 20, 30])),
            Step(_daily_rows([4, 5], [40, 50]), compare_with_polars=False, expected_values=[10, 20, 30, 40, 50]),
        ],
        expected_files={f"daily/theYear=2024/theMonth=06/theDay=0{d}.parquet": [10 * d] for d in range(1, 6)},
        only_expected_files=True,
        cache_kwargs={"time_unit": "daily", "partition_format": "theYear=$year/theMonth=$month/theDay=$day"},
    ),
    CacheFilesCase(
        "empty_result_from_cache",
        steps=[Step(_lazy_frame(date=[datetime.date(2024, 2, 1)], value=[1]), predicate=pl.col("date") == datetime.date(2025, 1, 1))],
        expected_files={"monthly/2025-01.parquet": []},
        only_expected_files=True,
    ),
    CacheFilesCase(
        "extra_partition_columns",
        steps=[Step(_lazy_frame(date=[datetime.date(2024, 1, 3), datetime.date(2024, 1, 4)] * 2, country=["US", "CA"] * 2, value=[1, 2, 3, 4]))],
        expected_files={"monthly/US/2024-01.parquet": [1, 3], "monthly/CA/2024-01.parquet": [2, 4]},
        only_expected_files=True,
        cache_kwargs={"extra_partition_cols": ["country"]},
    ),
    # no date column: one file per extra-column value under null/, or null/data.parquet
    CacheFilesCase(
        "cache_parquet_no_date_single_extra_col",
        steps=[
            Step(_lazy_frame(region=["US", "US", "EU", "EU", "APAC"], value=[100, 200, 150, 250, 180])),
            Step(_lazy_frame(region=["US"], value=[0]), compare_with_polars=False, expected_values=[100, 200, 150, 250, 180]),
        ],
        expected_files={"null/US.parquet": [100, 200], "null/EU.parquet": [150, 250], "null/APAC.parquet": [180]},
        only_expected_files=True,
        cache_kwargs={"date_column": None, "extra_partition_cols": "region"},
    ),
    CacheFilesCase(
        "cache_parquet_no_date_multiple_extra_cols",
        steps=[Step(_lazy_frame(region=["US", "US", "EU", "EU", "APAC", "APAC"], product=["A", "B"] * 3, value=[100, 200, 150, 250, 180, 220]))],
        expected_files={
            f"null/{r}/{p}.parquet": [v] for (r, p), v in zip([(r, p) for r in ["US", "EU", "APAC"] for p in "AB"], [100, 200, 150, 250, 180, 220])
        },
        only_expected_files=True,
        cache_kwargs={"date_column": None, "extra_partition_cols": ["region", "product"]},
    ),
    CacheFilesCase(
        "cache_parquet_no_date_no_extra_cols",
        steps=[
            Step(_lazy_frame(value=[100, 200, 150], name=["Alice", "Bob", "Charlie"])),
            Step(_lazy_frame(value=[0], name=["x"]), compare_with_polars=False, expected_values=[100, 200, 150]),
        ],
        expected_files={"null/data.parquet": [100, 200, 150]},
        only_expected_files=True,
        cache_kwargs={"date_column": None},
    ),
    CacheFilesCase(
        "cache_parquet_no_date_with_predicate",
        steps=[
            Step(_lazy_frame(region=["US", "US", "EU", "EU", "APAC"], category=list("XYXYX"), value=[100, 200, 150, 250, 180])),
            Step(
                _lazy_frame(region=["US", "US", "EU", "EU", "APAC"], category=list("XYXYX"), value=[100, 200, 150, 250, 180]).filter(
                    pl.col("region") == "US"
                ),
                compare_with_polars=False,
                expected_values=[100, 200, 150, 250, 180],
            ),
        ],
        expected_files={"null/US/X.parquet": [100], "null/APAC/X.parquet": [180]},
        cache_kwargs={"date_column": None, "extra_partition_cols": ["region", "category"]},
    ),
    CacheFilesCase(
        "cache_parquet_no_date_different_cache_modes",
        steps=[
            Step(_lazy_frame(category=list("ABAB"), value=[100, 200, 150, 250]), mode=CacheMode.IGNORE),
            Step(_lazy_frame(category=list("ABAB"), value=[100, 200, 150, 250]), mode=CacheMode.CACHE),
            Step(_lazy_frame(category=list("ABAB"), value=[100, 200, 150, 250]), mode=CacheMode.REBUILD),
        ],
        expected_files={"null/A.parquet": [100, 150], "null/B.parquet": [200, 250]},
        only_expected_files=True,
        cache_kwargs={"date_column": None, "extra_partition_cols": ["category"]},
    ),
    CacheFilesCase(
        "no_date_with_extra_unconstrained_queries_upstream_for_new_partitions",
        steps=[
            Step(_lazy_frame(region=["US"], value=[1])),
            Step(_lazy_frame(region=["EU"], value=[2]), compare_with_polars=False, expected_values=[1, 2]),
            Step(_lazy_frame(region=["EU"], value=[3]), compare_with_polars=False, expected_values=[1, 2]),
        ],
        expected_files={"null/US.parquet": [1], "null/EU.parquet": [2]},
        only_expected_files=True,
        cache_kwargs={"date_column": None, "extra_partition_cols": "region"},
    ),
    CacheFilesCase(
        "sequential_cache_parquet",
        steps=[Step(_lazy_frame(date=[datetime.date(2024, 1, 1), datetime.date(2024, 1, 2), datetime.date(2024, 2, 1)], value=[1, 3, 5]))],
        expected_files={"monthly/2024-01.parquet": [1, 3], "monthly/2024-02.parquet": [5]},
    ),
]


@pytest.mark.parametrize("case", as_params(STEPS))
def test_steps(tmp_path, case: CacheFilesCase):
    run_steps(tmp_path, case.steps, **{"date_column": "date"} | case.cache_kwargs)
    assert_cache_files(tmp_path, case.expected_files, exact=case.only_expected_files)


def test_sequential_cache_parquet_chained(tmp_path):
    base = _lazy_frame(date=[datetime.date(2024, 1, 1), datetime.date(2024, 1, 2), datetime.date(2024, 2, 1)], x=[1, 3, 5])
    assert base.piot.cache_parquet(tmp_path, "date").piot.cache_parquet(tmp_path, "date").collect().shape == (3, 2)


def test_cache_without_a_date_column_writes_null_time_unit_metadata(tmp_path):
    for name, extra, by in [("with_extra", ["region"], b"region"), ("single", None, b"")]:
        cache_parquet(_lazy_frame(region=["US", "EU"], value=[100, 200]), tmp_path / name, extra_partition_cols=extra).collect()
        for path in (tmp_path / name).rglob("*.parquet"):
            meta = pq.ParquetFile(path).metadata.metadata
            assert (meta[b"__piot__time_unit"], meta[b"__piot__partitioned_by"]) == (b"null", by)


# --- Section: Results match plain Polars in every mode and dtype ---

# What a read returns equals native Polars, cold and warm: every cache mode over the shared predicate shapes on a daily
# Date cache, and the default CACHE mode over two of those shapes on a monthly cache of each other temporal dtype.
# CACHE_ONLY returns nothing from an empty cache and serves a full one; IGNORE leaves no files.


@pytest.mark.parametrize("mode", [m for m in MODES if m != CacheMode.CACHE_ONLY], ids=lambda m: m.name)
@pytest.mark.parametrize("case", as_params(DATE_PREDICATES))
def test_mode_matches_native_by_predicate(tmp_path, mode, case: DatePredicateCase):
    assert_cache_matches_plain_polars(daily_frame().lazy(), tmp_path, case.predicate, date_column="date", time_unit="daily", cache_mode=mode)
    if mode == CacheMode.IGNORE:
        assert not read_cache_files(tmp_path)


@pytest.fixture(scope="module")
def full_daily_cache(tmp_path_factory):
    root = tmp_path_factory.mktemp("full_daily_cache")
    cache_parquet(daily_frame().lazy(), root, "date", time_unit="daily").collect()
    return root


@pytest.mark.parametrize("case", as_params(DATE_PREDICATES))
def test_cache_only_matches_native_by_predicate(tmp_path, full_daily_cache, case: DatePredicateCase):
    kwargs = {"date_column": "date", "time_unit": "daily", "cache_mode": CacheMode.CACHE_ONLY}
    assert collect_filtered(cache_parquet(daily_frame().lazy(), tmp_path, **kwargs), case.predicate).is_empty()
    assert_cache_matches_plain_polars(daily_frame().lazy(), full_daily_cache, case.predicate, **kwargs)


@pytest.mark.parametrize("dtype", [d for d in DATE_AND_DATETIME_DTYPES if d != pl.Date], ids=str)
@pytest.mark.parametrize("case", as_params(c for c in DATE_PREDICATES if c.id in ("between", "ne_in_window")))
def test_mode_matches_native_by_dtype(tmp_path, dtype, case: DatePredicateCase):
    assert_cache_matches_plain_polars(daily_frame(dtype=dtype).lazy(), tmp_path, case.predicate, date_column="date", time_unit="monthly")


# --- Section: Calls to the upstream source ---

# Upstream calls: what the cache asks its source for.


def test_callable_not_invoked_when_cache_satisfies(tmp_path):
    source = _lazy_frame(date=[datetime.date(2023, 1, 1), datetime.date(2023, 1, 2)], region=["US", "US"], value=[10, 20])
    kwargs = {"date_column": "date", "time_unit": "daily", "extra_partition_cols": ["region"]}
    cache_parquet(source, tmp_path, **kwargs).collect()
    calls = []

    def make_source():
        calls.append(1)
        return source

    covered = (pl.col("region") == "US") & pl.col("date").is_between(datetime.date(2023, 1, 1), datetime.date(2023, 1, 2))
    assert_cache_matches_plain_polars(source, tmp_path, covered, runs=1, **kwargs)
    assert sorted(cache_parquet(make_source, tmp_path, **kwargs).filter(covered).collect()["value"]) == [10, 20]
    assert calls == []
    # Without a date restriction new dates may exist upstream, so the source is built.
    assert sorted(cache_parquet(make_source, tmp_path, **kwargs).filter(pl.col("region") == "US").collect()["value"]) == [10, 20]
    assert calls == [1]


_MAY_31_TO_JUNE_6 = pl.DataFrame({"date": pl.date_range(datetime.date(2024, 5, 31), datetime.date(2024, 6, 6), eager=True)})


@dataclass(frozen=True)
class UpstreamCallsCase:
    """A seed cached first, then a read through the cache, and what the cache asks the upstream source for."""

    id: str
    _: KW_ONLY
    # cached first, if given
    seed: pl.DataFrame | None
    # the data of the upstream source
    upstream: pl.DataFrame
    predicate: pl.Expr | None
    # how many times the upstream source is read
    upstream_calls: int
    # the dates of ``_MAY_31_TO_JUNE_6`` that the filter the upstream source last received keeps
    upstream_keeps: list[datetime.date] | None = None
    # ``cache_parquet`` options on top of a daily cache on ``date``
    cache_options: dict = field(default_factory=dict)


_JUNE_1_2_3 = pl.DataFrame({"date": [datetime.date(2024, 6, 1), datetime.date(2024, 6, 2), datetime.date(2024, 6, 3)], "value": [1, 2, 3]})
_JUNE_3_4_5 = pl.DataFrame({"date": [datetime.date(2024, 6, 3), datetime.date(2024, 6, 4), datetime.date(2024, 6, 5)], "value": [3, 4, 5]})
UPSTREAM = [
    UpstreamCallsCase(
        "predicate_pushdown",
        seed=None,
        upstream=_JUNE_1_2_3,
        predicate=pl.col("date") == datetime.date(2024, 6, 1),
        upstream_calls=1,
        upstream_keeps=[datetime.date(2024, 6, 1)],
    ),
    UpstreamCallsCase(
        "write_uses_widened_predicate",
        seed=None,
        upstream=_JUNE_1_2_3,
        predicate=pl.col("date") == datetime.date(2024, 6, 15),
        upstream_calls=1,
        upstream_keeps=pl.date_range(datetime.date(2024, 6, 1), datetime.date(2024, 6, 6), eager=True).to_list(),
        cache_options={"time_unit": "monthly"},
    ),
    UpstreamCallsCase(
        "one_sided_lower_bound_upstream_predicate_shape",
        seed=_JUNE_1_2_3,
        upstream=pl.DataFrame({"date": [datetime.date(2024, 6, 5)], "value": [5]}),
        predicate=pl.col("date") >= datetime.date(2024, 6, 2),
        upstream_calls=1,
        upstream_keeps=[datetime.date(2024, 6, 4), datetime.date(2024, 6, 5), datetime.date(2024, 6, 6)],
    ),
    UpstreamCallsCase(
        "one_sided_lower_bound_upstream_contains_not_existing_predicate",
        seed=_JUNE_1_2_3.head(2),
        upstream=pl.DataFrame({"date": [datetime.date(2024, 6, 5)], "value": [5]}),
        predicate=pl.col("date") >= datetime.date(2024, 6, 2),
        upstream_calls=1,
        upstream_keeps=[datetime.date(2024, 6, 3), datetime.date(2024, 6, 4), datetime.date(2024, 6, 5), datetime.date(2024, 6, 6)],
    ),
    UpstreamCallsCase(
        "one_sided_upper_bound_upstream_predicate_shape",
        seed=_JUNE_3_4_5,
        upstream=pl.DataFrame({"date": [datetime.date(2024, 6, 2)], "value": [2]}),
        predicate=pl.col("date") <= datetime.date(2024, 6, 4),
        upstream_calls=1,
        upstream_keeps=[datetime.date(2024, 5, 31), datetime.date(2024, 6, 1), datetime.date(2024, 6, 2)],
    ),
    UpstreamCallsCase(
        "rebuild_queries_upstream_not_cache",
        seed=_JUNE_1_2_3,
        upstream=_JUNE_1_2_3.with_columns(value=pl.lit(9, pl.Int64)),
        predicate=pl.col("date") == datetime.date(2024, 6, 1),
        upstream_calls=1,
        upstream_keeps=[datetime.date(2024, 6, 1)],
        cache_options={"cache_mode": CacheMode.REBUILD},
    ),
    UpstreamCallsCase(
        "contradiction",
        seed=_JUNE_1_2_3,
        upstream=_JUNE_1_2_3,
        predicate=pl.col("date").is_between(datetime.date(2024, 6, 1), datetime.date(2024, 6, 2)),
        upstream_calls=0,
    ),
    UpstreamCallsCase(
        "no_date_no_extra_first_writes_second_reads_cache",
        seed=_JUNE_1_2_3,
        upstream=_JUNE_1_2_3,
        predicate=None,
        upstream_calls=0,
        cache_options={"date_column": None},
    ),
]


@pytest.mark.parametrize("case", as_params(UPSTREAM))
def test_upstream_calls(tmp_path, case: UpstreamCallsCase):
    cache_options = {"date_column": "date", "time_unit": "daily"} | case.cache_options
    if case.seed is not None:
        cache_parquet(case.seed.lazy(), tmp_path, **cache_options | {"cache_mode": CacheMode.CACHE}).collect()
    spy = RecordingSource(case.upstream)
    result = collect_filtered(cache_parquet(spy.lazy, tmp_path, **cache_options), case.predicate)
    assert spy.calls == case.upstream_calls
    if case.upstream_keeps is not None:
        assert spy.rows_kept_by_read(_MAY_31_TO_JUNE_6)["date"].to_list() == case.upstream_keeps
    # a seeded cache serves the seed's rows, unless REBUILD reads the upstream again
    if case.seed is None or cache_options.get("cache_mode") == CacheMode.REBUILD:
        assert_frame_equal(result, collect_filtered(case.upstream.lazy(), case.predicate), check_row_order=False)


def test_debug_source_sees_pushed_partition_predicates(tmp_path, capsys):
    """Through ``piot.debug`` upstream: a category predicate reaches the source, and a second unconstrained read of a
    no-date cache with partition columns still asks upstream for new partitions."""
    categories = _lazy_frame(
        date=[datetime.date(2025, 5, 1), datetime.date(2025, 5, 2), datetime.date(2025, 5, 3)], category=["a", "b", "a"], val=[10, 20, 30]
    ).piot.debug()
    categories.piot.cache_parquet(cache_path=tmp_path / "m", date_column="date", extra_partition_cols="category").filter(
        pl.col("category") == "a"
    ).collect()
    out = capsys.readouterr().out
    assert 'col("category")' in out and '"a"' in out

    cached = (
        _lazy_frame(region=["US", "EU", "US"], value=[1, 2, 3])
        .piot.debug()
        .piot.cache_parquet(cache_path=tmp_path / "r", extra_partition_cols="region")
    )
    first = cached.filter(pl.col("value") >= 2).sort(["region", "value"]).collect()
    assert first.to_dict(as_series=False) == {"region": ["EU", "US"], "value": [2, 3]}
    assert "predicate=" in capsys.readouterr().out
    assert_frame_equal(cached.filter(pl.col("value") >= 2).sort(["region", "value"]).collect(), first)
    assert "predicate=" in capsys.readouterr().out

    lf = _lazy_frame(value=[1, 2, 3]).piot.debug().piot.cache_parquet(cache_path=tmp_path / "n")
    lf.collect()
    assert "debug called" in capsys.readouterr().out
    lf.collect()
    assert "debug called" not in capsys.readouterr().out


def test_push_and_parse_exclusions_to_custom_io_source(tmp_path):
    """A wider second query asks upstream only for the region it has not cached."""
    from polars_io_tools.io_sources.range_visitor import convert_expr_to_datetime_range
    from polars_io_tools.io_sources.set_visitor import convert_expr_to_valid_values

    spy = RecordingSource(_MAY_10_NA_MAY_11_EU.collect())
    kwargs = {"date_column": "date", **_DAILY_BY_REGION}
    pred = (pl.col("region") == "NA") & (pl.col("date") == datetime.date(2024, 5, 10))
    assert_frame_equal(cache_parquet(spy.lazy, tmp_path, **kwargs).filter(pred).collect(), _MAY_10_NA_MAY_11_EU.filter(pred).collect())
    assert_cache_files(tmp_path, {"daily/NA/2024-05-10.parquet": [100]}, exact=True)
    wider = pl.col("region").is_in(["NA", "EU"]) & (pl.col("date") == datetime.date(2024, 5, 10))
    assert_frame_equal(cache_parquet(spy.lazy, tmp_path, **kwargs).filter(wider).collect(), _MAY_10_NA_MAY_11_EU.filter(wider).collect())
    pushed = spy.predicates[-1]
    assert {e.meta.serialize(format="json") for e in wider.meta.pop()} != {e.meta.serialize(format="json") for e in pushed.meta.pop()}
    assert convert_expr_to_datetime_range(pushed, "date", get_enclosure=False).lower == datetime.datetime(2024, 5, 10)
    assert convert_expr_to_valid_values(pushed, "region") == {"EU"}


@pytest.mark.parametrize("partition_format", [None, "theYear=$year/theMonth=$month/theDay=$day"], ids=["default_format", "custom_format"])
def test_open_ended_filters_query_expected_partitions(tmp_path, caplog, partition_format):
    import logging

    caplog.set_level(logging.DEBUG, logger="polars_io_tools.io_sources.lazy_cache_parquet")
    collected, appended, filtered = write_append_and_filter_daily_cache(cache_root=str(tmp_path), partition_format=partition_format)
    assert collected.to_dict(as_series=False) == {
        "date": [datetime.date(2024, 6, 1), datetime.date(2024, 6, 2), datetime.date(2024, 6, 3)],
        "value": [11, 22, 33],
    }
    assert appended.to_dict(as_series=False) == {"date": [datetime.date(2024, 6, 5)], "value": [44]}
    assert filtered.to_dict(as_series=False) == {"date": [datetime.date(2024, 6, 2), datetime.date(2024, 6, 3)], "value": [22, 33]}
    assert pl.concat(read_cache_files(tmp_path).values())["date"].sort().to_list() == [datetime.date(2024, 6, d) for d in (1, 2, 3, 5)]
    assert "Cache scan uses" in caplog.text and "enumerated" in caplog.text
    lower = cache_parquet(
        _lazy_frame(date=[datetime.date(2024, 6, 1), datetime.date(2024, 6, 2), datetime.date(2024, 6, 3)], value=[11, 22, 33]),
        tmp_path,
        "date",
        time_unit="daily",
        partition_format=partition_format,
    )
    assert lower.filter(pl.col("date") >= datetime.date(2024, 6, 2)).sort("date").collect()["value"].to_list() == [22, 33, 44]


@pytest.mark.parametrize("run_first", ["collect", "no_collect"])
def test_intermediate_collect(tmp_path, run_first):
    """An intermediate collect does not change what chained caches return or write."""

    def run(base, do_collect: bool):
        if base.exists():
            shutil.rmtree(base)
        ldf = _lazy_frame(a=[1, 2, 3], date=[datetime.date(2025, 7, 31)] * 3).piot.cache_parquet(base / "foo_test", "date", time_unit="daily")
        if do_collect:
            ldf.collect()
        ldf1 = ldf.with_columns(pl.lit("foo").alias("foo")).piot.cache_parquet(base / "foo_test1", "date", time_unit="daily")
        ldf2 = ldf.with_columns(pl.lit("bar").alias("bar")).piot.cache_parquet(base / "foo_test2", "date", time_unit="daily")
        return ldf1.join(ldf2, on="a").collect(), read_cache_files(base)

    order = [False, True] if run_first == "no_collect" else [True, False]
    (res_a, cache_a), (res_b, cache_b) = (run(tmp_path / f"scenario_{c}", c) for c in order)
    assert_frame_equal(res_a, res_b, check_row_order=False)
    assert set(cache_a) == set(cache_b)
    for k in cache_a:
        assert_frame_equal(cache_a[k], cache_b[k], check_row_order=False)


_REGION_DAYS = pl.LazyFrame({"date": pl.date_range(datetime.date(2020, 1, 1), datetime.date(2026, 1, 1), eager=True)}).with_columns(
    region=pl.col.date.dt.day().replace_strict({1: "NA", 2: "EU", 3: "CA", 4: "NA", 6: "NA"}, default="US", return_dtype=pl.Utf8), value=pl.lit(100)
)


@dataclass(frozen=True)
class ManyPartitionsQueryCase:
    """A partly cached query over many daily partitions, and what the log must show."""

    id: str
    _: KW_ONLY
    # the end of the first, cached query, from 2020-01-01
    small_end: datetime.date
    # the date range of the second query
    big: tuple[datetime.date, datetime.date]
    # text the debug log must hold; None checks nothing
    scan: str | None


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            ManyPartitionsQueryCase(
                "large_partition_query", small_end=datetime.date(2020, 6, 1), big=(datetime.date(2019, 12, 15), datetime.date(2020, 9, 14)), scan=None
            ),
            ManyPartitionsQueryCase(
                "huge_partition_query",
                small_end=datetime.date(2021, 2, 1),
                big=(datetime.date(2020, 1, 1), datetime.date(2024, 2, 1)),
                scan="Cache scan uses glob",
            ),
        ]
    ),
)
def test_partition_query_over_many_partitions(tmp_path, caplog, monkeypatch, case: ManyPartitionsQueryCase):
    """A partly cached query over many partitions returns every row; a huge one scans the cache by glob."""
    import logging

    # More partitions than the default open-sink cap (128).
    monkeypatch.setenv("POLARS_MAX_OPEN_SINKS", "8192")
    caplog.set_level(logging.DEBUG, logger="polars_io_tools.io_sources.lazy_cache_parquet")
    kwargs = {"date_column": "date", **_DAILY_BY_REGION}
    assert_cache_matches_plain_polars(
        _REGION_DAYS, tmp_path, pl.col.region.eq("US") & pl.col.date.is_between(datetime.date(2020, 1, 1), case.small_end), runs=1, **kwargs
    )
    assert_cache_matches_plain_polars(_REGION_DAYS, tmp_path, pl.col.region.is_in(["US", "NA"]) & pl.col.date.is_between(*case.big), runs=1, **kwargs)
    if case.scan:
        assert case.scan in caplog.text


def test_partition_count_exceeds_open_sinks_raises(tmp_path, monkeypatch):
    """Writing more partitions than the open-sink cap fails loud instead of dropping rows."""
    monkeypatch.setenv("POLARS_MAX_OPEN_SINKS", "4")
    df = _lazy_frame(date=pl.date_range(datetime.date(2020, 1, 1), datetime.date(2020, 1, 10), eager=True), value=range(10))
    with pytest.raises(Exception, match="open-sink cap"):
        cache_parquet(df, tmp_path, "date", time_unit="daily").filter(
            pl.col.date.is_between(datetime.date(2020, 1, 1), datetime.date(2020, 1, 10))
        ).collect()


def test_contradiction_writes_once(monkeypatch, tmp_path):
    """A repeated read of a cached range does not sink again."""
    base = daily_frame(31, datetime.date(2024, 1, 1)).lazy()
    pred = pl.col("date").is_between(datetime.date(2024, 1, 10), datetime.date(2024, 1, 20))
    sinks = []
    real_sink = pl.LazyFrame.sink_parquet
    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", lambda self, *a, **k: sinks.append(1) or real_sink(self, *a, **k))
    cached = cache_parquet(base, tmp_path, "date", time_unit="monthly")
    cached.filter(pred).collect()
    cached.filter(pred).collect()
    assert sinks == [1]


# --- Section: write_bounding_columns ---

# write_bounding_columns: an opt-in that writes only the rows the predicate keeps on the listed columns.

_BOUNDED_SOURCE = pl.DataFrame(
    {"date": [datetime.date(2023, 1, 10)] * 5, "x": [1, 2, 3, 4, 5], "y": [100, 200, 300, 400, 500], "value": [10, 20, 30, 40, 50]}
)


@dataclass(frozen=True)
class WriteBoundingCase:
    """A read with write bounding columns, filtered, and the rows that reach the result and the cache file."""

    id: str
    _: KW_ONLY
    predicate: pl.Expr
    # the ``x`` values of the rows the result holds
    expected_x: list
    # the ``x`` values in the cache file; None: no file is written; ``...``: not checked
    x_on_disk: list | None | EllipsisType
    mode: CacheMode = CacheMode.REBUILD
    # passed as ``write_bounding_columns``; empty or None turns the bounding off
    bounding_columns: tuple[str, ...] | None = ("x",)


WRITE_BOUNDING = [
    WriteBoundingCase("bounded_write_restricts_on_disk_rows", predicate=pl.col("x").is_in([1, 2]), expected_x=[1, 2], x_on_disk=[1, 2]),
    WriteBoundingCase(
        "unbounded_write_keeps_all_rows", predicate=pl.col("x").is_in([1, 2]), expected_x=[1, 2], x_on_disk=[1, 2, 3, 4, 5], bounding_columns=None
    ),
    WriteBoundingCase("bounded_and_unbounded_readback_identity", predicate=pl.col("x").is_in([2, 4]), expected_x=[2, 4], x_on_disk=[2, 4]),
    WriteBoundingCase(
        "cache_mode_allowed_with_write_bounding_columns",
        predicate=pl.col("x").is_in([1, 2]),
        expected_x=[1, 2],
        x_on_disk=[1, 2],
        mode=CacheMode.CACHE,
    ),
    WriteBoundingCase("rebuild_mode_allowed", predicate=pl.col("x").is_in([1]), expected_x=[1], x_on_disk=[1]),
    WriteBoundingCase("ignore_mode_passthrough", predicate=pl.col("x").is_in([1, 2]), expected_x=[1, 2], x_on_disk=None, mode=CacheMode.IGNORE),
    WriteBoundingCase(
        "predicate_not_restrictable_degrades_to_unbounded", predicate=pl.col("y").is_in([100, 200]), expected_x=[1, 2], x_on_disk=[1, 2, 3, 4, 5]
    ),
    WriteBoundingCase("empty_subset_pins_current_behavior", predicate=pl.col("x").is_in([]), expected_x=[], x_on_disk=None),
    # Whether an empty file is written depends on the Polars version.
    WriteBoundingCase("empty_subset_unbounded_is_clean", predicate=pl.col("x").is_in([]), expected_x=[], x_on_disk=..., bounding_columns=None),
]


@pytest.mark.parametrize("case", as_params(WRITE_BOUNDING))
def test_write_bounding_columns(tmp_path, case: WriteBoundingCase):
    bounding_columns = list(case.bounding_columns) if case.bounding_columns else None
    out = (
        cache_parquet(lambda: _BOUNDED_SOURCE.lazy(), tmp_path, "date", cache_mode=case.mode, write_bounding_columns=bounding_columns)
        .filter(case.predicate)
        .collect()
    )
    assert_frame_equal(out, _BOUNDED_SOURCE.filter(pl.col("x").is_in(case.expected_x)), check_row_order=False)
    files = read_cache_files(tmp_path)
    if case.x_on_disk is ...:
        return
    if case.x_on_disk is None:
        assert not files
    else:
        assert sorted(files) == ["monthly/2023-01.parquet"]
        assert sorted(files["monthly/2023-01.parquet"]["x"]) == case.x_on_disk


def test_write_bounding_columns_default_off_is_identical_to_not_passing(tmp_path):
    cache_parquet(lambda: _BOUNDED_SOURCE.lazy(), tmp_path / "omitted", "date").collect()
    cache_parquet(lambda: _BOUNDED_SOURCE.lazy(), tmp_path / "none", "date", write_bounding_columns=None).collect()
    omitted, none = read_cache_files(tmp_path / "omitted"), read_cache_files(tmp_path / "none")
    assert omitted and set(omitted) == set(none)
    for rel, frame in omitted.items():
        assert_frame_equal(frame, none[rel])


# --- Section: Filters that list no partitions, wildcards and REBUILD ---

# Non-enumerable partition predicates (!=, wildcards) on a no-date cache, and the CACHE-mode rule that existing
# partitions are never rewritten.

_SYMBOLS = pl.DataFrame(
    {"date": [datetime.date(2024, 1, 1)] * 5, "symbol": ["s0", "s1", "s2", "s3", "s4"], "region": ["a", "a", "b", "b", "b"], "v": [1, 2, 3, 4, 5]},
    schema={"date": pl.Date, "symbol": pl.String, "region": pl.String, "v": pl.Int64},
)
_NO_DATE_BY_SYMBOL_AND_REGION = {"date_column": None, "time_unit": "daily", "extra_partition_cols": ["symbol", "region"], "schema": _SYMBOLS.schema}


@pytest.mark.parametrize("populated", [True, False], ids=["populated_cache_read_correct", "incomplete_cache_populates_missing"])
@pytest.mark.parametrize(
    "pred",
    [
        pl.col("symbol") != "s0",
        pl.col("symbol").is_in(["s1", "s2", "s3", "s4"]),
        pl.col("v") > 2,
        (pl.col("symbol") != "s0") | (pl.col("symbol") == "s0"),
        (pl.col("symbol") != "s0") & (pl.col("region") == "b"),
        (pl.col("symbol") == "s0") & (pl.col("symbol") == "s1"),
    ],
    ids=["ne", "positive_control", "non_partition", "always_true_or", "partial_wildcard", "contradiction_returns_empty"],
)
def test_non_enumerable_predicate(tmp_path, populated, pred):
    if populated:
        cache_parquet(lambda: _SYMBOLS.lazy(), tmp_path, **_NO_DATE_BY_SYMBOL_AND_REGION).collect()
    assert_cache_matches_plain_polars(_SYMBOLS.lazy(), tmp_path, pred, runs=1, **_NO_DATE_BY_SYMBOL_AND_REGION)


def test_populated_cache_not_rewritten(tmp_path):
    """CACHE mode must not rewrite existing partition files on a filtered read."""
    cache_parquet(_SYMBOLS.lazy(), tmp_path, **_NO_DATE_BY_SYMBOL_AND_REGION).collect()
    _set_mtime(tmp_path, _utc_datetime(2026, 3, 1))
    before = _mtimes(tmp_path)
    for pred in (pl.col("v") > 2, pl.col("symbol") != "s0"):
        cache_parquet(_SYMBOLS.lazy(), tmp_path, **_NO_DATE_BY_SYMBOL_AND_REGION).filter(pred).collect()
    assert _mtimes(tmp_path) == before


def test_date_partitioned_negative_predicate_reads_cache(tmp_path):
    """On a populated date-partitioned cache, a != on an extra column reads the cache and not a changed upstream."""
    orig = _SYMBOLS.head(3).drop("region")
    state = {"df": orig}
    kwargs = {"date_column": "date", "time_unit": "daily", "extra_partition_cols": "symbol", "schema": orig.schema}
    cache_parquet(lambda: state["df"].lazy(), tmp_path, **kwargs).collect()
    state["df"] = orig.with_columns(pl.col("v") + 100)
    assert cache_parquet(lambda: state["df"].lazy(), tmp_path, **kwargs).filter(pl.col("symbol") != "s0").collect().sort("symbol")["v"].to_list() == [
        2,
        3,
    ]


_THREE_SYMBOLS = pl.DataFrame({"symbol": ["s0", "s1", "s2"], "v": [1, 2, 3]}, schema={"symbol": pl.String, "v": pl.Int64})


@dataclass(frozen=True)
class RebuildWildcardCase:
    """What upstream holds when REBUILD runs, and the values each symbol's cache file holds afterwards."""

    id: str
    _: KW_ONLY
    refreshed: pl.DataFrame
    expected: dict


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            RebuildWildcardCase(
                "rebuild_wildcard_predicate_leaves_out_of_scope_partition",
                refreshed=_THREE_SYMBOLS.with_columns(pl.col("v") + 100),
                expected={"s0": 1, "s1": 102, "s2": 103},
            ),
            RebuildWildcardCase(
                "rebuild_wildcard_removes_partition_that_disappeared_upstream",
                refreshed=pl.DataFrame({"symbol": ["s0", "s1"], "v": [1, 20]}),
                expected={"s0": 1, "s1": 20},
            ),
        ]
    ),
)
def test_rebuild_wildcard(tmp_path, case: RebuildWildcardCase):
    """REBUILD with ``symbol != 's0'`` refreshes in-scope partitions, drops those upstream lost, and leaves s0 alone."""
    state = {"df": _THREE_SYMBOLS}

    def mk(mode):
        return cache_parquet(lambda: state["df"].lazy(), tmp_path, extra_partition_cols="symbol", schema=_THREE_SYMBOLS.schema, cache_mode=mode)

    mk(CacheMode.CACHE).collect()
    state["df"] = case.refreshed
    mk(CacheMode.REBUILD).filter(pl.col("symbol") != "s0").collect()
    got = mk(CacheMode.CACHE).collect().sort("symbol")
    assert dict(zip(got["symbol"], got["v"])) == case.expected


def test_wildcard_read_matching_no_files_returns_empty(tmp_path):
    """A wildcard predicate that matches no cached files returns an empty frame, not an error."""
    data = _THREE_SYMBOLS.head(1)
    lf = cache_parquet(lambda: data.lazy(), tmp_path, extra_partition_cols="symbol", schema=data.schema)
    for pred in (pl.col("symbol") != "s0", pl.lit(False)):
        assert_frame_equal(lf.filter(pred).collect(), data.clear())


def test_rebuild_one_sided_empties_dropped_day_within_bounds(tmp_path):
    """Unlike a no-date cache, a dropped day inside the cached date bounds is kept as an empty file."""
    source = _daily_rows([1, 2, 3])
    cache_parquet(source, tmp_path, "date", time_unit="daily").collect()
    rebuilt = source.filter(pl.col("date") != datetime.date(2024, 6, 2))
    out = (
        cache_parquet(rebuilt, tmp_path, "date", time_unit="daily", cache_mode=CacheMode.REBUILD)
        .filter(pl.col("date") >= datetime.date(2024, 6, 1))
        .collect()
    )
    assert out.sort("date")["value"].to_list() == [1, 3]
    assert_cache_files(tmp_path, {"daily/2024-06-02.parquet": []})


# --- Section: Expiry (ttl and final_after) and refresh ---


def _utc_datetime(*args) -> datetime.datetime:
    return datetime.datetime(*args, tzinfo=UTC)


SEPTEMBER_1_2026 = datetime.date(2026, 9, 1)


def _daily_values_until(end: datetime.date, value: int = 1, start: datetime.date = SEPTEMBER_1_2026) -> pl.LazyFrame:
    dates = pl.date_range(start, end, eager=True)
    return pl.LazyFrame({"date": dates, "value": [value] * len(dates)})


def _set_mtime(cache: Path, when: datetime.datetime, pattern: str = "**/*.parquet") -> None:
    for path in cache.glob(pattern):
        os.utime(path, (when.timestamp(), when.timestamp()))


def _mtimes(cache: Path) -> dict[str, int]:
    return {p.relative_to(cache).as_posix(): p.stat().st_mtime_ns for p in cache.glob("**/*.parquet")}


@pytest.fixture
def set_now(monkeypatch):
    def setter(when: datetime.datetime) -> None:
        monkeypatch.setattr(lazy_cache_parquet, "_utcnow", lambda: when)

    return setter


OCTOBER_2026 = pl.col("date").is_between(datetime.date(2026, 10, 1), datetime.date(2026, 10, 31))


@dataclass(frozen=True)
class ExpiryCase:
    """When a file was written, its period, the time now and the expiry options, and whether ``_default_expired`` finds
    the file expired."""

    id: str
    _: KW_ONLY
    written_at: datetime.datetime
    period_end: datetime.datetime | None
    now: datetime.datetime
    ttl: timedelta | None
    final_after: timedelta | None
    expected: bool


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            ExpiryCase(
                "ttl_passed",
                written_at=_utc_datetime(2026, 10, 1),
                period_end=None,
                now=_utc_datetime(2026, 10, 2),
                ttl=timedelta(days=1),
                final_after=None,
                expected=True,
            ),
            ExpiryCase(
                "ttl_not_passed",
                written_at=_utc_datetime(2026, 10, 1),
                period_end=None,
                now=_utc_datetime(2026, 10, 1, 23),
                ttl=timedelta(days=1),
                final_after=None,
                expected=False,
            ),
            ExpiryCase(
                "written_after_period_was_final",
                written_at=_utc_datetime(2026, 11, 4),
                period_end=_utc_datetime(2026, 11, 1),
                now=_utc_datetime(2027, 1, 1),
                ttl=timedelta(hours=1),
                final_after=timedelta(days=3),
                expected=False,
            ),
            ExpiryCase(
                "written_before_period_was_final_ttl_passed",
                written_at=_utc_datetime(2026, 11, 3),
                period_end=_utc_datetime(2026, 11, 1),
                now=_utc_datetime(2026, 11, 3, 2),
                ttl=timedelta(hours=1),
                final_after=timedelta(days=3),
                expected=True,
            ),
            ExpiryCase(
                "no_ttl_written_before_period_was_final",
                written_at=_utc_datetime(2026, 11, 3),
                period_end=_utc_datetime(2026, 11, 1),
                now=_utc_datetime(2026, 11, 3),
                ttl=None,
                final_after=timedelta(days=3),
                expected=True,
            ),
            ExpiryCase(
                "no_ttl_no_period_end",
                written_at=_utc_datetime(2026, 11, 3),
                period_end=None,
                now=_utc_datetime(2026, 11, 3),
                ttl=None,
                final_after=timedelta(days=3),
                expected=True,
            ),
        ]
    ),
)
def test_default_expired(case: ExpiryCase):
    assert _default_expired(case.written_at, case.period_end, case.now, ttl=case.ttl, final_after=case.final_after) is case.expected


def test_partial_month_refreshes_until_settled(tmp_path, set_now):
    kwargs = {"time_unit": "monthly", "ttl": timedelta(hours=1), "final_after": timedelta(days=3)}
    set_now(_utc_datetime(2026, 10, 15, 12))
    assert cache_parquet(_daily_values_until(datetime.date(2026, 10, 15)), tmp_path, "date", **kwargs).filter(OCTOBER_2026).collect()[
        "date"
    ].max() == datetime.date(2026, 10, 15)
    _set_mtime(tmp_path, _utc_datetime(2026, 10, 15, 12))

    set_now(_utc_datetime(2026, 10, 15, 12, 30))
    assert cache_parquet(_daily_values_until(datetime.date(2026, 10, 17)), tmp_path, "date", **kwargs).filter(OCTOBER_2026).collect()[
        "date"
    ].max() == datetime.date(2026, 10, 15)

    set_now(_utc_datetime(2026, 10, 15, 14))
    assert cache_parquet(_daily_values_until(datetime.date(2026, 10, 17)), tmp_path, "date", **kwargs).filter(OCTOBER_2026).collect()[
        "date"
    ].max() == datetime.date(2026, 10, 17)

    # A rewrite three days after the month ends makes October permanent.
    _set_mtime(tmp_path, _utc_datetime(2026, 11, 4))
    set_now(_utc_datetime(2027, 1, 1))
    result = cache_parquet(_daily_values_until(datetime.date(2026, 10, 31), value=2), tmp_path, "date", **kwargs).filter(OCTOBER_2026).collect()
    assert result["date"].max() == datetime.date(2026, 10, 17)
    assert result["value"].unique().to_list() == [1]


def test_empty_future_partitions_refresh(tmp_path, set_now):
    kwargs = {"time_unit": "daily", "ttl": timedelta(hours=1), "final_after": timedelta(days=0)}
    window = pl.col("date").is_between(datetime.date(2026, 10, 15), datetime.date(2026, 10, 25))
    set_now(_utc_datetime(2026, 10, 19, 12))
    assert cache_parquet(_daily_values_until(datetime.date(2026, 10, 19)), tmp_path, "date", **kwargs).filter(window).collect()[
        "date"
    ].max() == datetime.date(2026, 10, 19)
    assert (tmp_path / "daily" / "2026-10-25.parquet").exists()
    _set_mtime(tmp_path, _utc_datetime(2026, 10, 19, 12))

    set_now(_utc_datetime(2026, 10, 23, 12))
    assert cache_parquet(_daily_values_until(datetime.date(2026, 10, 23)), tmp_path, "date", **kwargs).filter(window).collect()[
        "date"
    ].max() == datetime.date(2026, 10, 23)


def test_no_date_column_ttl(tmp_path, set_now):
    source = pl.LazyFrame({"value": [1, 2]})
    set_now(_utc_datetime(2026, 10, 19))
    cache_parquet(source, tmp_path, ttl=timedelta(days=1)).collect()
    _set_mtime(tmp_path, _utc_datetime(2026, 10, 19))

    set_now(_utc_datetime(2026, 10, 19, 12))
    assert cache_parquet(source.with_columns(value=pl.col("value") * 10), tmp_path, ttl=timedelta(days=1)).collect()["value"].to_list() == [1, 2]

    set_now(_utc_datetime(2026, 10, 20))
    assert cache_parquet(source.with_columns(value=pl.col("value") * 10), tmp_path, ttl=timedelta(days=1)).collect()["value"].to_list() == [10, 20]

    # Upstream returning no rows empties the expired file instead of serving its stale rows.
    _set_mtime(tmp_path, _utc_datetime(2026, 10, 19))
    assert cache_parquet(source.filter(pl.lit(False)), tmp_path, ttl=timedelta(days=1)).collect().is_empty()
    assert (tmp_path / "null" / "data.parquet").exists()

    # The empty file counts as cached, so it is not fetched again until it expires.
    _set_mtime(tmp_path, _utc_datetime(2026, 10, 19, 12))
    assert cache_parquet(source, tmp_path, ttl=timedelta(days=1)).collect().is_empty()


def test_refresh_is_limited_to_query_partitions(tmp_path, set_now):
    source = pl.LazyFrame(
        {
            "date": [datetime.date(2026, 1, 5), datetime.date(2026, 1, 6), datetime.date(2026, 2, 5), datetime.date(2026, 2, 6)],
            "sym": ["A", "B", "A", "B"],
            "value": [1, 1, 1, 1],
        }
    )
    kwargs = {"time_unit": "monthly", "extra_partition_cols": "sym", "ttl": timedelta(hours=1)}
    set_now(_utc_datetime(2026, 3, 1))
    cache_parquet(source, tmp_path, "date", **kwargs).filter(
        pl.col("date").is_between(datetime.date(2026, 1, 1), datetime.date(2026, 2, 28))
    ).collect()
    _set_mtime(tmp_path, _utc_datetime(2026, 3, 1))
    before = _mtimes(tmp_path)

    set_now(_utc_datetime(2026, 3, 2))
    jan = pl.col("date").is_between(datetime.date(2026, 1, 1), datetime.date(2026, 1, 31))
    feb = pl.col("date").is_between(datetime.date(2026, 2, 1), datetime.date(2026, 2, 28))
    predicate = (jan & (pl.col("sym") == "A")) | (feb & (pl.col("sym") == "B"))
    result = cache_parquet(source.with_columns(value=pl.lit(2, pl.Int64)), tmp_path, "date", **kwargs).filter(predicate).collect()
    assert result.sort("date")["value"].to_list() == [2, 2]
    after = _mtimes(tmp_path)
    assert {k for k in before if before[k] != after[k]} == {"monthly/A/2026-01.parquet", "monthly/B/2026-02.parquet"}


def test_refresh_keeps_null_partition_rows_outside_membership_scope(tmp_path, set_now, caplog):
    source = pl.LazyFrame(
        {
            "date": [datetime.date(2026, 1, 5), datetime.date(2026, 1, 6), datetime.date(2026, 2, 5)],
            "sym": ["A", None, "A"],
            "value": [1, 1, 1],
        }
    )
    kwargs = {"time_unit": "monthly", "extra_partition_cols": "sym", "ttl": timedelta(hours=1)}
    set_now(_utc_datetime(2026, 3, 1))
    cache_parquet(source, tmp_path, "date", **kwargs).filter(
        pl.col("date").is_between(datetime.date(2026, 1, 1), datetime.date(2026, 2, 28))
    ).collect()
    _set_mtime(tmp_path, _utc_datetime(2026, 3, 1))
    null_files = [k for k in _mtimes(tmp_path) if "/A/" not in k]
    assert null_files

    # The null partition is outside the query scope; refreshing it must not drop rows upstream still has.
    set_now(_utc_datetime(2026, 3, 2))
    for predicate in [pl.col("sym").is_in(["A"]), pl.col("sym") == "A", pl.col("sym") != "B"]:
        _set_mtime(tmp_path, _utc_datetime(2026, 3, 1))
        cache_parquet(source, tmp_path, "date", **kwargs).filter(predicate & (pl.col("date") < datetime.date(2026, 2, 1))).collect()
        assert pl.read_parquet(tmp_path / null_files[0]).height == 1

    # An expired file that upstream no longer fills is emptied with a warning.
    _set_mtime(tmp_path, _utc_datetime(2026, 3, 1))
    set_now(_utc_datetime(2026, 3, 2))
    cache_parquet(source.filter(pl.col("sym").is_not_null()), tmp_path, "date", **kwargs).collect()
    assert pl.read_parquet(tmp_path / null_files[0]).is_empty()
    assert any(r.levelname == "WARNING" and "expired cache file" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("predicate", [pl.col("sym").ne_missing("A"), ~pl.col("sym").eq_missing("A")], ids=["ne_missing", "not_eq_missing"])
def test_refresh_null_partition_matched_by_missing_aware_comparison(tmp_path, set_now, predicate):
    source = pl.LazyFrame({"date": [datetime.date(2026, 1, 5), datetime.date(2026, 1, 5)], "sym": ["B", None], "value": [1, 1]})
    kwargs = {"time_unit": "monthly", "extra_partition_cols": "sym", "ttl": timedelta(hours=1)}
    set_now(_utc_datetime(2026, 3, 1))
    cache_parquet(source, tmp_path, "date", **kwargs).collect()
    _set_mtime(tmp_path, _utc_datetime(2026, 3, 1))
    set_now(_utc_datetime(2026, 3, 2))
    jan = pl.col("date").is_between(datetime.date(2026, 1, 1), datetime.date(2026, 1, 31))
    out = cache_parquet(source.with_columns(value=pl.lit(2, pl.Int64)), tmp_path, "date", **kwargs).filter(jan & predicate).collect()
    assert out["value"].to_list() == [2, 2]


# --- Section: Open-sink cap, failed writes, write batches and listing ---


def test_unbounded_refresh_above_open_sink_cap(tmp_path, set_now, monkeypatch):
    kwargs = {"time_unit": "daily", "ttl": timedelta(hours=1)}
    set_now(_utc_datetime(2026, 9, 4))
    cache_parquet(_daily_values_until(datetime.date(2026, 9, 3)), tmp_path, "date", **kwargs).collect()
    _set_mtime(tmp_path, _utc_datetime(2026, 9, 4))

    monkeypatch.setattr(lazy_cache_parquet, "_max_open_sinks", lambda: 2)
    set_now(_utc_datetime(2026, 9, 5))
    assert cache_parquet(_daily_values_until(datetime.date(2026, 9, 3), value=2), tmp_path, "date", **kwargs).collect()["value"].to_list() == [
        2,
        2,
        2,
    ]


def test_bounded_refresh_above_open_sink_cap_raises_before_writing(tmp_path, set_now, monkeypatch):
    kwargs = {"time_unit": "daily", "ttl": timedelta(hours=1)}
    set_now(_utc_datetime(2026, 9, 4))
    cache_parquet(_daily_values_until(datetime.date(2026, 9, 3)), tmp_path, "date", **kwargs).collect()
    _set_mtime(tmp_path, _utc_datetime(2026, 9, 4))
    before = _mtimes(tmp_path)

    monkeypatch.setattr(lazy_cache_parquet, "_max_open_sinks", lambda: 2)
    set_now(_utc_datetime(2026, 9, 5))
    lf = cache_parquet(_daily_values_until(datetime.date(2026, 9, 3), value=2), tmp_path, "date", **kwargs).filter(
        pl.col("date").is_between(SEPTEMBER_1_2026, datetime.date(2026, 9, 3))
    )
    with pytest.raises(Exception, match="open-sink cap"):
        lf.collect()
    assert _mtimes(tmp_path) == before


@pytest.mark.parametrize("error", [RuntimeError("sink failed"), PanicException("sink panicked")], ids=["runtime_error", "panic"])
def test_failed_sink_deletes_partially_written_files(tmp_path, set_now, monkeypatch, error):
    # The second write's real mtime is long after October 2025 ends, so a leftover partial file would count as final.
    kwargs = {"time_unit": "monthly", "ttl": timedelta(hours=1), "final_after": timedelta(days=1)}
    source = _daily_values_until(datetime.date(2025, 10, 31), start=datetime.date(2025, 9, 1))
    cache_parquet(source, tmp_path, "date", **kwargs).collect()
    _set_mtime(tmp_path, _utc_datetime(2025, 10, 15))
    sep = tmp_path / "monthly" / "2025-09.parquet"
    sep_mtime = sep.stat().st_mtime_ns
    set_now(_utc_datetime(2025, 10, 16))

    real_sink = pl.LazyFrame.sink_parquet

    def partial_sink(self, target, **kwargs):
        real_sink(self.filter(pl.col("date").dt.day() <= 10), target, **kwargs)
        raise error

    with monkeypatch.context() as m:
        m.setattr(pl.LazyFrame, "sink_parquet", partial_sink)
        with pytest.raises(BaseException, match="sink"):
            cache_parquet(source.with_columns(value=pl.lit(2, pl.Int64)), tmp_path, "date", **kwargs).collect()
    assert not (tmp_path / "monthly" / "2025-10.parquet").exists()
    assert sep.stat().st_mtime_ns == sep_mtime

    oct_rows = (
        cache_parquet(source.with_columns(value=pl.lit(3, pl.Int64)), tmp_path, "date", **kwargs).collect().filter(pl.col("date").dt.month() == 10)
    )
    assert oct_rows.height == 31
    assert oct_rows["value"].unique().to_list() == [3]


@pytest.mark.parametrize("legacy", [False, True], ids=["partition_by", "legacy_partition_by_key"])
def test_sink_rejects_reopened_partition(tmp_path, monkeypatch, legacy):
    from types import SimpleNamespace

    providers = []

    def capture(**kwargs):
        providers.append(kwargs.get("file_path_provider") or kwargs.get("file_path"))

    monkeypatch.setattr(lazy_cache_parquet, "POLARS_HAS_PARTITION_BY", not legacy)
    monkeypatch.setattr(pl, "PartitionByKey" if legacy else "PartitionBy", capture, raising=False)

    def reopening_sink(self, target, **kwargs):
        if legacy:
            ctx = SimpleNamespace(keys=[SimpleNamespace(str_value="2026-09")])
        else:
            ctx = SimpleNamespace(partition_keys=pl.DataFrame({"k": ["2026-09"]}), index_in_partition=0)
        providers[0](ctx)
        providers[0](ctx)

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", reopening_sink)
    with pytest.raises(Exception, match="reopened a partition sink"):
        cache_parquet(_daily_values_until(datetime.date(2026, 9, 30)), tmp_path, "date").collect()


@pytest.mark.parametrize("syms", [[None], ["A", "B"]], ids=["null_symbol", "two_symbols"])
def test_write_batch_size_splits_sinks_below_open_sink_cap(tmp_path, monkeypatch, syms):
    monkeypatch.setattr(lazy_cache_parquet, "_max_open_sinks", lambda: 4)
    days = [d for d in pl.date_range(SEPTEMBER_1_2026, datetime.date(2026, 9, 6), eager=True) if d != datetime.date(2026, 9, 2)]
    source = pl.LazyFrame({"date": days}).join(pl.LazyFrame({"sym": syms}, schema={"sym": pl.String}), how="cross").with_columns(value=pl.lit(1))
    kwargs = {"time_unit": "daily", "extra_partition_cols": "sym" if syms != [None] else None}
    pred = pl.col("date").is_between(SEPTEMBER_1_2026, datetime.date(2026, 9, 6))
    if syms != [None]:
        pred &= pl.col("sym").is_in(syms)
    real_sink = pl.LazyFrame.sink_parquet
    sinks = []

    def counting_sink(self, target, **sink_kwargs):
        if not isinstance(target, (str, Path)):
            sinks.append(1)
        return real_sink(self, target, **sink_kwargs)

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", counting_sink)
    with pytest.raises(Exception, match="open-sink cap"):
        cache_parquet(source, tmp_path / "unbatched", "date", **kwargs).filter(pred).collect()
    out = cache_parquet(source, tmp_path / "batched", "date", write_batch_size=4, **kwargs).filter(pred).collect()
    assert_frame_equal(out.sort("date", "sym"), source.collect().sort("date", "sym"), check_column_order=False)
    assert len(sinks) == -(-6 * len(syms) // 4)


@pytest.mark.parametrize("refresh", [False, True], ids=["first_write", "refresh"])
def test_write_batch_size_keeps_batches_written_before_a_failure(tmp_path, set_now, monkeypatch, refresh):
    kwargs = {"time_unit": "daily", "ttl": timedelta(hours=1)}
    pred = pl.col("date").is_between(SEPTEMBER_1_2026, datetime.date(2026, 9, 6))
    if refresh:
        set_now(_utc_datetime(2026, 9, 7))
        cache_parquet(_daily_values_until(datetime.date(2026, 9, 6)), tmp_path, "date", **kwargs).filter(pred).collect()
        _set_mtime(tmp_path, _utc_datetime(2026, 9, 7))
        set_now(_utc_datetime(2026, 9, 8))
    real_sink = pl.LazyFrame.sink_parquet
    sinks = []

    def failing_second_sink(self, target, **sink_kwargs):
        if isinstance(target, (str, Path)):
            return real_sink(self, target, **sink_kwargs)
        sinks.append(1)
        if len(sinks) == 2:
            real_sink(self.head(1), target, **sink_kwargs)
            raise RuntimeError("sink failed")
        return real_sink(self, target, **sink_kwargs)

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", failing_second_sink)
    # 2026-09-02 is missing upstream, so batch one leaves an empty file for it.
    source = _daily_values_until(datetime.date(2026, 9, 6), value=2).filter(pl.col("date") != datetime.date(2026, 9, 2))
    with pytest.raises(Exception, match="sink failed"):
        cache_parquet(source, tmp_path, "date", write_batch_size=3, **kwargs).filter(pred).collect()
    # The failed batch deletes the file it opened; files it never opened keep their old data.
    expected = {"01", "02", "03"} | ({"05", "06"} if refresh else set())
    assert {p.stem[-2:] for p in (tmp_path / "daily").glob("*.parquet")} == expected
    assert pl.read_parquet(tmp_path / "daily" / "2026-09-02.parquet").is_empty()
    assert pl.read_parquet(tmp_path / "daily" / "2026-09-03.parquet")["value"].to_list() == [2]


def test_write_filter_selects_only_target_partitions(tmp_path):
    source = pl.LazyFrame(
        {
            "date": [SEPTEMBER_1_2026, SEPTEMBER_1_2026, datetime.date(2026, 9, 2), datetime.date(2026, 9, 2)],
            "sym": ["A", "B", "A", "B"],
            "value": [1, 2, 3, 4],
        }
    )
    pred = ((pl.col("date") == SEPTEMBER_1_2026) & (pl.col("sym") == "A")) | ((pl.col("date") == datetime.date(2026, 9, 2)) & (pl.col("sym") == "B"))
    out = cache_parquet(source, tmp_path, "date", time_unit="daily", extra_partition_cols="sym").filter(pred).collect()
    assert sorted(out["value"].to_list()) == [1, 4]
    assert len(list(tmp_path.glob("**/*.parquet"))) == 2


def test_write_filter_keeps_unrestricted_null_date_target(tmp_path):
    source = pl.LazyFrame({"date": [None, SEPTEMBER_1_2026], "sym": ["B", "A"], "value": [1, 2]}, schema_overrides={"date": pl.Date})
    pred = pl.col("date").is_null() | ((pl.col("date") == SEPTEMBER_1_2026) & (pl.col("sym") == "A"))
    out = cache_parquet(source, tmp_path, "date", time_unit="daily", extra_partition_cols="sym").filter(pred).collect()
    assert sorted(out["value"].to_list()) == [1, 2]


def test_listing_skips_file_deleted_after_glob(tmp_path, monkeypatch):
    import pyarrow.fs as pa_fs

    pl.DataFrame({"a": [1]}).write_parquet(tmp_path / "kept.parquet")
    real_glob = Path.glob
    monkeypatch.setattr(Path, "glob", lambda self, pattern: [*real_glob(self, pattern), self / "ghost.parquet"])
    mtimes = {}
    assert lazy_cache_parquet._extract_existing_files(pa_fs.LocalFileSystem(), str(tmp_path), mtimes=mtimes) == {"kept"}
    assert set(mtimes) == {"kept"}


# --- Section: More expiry and refresh: other cache modes, a callable ttl, null dates ---


def test_unbounded_query_refreshes_expired_latest_partition(tmp_path, set_now):
    kwargs = {"time_unit": "daily", "ttl": timedelta(hours=1), "final_after": timedelta(days=1)}
    set_now(_utc_datetime(2026, 10, 19, 12))
    cache_parquet(_daily_values_until(datetime.date(2026, 10, 19)), tmp_path, "date", **kwargs).filter(
        pl.col("date") >= datetime.date(2026, 10, 10)
    ).collect()
    _set_mtime(tmp_path, _utc_datetime(2026, 10, 21))
    _set_mtime(tmp_path, _utc_datetime(2026, 10, 19, 12), "daily/2026-10-19.parquet")

    set_now(_utc_datetime(2026, 10, 21, 1))
    result = (
        cache_parquet(_daily_values_until(datetime.date(2026, 10, 20), value=2), tmp_path, "date", **kwargs)
        .filter(pl.col("date") >= datetime.date(2026, 10, 10))
        .collect()
    )
    values = dict(zip(result["date"], result["value"], strict=True))
    assert values[datetime.date(2026, 10, 18)] == 1
    assert values[datetime.date(2026, 10, 19)] == 2
    assert values[datetime.date(2026, 10, 20)] == 2


def test_expired_empty_partition_still_bounds_gap_filling(tmp_path, set_now):
    kwargs = {"time_unit": "daily", "ttl": timedelta(hours=1)}
    source = pl.LazyFrame({"date": [SEPTEMBER_1_2026], "value": [1]})
    set_now(_utc_datetime(2026, 10, 19))
    for day in [SEPTEMBER_1_2026, datetime.date(2026, 9, 5)]:
        cache_parquet(source, tmp_path, "date", **kwargs).filter(pl.col("date") == day).collect()
    _set_mtime(tmp_path, _utc_datetime(2026, 10, 18))

    # Both files are expired and refreshed, but still bound the one-sided query: gaps up to the
    # empty 2026-09-05 file are filled, and nothing is written after it.
    result = cache_parquet(source, tmp_path, "date", **kwargs).filter(pl.col("date") >= SEPTEMBER_1_2026).collect()
    assert result["date"].to_list() == [SEPTEMBER_1_2026]
    assert sorted(_mtimes(tmp_path)) == [f"daily/2026-09-0{day}.parquet" for day in range(1, 6)]


@pytest.mark.parametrize("predicate", [OCTOBER_2026, None], ids=["october_2026", "no_filter"])
def test_read_only_reads_expired_partitions_from_upstream(tmp_path, set_now, predicate):
    kwargs = {"time_unit": "monthly", "ttl": timedelta(hours=1)}
    set_now(_utc_datetime(2026, 11, 1))
    cache_parquet(_daily_values_until(datetime.date(2026, 10, 31)), tmp_path, "date", **kwargs).collect()
    _set_mtime(tmp_path, _utc_datetime(2026, 11, 1))
    _set_mtime(tmp_path, _utc_datetime(2026, 10, 1), "monthly/2026-10.parquet")
    before = _mtimes(tmp_path)

    lf = cache_parquet(_daily_values_until(datetime.date(2026, 10, 31), value=2), tmp_path, "date", cache_mode=CacheMode.READ_ONLY, **kwargs)
    result = (lf.filter(predicate) if predicate is not None else lf).collect()
    oct_rows = result.filter(OCTOBER_2026)
    assert oct_rows.height == 31
    assert oct_rows["value"].unique().to_list() == [2]
    if predicate is None:
        assert result.filter(~OCTOBER_2026)["value"].unique().to_list() == [1]
    assert _mtimes(tmp_path) == before


def test_read_only_n_rows_skips_expired_files(tmp_path, set_now):
    kwargs = {"time_unit": "monthly", "ttl": timedelta(hours=1)}
    set_now(_utc_datetime(2026, 11, 1))
    cache_parquet(_daily_values_until(datetime.date(2026, 10, 31)), tmp_path, "date", **kwargs).collect()
    _set_mtime(tmp_path, _utc_datetime(2026, 11, 1))
    _set_mtime(tmp_path, _utc_datetime(2026, 10, 1), "monthly/2026-09.parquet")

    source = _daily_values_until(datetime.date(2026, 10, 31), value=2, start=datetime.date(2026, 10, 1))
    lf = cache_parquet(source, tmp_path, "date", cache_mode=CacheMode.READ_ONLY, read_kwargs={"n_rows": 1}, **kwargs)
    assert lf.collect()["date"].to_list() == [datetime.date(2026, 10, 1)]


def test_cache_only_serves_expired_files(tmp_path, set_now):
    kwargs = {"time_unit": "monthly", "ttl": timedelta(hours=1)}
    set_now(_utc_datetime(2026, 11, 1))
    cache_parquet(_daily_values_until(datetime.date(2026, 10, 31)), tmp_path, "date", **kwargs).filter(OCTOBER_2026).collect()
    _set_mtime(tmp_path, _utc_datetime(2026, 10, 1))
    result = (
        cache_parquet(_daily_values_until(datetime.date(2026, 10, 31), value=2), tmp_path, "date", cache_mode=CacheMode.CACHE_ONLY, **kwargs)
        .filter(OCTOBER_2026)
        .collect()
    )
    assert result["value"].unique().to_list() == [1]


def test_rebuild_does_not_call_ttl(tmp_path):
    def fail(*args):
        raise AssertionError("REBUILD must not evaluate expiry")

    cache_parquet(_daily_values_until(datetime.date(2026, 10, 31)), tmp_path, "date").filter(OCTOBER_2026).collect()
    result = (
        cache_parquet(_daily_values_until(datetime.date(2026, 10, 31), value=2), tmp_path, "date", ttl=fail, cache_mode=CacheMode.REBUILD)
        .filter(OCTOBER_2026)
        .collect()
    )
    assert result["value"].unique().to_list() == [2]


def test_callable_ttl_receives_local_period_end(tmp_path, set_now):
    calls = []

    def policy(written_at, period_end, now):
        calls.append((written_at.tzinfo, period_end, now))
        return False

    source = _daily_values_until(datetime.date(2026, 10, 31)).with_columns(pl.col("date").cast(pl.Datetime("us", "America/New_York")))
    set_now(_utc_datetime(2026, 11, 1))
    lf = cache_parquet(source, tmp_path, "date", ttl=policy).filter(pl.col("date").dt.month() == 10)
    lf.collect()
    lf.collect()
    assert {period_end for _, period_end, _ in calls} >= {_utc_datetime(2026, 11, 1, 4)}
    assert all(tz is UTC and now == _utc_datetime(2026, 11, 1) for tz, _, now in calls)


@pytest.mark.parametrize("dumps", [pickle.dumps, cloudpickle.dumps], ids=["pickle", "cloudpickle"])
def test_callable_ttl_pickles(tmp_path, dumps):
    lf = cache_parquet(_daily_values_until(datetime.date(2026, 10, 31)), tmp_path, "date", ttl=lambda written_at, period_end, now: False).filter(
        OCTOBER_2026
    )
    assert pickle.loads(dumps(lf)).collect().height == 31


@dataclass(frozen=True)
class InvalidExpiryCase:
    """Expiry options ``cache_parquet`` rejects, and the exception."""

    id: str
    _: KW_ONLY
    kwargs: dict
    error: type[Exception]


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            InvalidExpiryCase("final_after_without_date_column", kwargs={"date_column": None, "final_after": timedelta(days=1)}, error=ValueError),
            InvalidExpiryCase(
                "final_after_with_callable_ttl", kwargs={"ttl": lambda *args: True, "final_after": timedelta(days=1)}, error=ValueError
            ),
            InvalidExpiryCase("negative_ttl", kwargs={"ttl": timedelta(hours=-1)}, error=ValueError),
            InvalidExpiryCase("negative_final_after", kwargs={"final_after": timedelta(days=-1)}, error=ValueError),
            InvalidExpiryCase("integer_ttl", kwargs={"ttl": 3600}, error=TypeError),
        ]
    ),
)
def test_invalid_expiry_arguments(tmp_path, case: InvalidExpiryCase):
    kwargs = {"date_column": "date", **case.kwargs}
    with pytest.raises(case.error):
        cache_parquet(_daily_values_until(datetime.date(2026, 10, 31)), tmp_path, **kwargs)


def test_refresh_intersects_listed_dates_with_bounds(tmp_path, set_now):
    set_now(_utc_datetime(2026, 12, 1))
    cache_parquet(_daily_values_until(datetime.date(2026, 11, 30)), tmp_path, "date", ttl=timedelta(hours=1)).collect()
    _set_mtime(tmp_path, _utc_datetime(2026, 12, 1))
    set_now(_utc_datetime(2026, 12, 2))
    predicate = pl.col("date").is_in([datetime.date(2026, 10, 5), datetime.date(2026, 11, 5)]) & (pl.col("date") < datetime.date(2026, 11, 1))
    out = (
        cache_parquet(_daily_values_until(datetime.date(2026, 10, 31), value=2), tmp_path, "date", ttl=timedelta(hours=1)).filter(predicate).collect()
    )
    assert out["value"].to_list() == [2]
    assert pl.read_parquet(tmp_path / "monthly" / "2026-11.parquet")["value"].unique().to_list() == [1]


def test_refresh_null_partition_listed_in_is_in(tmp_path, set_now):
    source = pl.LazyFrame({"date": [datetime.date(2026, 1, 5), datetime.date(2026, 1, 5)], "sym": ["A", None], "value": [1, 1]})
    kwargs = {"time_unit": "monthly", "extra_partition_cols": "sym", "ttl": timedelta(hours=1)}
    set_now(_utc_datetime(2026, 3, 1))
    cache_parquet(source, tmp_path, "date", **kwargs).collect()
    _set_mtime(tmp_path, _utc_datetime(2026, 3, 1))
    set_now(_utc_datetime(2026, 3, 2))
    predicate = pl.col("sym").is_in(["A", None], nulls_equal=True) & (pl.col("date") < datetime.date(2026, 2, 1))
    out = cache_parquet(source.with_columns(value=pl.lit(2, pl.Int64)), tmp_path, "date", **kwargs).filter(predicate).collect()
    assert out["value"].to_list() == [2, 2]


@pytest.mark.parametrize("cache_mode", [CacheMode.CACHE, CacheMode.READ_ONLY])
@pytest.mark.parametrize("time_unit", ["daily", "monthly", "yearly"])
@pytest.mark.parametrize(
    "date_dtype", [pl.Date, pl.Datetime("us"), pl.Datetime("us", "America/New_York")], ids=["Date", "datetime_us", "datetime_us_new_york"]
)
@pytest.mark.parametrize("null_state", ["expired", "fresh", "removed"])
def test_refresh_nullable_date_membership_with_unbounded_or(tmp_path, set_now, cache_mode, time_unit, date_dtype, null_state):
    source = pl.DataFrame(
        {
            "date": [datetime.date(2024, 1, 10), None, datetime.date(2024, 3, 10), None],
            "sym": ["A", "A", "B", "B"],
            "value": [10, 20, 30, 40],
        }
    ).with_columns(pl.col("date").cast(date_dtype))
    kwargs = {"time_unit": time_unit, "extra_partition_cols": "sym"}
    cache_parquet(source.lazy(), tmp_path, "date", **kwargs).collect()
    _set_mtime(tmp_path, _utc_datetime(2026, 3, 1))
    if null_state == "fresh":
        for path in tmp_path.rglob("*.parquet"):
            if pl.read_parquet(path)["value"].to_list() == [20]:
                os.utime(path, (_utc_datetime(2026, 3, 2).timestamp(), _utc_datetime(2026, 3, 2).timestamp()))
    before = _mtimes(tmp_path)
    untouched = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*.parquet")
        if pl.read_parquet(path)["value"].to_list() in ([[20], [40]] if null_state == "fresh" else [[40]])
    }
    set_now(_utc_datetime(2026, 3, 2))
    predicate = (pl.col("date").is_in([source["date"][0], None], nulls_equal=True) & (pl.col("sym") == "A")) | (
        (pl.col("date") >= source["date"][2]) & (pl.col("sym") == "B")
    )
    upstream = source.with_columns(pl.col("value") + 100)
    if null_state == "removed":
        upstream = upstream.filter(pl.col("value") != 120)
    result = cache_parquet(upstream.lazy(), tmp_path, "date", ttl=timedelta(hours=1), cache_mode=cache_mode, **kwargs).filter(predicate).collect()
    expected = {"expired": [110, 120, 130], "fresh": [20, 110, 130], "removed": [110, 130]}[null_state]
    assert sorted(result["value"].to_list()) == expected
    persisted = cache_parquet(upstream.lazy(), tmp_path, "date", cache_mode=CacheMode.CACHE_ONLY, **kwargs).collect()
    assert sorted(persisted["value"].to_list()) == (sorted([40, *expected]) if cache_mode == CacheMode.CACHE else [10, 20, 30, 40])
    if cache_mode == CacheMode.READ_ONLY:
        assert _mtimes(tmp_path) == before
    for relative_path, contents in untouched.items():
        assert (tmp_path / relative_path).read_bytes() == contents
        assert (tmp_path / relative_path).stat().st_mtime_ns == before[relative_path]


@pytest.mark.parametrize("cache_mode", [CacheMode.CACHE, CacheMode.READ_ONLY])
def test_cold_cache_nullable_date_membership_with_unbounded_or(tmp_path, cache_mode):
    source = pl.LazyFrame({"date": [datetime.date(2024, 1, 10), None, datetime.date(2024, 3, 10)], "sym": ["A", "A", "B"], "value": [10, 20, 30]})
    predicate = (pl.col("date").is_in([datetime.date(2024, 1, 10), None], nulls_equal=True) & (pl.col("sym") == "A")) | (
        (pl.col("date") >= datetime.date(2024, 3, 10)) & (pl.col("sym") == "B")
    )
    result = cache_parquet(source, tmp_path, "date", extra_partition_cols="sym", cache_mode=cache_mode).filter(predicate).collect()
    assert sorted(result["value"].to_list()) == [10, 20, 30]


@pytest.mark.parametrize("engine", ["auto", "streaming"])
def test_read_only_unbounded_date_udf_rejecting_null(tmp_path, engine):
    source = pl.LazyFrame(
        {"date": [datetime.date(2024, 1, 10), datetime.date(2024, 3, 11), datetime.date(2024, 3, 16)], "sym": ["A", "B", "B"], "value": [1, 2, 3]}
    )

    def weekday(value):
        return value.weekday() < 5

    predicate = (pl.col("date") >= datetime.date(2024, 3, 1)) & pl.col("date").map_elements(weekday, return_dtype=pl.Boolean, skip_nulls=False)
    result = (
        cache_parquet(source, tmp_path, "date", extra_partition_cols="sym", cache_mode=CacheMode.READ_ONLY).filter(predicate).collect(engine=engine)
    )
    assert result["value"].to_list() == [2]


# --- Section: Retries of failed cache reads ---


@pytest.mark.parametrize(
    "error",
    [ComputeError("parquet: File out of specification"), OSError("range invalid"), PanicException("panicked")],
    ids=["compute_error", "os_error", "panic"],
)
def test_cache_read_retries_once_after_failure(tmp_path, monkeypatch, caplog, error):
    source = _daily_values_until(datetime.date(2026, 9, 30))
    cache_parquet(source, tmp_path, "date").collect()
    real = lazy_cache_parquet.collect_lf_in_io_source
    calls = []

    def flaky(lf, batch_size):
        calls.append(batch_size)
        if len(calls) == 1:
            raise error
        return real(lf, batch_size)

    monkeypatch.setattr(lazy_cache_parquet, "collect_lf_in_io_source", flaky)
    assert cache_parquet(source, tmp_path, "date").collect().height == 30
    assert len(calls) == 2
    assert any("attempt 1/2" in r.getMessage() for r in caplog.records)


def test_cache_read_retry_sees_replaced_file(tmp_path, monkeypatch):
    source = _daily_values_until(datetime.date(2026, 9, 30))
    cache_parquet(source, tmp_path, "date").collect()
    path = tmp_path / "monthly" / "2026-09.parquet"
    real = lazy_cache_parquet.collect_lf_in_io_source
    calls = []

    def replaced_mid_read(lf, batch_size):
        calls.append(lf)
        if len(calls) == 1:
            list(real(lf, batch_size))
            source.with_columns(value=pl.lit(2, pl.Int64)).collect().write_parquet(path.with_suffix(".tmp"), compression="uncompressed")
            os.replace(path.with_suffix(".tmp"), path)
            raise ComputeError("parquet: File out of specification")
        return real(lf, batch_size)

    monkeypatch.setattr(lazy_cache_parquet, "collect_lf_in_io_source", replaced_mid_read)
    assert cache_parquet(source, tmp_path, "date").collect()["value"].unique().to_list() == [2]


@dataclass(frozen=True)
class ReadRetriesCase:
    """A ``POLARS_IO_TOOLS_CACHE_READ_RETRIES`` setting, and how many times a failing cache read is tried."""

    id: str
    _: KW_ONLY
    # None leaves the variable unset
    retries: str | None
    attempts: int


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            ReadRetriesCase("unset_retries_once", retries=None, attempts=2),
            ReadRetriesCase("zero_no_retry", retries="0", attempts=1),
            ReadRetriesCase("three_retries", retries="3", attempts=4),
            ReadRetriesCase("malformed_retries_once", retries="bad", attempts=2),
        ]
    ),
)
def test_cache_read_raises_after_retries(tmp_path, monkeypatch, case: ReadRetriesCase):
    source = _daily_values_until(datetime.date(2026, 9, 30))
    cache_parquet(source, tmp_path, "date").collect()
    calls = []

    def broken(lf, batch_size):
        calls.append(batch_size)
        raise OSError("still broken")

    if case.retries is not None:
        monkeypatch.setenv("POLARS_IO_TOOLS_CACHE_READ_RETRIES", case.retries)
    monkeypatch.setattr(lazy_cache_parquet, "collect_lf_in_io_source", broken)
    with pytest.raises(Exception, match="still broken"):
        cache_parquet(source, tmp_path, "date").collect()
    assert len(calls) == case.attempts


def test_cache_read_not_retried_after_rows_returned(tmp_path, monkeypatch):
    source = _daily_values_until(datetime.date(2026, 9, 30))
    cache_parquet(source, tmp_path, "date").collect()
    real = lazy_cache_parquet.collect_lf_in_io_source
    calls = []

    def fails_after_first_batch(lf, batch_size):
        calls.append(batch_size)
        yield next(iter(real(lf, batch_size)))
        raise OSError("broken mid-read")

    monkeypatch.setattr(lazy_cache_parquet, "collect_lf_in_io_source", fails_after_first_batch)
    with pytest.raises(Exception, match="broken mid-read"):
        cache_parquet(source, tmp_path, "date").collect()
    assert len(calls) == 1


@dataclass(frozen=True)
class ReadOnlyRetryCase:
    """The state of the October partition of a READ_ONLY cache, and how many times a failing read of it is tried."""

    id: str
    _: KW_ONLY
    # "cached", "missing" (read from upstream) or "expired" (read from upstream)
    october: str
    attempts: int


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            ReadOnlyRetryCase("cached_partition_retried_once", october="cached", attempts=2),
            ReadOnlyRetryCase("missing_partition_not_retried", october="missing", attempts=1),
            ReadOnlyRetryCase("expired_partition_not_retried", october="expired", attempts=1),
        ]
    ),
)
def test_read_only_retries_only_cache_reads(tmp_path, monkeypatch, set_now, caplog, case: ReadOnlyRetryCase):
    kwargs = {"time_unit": "monthly", "ttl": timedelta(hours=1)}
    set_now(_utc_datetime(2026, 11, 1))
    cache_parquet(
        _daily_values_until(datetime.date(2026, 9, 30) if case.october == "missing" else datetime.date(2026, 10, 31)), tmp_path, "date", **kwargs
    ).collect()
    _set_mtime(tmp_path, _utc_datetime(2026, 11, 1))
    if case.october == "expired":
        _set_mtime(tmp_path, _utc_datetime(2026, 10, 1), "monthly/2026-10.parquet")
    calls = []

    def broken(lf, batch_size):
        calls.append(batch_size)
        raise OSError("still broken")

    monkeypatch.setattr(lazy_cache_parquet, "collect_lf_in_io_source", broken)
    lf = cache_parquet(_daily_values_until(datetime.date(2026, 10, 31), value=2), tmp_path, "date", cache_mode=CacheMode.READ_ONLY, **kwargs)
    with pytest.raises(Exception, match="still broken"):
        lf.filter(pl.col("date").is_between(SEPTEMBER_1_2026, datetime.date(2026, 10, 31))).collect()
    assert len(calls) == case.attempts
    assert any("attempt 1/2" in r.getMessage() for r in caplog.records) == (case.attempts == 2)


# --- Section: Write lock ---


def test_lock_taken_only_for_writes(tmp_path):
    names = []

    @contextlib.contextmanager
    def lock(name):
        names.append(name)
        yield

    predicate = pl.col("date").is_between(SEPTEMBER_1_2026, datetime.date(2026, 9, 30))
    for mode in (CacheMode.READ_ONLY, CacheMode.CACHE_ONLY, CacheMode.CACHE, CacheMode.CACHE):
        cache_parquet(_daily_values_until(datetime.date(2026, 9, 30)), tmp_path, "date", cache_mode=mode, lock=lock).filter(predicate).collect()
    assert names == [str((tmp_path / "monthly").resolve())]


def test_lock_plans_again_once_held(tmp_path):
    source = _daily_values_until(datetime.date(2026, 9, 30))

    @contextlib.contextmanager
    def lock(name):
        # Another process fills the cache while this one waits for the lock.
        cache_parquet(source.with_columns(value=pl.lit(2, pl.Int64)), tmp_path, "date").collect()
        yield

    predicate = pl.col("date").is_between(SEPTEMBER_1_2026, datetime.date(2026, 9, 30))
    out = cache_parquet(source, tmp_path, "date", lock=lock).filter(predicate).collect()
    assert out["value"].unique().to_list() == [2]


@pytest.mark.parametrize("fail", [False, True], ids=["write_succeeds", "write_fails"])
def test_lock_held_only_while_writing(tmp_path, monkeypatch, fail):
    held = False
    seen = []

    @contextlib.contextmanager
    def lock(name):
        nonlocal held
        held = True
        try:
            yield
        finally:
            held = False

    real_sink = pl.LazyFrame.sink_parquet
    real_collect = lazy_cache_parquet.collect_lf_in_io_source

    def sink(self, target, **kwargs):
        seen.append(("write", held))
        if fail:
            raise RuntimeError("sink failed")
        return real_sink(self, target, **kwargs)

    def collect(lf, batch_size):
        seen.append(("read", held))
        return real_collect(lf, batch_size)

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", sink)
    monkeypatch.setattr(lazy_cache_parquet, "collect_lf_in_io_source", collect)
    lf = cache_parquet(_daily_values_until(datetime.date(2026, 9, 30)), tmp_path, "date", lock=lock).filter(
        pl.col("date").is_between(SEPTEMBER_1_2026, datetime.date(2026, 9, 30))
    )
    if fail:
        with pytest.raises(ComputeError, match="sink failed"):
            lf.collect()
        assert seen == [("write", True)]
    else:
        lf.collect()
        assert seen == [("write", True), ("read", False)]
    assert not held


# --- Section: File visitor and write batches ---


def test_file_visitor_sees_each_written_file(tmp_path):
    events = []

    @contextlib.contextmanager
    def lock(name):
        events.append("lock")
        yield
        events.append("unlock")

    source = _daily_values_until(datetime.date(2026, 9, 30)).with_columns(sym=pl.lit("a%b"))
    predicate = pl.col("date").is_between(SEPTEMBER_1_2026, datetime.date(2026, 10, 31)) & (pl.col("sym") == "a%b")
    kwargs = {"extra_partition_cols": "sym", "lock": lock, "file_visitor": events.append}
    cache_parquet(source, tmp_path, "date", **kwargs).filter(predicate).collect()
    files = sorted(str(p) for p in (tmp_path / "monthly").resolve().rglob("*.parquet"))
    assert len(files) == 2
    assert events == ["lock", *files, "unlock"]
    events.clear()
    cache_parquet(source, tmp_path, "date", **kwargs).filter(predicate).collect()
    assert events == []


@pytest.mark.parametrize("present_keys", [[1, 2], [1], [2]], ids=["data-data", "data-empty", "empty-data"])
def test_write_batches_preserve_completed_physical_destinations(tmp_path, monkeypatch, present_keys):
    source = pl.LazyFrame({"key": present_keys, "value": [key * 10 for key in present_keys]})
    monkeypatch.setattr(lazy_cache_parquet, "_encode_partition_file_key", lambda key, windows: "collision")
    completed = {}

    def visitor(path):
        physical = Path(path)
        completed[path] = (physical.read_bytes(), physical.stat().st_mtime_ns)

    query = cache_parquet(source, tmp_path, extra_partition_cols="key", write_batch_size=1, file_visitor=visitor).filter(pl.col("key").is_in([1, 2]))
    with pytest.raises(Exception, match="completed partition path"):
        query.collect()
    assert len(completed) == 1
    files = list(tmp_path.rglob("*.parquet"))
    assert len(files) == 1
    for path, snapshot in completed.items():
        assert (Path(path).read_bytes(), Path(path).stat().st_mtime_ns) == snapshot
    assert pl.read_parquet(files[0])["value"].to_list() == ([10] if 1 in present_keys else [])


@pytest.mark.parametrize("layout", ["no-date", "default", "custom"])
def test_write_batches_reject_colliding_legacy_time_keys(tmp_path, layout):
    keys = [datetime.time(1, 2, 3, 100000), datetime.time(1, 2, 3, 200000)]
    source = pl.LazyFrame({"key": pl.Series(keys, dtype=pl.Time), "value": [1, 2]})
    predicate = pl.col("key").is_in(keys)
    options = {"extra_partition_cols": "key", "write_batch_size": 1}
    if layout != "no-date":
        source = source.with_columns(pl.lit(datetime.date(2024, 1, 10)).alias("date"))
        options.update(date_column="date", partition_format="$year/$month" if layout == "custom" else None)
        predicate &= pl.col("date") == datetime.date(2024, 1, 10)
    visited = []
    query = cache_parquet(source, tmp_path, file_visitor=visited.append, **options).filter(predicate)
    if lazy_cache_parquet.POLARS_HAS_PARTITION_BY:
        assert sorted(query.collect()["value"].to_list()) == [1, 2]
        assert len(visited) == 2
    else:
        with pytest.raises(Exception, match="completed partition path"):
            query.collect()
        assert len(visited) == 1
    persisted = pl.concat([pl.read_parquet(path) for path in tmp_path.rglob("*.parquet")])
    assert sorted(persisted["value"].to_list()) == ([1, 2] if lazy_cache_parquet.POLARS_HAS_PARTITION_BY else [1])


def test_write_batch_preserves_data_when_empty_destination_collides(tmp_path, monkeypatch):
    source = pl.LazyFrame({"key": [1, 3], "value": [10, 30]})
    monkeypatch.setattr(lazy_cache_parquet, "_encode_partition_file_key", lambda key, windows: "collision")
    visited = []
    query = cache_parquet(source, tmp_path, extra_partition_cols="key", write_batch_size=2, file_visitor=visited.append).filter(
        pl.col("key").is_in([1, 2, 3])
    )
    with pytest.raises(Exception, match="completed partition path"):
        query.collect()
    assert len(visited) == 1
    assert pl.read_parquet(visited[0])["value"].to_list() == [10]


def test_write_batch_checks_sink_cap_before_upstream_collection(tmp_path, monkeypatch):
    from polars_io_tools.io_sources.util import register_io_source_with_is_pure

    frame = pl.DataFrame({"key": [1, 2, 3, 4], "value": [10, 20, 30, 40]})
    calls = []

    def reader(with_columns, predicate, n_rows, batch_size):
        calls.append(1)
        yield frame.filter(predicate) if predicate is not None else frame

    monkeypatch.setattr(lazy_cache_parquet, "_max_open_sinks", lambda: 2)
    source = register_io_source_with_is_pure(reader, schema=frame.schema)
    query = cache_parquet(source, tmp_path, extra_partition_cols="key", write_batch_size=3).filter(pl.col("key").is_in([1, 2, 3, 4]))
    with pytest.raises(Exception, match="open-sink cap"):
        query.collect()
    assert calls == []
    assert list(tmp_path.rglob("*.parquet")) == []


# --- Section: Time partitions, and a cache over another cache ---


@pytest.mark.parametrize("literal_left", [False, True], ids=["column_left", "literal_left"])
@pytest.mark.parametrize("tick", [1, 999], ids=["1ns", "999ns"])
def test_time_partition_filter_preserves_nanoseconds(tmp_path, literal_left, tick):
    source = pl.DataFrame({"time": pl.Series([1, 999, 1000], dtype=pl.Int64).cast(pl.Time)}).lazy()
    literal = pl.lit(tick).cast(pl.Time)
    predicate = literal == pl.col("time") if literal_left else pl.col("time") == literal
    result = cache_parquet(source, tmp_path, extra_partition_cols="time").filter(predicate).collect()
    assert result["time"].cast(pl.Int64).to_list() == [tick]


def test_monthly_cache_over_daily_cache(tmp_path):
    calls = []

    def daily(end, cache_mode=CacheMode.CACHE):
        source = _daily_values_until(end)
        counted = source.map_batches(lambda df: calls.append(1) or df, schema=source.collect_schema())
        return cache_parquet(counted, tmp_path / "daily", "date", time_unit="daily", cache_mode=cache_mode)

    def monthly(upstream, cache_mode=CacheMode.CACHE):
        return cache_parquet(upstream, tmp_path / "monthly", "date", time_unit="monthly", cache_mode=cache_mode)

    def files(name):
        return len(list((tmp_path / name).rglob("*.parquet")))

    daily(datetime.date(2026, 10, 20)).filter(pl.col("date").is_between(SEPTEMBER_1_2026, datetime.date(2026, 10, 20))).collect()
    calls.clear()

    # Compacting September reads the daily cache only.
    september = (
        monthly(daily(datetime.date(2026, 10, 20), CacheMode.CACHE_ONLY))
        .filter(pl.col("date").is_between(SEPTEMBER_1_2026, datetime.date(2026, 9, 30)))
        .collect()
    )
    assert calls == []
    assert files("monthly") == 1
    assert_frame_equal(september, _daily_values_until(datetime.date(2026, 9, 30)).collect())

    # Reading through the monthly cache falls back to the daily cache for October, which fetches only the new days.
    query = pl.col("date").is_between(datetime.date(2026, 9, 10), datetime.date(2026, 10, 25))
    result = monthly(daily(datetime.date(2026, 10, 25)), CacheMode.READ_ONLY).filter(query).collect()
    assert_frame_equal(result.sort("date"), _daily_values_until(datetime.date(2026, 10, 25)).filter(query).collect())
    assert len(calls) == 1
    assert files("monthly") == 1
    assert files("daily") == 55


@pytest.mark.parametrize("cached_through", [datetime.date(2026, 9, 15), datetime.date(2026, 9, 30)], ids=["partial-daily", "complete-daily"])
def test_monthly_cache_retains_history_after_verified_finalization(tmp_path, set_now, cached_through):
    calls = []
    source = _daily_values_until(datetime.date(2026, 9, 30))
    expected = source.collect()
    counted = source.map_batches(lambda frame: calls.append(1) or frame, schema=source.collect_schema())
    history = pl.col("date").is_between(SEPTEMBER_1_2026, datetime.date(2026, 9, 30))
    daily_root, monthly_root = tmp_path / "daily", tmp_path / "monthly"

    def daily(cache_mode=CacheMode.CACHE):
        return cache_parquet(counted, daily_root, "date", time_unit="daily", cache_mode=cache_mode)

    def monthly():
        return cache_parquet(
            daily(CacheMode.CACHE_ONLY), monthly_root, "date", time_unit="monthly", ttl=timedelta(days=1), final_after=timedelta(days=1)
        )

    set_now(_utc_datetime(2026, 10, 1))
    daily().filter(pl.col("date") <= cached_through).collect()
    calls.clear()
    assert_frame_equal(daily().filter(history).collect().sort("date"), expected)
    assert len(calls) == (1 if cached_through < datetime.date(2026, 9, 30) else 0)
    calls.clear()
    assert_frame_equal(monthly().filter(history).collect().sort("date"), expected)
    _set_mtime(monthly_root, _utc_datetime(2026, 10, 1))
    before = _mtimes(monthly_root)

    set_now(_utc_datetime(2026, 10, 2))
    assert_frame_equal(monthly().filter(history).collect().sort("date"), expected)
    assert _mtimes(monthly_root) != before
    _set_mtime(monthly_root, _utc_datetime(2026, 10, 2))
    finalized = _mtimes(monthly_root)
    for path in daily_root.rglob("*.parquet"):
        path.unlink()

    set_now(_utc_datetime(2026, 10, 6))
    assert_frame_equal(monthly().filter(history).collect().sort("date"), expected)
    assert_frame_equal(pl.concat([pl.read_parquet(path) for path in monthly_root.rglob("*.parquet")]).sort("date"), expected)
    assert _mtimes(monthly_root) == finalized
    assert calls == []


# --- Section: Partition values no file can be named, and files outside the cache ---


def test_values_without_a_path_segment_of_their_own_are_read_from_upstream(tmp_path):
    src = pl.LazyFrame({"date": [datetime.date(2024, 1, 10)] * 6, "x": ["BRK/B", "None", "", "..", None, "ok"], "v": [1, 2, 3, 4, 5, 6]})
    jan = pl.col("date").is_between(datetime.date(2024, 1, 1), datetime.date(2024, 1, 31))
    for pred in [jan, jan & (pl.col("x") == "BRK/B"), jan & pl.col("x").is_in(["None", ""]), jan & pl.col("x").is_null(), jan]:
        out = cache_parquet(src, tmp_path, "date", extra_partition_cols="x").filter(pred).collect()
        assert_frame_equal(out, src.filter(pred).collect(), check_row_order=False)
    # Only the null and "ok" partitions are cached, and no other file is left behind.
    assert len(list(tmp_path.rglob("*.*"))) == 2


@pytest.mark.parametrize(
    "value",
    ["x" * 201, "%" * 100, "a.", "a ", "CON", "nul.txt", "Com1"],
    ids=["long", "long_when_percent_encoded", "dot", "space", "con", "nul_ext", "com1"],
)
def test_values_no_file_can_be_named_are_read_from_upstream(tmp_path, value):
    src = pl.LazyFrame({"date": [datetime.date(2024, 1, 10)] * 2, "x": [value, "ok"], "v": [1, 2]})
    for pred in [pl.lit(True), pl.col("x") == value]:
        out = cache_parquet(src, tmp_path, "date", extra_partition_cols="x").filter(pred).collect()
        assert_frame_equal(out, src.filter(pred).collect(), check_row_order=False)
    assert [p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.*")] == ["monthly/ok/2024-01.parquet"]


# Reading values no file can be named alongside cached files, with and without a date column.
DATE_COLUMN_OR_NONE = pytest.mark.parametrize("date_column", ["date", None], ids=["date", "no_date"])


@DATE_COLUMN_OR_NONE
def test_values_no_file_can_be_named_are_read_with_a_dropped_upstream_column(tmp_path, date_column):
    day = {"date": [datetime.date(2024, 1, 10)]}
    cache_parquet(pl.LazyFrame({**day, "x": ["ok"], "v": [1], "unused": [10]}), tmp_path, date_column, extra_partition_cols="x").collect()
    read_kwargs = {"missing_columns": "insert", "extra_columns": "ignore"}
    lf = cache_parquet(pl.LazyFrame({**day, "x": ["CON"], "v": [2]}), tmp_path, date_column, extra_partition_cols="x", read_kwargs=read_kwargs)
    assert sorted(lf.select("x", "v").collect().rows()) == [("CON", 2), ("ok", 1)]


def test_a_column_upstream_dropped_is_not_returned_with_a_declared_schema(tmp_path):
    frame = pl.DataFrame({"date": [datetime.date(2024, 1, 10)], "v": [1]})
    cache_parquet(frame.with_columns(dropped=pl.col("v")).lazy(), tmp_path, "date").collect()
    lf = cache_parquet(frame.lazy(), tmp_path, "date", schema=frame.schema, read_kwargs={"extra_columns": "ignore"})
    assert_frame_equal(lf.collect(), frame)


@DATE_COLUMN_OR_NONE
def test_n_rows_limits_cached_and_unnamed_rows_together(tmp_path, date_column):
    src = pl.LazyFrame({"date": [datetime.date(2024, 1, 10)] * 2, "x": ["CON", "ok"], "v": [7, 8]})
    assert cache_parquet(src, tmp_path, date_column, extra_partition_cols="x", read_kwargs={"n_rows": 1}).collect().height == 1


@dataclass(frozen=True)
class UnsafePartitionValueCase:
    """A nested or binary partition value, which ``cache_parquet`` must reject without writing or deleting a file."""

    id: str
    _: KW_ONLY
    value: object
    dtype: pl.DataType | type[pl.DataType]


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            UnsafePartitionValueCase("list", value=["../../victim"], dtype=pl.List(pl.String)),
            UnsafePartitionValueCase("binary", value=b"../../victim", dtype=pl.Binary),
            UnsafePartitionValueCase("struct", value={"a": "../../victim"}, dtype=pl.Struct({"a": pl.String})),
        ]
    ),
)
def test_a_nested_or_binary_partition_value_writes_and_deletes_nothing(tmp_path, case: UnsafePartitionValueCase):
    victim = tmp_path / "victim.parquet"
    victim.write_bytes(b"data")
    src = pl.LazyFrame({"date": [datetime.date(2024, 1, 10)], "x": [case.value], "v": [1]}, schema={"date": pl.Date, "x": case.dtype, "v": pl.Int64})
    with pytest.raises(ValueError, match="cannot partition on"):
        cache_parquet(src, tmp_path / "cache", "date", extra_partition_cols="x")
    assert [p.name for p in tmp_path.rglob("*")] == ["victim.parquet"]
    assert victim.read_bytes() == b"data"


def test_a_partition_path_outside_its_directory_is_refused(tmp_path):
    # The default monthly file of January 2024 sits next to the monthly directory.
    victim = tmp_path / "2024-01.parquet"
    victim.write_bytes(b"data")
    src = pl.LazyFrame({"date": [datetime.date(2024, 1, 10)], "v": [1]})
    with pytest.raises(Exception, match="refusing to write partition path '../2024-01'"):
        cache_parquet(src, tmp_path, "date", partition_format="../$year-$month").collect()
    assert victim.read_bytes() == b"data"


def test_cleanup_deletes_nothing_outside_the_cache_directory(tmp_path):
    from pyarrow.fs import LocalFileSystem

    from polars_io_tools.io_sources.lazy_cache_parquet import _delete_cache_file

    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "inside.parquet").write_bytes(b"data")
    (tmp_path / "victim.parquet").write_bytes(b"data")
    for path in ["../victim.parquet", "inside.parquet"]:
        _delete_cache_file(LocalFileSystem(), str(tmp_path / "cache"), path)
    assert [p.name for p in tmp_path.rglob("*.parquet")] == ["victim.parquet"]


@pytest.mark.parametrize(
    ("date_column", "legacy_file", "mode"),
    [(None, "null/a/b.parquet", CacheMode.CACHE), ("date", "monthly/a/b/2024-01.parquet", CacheMode.REBUILD)],
    ids=["no_date", "date"],
)
def test_files_with_names_that_are_not_cache_keys_are_not_read(tmp_path, date_column, legacy_file, mode):
    src = pl.LazyFrame({"date": [datetime.date(2024, 1, 10)] * 2, "x": ["a/b", "ok"], "v": [1, 2]})
    stale = tmp_path / legacy_file
    stale.parent.mkdir(parents=True)
    src.with_columns(v=pl.lit(0, pl.Int64)).collect().write_parquet(stale)
    out = cache_parquet(src, tmp_path, date_column, extra_partition_cols="x", cache_mode=mode).collect()
    assert_frame_equal(out, src.collect(), check_row_order=False)


def test_a_failed_temporary_file_deletion_ends_no_query(tmp_path, monkeypatch):
    import pyarrow.fs

    class UndeletableTemporaryFiles(pyarrow.fs.LocalFileSystem):
        def delete_file(self, path):
            if path.endswith(".tmp"):
                raise PermissionError(path)
            super().delete_file(path)

    def visitor(path):
        raise RuntimeError("visitor failed")

    monkeypatch.setattr(pyarrow.fs, "LocalFileSystem", UndeletableTemporaryFiles)
    src = pl.LazyFrame({"date": [datetime.date(2024, 1, 10)] * 2, "x": ["a/b", "ok"], "v": [1, 2]})
    assert_frame_equal(cache_parquet(src, tmp_path / "a", "date", extra_partition_cols="x").collect(), src.collect(), check_row_order=False)
    with pytest.raises(Exception, match="visitor failed"):
        cache_parquet(src, tmp_path / "b", "date", extra_partition_cols="x", file_visitor=visitor).collect()


def test_unnamed_partition_files_are_removed_when_the_file_visitor_raises(tmp_path):
    def visitor(path):
        raise RuntimeError("visitor failed")

    src = pl.LazyFrame({"date": [datetime.date(2024, 1, 10)] * 2, "x": ["a/b", "ok"], "v": [1, 2]})
    with pytest.raises(Exception, match="visitor failed"):
        cache_parquet(src, tmp_path, "date", extra_partition_cols="x", file_visitor=visitor).collect()
    assert not list(tmp_path.rglob("*.tmp"))


# --- Section: Property test, and the bugs it found ---


@pytest.mark.xfail(strict=True, reason="the day bounds of a nanosecond value near the Int64 minimum overflow")
def test_cache_parquet_on_a_nanosecond_date_column_near_its_minimum(tmp_path):
    frame = pl.DataFrame({"t": pl.Series([-(2**63), 0]).cast(pl.Datetime("ns")), "i": [0, 1]})
    pred = pl.col("i") >= 0
    assert_frame_equal(cache_parquet(frame.lazy(), tmp_path, "t", time_unit="daily").filter(pred).collect(), frame, check_row_order=False)


# Known bug, see test_cache_parquet_on_sub_second_time_keys: older Polars names Time partitions to the second.
_CACHE_PARQUET_POOLS = (
    PARTITION_KEY_POOLS
    if polars_has("time_partition_subsecond")
    else PARTITION_KEY_POOLS | {"time": pl.Series([0, 10**9, 61 * 10**9, None]).cast(pl.Time)}
)


@SETTINGS
@given(
    case=partitioned_frames_and_predicates(_CACHE_PARQUET_POOLS),
    time_unit=st.sampled_from(["daily", "monthly"]),
    dropped_column=st.booleans(),
    n_rows=st.integers(0, 3),
)
# A query that caches only null keys once broke later queries.
@example(
    case=(
        pl.DataFrame({"date": [datetime.date(2024, 1, 31), datetime.date(2024, 2, 1)], "x": [None, 1], "i": [0, 1]}),
        [pl.col("date") < datetime.date(2024, 2, 1), pl.col("x") == 1],
    ),
    time_unit="monthly",
    dropped_column=False,
    n_rows=0,
)
# A one-sided date query on a cache that holds only the null-date partition.
@example(
    case=(pl.DataFrame({"date": [None], "x": [1], "i": [0]}, schema_overrides={"date": pl.Date}), [pl.col("date") >= datetime.date(2024, 2, 1)]),
    time_unit="daily",
    dropped_column=False,
    n_rows=0,
)
def test_cache_parquet_keeps_the_rows_a_filter_keeps(tmp_path_factory, case, time_unit, dropped_column, n_rows):
    """``dropped_column`` first caches the frame with a column that upstream then drops, read with the schema options that allow it."""
    frame, predicates = case
    root = tmp_path_factory.mktemp("cache")
    options = {"time_unit": time_unit, "extra_partition_cols": "x", "read_kwargs": {"missing_columns": "insert", "extra_columns": "ignore"}}
    try:
        if dropped_column:
            cache_parquet(frame.with_columns(dropped=pl.col("i")).lazy(), root, "date", **options).collect()
            # Without a schema, the cache would take the dropped column from its files.
            options["schema"] = frame.schema
        for cache_mode in (CacheMode.CACHE, CacheMode.CACHE, CacheMode.READ_ONLY):
            check_cache_keeps_filtered_rows(cache_parquet(frame.lazy(), root, "date", cache_mode=cache_mode, **options), frame, predicates)
        limited = cache_parquet(frame.lazy(), root, "date", **{**options, "read_kwargs": {**options["read_kwargs"], "n_rows": n_rows}})
        assert limited.collect().height == min(n_rows, frame.height)
    finally:
        shutil.rmtree(root)


def test_cache_parquet_after_caching_only_null_partition_keys(tmp_path):
    frame = pl.DataFrame({"date": [datetime.date(2024, 1, 31), datetime.date(2024, 2, 1)], "x": [None, 1], "i": [0, 1]})
    lf = cache_parquet(frame.lazy(), tmp_path, "date", extra_partition_cols="x")
    for pred in [pl.col("date") < datetime.date(2024, 2, 1), pl.col("x") == 1]:
        assert_frame_equal(lf.filter(pred).collect(), frame.filter(pred))


@xfail_unless_polars_has("time_partition_subsecond", reason="Polars 1.37 names Time partitions to the second")
def test_cache_parquet_on_sub_second_time_keys(tmp_path):
    frame = pl.DataFrame({"date": [datetime.date(2024, 1, 31)] * 2, "x": pl.Series([0, 1000]).cast(pl.Time), "i": [0, 1]})
    lf = cache_parquet(frame.lazy(), tmp_path, "date", time_unit="daily", extra_partition_cols="x")
    for pred in [pl.col("x") == pl.lit(frame["x"].slice(0, 1)).first(), pl.lit(True)]:
        assert_frame_equal(lf.filter(pred).collect(), frame.filter(pred), check_row_order=False)


# --- Section: Threads and pickling ---


def test_cache_parquet_sink_preparation_python_scan_behavior(monkeypatch):
    """Sink preparation streams nested io_sources on fixed runtimes, materializes on old ones.

    On runtimes where nested streaming io_source collection is known-good
    (``POLARS_STREAMING_SUPPORTS_NESTED_IO_PLUGIN``, polars >= 1.43.2),
    ``_prepare_lf_for_sink_from_io_source`` must return the frame lazily (no eager collect of
    the source), so a large nested source streams into the sink instead of materializing. On
    older runtimes it must retain the previous behavior: materialize the source eagerly to
    dodge the partitioned-sink re-entrancy deadlock.
    """
    from polars_io_tools._compat import POLARS_STREAMING_SUPPORTS_NESTED_IO_PLUGIN

    calls = 0

    def reader(with_columns, predicate, n_rows, batch_size):
        nonlocal calls
        calls += 1
        yield pl.DataFrame({"a": [1]})

    monkeypatch.setattr(lazy_cache_parquet.pl, "thread_pool_size", lambda: 8)

    lf = register_io_source_with_is_pure(reader, schema={"a": pl.Int64})
    prepared = lazy_cache_parquet._prepare_lf_for_sink_from_io_source(lf)

    if POLARS_STREAMING_SUPPORTS_NESTED_IO_PLUGIN:
        # Streamed: the source is not collected during preparation.
        assert calls == 0
        assert prepared.collect().to_dict(as_series=False) == {"a": [1]}
    else:
        # Materialized: the source is collected eagerly during preparation.
        assert calls == 1
        assert prepared.collect().to_dict(as_series=False) == {"a": [1]}
        assert calls == 1


def test_cache_parquet_chained_write_does_not_deadlock_at_threads_one():
    """Regression test for chained ``cache_parquet`` writes under one Polars worker."""
    script = textwrap.dedent(
        """
        import datetime
        from tempfile import TemporaryDirectory

        import polars as pl
        import polars_io_tools  # noqa: F401

        assert pl.thread_pool_size() == 1, pl.thread_pool_size()

        def run(base, do_collect):
            src = pl.DataFrame(
                {
                    "a": [1, 2, 3],
                    "date": [datetime.date(2025, 7, 31)] * 3,
                }
            ).lazy()

            ldf = src.piot.cache_parquet(base / "foo_test", "date", time_unit="daily")
            if do_collect:
                ldf.collect()

            ldf1 = ldf.with_columns(pl.lit("foo").alias("foo")).piot.cache_parquet(base / "foo_test1", "date", time_unit="daily")
            ldf2 = ldf.with_columns(pl.lit("bar").alias("bar")).piot.cache_parquet(base / "foo_test2", "date", time_unit="daily")

            out = ldf1.join(ldf2, on="a").collect()
            assert out.height == 3, out

        with TemporaryDirectory() as tmpdir:
            from pathlib import Path

            base = Path(tmpdir)
            run(base / "scenario_with_collect", True)
            run(base / "scenario_no_collect", False)

        print("OK")
        """
    )
    try:
        result = run_in_subprocess(script)
    except subprocess.TimeoutExpired as e:
        pytest.fail(f"deadlock: subprocess did not finish in {SUBPROCESS_TIMEOUT_S}s\nstdout so far: {e.stdout!r}\nstderr so far: {e.stderr!r}")

    assert result.returncode == 0, f"subprocess failed: returncode={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    assert "OK" in result.stdout, result.stdout


class TestCacheParquetPickle:
    """Tests for cache_parquet pickle support."""

    def test_cache_parquet_pickle_basic(self, tmp_path):
        """cache_parquet LazyFrames can be pickled and unpickled."""
        cache_path = str(tmp_path / "cache")

        df = pl.DataFrame(
            {
                "date": [datetime.date(2025, 1, 1), datetime.date(2025, 1, 2), datetime.date(2025, 1, 3)],
                "value": [10, 20, 30],
            }
        )
        source_lf = df.lazy()

        # First collect to populate the cache
        lf = source_lf.piot.cache_parquet(
            cache_path=cache_path,
            date_column="date",
        )
        lf.collect()

        # Now create a new LazyFrame from the cache and test pickling
        lf2 = source_lf.piot.cache_parquet(
            cache_path=cache_path,
            date_column="date",
        )
        lf_unpickled = pickle_round_trip(lf2)

        result = lf_unpickled.collect()
        assert_frame_equal(result, df)


# --- Section: Regressions: the REGRESSIONS table ---

OLD_MTIME_NS = 1_000_000_000 * 10**9


def _age_files(root: Path) -> dict[Path, int]:
    """Set every cached file's mtime to a fixed past value and return the snapshot."""
    files = sorted(root.rglob("*.parquet"))
    assert files
    for f in files:
        os.utime(f, ns=(OLD_MTIME_NS, OLD_MTIME_NS))
    return {f: f.stat().st_mtime_ns for f in files}


def _rewritten(snapshot: dict[Path, int]) -> list[str]:
    return [str(f) for f, m in snapshot.items() if f.stat().st_mtime_ns != m]


def _sorted(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(df.columns)


# Two January rows and one February row.
JAN_JAN_FEB_SOURCE = pl.LazyFrame({"date": [datetime.date(2024, 1, 10), datetime.date(2024, 1, 15), datetime.date(2024, 2, 15)], "value": [0, 1, 2]})
_JANUARY_2024 = pl.col("date").is_between(datetime.date(2024, 1, 1), datetime.date(2024, 1, 31))


@dataclass(frozen=True)
class Read:
    """A cached read of ``predicate`` (``None`` reads everything) in cache ``mode``."""

    predicate: pl.Expr | None
    mode: CacheMode = CacheMode.CACHE


@dataclass(frozen=True)
class RegressionCase:
    """A cache filled by reading each filter in ``seed`` from ``seed_source`` (by default ``source``), then ``reads`` of
    ``source``.

    Each read runs ``runs`` times and must return what plain Polars returns. ``final_file_count`` is the number of cache
    files after the reads; ``cache_kwargs`` are passed to ``cache_parquet``.
    """

    id: str
    _: KW_ONLY
    source: Any
    reads: list[Read]
    seed: list = field(default_factory=list)
    seed_source: Any = None
    final_file_count: int | None = None
    runs: int = 1
    cache_kwargs: dict = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class NoRewriteCase(RegressionCase):
    """A ``RegressionCase`` whose seed files no read may rewrite or remove."""

    # the number of cache files after the seed
    seed_file_count: int | None = None
    # paths relative to the cache root, checked after the reads
    expected_files: tuple[str, ...] = ()
    absent_files: tuple[str, ...] = ()


_NULLABLE_DATES = pl.LazyFrame({"date": [datetime.date(2024, 1, 10), None, datetime.date(2024, 2, 5)], "x": ["a", "a", "a"], "value": [0, 1, 2]})
_NULLABLE_X = pl.LazyFrame(
    {"date": [datetime.date(2024, 1, 10), datetime.date(2024, 1, 11), datetime.date(2024, 1, 12)], "x": ["a", None, "b"], "value": [0, 1, 2]}
)
_MONTHLY_BY_X = {"time_unit": "monthly", "extra_partition_cols": "x"}
_TWO_SYMBOLS_IN_TWO_MONTHS = pl.LazyFrame(
    {
        "date": [datetime.date(2024, 1, 10), datetime.date(2024, 1, 10), datetime.date(2024, 2, 10), datetime.date(2024, 2, 10)],
        "sym": ["A", "B", "A", "B"],
        "value": [0, 1, 2, 3],
    }
)
_MONTHLY_BY_SYM = {"time_unit": "monthly", "extra_partition_cols": "sym"}
# Filters that treat null as a value (``eq_missing``, ``nulls_equal``).
_NULL_AWARE_FILTERS = {
    "ne-missing": pl.col("x").ne_missing("a"),
    "not-eq-missing": ~pl.col("x").eq_missing("a"),
    "in-null": pl.col("x").is_in(["b", None], nulls_equal=True),
    "date-ne-missing": _JANUARY_2024 & pl.col("x").ne_missing("a"),
    "date-in-null": _JANUARY_2024 & pl.col("x").is_in(["b", None], nulls_equal=True),
}
_X_A_B_AND_NULL = pl.LazyFrame({"date": [datetime.date(2024, 1, 10)] * 3, "x": ["a", "b", None], "value": [0, 1, 2]})
_NULL_DATE_AND_NULL_SYM = pl.LazyFrame(
    {"date": [datetime.date(2024, 1, 10), datetime.date(2024, 2, 10), None], "sym": ["A", None, "A"], "value": [1, 1, 1]}
)
_DATE_DTYPES = {"Date": pl.Date, "Datetime(us)": pl.Datetime("us"), "Datetime(us, NY)": pl.Datetime("us", "America/New_York")}

# Seeds and reads that once returned wrong rows.
REGRESSIONS = [
    # raised "truth value of an Expr is ambiguous"
    *(
        RegressionCase(f"one_sided_date_predicate_on_empty_cache-{k}", source=JAN_JAN_FEB_SOURCE, reads=[Read(p)])
        for k, p in [("le", pl.col("date") <= datetime.date(2024, 1, 31)), ("ge", pl.col("date") >= datetime.date(2024, 1, 1))]
    ),
    *(
        RegressionCase(
            f"not_equal_date_does_not_drop_enclosing_partition-{tu}",
            source=JAN_JAN_FEB_SOURCE.with_columns(pl.col("date").cast(dt)),
            reads=[Read(pl.col("date").is_between(datetime.date(2024, 1, 1), datetime.date(2024, 2, 29)) & (pl.col("date") != ex))],
            runs=2,
            cache_kwargs={"time_unit": tu},
        )
        for tu, dt, ex in [("monthly", pl.Date, datetime.date(2024, 1, 15)), ("daily", pl.Datetime("us"), datetime.datetime(2024, 1, 15, 12))]
    ),
    *(
        RegressionCase(
            f"null_or_value_predicate_keeps_null_rows-{k}-{'warm' if warm else 'cold'}",
            source=src,
            reads=[Read(p)],
            seed=[None] if warm else [],
            cache_kwargs=_MONTHLY_BY_X,
        )
        for k, src, p in [
            ("date_null_or_value", _NULLABLE_DATES, pl.col("date").is_null() | (pl.col("date") == datetime.date(2024, 1, 10))),
            ("date_null_or_range", _NULLABLE_DATES, pl.col("date").is_null() | _JANUARY_2024),
            (
                "date_null_or_value_with_x",
                _NULLABLE_DATES,
                (pl.col("x") == "a") & (pl.col("date").is_null() | (pl.col("date") == datetime.date(2024, 1, 10))),
            ),
            ("x_null_or_value", _NULLABLE_X, _JANUARY_2024 & (pl.col("x").is_null() | (pl.col("x") == "a"))),
            ("x_null_or_date", _NULLABLE_X, pl.col("x").is_null() | (pl.col("date") == datetime.date(2024, 1, 10))),
        ]
        for warm in (False, True)
    ),
    RegressionCase(
        "decimal_partition_finite_query_with_several_values",
        source=pl.LazyFrame(
            {
                "date": [datetime.date(2024, 1, 10), datetime.date(2024, 1, 11)],
                "x": pl.Series(["1.23", "2.34"]).cast(pl.Decimal(10, 2)),
                "value": [0, 1],
            }
        ),
        reads=[Read(_JANUARY_2024 & pl.col("x").is_in(pl.Series(["1.23", "2.34"]).cast(pl.Decimal(10, 2))))],
        cache_kwargs={"extra_partition_cols": "x"},
    ),
    *(
        RegressionCase(
            f"timezone_aware_finite_query_uses_local_partition_bounds-{tu}-{tz}",
            source=pl.LazyFrame(
                {
                    "ts": pl.Series(
                        [
                            datetime.datetime(2023, 12, 31, 23, 30),
                            datetime.datetime(2024, 1, 1, 0, 30),
                            datetime.datetime(2024, 1, 31, 23, 30),
                            datetime.datetime(2024, 2, 1, 0, 30),
                        ]
                    ).dt.replace_time_zone(tz),
                    "value": [0, 1, 2, 3],
                }
            ),
            reads=[
                Read(
                    pl.col("ts").is_between(
                        datetime.datetime(2024, 1, 1, tzinfo=ZoneInfo(tz)), datetime.datetime(2024, 1, 31, 23, 59, tzinfo=ZoneInfo(tz))
                    )
                )
            ],
            runs=2,
            cache_kwargs={"date_column": "ts", "time_unit": tu},
        )
        for tu in ["daily", "monthly"]
        for tz in ["America/New_York", "Asia/Tokyo"]
    ),
    *(
        RegressionCase(
            f"exclusive_datetime_bounds_keep_rest_of_bound_date-{tu}",
            source=pl.LazyFrame(
                {"ts": [datetime.datetime(2024, 1, 31, 13), datetime.datetime(2024, 2, 1, 13), datetime.datetime(2024, 2, 2, 11)], "value": [0, 1, 2]}
            ),
            reads=[Read((pl.col("ts") > datetime.datetime(2024, 1, 31, 12)) & (pl.col("ts") < datetime.datetime(2024, 2, 2, 12)))],
            runs=2,
            cache_kwargs={"date_column": "ts", "time_unit": tu},
        )
        for tu in ["daily", "monthly"]
    ),
    *(
        RegressionCase(
            f"one_sided_query_caches_whole_partitions-{op}-{k}",
            source=pl.LazyFrame(
                {
                    "ts": pl.Series([datetime.datetime(2024, 1, 5), datetime.datetime(2024, 1, 20), datetime.datetime(2024, 2, 5)]).cast(dt),
                    "value": [0, 1, 2],
                }
            ),
            reads=[
                Read(
                    pl.col("ts") >= pl.lit(datetime.datetime(2024, 1, 10)).cast(dt)
                    if op == "ge"
                    else pl.col("ts") <= pl.lit(datetime.datetime(2024, 1, 10)).cast(dt)
                ),
                Read(None),
            ],
            cache_kwargs={"date_column": "ts"},
        )
        for op in ["ge", "le"]
        for k, dt in _DATE_DTYPES.items()
    ),
    RegressionCase(
        "one_sided_query_writes_empty_files_only_for_queried_extra_values",
        source=pl.LazyFrame({"ts": [datetime.datetime(2024, 1, 10, 9), datetime.datetime(2024, 1, 21, 17)], "k": ["a", "a"], "value": [0, 1]}),
        reads=[
            Read(pl.col("ts").is_between(datetime.datetime(2024, 1, 21), datetime.datetime(2024, 1, 21, 23))),
            Read((pl.col("ts") > datetime.datetime(2024, 1, 1)) & (pl.col("k") == "b")),
            Read(None),
        ],
        cache_kwargs={"date_column": "ts", "time_unit": "daily", "extra_partition_cols": "k"},
    ),
    RegressionCase(
        "one_sided_exclusive_midnight_bound_does_not_mark_bound_date_cached",
        source=pl.LazyFrame({"ts": [datetime.datetime(2024, 1, 21, 17), datetime.datetime(2024, 2, 20, 6)], "value": [0, 1]}),
        reads=[
            Read(pl.col("ts").is_between(datetime.datetime(2024, 1, 21), datetime.datetime(2024, 1, 21, 23))),
            Read(pl.col("ts") < datetime.datetime(2024, 2, 20)),
            Read(None),
        ],
        cache_kwargs={"date_column": "ts", "time_unit": "daily"},
    ),
    RegressionCase(
        "or_with_one_sided_date_clause_reads_every_cached_match",
        source=pl.LazyFrame({"date": [datetime.date(2024, 1, 10), datetime.date(2024, 2, 10), datetime.date(2024, 3, 10)], "value": [1, 2, 3]}),
        reads=[Read((pl.col("date") == datetime.date(2024, 3, 10)) | (pl.col("date") < datetime.date(2024, 2, 15)))],
        seed=[None],
    ),
    # stepping past the partition that holds date.max overflowed
    *(
        RegressionCase(
            f"last_representable_date_partition-{mode.name}-{tu}",
            source=pl.LazyFrame({"date": [datetime.date.max, datetime.date(9999, 1, 5)], "value": [0, 1]}),
            reads=[Read(pl.col("date") == datetime.date.max, mode=mode)],
            runs=2,
            cache_kwargs={"time_unit": tu, "ttl": datetime.timedelta(days=1)},
        )
        for mode in [CacheMode.READ_ONLY, CacheMode.CACHE]
        for tu in ["daily", "monthly", "yearly"]
    ),
    *(
        RegressionCase(
            f"rebuild_datetime_scope_caches_whole_partition-{tz}",
            source=pl.LazyFrame(
                {
                    "ts": pl.Series(
                        [datetime.datetime(2024, 1, 5), datetime.datetime(2024, 1, 15), datetime.datetime(2024, 2, 5)], dtype=pl.Datetime("us", tz)
                    ),
                    "value": [0, 1, 2],
                }
            ),
            reads=[
                Read(
                    pl.col("ts").is_between(
                        datetime.datetime(2024, 1, 10, tzinfo=ZoneInfo(tz) if tz else None),
                        datetime.datetime(2024, 1, 20, tzinfo=ZoneInfo(tz) if tz else None),
                    ),
                    mode=CacheMode.REBUILD,
                ),
                Read(None),
            ],
            cache_kwargs={"date_column": "ts"},
        )
        for tz in [None, "America/New_York"]
    ),
    # REBUILD refreshes the partitions in scope from the current upstream
    *(
        RegressionCase(
            f"rebuild_missing_aware_predicate_refreshes_null_partition-{dc}-{k}",
            source=_X_A_B_AND_NULL.with_columns(pl.col("value") + 100),
            reads=[Read(p, mode=CacheMode.REBUILD)],
            seed=[None],
            seed_source=_X_A_B_AND_NULL,
            cache_kwargs={"date_column": dc, "extra_partition_cols": "x"},
        )
        for dc in [None, "date"]
        for k, p in _NULL_AWARE_FILTERS.items()
    ),
    *(
        RegressionCase(
            f"rebuild_unbounded_query_drops_rows_upstream_no_longer_has-{k}",
            source=_NULL_DATE_AND_NULL_SYM.clear(),
            reads=[Read(p, mode=CacheMode.REBUILD)],
            seed=[None],
            seed_source=_NULL_DATE_AND_NULL_SYM,
            cache_kwargs={"extra_partition_cols": "sym"},
        )
        for k, p in [
            ("unbounded", pl.col("date") >= datetime.date(2024, 1, 1)),
            ("sym_or_date", pl.col("sym").is_in(["A"]) | (pl.col("date") < datetime.date(2024, 1, 20))),
            ("null_or_date", pl.col("date").is_null() | (pl.col("date") < datetime.date(2024, 3, 1))),
        ]
    ),
    *(
        RegressionCase(
            f"rebuild_removes_stale_null_partition-{nc}",
            source=src.filter(pl.col("value") == 0),
            reads=[Read(None, mode=CacheMode.REBUILD)],
            seed=[None],
            seed_source=src,
            final_file_count=1,
            cache_kwargs={"extra_partition_cols": "x"},
        )
        for nc, src in [
            ("date", pl.LazyFrame({"date": [datetime.date(2024, 1, 10), None], "x": ["a", "a"], "value": [0, 1]})),
            ("x", pl.LazyFrame({"date": [datetime.date(2024, 1, 10), datetime.date(2024, 1, 11)], "x": ["a", None], "value": [0, 1]})),
        ]
    ),
]
# Seeds whose files no read may rewrite or remove.
NO_REWRITE_REGRESSIONS = [
    *(
        NoRewriteCase(
            f"cached_partitions_not_rewritten_by_open_queries-{tu}-{k}",
            source=JAN_JAN_FEB_SOURCE,
            reads=[Read(p)],
            seed=[_JANUARY_2024],
            expected_files=("monthly/2024-02.parquet",) if tu == "monthly" else (),
            cache_kwargs={"time_unit": tu},
        )
        for tu in ["monthly", "yearly"]
        for k, p in [("unbounded", pl.col("date") >= datetime.date(2024, 1, 1)), ("unconstrained", None)]
    ),
    NoRewriteCase(
        "daily_datetime_column_partitions_not_rewritten",
        source=pl.LazyFrame(
            {"date": [datetime.datetime(2024, 1, 1, 9), datetime.datetime(2024, 1, 1, 15), datetime.datetime(2024, 1, 2, 9)], "value": [0, 1, 2]}
        ),
        reads=[Read(pl.col("date") >= datetime.datetime(2024, 1, 1))],
        seed=[pl.col("date").is_between(datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 1, 23, 59))],
        cache_kwargs={"time_unit": "daily"},
    ),
    NoRewriteCase(
        "finite_query_does_not_rewrite_cached_partitions_between_missing_ones",
        source=_TWO_SYMBOLS_IN_TWO_MONTHS,
        reads=[Read(pl.col("date").is_between(datetime.date(2024, 1, 1), datetime.date(2024, 2, 29)) & pl.col("sym").is_in(["A", "B"]))],
        seed=[
            (pl.col("date") == datetime.date(2024, 1, 10)) & (pl.col("sym") == "B"),
            (pl.col("date") == datetime.date(2024, 2, 10)) & (pl.col("sym") == "A"),
        ],
        cache_kwargs=_MONTHLY_BY_SYM,
    ),
    NoRewriteCase(
        "extra_col_only_predicate_does_not_rewrite_cached_dates",
        source=_TWO_SYMBOLS_IN_TWO_MONTHS,
        reads=[Read(pl.col("sym") == "A")],
        seed=[_JANUARY_2024],
        expected_files=("monthly/A/2024-02.parquet",),
        absent_files=("monthly/B/2024-02.parquet",),
        cache_kwargs=_MONTHLY_BY_SYM,
    ),
    *(
        NoRewriteCase(
            f"custom_partition_format_cache_hit-{fmt}",
            source=JAN_JAN_FEB_SOURCE,
            reads=[Read(_JANUARY_2024)],
            seed=[_JANUARY_2024],
            expected_files=(f"monthly/{jan}",),
            cache_kwargs={"partition_format": fmt},
        )
        for fmt, jan in [
            ("$year/$month", "2024/01.parquet"),
            ("prefix_$year-$month", "prefix_2024-01.parquet"),
            ("year=$year/month=$month", "year=2024/month=01.parquet"),
        ]
    ),
    *(
        NoRewriteCase(
            f"null_date_partition_is_cached-{fmt}",
            source=pl.LazyFrame({"date": [datetime.date(2024, 1, 10), None], "value": [0, 1]}),
            reads=[Read(None)],
            seed=[None],
            seed_file_count=2,
            cache_kwargs={"partition_format": fmt},
        )
        for fmt in [None, "$year/$month"]
    ),
    # the read excluded the cached Decimal keys with is_in on a list Polars types as Decimal(38, 2), which is_in rejects
    NoRewriteCase(
        "decimal_partition_exclusion_without_date_column",
        source=pl.LazyFrame({"x": pl.Series(["1.23", "-2.50", "4.00"]).cast(pl.Decimal(10, 2)), "value": [0, 1, 2]}),
        reads=[Read(None)],
        seed=[pl.col("x") < 4],
        cache_kwargs={"date_column": None, "extra_partition_cols": "x"},
    ),
]


def _seed(tmp_path: Path, case: RegressionCase, seed_source: pl.LazyFrame) -> dict:
    kwargs = {"date_column": "date"} | case.cache_kwargs
    for pred in case.seed:
        collect_filtered(cache_parquet(seed_source, tmp_path, **kwargs), pred)
    return kwargs


def _read(tmp_path: Path, case: RegressionCase, kwargs: dict) -> None:
    for number, read in enumerate(case.reads, start=1):
        with failure_note(f"read {number} of {len(case.reads)}, in {read.mode.name} mode"):
            assert_cache_matches_plain_polars(case.source, tmp_path, read.predicate, runs=case.runs, **kwargs | {"cache_mode": read.mode})


@pytest.mark.parametrize("case", as_params(REGRESSIONS))
def test_regression(tmp_path, case: RegressionCase):
    _read(tmp_path, case, _seed(tmp_path, case, case.source if case.seed_source is None else case.seed_source))
    if case.final_file_count is not None:
        assert len(list(tmp_path.rglob("*.parquet"))) == case.final_file_count


@pytest.mark.parametrize("case", as_params(NO_REWRITE_REGRESSIONS))
def test_seed_files_are_not_rewritten(tmp_path, case: NoRewriteCase):
    kwargs = _seed(tmp_path, case, case.source)
    if case.seed_file_count is not None:
        assert len(list(tmp_path.rglob("*.parquet"))) == case.seed_file_count
    snapshot = _age_files(tmp_path)
    _read(tmp_path, case, kwargs)
    assert _rewritten(snapshot) == []
    assert all(path.exists() for path in snapshot)
    for path in case.expected_files:
        assert (tmp_path / path).exists(), path
    for path in case.absent_files:
        assert not (tmp_path / path).exists(), path


# --- Section: Partition keys: parsing, dtypes, spellings, and exclusion predicates ---

# Key parsing


def test_custom_partition_format_parsing():
    schema = pl.Schema({"sym": pl.String, "date": pl.Date})
    df = _construct_partitions_df_from_existing({"A/2024/03", "B/2023/12"}, schema, ["sym", "date"], "$year/$month", "date", "file")
    expected = pl.DataFrame({"sym": ["A", "B"], "date": [datetime.date(2024, 3, 1), datetime.date(2023, 12, 1)], "file": ["A/2024/03", "B/2023/12"]})
    assert_frame_equal(_sorted(df), expected)


def test_boolean_key_parsing_drops_unrecognized_keys():
    # "true"/"null" are the spellings of the polars < 1.38 writer
    schema = pl.Schema({"flag": pl.Boolean, "n": pl.Int64})
    keys = {"True/1", "False/None", "true/2", "false/null", "False/x", "yes/3"}
    df = _construct_partitions_df_from_existing(keys, schema, ["flag", "n"], None, None, "file").drop("file")
    assert sorted(df.rows(), key=str) == sorted([(True, 1), (False, None), (True, 2), (False, None)], key=str)


def test_existing_file_keys_preserve_percent_encoding(tmp_path):
    import pyarrow.fs as pa_fs

    # Polars writes "2020-01-01 12:00:00" as "2020-01-01 12%3A00%3A00" in local file names on Windows.
    (tmp_path / "2020-01-01 12%3A00%3A00").mkdir()
    (tmp_path / "2020-01-01 12%3A00%3A00" / "2024-01.parquet").touch()
    assert _extract_existing_files(pa_fs.LocalFileSystem(), str(tmp_path)) == {"2020-01-01 12%3A00%3A00/2024-01"}


# Extra partition column dtypes and nulls


@dataclass(frozen=True)
class PartitionDtypeCase:
    """Two values of an extra partition column dtype, which must survive a round trip through the cache."""

    id: str
    _: KW_ONLY
    values: list
    dtype: pl.DataType | type[pl.DataType]


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            PartitionDtypeCase("boolean", values=[True, False], dtype=pl.Boolean),
            PartitionDtypeCase("float", values=[0.5, 1e-05], dtype=pl.Float64),
            PartitionDtypeCase("date", values=[datetime.date(2020, 1, 1), datetime.date(2021, 1, 1)], dtype=pl.Date),
            PartitionDtypeCase(
                "datetime", values=[datetime.datetime(2020, 1, 1, 12), datetime.datetime(2020, 1, 2, 0, 0, 0, 500)], dtype=pl.Datetime("us")
            ),
            PartitionDtypeCase(
                "datetime_tz",
                values=[datetime.datetime(2020, 1, 1, 12, tzinfo=ZoneInfo("UTC")), datetime.datetime(2020, 1, 2, tzinfo=ZoneInfo("UTC"))],
                dtype=pl.Datetime("us", "Asia/Tokyo"),
            ),
            # Whole seconds: polars < 1.38 drops fractional seconds from Time keys
            PartitionDtypeCase("time", values=[datetime.time(12, 30), datetime.time(0, 0, 1)], dtype=pl.Time),
            PartitionDtypeCase("decimal", values=["1.23", "-2.50"], dtype=pl.Decimal(10, 2)),
        ]
    ),
)
def test_extra_partition_col_dtypes_round_trip(tmp_path, case: PartitionDtypeCase):
    series = pl.Series(case.values).cast(case.dtype) if isinstance(case.dtype, pl.Decimal) else pl.Series(case.values, dtype=case.dtype)
    src = pl.LazyFrame({"date": [datetime.date(2024, 1, 10), datetime.date(2024, 1, 11)], "x": series, "value": [0, 1]})
    kwargs = {"date_column": "date", **_MONTHLY_BY_X}
    assert_cache_matches_plain_polars(src, tmp_path, _JANUARY_2024 & (pl.col("x") == series[0]), **kwargs)
    cache_parquet(src, tmp_path, **kwargs).filter(_JANUARY_2024).collect()
    snapshot = _age_files(tmp_path)
    assert len(snapshot) == 2
    for pred in (_JANUARY_2024, None):
        assert_cache_matches_plain_polars(src, tmp_path, pred, **kwargs)
    assert _rewritten(snapshot) == []
    assert sorted(tmp_path.rglob("*.parquet")) == sorted(snapshot)


@pytest.mark.parametrize(
    "x",
    [pl.Series([datetime.timedelta(hours=1)]), pl.Series([datetime.datetime(2024, 1, 1)], dtype=pl.Datetime("ns"))],
    ids=["duration", "datetime_ns"],
)
def test_unsupported_extra_partition_col_is_rejected(tmp_path, x):
    src = pl.LazyFrame({"date": [datetime.date(2024, 1, 10)], "x": x})
    with pytest.raises(ValueError, match="cannot partition on Duration or nanosecond Datetime column x"):
        cache_parquet(src, tmp_path, "date", extra_partition_cols="x")


@dataclass(frozen=True)
class NullPartitionValueCase:
    """A value and a null of an extra partition column dtype, both of which must be cached."""

    id: str
    _: KW_ONLY
    values: list
    dtype: pl.DataType | type[pl.DataType]


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            NullPartitionValueCase("str", values=["a", None], dtype=pl.String),
            NullPartitionValueCase("int", values=[1, None], dtype=pl.Int64),
            NullPartitionValueCase("bool", values=[True, None], dtype=pl.Boolean),
            NullPartitionValueCase("enum", values=["a", None], dtype=pl.Enum(["a", "b"])),
            NullPartitionValueCase("categorical", values=["a", None], dtype=pl.Categorical),
        ]
    ),
)
@pytest.mark.parametrize("date_column", ["date", None], ids=["with_date_column", "no_date_column"])
def test_null_extra_partition_col_is_cached(tmp_path, case: NullPartitionValueCase, date_column):
    src = pl.LazyFrame(
        {"date": [datetime.date(2024, 1, 10), datetime.date(2024, 1, 11)], "x": pl.Series(case.values, dtype=case.dtype), "value": [0, 1]}
    )
    kwargs = {"date_column": date_column, **_MONTHLY_BY_X}
    cache_parquet(src, tmp_path, **kwargs).collect()
    snapshot = _age_files(tmp_path)
    assert len(snapshot) == 2
    for pred in (
        None,
        None,
        pl.col("x").is_null(),
        pl.col("x").is_null() & (pl.col("date") == datetime.date(2024, 1, 11)),
        pl.col("x") == case.values[0],
    ):
        assert_cache_matches_plain_polars(src, tmp_path, pred, **kwargs)
    assert _rewritten(snapshot) == []


@pytest.mark.parametrize("tz", ["Asia/Tokyo", "America/New_York"])
@pytest.mark.parametrize("time_unit", ["daily", "monthly"])
def test_timezone_aware_open_query_keeps_uncached_rows(tmp_path, tz, time_unit):
    dtype = pl.Datetime("us", tz)
    times = [
        datetime.datetime(2024, 1, 1, 12),
        datetime.datetime(2024, 1, 2, 0, 30),
        datetime.datetime(2024, 1, 2, 23, 30),
        datetime.datetime(2024, 2, 1, 0, 30),
    ]
    src = pl.LazyFrame({"ts": pl.Series(times).dt.replace_time_zone(tz), "value": [0, 1, 2, 3]})
    seed = pl.LazyFrame({"ts": pl.Series(times[:1]).dt.replace_time_zone(tz), "value": [0]})
    cache_parquet(seed, tmp_path, "ts", time_unit=time_unit).collect()
    snapshot = _age_files(tmp_path)
    pred = pl.col("ts") >= datetime.datetime(2023, 12, 1, tzinfo=ZoneInfo(tz))
    for _ in range(2):  # writes the uncached periods, then reads them back
        out = cache_parquet(src, tmp_path, "ts", time_unit=time_unit).filter(pred).collect()
        assert out.schema["ts"] == dtype
        assert sorted(out["value"].to_list()) == ([0, 1, 2, 3] if time_unit == "daily" else [0, 3])
    assert _rewritten(snapshot) == []


# Exclusion predicates over many cached partitions


def test_exclusion_predicate_scales_with_fragmented_cache():
    from polars_io_tools.io_sources.lazy_cache_parquet import _build_not_existing_partitions_pred

    days = pl.date_range(datetime.date(2000, 1, 1), datetime.date(2021, 12, 31), eager=True)
    cached = pl.DataFrame({"date": days.gather_every(2)})
    rows = pl.DataFrame({"date": days})
    start = time.perf_counter()
    pred = _build_not_existing_partitions_pred(cached, ["date"], "date", "daily", pl.Date)
    kept = rows.filter(pred)
    assert time.perf_counter() - start < 2
    assert kept["date"].to_list() == days.gather_every(2, offset=1).to_list()


def test_exclusion_predicate_scales_with_uneven_date_coverage():
    from polars_io_tools.io_sources.lazy_cache_parquet import _build_not_existing_partitions_pred

    # Every symbol caches a different subset of 10 dates
    days = [datetime.date(2024, 1, 1) + datetime.timedelta(days=j) for j in range(11)]
    cached = pl.DataFrame(
        [(f"s{i}", days[j]) for i in range(1, 1024) for j in range(10) if i & (1 << j)], schema={"sym": pl.String, "date": pl.Date}, orient="row"
    )
    rows = pl.DataFrame([(f"s{i}", d) for i in range(1025) for d in days], schema=cached.schema, orient="row")
    start = time.perf_counter()
    pred = _build_not_existing_partitions_pred(cached, ["sym", "date"], "date", "daily", pl.Date)
    kept = rows.filter(pred)
    assert time.perf_counter() - start < 1
    assert_frame_equal(_sorted(kept), _sorted(rows.join(cached, on=["sym", "date"], how="anti")))


# Key spellings: the writer of the installed Polars, the legacy writer, and Windows file names


def test_boolean_partition_finite_query_reads_back(tmp_path):
    src = pl.LazyFrame({"date": [datetime.date(2024, 1, 10), datetime.date(2024, 1, 10)], "flag": [True, False], "value": [0, 1]})
    pred = (pl.col("flag") == True) & (pl.col("date") == datetime.date(2024, 1, 10))
    assert_cache_matches_plain_polars(src, tmp_path, pred, date_column="date", time_unit="monthly", extra_partition_cols="flag")
    true_key = "True" if POLARS_HAS_PARTITION_BY else "true"
    assert sorted(p.parent.name for p in (tmp_path / "monthly").rglob("*.parquet")) == [true_key]


def test_custom_partition_format_extra_value_with_equals_sign(tmp_path):
    src = pl.LazyFrame({"date": [datetime.date(2024, 1, 10), datetime.date(2024, 1, 10)], "k": ["a=b", "c"], "value": [0, 1]})
    kwargs = {"time_unit": "monthly", "partition_format": "$year/$month", "extra_partition_cols": "k"}
    cache_parquet(src, tmp_path, "date", **kwargs).filter(_JANUARY_2024).collect()
    snapshot = _age_files(tmp_path)
    out = cache_parquet(src.with_columns(pl.col("value") + 10), tmp_path, "date", **kwargs).filter(_JANUARY_2024).collect()
    assert_frame_equal(_sorted(out), _sorted(src.collect()))
    assert _rewritten(snapshot) == []


@dataclass(frozen=True)
class OldKeySpellingCase:
    """Two values of a partition column dtype, and the keys an older writer spelled them as."""

    id: str
    _: KW_ONLY
    values: list
    dtype: pl.DataType | type[pl.DataType]
    old_keys: list[str]


OLD_KEY_SPELLINGS = [
    OldKeySpellingCase("boolean", values=[True, False], dtype=pl.Boolean, old_keys=["true", "false"]),
    OldKeySpellingCase(
        "datetime",
        values=[datetime.datetime(2020, 1, 1, 12), datetime.datetime(2020, 1, 2, 0, 0, 0, 500)],
        dtype=pl.Datetime("us"),
        old_keys=["2020-01-01 12:00:00.000000", "2020-01-02 00:00:00.000500"],
    ),
    OldKeySpellingCase("null", values=[True, None], dtype=pl.Boolean, old_keys=["true", "null"]),
]


def _write_old_key_files(tmp_path: Path, case: OldKeySpellingCase) -> tuple[pl.DataFrame, dict[Path, int]]:
    """Write one January row per old key; return the rows and the files' modification times."""
    src = pl.DataFrame(
        {"date": [datetime.date(2024, 1, 10), datetime.date(2024, 1, 11)], "x": pl.Series(case.values, dtype=case.dtype), "value": [0, 1]}
    )
    for index, key in enumerate(case.old_keys):
        if os.name == "nt":
            key = quote(key, safe=" -_.")
        path = tmp_path / "monthly" / key / "2024-01.parquet"
        path.parent.mkdir(parents=True)
        src.slice(index, 1).write_parquet(path)
    return src, _age_files(tmp_path)


@pytest.mark.parametrize("case", as_params(OLD_KEY_SPELLINGS))
def test_cache_reads_files_with_old_key_spellings(tmp_path, case: OldKeySpellingCase):
    src, snapshot = _write_old_key_files(tmp_path, case)
    predicate = _JANUARY_2024 & (pl.col("x") == case.values[0])
    for _ in range(2):
        out = cache_parquet(src.lazy(), tmp_path, "date", extra_partition_cols="x").filter(predicate).collect()
        assert_frame_equal(_sorted(out), _sorted(src.filter(predicate)))
    assert_frame_equal(_sorted(cache_parquet(src.lazy(), tmp_path, "date", extra_partition_cols="x").collect()), _sorted(src))
    assert sorted(tmp_path.rglob("*.parquet")) == sorted(snapshot)
    assert _rewritten(snapshot) == []


@pytest.mark.parametrize("case", as_params(OLD_KEY_SPELLINGS))
def test_rebuild_replaces_files_with_old_key_spellings(tmp_path, case: OldKeySpellingCase):
    src, snapshot = _write_old_key_files(tmp_path, case)
    current = src.with_columns(pl.col("value") + 100)
    for _ in range(2):
        out = cache_parquet(current.lazy(), tmp_path, "date", extra_partition_cols="x", cache_mode=CacheMode.REBUILD).collect()
        assert_frame_equal(_sorted(out), _sorted(current))
    assert_frame_equal(_sorted(cache_parquet(current.lazy(), tmp_path, "date", extra_partition_cols="x").collect()), _sorted(current))
    remaining = current.filter(pl.col("value") == 101)
    predicate = _JANUARY_2024 & (pl.col("x") == case.values[0])
    out = cache_parquet(remaining.lazy(), tmp_path, "date", extra_partition_cols="x", cache_mode=CacheMode.REBUILD).filter(predicate).collect()
    assert out.is_empty()
    assert_frame_equal(_sorted(cache_parquet(remaining.lazy(), tmp_path, "date", extra_partition_cols="x").collect()), _sorted(remaining))
    assert sorted(tmp_path.rglob("*.parquet")) == sorted(snapshot)


def _second_read_reuses_files(tmp_path, src, predicate, expanded, **kwargs):
    """Read ``src`` with ``predicate``; ``expanded`` read with the same predicate is served by the same unchanged files."""
    expected = src.filter(predicate).collect()
    assert_frame_equal(cache_parquet(src, tmp_path, **kwargs).filter(predicate).collect(), expected)
    snapshot = _age_files(tmp_path)
    assert_frame_equal(cache_parquet(expanded, tmp_path, **kwargs).filter(predicate).collect(), expected)
    assert sorted(tmp_path.rglob("*.parquet")) == sorted(snapshot)
    assert _rewritten(snapshot) == []


@pytest.mark.parametrize("value", ["%2541", "%25", "a%253Ab"])
def test_literal_percent_partition_values_round_trip(tmp_path, value):
    src = pl.LazyFrame({"x": [value], "value": [0]})
    _second_read_reuses_files(tmp_path, src, pl.col("x") == value, src, extra_partition_cols="x")


def _empty_marker_sources(value, empty):
    src = pl.LazyFrame({"date": [datetime.date(2024, 1, 10)], "x": ["other" if empty else value], "value": [0]})
    return src, pl.concat([src, pl.LazyFrame({"date": [datetime.date(2024, 1, 10)], "x": [value], "value": [99]})])


@pytest.mark.parametrize("value", ["New York", "a#b", "x?y", "%2541"])
def test_empty_marker_preserves_literal_partition_key(tmp_path, value):
    """The empty file for a value the source lacks keeps the value's key, so a later read finds it."""
    src, expanded = _empty_marker_sources(value, empty=True)
    _second_read_reuses_files(tmp_path, src, _JANUARY_2024 & (pl.col("x") == value), expanded, date_column="date", extra_partition_cols="x")


@dataclass(frozen=True)
class WindowsFileKeyCase:
    """A partition value, and the file key it gets on Windows."""

    id: str
    _: KW_ONLY
    value: str
    file_key: str


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            WindowsFileKeyCase("%2541-%252541", value="%2541", file_key="%252541"),
            WindowsFileKeyCase("%25-%2525", value="%25", file_key="%2525"),
            WindowsFileKeyCase("a%253Ab-a%25253Ab", value="a%253Ab", file_key="a%25253Ab"),
            WindowsFileKeyCase("x?y-x%3Fy", value="x?y", file_key="x%3Fy"),
            WindowsFileKeyCase("12:00-12%3A00", value="12:00", file_key="12%3A00"),
        ]
    ),
)
@pytest.mark.parametrize("empty", [False, True], ids=["value_present", "value_absent"])
def test_windows_partition_file_keys_round_trip_on_any_platform(tmp_path, monkeypatch, case: WindowsFileKeyCase, empty):
    module = importlib.import_module("polars_io_tools.io_sources.lazy_cache_parquet")
    monkeypatch.setattr(module, "os", SimpleNamespace(name="nt", environ=os.environ))
    src, expanded = _empty_marker_sources(case.value, empty)
    _second_read_reuses_files(tmp_path, src, _JANUARY_2024 & (pl.col("x") == case.value), expanded, date_column="date", extra_partition_cols="x")
    assert (tmp_path / "monthly" / case.file_key / "2024-01.parquet").exists()


# --- Section: REBUILD scope: what a rebuild may overwrite or remove ---


def test_rebuild_without_date_constraint_removes_stale_partitions(tmp_path):
    src = pl.LazyFrame(
        {"date": [datetime.date(2024, 1, 15), datetime.date(2024, 2, 15), datetime.date(2024, 1, 15)], "k": ["a", "a", "b"], "value": [0, 1, 2]}
    )
    kwargs = {"time_unit": "monthly", "extra_partition_cols": "k"}

    def cached_files():
        return sorted(p.relative_to(tmp_path / "monthly").as_posix() for p in (tmp_path / "monthly").rglob("*.parquet"))

    cache_parquet(src, tmp_path, "date", **kwargs).collect()
    rebuilt = src.filter(pl.col("value") == 1)
    out = cache_parquet(rebuilt, tmp_path, "date", cache_mode=CacheMode.REBUILD, **kwargs).filter(pl.col("k") == "a").collect()
    assert out["value"].to_list() == [1]
    assert cached_files() == ["a/2024-02.parquet", "b/2024-01.parquet"]

    out = cache_parquet(rebuilt, tmp_path, "date", cache_mode=CacheMode.REBUILD, **kwargs).collect()
    assert out["value"].to_list() == [1]
    assert cached_files() == ["a/2024-02.parquet"]


def test_rebuild_null_date_scope_preserves_other_partitions(tmp_path):
    seed = pl.LazyFrame({"date": [datetime.date(2024, 1, 10), datetime.date(2024, 2, 10)], "value": [0, 1]})
    cache_parquet(seed, tmp_path, "date").collect()
    snapshot = _age_files(tmp_path)
    current = pl.LazyFrame({"date": pl.Series([None], dtype=pl.Date), "value": [99]})
    out = cache_parquet(current, tmp_path, "date", cache_mode=CacheMode.REBUILD).filter(pl.col("date").is_null()).collect()
    assert_frame_equal(out, current.collect())
    assert all(path.exists() for path in snapshot)
    assert _rewritten(snapshot) == []
    assert sorted(cache_parquet(current, tmp_path, "date").collect()["value"].to_list()) == [0, 1, 99]


def _rebuild_keeps(tmp_path, src, keep_value, current, predicate, **kwargs):
    """Cache ``src``, REBUILD ``current`` over ``predicate``, and return the untouched file holding ``keep_value``."""
    cache_parquet(src, tmp_path, **kwargs).collect()
    snapshot = _age_files(tmp_path)
    kept = next(path for path in snapshot if pl.read_parquet(path)["value"].to_list() == [keep_value])
    out = cache_parquet(current, tmp_path, cache_mode=CacheMode.REBUILD, **kwargs).filter(predicate).collect()
    assert_frame_equal(out, current.filter(predicate).collect())
    assert pl.read_parquet(kept)["value"].to_list() == [keep_value]
    assert kept.stat().st_mtime_ns == snapshot[kept]


@dataclass(frozen=True)
class ExclusiveLowerBoundCase:
    """Two dates in neighbouring partitions and a bound between them; REBUILD with ``date > bound`` must keep the
    earlier partition."""

    id: str
    _: KW_ONLY
    time_unit: str
    dates: list[datetime.date]
    bound: datetime.date | datetime.datetime


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            ExclusiveLowerBoundCase(
                "daily",
                time_unit="daily",
                dates=[datetime.date(2024, 1, 31), datetime.date(2024, 2, 1)],
                bound=datetime.date(2024, 1, 31),
            ),
            ExclusiveLowerBoundCase(
                "monthly",
                time_unit="monthly",
                dates=[datetime.date(2024, 1, 10), datetime.date(2024, 2, 10)],
                bound=datetime.date(2024, 1, 31),
            ),
            ExclusiveLowerBoundCase(
                "yearly",
                time_unit="yearly",
                dates=[datetime.date(2023, 12, 10), datetime.date(2024, 2, 10)],
                bound=datetime.date(2023, 12, 31),
            ),
        ]
    ),
)
def test_rebuild_exclusive_lower_bound_preserves_previous_partition(tmp_path, case: ExclusiveLowerBoundCase):
    src = pl.LazyFrame({"date": case.dates, "value": [0, 1]})
    _rebuild_keeps(
        tmp_path, src, 0, src.with_columns(pl.col("value") + 100), pl.col("date") > case.bound, date_column="date", time_unit=case.time_unit
    )


@pytest.mark.parametrize("tz", ["UTC", "America/New_York", "Asia/Tokyo"])
@pytest.mark.parametrize("partition_format", [None, "$year/$month"], ids=["default_format", "year_slash_month"])
def test_rebuild_timezone_date_scope_with_unconstrained_extra_column(tmp_path, tz, partition_format):
    zone = ZoneInfo(tz)
    src = pl.LazyFrame(
        {
            "ts": pl.Series([datetime.datetime(2024, 1, 10, tzinfo=zone), datetime.datetime(2024, 2, 10, tzinfo=zone)], dtype=pl.Datetime("us", tz)),
            "x": ["a", "b"],
            "value": [0, 1],
        }
    )
    options = {"date_column": "ts", "extra_partition_cols": "x", "partition_format": partition_format}
    current = src.filter(pl.col("value") == 0).with_columns(pl.col("value") + 100)
    predicate = pl.col("ts").is_between(datetime.datetime(2024, 1, 1, tzinfo=zone), datetime.datetime(2024, 1, 31, tzinfo=zone))
    _rebuild_keeps(tmp_path, src, 1, current, predicate, **options)
    assert sorted(cache_parquet(current, tmp_path, **options).collect()["value"].to_list()) == [1, 100]


@pytest.mark.parametrize("nulls_equal", [False, True], ids=["nulls_not_equal", "nulls_equal"])
def test_rebuild_date_membership_respects_null_matching(tmp_path, nulls_equal):
    src = pl.LazyFrame({"date": [datetime.date(2024, 1, 10), None, datetime.date(2024, 2, 10)], "value": [0, 1, 2]})
    cache_parquet(src, tmp_path, "date").collect()
    snapshot = _age_files(tmp_path)
    null_path = next(path for path in snapshot if pl.read_parquet(path)["date"].null_count())
    current = src.filter(pl.col("value") == 0).with_columns(pl.col("value") + 100)
    predicate = pl.col("date").is_in([datetime.date(2024, 1, 10), None], nulls_equal=nulls_equal)
    out = cache_parquet(current, tmp_path, "date", cache_mode=CacheMode.REBUILD).filter(predicate).collect()
    assert_frame_equal(out, current.collect())
    assert null_path.exists() is not nulls_equal
    if not nulls_equal:
        assert null_path.stat().st_mtime_ns == snapshot[null_path]
    remaining = cache_parquet(current, tmp_path, "date").collect()["value"].to_list()
    assert sorted(remaining) == ([2, 100] if nulls_equal else [1, 2, 100])


def test_rebuild_scope_with_date_predicate_failing_on_null():
    pred = pl.col("date").map_elements(lambda d: d.weekday() < 5, return_dtype=pl.Boolean, skip_nulls=False)
    scope = _rebuild_partition_scope(pred, "date", [], "daily", pl.Schema({"date": pl.Date}))
    partitions = pl.DataFrame({"date": [datetime.date(2024, 3, 11), None]})
    assert scope is None or partitions.filter(scope).height == 2


# A REBUILD cache over another REBUILD cache rewrites only the inner partitions the query needs.
def _source_recording_predicates(seen: list, source_df: pl.DataFrame) -> pl.LazyFrame:
    def gen(with_columns, predicate, n_rows, batch_size):
        seen.append(predicate)
        frame = source_df if predicate is None else source_df.filter(predicate)
        yield frame.select(with_columns) if with_columns else frame

    return register_io_source_with_is_pure(gen, schema=source_df.schema, is_pure=True)


def _nested_rebuild_caches(tmp_path, seen: list, source_df: pl.DataFrame, time_unit: str) -> pl.LazyFrame:
    inner = cache_parquet(
        lambda: _source_recording_predicates(seen, source_df), tmp_path / "inner", "date", time_unit=time_unit, cache_mode=CacheMode.REBUILD
    )
    return cache_parquet(lambda: inner, tmp_path / "outer", "date", time_unit=time_unit, cache_mode=CacheMode.REBUILD)


@pytest.mark.parametrize(
    ("time_unit", "dates", "target"),
    [
        ("daily", [datetime.date(2025, 1, 1) + timedelta(days=i) for i in range(40)], datetime.date(2025, 1, 31)),
        ("monthly", [datetime.date(2025, m, 15) for m in range(1, 13)], datetime.date(2025, 6, 15)),
        (
            "monthly",
            [datetime.datetime(2025, m, 1, 9, tzinfo=ZoneInfo("America/New_York")) for m in range(1, 13)],
            datetime.datetime(2025, 6, 1, 9, tzinfo=ZoneInfo("America/New_York")),
        ),
    ],
    ids=["daily", "monthly", "monthly_tz"],
)
def test_nested_rebuild_rewrites_only_the_requested_inner_partition(tmp_path, time_unit, dates, target):
    source_df = pl.DataFrame({"date": dates, "value": list(range(len(dates)))})
    seen: list[pl.Expr | None] = []
    out = _nested_rebuild_caches(tmp_path, seen, source_df, time_unit).filter(pl.col("date") == target).collect()
    assert out["value"].to_list() == [dates.index(target)]
    assert len(list((tmp_path / "inner").rglob("*.parquet"))) == 1
    # The inner source was always scanned with a predicate that bounds the dates and keeps the target.
    assert seen and all(p is not None for p in seen)
    for pushed in seen:
        kept = source_df.filter(pushed)["date"].to_list()
        assert target in kept and len(kept) < len(dates)


def test_nested_rebuild_with_null_dates_rewrites_only_the_day_and_null_partitions(tmp_path):
    source_df = pl.DataFrame(
        {"date": [datetime.date(2025, 1, 10), datetime.date(2025, 1, 20), None], "value": [1, 2, 3]}, schema={"date": pl.Date, "value": pl.Int64}
    )
    seen: list[pl.Expr | None] = []
    outer = _nested_rebuild_caches(tmp_path, seen, source_df, "daily")
    out = outer.filter(pl.col("date").is_in([datetime.date(2025, 1, 10), None], nulls_equal=True)).collect().sort("value")
    assert out["value"].to_list() == [1, 3]
    inner_dates = pl.concat([pl.read_parquet(p) for p in sorted((tmp_path / "inner").rglob("*.parquet"))])["date"].to_list()
    assert set(inner_dates) == {datetime.date(2025, 1, 10), None}
    assert seen and all(p is not None for p in seen)


# --- Section: Signed zero keys: -0.0 and 0.0 are one partition ---


def _signed_zero(dtype, source_value, layout):
    source = pl.DataFrame({"key": pl.Series([source_value], dtype=dtype), "value": [7]})
    predicate = pl.col("key") == -source_value
    options = {"extra_partition_cols": "key"}
    if layout == "no-date":
        return source, predicate, options, 1
    source = source.with_columns(pl.lit(datetime.date(2024, 1, 10)).alias("date"))
    options.update(date_column="date", partition_format="$year/$month" if layout == "custom" else None)
    return source, predicate & pl.col("date").is_between(datetime.date(2024, 1, 1), datetime.date(2024, 2, 29)), options, 2


_signed_zero_axes = (
    pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64]),
    pytest.mark.parametrize("source_value", [-0.0, 0.0]),
    pytest.mark.parametrize("layout", ["no-date", "default", "custom"]),
)


@_signed_zero_axes[0]
@_signed_zero_axes[1]
@_signed_zero_axes[2]
@pytest.mark.parametrize("cache_mode", [CacheMode.CACHE, CacheMode.REBUILD])
@pytest.mark.parametrize("write_batch_size", [None, 1], ids=["unbatched", "batches_of_one"])
def test_equal_signed_zero_partition_does_not_create_false_gap(tmp_path, dtype, source_value, layout, cache_mode, write_batch_size):
    source, predicate, options, expected_files = _signed_zero(dtype, source_value, layout)
    visited = []
    lf = cache_parquet(source.lazy(), tmp_path, cache_mode=cache_mode, file_visitor=visited.append, write_batch_size=write_batch_size, **options)
    assert_frame_equal(lf.filter(predicate).collect(), source)
    assert len(list(tmp_path.rglob("*.parquet"))) == expected_files
    assert len(visited) == len(set(visited)) == expected_files


@_signed_zero_axes[0]
@_signed_zero_axes[1]
@_signed_zero_axes[2]
@pytest.mark.parametrize("refresh", ["rebuild", "ttl"])
def test_equal_signed_zero_refresh_preserves_written_rows(tmp_path, dtype, source_value, layout, refresh):
    source, predicate, options, expected_files = _signed_zero(dtype, source_value, layout)
    cache_parquet(source.lazy(), tmp_path, **options).collect()
    refresh_options = {"cache_mode": CacheMode.REBUILD} if refresh == "rebuild" else {"ttl": datetime.timedelta(0)}
    for value in [107, 207]:
        current = source.with_columns(pl.lit(value, pl.Int64).alias("value"))
        visited = []
        result = cache_parquet(current.lazy(), tmp_path, file_visitor=visited.append, **options, **refresh_options).filter(predicate).collect()
        assert_frame_equal(result, current)
        assert len(visited) == len(set(visited))
    assert_frame_equal(cache_parquet(source.lazy(), tmp_path, cache_mode=CacheMode.CACHE_ONLY, **options).collect(), current)
    assert len(list(tmp_path.rglob("*.parquet"))) == expected_files


# --- Section: Modes that never write: READ_ONLY and CACHE_ONLY ---

JANUARY_OR_FEBRUARY_2024 = pl.col("date").is_between(datetime.date(2024, 1, 1), datetime.date(2024, 2, 29))


def _snapshot(root) -> dict[str, tuple[int, int]]:
    """Map every file and directory under ``root`` to (size, mtime_ns), to detect any mutation."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = os.path.join(dirpath, name)
            st = os.stat(path)
            out[os.path.relpath(path, root)] = (st.st_size, st.st_mtime_ns)
    return out


def _three_dated_rows(offset: int = 0) -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "date": [datetime.date(2024, 1, 10), datetime.date(2024, 1, 15), datetime.date(2024, 2, 15)],
            "value": [offset + 0, offset + 1, offset + 2],
        }
    )


@pytest.fixture
def jan_cached(tmp_path):
    """A monthly cache holding only January, written from values 0 and 1."""
    cache_parquet(_three_dated_rows(), tmp_path, "date", time_unit="monthly").filter(
        pl.col("date").is_between(datetime.date(2024, 1, 1), datetime.date(2024, 1, 31))
    ).collect()
    return tmp_path


def _values(lf: pl.LazyFrame) -> list:
    return sorted(lf.collect()["value"].to_list())


@dataclass(frozen=True)
class NoWriteReadCase:
    """A filtered read of the January cache in the modes that do not write, and the values each mode returns."""

    id: str
    _: KW_ONLY
    predicate: pl.Expr | None
    # cache mode name -> the sorted ``value`` column the read returns
    expected: dict[str, list[int]]


# January is cached from values 0 and 1; upstream now holds 90, 91 and 92. READ_ONLY fills the gaps from upstream,
# CACHE_ONLY serves the cache alone, and neither changes the cache.
NO_WRITE_READS = [
    NoWriteReadCase("finite", predicate=JANUARY_OR_FEBRUARY_2024, expected={"READ_ONLY": [0, 1, 92], "CACHE_ONLY": [0, 1]}),
    NoWriteReadCase("unbounded", predicate=pl.col("date") >= datetime.date(2024, 1, 1), expected={"READ_ONLY": [0, 1, 92], "CACHE_ONLY": [0, 1]}),
    NoWriteReadCase("unconstrained", predicate=None, expected={"READ_ONLY": [0, 1, 92], "CACHE_ONLY": [0, 1]}),
    NoWriteReadCase(
        "not_equal_date_keeps_enclosing_partition",
        predicate=JANUARY_OR_FEBRUARY_2024 & (pl.col("date") != datetime.date(2024, 1, 15)),
        expected={"READ_ONLY": [0, 92], "CACHE_ONLY": [0]},
    ),
]


@pytest.mark.parametrize("mode", ["READ_ONLY", "CACHE_ONLY"])
@pytest.mark.parametrize("case", as_params(NO_WRITE_READS))
def test_no_write_mode_reads(jan_cached, mode, case: NoWriteReadCase):
    before = _snapshot(jan_cached)
    lf = cache_parquet(_three_dated_rows(offset=90), jan_cached, "date", time_unit="monthly", cache_mode=CacheMode[mode])
    assert _values(lf.filter(case.predicate) if case.predicate is not None else lf) == case.expected[mode]
    assert _snapshot(jan_cached) == before


@pytest.mark.parametrize("mode", [CacheMode.READ_ONLY, CacheMode.CACHE_ONLY])
def test_no_write_modes_do_not_mutate_cache(jan_cached, mode):
    before = _snapshot(jan_cached)
    cache_parquet(_three_dated_rows(offset=90), jan_cached, "date", time_unit="monthly", cache_mode=mode).filter(JANUARY_OR_FEBRUARY_2024).collect()
    cache_parquet(_three_dated_rows(offset=90), jan_cached, "date", time_unit="daily", cache_mode=mode).collect()
    assert _snapshot(jan_cached) == before


@pytest.mark.parametrize("mode", [CacheMode.READ_ONLY, CacheMode.CACHE_ONLY])
def test_no_write_modes_never_call_cache_writers(jan_cached, monkeypatch, mode):
    def fail(*args, **kwargs):
        raise AssertionError("no-write modes must not write")

    listing = lazy_cache_parquet._extract_existing_files

    def list_without_creating(fs, prefix, *, create_s3_dir=True, **kwargs):
        assert not create_s3_dir
        return listing(fs, prefix, create_s3_dir=create_s3_dir, **kwargs)

    monkeypatch.setattr(lazy_cache_parquet, "_sink_cache_partitions", fail)
    monkeypatch.setattr(lazy_cache_parquet, "_write_empty_parquet_file", fail)
    monkeypatch.setattr(lazy_cache_parquet, "_extract_existing_files", list_without_creating)
    lf = cache_parquet(_three_dated_rows(offset=90), jan_cached, "date", time_unit="monthly", cache_mode=mode)
    for predicate in [JANUARY_OR_FEBRUARY_2024, pl.col("date") >= datetime.date(2024, 1, 1), None]:
        (lf.filter(predicate) if predicate is not None else lf).collect()


@pytest.mark.parametrize("mode", [CacheMode.READ_ONLY, CacheMode.CACHE_ONLY])
def test_no_write_modes_on_missing_cache_directory(tmp_path, mode):
    cache_dir = tmp_path / "missing"
    result = cache_parquet(_three_dated_rows(), cache_dir, "date", cache_mode=mode).filter(JANUARY_OR_FEBRUARY_2024).collect()
    assert sorted(result["value"].to_list()) == ([0, 1, 2] if mode == CacheMode.READ_ONLY else [])
    assert result.schema == _three_dated_rows().collect_schema()
    assert not cache_dir.exists()


def test_read_only_cached_empty_partition_suppresses_upstream(tmp_path):
    # CACHE writes an empty file for February because upstream has no February rows yet.
    cache_parquet(_three_dated_rows().filter(pl.col("date") < datetime.date(2024, 2, 1)), tmp_path, "date", time_unit="monthly").filter(
        JANUARY_OR_FEBRUARY_2024
    ).collect()
    assert (tmp_path / "monthly" / "2024-02.parquet").exists()
    lf = cache_parquet(_three_dated_rows(offset=90), tmp_path, "date", time_unit="monthly", cache_mode=CacheMode.READ_ONLY)
    assert _values(lf.filter(JANUARY_OR_FEBRUARY_2024)) == [0, 1]


def test_cache_only_never_invokes_callable_source(jan_cached):
    def fail():
        raise AssertionError("upstream must not be called")

    assert _values(
        cache_parquet(fail, jan_cached, "date", time_unit="monthly", cache_mode=CacheMode.CACHE_ONLY).filter(JANUARY_OR_FEBRUARY_2024)
    ) == [0, 1]


def test_cache_only_callable_source_with_empty_cache(tmp_path):
    def fail():
        raise AssertionError("upstream must not be called")

    with pytest.raises(ValueError, match="requires `schema`"):
        cache_parquet(fail, tmp_path, "date", cache_mode=CacheMode.CACHE_ONLY)

    schema = _three_dated_rows().collect_schema()
    result = cache_parquet(fail, tmp_path, "date", schema=schema, cache_mode=CacheMode.CACHE_ONLY).collect()
    assert result.is_empty()
    assert result.schema == schema


def test_read_only_does_not_invoke_callable_when_source_not_needed_for_schema(jan_cached):
    calls = []

    def source():
        calls.append(1)
        return _three_dated_rows(offset=90)

    lf = cache_parquet(source, jan_cached, "date", time_unit="monthly", cache_mode=CacheMode.READ_ONLY)
    assert calls == []
    assert _values(lf.filter(JANUARY_OR_FEBRUARY_2024)) == [0, 1, 92]
    assert calls == [1]


@dataclass(frozen=True)
class ReadOnlyFormatCase:
    """A time unit and partition format for a READ_ONLY cache."""

    id: str
    _: KW_ONLY
    time_unit: str
    # None: the default format of ``time_unit``
    partition_format: str | None


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            ReadOnlyFormatCase("daily_default_format", time_unit="daily", partition_format=None),
            ReadOnlyFormatCase("yearly_default_format", time_unit="yearly", partition_format=None),
            ReadOnlyFormatCase("monthly_year_slash_month", time_unit="monthly", partition_format="$year/$month"),
            ReadOnlyFormatCase("monthly_with_prefix", time_unit="monthly", partition_format="prefix_$year-$month"),
        ]
    ),
)
def test_read_only_partition_formats(tmp_path, case: ReadOnlyFormatCase):
    source = pl.LazyFrame(
        {
            "date": [datetime.date(2023, 12, 30), datetime.date(2024, 1, 10), datetime.date(2024, 1, 15), datetime.date(2025, 3, 1)],
            "value": [0, 1, 2, 3],
        }
    )
    warm = pl.col("date").is_between(datetime.date(2023, 12, 30), datetime.date(2024, 1, 12))
    cache_parquet(source, tmp_path, "date", time_unit=case.time_unit, partition_format=case.partition_format).filter(warm).collect()
    upstream = source.with_columns(pl.col("value") + 90)
    lf = cache_parquet(upstream, tmp_path, "date", time_unit=case.time_unit, partition_format=case.partition_format, cache_mode=CacheMode.READ_ONLY)
    # CACHE writes whole partitions, so a monthly/yearly cache also holds 2024-01-15.
    expected = [0, 1, 92, 93] if case.time_unit == "daily" else [0, 1, 2, 93]
    assert _values(lf) == expected


def test_read_only_multiple_extra_partition_columns(tmp_path):
    source = pl.LazyFrame(
        {
            "region": ["A B", "A B", "C"],
            "bucket": [1, 2, 1],
            "value": [0, 1, 2],
        }
    )
    cols = ["region", "bucket"]
    cache_parquet(source, tmp_path, extra_partition_cols=cols).filter(pl.col("region") == "A B").collect()
    lf = cache_parquet(source.with_columns(pl.col("value") + 90), tmp_path, extra_partition_cols=cols, cache_mode=CacheMode.READ_ONLY)
    assert _values(lf) == [0, 1, 92]


@pytest.mark.parametrize("region", ["A B", "a:b", "x%41"])
@pytest.mark.parametrize("cache_mode", [CacheMode.READ_ONLY, CacheMode.CACHE_ONLY])
def test_no_write_modes_enumerate_keys_that_need_quoting(tmp_path, region, cache_mode):
    source = pl.LazyFrame({"region": [region, region], "bucket": [1, 2], "value": [0, 1]})
    cols = ["region", "bucket"]
    pred = (pl.col("region") == region) & pl.col("bucket").is_in([1, 2])
    cache_parquet(source, tmp_path, extra_partition_cols=cols).filter(pred & (pl.col("bucket") == 1)).collect()
    lf = cache_parquet(source.with_columns(pl.col("value") + 90), tmp_path, extra_partition_cols=cols, cache_mode=cache_mode)
    assert _values(lf.filter(pred)) == ([0, 91] if cache_mode == CacheMode.READ_ONLY else [0])


def test_read_only_row_limit_covers_upstream_rows(jan_cached):
    lf = cache_parquet(
        _three_dated_rows(offset=90), jan_cached, "date", time_unit="monthly", cache_mode=CacheMode.READ_ONLY, read_kwargs={"n_rows": 1}
    )
    assert _values(lf.filter(JANUARY_OR_FEBRUARY_2024)) == [0]


def test_read_only_without_partition_columns(tmp_path):
    cache_parquet(_three_dated_rows(), tmp_path).collect()
    lf = cache_parquet(_three_dated_rows(offset=90), tmp_path, cache_mode=CacheMode.READ_ONLY)
    assert _values(lf) == [0, 1, 2]


def test_read_only_casts_upstream_to_declared_schema(jan_cached):
    upstream = _three_dated_rows(offset=90).with_columns(pl.col("value").cast(pl.Float64))
    lf = cache_parquet(upstream, jan_cached, "date", time_unit="monthly", cache_mode=CacheMode.READ_ONLY)
    assert lf.collect_schema()["value"] == pl.Int64
    result = lf.filter(JANUARY_OR_FEBRUARY_2024).with_columns(half=pl.col("value") // 2).collect()
    assert result.schema["value"] == pl.Int64
    assert sorted(result["half"].to_list()) == [0, 0, 46]


@pytest.mark.parametrize(
    "predicate", [None, JANUARY_OR_FEBRUARY_2024, pl.col("date") >= datetime.date(2024, 1, 1)], ids=["unconstrained", "finite", "unbounded"]
)
@pytest.mark.parametrize("selection", [["date"], [pl.col("date").count()]], ids=["projection", "count"])
def test_read_only_ignores_missing_unrequested_upstream_columns(jan_cached, predicate, selection):
    before = _snapshot(jan_cached)
    upstream = _three_dated_rows().select("date")
    lf = cache_parquet(upstream, jan_cached, "date", cache_mode=CacheMode.READ_ONLY)
    if predicate is not None:
        lf = lf.filter(predicate)
    expected = pl.DataFrame({"date": [datetime.date(2024, 1, 10), datetime.date(2024, 1, 15), datetime.date(2024, 2, 15)]})
    assert_frame_equal(lf.select(selection).collect().sort(pl.all()), expected.select(selection).sort(pl.all()))
    assert _snapshot(jan_cached) == before


@pytest.mark.parametrize(
    "selection", [["value"], ["date"], ["region", "date"], [pl.col("value").count()]], ids=["value", "date", "reordered", "count"]
)
@pytest.mark.parametrize("engine", ["auto", "streaming"])
def test_read_only_projection_retains_filter_and_partition_columns(tmp_path, selection, engine):
    source = pl.LazyFrame(
        {
            "date": [datetime.date(2024, 1, 10), datetime.date(2024, 1, 15), datetime.date(2024, 2, 15), datetime.date(2024, 2, 15)],
            "region": ["A", "B", "A", "B"],
            "value": [0, 1, 2, 3],
            "unused": ["old"] * 4,
        }
    )
    cache_parquet(source, tmp_path, "date", extra_partition_cols="region").filter(
        (pl.col("date") == datetime.date(2024, 1, 10)) & (pl.col("region") == "A")
    ).collect()
    before = _snapshot(tmp_path)
    upstream = source.drop("unused").with_columns(pl.col("date").cast(pl.Datetime("ns")), (pl.col("value") + 90).cast(pl.String))
    lf = cache_parquet(upstream, tmp_path, "date", extra_partition_cols="region", cache_mode=CacheMode.READ_ONLY)
    expected = pl.DataFrame(
        {
            "date": [datetime.date(2024, 1, 10), datetime.date(2024, 2, 15), datetime.date(2024, 2, 15)],
            "region": ["A", "A", "B"],
            "value": [0, 92, 93],
        }
    )
    result = lf.filter(JANUARY_OR_FEBRUARY_2024 & (pl.col("value") != 91)).select(selection).collect(engine=engine)
    assert_frame_equal(result.sort(pl.all()), expected.select(selection).sort(pl.all()))
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("requirement", ["projection", "predicate"])
def test_read_only_does_not_ignore_missing_required_upstream_columns(jan_cached, requirement):
    lf = cache_parquet(_three_dated_rows().select("date"), jan_cached, "date", cache_mode=CacheMode.READ_ONLY)
    query = lf.select("value") if requirement == "projection" else lf.filter(pl.col("value") > 0).select("date")
    with pytest.raises(pl.exceptions.ComputeError, match="value"):
        query.collect()


def test_read_only_applies_predicate_to_conformed_upstream(tmp_path):
    upstream = pl.LazyFrame({"date": [datetime.date(2024, 1, 10), datetime.date(2024, 1, 15)], "value": ["1", "2"]})
    schema = pl.Schema({"date": pl.Date, "value": pl.Int64})
    lf = cache_parquet(upstream, tmp_path, "date", schema=schema, cache_mode=CacheMode.READ_ONLY)
    assert lf.filter(pl.col("value") == 1).collect()["value"].to_list() == [1]


@pytest.mark.parametrize("date_dtype", [pl.Datetime("ns"), pl.Datetime("ms")], ids=["datetime_ns", "datetime_ms"])
@pytest.mark.parametrize("predicate", [JANUARY_OR_FEBRUARY_2024, pl.col("date") >= datetime.date(2024, 1, 1)], ids=["finite", "unbounded"])
def test_read_only_filters_upstream_partitions_after_casting_to_cached_schema(jan_cached, date_dtype, predicate):
    upstream = _three_dated_rows(offset=90).with_columns(pl.col("date").cast(date_dtype))
    lf = cache_parquet(upstream, jan_cached, "date", time_unit="monthly", cache_mode=CacheMode.READ_ONLY)
    assert _values(lf.filter(predicate)) == [0, 1, 92]


def test_read_only_pushes_predicate_to_upstream(jan_cached):
    pushed = []
    source = _three_dated_rows(offset=90)

    def generator(with_columns, predicate, n_rows, batch_size):
        pushed.append(predicate)
        yield source.filter(predicate).collect() if predicate is not None else source.collect()

    upstream = register_io_source_with_is_pure(io_source=generator, schema=source.collect_schema())
    lf = cache_parquet(upstream, jan_cached, "date", time_unit="monthly", cache_mode=CacheMode.READ_ONLY)
    assert _values(lf.filter(JANUARY_OR_FEBRUARY_2024)) == [0, 1, 92]
    assert pushed and pushed[-1] is not None
    assert source.filter(pushed[-1]).collect()["value"].to_list() == [92]


def test_read_only_null_partition_key_cached(tmp_path):
    source = pl.LazyFrame({"k": ["a", None], "value": [0, 1]})
    cache_parquet(source, tmp_path, extra_partition_cols="k").collect()
    assert len(list((tmp_path / "null").glob("*.parquet"))) == 2
    lf = cache_parquet(source.with_columns(pl.col("value") + 90), tmp_path, extra_partition_cols="k", cache_mode=CacheMode.READ_ONLY)
    assert _values(lf) == [0, 1]


@dataclass(frozen=True)
class ReadOnlyCollectCase:
    """A way to collect a READ_ONLY read, and the number of rows it returns."""

    id: str
    _: KW_ONLY
    collect: Callable[[pl.LazyFrame], pl.DataFrame]
    height: int


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            ReadOnlyCollectCase("projection", collect=lambda lf: lf.select("value").collect(), height=3),
            ReadOnlyCollectCase("limit", collect=lambda lf: lf.head(2).collect(), height=2),
            ReadOnlyCollectCase("streaming", collect=lambda lf: lf.collect(engine="streaming"), height=3),
        ]
    ),
)
def test_read_only_projection_limit_and_streaming(jan_cached, case: ReadOnlyCollectCase):
    # January is read from the cache and February from upstream.
    expected = pl.concat(
        [_three_dated_rows().filter(pl.col("date").dt.month() == 1), _three_dated_rows(offset=90).filter(pl.col("date").dt.month() == 2)]
    ).collect()
    lf = cache_parquet(_three_dated_rows(offset=90), jan_cached, "date", time_unit="monthly", cache_mode=CacheMode.READ_ONLY).filter(
        JANUARY_OR_FEBRUARY_2024
    )
    got = case.collect(lf)
    assert got.height == case.height
    assert got.join(expected.select(got.columns), on=got.columns, how="anti").is_empty()
    assert not got.is_duplicated().any()


@pytest.mark.parametrize("name", ["READ_ONLY", "CACHE_ONLY"])
def test_new_modes_validate_from_string(name):
    class Config(pydantic.BaseModel):
        mode: CacheMode

    config = Config.model_validate({"mode": name})
    assert config.mode == CacheMode[name]
    assert config.model_dump(mode="json") == {"mode": name}


def test_read_only_over_chained_cache_does_not_deadlock_single_thread(tmp_path):
    script = textwrap.dedent(
        f"""
        import datetime as dt
        import polars as pl
        from polars_io_tools.io_sources.lazy_cache_parquet import CacheMode, cache_parquet

        src = pl.LazyFrame({{"date": [dt.date(2024, 1, 10), dt.date(2024, 2, 15)], "value": [0, 1]}})
        inner = cache_parquet(src, {str(tmp_path / "inner")!r}, "date")
        cache_parquet(inner, {str(tmp_path / "outer")!r}, "date").filter(pl.col("date").is_between(dt.date(2024, 1, 1), dt.date(2024, 1, 31))).collect()
        lf = cache_parquet(inner, {str(tmp_path / "outer")!r}, "date", cache_mode=CacheMode.READ_ONLY)
        lf = lf.filter(pl.col("date") >= dt.date(2024, 1, 1)).join(src.select("date"), on="date")
        print(sorted(lf.collect()["value"].to_list()), sorted(lf.collect(engine="streaming")["value"].to_list()))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "POLARS_MAX_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[0, 1] [0, 1]"


_MATRIX_DATES = [datetime.date(2024, 1, 10), datetime.date(2024, 1, 20), datetime.date(2024, 2, 10), datetime.date(2024, 3, 5), None]


@dataclass(frozen=True)
class ReadOnlyValuesCase:
    """Values of an extra partition column dtype, read with READ_ONLY and with CACHE."""

    id: str
    _: KW_ONLY
    values: list
    dtype: pl.DataType | type[pl.DataType]


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            ReadOnlyValuesCase("str", values=["A", "B", None], dtype=pl.String),
            ReadOnlyValuesCase("bool", values=[True, False, None], dtype=pl.Boolean),
            ReadOnlyValuesCase("categorical", values=["A", "B", None], dtype=pl.Categorical),
            ReadOnlyValuesCase("decimal", values=[Decimal("1.5"), Decimal("2.5"), None], dtype=pl.Decimal(10, 2)),
            ReadOnlyValuesCase("time", values=[datetime.time(1), datetime.time(2), None], dtype=pl.Time),
        ]
    ),
)
@pytest.mark.parametrize("partition_format", [None, "$year/$month"], ids=["default_format", "year_slash_month"])
@pytest.mark.parametrize(
    "predicate",
    [
        pytest.param(None, id="unconstrained"),
        pytest.param(pl.col("date") >= datetime.date(2024, 1, 1), id="unbounded"),
        pytest.param(
            pl.col("date").is_between(datetime.date(2024, 1, 1), datetime.date(2024, 3, 31)) & (pl.col("date") != datetime.date(2024, 1, 20)),
            id="not_equal",
        ),
        pytest.param(pl.col("x").is_null() | (pl.col("date") == datetime.date(2024, 2, 10)), id="null_or"),
    ],
)
def test_read_only_matches_cache_mode(tmp_path, case: ReadOnlyValuesCase, partition_format, predicate):
    rows = [(d, v) for d in _MATRIX_DATES for v in case.values]
    source = pl.LazyFrame({"date": [r[0] for r in rows], "x": pl.Series([r[1] for r in rows], dtype=case.dtype), "value": range(len(rows))})
    kwargs = {"date_column": "date", "time_unit": "monthly", "partition_format": partition_format, "extra_partition_cols": "x"}
    cache_parquet(source, tmp_path / "ro", **kwargs).filter(
        pl.col("date").is_between(datetime.date(2024, 1, 1), datetime.date(2024, 2, 29)) & (pl.col("x") == case.values[0])
    ).collect()
    shutil.copytree(tmp_path / "ro", tmp_path / "cache")
    before = _snapshot(tmp_path / "ro")
    upstream = source.with_columns(pl.col("value") + 1000)

    def run(path, mode):
        lf = cache_parquet(upstream, path, cache_mode=mode, **kwargs)
        return (lf.filter(predicate) if predicate is not None else lf).collect().sort(pl.all())

    cached = pl.col("date").is_between(datetime.date(2024, 1, 1), datetime.date(2024, 2, 29)) & (pl.col("x") == case.values[0])
    expected = source.with_columns(pl.when(cached).then(pl.col("value")).otherwise(pl.col("value") + 1000))
    expected = (expected.filter(predicate) if predicate is not None else expected).collect().sort(pl.all())

    read_only = run(tmp_path / "ro", CacheMode.READ_ONLY)
    assert_frame_equal(read_only, expected)
    assert_frame_equal(read_only, run(tmp_path / "cache", CacheMode.CACHE))
    assert (read_only["value"] < 1000).any() and (read_only["value"] >= 1000).any()
    assert _snapshot(tmp_path / "ro") == before


# --- Section: sort_by: a physical sort of each cache file ---
# sort_by orders the rows of each file, so the row-group statistics of the first key do not overlap and scan_parquet
# skips row groups, and so grouped temporal operations run on the cached read without a sort. The returned frame is
# not marked sorted, as the key is not sorted across partitions.

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
