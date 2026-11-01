"""cache_parquet: planning helpers as tables, then cache behavior as (steps -> result, files, upstream calls) tables.

The table-driven tests check results against native Polars with ``assert_cached_matches_native`` and the files left
behind with ``assert_cache_files``. Tests that patch writers, clocks or locks stay functions.
"""

import contextlib
import datetime
import os
import pickle
import shutil
import time
from dataclasses import KW_ONLY, dataclass, field
from datetime import timedelta
from pathlib import Path, PureWindowsPath
from typing import Any

import cloudpickle
import numpy as np
import polars as pl
import portion
import pyarrow.parquet as pq
import pytest
from polars.exceptions import ComputeError, PanicException
from polars.testing import assert_frame_equal
from pydantic import BaseModel

from polars_io_tools._compat import POLARS_GE_2
from polars_io_tools.io_sources import lazy_cache_parquet
from polars_io_tools.io_sources.lazy_cache_parquet import (
    CacheMode,
    Enumerability,
    PartitionInfo,
    _build_final_scan,
    _compute_clipped_expected_partitions,
    _compute_partitions_to_write,
    _default_expired,
    _get_expected_partitions_df,
    _get_fs_path_directory_info,
    cache_parquet,
)
from polars_io_tools.tests.helpers.cache_parquet_shared import (
    UpstreamSpy,
    assert_cache_files,
    assert_cached_matches_native,
    cache_files,
    collect_filtered,
    exercise_daily_cache_parquet,
)
from polars_io_tools.tests.helpers.cases import DATE_PREDICATES, TEMPORAL_DTYPES, Case, daily_frame, params

D = datetime.date
DT = datetime.datetime
UTC = datetime.UTC
MODES = [CacheMode.CACHE, CacheMode.READ_ONLY, CacheMode.CACHE_ONLY, CacheMode.REBUILD, CacheMode.IGNORE]


def _frame(**cols) -> pl.LazyFrame:
    return pl.LazyFrame(cols)


# _get_expected_partitions_df: (predicate, date column, extra columns, time unit) -> the partitions a query covers.
# Expected frames hold the extra columns, then the date column; an unrestricted column is null.

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


def parts(dates, col="date", **extra) -> pl.DataFrame:
    n = len(dates)
    cols = {k: v if isinstance(v, list) else [v] * n for k, v in extra.items()}
    return pl.DataFrame({**cols, col: dates}, schema={**{k: pl.String for k in extra}, col: SCHEMA[col]})


def EP(id, pred, expect, time_unit="monthly", extra=(), col="date"):
    return Case(id, pred, expect, {"extra": list(extra), "time_unit": time_unit, "col": col})


def _months(*ms, year=2024):
    return [D(year, m, 1) for m in ms]


_d, _r, _c = pl.col("date"), pl.col("region"), pl.col("category")
_RANGE = _d.is_between(D(2024, 3, 10), D(2024, 5, 20))
EXPECTED_PARTITIONS = [
    EP("none_predicate_behavior-date", None, parts([None])),
    EP("none_predicate_behavior-extra_cols", None, parts([None], region=None, country=None), "daily", ["region", "country"]),
    EP("simple_date_equality_monthly", _d == D(2024, 3, 15), parts(_months(3))),
    EP("simple_date_equality_daily", _d == D(2024, 5, 20), parts([D(2024, 5, 20)]), "daily"),
    EP("date_range_creates_multiple_monthly_partitions", _d.is_between(D(2024, 1, 15), D(2024, 4, 10)), parts(_months(1, 2, 3, 4))),
    EP("extra_columns_with_simple_predicates", (_d == D(2024, 7, 4)) & (_r == "US"), parts([D(2024, 7, 4)], region="US"), "daily", ["region"]),
    EP(
        "multiple_extra_columns_complex_predicate",
        (_d == D(2024, 8, 12)) & _r.is_in(["US", "EU", "APAC"]) & _c.is_in(["A", "B"]),
        parts([D(2024, 8, 12)] * 6, region=["US", "US", "EU", "EU", "APAC", "APAC"], category=["A", "B"] * 3),
        "daily",
        ["region", "category"],
    ),
    EP("or_conditions_create_multiple_partitions", (_d == D(2024, 9, 5)) | (_d == D(2024, 10, 15)), parts(_months(9, 10))),
    EP(
        "or_with_extra_columns_creates_all_combinations",
        (_d == D(2024, 11, 8)) & ((_r == "US") | (_r == "EU")) & ((_c == "X") | (_c == "Y")),
        parts([D(2024, 11, 8)] * 4, region=["US", "US", "EU", "EU"], category=["X", "Y"] * 2),
        "daily",
        ["region", "category"],
    ),
    EP("contradictory_predicates", (_d == D(2024, 1, 1)) & (_d == D(2024, 2, 1)), parts([])),
    EP("different_time_units_same_predicate-daily", _RANGE, parts(pl.date_range(D(2024, 3, 10), D(2024, 5, 20), eager=True).to_list()), "daily"),
    EP("different_time_units_same_predicate-monthly", _RANGE, parts(_months(3, 4, 5))),
    EP("different_time_units_same_predicate-yearly", _RANGE, parts(_months(1)), "yearly"),
    EP("open_ended_lower_bound_only", _d >= D(2024, 6, 5), parts([]), "daily"),
    EP("open_ended_upper_bound_only", _d <= D(2024, 6, 2), parts([]), "daily"),
    EP(
        "predicate_with_non_partition_columns_ignored",
        (_d == D(2024, 12, 25)) & (_r == "US") & (pl.col("value") > 100.0),
        parts([D(2024, 12, 25)], region="US"),
        "daily",
        ["region"],
    ),
    EP(
        "complex_nested_or_and_conditions",
        ((_d == D(2024, 6, 1)) & (_r == "US")) | ((_d == D(2024, 7, 1)) & (_r == "EU")),
        parts(_months(6, 7), region=["US", "EU"]),
        extra=["region"],
    ),
    EP(
        "timestamp_column_as_date_column",
        pl.col("timestamp") == DT(2024, 4, 15, 14, 30, 45),
        parts([DT(2024, 4, 15)], col="timestamp"),
        "daily",
        col="timestamp",
    ),
    EP(
        "large_date_ranges_yearly_partitioning",
        _d.is_between(D(2020, 1, 1), D(2024, 12, 31)) & _r.is_in(["US", "EU"]),
        parts([D(y, 1, 1) for y in range(2020, 2025) for _ in "ab"], region=["US", "EU"] * 5),
        "yearly",
        ["region"],
    ),
    EP("is_in_predicate_with_multiple_dates", _d.is_in([D(2024, 2, 14), D(2024, 3, 17), D(2024, 4, 22)]), parts(_months(2, 3, 4))),
]


@pytest.mark.parametrize(("pred", "expect", "col", "extra", "time_unit"), params(EXPECTED_PARTITIONS, "pred", "expect", "col", "extra", "time_unit"))
def test_get_expected_partitions_df(pred, expect, col, extra, time_unit):
    result = _get_expected_partitions_df(pred=pred, date_column=col, extra_cols=extra, time_unit=time_unit, schema=SCHEMA)
    assert_frame_equal(result, expect, check_row_order=False)


# _compute_clipped_expected_partitions: a one-sided interval is clipped to the cached date range, or None.

_inf = portion.inf


def CL(id, interval, existing, expect, time_unit="daily", region=None):
    return Case(id, interval, expect, {"existing": existing, "time_unit": time_unit, "region": region})


CLIPPED = [
    CL("returns_none_for_empty_interval", portion.empty(), [D(2024, 6, 1)], None),
    CL("returns_none_for_fully_bounded_interval", portion.closed(D(2024, 6, 1), D(2024, 6, 5)), [D(2024, 6, 1), D(2024, 6, 5)], None),
    CL("returns_none_for_fully_unbounded_interval", portion.open(-_inf, _inf), [D(2024, 6, 1)], None),
    CL("returns_none_for_empty_existing_partitions", portion.closedopen(D(2024, 6, 1), _inf), [], None),
    CL("returns_none_for_missing_date_column", portion.closedopen(D(2024, 6, 1), _inf), None, None),
    CL("returns_none_for_invalid_clipped_range", portion.closedopen(D(2024, 6, 10), _inf), [D(2024, 6, 1)], None),
    CL("unbounded_above_clips_to_existing_max", portion.closedopen(D(2024, 6, 2), _inf), [D(2024, 6, 1), D(2024, 6, 5)], [2, 3, 4, 5]),
    CL("unbounded_below_clips_to_existing_min", portion.openclosed(-_inf, D(2024, 6, 4)), [D(2024, 6, 2), D(2024, 6, 5)], [2, 3, 4]),
    CL("with_extra_cols_cross_product", portion.closedopen(D(2024, 6, 1), _inf), [D(2024, 6, 1), D(2024, 6, 2)], [1, 2], region=["US", "EU"]),
    CL("with_datetime_in_interval", portion.closedopen(DT(2024, 6, 2, 10, 30), _inf), [D(2024, 6, 1), D(2024, 6, 3)], [2, 3]),
    CL("monthly_time_unit", portion.closedopen(D(2024, 2, 15), _inf), [D(2024, 1, 15), D(2024, 4, 10)], _months(2, 3, 4), "monthly"),
    CL(
        "yearly_time_unit",
        portion.closedopen(D(2022, 6, 15), _inf),
        [D(2021, 1, 1), D(2024, 3, 10)],
        [D(y, 1, 1) for y in (2022, 2023, 2024)],
        "yearly",
    ),
    CL("single_date_existing", portion.closedopen(D(2024, 6, 1), _inf), [D(2024, 6, 1)], [1]),
]


@pytest.mark.parametrize(
    ("interval", "existing", "expect", "time_unit", "region"), params(CLIPPED, "pred", "existing", "expect", "time_unit", "region")
)
def test_compute_clipped_expected_partitions(interval, existing, expect, time_unit, region):
    if existing is None:
        existing_df = pl.DataFrame({"other_col": [1, 2, 3]})
    else:
        existing_df = pl.DataFrame({"date": existing}, schema={"date": pl.Date})
        if region:
            existing_df = existing_df.with_columns(region=pl.Series(region))
    extra = ["region"] if region else []
    schema = {"date": pl.Date, **({"region": pl.String} if region else {})}
    result = _compute_clipped_expected_partitions(
        predicate_interval=interval, existing_parts_df=existing_df, date_column="date", extra_cols=extra, time_unit=time_unit, schema=schema
    )
    if expect is None:
        assert result is None
        return
    dates = [D(2024, 6, d) if isinstance(d, int) else d for d in expect]
    want = pl.DataFrame({"date": dates}, schema={"date": pl.Date})
    if region:
        want = want.join(pl.DataFrame({"region": region}), how="cross")
    assert_frame_equal(result, want.select(result.columns), check_row_order=False)


# Small helpers


@pytest.mark.parametrize(
    ("expected_days", "existing_days", "remaining_days"),
    [([1, 2], [], [1, 2]), ([1, 2, 3], [1, 3], [2]), ([1, 2], [1, 2], [])],
    ids=["empty_existing", "some_existing", "all_existing"],
)
def test_compute_partitions_to_write(expected_days, existing_days, remaining_days):
    def days(ds):
        return pl.DataFrame({"date": [D(2024, 1, d) for d in ds]}, schema={"date": pl.Date})

    info = PartitionInfo(
        enumerability=Enumerability.FINITE, expected_parts_df=days(expected_days), existing_parts_df=days(existing_days), join_cols=["date"]
    )
    assert_frame_equal(_compute_partitions_to_write(info), days(remaining_days))


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

    dates = [D(2024, 1, 1) + timedelta(days=i) for i in range(_MAX_ENUMERATED_SCAN_PATHS + 1)]
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


@pytest.mark.parametrize(
    ("pred", "cols", "n_rows", "expected"),
    [
        (None, None, None, _SCAN),
        (pl.col("x") > 1, None, None, _SCAN.filter(pl.col("x") > 1)),
        (None, ["y"], None, _SCAN.select("y")),
        (None, None, 2, _SCAN.head(2)),
    ],
    ids=["cache_only", "with_predicate", "with_columns", "with_limit"],
)
def test_build_final_scan(pred, cols, n_rows, expected):
    assert_frame_equal(_build_final_scan(_SCAN.lazy(), pred, cols, n_rows).collect(), expected)


@pytest.mark.parametrize(
    ("root", "time_unit", "prefix", "time_dir"),
    [
        ("s3://mybucket/my/prefix/", "daily", "mybucket/my/prefix/daily", "s3://mybucket/my/prefix/daily"),
        ("s3://mybucket//double/leading/", "monthly", "mybucket//double/leading/monthly", "s3://mybucket//double/leading/monthly"),
        ("{tmp}", "daily", "{tmp}/daily", "{uri}/daily"),
        ("{uri}", "daily", "{tmp}/daily", "{uri}/daily"),
        ("{uri}", "monthly", "{tmp}/monthly", "{uri}/monthly"),
    ],
    ids=[
        "strips_trailing_slash_s3",
        "preserves_double_leading_slash_s3",
        "local_path_gets_file_prefix",
        "file_uri_not_double_prefixed",
        "local_and_file_uri_equivalent",
    ],
)
def test_get_fs_path_directory_info(tmp_path, root, time_unit, prefix, time_dir):
    def fill(s):
        return s.format(tmp=str(tmp_path), uri=tmp_path.as_uri())

    assert _get_fs_path_directory_info(fill(root), time_unit) == (str(Path(fill(prefix))) if "{tmp}" in prefix else prefix, fill(time_dir))


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


# CacheMode as a pydantic field


class _Model(BaseModel):
    cache_mode: CacheMode = CacheMode.CACHE


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("CACHE", CacheMode.CACHE),
        ("IGNORE", CacheMode.IGNORE),
        ("REBUILD", CacheMode.REBUILD),
        (1, CacheMode.CACHE),
        (2, CacheMode.IGNORE),
        (3, CacheMode.REBUILD),
        (CacheMode.CACHE, CacheMode.CACHE),
        (CacheMode.REBUILD, CacheMode.REBUILD),
        ("INVALID", KeyError),
        (999, ValueError),
        (3.14, ValueError),
    ],
    ids=[
        "validate_from_string-CACHE",
        "validate_from_string-IGNORE",
        "validate_from_string-REBUILD",
        "validate_from_int-1",
        "validate_from_int-2",
        "validate_from_int-3",
        "validate_from_enum-CACHE",
        "validate_from_enum-REBUILD",
        "validate_invalid_string",
        "validate_invalid_int",
        "validate_invalid_type",
    ],
)
def test_cachemode_validate(value, expected):
    if isinstance(expected, type):
        with pytest.raises(expected, match="Cannot convert value to CacheMode" if isinstance(value, float) else None):
            CacheMode.validate(value)
    else:
        assert CacheMode.validate(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("REBUILD", CacheMode.REBUILD), ("CACHE", CacheMode.CACHE), (1, CacheMode.CACHE), (3, CacheMode.REBUILD), (CacheMode.IGNORE, CacheMode.IGNORE)],
    ids=["from_string-REBUILD", "from_string-CACHE", "from_int-1", "from_int-3", "from_enum"],
)
def test_cachemode_in_pydantic_model(value, expected):
    model = _Model(cache_mode=value)
    assert model.cache_mode == expected
    assert model.model_dump(mode="python") == {"cache_mode": expected}
    assert model.model_dump(mode="json") == {"cache_mode": expected.name}
    assert model.model_dump_json() == f'{{"cache_mode":"{expected.name}"}}'


def test_cachemode_pydantic_json_schema_and_default():
    props = _Model.model_json_schema()["properties"]["cache_mode"]
    assert "enum" in props or "$ref" in props
    assert _Model().cache_mode == CacheMode.CACHE


# Layout: the files a write leaves and the metadata in them.


def LAY(id, time_unit, fmt, dates, files):
    return Case(id, None, files, {"time_unit": time_unit, "fmt": fmt, "dates": dates})


_JFM = [D(2023, 1, 15), D(2023, 2, 20), D(2023, 3, 25)]
LAYOUTS = [
    LAY("simple", "monthly", None, [D(2023, 1, 15)], ["monthly/2023-01.parquet"]),
    LAY("custom_partition_format-daily", "daily", "y=$year/m=$month/d=$day", [D(2023, 1, 15)], ["daily/y=2023/m=01/d=15.parquet"]),
    LAY("custom_partition_format-monthly", "monthly", "year=$year/month=$month", [D(2023, 2, 15)], ["monthly/year=2023/month=02.parquet"]),
    LAY("custom_partition_format-yearly", "yearly", "year=$year", [D(2023, 3, 15)], ["yearly/year=2023.parquet"]),
    LAY("multiple_partitions-monthly", "monthly", None, _JFM, ["monthly/2023-01.parquet", "monthly/2023-02.parquet", "monthly/2023-03.parquet"]),
    LAY("multiple_partitions-daily", "daily", None, _JFM, ["daily/2023-01-15.parquet", "daily/2023-02-20.parquet", "daily/2023-03-25.parquet"]),
    LAY("multiple_partitions-yearly", "yearly", None, _JFM, ["yearly/2023.parquet"]),
]
_DEFAULT_FORMAT = {"daily": "$year-$month-$day", "monthly": "$year-$month", "yearly": "$year"}


@pytest.mark.parametrize(("time_unit", "fmt", "dates", "files"), params(LAYOUTS, "time_unit", "fmt", "dates", "expect"))
def test_layout(tmp_path, time_unit, fmt, dates, files):
    source = _frame(date=dates, value=list(range(1, len(dates) + 1)))
    result = cache_parquet(source, tmp_path, "date", time_unit=time_unit, partition_format=fmt)
    assert isinstance(result, pl.LazyFrame)
    assert_frame_equal(result.collect(), source.collect(), check_row_order=False)
    written = cache_files(tmp_path)
    assert sorted(written) == files
    assert_frame_equal(pl.concat(written.values()), source.collect(), check_row_order=False)
    for path in files:
        meta = {k.decode(): v.decode() for k, v in pq.read_metadata(tmp_path / path).metadata.items()}
        assert meta["__piot__time_unit"] == time_unit
        assert meta["__piot__partition_format"] == (fmt or _DEFAULT_FORMAT[time_unit])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"partition_format": "$year"}, "missing"),
        ({"partition_format": "$year-$month-$day"}, "unexpected"),
        ({"partition_format": "$year-${month"}, None),
        ({"time_unit": "occasionally"}, "Invalid time unit"),
    ],
    ids=["missing", "unexpected", "malformed", "time_unit"],
)
def test_bad_partition_format(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _frame(date=[D(2023, 1, 15)], value=[42]).piot.cache_parquet(cache_path=tmp_path, date_column="date", **({"time_unit": "monthly"} | kwargs))


def test_complex(tmp_path):
    x = pl.datetime_range(DT(2023, 1, 1), DT(2024, 12, 31), interval="1d", eager=True)
    lf = pl.LazyFrame({"date": x, "feature": np.random.rand(x.shape[0]), "quantity": range(len(x)), "price": 100 + np.arange(x.shape[0])})
    pred = (pl.col("date") < DT(2024, 12, 15)) & (pl.col("date") > DT(2024, 10, 15))
    query = lf.piot.cache_parquet(cache_path=str(tmp_path), date_column="date").filter(pred).filter(pl.col("quantity") % 2 != 0)
    assert query.select(["date", "feature", "price"]).collect(engine="streaming").shape == (30, 3)
    assert {p: f.shape for p, f in cache_files(tmp_path).items()} == {
        f"monthly/2024-{m}.parquet": (n, 4) for m, n in [("10", 31), ("11", 30), ("12", 31)]
    }


# Steps: a cache is driven by a sequence of reads; then the files it holds are checked.


@dataclass(frozen=True)
class Step:
    """One read: ``source`` through the cache in ``mode``, filtered by ``predicate``.

    With ``compare_native`` the result equals native Polars on the same source and predicate; otherwise its ``value``
    column holds ``expected_values`` in any order.
    """

    source: Any
    _: KW_ONLY
    predicate: pl.Expr | None = None
    mode: CacheMode = CacheMode.CACHE
    compare_native: bool = True
    expected_values: list | None = None

    def __post_init__(self):
        assert self.compare_native == (self.expected_values is None), "set compare_native=False with expected_values"


@dataclass(frozen=True)
class Scenario:
    """A cache driven by ``steps`` in order, then holding ``expected_files`` (see ``assert_cache_files``)."""

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
        if step.compare_native:
            try:
                assert_frame_equal(got, collect_filtered(step.source, step.predicate), check_row_order=False)
            except AssertionError as exc:
                raise AssertionError(f"{context}: {exc}") from exc
        else:
            assert sorted(got["value"].to_list()) == sorted(step.expected_values), context


def _daily(days, values=None, month=6, **extra):
    days = [D(2024, month, d) for d in days]
    return _frame(date=days, value=values if values is not None else list(range(1, len(days) + 1)), **extra)


_RG = {"time_unit": "daily", "extra_partition_cols": "region"}
_MAY10 = _frame(date=[D(2024, 5, 10)] * 2, region=["NA", "EU"], value=[100, 200])
_MAY_GAP = _frame(date=[D(2024, 5, 10), D(2024, 5, 11)], region=["NA", "EU"], value=[100, 200])
_ONE_SIDED = _frame(date=[D(2024, 6, 1), D(2024, 6, 2), D(2024, 6, 5)], region=["US", "US", "EU"], value=[10, 20, 50])
_REGIONS = pl.col("region").is_in(["US", "EU", "APAC"])
_JAN_MAR = _frame(date=[D(2024, 1, 15), D(2024, 2, 15), D(2024, 3, 15)], value=[100, 200, 300])
_EMPTY = pl.LazyFrame(schema={"date": pl.Date, "value": pl.Int64})

STEPS = [
    # empty files mark queried partitions without rows
    Scenario(
        "missing_written",
        steps=[Step(_frame(date=[D(2023, 1, 15)], value=[1]), predicate=pl.col("date").is_between(D(2023, 1, 1), D(2023, 3, 31)))],
        expected_files={"monthly/2023-01.parquet": [1], "monthly/2023-02.parquet": [], "monthly/2023-03.parquet": []},
        only_expected_files=True,
    ),
    Scenario(
        "empty_partitions_with_extra_cols",
        steps=[Step(_MAY10, predicate=pl.col("region").is_in(["NA", "EU", "APAC"]) & (pl.col("date") == D(2024, 5, 10)))],
        expected_files={"daily/NA/2024-05-10.parquet": [100], "daily/EU/2024-05-10.parquet": [200], "daily/APAC/2024-05-10.parquet": []},
        only_expected_files=True,
        cache_kwargs=_RG,
    ),
    Scenario(
        "missing_written_daily_with_extra_cols",
        steps=[Step(_MAY10, predicate=pl.col("region").is_in(["NA", "EU", "APAC"]) & pl.col("date").is_between(D(2024, 5, 10), D(2024, 5, 11)))],
        expected_files={
            f"daily/{r}/2024-05-{d}.parquet": v
            for r, d, v in [("NA", 10, [100]), ("EU", 10, [200]), ("NA", 11, []), ("EU", 11, []), ("APAC", 10, []), ("APAC", 11, [])]
        },
        only_expected_files=True,
        cache_kwargs=_RG,
    ),
    Scenario(
        "missing_written_daily_with_extra_cols_gap",
        steps=[
            Step(
                _MAY_GAP,
                predicate=pl.col("region").is_in(["NA", "EU", "APAC"])
                & pl.col("date").is_between(D(2024, 5, 10), D(2024, 5, 14))
                & (pl.col("date") != D(2024, 5, 11)),
            )
        ],
        expected_files={f"daily/{r}/2024-05-11.parquet": None for r in ["NA", "EU", "APAC"]}
        | {"daily/NA/2024-05-10.parquet": [100], "daily/APAC/2024-05-14.parquet": []},
        cache_kwargs=_RG,
    ),
    Scenario(
        "missing_written_daily_with_extra_cols_gap2",
        steps=[Step(_MAY_GAP, predicate=pl.col("region").is_in(["EU", "NA"]) & pl.col("region").ne("NA") & pl.col("date").eq(D(2024, 5, 10)))],
        expected_files={"daily/NA/2024-05-10.parquet": None, "daily/EU/2024-05-10.parquet": []},
        only_expected_files=True,
        cache_kwargs=_RG,
    ),
    # one-sided reads write whole partitions that have rows, and no empty files beyond the cached range
    Scenario(
        "one_sided_with_extra_cols_no_empty_writes",
        steps=[Step(_ONE_SIDED), Step(_ONE_SIDED, predicate=(pl.col("date") >= D(2024, 6, 2)) & _REGIONS)],
        expected_files={
            "daily/US/2024-06-02.parquet": [20],
            "daily/EU/2024-06-05.parquet": [50],
            "daily/APAC/2024-06-02.parquet": None,
            "daily/APAC/2024-06-05.parquet": None,
        },
        cache_kwargs=_RG,
    ),
    Scenario(
        "one_sided_upper_bound_with_extra_cols",
        steps=[Step(_ONE_SIDED), Step(_ONE_SIDED, predicate=(pl.col("date") <= D(2024, 6, 2)) & _REGIONS)],
        expected_files={
            "daily/US/2024-06-01.parquet": [10],
            "daily/US/2024-06-02.parquet": [20],
            "daily/APAC/2024-06-01.parquet": None,
            "daily/APAC/2024-06-02.parquet": None,
        },
        cache_kwargs=_RG,
    ),
    Scenario(
        "one_sided_lower_bound_reads_cache_and_upstream",
        steps=[Step(_daily([1, 2])), Step(pl.concat([_daily([5], [5]), _daily([1, 2])]), predicate=pl.col("date") >= D(2024, 6, 2))],
        expected_files={"daily/2024-06-01.parquet": [1], "daily/2024-06-02.parquet": [2], "daily/2024-06-05.parquet": [5]},
        cache_kwargs={"time_unit": "daily"},
    ),
    Scenario(
        "unbounded_above_fills_gaps_up_to_existing_max",
        steps=[Step(_daily([1, 3, 5], [1, 3, 5])), Step(_daily([1, 3, 5], [1, 3, 5]), predicate=pl.col("date") >= D(2024, 6, 2))],
        expected_files={f"daily/2024-06-0{d}.parquet": v for d, v in [(1, [1]), (2, []), (3, [3]), (4, []), (5, [5]), (6, None), (7, None)]},
        only_expected_files=True,
        cache_kwargs={"time_unit": "daily"},
    ),
    Scenario(
        "unbounded_below_fills_gaps_down_to_existing_min",
        steps=[Step(_daily([3, 5, 7], [3, 5, 7])), Step(_daily([3, 5, 7], [3, 5, 7]), predicate=pl.col("date") <= D(2024, 6, 6))],
        expected_files={f"daily/2024-06-0{d}.parquet": v for d, v in [(1, None), (2, None), (3, [3]), (4, []), (5, [5]), (6, []), (7, [7])]},
        only_expected_files=True,
        cache_kwargs={"time_unit": "daily"},
    ),
    Scenario(
        "unbounded_with_extra_cols_fills_gaps",
        steps=[
            Step(_daily([1, 3], [1, 3], region=["US", "US"])),
            Step(_daily([1, 3], [1, 3], region=["US", "US"]), predicate=pl.col("date") >= D(2024, 6, 1)),
        ],
        expected_files={
            "daily/US/2024-06-01.parquet": [1],
            "daily/US/2024-06-02.parquet": [],
            "daily/US/2024-06-03.parquet": [3],
            "daily/US/2024-06-04.parquet": None,
        },
        only_expected_files=True,
        cache_kwargs=_RG,
    ),
    # REBUILD reads upstream, overwrites the partitions in scope, and keeps the others
    Scenario(
        "overwrite",
        steps=[Step(_frame(date=[D(2023, 1, 15)], value=[42])), Step(_frame(date=[D(2023, 1, 15)], value=[99]), mode=CacheMode.REBUILD)],
        expected_files={"monthly/2023-01.parquet": [99]},
        only_expected_files=True,
    ),
    Scenario(
        "rebuild_preserves_other_partitions",
        steps=[
            Step(_JAN_MAR),
            Step(
                _frame(date=[D(2024, 2, 20)], value=[999]), predicate=pl.col("date").is_between(D(2024, 2, 1), D(2024, 2, 28)), mode=CacheMode.REBUILD
            ),
        ],
        expected_files={"monthly/2024-01.parquet": [100], "monthly/2024-02.parquet": [999], "monthly/2024-03.parquet": [300]},
        only_expected_files=True,
        cache_kwargs={"time_unit": "monthly"},
    ),
    Scenario(
        "rebuild_overwrites_existing_partition",
        steps=[
            Step(_frame(date=[D(2024, 5, 10)], region=["US"], value=[100])),
            Step(
                _frame(date=[D(2024, 5, 10)], region=["US"], value=[999]),
                predicate=(pl.col("date") == D(2024, 5, 10)) & (pl.col("region") == "US"),
                mode=CacheMode.REBUILD,
            ),
        ],
        expected_files={"daily/US/2024-05-10.parquet": [999]},
        only_expected_files=True,
        cache_kwargs=_RG,
    ),
    Scenario(
        "rebuild_writes_empty_files_for_missing_data",
        steps=[Step(_daily([1, 3], [10, 30]), predicate=pl.col("date").is_between(D(2024, 6, 1), D(2024, 6, 3)), mode=CacheMode.REBUILD)],
        expected_files={"daily/2024-06-01.parquet": [10], "daily/2024-06-02.parquet": [], "daily/2024-06-03.parquet": [30]},
        only_expected_files=True,
        cache_kwargs={"time_unit": "daily"},
    ),
    Scenario(
        "rebuild_overwrites_previously_empty_partition",
        steps=[
            Step(_daily([1, 3], [10, 30]), predicate=pl.col("date").is_between(D(2024, 6, 1), D(2024, 6, 3))),
            Step(_daily([1, 2, 3], [10, 20, 30]), predicate=pl.col("date").is_between(D(2024, 6, 1), D(2024, 6, 3)), mode=CacheMode.REBUILD),
        ],
        expected_files={"daily/2024-06-01.parquet": [10], "daily/2024-06-02.parquet": [20], "daily/2024-06-03.parquet": [30]},
        only_expected_files=True,
        cache_kwargs={"time_unit": "daily"},
    ),
    Scenario(
        "rebuild_with_extra_cols_preserves_other_regions",
        steps=[
            Step(_frame(date=[D(2024, 8, 1)] * 2, region=["US", "EU"], value=[100, 200])),
            Step(
                _frame(date=[D(2024, 8, 1)], region=["US"], value=[999]),
                predicate=(pl.col("date") == D(2024, 8, 1)) & (pl.col("region") == "US"),
                mode=CacheMode.REBUILD,
            ),
        ],
        expected_files={"daily/US/2024-08-01.parquet": [999], "daily/EU/2024-08-01.parquet": [200]},
        only_expected_files=True,
        cache_kwargs=_RG,
    ),
    Scenario(
        "rebuild_no_date_column",
        steps=[
            Step(_frame(region=["US", "EU"], value=[100, 200])),
            Step(_frame(region=["US"], value=[999]), predicate=pl.col("region") == "US", mode=CacheMode.REBUILD),
        ],
        expected_files={"null/US.parquet": [999], "null/EU.parquet": [200]},
        only_expected_files=True,
        cache_kwargs={"date_column": None, "extra_partition_cols": "region"},
    ),
    Scenario(
        "rebuild_empty_to_data",
        steps=[
            Step(_EMPTY, predicate=pl.col("date") == D(2024, 9, 1), mode=CacheMode.REBUILD),
            Step(_frame(date=[D(2024, 9, 1)], value=[42]), predicate=pl.col("date") == D(2024, 9, 1), mode=CacheMode.REBUILD),
        ],
        expected_files={"daily/2024-09-01.parquet": [42]},
        only_expected_files=True,
        cache_kwargs={"time_unit": "daily"},
    ),
    Scenario(
        "rebuild_data_to_empty",
        steps=[Step(_frame(date=[D(2024, 10, 1)], value=[42])), Step(_EMPTY, predicate=pl.col("date") == D(2024, 10, 1), mode=CacheMode.REBUILD)],
        expected_files={"daily/2024-10-01.parquet": []},
        only_expected_files=True,
        cache_kwargs={"time_unit": "daily"},
    ),
    # reads served from the cache
    Scenario(
        "between_pred_uses_existing_partitions",
        steps=[
            Step(_frame(date=[D(2021, 3, 1), D(2021, 3, 2)], value=[10, 20]), predicate=pl.col("date") == D(2021, 3, 1)),
            Step(_frame(date=[D(2021, 3, 1), D(2021, 3, 2)], value=[10, 20]), predicate=pl.col("date") == D(2021, 3, 2)),
            Step(
                _frame(date=[D(2021, 3, 1), D(2021, 3, 2)], value=[0, 0]),
                predicate=pl.col("date").is_between(D(2021, 3, 1), D(2021, 3, 2)),
                compare_native=False,
                expected_values=[10, 20],
            ),
        ],
        expected_files={"daily/2021-03-01.parquet": [10], "daily/2021-03-02.parquet": [20]},
        only_expected_files=True,
        cache_kwargs={"time_unit": "daily"},
    ),
    Scenario(
        "custom_partition_format_cache_readback",
        steps=[Step(_daily([1, 2, 3], [10, 20, 30])), Step(_daily([4, 5], [40, 50]), compare_native=False, expected_values=[10, 20, 30, 40, 50])],
        expected_files={f"daily/theYear=2024/theMonth=06/theDay=0{d}.parquet": [10 * d] for d in range(1, 6)},
        only_expected_files=True,
        cache_kwargs={"time_unit": "daily", "partition_format": "theYear=$year/theMonth=$month/theDay=$day"},
    ),
    Scenario(
        "empty_result_from_cache",
        steps=[Step(_frame(date=[D(2024, 2, 1)], value=[1]), predicate=pl.col("date") == D(2025, 1, 1))],
        expected_files={"monthly/2025-01.parquet": []},
        only_expected_files=True,
    ),
    Scenario(
        "extra_partition_columns",
        steps=[Step(_frame(date=[D(2024, 1, 3), D(2024, 1, 4)] * 2, country=["US", "CA"] * 2, value=[1, 2, 3, 4]))],
        expected_files={"monthly/US/2024-01.parquet": [1, 3], "monthly/CA/2024-01.parquet": [2, 4]},
        only_expected_files=True,
        cache_kwargs={"extra_partition_cols": ["country"]},
    ),
    # no date column: one file per extra-column value under null/, or null/data.parquet
    Scenario(
        "cache_parquet_no_date_single_extra_col",
        steps=[
            Step(_frame(region=["US", "US", "EU", "EU", "APAC"], value=[100, 200, 150, 250, 180])),
            Step(_frame(region=["US"], value=[0]), compare_native=False, expected_values=[100, 200, 150, 250, 180]),
        ],
        expected_files={"null/US.parquet": [100, 200], "null/EU.parquet": [150, 250], "null/APAC.parquet": [180]},
        only_expected_files=True,
        cache_kwargs={"date_column": None, "extra_partition_cols": "region"},
    ),
    Scenario(
        "cache_parquet_no_date_multiple_extra_cols",
        steps=[Step(_frame(region=["US", "US", "EU", "EU", "APAC", "APAC"], product=["A", "B"] * 3, value=[100, 200, 150, 250, 180, 220]))],
        expected_files={
            f"null/{r}/{p}.parquet": [v] for (r, p), v in zip([(r, p) for r in ["US", "EU", "APAC"] for p in "AB"], [100, 200, 150, 250, 180, 220])
        },
        only_expected_files=True,
        cache_kwargs={"date_column": None, "extra_partition_cols": ["region", "product"]},
    ),
    Scenario(
        "cache_parquet_no_date_no_extra_cols",
        steps=[
            Step(_frame(value=[100, 200, 150], name=["Alice", "Bob", "Charlie"])),
            Step(_frame(value=[0], name=["x"]), compare_native=False, expected_values=[100, 200, 150]),
        ],
        expected_files={"null/data.parquet": [100, 200, 150]},
        only_expected_files=True,
        cache_kwargs={"date_column": None},
    ),
    Scenario(
        "cache_parquet_no_date_with_predicate",
        steps=[
            Step(_frame(region=["US", "US", "EU", "EU", "APAC"], category=list("XYXYX"), value=[100, 200, 150, 250, 180])),
            Step(
                _frame(region=["US", "US", "EU", "EU", "APAC"], category=list("XYXYX"), value=[100, 200, 150, 250, 180]).filter(
                    pl.col("region") == "US"
                ),
                compare_native=False,
                expected_values=[100, 200, 150, 250, 180],
            ),
        ],
        expected_files={"null/US/X.parquet": [100], "null/APAC/X.parquet": [180]},
        cache_kwargs={"date_column": None, "extra_partition_cols": ["region", "category"]},
    ),
    Scenario(
        "cache_parquet_no_date_different_cache_modes",
        steps=[
            Step(_frame(category=list("ABAB"), value=[100, 200, 150, 250]), mode=CacheMode.IGNORE),
            Step(_frame(category=list("ABAB"), value=[100, 200, 150, 250]), mode=CacheMode.CACHE),
            Step(_frame(category=list("ABAB"), value=[100, 200, 150, 250]), mode=CacheMode.REBUILD),
        ],
        expected_files={"null/A.parquet": [100, 150], "null/B.parquet": [200, 250]},
        only_expected_files=True,
        cache_kwargs={"date_column": None, "extra_partition_cols": ["category"]},
    ),
    Scenario(
        "no_date_with_extra_unconstrained_queries_upstream_for_new_partitions",
        steps=[
            Step(_frame(region=["US"], value=[1])),
            Step(_frame(region=["EU"], value=[2]), compare_native=False, expected_values=[1, 2]),
            Step(_frame(region=["EU"], value=[3]), compare_native=False, expected_values=[1, 2]),
        ],
        expected_files={"null/US.parquet": [1], "null/EU.parquet": [2]},
        only_expected_files=True,
        cache_kwargs={"date_column": None, "extra_partition_cols": "region"},
    ),
    Scenario(
        "sequential_cache_parquet",
        steps=[Step(_frame(date=[D(2024, 1, 1), D(2024, 1, 2), D(2024, 2, 1)], value=[1, 3, 5]))],
        expected_files={"monthly/2024-01.parquet": [1, 3], "monthly/2024-02.parquet": [5]},
    ),
]


@pytest.mark.parametrize("scenario", [pytest.param(sc, id=sc.id) for sc in STEPS])
def test_steps(tmp_path, scenario):
    run_steps(tmp_path, scenario.steps, **{"date_column": "date"} | scenario.cache_kwargs)
    assert_cache_files(tmp_path, scenario.expected_files, exact=scenario.only_expected_files)


def test_sequential_cache_parquet_chained(tmp_path):
    base = _frame(date=[D(2024, 1, 1), D(2024, 1, 2), D(2024, 2, 1)], x=[1, 3, 5])
    assert base.piot.cache_parquet(tmp_path, "date").piot.cache_parquet(tmp_path, "date").collect().shape == (3, 2)


def test_no_date_metadata(tmp_path):
    for name, extra, by in [("with_extra", ["region"], b"region"), ("single", None, b"")]:
        cache_parquet(_frame(region=["US", "EU"], value=[100, 200]), tmp_path / name, extra_partition_cols=extra).collect()
        for path in (tmp_path / name).rglob("*.parquet"):
            meta = pq.ParquetFile(path).metadata.metadata
            assert (meta[b"__piot__time_unit"], meta[b"__piot__partitioned_by"]) == (b"null", by)


# Every cache mode over the shared predicate shapes and temporal dtypes: what a read returns equals native Polars,
# cold and warm. CACHE_ONLY returns nothing from an empty cache and serves a full one; IGNORE leaves no files.


@pytest.mark.parametrize("mode", [m for m in MODES if m != CacheMode.CACHE_ONLY], ids=lambda m: m.name)
@pytest.mark.parametrize("pred", params(DATE_PREDICATES, "pred"))
def test_mode_matches_native_by_predicate(tmp_path, mode, pred):
    assert_cached_matches_native(daily_frame().lazy(), tmp_path, pred, date_column="date", time_unit="daily", cache_mode=mode)
    if mode == CacheMode.IGNORE:
        assert not cache_files(tmp_path)


@pytest.fixture(scope="module")
def full_daily_cache(tmp_path_factory):
    root = tmp_path_factory.mktemp("full_daily_cache")
    cache_parquet(daily_frame().lazy(), root, "date", time_unit="daily").collect()
    return root


@pytest.mark.parametrize("pred", params(DATE_PREDICATES, "pred"))
def test_cache_only_matches_native_by_predicate(tmp_path, full_daily_cache, pred):
    kwargs = {"date_column": "date", "time_unit": "daily", "cache_mode": CacheMode.CACHE_ONLY}
    assert collect_filtered(cache_parquet(daily_frame().lazy(), tmp_path, **kwargs), pred).is_empty()
    assert_cached_matches_native(daily_frame().lazy(), full_daily_cache, pred, **kwargs)


@pytest.mark.parametrize("dtype", [d for d in TEMPORAL_DTYPES if d != pl.Date], ids=str)
@pytest.mark.parametrize("pred", params([c for c in DATE_PREDICATES if c.id in ("between", "ne_in_window")], "pred"))
def test_mode_matches_native_by_dtype(tmp_path, dtype, pred):
    assert_cached_matches_native(daily_frame(dtype=dtype).lazy(), tmp_path, pred, date_column="date", time_unit="monthly")


# Upstream calls: what the cache asks its source for.


def test_callable_not_invoked_when_cache_satisfies(tmp_path):
    source = _frame(date=[D(2023, 1, 1), D(2023, 1, 2)], region=["US", "US"], value=[10, 20])
    kwargs = {"date_column": "date", "time_unit": "daily", "extra_partition_cols": ["region"]}
    cache_parquet(source, tmp_path, **kwargs).collect()
    calls = []

    def make_source():
        calls.append(1)
        return source

    covered = (pl.col("region") == "US") & pl.col("date").is_between(D(2023, 1, 1), D(2023, 1, 2))
    assert_cached_matches_native(source, tmp_path, covered, runs=1, **kwargs)
    assert sorted(cache_parquet(make_source, tmp_path, **kwargs).filter(covered).collect()["value"]) == [10, 20]
    assert calls == []
    # Without a date restriction new dates may exist upstream, so the source is built.
    assert sorted(cache_parquet(make_source, tmp_path, **kwargs).filter(pl.col("region") == "US").collect()["value"]) == [10, 20]
    assert calls == [1]


_PROBE = pl.DataFrame({"date": pl.date_range(D(2024, 5, 31), D(2024, 6, 6), eager=True)})


def UP(id, seed, upstream, pred, calls, kept=None, **kwargs):
    """``seed`` is cached first; then ``upstream`` is read through the cache with ``pred``.

    ``calls`` is the number of upstream calls; ``kept`` is what the last pushed predicate keeps of ``_PROBE``.
    """
    return Case(id, pred, (calls, kept), {"seed": seed, "upstream": upstream, "kwargs": {"date_column": "date", "time_unit": "daily"} | kwargs})


_D123 = pl.DataFrame({"date": [D(2024, 6, 1), D(2024, 6, 2), D(2024, 6, 3)], "value": [1, 2, 3]})
_D345 = pl.DataFrame({"date": [D(2024, 6, 3), D(2024, 6, 4), D(2024, 6, 5)], "value": [3, 4, 5]})
UPSTREAM = [
    UP("predicate_pushdown", None, _D123, pl.col("date") == D(2024, 6, 1), 1, [D(2024, 6, 1)]),
    UP(
        "write_uses_widened_predicate",
        None,
        _D123,
        pl.col("date") == D(2024, 6, 15),
        1,
        pl.date_range(D(2024, 6, 1), D(2024, 6, 6), eager=True).to_list(),
        time_unit="monthly",
    ),
    UP(
        "one_sided_lower_bound_upstream_predicate_shape",
        _D123,
        pl.DataFrame({"date": [D(2024, 6, 5)], "value": [5]}),
        pl.col("date") >= D(2024, 6, 2),
        1,
        [D(2024, 6, 4), D(2024, 6, 5), D(2024, 6, 6)],
    ),
    UP(
        "one_sided_lower_bound_upstream_contains_not_existing_predicate",
        _D123.head(2),
        pl.DataFrame({"date": [D(2024, 6, 5)], "value": [5]}),
        pl.col("date") >= D(2024, 6, 2),
        1,
        [D(2024, 6, 3), D(2024, 6, 4), D(2024, 6, 5), D(2024, 6, 6)],
    ),
    UP(
        "one_sided_upper_bound_upstream_predicate_shape",
        _D345,
        pl.DataFrame({"date": [D(2024, 6, 2)], "value": [2]}),
        pl.col("date") <= D(2024, 6, 4),
        1,
        [D(2024, 5, 31), D(2024, 6, 1), D(2024, 6, 2)],
    ),
    UP(
        "rebuild_queries_upstream_not_cache",
        _D123,
        _D123.with_columns(value=pl.lit(9, pl.Int64)),
        pl.col("date") == D(2024, 6, 1),
        1,
        [D(2024, 6, 1)],
        cache_mode=CacheMode.REBUILD,
    ),
    UP("contradiction", _D123, _D123, pl.col("date").is_between(D(2024, 6, 1), D(2024, 6, 2)), 0),
    UP("no_date_no_extra_first_writes_second_reads_cache", _D123, _D123, None, 0, date_column=None),
]


@pytest.mark.parametrize(("seed", "upstream", "pred", "expect", "kwargs"), params(UPSTREAM, "seed", "upstream", "pred", "expect", "kwargs"))
def test_upstream_calls(tmp_path, seed, upstream, pred, expect, kwargs):
    calls, kept = expect
    if seed is not None:
        cache_parquet(seed.lazy(), tmp_path, **kwargs | {"cache_mode": CacheMode.CACHE}).collect()
    spy = UpstreamSpy(upstream)
    result = collect_filtered(cache_parquet(spy.lazy, tmp_path, **kwargs), pred)
    assert spy.calls == calls
    if kept is not None:
        assert spy.kept(_PROBE)["date"].to_list() == kept
    if seed is None or kwargs.get("cache_mode") == CacheMode.REBUILD:
        assert_frame_equal(result, collect_filtered(upstream.lazy(), pred), check_row_order=False)


def test_debug_source_sees_pushed_partition_predicates(tmp_path, capsys):
    """Through ``piot.debug`` upstream: a model predicate reaches the source, and a second unconstrained read of a
    no-date cache with partition columns still asks upstream for new partitions."""
    model = _frame(date=[D(2025, 5, 1), D(2025, 5, 2), D(2025, 5, 3)], model=["USE4S", "OTHER", "USE4S"], val=[10, 20, 30]).piot.debug()
    model.piot.cache_parquet(cache_path=tmp_path / "m", date_column="date", extra_partition_cols="model").filter(pl.col("model") == "USE4S").collect()
    out = capsys.readouterr().out
    assert 'col("model")' in out and '"USE4S"' in out

    cached = (
        _frame(region=["US", "EU", "US"], value=[1, 2, 3]).piot.debug().piot.cache_parquet(cache_path=tmp_path / "r", extra_partition_cols="region")
    )
    first = cached.filter(pl.col("value") >= 2).sort(["region", "value"]).collect()
    assert first.to_dict(as_series=False) == {"region": ["EU", "US"], "value": [2, 3]}
    assert "predicate=" in capsys.readouterr().out
    assert_frame_equal(cached.filter(pl.col("value") >= 2).sort(["region", "value"]).collect(), first)
    assert "predicate=" in capsys.readouterr().out

    lf = _frame(value=[1, 2, 3]).piot.debug().piot.cache_parquet(cache_path=tmp_path / "n")
    lf.collect()
    assert "debug called" in capsys.readouterr().out
    lf.collect()
    assert "debug called" not in capsys.readouterr().out


def test_push_and_parse_exclusions_to_custom_io_source(tmp_path):
    """A wider second query asks upstream only for the region it has not cached."""
    from polars_io_tools.io_sources.range_visitor import convert_expr_to_datetime_range
    from polars_io_tools.io_sources.set_visitor import convert_expr_to_valid_values

    spy = UpstreamSpy(_MAY_GAP.collect())
    kwargs = {"date_column": "date", **_RG}
    pred = (pl.col("region") == "NA") & (pl.col("date") == D(2024, 5, 10))
    assert_frame_equal(cache_parquet(spy.lazy, tmp_path, **kwargs).filter(pred).collect(), _MAY_GAP.filter(pred).collect())
    assert_cache_files(tmp_path, {"daily/NA/2024-05-10.parquet": [100]}, exact=True)
    wider = pl.col("region").is_in(["NA", "EU"]) & (pl.col("date") == D(2024, 5, 10))
    assert_frame_equal(cache_parquet(spy.lazy, tmp_path, **kwargs).filter(wider).collect(), _MAY_GAP.filter(wider).collect())
    pushed = spy.predicates[-1]
    assert {e.meta.serialize(format="json") for e in wider.meta.pop()} != {e.meta.serialize(format="json") for e in pushed.meta.pop()}
    assert convert_expr_to_datetime_range(pushed, "date", get_enclosure=False).lower == DT(2024, 5, 10)
    assert convert_expr_to_valid_values(pushed, "region") == {"EU"}


@pytest.mark.parametrize("partition_format", [None, "theYear=$year/theMonth=$month/theDay=$day"])
def test_open_ended_filters_query_expected_partitions(tmp_path, caplog, partition_format):
    import logging

    caplog.set_level(logging.DEBUG, logger="polars_io_tools.io_sources.lazy_cache_parquet")
    collected, appended, filtered = exercise_daily_cache_parquet(cache_root=str(tmp_path), partition_format=partition_format)
    assert collected.to_dict(as_series=False) == {"date": [D(2024, 6, 1), D(2024, 6, 2), D(2024, 6, 3)], "value": [11, 22, 33]}
    assert appended.to_dict(as_series=False) == {"date": [D(2024, 6, 5)], "value": [44]}
    assert filtered.to_dict(as_series=False) == {"date": [D(2024, 6, 2), D(2024, 6, 3)], "value": [22, 33]}
    assert pl.concat(cache_files(tmp_path).values())["date"].sort().to_list() == [D(2024, 6, d) for d in (1, 2, 3, 5)]
    assert "Cache scan uses" in caplog.text and "enumerated" in caplog.text
    lower = cache_parquet(
        _frame(date=[D(2024, 6, 1), D(2024, 6, 2), D(2024, 6, 3)], value=[11, 22, 33]),
        tmp_path,
        "date",
        time_unit="daily",
        partition_format=partition_format,
    )
    assert lower.filter(pl.col("date") >= D(2024, 6, 2)).sort("date").collect()["value"].to_list() == [22, 33, 44]


@pytest.mark.parametrize("run_first", ["collect", "no_collect"])
def test_intermediate_collect(tmp_path, run_first):
    """An intermediate collect does not change what chained caches return or write."""

    def run(base, do_collect: bool):
        if base.exists():
            shutil.rmtree(base)
        ldf = _frame(a=[1, 2, 3], date=[D(2025, 7, 31)] * 3).piot.cache_parquet(base / "foo_test", "date", time_unit="daily")
        if do_collect:
            ldf.collect()
        ldf1 = ldf.with_columns(pl.lit("foo").alias("foo")).piot.cache_parquet(base / "foo_test1", "date", time_unit="daily")
        ldf2 = ldf.with_columns(pl.lit("bar").alias("bar")).piot.cache_parquet(base / "foo_test2", "date", time_unit="daily")
        return ldf1.join(ldf2, on="a").collect(), cache_files(base)

    order = [False, True] if run_first == "no_collect" else [True, False]
    (res_a, cache_a), (res_b, cache_b) = (run(tmp_path / f"scenario_{c}", c) for c in order)
    assert_frame_equal(res_a, res_b, check_row_order=False)
    assert set(cache_a) == set(cache_b)
    for k in cache_a:
        assert_frame_equal(cache_a[k], cache_b[k], check_row_order=False)


_REGION_DAYS = pl.LazyFrame({"date": pl.date_range(D(2020, 1, 1), D(2026, 1, 1), eager=True)}).with_columns(
    region=pl.col.date.dt.day().replace_strict({1: "NA", 2: "EU", 3: "CA", 4: "NA", 6: "NA"}, default="US", return_dtype=pl.Utf8), value=pl.lit(100)
)


@pytest.mark.parametrize(
    ("small_end", "big", "scan"),
    [
        (D(2020, 6, 1), (D(2019, 12, 15), D(2020, 9, 14)), None),
        (D(2021, 2, 1), (D(2020, 1, 1), D(2024, 2, 1)), "Cache scan uses glob"),
    ],
    ids=["large_partition_query", "huge_partition_query"],
)
def test_partition_query_over_many_partitions(tmp_path, caplog, monkeypatch, small_end, big, scan):
    """A partly cached query over many partitions returns every row; a huge one scans the cache by glob."""
    import logging

    # More partitions than the default open-sink cap (128).
    monkeypatch.setenv("POLARS_MAX_OPEN_SINKS", "8192")
    caplog.set_level(logging.DEBUG, logger="polars_io_tools.io_sources.lazy_cache_parquet")
    kwargs = {"date_column": "date", **_RG}
    assert_cached_matches_native(_REGION_DAYS, tmp_path, pl.col.region.eq("US") & pl.col.date.is_between(D(2020, 1, 1), small_end), runs=1, **kwargs)
    assert_cached_matches_native(_REGION_DAYS, tmp_path, pl.col.region.is_in(["US", "NA"]) & pl.col.date.is_between(*big), runs=1, **kwargs)
    if scan:
        assert scan in caplog.text


def test_partition_count_exceeds_open_sinks_raises(tmp_path, monkeypatch):
    """Writing more partitions than the open-sink cap fails loud instead of dropping rows."""
    monkeypatch.setenv("POLARS_MAX_OPEN_SINKS", "4")
    df = _frame(date=pl.date_range(D(2020, 1, 1), D(2020, 1, 10), eager=True), value=range(10))
    with pytest.raises(Exception, match="open-sink cap"):
        cache_parquet(df, tmp_path, "date", time_unit="daily").filter(pl.col.date.is_between(D(2020, 1, 1), D(2020, 1, 10))).collect()


def test_contradiction_writes_once(monkeypatch, tmp_path):
    """A repeated read of a cached range does not sink again."""
    base = daily_frame(31, D(2024, 1, 1)).lazy()
    pred = pl.col("date").is_between(D(2024, 1, 10), D(2024, 1, 20))
    sinks = []
    real_sink = pl.LazyFrame.sink_parquet
    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", lambda self, *a, **k: sinks.append(1) or real_sink(self, *a, **k))
    cached = cache_parquet(base, tmp_path, "date", time_unit="monthly")
    cached.filter(pred).collect()
    cached.filter(pred).collect()
    assert sinks == [1]


# write_bounding_columns: an opt-in that writes only the rows the predicate keeps on the listed columns.

_BOUNDED_SOURCE = pl.DataFrame({"date": [D(2023, 1, 10)] * 5, "x": [1, 2, 3, 4, 5], "y": [100, 200, 300, 400, 500], "value": [10, 20, 30, 40, 50]})


def WB(id, pred, result, on_disk, mode=CacheMode.REBUILD, bounding=("x",)):
    return Case(id, pred, (result, on_disk), {"mode": mode, "bounding": list(bounding) if bounding else None})


WRITE_BOUNDING = [
    WB("bounded_write_restricts_on_disk_rows", pl.col("x").is_in([1, 2]), [1, 2], [1, 2]),
    WB("unbounded_write_keeps_all_rows", pl.col("x").is_in([1, 2]), [1, 2], [1, 2, 3, 4, 5], bounding=None),
    WB("bounded_and_unbounded_readback_identity", pl.col("x").is_in([2, 4]), [2, 4], [2, 4]),
    WB("cache_mode_allowed_with_write_bounding_columns", pl.col("x").is_in([1, 2]), [1, 2], [1, 2], CacheMode.CACHE),
    WB("rebuild_mode_allowed", pl.col("x").is_in([1]), [1], [1]),
    WB("ignore_mode_passthrough", pl.col("x").is_in([1, 2]), [1, 2], None, CacheMode.IGNORE),
    WB("predicate_not_restrictable_degrades_to_unbounded", pl.col("y").is_in([100, 200]), [1, 2], [1, 2, 3, 4, 5]),
    WB("empty_subset_pins_current_behavior", pl.col("x").is_in([]), [], None),
    # Whether an empty file is written depends on the Polars version.
    WB("empty_subset_unbounded_is_clean", pl.col("x").is_in([]), [], ..., bounding=None),
]


@pytest.mark.parametrize(("pred", "expect", "mode", "bounding"), params(WRITE_BOUNDING, "pred", "expect", "mode", "bounding"))
def test_write_bounding_columns(tmp_path, pred, expect, mode, bounding):
    result, on_disk = expect
    out = cache_parquet(lambda: _BOUNDED_SOURCE.lazy(), tmp_path, "date", cache_mode=mode, write_bounding_columns=bounding).filter(pred).collect()
    assert_frame_equal(out, _BOUNDED_SOURCE.filter(pl.col("x").is_in(result)), check_row_order=False)
    files = cache_files(tmp_path)
    if on_disk is ...:
        return
    if on_disk is None:
        assert not files
    else:
        assert sorted(files) == ["monthly/2023-01.parquet"]
        assert sorted(files["monthly/2023-01.parquet"]["x"]) == on_disk


def test_write_bounding_columns_default_off_is_identical_to_not_passing(tmp_path):
    cache_parquet(lambda: _BOUNDED_SOURCE.lazy(), tmp_path / "omitted", "date").collect()
    cache_parquet(lambda: _BOUNDED_SOURCE.lazy(), tmp_path / "none", "date", write_bounding_columns=None).collect()
    omitted, none = cache_files(tmp_path / "omitted"), cache_files(tmp_path / "none")
    assert omitted and set(omitted) == set(none)
    for rel, frame in omitted.items():
        assert_frame_equal(frame, none[rel])


# Non-enumerable partition predicates (!=, wildcards) on a no-date cache, and the CACHE-mode rule that existing
# partitions are never rewritten.

_SYMBOLS = pl.DataFrame(
    {"date": [D(2024, 1, 1)] * 5, "symbol": ["s0", "s1", "s2", "s3", "s4"], "region": ["a", "a", "b", "b", "b"], "v": [1, 2, 3, 4, 5]},
    schema={"date": pl.Date, "symbol": pl.String, "region": pl.String, "v": pl.Int64},
)
_SYM_KW = {"date_column": None, "time_unit": "daily", "extra_partition_cols": ["symbol", "region"], "schema": _SYMBOLS.schema}
_s = pl.col("symbol")


@pytest.mark.parametrize("populated", [True, False], ids=["populated_cache_read_correct", "incomplete_cache_populates_missing"])
@pytest.mark.parametrize(
    "pred",
    [
        _s != "s0",
        _s.is_in(["s1", "s2", "s3", "s4"]),
        pl.col("v") > 2,
        (_s != "s0") | (_s == "s0"),
        (_s != "s0") & (pl.col("region") == "b"),
        (_s == "s0") & (_s == "s1"),
    ],
    ids=["ne", "positive_control", "non_partition", "always_true_or", "partial_wildcard", "contradiction_returns_empty"],
)
def test_non_enumerable_predicate(tmp_path, populated, pred):
    if populated:
        cache_parquet(lambda: _SYMBOLS.lazy(), tmp_path, **_SYM_KW).collect()
    assert_cached_matches_native(_SYMBOLS.lazy(), tmp_path, pred, runs=1, **_SYM_KW)


def test_populated_cache_not_rewritten(tmp_path):
    """CACHE mode must not rewrite existing partition files on a filtered read."""
    cache_parquet(_SYMBOLS.lazy(), tmp_path, **_SYM_KW).collect()
    before = {p: (tmp_path / p).stat().st_mtime_ns for p in cache_files(tmp_path)}
    time.sleep(0.05)
    for pred in (pl.col("v") > 2, _s != "s0"):
        cache_parquet(_SYMBOLS.lazy(), tmp_path, **_SYM_KW).filter(pred).collect()
    assert {p: (tmp_path / p).stat().st_mtime_ns for p in cache_files(tmp_path)} == before


def test_date_partitioned_negative_predicate_reads_cache(tmp_path):
    """On a populated date-partitioned cache, a != on an extra column reads the cache and not a changed upstream."""
    orig = _SYMBOLS.head(3).drop("region")
    state = {"df": orig}
    kwargs = {"date_column": "date", "time_unit": "daily", "extra_partition_cols": "symbol", "schema": orig.schema}
    cache_parquet(lambda: state["df"].lazy(), tmp_path, **kwargs).collect()
    state["df"] = orig.with_columns(pl.col("v") + 100)
    assert cache_parquet(lambda: state["df"].lazy(), tmp_path, **kwargs).filter(_s != "s0").collect().sort("symbol")["v"].to_list() == [2, 3]


_SYM3 = pl.DataFrame({"symbol": ["s0", "s1", "s2"], "v": [1, 2, 3]}, schema={"symbol": pl.String, "v": pl.Int64})


@pytest.mark.parametrize(
    ("refreshed", "expected"),
    [
        (_SYM3.with_columns(pl.col("v") + 100), {"s0": 1, "s1": 102, "s2": 103}),
        (pl.DataFrame({"symbol": ["s0", "s1"], "v": [1, 20]}), {"s0": 1, "s1": 20}),
    ],
    ids=["rebuild_wildcard_predicate_leaves_out_of_scope_partition", "rebuild_wildcard_removes_partition_that_disappeared_upstream"],
)
def test_rebuild_wildcard(tmp_path, refreshed, expected):
    """REBUILD with ``symbol != 's0'`` refreshes in-scope partitions, drops those upstream lost, and leaves s0 alone."""
    state = {"df": _SYM3}

    def mk(mode):
        return cache_parquet(lambda: state["df"].lazy(), tmp_path, extra_partition_cols="symbol", schema=_SYM3.schema, cache_mode=mode)

    mk(CacheMode.CACHE).collect()
    state["df"] = refreshed
    mk(CacheMode.REBUILD).filter(_s != "s0").collect()
    got = mk(CacheMode.CACHE).collect().sort("symbol")
    assert dict(zip(got["symbol"], got["v"])) == expected


def test_wildcard_read_matching_no_files_returns_empty(tmp_path):
    """A wildcard predicate that matches no cached files returns an empty frame, not an error."""
    data = _SYM3.head(1)
    lf = cache_parquet(lambda: data.lazy(), tmp_path, extra_partition_cols="symbol", schema=data.schema)
    for pred in (_s != "s0", pl.lit(False)):
        assert_frame_equal(lf.filter(pred).collect(), data.clear())


def test_rebuild_one_sided_empties_dropped_day_within_bounds(tmp_path):
    """Unlike a no-date cache, a dropped day inside the cached date bounds is kept as an empty file."""
    source = _daily([1, 2, 3])
    cache_parquet(source, tmp_path, "date", time_unit="daily").collect()
    rebuilt = source.filter(pl.col("date") != D(2024, 6, 2))
    out = cache_parquet(rebuilt, tmp_path, "date", time_unit="daily", cache_mode=CacheMode.REBUILD).filter(pl.col("date") >= D(2024, 6, 1)).collect()
    assert out.sort("date")["value"].to_list() == [1, 3]
    assert_cache_files(tmp_path, {"daily/2024-06-02.parquet": []})


def T(*args) -> datetime.datetime:
    return datetime.datetime(*args, tzinfo=UTC)


START = D(2026, 9, 1)


def _source(end: datetime.date, value: int = 1, start: datetime.date = START) -> pl.LazyFrame:
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


OCT = pl.col("date").is_between(D(2026, 10, 1), D(2026, 10, 31))


@pytest.mark.parametrize(
    ("written_at", "period_end", "now", "ttl", "final_after", "expected"),
    [
        (T(2026, 10, 1), None, T(2026, 10, 2), timedelta(days=1), None, True),
        (T(2026, 10, 1), None, T(2026, 10, 1, 23), timedelta(days=1), None, False),
        (T(2026, 11, 4), T(2026, 11, 1), T(2027, 1, 1), timedelta(hours=1), timedelta(days=3), False),
        (T(2026, 11, 3), T(2026, 11, 1), T(2026, 11, 3, 2), timedelta(hours=1), timedelta(days=3), True),
        (T(2026, 11, 3), T(2026, 11, 1), T(2026, 11, 3), None, timedelta(days=3), True),
        (T(2026, 11, 3), None, T(2026, 11, 3), None, timedelta(days=3), True),
    ],
)
def test_default_expired(written_at, period_end, now, ttl, final_after, expected):
    assert _default_expired(written_at, period_end, now, ttl=ttl, final_after=final_after) is expected


def test_partial_month_refreshes_until_settled(tmp_path, set_now):
    kwargs = {"time_unit": "monthly", "ttl": timedelta(hours=1), "final_after": timedelta(days=3)}
    set_now(T(2026, 10, 15, 12))
    assert cache_parquet(_source(D(2026, 10, 15)), tmp_path, "date", **kwargs).filter(OCT).collect()["date"].max() == D(2026, 10, 15)
    _set_mtime(tmp_path, T(2026, 10, 15, 12))

    set_now(T(2026, 10, 15, 12, 30))
    assert cache_parquet(_source(D(2026, 10, 17)), tmp_path, "date", **kwargs).filter(OCT).collect()["date"].max() == D(2026, 10, 15)

    set_now(T(2026, 10, 15, 14))
    assert cache_parquet(_source(D(2026, 10, 17)), tmp_path, "date", **kwargs).filter(OCT).collect()["date"].max() == D(2026, 10, 17)

    # A rewrite three days after the month ends makes October permanent.
    _set_mtime(tmp_path, T(2026, 11, 4))
    set_now(T(2027, 1, 1))
    result = cache_parquet(_source(D(2026, 10, 31), value=2), tmp_path, "date", **kwargs).filter(OCT).collect()
    assert result["date"].max() == D(2026, 10, 17)
    assert result["value"].unique().to_list() == [1]


def test_empty_future_partitions_refresh(tmp_path, set_now):
    kwargs = {"time_unit": "daily", "ttl": timedelta(hours=1), "final_after": timedelta(days=0)}
    window = pl.col("date").is_between(D(2026, 10, 15), D(2026, 10, 25))
    set_now(T(2026, 10, 19, 12))
    assert cache_parquet(_source(D(2026, 10, 19)), tmp_path, "date", **kwargs).filter(window).collect()["date"].max() == D(2026, 10, 19)
    assert (tmp_path / "daily" / "2026-10-25.parquet").exists()
    _set_mtime(tmp_path, T(2026, 10, 19, 12))

    set_now(T(2026, 10, 23, 12))
    assert cache_parquet(_source(D(2026, 10, 23)), tmp_path, "date", **kwargs).filter(window).collect()["date"].max() == D(2026, 10, 23)


def test_no_date_column_ttl(tmp_path, set_now):
    source = pl.LazyFrame({"value": [1, 2]})
    set_now(T(2026, 10, 19))
    cache_parquet(source, tmp_path, ttl=timedelta(days=1)).collect()
    _set_mtime(tmp_path, T(2026, 10, 19))

    set_now(T(2026, 10, 19, 12))
    assert cache_parquet(source.with_columns(value=pl.col("value") * 10), tmp_path, ttl=timedelta(days=1)).collect()["value"].to_list() == [1, 2]

    set_now(T(2026, 10, 20))
    assert cache_parquet(source.with_columns(value=pl.col("value") * 10), tmp_path, ttl=timedelta(days=1)).collect()["value"].to_list() == [10, 20]

    # Upstream returning no rows empties the expired file instead of serving its stale rows.
    _set_mtime(tmp_path, T(2026, 10, 19))
    assert cache_parquet(source.filter(pl.lit(False)), tmp_path, ttl=timedelta(days=1)).collect().is_empty()
    assert (tmp_path / "null" / "data.parquet").exists()

    # The empty file counts as cached, so it is not fetched again until it expires.
    _set_mtime(tmp_path, T(2026, 10, 19, 12))
    assert cache_parquet(source, tmp_path, ttl=timedelta(days=1)).collect().is_empty()


def test_refresh_is_limited_to_query_partitions(tmp_path, set_now):
    source = pl.LazyFrame(
        {
            "date": [D(2026, 1, 5), D(2026, 1, 6), D(2026, 2, 5), D(2026, 2, 6)],
            "sym": ["A", "B", "A", "B"],
            "value": [1, 1, 1, 1],
        }
    )
    kwargs = {"time_unit": "monthly", "extra_partition_cols": "sym", "ttl": timedelta(hours=1)}
    set_now(T(2026, 3, 1))
    cache_parquet(source, tmp_path, "date", **kwargs).filter(pl.col("date").is_between(D(2026, 1, 1), D(2026, 2, 28))).collect()
    _set_mtime(tmp_path, T(2026, 3, 1))
    before = _mtimes(tmp_path)

    set_now(T(2026, 3, 2))
    jan = pl.col("date").is_between(D(2026, 1, 1), D(2026, 1, 31))
    feb = pl.col("date").is_between(D(2026, 2, 1), D(2026, 2, 28))
    predicate = (jan & (pl.col("sym") == "A")) | (feb & (pl.col("sym") == "B"))
    result = cache_parquet(source.with_columns(value=pl.lit(2, pl.Int64)), tmp_path, "date", **kwargs).filter(predicate).collect()
    assert result.sort("date")["value"].to_list() == [2, 2]
    after = _mtimes(tmp_path)
    assert {k for k in before if before[k] != after[k]} == {"monthly/A/2026-01.parquet", "monthly/B/2026-02.parquet"}


def test_refresh_keeps_null_partition_rows_outside_membership_scope(tmp_path, set_now, caplog):
    source = pl.LazyFrame(
        {
            "date": [D(2026, 1, 5), D(2026, 1, 6), D(2026, 2, 5)],
            "sym": ["A", None, "A"],
            "value": [1, 1, 1],
        }
    )
    kwargs = {"time_unit": "monthly", "extra_partition_cols": "sym", "ttl": timedelta(hours=1)}
    set_now(T(2026, 3, 1))
    cache_parquet(source, tmp_path, "date", **kwargs).filter(pl.col("date").is_between(D(2026, 1, 1), D(2026, 2, 28))).collect()
    _set_mtime(tmp_path, T(2026, 3, 1))
    null_files = [k for k in _mtimes(tmp_path) if "/A/" not in k]
    assert null_files

    # The null partition is outside the query scope; refreshing it must not drop rows upstream still has.
    set_now(T(2026, 3, 2))
    for predicate in [pl.col("sym").is_in(["A"]), pl.col("sym") == "A", pl.col("sym") != "B"]:
        _set_mtime(tmp_path, T(2026, 3, 1))
        cache_parquet(source, tmp_path, "date", **kwargs).filter(predicate & (pl.col("date") < D(2026, 2, 1))).collect()
        assert pl.read_parquet(tmp_path / null_files[0]).height == 1

    # An expired file that upstream no longer fills is emptied with a warning.
    _set_mtime(tmp_path, T(2026, 3, 1))
    set_now(T(2026, 3, 2))
    cache_parquet(source.filter(pl.col("sym").is_not_null()), tmp_path, "date", **kwargs).collect()
    assert pl.read_parquet(tmp_path / null_files[0]).is_empty()
    assert any(r.levelname == "WARNING" and "expired cache file" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("predicate", [pl.col("sym").ne_missing("A"), ~pl.col("sym").eq_missing("A")])
def test_refresh_null_partition_matched_by_missing_aware_comparison(tmp_path, set_now, predicate):
    source = pl.LazyFrame({"date": [D(2026, 1, 5), D(2026, 1, 5)], "sym": ["B", None], "value": [1, 1]})
    kwargs = {"time_unit": "monthly", "extra_partition_cols": "sym", "ttl": timedelta(hours=1)}
    set_now(T(2026, 3, 1))
    cache_parquet(source, tmp_path, "date", **kwargs).collect()
    _set_mtime(tmp_path, T(2026, 3, 1))
    set_now(T(2026, 3, 2))
    jan = pl.col("date").is_between(D(2026, 1, 1), D(2026, 1, 31))
    out = cache_parquet(source.with_columns(value=pl.lit(2, pl.Int64)), tmp_path, "date", **kwargs).filter(jan & predicate).collect()
    assert out["value"].to_list() == [2, 2]


def test_unbounded_refresh_above_open_sink_cap(tmp_path, set_now, monkeypatch):
    kwargs = {"time_unit": "daily", "ttl": timedelta(hours=1)}
    set_now(T(2026, 9, 4))
    cache_parquet(_source(D(2026, 9, 3)), tmp_path, "date", **kwargs).collect()
    _set_mtime(tmp_path, T(2026, 9, 4))

    monkeypatch.setattr(lazy_cache_parquet, "_max_open_sinks", lambda: 2)
    set_now(T(2026, 9, 5))
    assert cache_parquet(_source(D(2026, 9, 3), value=2), tmp_path, "date", **kwargs).collect()["value"].to_list() == [2, 2, 2]


def test_bounded_refresh_above_open_sink_cap_raises_before_writing(tmp_path, set_now, monkeypatch):
    kwargs = {"time_unit": "daily", "ttl": timedelta(hours=1)}
    set_now(T(2026, 9, 4))
    cache_parquet(_source(D(2026, 9, 3)), tmp_path, "date", **kwargs).collect()
    _set_mtime(tmp_path, T(2026, 9, 4))
    before = _mtimes(tmp_path)

    monkeypatch.setattr(lazy_cache_parquet, "_max_open_sinks", lambda: 2)
    set_now(T(2026, 9, 5))
    lf = cache_parquet(_source(D(2026, 9, 3), value=2), tmp_path, "date", **kwargs).filter(pl.col("date").is_between(START, D(2026, 9, 3)))
    with pytest.raises(Exception, match="open-sink cap"):
        lf.collect()
    assert _mtimes(tmp_path) == before


@pytest.mark.parametrize("error", [RuntimeError("sink failed"), PanicException("sink panicked")])
def test_failed_sink_deletes_partially_written_files(tmp_path, set_now, monkeypatch, error):
    # The second write's real mtime is long after October 2025 ends, so a leftover partial file would count as final.
    kwargs = {"time_unit": "monthly", "ttl": timedelta(hours=1), "final_after": timedelta(days=1)}
    source = _source(D(2025, 10, 31), start=D(2025, 9, 1))
    cache_parquet(source, tmp_path, "date", **kwargs).collect()
    _set_mtime(tmp_path, T(2025, 10, 15))
    sep = tmp_path / "monthly" / "2025-09.parquet"
    sep_mtime = sep.stat().st_mtime_ns
    set_now(T(2025, 10, 16))

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


@pytest.mark.parametrize("legacy", [False, True])
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
        cache_parquet(_source(D(2026, 9, 30)), tmp_path, "date").collect()


@pytest.mark.parametrize("syms", [[None], ["A", "B"]])
def test_write_batch_size_splits_sinks_below_open_sink_cap(tmp_path, monkeypatch, syms):
    monkeypatch.setattr(lazy_cache_parquet, "_max_open_sinks", lambda: 4)
    days = [d for d in pl.date_range(START, D(2026, 9, 6), eager=True) if d != D(2026, 9, 2)]
    source = pl.LazyFrame({"date": days}).join(pl.LazyFrame({"sym": syms}, schema={"sym": pl.String}), how="cross").with_columns(value=pl.lit(1))
    kwargs = {"time_unit": "daily", "extra_partition_cols": "sym" if syms != [None] else None}
    pred = pl.col("date").is_between(START, D(2026, 9, 6))
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


@pytest.mark.parametrize("refresh", [False, True])
def test_write_batch_size_keeps_batches_written_before_a_failure(tmp_path, set_now, monkeypatch, refresh):
    kwargs = {"time_unit": "daily", "ttl": timedelta(hours=1)}
    pred = pl.col("date").is_between(START, D(2026, 9, 6))
    if refresh:
        set_now(T(2026, 9, 7))
        cache_parquet(_source(D(2026, 9, 6)), tmp_path, "date", **kwargs).filter(pred).collect()
        _set_mtime(tmp_path, T(2026, 9, 7))
        set_now(T(2026, 9, 8))
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
    source = _source(D(2026, 9, 6), value=2).filter(pl.col("date") != D(2026, 9, 2))
    with pytest.raises(Exception, match="sink failed"):
        cache_parquet(source, tmp_path, "date", write_batch_size=3, **kwargs).filter(pred).collect()
    # The failed batch deletes the file it opened; files it never opened keep their old data.
    expected = {"01", "02", "03"} | ({"05", "06"} if refresh else set())
    assert {p.stem[-2:] for p in (tmp_path / "daily").glob("*.parquet")} == expected
    assert pl.read_parquet(tmp_path / "daily" / "2026-09-02.parquet").is_empty()
    assert pl.read_parquet(tmp_path / "daily" / "2026-09-03.parquet")["value"].to_list() == [2]


def test_write_filter_selects_only_target_partitions(tmp_path):
    source = pl.LazyFrame({"date": [START, START, D(2026, 9, 2), D(2026, 9, 2)], "sym": ["A", "B", "A", "B"], "value": [1, 2, 3, 4]})
    pred = ((pl.col("date") == START) & (pl.col("sym") == "A")) | ((pl.col("date") == D(2026, 9, 2)) & (pl.col("sym") == "B"))
    out = cache_parquet(source, tmp_path, "date", time_unit="daily", extra_partition_cols="sym").filter(pred).collect()
    assert sorted(out["value"].to_list()) == [1, 4]
    assert len(list(tmp_path.glob("**/*.parquet"))) == 2


def test_write_filter_keeps_unrestricted_null_date_target(tmp_path):
    source = pl.LazyFrame({"date": [None, START], "sym": ["B", "A"], "value": [1, 2]}, schema_overrides={"date": pl.Date})
    pred = pl.col("date").is_null() | ((pl.col("date") == START) & (pl.col("sym") == "A"))
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


def test_unbounded_query_refreshes_expired_latest_partition(tmp_path, set_now):
    kwargs = {"time_unit": "daily", "ttl": timedelta(hours=1), "final_after": timedelta(days=1)}
    set_now(T(2026, 10, 19, 12))
    cache_parquet(_source(D(2026, 10, 19)), tmp_path, "date", **kwargs).filter(pl.col("date") >= D(2026, 10, 10)).collect()
    _set_mtime(tmp_path, T(2026, 10, 21))
    _set_mtime(tmp_path, T(2026, 10, 19, 12), "daily/2026-10-19.parquet")

    set_now(T(2026, 10, 21, 1))
    result = cache_parquet(_source(D(2026, 10, 20), value=2), tmp_path, "date", **kwargs).filter(pl.col("date") >= D(2026, 10, 10)).collect()
    values = dict(zip(result["date"], result["value"], strict=True))
    assert values[D(2026, 10, 18)] == 1
    assert values[D(2026, 10, 19)] == 2
    assert values[D(2026, 10, 20)] == 2


def test_expired_empty_partition_still_bounds_gap_filling(tmp_path, set_now):
    kwargs = {"time_unit": "daily", "ttl": timedelta(hours=1)}
    source = pl.LazyFrame({"date": [START], "value": [1]})
    set_now(T(2026, 10, 19))
    for day in [START, D(2026, 9, 5)]:
        cache_parquet(source, tmp_path, "date", **kwargs).filter(pl.col("date") == day).collect()
    _set_mtime(tmp_path, T(2026, 10, 18))

    # Both files are expired and refreshed, but still bound the one-sided query: gaps up to the
    # empty 2026-09-05 file are filled, and nothing is written after it.
    result = cache_parquet(source, tmp_path, "date", **kwargs).filter(pl.col("date") >= START).collect()
    assert result["date"].to_list() == [START]
    assert sorted(_mtimes(tmp_path)) == [f"daily/2026-09-0{day}.parquet" for day in range(1, 6)]


@pytest.mark.parametrize("predicate", [OCT, None])
def test_read_only_reads_expired_partitions_from_upstream(tmp_path, set_now, predicate):
    kwargs = {"time_unit": "monthly", "ttl": timedelta(hours=1)}
    set_now(T(2026, 11, 1))
    cache_parquet(_source(D(2026, 10, 31)), tmp_path, "date", **kwargs).collect()
    _set_mtime(tmp_path, T(2026, 11, 1))
    _set_mtime(tmp_path, T(2026, 10, 1), "monthly/2026-10.parquet")
    before = _mtimes(tmp_path)

    lf = cache_parquet(_source(D(2026, 10, 31), value=2), tmp_path, "date", cache_mode=CacheMode.READ_ONLY, **kwargs)
    result = (lf.filter(predicate) if predicate is not None else lf).collect()
    oct_rows = result.filter(OCT)
    assert oct_rows.height == 31
    assert oct_rows["value"].unique().to_list() == [2]
    if predicate is None:
        assert result.filter(~OCT)["value"].unique().to_list() == [1]
    assert _mtimes(tmp_path) == before


def test_read_only_n_rows_skips_expired_files(tmp_path, set_now):
    kwargs = {"time_unit": "monthly", "ttl": timedelta(hours=1)}
    set_now(T(2026, 11, 1))
    cache_parquet(_source(D(2026, 10, 31)), tmp_path, "date", **kwargs).collect()
    _set_mtime(tmp_path, T(2026, 11, 1))
    _set_mtime(tmp_path, T(2026, 10, 1), "monthly/2026-09.parquet")

    source = _source(D(2026, 10, 31), value=2, start=D(2026, 10, 1))
    lf = cache_parquet(source, tmp_path, "date", cache_mode=CacheMode.READ_ONLY, read_kwargs={"n_rows": 1}, **kwargs)
    assert lf.collect()["date"].to_list() == [D(2026, 10, 1)]


def test_cache_only_serves_expired_files(tmp_path, set_now):
    kwargs = {"time_unit": "monthly", "ttl": timedelta(hours=1)}
    set_now(T(2026, 11, 1))
    cache_parquet(_source(D(2026, 10, 31)), tmp_path, "date", **kwargs).filter(OCT).collect()
    _set_mtime(tmp_path, T(2026, 10, 1))
    result = cache_parquet(_source(D(2026, 10, 31), value=2), tmp_path, "date", cache_mode=CacheMode.CACHE_ONLY, **kwargs).filter(OCT).collect()
    assert result["value"].unique().to_list() == [1]


def test_rebuild_does_not_call_ttl(tmp_path):
    def fail(*args):
        raise AssertionError("REBUILD must not evaluate expiry")

    cache_parquet(_source(D(2026, 10, 31)), tmp_path, "date").filter(OCT).collect()
    result = cache_parquet(_source(D(2026, 10, 31), value=2), tmp_path, "date", ttl=fail, cache_mode=CacheMode.REBUILD).filter(OCT).collect()
    assert result["value"].unique().to_list() == [2]


def test_callable_ttl_receives_local_period_end(tmp_path, set_now):
    calls = []

    def policy(written_at, period_end, now):
        calls.append((written_at.tzinfo, period_end, now))
        return False

    source = _source(D(2026, 10, 31)).with_columns(pl.col("date").cast(pl.Datetime("us", "America/New_York")))
    set_now(T(2026, 11, 1))
    lf = cache_parquet(source, tmp_path, "date", ttl=policy).filter(pl.col("date").dt.month() == 10)
    lf.collect()
    lf.collect()
    assert {period_end for _, period_end, _ in calls} >= {T(2026, 11, 1, 4)}
    assert all(tz is UTC and now == T(2026, 11, 1) for tz, _, now in calls)


@pytest.mark.parametrize("dumps", [pickle.dumps, cloudpickle.dumps])
def test_callable_ttl_pickles(tmp_path, dumps):
    lf = cache_parquet(_source(D(2026, 10, 31)), tmp_path, "date", ttl=lambda written_at, period_end, now: False).filter(OCT)
    assert pickle.loads(dumps(lf)).collect().height == 31


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"date_column": None, "final_after": timedelta(days=1)}, ValueError),
        ({"ttl": lambda *args: True, "final_after": timedelta(days=1)}, ValueError),
        ({"ttl": timedelta(hours=-1)}, ValueError),
        ({"final_after": timedelta(days=-1)}, ValueError),
        ({"ttl": 3600}, TypeError),
    ],
)
def test_invalid_expiry_arguments(tmp_path, kwargs, error):
    kwargs = {"date_column": "date", **kwargs}
    with pytest.raises(error):
        cache_parquet(_source(D(2026, 10, 31)), tmp_path, **kwargs)


def test_refresh_intersects_listed_dates_with_bounds(tmp_path, set_now):
    set_now(T(2026, 12, 1))
    cache_parquet(_source(D(2026, 11, 30)), tmp_path, "date", ttl=timedelta(hours=1)).collect()
    _set_mtime(tmp_path, T(2026, 12, 1))
    set_now(T(2026, 12, 2))
    predicate = pl.col("date").is_in([D(2026, 10, 5), D(2026, 11, 5)]) & (pl.col("date") < D(2026, 11, 1))
    out = cache_parquet(_source(D(2026, 10, 31), value=2), tmp_path, "date", ttl=timedelta(hours=1)).filter(predicate).collect()
    assert out["value"].to_list() == [2]
    assert pl.read_parquet(tmp_path / "monthly" / "2026-11.parquet")["value"].unique().to_list() == [1]


def test_refresh_null_partition_listed_in_is_in(tmp_path, set_now):
    source = pl.LazyFrame({"date": [D(2026, 1, 5), D(2026, 1, 5)], "sym": ["A", None], "value": [1, 1]})
    kwargs = {"time_unit": "monthly", "extra_partition_cols": "sym", "ttl": timedelta(hours=1)}
    set_now(T(2026, 3, 1))
    cache_parquet(source, tmp_path, "date", **kwargs).collect()
    _set_mtime(tmp_path, T(2026, 3, 1))
    set_now(T(2026, 3, 2))
    predicate = pl.col("sym").is_in(["A", None], nulls_equal=True) & (pl.col("date") < D(2026, 2, 1))
    out = cache_parquet(source.with_columns(value=pl.lit(2, pl.Int64)), tmp_path, "date", **kwargs).filter(predicate).collect()
    assert out["value"].to_list() == [2, 2]


@pytest.mark.parametrize("cache_mode", [CacheMode.CACHE, CacheMode.READ_ONLY])
@pytest.mark.parametrize("time_unit", ["daily", "monthly", "yearly"])
@pytest.mark.parametrize("date_dtype", [pl.Date, pl.Datetime("us"), pl.Datetime("us", "America/New_York")])
@pytest.mark.parametrize("null_state", ["expired", "fresh", "removed"])
def test_refresh_nullable_date_membership_with_unbounded_or(tmp_path, set_now, cache_mode, time_unit, date_dtype, null_state):
    source = pl.DataFrame(
        {
            "date": [D(2024, 1, 10), None, D(2024, 3, 10), None],
            "sym": ["A", "A", "B", "B"],
            "value": [10, 20, 30, 40],
        }
    ).with_columns(pl.col("date").cast(date_dtype))
    kwargs = {"time_unit": time_unit, "extra_partition_cols": "sym"}
    cache_parquet(source.lazy(), tmp_path, "date", **kwargs).collect()
    _set_mtime(tmp_path, T(2026, 3, 1))
    if null_state == "fresh":
        for path in tmp_path.rglob("*.parquet"):
            if pl.read_parquet(path)["value"].to_list() == [20]:
                os.utime(path, (T(2026, 3, 2).timestamp(), T(2026, 3, 2).timestamp()))
    before = _mtimes(tmp_path)
    untouched = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*.parquet")
        if pl.read_parquet(path)["value"].to_list() in ([[20], [40]] if null_state == "fresh" else [[40]])
    }
    set_now(T(2026, 3, 2))
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
    source = pl.LazyFrame({"date": [D(2024, 1, 10), None, D(2024, 3, 10)], "sym": ["A", "A", "B"], "value": [10, 20, 30]})
    predicate = (pl.col("date").is_in([D(2024, 1, 10), None], nulls_equal=True) & (pl.col("sym") == "A")) | (
        (pl.col("date") >= D(2024, 3, 10)) & (pl.col("sym") == "B")
    )
    result = cache_parquet(source, tmp_path, "date", extra_partition_cols="sym", cache_mode=cache_mode).filter(predicate).collect()
    assert sorted(result["value"].to_list()) == [10, 20, 30]


@pytest.mark.parametrize("engine", ["auto", "streaming"])
def test_read_only_unbounded_date_udf_rejecting_null(tmp_path, engine):
    source = pl.LazyFrame({"date": [D(2024, 1, 10), D(2024, 3, 11), D(2024, 3, 16)], "sym": ["A", "B", "B"], "value": [1, 2, 3]})

    def weekday(value):
        return value.weekday() < 5

    predicate = (pl.col("date") >= D(2024, 3, 1)) & pl.col("date").map_elements(weekday, return_dtype=pl.Boolean, skip_nulls=False)
    result = (
        cache_parquet(source, tmp_path, "date", extra_partition_cols="sym", cache_mode=CacheMode.READ_ONLY).filter(predicate).collect(engine=engine)
    )
    assert result["value"].to_list() == [2]


@pytest.mark.parametrize("error", [ComputeError("parquet: File out of specification"), OSError("range invalid"), PanicException("panicked")])
def test_cache_read_retries_once_after_failure(tmp_path, monkeypatch, caplog, error):
    source = _source(D(2026, 9, 30))
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
    source = _source(D(2026, 9, 30))
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


@pytest.mark.parametrize(("retries", "attempts"), [(None, 2), ("0", 1), ("3", 4), ("bad", 2)])
def test_cache_read_raises_after_retries(tmp_path, monkeypatch, retries, attempts):
    source = _source(D(2026, 9, 30))
    cache_parquet(source, tmp_path, "date").collect()
    calls = []

    def broken(lf, batch_size):
        calls.append(batch_size)
        raise OSError("still broken")

    if retries is not None:
        monkeypatch.setenv("POLARS_IO_TOOLS_CACHE_READ_RETRIES", retries)
    monkeypatch.setattr(lazy_cache_parquet, "collect_lf_in_io_source", broken)
    with pytest.raises(Exception, match="still broken"):
        cache_parquet(source, tmp_path, "date").collect()
    assert len(calls) == attempts


def test_cache_read_not_retried_after_rows_returned(tmp_path, monkeypatch):
    source = _source(D(2026, 9, 30))
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


@pytest.mark.parametrize(("case", "attempts"), [("cached", 2), ("missing", 1), ("expired", 1)])
def test_read_only_retries_only_cache_reads(tmp_path, monkeypatch, set_now, caplog, case, attempts):
    kwargs = {"time_unit": "monthly", "ttl": timedelta(hours=1)}
    set_now(T(2026, 11, 1))
    cache_parquet(_source(D(2026, 9, 30) if case == "missing" else D(2026, 10, 31)), tmp_path, "date", **kwargs).collect()
    _set_mtime(tmp_path, T(2026, 11, 1))
    if case == "expired":
        _set_mtime(tmp_path, T(2026, 10, 1), "monthly/2026-10.parquet")
    calls = []

    def broken(lf, batch_size):
        calls.append(batch_size)
        raise OSError("still broken")

    monkeypatch.setattr(lazy_cache_parquet, "collect_lf_in_io_source", broken)
    lf = cache_parquet(_source(D(2026, 10, 31), value=2), tmp_path, "date", cache_mode=CacheMode.READ_ONLY, **kwargs)
    with pytest.raises(Exception, match="still broken"):
        lf.filter(pl.col("date").is_between(START, D(2026, 10, 31))).collect()
    assert len(calls) == attempts
    assert any("attempt 1/2" in r.getMessage() for r in caplog.records) == (attempts == 2)


def test_lock_taken_only_for_writes(tmp_path):
    names = []

    @contextlib.contextmanager
    def lock(name):
        names.append(name)
        yield

    predicate = pl.col("date").is_between(START, D(2026, 9, 30))
    for mode in (CacheMode.READ_ONLY, CacheMode.CACHE_ONLY, CacheMode.CACHE, CacheMode.CACHE):
        cache_parquet(_source(D(2026, 9, 30)), tmp_path, "date", cache_mode=mode, lock=lock).filter(predicate).collect()
    assert names == [str((tmp_path / "monthly").resolve())]


def test_lock_plans_again_once_held(tmp_path):
    source = _source(D(2026, 9, 30))

    @contextlib.contextmanager
    def lock(name):
        # Another process fills the cache while this one waits for the lock.
        cache_parquet(source.with_columns(value=pl.lit(2, pl.Int64)), tmp_path, "date").collect()
        yield

    predicate = pl.col("date").is_between(START, D(2026, 9, 30))
    out = cache_parquet(source, tmp_path, "date", lock=lock).filter(predicate).collect()
    assert out["value"].unique().to_list() == [2]


@pytest.mark.parametrize("fail", [False, True])
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
    lf = cache_parquet(_source(D(2026, 9, 30)), tmp_path, "date", lock=lock).filter(pl.col("date").is_between(START, D(2026, 9, 30)))
    if fail:
        with pytest.raises(ComputeError, match="sink failed"):
            lf.collect()
        assert seen == [("write", True)]
    else:
        lf.collect()
        assert seen == [("write", True), ("read", False)]
    assert not held


def test_file_visitor_sees_each_written_file(tmp_path):
    events = []

    @contextlib.contextmanager
    def lock(name):
        events.append("lock")
        yield
        events.append("unlock")

    source = _source(D(2026, 9, 30)).with_columns(sym=pl.lit("a%b"))
    predicate = pl.col("date").is_between(START, D(2026, 10, 31)) & (pl.col("sym") == "a%b")
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
        source = source.with_columns(pl.lit(D(2024, 1, 10)).alias("date"))
        options.update(date_column="date", partition_format="$year/$month" if layout == "custom" else None)
        predicate &= pl.col("date") == D(2024, 1, 10)
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


@pytest.mark.parametrize("literal_left", [False, True])
@pytest.mark.parametrize("tick", [1, 999])
def test_time_partition_filter_preserves_nanoseconds(tmp_path, literal_left, tick):
    source = pl.DataFrame({"time": pl.Series([1, 999, 1000], dtype=pl.Int64).cast(pl.Time)}).lazy()
    literal = pl.lit(tick).cast(pl.Time)
    predicate = literal == pl.col("time") if literal_left else pl.col("time") == literal
    result = cache_parquet(source, tmp_path, extra_partition_cols="time").filter(predicate).collect()
    assert result["time"].cast(pl.Int64).to_list() == [tick]


def test_monthly_cache_over_daily_cache(tmp_path):
    calls = []

    def daily(end, cache_mode=CacheMode.CACHE):
        source = _source(end)
        counted = source.map_batches(lambda df: calls.append(1) or df, schema=source.collect_schema())
        return cache_parquet(counted, tmp_path / "daily", "date", time_unit="daily", cache_mode=cache_mode)

    def monthly(upstream, cache_mode=CacheMode.CACHE):
        return cache_parquet(upstream, tmp_path / "monthly", "date", time_unit="monthly", cache_mode=cache_mode)

    def files(name):
        return len(list((tmp_path / name).rglob("*.parquet")))

    daily(D(2026, 10, 20)).filter(pl.col("date").is_between(START, D(2026, 10, 20))).collect()
    calls.clear()

    # Compacting September reads the daily cache only.
    september = monthly(daily(D(2026, 10, 20), CacheMode.CACHE_ONLY)).filter(pl.col("date").is_between(START, D(2026, 9, 30))).collect()
    assert calls == []
    assert files("monthly") == 1
    assert_frame_equal(september, _source(D(2026, 9, 30)).collect())

    # Reading through the monthly cache falls back to the daily cache for October, which fetches only the new days.
    query = pl.col("date").is_between(D(2026, 9, 10), D(2026, 10, 25))
    result = monthly(daily(D(2026, 10, 25)), CacheMode.READ_ONLY).filter(query).collect()
    assert_frame_equal(result.sort("date"), _source(D(2026, 10, 25)).filter(query).collect())
    assert len(calls) == 1
    assert files("monthly") == 1
    assert files("daily") == 55


@pytest.mark.parametrize("cached_through", [D(2026, 9, 15), D(2026, 9, 30)], ids=["partial-daily", "complete-daily"])
def test_monthly_cache_retains_history_after_verified_finalization(tmp_path, set_now, cached_through):
    calls = []
    source = _source(D(2026, 9, 30))
    expected = source.collect()
    counted = source.map_batches(lambda frame: calls.append(1) or frame, schema=source.collect_schema())
    history = pl.col("date").is_between(START, D(2026, 9, 30))
    daily_root, monthly_root = tmp_path / "daily", tmp_path / "monthly"

    def daily(cache_mode=CacheMode.CACHE):
        return cache_parquet(counted, daily_root, "date", time_unit="daily", cache_mode=cache_mode)

    def monthly():
        return cache_parquet(
            daily(CacheMode.CACHE_ONLY), monthly_root, "date", time_unit="monthly", ttl=timedelta(days=1), final_after=timedelta(days=1)
        )

    set_now(T(2026, 10, 1))
    daily().filter(pl.col("date") <= cached_through).collect()
    calls.clear()
    assert_frame_equal(daily().filter(history).collect().sort("date"), expected)
    assert len(calls) == (1 if cached_through < D(2026, 9, 30) else 0)
    calls.clear()
    assert_frame_equal(monthly().filter(history).collect().sort("date"), expected)
    _set_mtime(monthly_root, T(2026, 10, 1))
    before = _mtimes(monthly_root)

    set_now(T(2026, 10, 2))
    assert_frame_equal(monthly().filter(history).collect().sort("date"), expected)
    assert _mtimes(monthly_root) != before
    _set_mtime(monthly_root, T(2026, 10, 2))
    finalized = _mtimes(monthly_root)
    for path in daily_root.rglob("*.parquet"):
        path.unlink()

    set_now(T(2026, 10, 6))
    assert_frame_equal(monthly().filter(history).collect().sort("date"), expected)
    assert_frame_equal(pl.concat([pl.read_parquet(path) for path in monthly_root.rglob("*.parquet")]).sort("date"), expected)
    assert _mtimes(monthly_root) == finalized
    assert calls == []


def test_values_without_a_path_segment_of_their_own_are_read_from_upstream(tmp_path):
    src = pl.LazyFrame({"date": [D(2024, 1, 10)] * 6, "x": ["BRK/B", "None", "", "..", None, "ok"], "v": [1, 2, 3, 4, 5, 6]})
    jan = pl.col("date").is_between(D(2024, 1, 1), D(2024, 1, 31))
    for pred in [jan, jan & (pl.col("x") == "BRK/B"), jan & pl.col("x").is_in(["None", ""]), jan & pl.col("x").is_null(), jan]:
        out = cache_parquet(src, tmp_path, "date", extra_partition_cols="x").filter(pred).collect()
        assert_frame_equal(out, src.filter(pred).collect(), check_row_order=False)
    # Only the null and "ok" partitions are cached, and no other file is left behind.
    assert len(list(tmp_path.rglob("*.*"))) == 2


@pytest.mark.parametrize(
    ("date_column", "legacy_file", "mode"),
    [(None, "null/a/b.parquet", CacheMode.CACHE), ("date", "monthly/a/b/2024-01.parquet", CacheMode.REBUILD)],
    ids=["no_date", "date"],
)
def test_files_with_names_that_are_not_cache_keys_are_not_read(tmp_path, date_column, legacy_file, mode):
    src = pl.LazyFrame({"date": [D(2024, 1, 10)] * 2, "x": ["a/b", "ok"], "v": [1, 2]})
    stale = tmp_path / legacy_file
    stale.parent.mkdir(parents=True)
    src.with_columns(v=pl.lit(0, pl.Int64)).collect().write_parquet(stale)
    out = cache_parquet(src, tmp_path, date_column, extra_partition_cols="x", cache_mode=mode).collect()
    assert_frame_equal(out, src.collect(), check_row_order=False)


def test_unnamed_partition_files_are_removed_when_the_file_visitor_raises(tmp_path):
    def visitor(path):
        raise RuntimeError("visitor failed")

    src = pl.LazyFrame({"date": [D(2024, 1, 10)] * 2, "x": ["a/b", "ok"], "v": [1, 2]})
    with pytest.raises(Exception, match="visitor failed"):
        cache_parquet(src, tmp_path, "date", extra_partition_cols="x", file_visitor=visitor).collect()
    assert not list(tmp_path.rglob("*.tmp"))
