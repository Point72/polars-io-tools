"""Regressions for cache_parquet planning bugs (re-writes, lost rows, key parsing), as named rows.

``REGRESSIONS`` rows drive a cache with a seed and a list of reads and check every read against native Polars
cold and warm; rows can also require that the seed's files are not rewritten. Bugs that need a patched platform,
pre-written legacy files or a changing upstream stay functions.
"""

import datetime
import importlib
import os
import time
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_io_tools._compat import POLARS_HAS_PARTITION_BY
from polars_io_tools.io_sources.lazy_cache_parquet import (
    CacheMode,
    _construct_partitions_df_from_existing,
    _extract_existing_files,
    _rebuild_partition_scope,
    cache_parquet,
)
from polars_io_tools.tests.helpers.cache_parquet_shared import assert_cached_matches_native, collect_filtered

D = datetime.date
DT = datetime.datetime
OLD_NS = 1_000_000_000 * 10**9
CACHE, REBUILD, READ_ONLY = CacheMode.CACHE, CacheMode.REBUILD, CacheMode.READ_ONLY


def _age_files(root: Path) -> dict[Path, int]:
    """Set every cached file's mtime to a fixed past value and return the snapshot."""
    files = sorted(root.rglob("*.parquet"))
    assert files
    for f in files:
        os.utime(f, ns=(OLD_NS, OLD_NS))
    return {f: f.stat().st_mtime_ns for f in files}


def _rewritten(snapshot: dict[Path, int]) -> list[str]:
    return [str(f) for f, m in snapshot.items() if f.stat().st_mtime_ns != m]


def _sorted(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(df.columns)


MONTHLY_SRC = pl.LazyFrame({"date": [D(2024, 1, 10), D(2024, 1, 15), D(2024, 2, 15)], "value": [0, 1, 2]})
_JAN = pl.col("date").is_between(D(2024, 1, 1), D(2024, 1, 31))


@dataclass(frozen=True)
class Read:
    """A cached read of ``predicate`` (``None`` reads everything) in cache ``mode``."""

    predicate: pl.Expr | None
    mode: CacheMode = CACHE


@dataclass(frozen=True)
class Regression:
    """A cache seeded by reading ``seed`` from ``seed_src`` (default ``src``), then ``reads`` of ``src``.

    Each read runs ``runs`` times and must match native Polars. With ``keep_seed_files`` the seed's files are aged and
    no read may rewrite or remove them.
    """

    id: str
    src: Any
    _: KW_ONLY
    reads: list[Read]
    seed: list = field(default_factory=list)
    seed_src: Any = None
    keep_seed_files: bool = False
    seed_file_count: int | None = None
    final_file_count: int | None = None
    # paths relative to the cache root, checked after the reads
    expected_files: tuple[str, ...] = ()
    absent_files: tuple[str, ...] = ()
    runs: int = 1
    cache_kwargs: dict = field(default_factory=dict)


def _ts(values, dtype):
    return pl.Series(values).cast(dtype)


_NULLABLE_DATES = pl.LazyFrame({"date": [D(2024, 1, 10), None, D(2024, 2, 5)], "x": ["a", "a", "a"], "value": [0, 1, 2]})
_NULLABLE_X = pl.LazyFrame({"date": [D(2024, 1, 10), D(2024, 1, 11), D(2024, 1, 12)], "x": ["a", None, "b"], "value": [0, 1, 2]})
_MX = {"time_unit": "monthly", "extra_partition_cols": "x"}
_SYM = pl.LazyFrame({"date": [D(2024, 1, 10), D(2024, 1, 10), D(2024, 2, 10), D(2024, 2, 10)], "sym": ["A", "B", "A", "B"], "value": [0, 1, 2, 3]})
_MSYM = {"time_unit": "monthly", "extra_partition_cols": "sym"}
_MISSING_AWARE = {
    "ne-missing": pl.col("x").ne_missing("a"),
    "not-eq-missing": ~pl.col("x").eq_missing("a"),
    "in-null": pl.col("x").is_in(["b", None], nulls_equal=True),
    "date-ne-missing": _JAN & pl.col("x").ne_missing("a"),
    "date-in-null": _JAN & pl.col("x").is_in(["b", None], nulls_equal=True),
}
_MISSING_SRC = pl.LazyFrame({"date": [D(2024, 1, 10)] * 3, "x": ["a", "b", None], "value": [0, 1, 2]})
_DROPPED = pl.LazyFrame({"date": [D(2024, 1, 10), D(2024, 2, 10), None], "sym": ["A", None, "A"], "value": [1, 1, 1]})
_DT_TYPES = {"Date": pl.Date, "Datetime(us)": pl.Datetime("us"), "Datetime(us, NY)": pl.Datetime("us", "America/New_York")}

REGRESSIONS = [
    *(
        Regression(f"one_sided_date_predicate_on_empty_cache-{k}", MONTHLY_SRC, reads=[Read(p)])
        for k, p in [("le", pl.col("date") <= D(2024, 1, 31)), ("ge", pl.col("date") >= D(2024, 1, 1))]
    ),
    *(
        Regression(
            f"not_equal_date_does_not_drop_enclosing_partition-{tu}",
            MONTHLY_SRC.with_columns(pl.col("date").cast(dt)),
            reads=[Read(pl.col("date").is_between(D(2024, 1, 1), D(2024, 2, 29)) & (pl.col("date") != ex))],
            runs=2,
            cache_kwargs={"time_unit": tu},
        )
        for tu, dt, ex in [("monthly", pl.Date, D(2024, 1, 15)), ("daily", pl.Datetime("us"), DT(2024, 1, 15, 12))]
    ),
    *(
        Regression(
            f"cached_partitions_not_rewritten_by_open_queries-{tu}-{k}",
            MONTHLY_SRC,
            reads=[Read(p)],
            seed=[_JAN],
            keep_seed_files=True,
            expected_files=("monthly/2024-02.parquet",) if tu == "monthly" else (),
            cache_kwargs={"time_unit": tu},
        )
        for tu in ["monthly", "yearly"]
        for k, p in [("unbounded", pl.col("date") >= D(2024, 1, 1)), ("unconstrained", None)]
    ),
    Regression(
        "daily_datetime_column_partitions_not_rewritten",
        pl.LazyFrame({"date": [DT(2024, 1, 1, 9), DT(2024, 1, 1, 15), DT(2024, 1, 2, 9)], "value": [0, 1, 2]}),
        reads=[Read(pl.col("date") >= DT(2024, 1, 1))],
        seed=[pl.col("date").is_between(DT(2024, 1, 1), DT(2024, 1, 1, 23, 59))],
        keep_seed_files=True,
        cache_kwargs={"time_unit": "daily"},
    ),
    Regression(
        "finite_query_does_not_rewrite_cached_partitions_between_missing_ones",
        _SYM,
        reads=[Read(pl.col("date").is_between(D(2024, 1, 1), D(2024, 2, 29)) & pl.col("sym").is_in(["A", "B"]))],
        seed=[(pl.col("date") == D(2024, 1, 10)) & (pl.col("sym") == "B"), (pl.col("date") == D(2024, 2, 10)) & (pl.col("sym") == "A")],
        keep_seed_files=True,
        cache_kwargs=_MSYM,
    ),
    Regression(
        "extra_col_only_predicate_does_not_rewrite_cached_dates",
        _SYM,
        reads=[Read(pl.col("sym") == "A")],
        seed=[_JAN],
        keep_seed_files=True,
        expected_files=("monthly/A/2024-02.parquet",),
        absent_files=("monthly/B/2024-02.parquet",),
        cache_kwargs=_MSYM,
    ),
    *(
        Regression(
            f"custom_partition_format_cache_hit-{fmt}",
            MONTHLY_SRC,
            reads=[Read(_JAN)],
            seed=[_JAN],
            keep_seed_files=True,
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
        Regression(
            f"null_or_value_predicate_keeps_null_rows-{k}-{'warm' if warm else 'cold'}",
            src,
            reads=[Read(p)],
            seed=[None] if warm else [],
            cache_kwargs=_MX,
        )
        for k, src, p in [
            ("date_null_or_value", _NULLABLE_DATES, pl.col("date").is_null() | (pl.col("date") == D(2024, 1, 10))),
            ("date_null_or_range", _NULLABLE_DATES, pl.col("date").is_null() | _JAN),
            ("date_null_or_value_with_x", _NULLABLE_DATES, (pl.col("x") == "a") & (pl.col("date").is_null() | (pl.col("date") == D(2024, 1, 10)))),
            ("x_null_or_value", _NULLABLE_X, _JAN & (pl.col("x").is_null() | (pl.col("x") == "a"))),
            ("x_null_or_date", _NULLABLE_X, pl.col("x").is_null() | (pl.col("date") == D(2024, 1, 10))),
        ]
        for warm in (False, True)
    ),
    *(
        Regression(
            f"null_date_partition_is_cached-{fmt}",
            pl.LazyFrame({"date": [D(2024, 1, 10), None], "value": [0, 1]}),
            reads=[Read(None)],
            seed=[None],
            keep_seed_files=True,
            seed_file_count=2,
            cache_kwargs={"partition_format": fmt},
        )
        for fmt in [None, "$year/$month"]
    ),
    Regression(
        "decimal_partition_exclusion_without_date_column",
        pl.LazyFrame({"x": pl.Series(["1.23", "-2.50", "4.00"]).cast(pl.Decimal(10, 2)), "value": [0, 1, 2]}),
        reads=[Read(None)],
        seed=[pl.col("x") < 4],
        keep_seed_files=True,
        cache_kwargs={"date_column": None, "extra_partition_cols": "x"},
    ),
    Regression(
        "decimal_partition_finite_query_with_several_values",
        pl.LazyFrame({"date": [D(2024, 1, 10), D(2024, 1, 11)], "x": pl.Series(["1.23", "2.34"]).cast(pl.Decimal(10, 2)), "value": [0, 1]}),
        reads=[Read(_JAN & pl.col("x").is_in(pl.Series(["1.23", "2.34"]).cast(pl.Decimal(10, 2))))],
        cache_kwargs={"extra_partition_cols": "x"},
    ),
    *(
        Regression(
            f"timezone_aware_finite_query_uses_local_partition_bounds-{tu}-{tz}",
            pl.LazyFrame(
                {
                    "ts": pl.Series(
                        [DT(2023, 12, 31, 23, 30), DT(2024, 1, 1, 0, 30), DT(2024, 1, 31, 23, 30), DT(2024, 2, 1, 0, 30)]
                    ).dt.replace_time_zone(tz),
                    "value": [0, 1, 2, 3],
                }
            ),
            reads=[Read(pl.col("ts").is_between(DT(2024, 1, 1, tzinfo=ZoneInfo(tz)), DT(2024, 1, 31, 23, 59, tzinfo=ZoneInfo(tz))))],
            runs=2,
            cache_kwargs={"date_column": "ts", "time_unit": tu},
        )
        for tu in ["daily", "monthly"]
        for tz in ["America/New_York", "Asia/Tokyo"]
    ),
    *(
        Regression(
            f"exclusive_datetime_bounds_keep_rest_of_bound_date-{tu}",
            pl.LazyFrame({"ts": [DT(2024, 1, 31, 13), DT(2024, 2, 1, 13), DT(2024, 2, 2, 11)], "value": [0, 1, 2]}),
            reads=[Read((pl.col("ts") > DT(2024, 1, 31, 12)) & (pl.col("ts") < DT(2024, 2, 2, 12)))],
            runs=2,
            cache_kwargs={"date_column": "ts", "time_unit": tu},
        )
        for tu in ["daily", "monthly"]
    ),
    *(
        Regression(
            f"one_sided_query_caches_whole_partitions-{op}-{k}",
            pl.LazyFrame({"ts": _ts([DT(2024, 1, 5), DT(2024, 1, 20), DT(2024, 2, 5)], dt), "value": [0, 1, 2]}),
            reads=[
                Read(pl.col("ts") >= pl.lit(DT(2024, 1, 10)).cast(dt) if op == "ge" else pl.col("ts") <= pl.lit(DT(2024, 1, 10)).cast(dt)),
                Read(None),
            ],
            cache_kwargs={"date_column": "ts"},
        )
        for op in ["ge", "le"]
        for k, dt in _DT_TYPES.items()
    ),
    Regression(
        "one_sided_query_writes_empty_files_only_for_queried_extra_values",
        pl.LazyFrame({"ts": [DT(2024, 1, 10, 9), DT(2024, 1, 21, 17)], "k": ["a", "a"], "value": [0, 1]}),
        reads=[
            Read(pl.col("ts").is_between(DT(2024, 1, 21), DT(2024, 1, 21, 23))),
            Read((pl.col("ts") > DT(2024, 1, 1)) & (pl.col("k") == "b")),
            Read(None),
        ],
        cache_kwargs={"date_column": "ts", "time_unit": "daily", "extra_partition_cols": "k"},
    ),
    Regression(
        "one_sided_exclusive_midnight_bound_does_not_mark_bound_date_cached",
        pl.LazyFrame({"ts": [DT(2024, 1, 21, 17), DT(2024, 2, 20, 6)], "value": [0, 1]}),
        reads=[Read(pl.col("ts").is_between(DT(2024, 1, 21), DT(2024, 1, 21, 23))), Read(pl.col("ts") < DT(2024, 2, 20)), Read(None)],
        cache_kwargs={"date_column": "ts", "time_unit": "daily"},
    ),
    Regression(
        "or_with_one_sided_date_clause_reads_every_cached_match",
        pl.LazyFrame({"date": [D(2024, 1, 10), D(2024, 2, 10), D(2024, 3, 10)], "value": [1, 2, 3]}),
        reads=[Read((pl.col("date") == D(2024, 3, 10)) | (pl.col("date") < D(2024, 2, 15)))],
        seed=[None],
    ),
    *(
        Regression(
            f"last_representable_date_partition-{mode.name}-{tu}",
            pl.LazyFrame({"date": [D.max, D(9999, 1, 5)], "value": [0, 1]}),
            reads=[Read(pl.col("date") == D.max, mode=mode)],
            runs=2,
            cache_kwargs={"time_unit": tu, "ttl": datetime.timedelta(days=1)},
        )
        for mode in [READ_ONLY, CACHE]
        for tu in ["daily", "monthly", "yearly"]
    ),
    # REBUILD refreshes the partitions in scope from the current upstream
    *(
        Regression(
            f"rebuild_missing_aware_predicate_refreshes_null_partition-{dc}-{k}",
            _MISSING_SRC.with_columns(pl.col("value") + 100),
            reads=[Read(p, mode=REBUILD)],
            seed=[None],
            seed_src=_MISSING_SRC,
            cache_kwargs={"date_column": dc, "extra_partition_cols": "x"},
        )
        for dc in [None, "date"]
        for k, p in _MISSING_AWARE.items()
    ),
    *(
        Regression(
            f"rebuild_unbounded_query_drops_rows_upstream_no_longer_has-{k}",
            _DROPPED.clear(),
            reads=[Read(p, mode=REBUILD)],
            seed=[None],
            seed_src=_DROPPED,
            cache_kwargs={"extra_partition_cols": "sym"},
        )
        for k, p in [
            ("unbounded", pl.col("date") >= D(2024, 1, 1)),
            ("sym_or_date", pl.col("sym").is_in(["A"]) | (pl.col("date") < D(2024, 1, 20))),
            ("null_or_date", pl.col("date").is_null() | (pl.col("date") < D(2024, 3, 1))),
        ]
    ),
    *(
        Regression(
            f"rebuild_datetime_scope_caches_whole_partition-{tz}",
            pl.LazyFrame({"ts": pl.Series([DT(2024, 1, 5), DT(2024, 1, 15), DT(2024, 2, 5)], dtype=pl.Datetime("us", tz)), "value": [0, 1, 2]}),
            reads=[
                Read(
                    pl.col("ts").is_between(
                        DT(2024, 1, 10, tzinfo=ZoneInfo(tz) if tz else None), DT(2024, 1, 20, tzinfo=ZoneInfo(tz) if tz else None)
                    ),
                    mode=REBUILD,
                ),
                Read(None),
            ],
            cache_kwargs={"date_column": "ts"},
        )
        for tz in [None, "America/New_York"]
    ),
    *(
        Regression(
            f"rebuild_removes_stale_null_partition-{nc}",
            src.filter(pl.col("value") == 0),
            reads=[Read(None, mode=REBUILD)],
            seed=[None],
            seed_src=src,
            final_file_count=1,
            cache_kwargs={"extra_partition_cols": "x"},
        )
        for nc, src in [
            ("date", pl.LazyFrame({"date": [D(2024, 1, 10), None], "x": ["a", "a"], "value": [0, 1]})),
            ("x", pl.LazyFrame({"date": [D(2024, 1, 10), D(2024, 1, 11)], "x": ["a", None], "value": [0, 1]})),
        ]
    ),
]


@pytest.mark.parametrize("reg", [pytest.param(r, id=r.id) for r in REGRESSIONS])
def test_regression(tmp_path, reg):
    kwargs = {"date_column": "date"} | reg.cache_kwargs
    seed_src = reg.src if reg.seed_src is None else reg.seed_src
    for pred in reg.seed:
        collect_filtered(cache_parquet(seed_src, tmp_path, **kwargs), pred)
    if reg.seed_file_count is not None:
        assert len(list(tmp_path.rglob("*.parquet"))) == reg.seed_file_count
    snapshot = _age_files(tmp_path) if reg.keep_seed_files else None
    for read in reg.reads:
        assert_cached_matches_native(reg.src, tmp_path, read.predicate, runs=reg.runs, **kwargs | {"cache_mode": read.mode})
    if reg.keep_seed_files:
        assert _rewritten(snapshot) == []
        assert all(path.exists() for path in snapshot)
    if reg.final_file_count is not None:
        assert len(list(tmp_path.rglob("*.parquet"))) == reg.final_file_count
    for path in reg.expected_files:
        assert (tmp_path / path).exists(), path
    for path in reg.absent_files:
        assert not (tmp_path / path).exists(), path


# Key parsing


def test_custom_partition_format_parsing():
    schema = pl.Schema({"sym": pl.String, "date": pl.Date})
    df = _construct_partitions_df_from_existing({"A/2024/03", "B/2023/12"}, schema, ["sym", "date"], "$year/$month", "date", "file")
    expected = pl.DataFrame({"sym": ["A", "B"], "date": [D(2024, 3, 1), D(2023, 12, 1)], "file": ["A/2024/03", "B/2023/12"]})
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


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        ([True, False], pl.Boolean),
        ([0.5, 1e-05], pl.Float64),
        ([D(2020, 1, 1), D(2021, 1, 1)], pl.Date),
        ([DT(2020, 1, 1, 12), DT(2020, 1, 2, 0, 0, 0, 500)], pl.Datetime("us")),
        ([DT(2020, 1, 1, 12, tzinfo=ZoneInfo("UTC")), DT(2020, 1, 2, tzinfo=ZoneInfo("UTC"))], pl.Datetime("us", "Asia/Tokyo")),
        # Whole seconds: polars < 1.38 drops fractional seconds from Time keys
        ([datetime.time(12, 30), datetime.time(0, 0, 1)], pl.Time),
        (["1.23", "-2.50"], pl.Decimal(10, 2)),
    ],
    ids=["boolean", "float", "date", "datetime", "datetime_tz", "time", "decimal"],
)
def test_extra_partition_col_dtypes_round_trip(tmp_path, values, dtype):
    series = pl.Series(values).cast(dtype) if isinstance(dtype, pl.Decimal) else pl.Series(values, dtype=dtype)
    src = pl.LazyFrame({"date": [D(2024, 1, 10), D(2024, 1, 11)], "x": series, "value": [0, 1]})
    kwargs = {"date_column": "date", **_MX}
    assert_cached_matches_native(src, tmp_path, _JAN & (pl.col("x") == series[0]), **kwargs)
    cache_parquet(src, tmp_path, **kwargs).filter(_JAN).collect()
    snapshot = _age_files(tmp_path)
    assert len(snapshot) == 2
    for pred in (_JAN, None):
        assert_cached_matches_native(src, tmp_path, pred, **kwargs)
    assert _rewritten(snapshot) == []
    assert sorted(tmp_path.rglob("*.parquet")) == sorted(snapshot)


@pytest.mark.parametrize(
    "x", [pl.Series([datetime.timedelta(hours=1)]), pl.Series([DT(2024, 1, 1)], dtype=pl.Datetime("ns"))], ids=["duration", "datetime_ns"]
)
def test_unsupported_extra_partition_col_is_rejected(tmp_path, x):
    src = pl.LazyFrame({"date": [D(2024, 1, 10)], "x": x})
    with pytest.raises(ValueError, match="cannot partition on Duration or nanosecond Datetime column x"):
        cache_parquet(src, tmp_path, "date", extra_partition_cols="x")


@pytest.mark.parametrize(
    ("values", "dtype"),
    [(["a", None], pl.String), ([1, None], pl.Int64), ([True, None], pl.Boolean), (["a", None], pl.Enum(["a", "b"])), (["a", None], pl.Categorical)],
    ids=["str", "int", "bool", "enum", "categorical"],
)
@pytest.mark.parametrize("date_column", ["date", None])
def test_null_extra_partition_col_is_cached(tmp_path, values, dtype, date_column):
    src = pl.LazyFrame({"date": [D(2024, 1, 10), D(2024, 1, 11)], "x": pl.Series(values, dtype=dtype), "value": [0, 1]})
    kwargs = {"date_column": date_column, **_MX}
    cache_parquet(src, tmp_path, **kwargs).collect()
    snapshot = _age_files(tmp_path)
    assert len(snapshot) == 2
    for pred in (None, None, pl.col("x").is_null(), pl.col("x").is_null() & (pl.col("date") == D(2024, 1, 11)), pl.col("x") == values[0]):
        assert_cached_matches_native(src, tmp_path, pred, **kwargs)
    assert _rewritten(snapshot) == []


@pytest.mark.parametrize("tz", ["Asia/Tokyo", "America/New_York"])
@pytest.mark.parametrize("time_unit", ["daily", "monthly"])
def test_timezone_aware_open_query_keeps_uncached_rows(tmp_path, tz, time_unit):
    dtype = pl.Datetime("us", tz)
    times = [DT(2024, 1, 1, 12), DT(2024, 1, 2, 0, 30), DT(2024, 1, 2, 23, 30), DT(2024, 2, 1, 0, 30)]
    src = pl.LazyFrame({"ts": pl.Series(times).dt.replace_time_zone(tz), "value": [0, 1, 2, 3]})
    seed = pl.LazyFrame({"ts": pl.Series(times[:1]).dt.replace_time_zone(tz), "value": [0]})
    cache_parquet(seed, tmp_path, "ts", time_unit=time_unit).collect()
    snapshot = _age_files(tmp_path)
    pred = pl.col("ts") >= DT(2023, 12, 1, tzinfo=ZoneInfo(tz))
    for _ in range(2):  # writes the uncached periods, then reads them back
        out = cache_parquet(src, tmp_path, "ts", time_unit=time_unit).filter(pred).collect()
        assert out.schema["ts"] == dtype
        assert sorted(out["value"].to_list()) == ([0, 1, 2, 3] if time_unit == "daily" else [0, 3])
    assert _rewritten(snapshot) == []


# Exclusion predicates over many cached partitions


def test_exclusion_predicate_scales_with_fragmented_cache():
    from polars_io_tools.io_sources.lazy_cache_parquet import _build_not_existing_partitions_pred

    days = pl.date_range(D(2000, 1, 1), D(2021, 12, 31), eager=True)
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
    days = [D(2024, 1, 1) + datetime.timedelta(days=j) for j in range(11)]
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
    src = pl.LazyFrame({"date": [D(2024, 1, 10), D(2024, 1, 10)], "flag": [True, False], "value": [0, 1]})
    pred = (pl.col("flag") == True) & (pl.col("date") == D(2024, 1, 10))
    assert_cached_matches_native(src, tmp_path, pred, date_column="date", time_unit="monthly", extra_partition_cols="flag")
    true_key = "True" if POLARS_HAS_PARTITION_BY else "true"
    assert sorted(p.parent.name for p in (tmp_path / "monthly").rglob("*.parquet")) == [true_key]


def test_custom_partition_format_extra_value_with_equals_sign(tmp_path):
    src = pl.LazyFrame({"date": [D(2024, 1, 10), D(2024, 1, 10)], "k": ["a=b", "c"], "value": [0, 1]})
    kwargs = {"time_unit": "monthly", "partition_format": "$year/$month", "extra_partition_cols": "k"}
    cache_parquet(src, tmp_path, "date", **kwargs).filter(_JAN).collect()
    snapshot = _age_files(tmp_path)
    out = cache_parquet(src.with_columns(pl.col("value") + 10), tmp_path, "date", **kwargs).filter(_JAN).collect()
    assert_frame_equal(_sorted(out), _sorted(src.collect()))
    assert _rewritten(snapshot) == []


@pytest.mark.parametrize(
    ("values", "dtype", "old_keys"),
    [
        ([True, False], pl.Boolean, ["true", "false"]),
        ([DT(2020, 1, 1, 12), DT(2020, 1, 2, 0, 0, 0, 500)], pl.Datetime("us"), ["2020-01-01 12:00:00.000000", "2020-01-02 00:00:00.000500"]),
        ([True, None], pl.Boolean, ["true", "null"]),
    ],
    ids=["boolean", "datetime", "null"],
)
@pytest.mark.parametrize("cache_mode", [CACHE, REBUILD])
def test_old_partition_key_spellings_are_reused(tmp_path, values, dtype, old_keys, cache_mode):
    src = pl.DataFrame({"date": [D(2024, 1, 10), D(2024, 1, 11)], "x": pl.Series(values, dtype=dtype), "value": [0, 1]})
    for index, key in enumerate(old_keys):
        if os.name == "nt":
            key = quote(key, safe=" -_.")
        path = tmp_path / "monthly" / key / "2024-01.parquet"
        path.parent.mkdir(parents=True)
        src.slice(index, 1).write_parquet(path)
    snapshot = _age_files(tmp_path)
    predicate = _JAN & (pl.col("x") == values[0])
    current = src if cache_mode == CACHE else src.with_columns(pl.col("value") + 100)
    for _ in range(2):
        cached = cache_parquet(current.lazy(), tmp_path, "date", extra_partition_cols="x", cache_mode=cache_mode)
        out = cached.filter(predicate).collect() if cache_mode == CACHE else cached.collect()
        expected = current.filter(predicate) if cache_mode == CACHE else current
        assert_frame_equal(_sorted(out), _sorted(expected))
    assert_frame_equal(_sorted(cache_parquet(current.lazy(), tmp_path, "date", extra_partition_cols="x").collect()), _sorted(current))
    if cache_mode == REBUILD:
        remaining = current.filter(pl.col("value") == 101)
        out = cache_parquet(remaining.lazy(), tmp_path, "date", extra_partition_cols="x", cache_mode=REBUILD).filter(predicate).collect()
        assert out.is_empty()
        assert_frame_equal(_sorted(cache_parquet(remaining.lazy(), tmp_path, "date", extra_partition_cols="x").collect()), _sorted(remaining))
    assert sorted(tmp_path.rglob("*.parquet")) == sorted(snapshot)
    if cache_mode == CACHE:
        assert _rewritten(snapshot) == []


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
    src = pl.LazyFrame({"date": [D(2024, 1, 10)], "x": ["other" if empty else value], "value": [0]})
    return src, pl.concat([src, pl.LazyFrame({"date": [D(2024, 1, 10)], "x": [value], "value": [99]})])


@pytest.mark.parametrize("value", ["New York", "a#b", "x?y", "%2541"])
def test_empty_marker_preserves_literal_partition_key(tmp_path, value):
    """The empty file for a value the source lacks keeps the value's key, so a later read finds it."""
    src, expanded = _empty_marker_sources(value, empty=True)
    _second_read_reuses_files(tmp_path, src, _JAN & (pl.col("x") == value), expanded, date_column="date", extra_partition_cols="x")


@pytest.mark.parametrize(
    ("value", "file_key"),
    [("%2541", "%252541"), ("%25", "%2525"), ("a%253Ab", "a%25253Ab"), ("x?y", "x%3Fy"), ("12:00", "12%3A00")],
)
@pytest.mark.parametrize("empty", [False, True])
def test_windows_partition_file_keys_round_trip_on_any_platform(tmp_path, monkeypatch, value, file_key, empty):
    module = importlib.import_module("polars_io_tools.io_sources.lazy_cache_parquet")
    monkeypatch.setattr(module, "os", SimpleNamespace(name="nt", environ=os.environ))
    src, expanded = _empty_marker_sources(value, empty)
    _second_read_reuses_files(tmp_path, src, _JAN & (pl.col("x") == value), expanded, date_column="date", extra_partition_cols="x")
    assert (tmp_path / "monthly" / file_key / "2024-01.parquet").exists()


# REBUILD scope: what a rebuild may overwrite or remove


def test_rebuild_without_date_constraint_removes_stale_partitions(tmp_path):
    src = pl.LazyFrame({"date": [D(2024, 1, 15), D(2024, 2, 15), D(2024, 1, 15)], "k": ["a", "a", "b"], "value": [0, 1, 2]})
    kwargs = {"time_unit": "monthly", "extra_partition_cols": "k"}

    def cached_files():
        return sorted(p.relative_to(tmp_path / "monthly").as_posix() for p in (tmp_path / "monthly").rglob("*.parquet"))

    cache_parquet(src, tmp_path, "date", **kwargs).collect()
    rebuilt = src.filter(pl.col("value") == 1)
    out = cache_parquet(rebuilt, tmp_path, "date", cache_mode=REBUILD, **kwargs).filter(pl.col("k") == "a").collect()
    assert out["value"].to_list() == [1]
    assert cached_files() == ["a/2024-02.parquet", "b/2024-01.parquet"]

    out = cache_parquet(rebuilt, tmp_path, "date", cache_mode=REBUILD, **kwargs).collect()
    assert out["value"].to_list() == [1]
    assert cached_files() == ["a/2024-02.parquet"]


def test_rebuild_null_date_scope_preserves_other_partitions(tmp_path):
    seed = pl.LazyFrame({"date": [D(2024, 1, 10), D(2024, 2, 10)], "value": [0, 1]})
    cache_parquet(seed, tmp_path, "date").collect()
    snapshot = _age_files(tmp_path)
    current = pl.LazyFrame({"date": pl.Series([None], dtype=pl.Date), "value": [99]})
    out = cache_parquet(current, tmp_path, "date", cache_mode=REBUILD).filter(pl.col("date").is_null()).collect()
    assert_frame_equal(out, current.collect())
    assert all(path.exists() for path in snapshot)
    assert _rewritten(snapshot) == []
    assert sorted(cache_parquet(current, tmp_path, "date").collect()["value"].to_list()) == [0, 1, 99]


def _rebuild_keeps(tmp_path, src, keep_value, current, predicate, **kwargs):
    """Cache ``src``, REBUILD ``current`` over ``predicate``, and return the untouched file holding ``keep_value``."""
    cache_parquet(src, tmp_path, **kwargs).collect()
    snapshot = _age_files(tmp_path)
    kept = next(path for path in snapshot if pl.read_parquet(path)["value"].to_list() == [keep_value])
    out = cache_parquet(current, tmp_path, cache_mode=REBUILD, **kwargs).filter(predicate).collect()
    assert_frame_equal(out, current.filter(predicate).collect())
    assert pl.read_parquet(kept)["value"].to_list() == [keep_value]
    assert kept.stat().st_mtime_ns == snapshot[kept]


@pytest.mark.parametrize(
    ("time_unit", "dates", "bound"),
    [
        ("daily", [D(2024, 1, 31), D(2024, 2, 1)], D(2024, 1, 31)),
        ("monthly", [D(2024, 1, 10), D(2024, 2, 10)], D(2024, 1, 31)),
        ("yearly", [D(2023, 12, 10), D(2024, 2, 10)], D(2023, 12, 31)),
    ],
)
def test_rebuild_exclusive_lower_bound_preserves_previous_partition(tmp_path, time_unit, dates, bound):
    src = pl.LazyFrame({"date": dates, "value": [0, 1]})
    _rebuild_keeps(tmp_path, src, 0, src.with_columns(pl.col("value") + 100), pl.col("date") > bound, date_column="date", time_unit=time_unit)


@pytest.mark.parametrize("tz", ["UTC", "America/New_York", "Asia/Tokyo"])
@pytest.mark.parametrize("partition_format", [None, "$year/$month"])
def test_rebuild_timezone_date_scope_with_unconstrained_extra_column(tmp_path, tz, partition_format):
    zone = ZoneInfo(tz)
    src = pl.LazyFrame(
        {"ts": pl.Series([DT(2024, 1, 10, tzinfo=zone), DT(2024, 2, 10, tzinfo=zone)], dtype=pl.Datetime("us", tz)), "x": ["a", "b"], "value": [0, 1]}
    )
    options = {"date_column": "ts", "extra_partition_cols": "x", "partition_format": partition_format}
    current = src.filter(pl.col("value") == 0).with_columns(pl.col("value") + 100)
    predicate = pl.col("ts").is_between(DT(2024, 1, 1, tzinfo=zone), DT(2024, 1, 31, tzinfo=zone))
    _rebuild_keeps(tmp_path, src, 1, current, predicate, **options)
    assert sorted(cache_parquet(current, tmp_path, **options).collect()["value"].to_list()) == [1, 100]


@pytest.mark.parametrize("nulls_equal", [False, True])
def test_rebuild_date_membership_respects_null_matching(tmp_path, nulls_equal):
    src = pl.LazyFrame({"date": [D(2024, 1, 10), None, D(2024, 2, 10)], "value": [0, 1, 2]})
    cache_parquet(src, tmp_path, "date").collect()
    snapshot = _age_files(tmp_path)
    null_path = next(path for path in snapshot if pl.read_parquet(path)["date"].null_count())
    current = src.filter(pl.col("value") == 0).with_columns(pl.col("value") + 100)
    predicate = pl.col("date").is_in([D(2024, 1, 10), None], nulls_equal=nulls_equal)
    out = cache_parquet(current, tmp_path, "date", cache_mode=REBUILD).filter(predicate).collect()
    assert_frame_equal(out, current.collect())
    assert null_path.exists() is not nulls_equal
    if not nulls_equal:
        assert null_path.stat().st_mtime_ns == snapshot[null_path]
    remaining = cache_parquet(current, tmp_path, "date").collect()["value"].to_list()
    assert sorted(remaining) == ([2, 100] if nulls_equal else [1, 2, 100])


def test_rebuild_scope_with_date_predicate_failing_on_null():
    pred = pl.col("date").map_elements(lambda d: d.weekday() < 5, return_dtype=pl.Boolean, skip_nulls=False)
    scope = _rebuild_partition_scope(pred, "date", [], "daily", pl.Schema({"date": pl.Date}))
    partitions = pl.DataFrame({"date": [D(2024, 3, 11), None]})
    assert scope is None or partitions.filter(scope).height == 2


# Signed zero keys: -0.0 and 0.0 are one partition


def _signed_zero(dtype, source_value, layout):
    source = pl.DataFrame({"key": pl.Series([source_value], dtype=dtype), "value": [7]})
    predicate = pl.col("key") == -source_value
    options = {"extra_partition_cols": "key"}
    if layout == "no-date":
        return source, predicate, options, 1
    source = source.with_columns(pl.lit(D(2024, 1, 10)).alias("date"))
    options.update(date_column="date", partition_format="$year/$month" if layout == "custom" else None)
    return source, predicate & pl.col("date").is_between(D(2024, 1, 1), D(2024, 2, 29)), options, 2


_signed_zero_axes = (
    pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64]),
    pytest.mark.parametrize("source_value", [-0.0, 0.0]),
    pytest.mark.parametrize("layout", ["no-date", "default", "custom"]),
)


@_signed_zero_axes[0]
@_signed_zero_axes[1]
@_signed_zero_axes[2]
@pytest.mark.parametrize("cache_mode", [CACHE, REBUILD])
@pytest.mark.parametrize("write_batch_size", [None, 1])
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
    refresh_options = {"cache_mode": REBUILD} if refresh == "rebuild" else {"ttl": datetime.timedelta(0)}
    for value in [107, 207]:
        current = source.with_columns(pl.lit(value, pl.Int64).alias("value"))
        visited = []
        result = cache_parquet(current.lazy(), tmp_path, file_visitor=visited.append, **options, **refresh_options).filter(predicate).collect()
        assert_frame_equal(result, current)
        assert len(visited) == len(set(visited))
    assert_frame_equal(cache_parquet(source.lazy(), tmp_path, cache_mode=CacheMode.CACHE_ONLY, **options).collect(), current)
    assert len(list(tmp_path.rglob("*.parquet"))) == expected_files
