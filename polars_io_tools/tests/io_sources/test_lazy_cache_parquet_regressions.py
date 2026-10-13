"""Regression tests for cache_parquet planning bugs (re-writes, lost rows, key parsing)."""

import datetime
import os
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_io_tools._compat import POLARS_HAS_PARTITION_BY
from polars_io_tools.io_sources.lazy_cache_parquet import _construct_partitions_df_from_existing, cache_parquet

D = datetime.date
DT = datetime.datetime
OLD_NS = 1_000_000_000 * 10**9


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


@pytest.mark.parametrize("pred", [pl.col("date") <= D(2024, 1, 31), pl.col("date") >= D(2024, 1, 1)])
def test_one_sided_date_predicate_on_empty_cache(tmp_path, pred):
    out = cache_parquet(MONTHLY_SRC, tmp_path, "date", time_unit="monthly").filter(pred).collect()
    assert_frame_equal(_sorted(out), _sorted(MONTHLY_SRC.filter(pred).collect()))


@pytest.mark.parametrize(("time_unit", "dtype"), [("monthly", pl.Date), ("daily", pl.Datetime("us"))])
def test_not_equal_date_does_not_drop_enclosing_partition(tmp_path, time_unit, dtype):
    src = MONTHLY_SRC.with_columns(pl.col("date").cast(dtype))
    excluded = D(2024, 1, 15) if dtype == pl.Date else DT(2024, 1, 15, 12)
    pred = pl.col("date").is_between(D(2024, 1, 1), D(2024, 2, 29)) & (pl.col("date") != excluded)
    expected = _sorted(src.filter(pred).collect())
    for _ in range(2):  # cold, then warm
        out = cache_parquet(src, tmp_path, "date", time_unit=time_unit).filter(pred).collect()
        assert_frame_equal(_sorted(out), expected)


def _jan_cached(tmp_path, src, **kwargs):
    jan = pl.col("date").is_between(D(2024, 1, 1), D(2024, 1, 31))
    cache_parquet(src, tmp_path, "date", **kwargs).filter(jan).collect()
    return _age_files(tmp_path)


@pytest.mark.parametrize("time_unit", ["monthly", "yearly"])
@pytest.mark.parametrize("pred", [pl.col("date") >= D(2024, 1, 1), None], ids=["unbounded", "unconstrained"])
def test_cached_partitions_not_rewritten_by_open_queries(tmp_path, time_unit, pred):
    src = MONTHLY_SRC
    snapshot = _jan_cached(tmp_path, src, time_unit=time_unit)
    lf = cache_parquet(src, tmp_path, "date", time_unit=time_unit)
    out = (lf.filter(pred) if pred is not None else lf).collect()
    assert_frame_equal(_sorted(out), _sorted(src.filter(pred).collect() if pred is not None else src.collect()))
    assert _rewritten(snapshot) == []
    if time_unit == "monthly":
        assert (tmp_path / "monthly" / "2024-02.parquet").exists()


def test_daily_datetime_column_partitions_not_rewritten(tmp_path):
    src = pl.LazyFrame({"date": [DT(2024, 1, 1, 9), DT(2024, 1, 1, 15), DT(2024, 1, 2, 9)], "value": [0, 1, 2]})
    day1 = pl.col("date").is_between(DT(2024, 1, 1), DT(2024, 1, 1, 23, 59))
    cache_parquet(src, tmp_path, "date", time_unit="daily").filter(day1).collect()
    snapshot = _age_files(tmp_path)
    pred = pl.col("date") >= DT(2024, 1, 1)
    out = cache_parquet(src, tmp_path, "date", time_unit="daily").filter(pred).collect()
    assert_frame_equal(_sorted(out), _sorted(src.collect()))
    assert _rewritten(snapshot) == []


def test_extra_col_only_predicate_does_not_rewrite_cached_dates(tmp_path):
    src = pl.LazyFrame(
        {
            "date": [D(2024, 1, 10), D(2024, 2, 10), D(2024, 1, 10), D(2024, 2, 10)],
            "sym": ["A", "A", "B", "B"],
            "value": [0, 1, 2, 3],
        }
    )
    kwargs = {"time_unit": "monthly", "extra_partition_cols": "sym"}
    jan = pl.col("date").is_between(D(2024, 1, 1), D(2024, 1, 31))
    cache_parquet(src, tmp_path, "date", **kwargs).filter(jan).collect()
    snapshot = _age_files(tmp_path)
    pred = pl.col("sym") == "A"
    out = cache_parquet(src, tmp_path, "date", **kwargs).filter(pred).collect()
    assert_frame_equal(_sorted(out), _sorted(src.filter(pred).collect()))
    assert _rewritten(snapshot) == []
    assert (tmp_path / "monthly" / "A" / "2024-02.parquet").exists()
    assert not (tmp_path / "monthly" / "B" / "2024-02.parquet").exists()


@pytest.mark.parametrize(
    ("fmt", "jan_file"),
    [
        ("$year/$month", "2024/01.parquet"),
        ("prefix_$year-$month", "prefix_2024-01.parquet"),
        ("year=$year/month=$month", "year=2024/month=01.parquet"),
    ],
)
def test_custom_partition_format_cache_hit(tmp_path, fmt, jan_file):
    snapshot = _jan_cached(tmp_path, MONTHLY_SRC, partition_format=fmt)
    assert (tmp_path / "monthly" / jan_file).exists()
    jan = pl.col("date").is_between(D(2024, 1, 1), D(2024, 1, 31))
    out = cache_parquet(MONTHLY_SRC, tmp_path, "date", partition_format=fmt).filter(jan).collect()
    assert_frame_equal(_sorted(out), _sorted(MONTHLY_SRC.filter(jan).collect()))
    assert _rewritten(snapshot) == []


def test_custom_partition_format_parsing():
    schema = pl.Schema({"sym": pl.String, "date": pl.Date})
    df = _construct_partitions_df_from_existing(
        {"A/2024/03", "B/2023/12"}, schema, ["sym", "date"], "$year/$month", date_column="date", time_unit="monthly"
    )
    expected = pl.DataFrame({"sym": ["A", "B"], "date": [D(2024, 3, 1), D(2023, 12, 1)]})
    assert_frame_equal(_sorted(df), expected)


def test_boolean_key_parsing_drops_unrecognized_keys():
    # "true"/"null" are the spellings of the polars < 1.38 writer
    schema = pl.Schema({"flag": pl.Boolean, "n": pl.Int64})
    keys = {"True/1", "False/None", "true/2", "false/null", "False/x", "yes/3"}
    df = _construct_partitions_df_from_existing(keys, schema, ["flag", "n"], None)
    assert sorted(df.rows(), key=str) == sorted([(True, 1), (False, None), (True, 2), (False, None)], key=str)


_EXTRA_COL_CASES = [
    ([True, False], pl.Boolean),
    ([0.5, 1e-05], pl.Float64),
    ([D(2020, 1, 1), D(2021, 1, 1)], pl.Date),
    ([DT(2020, 1, 1, 12), DT(2020, 1, 2, 0, 0, 0, 500)], pl.Datetime("us")),
    ([DT(2020, 1, 1, 12, tzinfo=ZoneInfo("UTC")), DT(2020, 1, 2, tzinfo=ZoneInfo("UTC"))], pl.Datetime("us", "Asia/Tokyo")),
    # Whole seconds: polars < 1.38 drops fractional seconds from Time keys
    ([datetime.time(12, 30), datetime.time(0, 0, 1)], pl.Time),
    (["1.23", "-2.50"], pl.Decimal(10, 2)),
]


@pytest.mark.parametrize(("values", "dtype"), _EXTRA_COL_CASES, ids=["boolean", "float", "date", "datetime", "datetime_tz", "time", "decimal"])
def test_extra_partition_col_dtypes_round_trip(tmp_path, values, dtype):
    series = pl.Series(values).cast(dtype) if isinstance(dtype, pl.Decimal) else pl.Series(values, dtype=dtype)
    src = pl.LazyFrame({"date": [D(2024, 1, 10), D(2024, 1, 11)], "x": series, "value": [0, 1]})
    kwargs = {"time_unit": "monthly", "extra_partition_cols": "x"}
    jan = pl.col("date").is_between(D(2024, 1, 1), D(2024, 1, 31))
    one = jan & (pl.col("x") == series[0])
    for _ in range(2):  # cold, then warm
        out = cache_parquet(src, tmp_path, "date", **kwargs).filter(one).collect()
        assert_frame_equal(_sorted(out), _sorted(src.filter(one).collect()))
    cache_parquet(src, tmp_path, "date", **kwargs).filter(jan).collect()
    snapshot = _age_files(tmp_path)
    assert len(snapshot) == 2
    for pred in (jan, None):
        lf = cache_parquet(src, tmp_path, "date", **kwargs)
        out = (lf.filter(pred) if pred is not None else lf).collect()
        assert_frame_equal(_sorted(out), _sorted(src.collect()))
    assert _rewritten(snapshot) == []
    assert sorted(tmp_path.rglob("*.parquet")) == sorted(snapshot)


@pytest.mark.parametrize(
    "x", [pl.Series([datetime.timedelta(hours=1)]), pl.Series([DT(2024, 1, 1)], dtype=pl.Datetime("ns"))], ids=["duration", "datetime_ns"]
)
def test_unsupported_extra_partition_col_is_rejected(tmp_path, x):
    src = pl.LazyFrame({"date": [D(2024, 1, 10)], "x": x})
    with pytest.raises(ValueError, match="cannot partition on Duration or nanosecond Datetime column x"):
        cache_parquet(src, tmp_path, "date", extra_partition_cols="x")


_NULLABLE_DATES = pl.LazyFrame({"date": [D(2024, 1, 10), None, D(2024, 2, 5)], "x": ["a", "a", "a"], "value": [0, 1, 2]})
_NULLABLE_X = pl.LazyFrame({"date": [D(2024, 1, 10), D(2024, 1, 11), D(2024, 1, 12)], "x": ["a", None, "b"], "value": [0, 1, 2]})
_JAN = pl.col("date").is_between(D(2024, 1, 1), D(2024, 1, 31))


@pytest.mark.parametrize(
    ("src", "pred"),
    [
        (_NULLABLE_DATES, pl.col("date").is_null() | (pl.col("date") == D(2024, 1, 10))),
        (_NULLABLE_DATES, pl.col("date").is_null() | _JAN),
        (_NULLABLE_DATES, (pl.col("x") == "a") & (pl.col("date").is_null() | (pl.col("date") == D(2024, 1, 10)))),
        (_NULLABLE_X, _JAN & (pl.col("x").is_null() | (pl.col("x") == "a"))),
        (_NULLABLE_X, pl.col("x").is_null() | (pl.col("date") == D(2024, 1, 10))),
    ],
    ids=["date_null_or_value", "date_null_or_range", "date_null_or_value_with_x", "x_null_or_value", "x_null_or_date"],
)
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_null_or_value_predicate_keeps_null_rows(tmp_path, src, pred, warm):
    kwargs = {"time_unit": "monthly", "extra_partition_cols": "x"}
    if warm:
        cache_parquet(src, tmp_path, "date", **kwargs).collect()
    out = cache_parquet(src, tmp_path, "date", **kwargs).filter(pred).collect()
    assert_frame_equal(_sorted(out), _sorted(src.filter(pred).collect()))


@pytest.mark.parametrize("fmt", [None, "$year/$month"])
def test_null_date_partition_is_cached(tmp_path, fmt):
    src = pl.LazyFrame({"date": [D(2024, 1, 10), None], "value": [0, 1]})
    kwargs = {"time_unit": "monthly", "partition_format": fmt}
    cache_parquet(src, tmp_path, "date", **kwargs).collect()
    snapshot = _age_files(tmp_path)
    assert len(snapshot) == 2
    out = cache_parquet(src, tmp_path, "date", **kwargs).collect()
    assert_frame_equal(_sorted(out), _sorted(src.collect()))
    assert _rewritten(snapshot) == []


@pytest.mark.parametrize(
    ("values", "dtype"),
    [(["a", None], pl.String), ([1, None], pl.Int64), ([True, None], pl.Boolean), (["a", None], pl.Enum(["a", "b"])), (["a", None], pl.Categorical)],
    ids=["str", "int", "bool", "enum", "categorical"],
)
@pytest.mark.parametrize("date_column", ["date", None])
def test_null_extra_partition_col_is_cached(tmp_path, values, dtype, date_column):
    src = pl.LazyFrame({"date": [D(2024, 1, 10), D(2024, 1, 11)], "x": pl.Series(values, dtype=dtype), "value": [0, 1]})
    kwargs = {"time_unit": "monthly", "extra_partition_cols": "x"}
    cache_parquet(src, tmp_path, date_column, **kwargs).collect()
    snapshot = _age_files(tmp_path)
    assert len(snapshot) == 2
    for pred in (None, None, pl.col("x").is_null(), pl.col("x").is_null() & (pl.col("date") == D(2024, 1, 11)), pl.col("x") == values[0]):
        lf = cache_parquet(src, tmp_path, date_column, **kwargs)
        out = (lf.filter(pred) if pred is not None else lf).collect()
        assert_frame_equal(_sorted(out), _sorted(src.filter(pred).collect() if pred is not None else src.collect()))
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


def test_boolean_partition_finite_query_reads_back(tmp_path):
    src = pl.LazyFrame({"date": [D(2024, 1, 10), D(2024, 1, 10)], "flag": [True, False], "value": [0, 1]})
    kwargs = {"time_unit": "monthly", "extra_partition_cols": "flag"}
    pred = (pl.col("flag") == True) & (pl.col("date") == D(2024, 1, 10))
    for _ in range(2):
        out = cache_parquet(src, tmp_path, "date", **kwargs).filter(pred).collect()
        assert out["value"].to_list() == [0]
    true_key = "True" if POLARS_HAS_PARTITION_BY else "true"
    assert sorted(p.parent.name for p in (tmp_path / "monthly").rglob("*.parquet")) == [true_key]


def test_decimal_partition_exclusion_without_date_column(tmp_path):
    src = pl.LazyFrame({"x": pl.Series(["1.23", "-2.50", "4.00"]).cast(pl.Decimal(10, 2)), "value": [0, 1, 2]})
    cache_parquet(src, tmp_path, None, extra_partition_cols="x").filter(pl.col("x") < 4).collect()
    snapshot = _age_files(tmp_path)
    out = cache_parquet(src, tmp_path, None, extra_partition_cols="x").collect()
    assert_frame_equal(_sorted(out), _sorted(src.collect()))
    assert _rewritten(snapshot) == []


def test_decimal_partition_finite_query_with_several_values(tmp_path):
    src = pl.LazyFrame({"date": [D(2024, 1, 10), D(2024, 1, 11)], "x": pl.Series(["1.23", "2.34"]).cast(pl.Decimal(10, 2)), "value": [0, 1]})
    pred = pl.col("date").is_between(D(2024, 1, 1), D(2024, 1, 31)) & pl.col("x").is_in(src.collect()["x"])
    out = cache_parquet(src, tmp_path, "date", extra_partition_cols="x").filter(pred).collect()
    assert_frame_equal(_sorted(out), _sorted(src.collect()))
