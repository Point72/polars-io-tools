"""Tests for the no-write cache modes of ``cache_parquet``: ``READ_ONLY`` and ``CACHE_ONLY``."""

import datetime
import os
import shutil
import subprocess
import sys
import textwrap
from decimal import Decimal

import polars as pl
import pydantic
import pytest
from polars.testing import assert_frame_equal

from polars_io_tools.io_sources import lazy_cache_parquet
from polars_io_tools.io_sources.lazy_cache_parquet import CacheMode, cache_parquet
from polars_io_tools.io_sources.util import register_io_source_with_is_pure

D = datetime.date

JAN_FEB = pl.col("date").is_between(D(2024, 1, 1), D(2024, 2, 29))


def _snapshot(root) -> dict[str, tuple[int, int]]:
    """Map every file and directory under ``root`` to (size, mtime_ns), to detect any mutation."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = os.path.join(dirpath, name)
            st = os.stat(path)
            out[os.path.relpath(path, root)] = (st.st_size, st.st_mtime_ns)
    return out


def _source(offset: int = 0) -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "date": [D(2024, 1, 10), D(2024, 1, 15), D(2024, 2, 15)],
            "value": [offset + 0, offset + 1, offset + 2],
        }
    )


@pytest.fixture
def jan_cached(tmp_path):
    """A monthly cache holding only January, written from values 0 and 1."""
    cache_parquet(_source(), tmp_path, "date", time_unit="monthly").filter(pl.col("date").is_between(D(2024, 1, 1), D(2024, 1, 31))).collect()
    return tmp_path


def _values(lf: pl.LazyFrame) -> list:
    return sorted(lf.collect()["value"].to_list())


@pytest.mark.parametrize(
    "predicate",
    [
        pytest.param(JAN_FEB, id="finite"),
        pytest.param(pl.col("date") >= D(2024, 1, 1), id="unbounded"),
        pytest.param(None, id="unconstrained"),
    ],
)
def test_read_only_serves_cached_partitions_and_fills_gaps_from_upstream(jan_cached, predicate):
    lf = cache_parquet(_source(offset=90), jan_cached, "date", time_unit="monthly", cache_mode=CacheMode.READ_ONLY)
    if predicate is not None:
        lf = lf.filter(predicate)
    # January comes from the cache (0, 1), February from upstream (92); no duplicates.
    assert _values(lf) == [0, 1, 92]


@pytest.mark.parametrize(
    "predicate",
    [
        pytest.param(JAN_FEB, id="finite"),
        pytest.param(pl.col("date") >= D(2024, 1, 1), id="unbounded"),
        pytest.param(None, id="unconstrained"),
    ],
)
def test_cache_only_serves_cached_partitions_only(jan_cached, predicate):
    lf = cache_parquet(_source(offset=90), jan_cached, "date", time_unit="monthly", cache_mode=CacheMode.CACHE_ONLY)
    if predicate is not None:
        lf = lf.filter(predicate)
    assert _values(lf) == [0, 1]


def test_no_write_modes_not_equal_date_keeps_enclosing_partition(jan_cached):
    predicate = JAN_FEB & (pl.col("date") != D(2024, 1, 15))
    read_only = cache_parquet(_source(offset=90), jan_cached, "date", time_unit="monthly", cache_mode=CacheMode.READ_ONLY)
    cache_only = cache_parquet(_source(offset=90), jan_cached, "date", time_unit="monthly", cache_mode=CacheMode.CACHE_ONLY)
    assert _values(read_only.filter(predicate)) == [0, 92]
    assert _values(cache_only.filter(predicate)) == [0]


@pytest.mark.parametrize("mode", [CacheMode.READ_ONLY, CacheMode.CACHE_ONLY])
def test_no_write_modes_do_not_mutate_cache(jan_cached, mode):
    before = _snapshot(jan_cached)
    cache_parquet(_source(offset=90), jan_cached, "date", time_unit="monthly", cache_mode=mode).filter(JAN_FEB).collect()
    cache_parquet(_source(offset=90), jan_cached, "date", time_unit="daily", cache_mode=mode).collect()
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
    lf = cache_parquet(_source(offset=90), jan_cached, "date", time_unit="monthly", cache_mode=mode)
    for predicate in [JAN_FEB, pl.col("date") >= D(2024, 1, 1), None]:
        (lf.filter(predicate) if predicate is not None else lf).collect()


@pytest.mark.parametrize("mode", [CacheMode.READ_ONLY, CacheMode.CACHE_ONLY])
def test_no_write_modes_on_missing_cache_directory(tmp_path, mode):
    cache_dir = tmp_path / "missing"
    result = cache_parquet(_source(), cache_dir, "date", cache_mode=mode).filter(JAN_FEB).collect()
    assert sorted(result["value"].to_list()) == ([0, 1, 2] if mode == CacheMode.READ_ONLY else [])
    assert result.schema == _source().collect_schema()
    assert not cache_dir.exists()


def test_read_only_cached_empty_partition_suppresses_upstream(tmp_path):
    # CACHE writes an empty file for February because upstream has no February rows yet.
    cache_parquet(_source().filter(pl.col("date") < D(2024, 2, 1)), tmp_path, "date", time_unit="monthly").filter(JAN_FEB).collect()
    assert (tmp_path / "monthly" / "2024-02.parquet").exists()
    lf = cache_parquet(_source(offset=90), tmp_path, "date", time_unit="monthly", cache_mode=CacheMode.READ_ONLY)
    assert _values(lf.filter(JAN_FEB)) == [0, 1]


def test_cache_only_never_invokes_callable_source(jan_cached):
    def fail():
        raise AssertionError("upstream must not be called")

    assert _values(cache_parquet(fail, jan_cached, "date", time_unit="monthly", cache_mode=CacheMode.CACHE_ONLY).filter(JAN_FEB)) == [0, 1]


def test_cache_only_callable_source_with_empty_cache(tmp_path):
    def fail():
        raise AssertionError("upstream must not be called")

    with pytest.raises(ValueError, match="requires `schema`"):
        cache_parquet(fail, tmp_path, "date", cache_mode=CacheMode.CACHE_ONLY)

    schema = _source().collect_schema()
    result = cache_parquet(fail, tmp_path, "date", schema=schema, cache_mode=CacheMode.CACHE_ONLY).collect()
    assert result.is_empty()
    assert result.schema == schema


def test_read_only_does_not_invoke_callable_when_source_not_needed_for_schema(jan_cached):
    calls = []

    def source():
        calls.append(1)
        return _source(offset=90)

    lf = cache_parquet(source, jan_cached, "date", time_unit="monthly", cache_mode=CacheMode.READ_ONLY)
    assert calls == []
    assert _values(lf.filter(JAN_FEB)) == [0, 1, 92]
    assert calls == [1]


@pytest.mark.parametrize(
    ("time_unit", "partition_format"),
    [
        ("daily", None),
        ("yearly", None),
        ("monthly", "$year/$month"),
        ("monthly", "prefix_$year-$month"),
    ],
)
def test_read_only_partition_formats(tmp_path, time_unit, partition_format):
    source = pl.LazyFrame({"date": [D(2023, 12, 30), D(2024, 1, 10), D(2024, 1, 15), D(2025, 3, 1)], "value": [0, 1, 2, 3]})
    warm = pl.col("date").is_between(D(2023, 12, 30), D(2024, 1, 12))
    cache_parquet(source, tmp_path, "date", time_unit=time_unit, partition_format=partition_format).filter(warm).collect()
    upstream = source.with_columns(pl.col("value") + 90)
    lf = cache_parquet(upstream, tmp_path, "date", time_unit=time_unit, partition_format=partition_format, cache_mode=CacheMode.READ_ONLY)
    # CACHE writes whole partitions, so a monthly/yearly cache also holds 2024-01-15.
    expected = [0, 1, 92, 93] if time_unit == "daily" else [0, 1, 2, 93]
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
    lf = cache_parquet(_source(offset=90), jan_cached, "date", time_unit="monthly", cache_mode=CacheMode.READ_ONLY, read_kwargs={"n_rows": 1})
    assert _values(lf.filter(JAN_FEB)) == [0]


def test_read_only_without_partition_columns(tmp_path):
    cache_parquet(_source(), tmp_path).collect()
    lf = cache_parquet(_source(offset=90), tmp_path, cache_mode=CacheMode.READ_ONLY)
    assert _values(lf) == [0, 1, 2]


def test_read_only_casts_upstream_to_declared_schema(jan_cached):
    upstream = _source(offset=90).with_columns(pl.col("value").cast(pl.Float64))
    lf = cache_parquet(upstream, jan_cached, "date", time_unit="monthly", cache_mode=CacheMode.READ_ONLY)
    assert lf.collect_schema()["value"] == pl.Int64
    result = lf.filter(JAN_FEB).with_columns(half=pl.col("value") // 2).collect()
    assert result.schema["value"] == pl.Int64
    assert sorted(result["half"].to_list()) == [0, 0, 46]


@pytest.mark.parametrize("predicate", [None, JAN_FEB, pl.col("date") >= D(2024, 1, 1)], ids=["unconstrained", "finite", "unbounded"])
@pytest.mark.parametrize("selection", [["date"], [pl.col("date").count()]], ids=["projection", "count"])
def test_read_only_ignores_missing_unrequested_upstream_columns(jan_cached, predicate, selection):
    before = _snapshot(jan_cached)
    upstream = _source().select("date")
    lf = cache_parquet(upstream, jan_cached, "date", cache_mode=CacheMode.READ_ONLY)
    if predicate is not None:
        lf = lf.filter(predicate)
    expected = pl.DataFrame({"date": [D(2024, 1, 10), D(2024, 1, 15), D(2024, 2, 15)]})
    assert_frame_equal(lf.select(selection).collect().sort(pl.all()), expected.select(selection).sort(pl.all()))
    assert _snapshot(jan_cached) == before


@pytest.mark.parametrize(
    "selection", [["value"], ["date"], ["region", "date"], [pl.col("value").count()]], ids=["value", "date", "reordered", "count"]
)
@pytest.mark.parametrize("engine", ["auto", "streaming"])
def test_read_only_projection_retains_filter_and_partition_columns(tmp_path, selection, engine):
    source = pl.LazyFrame(
        {
            "date": [D(2024, 1, 10), D(2024, 1, 15), D(2024, 2, 15), D(2024, 2, 15)],
            "region": ["A", "B", "A", "B"],
            "value": [0, 1, 2, 3],
            "unused": ["old"] * 4,
        }
    )
    cache_parquet(source, tmp_path, "date", extra_partition_cols="region").filter(
        (pl.col("date") == D(2024, 1, 10)) & (pl.col("region") == "A")
    ).collect()
    before = _snapshot(tmp_path)
    upstream = source.drop("unused").with_columns(pl.col("date").cast(pl.Datetime("ns")), (pl.col("value") + 90).cast(pl.String))
    lf = cache_parquet(upstream, tmp_path, "date", extra_partition_cols="region", cache_mode=CacheMode.READ_ONLY)
    expected = pl.DataFrame({"date": [D(2024, 1, 10), D(2024, 2, 15), D(2024, 2, 15)], "region": ["A", "A", "B"], "value": [0, 92, 93]})
    result = lf.filter(JAN_FEB & (pl.col("value") != 91)).select(selection).collect(engine=engine)
    assert_frame_equal(result.sort(pl.all()), expected.select(selection).sort(pl.all()))
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("requirement", ["projection", "predicate"])
def test_read_only_does_not_ignore_missing_required_upstream_columns(jan_cached, requirement):
    lf = cache_parquet(_source().select("date"), jan_cached, "date", cache_mode=CacheMode.READ_ONLY)
    query = lf.select("value") if requirement == "projection" else lf.filter(pl.col("value") > 0).select("date")
    with pytest.raises(pl.exceptions.ComputeError, match="value"):
        query.collect()


def test_read_only_applies_predicate_to_conformed_upstream(tmp_path):
    upstream = pl.LazyFrame({"date": [D(2024, 1, 10), D(2024, 1, 15)], "value": ["1", "2"]})
    schema = pl.Schema({"date": pl.Date, "value": pl.Int64})
    lf = cache_parquet(upstream, tmp_path, "date", schema=schema, cache_mode=CacheMode.READ_ONLY)
    assert lf.filter(pl.col("value") == 1).collect()["value"].to_list() == [1]


@pytest.mark.parametrize("date_dtype", [pl.Datetime("ns"), pl.Datetime("ms")])
@pytest.mark.parametrize("predicate", [JAN_FEB, pl.col("date") >= D(2024, 1, 1)], ids=["finite", "unbounded"])
def test_read_only_filters_upstream_partitions_after_casting_to_cached_schema(jan_cached, date_dtype, predicate):
    upstream = _source(offset=90).with_columns(pl.col("date").cast(date_dtype))
    lf = cache_parquet(upstream, jan_cached, "date", time_unit="monthly", cache_mode=CacheMode.READ_ONLY)
    assert _values(lf.filter(predicate)) == [0, 1, 92]


def test_read_only_pushes_predicate_to_upstream(jan_cached):
    pushed = []
    source = _source(offset=90)

    def generator(with_columns, predicate, n_rows, batch_size):
        pushed.append(predicate)
        yield source.filter(predicate).collect() if predicate is not None else source.collect()

    upstream = register_io_source_with_is_pure(io_source=generator, schema=source.collect_schema())
    lf = cache_parquet(upstream, jan_cached, "date", time_unit="monthly", cache_mode=CacheMode.READ_ONLY)
    assert _values(lf.filter(JAN_FEB)) == [0, 1, 92]
    assert pushed and pushed[-1] is not None
    assert source.filter(pushed[-1]).collect()["value"].to_list() == [92]


def test_read_only_null_partition_key_cached(tmp_path):
    source = pl.LazyFrame({"k": ["a", None], "value": [0, 1]})
    cache_parquet(source, tmp_path, extra_partition_cols="k").collect()
    assert len(list((tmp_path / "null").glob("*.parquet"))) == 2
    lf = cache_parquet(source.with_columns(pl.col("value") + 90), tmp_path, extra_partition_cols="k", cache_mode=CacheMode.READ_ONLY)
    assert _values(lf) == [0, 1]


def test_read_only_projection_and_limit(jan_cached):
    lf = cache_parquet(_source(offset=90), jan_cached, "date", time_unit="monthly", cache_mode=CacheMode.READ_ONLY)
    projected = lf.filter(JAN_FEB).select("value").collect()
    assert projected.columns == ["value"]
    assert sorted(projected["value"].to_list()) == [0, 1, 92]
    assert lf.filter(JAN_FEB).head(2).collect().height == 2


def test_read_only_streaming_collect(jan_cached):
    lf = cache_parquet(_source(offset=90), jan_cached, "date", time_unit="monthly", cache_mode=CacheMode.READ_ONLY)
    assert sorted(lf.filter(JAN_FEB).collect(engine="streaming")["value"].to_list()) == [0, 1, 92]


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


_MATRIX_DATES = [D(2024, 1, 10), D(2024, 1, 20), D(2024, 2, 10), D(2024, 3, 5), None]


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        (["A", "B", None], pl.String),
        ([True, False, None], pl.Boolean),
        (["A", "B", None], pl.Categorical),
        ([Decimal("1.5"), Decimal("2.5"), None], pl.Decimal(10, 2)),
        ([datetime.time(1), datetime.time(2), None], pl.Time),
    ],
    ids=["str", "bool", "categorical", "decimal", "time"],
)
@pytest.mark.parametrize("partition_format", [None, "$year/$month"])
@pytest.mark.parametrize(
    "predicate",
    [
        pytest.param(None, id="unconstrained"),
        pytest.param(pl.col("date") >= D(2024, 1, 1), id="unbounded"),
        pytest.param(pl.col("date").is_between(D(2024, 1, 1), D(2024, 3, 31)) & (pl.col("date") != D(2024, 1, 20)), id="not_equal"),
        pytest.param(pl.col("x").is_null() | (pl.col("date") == D(2024, 2, 10)), id="null_or"),
    ],
)
def test_read_only_matches_cache_mode(tmp_path, values, dtype, partition_format, predicate):
    rows = [(d, v) for d in _MATRIX_DATES for v in values]
    source = pl.LazyFrame({"date": [r[0] for r in rows], "x": pl.Series([r[1] for r in rows], dtype=dtype), "value": range(len(rows))})
    kwargs = {"date_column": "date", "time_unit": "monthly", "partition_format": partition_format, "extra_partition_cols": "x"}
    cache_parquet(source, tmp_path / "ro", **kwargs).filter(
        pl.col("date").is_between(D(2024, 1, 1), D(2024, 2, 29)) & (pl.col("x") == values[0])
    ).collect()
    shutil.copytree(tmp_path / "ro", tmp_path / "cache")
    before = _snapshot(tmp_path / "ro")
    upstream = source.with_columns(pl.col("value") + 1000)

    def run(path, mode):
        lf = cache_parquet(upstream, path, cache_mode=mode, **kwargs)
        return (lf.filter(predicate) if predicate is not None else lf).collect().sort(pl.all())

    cached = pl.col("date").is_between(D(2024, 1, 1), D(2024, 2, 29)) & (pl.col("x") == values[0])
    expected = source.with_columns(pl.when(cached).then(pl.col("value")).otherwise(pl.col("value") + 1000))
    expected = (expected.filter(predicate) if predicate is not None else expected).collect().sort(pl.all())

    read_only = run(tmp_path / "ro", CacheMode.READ_ONLY)
    assert_frame_equal(read_only, expected)
    assert_frame_equal(read_only, run(tmp_path / "cache", CacheMode.CACHE))
    assert (read_only["value"] < 1000).any() and (read_only["value"] >= 1000).any()
    assert _snapshot(tmp_path / "ro") == before
