"""sink_delta and scan_delta: logical dtypes stored as integers, their mapping in the table description, and predicates
translated to the stored values and pruned to Delta partitions.

Tables: a frame written and read back -> the same frame and the mapping in the description; and (Delta table,
predicate) -> scan_delta filtered by the predicate equals Polars filtering the written frame, with optional checks of
the plan and of partition pruning.
"""

import base64
from datetime import date, datetime, time, timedelta
from unittest.mock import patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_io_tools as cpl
from polars_io_tools._compat import POLARS_HAS_COLLECT_BATCHES
from polars_io_tools.io_sources import delta_io
from polars_io_tools.io_sources.delta_io import (
    _MAPPING_BLOCK_TAG,
    _get_partition_uris,
    build_delta_write_exprs,
    infer_logical_mapping,
    metadata_to_mapping,
)
from polars_io_tools.io_sources.util import _storage_options_for, extract_description_block, inject_description_block
from polars_io_tools.tests.helpers.assertions import assert_matches_native, assert_text
from polars_io_tools.tests.helpers.cases import Case, params

deltalake = pytest.importorskip("deltalake")
DeltaTable, write_deltalake = deltalake.DeltaTable, deltalake.write_deltalake

NS, MS, US = pl.Datetime("ns"), pl.Duration("ms"), pl.Datetime("us")


def _seconds(n: int, dtype=NS, start=datetime(2025, 1, 1)) -> pl.Series:
    return pl.Series([start + timedelta(seconds=i) for i in range(n)]).cast(dtype)


SAMPLE = pl.DataFrame(
    {
        "ts": pl.Series([datetime(2025, 1, 1) + timedelta(hours=i) for i in range(5)]).cast(NS),
        "dur_ms": pl.Series([0, 500, 1500, 2500, 4000], dtype=MS),
        "value": pl.Series([10, 20, 30, 40, 50], dtype=pl.Int64),
    }
)
SAMPLE_MAPPING = {"ts": NS, "dur_ms": MS}


@pytest.mark.parametrize(
    ("df", "mapping"),
    [
        pytest.param(SAMPLE, SAMPLE_MAPPING, id="infer_logical_mapping_from_schema_basic"),
        pytest.param(pl.DataFrame({"x": [1, 2], "y": ["a", "b"]}), {}, id="infer_logical_mapping_ignores_non_temporal"),
    ],
)
def test_infer_logical_mapping(df, mapping):
    assert infer_logical_mapping(df.lazy().collect_schema()) == mapping


def test_build_delta_write_exprs_shapes_temporal():
    schema = SAMPLE.lazy().collect_schema()
    out = SAMPLE.lazy().select(build_delta_write_exprs(schema, infer_logical_mapping(schema))).collect()
    assert dict(out.schema) == {"ts": pl.Int64, "dur_ms": pl.Int64, "value": pl.Int64}


def _description_mapping(path: str) -> dict | None:
    """The logical mapping in the table description, or None without a mapping block."""
    desc = DeltaTable(path).metadata().description
    payload = extract_description_block(desc, _MAPPING_BLOCK_TAG) if desc else None
    if payload is not None:
        assert_text(desc, [f"[{_MAPPING_BLOCK_TAG}:begin]", f"[{_MAPPING_BLOCK_TAG}:end]"])
    return None if payload is None else metadata_to_mapping(base64.b64decode(payload))


def _namespace_sink(lf, path, **kw):
    lf.piot.sink_delta(path, **kw)


def _polars_sink(lf, path, **kw):
    lf.collect().write_delta(path, **kw)


def R(id, df, mapping, sink=cpl.sink_delta, sink_kw=None, scan_kw=None):
    return Case(id, df, mapping, {"sink": sink, "sink_kw": sink_kw or {}, "scan_kw": scan_kw or {}})


ROUNDTRIPS = [
    *[
        R(
            f"delta_roundtrip_schema_and_values-{rechunk}-{pushdown}",
            SAMPLE,
            SAMPLE_MAPPING,
            _namespace_sink,
            scan_kw={"rechunk": rechunk, "pushdown_predicate_deltalake": pushdown},
        )
        for rechunk in (None, False, True)
        for pushdown in (False, True)
    ],
    R("delta_roundtrip_infer_mapping", SAMPLE, SAMPLE_MAPPING),
    # microsecond datetimes are stored natively, without a mapping
    R(
        "delta_microseconds_passthrough",
        pl.DataFrame({"ts": _seconds(2, US), "dur": pl.Series([100_000, 200_000], dtype=pl.Int64), "x": [1, 2]}),
        None,
    ),
    R("datetime_unit_fidelity_roundtrip", pl.DataFrame({"ts": pl.Series([datetime(2025, 1, 1, 0, 0, s) for s in (0, 1)]).cast(US)}), None),
    R("delta_metadata_us_precision_has_no_block", pl.DataFrame({"ts": _seconds(5, US), "x": [1, 2, 3, 4, 5]}), None),
    R("mixed_datetime_us_and_ns_roundtrip", pl.DataFrame({"ts_us": _seconds(2, US), "ts_ns": _seconds(2), "value": [10, 20]}), {"ts_ns": NS}),
    R("delta_streaming_write_batches", SAMPLE, SAMPLE_MAPPING, sink_kw={"chunk_size": 2}),
    R("scan_delta_reads_description_mapping", pl.DataFrame({"ts": _seconds(5)}), {"ts": NS}),
    R(
        "scan_delta_propagates_storage_options_passthrough",
        pl.DataFrame({"ts": _seconds(3)}),
        {"ts": NS},
        scan_kw={"storage_options": {"endpoint_url": "http://localhost:9000", "unused_key": "value"}},
    ),
    R(
        "delta_metadata_ms_precision_includes_block",
        pl.DataFrame({"dur_ms": pl.Series([0, 500, 1500, 2500, 4000], dtype=MS), "x": [1, 2, 3, 4, 5]}),
        {"dur_ms": MS},
    ),
    # a table written by Polars has no mapping and reads back as stored
    R("scan_delta_without_mapping_returns_plain", pl.DataFrame({"x": [1, 2, 3]}), None, _polars_sink),
]


@pytest.mark.parametrize(("df", "mapping", "sink", "sink_kw", "scan_kw"), params(ROUNDTRIPS, "pred", "expect", "sink", "sink_kw", "scan_kw"))
def test_roundtrip(tmp_path, df, mapping, sink, sink_kw, scan_kw):
    path = str(tmp_path / "t")
    sink(df.lazy(), path, mode="overwrite", **sink_kw)
    assert_frame_equal(cpl.scan_delta(path, **scan_kw).collect(), df)
    assert _description_mapping(path) == mapping


@pytest.mark.skipif(not POLARS_HAS_COLLECT_BATCHES, reason="collect_batches requires Polars >= 1.34.0")
@pytest.mark.parametrize(("chunk_size", "expect_batches"), [(-1, False), (2, True), (None, True)])
def test_delta_chunk_size_controls_collect_batches(tmp_path, chunk_size, expect_batches):
    """An append streams batches unless chunk_size is -1, for writers that need one atomic write."""
    path = str(tmp_path / "t")
    cpl.sink_delta(SAMPLE.lazy(), path, mode="overwrite")
    original = pl.LazyFrame.collect_batches
    called = []

    def detecting(self, *args, **kwargs):
        called.append(True)
        return original(self, *args, **kwargs)

    # chunking only applies to append/error/ignore modes
    with patch.object(pl.LazyFrame, "collect_batches", detecting):
        cpl.sink_delta(SAMPLE.lazy(), path, mode="append", chunk_size=chunk_size)
    assert bool(called) == expect_batches
    assert_frame_equal(cpl.scan_delta(path).collect(), pl.concat([SAMPLE, SAMPLE]), check_row_order=False)


@pytest.mark.parametrize(
    ("desc", "tag", "want"),
    [
        pytest.param(
            inject_description_block(None, "foo.block", "aGVsbG8="), "foo.block", "aGVsbG8=", id="inject_and_extract_roundtrip_block_helpers"
        ),
        pytest.param("some text without blocks", "foo", None, id="extract_missing_returns_none_block_helpers-text"),
        pytest.param(None, "foo", None, id="extract_missing_returns_none_block_helpers-none"),
        pytest.param("\n[foo:begin]\n  abc\n  def\n[foo:end]", "foo", "abc", id="extract_ignores_whitespace_and_takes_first_line_block_helpers"),
    ],
)
def test_extract_description_block(desc, tag, want):
    assert extract_description_block(desc, tag) == want


def test_scan_delta_invalid_block_raises(tmp_path):
    path = str(tmp_path / "t")
    pl.DataFrame({"x": [1, 2, 3]}).write_delta(
        path, mode="overwrite", delta_write_options={"description": f"\n[{_MAPPING_BLOCK_TAG}:begin]\nnot-base64\n[{_MAPPING_BLOCK_TAG}:end]"}
    )
    with pytest.raises(ValueError):
        cpl.scan_delta(path).collect()


# Tables the predicate rows read: name -> (frame, how it is written).
_NS_BASE = 1704198600000000000  # 2024-01-02 12:30:00
_NOON_30 = datetime(2024, 1, 2, 12, 30)
_MAPPED_PREDICATES = {
    "arithmetic": (pl.col("d") * 2) < timedelta(minutes=5),
    "column_difference": (pl.col("t") - pl.col("tm")) <= timedelta(0),
    "cast_to_string": pl.col("t").cast(pl.String).str.starts_with("2024-01-02"),
    "epoch": pl.col("t").dt.epoch("s") <= 1704198600,
    "year": pl.col("t").dt.year() == 2024,
    "time_hour": pl.col("tt").dt.hour() == 12,
    "total_seconds": pl.col("d").dt.total_seconds() >= 60,
    "truncate": pl.col("t").dt.truncate("1h") == datetime(2024, 1, 2, 12),
    "offset_by": pl.col("t").dt.offset_by("1h") > _NOON_30,
    "strftime": pl.col("t").dt.strftime("%Y") == "2024",
    "replace_time_zone": pl.col("t").dt.replace_time_zone("UTC") >= datetime(2024, 1, 1).astimezone(),
    "subtract_literal": (pl.col("tm") - _NOON_30) >= timedelta(0),
    "date_is_in": pl.col("t").dt.date().is_in([date(2024, 1, 2)]),
    "date_is_between": pl.col("t").dt.date().is_between(date(2024, 1, 1), date(2024, 1, 3)),
    "date_eq": pl.col("t").dt.date() == date(2024, 1, 2),
    "cast_date_le": pl.col("t").cast(pl.Date) <= date(2024, 1, 2),
    "cast_ms_eq": pl.col("t").cast(pl.Datetime("ms")) == _NOON_30,
    "cast_duration_us_eq": pl.col("d").cast(pl.Duration("us")) == timedelta(minutes=1),
    "cast_time_to_duration": pl.col("tt").cast(pl.Duration("ns")) <= timedelta(hours=13),
    "ns_upper_bound": pl.col("t") <= _NOON_30,
    "ns_literal_eq": pl.col("t") == pl.lit(_NS_BASE + 500).cast(pl.Datetime("ns")),
    "ns_is_in": pl.col("t").is_in(pl.Series([_NS_BASE + 500]).cast(pl.Datetime("ns")).implode()),
    "duration_upper_bound": pl.col("d") <= timedelta(minutes=1),
    "ne_missing": pl.col("x").ne_missing(1),
    "not_eq_missing": ~pl.col("x").eq_missing(1),
    "eq_missing_column": pl.col("x").eq_missing(pl.col("x")),
    "is_in_nulls_equal": pl.col("x").is_in([1, None], nulls_equal=True),
    "is_between_string_literals": pl.col("s").is_between(pl.lit("a"), pl.lit("b")),
    "is_between_string_not_columns": pl.col("s").is_between(pl.lit("q"), pl.lit("r")),
    "is_null_non_strict_cast": pl.col("ss").cast(pl.Int64, strict=False).is_null(),
    "and_untranslatable_side": (pl.col("t").dt.hour() == 12) & (pl.col("x") > 1),
    "or_untranslatable_side": (pl.col("t").dt.hour() == 12) | (pl.col("x") > 1),
    "not_untranslatable": ~((pl.col("t").dt.hour() == 12) & (pl.col("x") > 1)),
}

_MAPPED = pl.DataFrame(
    {
        "t": pl.Series([_NS_BASE, _NS_BASE + 500, _NS_BASE + 1500, None]).cast(NS),
        "tm": pl.Series([_NS_BASE // 10**6, _NS_BASE // 10**6 - 1, _NS_BASE // 10**6 + 1, _NS_BASE // 10**6]).cast(pl.Datetime("ms")),
        "d": pl.Series([60 * 10**9, 60 * 10**9 + 500, 30 * 10**9, None]).cast(pl.Duration("ns")),
        "tt": pl.Series([45000 * 10**9, 45000 * 10**9 + 500, 46800 * 10**9, None]).cast(pl.Time),
        "x": [1, 2, None, 3],
        "s": ["a", "ab", "b", "c"],
        "ss": ["1", "x", None, "2"],
        "a": ["zz"] * 4,
        "b": ["zzz"] * 4,
    }
)
_BASE = datetime(2025, 1, 1)
_BASE_NS = 1735689600 * 10**9
TABLES = {
    "sample": (SAMPLE, None),
    "mapped": (_MAPPED, None),
    # p in {"A", "B", None}, partitioned by p
    "partitioned": (pl.DataFrame({"x": [1, 2, 10, 20, 100], "p": ["A", "A", "B", "B", None]}), ["p"]),
    "int_partitioned": (
        pl.DataFrame({"a": [1, 1, 2, None, 257], "b": ["xx", "xyz", "xx", "xx", "xx"]}, schema={"a": pl.Int64, "b": pl.String}),
        ["a"],
    ),
    # Delta stores timestamps in microseconds
    "ts_partitioned": (pl.DataFrame({"ts": _seconds(2, US), "x": [1, 2]}), ["ts"]),
    # timestamps one nanosecond apart
    "ns": (pl.DataFrame({"ts": pl.Series([_BASE_NS, _BASE_NS + 1, _BASE_NS + 2]).cast(NS), "id": [10, 20, 30]}), None),
    "readme": (
        pl.DataFrame(
            {
                "ts": _seconds(4),
                "dur_ns": pl.Series([0, 500, 1500, 2500], dtype=pl.Duration("ns")),
                "t": pl.Series([time(0, 0, s) for s in range(4)], dtype=pl.Time),
            }
        ),
        None,
    ),
    **{
        f"mapping_{u}": (
            pl.DataFrame({"ts": [datetime(2024, 1, 1), datetime(2024, 1, 2)], "val": [10, 20]}, schema_overrides={"ts": pl.Datetime(u)}),
            None,
        )
        for u in ("us", "ns")
    },
}


@pytest.fixture(scope="module")
def delta_tables(tmp_path_factory):
    """Each table of TABLES written once: by sink_delta, or by deltalake partitioned by the given columns."""
    root = tmp_path_factory.mktemp("delta")
    paths = {}
    for name, (df, partition_by) in TABLES.items():
        paths[name] = str(root / name)
        if partition_by is None:
            cpl.sink_delta(df.lazy(), paths[name], mode="overwrite")
        else:
            write_deltalake(paths[name], df.to_arrow(), mode="overwrite", partition_by=partition_by)
    return paths


def _lit_ns(ns: int) -> pl.Expr:
    """A nanosecond datetime literal built without passing through a Python datetime."""
    return pl.lit(0, dtype=NS) + pl.duration(nanoseconds=ns)


def P(id, table, pred, op=None, plan=False, uris=None, extra=None, **scan):
    """``plan``: the filter is pushed into the scan (SELECTION, no FILTER). ``uris``: how deltalake listed files —
    "unused" (no partition pruning attempted), "used", or "unpruned" (listed without partition filters). ``extra``:
    rows the scan may also return beyond Polars' filter, depending on the Polars version."""
    return Case(id, pred, kw={"table": table, "op": op, "plan": plan, "uris": uris, "extra": extra, "scan": scan})


p, ts = pl.col("p"), pl.col("ts")
PREDICATES = [
    P("delta_predicate_rewrite_datetime", "sample", ts >= datetime(2025, 1, 1, 2), plan=True),
    P("delta_predicate_rewrite_duration", "sample", pl.col("dur_ms") < timedelta(milliseconds=2000), plan=True),
    P("scan_delta_no_pushdown_calls_file_uris", "partitioned", p == "A", uris="unused", pushdown_predicate_deltalake=False),
    P("scan_delta_pushdown_calls_file_uris", "partitioned", p == "A", uris="used", pushdown_predicate_deltalake=True),
    P("delta_partition_pushdown_eq", "partitioned", p == "A"),
    P("delta_partition_pushdown_eq-cast_literal", "partitioned", p == pl.lit("A").cast(pl.Utf8)),
    P("delta_partition_pushdown_in", "partitioned", p.is_in(["B"])),
    # a range on a partition column cannot prune
    # the null partition may come back too
    P(
        "delta_partition_pushdown_all_unsupported_ops_fallback",
        "partitioned",
        p < "B",
        uris="unpruned",
        extra=p.is_null(),
        pushdown_predicate_deltalake=True,
    ),
    P(
        "delta_partition_pushdown_mixed_or_clause_unsafe_prune",
        "partitioned",
        (p < "B") | (p == "B") | p.is_null(),
        uris="unpruned",
        pushdown_predicate_deltalake=True,
    ),
    P("delta_partition_pushdown_not_in", "partitioned", ~p.is_in(["A"]) | p.is_null()),
    # "" is Delta's null partition token, so it must not reach file_uris inside a "not in"
    *[
        P(f"delta_partition_pushdown_not_in_empty_string-{n}", "partitioned", q)
        for n, q in [("empty", ~p.is_in([""])), ("empty-and-value", ~p.is_in(["", "B"])), ("or-empty", (p == "A") | ~p.is_in([""]))]
    ],
    P("delta_partition_pushdown_is_null", "partitioned", p.is_null()),
    P("scan_delta_ne_missing_keeps_null_partition", "int_partitioned", pl.col("a").ne_missing(1)),
    # 257 wraps to 1, so its partition must stay
    P("scan_delta_narrowing_cast_does_not_prune_wrapped_partition", "int_partitioned", pl.col("a").cast(pl.Int8, wrap_numerical=True) == 1),
    P("delta_partition_pushdown_datetime_mapped_eq", "ts_partitioned", ts == pl.lit(_BASE + timedelta(seconds=1)).cast(NS)),
    *[
        P(
            f"delta_ns_precision_unit-{n}",
            "ns",
            q,
            plan=plan,
            storage_options=_storage_options_for("unused", aws_profile=None).polars,
            aws_profile=None,
        )
        for n, q, plan in [
            ("ge", ts >= _lit_ns(_BASE_NS + 1), True),
            ("le", ts <= _lit_ns(_BASE_NS + 1), True),
            ("gt", ts > _lit_ns(_BASE_NS + 1), False),
            ("lt", ts < _lit_ns(_BASE_NS + 2), False),
            *[
                (f"between_{c}", ts.is_between(_lit_ns(_BASE_NS + 1), _lit_ns(_BASE_NS + 2), closed=c), True)
                for c in ("both", "left", "right", "none")
            ],
        ]
    ],
    P(
        "readme_delta_example",
        "readme",
        (ts >= datetime(2025, 1, 1, 0, 0, 2)) & (pl.col("dur_ns") >= pl.duration(nanoseconds=1500)) & (pl.col("t") >= time(0, 0, 1)),
    ),
    *[
        P(f"scan_delta_mapping_filter_then_select-{u}", f"mapping_{u}", ts == datetime(2024, 1, 2), op=lambda lf: lf.select("val"))
        for u in ("us", "ns")
    ],
    # file pruning off: only the translated predicate pushed to the Parquet scan is exercised
    *[P(f"scan_delta_mapped_predicate_matches_eager-{n}", "mapped", q, pushdown_predicate_deltalake=False) for n, q in _MAPPED_PREDICATES.items()],
]


@pytest.mark.parametrize(("pred", "kw"), [pytest.param(c.pred, c.kw, id=c.id) for c in PREDICATES])
def test_scan_delta_matches_polars(delta_tables, monkeypatch, pred, kw):
    calls = {"partition_uris": 0, "file_uris_filters": []}
    get_uris, file_uris = delta_io._get_partition_uris, DeltaTable.file_uris

    def spy_uris(*args, **kwargs):
        calls["partition_uris"] += 1
        return get_uris(*args, **kwargs)

    def spy_file_uris(self, *args, **kwargs):
        calls["file_uris_filters"].append(kwargs.get("partition_filters"))
        return file_uris(self, *args, **kwargs)

    monkeypatch.setattr(delta_io, "_get_partition_uris", spy_uris)
    monkeypatch.setattr(DeltaTable, "file_uris", spy_file_uris)
    op = kw["op"] or (lambda lf: lf)
    lf = cpl.scan_delta(delta_tables[kw["table"]], credential_provider=None, **kw["scan"])
    native = TABLES[kw["table"]][0].lazy()
    if kw["extra"] is None:
        assert assert_matches_native(op(lf.filter(pred)), op(native.filter(pred)), None) is not None
    else:

        def rows(lf):
            df = lf.collect()
            return set(df.select(sorted(df.columns)).rows())

        assert rows(op(native.filter(pred))) <= rows(op(lf.filter(pred))) <= rows(op(native.filter(pred | kw["extra"])))
    if kw["plan"]:
        assert_text(lf.filter(pred).explain(), "SELECTION", omits="FILTER")
    if kw["uris"] == "unused":
        assert calls["partition_uris"] == 0
    elif kw["uris"] == "used":
        assert calls["partition_uris"] >= 1
    elif kw["uris"] == "unpruned":
        assert None in calls["file_uris_filters"]


def test_delta_partition_pushdown_datetime_mapped_eq_prunes_uris(delta_tables):
    """An equality on a datetime partition column lists only the matching partition's files."""
    dt = DeltaTable(delta_tables["ts_partitioned"])
    pruned = _get_partition_uris(dt, ts == pl.lit(_BASE + timedelta(seconds=1)).cast(NS), mapping=None)
    assert 1 <= len(pruned) < len(dt.file_uris())


def test_delta_partition_pushdown_ne_keeps_empty_string(tmp_path):
    """delta-rs reads the "" partition as null, so "!=" must not prune it; the source returns it."""
    path = str(tmp_path / "t")
    write_deltalake(path, pl.DataFrame({"c": ["a", "", "b"], "i": [0, 1, 2]}).to_arrow(), mode="overwrite", partition_by=["c"])
    assert sorted(cpl.scan_delta(path, credential_provider=None).filter(pl.col("c") != "a").collect()["i"]) == [1, 2]


def _run_delta_io_roundtrip(path: str, aws_profile: str | None) -> None:
    """Overwrite, append through the streaming path, and filter like a native scan of the stored integers.

    Also run against S3 by the integration tests.
    """
    cred = None if aws_profile is None else pl.CredentialProviderAWS(profile_name=aws_profile)
    df = pl.DataFrame({"ts": _seconds(3, start=datetime(2024, 6, 1)), "dur_ms": pl.Series([0, 500, 1500], dtype=MS), "value": [11, 22, 33]})
    df.lazy().piot.sink_delta(path, mode="overwrite", credential_provider=cred)
    more = pl.DataFrame({"ts": _seconds(1, start=datetime(2024, 6, 1, 0, 0, 3)), "dur_ms": pl.Series([2000], dtype=MS), "value": [44]})
    more.lazy().piot.sink_delta(path, mode="append", credential_provider=cred, chunk_size=2)
    assert_frame_equal(cpl.scan_delta(path, credential_provider=cred).sort("ts").collect(), pl.concat([df, more]))

    threshold = datetime(2024, 6, 1, 0, 0, 2)
    storage_options = _storage_options_for(path, aws_profile=aws_profile).polars
    native = pl.scan_delta(path, storage_options=storage_options, credential_provider=cred).filter(
        ts >= pl.Series([threshold], dtype=NS).dt.timestamp("ns").item()
    )
    logical = cpl.scan_delta(path, credential_provider=cred).filter(ts >= threshold)
    assert logical.sort("ts").collect()["value"].to_list() == native.sort("ts").collect()["value"].to_list() == [33, 44]


def test_delta_io_roundtrip(tmp_path):
    _run_delta_io_roundtrip(str(tmp_path / "t"), aws_profile=None)
