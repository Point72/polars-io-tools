"""sink_delta and scan_delta: logical dtypes stored as integers, their mapping in the table description, and predicates
translated to the stored values and pruned to Delta partitions.

Tables: frames written and read back, which must come back unchanged with the expected dtype mapping stored in the
table description; and filters on Delta tables, where scan_delta must return what Polars returns when it filters the
written frame, with optional checks of the query plan and of which partitions are listed.

The property test at the end (using the Hypothesis library) checks that partition pruning keeps every row a random
filter keeps.
"""

import base64
import math
import shutil
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field
from datetime import date, datetime, time, timedelta
from unittest.mock import patch

import polars as pl
import pytest
from hypothesis import given, note, strategies as st
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
    scan_delta,
    sink_delta,
)
from polars_io_tools.io_sources.util import _storage_options_for, extract_description_block
from polars_io_tools.tests.helpers.assertions import assert_matches_plain_polars, assert_translated_text
from polars_io_tools.tests.helpers.cases import as_params, epoch_ns
from polars_io_tools.tests.helpers.pickling import pickle_round_trip
from polars_io_tools.tests.helpers.property_checks import SETTINGS, show
from polars_io_tools.tests.helpers.strategies import (
    POLARS_ERRORS,
    random_filter,
    series_from_raw,
)

deltalake = pytest.importorskip("deltalake")
DeltaTable, write_deltalake = deltalake.DeltaTable, deltalake.write_deltalake

NANOSECOND_DATETIME, MILLISECOND_DURATION, MICROSECOND_DATETIME = pl.Datetime("ns"), pl.Duration("ms"), pl.Datetime("us")


def _consecutive_seconds(n: int, dtype=NANOSECOND_DATETIME, start=datetime(2025, 1, 1)) -> pl.Series:
    return pl.Series([start + timedelta(seconds=i) for i in range(n)]).cast(dtype)


SAMPLE = pl.DataFrame(
    {
        "ts": pl.Series([datetime(2025, 1, 1) + timedelta(hours=i) for i in range(5)]).cast(NANOSECOND_DATETIME),
        "dur_ms": pl.Series([0, 500, 1500, 2500, 4000], dtype=MILLISECOND_DURATION),
        "value": pl.Series([10, 20, 30, 40, 50], dtype=pl.Int64),
    }
)
SAMPLE_MAPPING = {"ts": NANOSECOND_DATETIME, "dur_ms": MILLISECOND_DURATION}


@dataclass(frozen=True)
class LogicalMappingCase:
    """A frame, and the dtype mapping ``infer_logical_mapping`` infers from its schema."""

    id: str
    _: KW_ONLY
    df: pl.DataFrame
    mapping: dict | None


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            LogicalMappingCase("infer_logical_mapping_from_schema_basic", df=SAMPLE, mapping=SAMPLE_MAPPING),
            LogicalMappingCase("infer_logical_mapping_ignores_non_temporal", df=pl.DataFrame({"x": [1, 2], "y": ["a", "b"]}), mapping={}),
        ]
    ),
)
def test_infer_logical_mapping(case: LogicalMappingCase):
    assert infer_logical_mapping(case.df.lazy().collect_schema()) == case.mapping


def test_build_delta_write_exprs_shapes_temporal():
    schema = SAMPLE.lazy().collect_schema()
    out = SAMPLE.lazy().select(build_delta_write_exprs(schema, infer_logical_mapping(schema))).collect()
    assert dict(out.schema) == {"ts": pl.Int64, "dur_ms": pl.Int64, "value": pl.Int64}


def _description_mapping(path: str) -> dict | None:
    """The logical mapping in the table description, or None without a mapping block."""
    desc = DeltaTable(path).metadata().description
    payload = extract_description_block(desc, _MAPPING_BLOCK_TAG) if desc else None
    if payload is not None:
        assert_translated_text(desc, [f"[{_MAPPING_BLOCK_TAG}:begin]", f"[{_MAPPING_BLOCK_TAG}:end]"])
    return None if payload is None else metadata_to_mapping(base64.b64decode(payload))


@dataclass(frozen=True)
class DeltaRoundtripCase:
    """A frame written to a Delta table and read back, which must give the frame again."""

    id: str
    _: KW_ONLY
    frame: pl.DataFrame
    # the dtype each stored column is read as, which the table description must hold; None: no mapping is stored
    expected_mapping: dict | None
    # called as ``sink(lazy_frame, path, **sink_options)``
    sink: Callable = cpl.sink_delta
    sink_options: dict = field(default_factory=dict)
    # keyword arguments for ``scan_delta``
    scan_options: dict = field(default_factory=dict)


_RECHUNK_IDS = {None: "default_rechunk", False: "no_rechunk", True: "rechunk"}
ROUNDTRIPS = [
    *[
        DeltaRoundtripCase(
            f"delta_roundtrip_schema_and_values-{_RECHUNK_IDS[rechunk]}-{'deltalake_pushdown' if pushdown else 'no_deltalake_pushdown'}",
            frame=SAMPLE,
            expected_mapping=SAMPLE_MAPPING,
            sink=lambda lf, path, **kw: lf.piot.sink_delta(path, **kw),
            scan_options={"rechunk": rechunk, "pushdown_predicate_deltalake": pushdown},
        )
        for rechunk in (None, False, True)
        for pushdown in (False, True)
    ],
    DeltaRoundtripCase("delta_roundtrip_infer_mapping", frame=SAMPLE, expected_mapping=SAMPLE_MAPPING),
    # microsecond datetimes are stored natively, without a mapping
    DeltaRoundtripCase(
        "delta_microseconds_passthrough",
        frame=pl.DataFrame({"ts": _consecutive_seconds(2, MICROSECOND_DATETIME), "dur": pl.Series([100_000, 200_000], dtype=pl.Int64), "x": [1, 2]}),
        expected_mapping=None,
    ),
    DeltaRoundtripCase(
        "datetime_unit_fidelity_roundtrip",
        frame=pl.DataFrame({"ts": pl.Series([datetime(2025, 1, 1, 0, 0, s) for s in (0, 1)]).cast(MICROSECOND_DATETIME)}),
        expected_mapping=None,
    ),
    DeltaRoundtripCase(
        "delta_metadata_us_precision_has_no_block",
        frame=pl.DataFrame({"ts": _consecutive_seconds(5, MICROSECOND_DATETIME), "x": [1, 2, 3, 4, 5]}),
        expected_mapping=None,
    ),
    DeltaRoundtripCase(
        "mixed_datetime_us_and_ns_roundtrip",
        frame=pl.DataFrame({"ts_us": _consecutive_seconds(2, MICROSECOND_DATETIME), "ts_ns": _consecutive_seconds(2), "value": [10, 20]}),
        expected_mapping={"ts_ns": NANOSECOND_DATETIME},
    ),
    DeltaRoundtripCase("delta_streaming_write_batches", frame=SAMPLE, expected_mapping=SAMPLE_MAPPING, sink_options={"chunk_size": 2}),
    DeltaRoundtripCase(
        "scan_delta_reads_description_mapping", frame=pl.DataFrame({"ts": _consecutive_seconds(5)}), expected_mapping={"ts": NANOSECOND_DATETIME}
    ),
    DeltaRoundtripCase(
        "scan_delta_propagates_storage_options_passthrough",
        frame=pl.DataFrame({"ts": _consecutive_seconds(3)}),
        expected_mapping={"ts": NANOSECOND_DATETIME},
        scan_options={"storage_options": {"endpoint_url": "http://localhost:9000", "unused_key": "value"}},
    ),
    DeltaRoundtripCase(
        "delta_metadata_ms_precision_includes_block",
        frame=pl.DataFrame({"dur_ms": pl.Series([0, 500, 1500, 2500, 4000], dtype=MILLISECOND_DURATION), "x": [1, 2, 3, 4, 5]}),
        expected_mapping={"dur_ms": MILLISECOND_DURATION},
    ),
    # a table written by Polars has no mapping and reads back as stored
    DeltaRoundtripCase(
        "scan_delta_without_mapping_returns_plain",
        frame=pl.DataFrame({"x": [1, 2, 3]}),
        expected_mapping=None,
        sink=lambda lf, path, **kw: lf.collect().write_delta(path, **kw),
    ),
]


@pytest.mark.parametrize("case", as_params(ROUNDTRIPS))
def test_roundtrip(tmp_path, case: DeltaRoundtripCase):
    path = str(tmp_path / "t")
    case.sink(case.frame.lazy(), path, mode="overwrite", **case.sink_options)
    assert_frame_equal(cpl.scan_delta(path, **case.scan_options).collect(), case.frame)
    assert _description_mapping(path) == case.expected_mapping


@dataclass(frozen=True)
class ChunkSizeCase:
    """A ``chunk_size`` for an append, and whether the append streams batches."""

    id: str
    _: KW_ONLY
    chunk_size: int | None
    expect_batches: bool


@pytest.mark.skipif(not POLARS_HAS_COLLECT_BATCHES, reason="collect_batches requires Polars >= 1.34.0")
@pytest.mark.parametrize(
    "case",
    as_params(
        [
            ChunkSizeCase("minus_one_writes_at_once", chunk_size=-1, expect_batches=False),
            ChunkSizeCase("two_streams_batches", chunk_size=2, expect_batches=True),
            ChunkSizeCase("default_streams_batches", chunk_size=None, expect_batches=True),
        ]
    ),
)
def test_delta_chunk_size_controls_collect_batches(tmp_path, case: ChunkSizeCase):
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
        cpl.sink_delta(SAMPLE.lazy(), path, mode="append", chunk_size=case.chunk_size)
    assert bool(called) == case.expect_batches
    assert_frame_equal(cpl.scan_delta(path).collect(), pl.concat([SAMPLE, SAMPLE]), check_row_order=False)


def test_scan_delta_invalid_block_raises(tmp_path):
    path = str(tmp_path / "t")
    pl.DataFrame({"x": [1, 2, 3]}).write_delta(
        path, mode="overwrite", delta_write_options={"description": f"\n[{_MAPPING_BLOCK_TAG}:begin]\nnot-base64\n[{_MAPPING_BLOCK_TAG}:end]"}
    )
    with pytest.raises(ValueError):
        cpl.scan_delta(path).collect()


# Tables the predicate rows read: name -> (frame, how it is written).
_JAN_2_2024_12_30 = datetime(2024, 1, 2, 12, 30)
_JAN_2_2024_12_30_NS = epoch_ns(_JAN_2_2024_12_30)
_MAPPED_PREDICATES = {
    "arithmetic": (pl.col("d") * 2) < timedelta(minutes=5),
    "column_difference": (pl.col("t") - pl.col("tm")) <= timedelta(0),
    "cast_to_string": pl.col("t").cast(pl.String).str.starts_with("2024-01-02"),
    "epoch": pl.col("t").dt.epoch("s") <= 1704198600,
    "year": pl.col("t").dt.year() == 2024,
    "time_hour": pl.col("tt").dt.hour() == 12,
    "total_seconds": pl.col("d").dt.total_seconds() >= 60,
    "truncate": pl.col("t").dt.truncate("1h") == datetime(2024, 1, 2, 12),
    "offset_by": pl.col("t").dt.offset_by("1h") > _JAN_2_2024_12_30,
    "strftime": pl.col("t").dt.strftime("%Y") == "2024",
    "replace_time_zone": pl.col("t").dt.replace_time_zone("UTC") >= datetime(2024, 1, 1).astimezone(),
    "subtract_literal": (pl.col("tm") - _JAN_2_2024_12_30) >= timedelta(0),
    "date_is_in": pl.col("t").dt.date().is_in([date(2024, 1, 2)]),
    "date_is_between": pl.col("t").dt.date().is_between(date(2024, 1, 1), date(2024, 1, 3)),
    "date_eq": pl.col("t").dt.date() == date(2024, 1, 2),
    "cast_date_le": pl.col("t").cast(pl.Date) <= date(2024, 1, 2),
    "cast_ms_eq": pl.col("t").cast(pl.Datetime("ms")) == _JAN_2_2024_12_30,
    "cast_duration_us_eq": pl.col("d").cast(pl.Duration("us")) == timedelta(minutes=1),
    "cast_time_to_duration": pl.col("tt").cast(pl.Duration("ns")) <= timedelta(hours=13),
    "ns_upper_bound": pl.col("t") <= _JAN_2_2024_12_30,
    "ns_literal_eq": pl.col("t") == pl.lit(_JAN_2_2024_12_30_NS + 500).cast(pl.Datetime("ns")),
    "ns_is_in": pl.col("t").is_in(pl.Series([_JAN_2_2024_12_30_NS + 500]).cast(pl.Datetime("ns")).implode()),
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

# Columns of many dtypes that are stored as plain values and read back through the mapping.
_MAPPED = pl.DataFrame(
    {
        "t": pl.Series([_JAN_2_2024_12_30_NS, _JAN_2_2024_12_30_NS + 500, _JAN_2_2024_12_30_NS + 1500, None]).cast(NANOSECOND_DATETIME),
        "tm": pl.Series(
            [_JAN_2_2024_12_30_NS // 10**6, _JAN_2_2024_12_30_NS // 10**6 - 1, _JAN_2_2024_12_30_NS // 10**6 + 1, _JAN_2_2024_12_30_NS // 10**6]
        ).cast(pl.Datetime("ms")),
        "d": pl.Series([60 * 10**9, 60 * 10**9 + 500, 30 * 10**9, None]).cast(pl.Duration("ns")),
        "tt": pl.Series([45000 * 10**9, 45000 * 10**9 + 500, 46800 * 10**9, None]).cast(pl.Time),
        "x": [1, 2, None, 3],
        "s": ["a", "ab", "b", "c"],
        "ss": ["1", "x", None, "2"],
        "a": ["zz"] * 4,
        "b": ["zzz"] * 4,
    }
)
_JAN_1_2025 = datetime(2025, 1, 1)
_JAN_1_2025_NS = epoch_ns(_JAN_1_2025)
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
    "ts_partitioned": (pl.DataFrame({"ts": _consecutive_seconds(2, MICROSECOND_DATETIME), "x": [1, 2]}), ["ts"]),
    # timestamps one nanosecond apart
    "ns": (
        pl.DataFrame({"ts": pl.Series([_JAN_1_2025_NS, _JAN_1_2025_NS + 1, _JAN_1_2025_NS + 2]).cast(NANOSECOND_DATETIME), "id": [10, 20, 30]}),
        None,
    ),
    "readme": (
        pl.DataFrame(
            {
                "ts": _consecutive_seconds(4),
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


def _nanosecond_literal(ns: int) -> pl.Expr:
    """A nanosecond datetime literal built without passing through a Python datetime."""
    return pl.lit(0, dtype=NANOSECOND_DATETIME) + pl.duration(nanoseconds=ns)


@dataclass(frozen=True)
class DeltaScanCase:
    """A filter on a Delta table read with ``scan_delta``; the result must equal plain Polars filtering the written
    frame."""

    id: str
    _: KW_ONLY
    # the name of a table in ``TABLES``
    table: str
    predicate: pl.Expr
    # applied after the filter
    query: Callable[[pl.LazyFrame], pl.LazyFrame] | None = None
    # the filter must reach the scan (``SELECTION`` and no ``FILTER`` in the plan)
    check_plan: bool = False
    # how the partitions were used: "not_attempted" (no partition pruning), "attempted", or "without_partition_filters"
    # (files were listed with no partition filter)
    partition_listing: str | None = None
    # a filter for rows the scan may return as well, depending on the Polars version
    may_also_return: pl.Expr | None = None
    # keyword arguments for ``scan_delta``
    scan_options: dict = field(default_factory=dict)


PREDICATES = [
    DeltaScanCase("delta_predicate_rewrite_datetime", table="sample", predicate=pl.col("ts") >= datetime(2025, 1, 1, 2), check_plan=True),
    DeltaScanCase("delta_predicate_rewrite_duration", table="sample", predicate=pl.col("dur_ms") < timedelta(milliseconds=2000), check_plan=True),
    DeltaScanCase(
        "scan_delta_no_pushdown_calls_file_uris",
        table="partitioned",
        predicate=pl.col("p") == "A",
        partition_listing="not_attempted",
        scan_options={"pushdown_predicate_deltalake": False},
    ),
    DeltaScanCase(
        "scan_delta_pushdown_calls_file_uris",
        table="partitioned",
        predicate=pl.col("p") == "A",
        partition_listing="attempted",
        scan_options={"pushdown_predicate_deltalake": True},
    ),
    DeltaScanCase("delta_partition_pushdown_eq", table="partitioned", predicate=pl.col("p") == "A"),
    DeltaScanCase("delta_partition_pushdown_eq-cast_literal", table="partitioned", predicate=pl.col("p") == pl.lit("A").cast(pl.Utf8)),
    DeltaScanCase("delta_partition_pushdown_in", table="partitioned", predicate=pl.col("p").is_in(["B"])),
    # a range on a partition column cannot prune
    # the null partition may come back too
    DeltaScanCase(
        "delta_partition_pushdown_all_unsupported_ops_fallback",
        table="partitioned",
        predicate=pl.col("p") < "B",
        partition_listing="without_partition_filters",
        may_also_return=pl.col("p").is_null(),
        scan_options={"pushdown_predicate_deltalake": True},
    ),
    DeltaScanCase(
        "delta_partition_pushdown_mixed_or_clause_unsafe_prune",
        table="partitioned",
        predicate=(pl.col("p") < "B") | (pl.col("p") == "B") | pl.col("p").is_null(),
        partition_listing="without_partition_filters",
        scan_options={"pushdown_predicate_deltalake": True},
    ),
    DeltaScanCase("delta_partition_pushdown_not_in", table="partitioned", predicate=~pl.col("p").is_in(["A"]) | pl.col("p").is_null()),
    # "" is Delta's null partition token, so it must not reach file_uris inside a "not in"
    *[
        DeltaScanCase(f"delta_partition_pushdown_not_in_empty_string-{n}", table="partitioned", predicate=q)
        for n, q in [
            ("empty", ~pl.col("p").is_in([""])),
            ("empty-and-value", ~pl.col("p").is_in(["", "B"])),
            ("or-empty", (pl.col("p") == "A") | ~pl.col("p").is_in([""])),
        ]
    ],
    DeltaScanCase("delta_partition_pushdown_is_null", table="partitioned", predicate=pl.col("p").is_null()),
    DeltaScanCase("scan_delta_ne_missing_keeps_null_partition", table="int_partitioned", predicate=pl.col("a").ne_missing(1)),
    # 257 wraps to 1, so its partition must stay
    DeltaScanCase(
        "scan_delta_narrowing_cast_does_not_prune_wrapped_partition",
        table="int_partitioned",
        predicate=pl.col("a").cast(pl.Int8, wrap_numerical=True) == 1,
    ),
    DeltaScanCase(
        "delta_partition_pushdown_datetime_mapped_eq",
        table="ts_partitioned",
        predicate=pl.col("ts") == pl.lit(_JAN_1_2025 + timedelta(seconds=1)).cast(NANOSECOND_DATETIME),
    ),
    *[
        DeltaScanCase(
            f"delta_ns_precision_unit-{n}",
            table="ns",
            predicate=q,
            check_plan=plan,
            scan_options={"storage_options": _storage_options_for("unused", aws_profile=None).polars, "aws_profile": None},
        )
        for n, q, plan in [
            ("ge", pl.col("ts") >= _nanosecond_literal(_JAN_1_2025_NS + 1), True),
            ("le", pl.col("ts") <= _nanosecond_literal(_JAN_1_2025_NS + 1), True),
            ("gt", pl.col("ts") > _nanosecond_literal(_JAN_1_2025_NS + 1), False),
            ("lt", pl.col("ts") < _nanosecond_literal(_JAN_1_2025_NS + 2), False),
            *[
                (
                    f"between_{c}",
                    pl.col("ts").is_between(_nanosecond_literal(_JAN_1_2025_NS + 1), _nanosecond_literal(_JAN_1_2025_NS + 2), closed=c),
                    True,
                )
                for c in ("both", "left", "right", "none")
            ],
        ]
    ],
    DeltaScanCase(
        "readme_delta_example",
        table="readme",
        predicate=(pl.col("ts") >= datetime(2025, 1, 1, 0, 0, 2))
        & (pl.col("dur_ns") >= pl.duration(nanoseconds=1500))
        & (pl.col("t") >= time(0, 0, 1)),
    ),
    *[
        DeltaScanCase(
            f"scan_delta_mapping_filter_then_select-{u}",
            table=f"mapping_{u}",
            predicate=pl.col("ts") == datetime(2024, 1, 2),
            query=lambda lf: lf.select("val"),
        )
        for u in ("us", "ns")
    ],
    # file pruning off: only the translated predicate pushed to the Parquet scan is exercised
    *[
        DeltaScanCase(
            f"scan_delta_mapped_predicate_matches_eager-{n}", table="mapped", predicate=q, scan_options={"pushdown_predicate_deltalake": False}
        )
        for n, q in _MAPPED_PREDICATES.items()
    ],
]


@pytest.mark.parametrize("case", as_params(PREDICATES))
def test_scan_delta_matches_polars(delta_tables, monkeypatch, case: DeltaScanCase):
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
    op = case.query or (lambda lf: lf)
    lf = cpl.scan_delta(delta_tables[case.table], credential_provider=None, **case.scan_options)
    native = TABLES[case.table][0].lazy()
    if case.may_also_return is None:
        assert assert_matches_plain_polars(op(lf.filter(case.predicate)), op(native.filter(case.predicate)), None) is not None
    else:

        def rows(lf):
            df = lf.collect()
            return set(df.select(sorted(df.columns)).rows())

        assert (
            rows(op(native.filter(case.predicate)))
            <= rows(op(lf.filter(case.predicate)))
            <= rows(op(native.filter(case.predicate | case.may_also_return)))
        )
    if case.check_plan:
        assert_translated_text(lf.filter(case.predicate).explain(), "SELECTION", omits="FILTER")
    if case.partition_listing == "not_attempted":
        assert calls["partition_uris"] == 0
    elif case.partition_listing == "attempted":
        assert calls["partition_uris"] >= 1
    elif case.partition_listing == "without_partition_filters":
        assert None in calls["file_uris_filters"]


def test_delta_partition_pushdown_datetime_mapped_eq_prunes_uris(delta_tables):
    """An equality on a datetime partition column lists only the matching partition's files."""
    dt = DeltaTable(delta_tables["ts_partitioned"])
    pruned = _get_partition_uris(dt, pl.col("ts") == pl.lit(_JAN_1_2025 + timedelta(seconds=1)).cast(NANOSECOND_DATETIME), mapping=None)
    assert 1 <= len(pruned) < len(dt.file_uris())


def test_delta_partition_pushdown_ne_keeps_empty_string(tmp_path):
    """delta-rs reads the "" partition as null, so "!=" must not prune it; the source returns it."""
    path = str(tmp_path / "t")
    write_deltalake(path, pl.DataFrame({"c": ["a", "", "b"], "i": [0, 1, 2]}).to_arrow(), mode="overwrite", partition_by=["c"])
    assert sorted(cpl.scan_delta(path, credential_provider=None).filter(pl.col("c") != "a").collect()["i"]) == [1, 2]


_NAN = float("nan")


@pytest.mark.parametrize(
    "pred",
    [
        pl.col("c0") == -0.0,
        pl.col("c0") == 0.0,
        pl.col("c0").cast(pl.Float64) == -0.0,
        pl.col("c0").is_in([-0.0]),
        pl.col("c0").is_in([0.0, 1.5]),
        pl.col("c0") != 0.0,
        ~pl.col("c0").is_in([-0.0]),
        pl.col("c0") == _NAN,
        pl.col("c0").is_in([_NAN]),
        pl.col("c0") != _NAN,
        ~pl.col("c0").is_in([_NAN, 0.0]),
    ],
    ids=["eq_neg", "eq_pos", "cast_eq_neg", "in_neg", "in_pos", "ne_pos", "not_in_neg", "eq_nan", "in_nan", "ne_nan", "not_in_nan"],
)
def test_delta_partition_pushdown_on_signed_zero_and_nan_float_partitions(tmp_path, pred):
    """Polars matches 0.0 with -0.0 and NaN with NaN, which delta-rs keeps in separate partitions.

    Pruning keeps every row the predicate keeps, and only rows a lazy filter of every file keeps (before Polars 2 that
    also keeps the null partition for some predicates).
    """
    path = str(tmp_path / "t")
    df = pl.DataFrame({"c0": [0.0, -0.0, _NAN, 1.5, None], "v": range(5)})
    df.write_delta(path, delta_write_options={"partition_by": ["c0"]})
    unpruned = cpl.scan_delta(path, credential_provider=None, pushdown_predicate_deltalake=False)
    out = set(cpl.scan_delta(path, credential_provider=None).filter(pred).collect()["v"])
    assert set(df.filter(pred)["v"]) <= out <= set(unpruned.filter(pred).collect()["v"])


def test_delta_partition_pushdown_ne_keeps_the_rows_of_a_native_scan_of_a_null_integer_partition(tmp_path):
    path = str(tmp_path / "t")
    pl.DataFrame({"q": [1, 2, None], "i": [0, 1, 2]}).write_delta(path, delta_write_options={"partition_by": ["q"]})
    pred = pl.col("q") != 1
    out = cpl.scan_delta(path, credential_provider=None).filter(pred).collect()
    assert sorted(out["i"]) == sorted(pl.scan_delta(path).filter(pred).collect()["i"])


def test_scan_delta_collects_the_columns_of_its_schema_in_order(tmp_path):
    path = str(tmp_path / "t")
    write_deltalake(path, pl.DataFrame({"p": ["a", "b"], "v": [1, 2]}).to_arrow(), mode="overwrite", partition_by=["p"])
    lf = cpl.scan_delta(path, credential_provider=None)
    assert lf.collect().columns == lf.collect_schema().names() == ["p", "v"]


def _run_delta_io_roundtrip(path: str, aws_profile: str | None) -> None:
    """Overwrite, append through the streaming path, and filter like a native scan of the stored integers.

    Also run against S3 by the integration tests.
    """
    cred = None if aws_profile is None else pl.CredentialProviderAWS(profile_name=aws_profile)
    df = pl.DataFrame(
        {
            "ts": _consecutive_seconds(3, start=datetime(2024, 6, 1)),
            "dur_ms": pl.Series([0, 500, 1500], dtype=MILLISECOND_DURATION),
            "value": [11, 22, 33],
        }
    )
    df.lazy().piot.sink_delta(path, mode="overwrite", credential_provider=cred)
    more = pl.DataFrame(
        {"ts": _consecutive_seconds(1, start=datetime(2024, 6, 1, 0, 0, 3)), "dur_ms": pl.Series([2000], dtype=MILLISECOND_DURATION), "value": [44]}
    )
    more.lazy().piot.sink_delta(path, mode="append", credential_provider=cred, chunk_size=2)
    assert_frame_equal(cpl.scan_delta(path, credential_provider=cred).sort("ts").collect(), pl.concat([df, more]))

    threshold = datetime(2024, 6, 1, 0, 0, 2)
    storage_options = _storage_options_for(path, aws_profile=aws_profile).polars
    native = pl.scan_delta(path, storage_options=storage_options, credential_provider=cred).filter(
        pl.col("ts") >= pl.Series([threshold], dtype=NANOSECOND_DATETIME).dt.timestamp("ns").item()
    )
    logical = cpl.scan_delta(path, credential_provider=cred).filter(pl.col("ts") >= threshold)
    assert logical.sort("ts").collect()["value"].to_list() == native.sort("ts").collect()["value"].to_list() == [33, 44]


def test_delta_io_roundtrip(tmp_path):
    _run_delta_io_roundtrip(str(tmp_path / "t"), aws_profile=None)


_HOUR_US = 3_600 * 10**6


_DELTA_VALUES = {
    pl.Int64(): st.integers(-5, 300),
    pl.Int8(): st.integers(-5, 100),
    pl.Boolean(): st.booleans(),
    pl.String(): st.sampled_from(["", "a", "ab", "b", "B", " x", "é", "a/b", "None"]),
    pl.Date(): st.integers(19_000, 19_040),
    pl.Datetime("us"): st.integers(0, 60).map(lambda h: 1_704_067_200 * 10**6 + h * _HOUR_US),
    pl.Datetime("us", "UTC"): st.integers(0, 60).map(lambda h: 1_704_067_200 * 10**6 + h * _HOUR_US),
    pl.Datetime("ns"): st.tuples(st.integers(0, 60), st.sampled_from([0, 500])).map(
        lambda p: (1_704_067_200 * 10**6 + p[0] * _HOUR_US) * 1000 + p[1]
    ),
    pl.Float64(): st.sampled_from([0.0, 1.5, -2.0, math.nan]),
    pl.Decimal(10, 2): st.integers(0, 500),  # deltalake cannot write a negative Decimal partition value
}


@st.composite
def _delta_cases(draw):
    dtypes = draw(st.lists(st.sampled_from(list(_DELTA_VALUES)), min_size=1, max_size=3))
    schema = {f"c{i}": d for i, d in enumerate(dtypes)}
    pools = {name: draw(st.lists(_DELTA_VALUES[d], min_size=1, max_size=3)) for name, d in schema.items()}
    columns = {name: series_from_raw(d, draw(st.lists(st.sampled_from([*pools[name], None]), min_size=5, max_size=5))) for name, d in schema.items()}
    frame = pl.DataFrame({**columns, "v": list(range(5))})
    partition_by = draw(st.lists(st.sampled_from(list(schema)), min_size=1, max_size=2, unique=True))
    return frame, partition_by, random_filter(draw, schema, pools, depth=2)


@SETTINGS
@given(case=_delta_cases())
def test_scan_delta_pruning_keeps_the_rows_a_filter_keeps(tmp_path_factory, case):
    """Partition pruning keeps every row the predicate keeps and only rows a read of every file gives."""
    pytest.importorskip("deltalake")
    frame, partition_by, pred = case
    note(f"partition_by={partition_by} pred={show(pred)}")
    root = tmp_path_factory.mktemp("delta")
    table_path = str(root / "table")
    try:
        sink_delta(frame.lazy(), table_path, mode="overwrite", delta_write_options={"partition_by": partition_by})
        unpruned = scan_delta(table_path, credential_provider=None, pushdown_predicate_deltalake=False)
        try:
            kept, unpruned_kept = set(unpruned.collect().filter(pred)["v"]), set(unpruned.filter(pred).collect()["v"])
        except POLARS_ERRORS:
            return
        # Filtering a scan of hive partitions may keep null partition values that filtering the collected frame drops.
        assert kept <= set(scan_delta(table_path, credential_provider=None).filter(pred).collect()["v"]) <= unpruned_kept
    finally:
        shutil.rmtree(root)


class TestScanDeltaPickle:
    """Tests for cpl.scan_delta pickle support."""

    def test_scan_delta_pickle_basic(self, tmp_path):
        """scan_delta LazyFrames can be pickled and unpickled."""
        path = str(tmp_path / "delta_table")

        # Create a simple delta table
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        df.write_delta(path)

        # Create LazyFrame and pickle roundtrip
        lf = cpl.scan_delta(path)
        lf_unpickled = pickle_round_trip(lf)

        # Verify results match
        expected = df
        result = lf_unpickled.collect()
        assert_frame_equal(result, expected)

    def test_scan_delta_pickle_with_mapping(self, tmp_path):
        """scan_delta with type mappings can be pickled."""
        path = str(tmp_path / "delta_table")

        # Create delta table with Datetime[ns] (requires mapping)
        df = pl.DataFrame(
            {
                "ts": pl.Series([datetime(2025, 1, 1)]).cast(pl.Datetime("ns")),
                "value": [42],
            }
        )
        cpl.sink_delta(df.lazy(), path, mode="overwrite")

        lf = cpl.scan_delta(path)
        lf_unpickled = pickle_round_trip(lf)

        result = lf_unpickled.collect()
        assert result.schema["ts"] == pl.Datetime("ns")
        assert result["ts"].to_list() == [datetime(2025, 1, 1)]

    def test_scan_delta_pickle_with_filter(self, tmp_path):
        """scan_delta with filters can be pickled."""
        path = str(tmp_path / "delta_table")

        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["a", "b", "c", "d", "e"]})
        df.write_delta(path)

        lf = cpl.scan_delta(path).filter(pl.col("a") > 2)
        lf_unpickled = pickle_round_trip(lf)

        expected = df.filter(pl.col("a") > 2)
        result = lf_unpickled.collect()
        assert_frame_equal(result, expected)

    def test_scan_delta_pickle_with_select(self, tmp_path):
        """scan_delta with column selection can be pickled."""
        path = str(tmp_path / "delta_table")

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [10, 20, 30]})
        df.write_delta(path)

        lf = cpl.scan_delta(path).select(["a", "b"])
        lf_unpickled = pickle_round_trip(lf)

        expected = df.select(["a", "b"])
        result = lf_unpickled.collect()
        assert_frame_equal(result, expected)
