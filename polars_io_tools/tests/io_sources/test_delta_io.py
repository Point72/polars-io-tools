import base64
import os
from datetime import datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_io_tools as cpl
from polars_io_tools._compat import POLARS_HAS_COLLECT_BATCHES
from polars_io_tools.io_sources.delta_io import (
    _MAPPING_BLOCK_TAG,
    _get_partition_uris,
    build_delta_write_exprs,
    infer_logical_mapping,
    metadata_to_mapping,
)
from polars_io_tools.io_sources.util import _storage_options_for, extract_description_block, inject_description_block

DeltaTable = pytest.importorskip("deltalake").DeltaTable
write_deltalake = pytest.importorskip("deltalake").write_deltalake


def _build_sample_df() -> pl.DataFrame:
    ts = [datetime(2025, 1, 1, 0, 0, 0) + timedelta(hours=i) for i in range(5)]
    ts_s = pl.Series("ts", ts).cast(pl.Datetime("ns"))
    dur = [0, 500, 1500, 2500, 4000]
    dur_s = pl.Series("dur_ms", dur, dtype=pl.Duration("ms"))
    val_s = pl.Series("value", [10, 20, 30, 40, 50], dtype=pl.Int64)
    return pl.DataFrame([ts_s, dur_s, val_s])


def test_infer_logical_mapping_from_schema_basic():
    df = _build_sample_df()
    schema = df.lazy().collect_schema()
    mapping = infer_logical_mapping(schema)
    assert mapping == {"ts": pl.Datetime("ns"), "dur_ms": pl.Duration("ms")}


def test_infer_logical_mapping_ignores_non_temporal():
    df = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    schema = df.lazy().collect_schema()
    mapping = infer_logical_mapping(schema)
    assert mapping == {}


def test_build_delta_write_exprs_shapes_temporal():
    df = _build_sample_df()
    lf = df.lazy()
    schema = lf.collect_schema()
    mapping = infer_logical_mapping(schema)
    exprs = build_delta_write_exprs(schema, mapping)
    out = lf.select(exprs).collect()
    # Temporal types cast to Int64; plain value stays Int64
    assert out.schema["ts"] == pl.Int64
    assert out.schema["dur_ms"] == pl.Int64
    assert out.schema["value"] == pl.Int64


def test_delta_roundtrip_schema_and_values(tmp_path):
    df = _build_sample_df()
    table_path = os.path.join(tmp_path, "delta_table")
    # Namespace API should work
    df.lazy().piot.sink_delta(table_path, mode="overwrite")

    lf = cpl.scan_delta(table_path)
    out = lf.collect()

    # Schema types should match target logical types
    schema = out.schema
    assert schema["ts"] == pl.Datetime("ns")
    assert schema["dur_ms"] == pl.Duration("ms")

    # Values round-trip
    assert_frame_equal(out.select(["ts", "dur_ms", "value"]), df.select(["ts", "dur_ms", "value"]))


def test_delta_microseconds_passthrough(tmp_path):
    # Datetime/us should be written natively without mapping; durations are mapped to ints
    ts_us = pl.datetime_range(start=pl.datetime(2025, 1, 1), end=pl.datetime(2025, 1, 1, 0, 0, 2), interval="1s", eager=True).cast(pl.Datetime("us"))
    dur_us_counts = pl.Series([100_000, 200_000], dtype=pl.Int64)
    df = pl.DataFrame({"ts": ts_us.head(2), "dur": dur_us_counts, "x": [1, 2]})
    table_path = os.path.join(tmp_path, "delta_us_passthrough")
    cpl.sink_delta(df.lazy(), table_path, mode="overwrite")

    # scan_delta should expose native types without casting
    out = cpl.scan_delta(table_path).collect()
    assert out.schema["ts"] == pl.Datetime("us")
    assert out.schema["dur"] == pl.Int64

    # And the mapping should not include these columns
    from deltalake import DeltaTable

    dt = DeltaTable(table_path)
    meta = dt.metadata()
    desc = getattr(meta, "description", None)
    if desc:
        payload = extract_description_block(desc, _MAPPING_BLOCK_TAG)
        if payload:
            mapping = metadata_to_mapping(base64.b64decode(payload))
            # ts (us) should not be mapped
            assert "ts" not in mapping


def test_datetime_unit_fidelity_roundtrip(tmp_path):
    # Build microsecond-precision datetimes, add a nanosecond duration delta
    base = pl.Series("ts", [datetime(2025, 1, 1, 0, 0, 0), datetime(2025, 1, 1, 0, 0, 1)]).cast(pl.Datetime("us"))
    # Add 500 nanoseconds to each timestamp via arithmetic on ns cast
    df = pl.DataFrame({"ts": base})
    df = df.with_columns(pl.col("ts").cast(pl.Datetime("ns")).dt.offset_by("500ns").cast(pl.Datetime("us")).alias("ts"))
    table_path = os.path.join(tmp_path, "delta_unit_fidelity")
    # Write and read back through our pipeline
    cpl.sink_delta(df.lazy(), table_path, mode="overwrite")
    out = cpl.scan_delta(table_path).collect()
    # Expect the microsecond precision retained without losing fidelity
    assert out.schema["ts"] == pl.Datetime("us")
    assert_frame_equal(out.select(["ts"]), df.select(["ts"]))


def test_mixed_datetime_us_and_ns_roundtrip(tmp_path):
    """Test that Datetime[us] columns are preserved when written alongside Datetime[ns] columns."""
    df = pl.DataFrame(
        {
            "ts_us": pl.Series([datetime(2025, 1, 1, 0, 0, 0), datetime(2025, 1, 1, 0, 0, 1)]).cast(pl.Datetime("us")),
            "ts_ns": pl.Series([datetime(2025, 1, 1, 0, 0, 0), datetime(2025, 1, 1, 0, 0, 1)]).cast(pl.Datetime("ns")),
            "value": [10, 20],
        }
    )
    table_path = os.path.join(tmp_path, "delta_mixed_datetime")
    cpl.sink_delta(df.lazy(), table_path, mode="overwrite")

    out = cpl.scan_delta(table_path).collect()

    # Both datetime columns should preserve their original time units
    assert out.schema["ts_us"] == pl.Datetime("us")
    assert out.schema["ts_ns"] == pl.Datetime("ns")
    assert_frame_equal(out, df)

    # Verify mapping only includes ns column, not us column
    dt = DeltaTable(table_path)
    meta = dt.metadata()
    desc = getattr(meta, "description", None)
    assert isinstance(desc, str) and f"[{_MAPPING_BLOCK_TAG}:begin]" in desc
    payload = extract_description_block(desc, _MAPPING_BLOCK_TAG)
    mapping = metadata_to_mapping(base64.b64decode(payload))
    assert "ts_ns" in mapping
    assert "ts_us" not in mapping


def test_delta_streaming_write_batches(tmp_path):
    # Force streaming path even if collect_batches exists, by setting chunk_size
    df = _build_sample_df()
    table_path = os.path.join(tmp_path, "delta_streaming")
    # Write with chunking to exercise collect_batches path
    cpl.sink_delta(df.lazy(), table_path, mode="overwrite", chunk_size=2)

    lf = cpl.scan_delta(table_path)
    out = lf.collect()
    assert_frame_equal(out.select(["ts", "dur_ms", "value"]), df.select(["ts", "dur_ms", "value"]))
    # Description mapping should be present and valid in the Delta table
    dt = DeltaTable(table_path)
    meta = dt.metadata()
    desc = getattr(meta, "description", None)
    assert isinstance(desc, str) and f"[{_MAPPING_BLOCK_TAG}:begin]" in desc and f"[{_MAPPING_BLOCK_TAG}:end]" in desc


@pytest.mark.skipif(not POLARS_HAS_COLLECT_BATCHES, reason="collect_batches requires Polars >= 1.34.0")
@pytest.mark.parametrize(
    "chunk_size,expect_batches",
    [
        (-1, False),  # chunk_size=-1 should use single collect(), not collect_batches
        (2, True),  # chunk_size=2 should use collect_batches for chunked writes
        (None, True),  # chunk_size=None (default) should use collect_batches
    ],
)
def test_delta_chunk_size_controls_collect_batches(tmp_path, chunk_size, expect_batches):
    """Verify chunk_size controls whether collect_batches is used.

    This is important for users who need atomic/consistent writes where
    multiple write operations would break consistency guarantees. The code logic is:

        if POLARS_HAS_COLLECT_BATCHES and chunk_size != -1:
            for batch_df in lf.collect_batches(...):  # chunked path
                ...
        else:
            df = lf.collect()  # single-write path
            df.write_delta(...)

    Note: chunking only applies to append/error/ignore modes, not overwrite/merge.
    By verifying collect_batches is/isn't called based on chunk_size, we ensure
    the correct code path is taken regardless of how polars implements things internally.
    """
    from unittest.mock import patch

    df = _build_sample_df()
    table_path = os.path.join(tmp_path, f"delta_chunk_{chunk_size}")

    # First create the table so we can append to it
    cpl.sink_delta(df.lazy(), table_path, mode="overwrite")

    collect_batches_called = {"called": False}
    original_collect_batches = pl.LazyFrame.collect_batches

    def detecting_collect_batches(self, *args, **kwargs):
        collect_batches_called["called"] = True
        return original_collect_batches(self, *args, **kwargs)

    # Use mode="append" because chunking only applies to append/error/ignore modes
    with patch.object(pl.LazyFrame, "collect_batches", detecting_collect_batches):
        cpl.sink_delta(df.lazy(), table_path, mode="append", chunk_size=chunk_size)

    if expect_batches:
        assert collect_batches_called["called"], (
            f"collect_batches SHOULD be called when chunk_size={chunk_size}; the chunked write code path should use collect_batches"
        )
    else:
        assert not collect_batches_called["called"], (
            f"collect_batches should NOT be called when chunk_size={chunk_size}; the single-write code path should use collect() instead"
        )

    # Verify data was written correctly (should have 2x rows after append)
    out = cpl.scan_delta(table_path).collect()
    assert out.shape[0] == df.shape[0] * 2


def test_delta_predicate_rewrite_datetime(tmp_path):
    df = _build_sample_df()
    table_path = os.path.join(tmp_path, "delta_table_pred_dt")
    cpl.sink_delta(df.lazy(), table_path, mode="overwrite")

    lf = cpl.scan_delta(table_path)
    threshold = datetime(2025, 1, 1, 2, 0, 0)
    # Validate end-to-end through the translated source
    res = lf.filter(pl.col("ts") >= threshold).collect()
    exp = df.filter(pl.col("ts") >= threshold)
    # Results should match
    assert_frame_equal(res, exp)
    # Verify pushdown happened: explain should show SELECTION not FILTER at top
    plan = lf.filter(pl.col("ts") >= threshold).explain()
    assert "SELECTION" in plan
    assert "FILTER" not in plan
    # Ensure the translated schema is still exposed
    assert lf.collect().schema["ts"] == pl.Datetime("ns")


def test_delta_predicate_rewrite_duration(tmp_path):
    df = _build_sample_df()
    table_path = os.path.join(tmp_path, "delta_table_pred_dur")
    cpl.sink_delta(df.lazy(), table_path, mode="overwrite")

    lf = cpl.scan_delta(table_path)
    res = lf.filter(pl.col("dur_ms") < timedelta(milliseconds=2000)).collect()
    exp = df.filter(pl.col("dur_ms") < timedelta(milliseconds=2000))
    assert_frame_equal(res, exp)
    # Verify pushdown happened: explain should show SELECTION not FILTER
    plan = lf.filter(pl.col("dur_ms") < timedelta(milliseconds=2000)).explain()
    assert "SELECTION" in plan
    assert "FILTER" not in plan


def test_delta_roundtrip_infer_mapping(tmp_path):
    df = _build_sample_df()
    table_path = os.path.join(tmp_path, "delta_table_infer")
    # No mapping provided: infer
    cpl.sink_delta(df.lazy(), table_path, mode="overwrite")

    lf = cpl.scan_delta(table_path)
    out = lf.collect()

    # Schema types should match target logical types (inferred)
    schema = out.schema
    assert schema["ts"] == pl.Datetime("ns")
    assert schema["dur_ms"] == pl.Duration("ms")
    assert_frame_equal(out.select(["ts", "dur_ms", "value"]), df.select(["ts", "dur_ms", "value"]))


def test_delta_metadata_us_precision_has_no_block(tmp_path):
    # For microsecond precision datetime, no mapping block is required
    df = pl.DataFrame(
        {
            "ts": pl.datetime_range(start=pl.datetime(2025, 1, 1), end=pl.datetime(2025, 1, 1, 0, 0, 4), interval="1s", eager=True).cast(
                pl.Datetime("us")
            ),
            "x": [1, 2, 3, 4, 5],
        }
    )
    table_path = os.path.join(tmp_path, "delta_meta_us")
    cpl.sink_delta(df.lazy(), table_path, mode="overwrite")

    dt = DeltaTable(table_path)
    meta = dt.metadata()
    desc = getattr(meta, "description", None)
    # No mapping description expected for us-precision datetime
    assert desc is None or f"[{_MAPPING_BLOCK_TAG}:begin]" not in desc


def test_delta_metadata_ms_precision_includes_block(tmp_path):
    # For non-us precision temporal types, mapping block should be present
    df = pl.DataFrame(
        {
            "dur_ms": pl.Series([0, 500, 1500, 2500, 4000], dtype=pl.Duration("ms")),
            "x": [1, 2, 3, 4, 5],
        }
    )
    table_path = os.path.join(tmp_path, "delta_meta_ms")
    cpl.sink_delta(df.lazy(), table_path, mode="overwrite")

    dt = DeltaTable(table_path)
    meta = dt.metadata()
    desc = getattr(meta, "description", None)
    assert isinstance(desc, str)
    assert f"[{_MAPPING_BLOCK_TAG}:begin]" in desc and f"[{_MAPPING_BLOCK_TAG}:end]" in desc
    payload = extract_description_block(desc, _MAPPING_BLOCK_TAG)
    mapping = metadata_to_mapping(base64.b64decode(payload))
    assert mapping.get("dur_ms") == pl.Duration("ms")


@pytest.fixture()
def _partitioned_delta_table(tmp_path):
    # Create a small partitioned Delta table with p in {"A", "B", None}
    table_path = os.path.join(tmp_path, "delta_partition_pushdown")
    df = pl.DataFrame(
        {
            "x": [1, 2, 10, 20, 100],
            "p": ["A", "A", "B", "B", None],
        }
    )
    # Write via Deltalake with partitioning on column 'p'
    write_deltalake(table_path, df.to_arrow(), mode="overwrite", partition_by=["p"])
    return table_path


def test_scan_delta_no_pushdown_calls_file_uris(_partitioned_delta_table, monkeypatch):
    # Ensure that when pushdown is disabled, _get_partition_uris is not called
    calls = {"count": 0}

    # Monkeypatch _get_partition_uris to increment count if called
    import polars_io_tools.io_sources.delta_io as dio

    orig_get = dio._get_partition_uris

    def _spy(*args, **kwargs):
        calls["count"] += 1
        return orig_get(*args, **kwargs)

    monkeypatch.setattr(dio, "_get_partition_uris", _spy)

    # Build a simple filtered scan with pushdown disabled
    lf = cpl.scan_delta(_partitioned_delta_table, credential_provider=None, pushdown_predicate_deltalake=False)
    out = lf.filter(pl.col("p") == "A").collect()
    assert out.shape == (2, 2)
    # Verify _get_partition_uris was never called
    assert calls["count"] == 0


def test_scan_delta_pushdown_calls_file_uris(_partitioned_delta_table, monkeypatch):
    # Ensure that when pushdown is enabled, _get_partition_uris is called
    calls = {"count": 0}

    import polars_io_tools.io_sources.delta_io as dio

    orig_get = dio._get_partition_uris

    def _spy(*args, **kwargs):
        calls["count"] += 1
        return orig_get(*args, **kwargs)

    monkeypatch.setattr(dio, "_get_partition_uris", _spy)

    lf = cpl.scan_delta(_partitioned_delta_table, credential_provider=None, pushdown_predicate_deltalake=True)
    out = lf.filter(pl.col("p") == "A").collect()
    assert out.shape == (2, 2)
    # Verify _get_partition_uris was called at least once
    assert calls["count"] >= 1


def test_delta_partition_pushdown_eq(_partitioned_delta_table):
    lf = cpl.scan_delta(_partitioned_delta_table, credential_provider=None)
    out = lf.filter(pl.col("p") == "A").sort("x").collect()
    assert out.shape == (2, 2)
    assert set(out["x"].to_list()) == {1, 2}
    assert set(out["p"].to_list()) == {"A"}

    # Literal cast to ensure proper encoding for equality
    out2 = cpl.scan_delta(_partitioned_delta_table, credential_provider=None).filter(pl.col("p") == pl.lit("A").cast(pl.Utf8)).sort("x").collect()
    assert out2.shape == (2, 2)


def test_delta_partition_pushdown_in(_partitioned_delta_table):
    lf = cpl.scan_delta(_partitioned_delta_table, credential_provider=None)
    out = lf.filter(pl.col("p").is_in(["B"]))
    df = out.sort("x").collect()
    assert df.shape == (2, 2)
    assert set(df["x"].to_list()) == {10, 20}
    assert set(df["p"].to_list()) == {"B"}


def test_delta_partition_pushdown_all_unsupported_ops_fallback(_partitioned_delta_table, monkeypatch):
    """If all DNF clauses have only unsupported ops, ensure we fall back to dt.file_uris() (pf None)."""
    calls = {"filters": []}

    import deltalake

    orig = deltalake.DeltaTable.file_uris

    def _spy(self, *args, **kwargs):
        calls["filters"].append(kwargs.get("partition_filters", None))
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(deltalake.DeltaTable, "file_uris", _spy, raising=True)

    lf = cpl.scan_delta(_partitioned_delta_table, credential_provider=None, pushdown_predicate_deltalake=True)
    # Use a range operator on partition column 'p' (unsupported for pruning)
    out = lf.filter(pl.col("p") < "B").sort("x").collect()
    # Expect only 'A' rows; no 'B' rows
    assert out.filter(pl.col("p") == "A").shape == (2, 2)
    assert out.filter(pl.col("p") == "B").is_empty()

    # Verify that file_uris was called without partition_filters at least once (pf None)
    assert any(f is None for f in calls["filters"]) and len(calls["filters"]) >= 1


def test_delta_partition_pushdown_not_in(_partitioned_delta_table):
    lf = cpl.scan_delta(_partitioned_delta_table, credential_provider=None)
    out = lf.filter(~pl.col("p").is_in(["A"]))
    df = out.sort(["p", "x"]).collect()
    # Expect B rows and the null partition row
    assert df.shape == (3, 2)
    assert set(df["p"].to_list()) == {"B", None}
    assert set(df["x"].to_list()) == {10, 20, 100}


def test_delta_partition_pushdown_is_null(_partitioned_delta_table):
    lf = cpl.scan_delta(_partitioned_delta_table, credential_provider=None)
    # Match the null partition using is_null, and equality against None
    out = lf.filter(pl.col("p").is_null())
    df = out.collect()
    assert df.shape == (1, 2)
    assert df["p"].to_list() == [None]
    assert df["x"].to_list() == [100]


def test_scan_delta_reads_description_mapping(tmp_path):
    # Build a table and rely on Delta metadata description
    ts = pl.datetime_range(start=pl.datetime(2025, 1, 1), end=pl.datetime(2025, 1, 1, 0, 0, 4), interval="1s", eager=True).cast(pl.Datetime("ns"))
    df = pl.DataFrame({"ts": ts})
    table_path = os.path.join(tmp_path, "delta_desc")
    cpl.sink_delta(df.lazy(), table_path, mode="overwrite")
    out = cpl.scan_delta(table_path).collect()
    assert out.schema["ts"] == pl.Datetime("ns")


def test_scan_delta_without_mapping_returns_plain(tmp_path):
    # Create a delta table via polars, but without our configuration block
    df = pl.DataFrame({"x": [1, 2, 3]})
    table_path = os.path.join(tmp_path, "delta_no_mapping")
    df.write_delta(table_path, mode="overwrite")

    # scan_delta should fall back to plain scan (no translated types)
    lf = cpl.scan_delta(table_path)
    out = lf.collect()
    assert out.schema["x"] == pl.Int64


def test_scan_delta_reads_block_mapping_when_present(tmp_path):
    df = _build_sample_df()
    table_path = os.path.join(tmp_path, "delta_desc_fallback")
    cpl.sink_delta(df.lazy(), table_path, mode="overwrite")
    out = cpl.scan_delta(table_path).collect()
    assert out.schema["ts"] == pl.Datetime("ns")


def test_delta_partition_pushdown_datetime_mapped_eq(tmp_path):
    # Create a Delta table partitioned by a Datetime column and verify
    # that an equality filter is pushed down to deltalake (via URIs pruning).
    ts_vals = [
        datetime(2025, 1, 1, 0, 0, 0),
        datetime(2025, 1, 1, 0, 0, 1),
    ]
    df = pl.DataFrame(
        {
            "ts": pl.Series(ts_vals).cast(pl.Datetime("ns")),
            "x": [1, 2],
        }
    )
    table_path = os.path.join(tmp_path, "delta_partition_ts")
    write_deltalake(table_path, df.to_arrow(), mode="overwrite", partition_by=["ts"])

    # Build a predicate that includes an explicit cast to the mapped dtype
    target = datetime(2025, 1, 1, 0, 0, 1)
    pred = pl.col("ts") == pl.lit(target).cast(pl.Datetime("ns"))

    # Verify pruning via file_uris returns only the matching partition URIs
    dt = DeltaTable(table_path)
    all_uris = dt.file_uris()
    pruned_uris = _get_partition_uris(dt, pred, mapping=None)
    assert len(pruned_uris) < len(all_uris)
    assert len(pruned_uris) >= 1

    # Ensure the scan returns the correct single row
    out = cpl.scan_delta(table_path, credential_provider=None).filter(pred).collect()
    assert out.shape == (1, 2)
    assert out["ts"].to_list() == [target]
    assert out["x"].to_list() == [2]
    # No other mapped columns present in this test


def test_scan_delta_propagates_storage_options_passthrough(tmp_path):
    # Create a small table with mapping embedded
    ts = pl.datetime_range(start=pl.datetime(2025, 1, 1), end=pl.datetime(2025, 1, 1, 0, 0, 2), interval="1s", eager=True).cast(pl.Datetime("ns"))
    df = pl.DataFrame({"ts": ts})
    table_path = os.path.join(tmp_path, "delta_opts")
    cpl.sink_delta(df.lazy(), table_path, mode="overwrite")

    # Pass user storage_options and ensure scan_delta still exposes mapped schema
    user_opts = {"endpoint_url": "http://localhost:9000", "unused_key": "value"}
    out = cpl.scan_delta(table_path, storage_options=user_opts).collect()
    assert out.schema["ts"] == pl.Datetime("ns")


def test_delta_mapping_uses_ns_not_us(tmp_path):
    # Create a simple frame with datetime[ns]
    df = pl.DataFrame(
        {
            "ts": pl.datetime_range(start=pl.datetime(2025, 1, 1), end=pl.datetime(2025, 1, 1, 0, 0, 2), interval="1s", eager=True).cast(
                pl.Datetime("ns")
            ),
        }
    )
    table_path = os.path.join(tmp_path, "delta_mapping_units")
    cpl.sink_delta(df.lazy(), table_path, mode="overwrite")

    # Read via scan_delta and verify schema exposes ns
    out = cpl.scan_delta(table_path).collect()
    assert out.schema["ts"] == pl.Datetime("ns")

    # Also assert that the embedded mapping metadata decodes to ns
    dt = DeltaTable(table_path)
    meta = dt.metadata()
    desc = getattr(meta, "description", None)
    assert isinstance(desc, str) and f"[{_MAPPING_BLOCK_TAG}:begin]" in desc
    payload = extract_description_block(desc, _MAPPING_BLOCK_TAG)

    mapping = metadata_to_mapping(base64.b64decode(payload))
    assert mapping["ts"] == pl.Datetime("ns")


def test_scan_delta_invalid_block_raises(tmp_path):
    # Create a delta table with an invalid mapping description block
    df = pl.DataFrame({"x": [1, 2, 3]})
    table_path = os.path.join(tmp_path, "delta_bad_desc")
    df.write_delta(
        table_path, mode="overwrite", delta_write_options={"description": f"\n[{_MAPPING_BLOCK_TAG}:begin]\nnot-base64\n[{_MAPPING_BLOCK_TAG}:end]"}
    )

    # No sidecar present, fallback reads description and should raise on invalid base64
    with pytest.raises(ValueError):
        cpl.scan_delta(table_path).collect()


def test_inject_and_extract_roundtrip_block_helpers():
    desc = None
    payload = base64.b64encode(b"hello").decode("ascii")
    desc = inject_description_block(desc, "foo.block", payload)
    assert "[foo.block:begin]" in desc and "[foo.block:end]" in desc
    out = extract_description_block(desc, "foo.block")
    assert out == payload


def test_extract_missing_returns_none_block_helpers():
    assert extract_description_block("some text without blocks", "foo") is None
    assert extract_description_block(None, "foo") is None


def test_extract_ignores_whitespace_and_takes_first_line_block_helpers():
    desc = "\n[foo:begin]\n  abc\n  def\n[foo:end]"
    out = extract_description_block(desc, "foo")
    assert out == "abc"


def _run_delta_ns_precision_common(cache_root: str, aws_profile: str | None) -> None:
    # Build underlying data as Datetime[ns] with nanosecond offsets around a base timestamp
    import datetime as _dt

    base_dt = _dt.datetime(2025, 1, 1, tzinfo=_dt.timezone.utc)
    base_ns = int(base_dt.timestamp() * 1_000_000_000)

    df_inner = (
        pl.DataFrame({"id": [10, 20, 30], "offset_ns": [0, 1, 2]})
        .with_columns(ts=(pl.lit(base_dt).cast(pl.Datetime("ns")) + pl.col("offset_ns").cast(pl.Duration("ns"))))
        .select(["ts", "id"])
    )

    # Write via sink_delta
    cp = pl.CredentialProviderAWS(profile_name=aws_profile) if aws_profile else None
    cpl.sink_delta(df_inner.lazy(), cache_root, mode="overwrite", credential_provider=cp)

    # Read via scan_delta
    polars_opts = _storage_options_for(cache_root, aws_profile=aws_profile).polars
    lf = cpl.scan_delta(cache_root, storage_options=polars_opts, aws_profile=aws_profile)
    out = lf.collect()
    assert out.schema["ts"] == pl.Datetime("ns")

    # Build ns-precision datetime literals without losing precision using durations
    def lit_ns(ns: int) -> pl.Expr:
        return pl.lit(0, dtype=pl.Datetime("ns")) + pl.duration(nanoseconds=ns)

    # Lower bound: >= base+1ns should match ids [20, 30]
    lf_ge = lf.filter(pl.col("ts") >= lit_ns(base_ns + 1))
    plan_ge = lf_ge.explain()
    assert "SELECTION" in plan_ge and "FILTER" not in plan_ge
    out_ge = lf_ge.collect()
    assert out_ge["id"].to_list() == [20, 30]

    # Upper bound: <= base+1ns should match ids [10, 20]
    lf_le = lf.filter(pl.col("ts") <= lit_ns(base_ns + 1))
    plan_le = lf_le.explain()
    assert "SELECTION" in plan_le and "FILTER" not in plan_le
    out_le = lf_le.collect()
    assert out_le["id"].to_list() == [10, 20]

    # Strict bounds: > base+1ns should match [30]; < base+2ns should match [10, 20]
    out_gt = lf.filter(pl.col("ts") > lit_ns(base_ns + 1)).collect()
    assert out_gt["id"].to_list() == [30]
    out_lt = lf.filter(pl.col("ts") < lit_ns(base_ns + 2)).collect()
    assert out_lt["id"].to_list() == [10, 20]

    # Between tests on [base+1ns, base+2ns]
    lf_both = lf.filter(pl.col("ts").is_between(lit_ns(base_ns + 1), lit_ns(base_ns + 2), closed="both"))
    assert "SELECTION" in lf_both.explain()
    assert lf_both.collect()["id"].to_list() == [20, 30]

    lf_left = lf.filter(pl.col("ts").is_between(lit_ns(base_ns + 1), lit_ns(base_ns + 2), closed="left"))
    assert "SELECTION" in lf_left.explain()
    assert lf_left.collect()["id"].to_list() == [20]

    lf_right = lf.filter(pl.col("ts").is_between(lit_ns(base_ns + 1), lit_ns(base_ns + 2), closed="right"))
    assert "SELECTION" in lf_right.explain()
    assert lf_right.collect()["id"].to_list() == [30]

    lf_none = lf.filter(pl.col("ts").is_between(lit_ns(base_ns + 1), lit_ns(base_ns + 2), closed="none"))
    assert "SELECTION" in lf_none.explain()
    assert lf_none.collect()["id"].to_list() == []


def _run_delta_io_roundtrip(cache_root, aws_profile):
    table_root = cache_root

    # Sample data with logical types
    lf = pl.DataFrame(
        {
            "ts": pl.datetime_range(start=pl.datetime(2024, 6, 1), end=pl.datetime(2024, 6, 1, 0, 0, 2), interval="1s", eager=True).cast(
                pl.Datetime("ns")
            ),
            "dur_ms": pl.Series([0, 500, 1500], dtype=pl.Duration("ms")),
            "value": [11, 22, 33],
        }
    ).lazy()

    cred_prov = None if aws_profile is None else pl.CredentialProviderAWS(profile_name=aws_profile)

    # Overwrite table
    lf.piot.sink_delta(
        table_root,
        mode="overwrite",
        credential_provider=cred_prov,
    )

    # Append a row via streaming write path
    lf_append = pl.LazyFrame(
        {
            "ts": [datetime(2024, 6, 1, 0, 0, 3)],
            "dur_ms": pl.Series([2000], dtype=pl.Duration("ms")),
            "value": [44],
        },
        schema_overrides={"ts": pl.Datetime(time_unit="ns")},
    )

    lf_append.piot.sink_delta(
        table_root,
        mode="append",
        credential_provider=cred_prov,
        chunk_size=2,
    )

    # Scan with translation applied
    out = (
        cpl.scan_delta(
            table_root,
            credential_provider=cred_prov,
        )
        .sort("ts")
        .collect()
    )

    assert out.shape[0] == 4
    assert out["ts"].dtype == pl.Datetime("ns")
    assert out["dur_ms"].dtype == pl.Duration("ms")
    assert out["value"].to_list() == [11, 22, 33, 44]

    # Verify predicate pushdown via equivalence to underlying int-domain filters
    # For native pl.scan_delta, we need storage_options with allow_http for on-prem S3
    polars_opts = _storage_options_for(table_root, aws_profile=aws_profile).polars
    native = pl.scan_delta(
        table_root,
        storage_options=polars_opts,
        credential_provider=cred_prov,
    )
    # 1) Datetime >= threshold
    threshold = datetime(2024, 6, 1, 0, 0, 2)
    threshold_ns = int(pl.Series([threshold], dtype=pl.Datetime("ns")).dt.timestamp("ns").to_list()[0])
    filtered_logical = cpl.scan_delta(table_root, credential_provider=cred_prov).filter(pl.col("ts") >= threshold).sort("ts").collect()
    filtered_native = native.filter(pl.col("ts") >= threshold_ns).sort("ts").collect()
    assert filtered_logical["value"].to_list() == filtered_native["value"].to_list() == [33, 44]


def test_delta_ns_precision_unit(tmp_path):
    _run_delta_ns_precision_common(str(tmp_path / "delta_ns"), aws_profile=None)


def test_delta_io_roundtrip(tmp_path):
    _run_delta_io_roundtrip(str(tmp_path / "delta_ns"), aws_profile=None)


def test_readme_delta_example():
    # EXACT example from README: write ns datetime, ns duration, time; then scan and filter
    import tempfile
    from datetime import datetime, time as dtime

    lf = pl.DataFrame(
        {
            "ts": pl.datetime_range(datetime(2025, 1, 1), datetime(2025, 1, 1, 0, 0, 3), interval="1s", eager=True).cast(pl.Datetime("ns")),
            "dur_ns": pl.Series([0, 500, 1500, 2500], dtype=pl.Duration("ns")),
            "t": pl.Series([dtime(0, 0, 0), dtime(0, 0, 1), dtime(0, 0, 2), dtime(0, 0, 3)], dtype=pl.Time),
        }
    ).lazy()

    root = tempfile.mkdtemp(prefix="cpl_delta_")
    table_path = os.path.join(root, "delta_table")
    cpl.sink_delta(lf, table_path, mode="overwrite")

    out = (
        cpl.scan_delta(table_path)
        .filter(pl.col("ts") >= datetime(2025, 1, 1, 0, 0, 2))
        .filter(pl.col("dur_ns") >= pl.duration(nanoseconds=1500))
        .filter(pl.col("t") >= dtime(0, 0, 1))
        .collect()
    )

    # Expect the last two rows only
    assert out.shape == (2, 3)
    assert out["ts"].to_list() == [datetime(2025, 1, 1, 0, 0, 2), datetime(2025, 1, 1, 0, 0, 3)]
    assert out["t"].to_list() == [dtime(0, 0, 2), dtime(0, 0, 3)]
    # Duration(ns) should compare as integer nanosecond counts when cast
    assert out["dur_ns"].dtype == pl.Duration("ns")
    assert out.select(pl.col("dur_ns").cast(pl.Int64))["dur_ns"].to_list() == [1500, 2500]


@pytest.mark.parametrize("time_unit", ["us", "ns"])
def test_scan_delta_mapping_filter_then_select(tmp_path, time_unit):
    df = pl.DataFrame(
        {"ts": [datetime(2024, 1, 1), datetime(2024, 1, 2)], "val": [10, 20]}, schema_overrides={"ts": pl.Datetime(time_unit=time_unit)}
    ).lazy()
    table_path = os.path.join(tmp_path, "mapping_filter_select")
    # Write with translation (ts mapped to integer backing type)
    cpl.sink_delta(df, table_path, mode="overwrite")

    # Filter on mapped column, then select a different column and collect
    lf = cpl.scan_delta(table_path)
    out = lf.filter(pl.col("ts") == datetime(2024, 1, 2)).select("val").collect()
    assert out.shape == (1, 1)
    assert out["val"].to_list() == [20]


def test_delta_partition_pushdown_mixed_or_clause_unsafe_prune(_partitioned_delta_table, monkeypatch):
    """OR of an unsupported clause and a supported clause should not prune at all (pf None)."""
    calls = {"filters": []}

    import deltalake

    orig = deltalake.DeltaTable.file_uris

    def _spy(self, *args, **kwargs):
        calls["filters"].append(kwargs.get("partition_filters", None))
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(deltalake.DeltaTable, "file_uris", _spy, raising=True)

    lf = cpl.scan_delta(_partitioned_delta_table, credential_provider=None, pushdown_predicate_deltalake=True)
    # Construct OR of unsupported (<) and supported (=) predicate
    pred = (pl.col("p") < "B") | (pl.col("p") == "B")
    out = lf.filter(pred).sort("x").collect()

    # Expect rows for p == 'A' (from first) and p == 'B' (from second); null may pass through via filter semantics
    assert set(out["p"].to_list()) == {"A", "B", None}
    # Verify that file_uris was called without partition_filters at least once (pf None)
    assert any(f is None for f in calls["filters"]) and len(calls["filters"]) >= 1
