from datetime import datetime, timedelta

import polars as pl
import pytest

from polars_io_tools.io_sources.base import get_parsed_expr
from polars_io_tools.io_sources.translated_source import (
    TranslatedPredicateVisitor,
    mapping_to_metadata,
    metadata_to_mapping,
)


def _ns_epoch(dt: datetime) -> int:
    s = pl.Series([dt]).cast(pl.Datetime("ns"), strict=False)
    return s.dt.timestamp("ns").cast(pl.Int64).to_list()[0]


def test_translated_visitor_eq_datetime_ns():
    # Underlying ints for two timestamps
    t0 = datetime(2025, 1, 1, 0, 0, 0)
    t1 = datetime(2025, 1, 1, 0, 0, 1)
    df = pl.DataFrame(
        {
            "ts": [
                _ns_epoch(t0),
                _ns_epoch(t1),
            ]
        }
    )
    # Mapping declares logical dtype for ts
    mapping = {"ts": pl.Datetime("ns")}
    # Predicate comparing against a datetime literal
    pred = pl.col("ts") == pl.lit(t1)
    node = get_parsed_expr(pred)
    vis = TranslatedPredicateVisitor(mapping)
    node.accept(vis)
    rewritten = vis.process_results()
    out = df.filter(rewritten)
    assert out.shape == (1, 1)
    assert out["ts"].to_list() == [_ns_epoch(t1)]


@pytest.mark.xfail(reason="Polars handles is_in differently than equality here, see this issue: https://github.com/pola-rs/polars/issues/22824")
def test_translated_visitor_in_datetime_ns():
    t0 = datetime(2025, 1, 1, 0, 0, 0)
    t1 = datetime(2025, 1, 1, 0, 0, 1)
    t2 = datetime(2025, 1, 1, 0, 0, 2)
    df = pl.DataFrame({"ts": [_ns_epoch(t0), _ns_epoch(t1), _ns_epoch(t2)]})
    mapping = {"ts": pl.Datetime("ns")}
    pred = pl.col("ts").is_in([t0, t2])
    node = get_parsed_expr(pred)
    vis = TranslatedPredicateVisitor(mapping)
    node.accept(vis)
    rewritten = vis.process_results()
    out = df.filter(rewritten).sort("ts")
    assert out.shape == (2, 1)
    assert out["ts"].to_list() == [_ns_epoch(t0), _ns_epoch(t2)]


def test_translated_visitor_eq_datetime_literal_cast():
    # Column mapped to Datetime(ns); literal has an explicit cast
    t0 = datetime(2025, 1, 1, 0, 0, 0)
    t1 = datetime(2025, 1, 1, 0, 0, 1)
    df = pl.DataFrame({"ts": [_ns_epoch(t0), _ns_epoch(t1)]})
    mapping = {"ts": pl.Datetime("ns")}
    lit_cast = pl.lit(t1).cast(pl.Datetime("ns"), strict=False)
    pred = pl.col("ts") == lit_cast
    node = get_parsed_expr(pred)
    vis = TranslatedPredicateVisitor(mapping)
    node.accept(vis)
    rewritten = vis.process_results()
    out = df.filter(rewritten)
    assert out["ts"].to_list() == [_ns_epoch(t1)]


def test_translated_visitor_between_datetime_ns():
    base = datetime(2025, 1, 1, 0, 0, 0)
    ts = [base, base.replace(second=1), base.replace(second=2), base.replace(second=3)]
    df = pl.DataFrame({"ts": [_ns_epoch(t) for t in ts]})
    mapping = {"ts": pl.Datetime("ns")}
    lower = base.replace(second=1)
    upper = base.replace(second=2)
    pred = pl.col("ts").is_between(pl.lit(lower), pl.lit(upper), closed="both")
    node = get_parsed_expr(pred)
    vis = TranslatedPredicateVisitor(mapping)
    node.accept(vis)
    rewritten = vis.process_results()
    out = df.filter(rewritten).sort("ts")
    assert out["ts"].to_list() == [_ns_epoch(lower), _ns_epoch(upper)]


def test_translated_visitor_between_closed_variants():
    base = datetime(2025, 1, 1, 0, 0, 0)
    ts = [base, base.replace(second=1), base.replace(second=2), base.replace(second=3)]
    df = pl.DataFrame({"ts": [_ns_epoch(t) for t in ts]})
    mapping = {"ts": pl.Datetime("ns")}
    lower = base.replace(second=1)
    upper = base.replace(second=2)
    # left-closed: include lower, exclude upper
    pred_left = pl.col("ts").is_between(lower, upper, closed="left")
    node_left = get_parsed_expr(pred_left)
    vis_left = TranslatedPredicateVisitor(mapping)
    node_left.accept(vis_left)
    out_left = df.filter(vis_left.process_results()).sort("ts")
    assert out_left["ts"].to_list() == [_ns_epoch(lower)]
    # right-closed: exclude lower, include upper
    pred_right = pl.col("ts").is_between(lower, upper, closed="right")
    node_right = get_parsed_expr(pred_right)
    vis_right = TranslatedPredicateVisitor(mapping)
    node_right.accept(vis_right)
    out_right = df.filter(vis_right.process_results()).sort("ts")
    assert out_right["ts"].to_list() == [_ns_epoch(upper)]
    # none: exclude both bounds
    pred_none = pl.col("ts").is_between(lower, upper, closed="none")
    node_none = get_parsed_expr(pred_none)
    vis_none = TranslatedPredicateVisitor(mapping)
    node_none.accept(vis_none)
    out_none = df.filter(vis_none.process_results()).sort("ts")
    assert out_none["ts"].to_list() == []


def test_translated_visitor_duration_ms_gt():
    # Underlying duration counts in milliseconds
    df = pl.DataFrame({"dur": [0, 500, 1500, 2500]})
    mapping = {"dur": pl.Duration("ms")}
    pred = pl.col("dur") > pl.lit(timedelta(milliseconds=1000))
    node = get_parsed_expr(pred)
    vis = TranslatedPredicateVisitor(mapping)
    node.accept(vis)
    rewritten = vis.process_results()
    out = df.filter(rewritten).sort("dur")
    assert out["dur"].to_list() == [1500, 2500]


def test_translated_visitor_time_eq():
    # Underlying time stored as nanoseconds since midnight
    times = [
        datetime(2025, 1, 1, 12, 0, 0).time(),
        datetime(2025, 1, 1, 12, 0, 1).time(),
    ]
    # Build underlying ints via Polars cast
    ns_counts = pl.Series(times, dtype=pl.Time).cast(pl.Int64).to_list()
    df = pl.DataFrame({"t": ns_counts})
    mapping = {"t": pl.Time}
    pred = pl.col("t") == pl.lit(times[1])
    node = get_parsed_expr(pred)
    vis = TranslatedPredicateVisitor(mapping)
    node.accept(vis)
    rewritten = vis.process_results()
    out = df.filter(rewritten)
    assert out.shape == (1, 1)


def test_translated_visitor_time_is_in():
    # Mapped time column; check is_in with Python time objects
    t0 = datetime(2025, 1, 1, 12, 0, 0).time()
    t1 = datetime(2025, 1, 1, 12, 0, 1).time()
    t2 = datetime(2025, 1, 1, 12, 0, 2).time()
    ns_counts = pl.Series([t0, t1, t2], dtype=pl.Time).cast(pl.Int64).to_list()
    df = pl.DataFrame({"t": ns_counts})
    mapping = {"t": pl.Time}
    pred = pl.col("t").is_in([t0, t2])
    node = get_parsed_expr(pred)
    vis = TranslatedPredicateVisitor(mapping)
    node.accept(vis)
    out = df.filter(vis.process_results()).sort("t")
    expected = pl.Series([t0, t2], dtype=pl.Time).cast(pl.Int64).to_list()
    assert out["t"].to_list() == expected


def test_translated_visitor_time_neq_value():
    # Times and underlying ns counts
    t0 = datetime(2025, 1, 1, 12, 0, 0).time()
    t1 = datetime(2025, 1, 1, 12, 0, 1).time()
    t2 = datetime(2025, 1, 1, 12, 0, 2).time()
    ns_counts = pl.Series([t0, t1, t2], dtype=pl.Time).cast(pl.Int64).to_list()
    df = pl.DataFrame({"t": ns_counts})
    mapping = {"t": pl.Time}
    pred = pl.col("t") != t0
    node = get_parsed_expr(pred)
    vis = TranslatedPredicateVisitor(mapping)
    node.accept(vis)
    rewritten = vis.process_results()
    out = df.filter(rewritten).sort("t")
    expected = pl.Series([t1, t2], dtype=pl.Time).cast(pl.Int64).to_list()
    assert out["t"].to_list() == expected


def test_translated_visitor_duration_neq_values():
    # Underlying duration counts in ms; exclude a specific timedelta
    vals = [0, 500, 1500, 2500]
    df = pl.DataFrame({"dur": vals})
    mapping = {"dur": pl.Duration("ms")}
    pred = pl.col("dur") != timedelta(milliseconds=500)
    node = get_parsed_expr(pred)
    vis = TranslatedPredicateVisitor(mapping)
    node.accept(vis)
    rewritten = vis.process_results()
    out = df.filter(rewritten).sort("dur")
    assert out["dur"].to_list() == [0, 1500, 2500]


@pytest.mark.xfail(reason="Polars does not support nested object types so the predicate is_in cannot be created.")
def test_translated_visitor_duration_is_in():
    vals = [0, 500, 1500, 2500]
    df = pl.DataFrame({"dur": vals})
    mapping = {"dur": pl.Duration("ms")}
    lit1 = pl.lit(timedelta(milliseconds=500), dtype=pl.Duration(time_unit="ms"))
    lit2 = pl.lit(timedelta(milliseconds=2500), dtype=pl.Duration(time_unit="ms"))
    pred = pl.col("dur").is_in([lit1, lit2])
    node = get_parsed_expr(pred)
    vis = TranslatedPredicateVisitor(mapping)
    node.accept(vis)
    out = df.filter(vis.process_results()).sort("dur")
    assert out["dur"].to_list() == [500, 2500]


def test_translated_visitor_user_cast_layering():
    # User casts the column; visitor should apply mapping then user cast
    base = datetime(2025, 1, 1, 0, 0, 0)
    t0 = base
    t1 = base.replace(second=1)
    df = pl.DataFrame({"ts": [_ns_epoch(t0), _ns_epoch(t1)]})
    mapping = {"ts": pl.Datetime("ns")}
    pred = pl.col("ts").cast(pl.Datetime("ns"), strict=False) == t1
    node = get_parsed_expr(pred)
    vis = TranslatedPredicateVisitor(mapping)
    node.accept(vis)
    out = df.filter(vis.process_results())
    assert out["ts"].to_list() == [_ns_epoch(t1)]


def test_translated_visitor_datetime_not_in_values():
    base = datetime(2025, 1, 1, 0, 0, 0)
    t0 = base
    t1 = base.replace(second=1)
    t2 = base.replace(second=2)
    df = pl.DataFrame({"ts": [_ns_epoch(t0), _ns_epoch(t1), _ns_epoch(t2)]})
    mapping = {"ts": pl.Datetime("ns")}
    pred = (pl.col("ts") != t0) & (pl.col("ts") != t2)
    node = get_parsed_expr(pred)
    vis = TranslatedPredicateVisitor(mapping)
    node.accept(vis)
    rewritten = vis.process_results()
    out = df.filter(rewritten)
    assert out["ts"].to_list() == [_ns_epoch(t1)]


def test_translated_visitor_duration_not_in_values():
    vals = [0, 500, 1500, 2500]
    df = pl.DataFrame({"dur": vals})
    mapping = {"dur": pl.Duration("ms")}
    pred = (pl.col("dur") != timedelta(milliseconds=500)) & (pl.col("dur") != timedelta(milliseconds=2500))
    node = get_parsed_expr(pred)
    vis = TranslatedPredicateVisitor(mapping)
    node.accept(vis)
    rewritten = vis.process_results()
    out = df.filter(rewritten).sort("dur")
    assert out["dur"].to_list() == [0, 1500]


def test_translated_visitor_unmapped_not_in_values():
    # Unmapped Int64 column; NOT IN via unary negation of is_in
    df = pl.DataFrame({"x": [1, 2, 3]})
    mapping = {}  # unmapped
    pred = ~pl.col("x").is_in([1, 3])
    node = get_parsed_expr(pred)
    vis = TranslatedPredicateVisitor(mapping)
    node.accept(vis)
    rewritten = vis.process_results()
    out = df.filter(rewritten)
    assert out["x"].to_list() == [2]


@pytest.mark.xfail(reason="Polars handles is_in differently than equality here, see this issue: https://github.com/pola-rs/polars/issues/22824")
def test_translated_visitor_datetime_not_in_unary():
    # Mapped datetime column; verify ~is_in form works via visitor
    base = datetime(2025, 1, 1, 0, 0, 0)
    t0 = base
    t1 = base.replace(second=1)
    t2 = base.replace(second=2)
    df = pl.DataFrame({"ts": [_ns_epoch(t0), _ns_epoch(t1), _ns_epoch(t2)]})
    mapping = {"ts": pl.Datetime("ns")}
    pred = ~pl.col("ts").is_in([t0, t2])
    node = get_parsed_expr(pred)
    vis = TranslatedPredicateVisitor(mapping)
    node.accept(vis)
    rewritten = vis.process_results()
    # Evaluate against underlying ints; expect to keep only t1
    out = df.filter(rewritten)
    assert out["ts"].to_list() == [_ns_epoch(t1)]


def test_metadata_roundtrip_datetime_with_timezone():
    """Test that metadata_to_mapping correctly handles time_zone field.

    This test ensures the fix for the spec.timezone -> spec.time_zone typo
    is working correctly. Without this fix, timezones would be silently lost.
    """
    # Create a mapping with timezone-aware datetime
    original_mapping = {
        "ts_utc": pl.Datetime("ns", time_zone="UTC"),
        "ts_ny": pl.Datetime("us", time_zone="America/New_York"),
        "ts_no_tz": pl.Datetime("ms"),  # No timezone
    }

    # Roundtrip through metadata
    meta_bytes = mapping_to_metadata(original_mapping)
    recovered_mapping = metadata_to_mapping(meta_bytes)

    # Verify timezone is preserved
    assert recovered_mapping["ts_utc"].time_zone == "UTC"
    assert recovered_mapping["ts_ny"].time_zone == "America/New_York"
    assert recovered_mapping["ts_no_tz"].time_zone is None

    # Verify time units are preserved
    assert recovered_mapping["ts_utc"].time_unit == "ns"
    assert recovered_mapping["ts_ny"].time_unit == "us"
    assert recovered_mapping["ts_no_tz"].time_unit == "ms"


def test_metadata_roundtrip_duration():
    """Test that metadata_to_mapping correctly handles Duration types with various units."""
    original_mapping = {
        "dur_ns": pl.Duration("ns"),
        "dur_us": pl.Duration("us"),
        "dur_ms": pl.Duration("ms"),
    }

    # Roundtrip through metadata
    meta_bytes = mapping_to_metadata(original_mapping)
    recovered_mapping = metadata_to_mapping(meta_bytes)

    # Verify duration types are preserved
    assert isinstance(recovered_mapping["dur_ns"], pl.Duration)
    assert isinstance(recovered_mapping["dur_us"], pl.Duration)
    assert isinstance(recovered_mapping["dur_ms"], pl.Duration)

    # Verify time units are preserved
    assert recovered_mapping["dur_ns"].time_unit == "ns"
    assert recovered_mapping["dur_us"].time_unit == "us"
    assert recovered_mapping["dur_ms"].time_unit == "ms"
