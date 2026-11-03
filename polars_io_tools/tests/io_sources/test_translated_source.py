"""Translating a predicate on logical dtypes into one on the stored physical values.

One table: stored data, the mapping to logical dtypes, a predicate on the logical columns, and what filtering the stored
data by the translated predicate must keep, compared with Polars filtering the data cast to its logical dtypes.
"""

from datetime import datetime, time, timedelta

import polars as pl
import pytest

from polars_io_tools.io_sources.base import get_parsed_expr
from polars_io_tools.io_sources.translated_source import (
    TranslatedPredicateVisitor,
    mapping_to_metadata,
    metadata_to_mapping,
    translate_polars_predicate,
)
from polars_io_tools.tests.helpers.cases import Case, params
from polars_io_tools.tests.helpers.polars_versions import xfail_unless


def _ns(dt: datetime) -> int:
    return pl.Series([dt]).cast(pl.Datetime("ns")).dt.timestamp("ns").item()


def _time_ns(*ts: time) -> list[int]:
    return pl.Series(ts, dtype=pl.Time).cast(pl.Int64).to_list()


T0, T1, T2, T3 = (datetime(2025, 1, 1, 0, 0, s) for s in range(4))
NOON = [time(12, 0, s) for s in range(3)]
NS = {"ts": pl.Datetime("ns")}
MS = {"dur": pl.Duration("ms")}
TIME = {"t": pl.Time}
_TS3 = {"ts": [_ns(T0), _ns(T1), _ns(T2)]}
_TS4 = {"ts": [_ns(t) for t in (T0, T1, T2, T3)]}
_DUR = {"dur": [0, 500, 1500, 2500]}
_TIMES = {"t": _time_ns(*NOON)}
ts, dur, t = pl.col("ts"), pl.col("dur"), pl.col("t")
_IS_IN_ISSUE = xfail_unless("is_in_coerces_temporal_units")


def V(id, stored, mapping, pred, expect="exact", via="visitor", marks=()):
    """``expect``: "exact" keeps the rows the logical filter keeps, "superset" at least those ("nonempty": and there are some), a list the
    given row numbers, None means nothing is translated. ``pred`` may be a function, for one that fails to build."""
    return Case(id, pred=pred, stored=stored, mapping=mapping, via=via, expect=expect, marks=marks)


CASES = [
    V("translated_visitor_eq_datetime_ns", {"ts": [_ns(T0), _ns(T1)]}, NS, ts == pl.lit(T1)),
    V("translated_visitor_in_datetime_ns", _TS3, NS, ts.is_in([T0, T2]), [0, 2], marks=_IS_IN_ISSUE),
    V("translated_visitor_eq_datetime_literal_cast", {"ts": [_ns(T0), _ns(T1)]}, NS, ts == pl.lit(T1).cast(pl.Datetime("ns"), strict=False)),
    # the cast gives the integer its time unit, so it must stay when comparing against a ms column
    V(
        "translated_visitor_keeps_cast_on_integer_literal",
        {"ts": [_ns(T1) // 10**6]},
        {"ts": pl.Datetime("ms")},
        ts == pl.lit(_ns(T1)).cast(pl.Datetime("ns")),
        via="translate",
    ),
    V(
        "translate_predicate_drops_only_untranslatable_conjunct",
        {"ts": [_ns(datetime(2025, 1, 1, h)) for h in (1, 2, 3)], "x": [1, 2, 3]},
        NS,
        (ts.dt.hour() == 2) & (pl.col("x") > 1),
        [1, 2],
        via="translate",
    ),
    V(
        "translate_predicate_drops_only_untranslatable_conjunct-negated",
        {"ts": [0], "x": [1]},
        NS,
        ~((ts.dt.hour() == 2) & (pl.col("x") > 1)),
        None,
        via="translate",
    ),
    V("translated_visitor_between_datetime_ns", _TS4, NS, ts.is_between(pl.lit(T1), pl.lit(T2), closed="both")),
    *[V(f"translated_visitor_between_closed_variants-{c}", _TS4, NS, ts.is_between(T1, T2, closed=c)) for c in ("left", "right", "none")],
    V("translated_visitor_duration_ms_gt", _DUR, MS, dur > pl.lit(timedelta(milliseconds=1000))),
    V("translated_visitor_time_eq", {"t": _time_ns(*NOON[:2])}, TIME, t == pl.lit(NOON[1])),
    V("translated_visitor_time_is_in", _TIMES, TIME, t.is_in([NOON[0], NOON[2]])),
    V("translated_visitor_time_neq_value", _TIMES, TIME, t != NOON[0]),
    V("translated_visitor_duration_neq_values", _DUR, MS, dur != timedelta(milliseconds=500)),
    V(
        "translated_visitor_duration_is_in",
        _DUR,
        MS,
        lambda: dur.is_in([pl.lit(timedelta(milliseconds=v), dtype=pl.Duration("ms")) for v in (500, 2500)]),
        marks=pytest.mark.xfail(reason="Polars does not support nested object types so the predicate is_in cannot be created.", strict=True),
    ),
    # the mapping applies first, then the user's cast
    V("translated_visitor_user_cast_layering", {"ts": [_ns(T0), _ns(T1)]}, NS, ts.cast(pl.Datetime("ns"), strict=False) == T1),
    V("translated_visitor_datetime_not_in_values", _TS3, NS, (ts != T0) & (ts != T2)),
    V("translated_visitor_duration_not_in_values", _DUR, MS, (dur != timedelta(milliseconds=500)) & (dur != timedelta(milliseconds=2500))),
    V("translated_visitor_unmapped_not_in_values", {"x": [1, 2, 3]}, {}, ~pl.col("x").is_in([1, 3])),
    V("translated_visitor_datetime_not_in_unary", _TS3, NS, ~ts.is_in([T0, T2]), [1], marks=_IS_IN_ISSUE),
    *[
        V(
            f"translated_visitor_in_datetime_ns_matches_polars-{n}",
            {"ts": [_ns(datetime(2025, 1, d)) for d in (1, 2)]},
            NS,
            p,
            "superset",
            via="translate",
        )
        for n, p in [("in", ts.is_in([datetime(2025, 1, 1)])), ("not_in", ~ts.is_in([datetime(2025, 1, 1)]))]
    ],
    *[
        V(f"translated_predicate_matches_logical_filter-{i}", stored, mapping, p, "nonempty", via="translate")
        for i, (stored, mapping, p) in enumerate(
            [
                ({"x": [257, 1], "ts": [0, 1]}, NS, pl.col("x").cast(pl.UInt8, wrap_numerical=True) == 1),
                ({"x": ["z", "a"]}, {"x": pl.Enum(["z", "a"])}, pl.col("x") < "a"),
                ({"x": [0, 1]}, {"x": pl.Date}, pl.col("x").cast(pl.String) == "1970-01-02"),
                ({"x": [2, 0]}, {"x": pl.Boolean}, pl.col("x") == True),
            ]
        )
    ],
]


def _translate(pred: pl.Expr, mapping: dict, via: str) -> pl.Expr | None:
    if via == "translate":
        return translate_polars_predicate(pred, mapping)
    visitor = TranslatedPredicateVisitor(mapping)
    get_parsed_expr(pred).accept(visitor)
    return visitor.process_results()


@pytest.mark.parametrize(("pred", "stored", "mapping", "via", "expect"), params(CASES, "pred", "stored", "mapping", "via", "expect"))
def test_translated_predicate_keeps_logical_rows(pred, stored, mapping, via, expect):
    pred = pred() if callable(pred) else pred
    df = pl.DataFrame(stored).with_row_index("__row")
    rewritten = _translate(pred, mapping, via)
    if expect is None:
        assert rewritten is None, rewritten
        return
    got = df["__row"].to_list() if rewritten is None else df.filter(rewritten)["__row"].to_list()
    if isinstance(expect, list):
        assert got == expect
        return
    want = df.cast(mapping).filter(pred)["__row"].to_list()
    if expect == "exact":
        assert got == want
    else:
        assert set(want) <= set(got)
        assert want or expect == "superset"


@pytest.mark.parametrize(
    "mapping",
    [
        pytest.param(
            {"ts_utc": pl.Datetime("ns", time_zone="UTC"), "ts_ny": pl.Datetime("us", time_zone="America/New_York"), "ts_no_tz": pl.Datetime("ms")},
            id="metadata_roundtrip_datetime_with_timezone",
        ),
        pytest.param({"dur_ns": pl.Duration("ns"), "dur_us": pl.Duration("us"), "dur_ms": pl.Duration("ms")}, id="metadata_roundtrip_duration"),
    ],
)
def test_metadata_roundtrip(mapping):
    """Time zones and time units survive the metadata round trip."""
    recovered = metadata_to_mapping(mapping_to_metadata(mapping))
    assert {k: (v, getattr(v, "time_zone", None), v.time_unit) for k, v in recovered.items()} == {
        k: (v, getattr(v, "time_zone", None), v.time_unit) for k, v in mapping.items()
    }


@pytest.mark.parametrize("dtype", [pl.Date, pl.Decimal(10, 2), pl.Categorical, pl.Enum(["a"]), pl.Int8])
def test_mapping_to_metadata_rejects_unsupported_dtype(dtype):
    with pytest.raises(ValueError, match="Unsupported dtype"):
        mapping_to_metadata({"c": dtype})
