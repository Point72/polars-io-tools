"""Translating a filter on the dtypes a user sees (for example nanosecond datetimes) into one on the values as stored
(for example integers).

Each row gives the stored data, the dtype each column is read as, and a filter on the columns as read. The stored data
filtered by the translated filter must keep the rows that Polars keeps when it filters the data cast to those dtypes.
"""

from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from datetime import datetime, time, timedelta
from enum import Enum, auto

import polars as pl
import pytest

from polars_io_tools.io_sources.base import get_parsed_expr
from polars_io_tools.io_sources.translated_source import (
    TranslatedPredicateVisitor,
    mapping_to_metadata,
    metadata_to_mapping,
    translate_polars_predicate,
)
from polars_io_tools.tests.helpers.cases import as_params
from polars_io_tools.tests.helpers.polars_versions import xfail_unless_polars_has


def _nanoseconds(moment: datetime) -> int:
    return pl.Series([moment]).cast(pl.Datetime("ns")).dt.timestamp("ns").item()


def _time_nanoseconds(*times: time) -> list[int]:
    return pl.Series(times, dtype=pl.Time).cast(pl.Int64).to_list()


# 2025-01-01 at 0, 1, 2 and 3 seconds past midnight.
SECOND_0, SECOND_1, SECOND_2, SECOND_3 = (datetime(2025, 1, 1, 0, 0, s) for s in range(4))
NOON_AND_TWO_SECONDS_AFTER = [time(12, 0, s) for s in range(3)]
TS_AS_NANOSECOND_DATETIME = {"ts": pl.Datetime("ns")}
DUR_AS_MILLISECOND_DURATION = {"dur": pl.Duration("ms")}
T_AS_TIME = {"t": pl.Time}
_THREE_SECONDS_STORED = {"ts": [_nanoseconds(SECOND_0), _nanoseconds(SECOND_1), _nanoseconds(SECOND_2)]}
_FOUR_SECONDS_STORED = {"ts": [_nanoseconds(moment) for moment in (SECOND_0, SECOND_1, SECOND_2, SECOND_3)]}
_MILLISECONDS_STORED = {"dur": [0, 500, 1500, 2500]}
_NOON_TIMES_STORED = {"t": _time_nanoseconds(*NOON_AND_TWO_SECONDS_AFTER)}
xfail_unless_is_in_converts_time_units = xfail_unless_polars_has("is_in_coerces_temporal_units")


class Kept(Enum):
    """Which stored rows the translated filter keeps, compared with the rows Polars keeps on the data as read."""

    SAME_ROWS = auto()
    SAME_ROWS_OR_MORE = auto()
    # and Polars keeps at least one row, so the check is not empty
    SAME_ROWS_OR_MORE_OF_AT_LEAST_ONE = auto()
    NOT_TRANSLATED = auto()


@dataclass(frozen=True)
class StoredValueTranslationCase:
    """A filter on columns read with other dtypes than they are stored with, and which stored rows its translation
    keeps."""

    id: str
    _: KW_ONLY
    # the data as stored, as plain values
    stored: dict[str, list]
    # column -> the dtype it is read as
    mapping: dict[str, pl.DataType]
    # a function returning the filter, for a filter that fails to build
    predicate: pl.Expr | Callable[[], pl.Expr]
    keeps: Kept = Kept.SAME_ROWS
    # when set, the row numbers the translated filter keeps, checked instead of ``keeps``
    row_numbers: list[int] | None = None
    # "visitor" (``TranslatedPredicateVisitor``) or "function" (``translate_polars_predicate``)
    translate_with: str = "visitor"
    marks: tuple = ()


CASES = [
    StoredValueTranslationCase(
        "translated_visitor_eq_datetime_ns",
        stored={"ts": [_nanoseconds(SECOND_0), _nanoseconds(SECOND_1)]},
        mapping=TS_AS_NANOSECOND_DATETIME,
        predicate=pl.col("ts") == pl.lit(SECOND_1),
    ),
    StoredValueTranslationCase(
        "translated_visitor_in_datetime_ns",
        stored=_THREE_SECONDS_STORED,
        mapping=TS_AS_NANOSECOND_DATETIME,
        predicate=pl.col("ts").is_in([SECOND_0, SECOND_2]),
        row_numbers=[0, 2],
        marks=(xfail_unless_is_in_converts_time_units,),
    ),
    StoredValueTranslationCase(
        "translated_visitor_eq_datetime_literal_cast",
        stored={"ts": [_nanoseconds(SECOND_0), _nanoseconds(SECOND_1)]},
        mapping=TS_AS_NANOSECOND_DATETIME,
        predicate=pl.col("ts") == pl.lit(SECOND_1).cast(pl.Datetime("ns"), strict=False),
    ),
    # the cast gives the integer its time unit, so it must stay when comparing against a ms column
    StoredValueTranslationCase(
        "translated_visitor_keeps_cast_on_integer_literal",
        stored={"ts": [_nanoseconds(SECOND_1) // 10**6]},
        mapping={"ts": pl.Datetime("ms")},
        predicate=pl.col("ts") == pl.lit(_nanoseconds(SECOND_1)).cast(pl.Datetime("ns")),
        translate_with="function",
    ),
    StoredValueTranslationCase(
        "translate_predicate_drops_only_untranslatable_conjunct",
        stored={"ts": [_nanoseconds(datetime(2025, 1, 1, h)) for h in (1, 2, 3)], "x": [1, 2, 3]},
        mapping=TS_AS_NANOSECOND_DATETIME,
        predicate=(pl.col("ts").dt.hour() == 2) & (pl.col("x") > 1),
        row_numbers=[1, 2],
        translate_with="function",
    ),
    StoredValueTranslationCase(
        "translate_predicate_drops_only_untranslatable_conjunct-negated",
        stored={"ts": [0], "x": [1]},
        mapping=TS_AS_NANOSECOND_DATETIME,
        predicate=~((pl.col("ts").dt.hour() == 2) & (pl.col("x") > 1)),
        keeps=Kept.NOT_TRANSLATED,
        translate_with="function",
    ),
    StoredValueTranslationCase(
        "translated_visitor_between_datetime_ns",
        stored=_FOUR_SECONDS_STORED,
        mapping=TS_AS_NANOSECOND_DATETIME,
        predicate=pl.col("ts").is_between(pl.lit(SECOND_1), pl.lit(SECOND_2), closed="both"),
    ),
    *[
        StoredValueTranslationCase(
            f"translated_visitor_between_closed_variants-{c}",
            stored=_FOUR_SECONDS_STORED,
            mapping=TS_AS_NANOSECOND_DATETIME,
            predicate=pl.col("ts").is_between(SECOND_1, SECOND_2, closed=c),
        )
        for c in ("left", "right", "none")
    ],
    StoredValueTranslationCase(
        "translated_visitor_duration_ms_gt",
        stored=_MILLISECONDS_STORED,
        mapping=DUR_AS_MILLISECOND_DURATION,
        predicate=pl.col("dur") > pl.lit(timedelta(milliseconds=1000)),
    ),
    StoredValueTranslationCase(
        "translated_visitor_time_eq",
        stored={"t": _time_nanoseconds(*NOON_AND_TWO_SECONDS_AFTER[:2])},
        mapping=T_AS_TIME,
        predicate=pl.col("t") == pl.lit(NOON_AND_TWO_SECONDS_AFTER[1]),
    ),
    StoredValueTranslationCase(
        "translated_visitor_time_is_in",
        stored=_NOON_TIMES_STORED,
        mapping=T_AS_TIME,
        predicate=pl.col("t").is_in([NOON_AND_TWO_SECONDS_AFTER[0], NOON_AND_TWO_SECONDS_AFTER[2]]),
    ),
    StoredValueTranslationCase(
        "translated_visitor_time_neq_value", stored=_NOON_TIMES_STORED, mapping=T_AS_TIME, predicate=pl.col("t") != NOON_AND_TWO_SECONDS_AFTER[0]
    ),
    StoredValueTranslationCase(
        "translated_visitor_duration_neq_values",
        stored=_MILLISECONDS_STORED,
        mapping=DUR_AS_MILLISECOND_DURATION,
        predicate=pl.col("dur") != timedelta(milliseconds=500),
    ),
    StoredValueTranslationCase(
        "translated_visitor_duration_is_in",
        stored=_MILLISECONDS_STORED,
        mapping=DUR_AS_MILLISECOND_DURATION,
        predicate=lambda: pl.col("dur").is_in([pl.lit(timedelta(milliseconds=v), dtype=pl.Duration("ms")) for v in (500, 2500)]),
        marks=(pytest.mark.xfail(reason="Polars does not support nested object types so the predicate is_in cannot be created.", strict=True),),
    ),
    # the mapping applies first, then the user's cast
    StoredValueTranslationCase(
        "translated_visitor_user_cast_layering",
        stored={"ts": [_nanoseconds(SECOND_0), _nanoseconds(SECOND_1)]},
        mapping=TS_AS_NANOSECOND_DATETIME,
        predicate=pl.col("ts").cast(pl.Datetime("ns"), strict=False) == SECOND_1,
    ),
    StoredValueTranslationCase(
        "translated_visitor_datetime_not_in_values",
        stored=_THREE_SECONDS_STORED,
        mapping=TS_AS_NANOSECOND_DATETIME,
        predicate=(pl.col("ts") != SECOND_0) & (pl.col("ts") != SECOND_2),
    ),
    StoredValueTranslationCase(
        "translated_visitor_duration_not_in_values",
        stored=_MILLISECONDS_STORED,
        mapping=DUR_AS_MILLISECOND_DURATION,
        predicate=(pl.col("dur") != timedelta(milliseconds=500)) & (pl.col("dur") != timedelta(milliseconds=2500)),
    ),
    StoredValueTranslationCase(
        "translated_visitor_unmapped_not_in_values", stored={"x": [1, 2, 3]}, mapping={}, predicate=~pl.col("x").is_in([1, 3])
    ),
    StoredValueTranslationCase(
        "translated_visitor_datetime_not_in_unary",
        stored=_THREE_SECONDS_STORED,
        mapping=TS_AS_NANOSECOND_DATETIME,
        predicate=~pl.col("ts").is_in([SECOND_0, SECOND_2]),
        row_numbers=[1],
        marks=(xfail_unless_is_in_converts_time_units,),
    ),
    *[
        StoredValueTranslationCase(
            f"translated_visitor_in_datetime_ns_matches_polars-{n}",
            stored={"ts": [_nanoseconds(datetime(2025, 1, d)) for d in (1, 2)]},
            mapping=TS_AS_NANOSECOND_DATETIME,
            predicate=p,
            keeps=Kept.SAME_ROWS_OR_MORE,
            translate_with="function",
        )
        for n, p in [("in", pl.col("ts").is_in([datetime(2025, 1, 1)])), ("not_in", ~pl.col("ts").is_in([datetime(2025, 1, 1)]))]
    ],
    *[
        StoredValueTranslationCase(
            f"translated_predicate_matches_logical_filter-{name}",
            stored=stored,
            mapping=mapping,
            predicate=p,
            keeps=Kept.SAME_ROWS_OR_MORE_OF_AT_LEAST_ONE,
            translate_with="function",
        )
        for name, stored, mapping, p in [
            ("uint8_wrapping_cast", {"x": [257, 1], "ts": [0, 1]}, TS_AS_NANOSECOND_DATETIME, pl.col("x").cast(pl.UInt8, wrap_numerical=True) == 1),
            ("enum_order_lt", {"x": ["z", "a"]}, {"x": pl.Enum(["z", "a"])}, pl.col("x") < "a"),
            ("date_cast_to_string", {"x": [0, 1]}, {"x": pl.Date}, pl.col("x").cast(pl.String) == "1970-01-02"),
            ("int_eq_true_as_boolean", {"x": [2, 0]}, {"x": pl.Boolean}, pl.col("x") == True),
        ]
    ],
]


def _translate(pred: pl.Expr, mapping: dict, translate_with: str) -> pl.Expr | None:
    if translate_with == "function":
        return translate_polars_predicate(pred, mapping)
    visitor = TranslatedPredicateVisitor(mapping)
    get_parsed_expr(pred).accept(visitor)
    return visitor.process_results()


@pytest.mark.parametrize("case", as_params(CASES))
def test_translated_predicate_keeps_logical_rows(case: StoredValueTranslationCase):
    predicate = case.predicate() if callable(case.predicate) else case.predicate
    df = pl.DataFrame(case.stored).with_row_index("__row")
    rewritten = _translate(predicate, case.mapping, case.translate_with)
    if case.keeps is Kept.NOT_TRANSLATED:
        assert rewritten is None, rewritten
        return
    got = df["__row"].to_list() if rewritten is None else df.filter(rewritten)["__row"].to_list()
    if case.row_numbers is not None:
        assert got == case.row_numbers
        return
    want = df.cast(case.mapping).filter(predicate)["__row"].to_list()
    if case.keeps is Kept.SAME_ROWS:
        assert got == want
    else:
        assert set(want) <= set(got)
        assert want or case.keeps is Kept.SAME_ROWS_OR_MORE


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


@pytest.mark.parametrize(
    "dtype", [pl.Date, pl.Decimal(10, 2), pl.Categorical, pl.Enum(["a"]), pl.Int8], ids=["Date", "decimal", "Categorical", "enum", "Int8"]
)
def test_mapping_to_metadata_rejects_unsupported_dtype(dtype):
    with pytest.raises(ValueError, match="Unsupported dtype"):
        mapping_to_metadata({"c": dtype})
