"""concat_named: which sources a query reads, and how identifiers become typed key columns.

A pruning row states, per source, whether it may be read and whether it must receive a filter. An identifier row
states the typed value a string or Python identifier becomes, or for full temporal precision the integer that value
is stored as.
"""

import datetime
import decimal
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_io_tools as cpl
from polars_io_tools._compat import POLARS_HAS_COLLECT_BATCHES
from polars_io_tools.testing import io_source_assert
from polars_io_tools.tests.helpers.cases import as_params
from polars_io_tools.tests.helpers.pickling import pickle_round_trip

# How a source may be read.
READ_WITHOUT_FILTER = "read without a filter"
READ_WITH_FILTER = "read with a filter"
READ_WITH_OR_WITHOUT_FILTER = "read, with or without a filter"
NOT_READ = "not read"
JAN_1_2023, FEB_1_2023 = datetime.date(2023, 1, 1), datetime.date(2023, 2, 1)
TWO_ROW_FRAMES = [pl.DataFrame({"a": [1 + 2 * i, 2 + 2 * i], "b": [11 + 2 * i, 12 + 2 * i]}) for i in range(4)]
THREE_ROW_FRAMES = [pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), pl.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]})]


@dataclass(frozen=True)
class PruningCase:
    """A filter on the key columns of ``concat_named``, the result, and which sources are read."""

    id: str
    _: KW_ONLY
    # identifier (a tuple of key values) -> its frame and how it may be read (``READ_...`` or ``NOT_READ``)
    sources: dict[tuple, tuple[pl.DataFrame, object]]
    # the key column names, each optionally with a dtype
    key_columns: list
    predicate: pl.Expr | None
    expected: pl.DataFrame
    # applied after the filter
    query: Callable[[pl.LazyFrame], pl.LazyFrame] | None = None


PRUNING = [
    PruningCase(
        "concat_named_basic",
        sources={("foo",): (THREE_ROW_FRAMES[0], READ_WITHOUT_FILTER), ("bar",): (THREE_ROW_FRAMES[1], READ_WITHOUT_FILTER)},
        key_columns=["source"],
        predicate=None,
        expected=pl.DataFrame({"a": [1, 2, 3, 7, 8, 9], "b": [4, 5, 6, 10, 11, 12], "source": ["foo"] * 3 + ["bar"] * 3}),
    ),
    PruningCase(
        "concat_named_filter_pushdown",
        sources={("foo",): (THREE_ROW_FRAMES[0], READ_WITHOUT_FILTER), ("bar",): (THREE_ROW_FRAMES[1], NOT_READ)},
        key_columns=["source"],
        predicate=pl.col("source") == "foo",
        expected=pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "source": ["foo"] * 3}),
    ),
    PruningCase(
        "concat_named_filter_selects_no_sources",
        sources={("foo",): (pl.DataFrame({"value": [1]}), NOT_READ), ("bar",): (pl.DataFrame({"value": [2]}), NOT_READ)},
        key_columns=["source"],
        predicate=pl.col("source") == "missing",
        expected=pl.DataFrame(schema={"value": pl.Int64, "source": pl.String}),
    ),
    PruningCase(
        "concat_named_multiple_unique_columns",
        sources={
            ("east", "A"): (TWO_ROW_FRAMES[0], READ_WITHOUT_FILTER),
            ("east", "B"): (TWO_ROW_FRAMES[1], NOT_READ),
            ("west", "A"): (TWO_ROW_FRAMES[2], NOT_READ),
            ("west", "B"): (TWO_ROW_FRAMES[3], NOT_READ),
        },
        key_columns=["region", "type"],
        predicate=(pl.col("region") == "east") & (pl.col("type") == "A"),
        expected=pl.DataFrame({"a": [1, 2], "b": [11, 12], "region": ["east"] * 2, "type": ["A"] * 2}),
    ),
    PruningCase(
        "concat_named_different_data_types",
        sources={(JAN_1_2023, 100): (TWO_ROW_FRAMES[0], READ_WITHOUT_FILTER), (FEB_1_2023, 200): (TWO_ROW_FRAMES[1], NOT_READ)},
        key_columns=["date", ("id", pl.Float64)],
        predicate=pl.col("date") == JAN_1_2023,
        expected=pl.DataFrame({"a": [1, 2], "b": [11, 12], "date": [JAN_1_2023, JAN_1_2023], "id": [100.0, 100.0]}),
    ),
    PruningCase(
        "concat_named_or_condition",
        sources={
            ("foo",): (TWO_ROW_FRAMES[0], READ_WITH_FILTER),
            ("bar",): (TWO_ROW_FRAMES[1], READ_WITH_FILTER),
            ("baz",): (TWO_ROW_FRAMES[2], NOT_READ),
        },
        key_columns=["source"],
        predicate=((pl.col("source") == "foo") | (pl.col("source") == "bar")) & (pl.col("a") > 1),
        expected=pl.DataFrame({"a": [2, 3, 4], "b": [12, 13, 14], "source": ["foo", "bar", "bar"]}),
    ),
    PruningCase(
        "concat_named_is_in_filter",
        sources={
            ("foo",): (TWO_ROW_FRAMES[0], READ_WITHOUT_FILTER),
            ("bar",): (TWO_ROW_FRAMES[1], READ_WITHOUT_FILTER),
            ("baz",): (TWO_ROW_FRAMES[2], NOT_READ),
        },
        key_columns=["source"],
        predicate=pl.col("source").is_in(["foo", "bar"]),
        expected=pl.DataFrame({"a": [1, 2, 3, 4], "b": [11, 12, 13, 14], "source": ["foo", "foo", "bar", "bar"]}),
    ),
    # A filter on a column the sources share reads every source.
    PruningCase(
        "concat_named_filter_non_unique_column",
        sources={("foo",): (THREE_ROW_FRAMES[0], READ_WITH_OR_WITHOUT_FILTER), ("bar",): (THREE_ROW_FRAMES[1], READ_WITH_OR_WITHOUT_FILTER)},
        key_columns=["source"],
        predicate=pl.col("a") == 1,
        expected=pl.DataFrame({"a": [1], "b": [4], "source": ["foo"]}),
    ),
    PruningCase(
        "concat_named_complex_predicate",
        sources={
            ("foo",): (THREE_ROW_FRAMES[0], READ_WITH_OR_WITHOUT_FILTER),
            ("bar",): (THREE_ROW_FRAMES[1].with_columns(a=pl.Series([1, 8, 9])), NOT_READ),
        },
        key_columns=["source"],
        predicate=(pl.col("source") == "foo") & (pl.col("a") >= 2),
        expected=pl.DataFrame({"a": [2, 3], "b": [5, 6], "source": ["foo", "foo"]}),
    ),
    PruningCase(
        "concat_named_with_column_selection",
        sources={
            ("foo",): (TWO_ROW_FRAMES[0].with_columns(c=pl.Series([5, 6])), READ_WITHOUT_FILTER),
            ("bar",): (TWO_ROW_FRAMES[1].with_columns(c=pl.Series([11, 12])), NOT_READ),
        },
        key_columns=["source"],
        predicate=pl.col("source") == "foo",
        expected=pl.DataFrame({"a": [1, 2], "c": [5, 6]}),
        query=lambda lf: lf.select("a", "c"),
    ),
    PruningCase(
        "concat_named_filter_with_inequality",
        sources={
            (datetime.date(2025, 5, 1),): (TWO_ROW_FRAMES[0], READ_WITH_OR_WITHOUT_FILTER),
            (datetime.date(2025, 5, 2),): (TWO_ROW_FRAMES[1], NOT_READ),
        },
        key_columns=["source"],
        predicate=pl.col("source") < datetime.date(2025, 5, 2),
        expected=pl.DataFrame({"a": [1, 2], "b": [11, 12], "source": [datetime.date(2025, 5, 1)] * 2}),
    ),
    PruningCase(
        "concat_named_python_dtypes_prune_multiple_sources",
        sources={
            ("rates", "1"): (pl.DataFrame({"value": [30]}), NOT_READ),
            ("rates", "2"): (pl.DataFrame({"value": [10, 20]}), READ_WITH_OR_WITHOUT_FILTER),
            ("credit", "2"): (pl.DataFrame({"value": [30]}), NOT_READ),
        },
        key_columns=[("desk", str), ("bucket", int)],
        predicate=(pl.col("desk") == "rates") & (pl.col("bucket") == 2),
        expected=pl.DataFrame({"value": [10, 20], "desk": ["rates", "rates"], "bucket": [2, 2]}),
    ),
]


@pytest.mark.parametrize("case", as_params(PRUNING))
def test_concat_named_pruning(case: PruningCase):
    seen = {key: [] for key in case.sources}
    lf = cpl.concat_named({key: io_source_assert(df, seen[key].append) for key, (df, _) in case.sources.items()}, case.key_columns)
    lf = lf if case.predicate is None else lf.filter(case.predicate)
    assert_frame_equal((case.query or (lambda frame: frame))(lf).collect(), case.expected)
    for key, (_, how_read) in case.sources.items():
        if how_read is NOT_READ:
            assert not seen[key], f"{key} was read"
            continue
        assert seen[key], f"{key} was not read"
        if how_read is not READ_WITH_OR_WITHOUT_FILTER:
            assert all((p is not None) == (how_read is READ_WITH_FILTER) for p in seen[key]), (key, seen[key])


class StoredInteger(int):
    """An identifier's expected value given as the integer its temporal dtype stores."""


@dataclass(frozen=True)
class IdentifierCase:
    """An identifier declared with a dtype, and the value and column dtype it becomes."""

    id: str
    _: KW_ONLY
    identifier: object
    dtype: object
    expected_value: object
    # None: the column has ``dtype``
    expected_polars_dtype: pl.DataType | None = None


IDENTIFIERS = [
    IdentifierCase(
        "concat_named_temporal_string_identifiers-date_string", identifier="2023-01-01", dtype=pl.Date, expected_value=datetime.date(2023, 1, 1)
    ),
    IdentifierCase(
        "concat_named_temporal_string_identifiers-datetime_string_ms",
        identifier="2023-01-01T12:30:00",
        dtype=pl.Datetime("ms"),
        expected_value=datetime.datetime(2023, 1, 1, 12, 30),
    ),
    IdentifierCase(
        "concat_named_temporal_string_identifiers-time_string", identifier="12:30:00", dtype=pl.Time, expected_value=datetime.time(12, 30)
    ),
    IdentifierCase(
        "concat_named_python_dtype_identifiers-str_label", identifier="label", dtype=str, expected_value="label", expected_polars_dtype=pl.String
    ),
    IdentifierCase(
        "concat_named_python_dtype_identifiers-int_from_string", identifier="42", dtype=int, expected_value=42, expected_polars_dtype=pl.Int64
    ),
    IdentifierCase(
        "concat_named_python_dtype_identifiers-float_from_string", identifier="3.5", dtype=float, expected_value=3.5, expected_polars_dtype=pl.Float64
    ),
    IdentifierCase(
        "concat_named_python_dtype_identifiers-date_from_string",
        identifier="2023-01-01",
        dtype=datetime.date,
        expected_value=datetime.date(2023, 1, 1),
        expected_polars_dtype=pl.Date,
    ),
    IdentifierCase(
        "concat_named_python_dtype_identifiers-datetime_from_string",
        identifier="2023-01-01T12:30:00",
        dtype=datetime.datetime,
        expected_value=datetime.datetime(2023, 1, 1, 12, 30),
        expected_polars_dtype=pl.Datetime("us"),
    ),
    IdentifierCase(
        "concat_named_python_dtype_identifiers-time_from_string",
        identifier="12:30:00",
        dtype=datetime.time,
        expected_value=datetime.time(12, 30),
        expected_polars_dtype=pl.Time,
    ),
    IdentifierCase("concat_named_python_dtype_identifiers-int", identifier=42, dtype=int, expected_value=42, expected_polars_dtype=pl.Int64),
    IdentifierCase("concat_named_python_dtype_identifiers-bool", identifier=True, dtype=bool, expected_value=True, expected_polars_dtype=pl.Boolean),
    IdentifierCase(
        "concat_named_python_dtype_identifiers-datetime_as_pl_datetime",
        identifier=datetime.datetime(2023, 1, 1),
        dtype=pl.Datetime,
        expected_value=datetime.datetime(2023, 1, 1),
        expected_polars_dtype=pl.Datetime("us"),
    ),
    IdentifierCase(
        "concat_named_python_dtype_identifiers-datetime_string_as_pl_datetime",
        identifier="2023-01-01T12:30:00",
        dtype=pl.Datetime,
        expected_value=datetime.datetime(2023, 1, 1, 12, 30),
        expected_polars_dtype=pl.Datetime("us"),
    ),
    IdentifierCase(
        "concat_named_python_dtype_identifiers-timedelta_as_pl_duration",
        identifier=datetime.timedelta(days=1),
        dtype=pl.Duration,
        expected_value=datetime.timedelta(days=1),
        expected_polars_dtype=pl.Duration("us"),
    ),
    IdentifierCase(
        "concat_named_python_dtype_identifiers-timedelta_as_python_timedelta",
        identifier=datetime.timedelta(days=1),
        dtype=datetime.timedelta,
        expected_value=datetime.timedelta(days=1),
        expected_polars_dtype=pl.Duration("us"),
    ),
    IdentifierCase(
        "concat_named_python_dtype_identifiers-categorical",
        identifier="rates",
        dtype=pl.Categorical,
        expected_value="rates",
        expected_polars_dtype=pl.Categorical(),
    ),
    IdentifierCase(
        "concat_named_temporal_identifier_precision-epoch_milliseconds_int",
        identifier=1672531200000,
        dtype=pl.Datetime("ms"),
        expected_value=StoredInteger(1672531200000),
    ),
    IdentifierCase(
        "concat_named_temporal_identifier_precision-ns_string_keeps_nanoseconds",
        identifier="2023-01-01T12:30:00.123456789",
        dtype=pl.Datetime("ns"),
        expected_value=StoredInteger(1672576200123456789),
    ),
    IdentifierCase(
        "concat_named_temporal_identifier_precision-ms_string_in_new_york",
        identifier="2023-01-01T12:30:00",
        dtype=pl.Datetime("ms", "America/New_York"),
        expected_value=StoredInteger(1672594200000),
    ),
    # Offsets, padding and sub-microsecond digits in temporal strings.
    *[
        IdentifierCase(
            f"concat_named_legacy_temporal_strings-{name}", identifier=s, dtype=pl.Datetime("us"), expected_value=StoredInteger(1672576200000000)
        )
        for name, s in [
            ("z_suffix", "2023-01-01T12:30:00Z"),
            ("offset_hh_mm", "2023-01-01T12:30:00+05:30"),
            ("offset_without_seconds_in_time", "2023-01-01T12:30+05:30"),
            ("offset_hh", "2023-01-01T12:30:00+05"),
            ("offset_with_seconds", "2023-01-01T12:30:00+05:30:15"),
            ("space_before_offset", "2023-01-01T12:30:00 +05:30"),
        ]
    ],
    IdentifierCase(
        "concat_named_legacy_temporal_strings-ns_string_with_negative_offset",
        identifier="2023-01-01T12:30:00.123456789-05:30",
        dtype=pl.Datetime("ns"),
        expected_value=StoredInteger(1672576200123456789),
    ),
    IdentifierCase("concat_named_legacy_temporal_strings-padded_date", identifier=" 2023-01-01 ", dtype=pl.Date, expected_value=StoredInteger(19358)),
    IdentifierCase(
        "concat_named_legacy_temporal_strings-padded_time", identifier=" 12:30:00 ", dtype=pl.Time, expected_value=StoredInteger(45000000000000)
    ),
    IdentifierCase(
        "concat_named_legacy_temporal_strings-padded_datetime_ms",
        identifier=" 2023-01-01T12:30:00 ",
        dtype=pl.Datetime("ms"),
        expected_value=StoredInteger(1672576200000),
    ),
    IdentifierCase(
        "concat_named_legacy_temporal_strings-offset_into_new_york",
        identifier="2023-01-01T12:30:00+05:30",
        dtype=pl.Datetime("us", "America/New_York"),
        expected_value=StoredInteger(1672556400000000),
    ),
    *[
        IdentifierCase(f"concat_named_accepts_nanosecond_boundaries-{name}", identifier=s, dtype=pl.Datetime("ns"), expected_value=StoredInteger(n))
        for name, s, n in [
            ("minimum", "1677-09-21T00:12:43.145224192", -9223372036854775808),
            ("maximum", "2262-04-11T23:47:16.854775807", 9223372036854775807),
            ("one_microsecond_above_minimum", "1677-09-21T00:12:43.145225192", -9223372036854774808),
            ("one_ns_before_epoch", "1969-12-31T23:59:59.999999999", -1),
        ]
    ],
]


@pytest.mark.parametrize("case", as_params(IDENTIFIERS))
def test_concat_named_identifier(case: IdentifierCase):
    value, polars_dtype = case.expected_value, case.expected_polars_dtype or case.dtype
    lf = cpl.concat_named({(case.identifier,): pl.LazyFrame({"value": [1, 2]})}, [("identifier", case.dtype)])
    if isinstance(value, StoredInteger):
        result = lf.filter(pl.col("identifier").cast(pl.Int64) == value).collect()
        assert result["value"].to_list() == [1, 2]
        assert result.schema["identifier"] == polars_dtype
        assert result["identifier"].cast(pl.Int64).to_list() == [value, value]
    else:
        result = lf.filter(pl.col("identifier") == value).collect()
        assert_frame_equal(result, pl.DataFrame({"value": [1, 2], "identifier": pl.Series([value, value], dtype=polars_dtype)}))


@dataclass(frozen=True)
class ConcatNamedErrorCase:
    """Identifiers and key columns that ``concat_named`` must reject, and the error."""

    id: str
    _: KW_ONLY
    # the identifiers; their frames do not matter
    keys: dict
    error: type[Exception]
    key_columns: list
    # a pattern the error message must match; None checks only the type
    match: str | None


ERRORS = [
    *[
        ConcatNamedErrorCase(
            f"concat_named_unspecified_decimal_does_not_round-{dtype}",
            keys={(decimal.Decimal("1.5"),): None},
            error=TypeError,
            key_columns=[("identifier", dtype)],
            match=None,
        )
        for dtype in (pl.Decimal, decimal.Decimal)
    ],
    *[
        ConcatNamedErrorCase(
            f"concat_named_rejects_out_of_range_nanoseconds-{dtype}-{s}",
            keys={(s,): None},
            error=pl.exceptions.InvalidOperationError,
            key_columns=[("identifier", dtype)],
            match="nanosecond.*range",
        )
        for dtype in (pl.Datetime("ns"), pl.Datetime("ns", "UTC"))
        for s in ["1600-01-01T00:00:00", "1600-01-01T00:00:00+05:30", "2500-01-01T00:00:00Z"]
    ],
    ConcatNamedErrorCase("concat_named_empty_dict", keys={}, error=ValueError, key_columns=["source"], match=None),
    ConcatNamedErrorCase(
        "concat_named_mismatched_columns_keys",
        keys={("foo",): None, ("bar",): None},
        error=ValueError,
        key_columns=["source", "type"],
        match="Number of unique columns 2 does not match number of keys 1",
    ),
]


@pytest.mark.parametrize("case", as_params(ERRORS))
def test_concat_named_error(case: ConcatNamedErrorCase):
    with pytest.raises(case.error, match=case.match):
        cpl.concat_named({key: pl.LazyFrame({"value": [1]}) for key in case.keys}, case.key_columns).collect()


@dataclass(frozen=True)
class PruningMatchesConcatCase:
    """Frames by identifier, concat options and a filter; ``concat_named`` must return what ``pl.concat`` returns."""

    id: str
    _: KW_ONLY
    # identifier -> frame
    lf_dict: dict[tuple, pl.LazyFrame]
    # keyword arguments for both concats
    kwargs: dict
    pred: pl.Expr


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            # 2**31 - 1 is an Int32 literal, so ``k + 1`` wraps in the data and must wrap in the pruning index too.
            PruningMatchesConcatCase(
                "int32_key_wraps_in_pruning_index", lf_dict={(2**31 - 1,): pl.LazyFrame({"v": [42]})}, kwargs={}, pred=(pl.col("k") + 1) < 0
            ),
            PruningMatchesConcatCase(
                "diagonal_filter_on_column_of_one_source",
                lf_dict={("a",): pl.LazyFrame({"x": [1]}), ("b",): pl.LazyFrame({"y": [2]})},
                kwargs={"how": "diagonal"},
                pred=pl.col("y") == 2,
            ),
            PruningMatchesConcatCase(
                "diagonal_filter_on_key",
                lf_dict={("a",): pl.LazyFrame({"x": [1]}), ("b",): pl.LazyFrame({"y": [2]})},
                kwargs={"how": "diagonal"},
                pred=pl.col("k") == "b",
            ),
            PruningMatchesConcatCase(
                "vertical_relaxed_filter_on_key",
                lf_dict={("a",): pl.LazyFrame({"x": [1]}), ("b",): pl.LazyFrame({"x": [2.5]})},
                kwargs={"how": "vertical_relaxed"},
                pred=pl.col("k") == "a",
            ),
        ]
    ),
)
def test_concat_named_pruning_matches_concat(case: PruningMatchesConcatCase):
    expected = pl.concat([lf.with_columns(k=pl.lit(key[0])) for key, lf in case.lf_dict.items()], **case.kwargs).filter(case.pred).collect()
    assert_frame_equal(cpl.concat_named(case.lf_dict, ["k"], **case.kwargs).filter(case.pred).collect(), expected)


@pytest.mark.skipif(not POLARS_HAS_COLLECT_BATCHES, reason="collect_batches requires Polars >= 1.34.0")
def test_collect_batches_concat_named_yields_equal():
    lf1 = pl.LazyFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    lf2 = pl.LazyFrame({"a": [4, 5, 6], "b": [40, 50, 60]})

    lf = cpl.concat_named({("foo",): lf1, ("bar",): lf2}, ["source"]).select(["a", "b", "source"]).sort(["source", "a"])  # stable compare

    chunks = list(lf.collect_batches(chunk_size=2))
    combined = pl.concat(chunks, how="vertical").sort(["source", "a"])  # ensure identical ordering
    expected = lf.collect()

    assert_frame_equal(combined, expected)


class TestConcatNamedPickle:
    """Tests for concat_named pickle support."""

    def test_concat_named_pickle_basic(self):
        """concat_named LazyFrames can be pickled and unpickled."""
        df1 = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        df2 = pl.DataFrame({"a": [3, 4], "b": ["z", "w"]})

        # concat_named requires identifier_cols which must match the key tuple structure
        lf = cpl.concat_named(
            {("first",): df1.lazy(), ("second",): df2.lazy()},
            identifier_cols=["source"],
        )
        lf_unpickled = pickle_round_trip(lf)

        # Verify results match exactly (sort for deterministic order)
        expected = lf.sort("a").collect()
        result = lf_unpickled.sort("a").collect()
        assert_frame_equal(result, expected)
