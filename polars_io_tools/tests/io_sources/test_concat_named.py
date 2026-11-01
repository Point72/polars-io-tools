"""concat_named: which sources a query reads, and how identifiers become typed key columns.

A pruning row names, per source, whether it may be read and what predicate it must receive: ``None`` none, ``SOME`` a
predicate, ``ANY`` anything, ``NEVER`` no read at all. An identifier row names the typed value a string or Python
identifier becomes, or its physical integer for temporal precision.
"""

import datetime
import decimal

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_io_tools as cpl
from polars_io_tools.tests.helpers.catalogue import Case, params

from .conftest import io_source_assert

c = pl.col
SOME, ANY, NEVER = "some predicate", "any predicate", "never read"
D1, D2 = datetime.date(2023, 1, 1), datetime.date(2023, 2, 1)
AB = [pl.DataFrame({"a": [1 + 2 * i, 2 + 2 * i], "b": [11 + 2 * i, 12 + 2 * i]}) for i in range(4)]
ABC = [pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), pl.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]})]


def P(id, sources, names, pred, expect, query=None):
    """A pruning row: ``sources`` maps an identifier to (frame, what it must receive); ``expect`` is the result."""
    return Case(id, pred, pl.DataFrame(expect) if isinstance(expect, dict) else expect, {"sources": sources, "names": names, "query": query})


PRUNING = [
    P(
        "concat_named_basic",
        {("foo",): (ABC[0], None), ("bar",): (ABC[1], None)},
        ["source"],
        None,
        {"a": [1, 2, 3, 7, 8, 9], "b": [4, 5, 6, 10, 11, 12], "source": ["foo"] * 3 + ["bar"] * 3},
    ),
    P(
        "concat_named_filter_pushdown",
        {("foo",): (ABC[0], None), ("bar",): (ABC[1], NEVER)},
        ["source"],
        c("source") == "foo",
        {"a": [1, 2, 3], "b": [4, 5, 6], "source": ["foo"] * 3},
    ),
    P(
        "concat_named_filter_selects_no_sources",
        {("foo",): (pl.DataFrame({"value": [1]}), NEVER), ("bar",): (pl.DataFrame({"value": [2]}), NEVER)},
        ["source"],
        c("source") == "missing",
        pl.DataFrame(schema={"value": pl.Int64, "source": pl.String}),
    ),
    P(
        "concat_named_multiple_unique_columns",
        {("east", "A"): (AB[0], None), ("east", "B"): (AB[1], NEVER), ("west", "A"): (AB[2], NEVER), ("west", "B"): (AB[3], NEVER)},
        ["region", "type"],
        (c("region") == "east") & (c("type") == "A"),
        {"a": [1, 2], "b": [11, 12], "region": ["east"] * 2, "type": ["A"] * 2},
    ),
    P(
        "concat_named_different_data_types",
        {(D1, 100): (AB[0], None), (D2, 200): (AB[1], NEVER)},
        ["date", ("id", pl.Float64)],
        c("date") == D1,
        {"a": [1, 2], "b": [11, 12], "date": [D1, D1], "id": [100.0, 100.0]},
    ),
    P(
        "concat_named_or_condition",
        {("foo",): (AB[0], SOME), ("bar",): (AB[1], SOME), ("baz",): (AB[2], NEVER)},
        ["source"],
        ((c("source") == "foo") | (c("source") == "bar")) & (c("a") > 1),
        {"a": [2, 3, 4], "b": [12, 13, 14], "source": ["foo", "bar", "bar"]},
    ),
    P(
        "concat_named_is_in_filter",
        {("foo",): (AB[0], None), ("bar",): (AB[1], None), ("baz",): (AB[2], NEVER)},
        ["source"],
        c("source").is_in(["foo", "bar"]),
        {"a": [1, 2, 3, 4], "b": [11, 12, 13, 14], "source": ["foo", "foo", "bar", "bar"]},
    ),
    # A filter on a column the sources share reads every source.
    P(
        "concat_named_filter_non_unique_column",
        {("foo",): (ABC[0], ANY), ("bar",): (ABC[1], ANY)},
        ["source"],
        c("a") == 1,
        {"a": [1], "b": [4], "source": ["foo"]},
    ),
    P(
        "concat_named_complex_predicate",
        {("foo",): (ABC[0], ANY), ("bar",): (ABC[1].with_columns(a=pl.Series([1, 8, 9])), NEVER)},
        ["source"],
        (c("source") == "foo") & (c("a") >= 2),
        {"a": [2, 3], "b": [5, 6], "source": ["foo", "foo"]},
    ),
    P(
        "concat_named_with_column_selection",
        {("foo",): (AB[0].with_columns(c=pl.Series([5, 6])), None), ("bar",): (AB[1].with_columns(c=pl.Series([11, 12])), NEVER)},
        ["source"],
        c("source") == "foo",
        {"a": [1, 2], "c": [5, 6]},
        query=lambda lf: lf.select("a", "c"),
    ),
    P(
        "concat_named_filter_with_inequality",
        {(datetime.date(2025, 5, 1),): (AB[0], ANY), (datetime.date(2025, 5, 2),): (AB[1], NEVER)},
        ["source"],
        c("source") < datetime.date(2025, 5, 2),
        {"a": [1, 2], "b": [11, 12], "source": [datetime.date(2025, 5, 1)] * 2},
    ),
    P(
        "concat_named_python_dtypes_prune_multiple_sources",
        {
            ("rates", "1"): (pl.DataFrame({"value": [30]}), NEVER),
            ("rates", "2"): (pl.DataFrame({"value": [10, 20]}), ANY),
            ("credit", "2"): (pl.DataFrame({"value": [30]}), NEVER),
        },
        [("desk", str), ("bucket", int)],
        (c("desk") == "rates") & (c("bucket") == 2),
        {"value": [10, 20], "desk": ["rates", "rates"], "bucket": [2, 2]},
    ),
]


@pytest.mark.parametrize(("pred", "expect", "kw"), params(PRUNING, "pred", "expect", "kw"))
def test_concat_named_pruning(pred, expect, kw):
    seen = {key: [] for key in kw["sources"]}
    lf = cpl.concat_named({key: io_source_assert(df, seen[key].append) for key, (df, _) in kw["sources"].items()}, kw["names"])
    lf = lf if pred is None else lf.filter(pred)
    assert_frame_equal((kw["query"] or (lambda frame: frame))(lf).collect(), expect)
    for key, (_, want) in kw["sources"].items():
        if want is NEVER:
            assert not seen[key], f"{key} was read"
            continue
        assert seen[key], f"{key} was not read"
        if want is not ANY:
            assert all((p is not None) == (want is SOME) for p in seen[key]), (key, seen[key])


class Physical(int):
    """An identifier's expected value as the physical integer of its temporal dtype."""


def I(group, identifier, dtype, value, polars_dtype=None):
    return (group, identifier, dtype, value, polars_dtype or dtype)


_IDENTIFIERS = [
    I("concat_named_temporal_string_identifiers", "2023-01-01", pl.Date, datetime.date(2023, 1, 1)),
    I("concat_named_temporal_string_identifiers", "2023-01-01T12:30:00", pl.Datetime("ms"), datetime.datetime(2023, 1, 1, 12, 30)),
    I("concat_named_temporal_string_identifiers", "12:30:00", pl.Time, datetime.time(12, 30)),
    I("concat_named_python_dtype_identifiers", "label", str, "label", pl.String),
    I("concat_named_python_dtype_identifiers", "42", int, 42, pl.Int64),
    I("concat_named_python_dtype_identifiers", "3.5", float, 3.5, pl.Float64),
    I("concat_named_python_dtype_identifiers", "2023-01-01", datetime.date, datetime.date(2023, 1, 1), pl.Date),
    I("concat_named_python_dtype_identifiers", "2023-01-01T12:30:00", datetime.datetime, datetime.datetime(2023, 1, 1, 12, 30), pl.Datetime("us")),
    I("concat_named_python_dtype_identifiers", "12:30:00", datetime.time, datetime.time(12, 30), pl.Time),
    I("concat_named_python_dtype_identifiers", 42, int, 42, pl.Int64),
    I("concat_named_python_dtype_identifiers", True, bool, True, pl.Boolean),
    I("concat_named_python_dtype_identifiers", datetime.datetime(2023, 1, 1), pl.Datetime, datetime.datetime(2023, 1, 1), pl.Datetime("us")),
    I("concat_named_python_dtype_identifiers", "2023-01-01T12:30:00", pl.Datetime, datetime.datetime(2023, 1, 1, 12, 30), pl.Datetime("us")),
    I("concat_named_python_dtype_identifiers", datetime.timedelta(days=1), pl.Duration, datetime.timedelta(days=1), pl.Duration("us")),
    I("concat_named_python_dtype_identifiers", datetime.timedelta(days=1), datetime.timedelta, datetime.timedelta(days=1), pl.Duration("us")),
    I("concat_named_python_dtype_identifiers", "rates", pl.Categorical, "rates", pl.Categorical()),
    I("concat_named_temporal_identifier_precision", 1672531200000, pl.Datetime("ms"), Physical(1672531200000)),
    I("concat_named_temporal_identifier_precision", "2023-01-01T12:30:00.123456789", pl.Datetime("ns"), Physical(1672576200123456789)),
    I("concat_named_temporal_identifier_precision", "2023-01-01T12:30:00", pl.Datetime("ms", "America/New_York"), Physical(1672594200000)),
    # Offsets, padding and sub-microsecond digits in temporal strings.
    *[
        I("concat_named_legacy_temporal_strings", s, pl.Datetime("us"), Physical(1672576200000000))
        for s in [
            "2023-01-01T12:30:00Z",
            "2023-01-01T12:30:00+05:30",
            "2023-01-01T12:30+05:30",
            "2023-01-01T12:30:00+05",
            "2023-01-01T12:30:00+05:30:15",
            "2023-01-01T12:30:00 +05:30",
        ]
    ],
    I("concat_named_legacy_temporal_strings", "2023-01-01T12:30:00.123456789-05:30", pl.Datetime("ns"), Physical(1672576200123456789)),
    I("concat_named_legacy_temporal_strings", " 2023-01-01 ", pl.Date, Physical(19358)),
    I("concat_named_legacy_temporal_strings", " 12:30:00 ", pl.Time, Physical(45000000000000)),
    I("concat_named_legacy_temporal_strings", " 2023-01-01T12:30:00 ", pl.Datetime("ms"), Physical(1672576200000)),
    I("concat_named_legacy_temporal_strings", "2023-01-01T12:30:00+05:30", pl.Datetime("us", "America/New_York"), Physical(1672556400000000)),
    *[
        I("concat_named_accepts_nanosecond_boundaries", s, pl.Datetime("ns"), Physical(n))
        for s, n in [
            ("1677-09-21T00:12:43.145224192", -9223372036854775808),
            ("2262-04-11T23:47:16.854775807", 9223372036854775807),
            ("1677-09-21T00:12:43.145225192", -9223372036854774808),
            ("1969-12-31T23:59:59.999999999", -1),
        ]
    ],
]
IDENTIFIERS = [pytest.param(*row[1:], id=f"{row[0]}-{i}") for i, row in enumerate(_IDENTIFIERS)]


@pytest.mark.parametrize(("identifier", "dtype", "value", "polars_dtype"), IDENTIFIERS)
def test_concat_named_identifier(identifier, dtype, value, polars_dtype):
    lf = cpl.concat_named({(identifier,): pl.LazyFrame({"value": [1, 2]})}, [("identifier", dtype)])
    if isinstance(value, Physical):
        result = lf.filter(c("identifier").cast(pl.Int64) == value).collect()
        assert result["value"].to_list() == [1, 2]
        assert result.schema["identifier"] == polars_dtype
        assert result["identifier"].cast(pl.Int64).to_list() == [value, value]
    else:
        result = lf.filter(c("identifier") == value).collect()
        assert_frame_equal(result, pl.DataFrame({"value": [1, 2], "identifier": pl.Series([value, value], dtype=polars_dtype)}))


ERRORS = [
    *[
        Case(
            f"concat_named_unspecified_decimal_does_not_round-{dtype}",
            {(decimal.Decimal("1.5"),): None},
            TypeError,
            {"names": [("identifier", dtype)]},
        )
        for dtype in (pl.Decimal, decimal.Decimal)
    ],
    *[
        Case(
            f"concat_named_rejects_out_of_range_nanoseconds-{dtype}-{s}",
            {(s,): None},
            pl.exceptions.InvalidOperationError,
            {"names": [("identifier", dtype)], "match": "nanosecond.*range"},
        )
        for dtype in (pl.Datetime("ns"), pl.Datetime("ns", "UTC"))
        for s in ["1600-01-01T00:00:00", "1600-01-01T00:00:00+05:30", "2500-01-01T00:00:00Z"]
    ],
    Case("concat_named_empty_dict", {}, ValueError, {"names": ["source"]}),
    Case(
        "concat_named_mismatched_columns_keys",
        {("foo",): None, ("bar",): None},
        ValueError,
        {"names": ["source", "type"], "match": "Number of unique columns 2 does not match number of keys 1"},
    ),
]


@pytest.mark.parametrize(("keys", "error", "kw"), params(ERRORS, "pred", "expect", "kw"))
def test_concat_named_error(keys, error, kw):
    with pytest.raises(error, match=kw.get("match")):
        cpl.concat_named({key: pl.LazyFrame({"value": [1]}) for key in keys}, kw["names"]).collect()


@pytest.mark.parametrize(
    ("lf_dict", "kwargs", "pred"),
    [
        # 2**31 - 1 is an Int32 literal, so ``k + 1`` wraps in the data and must wrap in the pruning index too.
        ({(2**31 - 1,): pl.LazyFrame({"v": [42]})}, {}, (c("k") + 1) < 0),
        ({("a",): pl.LazyFrame({"x": [1]}), ("b",): pl.LazyFrame({"y": [2]})}, {"how": "diagonal"}, c("y") == 2),
        ({("a",): pl.LazyFrame({"x": [1]}), ("b",): pl.LazyFrame({"y": [2]})}, {"how": "diagonal"}, c("k") == "b"),
        ({("a",): pl.LazyFrame({"x": [1]}), ("b",): pl.LazyFrame({"x": [2.5]})}, {"how": "vertical_relaxed"}, c("k") == "a"),
    ],
)
def test_concat_named_pruning_matches_concat(lf_dict, kwargs, pred):
    expected = pl.concat([lf.with_columns(k=pl.lit(key[0])) for key, lf in lf_dict.items()], **kwargs).filter(pred).collect()
    assert_frame_equal(cpl.concat_named(lf_dict, ["k"], **kwargs).filter(pred).collect(), expected)
