"""from_narwhals and the Polars-to-Narwhals predicate translation.

Frame types are not tested one by one: Narwhals does the conversion. In the main table each row is a native source
(pandas, pyarrow, ...) and a filter; the reader filtered by it must return what Polars returns on the same data.
"""

import re
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import cloudpickle
import narwhals as nw
import narwhals.stable.v1 as nw_stable
import polars as pl
import portion
import pyarrow as pa
import pytest
from polars.testing import assert_frame_equal

import polars_io_tools as cpl
from polars_io_tools.io_sources.lazy_narwhals_reader import polars_to_nw
from polars_io_tools.io_sources.range_visitor import convert_expr_to_datetime_range
from polars_io_tools.io_sources.set_visitor import convert_expr_to_valid_values
from polars_io_tools.io_sources.sql_utils import convert_predicate_to_sql
from polars_io_tools.testing import PredicateTracker
from polars_io_tools.tests.helpers.assertions import assert_matches_plain_polars, assert_translated_text
from polars_io_tools.tests.helpers.cases import as_params
from polars_io_tools.tests.helpers.pickling import pickle_round_trip

pd = pytest.importorskip("pandas", exc_type=ImportError)

THREE_COLUMN_DATA = {"a": [1, 2, 3, 4, 5, 6], "b": [10, 20, 30, 40, 50, 60], "cat": ["x", "y", "x", "y", "x", "y"]}


def filter_group_and_sum(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter(pl.col("a") > 3).select("cat", "b").group_by("cat").agg(pl.col("b").sum().alias("b_sum")).sort("cat")


@dataclass(frozen=True)
class NarwhalsExpressionCase:
    """A Polars expression, and the Narwhals expression ``polars_to_nw`` makes of it."""

    id: str
    _: KW_ONLY
    polars_expr: pl.Expr
    narwhals_expr: nw.Expr


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            NarwhalsExpressionCase("is_null", polars_expr=pl.col("a").is_null(), narwhals_expr=nw.col("a").is_null()),
            NarwhalsExpressionCase("is_not_null", polars_expr=pl.col("a").is_not_null(), narwhals_expr=~nw.col("a").is_null()),
            NarwhalsExpressionCase("is_in", polars_expr=pl.col("a").is_in([1, 2, 3]), narwhals_expr=nw.col("a").is_in([1, 2, 3])),
        ]
    ),
)
def test_narwhals_expression_creation(case: NarwhalsExpressionCase):
    assert str(polars_to_nw(case.polars_expr)) == str(case.narwhals_expr)


def test_non_narwhals_input():
    with pytest.raises(TypeError, match="Expected a Narwhals DataFrame or LazyFrame"):
        cpl.from_narwhals(pl.DataFrame(THREE_COLUMN_DATA))


@pytest.mark.parametrize(
    "native", [pl.DataFrame(THREE_COLUMN_DATA), pl.LazyFrame(THREE_COLUMN_DATA)], ids=["polars_shortcut-eager", "polars_shortcut-lazy"]
)
def test_polars_shortcut(native):
    """A Polars-backed frame comes back as the same kind of Polars frame."""
    nw_frame = nw.from_native(native)
    assert nw_frame.implementation == nw.Implementation.POLARS
    returned = cpl.from_narwhals(nw_frame)
    assert type(returned) is type(native)
    assert_frame_equal(returned, native)


def _read_lazily(native, version=nw):
    return cpl.from_narwhals(version.from_native(native).lazy())


def _pickle_round_trip(lf):
    return cloudpickle.loads(cloudpickle.dumps(lf))


@dataclass(frozen=True)
class ConversionCase:
    """A native source, how it becomes a Polars frame, and whether that frame is eager; the same query must give the
    same result as on Polars."""

    id: str
    _: KW_ONLY
    source: object
    eager: bool
    convert: Callable[[object], pl.DataFrame | pl.LazyFrame]


CONVERSIONS = [
    ConversionCase(
        "from_narwhals_dataframe", source=pd.DataFrame(THREE_COLUMN_DATA), eager=True, convert=lambda src: cpl.from_narwhals(nw.from_native(src))
    ),
    ConversionCase("from_narwhals_lazyframe", source=pd.DataFrame(THREE_COLUMN_DATA), eager=False, convert=_read_lazily),
    ConversionCase(
        "from_narwhals_stable_lazyframe", source=pd.DataFrame(THREE_COLUMN_DATA), eager=False, convert=lambda src: _read_lazily(src, nw_stable)
    ),
    ConversionCase(
        "from_narwhals_pickle_pandas_backed",
        source=pd.DataFrame(THREE_COLUMN_DATA),
        eager=False,
        convert=lambda src: _pickle_round_trip(_read_lazily(src)),
    ),
    ConversionCase(
        "from_narwhals_pickle_pandas_dataframe",
        source=pd.DataFrame(THREE_COLUMN_DATA),
        eager=False,
        convert=lambda src: _pickle_round_trip(cpl.from_narwhals(nw.from_native(src)).lazy()),
    ),
    ConversionCase(
        "from_narwhals_pickle_pyarrow_backed",
        source=pa.table(THREE_COLUMN_DATA),
        eager=False,
        convert=lambda src: _pickle_round_trip(cpl.from_narwhals(nw.from_native(src)).lazy()),
    ),
    ConversionCase(
        "from_narwhals_pickle_with_filter",
        source=pd.DataFrame(THREE_COLUMN_DATA),
        eager=False,
        convert=lambda src: _pickle_round_trip(_read_lazily(src).filter(pl.col("a") > 3)),
    ),
]


@pytest.mark.parametrize("case", as_params(CONVERSIONS))
def test_conversion(case: ConversionCase):
    frame = case.convert(case.source)
    assert isinstance(frame, pl.DataFrame if case.eager else pl.LazyFrame)
    for _ in range(2):  # collecting again gives the same result
        assert_frame_equal(filter_group_and_sum(frame.lazy()).collect(), filter_group_and_sum(pl.LazyFrame(THREE_COLUMN_DATA)).collect())


_MANY_DTYPES = pl.DataFrame(
    {
        "id": [0, 1, 2, 3, 4],
        "a": [1, 2, 3, 10, 1000],
        "b": [5, 4, 3, 9, 1],
        "n": [1.0, None, 3.0, 0.0, 2.0],
        "f": [1.5, float("nan"), 0.5, None, 2.0],
        "s": ["x", "y", None, "x", "3"],
        "ts": [datetime(2020, 1, 1, 0, 0, 0, 123456)] * 5,
        "dur": [timedelta(microseconds=1500)] * 5,
        "ns": pl.Series([0, 500, 1000, 1500, None]).cast(pl.Datetime("ns")),
        "d": pl.Series([0, 1, 2, 3, None]).cast(pl.Date),
        "u": pl.Series([2**63, None, 1, 2, 3], dtype=pl.UInt64),
    }
)
_BACKENDS = {
    "pandas": lambda: _MANY_DTYPES.to_pandas(),
    "pandas_pyarrow": lambda: _MANY_DTYPES.to_pandas(use_pyarrow_extension_array=True),
    "pyarrow": lambda: _MANY_DTYPES.to_arrow(),
    "dask": lambda: pytest.importorskip("dask.dataframe").from_pandas(_MANY_DTYPES.to_pandas(), npartitions=2),
}
_500_NANOSECONDS = pl.lit(500).cast(pl.Datetime("ns"))
# Filters that each native source must answer as Polars does.
_PUSHDOWN_PREDICATES = {
    "gt_when_then_otherwise": pl.col("a") > pl.when(pl.col("b") > 2).then(pl.col("b")).otherwise(0),
    "or_with_when_boolean": (pl.col("a") > 4) | pl.when(pl.col("b") > 2).then(True).otherwise(False),
    "gt_map_elements": pl.col("a") > pl.col("b").map_elements(lambda v: v - 10, return_dtype=pl.Int64),
    "float_cast_to_boolean_eq_true": pl.col("n").cast(pl.Boolean) == True,
    "datetime_cast_to_ms_eq": pl.col("ts").cast(pl.Datetime("ms")) == datetime(2020, 1, 1, 0, 0, 0, 123000),
    "duration_cast_to_ms_eq": pl.col("dur").cast(pl.Duration("ms")) == timedelta(milliseconds=1),
    "int128_column_comparison": pl.col("a").cast(pl.Int128) > pl.col("b").cast(pl.Int128),
    "int128_gt_literal": pl.col("a").cast(pl.Int128) > 5,
    "int8_non_strict_cast_is_null": pl.col("a").cast(pl.Int8, strict=False).is_null(),
    "string_non_strict_cast_to_int_ge": pl.col("s").cast(pl.Int64, strict=False) >= 3,
    "float_non_strict_cast_to_int_gt": pl.col("f").cast(pl.Int64, strict=False) > 0,
    "comparison_is_null": (pl.col("n") > 2).is_null(),
    "comparison_ne_missing_false": (pl.col("n") > 2).ne_missing(False),
    "is_in_is_null": pl.col("n").is_in([3.0]).is_null(),
    "is_in_ne_missing_false": pl.col("n").is_in([3.0]).ne_missing(False),
    "is_in_nulls_equal_eq_missing_false": pl.col("n").is_in([1.0], nulls_equal=True).eq_missing(False),
    "float_gt": pl.col("f") > 1,
    "float_ge": pl.col("f") >= 1,
    "float_le_itself": pl.col("f") <= pl.col("f"),
    "lt_eq_false": (pl.col("f") < 1) == False,
    "lt_is_in_false": (pl.col("f") < 1).is_in([False]),
    "ns_le_500ns": pl.col("ns") <= _500_NANOSECONDS,
    "ns_is_in_concat_list": pl.col("ns").is_in(pl.concat_list(_500_NANOSECONDS)),
    "float_eq_nan": pl.col("f") == pl.lit(float("nan")),
    "float_eq_missing_nan": pl.col("f").eq_missing(float("nan")),
    "float_ne_missing_nan": pl.col("f").ne_missing(float("nan")),
    "string_ne_missing": pl.col("s").ne_missing("x"),
    "ns_eq_integer": pl.col("ns") == 500,
    "date_eq_integer": pl.col("d") == 1,
    "duration_eq_integer": pl.col("dur") == 1500,
    "uint64_gt_int64_max": pl.col("u") > 2**63 - 1,
}

_PEOPLE_WITH_NULLS = {
    "id": [1, 2, 3, 4, 5, 6, 7, 8],
    "name": ["Alice", None, "Charlie", "David", None, "Frank", "Grace", None],
    "category": ["A", "B", "A", "C", "B", "C", "A", "B"],
    "score": [85.5, 92.0, None, 78.0, 88.5, None, 95.0, 82.0],
    "active": [True, False, None, True, False, True, None, False],
}


def _polars_conversion(src, lf: pl.LazyFrame) -> pl.LazyFrame:
    """Polars' own conversion of the native source."""
    return (pl.from_arrow(src) if isinstance(src, pa.Table) else pl.from_pandas(src)).lazy()


@dataclass(frozen=True)
class NarwhalsReaderCase:
    """A native source and a filter; ``from_narwhals`` must return what the reference returns for the filter."""

    id: str
    _: KW_ONLY
    # makes the native source
    source: Callable[[], object]
    predicate: pl.Expr
    # receives the native source and the reader's frame, and returns the frame to compare with
    reference: Callable[[object, pl.LazyFrame], pl.LazyFrame] = _polars_conversion
    # read through an eager Narwhals frame
    eager: bool = False
    # passed to the reader if given
    fetch_size: int | None = None


SOURCES = [
    # the reference is the reader's own unfiltered result: the conversion may legitimately differ from Polars'
    *[
        NarwhalsReaderCase(
            f"from_narwhals_predicate_pushdown_matches_polars-{backend}-{name}",
            source=_BACKENDS[backend],
            predicate=p,
            reference=lambda src, lf: lf.collect().lazy(),
        )
        for name, p in _PUSHDOWN_PREDICATES.items()
        for backend in _BACKENDS
    ],
    *[
        NarwhalsReaderCase(f"from_narwhals_native_source_matches_polars-{name}", source=lambda src=src: src, predicate=p)
        for name, src, p in [
            ("enum_gt", pl.DataFrame({"x": pl.Series(["z", "a"]).cast(pl.Enum(["z", "a"]))}).to_arrow(), pl.col("x") > "z"),
            (
                "ordered_categorical_is_in",
                pd.DataFrame({"x": pd.Categorical(["lo", "hi"], categories=["lo", "hi"], ordered=True)}),
                pl.col("x").is_in(["hi"]),
            ),
            (
                "utc_datetime_is_in_naive",
                pl.DataFrame({"x": [datetime(2020, 1, 1, tzinfo=UTC)]}).to_pandas(),
                pl.col("x").is_in([datetime(2020, 1, 1)]),
            ),
            ("nullable_date_eq_missing_day", pd.DataFrame({"x": [date(2020, 1, 1), None]}), pl.col("x") == date(2020, 1, 2)),
            (
                "new_york_datetime_gt_date",
                pd.DataFrame({"x": pd.date_range("2020-01-01", periods=2, tz="America/New_York")}),
                pl.col("x") > date(2020, 1, 1),
            ),
            (
                "second_timestamp_gt",
                pa.table({"x": pa.array([datetime(2020, 1, 1), datetime(2020, 1, 2)], pa.timestamp("s"))}),
                pl.col("x") > datetime(2020, 1, 1),
            ),
            ("list_of_date64_len", pa.table({"x": pa.array([[date(2020, 1, 1)], []], pa.list_(pa.date64()))}), pl.col("x").list.len() > 0),
            ("nullable_bool_or_column", pd.DataFrame({"x": [True, False, None], "u": [1.0, 1.0, 1.0]}), pl.col("x") | (pl.col("u") > 0)),
            ("nullable_bool_is_null", pd.DataFrame({"x": [True, None], "u": [1, 2]}), pl.col("x").is_null()),
            ("null_then_date_eq", pd.DataFrame({"x": [None, date(2020, 1, 1)]}), pl.col("x") == date(2020, 1, 1)),
            ("date_columns_is_not_null", pd.DataFrame({"x": [None, date(2020, 1, 1)], "y": [date(2020, 1, 1), None]}), pl.col("x").is_not_null()),
            ("list_column_beside_filter", pd.DataFrame({"x": [[], [1]], "u": [1, 2]}), pl.col("u") == 2),
            ("decimal_column_beside_filter", pd.DataFrame({"x": [Decimal("1.5"), Decimal("2.25")], "u": [1, 2]}), pl.col("u") == 1),
        ]
    ],
    *[
        NarwhalsReaderCase(
            f"from_narwhals_polars_predicate_pushdown-{name}", source=lambda: pd.DataFrame(_PEOPLE_WITH_NULLS), predicate=p, eager=True
        )
        for name, p in [
            ("is_null", pl.col("name").is_null()),
            ("is_not_null", pl.col("score").is_not_null()),
            ("is_in", pl.col("category").is_in(["A", "C"])),
            (
                "combined",
                pl.col("name").is_not_null() & pl.col("category").is_in(["A", "C"]) & (pl.col("score").is_null() | (pl.col("score") > 80.0)),
            ),
            ("is_in_single", pl.col("category").is_in(["C"])),
        ]
    ],
    NarwhalsReaderCase("from_narwhals_with_predicate_pushdown", source=lambda: pd.DataFrame(THREE_COLUMN_DATA), predicate=pl.col("a") > 3),
    # a pandas filter leaves a non-RangeIndex behind
    NarwhalsReaderCase(
        "from_narwhals_filtered_pandas_index_is_not_a_column",
        source=lambda: pd.DataFrame({"id": [0, 1, 2, 3], "x": [0, 1, 2, 3]}),
        predicate=pl.col("x") != 1,
    ),
    NarwhalsReaderCase(
        "from_narwhals_custom_batch_size",
        source=lambda: pa.table({"a": list(range(1000)), "b": list(range(1000, 2000)), "cat": ["x", "y"] * 500}),
        predicate=pl.col("a") < 10,
        fetch_size=10,
    ),
]


@pytest.mark.parametrize("case", as_params(SOURCES))
def test_from_narwhals_matches_polars(case: NarwhalsReaderCase):
    src = case.source()
    if case.eager:
        lf = cpl.from_narwhals(nw.from_native(src)).lazy()
    else:
        lf = cpl.from_narwhals(nw.from_native(src).lazy(), **({"fetch_size": case.fetch_size} if case.fetch_size is not None else {}))
    assert_matches_plain_polars(lf, case.reference(src, lf), case.predicate, check_row_order=True)


@pytest.mark.parametrize(
    "pred",
    [
        pl.col("x").ne_missing(1),
        pl.col("x").eq_missing(pl.col("y")),
        pl.col("x").ne_missing(pl.col("y")),
        pl.col("x").is_in([None, 1], nulls_equal=True),
        pl.col("x").is_in(None, nulls_equal=True),
        pl.col("x").is_in(pl.lit(1, dtype=pl.Int64), nulls_equal=True),
    ],
    ids=["ne_missing_literal", "eq_missing_column", "ne_missing_column", "is_in_list_with_null", "is_in_null", "is_in_literal"],
)
def test_from_narwhals_null_aware_predicate_matches_polars(pred: pl.Expr):
    """The translated Narwhals predicate keeps the rows Polars keeps, and so does the reader."""
    source = pa.table({"id": [0, 1, 2, 3], "x": [None, 1, 2, None], "y": [None, 1, None, 2]})
    expected = pl.from_arrow(source).filter(pred)
    translated = polars_to_nw(pred, expected.schema)
    assert translated is not None
    assert nw.from_native(source).lazy().filter(translated).collect().to_arrow()["id"].to_pylist() == expected["id"].to_list()
    assert_matches_plain_polars(_read_lazily(source), pl.from_arrow(source).lazy(), pred, check_row_order=True)


def test_from_narwhals_unconvertible_object_column():
    lf = _read_lazily(pd.DataFrame({"x": pd.Series([1, "a"], dtype=object), "u": [0, 1]}))
    assert_frame_equal(lf.select("u").filter(pl.col("u") > 0).collect(), pl.DataFrame({"u": [1]}))


def test_from_narwhals_duckdb_hugeint():
    duckdb = pytest.importorskip("duckdb")
    pytest.importorskip("narwhals._duckdb.utils", reason="narwhals does not support this duckdb version")
    rel = duckdb.sql("SELECT 1::HUGEINT AS x UNION ALL SELECT 2")
    assert_matches_plain_polars(_read_lazily(rel), rel.pl().lazy(), pl.col("x") > 1)


_SIX_COLUMNS = pl.DataFrame(
    {
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": [True, False, None],
        "d": ["a", "b", "c"],
        "e": [1.1, 2.2, 3.3],
        "date": [date(2023, 10, 1), date(2023, 10, 2), date(2023, 10, 3)],
    }
)


def test_narwhals_wrapping_polars():
    """Narwhals filters on a Polars IO source reach it as a predicate the visitors can read."""
    tracker = PredicateTracker(_SIX_COLUMNS)
    res = (
        nw.from_native(tracker.lazy_frame)
        .filter(nw.col("a").is_in([1, 5, 2]) & ~nw.col("a").is_in([5, 2]))
        .filter(~(nw.col("a").is_in([4, 5]) | nw.col("date").is_between(date(2023, 10, 1), date(2023, 10, 8), closed="right")))
        .select("a", "c", "date")
        .collect()
        .to_polars()
    )
    assert_frame_equal(res, pl.DataFrame({"a": [1], "c": [True], "date": [date(2023, 10, 1)]}))
    assert convert_expr_to_valid_values(tracker.last_predicate, "a") == {1}
    expected = portion.closed(-portion.inf, datetime(2023, 10, 1)) | portion.openclosed(datetime(2023, 10, 8), portion.inf)
    assert convert_expr_to_datetime_range(tracker.last_predicate, "date", get_enclosure=False) == expected


def test_alias_filter_predicate_pushdown_to_original_column():
    """A Narwhals filter on an aliased column reaches the source on the original column."""
    tracker = PredicateTracker(pl.DataFrame({"original_col": [1, 2, 3, 4, 5], "other_col": ["a", "b", "c", "d", "e"]}))
    result = (
        nw.from_native(tracker.lazy_frame)
        .with_columns(nw.col("original_col").alias("renamed_col"))
        .select("renamed_col", "other_col")
        .filter(nw.col("renamed_col") > 2)
        .collect()
        .to_polars()
    )
    assert_frame_equal(result, pl.DataFrame({"renamed_col": [3, 4, 5], "other_col": ["c", "d", "e"]}))
    assert tracker.call_count == 1
    assert_translated_text(None if tracker.last_predicate is None else str(tracker.last_predicate), "original_col", omits="renamed_col")


def test_sql_lazy_frame():
    """A predicate through from_narwhals over a Polars source still translates to SQL."""
    tracker = PredicateTracker(_SIX_COLUMNS)
    pred = pl.col("a").cast(pl.Float64) > 1
    assert_matches_plain_polars(cpl.from_narwhals(nw.from_native(tracker.lazy_frame)), _SIX_COLUMNS.lazy(), pred, check_row_order=True)
    assert convert_predicate_to_sql(tracker.last_predicate, dialect="mssql") is not None


@dataclass(frozen=True)
class BooleanCastSqlCase:
    """A filter wrapped in a Boolean cast, and what its SQL must hold."""

    id: str
    _: KW_ONLY
    # the filter without the cast
    plain: pl.Expr
    # the SQL equals that of the filter without the cast
    same: bool
    omits: tuple[str, ...]


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            BooleanCastSqlCase("boolean_cast_is_removed_from_sql-comparison", plain=pl.col("data_date") >= date(2025, 7, 7), same=True, omits=()),
            BooleanCastSqlCase(
                "boolean_cast_is_removed_from_sql-between",
                plain=pl.col("TradeDate").is_between(date(2025, 7, 7), date(2025, 7, 10)),
                same=False,
                omits=("CAST",),
            ),
        ]
    ),
)
def test_boolean_cast_is_removed_from_sql(case: BooleanCastSqlCase):
    """A Boolean cast around a predicate (as Narwhals adds) translates like the bare predicate."""

    def sql(pred):
        return re.sub(r"\[([^\]]+)]", r"\1", convert_predicate_to_sql(pred, dialect="tsql").sql(dialect="tsql"))

    assert_translated_text(sql(case.plain.cast(pl.Boolean)), equals=sql(case.plain) if case.same else None, omits=case.omits)


class TestScanNarwhalsPickle:
    """Tests for scan_narwhals / from_narwhals pickle support."""

    def test_scan_narwhals_lazy_pickle_basic(self):
        """scan_narwhals LazyFrames (lazy backend) can be pickled and unpickled."""
        import narwhals as nw

        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        nw_lf = nw.from_native(df.lazy())

        lf = cpl.from_narwhals(nw_lf)
        assert isinstance(lf, pl.LazyFrame)
        lf_unpickled = pickle_round_trip(lf)

        expected = lf.collect()
        result = lf_unpickled.collect()
        assert_frame_equal(result, expected)

    def test_scan_narwhals_pickle_with_filter_and_select(self):
        """scan_narwhals LazyFrames with downstream ops can be pickled and unpickled."""
        import narwhals as nw

        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50], "c": ["x", "y", "x", "y", "x"]})
        nw_lf = nw.from_native(df.lazy())

        lf = cpl.from_narwhals(nw_lf).filter(pl.col("a") > 2).select(["a", "b"])
        lf_unpickled = pickle_round_trip(lf)

        expected = df.filter(pl.col("a") > 2).select(["a", "b"])
        result = lf_unpickled.collect()
        assert_frame_equal(result, expected)
