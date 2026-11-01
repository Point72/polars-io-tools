"""from_narwhals and the Polars-to-Narwhals predicate translation.

Frame types are not tested one by one: Narwhals does the conversion. The main table is (native source, predicate):
the reader filtered by the predicate equals Polars filtering the same data.
"""

import re
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
from polars_io_tools.tests.helpers.assertions import assert_matches_native, assert_text
from polars_io_tools.tests.helpers.cases import Case, params

pd = pytest.importorskip("pandas", exc_type=ImportError)

DATA = {"a": [1, 2, 3, 4, 5, 6], "b": [10, 20, 30, 40, 50, 60], "cat": ["x", "y", "x", "y", "x", "y"]}


def transformation(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter(pl.col("a") > 3).select("cat", "b").group_by("cat").agg(pl.col("b").sum().alias("b_sum")).sort("cat")


@pytest.mark.parametrize(
    ("polars_expr", "narwhals_expr"),
    [
        (pl.col("a").is_null(), nw.col("a").is_null()),
        (pl.col("a").is_not_null(), ~nw.col("a").is_null()),
        (pl.col("a").is_in([1, 2, 3]), nw.col("a").is_in([1, 2, 3])),
    ],
)
def test_narwhals_expression_creation(polars_expr: pl.Expr, narwhals_expr: nw.Expr):
    assert str(polars_to_nw(polars_expr)) == str(narwhals_expr)


def test_non_narwhals_input():
    with pytest.raises(TypeError, match="Expected a Narwhals DataFrame or LazyFrame"):
        cpl.from_narwhals(pl.DataFrame(DATA))


@pytest.mark.parametrize("native", [pl.DataFrame(DATA), pl.LazyFrame(DATA)], ids=["polars_shortcut-eager", "polars_shortcut-lazy"])
def test_polars_shortcut(native):
    """A Polars-backed frame comes back as the same kind of Polars frame."""
    nw_frame = nw.from_native(native)
    assert nw_frame.implementation == nw.Implementation.POLARS
    returned = cpl.from_narwhals(nw_frame)
    assert type(returned) is type(native)
    assert_frame_equal(returned, native)


def _lazy(native, version=nw):
    return cpl.from_narwhals(version.from_native(native).lazy())


def _pickled(lf):
    return cloudpickle.loads(cloudpickle.dumps(lf))


# (native source, how it becomes a Polars LazyFrame, eager result expected) -> the transformation equals Polars'.
CONVERSIONS = [
    Case("from_narwhals_dataframe", pd.DataFrame(DATA), True, {"convert": lambda src: cpl.from_narwhals(nw.from_native(src))}),
    Case("from_narwhals_lazyframe", pd.DataFrame(DATA), False, {"convert": _lazy}),
    Case("from_narwhals_stable_lazyframe", pd.DataFrame(DATA), False, {"convert": lambda src: _lazy(src, nw_stable)}),
    Case("from_narwhals_pickle_pandas_backed", pd.DataFrame(DATA), False, {"convert": lambda src: _pickled(_lazy(src))}),
    Case(
        "from_narwhals_pickle_pandas_dataframe",
        pd.DataFrame(DATA),
        False,
        {"convert": lambda src: _pickled(cpl.from_narwhals(nw.from_native(src)).lazy())},
    ),
    Case(
        "from_narwhals_pickle_pyarrow_backed", pa.table(DATA), False, {"convert": lambda src: _pickled(cpl.from_narwhals(nw.from_native(src)).lazy())}
    ),
    Case("from_narwhals_pickle_with_filter", pd.DataFrame(DATA), False, {"convert": lambda src: _pickled(_lazy(src).filter(pl.col("a") > 3))}),
]


@pytest.mark.parametrize(("source", "eager", "convert"), params(CONVERSIONS, "pred", "expect", "convert"))
def test_conversion(source, eager, convert):
    frame = convert(source)
    assert isinstance(frame, pl.DataFrame if eager else pl.LazyFrame)
    for _ in range(2):  # collecting again gives the same result
        assert_frame_equal(transformation(frame.lazy()).collect(), transformation(pl.LazyFrame(DATA)).collect())


_PUSHDOWN_DATA = pl.DataFrame(
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
    "pandas": lambda: _PUSHDOWN_DATA.to_pandas(),
    "pandas_pyarrow": lambda: _PUSHDOWN_DATA.to_pandas(use_pyarrow_extension_array=True),
    "pyarrow": lambda: _PUSHDOWN_DATA.to_arrow(),
    "dask": lambda: pytest.importorskip("dask.dataframe").from_pandas(_PUSHDOWN_DATA.to_pandas(), npartitions=2),
}
a, b, f, n, x = pl.col("a"), pl.col("b"), pl.col("f"), pl.col("n"), pl.col("x")
_NS500 = pl.lit(500).cast(pl.Datetime("ns"))
_PUSHDOWN_PREDICATES = [
    a > pl.when(b > 2).then(b).otherwise(0),
    (a > 4) | pl.when(b > 2).then(True).otherwise(False),
    a > b.map_elements(lambda v: v - 10, return_dtype=pl.Int64),
    n.cast(pl.Boolean) == True,
    pl.col("ts").cast(pl.Datetime("ms")) == datetime(2020, 1, 1, 0, 0, 0, 123000),
    pl.col("dur").cast(pl.Duration("ms")) == timedelta(milliseconds=1),
    a.cast(pl.Int128) > b.cast(pl.Int128),
    a.cast(pl.Int128) > 5,
    a.cast(pl.Int8, strict=False).is_null(),
    pl.col("s").cast(pl.Int64, strict=False) >= 3,
    f.cast(pl.Int64, strict=False) > 0,
    (n > 2).is_null(),
    (n > 2).ne_missing(False),
    n.is_in([3.0]).is_null(),
    n.is_in([3.0]).ne_missing(False),
    n.is_in([1.0], nulls_equal=True).eq_missing(False),
    f > 1,
    f >= 1,
    f <= pl.col("f"),
    (f < 1) == False,
    (f < 1).is_in([False]),
    pl.col("ns") <= _NS500,
    pl.col("ns").is_in(pl.concat_list(_NS500)),
    f == pl.lit(float("nan")),
    f.eq_missing(float("nan")),
    f.ne_missing(float("nan")),
    pl.col("s").ne_missing("x"),
    pl.col("ns") == 500,
    pl.col("d") == 1,
    pl.col("dur") == 1500,
    pl.col("u") > 2**63 - 1,
]

_TEST_DATA = {
    "id": [1, 2, 3, 4, 5, 6, 7, 8],
    "name": ["Alice", None, "Charlie", "David", None, "Frank", "Grace", None],
    "category": ["A", "B", "A", "C", "B", "C", "A", "B"],
    "score": [85.5, 92.0, None, 78.0, 88.5, None, 95.0, 82.0],
    "active": [True, False, None, True, False, True, None, False],
}


def _from_native(src):
    return pl.from_arrow(src) if isinstance(src, pa.Table) else pl.from_pandas(src)


def F(id, source, pred, ref=None, **kw):
    """A row: the reader over ``source()`` filtered by ``pred`` equals ``ref(source, reader)`` filtered by ``pred``.

    By default the reference is Polars' own conversion of the native source.
    """
    return Case(id, pred, kw={"source": source, "ref": ref or (lambda src, lf: _from_native(src).lazy()), **kw})


SOURCES = [
    # the reference is the reader's own unfiltered result: the conversion may legitimately differ from Polars'
    *[
        F(f"from_narwhals_predicate_pushdown_matches_polars-{backend}-{i}", _BACKENDS[backend], p, ref=lambda src, lf: lf.collect().lazy())
        for i, p in enumerate(_PUSHDOWN_PREDICATES)
        for backend in _BACKENDS
    ],
    *[
        F(f"from_narwhals_native_source_matches_polars-{i}", lambda src=src: src, p)
        for i, (src, p) in enumerate(
            [
                (pl.DataFrame({"x": pl.Series(["z", "a"]).cast(pl.Enum(["z", "a"]))}).to_arrow(), x > "z"),
                (pd.DataFrame({"x": pd.Categorical(["lo", "hi"], categories=["lo", "hi"], ordered=True)}), x.is_in(["hi"])),
                (pl.DataFrame({"x": [datetime(2020, 1, 1, tzinfo=UTC)]}).to_pandas(), x.is_in([datetime(2020, 1, 1)])),
                (pd.DataFrame({"x": [date(2020, 1, 1), None]}), x == date(2020, 1, 2)),
                (pd.DataFrame({"x": pd.date_range("2020-01-01", periods=2, tz="America/New_York")}), x > date(2020, 1, 1)),
                (pa.table({"x": pa.array([datetime(2020, 1, 1), datetime(2020, 1, 2)], pa.timestamp("s"))}), x > datetime(2020, 1, 1)),
                (pa.table({"x": pa.array([[date(2020, 1, 1)], []], pa.list_(pa.date64()))}), x.list.len() > 0),
                (pd.DataFrame({"x": [True, False, None], "u": [1.0, 1.0, 1.0]}), x | (pl.col("u") > 0)),
                (pd.DataFrame({"x": [True, None], "u": [1, 2]}), x.is_null()),
                (pd.DataFrame({"x": [None, date(2020, 1, 1)]}), x == date(2020, 1, 1)),
                (pd.DataFrame({"x": [None, date(2020, 1, 1)], "y": [date(2020, 1, 1), None]}), x.is_not_null()),
                (pd.DataFrame({"x": [[], [1]], "u": [1, 2]}), pl.col("u") == 2),
                (pd.DataFrame({"x": [Decimal("1.5"), Decimal("2.25")], "u": [1, 2]}), pl.col("u") == 1),
            ]
        )
    ],
    *[
        F(f"from_narwhals_polars_predicate_pushdown-{name}", lambda: pd.DataFrame(_TEST_DATA), p, eager=True)
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
    F("from_narwhals_with_predicate_pushdown", lambda: pd.DataFrame(DATA), a > 3),
    # a pandas filter leaves a non-RangeIndex behind
    F("from_narwhals_filtered_pandas_index_is_not_a_column", lambda: pd.DataFrame({"id": [0, 1, 2, 3], "x": [0, 1, 2, 3]}), x != 1),
    F(
        "from_narwhals_custom_batch_size",
        lambda: pa.table({"a": list(range(1000)), "b": list(range(1000, 2000)), "cat": ["x", "y"] * 500}),
        a < 10,
        fetch_size=10,
    ),
]


@pytest.mark.parametrize(
    ("pred", "source", "ref", "kw"),
    params([Case(c.id, c.pred, kw={k: c.kw[k] for k in ("source", "ref")} | {"kw": c.kw}) for c in SOURCES], "pred", "source", "ref", "kw"),
)
def test_from_narwhals_matches_polars(pred, source, ref, kw):
    src = source()
    if kw.get("eager"):
        lf = cpl.from_narwhals(nw.from_native(src)).lazy()
    else:
        lf = cpl.from_narwhals(nw.from_native(src).lazy(), **({"fetch_size": kw["fetch_size"]} if "fetch_size" in kw else {}))
    assert_matches_native(lf, ref(src, lf), pred, check_row_order=True)


@pytest.mark.parametrize(
    "pred",
    [
        x.ne_missing(1),
        x.eq_missing(pl.col("y")),
        x.ne_missing(pl.col("y")),
        x.is_in([None, 1], nulls_equal=True),
        x.is_in(None, nulls_equal=True),
        x.is_in(pl.lit(1, dtype=pl.Int64), nulls_equal=True),
    ],
)
def test_from_narwhals_null_aware_predicate_matches_polars(pred: pl.Expr):
    """The translated Narwhals predicate keeps the rows Polars keeps, and so does the reader."""
    source = pa.table({"id": [0, 1, 2, 3], "x": [None, 1, 2, None], "y": [None, 1, None, 2]})
    expected = pl.from_arrow(source).filter(pred)
    translated = polars_to_nw(pred, expected.schema)
    assert translated is not None
    assert nw.from_native(source).lazy().filter(translated).collect().to_arrow()["id"].to_pylist() == expected["id"].to_list()
    assert_matches_native(_lazy(source), pl.from_arrow(source).lazy(), pred, check_row_order=True)


def test_from_narwhals_unconvertible_object_column():
    lf = _lazy(pd.DataFrame({"x": pd.Series([1, "a"], dtype=object), "u": [0, 1]}))
    assert_frame_equal(lf.select("u").filter(pl.col("u") > 0).collect(), pl.DataFrame({"u": [1]}))


def test_from_narwhals_duckdb_hugeint():
    duckdb = pytest.importorskip("duckdb")
    pytest.importorskip("narwhals._duckdb.utils", reason="narwhals does not support this duckdb version")
    rel = duckdb.sql("SELECT 1::HUGEINT AS x UNION ALL SELECT 2")
    assert_matches_native(_lazy(rel), rel.pl().lazy(), x > 1)


_SOURCE_DF = pl.DataFrame(
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
    tracker = PredicateTracker(_SOURCE_DF)
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
    assert_text(None if tracker.last_predicate is None else str(tracker.last_predicate), "original_col", omits="renamed_col")


def test_sql_lazy_frame():
    """A predicate through from_narwhals over a Polars source still translates to SQL."""
    tracker = PredicateTracker(_SOURCE_DF)
    pred = pl.col("a").cast(pl.Float64) > 1
    assert_matches_native(cpl.from_narwhals(nw.from_native(tracker.lazy_frame)), _SOURCE_DF.lazy(), pred, check_row_order=True)
    assert convert_predicate_to_sql(tracker.last_predicate, dialect="mssql") is not None


@pytest.mark.parametrize(
    ("plain", "same", "omits"),
    [
        pytest.param(pl.col("data_date") >= date(2025, 7, 7), True, (), id="boolean_cast_is_removed_from_sql-comparison"),
        pytest.param(
            pl.col("TradeDate").is_between(date(2025, 7, 7), date(2025, 7, 10)), False, ("CAST",), id="boolean_cast_is_removed_from_sql-between"
        ),
    ],
)
def test_boolean_cast_is_removed_from_sql(plain, same, omits):
    """A Boolean cast around a predicate (as Narwhals adds) translates like the bare predicate."""

    def sql(pred):
        return re.sub(r"\[([^\]]+)]", r"\1", convert_predicate_to_sql(pred, dialect="tsql").sql(dialect="tsql"))

    assert_text(sql(plain.cast(pl.Boolean)), equals=sql(plain) if same else None, omits=omits)
