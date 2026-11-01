"""ts_with_columns: a filter on the index column or a linked column is widened by the lookback and lookahead before it
reaches the source, so window expressions see the rows they need, and the result is then trimmed to the filter.

The source sits behind a ``PredicateTracker``. Most rows forward-fill ``Value`` (non-null only on the first day) and
name the expected fill; a lookback that does not reach the first day leaves it null.
"""

from datetime import date, datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_io_tools.testing import PredicateTracker
from polars_io_tools.tests.helpers.assertions import assert_pushed_bounds
from polars_io_tools.tests.helpers.cases import Case, params

c = pl.col
FFILL = c("Value").forward_fill(3).alias("ForwardFill")
LINKED = {"index_col": "Date", "lookback": timedelta(days=3), "linked_cols": ["EventDate"]}
SELECTED = ["Date", "EventDate", "PointID", "Value", "ForwardFill"]


def _frame(index_dtype, days: int = 3) -> pl.DataFrame:
    """``days`` rows from 1 Jan 2023: EventDate is the next day, and Value is 100 on the first day, 200 on a fourth."""
    func = datetime if index_dtype == pl.Datetime else date
    return pl.DataFrame(
        {
            "Date": [func(2023, 1, d) for d in range(1, days + 1)],
            "EventDate": [date(2023, 1, d + 1) for d in range(1, days + 1)],
            "PointID": [1] * days,
            "Value": [100, None, None, 200][:days],
            "DummyValue": list(range(1, days + 1)),
        },
        schema={"Date": index_dtype, "EventDate": pl.Date, "PointID": pl.Int64, "Value": pl.Int64, "DummyValue": pl.Int64},
    )


def _filled(index_dtype, ffill, dtype=pl.Int64, empty=False) -> pl.DataFrame:
    """The 3 Jan row (EventDate 4 Jan) with ``ForwardFill`` set to ``ffill``, or no row."""
    schema = {**_frame(index_dtype).drop("DummyValue").schema, "ForwardFill": dtype}
    if empty:
        return pl.DataFrame(schema=schema)
    day = datetime(2023, 1, 3) if index_dtype == pl.Datetime else date(2023, 1, 3)
    return pl.DataFrame({"Date": [day], "EventDate": [date(2023, 1, 4)], "PointID": [1], "Value": [None], "ForwardFill": [ffill]}, schema=schema)


@pytest.mark.parametrize(
    ("index_col", "match"),
    [("Date_MISSING", "missing from the schema"), ("DummyValue", "but found type")],
    ids=["errors-missing", "errors-not_temporal"],
)
def test_errors(index_col, match):
    # Validation is deferred to collect time, where Polars wraps the error in a ComputeError.
    lf = _frame(pl.Date).lazy().piot.ts_with_columns([FFILL], index_col=index_col, lookback=timedelta(days=3), linked_cols=["EventDate"])
    with pytest.raises((ValueError, pl.exceptions.ComputeError), match=match):
        lf.collect()


# A filter on BusinessDatePlus2 (the date two business days on) becomes a Date filter widened by the window.
@pytest.mark.parametrize(
    ("lookahead", "upper"), [(None, date(2023, 1, 4)), (timedelta(days=1), date(2023, 1, 5))], ids=["simple_lookback", "simple_lookahead"]
)
def test_linked_business_days(lookahead, upper):
    tracker = PredicateTracker(pl.DataFrame({"Date": [date(2023, 1, d) for d in (1, 2, 3)], "Value": [1, 2, 3]}))
    lf = tracker.lazy_frame.with_columns(c("Date").dt.add_business_days(2, roll="forward").alias("BusinessDatePlus2"))
    result = (
        lf.piot.ts_with_columns([], "Date", linked_cols=["BusinessDatePlus2"], lookback=timedelta(days=3), lookahead=lookahead)
        .filter(c("BusinessDatePlus2") == date(2023, 1, 4))
        .filter((c("Date") == c("Date").max()).over("BusinessDatePlus2"))
        .collect()
    )
    assert_frame_equal(result, pl.DataFrame({"Date": [date(2023, 1, 2)], "Value": [2], "BusinessDatePlus2": [date(2023, 1, 4)]}))
    # Some Polars versions push nothing here.
    if tracker.last_predicate is not None:
        assert_pushed_bounds(tracker, "Date", date(2023, 1, 1), upper)


# Forward fill through a linked column (EventDate == 4 Jan becomes Date == 4 Jan) or the index itself (Date == 3 Jan, on
# four rows). The lookback must reach 1 Jan; a sub-day remainder reaches it only on a Date index, rounded out to whole days.
WINDOWED_FILLS = [
    Case(
        "ts_ffill",
        via="linked",
        expect=None,
        kw={"days": 3, "reach": 3, "ts": {"linked_cols": ["EventDate"]}, "filter": c("EventDate") == date(2023, 1, 4)},
    ),
    Case("ts_ffill_no_indexed_column", via="index", expect=None, kw={"days": 4, "reach": 2, "ts": {}, "filter": c("Date") == date(2023, 1, 3)}),
]


@pytest.mark.parametrize("index_dtype", [pl.Date, pl.Datetime], ids=["Date", "Datetime"])
@pytest.mark.parametrize("short", [None, timedelta(days=1), timedelta(hours=6), timedelta()], ids=["none", "day_short", "hours_short", "reaches"])
@pytest.mark.parametrize(("via", "kw"), params(WINDOWED_FILLS, "via", "kw"))
def test_windowed_forward_fill(via, kw, short, index_dtype):
    """``short`` is how far the lookback falls short of 1 Jan; ``None`` is no lookback."""
    lookback = None if short is None else timedelta(days=kw["reach"]) - short
    result = (
        _frame(index_dtype, kw["days"])
        .lazy()
        .piot.ts_with_columns([FFILL], index_col="Date", lookback=lookback, **kw["ts"])
        .filter(kw["filter"])
        .select(SELECTED)
        .collect()
    )
    if short is None:
        # Without a lookback a linked filter becomes Date == 4 Jan, which has no row.
        expected = _filled(index_dtype, None, empty=via == "linked")
    else:
        expected = _filled(index_dtype, 100 if not short or (short < timedelta(days=1) and index_dtype == pl.Date) else None)
    assert_frame_equal(result, expected)


def _ffill_callable(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(FFILL.cast(pl.Float64))


# The window reaches back for the fill even where a later filter drops the row it fills from.
FILLS = [
    Case("ts_ffill_multiple_filters-date", exprs=[FFILL.cast(pl.Float64)], extra=c("Date") > date(2023, 1, 1)),
    Case("ts_ffill_multiple_filters-dummy", exprs=[FFILL.cast(pl.Float64)], extra=c("DummyValue") > 1),
    Case("ts_with_columns_callable_expression", exprs=_ffill_callable, extra=None),
]


@pytest.mark.parametrize(("exprs", "extra"), params(FILLS, "exprs", "extra"))
def test_forward_fill_reaches_back(exprs, extra):
    lf = _frame(pl.Date).lazy().piot.ts_with_columns(exprs, **LINKED).filter(c("EventDate") == date(2023, 1, 4))
    lf = lf if extra is None else lf.filter(extra)
    assert_frame_equal(lf.select(SELECTED).collect(), _filled(pl.Date, 100.0, pl.Float64))


def _between_3_and_4(**cols) -> pl.DataFrame:
    base = _frame(pl.Date).slice(1, 2).with_columns(Value=pl.lit(None, dtype=pl.Int64))
    return base.with_columns(**{name: pl.Series(values) for name, values in cols.items()})


# (positional arguments to ts_with_columns) -> the rows with EventDate in [3 Jan, 4 Jan]
BETWEEN = [
    Case("ts_rolling_mean", args=[[c("DummyValue").sum().rolling(index_column="Date", period="2d")]], expected=_between_3_and_4(DummyValue=[3, 5])),
    Case(
        "ts_with_columns_args-list",
        args=[[c("DummyValue").alias("col1"), c("DummyValue").alias("col2")]],
        expected=_between_3_and_4(col1=[2, 3], col2=[2, 3]),
    ),
    Case(
        "ts_with_columns_args-varargs",
        args=[c("DummyValue").alias("col1"), c("DummyValue").alias("col2")],
        expected=_between_3_and_4(col1=[2, 3], col2=[2, 3]),
    ),
]


@pytest.mark.parametrize(("args", "expected"), params(BETWEEN, "args", "expected"))
def test_linked_range(args, expected):
    result = (
        _frame(pl.Date).lazy().piot.ts_with_columns(*args, **LINKED).filter(c("EventDate").is_between(date(2023, 1, 3), date(2023, 1, 4))).collect()
    )
    assert_frame_equal(result, expected)


def test_ts_with_columns_callable_no_filters():
    """Without a filter a callable behaves like ``pipe``."""
    lf = pl.LazyFrame({"Date": [date(2023, 1, 1), date(2023, 1, 2)], "val": [1, 2]})
    assert_frame_equal(
        lf.piot.ts_with_columns(lambda x: x.with_columns(pl.lit(42).alias("answer")), index_col="Date").collect(),
        lf.with_columns(answer=pl.lit(42)).collect(),
    )


@pytest.mark.parametrize("lookback", [timedelta(days=1), timedelta(days=3)])
@pytest.mark.parametrize("index_dtype", [pl.Date, pl.Datetime])
def test_ts_ffill_over_with_filtering(lookback, index_dtype):
    """A forward fill over Symbol, then a filter on Symbol after ts_with_columns: Polars pushes both Date and Symbol.

    A has a value on day 1 only, so its day-3 fill needs a 2-day lookback; B has one on day 3.
    """
    func = datetime if index_dtype == pl.Datetime else date
    schema = {"Date": index_dtype, "EventDate": pl.Date, "Symbol": pl.Utf8, "Value": pl.Int64}
    days = [func(2023, 1, d) for d in (1, 2, 3)] * 2
    df = pl.DataFrame(
        {
            "Date": days,
            "EventDate": [date(2023, 1, d) for d in (1, 2, 3)] * 2,
            "Symbol": ["A"] * 3 + ["B"] * 3,
            "Value": [100, None, None, None, None, 300],
        },
        schema=schema,
    )
    tracker = PredicateTracker(df)
    result = (
        tracker.lazy_frame.piot.ts_with_columns(
            [c("Value").forward_fill(3).over("Symbol").alias("ForwardFill").cast(pl.Float64)],
            index_col="Date",
            lookback=lookback,
            linked_cols=["EventDate"],
        )
        .filter(c("EventDate") == date(2023, 1, 3))
        .select("Date", "EventDate", "Symbol", "Value", "ForwardFill")
    )
    for symbol, value, ffill in [("A", None, 100.0 if lookback >= timedelta(days=2) else None), ("B", 300, 300.0)]:
        expected = pl.DataFrame(
            {"Date": [func(2023, 1, 3)], "EventDate": [date(2023, 1, 3)], "Symbol": [symbol], "Value": [value], "ForwardFill": [ffill]},
            schema={**schema, "ForwardFill": pl.Float64},
        )
        assert_frame_equal(result.filter(c("Symbol") == symbol).collect(), expected)
        assert sorted(tracker.last_predicate.meta.root_names()) == ["Date", "Symbol"]
