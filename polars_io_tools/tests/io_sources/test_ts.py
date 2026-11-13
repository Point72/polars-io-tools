"""ts_with_columns: a filter on the index column or a linked column is widened by the lookback and lookahead before it
reaches the source, so window expressions see the rows they need, and the result is then trimmed to the filter.

The source sits behind a ``PredicateTracker``. Most rows forward-fill ``Value`` (non-null only on the first day) and
name the expected fill; a lookback that does not reach the first day leaves it null.
"""

from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from datetime import date, datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_io_tools._compat import POLARS_HAS_COLLECT_BATCHES
from polars_io_tools.testing import PredicateTracker
from polars_io_tools.tests.helpers.assertions import assert_pushed_time_bounds
from polars_io_tools.tests.helpers.cases import as_params
from polars_io_tools.tests.helpers.pickling import pickle_round_trip

FORWARD_FILL_VALUE = pl.col("Value").forward_fill(3).alias("ForwardFill")
DATE_INDEX_LINKED_TO_EVENT_DATE = {"index_col": "Date", "lookback": timedelta(days=3), "linked_cols": ["EventDate"]}
OUTPUT_COLUMNS = ["Date", "EventDate", "PointID", "Value", "ForwardFill"]


def _daily_frame(index_dtype, days: int = 3) -> pl.DataFrame:
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


def _jan_3_row(index_dtype, forward_fill, dtype=pl.Int64, empty=False) -> pl.DataFrame:
    """The 3 Jan row (EventDate 4 Jan) with ``ForwardFill`` set to ``forward_fill``, or no row if ``empty``."""
    schema = {**_daily_frame(index_dtype).drop("DummyValue").schema, "ForwardFill": dtype}
    if empty:
        return pl.DataFrame(schema=schema)
    day = datetime(2023, 1, 3) if index_dtype == pl.Datetime else date(2023, 1, 3)
    return pl.DataFrame(
        {"Date": [day], "EventDate": [date(2023, 1, 4)], "PointID": [1], "Value": [None], "ForwardFill": [forward_fill]}, schema=schema
    )


@dataclass(frozen=True)
class IndexColumnErrorCase:
    """An index column ``ts_with_columns`` rejects, and the error message pattern."""

    id: str
    _: KW_ONLY
    index_col: str
    match: str


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            IndexColumnErrorCase("errors-missing", index_col="Date_MISSING", match="missing from the schema"),
            IndexColumnErrorCase("errors-not_temporal", index_col="DummyValue", match="but found type"),
        ]
    ),
)
def test_errors(case: IndexColumnErrorCase):
    # Validation is deferred to collect time, where Polars wraps the error in a ComputeError.
    lf = (
        _daily_frame(pl.Date)
        .lazy()
        .piot.ts_with_columns([FORWARD_FILL_VALUE], index_col=case.index_col, lookback=timedelta(days=3), linked_cols=["EventDate"])
    )
    with pytest.raises((ValueError, pl.exceptions.ComputeError), match=case.match):
        lf.collect()


# A filter on BusinessDatePlus2 (the date two business days on) becomes a Date filter widened by the window.
@dataclass(frozen=True)
class BusinessDaysCase:
    """A lookahead, and the upper bound of the date filter the source receives."""

    id: str
    _: KW_ONLY
    lookahead: timedelta | None
    upper: date


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            BusinessDaysCase("simple_lookback", lookahead=None, upper=date(2023, 1, 4)),
            BusinessDaysCase("simple_lookahead", lookahead=timedelta(days=1), upper=date(2023, 1, 5)),
        ]
    ),
)
def test_linked_business_days(case: BusinessDaysCase):
    tracker = PredicateTracker(pl.DataFrame({"Date": [date(2023, 1, d) for d in (1, 2, 3)], "Value": [1, 2, 3]}))
    lf = tracker.lazy_frame.with_columns(pl.col("Date").dt.add_business_days(2, roll="forward").alias("BusinessDatePlus2"))
    result = (
        lf.piot.ts_with_columns([], "Date", linked_cols=["BusinessDatePlus2"], lookback=timedelta(days=3), lookahead=case.lookahead)
        .filter(pl.col("BusinessDatePlus2") == date(2023, 1, 4))
        .filter((pl.col("Date") == pl.col("Date").max()).over("BusinessDatePlus2"))
        .collect()
    )
    assert_frame_equal(result, pl.DataFrame({"Date": [date(2023, 1, 2)], "Value": [2], "BusinessDatePlus2": [date(2023, 1, 4)]}))
    # Some Polars versions push nothing here.
    if tracker.last_predicate is not None:
        assert_pushed_time_bounds(tracker, "Date", date(2023, 1, 1), case.upper)


@dataclass(frozen=True)
class WindowedFillCase:
    """A forward fill filtered on a linked column or on the index, and the shortest lookback that reaches 1 Jan.

    The linked filter ``EventDate == 4 Jan`` becomes ``Date == 4 Jan``; the index filter is ``Date == 3 Jan``, on four
    rows. A sub-day remainder of the lookback reaches 1 Jan only on a Date index, which rounds it out to whole days.
    """

    id: str
    _: KW_ONLY
    filter_on_linked_column: bool
    # how many days of data the source holds
    days: int
    lookback_to_jan_1: timedelta
    # extra keyword arguments for ``ts_with_columns``
    ts_options: dict
    predicate: pl.Expr


WINDOWED_FILLS = [
    WindowedFillCase(
        "ts_ffill",
        filter_on_linked_column=True,
        days=3,
        lookback_to_jan_1=timedelta(days=3),
        ts_options={"linked_cols": ["EventDate"]},
        predicate=pl.col("EventDate") == date(2023, 1, 4),
    ),
    WindowedFillCase(
        "ts_ffill_no_indexed_column",
        filter_on_linked_column=False,
        days=4,
        lookback_to_jan_1=timedelta(days=2),
        ts_options={},
        predicate=pl.col("Date") == date(2023, 1, 3),
    ),
]


@pytest.mark.parametrize("index_dtype", [pl.Date, pl.Datetime], ids=["Date", "Datetime"])
@pytest.mark.parametrize("short", [None, timedelta(days=1), timedelta(hours=6), timedelta()], ids=["none", "day_short", "hours_short", "reaches"])
@pytest.mark.parametrize("case", as_params(WINDOWED_FILLS))
def test_windowed_forward_fill(case: WindowedFillCase, short, index_dtype):
    """``short`` is how far the lookback falls short of 1 Jan; ``None`` is no lookback."""
    lookback = None if short is None else case.lookback_to_jan_1 - short
    result = (
        _daily_frame(index_dtype, case.days)
        .lazy()
        .piot.ts_with_columns([FORWARD_FILL_VALUE], index_col="Date", lookback=lookback, **case.ts_options)
        .filter(case.predicate)
        .select(OUTPUT_COLUMNS)
        .collect()
    )
    if short is None:
        # Without a lookback a linked filter becomes Date == 4 Jan, which has no row.
        expected = _jan_3_row(index_dtype, None, empty=case.filter_on_linked_column)
    else:
        # a Date index steps in whole days, so a lookback less than a day short still reaches 1 Jan
        expected = _jan_3_row(index_dtype, 100 if not short or (short < timedelta(days=1) and index_dtype == pl.Date) else None)
    assert_frame_equal(result, expected)


@dataclass(frozen=True)
class ForwardFillCase:
    """A forward fill and a further filter; the window must reach back for the fill even where the filter drops the row
    it fills from."""

    id: str
    _: KW_ONLY
    # the expressions, or a function of the frame, passed to ``ts_with_columns``
    exprs: list[pl.Expr] | Callable[[pl.LazyFrame], pl.LazyFrame]
    extra_filter: pl.Expr | None


FILLS = [
    ForwardFillCase("ts_ffill_multiple_filters-date", exprs=[FORWARD_FILL_VALUE.cast(pl.Float64)], extra_filter=pl.col("Date") > date(2023, 1, 1)),
    ForwardFillCase("ts_ffill_multiple_filters-dummy", exprs=[FORWARD_FILL_VALUE.cast(pl.Float64)], extra_filter=pl.col("DummyValue") > 1),
    ForwardFillCase(
        "ts_with_columns_callable_expression",
        exprs=lambda lf: lf.with_columns(FORWARD_FILL_VALUE.cast(pl.Float64)),
        extra_filter=None,
    ),
]


@pytest.mark.parametrize("case", as_params(FILLS))
def test_forward_fill_reaches_back(case: ForwardFillCase):
    lf = (
        _daily_frame(pl.Date)
        .lazy()
        .piot.ts_with_columns(case.exprs, **DATE_INDEX_LINKED_TO_EVENT_DATE)
        .filter(pl.col("EventDate") == date(2023, 1, 4))
    )
    lf = lf if case.extra_filter is None else lf.filter(case.extra_filter)
    assert_frame_equal(lf.select(OUTPUT_COLUMNS).collect(), _jan_3_row(pl.Date, 100.0, pl.Float64))


def _rows_for_jan_3_and_4(**cols) -> pl.DataFrame:
    base = _daily_frame(pl.Date).slice(1, 2).with_columns(Value=pl.lit(None, dtype=pl.Int64))
    return base.with_columns(**{name: pl.Series(values) for name, values in cols.items()})


@dataclass(frozen=True)
class LinkedRangeCase:
    """Positional arguments to ``ts_with_columns``, and the rows with ``EventDate`` in [3 Jan, 4 Jan]."""

    id: str
    _: KW_ONLY
    args: list
    expected: pl.DataFrame


BETWEEN = [
    LinkedRangeCase(
        "ts_rolling_mean",
        args=[[pl.col("DummyValue").sum().rolling(index_column="Date", period="2d")]],
        expected=_rows_for_jan_3_and_4(DummyValue=[3, 5]),
    ),
    LinkedRangeCase(
        "ts_with_columns_args-list",
        args=[[pl.col("DummyValue").alias("col1"), pl.col("DummyValue").alias("col2")]],
        expected=_rows_for_jan_3_and_4(col1=[2, 3], col2=[2, 3]),
    ),
    LinkedRangeCase(
        "ts_with_columns_args-varargs",
        args=[pl.col("DummyValue").alias("col1"), pl.col("DummyValue").alias("col2")],
        expected=_rows_for_jan_3_and_4(col1=[2, 3], col2=[2, 3]),
    ),
]


@pytest.mark.parametrize("case", as_params(BETWEEN))
def test_linked_range(case: LinkedRangeCase):
    result = (
        _daily_frame(pl.Date)
        .lazy()
        .piot.ts_with_columns(*case.args, **DATE_INDEX_LINKED_TO_EVENT_DATE)
        .filter(pl.col("EventDate").is_between(date(2023, 1, 3), date(2023, 1, 4)))
        .collect()
    )
    assert_frame_equal(result, case.expected)


def test_ts_with_columns_callable_no_filters():
    """Without a filter a callable behaves like ``pipe``."""
    lf = pl.LazyFrame({"Date": [date(2023, 1, 1), date(2023, 1, 2)], "val": [1, 2]})
    assert_frame_equal(
        lf.piot.ts_with_columns(lambda x: x.with_columns(pl.lit(42).alias("answer")), index_col="Date").collect(),
        lf.with_columns(answer=pl.lit(42)).collect(),
    )


@pytest.mark.parametrize("lookback", [timedelta(days=1), timedelta(days=3)], ids=["one_day", "three_days"])
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
            [pl.col("Value").forward_fill(3).over("Symbol").alias("ForwardFill").cast(pl.Float64)],
            index_col="Date",
            lookback=lookback,
            linked_cols=["EventDate"],
        )
        .filter(pl.col("EventDate") == date(2023, 1, 3))
        .select("Date", "EventDate", "Symbol", "Value", "ForwardFill")
    )
    for symbol, value, forward_fill in [("A", None, 100.0 if lookback >= timedelta(days=2) else None), ("B", 300, 300.0)]:
        expected = pl.DataFrame(
            {"Date": [func(2023, 1, 3)], "EventDate": [date(2023, 1, 3)], "Symbol": [symbol], "Value": [value], "ForwardFill": [forward_fill]},
            schema={**schema, "ForwardFill": pl.Float64},
        )
        assert_frame_equal(result.filter(pl.col("Symbol") == symbol).collect(), expected)
        assert sorted(tracker.last_predicate.meta.root_names()) == ["Date", "Symbol"]


@pytest.mark.skipif(not POLARS_HAS_COLLECT_BATCHES, reason="collect_batches requires Polars >= 1.34.0")
def test_collect_batches_ts_with_columns_yields_equal():
    dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    vals = [1, 2, 3, 4]
    base = pl.LazyFrame({"Date": dates, "val": vals})

    lf = base.piot.ts_with_columns([pl.col("val") * 2], index_col="Date")
    chunks = list(lf.collect_batches(chunk_size=2))
    combined = pl.concat(chunks, how="vertical").sort("Date")
    expected = lf.collect().sort("Date")

    assert_frame_equal(combined, expected)


class TestTsWithColumnsPickle:
    """Tests for ts_with_columns pickle support."""

    def test_ts_with_columns_pickle_basic(self):
        """ts_with_columns LazyFrames can be pickled and unpickled."""
        df = pl.DataFrame(
            {
                "date": [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)],
                "value": [10, 20, 30],
            }
        )

        # ts_with_columns uses index_col parameter (not ts_col)
        lf = df.lazy().piot.ts_with_columns(
            [pl.col("value").shift(1).alias("prev_value")],
            index_col="date",
            lookback=timedelta(days=1),
        )
        lf_unpickled = pickle_round_trip(lf)

        # Verify results match exactly
        expected = lf.collect()
        result = lf_unpickled.collect()
        assert_frame_equal(result, expected)
