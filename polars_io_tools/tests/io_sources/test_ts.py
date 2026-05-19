from datetime import date, datetime, timedelta
from typing import Dict, Iterator, List, Optional, Tuple

import polars as pl
import pytest
from polars.io.plugins import register_io_source
from polars.testing import assert_frame_equal

from polars_io_tools.io_sources.base import CastNode, ColumnNode, FunctionNode, get_parsed_expr
from polars_io_tools.io_sources.enum import BooleanFunctionType


def _get_column_name_from_node(node):
    """Extract the column name from a node, handling both ColumnNode and CastNode wrapping ColumnNode."""
    if isinstance(node, ColumnNode):
        return node.name
    elif isinstance(node, CastNode) and isinstance(node.input, ColumnNode):
        return node.input.name
    else:
        raise ValueError(f"Cannot extract column name from node type: {type(node).__name__}")


def _assert_date_between_predicate_if_pushed(last_predicate, expected_lower: date, expected_upper: date) -> None:
    """Assert the pushed predicate shape when Polars chooses to expose one."""
    if last_predicate is None:
        return

    parsed_last_predicate = get_parsed_expr(last_predicate)
    assert isinstance(parsed_last_predicate, FunctionNode)
    assert parsed_last_predicate.function_type == BooleanFunctionType.IS_BETWEEN
    assert _get_column_name_from_node(parsed_last_predicate.inputs[0]) == "Date"

    lower_val = parsed_last_predicate.inputs[1].value
    upper_val = parsed_last_predicate.inputs[2].value
    lower_date = lower_val.date() if isinstance(lower_val, datetime) else lower_val
    upper_date = upper_val.date() if isinstance(upper_val, datetime) else upper_val
    assert lower_date == expected_lower
    assert upper_date == expected_upper


def generate_lazy_frame_and_container_with_pushed_predicate(df: pl.DataFrame) -> Tuple[pl.LazyFrame, Dict[str, pl.Expr]]:
    container = {}

    def my_scan(df):
        def source_generator(
            with_columns: Optional[List[str]],
            predicate: Optional[pl.Expr],
            n_rows: Optional[int],
            batch_size: Optional[int],
        ) -> Iterator[pl.DataFrame]:
            # Capture the predicate
            container["last_predicate"] = predicate
            df2 = df.clone()
            # Apply column projection if requested
            if with_columns is not None:
                df2 = df2.select(with_columns)

            # Apply predicate filtering if provided
            if predicate is not None:
                df2 = df2.filter(predicate)

            # Apply row limiting if requested
            if n_rows is not None:
                df2 = df2.head(n_rows)

            yield df2

        return register_io_source(io_source=source_generator, schema=df.schema)

    return my_scan(df), container


def _generate_df(index_col_typ: pl.DataType) -> pl.DataFrame:
    schema = {"Date": index_col_typ, "EventDate": pl.Date, "PointID": pl.Int64, "Value": pl.Int64, "DummyValue": pl.Int64}
    func = datetime if index_col_typ == pl.Datetime else date
    datetime_vals = [
        func(2023, 1, 1),
        func(2023, 1, 2),
        func(2023, 1, 3),
    ]
    df = pl.LazyFrame(
        {
            "Date": datetime_vals,
            "EventDate": [date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)],
            "PointID": [1, 1, 1],
            "Value": [100, None, None],
            "DummyValue": [1, 2, 3],
        },
        schema=schema,
    )
    return df


def test_errors():
    df = _generate_df(pl.Date).lazy()

    # Validation is deferred until collect time. Polars wraps user-raised exceptions
    # from the schema callable in a ComputeError, preserving the original message.
    with pytest.raises((ValueError, pl.exceptions.ComputeError), match="missing from the schema"):
        df.piot.ts_with_columns(
            [pl.col("Value").forward_fill(3).alias("ForwardFill")],
            index_col="Date_MISSING",
            lookback=timedelta(days=3),
            linked_cols=["EventDate"],
        ).collect()

    with pytest.raises((ValueError, pl.exceptions.ComputeError), match="but found type"):
        df.piot.ts_with_columns(
            [pl.col("Value").forward_fill(3).alias("ForwardFill")],
            index_col="DummyValue",
            lookback=timedelta(days=3),
            linked_cols=["EventDate"],
        ).collect()


def test_simple_lookback():
    dates = pl.Series("Date", [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)])
    values = pl.Series("Value", [1, 2, 3])

    df = pl.DataFrame([dates, values])
    lf, container = generate_lazy_frame_and_container_with_pushed_predicate(df)

    add_business_days = pl.col.Date.dt.add_business_days(2, roll="forward")
    lf = lf.with_columns(add_business_days.alias("BusinessDatePlus2"))
    # Now, we expect the Dates to be [2023-01-04, 2023-01-04, 2023-01-05]
    assert dates.dt.add_business_days(2, roll="forward").to_list() == [date(2023, 1, 4), date(2023, 1, 4), date(2023, 1, 5)]

    # Now, we can use lookback to convert filters on BusinessDatePlus2 to Date filters
    result = (
        lf.piot.ts_with_columns(
            [],
            "Date",
            linked_cols=["BusinessDatePlus2"],
            lookback=timedelta(days=3),  # Look back 3 days
        )
        .filter(pl.col("BusinessDatePlus2") == date(2023, 1, 4))  # This filter will be converted to a Date filter
        .filter((pl.col("Date") == pl.col.Date.max()).over("BusinessDatePlus2"))  # we then filter for the max Date per BusinessDatePlus2
        .collect()
    )
    expected = pl.DataFrame(
        {
            "Date": [date(2023, 1, 2)],
            "Value": [2],
            "BusinessDatePlus2": [date(2023, 1, 4)],
        }
    )
    assert_frame_equal(result, expected)
    _assert_date_between_predicate_if_pushed(container["last_predicate"], date(2023, 1, 1), date(2023, 1, 4))


def test_simple_lookahead():
    dates = pl.Series("Date", [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)])
    values = pl.Series("Value", [1, 2, 3])

    df = pl.DataFrame([dates, values])
    lf, container = generate_lazy_frame_and_container_with_pushed_predicate(df)

    add_business_days = pl.col.Date.dt.add_business_days(2, roll="forward")
    lf = lf.with_columns(add_business_days.alias("BusinessDatePlus2"))
    assert dates.dt.add_business_days(2, roll="forward").to_list() == [date(2023, 1, 4), date(2023, 1, 4), date(2023, 1, 5)]

    result = (
        lf.piot.ts_with_columns(
            [],
            "Date",
            linked_cols=["BusinessDatePlus2"],
            lookback=timedelta(days=3),
            lookahead=timedelta(days=1),  # Extend the upper bound by 1 day
        )
        .filter(pl.col("BusinessDatePlus2") == date(2023, 1, 4))
        .filter((pl.col("Date") == pl.col.Date.max()).over("BusinessDatePlus2"))
        .collect()
    )

    expected = pl.DataFrame(
        {
            "Date": [date(2023, 1, 2)],
            "Value": [2],
            "BusinessDatePlus2": [date(2023, 1, 4)],
        }
    )
    assert_frame_equal(result, expected)
    _assert_date_between_predicate_if_pushed(container["last_predicate"], date(2023, 1, 1), date(2023, 1, 5))


@pytest.mark.parametrize("lookback", [None, timedelta(days=2), timedelta(days=2, hours=18), timedelta(days=3)])
@pytest.mark.parametrize("index_col_typ", [pl.Date, pl.Datetime])
def test_ts_ffill(lookback, index_col_typ):
    func = datetime if index_col_typ == pl.Datetime else date
    target_val = func(2023, 1, 3)
    df = _generate_df(index_col_typ).lazy()

    result = (
        df.piot.ts_with_columns(
            [pl.col("Value").forward_fill(3).alias("ForwardFill")],
            index_col="Date",
            lookback=lookback,
            linked_cols=["EventDate"],
        )
        .filter(pl.col("EventDate") == date(2023, 1, 4))  # This filter will be converted to a Date filter
        .select("Date", "EventDate", "PointID", "Value", "ForwardFill")
        .collect()
    )
    expected_schema = {**df.collect_schema(), **{"ForwardFill": pl.Int64}}
    expected_schema.pop("DummyValue")  # DummyValue is not in the result

    # We look back 3 rows, but we need our lookback to contain the only non-null value in our table.
    # That non-null-value has EventDate = 2023-01-02, which is 2 days before our target date of 2023-01-04.
    # However, we utililze "Date" to determine the lookback
    if lookback is None:
        # If lookback is None, we try to filter for Date = 2023-01-04, but that is empty so we get an empty result.
        expected = pl.DataFrame({}, schema=expected_schema)
    else:
        if lookback >= timedelta(days=3):
            ffill = 100
        elif lookback > timedelta(days=2):
            # We only forward fill if the index_col type is Date, so we round down to the nearest day.
            ffill = 100 if index_col_typ == pl.Date else None
        else:
            ffill = None
        expected = pl.DataFrame(
            {"Date": [target_val], "EventDate": [date(2023, 1, 4)], "PointID": [1], "Value": [None], "ForwardFill": [ffill]}, schema=expected_schema
        )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("lookback", [None, timedelta(days=1), timedelta(days=1, hours=18), timedelta(days=2)])
@pytest.mark.parametrize("index_col_typ", [pl.Date, pl.Datetime])
def test_ts_ffill_no_indexed_column(lookback, index_col_typ):
    schema = {"Date": index_col_typ, "EventDate": pl.Date, "PointID": pl.Int64, "Value": pl.Int64, "DummyValue": pl.Int64}
    func = datetime if index_col_typ == pl.Datetime else date
    datetime_vals = [
        func(2023, 1, 1),
        func(2023, 1, 2),
        func(2023, 1, 3),
        func(2023, 1, 4),  # Added an extra date to ensure we can test lookback correctly
    ]
    target_val = func(2023, 1, 3)
    df = pl.LazyFrame(
        {
            "Date": datetime_vals,
            "EventDate": [date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4), date(2023, 1, 5)],
            "PointID": [1, 1, 1, 1],
            "Value": [100, None, None, 200],
            "DummyValue": [1, 2, 3, 4],
        },
        schema=schema,
    )
    result = (
        df.piot.ts_with_columns(
            [pl.col("Value").forward_fill(3).alias("ForwardFill")],
            index_col="Date",
            lookback=lookback,
        )
        .filter(pl.col("Date") == date(2023, 1, 3))
        .select("Date", "EventDate", "PointID", "Value", "ForwardFill")
        .collect()
    )
    expected_schema = {**df.collect_schema(), **{"ForwardFill": pl.Int64}}
    expected_schema.pop("DummyValue")  # DummyValue is not in the result

    # We look back 3 rows, but we need our lookback to contain the only non-null value in our table.
    if lookback is None:
        # No lookback
        ffill = None
    elif lookback >= timedelta(days=2):
        # We always lookback
        ffill = 100
    elif lookback > timedelta(days=1):
        # Here we have timedeltas between 1 and 2 days.
        # We only forward fill if the index_col type is Date, since we expand our range to the nearest day.
        ffill = 100 if index_col_typ == pl.Date else None
    else:
        ffill = None
    expected = pl.DataFrame(
        {"Date": [target_val], "EventDate": [date(2023, 1, 4)], "PointID": [1], "Value": [None], "ForwardFill": [ffill]}, schema=expected_schema
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("expr", [pl.col("Date") > date(2023, 1, 1), pl.col("DummyValue") > 1])
def test_ts_ffill_multiple_filters(expr):
    df = _generate_df(pl.Date).lazy()
    result = (
        df.piot.ts_with_columns(
            [pl.col("Value").forward_fill(3).alias("ForwardFill").cast(pl.Float64)],
            index_col="Date",
            lookback=timedelta(days=3),
            linked_cols=["EventDate"],
        )
        .filter(pl.col("EventDate") == date(2023, 1, 4))  # This filter will be converted to a Date filter
        .filter(expr)
        .select("Date", "EventDate", "PointID", "Value", "ForwardFill")
        .collect()
    )
    expected_schema = {**df.collect_schema(), **{"ForwardFill": pl.Float64}}
    expected_schema.pop("DummyValue")  # DummyValue is not in the result

    # Now, in either case we get the forward fill, because even though our filter on "Date" filters out the first row,
    # the lookback logic we apply extends our logic to include the first row, which gets filtered out afterwards.
    ffill = 100.0
    expected = pl.DataFrame(
        {"Date": [date(2023, 1, 3)], "EventDate": [date(2023, 1, 4)], "PointID": [1], "Value": [None], "ForwardFill": [ffill]}, schema=expected_schema
    )
    assert_frame_equal(result, expected)


def test_ts_rolling_mean():
    df = _generate_df(pl.Date).lazy()
    result = (
        df.piot.ts_with_columns(
            [pl.col("DummyValue").sum().rolling(index_column="Date", period="2d")],
            index_col="Date",
            lookback=timedelta(days=3),
            linked_cols=["EventDate"],
        )
        .filter(pl.col("EventDate").is_between(date(2023, 1, 3), date(2023, 1, 4)))  # This filter will be converted to a Date filter
        .collect()
    )
    expected_schema = df.collect_schema()

    # Now, in either case we get the forward fill, because even though our filter on "Date" filters out the first row,
    # the lookback logic we apply extends our logic to include the first row, which gets filtered out afterwards.
    expected = pl.DataFrame(
        {
            "Date": [date(2023, 1, 2), date(2023, 1, 3)],
            "EventDate": [date(2023, 1, 3), date(2023, 1, 4)],
            "PointID": [1, 1],
            "Value": [None, None],
            "DummyValue": [3, 5],  # note that we sum the DummyValue over a 2-day rolling window
        },
        schema=expected_schema,
    )
    assert_frame_equal(result, expected)


def test_ts_with_columns_args():
    df = _generate_df(pl.Date).lazy()
    kwargs = dict(
        index_col="Date",
        lookback=timedelta(days=3),
        linked_cols=["EventDate"],
    )
    res = [
        # Just pass list as first args.
        df.piot.ts_with_columns(
            [pl.col("DummyValue").alias("col1"), pl.col("DummyValue").alias("col2")],
            **kwargs,
        ),
        # As args.
        df.piot.ts_with_columns(
            pl.col("DummyValue").alias("col1"),
            pl.col("DummyValue").alias("col2"),
            **kwargs,
        ),
    ]

    for r in res:
        result = r.filter(
            pl.col("EventDate").is_between(date(2023, 1, 3), date(2023, 1, 4))
        ).collect()  # This filter will be converted to a Date filter
        expected_schema = df.collect_schema()
        expected_schema["col1"] = pl.Int64
        expected_schema["col2"] = pl.Int64

        # Now, in either case we get the forward fill, because even though our filter on "Date" filters out the first row,
        # the lookback logic we apply extends our logic to include the first row, which gets filtered out afterwards.
        expected = pl.DataFrame(
            {
                "Date": [date(2023, 1, 2), date(2023, 1, 3)],
                "EventDate": [date(2023, 1, 3), date(2023, 1, 4)],
                "PointID": [1, 1],
                "Value": [None, None],
                "DummyValue": [2, 3],
                "col1": [2, 3],
                "col2": [2, 3],
            },
            schema=expected_schema,
        )
        assert_frame_equal(result, expected)


def test_ts_with_columns_callable_expression():
    """Ensure callable expressions are supported and integrated with lookback pushdown."""
    df = _generate_df(pl.Date).lazy()

    def expr_fn(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(pl.col("Value").forward_fill(3).alias("ForwardFill").cast(pl.Float64))

    result = (
        df.piot.ts_with_columns(
            expr_fn,
            index_col="Date",
            lookback=timedelta(days=3),
            linked_cols=["EventDate"],
        )
        .filter(pl.col("EventDate") == date(2023, 1, 4))
        .select("Date", "EventDate", "PointID", "Value", "ForwardFill")
        .collect()
    )

    expected_schema = {**df.collect_schema(), **{"ForwardFill": pl.Float64}}
    expected_schema.pop("DummyValue")
    assert result.collect_schema() == expected_schema
    expected = pl.DataFrame(
        {"Date": [date(2023, 1, 3)], "EventDate": [date(2023, 1, 4)], "PointID": [1], "Value": [None], "ForwardFill": [100.0]},
        schema=expected_schema,
    )
    assert_frame_equal(result, expected)


def test_ts_with_columns_callable_no_filters():
    """Callable expressions should behave like .pipe when no filters are present."""
    df = pl.DataFrame(
        {
            "Date": [date(2023, 1, 1), date(2023, 1, 2)],
            "val": [1, 2],
        }
    ).lazy()

    def expr_fn(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(pl.lit(42).alias("answer"))

    res = df.piot.ts_with_columns(expr_fn, index_col="Date").collect()
    exp = df.pipe(expr_fn).collect()
    assert_frame_equal(res, exp)


@pytest.mark.parametrize("lookback", [timedelta(days=1), timedelta(days=3)])
@pytest.mark.parametrize("index_col_typ", [pl.Date, pl.Datetime])
def test_ts_ffill_over_with_filtering(lookback, index_col_typ):
    """
    Test forward fill with over() clause and subsequent symbol filtering.

    This test verifies that:
    1. Forward fill works correctly for each symbol independently using the over() clause
    2. Filters on symbol applied AFTER ts_with_columns work correctly
    3. The lookback parameter correctly determines which values are included

    The test creates data where:
    - Symbol A has a value only on day 1
    - Symbol B has a value only on day 3

    Then tests if filtering to day 3 and by symbol produces the correct forward fill results.
    """
    schema = {"Date": index_col_typ, "EventDate": pl.Date, "Symbol": pl.Utf8, "Value": pl.Int64}
    func = datetime if index_col_typ == pl.Datetime else date

    # Create test data with two symbols A and B with different value patterns
    df = pl.DataFrame(
        {
            "Date": [
                # Symbol A data - Value on day 1 only
                func(2023, 1, 1),
                func(2023, 1, 2),
                func(2023, 1, 3),
                # Symbol B data - Value on day 3 only
                func(2023, 1, 1),
                func(2023, 1, 2),
                func(2023, 1, 3),
            ],
            "EventDate": [
                # Event dates matching Date column for simplicity
                date(2023, 1, 1),
                date(2023, 1, 2),
                date(2023, 1, 3),
                date(2023, 1, 1),
                date(2023, 1, 2),
                date(2023, 1, 3),
            ],
            "Symbol": ["A", "A", "A", "B", "B", "B"],
            "Value": [100, None, None, None, None, 300],
        },
        schema=schema,
    )

    lf, container = generate_lazy_frame_and_container_with_pushed_predicate(df)

    # Apply ts_with_columns with forward_fill().over("Symbol") expression
    result = (
        lf.clone()
        .piot.ts_with_columns(
            [
                # Forward fill values within each Symbol group
                pl.col("Value").forward_fill(3).over("Symbol").alias("ForwardFill").cast(pl.Float64)
            ],
            index_col="Date",
            lookback=lookback,
            linked_cols=["EventDate"],
        )
        .filter(pl.col("EventDate") == date(2023, 1, 3))  # Filter to just day 3
        .select("Date", "EventDate", "Symbol", "Value", "ForwardFill")
    )

    # Filter the result by Symbol after collection
    result_a = result.clone().filter(pl.col("Symbol") == "A").collect()

    # For Symbol A, value from day 1 should be forward filled to day 3,
    # but only if lookback is sufficient to include day 1
    expected_schema = {"Date": index_col_typ, "EventDate": pl.Date, "Symbol": pl.Utf8, "Value": pl.Int64, "ForwardFill": pl.Float64}

    # For Symbol A on day 3, we need lookback >= 2 days to reach the value on day 1
    if lookback >= timedelta(days=2):
        ffill_a = 100.0  # Lookback includes day 1, so value gets forward filled
    else:
        ffill_a = None  # Lookback doesn't reach day 1, so no value to forward fill

    expected_a = pl.DataFrame(
        {
            "Date": [func(2023, 1, 3)],
            "EventDate": [date(2023, 1, 3)],
            "Symbol": ["A"],
            "Value": [None],
            "ForwardFill": [ffill_a],
        },
        schema=expected_schema,
    )
    assert_frame_equal(result_a, expected_a)
    pushed_down_predicate = container.get("last_predicate")
    # We push down filters to Date and Symbol, Polars handles this for us since
    # We are performing the "over" operation on Symbol
    assert sorted(pushed_down_predicate.meta.root_names()) == ["Date", "Symbol"]

    # Filter to Symbol B
    result_b = result.clone().filter(pl.col("Symbol") == "B").collect()

    # For Symbol B, value is on the current day (day 3), so always available regardless of lookback
    expected_b = pl.DataFrame(
        {
            "Date": [func(2023, 1, 3)],
            "EventDate": [date(2023, 1, 3)],
            "Symbol": ["B"],
            "Value": [300],
            "ForwardFill": [300.0],  # Value is from the current day
        },
        schema=expected_schema,
    )
    assert_frame_equal(result_b, expected_b)
    pushed_down_predicate = container.get("last_predicate")
    assert sorted(pushed_down_predicate.meta.root_names()) == ["Date", "Symbol"]
