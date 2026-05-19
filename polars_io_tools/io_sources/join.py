import datetime
import logging
from typing import Iterator, List, Literal, Optional, Tuple, Union

import polars as pl
from packaging import version

from .base import get_parsed_expr
from .range_visitor import (
    _convert_interval_to_polars_expr,
    _extend_dates_to_full_datetimes,
    _extend_to_full_dates,
    _lookahead_interval,
    _lookback_interval,
    _promote_dates_to_datetimes,
    convert_expr_to_datetime_range,
)
from .restrict_visitor import restrict_expr_to_columns
from .util import collect_lf_in_io_source, register_io_source_with_is_pure

__all__ = ("filtered_join", "filtered_join_asof", "join_between")

log = logging.getLogger(__name__)

# In polars versions >1.31.0
# polars will push down filters on join columns for us.
_POLARS_PUSHDOWN_FILTERS_JOIN_COLUMNS = version.parse(pl.__version__) > version.parse("1.31.0")


def _normalize_join_columns(
    on: Optional[Union[str, List[str]]],
    left_on: Optional[Union[str, List[str]]],
    right_on: Optional[Union[str, List[str]]],
) -> Tuple[List[str], List[str]]:
    """Normalize join column specifications to lists for our optimization logic."""
    if on is not None:
        if left_on is not None or right_on is not None:
            raise ValueError("Cannot specify both 'on' and 'left_on'/'right_on'")
        on_list = [on] if isinstance(on, str) else list(on)
        return on_list, on_list
    if left_on is None or right_on is None:
        raise ValueError("Must specify either 'on' or both 'left_on' and 'right_on'")
    left_list = [left_on] if isinstance(left_on, str) else list(left_on)
    right_list = [right_on] if isinstance(right_on, str) else list(right_on)
    if len(left_list) != len(right_list):
        raise ValueError("left_on and right_on must be the same length")
    return left_list, right_list


def _rename_columns_in_filters(
    left_predicate: pl.Expr,
    left_on: List[str],
    right_on: List[str],
    right_schema: dict,
) -> Optional[pl.Expr]:
    """
    Convert left_predicate to a predicate on a LazyFrame with 'right_schema' where
    we map left_on columns to right_on columns, in order.
    So if we have left_on=[A, B, C], right_on=[A1, B1, C1]
    We will convert the left_predicate to the right predicate using the mapping
    {A: A1, B: B1, C: C1}

    This operation is effectively renaming the columns in the left predicate to match the right schema.
    """
    container = {}
    # Since we are only dealing with columns specified in left_on and right_on,
    # we know they exist on the original tables as-is. Thus, we can avoid needing
    # to handle suffixes that might have been added by polars.
    schema = {k: v for k, v in right_schema.items() if k in right_on}

    def _dummy_source(
        with_columns: Optional[List[str]],
        predicate: Optional[pl.Expr],
        n_rows: Optional[int],
        batch_size: Optional[int],
    ) -> Iterator[pl.DataFrame]:
        container["predicate"] = predicate
        df = pl.DataFrame({}, schema=schema)
        if predicate is not None:
            df.filter(predicate)
        if with_columns is not None:
            df = df.select(with_columns)
        yield df

    lf = register_io_source_with_is_pure(_dummy_source, schema=schema)
    # We apply the rename here to have polars change the column names for us
    # NOTE: we are performing the rename right to left here. This might seem a bit
    # counterintuitive, but we are renaming the right columns to the left columns.
    # So, when we give our filters to this LazyFrame, the Polars engine will give our
    # custom io source the renamed columns which match the schema before the rename.
    rename_mapping = {right: left for left, right in zip(left_on, right_on)}
    renamed_lf = lf.rename({right: left for left, right in zip(left_on, right_on)})
    log.debug(f"Renaming LazyFrame schema {right_schema} with the mapping: {rename_mapping}")
    _ = renamed_lf.filter(left_predicate).collect()  # Trigger the lazy evaluation to apply the renames
    return container.get("predicate", None)


def filtered_join(
    lf1: pl.LazyFrame,
    lf2: pl.LazyFrame,
    on: Optional[Union[str, List[str]]] = None,
    how: Literal["inner", "left"] = "inner",
    *,
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    nulls_equal: bool = False,
    log_explain: bool = False,
    **join_kwargs,
) -> pl.LazyFrame:
    """
    When performing an inner join, we can push down filters to both the left and right dataframe. However, the join itself will filter out rows from the right dataframe that do not have a match in the left dataframe. Polars does not convert this join into a filter for us.

    This function will perform the same join as polars, but will materialize the left lazyframe first, then convert the join logic into a filter that will get pushed down to the right lazyframe.

    Examples:
        Simple usage example:
            >>> df = pl.LazyFrame({"x": [1, 2, 3], "y": [4, 5, 6]}).lazy()
            >>> df2 = pl.LazyFrame({"x": [-1, -2, 3], "z": [7, 8, 9]}).lazy()
            >>> df.piot.filtered_join(df2, on="x").collect()
            shape: (1, 3)
            ┌─────┬─────┬─────┐
            │ x   ┆ y   ┆ z   │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ i64 │
            ╞═════╪═════╪═════╡
            │ 3   ┆ 6   ┆ 9   │
            └─────┴─────┴─────┘

            This works like a regular inner join, but we actually pushed down an additional filter
            to the right lazyframe equivalent to ``pl.col("x").is_in([3])`` before performing the join.
    """
    if nulls_equal:
        raise NotImplementedError("nulls_equal set to True is not supported for now")

    left_on, right_on = _normalize_join_columns(on, left_on, right_on)

    if log_explain:
        log.debug(f"filtered_join: Left LazyFrame plan:\n{str(lf1.explain())}")
        log.debug(f"filtered_join: Right LazyFrame plan:\n{str(lf2.explain())}")

    l1_schema = lf1.collect_schema()
    l2_schema = lf2.collect_schema()

    # To get the schema, we match polars behavior by performing the join
    # on empty lazyframes with the same schemas and collecting the resulting schema
    schema = (
        pl.LazyFrame({}, schema=l1_schema)
        .join(
            pl.LazyFrame({}, schema=l2_schema),
            how=how,
            left_on=left_on,
            right_on=right_on,
            nulls_equal=nulls_equal,
            **join_kwargs,
        )
        .collect_schema()
    )

    def source_generator(
        with_columns: Optional[List[str]],
        predicate: Optional[pl.Expr],
        n_rows: Optional[int],
        batch_size: Optional[int],
    ) -> Iterator[pl.DataFrame]:
        left_columns = l1_schema.keys()
        left_predicate = None
        if predicate is not None:
            # We only need the left predicate, we can leave the right
            # lazyframe alone, polars will handle the filtering there for us
            # Since no columns in the left lazyframe are renamed after a join, we don't have to worry about column renames
            left_predicate = restrict_expr_to_columns(predicate, left_columns)
        left_lf = lf1.filter(left_predicate) if left_predicate is not None else lf1

        # If polars told us to only return a subset of columns, we don't
        # need to request them all. We can just request the columns polars
        # wants from us, and the left_on columns for the join.
        if with_columns is not None:
            all_columns = set([*with_columns, *left_on])
            # If with_columns includes suffixed columns (e.g. "quantity_previous"),
            # include the base column (e.g. "quantity") so a name conflict occurs
            # and the suffix gets applied to the right side.
            suffix = join_kwargs.get("suffix", "_right")
            for col in with_columns:
                if col.endswith(suffix):
                    all_columns.add(col[: -len(suffix)])
            true_left_columns = [col for col in l1_schema if col in all_columns]
            left_lf = left_lf.select(true_left_columns)

        if log_explain:
            log.debug(f"filtered_join: Left LazyFrame plan:\n{str(left_lf.explain())}")
        try:
            left_df = left_lf.collect()
        except Exception as e:
            raise RuntimeError(
                f"Failed to collect left LazyFrame for filtered join with plan: \n{str(left_lf.explain())}\n\nWhile running the above, {e}"
            ) from e
        # If left_df is empty here, we can just return an empty dataframe
        with_columns_set = set(with_columns) if with_columns is not None else None
        if left_df.is_empty():
            if with_columns_set is not None:
                # We need to make sure we return the columns that were requested
                # even if they are empty
                true_df = pl.DataFrame({}, schema={k: v for k, v in schema.items() if k in with_columns_set})
            else:
                true_df = pl.DataFrame({}, schema=schema)
            yield true_df
            # Return is a stop iteration here, so we break out of the generator
            return
        # TODO: We could make this more precise by using pl.Struct to
        # combine the different columns. That should theoretically get pushed down, but it's not covered by the parsers.
        new_filters = []
        for left_col, right_col in zip(left_on, right_on):
            left_col_ser = left_df[left_col].drop_nulls().unique()
            if len(left_col_ser) == 1:
                # We need this to make sure we treat the single value as a list
                new_filters.append(pl.col(right_col) == left_col_ser[0])
            else:
                # We convert to a list because otherwise polars 1.28.0 doesnt
                # push any filters down if we passed the series directly.
                new_filters.append(pl.col(right_col).is_in(left_col_ser.to_list()))

        extra_filter = pl.all_horizontal(*new_filters) if len(left_on) > 1 else new_filters[0]
        # Filter the right lazyframe based on the join column in the materialized left df
        # Do not collect yet
        right_lf = lf2.filter(extra_filter)
        df = left_df.lazy().join(right_lf, how=how, left_on=left_on, right_on=right_on, nulls_equal=nulls_equal, **join_kwargs)
        if predicate is not None:
            # This filter application will push the filter down to the right lazyframe for us
            # NOTE: This will re-run the filter on the left dataframe, but it should be fast since we already have the filtered dataframe
            df = df.filter(predicate)

        if with_columns_set is not None:
            df = df.select([col for col in schema.keys() if col in with_columns_set])
        else:
            df = df.select(schema.keys())

        if n_rows is not None:
            df = df.head(n_rows)

        if log_explain:
            log.debug(f"filtered_join: LazyFrame plan:\n{str(df.explain())}")
        try:
            yield from collect_lf_in_io_source(df, batch_size)
        except Exception as e:
            err_msg = f"Failed during collection for filtered join. Plan: \n{str(df.explain())}"
            err_msg += f"\n\nError: {e.__class__.__name__}:{e}"
            raise RuntimeError(err_msg) from e

    return register_io_source_with_is_pure(source_generator, schema=schema)


def filtered_join_asof(
    lf1: pl.LazyFrame,
    lf2: pl.LazyFrame,
    *,  # this matches Polars behavior for pl.join_asof
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    on: Optional[Union[str, List[str]]] = None,
    by: Optional[Union[str, List[str]]] = None,
    by_left: Optional[Union[str, List[str]]] = None,
    by_right: Optional[Union[str, List[str]]] = None,
    strategy: Literal["backward", "forward", "nearest"] = "backward",
    tolerance: Optional[Union[str, int, float, datetime.timedelta]] = None,  # TODO: Only timedelta is supported for now
    log_explain: bool = True,
    **join_kwargs,
) -> pl.LazyFrame:
    """
    Perform an optimized asof join with filter pushdown to minimize data loading.

    This function performs the same asof join as Polars but with optimizations
    that push filters down to the source data before performing the join. This can dramatically
    improve performance when working with large time-series datasets by reducing the amount
    of data that needs to be loaded and processed.

    The optimization works through multiple layers:
        1. **Standard Filter Pushdown**: Filters applicable to left-side columns are pushed down
        since asof joins preserve all left rows (like left joins)
        2. **By-Column Filter Pushdown**: When using exact-match columns (by/by_left/by_right),
        filters on these columns are pushed to both dataframes
        3. **Temporal Range Expansion**: When tolerance is specified, temporal filters are expanded based on the join strategy to ensure all potentially
        matching records are enclosed in the expanded filter, and that filter is pushed down to source.

    Args:
        lf1 (pl.LazyFrame): Left LazyFrame for the asof join
        lf2 (pl.LazyFrame): Right LazyFrame for the asof join
        left_on (str or List[str], optional): Column name(s) to join on from the left DataFrame. Must be specified if `on` is None.
        right_on (str or List[str], optional): Column name(s) to join on from the right DataFrame. Must be specified if `on` is None.
        on (str or List[str], optional): Column name(s) to join on when column names are the same in both DataFrames.
            Cannot be used with left_on/right_on.
        by (str or List[str], optional): Additional column(s) that must match exactly (not subject to asof matching).
            Cannot be used with by_left/by_right.
        by_left (str or List[str], optional): Left DataFrame columns for exact matching. Must be used with by_right.
        by_right (str or List[str], optional): Right DataFrame columns for exact matching. Must be used with by_left.
        strategy ({"backward", "forward", "nearest"}, default "backward"): Asof join strategy:
                - "backward": Match with the last value in right DataFrame that is <= left value
                - "forward": Match with the first value in right DataFrame that is >= left value
                - "nearest": Match with the closest value in right DataFrame
        tolerance (timedelta, optional): Maximum time difference allowed for a match. Currently only timedelta is supported.
            When specified, enables temporal range expansion optimization.
        log_explain (bool, default True): Whether to log detailed execution plans for debugging
        **join_kwargs: Additional keyword arguments passed to the underlying join_asof operation

    Returns:
        pl.LazyFrame: A LazyFrame representing the optimized asof join operation

    Examples:
        Basic temporal join with filter pushdown:
            >>> from datetime import date, timedelta
            >>> import polars_io_tools.io_sources  # register .piot namespace
            >>> df = pl.LazyFrame({"date": [date(2025, 1, 6), date(2025, 1, 7)], "z": [7, 8]}).lazy()
            >>> df2 = pl.LazyFrame({"date": [date(2024, 12, 30), date(2025, 1, 3)], "y": [4, 5]}).lazy()
            >>> df.piot.filtered_join_asof(df2, on="date", tolerance=timedelta(days=4)).collect()
            shape: (2, 3)
            ┌────────────┬─────┬─────┐
            │ date       ┆ z   ┆ y   │
            │ ---        ┆ --- ┆ --- │
            │ date       ┆ i64 ┆ i64 │
            ╞════════════╪═════╪═════╡
            │ 2025-01-06 ┆ 7   ┆ 5   │
            │ 2025-01-07 ┆ 8   ┆ 5   │
            └────────────┴─────┴─────┘

    Notes:
        - Performance gains are most significant when the left DataFrame is much smaller
          than the right DataFrame and/or when working with time-series data where
          temporal filtering can eliminate large portions of the right DataFrame
        - The temporal range expansion optimization requires that join columns contain
          datetime-like data and that filters contain temporal predicates
        - When using tolerance with "nearest" strategy, the function expands search
          ranges in both directions; "backward"/"forward" only expand in one direction
        - Filter pushdown works by materializing parts of the query plan and analyzing
          the predicates, so very complex filter expressions may not be fully optimized

    Warnings:
        - Currently only timedelta tolerance is supported; string and numeric tolerances
          are not yet implemented
        - Complex nested filter expressions may not be fully analyzed for pushdown
        - The optimization requires additional memory to materialize intermediate results
          for filter analysis

    See Also:
        filtered_join : Optimized inner/left joins with filter pushdown
        pl.LazyFrame.join_asof : Standard Polars asof join without optimization
    """
    left_on, right_on = _normalize_join_columns(on, left_on, right_on)
    if by is not None or by_left is not None or by_right is not None:
        by_left, by_right = _normalize_join_columns(by, by_left, by_right)

    if log_explain:
        log.debug(f"filtered_join_asof: Left LazyFrame plan:\n{str(lf1.explain())}")
        log.debug(f"filtered_join_asof: Right LazyFrame plan:\n{str(lf2.explain())}")

    l1_schema = lf1.collect_schema()
    l2_schema = lf2.collect_schema()

    # To get the schema, we match polars behavior by performing the join
    # on empty lazyframes with the same schemas and collecting the resulting schema
    kwargs = dict(
        by_left=by_left,
        by_right=by_right,
        strategy=strategy,
        tolerance=tolerance,
        left_on=(left_on if len(left_on) > 1 else left_on[0]),
        right_on=(right_on if len(right_on) > 1 else right_on[0]),
        **join_kwargs,
    )
    schema = (
        pl.LazyFrame({}, schema=l1_schema)
        .join_asof(
            pl.LazyFrame({}, schema=l2_schema),
            **kwargs,  # type: ignore[arg-type]
        )
        .collect_schema()
    )

    def source_generator(
        with_columns: Optional[List[str]],
        predicate: Optional[pl.Expr],
        n_rows: Optional[int],
        batch_size: Optional[int],
    ) -> Iterator[pl.DataFrame]:
        nonlocal lf1
        nonlocal lf2
        right_filters_from_by = None
        parsed_predicate = get_parsed_expr(predicate) if predicate is not None else None
        with_columns_set = set(with_columns) if with_columns is not None else None

        if parsed_predicate is not None:
            # A join_asof is like a left join but where we match on nearest keys. Since a left join
            # keeps all rows from the left dataframe and pulls in matches from the right, we can push
            # all filters to the left side down.
            if not _POLARS_PUSHDOWN_FILTERS_JOIN_COLUMNS:
                full_left_predicate = restrict_expr_to_columns(parsed_predicate, l1_schema.keys())
                if full_left_predicate is not None:
                    # If we have a predicate, we can push it down to the left lazyframe
                    # We only do so if polars does not
                    lf1 = lf1.filter(full_left_predicate)

                if by_left is not None and by_right is not None:
                    # If we have set by_left and by_right, then we can, additionally, push down filters
                    # on the by columns to both lazyframes. Since we already applied everything we can to the left,
                    # we focus here on the right LazyFrame. Now, we restrict the left lazyframe to just the by_left columns
                    left_predicate = restrict_expr_to_columns(parsed_predicate, by_left)
                    if left_predicate is not None:
                        if by_left == by_right:
                            # If by_left and by_right are the same, we can just use the left predicate
                            # to filter the right lazyframe as well, we know columns are in both.
                            right_filters_from_by = left_predicate
                        else:
                            # In this case, we have to "rename" the left predicate to match the right schema.
                            # This is because we must apply this filter BEFORE we pass in our LazyFrame to polars
                            # to perform a true join_asof. Now, this isn't comprehensive. There could be columns that have the
                            # suffix added to them that we would miss in the renaming. However, since we pass the same filters to
                            # polars AFTER we let polars perform the join_asof, we know the result is the same.
                            right_filters_from_by = _rename_columns_in_filters(
                                left_predicate,
                                left_on=by_left,
                                right_on=by_right,
                                right_schema=l2_schema,
                            )
                else:
                    right_filters_from_by = None

            lf2 = lf2.filter(right_filters_from_by) if right_filters_from_by is not None else lf2
            right_extended_filters = []
            empty_interval = False

            # If we are given a tolerance
            if tolerance is not None and isinstance(tolerance, datetime.timedelta):
                if strategy == "nearest":
                    # We can restrict nearest to cover both lookback and lookahead intervals.
                    target_funcs = [_lookback_interval, _lookahead_interval]
                elif strategy == "backward":
                    target_funcs = [_lookback_interval]
                elif strategy == "forward":
                    target_funcs = [_lookahead_interval]
                else:
                    raise ValueError(f"Unknown strategy: {strategy}. Must be one of 'backward', 'forward', or 'nearest'.")
                for left_col, right_col in zip(left_on, right_on):
                    try:
                        index_col_interval = convert_expr_to_datetime_range(parsed_predicate, left_col, get_enclosure=False, preserve_dates=True)
                    except TypeError:
                        # Predicate mixes ``date`` and ``datetime`` literals against the same column; ``preserve_dates``
                        # can produce uncomparable bounds. Fall back to coercing to datetime — pushdown may be slightly
                        # looser but the original predicate is still reapplied by the caller for correctness.
                        index_col_interval = convert_expr_to_datetime_range(parsed_predicate, left_col, get_enclosure=False)
                    index_col_typ = l2_schema[right_col]
                    if index_col_typ == pl.Date:
                        # Date right column: promote preserved date bounds to midnight datetimes so sub-day tolerances
                        # still cross day boundaries; floored back below.
                        index_col_interval = _promote_dates_to_datetimes(index_col_interval)
                    elif isinstance(index_col_typ, pl.Datetime):
                        # Datetime right column with date-typed filter literals: widen date bounds to full-day datetime
                        # ranges before applying tolerance. ``date - timedelta(hours=18)`` truncates to whole days in
                        # Python, so sub-day shifts would otherwise be silently dropped, and the upper bound would
                        # collapse to midnight and drop intraday rows on the bound day.
                        index_col_interval = _extend_dates_to_full_datetimes(index_col_interval)
                    # We apply the range expansion transformations
                    for target_func in target_funcs:
                        index_col_interval = target_func(index_col_interval, tolerance)
                    if index_col_typ == pl.Date:
                        # We have to restrict only on dates.
                        index_col_interval = _extend_to_full_dates(index_col_interval)
                    new_filter = _convert_interval_to_polars_expr(index_col_interval, right_col)
                    if new_filter is False:
                        empty_interval = True
                    elif new_filter is not None:
                        right_extended_filters.append(new_filter)

            if right_extended_filters:
                # Apply temporal filters first so they get pushed down unobstructed.
                lf2 = lf2.filter(pl.all_horizontal(*right_extended_filters) if len(right_extended_filters) > 1 else right_extended_filters[0])

            if empty_interval:
                # Empty interval — no right-side rows can match.
                # Applied after filters so the temporal bounds still act as a
                # safety net if head(0) doesn't get pushed down to the source.
                lf2 = lf2.head(0)

        lf_joined = lf1.join_asof(
            lf2,
            **kwargs,  # type: ignore[arg-type]
        )

        if predicate is not None:
            lf_joined = lf_joined.filter(predicate)

        # TODO: Remove when this does not interfere with parquet. Should be
        # release >1.30.0
        if with_columns_set is not None:
            lf_joined = lf_joined.select([col for col in schema.keys() if col in with_columns_set])
        else:
            lf_joined = lf_joined.select(schema.keys())

        if n_rows is not None:
            lf_joined = lf_joined.head(n_rows)

        if log_explain:
            log.debug(f"filtered_join_asof: LazyFrame plan:\n{str(lf_joined.explain())}")
        try:
            yield from collect_lf_in_io_source(lf_joined, batch_size)
        except Exception as e:
            err_msg = f"Failed during collection for filtered join_asof. Plan: \n{str(lf_joined.explain())}"
            err_msg += f"\n\nError: {e.__class__.__name__}:{e}"
            raise RuntimeError(err_msg) from e

    return register_io_source_with_is_pure(source_generator, schema=schema)


def join_between(
    left: pl.LazyFrame,
    right: pl.LazyFrame,
    left_on: str,
    right_on_start: str,
    right_on_end: str,
    by: Union[str, List[str], None] = None,
    how: Literal["left", "inner"] = "left",
) -> pl.LazyFrame:
    """Join left table to right table where left_on is between right_on_start and right_on_end.

    This implements a range join using a single asof join plus validation:

    1. Asof backward: finds rows where ``left_on >= right_on_start``
       (with equi-join on *by* if specified)
    2. Validation: ensures ``left_on <= right_on_end``

    The *by* parameter specifies columns that exist in **both** tables. When
    provided, the join will only match rows where:

    - The *by* columns have equal values (equi-join condition)
    - AND ``left_on`` is between ``right_on_start`` and ``right_on_end``
      (range condition)

    Args:
        left (pl.LazyFrame): Left LazyFrame with the point-in-time column.
        right (pl.LazyFrame): Right LazyFrame with range start/end columns.
        left_on (str): Column name in left table containing the date/value to match.
        right_on_start (str): Column name in right table for range start (inclusive).
        right_on_end (str): Column name in right table for range end (inclusive).
        by (str | list[str] | None): Column name(s) that exist in both tables for equi-join.
        how ({"left", "inner"}): Join type — ``"left"`` preserves all left rows, ``"inner"`` keeps
            only matches.

    Returns:
        pl.LazyFrame: Joined LazyFrame with all columns from both tables.
            For ``"left"`` join, right columns are null when no matching range
            exists.

    Notes:
        **Non-overlapping intervals only.** This function returns at most one
        right-side match per left row (the nearest ``right_on_start`` via a
        backward asof join). If the right table has overlapping intervals, only
        the interval with the latest start that is not after ``left_on`` is
        considered. For many-to-many interval joins where overlapping ranges
        should each produce a row, use :meth:`polars.LazyFrame.join_where`::

            left.join_where(
                right,
                pl.col("left_on") >= pl.col("right_on_start"),
                pl.col("left_on") <= pl.col("right_on_end"),
            )

        See also `pola-rs/polars#24091 <https://github.com/pola-rs/polars/issues/24091>`_
        for discussion of a general sorted-data range join in Polars.

    Examples:
        Resolve observations to the contract record effective on each date:

        >>> import polars as pl
        >>> from datetime import date
        >>> observations = pl.LazyFrame({
        ...     "symbol": ["ESH4", "ESH4"],
        ...     "obs_date": [date(2024, 1, 15), date(2024, 3, 10)],
        ... })
        >>> contracts = pl.LazyFrame({
        ...     "symbol": ["ESH4", "ESH4"],
        ...     "eff_start": [date(2024, 1, 1), date(2024, 3, 1)],
        ...     "eff_end": [date(2024, 2, 28), date(2024, 3, 15)],
        ...     "contract_id": ["H4-v1", "H4-v2"],
        ... })
        >>> result = join_between(
        ...     observations, contracts,
        ...     left_on="obs_date",
        ...     right_on_start="eff_start",
        ...     right_on_end="eff_end",
        ...     by="symbol",
        ... )
    """
    by_cols = [by] if isinstance(by, str) else (list(by) if by else None)

    # Sort for asof join
    left_sorted = left.sort(by_cols + [left_on] if by_cols else left_on)
    right_sorted = right.sort(by_cols + [right_on_start] if by_cols else right_on_start)

    # Single asof backward join: finds rows where by matches AND right_on_start <= left_on
    # check_sortedness=False: data is explicitly sorted above; Polars can't verify per-group sortedness
    suffix = "_right"
    result = left_sorted.join_asof(
        right_sorted,
        left_on=left_on,
        right_on=right_on_start,
        by=by_cols,
        strategy="backward",
        check_sortedness=False,
        suffix=suffix,
    )

    # After join_asof, Polars keeps left columns unsuffixed and suffixes right columns
    # that collide with left names. Build a mapping from pre-join right names to post-join names.
    left_names = set(left.collect_schema().names())
    # right_on_start is consumed by join_asof (merged into left_on), but may reappear
    # with suffix if it collides with a left column name
    right_names = right.collect_schema().names()

    def _post_join_name(col: str) -> str:
        """Map a pre-join right column name to its post-join name."""
        # by columns come from left, so right by-cols are dropped
        if by_cols and col in by_cols:
            return col
        # right_on_start is merged into left_on by join_asof
        if col == right_on_start and right_on_start == left_on:
            return col
        # If right column name collides with any left column, Polars adds suffix
        if col in left_names:
            return col + suffix
        return col

    # Validate: left_on must also be <= right_on_end (using post-join name)
    end_col = _post_join_name(right_on_end)
    valid_match = pl.col(left_on) <= pl.col(end_col)

    # Null out right-only columns when validation fails. When right_on_start == left_on,
    # the asof join key is the preserved left key rather than a right-side output column.
    right_only_cols = [c for c in right_names if c not in (by_cols or []) and c != right_on_start]
    nullable_right_cols = list(right_only_cols)
    if right_on_start != left_on:
        nullable_right_cols.append(right_on_start)
    null_exprs = [pl.when(valid_match).then(pl.col(_post_join_name(c))).otherwise(None).alias(_post_join_name(c)) for c in nullable_right_cols]
    result = result.with_columns(null_exprs)

    # For inner join, filter out invalid matches
    if how == "inner":
        result = result.filter(valid_match)

    return result
