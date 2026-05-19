import datetime
import logging
from typing import Callable, List, Optional

import polars as pl

from .multi_source import FilterSpec, multi_source

__all__ = ("ts_with_columns",)


log = logging.getLogger(__name__)


def ts_with_columns(
    self: pl.LazyFrame,
    *exprs: pl.Expr | List[pl.Expr] | Callable[[pl.LazyFrame], pl.LazyFrame],
    index_col: str = None,
    linked_cols: Optional[List[str]] = None,
    lookback: Optional[datetime.timedelta] = None,
    lookahead: Optional[datetime.timedelta] = None,
    log_explain: bool = False,
    _disable_optimizations: bool = False,
    expressions: List[pl.Expr] | Callable[[pl.LazyFrame], pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Apply expressions to a time series LazyFrame with optimized date-based predicate pushdown.

    The flow is as follows:
        1. We extract filters on linked_cols.
        2. We convert them to filters on the "index_col", expanded backwards by lookback and forward by lookahead.
        3. We apply this expanded filter to the original LazyFrame.
        4. We apply the expressions to the lazy frame, then the original filters and the specified with_columns.
        5. We collect and return the LazyFrame.

    This is useful for cases involving rolling windows, forward fills, or other expressions that depend on a history, or prevent pushdown of filters. Normally, polars will stop all pushdowns with these operations specified. However, we define this custom io source to maintain laziness, and allow extracting filters on "index_col" or on "linked_cols" to get applied before any of the predicate pushdown-blocking expressions are applied.

    Additionally, we can utilize "linked_cols" with no expressions specified to convert filters from the linked columns to filters on the "index_col" with lookback and/or lookahead. This is useful for cases where "index_col" is the index column for time-based filtering, but we want to restrict to another column. A common case is where "Date" is the index column, but we want to filter for "NextOpenDate", where "Date" represents the date data was added to a database, while "NextOpenDate" represents the next date when the data is valid for trading. In this case, we can use "NextOpenDate" as a linked column, and convert filters on it to filters on "Date", expanded by lookback and optionally by lookahead to include near-future dates.

    Simple usage example:
        >>> from datetime import date, timedelta
        >>> import polars_io_tools.io_sources  # register .piot namespace
        >>> df = pl.LazyFrame({
        ...     "Date": [date(2025, 1, i) for i in range(1, 6)],
        ...     "EventDate": [date(2025, 1, i) for i in range(2, 7)],
        ...     "Value": [10, 20, 30, 40, 50],
        ... })
        >>> result = df.piot.ts_with_columns(
        ...     pl.col("Value").cum_sum().alias("CumValue"),
        ...     index_col="Date",
        ...     lookback=timedelta(days=3),
        ...     linked_cols=["EventDate"],
        ... )
        >>> result.filter(pl.col("EventDate") >= date(2025, 1, 5)).collect()
        shape: (2, 4)
        ┌────────────┬────────────┬───────┬──────────┐
        │ Date       ┆ EventDate  ┆ Value ┆ CumValue │
        │ ---        ┆ ---        ┆ ---   ┆ ---      │
        │ date       ┆ date       ┆ i64   ┆ i64      │
        ╞════════════╪════════════╪═══════╪══════════╡
        │ 2025-01-04 ┆ 2025-01-05 ┆ 40    ┆ 90       │
        │ 2025-01-05 ┆ 2025-01-06 ┆ 50    ┆ 140      │
        └────────────┴────────────┴───────┴──────────┘

    In the above example, we filter for ``EventDate >= 2025-01-05``.  The filter
    on "EventDate" is converted to a filter on "Date", expanded by 3 days
    backwards (``Date >= 2025-01-05 - 3d = 2025-01-02``), and the cumulative sum
    expression is applied to "Value".  Then, the original filter on "EventDate"
    is applied.  This means the cum_sum sees rows starting from Date 2025-01-02
    (Values 20, 30, 40, 50), giving CumValues of 90 (20+30+40) and 140
    (20+30+40+50).  A plain ``with_columns`` + filter would instead compute the
    cum_sum over all rows first (giving 100 and 150, including Value=10 from
    2025-01-01).

    Args:
        self (pl.LazyFrame): The input LazyFrame
        *exprs (pl.Expr | List[pl.Expr] | Callable[[pl.LazyFrame], pl.LazyFrame]): Expression(s) to apply to the dataframe (like what can be passed into `df.with_columns`) or a callable taking a LazyFrame and returning a LazyFrame (applied via `pipe`). These expressions are applied for filters on `index_col`, but before the rest of the predicate pushed down by Polars. This can be an empty list if no extra expressions are needed.
        index_col (str): The main date column for time-based filtering. We will extract filters on this column from linked_cols and convert them to filters on the "index_col", with lookback.
        lookback (timedelta): How far back to look from the filter dates. We convert filters on linked_cols into filters on the "index_col" by expanding the date range by this lookback period. This expanded range is then used as a filter before applying the expressions.
        lookahead (timedelta): How far forward to extend from the filter dates. Similar to lookback, this expands the upper bound of the converted intervals so downstream operations can access near-future data when needed.
        linked_cols (List[str]): Columns whose filters should be converted to index_col filters. Specifically, all filters on these columns will be relaxed by looking backwards by `lookback` amount and applied to the `index_col` instead.
        log_explain (bool, default False): If True, logs the query plans for debugging purposes.
        _disable_optimizations (bool, default False): If True, replaces ts_with_columns by with_columns, disabling the push down optimizations we perform.
        expressions (pl.Expr | List[pl.Expr] | Callable[[pl.LazyFrame], pl.LazyFrame], default None): Deprecated.

    Returns:
        pl.LazyFrame: A LazyFrame with the expressions applied after efficient filtering
    """
    if index_col is None:
        if len(exprs) == 0:
            raise ValueError("index_col must be specified.")

        index_col = exprs[-1]
        exprs = exprs[:-1]
        assert isinstance(index_col, str)

    if expressions is not None:
        if len(exprs):
            raise ValueError("expressions and exprs can't both be used as the same time.")

        log.warning("Calling ts_with_columns with keyword argument expressions is now deprecated.")

    if len(exprs) == 1 and (isinstance(exprs[0], list) or callable(exprs[0])):
        expressions = exprs[0]

    if expressions is None:
        expressions = list(exprs)

    if _disable_optimizations:
        return self.with_columns(expressions) if not callable(expressions) else self.pipe(expressions)

    # Build FilterSpecs: all linked_cols + index_col map to the same source column (index_col)
    # with the same lookback/lookahead. This is equivalent to the old behavior where filters
    # from all these columns were intersected and applied to index_col.
    lb = lookback or datetime.timedelta()
    la = lookahead or datetime.timedelta()

    filter_specs: dict[str, FilterSpec] = {
        index_col: FilterSpec(lookback=lb, lookahead=la),
    }
    for col in linked_cols or []:
        filter_specs[col] = FilterSpec(source_col=index_col, lookback=lb, lookahead=la)

    def combine(sources: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
        lf = sources["main"]
        # Validate index_col lazily (at collect time) to avoid forcing schema
        # resolution on the input LazyFrame when ts_with_columns is constructed.
        schema = lf.collect_schema()
        if index_col not in schema or schema[index_col] not in (pl.Date, pl.Datetime):
            error = f"Expected index_col '{index_col}' to be in the schema with type Date or Datetime, "
            if index_col not in schema:
                error += "but found it was missing from the schema."
            else:
                error += f"but found type {schema[index_col]}."
            raise ValueError(error)

        if expressions:
            if callable(expressions):
                lf = lf.pipe(expressions)
            else:
                lf = lf.with_columns(expressions)
        return lf

    return multi_source(
        sources={"main": (self, filter_specs)},
        combine=combine,
        log_explain=log_explain,
    )
