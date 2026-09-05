"""
Coordinated filter pushdown across a user-defined combine of multiple sources.

This module provides the ``pushdown_combine`` function for creating LazyFrames that
combine multiple data sources while automatically propagating and transforming
filters to each source appropriately.

Example usage::

    import polars_io_tools as cpl
    from polars_io_tools import pushdown_combine, FilterSpec

    lf = pushdown_combine(
        sources={
            "left": (left_lf, {
                "date": FilterSpec(),
                "id": FilterSpec(),
            }),
            "right": (right_lf, {
                "date": FilterSpec(lookback=timedelta(days=5)),
                "id": FilterSpec(source_col="identifier"),
            }),
        },
        combine=lambda s: s["left"].join(s["right"], on=["date", "id"]),
    )

    # Filters automatically propagate to both sources with transformations
    result = lf.filter(pl.col("date") > date(2024, 1, 1)).collect()
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any

import polars as pl
import portion

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
from .set_visitor import convert_expr_to_valid_values
from .util import collect_lf_in_io_source, register_io_source_with_is_pure

__all__ = ("FilterSpec", "IntervalFilterSpec", "pushdown_combine")

log = logging.getLogger(__name__)


@dataclass
class FilterSpec:
    """
    Specification for how to transform a filter on an output column for a specific source.

    When the user filters on an output column, this spec describes how to apply
    that filter to this particular source.

    Args:
        source_col (str | None): Column name in the source LazyFrame to filter on.
            If None, defaults to the output column name (the dict key).
            This makes the common case (same column name) very concise.

        lookback (timedelta): Expand the filter range backward by this amount.
            Used when the source needs historical data beyond the requested range
            (e.g., for rolling windows, forward fills, or return calculations).
            Only applies to temporal (date/datetime) filters.

        lookahead (timedelta): Expand the filter range forward by this amount.
            Used when the source needs future data beyond the requested range.
            Only applies to temporal (date/datetime) filters.

        value_mapping (dict[Any, Any] | Callable[[Any], Any] | None): Transform filter values before applying to source.
            Used when the source uses different values than the output column.
            Can be a dict for direct mapping or a callable for custom logic.
            Only applies to equality (==) and is_in() filters.

    Examples:
        Same column name, no transformation::

            {"date": FilterSpec()}

        Different column name in source::

            {"date": FilterSpec(source_col="DataDate")}

        Need 5 days of lookback::

            {"date": FilterSpec(lookback=timedelta(days=5))}

        Value mapping with dict::

            {"region": FilterSpec(source_col="region_code", value_mapping={"NORTH_AMERICA": "NA"})}

        Value mapping with callable::

            {"ticker": FilterSpec(value_mapping=str.upper)}
    """

    source_col: str | None = None
    lookback: timedelta = field(default_factory=timedelta)
    lookahead: timedelta = field(default_factory=timedelta)
    value_mapping: dict[Any, Any] | Callable[[Any], Any] | None = None


@dataclass
class IntervalFilterSpec:
    """
    Specification for pushing a filter into a validity-interval (history-of-change) source.

    Where :class:`FilterSpec` maps one output column to one source column, this maps a single
    output (request-date) column to a pair of source columns ``[start_col, end_col]`` that
    describe the window over which each source row is valid. When the user filters the output
    column to a range ``[lo, hi]``, the correct predicate is an *overlap*::

        start_col <= hi AND end_col >= lo          # closed="both"

    which cannot be expressed with a single-column ``FilterSpec``. This spec pushes that overlap
    to the source (so SQL/TickStore range-predicate pushdown is preserved) and, because the
    overlap is fully determined by each source row, resolves membership exactly at the source --
    consumers no longer re-implement ``start <= d <= end`` in their ``combine``.

    Args:
        start_col (str): Source column holding the window's inclusive lower bound.

        end_col (str): Source column holding the window's upper bound (see ``closed``).

        closed (str): Interval endpoint semantics -- one of ``"both"``, ``"left"``, ``"right"``,
            ``"none"``. The half-open variants flip the boundary comparison at the exact endpoints:

              - ``"both"``  (``[start, end]``): ``end >= lo`` and ``start <= hi``
              - ``"left"``  (``[start, end)``): ``end >  lo`` and ``start <= hi``
              - ``"right"`` (``(start, end]``): ``end >= lo`` and ``start <  hi``
              - ``"none"``  (``(start, end)``): ``end >  lo`` and ``start <  hi``

        value_mapping (dict[Any, Any] | Callable[[Any], Any] | None): Transform the request
            bounds ``lo``/``hi`` before pushdown, exactly as :class:`FilterSpec` does for discrete
            values (e.g. a business-day offset applied to the request date). A callable is applied to
            each finite bound; a dict maps finite bounds present in it and leaves absent bounds
            unmapped (that bound's predicate is then not pushed). ``None`` passes bounds through.

    Notes:
        The pushdown is **conservative** -- it may keep non-overlapping rows, but never drops an
        overlapping one. The request column is virtual (produced by another source in ``combine``,
        not by the interval source), so unlike :class:`FilterSpec` there is no post-combine filter on
        it to trim the surplus; exact per-row membership is the consumer's ``combine``. Two cases
        over-select at the source:

          - **Disjoint requests** (e.g. ``date in [Jan1..Jan5] OR [Feb1..Feb5]``) collapse to their
            outer hull ``[Jan1, Feb5]``, so windows overlapping only the gap survive the pushdown.
          - **A dict ``value_mapping`` missing a bound** skips that side's predicate entirely, so that
            side is left unconstrained.
    """

    start_col: str
    end_col: str
    closed: str = "both"
    value_mapping: dict[Any, Any] | Callable[[Any], Any] | None = None

    def __post_init__(self) -> None:
        if self.closed not in _CLOSED_MODES:
            raise ValueError(f"closed must be one of {sorted(_CLOSED_MODES)}, got {self.closed!r}")


def _map_bound(value: Any, mapping: dict | Callable | None) -> tuple[Any, bool]:
    """Map a single interval request bound through ``value_mapping``.

    Returns ``(mapped_value, ok)``. ``ok`` is False only for a dict mapping that lacks the key,
    signalling the caller not to push that bound's predicate (mirrors the ``FilterSpec`` dict-miss
    behavior of deferring rather than pushing an incomplete filter).
    """
    if mapping is None:
        return value, True
    if callable(mapping):
        return mapping(value), True
    if value in mapping:
        return mapping[value], True
    return value, False


def _apply_value_mapping(values: set[Any], mapping: dict | Callable | None) -> tuple[set[Any], set[Any]]:
    """Transform filter values using the provided mapping.

    Args:
        values (set[Any]): The values to transform
        mapping (dict | Callable | None): Either a dictionary for direct lookup, a callable to apply to each value,
            or None (returns values unchanged)

    Returns:
        tuple[set[Any], set[Any]]: A tuple of (mapped_values, unmapped_values).
            - For None mapping: all values are considered mapped (passthrough)
            - For callable mapping: all values are mapped via the callable
            - For dict mapping: only values present in the dict are mapped;
              values not in the dict are returned as unmapped (not pushed down)
    """
    if mapping is None:
        return values, set()
    if callable(mapping):
        return {mapping(v) for v in values}, set()
    # Dict mapping - only map values that are explicitly in the mapping
    # Values not in the mapping are NOT pushed down (safer behavior)
    mapped = set()
    unmapped = set()
    for v in values:
        if v in mapping:
            mapped.add(mapping[v])
        else:
            unmapped.add(v)
    return mapped, unmapped


# Endpoint comparison operators per ``closed`` mode. ``lower`` is the ``end_col >= lo`` side (left
# endpoint of the request overlap); ``upper`` is the ``start_col <= hi`` side (right endpoint).
_CLOSED_TO_LEFT = {"both": portion.CLOSED, "left": portion.OPEN, "right": portion.CLOSED, "none": portion.OPEN}
_CLOSED_TO_RIGHT = {"both": portion.CLOSED, "left": portion.CLOSED, "right": portion.OPEN, "none": portion.OPEN}
_CLOSED_MODES = frozenset(_CLOSED_TO_LEFT)


def _bound_expr(value: Any, source_col: str, source_dtype: pl.DataType, *, is_lower: bool, boundary) -> pl.Expr | None:
    """Build a single-sided comparison predicate for an interval bound.

    ``is_lower=True`` emits ``end_col >= / > value`` (closed / open ``boundary``); ``is_lower=False``
    emits ``start_col <= / < value``.

    When ``value`` is a ``date`` (not ``datetime``) but the source column is ``Datetime``, the request
    is a whole *day*, so the comparison point is resolved to a datetime that keeps full-day overlap
    semantics -- independently of the endpoint open/closed-ness, which only governs the operator:

      - lower side (``end_col`` vs the request-day start): compare against ``datetime(day, 00:00)``.
        End-closed (``both``/``right``) -> ``>=``; end-open (``left``/``none``) -> ``>`` so a window
        ending exactly at that midnight is dropped.
      - upper side (``start_col`` vs the request-day end): a window starting anywhere within the day
        overlaps it, so always compare ``start < datetime(day + 1, 00:00)`` regardless of ``boundary``.

    Passing an open-boundary *date* interval through the day-granular widener would instead shift the
    value a full day (reading "open" as "strictly after day d"), silently dropping boundary-day rows.
    """
    is_date_on_datetime = isinstance(source_dtype, pl.Datetime) and isinstance(value, date) and not isinstance(value, datetime)
    if is_date_on_datetime:
        if is_lower:
            point = datetime.combine(value, time.min)
            interval = portion.Interval.from_atomic(boundary, point, portion.inf, portion.OPEN)
        else:
            # Full-day upper: start strictly before the day after ``value`` (endpoint closed-ness is
            # immaterial for a day-wide request).
            point = datetime.combine(value + timedelta(days=1), time.min)
            interval = portion.Interval.from_atomic(portion.OPEN, -portion.inf, point, portion.OPEN)
    elif is_lower:
        interval = portion.Interval.from_atomic(boundary, value, portion.inf, portion.OPEN)
    else:
        interval = portion.Interval.from_atomic(portion.OPEN, -portion.inf, value, boundary)
    expr = _convert_interval_to_polars_expr(interval, source_col)
    # A one-sided finite interval is never empty/universe here, but stay defensive.
    return expr if isinstance(expr, pl.Expr) else None


def _apply_interval_filter(
    filtered_lf: pl.LazyFrame,
    spec: IntervalFilterSpec,
    date_range: portion.Interval,
    source_schema: dict[str, pl.DataType],
) -> pl.LazyFrame:
    """Apply an interval-overlap filter to a source LazyFrame.

    Given the request range ``date_range`` extracted for the output column, push the overlap
    ``start_col <= hi AND end_col >= lo`` (adjusted for ``spec.closed``) onto the source, mapping the
    request bounds through ``spec.value_mapping`` first. The overlap is fully determined per source
    row, so the surviving rows are exactly the overlapping windows -- no post-combine membership
    predicate is needed.
    """
    if spec.start_col not in source_schema or spec.end_col not in source_schema:
        log.debug(f"Interval columns {spec.start_col!r}/{spec.end_col!r} not both in source, skipping interval pushdown")
        return filtered_lf

    enclosure = date_range.enclosure
    lo, hi = enclosure.lower, enclosure.upper

    conditions: list[pl.Expr] = []

    # end_col >= lo  (drops expired/seed rows). Skipped for an unbounded lower request bound.
    if lo != -portion.inf:
        lo_mapped, ok = _map_bound(lo, spec.value_mapping)
        if ok:
            expr = _bound_expr(lo_mapped, spec.end_col, source_schema[spec.end_col], is_lower=True, boundary=_CLOSED_TO_LEFT[spec.closed])
            if expr is not None:
                conditions.append(expr)

    # start_col <= hi  (drops windows starting after the request). Skipped for an unbounded upper bound.
    if hi != portion.inf:
        hi_mapped, ok = _map_bound(hi, spec.value_mapping)
        if ok:
            expr = _bound_expr(hi_mapped, spec.start_col, source_schema[spec.start_col], is_lower=False, boundary=_CLOSED_TO_RIGHT[spec.closed])
            if expr is not None:
                conditions.append(expr)

    for cond in conditions:
        filtered_lf = filtered_lf.filter(cond)

    return filtered_lf


def _call_combine(
    combine: Callable[..., pl.LazyFrame],
    filtered_sources: dict[str, pl.LazyFrame],
    combine_kwargs: dict[str, Any] | None,
    sources_as_kwargs: bool,
) -> pl.LazyFrame:
    """Call the combine function with the appropriate signature.

    Args:
        combine (Callable[..., pl.LazyFrame]): The combine function
        filtered_sources (dict[str, pl.LazyFrame]): The filtered source LazyFrames
        combine_kwargs (dict[str, Any] | None): Additional keyword arguments to pass to combine
        sources_as_kwargs (bool): If True, pass sources as individual kwargs; if False, pass as a dict

    Returns:
        pl.LazyFrame: The combined LazyFrame
    """
    kwargs = combine_kwargs or {}
    if sources_as_kwargs:
        # Pass each source as a keyword argument
        return combine(**filtered_sources, **kwargs)
    else:
        # Pass sources as a single dict argument
        return combine(filtered_sources, **kwargs)


def _compute_output_schema(
    sources: dict[str, tuple[pl.LazyFrame, dict[str, FilterSpec | IntervalFilterSpec]]],
    combine: Callable[..., pl.LazyFrame],
    combine_kwargs: dict[str, Any] | None = None,
    sources_as_kwargs: bool = False,
) -> dict[str, pl.DataType]:
    """Compute the output schema by running combine on empty frames.

    This creates schema-only LazyFrames from each source and passes them
    through the combine function to determine the output schema without
    actually processing any data.

    Args:
        sources (dict[str, tuple[pl.LazyFrame, dict[str, FilterSpec | IntervalFilterSpec]]]): The source specifications
        combine (Callable[..., pl.LazyFrame]): The combine function
        combine_kwargs (dict[str, Any] | None): Additional keyword arguments to pass to combine
        sources_as_kwargs (bool): If True, pass sources as individual kwargs; if False, pass as a dict

    Returns:
        dict[str, pl.DataType]: The output schema
    """
    empty_sources = {name: pl.LazyFrame(schema=lf.collect_schema()) for name, (lf, _) in sources.items()}
    return _call_combine(combine, empty_sources, combine_kwargs, sources_as_kwargs).collect_schema()


def _get_source_col(output_col: str, spec: FilterSpec) -> str:
    """Get the source column name, defaulting to output column if not specified."""
    return spec.source_col if spec.source_col is not None else output_col


def _is_temporal_dtype(dtype: pl.DataType) -> bool:
    """Check if a dtype is temporal (Date or Datetime)."""
    return dtype in (pl.Date, pl.Datetime) or isinstance(dtype, pl.Datetime)


def pushdown_combine(
    sources: dict[str, tuple[pl.LazyFrame, dict[str, FilterSpec | IntervalFilterSpec]]],
    combine: Callable[..., pl.LazyFrame],
    *,
    combine_kwargs: dict[str, Any] | None = None,
    sources_as_kwargs: bool = False,
    log_explain: bool = False,
) -> pl.LazyFrame:
    """
    Create a LazyFrame from multiple sources with coordinated filter pushdown.

    When the returned LazyFrame is filtered and collected, filters are automatically
    transformed and applied to each source according to their FilterSpecs before
    the combine function is called.

    Args:
        sources (dict[str, tuple[pl.LazyFrame, dict[str, FilterSpec | IntervalFilterSpec]]]): Dictionary mapping source names to (LazyFrame, filter_specs) tuples.

            The filter_specs dict maps OUTPUT column names to FilterSpec objects
            that describe how to transform filters on that column for this source.

        combine (Callable[..., pl.LazyFrame]): Function that combines the filtered source LazyFrames into a single result.

            If ``sources_as_kwargs=False`` (default): Takes a dict of source LazyFrames
            as its first argument, plus any additional kwargs from ``combine_kwargs``.

            If ``sources_as_kwargs=True``: Takes each source LazyFrame as a keyword
            argument (using the source names as parameter names), plus any additional
            kwargs from ``combine_kwargs``.

            This function defines the join/transform logic and is called AFTER
            filters have been applied to each source.

            Common operations in combine: joins, with_columns, select, rename,
            drop, cast, group_by, unique, filter, etc. The final predicate is
            always applied after combine to ensure correct output.

        combine_kwargs (dict[str, Any] | None, default None): Additional keyword arguments to pass to the combine function.
            This allows parameterizing the combine logic without closures.

        sources_as_kwargs (bool, default False): If True, pass each source LazyFrame as a keyword argument to combine()
            using the source names as parameter names. If False (default), pass all
            sources as a single dict argument.

            Example with ``sources_as_kwargs=False`` (default)::

                def combine(sources, multiplier):
                    return sources["prices"].join(sources["rates"], on="date")

            Example with ``sources_as_kwargs=True``::

                def combine(prices, rates, multiplier):
                    return prices.join(rates, on="date")

        log_explain (bool, default False): If True, logs the query plans for debugging purposes.

    Returns:
        pl.LazyFrame: A LazyFrame that, when filtered and collected, will:
            1. Intercept filter predicates
            2. Transform and apply EXPANDED filters to each source (with lookback/lookahead)
            3. Call combine() with filtered sources and combine_kwargs
            4. Apply ORIGINAL predicate to final result (trims lookback rows to exact request)

    Notes:
        **Filter transformation pattern:**

        In all cases, TRANSFORMED filters are applied to sources, and the ORIGINAL filter
        is applied at the end after combine() runs. This ensures sources fetch the right data
        while the final output matches exactly what the user requested.

        **Lookback/lookahead (temporal expansion):**

        When a FilterSpec has lookback or lookahead, the source receives an EXPANDED filter
        to fetch additional data needed for calculations (such as rolling ones). After combine() runs,
        the ORIGINAL filter is applied to trim back to exactly what the user requested.

        Example: User filters ``date == "2024-01-05"`` with ``lookback=timedelta(days=3)``:

        1. Source receives filter ``date.is_between("2024-01-02", "2024-01-05")`` (lower bound expanded by 3 days)
        2. combine() runs on Jan 2-5 data (can compute lag/rolling values)
        3. Original filter ``date == "2024-01-05"`` applied, returning only Jan 5 row
           (but with lag values correctly computed from the historical data)

        **Value mapping (value transformation):**

        When a FilterSpec has value_mapping and/or source_col, the source receives a filter
        with MAPPED values on the SOURCE column. After combine() runs (which typically creates
        the output column), the ORIGINAL filter is applied.

        Example: User filters ``region == "NORTH_AMERICA"`` with
        ``FilterSpec(source_col="region_code", value_mapping={"NORTH_AMERICA": "NA"})``:

        1. Source receives filter ``region_code == "NA"`` (mapped value on source column)
        2. combine() runs and creates ``region`` column from ``region_code``
        3. Original filter ``region == "NORTH_AMERICA"`` applied, ensuring correctness

    Examples:
        Basic usage with lookback::

            lf = pushdown_combine(
                sources={
                    "prices": (prices_lf, {
                        "date": FilterSpec(lookback=timedelta(days=5)),
                        "symbol": FilterSpec(),
                    }),
                    "fundamentals": (fundamentals_lf, {
                        "date": FilterSpec(),
                        "symbol": FilterSpec(source_col="ticker"),
                    }),
                },
                combine=lambda s: s["prices"].join(s["fundamentals"], on=["date", "symbol"]),
            )

            # Filter propagates: prices gets 5-day lookback, fundamentals gets exact range
            result = lf.filter(pl.col("date").is_between(start, end)).collect()

        Using combine_kwargs for parameterized combine logic::

            def combine_with_region_mapping(sources, region_to_code):
                return sources["primary"].join(
                    sources["reference"].with_columns(
                        pl.col("region_code").replace(region_to_code).alias("region")
                    ),
                    on=["date", "region"],
                )

            REGION_TO_CODE = {"NORTH_AMERICA": "NA", "EUROPE": "EU"}

            lf = pushdown_combine(
                sources={
                    "primary": (primary_lf, {"date": FilterSpec(), "region": FilterSpec()}),
                    "reference": (reference_lf, {"date": FilterSpec(lookback=timedelta(days=5))}),
                },
                combine=combine_with_region_mapping,
                combine_kwargs={"region_to_code": REGION_TO_CODE},
            )

        Validity-interval source via :class:`IntervalFilterSpec`::

            lf = pushdown_combine(
                sources={
                    # window rows valid over [valid_from, valid_to]
                    "schedule": (schedule_lf, {"date": IntervalFilterSpec(start_col="valid_from", end_col="valid_to")}),
                },
                combine=lambda s: s["schedule"],
            )

            # Pushes the overlap start<=hi AND end>=lo to the source; only covering windows survive.
            result = lf.filter(pl.col("date").is_between(start, end)).collect()
    """
    # Compute output schema for the IO source.
    # Wrap in a lambda so that the schema (and the per-source `lf.collect_schema()`
    # calls it requires) is only resolved when Polars actually needs it (i.e. at
    # collect time), rather than eagerly when `pushdown_combine` is constructed.
    output_schema = lambda: _compute_output_schema(sources, combine, combine_kwargs, sources_as_kwargs)

    # Collect all output columns that have FilterSpecs across all sources
    all_output_cols: set[str] = set()
    for _, specs in sources.values():
        all_output_cols.update(specs.keys())

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        # Parse predicate to extract filter information
        parsed_predicate = get_parsed_expr(predicate) if predicate is not None else None

        # Extract filter values for each output column
        # We extract both temporal ranges and discrete values for each column
        extracted_ranges: dict[str, portion.Interval] = {}
        extracted_values: dict[str, set[Any] | None] = {}

        if parsed_predicate is not None:
            for col in all_output_cols:
                # Try to extract datetime range (returns universe interval if no constraints)
                try:
                    date_range = convert_expr_to_datetime_range(parsed_predicate, col, get_enclosure=False, preserve_dates=True)
                    # Only store if it's constraining (not the full universe)
                    if date_range != portion.closed(-portion.inf, portion.inf):
                        extracted_ranges[col] = date_range
                except Exception:  # noqa: BLE001 -- intentional broad catch (defensive fallback)
                    log.debug(f"Failed to extract date range for column {col}")

                # Try to extract discrete values
                try:
                    discrete_values = convert_expr_to_valid_values(parsed_predicate, col)
                    if discrete_values is not None:
                        extracted_values[col] = discrete_values
                except Exception:  # noqa: BLE001 -- intentional broad catch (defensive fallback)
                    log.debug(f"Failed to extract discrete values for column {col}")

        log.debug(f"Extracted ranges: {extracted_ranges}")
        log.debug(f"Extracted values: {extracted_values}")

        # Get source schemas for dtype checking. Resolved lazily here (at
        # collect time) rather than eagerly at pushdown_combine construction so
        # we don't force schema resolution on each source until we actually
        # need it.
        source_schemas: dict[str, dict[str, pl.DataType]] = {name: lf.collect_schema() for name, (lf, _) in sources.items()}

        # Apply transformed filters to each source
        # IMPORTANT: For temporal columns with lookback/lookahead, we apply the EXPANDED
        # filter here (e.g., date >= Jan 2 for a lookback of 3 days when user filters >= Jan 5).
        # This ensures the source fetches enough historical/future data for rolling calculations.
        # The ORIGINAL filter is applied at the end (after combine) to trim back to exactly
        # what the user requested. We skip discrete filters for temporal columns when there's
        # expansion, as they would undo the lookback/lookahead.
        filtered_sources: dict[str, pl.LazyFrame] = {}
        for source_name, (source_lf, specs) in sources.items():
            filtered_lf = source_lf
            source_schema = source_schemas[source_name]
            empty_temporal_range = False

            for output_col, spec in specs.items():
                # Interval specs reference two source columns and push an overlap predicate; handle them
                # separately from the single-column FilterSpec path below.
                if isinstance(spec, IntervalFilterSpec):
                    if output_col in extracted_ranges:
                        filtered_lf = _apply_interval_filter(filtered_lf, spec, extracted_ranges[output_col], source_schema)
                    continue

                source_col = _get_source_col(output_col, spec)

                # Skip if source column doesn't exist in this source
                if source_col not in source_schema:
                    log.debug(f"Source column {source_col} not in source {source_name}, skipping")
                    continue

                source_dtype = source_schema[source_col]

                # Track whether we applied a temporal filter with expansion. If so, we skip the discrete filter for this
                # column (it would undo the lookback/lookahead expansion).
                applied_temporal_with_expansion = False

                # Apply temporal filter with lookback/lookahead
                if output_col in extracted_ranges:
                    date_range = extracted_ranges[output_col]

                    # Only apply lookback/lookahead to temporal columns
                    if _is_temporal_dtype(source_dtype):
                        expanded_range = date_range
                        has_expansion = bool(spec.lookback) or bool(spec.lookahead)

                        if source_dtype == pl.Date:
                            # Date source: promote date bounds to midnight datetimes so sub-day lookback/lookahead nudges
                            # them across day boundaries; ``_extend_to_full_dates`` below floors them back to dates.
                            expanded_range = _promote_dates_to_datetimes(expanded_range)
                        elif isinstance(source_dtype, pl.Datetime):
                            # Datetime source with date-typed filter literal: widen date bounds to full-day datetime ranges
                            # before lookback/lookahead. ``date - timedelta(hours=18)`` truncates to whole days in Python,
                            # so sub-day shifts would otherwise be silently dropped, and the upper bound would collapse to
                            # midnight and drop intraday rows on the bound day.
                            expanded_range = _extend_dates_to_full_datetimes(expanded_range)

                        if spec.lookback:
                            expanded_range = _lookback_interval(expanded_range, spec.lookback)
                        if spec.lookahead:
                            expanded_range = _lookahead_interval(expanded_range, spec.lookahead)

                        # Floor interval bounds back to dates for Date sources so sub-day lookback/lookahead values are
                        # rounded to include the full day boundary (e.g., 2d 18h lookback includes the start of day 3).
                        if source_dtype == pl.Date:
                            expanded_range = _extend_to_full_dates(expanded_range)

                        temporal_filter = _convert_interval_to_polars_expr(expanded_range, source_col)
                        if temporal_filter is False:
                            # Empty temporal range — no rows can match.
                            # Defer head(0) until after all filters are applied so
                            # they still act as a safety net if head(0) doesn't
                            # get pushed down to the underlying source.
                            empty_temporal_range = True
                            log.debug(f"Empty temporal range for {source_name}.{source_col}, will apply head(0) after filters")
                            applied_temporal_with_expansion = has_expansion or isinstance(source_dtype, pl.Datetime)
                        elif temporal_filter is not None:
                            filtered_lf = filtered_lf.filter(temporal_filter)
                            log.debug(f"Applied temporal filter to {source_name}.{source_col}: {temporal_filter}")
                            applied_temporal_with_expansion = has_expansion or isinstance(source_dtype, pl.Datetime)

                # Apply discrete filter with value mapping
                # Skip for temporal columns if we already applied an expanded temporal filter (the discrete filter would
                # undo the lookback/lookahead expansion). Also skip for Datetime sources unconditionally: the temporal
                # filter already encodes the constraint with full-day widening for date literals, and a discrete filter
                # would re-promote date values to midnight and silently drop intraday rows.
                if output_col in extracted_values and not applied_temporal_with_expansion:
                    discrete_values = extracted_values[output_col]
                    if discrete_values is not None and len(discrete_values) > 0:
                        mapped_values, unmapped_values = _apply_value_mapping(discrete_values, spec.value_mapping)

                        # If there are ANY unmapped values, we cannot safely push down the filter.
                        # Pushing down only mapped values would exclude data that might match
                        # the unmapped values after combine() transforms the data.
                        # Let the final predicate handle all filtering after combine().
                        if unmapped_values:
                            log.debug(
                                f"Values {unmapped_values} for column '{output_col}' not found in "
                                f"value_mapping for source '{source_name}'; skipping filter pushdown "
                                f"entirely for this column (will be applied after combine)"
                            )
                        elif len(mapped_values) == 1:
                            # All values mapped, safe to push down equality filter
                            val = next(iter(mapped_values))
                            filtered_lf = filtered_lf.filter(pl.col(source_col) == val)
                            log.debug(f"Applied equality filter to {source_name}.{source_col}: == {val}")
                        elif len(mapped_values) > 1:
                            # All values mapped, safe to push down is_in filter
                            filtered_lf = filtered_lf.filter(pl.col(source_col).is_in(list(mapped_values)))
                            log.debug(f"Applied is_in filter to {source_name}.{source_col}: {mapped_values}")

            if empty_temporal_range:
                filtered_lf = filtered_lf.head(0)

            filtered_sources[source_name] = filtered_lf

        # Call user's combine function with filtered sources and any kwargs
        result_lf = _call_combine(combine, filtered_sources, combine_kwargs, sources_as_kwargs)

        if log_explain:
            log.debug(f"Combined LazyFrame plan before final filter:\n{result_lf.explain()}")

        # Apply original predicate to ensure exact output matches user's filter
        # This is critical: it trims any lookback/lookahead rows that were fetched for
        # rolling calculations, and handles any filters we couldn't push down.
        # Example: if user filters date == Jan 5 with 3-day lookback, the source fetched
        # Jan 2-5, combine ran (e.g., computed lag values), and now we filter to just Jan 5.
        #
        # An IntervalFilterSpec output column is a *virtual* request axis (e.g. "date") that maps to
        # source [start_col, end_col] and is not produced by combine, so it is absent from the output.
        # Its overlap was already resolved exactly at the source, so restrict the final predicate to
        # columns that actually exist in the output before applying it (a no-op for FilterSpec columns,
        # which combine does produce).
        if predicate is not None:
            output_cols = result_lf.collect_schema().names()
            restricted_predicate = restrict_expr_to_columns(predicate, output_cols)
            if restricted_predicate is not None:
                result_lf = result_lf.filter(restricted_predicate)

        # Select requested columns
        if with_columns is not None:
            result_lf = result_lf.select(with_columns)

        # Apply row limit
        if n_rows is not None:
            result_lf = result_lf.head(n_rows)

        if log_explain:
            log.debug(f"Final LazyFrame plan:\n{result_lf.explain()}")

        # Collect and yield
        try:
            yield from collect_lf_in_io_source(result_lf, batch_size)
        except Exception as e:
            err_msg = f"Failed during collection in pushdown_combine.\nPolars plan:\n{result_lf.explain()}\nError: {e.__class__.__name__}: {e}"
            raise RuntimeError(err_msg) from e

    return register_io_source_with_is_pure(source_generator, schema=output_schema)
