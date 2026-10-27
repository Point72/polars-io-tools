import datetime
import logging
from functools import partial
from typing import Annotated

import polars as pl
import portion
from portion import Bound, Interval
from pydantic import AfterValidator, Field, TypeAdapter

# Import from original code
from .base import (
    FAILED_LITERAL_RESULT,
    AliasNode,
    BaseExprNode,
    BinaryExprNode,
    CastNode,
    ColumnNode,
    ExprVisitor,
    FunctionNode,
    TernaryNode,
    get_parsed_expr,
    has_exact_value,
)
from .enum import (
    BooleanFunctionType,
    OperatorType,
    TemporalFunctionType,
)

# Configure logging
log = logging.getLogger(__name__)

# Export only what's needed publicly
__all__ = (
    "convert_expr_to_datetime_range",
    "convert_expr_to_range",
)

# Type definition for validated datetime
ValidatedDatetime = Annotated[
    datetime.datetime,
    AfterValidator(lambda v: v if v.tzinfo is None else v.astimezone(datetime.UTC).replace(tzinfo=None)),
    Field(description="Validated datetime object, timezone-naive but interpreted as UTC"),
]

_VALIDATED_DATETIME = TypeAdapter(ValidatedDatetime)


# Special function for datetime interval creation with date expansion
def _create_datetime_point(value, expand_dates: bool = False, preserve_dates: bool = False) -> Interval:
    """
    Create a datetime interval with special handling for date objects.

    Args:
        value: The datetime or date value
        expand_dates: Whether to expand dates to full-day intervals
        preserve_dates: When True, ``date`` (not ``datetime``) values are kept as ``date`` singletons rather than
            promoted to midnight ``datetime``. Lets callers distinguish bounds that originated from a date literal so
            they can widen them appropriately when comparing against a ``Datetime`` column. ``preserve_dates`` takes
            precedence over ``expand_dates`` when both are set.
    """
    # Expand dates to full-day ranges if needed
    # We only do this on values that are dates, not values that could validate to a date.
    if isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
        is_date = True
    else:
        is_date = False

    if preserve_dates and is_date:
        return _create_point_interval(value)

    dt_val = _VALIDATED_DATETIME.validate_python(value)

    if expand_dates and is_date:
        start_value = dt_val
        end_value = datetime.datetime.combine(value + datetime.timedelta(days=1), datetime.time())
        return Interval.from_atomic(Bound.CLOSED, start_value, end_value, Bound.OPEN)

    return _create_point_interval(dt_val)


# Helper functions for interval creation
def _create_point_interval(value) -> Interval:
    """Create an interval containing only the given value."""
    return portion.singleton(value)


_DAY = datetime.timedelta(days=1)


def _extract_column(node: BaseExprNode) -> tuple[str | None, datetime.timedelta]:
    """Column under aliases and monotonic casts, and the unit (or zero) that a cast or ``dt.date()`` floored it to.

    Other casts can wrap or reorder values, so no bound on the raw column follows from them.
    """
    floor = datetime.timedelta(0)
    cur = node
    while len(cur.expr.meta.root_names()) == 1:
        if isinstance(cur, ColumnNode):
            return cur.expr.meta.root_names()[0], floor
        if isinstance(cur, FunctionNode) and cur.function_type == TemporalFunctionType.DATE and len(cur.inputs) == 1:
            floor, cur = _DAY, cur.inputs[0]
        elif isinstance(cur, CastNode) and cur.dtype == pl.Date:
            floor, cur = _DAY, cur.input
        elif isinstance(cur, CastNode) and isinstance(cur.dtype, pl.Datetime) and cur.dtype.time_unit in ("us", "ns"):
            # A cast to us floors an ns column, and a cast from a coarser unit makes literals finer than the column.
            floor = max(floor, datetime.timedelta(microseconds=1))
            cur = cur.input
        elif isinstance(cur, AliasNode) or (isinstance(cur, CastNode) and cur.dtype == pl.Float64):
            cur = cur.input
        else:
            break
    return None, datetime.timedelta(0)


_US = datetime.timedelta(microseconds=1)


def _next_microsecond(value: datetime.datetime) -> datetime.datetime:
    """The instant one microsecond after ``value``, added on the UTC timeline when ``value`` is timezone-aware.

    Aware arithmetic in Python is wall-clock and resets ``fold``, so inside a repeated DST hour it would move the instant.
    """
    if value.tzinfo is not None:
        value = value.astimezone(datetime.UTC)
    return value + _US


def _is_bound(value, floor: datetime.timedelta) -> bool:
    """Whether a non-null literal can bound the column: a ``date`` or ``datetime`` matching the column's floor, if any."""
    if value is None:
        return False
    if floor == _DAY:
        return isinstance(value, datetime.date) and not isinstance(value, datetime.datetime)
    return not floor or isinstance(value, datetime.datetime)


def _rounded_bound_value(node: BaseExprNode, *, round_up: bool):
    """A literal's value usable as a range bound, or ``None`` if it cannot bound the column.

    An exact literal yields its value unchanged. A Python ``datetime`` holds only microseconds, so a nanosecond literal
    with a sub-microsecond part is not exact; Polars floors its extracted value toward the past. For a range bound such
    a value can still be used by rounding to microseconds away from the kept side: up to the next microsecond for an
    upper bound (``round_up``) and left at the floored value for a lower bound. Either keeps the bound a superset of the
    true one. Other inexact literals return ``None`` (unusable): ``NaN`` has no position, and ``timedelta`` is excluded
    because Polars converts a negative ``Duration`` toward zero rather than flooring, breaking the superset guarantee.
    """
    if has_exact_value(node):
        return node.value
    value = node.value
    if isinstance(value, datetime.datetime):
        return _next_microsecond(value) if round_up else value
    return None


def _unfloor(interval: Interval, floor: datetime.timedelta) -> Interval:
    """Widen bounds on a column floored to ``floor`` to bounds on the column: a closed upper bound ``v`` becomes ``< v + floor``.

    Under a sub-day floor the upper bound is kept closed (``<= v + floor``): the column may have a coarser unit than the
    literal, and Polars truncates the literal to the column's unit, which would turn ``< t + 1us`` into ``< t``.
    """

    def widen(atomic: Interval) -> Interval:
        if not floor or atomic.upper == portion.inf:
            return atomic
        upper = atomic.upper
        if atomic.right == Bound.CLOSED:
            try:
                upper = upper + floor
            except OverflowError:
                upper = portion.inf
        return atomic.replace(upper=upper, right=Bound.OPEN if floor == _DAY else Bound.CLOSED)

    return interval.apply(widen)


class IntervalVisitor(ExprVisitor[Interval]):
    """
    Visitor that extracts interval constraints for a specific column from expressions.
    """

    def __init__(self, target_column: str, create_point_func=None, validate_value_func=None, constraints=None):
        """
        Initialize with a target column and point creation function.

        Args:
            target_column: The name of the column to extract range constraints for
            create_point_func: Function to create a point interval (with special handling if needed)
        """
        self.target_column = target_column
        self.create_point = create_point_func or _create_point_interval
        self.validate_value_func = validate_value_func
        # Start with universe interval (-infinity to infinity) if interval not passed.
        self.constraints = portion.closed(-portion.inf, portion.inf) if constraints is None else constraints

    def copy(self) -> "IntervalVisitor":
        """Create a copy of the visitor with the same constraints."""
        return IntervalVisitor(self.target_column, self.create_point, self.validate_value_func, self.constraints)

    def default_result(self) -> Interval:
        """Return the current constraints."""
        return self.constraints

    def _microsecond_window(self, value) -> Interval | None:
        """The closed microsecond window ``[floor, floor + 1us]`` around a sub-microsecond datetime value, else ``None``.

        A nanosecond ``datetime`` literal with a sub-microsecond part has no exact Python ``datetime``; Polars floors its
        extracted value to the microsecond. The window from that floor to the next microsecond contains the true value,
        so an equality (or ``is_in`` membership) against it stays a superset while still pruning. Returns ``None`` for
        non-datetime values (e.g. ``NaN``, or a ``timedelta`` whose sign-dependent rounding would break the superset),
        leaving the column unconstrained.
        """
        if not isinstance(value, datetime.datetime):
            return None
        lower = value.astimezone(datetime.UTC) if value.tzinfo is not None else value
        upper = _next_microsecond(value)
        if self.validate_value_func is not None:
            lower, upper = self.validate_value_func(lower), self.validate_value_func(upper)
        return Interval.from_atomic(Bound.CLOSED, lower, upper, Bound.CLOSED)

    def visit_binary_expr(self, node: BinaryExprNode) -> None:
        """
        Visit binary expressions (comparisons, logical operations).
        """
        # Skip if not related to our column
        if self.target_column not in node.expr.meta.root_names():
            return

        if node.op.is_comparison():
            # Handle comparisons between column and literal
            column, floor = _extract_column(node.left)
            if column != self.target_column or not _is_bound(node.right.value, floor):
                return
            raw_value = node.right.value
            is_eq = node.op in (OperatorType.EQ, OperatorType.EQ_VALIDITY)
            if not is_eq and isinstance(raw_value, str):
                # An Enum column orders strings by category, not lexically.
                return

            try:
                if is_eq:
                    if has_exact_value(node.right):
                        # Use create_point to handle date expansion when needed. We use the raw value here to allow
                        # for custom behavior before validation.
                        interval = self.create_point(raw_value)
                    else:
                        # A sub-microsecond nanosecond literal has no exact Python datetime. Rather than drop the
                        # equality (losing all pruning), widen the point to its enclosing microsecond window
                        # [floor, floor + 1us], which still contains the true value.
                        interval = self._microsecond_window(node.right.value)
                        if interval is None:
                            return
                else:
                    # An inequality is a one-sided bound, so an inexact temporal literal can be rounded to microseconds
                    # toward the dropped side and still keep every matching row.
                    bound_value = _rounded_bound_value(node.right, round_up=node.op in (OperatorType.LT, OperatorType.LT_EQ))
                    if bound_value is None:
                        return
                    value = self.validate_value_func(bound_value) if self.validate_value_func is not None else bound_value
                    if node.op == OperatorType.GT:
                        # x > value (exclusive lower bound)
                        interval = Interval.from_atomic(Bound.OPEN, value, portion.inf, Bound.OPEN)
                    elif node.op == OperatorType.GT_EQ:
                        # x >= value (inclusive lower bound)
                        interval = Interval.from_atomic(Bound.CLOSED, value, portion.inf, Bound.CLOSED)
                    elif node.op == OperatorType.LT:
                        # x < value (exclusive upper bound)
                        interval = Interval.from_atomic(Bound.CLOSED, -portion.inf, value, Bound.OPEN)
                    elif node.op == OperatorType.LT_EQ:
                        # x <= value (inclusive upper bound)
                        interval = Interval.from_atomic(Bound.CLOSED, -portion.inf, value, Bound.CLOSED)
                    else:
                        return
                self.constraints &= _unfloor(interval, floor)

            except Exception as e:  # noqa: BLE001 -- intentional broad catch (defensive fallback)
                log.info(f"Failed to create comparison constraint for value {raw_value}: {e}")

        elif node.op in [OperatorType.AND, OperatorType.LOGICAL_AND]:
            # For AND, visit both sides with the current constraints
            left_visitor = self.copy()
            left_visitor.visit(node.left)

            # Update constraints from left side first
            self.constraints = left_visitor.constraints

            # Now visit right side with updated constraints
            right_visitor = self.copy()
            right_visitor.visit(node.right)

            # Update with results from right side
            self.constraints = right_visitor.constraints

        elif node.op in [OperatorType.OR, OperatorType.LOGICAL_OR]:
            # For OR, visit both sides independently and take union
            left_visitor = self.copy()
            left_visitor.visit(node.left)

            right_visitor = self.copy()
            right_visitor.visit(node.right)

            # Take union of both sides (uses portion's | operator)
            self.constraints = left_visitor.constraints | right_visitor.constraints

    def visit_function(self, node: FunctionNode) -> None:
        """
        Visit function nodes (IS_BETWEEN, IS_IN).
        """
        if self.target_column not in node.expr.meta.root_names():
            return

        if not isinstance(node.function_type, BooleanFunctionType):
            return

        if node.function_type == BooleanFunctionType.IS_BETWEEN:
            # Handle BETWEEN function
            column, floor = _extract_column(node.inputs[0])
            if column != self.target_column or not all(_is_bound(n.value, floor) and not isinstance(n.value, str) for n in node.inputs[1:3]):
                return

            # Round an inexact temporal bound toward the dropped side (lower down, upper up) to keep a superset.
            lower_value = _rounded_bound_value(node.inputs[1], round_up=False)
            upper_value = _rounded_bound_value(node.inputs[2], round_up=True)
            if lower_value is None or upper_value is None:
                return

            try:
                if self.validate_value_func is not None:
                    lower = self.validate_value_func(lower_value)
                    upper = self.validate_value_func(upper_value)
                else:
                    lower = lower_value
                    upper = upper_value
                closed = node.options.get("closed", "Both")
                # Create range based on closed parameter
                if closed == "Both":
                    between_interval = Interval.from_atomic(Bound.CLOSED, lower, upper, Bound.CLOSED)
                elif closed == "Left":
                    between_interval = Interval.from_atomic(Bound.CLOSED, lower, upper, Bound.OPEN)
                elif closed == "Right":
                    between_interval = Interval.from_atomic(Bound.OPEN, lower, upper, Bound.CLOSED)
                else:  # "Neither"
                    between_interval = Interval.from_atomic(Bound.OPEN, lower, upper, Bound.OPEN)

                self.constraints &= _unfloor(between_interval, floor)

            except Exception as e:  # noqa: BLE001 -- intentional broad catch (defensive fallback)
                log.info(f"Failed to create BETWEEN constraint: {e}")

        elif node.function_type == BooleanFunctionType.IS_IN:
            # Handle IN function
            column, floor = _extract_column(node.inputs[0])
            if column != self.target_column or not node.inputs[1].can_extract_literal or node.inputs[1].value is FAILED_LITERAL_RESULT:
                return

            # We do not validate the values here since we pass them to
            # create_point directly
            values = node.inputs[1].value
            # A one-element series literal extracts as a scalar.
            values = values if isinstance(values, list) else [values]
            if not values:
                # Empty IN list is always false - set to empty
                self.constraints = portion.empty()
                return
            # A null (matched by ``nulls_equal=True``) or NaN value has no interval, so leave the column unconstrained.
            if not all(_is_bound(value, floor) for value in values):
                return
            # Per-member exactness, aligned with ``values``: a sub-microsecond nanosecond member widens to its
            # enclosing microsecond window (like an equality) instead of forcing the whole membership to be dropped.
            series = pl.select(node.inputs[1].expr).to_series()
            if isinstance(series.dtype, pl.List):
                if len(series) != 1:
                    return
                series = series[0]
            if len(series) != len(values) or (series.dtype.is_float() and series.is_nan().any()):
                return
            member_is_exact = (series == pl.Series(series.to_list(), dtype=series.dtype)).to_list() if series.dtype.is_temporal() else None

            # Create union of intervals for each value
            in_interval = portion.empty()
            for i, value in enumerate(values):
                try:
                    if member_is_exact is None or member_is_exact[i]:
                        # Use create_point to handle date expansion when needed
                        point_interval = self.create_point(value)
                    else:
                        point_interval = self._microsecond_window(value)
                        if point_interval is None:
                            return
                    in_interval |= point_interval
                except Exception as e:  # noqa: BLE001 -- intentional broad catch (defensive fallback)
                    log.info(f"Failed to add IN value {value}: {e}")
                    return

            self.constraints &= _unfloor(in_interval, floor)

    def visit_ternary(self, node: TernaryNode) -> None:
        """
        Visit ternary expressions (when/then/otherwise).
        """
        if self.target_column not in node.expr.meta.root_names():
            return

        # For simplicity, we currently ignore the restrictions in the predicate.

        # Truthy branch with predicate constraints
        truthy_visitor = self.copy()
        truthy_visitor.visit(node.truthy)

        # Handle falsy branch separately
        falsy_visitor = self.copy()
        falsy_visitor.visit(node.falsy)

        # Combine the branches with OR (union)
        self.constraints = truthy_visitor.constraints | falsy_visitor.constraints

    def visit_cast(self, node: CastNode) -> None:
        """Handle cast expressions."""
        if self.target_column not in node.expr.meta.root_names():
            return
        if node.dtype == pl.Boolean:
            # If casting to boolean, process the underlying expression
            # The constraint modification will happen based on the input node.
            self.visit(node.input)
        # For non-boolean casts, we do not modify the constraints.


# Public facing functions
def convert_expr_to_range(expr_or_node: pl.Expr | BaseExprNode, column_name: str, create_point_func=None, validate_value_func=None) -> Interval:
    """
    Extract valid ranges for a specific column from filters.

    Args:
        expr_or_node: Either a Polars expression or a parsed expression node
        column_name: Column to extract ranges for
        create_point_func: Function to create point intervals (with special handling if needed)

    Returns:
        A Interval representing the valid ranges for the column
    """
    node = get_parsed_expr(expr_or_node)
    visitor = IntervalVisitor(
        column_name,
        create_point_func,
        validate_value_func,
    )
    visitor.visit(node)
    return visitor.process_results()


def convert_expr_to_datetime_range(
    expr_or_node: pl.Expr | BaseExprNode,
    column_name: str,
    coerce_date_to_datetime: bool = True,
    get_enclosure: bool = True,
    preserve_dates: bool = False,
) -> Interval:
    """
    Extract the valid datetime range for a specific column from filters. This returns the raw Interval, which will be a disjunction of non-overlapping bounds.

    Args:
        expr_or_node: Either a Polars expression or a parsed expression node
        column_name: Column to extract date range for
        coerce_date_to_datetime: Whether to treat date equality as a range spanning the whole day
        get_enclosure: Whether we return the interval as is, or an atomic interval covering the entire range.
        preserve_dates: When True, ``date`` literals are kept as ``date`` bounds rather than silently promoted to
            midnight ``datetime``. Used by callers that compare against a ``Datetime`` source column and need to widen
            date bounds to full-day datetime ranges (see :func:`_extend_dates_to_full_datetimes`); other callers should
            leave this False to preserve backwards-compatible datetime-typed bounds.
    """
    _dt_to_point_interval = partial(_create_datetime_point, expand_dates=not coerce_date_to_datetime, preserve_dates=preserve_dates)

    def _validate(value):
        if preserve_dates and isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
            return value
        return _VALIDATED_DATETIME.validate_python(value)

    interval = convert_expr_to_range(expr_or_node, column_name, _dt_to_point_interval, _validate)
    if interval.empty or not get_enclosure:
        return interval
    return interval.enclosure


def _lookback_interval(interval: Interval, lookback: datetime.timedelta) -> Interval:
    """Extend intervals backward by ``lookback``.

    For each atomic interval, if the lower bound is finite, shift it earlier by
    the provided timedelta. Infinite lower bounds are left unchanged.

    This allows upstream predicate pushdown to include historical data needed
    for operations like rolling windows or forward-fill.
    """

    def _internal_apply(atomic_interval: Interval) -> Interval:
        if atomic_interval.lower == -portion.inf:
            return atomic_interval  # return as unchanged
        try:
            lower = atomic_interval.lower - lookback
        except OverflowError:
            lower = -portion.inf
        return atomic_interval.replace(lower=lower)

    return interval.apply(_internal_apply)


def _lookahead_interval(interval: Interval, lookahead: datetime.timedelta) -> Interval:
    """Extend intervals forward by ``lookahead``.

    For each atomic interval, if the upper bound is finite, shift it later by
    the provided timedelta. Infinite upper bounds are left unchanged.

    This complements ``lookback`` by allowing predicate pushdown to include
    near-future data needed for endpoint-sensitive computations.
    """

    def _internal_apply(atomic_interval: Interval) -> Interval:
        if atomic_interval.upper == portion.inf:
            return atomic_interval  # return as unchanged
        try:
            upper = atomic_interval.upper + lookahead
        except OverflowError:
            upper = portion.inf
        return atomic_interval.replace(upper=upper)

    return interval.apply(_internal_apply)


def _extend_to_full_dates(interval: Interval) -> Interval:
    """Floor datetime bounds to dates for a ``Date`` column, closing an open upper bound after midnight so its date is kept."""

    def _internal_apply(atomic_interval: Interval) -> Interval:
        lower = atomic_interval.lower.date() if isinstance(atomic_interval.lower, datetime.datetime) else atomic_interval.lower
        upper, right = atomic_interval.upper, atomic_interval.right
        if isinstance(upper, datetime.datetime):
            if upper.time() != datetime.time.min:
                right = portion.CLOSED  # ``d < 2024-01-01 12:00`` keeps the date 2024-01-01.
            upper = upper.date()
        return atomic_interval.replace(lower=lower, upper=upper, right=right)

    return interval.apply(_internal_apply)


def _promote_dates_to_datetimes(interval: Interval) -> Interval:
    """Promote ``date`` (not ``datetime``) bounds to midnight ``datetime`` bounds.

    Inverse of :func:`convert_expr_to_datetime_range`'s ``preserve_dates=True`` mode: callers that ultimately want
    datetime semantics (e.g., to apply sub-day ``lookback`` arithmetic) can re-promote bounds with this helper.
    Datetime and infinite bounds pass through unchanged.
    """

    def _is_pure_date(value) -> bool:
        return isinstance(value, datetime.date) and not isinstance(value, datetime.datetime)

    def _internal_apply(atomic_interval: Interval) -> Interval:
        lower = atomic_interval.lower
        upper = atomic_interval.upper
        if _is_pure_date(lower):
            lower = datetime.datetime.combine(lower, datetime.time.min)
        if _is_pure_date(upper):
            upper = datetime.datetime.combine(upper, datetime.time.min)
        return atomic_interval.replace(lower=lower, upper=upper)

    return interval.apply(_internal_apply)


def _extend_dates_to_full_datetimes(interval: Interval) -> Interval:
    """Widen ``date`` (not ``datetime``) bounds so they cover full days when the target column is ``Datetime``.

    Polars promotes a bare ``date`` literal to midnight ``datetime`` when comparing against a ``Datetime`` column, which
    silently drops every intraday row on the bound day for upper-closed bounds (and includes nothing for upper-open
    bounds). This helper rewrites date bounds into the equivalent datetime bounds that preserve full-day semantics:

      - lower closed at date ``d``  ->  closed at ``datetime(d, 00:00)``
      - lower open   at date ``d``  ->  closed at ``datetime(d + 1d, 00:00)``
      - upper closed at date ``d``  ->  open   at ``datetime(d + 1d, 00:00)``
      - upper open   at date ``d``  ->  open   at ``datetime(d, 00:00)``

    Datetime and infinite bounds pass through unchanged, so a date lookback that produced a sub-day datetime bound
    retains its sub-day precision.
    """

    def _is_pure_date(value) -> bool:
        return isinstance(value, datetime.date) and not isinstance(value, datetime.datetime)

    def _internal_apply(atomic_interval: Interval) -> Interval:
        lower = atomic_interval.lower
        upper = atomic_interval.upper
        left = atomic_interval.left
        right = atomic_interval.right

        if _is_pure_date(lower):
            if left == portion.CLOSED:
                lower = datetime.datetime.combine(lower, datetime.time.min)
            else:
                lower = datetime.datetime.combine(lower + datetime.timedelta(days=1), datetime.time.min)
                left = portion.CLOSED

        if _is_pure_date(upper):
            if right == portion.CLOSED:
                upper = datetime.datetime.combine(upper + datetime.timedelta(days=1), datetime.time.min)
                right = portion.OPEN
            else:
                upper = datetime.datetime.combine(upper, datetime.time.min)

        return atomic_interval.replace(lower=lower, upper=upper, left=left, right=right)

    return interval.apply(_internal_apply)


def _convert_atomic_interval_to_polars_expr(
    atomic_interval: Interval,
    index_col: str,
) -> pl.Expr | None:
    """Convert a single atomic portion.Interval to a Polars expression for filtering on a date column. We assume the atomic interval already has the correct types for the bounds."""
    if atomic_interval.lower == -portion.inf and atomic_interval.upper == portion.inf:
        return None  # No restriction, always true
    left, right = atomic_interval.left, atomic_interval.right
    if atomic_interval.lower == -portion.inf:
        if right == portion.OPEN:
            return pl.col(index_col) < atomic_interval.upper
        return pl.col(index_col) <= atomic_interval.upper
    elif atomic_interval.upper == portion.inf:
        if left == portion.OPEN:
            return pl.col(index_col) > atomic_interval.lower
        return pl.col(index_col) >= atomic_interval.lower

    if left == portion.OPEN and right == portion.OPEN:
        closed = "none"
    elif left == portion.OPEN and right == portion.CLOSED:
        closed = "right"
    elif left == portion.CLOSED and right == portion.OPEN:
        closed = "left"
    else:
        closed = "both"
    return pl.col(index_col).is_between(atomic_interval.lower, atomic_interval.upper, closed=closed)


def _convert_interval_to_polars_expr(
    interval: Interval,
    index_col: str,
) -> pl.Expr | None | bool:
    """Convert a portion.Interval to a Polars expression for filtering.

    The interval is composed of disjoint atomic intervals.

    Returns:
        pl.Expr: A filter expression for bounded intervals.
        None: Universe interval (no restriction needed).
        False: Empty interval (nothing can match — caller decides how to handle).
    """
    if interval.empty:
        return False

    conditions = []
    for atomic_interval in interval:
        new_condition = _convert_atomic_interval_to_polars_expr(atomic_interval, index_col)
        if new_condition is not None:
            conditions.append(new_condition)

    if not conditions:
        return None  # Universe interval — no restriction
    return pl.any_horizontal(conditions) if len(conditions) > 1 else conditions[0]
