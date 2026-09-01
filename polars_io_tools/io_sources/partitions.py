"""Backend-neutral read-partition specs shared by the SQL and distributed readers.

A *read partition* is a slice of the input domain, defined by a polars predicate, that can be
read or executed independently. This differs from :class:`polars.io.partition.PartitionBy`,
which is write-side (it routes materialised rows to output files); these describe how to
*split a read* so a database scan or a distributed executor can process slices in parallel.

Partitions come from two places:

* **Eagerly** -- the caller enumerates them (an explicit value list, a hand-built
  ``list[ReadPartition]``).
* **Derived** -- the split depends on the predicate Polars pushes down at scan time (calendar
  windows from a date range, distinct values from an ``IN`` list). A :class:`Partitioner`
  builds the concrete list from that predicate.

Both collapse to the same currency -- a list of :class:`ReadPartition` -- so one vocabulary
drives both ``scan_db(..., partitions=...)`` and the Ray executor. A consumer that cannot honour
a partition's predicate (a database cannot run an arbitrary UDF, for instance) is expected to
reject it rather than silently drop the slice.
"""

from __future__ import annotations

import datetime
import itertools
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import polars as pl
import portion

from .range_visitor import convert_expr_to_datetime_range, convert_expr_to_range
from .set_visitor import convert_expr_to_valid_values

__all__ = (
    "KeyPartitions",
    "Partitioner",
    "ReadPartition",
    "by_key",
    "by_range",
    "by_time",
    "by_value",
    "cartesian_partitions",
    "discrete_partitions",
)


@dataclass(frozen=True)
class ReadPartition:
    """A single read partition: a predicate defining a disjoint slice of the input, plus a label.

    Args:
        predicate: The filter that defines this slice. Consumers apply it as an ordinary polars
            filter (distributed executor) or translate it to a SQL ``WHERE`` (database reader).
        key: An optional label used for ordering, error messages, and per-partition option
            lookup. For a multi-column key this is typically a tuple.
    """

    predicate: pl.Expr
    key: Any = None


@runtime_checkable
class Partitioner(Protocol):
    """Builds a list of :class:`ReadPartition` from the predicate pushed down at scan time.

    ``on`` names the column(s) the partitions filter on, so a consumer can keep them through
    projection pushdown. ``build`` returns ``None`` when no partitioning is possible (e.g. the
    predicate carries no bounded range on ``on``), signalling the consumer to fall back to a
    single unpartitioned read.
    """

    on: str

    def build(self, predicate: pl.Expr | None) -> list[ReadPartition] | None: ...


_WORD_TO_INTERVAL = {"day": "1d", "week": "1w", "month": "1mo", "quarter": "1q", "year": "1y"}
_INTERVAL_RE = re.compile(r"\d+(?:d|w|mo|q|y)")
_MAX_WINDOWS = 100_000


def _interval_str(every: str | int) -> str:
    """Normalise a bucket size to a validated Polars interval string (e.g. ``"1mo"``, ``"5d"``)."""
    if isinstance(every, bool):
        raise TypeError("every must be an int or interval string, not bool")
    if isinstance(every, int):
        if every <= 0:
            raise ValueError(f"every (days) must be positive, got {every}")
        return f"{every}d"
    text = _WORD_TO_INTERVAL.get(str(every).strip().lower(), str(every).strip().lower())
    if text[:1].isalpha():  # bare unit, e.g. "mo" -> "1mo"
        text = "1" + text
    if not _INTERVAL_RE.fullmatch(text):
        raise ValueError(f"Could not parse every={every!r}; use e.g. '1mo', '2w', '5d', '1q', '1y' or an int of days")
    return text


def _date_windows(interval: portion.Interval, every: str | int) -> list[tuple[Any, Any]] | None:
    """Split a date ``interval`` into contiguous half-open ``[lo, hi)`` windows.

    Boundaries come from Polars' calendar-aware temporal functions (``dt.truncate`` +
    ``date_range``), so month/quarter/year arithmetic and the interval-string grammar are not
    reimplemented here. A disjoint interval is handled per atomic component (windows are generated
    only over the ranges the predicate actually selects, not across the gaps between them), and the
    ``_MAX_WINDOWS`` cap applies to the emitted windows rather than the enclosing span.
    """
    if interval.empty:
        return None
    iv = _interval_str(every)
    windows: dict[tuple[Any, Any], None] = {}  # dict = ordered, de-duplicated
    for atomic in interval:
        lower, upper = atomic.lower, atomic.upper
        if lower == -portion.inf or upper == portion.inf:
            return None
        start = pl.Series([lower]).dt.truncate(iv).item()
        end = pl.Series([upper]).dt.offset_by(iv).item()  # one step past ``upper`` closes the final window
        is_date = isinstance(start, datetime.date) and not isinstance(start, datetime.datetime)
        make_range = pl.date_range if is_date else pl.datetime_range
        bounds = make_range(start, end, interval=iv, closed="both", eager=True)
        inclusive_upper = atomic.right == portion.CLOSED
        for lo, hi in itertools.pairwise(bounds):
            if (lo <= upper) if inclusive_upper else (lo < upper):
                windows[(lo, hi)] = None
                if len(windows) > _MAX_WINDOWS:
                    return None
    return list(windows) or None


@dataclass(frozen=True)
class _ByTime:
    on: str
    every: str | int

    def build(self, predicate: pl.Expr | None) -> list[ReadPartition] | None:
        if predicate is None:
            return None
        interval = convert_expr_to_datetime_range(predicate, self.on, get_enclosure=False)
        if interval.empty:
            return []  # contradictory predicate selects nothing
        windows = _date_windows(interval, self.every)
        if windows is None:
            return None  # unbounded range -- cannot partition
        col = pl.col(self.on)
        return [ReadPartition(predicate=(col >= lo) & (col < hi), key=(lo, hi)) for lo, hi in windows]


def by_time(column: str, every: str | int = "1mo") -> Partitioner:
    """Partition on a date/datetime column by calendar windows derived from the pushed-down range.

    Note: the pushed-down bounds are normalised to timezone-naive values, so partitioning a
    timezone-aware ``Datetime`` column is not currently supported.

    Args:
        column: The date/datetime column to partition on.
        every: Window size -- an interval string (``"1mo"``, ``"2w"``, ``"5d"``, ``"1q"``, ``"1y"``)
            or an integer number of days.
    """
    _interval_str(every)  # validate eagerly
    return _ByTime(on=column, every=every)


def _membership_predicate(target: pl.Expr, value: Any) -> pl.Expr:
    """Build a predicate matching ``value`` on ``target``.

    A scalar becomes ``target == value``, or ``target.is_null()`` when ``value`` is ``None``
    (``target == None`` would match nothing). A list/tuple/set becomes ``target.is_in(members)``,
    OR-ed with ``target.is_null()`` when the members include ``None``.
    """
    if isinstance(value, (list, tuple, set, frozenset)):
        members = list(value)
        non_null = [m for m in members if m is not None]
        pred = target.is_in(non_null)
        if len(non_null) != len(members):
            pred = pred | target.is_null()
        return pred
    if value is None:
        return target.is_null()
    return target == value


@dataclass(frozen=True)
class _ByValue:
    on: str
    values: tuple[Any, ...] | None

    def build(self, predicate: pl.Expr | None) -> list[ReadPartition] | None:
        if self.values is not None:
            return [
                ReadPartition(
                    predicate=_membership_predicate(pl.col(self.on), v), key=v if not isinstance(v, (list, tuple, set, frozenset)) else tuple(v)
                )
                for v in self.values
            ]
        if predicate is None:
            return None
        allowed = convert_expr_to_valid_values(predicate, self.on)
        if allowed is None:
            return None  # cannot determine the value set -- fall back
        col = pl.col(self.on)
        return [ReadPartition(predicate=_membership_predicate(col, v), key=v) for v in allowed]


def by_value(column: str, values: Iterable[Any] | None = None) -> Partitioner:
    """Partition on a discrete column, one partition per value or per member group.

    Args:
        column: The column to partition on.
        values: The partition values. Each element is either a scalar (``col == v``) or a
            collection (``col.is_in(v)``). When ``None``, the distinct values are read from the
            ``IN`` / equality filter pushed down on ``column``; if none can be determined, the
            read runs unpartitioned.
    """
    materialized = None if values is None else tuple(values)
    return _ByValue(on=column, values=materialized)


@dataclass(frozen=True)
class _ByRange:
    on: str
    every: float | int

    def build(self, predicate: pl.Expr | None) -> list[ReadPartition] | None:
        if predicate is None:
            return None
        interval = convert_expr_to_range(predicate, self.on)
        if interval.empty:
            return []  # contradictory predicate selects nothing
        col = pl.col(self.on)
        parts: list[ReadPartition] = []
        for atomic in interval:  # bucket each selected range; skip the gaps between disjoint ranges
            if atomic.lower == -portion.inf or atomic.upper == portion.inf:
                return None  # unbounded range -- cannot partition
            inclusive_upper = atomic.right == portion.CLOSED
            lo = atomic.lower
            while (lo <= atomic.upper) if inclusive_upper else (lo < atomic.upper):
                hi = lo + self.every
                parts.append(ReadPartition(predicate=(col >= lo) & (col < hi), key=(lo, hi)))
                lo = hi
                if len(parts) > _MAX_WINDOWS:
                    return None
        return parts or None


def by_range(column: str, every: float) -> Partitioner:
    """Partition on a numeric column into fixed-width ``[lo, lo+every)`` buckets.

    The numeric range is read from the bounded filter pushed down on ``column``; if the filter
    leaves the range unbounded, the read runs unpartitioned.

    Args:
        column: The numeric column to partition on.
        every: Bucket width.
    """
    if every <= 0:
        raise ValueError(f"every must be positive, got {every}")
    return _ByRange(on=column, every=every)


@dataclass(frozen=True)
class KeyPartitions:
    """Enumerated equality partitioning by caller-supplied keys.

    A spec (not a :class:`Partitioner`): the concrete partitions depend on the target frame's
    schema and the pushed-down predicate, so a backend that supports it resolves them at scan
    time. ``partition_remote_options`` is honoured only by distributed executors.

    Args:
        partitions: One key per partition -- a frame/series/sequence of keys.
        by: The key column name(s), selector, or expression (e.g. ``pl.col("id").hash() % n``).
        partition_remote_options: Optional per-partition executor options (distributed backends only).
    """

    partitions: Any
    by: Any
    partition_remote_options: Any = None


def by_key(partitions: Any, by: Any, *, partition_remote_options: Any = None) -> KeyPartitions:
    """Partition by equality on caller-enumerated keys (one partition per key).

    Args:
        partitions: A frame/series/sequence with one key per partition.
        by: Column name(s), a selector, or a ``pl.Expr`` identifying the partition key.
        partition_remote_options: Optional per-partition executor options (distributed backends only).
    """
    return KeyPartitions(partitions=partitions, by=by, partition_remote_options=partition_remote_options)


def discrete_partitions(column: str, groups: Iterable[Sequence] | Mapping[Any, Sequence]) -> list[ReadPartition]:
    """Build ``column.is_in(members)`` partitions -- one per group / member list.

    Args:
        column: The column to partition on.
        groups: Either an iterable of member lists (labelled by position) or a mapping
            ``{label: members}``.
    """
    items = groups.items() if isinstance(groups, Mapping) else enumerate(groups)
    return [ReadPartition(predicate=_membership_predicate(pl.col(column), list(members)), key=label) for label, members in items]


def cartesian_partitions(
    *,
    date_windows: Iterable[tuple],
    buckets: Iterable | Mapping,
    date_column: str,
    bucket: Any,
) -> list[ReadPartition]:
    """Build a flat ``{date_windows} x {buckets}`` partition set in a single pass.

    Each cell's predicate is ``(date_column in window) & (bucket predicate)``, applied as one
    filter so the date-window component still pushes down into inner sources.

    Args:
        date_windows: Iterable of ``(lower, upper)`` half-open ``[lower, upper)`` windows.
        buckets: Either an iterable of bucket values (labelled by position) or a mapping
            ``{label: value}``. A list/tuple/set becomes ``target.is_in(value)``; a scalar becomes
            ``target == value``; ``None`` matches nulls.
        date_column: The datetime column for the window predicate.
        bucket: The bucket target -- a column name (``str``) or a ``pl.Expr``.
    """
    target = pl.col(bucket) if isinstance(bucket, str) else bucket
    bucket_items = list(buckets.items()) if isinstance(buckets, Mapping) else list(enumerate(buckets))
    parts: list[ReadPartition] = []
    for lower, upper in date_windows:
        date_pred = (pl.col(date_column) >= lower) & (pl.col(date_column) < upper)
        for label, value in bucket_items:
            parts.append(ReadPartition(predicate=date_pred & _membership_predicate(target, value), key=((lower, upper), label)))
    return parts


def as_partition_list(partitions: Partitioner | Iterable[ReadPartition], predicate: pl.Expr | None) -> list[ReadPartition] | None:
    """Resolve ``partitions`` to a concrete list.

    Returns ``None`` only when a partitioner cannot derive a bounded split (the caller may then
    fall back to a single unpartitioned read). An explicit iterable always resolves to a list --
    an empty one stays empty (a known-empty partition set), never ``None``.
    """
    if isinstance(partitions, Partitioner):
        return partitions.build(predicate)
    resolved = list(partitions)
    if any(not isinstance(p, ReadPartition) for p in resolved):
        raise TypeError("partitions must be a Partitioner (e.g. by_time/by_value/by_range) or an iterable of ReadPartition")
    return resolved


def retained_columns(predicates: Iterable[pl.Expr], with_columns: list[str] | None) -> list[str] | None:
    """Projection a consumer must request so every predicate is evaluable, or ``None`` to keep all.

    Partition (and pushed-down) predicates are applied after the read, so the columns they touch
    must survive projection pushdown even when the caller did not request them. Returns ``None``
    (meaning "keep every column") when nothing was projected or when a predicate exposes no
    concrete root names (e.g. a selector), so a projection can never drop a column a filter needs.
    """
    if with_columns is None:
        return None
    required: set[str] = set()
    unresolved = False
    for pred in predicates:
        roots = pred.meta.root_names()
        if not roots:
            unresolved = True
        required.update(roots)
    if unresolved:
        return None
    return list(dict.fromkeys([*with_columns, *(c for c in required if c not in with_columns)]))
