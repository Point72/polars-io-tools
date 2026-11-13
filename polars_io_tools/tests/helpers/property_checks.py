"""Checks shared by the property tests, which use the Hypothesis library to generate frames and filters.

They work out which rows plain Polars keeps for a filter, which filter Polars hands to an IO source, and whether a
cached frame keeps the rows a plain filter keeps.
"""

import functools

import polars as pl
from hypothesis import note, settings
from polars.io.plugins import register_io_source
from polars.testing import assert_frame_equal

from polars_io_tools.tests.helpers.assertions import failure_note
from polars_io_tools.tests.helpers.polars_versions import polars_has
from polars_io_tools.tests.helpers.strategies import POLARS_ERRORS


def _filtered(frame: pl.DataFrame, expr: pl.Expr) -> pl.Series:
    """For each row of ``frame``, whether ``frame.filter(expr)`` keeps it."""
    kept = frame.with_row_index("__row").filter(expr)["__row"]
    return pl.int_range(frame.height, dtype=kept.dtype, eager=True).is_in(kept.implode() if polars_has("is_in_takes_a_list_value") else kept)


def _projected(frame: pl.DataFrame, expr: pl.Expr) -> pl.Series:
    """For each row of ``frame``, whether ``expr`` computed as a column (a projection) is true; null counts as false."""
    out = frame.with_columns(expr.alias("__m"))["__m"]
    if out.dtype not in (pl.Boolean, pl.Null):
        raise TypeError(out.dtype)
    return out.cast(pl.Boolean).fill_null(False)


def mask(frame: pl.DataFrame, expr: pl.Expr) -> pl.Series:
    """A Boolean Series with one value per row of ``frame``: whether a predicate built by library code keeps the row.

    The predicate is run two ways: as a filter, and as a projection (computed as a new column). Some Polars versions
    simplify a predicate differently in the two, so a row counts as kept if either the filter or the projection keeps
    it.
    """
    return _filtered(frame, expr) | _projected(frame, expr)


def show(expr: pl.Expr) -> str:
    """``str(expr)``; Polars panics formatting a temporal literal Python cannot hold."""
    try:
        return str(expr)
    except POLARS_ERRORS:
        return repr(expr)


def rows_plain_polars_keeps(frame: pl.DataFrame, pred: pl.Expr) -> pl.Series | None:
    """The rows plain Polars keeps for ``pred``; None if Polars rejects the predicate, or if a filter and a projection
    disagree (a Polars bug, such as 1.44 simplifying ``x.is_in([]).is_null()`` to false in a filter only)."""
    try:
        filtered, projected = _filtered(frame, pred), _projected(frame, pred)
    except POLARS_ERRORS:
        return None
    return filtered if filtered.equals(projected) else None


def filter_a_source_receives(frame: pl.DataFrame, pred: pl.Expr) -> pl.Expr | None:
    """The predicate Polars hands an IO source scanning ``frame`` filtered by ``pred``."""
    received: list = []

    def source(with_columns, predicate, n_rows, batch_size):
        received.append(predicate)
        yield frame

    try:
        register_io_source(source, schema=frame.schema).filter(pred).collect()
    except POLARS_ERRORS:
        return None
    return received[0] if received else None


# Settings for every property test: no time limit per example, since some examples fill a cache.
SETTINGS = settings(deadline=None)


def no_panic(fn):
    """Report a Rust panic, which is not an ``Exception``, as a failure Hypothesis can shrink."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except pl.exceptions.PanicException as e:
            raise AssertionError(f"panic: {e}") from e

    return wrapper


def note_case(frame: pl.DataFrame, pred: pl.Expr) -> None:
    """Record ``frame`` and ``pred`` in the Hypothesis report of a failing example."""
    note(f"physical={ {c: frame[c].to_physical().to_list() for c in frame.columns} } schema={frame.schema} pred={show(pred)}")


@no_panic
def check_cache_keeps_filtered_rows(lf: pl.LazyFrame, frame: pl.DataFrame, predicates: list) -> None:
    """Each predicate, applied in turn to the cached frame (cold, then warm), keeps the rows a plain filter keeps."""
    for i, pred in enumerate([*predicates, pl.lit(True), *predicates]):
        try:
            expected = frame.lazy().filter(pred).collect()
        except POLARS_ERRORS:
            continue
        note_case(frame, pred)
        with failure_note("warm cache" if i > len(predicates) else "cold cache"):
            assert_frame_equal(lf.filter(pred).collect(), expected, check_row_order=False)
