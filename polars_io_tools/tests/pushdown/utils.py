"""Shared helpers for the pushdown characterization suite.

The suite locks in observations of polars' optimizer behavior on plain
``pl.LazyFrame`` pipelines, using ``PredicateTracker`` as a sentinel scan.
Each lock-in test (``test_*_NOT_pushed``) is intended to be self-contained
so that its body can be lifted directly into a polars GitHub issue as a
minimal reproducer.
"""

from __future__ import annotations

from typing import Optional

import polars as pl

from polars_io_tools.testing import PredicateTracker


def _check(
    tracker: PredicateTracker,
    lf: pl.LazyFrame,
    *,
    predicate_pushed: Optional[bool] = None,
    projection_pushed: Optional[bool] = None,
    pushed_columns: Optional[set[str]] = None,
    call_count: Optional[int] = None,
) -> None:
    """Collect ``lf`` and assert pushdown observations on ``tracker``.

    ``None`` for any axis means "don't care". ``call_count`` lets CSE-style
    tests assert how many times the source was invoked.
    """
    tracker.reset()
    lf.collect()
    if predicate_pushed is True:
        assert tracker.last_predicate is not None, "predicate was not pushed"
    elif predicate_pushed is False:
        assert tracker.last_predicate is None, f"predicate was unexpectedly pushed: {tracker.last_predicate!r}"
    if projection_pushed is True:
        assert tracker.last_with_columns is not None, "projection was not pushed"
        if pushed_columns is not None:
            assert set(tracker.last_with_columns) == pushed_columns, f"pushed projection {tracker.last_with_columns!r} != {pushed_columns!r}"
    elif projection_pushed is False:
        assert tracker.last_with_columns is None, f"projection was unexpectedly pushed: {tracker.last_with_columns!r}"
    if call_count is not None:
        assert tracker.call_count == call_count, f"call_count {tracker.call_count} != expected {call_count}"
