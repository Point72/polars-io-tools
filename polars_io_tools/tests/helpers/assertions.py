"""Assertions shared by the pushdown tests.

"Pushdown" means Polars hands part of a query (a filter or a column selection) to the source that reads the data, so
the source can skip rows or columns. The assertions check that a query gives the same rows as plain Polars, what the
source was handed, and what text (SQL or a filter string) a filter was translated to.
"""

import contextlib
from dataclasses import dataclass

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_io_tools.testing import PredicateAnalyzer, PredicateTracker


@contextlib.contextmanager
def failure_note(note: str):
    """Add ``note`` to the report of an assertion that fails inside, to say which step of a test failed."""
    try:
        yield
    except AssertionError as error:
        error.add_note(note)
        raise


def _filter_if_given(lf: pl.LazyFrame, pred: pl.Expr | None) -> pl.LazyFrame:
    return lf if pred is None else lf.filter(pred)


def _sorted(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(df.columns, nulls_last=True)


def assert_matches_plain_polars(actual: pl.LazyFrame, reference: pl.LazyFrame, pred: pl.Expr | None, *, check_row_order: bool = False):
    """Filter both frames by ``pred`` (``None`` keeps every row), assert they give the same rows and return ``actual``'s.

    ``reference`` is the same data read by plain Polars. Rows are compared in any order unless ``check_row_order``. If
    Polars rejects the query on ``reference``, ``actual`` must raise the same exception type, and ``None`` is returned.
    """
    try:
        want = _filter_if_given(reference, pred).collect()
    except pl.exceptions.PolarsError as exc:
        with pytest.raises(type(exc)):
            _filter_if_given(actual, pred).collect()
        return None
    got = _filter_if_given(actual, pred).collect()
    if check_row_order:
        assert_frame_equal(got, want)
    else:
        assert_frame_equal(_sorted(got), _sorted(want))
    return got


def assert_translated_text(
    text: str | None, contains=(), omits=(), *, equals: str | None = None, startswith: str | None = None, ignore_case: bool = False
) -> None:
    """Check the text a filter was translated to (SQL, or a filter string).

    The text must exist, equal ``equals`` and start with ``startswith`` when those are given, include every item of
    ``contains`` and none of ``omits``. A string counts as one item. An item of ``contains`` that is a tuple is met by
    any one of its strings, for output that may be written more than one way.
    """
    assert text is not None, "nothing was translated"
    normalize = str.upper if ignore_case else str
    got = normalize(text)
    if equals is not None:
        assert text == equals, f"{text!r} != {equals!r}"
    if startswith is not None:
        assert got.lstrip().startswith(normalize(startswith)), f"{text!r} does not start with {startswith!r}"
    for item in [contains] if isinstance(contains, str) else contains:
        alternatives = item if isinstance(item, tuple) else (item,)
        assert any(normalize(a) in got for a in alternatives), f"{item!r} not in {text!r}"
    for item in [omits] if isinstance(omits, str) else omits:
        assert normalize(item) not in got, f"{item!r} in {text!r}"


def assert_pushed_filter_keeps_rows(df: pl.DataFrame, pred: pl.Expr, pushed: pl.Expr | None) -> None:
    """Every row of ``df`` that the user's filter ``pred`` keeps is also kept by ``pushed``, the filter handed to the source.

    The source may return extra rows (Polars applies ``pred`` again afterwards) but must not drop a wanted row.
    ``None`` means no filter was handed over, which keeps every row.
    """
    if pushed is None:
        return
    rows = df.with_row_index("__row")
    kept = set(rows.filter(pred)["__row"])
    lost = kept - set(rows.filter(pushed)["__row"])
    assert not lost, f"{pushed} drops rows {sorted(lost)} kept by {pred}:\n{df[sorted(lost)]}"


def assert_pushed_time_bounds(tracker: PredicateTracker, col: str, lower=..., upper=...) -> None:
    """The last filter ``tracker``'s source received limits the date or time column ``col`` to ``lower`` and ``upper``.

    ``...`` accepts any bound; a tuple accepts any of its items.
    """
    pushed = tracker.last_predicate
    assert pushed is not None, "nothing was pushed"
    analyzer = PredicateAnalyzer(pushed)
    node = analyzer.find_temporal_filter(col)
    assert node is not None, f"no temporal filter on {col!r} in {pushed}"
    lo, hi = analyzer.extract_temporal_bounds(node)
    for name, got, want in (("lower", lo, lower), ("upper", hi, upper)):
        if want is not ...:
            assert got in (want if isinstance(want, tuple) else (want,)), f"{name} bound {got!r} != {want!r}; pushed {pushed}"


@dataclass(frozen=True, kw_only=True)
class ExpectedPushdown:
    """What a source should be handed when a query is collected. A field left as ``None`` is not checked."""

    # whether a filter reached the source
    predicate: bool | None = None
    # whether a column selection reached the source, or the exact set of columns it was asked for
    projection: bool | set[str] | None = None
    # how many times the source was scanned
    calls: int | None = None


def assert_pushdown(tracker: PredicateTracker, expected: ExpectedPushdown) -> None:
    """Check that ``tracker``'s source was handed what ``expected`` states."""
    if expected.predicate is not None:
        assert (tracker.last_predicate is not None) == expected.predicate, f"pushed predicate: {tracker.last_predicate!r}"
    if isinstance(expected.projection, bool):
        assert (tracker.last_with_columns is not None) == expected.projection, f"pushed projection: {tracker.last_with_columns!r}"
    elif expected.projection is not None:
        assert tracker.last_with_columns is not None and set(tracker.last_with_columns) == expected.projection, (
            f"pushed projection: {tracker.last_with_columns!r}"
        )
    if expected.calls is not None:
        assert tracker.call_count == expected.calls, f"{tracker.call_count} scans, expected {expected.calls}"


def sql_text(*contains, omits=(), equals=None) -> dict:
    """What a SQL string must hold, as keyword arguments for ``assert_translated_text``: every item of ``contains``, none
    of ``omits``, and exactly ``equals`` if given."""
    return {"contains": contains, "omits": omits, "equals": equals}
