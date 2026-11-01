"""One assertion per pushdown contract, and adapters that put a source behind each entry point.

The contracts:

- matches native: an entry point filtered by a predicate returns what plain Polars returns.
- superset: what a source is asked for keeps every row the user's predicate keeps; pushdown is never lossy.
- pushed bounds / pushed discrete: the predicate a source receives has the expected temporal bounds or values.
"""

from collections.abc import Callable

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_io_tools.testing import PredicateAnalyzer, PredicateTracker


def _apply(lf: pl.LazyFrame, pred: pl.Expr | None) -> pl.LazyFrame:
    return lf if pred is None else lf.filter(pred)


def _sorted(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(df.columns, nulls_last=True)


def assert_matches_native(actual: pl.LazyFrame, reference: pl.LazyFrame, pred: pl.Expr | None, *, check_row_order: bool = False):
    """``actual`` filtered by ``pred`` equals ``reference`` filtered by ``pred``, and is returned.

    If Polars rejects the query on ``reference``, ``actual`` must raise the same exception type; ``None`` is returned.
    """
    try:
        want = _apply(reference, pred).collect()
    except pl.exceptions.PolarsError as exc:
        with pytest.raises(type(exc)):
            _apply(actual, pred).collect()
        return None
    got = _apply(actual, pred).collect()
    if check_row_order:
        assert_frame_equal(got, want.select(got.columns))
    else:
        assert_frame_equal(_sorted(got), _sorted(want.select(got.columns)))
    return got


def assert_keeps_rows(df: pl.DataFrame, pred: pl.Expr, pushed: pl.Expr | None) -> None:
    """Every row of ``df`` that ``pred`` keeps is kept by ``pushed``; ``None`` keeps every row."""
    if pushed is None:
        return
    rows = df.with_row_index("__row")
    kept = set(rows.filter(pred)["__row"])
    lost = kept - set(rows.filter(pushed)["__row"])
    assert not lost, f"{pushed} drops rows {sorted(lost)} kept by {pred}:\n{df[sorted(lost)]}"


def assert_superset(tracker: PredicateTracker, pred: pl.Expr | None) -> None:
    """The last predicate pushed to ``tracker`` keeps every row of its frame that ``pred`` keeps."""
    if pred is not None:
        assert_keeps_rows(tracker.df, pred, tracker.last_predicate)


def assert_pushed_bounds(tracker: PredicateTracker, col: str, lower=..., upper=...) -> None:
    """The temporal filter pushed on ``col`` has the given bounds; ``...`` is any bound, a tuple is any of its items."""
    pushed = tracker.last_predicate
    assert pushed is not None, "nothing was pushed"
    analyzer = PredicateAnalyzer(pushed)
    node = analyzer.find_temporal_filter(col)
    assert node is not None, f"no temporal filter on {col!r} in {pushed}"
    lo, hi = analyzer.extract_temporal_bounds(node)
    for name, got, want in (("lower", lo, lower), ("upper", hi, upper)):
        if want is not ...:
            assert got in (want if isinstance(want, tuple) else (want,)), f"{name} bound {got!r} != {want!r}; pushed {pushed}"


def assert_pushed_discrete(tracker: PredicateTracker, col: str, values: set) -> None:
    """The discrete filter pushed on ``col`` admits exactly ``values``."""
    from polars_io_tools.io_sources.set_visitor import convert_expr_to_valid_values

    pushed = tracker.last_predicate
    assert pushed is not None and PredicateAnalyzer(pushed).find_discrete_filter(col) is not None, f"no discrete filter on {col!r}: {pushed}"
    assert convert_expr_to_valid_values(pushed, col) == values, pushed


# Entry points: (source frame, tmp_path) -> (lazy frame under test, tracker on the source)
EntryPoint = Callable[[pl.DataFrame, object], tuple[pl.LazyFrame, PredicateTracker]]


def ep_io_source(df: pl.DataFrame, _tmp) -> tuple[pl.LazyFrame, PredicateTracker]:
    tracker = PredicateTracker(df)
    return tracker.lazy_frame, tracker


def ep_pushdown_combine(df: pl.DataFrame, _tmp) -> tuple[pl.LazyFrame, PredicateTracker]:
    from polars_io_tools.io_sources.pushdown_combine import FilterSpec, pushdown_combine

    tracker = PredicateTracker(df)
    specs = {c: FilterSpec() for c in df.columns if c != "val"}
    return pushdown_combine(sources={"data": (tracker.lazy_frame, specs)}, combine=lambda s: s["data"]), tracker


def ep_cache_parquet(df: pl.DataFrame, tmp) -> tuple[pl.LazyFrame, PredicateTracker]:
    from polars_io_tools.io_sources.lazy_cache_parquet import cache_parquet

    tracker = PredicateTracker(df)
    return cache_parquet(tracker.lazy_frame, tmp, "date", time_unit="daily"), tracker


ENTRY_POINTS: dict[str, EntryPoint] = {"io_source": ep_io_source, "pushdown_combine": ep_pushdown_combine, "cache_parquet": ep_cache_parquet}
