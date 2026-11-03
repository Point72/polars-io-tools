"""pushdown_unpivot against bare ``LazyFrame.unpivot``, and what each pushes to the source.

Every row runs twice: on plain ``unpivot`` ("bare") and on ``pushdown_unpivot`` ("wrapped"). Both must return what plain
unpivot returns, and each states whether a filter or a column selection reached the source. Bare runs of rows with
``bare_misses_pushdown`` are marked ``pytest.mark.gap``: Polars could hand something to the source there but does not.
For example ``filter(variable == "A")`` could become the column selection ``[id, A]`` on the source, and
``filter(value > X)`` a looser source filter ``(A > X) | (B > X) | ...`` with the exact filter kept after the unpivot.
The wrapper does the first.
"""

from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field

import polars as pl
import pytest

from polars_io_tools import pushdown_unpivot
from polars_io_tools.testing import PredicateTracker
from polars_io_tools.tests.helpers.assertions import ExpectedPushdown, assert_matches_plain_polars, assert_pushdown
from polars_io_tools.tests.helpers.cases import as_params

WIDE = pl.DataFrame({"id": [1, 2, 3], "A": [10, 20, 30], "B": [100, 200, 300], "C": [1000, 2000, 3000]})
NOTHING_PUSHED = ExpectedPushdown(predicate=False, projection=False)


@dataclass(frozen=True)
class UnpivotCase:
    """A query on ``WIDE.unpivot(index="id")``, and what the source should receive with and without the wrapper."""

    id: str
    _: KW_ONLY
    # receives the unpivoted frame and returns the query to collect
    query: Callable[[pl.LazyFrame], pl.LazyFrame]
    # what the source should receive from plain ``unpivot``
    expected_bare: ExpectedPushdown = field(default_factory=ExpectedPushdown)
    # what the source should receive from ``pushdown_unpivot``
    expected_wrapped: ExpectedPushdown = field(default_factory=ExpectedPushdown)
    # Polars could hand something to the source in the bare run but does not; that run is marked ``gap``
    bare_misses_pushdown: bool = False
    # the columns of the result, if checked
    expected_columns: list[str] | None = None


# Queries on ``WIDE.unpivot(index="id")``, and what reaches the source with and without the wrapper.
ROWS = [
    UnpivotCase(
        "filter_on_id_is_pushed",
        query=lambda lf: lf.filter(pl.col("id") > 1),
        expected_bare=ExpectedPushdown(predicate=True),
        expected_wrapped=ExpectedPushdown(predicate=True),
    ),
    UnpivotCase(
        "filter_on_variable_eq",
        query=lambda lf: lf.filter(pl.col("variable") == "A"),
        expected_bare=NOTHING_PUSHED,
        expected_wrapped=ExpectedPushdown(projection=True),
        bare_misses_pushdown=True,
    ),
    UnpivotCase(
        "filter_on_variable_isin",
        query=lambda lf: lf.filter(pl.col("variable").is_in(["A", "B"])),
        expected_bare=NOTHING_PUSHED,
        expected_wrapped=ExpectedPushdown(projection=True),
        bare_misses_pushdown=True,
    ),
    # Not pushable in general; the wrapper keeps the filter after the unpivot.
    UnpivotCase(
        "filter_on_value_NOT_pushed",
        query=lambda lf: lf.filter(pl.col("value") > 50),
        expected_bare=ExpectedPushdown(predicate=False),
        expected_wrapped=ExpectedPushdown(predicate=False),
        bare_misses_pushdown=True,
    ),
    # Only the count of value columns matters for select("id"); select("variable") needs only their names.
    UnpivotCase(
        "select_id_only",
        query=lambda lf: lf.select("id"),
        expected_bare=ExpectedPushdown(projection=False),
        expected_wrapped=ExpectedPushdown(projection=True),
        bare_misses_pushdown=True,
    ),
    UnpivotCase(
        "select_variable_only",
        query=lambda lf: lf.select("variable"),
        expected_bare=ExpectedPushdown(projection=False),
        expected_wrapped=ExpectedPushdown(projection=True),
        bare_misses_pushdown=True,
    ),
    UnpivotCase(
        "select_value_only",
        query=lambda lf: lf.select("value"),
        expected_bare=ExpectedPushdown(projection=False),
        expected_wrapped=ExpectedPushdown(projection=True),
        bare_misses_pushdown=True,
    ),
    UnpivotCase(
        "filter_id_and_select_value",
        query=lambda lf: lf.select("id", "value").filter(pl.col("id") > 1),
        expected_bare=ExpectedPushdown(predicate=True, projection=False),
        expected_wrapped=ExpectedPushdown(predicate=True),
        bare_misses_pushdown=True,
    ),
    # An OR across id and variable must not be split into pushable parts.
    UnpivotCase("or_predicate_spanning_columns", query=lambda lf: lf.filter((pl.col("id") > 2) | (pl.col("variable") == "A"))),
    UnpivotCase("predicate_references_value_and_variable", query=lambda lf: lf.filter((pl.col("variable") == "B") & (pl.col("value") > 100))),
    UnpivotCase("empty_variable_set", query=lambda lf: lf.filter(pl.col("variable").is_in([])), expected_columns=["id", "variable", "value"]),
]


def _one_param_per_mode():
    for case in ROWS:
        for mode in ("bare", "wrapped"):
            marks = pytest.mark.gap if mode == "bare" and case.bare_misses_pushdown else ()
            yield pytest.param(mode, case, id=f"{mode}-{case.id}", marks=marks)


@pytest.mark.parametrize(("mode", "case"), list(_one_param_per_mode()))
def test_unpivot(mode: str, case: UnpivotCase):
    tracker = PredicateTracker(WIDE)
    lf = tracker.lazy_frame.unpivot(index=["id"]) if mode == "bare" else pushdown_unpivot(tracker.lazy_frame, index=["id"])
    reference = case.query(tracker.lazy_frame.unpivot(index=["id"])).collect()
    tracker.reset()
    got = assert_matches_plain_polars(case.query(lf), reference.lazy(), None)
    assert_pushdown(tracker, case.expected_bare if mode == "bare" else case.expected_wrapped)
    if case.expected_columns:
        assert got.columns == case.expected_columns


@dataclass(frozen=True)
class UnpivotMatchesBareCase:
    """A frame, the index columns to unpivot it by, and a query whose result the wrapper and plain unpivot must agree
    on."""

    id: str
    _: KW_ONLY
    df: pl.DataFrame
    # receives the unpivoted frame and returns the query to collect
    query: Callable[[pl.LazyFrame], pl.LazyFrame]
    index: str | list[str]


MATCHES_BARE = [
    # Mixed value dtypes promote to a common supertype even when a variable filter keeps only the Int column.
    UnpivotMatchesBareCase(
        "mixed_dtype_value_unification_wrapped",
        df=pl.DataFrame({"id": [1, 2], "i": [10, 20], "f": [1.5, 2.5]}),
        query=lambda lf: lf.filter(pl.col("variable") == "i"),
        index=["id"],
    ),
    UnpivotMatchesBareCase("select_id_only_row_count_invariant_wrapped", df=WIDE.drop("C"), query=lambda lf: lf.select("id"), index=["id"]),
    UnpivotMatchesBareCase(
        "restricted_unpivot_matches_bare-no_index",
        df=pl.DataFrame({"A": [10, 20], "B": [30, 40]}),
        query=lambda lf: lf.filter(pl.col("variable") == "A").select("variable"),
        index=[],
    ),
    # The subset (i, f) has supertype Float64, but the full unpivot casts i straight to String.
    UnpivotMatchesBareCase(
        "restricted_unpivot_matches_bare-string_supertype",
        df=pl.DataFrame({"id": [1], "i": [10], "f": [1.5], "s": ["abc"]}),
        query=lambda lf: lf.filter(pl.col("variable").is_in(["i", "f"]) & (pl.col("value") == "10")),
        index="id",
    ),
]


@pytest.mark.parametrize("case", as_params(MATCHES_BARE))
def test_wrapped_matches_bare(case: UnpivotMatchesBareCase):
    wrapped, bare = pushdown_unpivot(case.df.lazy(), index=case.index), case.df.lazy().unpivot(index=case.index)
    assert_matches_plain_polars(case.query(wrapped), case.query(bare), None)


def test_value_filter_remains_in_wrapped_plan():
    """``filter(value > X)`` cannot replace the synthesized source projection, so it stays after the unpivot."""
    lf = pl.LazyFrame({"id": [1, 2], "A": [10, 200], "B": [20, 300]})
    wrapped = pushdown_unpivot(lf, index=["id"]).filter(pl.col("value") > 50)
    plan = wrapped.explain()
    assert "value" in plan and ("> 50" in plan or "(50)" in plan), plan
    assert_matches_plain_polars(wrapped, lf.unpivot(index=["id"]).filter(pl.col("value") > 50), None)
