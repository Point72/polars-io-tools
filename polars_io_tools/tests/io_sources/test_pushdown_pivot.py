"""pushdown_pivot against bare ``LazyFrame.pivot``, and what each pushes to the source.

Lazy pivot is a group_by with one ``filter().item()`` aggregation per ``on_columns`` value. Bare, a filter on the index
reaches the source and selecting only the index prunes the source to it; nothing else is pushed, which is correct:

* a filter on a pivoted column is a HAVING on an aggregate, so pushing ``score > 2`` would drop rows a group needs;
* ``is_not_null`` on a pivoted column rewritten as ``subject == "m"`` would null the other pivoted columns;
* selecting ``[name, m]`` rewritten as the source filter ``subject == "m"`` drops index values with no "m" row when the
  source is sparse over ``on``, which bare pivot keeps with ``m = null``.

The wrapper hands that filter to the source when the caller promises every index value has a row for every ``on``
value (``dense_on=True``, mode ``dense``). By default (mode ``smart``) a second scan of the index column alone recovers
the index values the filter dropped. Each row runs in the modes it lists ("bare" is plain ``LazyFrame.pivot``), and
every wrapped mode except ``dense`` on data with missing rows must return what plain pivot returns.
"""

from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from polars_io_tools import pushdown_pivot
from polars_io_tools.testing import PredicateTracker
from polars_io_tools.tests.helpers.assertions import ExpectedPushdown, assert_matches_plain_polars, assert_pushdown
from polars_io_tools.tests.helpers.cases import as_params

# Every name has a row for every subject.
DENSE = pl.DataFrame({"name": ["A", "A", "A", "B", "B", "B"], "subject": ["m", "p", "x", "m", "p", "x"], "score": [1, 2, 3, 4, 5, 6]})
# A has no "m", B no "x", and C only "p".
SPARSE = pl.DataFrame({"name": ["A", "A", "B", "B", "C"], "subject": ["p", "x", "m", "p", "p"], "score": [1, 2, 3, 4, 5]})
# DENSE with two columns the pivot does not use.
WIDE = DENSE.with_columns(pl.lit(0.0).alias("extra1"), pl.lit(0.0).alias("extra2"))
# Names and subjects not in sorted order, for ``maintain_order``.
ORDERED = pl.DataFrame({"name": ["B", "A", "A", "B"], "subject": ["m", "p", "m", "p"], "score": [1, 2, 3, 4]})
PIVOT_ARGUMENTS = {"on": "subject", "on_columns": ["m", "p", "x"], "index": "name", "values": "score"}
DENSE_ON_BY_MODE = {"smart": False, "dense": True}


@dataclass(frozen=True)
class PivotCase:
    """A query on the pivoted frame, and what the source should receive in each mode the row runs in."""

    id: str
    _: KW_ONLY
    # receives the pivoted frame and returns the query to collect
    query: Callable[[pl.LazyFrame], pl.LazyFrame]
    # what the source should receive, by mode: "bare", "smart" or "dense"
    expected_by_mode: dict[str, ExpectedPushdown]
    # the data the source serves
    source: pl.DataFrame = field(default_factory=lambda: DENSE)
    pivot_arguments: dict = field(default_factory=lambda: PIVOT_ARGUMENTS)
    # wrapped modes must return what plain pivot returns
    compare_with_bare: bool = True
    # asserts on the result
    result_check: Callable[[pl.DataFrame], None] | None = None


def _m_column_by_name_is(expected: dict):
    def check(got):
        assert dict(zip(got["name"], got["m"])) == expected

    return check


# Queries on the pivoted frame, and what reaches the source in each mode.
ROWS = [
    PivotCase(
        "filter_on_index_is_pushed",
        query=lambda lf: lf.filter(pl.col("name") == "A"),
        expected_by_mode={m: ExpectedPushdown(predicate=True) for m in ("bare", "smart", "dense")},
    ),
    PivotCase(
        "filter_on_pivoted_output_NOT_pushed",
        query=lambda lf: lf.filter(pl.col("m") > 2),
        expected_by_mode={"bare": ExpectedPushdown(predicate=False)},
    ),
    PivotCase(
        "filter_on_pivoted_output_is_not_null_NOT_pushed",
        query=lambda lf: lf.filter(pl.col("m").is_not_null()),
        expected_by_mode={"bare": ExpectedPushdown(predicate=False)},
    ),
    PivotCase(
        "select_index_only",
        query=lambda lf: lf.select("name"),
        expected_by_mode={
            "bare": ExpectedPushdown(projection={"name"}),
            "smart": ExpectedPushdown(projection=True),
            "dense": ExpectedPushdown(projection=True),
        },
    ),
    PivotCase("full_select_matches_bare_dense", query=lambda lf: lf, expected_by_mode={"smart": ExpectedPushdown(), "dense": ExpectedPushdown()}),
    # The source holds exactly the columns the pivot needs, so no projection is pushed.
    PivotCase(
        "select_subset",
        query=lambda lf: lf.select("name", "m"),
        expected_by_mode={"bare": ExpectedPushdown(projection=False), "smart": ExpectedPushdown(), "dense": ExpectedPushdown(predicate=True)},
    ),
    PivotCase(
        "select_two_pivoted_outputs",
        query=lambda lf: lf.select("name", "m", "p"),
        expected_by_mode={"bare": ExpectedPushdown(projection=False), "dense": ExpectedPushdown(predicate=True)},
    ),
    PivotCase(
        "select_subset_dense_on_True_pushes_projection_on_wide_source",
        query=lambda lf: lf.select("name", "m"),
        expected_by_mode={"dense": ExpectedPushdown(predicate=True, projection={"name", "subject", "score"})},
        source=WIDE,
    ),
    PivotCase(
        "sparse_data_dense_on_False_matches_bare",
        query=lambda lf: lf.select("name", "m"),
        expected_by_mode={"smart": ExpectedPushdown()},
        source=SPARSE,
        result_check=_m_column_by_name_is({"A": None, "B": 3, "C": None}),
    ),
    # Opting into the fast path on sparse data drops A and C, as documented.
    PivotCase(
        "sparse_data_dense_on_True_drops_rows_as_documented",
        query=lambda lf: lf.select("name", "m"),
        expected_by_mode={"dense": ExpectedPushdown()},
        source=SPARSE,
        compare_with_bare=False,
        result_check=_m_column_by_name_is({"B": 3}),
    ),
    # The on-filter could change the order of first appearance, so it is not pushed.
    PivotCase(
        "maintain_order_suppresses_on_filter",
        query=lambda lf: lf.select("name", "m"),
        expected_by_mode={"smart": ExpectedPushdown(predicate=False), "dense": ExpectedPushdown(predicate=False)},
        source=ORDERED,
        pivot_arguments={"on": "subject", "on_columns": ["m", "p"], "index": "name", "values": "score", "maintain_order": True},
    ),
    # A filter on p with select("name") still materializes p and pushes the on-filter.
    PivotCase(
        "predicate_on_dropped_pivoted_column_dense_on_True",
        query=lambda lf: lf.filter(pl.col("p") > 3).select("name"),
        expected_by_mode={"dense": ExpectedPushdown(predicate=True)},
    ),
]


@pytest.mark.parametrize(("mode", "case"), [pytest.param(mode, case, id=f"{mode}-{case.id}") for case in ROWS for mode in case.expected_by_mode])
def test_pivot(mode: str, case: PivotCase):
    tracker = PredicateTracker(case.source)
    pivot = case.pivot_arguments
    lf = tracker.lazy_frame.pivot(**pivot) if mode == "bare" else pushdown_pivot(tracker.lazy_frame, dense_on=DENSE_ON_BY_MODE[mode], **pivot)
    if mode != "bare" and case.compare_with_bare:
        reference = case.query(tracker.lazy_frame.pivot(**pivot)).collect()
        tracker.reset()
        got = assert_matches_plain_polars(case.query(lf), reference.lazy(), None, check_row_order=pivot.get("maintain_order", False))
    else:
        got = case.query(lf).collect()
    assert_pushdown(tracker, case.expected_by_mode[mode])
    if case.result_check:
        case.result_check(got)


def test_aggregate_function_not_implemented():
    with pytest.raises(NotImplementedError):
        pushdown_pivot(DENSE.lazy(), on="subject", on_columns=["m"], index="name", values="score", aggregate_function="sum")


@dataclass(frozen=True)
class PivotMatchesBareCase:
    """A frame pivoted on ``kind`` with index ``id`` and values ``v``, and a query whose result the wrapper and plain
    pivot must agree on."""

    id: str
    _: KW_ONLY
    df: pl.DataFrame
    on_columns: list
    # receives the pivoted frame and the names of its columns, and returns the query to collect
    query: Callable[[pl.LazyFrame, list[str]], pl.LazyFrame]


def _filter_and_keep_first_pivoted_column(lf: pl.LazyFrame, columns: list[str]) -> pl.LazyFrame:
    return lf.filter(pl.col(columns[1]) < 15).select("id", columns[1])


# 01:30 on the day New York's clocks go back, which happens twice; Python's ``fold=1`` picks the second time.
_NEW_YORK_REPEATED_HOUR = datetime(2024, 11, 3, 1, 30, tzinfo=ZoneInfo("America/New_York"))
_JAN_1_2_3_2024 = pl.Series([datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)])
_NULL_KIND = pl.DataFrame({"id": [1, 1], "kind": [None, "B"], "v": [10, 20]})
_DECIMAL_KIND = pl.DataFrame({"id": [1, 1, 1], "kind": pl.Series([1, 2, 3], dtype=pl.Decimal(3, 0)), "v": [10, 20, 30]})
# Frames, queries and pivot arguments for which the default wrapper must return what plain pivot returns.
MATCHES_BARE = [
    # A null index value survives the recovery join.
    PivotMatchesBareCase(
        "filter_on_first_pivoted_column_matches_bare-null_index",
        df=pl.DataFrame({"id": [None, None, 1, 1], "kind": ["A", "B", "A", "B"], "v": [10, 20, 30, 40]}),
        on_columns=["A", "B"],
        query=_filter_and_keep_first_pivoted_column,
    ),
    # A null ``on`` category survives the synthesized ``on`` filter.
    PivotMatchesBareCase(
        "filter_on_first_pivoted_column_matches_bare-null_on",
        df=_NULL_KIND,
        on_columns=[None, "B"],
        query=_filter_and_keep_first_pivoted_column,
    ),
    PivotMatchesBareCase(
        "filter_on_first_pivoted_column_matches_bare-null_on_extra_column",
        df=_NULL_KIND,
        on_columns=[None, "B", "C"],
        query=_filter_and_keep_first_pivoted_column,
    ),
    # The two 01:30s of the repeated hour when clocks go back compare equal in Python but are distinct categories.
    PivotMatchesBareCase(
        "filter_on_first_pivoted_column_matches_bare-dst_fold",
        df=pl.DataFrame(
            {
                "id": [1, 1],
                "kind": pl.Series([_NEW_YORK_REPEATED_HOUR, _NEW_YORK_REPEATED_HOUR.replace(fold=1)]).cast(pl.Datetime("us", "America/New_York")),
                "v": [1, 2],
            }
        ),
        on_columns=[_NEW_YORK_REPEATED_HOUR, _NEW_YORK_REPEATED_HOUR.replace(fold=1)],
        query=_filter_and_keep_first_pivoted_column,
    ),
    # Pivot on a Decimal column and compare two pivoted columns.
    PivotMatchesBareCase(
        "decimal_on_values_filter_matches_bare-in_range",
        df=_DECIMAL_KIND,
        on_columns=[Decimal(1), Decimal(2), Decimal(3)],
        query=lambda lf, columns: lf.filter(pl.col("1") < pl.col("2")).select("id", "1", "2"),
    ),
    PivotMatchesBareCase(
        "decimal_on_values_filter_matches_bare-out_of_range",
        df=_DECIMAL_KIND,
        on_columns=[Decimal(1), Decimal(2), Decimal(3), Decimal(1000)],
        query=lambda lf, columns: lf.filter(pl.col("1") < pl.col("2")).select("id", "1", "2", "1000"),
    ),
    # Pivot on datetimes; the pivoted columns are named in the time unit of the datetimes.
    *[
        PivotMatchesBareCase(
            f"series_on_columns_are_named_in_their_unit-{name}",
            df=pl.DataFrame({"id": [1, 1, 1], "kind": days, "v": [0, 1, 2]}),
            on_columns=days,
            query=lambda lf, columns: lf.select(columns[:3]),
        )
        for name, days in [
            ("ms", _JAN_1_2_3_2024.cast(pl.Datetime("ms"))),
            ("ns", _JAN_1_2_3_2024.cast(pl.Datetime("ns"))),
            ("ns_sub_us", pl.Series([1000, 1001, 1002]).cast(pl.Datetime("ns"))),
        ]
    ],
]


@pytest.mark.parametrize("case", as_params(MATCHES_BARE))
def test_wrapped_matches_bare(case: PivotMatchesBareCase):
    kw = {"on": "kind", "on_columns": case.on_columns, "index": "id", "values": "v"}
    columns = case.df.lazy().pivot(**kw).collect_schema().names()
    wrapped, bare = pushdown_pivot(case.df.lazy(), **kw), case.df.lazy().pivot(**kw)
    assert_matches_plain_polars(case.query(wrapped, columns), case.query(bare, columns), None)
