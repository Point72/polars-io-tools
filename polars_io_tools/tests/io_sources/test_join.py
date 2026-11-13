"""filtered_join, filtered_join_asof and join_between.

filtered_join and filtered_join_asof must return what the plain Polars join returns. A filtered_join row may also state
the filter the right source receives: per column, the keys of an ``is_in``, the value of an ``==``, or ``True`` for a
bare Boolean column. join_between rows compare the joined columns with an expected frame.
"""

from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_io_tools import join_between
from polars_io_tools._compat import POLARS_HAS_COLLECT_BATCHES
from polars_io_tools.io_sources.base import BinaryExprNode, ColumnNode, FunctionNode, LiteralNode, get_literal_value, get_parsed_expr
from polars_io_tools.io_sources.enum import BooleanFunctionType, OperatorType
from polars_io_tools.io_sources.join import filtered_join_asof
from polars_io_tools.testing import io_source_assert
from polars_io_tools.tests.helpers.assertions import assert_matches_plain_polars
from polars_io_tools.tests.helpers.cases import as_params
from polars_io_tools.tests.helpers.pickling import pickle_round_trip
from polars_io_tools.tests.helpers.polars_versions import polars_has


def _flatten_conjuncts(node):
    """Yield the atomic (non-AND) nodes of an AND-tree of predicates."""
    if isinstance(node, BinaryExprNode) and node.op in (OperatorType.AND, OperatorType.LOGICAL_AND):
        yield from _flatten_conjuncts(node.left)
        yield from _flatten_conjuncts(node.right)
    else:
        yield node


def _constraint(atom) -> tuple[str, object] | None:
    """``(column, keys)`` for ``col.is_in(keys)``, ``(column, ("==", value))`` for ``col == value``, ``(column, True)`` for a bare column."""
    if isinstance(atom, FunctionNode) and atom.function_type == BooleanFunctionType.IS_IN:
        return atom.inputs[0].expr.meta.output_name(), sorted(get_literal_value(atom.inputs[1].expr))
    if isinstance(atom, BinaryExprNode) and atom.op == OperatorType.EQ and isinstance(atom.right, LiteralNode):
        return atom.left.expr.meta.output_name(), ("==", get_literal_value(atom.right.expr))
    if isinstance(atom, ColumnNode):
        return atom.name, True
    return None


def assert_right_receives(predicates: list, expected_conditions: dict, only_these: bool) -> None:
    """Every filter the right source received holds ``expected_conditions``; with ``only_these``, it holds nothing else."""
    for predicate in predicates:
        found = [_constraint(atom) for atom in _flatten_conjuncts(get_parsed_expr(predicate))]
        for col, want in expected_conditions.items():
            assert (col, want) in found, f"{col}: {want!r} not in {predicate}"
        if only_these:
            assert len(found) == len(expected_conditions), predicate


LEFT_FOO_BAR_HAM = pl.DataFrame({"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0], "ham": ["a", "b", "c"]})
RIGHT_APPLE_HAM_BAR_FOO2 = pl.DataFrame({"apple": ["x", "y", "z"], "ham": ["a", "b", "d"], "bar": ["a", "b", "c"], "foo2": [1, 2, 3]})
FOUR_IDS = pl.DataFrame({"id": [1, 2, 3, 4], "value": ["a", "b", "c", "d"]})
LEFT_WITH_COMMON = pl.DataFrame({"id": [1, 2, 3], "common": ["a", "b", "c"]})
RIGHT_WITH_COMMON = pl.DataFrame({"id": [1, 2, 4], "common": ["x", "y", "z"]})
# Microsecond UTC keys past 9999-12-31, which a Python datetime cannot hold.
KEYS_BEYOND_PYTHON_DATETIME = pl.Series("k", [253402318799999998, 253402318799999999], dtype=pl.Datetime("us", "UTC")).to_frame()


@dataclass(frozen=True)
class FilteredJoinCase:
    """A ``filtered_join`` of two frames, which must equal the plain Polars join, and the filter the right source
    receives."""

    id: str
    _: KW_ONLY
    left: pl.DataFrame
    right: pl.DataFrame
    # keyword arguments for ``filtered_join``
    join_arguments: dict
    # a right column -> the condition the right source's filter must hold on it (see ``assert_right_receives``); None
    # checks nothing
    right_receives: dict | None = None
    right_not_read: bool = False
    # the filter holds no condition other than those of ``right_receives``
    only_these_conditions: bool = True
    # applied to the joined frame before the comparison
    query: Callable[[pl.LazyFrame], pl.LazyFrame] | None = None
    marks: tuple = ()


FILTERED_JOINS = [
    FilteredJoinCase(
        "filtered_join_basic",
        left=LEFT_FOO_BAR_HAM,
        right=RIGHT_APPLE_HAM_BAR_FOO2,
        join_arguments={"on": "ham", "maintain_order": "left"},
        right_receives={"ham": ["a", "b", "c"]},
    ),
    *[
        FilteredJoinCase(
            f"filtered_join_basic_no_results_left_side-{how}",
            left=LEFT_FOO_BAR_HAM,
            right=RIGHT_APPLE_HAM_BAR_FOO2.with_columns(bar=pl.Series(["a", "b", "d"])),
            join_arguments={"on": "ham", "how": how, "maintain_order": "left"},
            right_not_read=True,
            query=lambda lf: lf.filter(pl.col("ham") == "d"),
            marks=(pytest.mark.skipif(polars_has("pushes_join_filters"), reason="Polars pushes the filter to the right source itself"),),
        )
        for how in ("inner", "left")
    ],
    FilteredJoinCase(
        "filtered_join_single_filter_value",
        left=LEFT_FOO_BAR_HAM.head(1),
        right=RIGHT_APPLE_HAM_BAR_FOO2,
        join_arguments={"on": "ham", "maintain_order": "left"},
        right_receives={"ham": ("==", "a")},
    ),
    FilteredJoinCase(
        "filtered_join_column_selection",
        left=LEFT_FOO_BAR_HAM,
        right=RIGHT_APPLE_HAM_BAR_FOO2,
        join_arguments={"on": "ham", "maintain_order": "left"},
        right_receives={"ham": ["a", "b", "c"]},
        query=lambda lf: lf.select("foo", "bar"),
    ),
    FilteredJoinCase(
        "filtered_join_multiple_columns",
        left=LEFT_FOO_BAR_HAM,
        right=RIGHT_APPLE_HAM_BAR_FOO2,
        join_arguments={"left_on": ["ham", "foo"], "right_on": ["ham", "foo2"], "maintain_order": "left"},
        right_receives={"ham": ["a", "b", "c"], "foo2": [1, 2, 3]},
    ),
    # The join keys and the user's filter all reach the source; Polars >= 2.1 lowers ``active == True`` to ``active``.
    FilteredJoinCase(
        "filtered_join_multiple_columns_with_extra_filter",
        left=pl.DataFrame({"id": [1, 2, 3, 4], "category": ["A", "B", "A", "C"], "value": [10, 20, 30, 40]}),
        right=pl.DataFrame({"id": [1, 2, 3, 5], "category": ["A", "B", "A", "D"], "score": [15, 25, 35, 55], "active": [True, False, True, False]}),
        join_arguments={"on": ["id", "category"], "maintain_order": "left_right"},
        right_receives={"id": [1, 2, 3, 4], "category": ["A", "B", "C"], "active": ("==", True) if not polars_has("simplifies_eq_true") else True},
        only_these_conditions=False,
        query=lambda lf: lf.filter(pl.col("active") == pl.lit(True)),
    ),
    FilteredJoinCase(
        "filtered_join_different_join_types",
        left=FOUR_IDS,
        right=pl.DataFrame({"id": [1, 2, 5, 6], "desc": ["x", "y", "p", "q"]}),
        join_arguments={"on": "id", "how": "inner", "maintain_order": "left"},
        right_receives={"id": [1, 2, 3, 4]},
    ),
    # Null keys never match, so they are not pushed.
    FilteredJoinCase(
        "filtered_join_with_nulls",
        left=FOUR_IDS.with_columns(id=pl.Series([1, 2, None, 4])),
        right=pl.DataFrame({"id": [1, None, 3, 5], "desc": ["x", "y", "z", "p"]}),
        join_arguments={"on": "id", "maintain_order": "left"},
        right_receives={"id": [1, 2, 4]},
    ),
    FilteredJoinCase(
        "filtered_join_with_various_data_types",
        left=pl.DataFrame({"id": [1, 2, 3], "date": [date(2023, 1, d) for d in (1, 2, 3)], "flag": [True, False, True]}),
        right=pl.DataFrame({"id": [1, 2, 4], "date": [date(2023, 1, d) for d in (1, 2, 4)], "desc": ["x", "y", "z"]}),
        join_arguments={"on": ["id", "date"], "maintain_order": "left"},
        right_receives={"id": [1, 2, 3], "date": [date(2023, 1, d) for d in (1, 2, 3)]},
    ),
    FilteredJoinCase(
        "filtered_join_empty_dataframe",
        left=pl.DataFrame(schema={"id": pl.Int64, "value": pl.Float64}),
        right=pl.DataFrame({"id": [1, 2, 3], "desc": ["x", "y", "z"]}),
        join_arguments={"on": "id", "maintain_order": "left"},
        right_receives={"id": []},
    ),
    FilteredJoinCase(
        "filtered_join_suffix-default",
        left=LEFT_WITH_COMMON,
        right=RIGHT_WITH_COMMON,
        join_arguments={"on": "id", "maintain_order": "left"},
        right_receives={"id": [1, 2, 3]},
    ),
    FilteredJoinCase(
        "filtered_join_suffix-custom",
        left=LEFT_WITH_COMMON,
        right=RIGHT_WITH_COMMON,
        join_arguments={"on": "id", "suffix": "_custom", "maintain_order": "left"},
        right_receives={"id": [1, 2, 3]},
    ),
    # Projection pushdown must keep a left column that makes the right one take the suffix the query reads.
    FilteredJoinCase(
        "filtered_join_suffixed_column_in_derived_expression",
        left=pl.DataFrame(
            {
                "data_date": [date(2024, 1, d) for d in (2, 3, 4)],
                "data_date_previous": [date(2024, 1, d) for d in (1, 2, 3)],
                "id": [1, 2, 3],
                "quantity": [100, 200, 300],
                "price": [10.0, 20.0, 30.0],
            }
        ),
        right=pl.DataFrame(
            {"data_date": [date(2024, 1, d) for d in (1, 2, 3)], "id": [1, 2, 3], "quantity": [90, 190, 290], "price": [9.0, 19.0, 29.0]}
        ),
        join_arguments={
            "left_on": ["id", "data_date_previous"],
            "right_on": ["id", "data_date"],
            "how": "left",
            "suffix": "_previous",
            "maintain_order": "left",
        },
        query=lambda lf: lf.with_columns(((pl.col("price") - pl.col("price_previous")) * pl.col("quantity_previous")).alias("value_change")).select(
            "data_date", "id", "value_change"
        ),
    ),
    FilteredJoinCase(
        "filtered_join_suffixed_column_multiple_conflicts",
        left=pl.DataFrame({"key": [1, 2], "a": [10, 20], "b": [100, 200], "c": [1000, 2000]}),
        right=pl.DataFrame({"key": [1, 2], "a": [11, 21], "b": [110, 210], "c": [1100, 2100]}),
        join_arguments={"on": "key", "suffix": "_right", "maintain_order": "left"},
        query=lambda lf: lf.with_columns((pl.col("a_right") + pl.col("b_right") + pl.col("c_right")).alias("sum_right")).select("key", "sum_right"),
    ),
    # Keys must not lose precision through Python values.
    *[
        FilteredJoinCase(
            f"filtered_join_keys_keep_dtype-{name}",
            left=pl.DataFrame({"k": keys, "v": range(len(keys))}),
            right=pl.DataFrame({"k": keys, "w": range(len(keys))}),
            join_arguments={"on": "k"},
        )
        for name, keys in [
            ("datetime_ns", pl.Series([1001, 2002]).cast(pl.Datetime("ns"))),
            ("duration_ns", pl.Series([1001, 2002]).cast(pl.Duration("ns"))),
            ("time", pl.Series([1001]).cast(pl.Time)),
            ("decimal", pl.Series([Decimal("1.25"), Decimal("2.50")], dtype=pl.Decimal(10, 2))),
        ]
    ],
    # Keys a Python list holds exactly are pushed as a list, so the pushed predicate shows the values.
    FilteredJoinCase(
        "filtered_join_pushes_integer_keys_as_a_list",
        left=pl.DataFrame({"k": [2, 3], "v": [0, 1]}),
        right=pl.DataFrame({"k": [1, 2, 3, 4], "w": [10, 20, 30, 40]}),
        join_arguments={"on": "k"},
        right_receives={"k": [2, 3]},
    ),
    FilteredJoinCase(
        "filtered_join_keys_beyond_python_datetime_range",
        left=KEYS_BEYOND_PYTHON_DATETIME,
        right=KEYS_BEYOND_PYTHON_DATETIME,
        join_arguments={"on": "k"},
    ),
]


@pytest.mark.parametrize("case", as_params(FILTERED_JOINS))
def test_filtered_join(case: FilteredJoinCase):
    seen = []
    right_source = io_source_assert(case.right, seen.append)
    query = case.query or (lambda lf: lf)
    row_order = "maintain_order" in case.join_arguments
    reference = query(case.left.lazy().join(case.right.lazy(), **case.join_arguments))
    assert_matches_plain_polars(
        query(case.left.lazy().piot.filtered_join(right_source, **case.join_arguments)), reference, None, check_row_order=row_order
    )
    if case.right_not_read:
        assert not seen, seen
    elif case.right_receives is not None:
        assert_right_receives(seen, case.right_receives, case.only_these_conditions)


@pytest.mark.parametrize("how", ["inner", "left"])
def test_filtered_join_complex_workflow(how):
    """Only the join keys and the filter on the right column reach the right source, not the filters on derived columns."""
    if how == "left" and not polars_has("rewrites_filtered_left_join_as_inner"):
        pytest.skip("Polars before 1.32 does not rewrite this left join as an inner join, so only the keys are pushed")
    left, right_df = pl.DataFrame({"id": [1, 2, 3, 4], "value": [10, 20, 30, 40]}), pl.DataFrame({"id": [1, 2, 3, 5], "multiplier": [2, 3, 4, 5]})
    seen = []

    def downstream(joined):
        product = joined.with_columns((pl.col("value") * pl.col("multiplier")).alias("product")).filter(pl.col("product") > 50)
        return product.group_by("multiplier").agg(pl.sum("product").alias("sum_product")).sort("multiplier").filter(pl.col("multiplier") < 4)

    result = downstream(left.lazy().piot.filtered_join(io_source_assert(right_df, seen.append), on="id", how=how, maintain_order="left"))
    assert_matches_plain_polars(
        result, downstream(left.lazy().join(right_df.lazy(), on="id", how=how, maintain_order="left")), None, check_row_order=True
    )
    for predicate in seen:
        atoms = list(_flatten_conjuncts(get_parsed_expr(predicate)))
        assert len(atoms) == 2 and ("id", [1, 2, 3, 4]) in [_constraint(a) for a in atoms], predicate
        (lt,) = [a for a in atoms if isinstance(a, BinaryExprNode)]
        assert lt.op == OperatorType.LT and lt.left.expr.meta == pl.col("multiplier") and get_literal_value(lt.right.expr) == 4, predicate


def test_filtered_join_sink_parquet(tmp_path):
    seen = []
    path = tmp_path / "join_data.parquet"
    LEFT_FOO_BAR_HAM.lazy().piot.filtered_join(io_source_assert(RIGHT_APPLE_HAM_BAR_FOO2, seen.append), on="ham", maintain_order="left").select(
        "bar", "foo"
    ).sink_parquet(path)
    assert_frame_equal(pl.read_parquet(path), pl.DataFrame({"bar": [6.0, 7.0], "foo": [1, 2]}))
    assert_right_receives(seen, {"ham": ["a", "b", "c"]}, only_these=True)


# filtered_join_asof: the tolerance bound pushed to the right source, across strategies, keys and filters.
@dataclass(frozen=True)
class AsofColumnsCase:
    """Whether the join selects columns afterwards, and whether it coalesces the join key."""

    id: str
    _: KW_ONLY
    select_columns: bool
    coalesce: bool


def _at(hour: int, minute: int) -> datetime:
    return datetime(2023, 1, 1, hour, minute)


# Trades (left) and quotes (right); ``timestamp2`` is a quote's ``timestamp`` plus one second.
_TRADES = pl.DataFrame(
    {"timestamp": [_at(9, 0), _at(9, 5), _at(9, 10)], "trade_id": [1, 2, 2], "price": [100.0, 105.0, 102.0], "shared_col": [1, 2, 3]}
).with_columns(pl.col("timestamp").set_sorted())
_QUOTES = pl.DataFrame(
    {
        "timestamp": [_at(8, 58), _at(9, 2), _at(9, 7), _at(9, 12)],
        "timestamp2": [datetime(2023, 1, 1, h, m, 1) for h, m in [(8, 58), (9, 2), (9, 7), (9, 12)]],
        "quote_id": [1, 2, 2, 4],
        "bid": [99.0, 103.0, 104.0, 101.0],
        "shared_col": [-1, -2, -3, -4],
    }
).with_columns(pl.col("timestamp").set_sorted())
_ASOF_TOLERANCE = timedelta(minutes=3)
# The bound the quotes receive for the user's "after 09:07" and "until 09:03": moved by the tolerance on the side where a
# match can lie beyond the user's bound.
_LOWER_BOUND_PUSHED = {"backward": _at(9, 4), "nearest": _at(9, 4), "forward": _at(9, 7)}
_UPPER_BOUND_PUSHED = {"backward": _at(9, 3), "nearest": _at(9, 6), "forward": _at(9, 6)}
_ASOF_VARIANTS = [
    pytest.mark.parametrize("by", [True, False], ids=["with_id_column", "without_id_column"]),
    pytest.mark.parametrize("on_same_column", [True, False], ids=["same_on_column", "different_on_columns"]),
    pytest.mark.parametrize("strategy", ["backward", "forward", "nearest"]),
    pytest.mark.parametrize(
        "case",
        as_params(
            [
                AsofColumnsCase("select_columns_without_coalesce", select_columns=True, coalesce=False),
                AsofColumnsCase("coalesce_without_select", select_columns=False, coalesce=True),
            ]
        ),
    ),
]


def _asof_variants(test):
    for mark in reversed(_ASOF_VARIANTS):
        test = mark(test)
    return test


def _join_trades_and_quotes(case: AsofColumnsCase, strategy: str, on_same_column: bool, by: bool, filters: pl.Expr | None) -> list:
    """Assert ``filtered_join_asof`` of the trades and quotes, filtered by ``filters``, equals plain Polars, and return
    the filters the quotes source received."""
    on, left_on, right_on = ("timestamp", None, None) if on_same_column else (None, "timestamp", "timestamp2")
    kwargs = {"on": on, "left_on": left_on, "right_on": right_on, "strategy": strategy, "tolerance": _ASOF_TOLERANCE, "coalesce": case.coalesce}
    if by:
        kwargs.update({"by_left": "trade_id", "by_right": "quote_id", "check_sortedness": False})
    pushed = []
    result = _TRADES.lazy().piot.filtered_join_asof(io_source_assert(_QUOTES, pushed.append), **kwargs)
    expected = _TRADES.lazy().join_asof(_QUOTES.lazy(), **kwargs)
    if filters is not None:
        result, expected = result.filter(filters), expected.filter(filters)
    if case.select_columns:
        result, expected = result.select(["timestamp", "price", "shared_col"]), expected.select(["timestamp", "price", "shared_col"])
    else:
        # The two joins may order their columns differently.
        result, expected = result.select(sorted(result.collect_schema().keys())), expected.select(sorted(expected.collect_schema().keys()))
    assert_frame_equal(result.collect(), expected.collect())
    return pushed


def _assert_some_conjunct_is_bound(predicate: pl.Expr, column: str, bound: datetime) -> None:
    """Polars 2.1 and later may add its own filter next to the bound, so any conjunct may hold it."""
    atoms = list(_flatten_conjuncts(get_parsed_expr(predicate)))
    assert any(
        isinstance(atom, BinaryExprNode) and isinstance(atom.right, LiteralNode) and atom.left.name == column and atom.right.value == bound
        for atom in atoms
    ), f"tolerance bound not pushed: {predicate!r}"


@_asof_variants
def test_filtered_join_asof_pushes_no_filter_without_a_user_filter(by, on_same_column, strategy, case: AsofColumnsCase):
    for predicate in _join_trades_and_quotes(case, strategy, on_same_column, by, filters=None):
        assert predicate is None


@_asof_variants
def test_filtered_join_asof_widens_a_lower_bound_by_the_tolerance(by, on_same_column, strategy, case: AsofColumnsCase):
    for predicate in _join_trades_and_quotes(case, strategy, on_same_column, by, filters=pl.col("timestamp") > _at(9, 7)):
        _assert_some_conjunct_is_bound(predicate, "timestamp" if on_same_column else "timestamp2", _LOWER_BOUND_PUSHED[strategy])


@_asof_variants
def test_filtered_join_asof_widens_an_upper_bound_by_the_tolerance(by, on_same_column, strategy, case: AsofColumnsCase):
    filters = (pl.col("timestamp") <= _at(9, 3)) & (pl.col("trade_id") != 2)
    for predicate in _join_trades_and_quotes(case, strategy, on_same_column, by, filters=filters):
        if by:
            # with ``by`` columns, the quotes receive exactly the bound and ``quote_id != 2``
            parsed = get_parsed_expr(predicate)
            assert {parsed.left.right.value, parsed.right.right.value} == {_UPPER_BOUND_PUSHED[strategy], 2}, predicate
        else:
            _assert_some_conjunct_is_bound(predicate, "timestamp" if on_same_column else "timestamp2", _UPPER_BOUND_PUSHED[strategy])


def _join_without_tolerance(strategy: str) -> list:
    """Assert ``filtered_join_asof`` without a tolerance, filtered to rows after 09:05, equals plain Polars, and return
    the filters the right source received.

    The 09:05 left row is dropped by the filter. The 09:06 row is kept, and its backward or nearest match (09:03) lies
    before 09:05, so a right-side filter "after 09:05" would lose that match.
    """
    left = pl.DataFrame({"timestamp": [_at(9, 5), _at(9, 6)], "id": [1, 2]}).with_columns(pl.col("timestamp").set_sorted())
    right = pl.DataFrame({"timestamp": [_at(9, 0), _at(9, 3), _at(9, 8)], "value": [10, 20, 30]}).with_columns(pl.col("timestamp").set_sorted())
    after_9_05 = pl.col("timestamp") > _at(9, 5)
    pushed = []
    result = left.lazy().piot.filtered_join_asof(io_source_assert(right, pushed.append), on="timestamp", strategy=strategy)
    expected = left.lazy().join_asof(right.lazy(), on="timestamp", strategy=strategy)
    assert_frame_equal(result.filter(after_9_05).collect(), expected.filter(after_9_05).collect())
    return pushed


@pytest.mark.parametrize("strategy", ["backward", "nearest"])
def test_filtered_join_asof_without_tolerance_pushes_nothing(strategy):
    for predicate in _join_without_tolerance(strategy):
        assert predicate is None, f"unexpected right-side predicate for {strategy}: {predicate!r}"


def test_filtered_join_asof_without_tolerance_forward_pushes_only_the_user_filter():
    """A forward match lies at or after its left row, so "after 09:05" is safe on the right side. Polars 2.1 and later
    push it there themselves, unchanged; ``filtered_join_asof`` adds nothing."""
    for predicate in _join_without_tolerance("forward"):
        if predicate is None:
            continue
        assert polars_has("pushes_forward_asof_filter"), f"unexpected right-side predicate: {predicate!r}"
        for atom in _flatten_conjuncts(get_parsed_expr(predicate)):
            assert isinstance(atom, BinaryExprNode) and isinstance(atom.right, LiteralNode)
            assert atom.left.name == "timestamp"
            assert atom.op in (OperatorType.GT, OperatorType.GT_EQ)
            assert atom.right.value == _at(9, 5)


# Left and right frames, join_asof arguments, and filters applied in turn to the same joined frame.
_LEFT_DATETIMES = pl.DataFrame({"t": [datetime(2024, 1, 1, 0, 0, 0, 500), datetime(2024, 1, 1, 13)]})
_RIGHT_DATETIMES = pl.DataFrame({"t": [datetime(2023, 12, 31, 23), datetime(2024, 1, 1, 0, 0, 0, 400), datetime(2024, 1, 1, 12, 30)], "y": [0, 1, 2]})
_LEFT_DATES = pl.DataFrame({"t": [date(2024, 1, 1)]})
_RIGHT_DATES = pl.DataFrame({"t": [date(2023, 12, 31), date(2024, 1, 1)], "y": [0, 1]})
_T_AS_UTC = {"t": pl.Datetime("us", "UTC")}
_ON_T_WITHIN_ONE_HOUR = {"on": "t", "tolerance": timedelta(hours=1)}
_NO_TIMESTAMPS = pl.DataFrame(schema={"timestamp": pl.Datetime, "value": pl.Float64})
_TWO_QUOTES = pl.DataFrame({"timestamp": [datetime(2023, 1, 1, 9, 0, 0), datetime(2023, 1, 1, 9, 5, 0)], "data": ["a", "b"]})


@dataclass(frozen=True)
class AsofMatchCase:
    """Two frames and ``join_asof`` arguments; ``filtered_join_asof`` must equal plain ``join_asof`` for each filter in
    turn."""

    id: str
    _: KW_ONLY
    # applied one after another to the same joined frame; None is no filter
    filters_in_turn: list[pl.Expr | None]
    # the left and right frames
    frames: tuple[pl.DataFrame, pl.DataFrame]
    join_asof_arguments: dict


ASOF_MATCHES = [
    # A joined frame collected with one filter and then another must not keep the first filter.
    AsofMatchCase(
        "filtered_join_asof_collects_do_not_retain_previous_filters",
        filters_in_turn=[pl.col("t") >= date(2025, 1, 10), pl.col("t") <= date(2025, 1, 2), None],
        frames=(
            pl.DataFrame({"t": [date(2025, 1, 2), date(2025, 1, 11)]}),
            pl.DataFrame({"t": [date(2025, 1, 1), date(2025, 1, 10)], "value": [10, 20]}),
        ),
        join_asof_arguments={"on": "t", "tolerance": timedelta(days=2)},
    ),
    AsofMatchCase(
        "filtered_join_asof_empty_left_dataframe",
        filters_in_turn=[None],
        frames=(_NO_TIMESTAMPS, _TWO_QUOTES),
        join_asof_arguments={"on": "timestamp", "strategy": "backward"},
    ),
    AsofMatchCase(
        "filtered_join_asof_empty_right_dataframe",
        filters_in_turn=[None],
        frames=(
            pl.DataFrame({"timestamp": [datetime(2023, 1, 1, 9)], "value": [1]}),
            pl.DataFrame(schema={"timestamp": pl.Datetime, "data": pl.Utf8}),
        ),
        join_asof_arguments={"on": "timestamp", "strategy": "backward"},
    ),
    AsofMatchCase(
        "filtered_join_asof_keeps_matches-cast_date",
        filters_in_turn=[pl.col("t").cast(pl.Date) < datetime(2024, 1, 1, 12)],
        frames=(_LEFT_DATETIMES, _RIGHT_DATETIMES),
        join_asof_arguments=_ON_T_WITHIN_ONE_HOUR,
    ),
    AsofMatchCase(
        "filtered_join_asof_keeps_matches-cast_ms",
        filters_in_turn=[pl.col("t").cast(pl.Datetime("ms")) <= datetime(2024, 1, 1)],
        frames=(_LEFT_DATETIMES, _RIGHT_DATETIMES),
        join_asof_arguments=_ON_T_WITHIN_ONE_HOUR,
    ),
    AsofMatchCase(
        "filtered_join_asof_keeps_matches-date_forward",
        filters_in_turn=[pl.col("t") < datetime(2024, 1, 1, 12)],
        frames=(_LEFT_DATES, _RIGHT_DATES),
        join_asof_arguments={"on": "t", "strategy": "forward", "tolerance": timedelta(hours=1)},
    ),
    AsofMatchCase(
        "filtered_join_asof_keeps_matches-date_backward",
        filters_in_turn=[pl.col("t") < datetime(2024, 1, 1, 12)],
        frames=(_LEFT_DATES, _RIGHT_DATES),
        join_asof_arguments={"on": "t", "strategy": "backward", "tolerance": timedelta(days=1)},
    ),
    AsofMatchCase(
        "filtered_join_asof_keeps_matches-utc",
        filters_in_turn=[pl.col("t") >= datetime(2024, 1, 1, tzinfo=UTC)],
        frames=(_LEFT_DATETIMES.cast(_T_AS_UTC), _RIGHT_DATETIMES.cast(_T_AS_UTC)),
        join_asof_arguments=_ON_T_WITHIN_ONE_HOUR,
    ),
]


@pytest.mark.parametrize("case", as_params(ASOF_MATCHES))
def test_filtered_join_asof_matches_native(case: AsofMatchCase):
    on = case.join_asof_arguments["on"]
    left, right = (df.lazy().set_sorted(on) for df in case.frames)
    joined, reference = left.piot.filtered_join_asof(right, **case.join_asof_arguments), left.join_asof(right, **case.join_asof_arguments)
    for pred in case.filters_in_turn:
        assert_matches_plain_polars(joined, reference, pred, check_row_order=True)


# join_between: left and right frames and join arguments, and the expected joined columns (compared after sorting by
# the first).
def _trades(dates, securities="AAPL", **extra):
    securities = [securities] * len(dates) if isinstance(securities, str) else securities
    return pl.DataFrame({"trade_id": range(1, len(dates) + 1), "security_id": securities, "trade_date": dates, **extra})


def _prices(starts, ends, prices, securities="AAPL", **extra):
    securities = [securities] * len(starts) if isinstance(securities, str) else securities
    return pl.DataFrame({"security_id": securities, "valid_from": starts, "valid_to": ends, "price": prices, **extra})


def _expected_columns(**cols):
    return pl.DataFrame(cols)


JANUARY_2024, MARCH_2024 = (date(2024, 1, 1), date(2024, 1, 31)), (date(2024, 3, 1), date(2024, 3, 31))
TRADE_DATE_IN_PRICE_WINDOW = {"left_on": "trade_date", "right_on_start": "valid_from", "right_on_end": "valid_to"}
TRADE_DATE_IN_PRICE_WINDOW_BY_SECURITY = {**TRADE_DATE_IN_PRICE_WINDOW, "by": "security_id"}
JANUARY_PRICES_FOR_TWO_SECURITIES = _prices([JANUARY_2024[0]] * 2, [JANUARY_2024[1]] * 2, [150.0, 140.0], securities=["AAPL", "GOOG"])
JANUARY_AND_MARCH_PRICES = _prices([JANUARY_2024[0], MARCH_2024[0]], [JANUARY_2024[1], MARCH_2024[1]], [10.0, 30.0])
JANUARY_PRICES = _prices([JANUARY_2024[0]], [JANUARY_2024[1]], [10.0])
EVENT_WINDOWS = pl.DataFrame({"event_id": ["A", "B"], "window_start": [datetime(2023, 1, 1, 10, 0, 0), datetime(2023, 1, 1, 10, 0, 3)]})


@dataclass(frozen=True)
class JoinBetweenCase:
    """Trades or events joined to the price or session window that holds them, and the expected joined columns."""

    id: str
    _: KW_ONLY
    # the left and right frames
    frames: tuple[pl.DataFrame, pl.DataFrame]
    # compared after sorting by its first column
    expected: pl.DataFrame
    # keyword arguments for ``join_between``
    join_arguments: dict


JOIN_BETWEEN = [
    JoinBetweenCase(
        "basic_range_join_no_by",
        frames=(
            _trades([date(2024, 1, 15), date(2024, 2, 15), date(2024, 3, 15)]).drop("security_id"),
            _prices(
                [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)], [date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 31)], [10.0, 20.0, 30.0]
            ).drop("security_id"),
        ),
        expected=_expected_columns(trade_id=[1, 2, 3], price=[10.0, 20.0, 30.0]),
        join_arguments=TRADE_DATE_IN_PRICE_WINDOW,
    ),
    JoinBetweenCase(
        "equi_join_with_by_column",
        frames=(
            _trades([date(2024, 1, 15), date(2024, 2, 15)] * 2, ["AAPL", "AAPL", "GOOG", "GOOG"]),
            _prices(
                [date(2024, 1, 1), date(2024, 2, 1)] * 2,
                [date(2024, 1, 31), date(2024, 2, 29)] * 2,
                [150.0, 155.0, 140.0, 145.0],
                ["AAPL", "AAPL", "GOOG", "GOOG"],
            ),
        ),
        expected=_expected_columns(trade_id=[1, 2, 3, 4], price=[150.0, 155.0, 140.0, 145.0]),
        join_arguments=TRADE_DATE_IN_PRICE_WINDOW_BY_SECURITY,
    ),
    JoinBetweenCase(
        "left_join_preserves_rows_when_by_not_in_right",
        frames=(_trades([date(2024, 1, 15)] * 3, ["AAPL", "MSFT", "GOOG"]), JANUARY_PRICES_FOR_TWO_SECURITIES),
        expected=_expected_columns(trade_id=[1, 2, 3], price=[150.0, None, 140.0]),
        join_arguments={**TRADE_DATE_IN_PRICE_WINDOW_BY_SECURITY, "how": "left"},
    ),
    JoinBetweenCase(
        "inner_join_drops_rows_when_by_not_in_right",
        frames=(_trades([date(2024, 1, 15)] * 3, ["AAPL", "MSFT", "GOOG"]), JANUARY_PRICES_FOR_TWO_SECURITIES),
        expected=_expected_columns(trade_id=[1, 3], price=[150.0, 140.0]),
        join_arguments={**TRADE_DATE_IN_PRICE_WINDOW_BY_SECURITY, "how": "inner"},
    ),
    JoinBetweenCase(
        "equi_join_with_composite_key",
        frames=(
            _trades([date(2024, 1, 15)] * 2, "ABC", source=["Bloomberg", "Reuters"]),
            _prices([JANUARY_2024[0]] * 2, [JANUARY_2024[1]] * 2, [100.0, 101.0], "ABC", source=["Bloomberg", "Reuters"]),
        ),
        expected=_expected_columns(trade_id=[1, 2], price=[100.0, 101.0]),
        join_arguments={**TRADE_DATE_IN_PRICE_WINDOW, "by": ["security_id", "source"]},
    ),
    JoinBetweenCase(
        "left_join_preserves_unmatched_range",
        frames=(_trades([date(2024, 1, 15), date(2024, 6, 15), date(2024, 3, 15)]), JANUARY_AND_MARCH_PRICES),
        expected=_expected_columns(trade_id=[1, 2, 3], price=[10.0, None, 30.0]),
        join_arguments={**TRADE_DATE_IN_PRICE_WINDOW_BY_SECURITY, "how": "left"},
    ),
    JoinBetweenCase(
        "inner_join_excludes_unmatched_range",
        frames=(_trades([date(2024, 1, 15), date(2024, 6, 15), date(2024, 3, 15)]), JANUARY_AND_MARCH_PRICES),
        expected=_expected_columns(trade_id=[1, 3], price=[10.0, 30.0]),
        join_arguments={**TRADE_DATE_IN_PRICE_WINDOW_BY_SECURITY, "how": "inner"},
    ),
    JoinBetweenCase(
        "gap_between_ranges_returns_null",
        frames=(_trades([date(2024, 2, 15)]), JANUARY_AND_MARCH_PRICES),
        expected=_expected_columns(trade_id=[1], price=[None]),
        join_arguments={**TRADE_DATE_IN_PRICE_WINDOW_BY_SECURITY, "how": "left"},
    ),
    JoinBetweenCase(
        "date_before_all_ranges",
        frames=(_trades([date(2023, 12, 15)]), JANUARY_PRICES),
        expected=_expected_columns(trade_id=[1], price=[None]),
        join_arguments={**TRADE_DATE_IN_PRICE_WINDOW_BY_SECURITY, "how": "left"},
    ),
    JoinBetweenCase(
        "date_after_all_ranges",
        frames=(_trades([date(2024, 12, 15)]), JANUARY_PRICES),
        expected=_expected_columns(trade_id=[1], price=[None]),
        join_arguments={**TRADE_DATE_IN_PRICE_WINDOW_BY_SECURITY, "how": "left"},
    ),
    JoinBetweenCase(
        "boundary_dates_inclusive",
        frames=(_trades([date(2024, 1, 1), date(2024, 1, 15), date(2024, 1, 31)]), JANUARY_PRICES),
        expected=_expected_columns(trade_id=[1, 2, 3], price=[10.0] * 3),
        join_arguments=TRADE_DATE_IN_PRICE_WINDOW_BY_SECURITY,
    ),
    JoinBetweenCase(
        "with_datetime_columns",
        frames=(
            pl.DataFrame(
                {"event_id": [1, 2, 3], "session_type": ["trading"] * 3, "event_time": [datetime(2024, 1, 15, h, 30) for h in (10, 14, 18)]}
            ),
            pl.DataFrame(
                {
                    "session_type": ["trading"] * 2,
                    "session_start": [datetime(2024, 1, 15, 9), datetime(2024, 1, 15, 13)],
                    "session_end": [datetime(2024, 1, 15, 12), datetime(2024, 1, 15, 17)],
                    "session_name": ["Morning", "Afternoon"],
                }
            ),
        ),
        expected=_expected_columns(event_id=[1, 2, 3], session_name=["Morning", "Afternoon", None]),
        join_arguments={
            "left_on": "event_time",
            "right_on_start": "session_start",
            "right_on_end": "session_end",
            "by": "session_type",
            "how": "left",
        },
    ),
    # Regressions: a name shared by both sides, or by right_on_end or right_on_start and left_on.
    JoinBetweenCase(
        "column_name_collision_nulls_right_not_left",
        frames=(pl.DataFrame({"ts": [5], "value": ["left"]}), pl.DataFrame({"start": [1], "end": [3], "value": ["right"]})),
        expected=_expected_columns(value=["left"], value_right=[None]),
        join_arguments={"left_on": "ts", "right_on_start": "start", "right_on_end": "end", "how": "left"},
    ),
    # Window B ends at 10:00:04, before the reading at 10:00:05.
    JoinBetweenCase(
        "right_on_end_name_collision_validation",
        frames=(
            pl.DataFrame({"timestamp": [datetime(2023, 1, 1, 10, 0, 5)], "value": [99]}),
            EVENT_WINDOWS.with_columns(timestamp=pl.Series([datetime(2023, 1, 1, 10, 0, 10), datetime(2023, 1, 1, 10, 0, 4)])),
        ),
        expected=_expected_columns(value=pl.Series([], dtype=pl.Int64)),
        join_arguments={"left_on": "timestamp", "right_on_start": "window_start", "right_on_end": "timestamp", "how": "inner"},
    ),
    JoinBetweenCase(
        "right_on_start_equal_left_on_preserves_left_key",
        frames=(pl.DataFrame({"x": [10]}), pl.DataFrame({"x": [1], "end": [5], "payload": [99]})),
        expected=_expected_columns(x=[10], end=pl.Series([None], dtype=pl.Int64), payload=pl.Series([None], dtype=pl.Int64)),
        join_arguments={"left_on": "x", "right_on_start": "x", "right_on_end": "end", "how": "left"},
    ),
    # At most one match per left row: the window with the nearest start (B), though A contains 10:00:04 too.
    JoinBetweenCase(
        "overlapping_intervals_returns_single_match",
        frames=(
            pl.DataFrame({"timestamp": [datetime(2023, 1, 1, 10, 0, 4)], "value": [99]}),
            EVENT_WINDOWS.with_columns(window_end=pl.Series([datetime(2023, 1, 1, 10, 0, 5), datetime(2023, 1, 1, 10, 0, 8)])),
        ),
        expected=_expected_columns(event_id=["B"]),
        join_arguments={"left_on": "timestamp", "right_on_start": "window_start", "right_on_end": "window_end", "how": "inner"},
    ),
]


@pytest.mark.parametrize("case", as_params(JOIN_BETWEEN))
def test_join_between(case: JoinBetweenCase):
    left, right = (df.lazy() for df in case.frames)
    result = join_between(left, right, **case.join_arguments).collect()
    assert_frame_equal(result.select(case.expected.columns).sort(case.expected.columns[0]), case.expected, check_dtypes=False)


@pytest.mark.skipif(not POLARS_HAS_COLLECT_BATCHES, reason="collect_batches requires Polars >= 1.34.0")
def test_collect_batches_filtered_join_yields_equal():
    left = pl.LazyFrame({"id": [1, 2, 3, 4], "x": [10, 20, 30, 40]})
    right = pl.LazyFrame({"id": [1, 2, 3, 4], "y": [100, 200, 300, 400]})

    lf = left.piot.filtered_join(right, on="id", maintain_order="left")
    chunks = list(lf.collect_batches(chunk_size=2))
    combined = pl.concat(chunks, how="vertical").sort("id")
    expected = lf.collect().sort("id")

    assert_frame_equal(combined, expected)


@dataclass(frozen=True)
class AsofKeepsMatchesCase:
    """A one-row frame joined as of itself, and a filter whose matches the join must keep."""

    id: str
    _: KW_ONLY
    # the dtype of the ``t`` column
    dtype: pl.DataType | type[pl.DataType]
    t: datetime
    pred: pl.Expr
    # more ``join_asof`` arguments
    kwargs: dict


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            # A ``Datetime("ms")`` column cast to ``us`` is exact; its bound must survive truncation to milliseconds.
            AsofKeepsMatchesCase(
                "ms_cast_to_us",
                dtype=pl.Datetime("ms"),
                t=datetime(2024, 1, 1, 12),
                pred=pl.col("t").cast(pl.Datetime("us")) == datetime(2024, 1, 1, 12),
                kwargs={"tolerance": timedelta(seconds=1)},
            ),
            AsofKeepsMatchesCase(
                "forward_at_max",
                dtype=pl.Datetime("us"),
                t=datetime(9999, 12, 31),
                pred=pl.col("t") == datetime(9999, 12, 31),
                kwargs={"strategy": "forward", "tolerance": timedelta(days=1)},
            ),
            AsofKeepsMatchesCase(
                "backward_at_min",
                dtype=pl.Datetime("us"),
                t=datetime(1, 1, 1),
                pred=pl.col("t") == datetime(1, 1, 1),
                kwargs={"strategy": "backward", "tolerance": timedelta(days=1)},
            ),
        ]
    ),
)
def test_filtered_join_asof_keeps_matches(case: AsofKeepsMatchesCase):
    left = pl.DataFrame({"t": pl.Series([case.t], dtype=case.dtype)}).lazy().set_sorted("t")
    right = left.with_columns(v=pl.lit(7)).set_sorted("t")
    assert_matches_plain_polars(
        filtered_join_asof(left, right, on="t", log_explain=False, **case.kwargs),
        left.join_asof(right, on="t", **case.kwargs),
        case.pred & (pl.col("v") == 7),
        check_row_order=True,
    )


class TestFilteredJoinPickle:
    """Tests for filtered_join pickle support."""

    def test_filtered_join_pickle_basic(self):
        """filtered_join LazyFrames can be pickled and unpickled."""
        left_df = pl.DataFrame({"id": [1, 2, 3], "value_left": ["a", "b", "c"]})
        right_df = pl.DataFrame({"id": [2, 3, 4], "value_right": ["x", "y", "z"]})

        lf = left_df.lazy().piot.filtered_join(right_df.lazy(), on="id", how="inner")
        lf_unpickled = pickle_round_trip(lf)

        expected = left_df.join(right_df, on="id", how="inner")
        result = lf_unpickled.collect()
        assert_frame_equal(result, expected, check_row_order=False)
