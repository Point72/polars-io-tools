"""filtered_join, filtered_join_asof and join_between.

filtered_join and filtered_join_asof must return what the plain Polars join returns. A filtered_join row may also pin
the predicate the right source receives: per column, the keys of an ``is_in``, the value of an ``==``, or ``True`` for
a bare Boolean column. join_between rows compare the joined columns with an expected frame.
"""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_io_tools import join_between
from polars_io_tools.io_sources.base import BinaryExprNode, ColumnNode, FunctionNode, LiteralNode, get_literal_value, get_parsed_expr
from polars_io_tools.io_sources.enum import BooleanFunctionType, OperatorType
from polars_io_tools.tests.helpers.assertions import assert_matches_native
from polars_io_tools.tests.helpers.cases import Case, params
from polars_io_tools.tests.helpers.polars_versions import polars_has

from .conftest import io_source_assert

c = pl.col


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


def assert_right_receives(predicates: list, expect: dict, exact: bool) -> None:
    """Every predicate the right source received has the ``expect`` constraints; ``exact`` allows no other conjunct."""
    for predicate in predicates:
        found = [_constraint(atom) for atom in _flatten_conjuncts(get_parsed_expr(predicate))]
        for col, want in expect.items():
            assert (col, want) in found, f"{col}: {want!r} not in {predicate}"
        if exact:
            assert len(found) == len(expect), predicate


ABC = pl.DataFrame({"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0], "ham": ["a", "b", "c"]})
ABD = pl.DataFrame({"apple": ["x", "y", "z"], "ham": ["a", "b", "d"], "bar": ["a", "b", "c"], "foo2": [1, 2, 3]})
IDS = pl.DataFrame({"id": [1, 2, 3, 4], "value": ["a", "b", "c", "d"]})
COMMON = (pl.DataFrame({"id": [1, 2, 3], "common": ["a", "b", "c"]}), pl.DataFrame({"id": [1, 2, 4], "common": ["x", "y", "z"]}))
NEVER = "never scanned"


def J(id, left, right, kwargs, expect=None, exact=True, query=None, marks=()):
    """A filtered_join row: ``expect`` maps right columns to their constraint, or is NEVER; ``query`` runs on the result."""
    return Case(id, query=query, expect=expect, kw={"left": left, "right": right, "join": kwargs, "exact": exact}, marks=marks)


FILTERED_JOINS = [
    J("filtered_join_basic", ABC, ABD, {"on": "ham", "maintain_order": "left"}, {"ham": ["a", "b", "c"]}),
    *[
        J(
            f"filtered_join_basic_no_results_left_side-{how}",
            ABC,
            ABD.with_columns(bar=pl.Series(["a", "b", "d"])),
            {"on": "ham", "how": how, "maintain_order": "left"},
            NEVER,
            query=lambda lf: lf.filter(c("ham") == "d"),
            marks=pytest.mark.skipif(polars_has("pushes_join_filters"), reason="Polars pushes the filter to the right source itself"),
        )
        for how in ("inner", "left")
    ],
    J("filtered_join_single_filter_value", ABC.head(1), ABD, {"on": "ham", "maintain_order": "left"}, {"ham": ("==", "a")}),
    J(
        "filtered_join_column_selection",
        ABC,
        ABD,
        {"on": "ham", "maintain_order": "left"},
        {"ham": ["a", "b", "c"]},
        query=lambda lf: lf.select("foo", "bar"),
    ),
    J(
        "filtered_join_multiple_columns",
        ABC,
        ABD,
        {"left_on": ["ham", "foo"], "right_on": ["ham", "foo2"], "maintain_order": "left"},
        {"ham": ["a", "b", "c"], "foo2": [1, 2, 3]},
    ),
    # The join keys and the user's filter all reach the source; Polars >= 2.1 lowers ``active == True`` to ``active``.
    J(
        "filtered_join_multiple_columns_with_extra_filter",
        pl.DataFrame({"id": [1, 2, 3, 4], "category": ["A", "B", "A", "C"], "value": [10, 20, 30, 40]}),
        pl.DataFrame({"id": [1, 2, 3, 5], "category": ["A", "B", "A", "D"], "score": [15, 25, 35, 55], "active": [True, False, True, False]}),
        {"on": ["id", "category"], "maintain_order": "left_right"},
        {"id": [1, 2, 3, 4], "category": ["A", "B", "C"], "active": ("==", True) if not polars_has("pushes_asof_filter") else True},
        exact=False,
        query=lambda lf: lf.filter(c("active") == pl.lit(True)),
    ),
    J(
        "filtered_join_different_join_types",
        IDS,
        pl.DataFrame({"id": [1, 2, 5, 6], "desc": ["x", "y", "p", "q"]}),
        {"on": "id", "how": "inner", "maintain_order": "left"},
        {"id": [1, 2, 3, 4]},
    ),
    # Null keys never match, so they are not pushed.
    J(
        "filtered_join_with_nulls",
        IDS.with_columns(id=pl.Series([1, 2, None, 4])),
        pl.DataFrame({"id": [1, None, 3, 5], "desc": ["x", "y", "z", "p"]}),
        {"on": "id", "maintain_order": "left"},
        {"id": [1, 2, 4]},
    ),
    J(
        "filtered_join_with_various_data_types",
        pl.DataFrame({"id": [1, 2, 3], "date": [date(2023, 1, d) for d in (1, 2, 3)], "flag": [True, False, True]}),
        pl.DataFrame({"id": [1, 2, 4], "date": [date(2023, 1, d) for d in (1, 2, 4)], "desc": ["x", "y", "z"]}),
        {"on": ["id", "date"], "maintain_order": "left"},
        {"id": [1, 2, 3], "date": [date(2023, 1, d) for d in (1, 2, 3)]},
    ),
    J(
        "filtered_join_empty_dataframe",
        pl.DataFrame(schema={"id": pl.Int64, "value": pl.Float64}),
        pl.DataFrame({"id": [1, 2, 3], "desc": ["x", "y", "z"]}),
        {"on": "id", "maintain_order": "left"},
        {"id": []},
    ),
    J("filtered_join_suffix-default", *COMMON, {"on": "id", "maintain_order": "left"}, {"id": [1, 2, 3]}),
    J("filtered_join_suffix-custom", *COMMON, {"on": "id", "suffix": "_custom", "maintain_order": "left"}, {"id": [1, 2, 3]}),
    # Projection pushdown must keep a left column that makes the right one take the suffix the query reads.
    J(
        "filtered_join_suffixed_column_in_derived_expression",
        pl.DataFrame(
            {
                "data_date": [date(2024, 1, d) for d in (2, 3, 4)],
                "data_date_previous": [date(2024, 1, d) for d in (1, 2, 3)],
                "id": [1, 2, 3],
                "quantity": [100, 200, 300],
                "price": [10.0, 20.0, 30.0],
            }
        ),
        pl.DataFrame({"data_date": [date(2024, 1, d) for d in (1, 2, 3)], "id": [1, 2, 3], "quantity": [90, 190, 290], "price": [9.0, 19.0, 29.0]}),
        {"left_on": ["id", "data_date_previous"], "right_on": ["id", "data_date"], "how": "left", "suffix": "_previous", "maintain_order": "left"},
        query=lambda lf: lf.with_columns(((c("price") - c("price_previous")) * c("quantity_previous")).alias("value_change")).select(
            "data_date", "id", "value_change"
        ),
    ),
    J(
        "filtered_join_suffixed_column_multiple_conflicts",
        pl.DataFrame({"key": [1, 2], "a": [10, 20], "b": [100, 200], "c": [1000, 2000]}),
        pl.DataFrame({"key": [1, 2], "a": [11, 21], "b": [110, 210], "c": [1100, 2100]}),
        {"on": "key", "suffix": "_right", "maintain_order": "left"},
        query=lambda lf: lf.with_columns((c("a_right") + c("b_right") + c("c_right")).alias("sum_right")).select("key", "sum_right"),
    ),
    # Keys must not lose precision through Python values.
    *[
        J(
            f"filtered_join_keys_keep_dtype-{name}",
            pl.DataFrame({"k": keys, "v": range(len(keys))}),
            pl.DataFrame({"k": keys, "w": range(len(keys))}),
            {"on": "k"},
        )
        for name, keys in [
            ("datetime_ns", pl.Series([1001, 2002]).cast(pl.Datetime("ns"))),
            ("duration_ns", pl.Series([1001, 2002]).cast(pl.Duration("ns"))),
            ("time", pl.Series([1001]).cast(pl.Time)),
            ("decimal", pl.Series([Decimal("1.25"), Decimal("2.50")], dtype=pl.Decimal(10, 2))),
        ]
    ],
    # Keys a Python list holds exactly are pushed as a list, so the pushed predicate shows the values.
    J(
        "filtered_join_pushes_integer_keys_as_a_list",
        pl.DataFrame({"k": [2, 3], "v": [0, 1]}),
        pl.DataFrame({"k": [1, 2, 3, 4], "w": [10, 20, 30, 40]}),
        {"on": "k"},
        {"k": [2, 3]},
    ),
    J(
        "filtered_join_keys_beyond_python_datetime_range",
        *[pl.Series("k", [253402318799999998, 253402318799999999], dtype=pl.Datetime("us", "UTC")).to_frame()] * 2,
        {"on": "k"},
    ),
]


@pytest.mark.parametrize(("query", "expect", "kw"), params(FILTERED_JOINS, "query", "expect", "kw"))
def test_filtered_join(query, expect, kw):
    seen = []
    right = io_source_assert(kw["right"], seen.append)
    query = query or (lambda lf: lf)
    row_order = "maintain_order" in kw["join"]
    reference = query(kw["left"].lazy().join(kw["right"].lazy(), **kw["join"]))
    assert_matches_native(query(kw["left"].lazy().piot.filtered_join(right, **kw["join"])), reference, None, check_row_order=row_order)
    if expect is NEVER:
        assert not seen, seen
    elif expect is not None:
        assert_right_receives(seen, expect, kw["exact"])


@pytest.mark.parametrize("how", ["inner", "left"])
def test_filtered_join_complex_workflow(how):
    """Only the join keys and the filter on the right column reach the right source, not the filters on derived columns."""
    if how == "left" and not polars_has("pushes_join_filters"):
        pytest.skip("Polars before 1.32 does not rewrite this left join as an inner join, so only the keys are pushed")
    left, right_df = pl.DataFrame({"id": [1, 2, 3, 4], "value": [10, 20, 30, 40]}), pl.DataFrame({"id": [1, 2, 3, 5], "multiplier": [2, 3, 4, 5]})
    seen = []

    def downstream(joined):
        product = joined.with_columns((c("value") * c("multiplier")).alias("product")).filter(c("product") > 50)
        return product.group_by("multiplier").agg(pl.sum("product").alias("sum_product")).sort("multiplier").filter(c("multiplier") < 4)

    result = downstream(left.lazy().piot.filtered_join(io_source_assert(right_df, seen.append), on="id", how=how, maintain_order="left"))
    assert_matches_native(result, downstream(left.lazy().join(right_df.lazy(), on="id", how=how, maintain_order="left")), None, check_row_order=True)
    for predicate in seen:
        atoms = list(_flatten_conjuncts(get_parsed_expr(predicate)))
        assert len(atoms) == 2 and ("id", [1, 2, 3, 4]) in [_constraint(a) for a in atoms], predicate
        (lt,) = [a for a in atoms if isinstance(a, BinaryExprNode)]
        assert lt.op == OperatorType.LT and lt.left.expr.meta == c("multiplier") and get_literal_value(lt.right.expr) == 4, predicate


def test_filtered_join_sink_parquet(tmp_path):
    seen = []
    path = tmp_path / "join_data.parquet"
    ABC.lazy().piot.filtered_join(io_source_assert(ABD, seen.append), on="ham", maintain_order="left").select("bar", "foo").sink_parquet(path)
    assert_frame_equal(pl.read_parquet(path), pl.DataFrame({"bar": [6.0, 7.0], "foo": [1, 2]}))
    assert_right_receives(seen, {"ham": ["a", "b", "c"]}, exact=True)


# filtered_join_asof: the tolerance bound pushed to the right source, across strategies, keys and filters.
@pytest.mark.parametrize("include_id", [True, False])
@pytest.mark.parametrize(
    "filters",
    [
        None,
        pl.col("timestamp") > datetime(2023, 1, 1, 9, 7, 0),
        ((pl.col("timestamp") <= datetime(2023, 1, 1, 9, 3, 0)) & (pl.col("trade_id") != 2)),
    ],
)
@pytest.mark.parametrize("on_same_column", [True, False])
@pytest.mark.parametrize("strategy", ["backward", "forward", "nearest"])
@pytest.mark.parametrize("select_columns,coalesce", [[True, False], [False, True]])
def test_filtered_join_asof_basic_backward(include_id, filters, on_same_column, strategy, select_columns, coalesce):
    """Test basic functionality of filtered_join_asof with backward strategy."""
    # Left dataframe with timestamps - represents trades
    on, left_on, right_on = "timestamp", None, None
    if on_same_column:
        on, left_on, right_on = "timestamp", None, None
    else:
        on, left_on, right_on = None, "timestamp", "timestamp2"
    df_left = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 1, 9, 0, 0),
                datetime(2023, 1, 1, 9, 5, 0),
                datetime(2023, 1, 1, 9, 10, 0),
            ],
            "trade_id": [1, 2, 2],
            "price": [100.0, 105.0, 102.0],
            "shared_col": [1, 2, 3],
        }
    ).with_columns(pl.col("timestamp").set_sorted())

    # Right dataframe with quotes - represents market data
    df_right = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 1, 8, 58, 0),
                datetime(2023, 1, 1, 9, 2, 0),
                datetime(2023, 1, 1, 9, 7, 0),
                datetime(2023, 1, 1, 9, 12, 0),
            ],
            "timestamp2": [
                datetime(2023, 1, 1, 8, 58, 1),
                datetime(2023, 1, 1, 9, 2, 1),
                datetime(2023, 1, 1, 9, 7, 1),
                datetime(2023, 1, 1, 9, 12, 1),
            ],
            "quote_id": [1, 2, 2, 4],
            "bid": [99.0, 103.0, 104.0, 101.0],
            "shared_col": [-1, -2, -3, -4],
        }
    ).with_columns(pl.col("timestamp").set_sorted())

    tolerance = timedelta(minutes=3)

    def assert_func(predicate):
        """Validate that filters are pushed down correctly to the right dataframe."""
        # We make sure we do not add predicates unnecessarily
        if filters is None:
            assert predicate is None
            return
        parsed_expr = get_parsed_expr(predicate)
        parsed_filter = get_parsed_expr(filters)

        assert isinstance(parsed_expr, BinaryExprNode)
        assert isinstance(parsed_filter, BinaryExprNode)

        filter_right_binary = isinstance(parsed_filter.right, BinaryExprNode)
        if filter_right_binary:
            filter_left_value = parsed_filter.left.right.value
            filter_right_value = parsed_filter.right.right.value
            filter_values = {filter_left_value, filter_right_value}
        else:
            filter_values = {parsed_filter.right.value}

        if include_id and filter_right_binary:
            expr_left_value = parsed_expr.left.right.value
            expr_right_value = parsed_expr.right.right.value
            if strategy in ["forward", "nearest"]:
                # In this case, the upper bound is affected
                # We subtract the tolerance from it to move backwards.
                if isinstance(expr_left_value, (datetime, date)):
                    expr_left_value -= tolerance
                else:
                    expr_right_value -= tolerance

            filter_left_value = parsed_filter.left.right.value
            filter_right_value = parsed_filter.right.right.value

            # Order does not matter, so we use set to compare
            assert {filter_left_value, filter_right_value} == {expr_left_value, expr_right_value}
            return
        # If we only have 1 filter, we should have 1 value pushed down. Polars >= 2.1 may AND its own
        # pushdown of the outer filter alongside ours, so search the conjuncts for our tolerance bound.
        target_col = right_on if right_on is not None else on

        def is_our_bound(atom) -> bool:
            if not (isinstance(atom, BinaryExprNode) and isinstance(atom.right, LiteralNode)):
                return False
            if atom.left.name != target_col:
                return False
            parsed_value = atom.right.value
            if filter_right_binary:
                if strategy in ["forward", "nearest"]:
                    # then we shift the upper bound in the 2 filters case
                    parsed_value -= tolerance
            else:
                if strategy in ["backward", "nearest"]:
                    # We shift the lower bound in the 1 filter case
                    parsed_value += tolerance
            return parsed_value in filter_values

        assert any(is_our_bound(atom) for atom in _flatten_conjuncts(parsed_expr)), f"tolerance bound not pushed: {predicate!r}"

    right_lf = io_source_assert(df_right, assert_func)

    # Perform filtered as-of join with 3-minute tolerance
    kwargs = {
        "on": on,
        "left_on": left_on,
        "right_on": right_on,
        "strategy": strategy,
        "tolerance": tolerance,
        "coalesce": coalesce,
    }

    if include_id:
        kwargs.update(
            {
                "by_left": "trade_id",
                "by_right": "quote_id",
                "check_sortedness": False,
            }
        )

    result = df_left.lazy().clone().piot.filtered_join_asof(right_lf, **kwargs)
    expected = df_left.lazy().clone().join_asof(df_right.lazy(), **kwargs)
    if filters is not None:
        result = result.filter(filters)
        expected = expected.filter(filters)

    if select_columns:
        result = result.select(["timestamp", "price", "shared_col"])
        expected = expected.select(["timestamp", "price", "shared_col"])
    else:
        # The orders of these might be different because we manually set the order of the columns
        # in the expected dataframe, so we sort them to compare. Once the Polars release after 1.30.0
        # comes out, we should be able to remove this and assert the orders are the same, we won't
        # need to adjust to the order of the schema (was causing issues sinking to parquet)
        result = result.select(sorted(result.collect_schema().keys()))
        expected = expected.select(sorted(expected.collect_schema().keys()))
    expected = expected.collect()
    result = result.collect()
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("strategy", ["backward", "forward", "nearest"])
def test_filtered_join_asof_no_tolerance(strategy):
    """Test filtered_join_asof without tolerance parameter."""
    df_left = pl.DataFrame(
        {
            # 09:05 is dropped by the filter; 09:06 survives and its backward/nearest match (09:03)
            # sits below the filter threshold, so unsafe right-side pruning would change the result.
            "timestamp": [datetime(2023, 1, 1, 9, 5, 0), datetime(2023, 1, 1, 9, 6, 0)],
            "id": [1, 2],
        }
    ).with_columns(pl.col("timestamp").set_sorted())

    df_right = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 1, 9, 0, 0),
                datetime(2023, 1, 1, 9, 3, 0),
                datetime(2023, 1, 1, 9, 8, 0),
            ],
            "value": [10, 20, 30],
        }
    ).with_columns(pl.col("timestamp").set_sorted())

    filters = [pl.col("timestamp") > datetime(2023, 1, 1, 9, 5, 0)]

    def assert_func(predicate):
        # Without tolerance we add no right-side bound. Polars >= 2.1 may itself push the post-join filter
        # into the right side of a *forward* asof join (where the matched right key is >= the left key);
        # that is unsound for backward/nearest, so only forward may carry a pushed bound, and it must be a
        # lower bound on the raw filter value (never tolerance-widened).
        if predicate is None:
            return
        assert polars_has("pushes_asof_filter") and strategy == "forward", f"unexpected right-side predicate for {strategy}: {predicate!r}"
        for atom in _flatten_conjuncts(get_parsed_expr(predicate)):
            assert isinstance(atom, BinaryExprNode) and isinstance(atom.right, LiteralNode)
            assert atom.left.name == "timestamp"
            assert atom.op in (OperatorType.GT, OperatorType.GT_EQ)
            assert atom.right.value == datetime(2023, 1, 1, 9, 5, 0)

    right_lf = io_source_assert(df_right, assert_func)

    result = df_left.lazy().piot.filtered_join_asof(
        right_lf,
        on="timestamp",
        strategy=strategy,
        # No tolerance parameter
    )

    expected = df_left.lazy().join_asof(df_right.lazy(), on="timestamp", strategy=strategy)

    expected = expected.filter(filters).collect()
    result = result.filter(filters).collect()

    assert_frame_equal(result, expected)


# (left, right, join_asof arguments, predicates collected in turn from the same joined frame)
_DT_LEFT = pl.DataFrame({"t": [datetime(2024, 1, 1, 0, 0, 0, 500), datetime(2024, 1, 1, 13)]})
_DT_RIGHT = pl.DataFrame({"t": [datetime(2023, 12, 31, 23), datetime(2024, 1, 1, 0, 0, 0, 400), datetime(2024, 1, 1, 12, 30)], "y": [0, 1, 2]})
_DATE_LEFT = pl.DataFrame({"t": [date(2024, 1, 1)]})
_DATE_RIGHT = pl.DataFrame({"t": [date(2023, 12, 31), date(2024, 1, 1)], "y": [0, 1]})
_UTC = {"t": pl.Datetime("us", "UTC")}
_HOUR = {"on": "t", "tolerance": timedelta(hours=1)}
_EMPTY_TS = pl.DataFrame(schema={"timestamp": pl.Datetime, "value": pl.Float64})
_TWO_QUOTES = pl.DataFrame({"timestamp": [datetime(2023, 1, 1, 9, 0, 0), datetime(2023, 1, 1, 9, 5, 0)], "data": ["a", "b"]})
ASOF_MATCHES = [
    # A joined frame collected with one filter and then another must not keep the first filter.
    Case(
        "filtered_join_asof_collects_do_not_retain_previous_filters",
        preds=[c("t") >= date(2025, 1, 10), c("t") <= date(2025, 1, 2), None],
        frames=(
            pl.DataFrame({"t": [date(2025, 1, 2), date(2025, 1, 11)]}),
            pl.DataFrame({"t": [date(2025, 1, 1), date(2025, 1, 10)], "value": [10, 20]}),
        ),
        kwargs={"on": "t", "tolerance": timedelta(days=2)},
    ),
    Case(
        "filtered_join_asof_empty_left_dataframe", preds=[None], frames=(_EMPTY_TS, _TWO_QUOTES), kwargs={"on": "timestamp", "strategy": "backward"}
    ),
    Case(
        "filtered_join_asof_empty_right_dataframe",
        preds=[None],
        frames=(
            pl.DataFrame({"timestamp": [datetime(2023, 1, 1, 9)], "value": [1]}),
            pl.DataFrame(schema={"timestamp": pl.Datetime, "data": pl.Utf8}),
        ),
        kwargs={"on": "timestamp", "strategy": "backward"},
    ),
    Case(
        "filtered_join_asof_keeps_matches-cast_date",
        preds=[c("t").cast(pl.Date) < datetime(2024, 1, 1, 12)],
        frames=(_DT_LEFT, _DT_RIGHT),
        kwargs=_HOUR,
    ),
    Case(
        "filtered_join_asof_keeps_matches-cast_ms",
        preds=[c("t").cast(pl.Datetime("ms")) <= datetime(2024, 1, 1)],
        frames=(_DT_LEFT, _DT_RIGHT),
        kwargs=_HOUR,
    ),
    Case(
        "filtered_join_asof_keeps_matches-date_forward",
        preds=[c("t") < datetime(2024, 1, 1, 12)],
        frames=(_DATE_LEFT, _DATE_RIGHT),
        kwargs={"on": "t", "strategy": "forward", "tolerance": timedelta(hours=1)},
    ),
    Case(
        "filtered_join_asof_keeps_matches-date_backward",
        preds=[c("t") < datetime(2024, 1, 1, 12)],
        frames=(_DATE_LEFT, _DATE_RIGHT),
        kwargs={"on": "t", "strategy": "backward", "tolerance": timedelta(days=1)},
    ),
    Case(
        "filtered_join_asof_keeps_matches-utc",
        preds=[c("t") >= datetime(2024, 1, 1, tzinfo=UTC)],
        frames=(_DT_LEFT.cast(_UTC), _DT_RIGHT.cast(_UTC)),
        kwargs=_HOUR,
    ),
]


@pytest.mark.parametrize(("preds", "frames", "kwargs"), params(ASOF_MATCHES, "preds", "frames", "kwargs"))
def test_filtered_join_asof_matches_native(preds, frames, kwargs):
    on = kwargs["on"]
    left, right = (df.lazy().set_sorted(on) for df in frames)
    joined, reference = left.piot.filtered_join_asof(right, **kwargs), left.join_asof(right, **kwargs)
    for pred in preds:
        assert_matches_native(joined, reference, pred, check_row_order=True)


# join_between: (left, right, keyword arguments) -> the expected joined columns, compared after sorting by the first.
def _trades(dates, securities="AAPL", **extra):
    securities = [securities] * len(dates) if isinstance(securities, str) else securities
    return pl.DataFrame({"trade_id": range(1, len(dates) + 1), "security_id": securities, "trade_date": dates, **extra})


def _prices(starts, ends, prices, securities="AAPL", **extra):
    securities = [securities] * len(starts) if isinstance(securities, str) else securities
    return pl.DataFrame({"security_id": securities, "valid_from": starts, "valid_to": ends, "price": prices, **extra})


def _expect(**cols):
    return pl.DataFrame(cols)


D = date
JAN, MAR = (D(2024, 1, 1), D(2024, 1, 31)), (D(2024, 3, 1), D(2024, 3, 31))
TRADE = {"left_on": "trade_date", "right_on_start": "valid_from", "right_on_end": "valid_to"}
BY_SECURITY = {**TRADE, "by": "security_id"}
TWO_SECURITIES = _prices([JAN[0]] * 2, [JAN[1]] * 2, [150.0, 140.0], securities=["AAPL", "GOOG"])
JAN_MAR = _prices([JAN[0], MAR[0]], [JAN[1], MAR[1]], [10.0, 30.0])
JAN_ONLY = _prices([JAN[0]], [JAN[1]], [10.0])
WINDOWS = pl.DataFrame({"event_id": ["A", "B"], "window_start": [datetime(2023, 1, 1, 10, 0, 0), datetime(2023, 1, 1, 10, 0, 3)]})
JOIN_BETWEEN = [
    Case(
        "basic_range_join_no_by",
        frames=(
            _trades([D(2024, 1, 15), D(2024, 2, 15), D(2024, 3, 15)]).drop("security_id"),
            _prices([D(2024, 1, 1), D(2024, 2, 1), D(2024, 3, 1)], [D(2024, 1, 31), D(2024, 2, 29), D(2024, 3, 31)], [10.0, 20.0, 30.0]).drop(
                "security_id"
            ),
        ),
        expect=_expect(trade_id=[1, 2, 3], price=[10.0, 20.0, 30.0]),
        kwargs=TRADE,
    ),
    Case(
        "equi_join_with_by_column",
        frames=(
            _trades([D(2024, 1, 15), D(2024, 2, 15)] * 2, ["AAPL", "AAPL", "GOOG", "GOOG"]),
            _prices(
                [D(2024, 1, 1), D(2024, 2, 1)] * 2,
                [D(2024, 1, 31), D(2024, 2, 29)] * 2,
                [150.0, 155.0, 140.0, 145.0],
                ["AAPL", "AAPL", "GOOG", "GOOG"],
            ),
        ),
        expect=_expect(trade_id=[1, 2, 3, 4], price=[150.0, 155.0, 140.0, 145.0]),
        kwargs=BY_SECURITY,
    ),
    Case(
        "left_join_preserves_rows_when_by_not_in_right",
        frames=(_trades([D(2024, 1, 15)] * 3, ["AAPL", "MSFT", "GOOG"]), TWO_SECURITIES),
        expect=_expect(trade_id=[1, 2, 3], price=[150.0, None, 140.0]),
        kwargs={**BY_SECURITY, "how": "left"},
    ),
    Case(
        "inner_join_drops_rows_when_by_not_in_right",
        frames=(_trades([D(2024, 1, 15)] * 3, ["AAPL", "MSFT", "GOOG"]), TWO_SECURITIES),
        expect=_expect(trade_id=[1, 3], price=[150.0, 140.0]),
        kwargs={**BY_SECURITY, "how": "inner"},
    ),
    Case(
        "equi_join_with_composite_key",
        frames=(
            _trades([D(2024, 1, 15)] * 2, "ABC", source=["Bloomberg", "Reuters"]),
            _prices([JAN[0]] * 2, [JAN[1]] * 2, [100.0, 101.0], "ABC", source=["Bloomberg", "Reuters"]),
        ),
        expect=_expect(trade_id=[1, 2], price=[100.0, 101.0]),
        kwargs={**TRADE, "by": ["security_id", "source"]},
    ),
    Case(
        "left_join_preserves_unmatched_range",
        frames=(_trades([D(2024, 1, 15), D(2024, 6, 15), D(2024, 3, 15)]), JAN_MAR),
        expect=_expect(trade_id=[1, 2, 3], price=[10.0, None, 30.0]),
        kwargs={**BY_SECURITY, "how": "left"},
    ),
    Case(
        "inner_join_excludes_unmatched_range",
        frames=(_trades([D(2024, 1, 15), D(2024, 6, 15), D(2024, 3, 15)]), JAN_MAR),
        expect=_expect(trade_id=[1, 3], price=[10.0, 30.0]),
        kwargs={**BY_SECURITY, "how": "inner"},
    ),
    Case(
        "gap_between_ranges_returns_null",
        frames=(_trades([D(2024, 2, 15)]), JAN_MAR),
        expect=_expect(trade_id=[1], price=[None]),
        kwargs={**BY_SECURITY, "how": "left"},
    ),
    Case(
        "date_before_all_ranges",
        frames=(_trades([D(2023, 12, 15)]), JAN_ONLY),
        expect=_expect(trade_id=[1], price=[None]),
        kwargs={**BY_SECURITY, "how": "left"},
    ),
    Case(
        "date_after_all_ranges",
        frames=(_trades([D(2024, 12, 15)]), JAN_ONLY),
        expect=_expect(trade_id=[1], price=[None]),
        kwargs={**BY_SECURITY, "how": "left"},
    ),
    Case(
        "boundary_dates_inclusive",
        frames=(_trades([D(2024, 1, 1), D(2024, 1, 15), D(2024, 1, 31)]), JAN_ONLY),
        expect=_expect(trade_id=[1, 2, 3], price=[10.0] * 3),
        kwargs=BY_SECURITY,
    ),
    Case(
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
        expect=_expect(event_id=[1, 2, 3], session_name=["Morning", "Afternoon", None]),
        kwargs={"left_on": "event_time", "right_on_start": "session_start", "right_on_end": "session_end", "by": "session_type", "how": "left"},
    ),
    # Regressions: a name shared by both sides, or by right_on_end or right_on_start and left_on.
    Case(
        "column_name_collision_nulls_right_not_left",
        frames=(pl.DataFrame({"ts": [5], "value": ["left"]}), pl.DataFrame({"start": [1], "end": [3], "value": ["right"]})),
        expect=_expect(value=["left"], value_right=[None]),
        kwargs={"left_on": "ts", "right_on_start": "start", "right_on_end": "end", "how": "left"},
    ),
    # Window B ends at 10:00:04, before the reading at 10:00:05.
    Case(
        "right_on_end_name_collision_validation",
        frames=(
            pl.DataFrame({"timestamp": [datetime(2023, 1, 1, 10, 0, 5)], "value": [99]}),
            WINDOWS.with_columns(timestamp=pl.Series([datetime(2023, 1, 1, 10, 0, 10), datetime(2023, 1, 1, 10, 0, 4)])),
        ),
        expect=_expect(value=pl.Series([], dtype=pl.Int64)),
        kwargs={"left_on": "timestamp", "right_on_start": "window_start", "right_on_end": "timestamp", "how": "inner"},
    ),
    Case(
        "right_on_start_equal_left_on_preserves_left_key",
        frames=(pl.DataFrame({"x": [10]}), pl.DataFrame({"x": [1], "end": [5], "payload": [99]})),
        expect=_expect(x=[10], end=pl.Series([None], dtype=pl.Int64), payload=pl.Series([None], dtype=pl.Int64)),
        kwargs={"left_on": "x", "right_on_start": "x", "right_on_end": "end", "how": "left"},
    ),
    # At most one match per left row: the window with the nearest start (B), though A contains 10:00:04 too.
    Case(
        "overlapping_intervals_returns_single_match",
        frames=(
            pl.DataFrame({"timestamp": [datetime(2023, 1, 1, 10, 0, 4)], "value": [99]}),
            WINDOWS.with_columns(window_end=pl.Series([datetime(2023, 1, 1, 10, 0, 5), datetime(2023, 1, 1, 10, 0, 8)])),
        ),
        expect=_expect(event_id=["B"]),
        kwargs={"left_on": "timestamp", "right_on_start": "window_start", "right_on_end": "window_end", "how": "inner"},
    ),
]


@pytest.mark.parametrize(("frames", "expect", "kwargs"), params(JOIN_BETWEEN, "frames", "expect", "kwargs"))
def test_join_between(frames, expect, kwargs):
    left, right = (df.lazy() for df in frames)
    result = join_between(left, right, **kwargs).collect()
    assert_frame_equal(result.select(expect.columns).sort(expect.columns[0]), expect, check_dtypes=False)
