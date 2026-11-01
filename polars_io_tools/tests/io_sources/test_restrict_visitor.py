"""restrict_expr_to_columns: a table of (predicate, allowed columns) -> the weaker predicate on those columns only.

``None`` means nothing can be kept. Restricting may keep extra rows but must never drop a row the predicate keeps.
"""

from collections import Counter

import polars as pl
import pytest

import polars_io_tools as cpl
from polars_io_tools.io_sources.base import BinaryExprNode, ColumnNode, get_parsed_expr
from polars_io_tools.io_sources.enum import OperatorType
from polars_io_tools.io_sources.restrict_visitor import RestrictPredicateVisitor, restrict_expr_to_columns
from polars_io_tools.tests.helpers.assertions import assert_matches_native
from polars_io_tools.tests.helpers.cases import Case, params

c = pl.col
p, q, xc = c("p"), c("q"), c("x")
sym, price, vol = c("symbol"), c("price"), c("volume")
B = pl.Boolean


def R(id, pred, columns, expect, node=False, undo_aliases=False):
    """``node`` passes the parsed node instead of the expression; ``undo_aliases`` compares the result without aliases."""
    return Case(id, pred=pred, columns=columns, node=node, undo_aliases=undo_aliases, expect=expect)


_COMPLEX = ((sym == "US") & (price > 100)) | ((sym == "EU") & (price > 200))
_TERNARY_IRRELEVANT_PREDICATE = pl.when(c("date") > "2023-01-01").then(price * 1.2).otherwise(vol * 2)
_ALIASED = (price.alias("p1").alias("p2") > 100) & (vol.alias("v1") > 1000)
_DATE_AND_VAL = (c("date") > "2024-01-01", c("val") > 5)
_NOT_EVEN = (c("A").mod(2) == 0).not_().fill_null(True)

CASES = [
    # basics
    R("restrict_predicate-column", p, ["p"], p),
    R("restrict_predicate-column_with_spare_allowed", p, ["p", "q"], p),
    R("restrict_predicate-is_in", p.is_in([1, 2]), ["p"], p.is_in([1, 2])),
    R("restrict_predicate-and_drops_other_column", p & xc, ["p"], p),
    R("restrict_predicate-and_keeps_both_allowed", p.is_in([1, 2]) & xc & p.is_in([4, 5]), ["p"], p.is_in([1, 2]) & p.is_in([4, 5])),
    R("restrict_predicate-or_with_other_column", p | xc, ["p"], None),
    R("restrict_predicate-and_two_allowed_columns", p & xc & q, ["p", "q"], p & q),
    R("restrict_predicate-or_two_allowed_columns", p | xc | q, ["p", "q"], None),
    R("restrict_predicate-arithmetic_on_allowed", (p * q) < 4, ["p", "q"], (p * q) < 4),
    R("restrict_predicate-arithmetic_with_other", (p + xc) < 4, ["p", "q"], None),
    R("restrict_predicate-alias", p.alias("q"), ["p"], p.alias("q")),
    R("restrict_predicate-function_with_other", pl.arctan2(q, p) > 12, ["p"], None),
    R("basic_comparison_restriction", sym == "US", {"symbol"}, sym == "US"),
    R("simple_and_restriction", (sym == "US") & (price > 100), {"symbol"}, sym == "US"),
    R(
        "and_multiple_restriction",
        (sym == "US") & (price > 100) & (c("exchange") == "NYSE"),
        {"symbol", "exchange"},
        (sym == "US") & (c("exchange") == "NYSE"),
    ),
    R("or_no_restriction", (sym == "US") | (price > 100), {"symbol"}, None),
    R("or_with_restriction", (sym == "US") | (sym == "EU"), {"symbol"}, (sym == "US") | (sym == "EU")),
    R("complex_nested_expression-expr", _COMPLEX, {"symbol"}, (sym == "US") | (sym == "EU")),
    R("complex_nested_expression-node", _COMPLEX, {"symbol"}, (sym == "US") | (sym == "EU"), node=True),
    R("complex_nested_expression-all_columns_expr", _COMPLEX, {"symbol", "price"}, _COMPLEX),
    R("complex_nested_expression-all_columns_node", _COMPLEX, {"symbol", "price"}, _COMPLEX, node=True),
    # De Morgan is not applied: the Polars optimizer pushes NOT down before a source sees the predicate.
    R("not_node_restriction", ~(sym == "US") & (price > 100), {"symbol"}, ~(sym == "US")),
    R("is_null_restriction", sym.is_null() & (price > 100), {"symbol"}, sym.is_null()),
    R("is_in_restriction", sym.is_in(["US", "EU"]) & (price > 100), {"symbol"}, sym.is_in(["US", "EU"])),
    R("string_function_restriction", sym.str.contains("US") & (price > 100), {"symbol"}, sym.str.contains("US")),
    R("is_between_restriction", sym.is_between(5, 10) & (price > 100), {"symbol"}, sym.is_between(5, 10)),
    R("nested_and_or_restriction", (sym == "US") & ((price > 100) | (vol > 1000)), {"symbol"}, sym == "US"),
    R("no_valid_restriction", (price > 100) | (vol > 1000), {"symbol"}, None),
    # ternaries
    R("ternary_truthy_relevant", pl.when(c("id") == 10).then(c("int_val") > 90).otherwise(c("int_val2") > 100), {"int_val"}, None),
    R(
        "ternary_truthy_and_falsy_relevant",
        pl.when(c("id") == 10).then(c("int_val") > 90).otherwise(c("int_val") > 100),
        {"int_val"},
        (c("int_val") > 90) | (c("int_val") > 100),
    ),
    R(
        "ternary_all_relevant",
        pl.when(c("int_val").mod(2) == 0).then(c("int_val") > 90).otherwise(c("int_val") > 100),
        {"int_val"},
        pl.when(c("int_val").mod(2) == 0).then(c("int_val") > 90).otherwise(c("int_val") > 100),
    ),
    R(
        "ternary_complex_predicate",
        pl.when((sym == "AAPL") & (price > 150)).then(vol * 2).otherwise(vol),
        {"symbol", "volume"},
        (sym == "AAPL").and_(vol * 2) | vol,
    ),
    R(
        "ternary_nested",
        pl.when(c("region") == "US").then(pl.when(c("sector") == "Tech").then((price * 1.1) > 1).otherwise(price < 12)).otherwise((price * 0.9) > 1),
        {"region", "price"},
        ((c("region") == "US") & ((price * 1.1 > 1) | (price < 12))) | ((c("region") == "US").not_().fill_null(True) & ((price * 0.9) > 1)),
    ),
    R("ternary_with_irrelevant_predicate", _TERNARY_IRRELEVANT_PREDICATE, {"price", "volume"}, (price * 1.2) | (vol * 2)),
    R("ternary_only_relevant_truthy", _TERNARY_IRRELEVANT_PREDICATE, {"price"}, None),
    R("ternary_only_relevant_falsy", pl.when(c("date") > "2023-01-01").then(price * 1.2).otherwise(sym.str.to_uppercase()), {"symbol"}, None),
    R(
        "ternary_combined_with_binary_op",
        pl.when(c("region") == "US").then(price > 100).otherwise(vol > 1000) & (c("date") > "2023-01-01"),
        {"region", "price"},
        ((c("region") == "US") & (price > 100)) | (c("region") == "US").not_().fill_null(True),
    ),
    # both branches unrepresentable: the predicate or its negation is trivially true
    R(
        "ternary_with_complex_expressions",
        pl.when((c("region") == "US") | (c("region") == "EU")).then((price * vol).alias("value") > 100).otherwise((price - c("discount")) > 5),
        {"region", "price"},
        None,
    ),
    R("ternary_all_parts_irrelevant", _TERNARY_IRRELEVANT_PREDICATE, {"symbol"}, None),
    R(
        "ternary_predicate_str_function",
        pl.when(sym.str.contains("HI").alias("hey")).then(price > 3).otherwise(vol != 13),
        {"symbol", "price"},
        (sym.str.contains("HI") & (price > 3)) | sym.str.contains("HI").not_().fill_null(True),
        undo_aliases=True,
    ),
    R(
        "ternary_complex_comparison",
        pl.when((c("open") <= c("close")) & (vol > 1000)).then(sym.str.contains("UP")).otherwise(c("open") > 100),
        {"open", "close"},
        (c("open") <= c("close")) | (c("open") > 100),
    ),
    # An odd A depends on B and C, which are not available, so every odd A is kept.
    R("not_pred_valid_pred_invalid", pl.when((c("A").mod(2) == 0) | c("B")).then(c("A") > 15).otherwise(c("C")), {"A"}, (c("A") > 15) | _NOT_EVEN),
    R(
        "when_then_otherwise_with_definitive_exclusions",
        pl.when(c("B") > 5).then((c("A") < 20) & (c("C") > 0)).otherwise(c("A").mod(3) > 0),
        {"A"},
        (c("A") < 20) | (c("A").mod(3) > 0),
    ),
    # casts
    R("simple_cast-boolean", c("flag").cast(B), {"flag"}, c("flag").cast(B)),
    R("simple_cast-non_boolean", c("value").cast(pl.Int64), {"value"}, c("value").cast(pl.Int64)),
    R("simple_cast-other_column", (c("value") + c("other")).cast(pl.Int64), {"value"}, None),
    # unknown dtypes: the AND may be bitwise and cannot be restricted
    R("cast_with_binary_expr-cast_of_and", (c("flag") & c("active")).cast(B), {"flag"}, None),
    R("cast_with_binary_expr-cast_inside_and", c("flag").cast(B) & c("active"), {"flag"}, c("flag").cast(B)),
    R("cast_with_binary_expr-cast_inside_or", c("flag").cast(B) | c("active").cast(B), {"flag", "active"}, c("flag").cast(B) | c("active").cast(B)),
    R("multiple_casts-nested", c("value").cast(pl.Int64).cast(B), {"value"}, c("value").cast(pl.Int64).cast(B)),
    R("multiple_casts-nested_comparison", (c("value") > 0).cast(pl.Int64).cast(B), {"value"}, (c("value") > 0).cast(pl.Int64).cast(B)),
    # the outer Boolean cast is stripped
    R("multiple_casts-and", ((c("value") > 0).cast(B) & (c("other") > 10).cast(B)).cast(B), {"value"}, (c("value") > 0).cast(B)),
    R("cast_with_not-not_after_cast", ~c("flag").cast(B), {"flag"}, ~c("flag").cast(B)),
    R("cast_with_not-cast_after_not", (~c("flag")).cast(B), {"flag"}, (~c("flag")).cast(B)),
    R("cast_with_not-and", ~c("flag").cast(B) & c("value"), {"flag"}, ~c("flag").cast(B)),
    R(
        "cast_in_ternary-predicate",
        pl.when(c("flag").cast(B)).then(c("value") > 0).otherwise(price > 100),
        {"flag", "value"},
        (c("flag").cast(B) & (c("value") > 0)) | c("flag").cast(B).not_().fill_null(True),
    ),
    R(
        "cast_in_ternary-branches",
        pl.when(c("flag")).then((c("value") > 0).cast(B)).otherwise((price > 100).cast(B)),
        {"flag", "value"},
        (c("flag") & (c("value") > 0).cast(B)) | c("flag").not_().fill_null(True),
    ),
    R("cast_boolean_complex_logic-not_or", ~((c("a") > 0).cast(B) | (c("b") < 10).cast(B)), {"a"}, ~(c("a") > 0).cast(B)),
    R(
        "cast_boolean_complex_logic-or_of_ands",
        ((c("a") > 0).cast(B) & (c("b") < 10).cast(B)) | ((c("c") == 5).cast(B) & (c("a") != 0).cast(B)),
        {"a"},
        (c("a") > 0).cast(B) | (c("a") != 0).cast(B),
    ),
    R(
        "cast_between_operations",
        ((c("a") > 0).cast(B) & c("b").is_null()).cast(B) | (c("a") < 0).cast(B),
        {"a"},
        (c("a") > 0).cast(B) | (c("a") < 0).cast(B),
    ),
    R("cast_non_boolean_handling-and", (c("a").cast(pl.Int32) > 0) & c("b"), {"a"}, c("a").cast(pl.Int32) > 0),
    R("cast_non_boolean_handling-other_column", (c("a") + c("b")).cast(pl.Int32) > 0, {"a"}, None),
    # aliases
    R("nested_alias_handling-nested", price.alias("p1").alias("p2") > 100, {"price"}, price.alias("p1").alias("p2") > 100),
    R(
        "nested_alias_handling-with_cast",
        price.cast(pl.Float64).alias("p1").alias("p2") > 100,
        {"price"},
        price.cast(pl.Float64).alias("p1").alias("p2") > 100,
    ),
    R("nested_alias_handling-and_all_columns", _ALIASED, {"price", "volume"}, _ALIASED),
    R("nested_alias_handling-and_subset", _ALIASED, {"price"}, price.alias("p1").alias("p2") > 100),
    # horizontal functions
    R("all_horizontal_restriction-subset", pl.all_horizontal(*_DATE_AND_VAL), {"date"}, _DATE_AND_VAL[0]),
    R("all_horizontal_restriction-all_columns", pl.all_horizontal(*_DATE_AND_VAL), {"date", "val"}, pl.all_horizontal(*_DATE_AND_VAL)),
    R("all_horizontal_restriction-none", pl.all_horizontal(*_DATE_AND_VAL), {"other"}, None),
    R("all_horizontal_all_same_column", pl.all_horizontal(xc > 0, xc < 100), {"x"}, pl.all_horizontal(xc > 0, xc < 100)),
    R("any_horizontal_restriction-subset", pl.any_horizontal(*_DATE_AND_VAL), {"date"}, None),
    R("any_horizontal_restriction-all_columns", pl.any_horizontal(*_DATE_AND_VAL), {"date", "val"}, pl.any_horizontal(*_DATE_AND_VAL)),
    R("any_horizontal_all_same_column", pl.any_horizontal(xc == 1, xc == 2), {"x"}, pl.any_horizontal(xc == 1, xc == 2)),
    R(
        "all_horizontal_nested",
        pl.all_horizontal(pl.all_horizontal(c("a") > 0, c("b") > 0), c("c") > 0),
        {"a", "b"},
        pl.all_horizontal(c("a") > 0, c("b") > 0),
    ),
]


@pytest.mark.parametrize(("pred", "columns", "node", "undo_aliases", "expect"), params(CASES, "pred", "columns", "node", "undo_aliases", "expect"))
def test_restrict(pred, columns, node, undo_aliases, expect):
    result = restrict_expr_to_columns(get_parsed_expr(pred) if node else pred, columns)
    if expect is None:
        assert result is None
    else:
        assert result is not None and (result.meta.undo_aliases() if undo_aliases else result).meta.eq(expect), result


def test_restrict_predicate_visitor_direct():
    visitor = RestrictPredicateVisitor({"symbol"})
    visitor.visit(get_parsed_expr((sym == "US") & (price > 100)))
    assert visitor.process_results().meta.eq(sym == "US")


def test_not_node_restriction_pushed_down(tester):
    """The predicate a source receives has NOT pushed down already, so restricting it needs no De Morgan step."""

    def restricts_to_id_ne_10(original_expr, pushed_expr):
        # ``meta.eq`` does not compare a pushed predicate reliably, so the parsed node is checked.
        node = get_parsed_expr(restrict_expr_to_columns(pushed_expr, {"id"}))
        assert isinstance(node, BinaryExprNode) and node.op == OperatorType.NOT_EQ
        assert isinstance(node.left, ColumnNode) and node.left.name == "id"
        assert node.right.can_extract_literal and node.right.value == 10

    tester.assert_predicate_pushed_down(~((c("id") == 10) | (c("int_val") > 90)), restricts_to_id_ne_10)

    # Pushed as ``(id != 10) | (int_val <= 90)``, which cannot be restricted to ``id``.
    def unrestricted(original_expr, pushed_expr):
        assert restrict_expr_to_columns(pushed_expr, {"id"}) is None

    tester.assert_predicate_pushed_down(~((c("id") == 10) & (c("int_val") > 90)), unrestricted)


_NULLABLE_DATA = pl.DataFrame(
    {"a": [None, None, 5, 0, 5, 2], "b": [None, 1, 0, 1, 0, None], "ab": [1, 2, 300, 4, 0, 7]},
    schema={"a": pl.Int64, "b": pl.Int64, "ab": pl.Int64},
)


@pytest.mark.parametrize(
    ("expr", "columns"),
    [
        pytest.param(pl.when(c("a") > 1).then(c("b") > 100).otherwise(c("a").fill_null(0) < 3), {"a"}, id="ternary_null_condition"),
        pytest.param(~pl.when(c("b") == 1).then(c("a") > 1).otherwise((c("a") > 0) & (c("ab") > 100)), {"a", "b"}, id="negated_ternary"),
        pytest.param(~pl.all_horizontal(c("a") > 1, c("b") > 0), {"a"}, id="negated_all_horizontal"),
        pytest.param((c("a") & c("b")).cast(B), {"a"}, id="bitwise_and_cast"),
        pytest.param(~((c("a") | c("b")).cast(B)), {"a"}, id="negated_bitwise_or_cast"),
        pytest.param(pl.when(c("a") > 1).then(c("b") * 2).otherwise(1).cast(B), {"a"}, id="numeric_ternary_cast"),
        pytest.param(pl.any_horizontal(pl.all() > 0), {"a"}, id="any_horizontal_all"),
        pytest.param(pl.any_horizontal(c("^a.*$") > 0), {"a"}, id="any_horizontal_regex"),
        pytest.param(pl.all_horizontal(c("a", "b") > 0), {"a"}, id="all_horizontal_multi_column"),
        pytest.param(pl.all_horizontal(c("a"), c("b") > 0), {"a"}, id="all_horizontal_bare_column"),
    ],
)
def test_restricted_predicate_keeps_all_matching_rows(expr, columns):
    """The restricted predicate, evaluated on the kept columns only, must keep every row the original keeps."""
    visitor = RestrictPredicateVisitor(columns)
    visitor.visit(get_parsed_expr(expr))
    cols = sorted(columns)
    kept = Counter(_NULLABLE_DATA.filter(expr).select(cols).rows())
    for restricted in (restrict_expr_to_columns(expr, columns), visitor.process_results()):
        if restricted is not None:
            restricted_kept = Counter(_NULLABLE_DATA.select(cols).filter(restricted).rows())
            assert not kept - restricted_kept


def test_ternary_with_null_condition_end_to_end():
    data = pl.DataFrame({"k": [None, None, 5, 0], "x": [1, 2, 300, 4]})
    pred = pl.when(c("k") > 1).then(c("x") > 100).otherwise(c("k").fill_null(0) < 3)
    assert_matches_native(cpl.cache_memory(lambda: data.lazy(), schema=data.schema, partition_cols="k"), data.lazy(), pred)


def test_negated_ternary_end_to_end():
    frames = {(None, 1): pl.DataFrame({"x": [1, 200]}), (5, 0): pl.DataFrame({"x": [2, 300]})}
    pred = ~pl.when(c("k2") == 1).then(c("k") > 1).otherwise((c("k") > 0) & (c("x") > 100))
    lf = cpl.concat_named({key: df.lazy() for key, df in frames.items()}, [("k", pl.Int64), ("k2", pl.Int64)])
    assert_matches_native(lf, lf.collect().lazy(), pred)
