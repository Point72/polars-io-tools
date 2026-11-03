"""restrict_expr_to_columns: turn a filter into one that reads only some of its columns.

The result may keep extra rows but must never drop a row the original filter keeps. ``None`` means no filter on the
allowed columns can do that, so every row has to be read.
"""

from collections import Counter
from dataclasses import KW_ONLY, dataclass

import polars as pl
import pytest

import polars_io_tools as cpl
from polars_io_tools.io_sources.base import BinaryExprNode, ColumnNode, get_parsed_expr
from polars_io_tools.io_sources.enum import OperatorType
from polars_io_tools.io_sources.restrict_visitor import RestrictPredicateVisitor, restrict_expr_to_columns
from polars_io_tools.tests.helpers.assertions import assert_matches_plain_polars
from polars_io_tools.tests.helpers.cases import as_params


@dataclass(frozen=True)
class RestrictCase:
    """A filter, the columns it may keep, and the filter it becomes."""

    id: str
    _: KW_ONLY
    predicate: pl.Expr
    allowed_columns: list[str]
    # None: the filter cannot be restricted to the allowed columns
    expected: pl.Expr | None
    # hand over the parsed form of the filter instead of the Polars expression
    pass_parsed_node: bool = False
    # remove ``.alias`` calls from the result before comparing it
    compare_without_aliases: bool = False


_SYMBOL_AND_PRICE_RULES = ((pl.col("symbol") == "US") & (pl.col("price") > 100)) | ((pl.col("symbol") == "EU") & (pl.col("price") > 200))
_WHEN_DATE_THEN_PRICE_ELSE_VOLUME = pl.when(pl.col("date") > "2023-01-01").then(pl.col("price") * 1.2).otherwise(pl.col("volume") * 2)
_ALIASED_PRICE_AND_VOLUME = (pl.col("price").alias("p1").alias("p2") > 100) & (pl.col("volume").alias("v1") > 1000)
_DATE_AND_VAL_CONDITIONS = (pl.col("date") > "2024-01-01", pl.col("val") > 5)
_A_NOT_EVEN_OR_NULL = (pl.col("A").mod(2) == 0).not_().fill_null(True)

CASES = [
    # basics
    RestrictCase("restrict_predicate-column", predicate=pl.col("p"), allowed_columns=["p"], expected=pl.col("p")),
    RestrictCase("restrict_predicate-column_with_spare_allowed", predicate=pl.col("p"), allowed_columns=["p", "q"], expected=pl.col("p")),
    RestrictCase("restrict_predicate-is_in", predicate=pl.col("p").is_in([1, 2]), allowed_columns=["p"], expected=pl.col("p").is_in([1, 2])),
    RestrictCase("restrict_predicate-and_drops_other_column", predicate=pl.col("p") & pl.col("x"), allowed_columns=["p"], expected=pl.col("p")),
    RestrictCase(
        "restrict_predicate-and_keeps_both_allowed",
        predicate=pl.col("p").is_in([1, 2]) & pl.col("x") & pl.col("p").is_in([4, 5]),
        allowed_columns=["p"],
        expected=pl.col("p").is_in([1, 2]) & pl.col("p").is_in([4, 5]),
    ),
    RestrictCase("restrict_predicate-or_with_other_column", predicate=pl.col("p") | pl.col("x"), allowed_columns=["p"], expected=None),
    RestrictCase(
        "restrict_predicate-and_two_allowed_columns",
        predicate=pl.col("p") & pl.col("x") & pl.col("q"),
        allowed_columns=["p", "q"],
        expected=pl.col("p") & pl.col("q"),
    ),
    RestrictCase(
        "restrict_predicate-or_two_allowed_columns", predicate=pl.col("p") | pl.col("x") | pl.col("q"), allowed_columns=["p", "q"], expected=None
    ),
    RestrictCase(
        "restrict_predicate-arithmetic_on_allowed",
        predicate=(pl.col("p") * pl.col("q")) < 4,
        allowed_columns=["p", "q"],
        expected=(pl.col("p") * pl.col("q")) < 4,
    ),
    RestrictCase("restrict_predicate-arithmetic_with_other", predicate=(pl.col("p") + pl.col("x")) < 4, allowed_columns=["p", "q"], expected=None),
    RestrictCase("restrict_predicate-alias", predicate=pl.col("p").alias("q"), allowed_columns=["p"], expected=pl.col("p").alias("q")),
    RestrictCase("restrict_predicate-function_with_other", predicate=pl.arctan2(pl.col("q"), pl.col("p")) > 12, allowed_columns=["p"], expected=None),
    RestrictCase("basic_comparison_restriction", predicate=pl.col("symbol") == "US", allowed_columns={"symbol"}, expected=pl.col("symbol") == "US"),
    RestrictCase(
        "simple_and_restriction",
        predicate=(pl.col("symbol") == "US") & (pl.col("price") > 100),
        allowed_columns={"symbol"},
        expected=pl.col("symbol") == "US",
    ),
    RestrictCase(
        "and_multiple_restriction",
        predicate=(pl.col("symbol") == "US") & (pl.col("price") > 100) & (pl.col("exchange") == "NYSE"),
        allowed_columns={"symbol", "exchange"},
        expected=(pl.col("symbol") == "US") & (pl.col("exchange") == "NYSE"),
    ),
    RestrictCase("or_no_restriction", predicate=(pl.col("symbol") == "US") | (pl.col("price") > 100), allowed_columns={"symbol"}, expected=None),
    RestrictCase(
        "or_with_restriction",
        predicate=(pl.col("symbol") == "US") | (pl.col("symbol") == "EU"),
        allowed_columns={"symbol"},
        expected=(pl.col("symbol") == "US") | (pl.col("symbol") == "EU"),
    ),
    RestrictCase(
        "complex_nested_expression-expr",
        predicate=_SYMBOL_AND_PRICE_RULES,
        allowed_columns={"symbol"},
        expected=(pl.col("symbol") == "US") | (pl.col("symbol") == "EU"),
    ),
    RestrictCase(
        "complex_nested_expression-node",
        predicate=_SYMBOL_AND_PRICE_RULES,
        allowed_columns={"symbol"},
        expected=(pl.col("symbol") == "US") | (pl.col("symbol") == "EU"),
        pass_parsed_node=True,
    ),
    RestrictCase(
        "complex_nested_expression-all_columns_expr",
        predicate=_SYMBOL_AND_PRICE_RULES,
        allowed_columns={"symbol", "price"},
        expected=_SYMBOL_AND_PRICE_RULES,
    ),
    RestrictCase(
        "complex_nested_expression-all_columns_node",
        predicate=_SYMBOL_AND_PRICE_RULES,
        allowed_columns={"symbol", "price"},
        expected=_SYMBOL_AND_PRICE_RULES,
        pass_parsed_node=True,
    ),
    # ``~(a == b) & c`` is not rewritten as ``(a != b) & c``: Polars already moves ``~`` inward before a source sees the filter.
    RestrictCase(
        "not_node_restriction",
        predicate=~(pl.col("symbol") == "US") & (pl.col("price") > 100),
        allowed_columns={"symbol"},
        expected=~(pl.col("symbol") == "US"),
    ),
    RestrictCase(
        "is_null_restriction",
        predicate=pl.col("symbol").is_null() & (pl.col("price") > 100),
        allowed_columns={"symbol"},
        expected=pl.col("symbol").is_null(),
    ),
    RestrictCase(
        "is_in_restriction",
        predicate=pl.col("symbol").is_in(["US", "EU"]) & (pl.col("price") > 100),
        allowed_columns={"symbol"},
        expected=pl.col("symbol").is_in(["US", "EU"]),
    ),
    RestrictCase(
        "string_function_restriction",
        predicate=pl.col("symbol").str.contains("US") & (pl.col("price") > 100),
        allowed_columns={"symbol"},
        expected=pl.col("symbol").str.contains("US"),
    ),
    RestrictCase(
        "is_between_restriction",
        predicate=pl.col("symbol").is_between(5, 10) & (pl.col("price") > 100),
        allowed_columns={"symbol"},
        expected=pl.col("symbol").is_between(5, 10),
    ),
    RestrictCase(
        "nested_and_or_restriction",
        predicate=(pl.col("symbol") == "US") & ((pl.col("price") > 100) | (pl.col("volume") > 1000)),
        allowed_columns={"symbol"},
        expected=pl.col("symbol") == "US",
    ),
    RestrictCase("no_valid_restriction", predicate=(pl.col("price") > 100) | (pl.col("volume") > 1000), allowed_columns={"symbol"}, expected=None),
    # when/then/otherwise
    RestrictCase(
        "ternary_truthy_relevant",
        predicate=pl.when(pl.col("id") == 10).then(pl.col("int_val") > 90).otherwise(pl.col("int_val2") > 100),
        allowed_columns={"int_val"},
        expected=None,
    ),
    RestrictCase(
        "ternary_truthy_and_falsy_relevant",
        predicate=pl.when(pl.col("id") == 10).then(pl.col("int_val") > 90).otherwise(pl.col("int_val") > 100),
        allowed_columns={"int_val"},
        expected=(pl.col("int_val") > 90) | (pl.col("int_val") > 100),
    ),
    RestrictCase(
        "ternary_all_relevant",
        predicate=pl.when(pl.col("int_val").mod(2) == 0).then(pl.col("int_val") > 90).otherwise(pl.col("int_val") > 100),
        allowed_columns={"int_val"},
        expected=pl.when(pl.col("int_val").mod(2) == 0).then(pl.col("int_val") > 90).otherwise(pl.col("int_val") > 100),
    ),
    RestrictCase(
        "ternary_complex_predicate",
        predicate=pl.when((pl.col("symbol") == "AAPL") & (pl.col("price") > 150)).then(pl.col("volume") * 2).otherwise(pl.col("volume")),
        allowed_columns={"symbol", "volume"},
        expected=(pl.col("symbol") == "AAPL").and_(pl.col("volume") * 2) | pl.col("volume"),
    ),
    RestrictCase(
        "ternary_nested",
        predicate=pl.when(pl.col("region") == "US")
        .then(pl.when(pl.col("sector") == "Tech").then((pl.col("price") * 1.1) > 1).otherwise(pl.col("price") < 12))
        .otherwise((pl.col("price") * 0.9) > 1),
        allowed_columns={"region", "price"},
        expected=((pl.col("region") == "US") & ((pl.col("price") * 1.1 > 1) | (pl.col("price") < 12)))
        | ((pl.col("region") == "US").not_().fill_null(True) & ((pl.col("price") * 0.9) > 1)),
    ),
    RestrictCase(
        "ternary_with_irrelevant_predicate",
        predicate=_WHEN_DATE_THEN_PRICE_ELSE_VOLUME,
        allowed_columns={"price", "volume"},
        expected=(pl.col("price") * 1.2) | (pl.col("volume") * 2),
    ),
    RestrictCase("ternary_only_relevant_truthy", predicate=_WHEN_DATE_THEN_PRICE_ELSE_VOLUME, allowed_columns={"price"}, expected=None),
    RestrictCase(
        "ternary_only_relevant_falsy",
        predicate=pl.when(pl.col("date") > "2023-01-01").then(pl.col("price") * 1.2).otherwise(pl.col("symbol").str.to_uppercase()),
        allowed_columns={"symbol"},
        expected=None,
    ),
    RestrictCase(
        "ternary_combined_with_binary_op",
        predicate=pl.when(pl.col("region") == "US").then(pl.col("price") > 100).otherwise(pl.col("volume") > 1000) & (pl.col("date") > "2023-01-01"),
        allowed_columns={"region", "price"},
        expected=((pl.col("region") == "US") & (pl.col("price") > 100)) | (pl.col("region") == "US").not_().fill_null(True),
    ),
    # Neither branch can be written with the allowed columns, so the result could only be ``True``: nothing is kept.
    RestrictCase(
        "ternary_with_complex_expressions",
        predicate=pl.when((pl.col("region") == "US") | (pl.col("region") == "EU"))
        .then((pl.col("price") * pl.col("volume")).alias("value") > 100)
        .otherwise((pl.col("price") - pl.col("discount")) > 5),
        allowed_columns={"region", "price"},
        expected=None,
    ),
    RestrictCase("ternary_all_parts_irrelevant", predicate=_WHEN_DATE_THEN_PRICE_ELSE_VOLUME, allowed_columns={"symbol"}, expected=None),
    RestrictCase(
        "ternary_predicate_str_function",
        predicate=pl.when(pl.col("symbol").str.contains("HI").alias("hey")).then(pl.col("price") > 3).otherwise(pl.col("volume") != 13),
        allowed_columns={"symbol", "price"},
        expected=(pl.col("symbol").str.contains("HI") & (pl.col("price") > 3)) | pl.col("symbol").str.contains("HI").not_().fill_null(True),
        compare_without_aliases=True,
    ),
    RestrictCase(
        "ternary_complex_comparison",
        predicate=pl.when((pl.col("open") <= pl.col("close")) & (pl.col("volume") > 1000))
        .then(pl.col("symbol").str.contains("UP"))
        .otherwise(pl.col("open") > 100),
        allowed_columns={"open", "close"},
        expected=(pl.col("open") <= pl.col("close")) | (pl.col("open") > 100),
    ),
    # An odd A depends on B and C, which are not available, so every odd A is kept.
    RestrictCase(
        "not_pred_valid_pred_invalid",
        predicate=pl.when((pl.col("A").mod(2) == 0) | pl.col("B")).then(pl.col("A") > 15).otherwise(pl.col("C")),
        allowed_columns={"A"},
        expected=(pl.col("A") > 15) | _A_NOT_EVEN_OR_NULL,
    ),
    RestrictCase(
        "when_then_otherwise_with_definitive_exclusions",
        predicate=pl.when(pl.col("B") > 5).then((pl.col("A") < 20) & (pl.col("C") > 0)).otherwise(pl.col("A").mod(3) > 0),
        allowed_columns={"A"},
        expected=(pl.col("A") < 20) | (pl.col("A").mod(3) > 0),
    ),
    # casts
    RestrictCase(
        "simple_cast-boolean", predicate=pl.col("flag").cast(pl.Boolean), allowed_columns={"flag"}, expected=pl.col("flag").cast(pl.Boolean)
    ),
    RestrictCase(
        "simple_cast-non_boolean", predicate=pl.col("value").cast(pl.Int64), allowed_columns={"value"}, expected=pl.col("value").cast(pl.Int64)
    ),
    RestrictCase("simple_cast-other_column", predicate=(pl.col("value") + pl.col("other")).cast(pl.Int64), allowed_columns={"value"}, expected=None),
    # The dtypes are unknown, so ``&`` may be a bitwise AND of integers and cannot be split.
    RestrictCase(
        "cast_with_binary_expr-cast_of_and", predicate=(pl.col("flag") & pl.col("active")).cast(pl.Boolean), allowed_columns={"flag"}, expected=None
    ),
    RestrictCase(
        "cast_with_binary_expr-cast_inside_and",
        predicate=pl.col("flag").cast(pl.Boolean) & pl.col("active"),
        allowed_columns={"flag"},
        expected=pl.col("flag").cast(pl.Boolean),
    ),
    RestrictCase(
        "cast_with_binary_expr-cast_inside_or",
        predicate=pl.col("flag").cast(pl.Boolean) | pl.col("active").cast(pl.Boolean),
        allowed_columns={"flag", "active"},
        expected=pl.col("flag").cast(pl.Boolean) | pl.col("active").cast(pl.Boolean),
    ),
    RestrictCase(
        "multiple_casts-nested",
        predicate=pl.col("value").cast(pl.Int64).cast(pl.Boolean),
        allowed_columns={"value"},
        expected=pl.col("value").cast(pl.Int64).cast(pl.Boolean),
    ),
    RestrictCase(
        "multiple_casts-nested_comparison",
        predicate=(pl.col("value") > 0).cast(pl.Int64).cast(pl.Boolean),
        allowed_columns={"value"},
        expected=(pl.col("value") > 0).cast(pl.Int64).cast(pl.Boolean),
    ),
    # the outer Boolean cast is stripped
    RestrictCase(
        "multiple_casts-and",
        predicate=((pl.col("value") > 0).cast(pl.Boolean) & (pl.col("other") > 10).cast(pl.Boolean)).cast(pl.Boolean),
        allowed_columns={"value"},
        expected=(pl.col("value") > 0).cast(pl.Boolean),
    ),
    RestrictCase(
        "cast_with_not-not_after_cast",
        predicate=~pl.col("flag").cast(pl.Boolean),
        allowed_columns={"flag"},
        expected=~pl.col("flag").cast(pl.Boolean),
    ),
    RestrictCase(
        "cast_with_not-cast_after_not",
        predicate=(~pl.col("flag")).cast(pl.Boolean),
        allowed_columns={"flag"},
        expected=(~pl.col("flag")).cast(pl.Boolean),
    ),
    RestrictCase(
        "cast_with_not-and",
        predicate=~pl.col("flag").cast(pl.Boolean) & pl.col("value"),
        allowed_columns={"flag"},
        expected=~pl.col("flag").cast(pl.Boolean),
    ),
    RestrictCase(
        "cast_in_ternary-predicate",
        predicate=pl.when(pl.col("flag").cast(pl.Boolean)).then(pl.col("value") > 0).otherwise(pl.col("price") > 100),
        allowed_columns={"flag", "value"},
        expected=(pl.col("flag").cast(pl.Boolean) & (pl.col("value") > 0)) | pl.col("flag").cast(pl.Boolean).not_().fill_null(True),
    ),
    RestrictCase(
        "cast_in_ternary-branches",
        predicate=pl.when(pl.col("flag")).then((pl.col("value") > 0).cast(pl.Boolean)).otherwise((pl.col("price") > 100).cast(pl.Boolean)),
        allowed_columns={"flag", "value"},
        expected=(pl.col("flag") & (pl.col("value") > 0).cast(pl.Boolean)) | pl.col("flag").not_().fill_null(True),
    ),
    RestrictCase(
        "cast_boolean_complex_logic-not_or",
        predicate=~((pl.col("a") > 0).cast(pl.Boolean) | (pl.col("b") < 10).cast(pl.Boolean)),
        allowed_columns={"a"},
        expected=~(pl.col("a") > 0).cast(pl.Boolean),
    ),
    RestrictCase(
        "cast_boolean_complex_logic-or_of_ands",
        predicate=((pl.col("a") > 0).cast(pl.Boolean) & (pl.col("b") < 10).cast(pl.Boolean))
        | ((pl.col("c") == 5).cast(pl.Boolean) & (pl.col("a") != 0).cast(pl.Boolean)),
        allowed_columns={"a"},
        expected=(pl.col("a") > 0).cast(pl.Boolean) | (pl.col("a") != 0).cast(pl.Boolean),
    ),
    RestrictCase(
        "cast_between_operations",
        predicate=((pl.col("a") > 0).cast(pl.Boolean) & pl.col("b").is_null()).cast(pl.Boolean) | (pl.col("a") < 0).cast(pl.Boolean),
        allowed_columns={"a"},
        expected=(pl.col("a") > 0).cast(pl.Boolean) | (pl.col("a") < 0).cast(pl.Boolean),
    ),
    RestrictCase(
        "cast_non_boolean_handling-and",
        predicate=(pl.col("a").cast(pl.Int32) > 0) & pl.col("b"),
        allowed_columns={"a"},
        expected=pl.col("a").cast(pl.Int32) > 0,
    ),
    RestrictCase(
        "cast_non_boolean_handling-other_column", predicate=(pl.col("a") + pl.col("b")).cast(pl.Int32) > 0, allowed_columns={"a"}, expected=None
    ),
    # aliases
    RestrictCase(
        "nested_alias_handling-nested",
        predicate=pl.col("price").alias("p1").alias("p2") > 100,
        allowed_columns={"price"},
        expected=pl.col("price").alias("p1").alias("p2") > 100,
    ),
    RestrictCase(
        "nested_alias_handling-with_cast",
        predicate=pl.col("price").cast(pl.Float64).alias("p1").alias("p2") > 100,
        allowed_columns={"price"},
        expected=pl.col("price").cast(pl.Float64).alias("p1").alias("p2") > 100,
    ),
    RestrictCase(
        "nested_alias_handling-and_all_columns",
        predicate=_ALIASED_PRICE_AND_VOLUME,
        allowed_columns={"price", "volume"},
        expected=_ALIASED_PRICE_AND_VOLUME,
    ),
    RestrictCase(
        "nested_alias_handling-and_subset",
        predicate=_ALIASED_PRICE_AND_VOLUME,
        allowed_columns={"price"},
        expected=pl.col("price").alias("p1").alias("p2") > 100,
    ),
    # horizontal functions
    RestrictCase(
        "all_horizontal_restriction-subset",
        predicate=pl.all_horizontal(*_DATE_AND_VAL_CONDITIONS),
        allowed_columns={"date"},
        expected=_DATE_AND_VAL_CONDITIONS[0],
    ),
    RestrictCase(
        "all_horizontal_restriction-all_columns",
        predicate=pl.all_horizontal(*_DATE_AND_VAL_CONDITIONS),
        allowed_columns={"date", "val"},
        expected=pl.all_horizontal(*_DATE_AND_VAL_CONDITIONS),
    ),
    RestrictCase("all_horizontal_restriction-none", predicate=pl.all_horizontal(*_DATE_AND_VAL_CONDITIONS), allowed_columns={"other"}, expected=None),
    RestrictCase(
        "all_horizontal_all_same_column",
        predicate=pl.all_horizontal(pl.col("x") > 0, pl.col("x") < 100),
        allowed_columns={"x"},
        expected=pl.all_horizontal(pl.col("x") > 0, pl.col("x") < 100),
    ),
    RestrictCase(
        "any_horizontal_restriction-subset", predicate=pl.any_horizontal(*_DATE_AND_VAL_CONDITIONS), allowed_columns={"date"}, expected=None
    ),
    RestrictCase(
        "any_horizontal_restriction-all_columns",
        predicate=pl.any_horizontal(*_DATE_AND_VAL_CONDITIONS),
        allowed_columns={"date", "val"},
        expected=pl.any_horizontal(*_DATE_AND_VAL_CONDITIONS),
    ),
    RestrictCase(
        "any_horizontal_all_same_column",
        predicate=pl.any_horizontal(pl.col("x") == 1, pl.col("x") == 2),
        allowed_columns={"x"},
        expected=pl.any_horizontal(pl.col("x") == 1, pl.col("x") == 2),
    ),
    RestrictCase(
        "all_horizontal_nested",
        predicate=pl.all_horizontal(pl.all_horizontal(pl.col("a") > 0, pl.col("b") > 0), pl.col("c") > 0),
        allowed_columns={"a", "b"},
        expected=pl.all_horizontal(pl.col("a") > 0, pl.col("b") > 0),
    ),
]


@pytest.mark.parametrize("case", as_params(CASES))
def test_restrict(case: RestrictCase):
    result = restrict_expr_to_columns(get_parsed_expr(case.predicate) if case.pass_parsed_node else case.predicate, case.allowed_columns)
    if case.expected is None:
        assert result is None
    else:
        assert result is not None and (result.meta.undo_aliases() if case.compare_without_aliases else result).meta.eq(case.expected), result


def test_restrict_predicate_visitor_direct():
    visitor = RestrictPredicateVisitor({"symbol"})
    visitor.visit(get_parsed_expr((pl.col("symbol") == "US") & (pl.col("price") > 100)))
    assert visitor.process_results().meta.eq(pl.col("symbol") == "US")


def test_not_node_restriction_pushed_down(tracked_sample_data):
    """Polars moves ``~`` inward before a source receives the filter, so the restriction never has to do it."""

    def restricts_to_id_ne_10(original_expr, pushed_expr):
        # ``meta.eq`` does not compare a pushed predicate reliably, so the parsed node is checked.
        node = get_parsed_expr(restrict_expr_to_columns(pushed_expr, {"id"}))
        assert isinstance(node, BinaryExprNode) and node.op == OperatorType.NOT_EQ
        assert isinstance(node.left, ColumnNode) and node.left.name == "id"
        assert node.right.can_extract_literal and node.right.value == 10

    tracked_sample_data.assert_predicate_pushed_down(~((pl.col("id") == 10) | (pl.col("int_val") > 90)), restricts_to_id_ne_10)

    # Pushed as ``(id != 10) | (int_val <= 90)``, which cannot be restricted to ``id``.
    def cannot_restrict_to_id(original_expr, pushed_expr):
        assert restrict_expr_to_columns(pushed_expr, {"id"}) is None

    tracked_sample_data.assert_predicate_pushed_down(~((pl.col("id") == 10) & (pl.col("int_val") > 90)), cannot_restrict_to_id)


_NULLABLE_DATA = pl.DataFrame(
    {"a": [None, None, 5, 0, 5, 2], "b": [None, 1, 0, 1, 0, None], "ab": [1, 2, 300, 4, 0, 7]},
    schema={"a": pl.Int64, "b": pl.Int64, "ab": pl.Int64},
)


@dataclass(frozen=True)
class RestrictedKeepsRowsCase:
    """A filter and the columns it is restricted to; the result must keep every row the filter keeps."""

    id: str
    _: KW_ONLY
    expr: pl.Expr
    columns: set[str]


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            RestrictedKeepsRowsCase(
                "ternary_null_condition", expr=pl.when(pl.col("a") > 1).then(pl.col("b") > 100).otherwise(pl.col("a").fill_null(0) < 3), columns={"a"}
            ),
            RestrictedKeepsRowsCase(
                "negated_ternary",
                expr=~pl.when(pl.col("b") == 1).then(pl.col("a") > 1).otherwise((pl.col("a") > 0) & (pl.col("ab") > 100)),
                columns={"a", "b"},
            ),
            RestrictedKeepsRowsCase("negated_all_horizontal", expr=~pl.all_horizontal(pl.col("a") > 1, pl.col("b") > 0), columns={"a"}),
            RestrictedKeepsRowsCase("bitwise_and_cast", expr=(pl.col("a") & pl.col("b")).cast(pl.Boolean), columns={"a"}),
            RestrictedKeepsRowsCase("negated_bitwise_or_cast", expr=~((pl.col("a") | pl.col("b")).cast(pl.Boolean)), columns={"a"}),
            RestrictedKeepsRowsCase(
                "numeric_ternary_cast", expr=pl.when(pl.col("a") > 1).then(pl.col("b") * 2).otherwise(1).cast(pl.Boolean), columns={"a"}
            ),
            RestrictedKeepsRowsCase("any_horizontal_all", expr=pl.any_horizontal(pl.all() > 0), columns={"a"}),
            RestrictedKeepsRowsCase("any_horizontal_regex", expr=pl.any_horizontal(pl.col("^a.*$") > 0), columns={"a"}),
            RestrictedKeepsRowsCase("all_horizontal_multi_column", expr=pl.all_horizontal(pl.col("a", "b") > 0), columns={"a"}),
            RestrictedKeepsRowsCase("all_horizontal_bare_column", expr=pl.all_horizontal(pl.col("a"), pl.col("b") > 0), columns={"a"}),
        ]
    ),
)
def test_restricted_predicate_keeps_all_matching_rows(case: RestrictedKeepsRowsCase):
    """The restricted predicate, evaluated on the kept columns only, must keep every row the original keeps."""
    visitor = RestrictPredicateVisitor(case.columns)
    visitor.visit(get_parsed_expr(case.expr))
    cols = sorted(case.columns)
    kept = Counter(_NULLABLE_DATA.filter(case.expr).select(cols).rows())
    for restricted in (restrict_expr_to_columns(case.expr, case.columns), visitor.process_results()):
        if restricted is not None:
            restricted_kept = Counter(_NULLABLE_DATA.select(cols).filter(restricted).rows())
            assert not kept - restricted_kept


def test_ternary_with_null_condition_end_to_end():
    data = pl.DataFrame({"k": [None, None, 5, 0], "x": [1, 2, 300, 4]})
    pred = pl.when(pl.col("k") > 1).then(pl.col("x") > 100).otherwise(pl.col("k").fill_null(0) < 3)
    assert_matches_plain_polars(cpl.cache_memory(lambda: data.lazy(), schema=data.schema, partition_cols="k"), data.lazy(), pred)


def test_negated_ternary_end_to_end():
    frames = {(None, 1): pl.DataFrame({"x": [1, 200]}), (5, 0): pl.DataFrame({"x": [2, 300]})}
    pred = ~pl.when(pl.col("k2") == 1).then(pl.col("k") > 1).otherwise((pl.col("k") > 0) & (pl.col("x") > 100))
    lf = cpl.concat_named({key: df.lazy() for key, df in frames.items()}, [("k", pl.Int64), ("k2", pl.Int64)])
    assert_matches_plain_polars(lf, lf.collect().lazy(), pred)
