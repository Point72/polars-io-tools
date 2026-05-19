from typing import Optional

import polars as pl
from packaging import version

from polars_io_tools.io_sources.base import BinaryExprNode, ColumnNode, ExprParser, get_parsed_expr
from polars_io_tools.io_sources.enum import OperatorType
from polars_io_tools.io_sources.restrict_visitor import RestrictPredicateVisitor, restrict_expr_to_columns


def assert_expr_equal(a: Optional[pl.Expr], b: Optional[pl.Expr]):
    """Assert that two polars expressions are equal"""
    if a is None:
        assert b is None
    elif b is None:
        assert a is None
    else:
        assert a.meta.eq(b)


def test_restrict_predicate():
    assert_expr_equal(restrict_expr_to_columns(pl.col("p"), ["p"]), pl.col("p"))
    assert_expr_equal(restrict_expr_to_columns(pl.col("p"), ["p", "q"]), pl.col("p"))
    assert_expr_equal(restrict_expr_to_columns(pl.col("p").is_in([1, 2]), ["p"]), pl.col("p").is_in([1, 2]))
    assert_expr_equal(restrict_expr_to_columns(pl.col("p") & pl.col("x"), ["p"]), pl.col("p"))
    assert_expr_equal(
        restrict_expr_to_columns(pl.col("p").is_in([1, 2]) & pl.col("x") & pl.col("p").is_in([4, 5]), ["p"]),
        pl.col("p").is_in([1, 2]) & pl.col("p").is_in([4, 5]),
    )
    assert_expr_equal(restrict_expr_to_columns(pl.col("p") | pl.col("x"), ["p"]), None)

    assert_expr_equal(restrict_expr_to_columns(pl.col("p") & pl.col("x") & pl.col("q"), ["p", "q"]), pl.col("p") & pl.col("q"))
    assert_expr_equal(restrict_expr_to_columns(pl.col("p") | pl.col("x") | pl.col("q"), ["p", "q"]), None)

    assert_expr_equal(restrict_expr_to_columns((pl.col("p") * pl.col("q")) < 4, ["p", "q"]), (pl.col("p") * pl.col("q")) < 4)
    assert_expr_equal(restrict_expr_to_columns((pl.col("p") + pl.col("x")) < 4, ["p", "q"]), None)

    assert_expr_equal(restrict_expr_to_columns(pl.col("p").alias("q"), ["p"]), pl.col("p").alias("q"))
    assert_expr_equal(restrict_expr_to_columns(pl.arctan2(pl.col("q"), pl.col("p")) > 12, ["p"]), None)


def test_restrict_predicate_visitor_direct():
    """Test using RestrictPredicateVisitor directly."""
    # Create an expression: (symbol == "US") & (price > 100)
    expr = (pl.col("symbol") == "US") & (pl.col("price") > 100)

    # Parse the expression
    parser = ExprParser()
    node = parser.parse(expr)

    # Apply the visitor, restricting to just "symbol"
    visitor = RestrictPredicateVisitor({"symbol"})
    visitor.visit(node)
    result = visitor.process_results()

    # Verify the result is equivalent to (symbol == "US")
    expected = pl.col("symbol") == "US"
    assert result.meta.eq(expected)


def test_basic_comparison_restriction():
    """Test restricting a simple comparison."""
    # Original: symbol == "US"
    expr = pl.col("symbol") == "US"

    # Should return the original expression since it already only uses "symbol"
    result = restrict_expr_to_columns(expr, {"symbol"})
    assert result.meta.eq(expr)


def test_simple_and_restriction():
    """Test restricting an AND expression by dropping irrelevant columns."""
    # Original: (symbol == "US") & (price > 100)
    expr = (pl.col("symbol") == "US") & (pl.col("price") > 100)

    # Restrict to just "symbol" - should drop the price condition
    result = restrict_expr_to_columns(expr, {"symbol"})
    expected = pl.col("symbol") == "US"
    assert result.meta.eq(expected)


def test_and_multiple_restriction():
    """Test restricting an AND expression with multiple relevant columns."""
    # Original: (symbol == "US") & (price > 100) & (exchange == "NYSE")
    expr = (pl.col("symbol") == "US") & (pl.col("price") > 100) & (pl.col("exchange") == "NYSE")

    # Restrict to "symbol" and "exchange"
    result = restrict_expr_to_columns(expr, {"symbol", "exchange"})

    # Expected: (symbol == "US") & (exchange == "NYSE") - price condition is dropped
    expected = (pl.col("symbol") == "US") & (pl.col("exchange") == "NYSE")
    assert result.meta.eq(expected)


def test_or_no_restriction():
    """Test that OR expressions can't be restricted if any part uses excluded columns."""
    # Original: (symbol == "US") | (price > 100)
    expr = (pl.col("symbol") == "US") | (pl.col("price") > 100)

    # Try to restrict to just "symbol" - should be impossible
    result = restrict_expr_to_columns(expr, {"symbol"})

    # Expected: None (can't restrict OR expressions if any part uses excluded columns)
    assert result is None


def test_or_with_restriction():
    """Test that OR expressions can be restricted if all parts use included columns."""
    # Original: (symbol == "US") | (symbol == "EU")
    expr = (pl.col("symbol") == "US") | (pl.col("symbol") == "EU")

    # Restrict to just "symbol" - should be possible since both sides only use "symbol"
    result = restrict_expr_to_columns(expr, {"symbol"})

    # Expected: Same expression
    assert result.meta.eq(expr)


def test_complex_nested_expression():
    """Test a complex nested expression with AND and OR operators."""
    # Original: ((symbol == "US") & (price > 100)) | ((symbol == "EU") & (price > 200))
    expr = ((pl.col("symbol") == "US") & (pl.col("price") > 100)) | ((pl.col("symbol") == "EU") & (pl.col("price") > 200))

    # Can't restrict to just "symbol" because of OR with mixed columns
    result = restrict_expr_to_columns(expr, {"symbol"})
    # Expected: (symbol == "US") | (symbol == "EU")
    expected = (pl.col("symbol") == "US") | (pl.col("symbol") == "EU")
    assert result.meta.eq(expected)

    node = get_parsed_expr(expr)
    result_node = restrict_expr_to_columns(node, {"symbol"})
    assert result_node.meta.eq(expected)

    # But we can restrict to both "symbol" and "price" - should return original expr
    result = restrict_expr_to_columns(expr, {"symbol", "price"})
    assert result.meta.eq(expr)
    node = get_parsed_expr(expr)
    result_node = restrict_expr_to_columns(node, {"symbol", "price"})
    assert result_node.meta.eq(expr)


def test_not_node_restriction():
    """Test restriction of NOT expressions."""
    # NOTE: The restriction here is not logically sound.
    # ~(A & B) becomes (~A | ~B) in DNF, but we don't apply DeMorgan's laws here.
    # This test demonstrates that behavior. The reason we can skip applying
    # DeMorgan's laws is because the polars optimization process will handle this
    # for us.

    # Original: ~(symbol == "US") & (price > 100)
    expr = ~(pl.col("symbol") == "US") & (pl.col("price") > 100)

    # Restrict to just "symbol"
    result = restrict_expr_to_columns(expr, {"symbol"})

    # Expected: ~(symbol == "US")
    expected = ~(pl.col("symbol") == "US")
    assert result.meta.eq(expected)


def test_not_node_restriction_pushed_down(tester):
    """Test restriction of NOT expressions."""
    # We test that the pushed down predicate is optimized
    # already, so we do not need to apply DeMorgan's laws

    expr = ~((pl.col("id") == 10) | (pl.col("int_val") > 90))

    def assert_func(original_expr, pushed_expr):
        # Check that the original expression is equivalent to the pushed expression
        # expected_pushed = (pl.col("id") != 10) & (pl.col("int_val") <= 90)
        restricted_expr = restrict_expr_to_columns(pushed_expr, {"id"})

        # We check the expression this way since the .meta.eq
        # doesnt work properly between pushed down predicates and
        # not pushed down predicates
        restricted_node = get_parsed_expr(restricted_expr)
        assert isinstance(restricted_node, BinaryExprNode)
        assert restricted_node.op == OperatorType.NOT_EQ
        assert isinstance(restricted_node.left, ColumnNode)
        assert restricted_node.left.name == "id"
        assert restricted_node.right.can_extract_literal
        assert restricted_node.right.value == 10

    tester.assert_predicate_pushed_down(expr, assert_func)

    # we cannot extract the restriction from the pushed down predicate
    # since DeMorgan's laws turns this into
    # expected_pushed = (pl.col("id") != 10) | (pl.col("int_val") <= 90)
    expr = ~((pl.col("id") == 10) & (pl.col("int_val") > 90))

    def assert_func(original_expr, pushed_expr):
        # Check that the original expression is equivalent to the pushed expression
        # expected_pushed = (pl.col("id") != 10) | (pl.col("int_val") <= 90)
        restricted_expr = restrict_expr_to_columns(pushed_expr, {"id"})

        assert restricted_expr is None

    tester.assert_predicate_pushed_down(expr, assert_func)


def test_is_null_restriction():
    """Test restriction of IS NULL expressions."""
    # Original: symbol.is_null() & (price > 100)
    expr = pl.col("symbol").is_null() & (pl.col("price") > 100)

    # Restrict to just "symbol"
    result = restrict_expr_to_columns(expr, {"symbol"})

    # Expected: symbol.is_null()
    expected = pl.col("symbol").is_null()
    assert result.meta.eq(expected)


def test_is_in_restriction():
    """Test restriction of IS IN expressions."""
    # Original: symbol.is_in(["US", "EU"]) & (price > 100)
    expr = pl.col("symbol").is_in(["US", "EU"]) & (pl.col("price") > 100)

    # Restrict to just "symbol"
    result = restrict_expr_to_columns(expr, {"symbol"})

    # Expected: symbol.is_in(["US", "EU"])
    expected = pl.col("symbol").is_in(["US", "EU"])
    assert result.meta.eq(expected)


def test_string_function_restriction():
    """Test restriction of string function expressions."""
    # Original: symbol.str.contains("US") & (price > 100)
    expr = pl.col("symbol").str.contains("US") & (pl.col("price") > 100)

    # Restrict to just "symbol"
    result = restrict_expr_to_columns(expr, {"symbol"})

    # Expected: symbol.str.contains("US")
    expected = pl.col("symbol").str.contains("US")
    assert result.meta.eq(expected)


def test_is_between_restriction():
    """Test restriction of IS BETWEEN expressions."""
    # Original: symbol.is_between(5, 10) & (price > 100)
    expr = pl.col("symbol").is_between(5, 10) & (pl.col("price") > 100)

    # Restrict to just "symbol"
    result = restrict_expr_to_columns(expr, {"symbol"})

    # Expected: symbol.is_between(5, 10)
    expected = pl.col("symbol").is_between(5, 10)
    assert result.meta.eq(expected)


def test_nested_and_or_restriction():
    """Test nested AND/OR expressions with complex structure."""
    # Original: (symbol == "US") & ((price > 100) | (volume > 1000))
    expr = (pl.col("symbol") == "US") & ((pl.col("price") > 100) | (pl.col("volume") > 1000))

    # Restrict to "symbol"
    result = restrict_expr_to_columns(expr, {"symbol"})

    # Expected: symbol == "US" (the OR part is dropped as it doesn't involve "symbol")
    expected = pl.col("symbol") == "US"
    assert result.meta.eq(expected)


def test_no_valid_restriction():
    """Test case where the entire expression must be dropped."""
    # Original: (price > 100) | (volume > 1000)
    expr = (pl.col("price") > 100) | (pl.col("volume") > 1000)

    # Try to restrict to just "symbol" - no part involves "symbol"
    result = restrict_expr_to_columns(expr, {"symbol"})

    # Expected: None (nothing can be preserved)
    assert result is None


def test_ternary_truthy_relevant():
    expr = pl.when(pl.col("id") == 10).then(pl.col("int_val") > 90).otherwise(pl.col("int_val2") > 100)
    result = restrict_expr_to_columns(expr, {"int_val"})
    assert result is None  # We cannot restrict


def test_ternary_truthy_and_falsy_relevant():
    expr = pl.when(pl.col("id") == 10).then(pl.col("int_val") > 90).otherwise(pl.col("int_val") > 100)
    expected = (pl.col("int_val") > 90) | (pl.col("int_val") > 100)
    result = restrict_expr_to_columns(expr, {"int_val"})
    assert result.meta.eq(expected)


def test_ternary_all_relevant():
    expr = pl.when(pl.col("int_val").mod(2) == 0).then(pl.col("int_val") > 90).otherwise(pl.col("int_val") > 100)
    result = restrict_expr_to_columns(expr, {"int_val"})
    assert result.meta.eq(expr)


def test_ternary_complex_predicate():
    # Ternary with complex predicate involving multiple columns
    expr = pl.when((pl.col("symbol") == "AAPL") & (pl.col("price") > 150)).then(pl.col("volume") * 2).otherwise(pl.col("volume"))

    # Restrict to just "symbol" and "volume"
    result = restrict_expr_to_columns(expr, {"symbol", "volume"})

    # Expected: (symbol == "AAPL" & volume * 2) | volume
    expected = ((pl.col("symbol") == "AAPL").and_(pl.col("volume") * 2)) | pl.col("volume")

    assert result.meta.eq(expected)


def test_ternary_nested():
    # Nested ternary expressions
    expr = (
        pl.when(pl.col("region") == "US")
        .then(pl.when(pl.col("sector") == "Tech").then((pl.col("price") * 1.1) > 1).otherwise(pl.col("price") < 12))
        .otherwise((pl.col("price") * 0.9) > 1)
    )

    # Restrict to just "region" and "price"
    result = restrict_expr_to_columns(expr, {"region", "price"})

    # Expected result when restricting to "region" and "price"
    #
    expected_true = (pl.col("region") == "US") & ((pl.col("price") * 1.1 > 1) | (pl.col("price") < 12))
    expected_false = (pl.col("region") == "US").not_() & ((pl.col("price") * 0.9) > 1)

    expected = expected_true | expected_false

    assert result.meta.eq(expected)


def test_ternary_with_irrelevant_predicate():
    # Ternary where predicate is completely irrelevant
    expr = pl.when(pl.col("date") > "2023-01-01").then(pl.col("price") * 1.2).otherwise(pl.col("volume") * 2)

    # Restrict to just "price" and "volume"
    result = restrict_expr_to_columns(expr, {"price", "volume"})

    # Expected: (price * 1.2) | (volume * 2)
    # Since predicate is not relevant, both branches remain
    expected = (pl.col("price") * 1.2) | (pl.col("volume") * 2)

    assert result.meta.eq(expected)


def test_ternary_only_relevant_truthy():
    # Only the truthy branch contains relevant columns
    expr = pl.when(pl.col("date") > "2023-01-01").then(pl.col("price") * 1.2).otherwise(pl.col("volume") * 2)

    # Restrict to just "price"
    result = restrict_expr_to_columns(expr, {"price"})
    # since the predicate and falsy are not relevant,
    # we cannot just restrict to truthy, so we return None
    assert result is None


def test_ternary_only_relevant_falsy():
    # Only the falsy branch contains relevant columns
    expr = pl.when(pl.col("date") > "2023-01-01").then(pl.col("price") * 1.2).otherwise(pl.col("symbol").str.to_uppercase())

    # Restrict to just "symbol"
    result = restrict_expr_to_columns(expr, {"symbol"})

    # Expected: symbol.str.to_uppercase()
    # Since the predicate and truthy are not relevant,
    # we cannot just restrict to Falsy
    assert result is None


def test_ternary_combined_with_binary_op():
    # Ternary combined with binary operators
    expr = (pl.when(pl.col("region") == "US").then(pl.col("price") > 100).otherwise(pl.col("volume") > 1000)) & (pl.col("date") > "2023-01-01")

    # Restrict to just "region" and "price"
    result = restrict_expr_to_columns(expr, {"region", "price"})

    # Ok so, ignore the "date" one, it's just to test that nothing breaks

    # But we restrict to region and price
    # our filter says, we keep the region == "US" and price > 100
    # and we keep region != "US" and volume > 1000
    # Since we ignore volume, we can drop that restriction

    expected = (pl.col("region") == "US") & (pl.col("price") > 100)
    expected = expected | (pl.col("region") == "US").not_()
    assert result.meta.eq(expected)


def test_ternary_with_complex_expressions():
    # Ternary with complex expressions in all parts
    expr = (
        pl.when((pl.col("region") == "US") | (pl.col("region") == "EU"))
        .then(((pl.col("price") * pl.col("volume")).alias("value") > 100))
        .otherwise((pl.col("price") - pl.col("discount")) > 5)
    )

    # Restrict to just "region" and "price"
    result = restrict_expr_to_columns(expr, {"region", "price"})
    # Now, since the true and false paths are both unrepresentable
    # We would just have the predicate and it's negation, which is trivially true.
    # Thus, we can't make a less restrictive predicate
    assert result is None


def test_ternary_all_parts_irrelevant():
    # No part of the ternary is relevant
    expr = pl.when(pl.col("date") > "2023-01-01").then(pl.col("price") * 1.2).otherwise(pl.col("volume") * 2)

    # Restrict to just "symbol"
    result = restrict_expr_to_columns(expr, {"symbol"})

    # Expected: None (no part is relevant)
    assert result is None


def test_ternary_predicate_str_function():
    # No part of the ternary is relevant
    expr = pl.when(pl.col("symbol").str.contains("HI").alias("hey")).then(pl.col("price") > 3).otherwise(pl.col("volume") != 13)

    result = restrict_expr_to_columns(expr, {"symbol", "price"})
    lower_bound = version.parse("1.30.0")
    upper_bound = version.parse("1.32.0")
    cur_version = version.parse(pl.__version__)
    if lower_bound < cur_version < upper_bound:
        # In versions between 1.30.0 and 1.32.0, the alias is not preserved in the expression
        # This is a bug
        expected = (pl.col("symbol").str.contains("HI").alias("hey") & (pl.col("price") > 3)) | (
            pl.col("symbol").str.contains("HI").alias("hey").not_()
        )
        assert result.meta.eq(expected)
    else:
        expected = ((pl.col("symbol").str.contains("HI").alias("hey")) & (pl.col("price") > 3)) | (
            pl.col("symbol").str.contains("HI").alias("hey").not_()
        )
        assert result.meta.eq(expected)


def test_ternary_complex_comparison():
    """Test ternary with complex comparisons in predicate."""
    expr = (
        pl.when((pl.col("open") <= pl.col("close")) & (pl.col("volume") > 1000))
        .then(pl.col("symbol").str.contains("UP"))
        .otherwise(pl.col("open") > 100)
    )

    # Restrict to "open" and "close"
    result = restrict_expr_to_columns(expr, {"open", "close"})
    expected = (pl.col("open") <= pl.col("close")) | (pl.col("open") > 100)
    assert result.meta.eq(expected)


def test_not_pred_valid_pred_invalid():
    expr = pl.when((pl.col("A").mod(2) == 0) | pl.col("B")).then(pl.col("A") > 15).otherwise(pl.col("C"))

    result = restrict_expr_to_columns(expr, {"A"})

    # We need to create an expression that only uses column "A" but preserves the original logic as much as possible.
    # In the original expression:
    # 1. When (A is even OR B is true), we return (A > 15)
    # 2. Otherwise, we return the value in column C
    #
    # Since we can only use column "A":
    # - For rows where A is even, the condition is definitely true (regardless of B's value),
    #   so the result is (A > 15)
    # - For rows where A is odd, the condition depends on column B which we can't access,
    #   and the result depends on column C, which we also can't access.
    #   Since we can't determine the result using only A in this case, we must include all rows where A is odd
    #
    # Therefore, our restricted expression becomes:
    # (A > 15) OR (A is odd)
    # This ensures we include all rows that definitely match the original expression,
    # even if it means including some rows that might not have matched in the complete expression.

    expected = (pl.col("A") > 15) | (pl.col("A").mod(2) == 0).not_()
    assert result.meta.eq(expected)


def test_when_then_otherwise_with_definitive_exclusions():
    expr = pl.when((pl.col("B") > 5)).then((pl.col("A") < 20) & (pl.col("C") > 0)).otherwise(pl.col("A").mod(3) > 0)

    result = restrict_expr_to_columns(expr, {"A"})

    expected = (pl.col("A") < 20) | (pl.col("A").mod(3) > 0)
    assert result.meta.eq(expected)


def test_simple_cast():
    """Test simple cast expressions with column restriction."""
    # Cast to boolean
    expr = pl.col("flag").cast(pl.Boolean)
    result = restrict_expr_to_columns(expr, {"flag"})
    assert result.meta.eq(expr)

    # Cast to non-boolean
    expr = pl.col("value").cast(pl.Int64)
    result = restrict_expr_to_columns(expr, {"value"})
    assert result.meta.eq(expr)

    # Cast involving columns we're restricting out
    expr = (pl.col("value") + pl.col("other")).cast(pl.Int64)
    result = restrict_expr_to_columns(expr, {"value"})
    # Cannot represent this expression with just "value"
    assert result is None


def test_cast_with_binary_expr():
    """Test cast with binary expressions."""
    # Cast AND expression to boolean
    expr = (pl.col("flag") & pl.col("active")).cast(pl.Boolean)

    # When restricting to just one column of a binary op that's cast to boolean
    result = restrict_expr_to_columns(expr, {"flag"})
    # Cannot represent the AND with just flag
    assert result.meta.eq(pl.col("flag"))

    # Cast inside AND - this tests the specific functionality you added
    expr = pl.col("flag").cast(pl.Boolean) & pl.col("active")
    result = restrict_expr_to_columns(expr, {"flag"})
    expected = pl.col("flag").cast(pl.Boolean)
    assert result.meta.eq(expected)

    # Cast inside OR with relevant columns
    expr = pl.col("flag").cast(pl.Boolean) | pl.col("active").cast(pl.Boolean)
    result = restrict_expr_to_columns(expr, {"flag", "active"})
    assert result.meta.eq(expr)


def test_multiple_casts():
    """Test multiple casts in an expression."""
    # Nested casts (this is a key test for your implementation)
    expr = pl.col("value").cast(pl.Int64).cast(pl.Boolean)
    result = restrict_expr_to_columns(expr, {"value"})
    assert result.meta.eq(expr)

    # More complex nested cast with operation
    expr = (pl.col("value") > 0).cast(pl.Int64).cast(pl.Boolean)
    result = restrict_expr_to_columns(expr, {"value"})
    assert result.meta.eq(expr)

    # Complex nested cast with AND
    expr = ((pl.col("value") > 0).cast(pl.Boolean) & (pl.col("other") > 10).cast(pl.Boolean)).cast(pl.Boolean)
    result = restrict_expr_to_columns(expr, {"value"})
    # We strip the outer level of boolean in the restrict visitor.
    expected = (pl.col("value") > 0).cast(pl.Boolean)
    assert result.meta.eq(expected)


def test_cast_with_not():
    """Test casts with NOT operations."""
    # NOT after cast
    expr = ~(pl.col("flag").cast(pl.Boolean))
    result = restrict_expr_to_columns(expr, {"flag"})
    assert result.meta.eq(expr)

    # Cast after NOT
    expr = (~pl.col("flag")).cast(pl.Boolean)
    result = restrict_expr_to_columns(expr, {"flag"})
    assert result.meta.eq(expr)

    # Combining NOT, cast and AND
    expr = ~(pl.col("flag").cast(pl.Boolean)) & pl.col("value")
    result = restrict_expr_to_columns(expr, {"flag"})
    expected = ~(pl.col("flag").cast(pl.Boolean))
    assert result.meta.eq(expected)


def test_cast_in_ternary():
    """Test casts in ternary expressions."""
    # Cast in predicate of ternary
    expr = pl.when(pl.col("flag").cast(pl.Boolean)).then(pl.col("value") > 0).otherwise(pl.col("price") > 100)

    # Restrict to "flag" and "value"
    result = restrict_expr_to_columns(expr, {"flag", "value"})
    # The expected result combines the predicate and truthy branch
    expected = (pl.col("flag").cast(pl.Boolean) & (pl.col("value") > 0)) | (pl.col("flag").cast(pl.Boolean).not_())
    assert result.meta.eq(expected)

    # Cast in both branches of ternary
    expr = pl.when(pl.col("flag")).then((pl.col("value") > 0).cast(pl.Boolean)).otherwise((pl.col("price") > 100).cast(pl.Boolean))

    # Restrict to "flag" and "value"
    result = restrict_expr_to_columns(expr, {"flag", "value"})
    expected = (pl.col("flag") & (pl.col("value") > 0).cast(pl.Boolean)) | (pl.col("flag").not_())
    assert result.meta.eq(expected)


def test_cast_boolean_complex_logic():
    """Test complex boolean logic with casts."""
    # Test for DeMorgan's law with casts
    expr = ~((pl.col("a") > 0).cast(pl.Boolean) | (pl.col("b") < 10).cast(pl.Boolean))

    # Restrict to just "a"
    result = restrict_expr_to_columns(expr, {"a"})
    expected = ~(pl.col("a") > 0).cast(pl.Boolean)
    assert result.meta.eq(expected)

    # This tests the cast node handling embedded in complex boolean logic
    expr = ((pl.col("a") > 0).cast(pl.Boolean) & (pl.col("b") < 10).cast(pl.Boolean)) | (
        (pl.col("c") == 5).cast(pl.Boolean) & (pl.col("a") != 0).cast(pl.Boolean)
    )

    result = restrict_expr_to_columns(expr, {"a"})
    expected = (pl.col("a") > 0).cast(pl.Boolean) | (pl.col("a") != 0).cast(pl.Boolean)
    assert result.meta.eq(expected)


def test_cast_between_operations():
    """Test casting between logical operations."""
    # This tests that we handle casts correctly when they're between operations
    expr = ((pl.col("a") > 0).cast(pl.Boolean) & pl.col("b").is_null()).cast(pl.Boolean) | (pl.col("a") < 0).cast(pl.Boolean)

    result = restrict_expr_to_columns(expr, {"a"})
    expected = (pl.col("a") > 0).cast(pl.Boolean) | (pl.col("a") < 0).cast(pl.Boolean)
    assert result.meta.eq(expected)


def test_cast_non_boolean_handling():
    """Test that non-boolean casts are properly rejected in complex expressions."""
    # Cast to non-boolean inside a logical operation
    expr = (pl.col("a").cast(pl.Int32) > 0) & pl.col("b")

    result = restrict_expr_to_columns(expr, {"a"})
    expected = pl.col("a").cast(pl.Int32) > 0
    assert result.meta.eq(expected)

    # Cast to non-boolean that cannot be represented
    expr = (pl.col("a") + pl.col("b")).cast(pl.Int32) > 0

    result = restrict_expr_to_columns(expr, {"a"})
    assert result is None


def test_nested_alias_handling():
    """Test that restrict_visitor handles nested aliases correctly."""

    # Test simple nested alias
    expr1 = pl.col("price").alias("p1").alias("p2") > 100
    result1 = restrict_expr_to_columns(expr1, {"price"})
    assert_expr_equal(result1, expr1)  # Should preserve the expression

    # Test nested alias with cast
    expr2 = pl.col("price").cast(pl.Float64).alias("p1").alias("p2") > 100
    result2 = restrict_expr_to_columns(expr2, {"price"})
    assert_expr_equal(result2, expr2)  # Should preserve the expression

    # Test complex expression with nested aliases
    expr3 = (pl.col("price").alias("p1").alias("p2") > 100) & (pl.col("volume").alias("v1") > 1000)
    result3 = restrict_expr_to_columns(expr3, {"price", "volume"})
    assert_expr_equal(result3, expr3)  # Should preserve the entire expression

    # Test restriction to subset of columns
    result4 = restrict_expr_to_columns(expr3, {"price"})
    expected4 = pl.col("price").alias("p1").alias("p2") > 100
    assert_expr_equal(result4, expected4)  # Should only keep price part


def test_all_horizontal_restriction():
    """Test restrict_visitor handles all_horizontal expressions."""
    pred1 = pl.col("date") > "2024-01-01"
    pred2 = pl.col("val") > 5
    combined = pl.all_horizontal(pred1, pred2)

    result = restrict_expr_to_columns(combined, {"date"})
    expected = pl.col("date") > "2024-01-01"
    assert_expr_equal(result, expected)

    result_both = restrict_expr_to_columns(combined, {"date", "val"})
    assert_expr_equal(result_both, combined)

    result_none = restrict_expr_to_columns(combined, {"other"})
    assert result_none is None


def test_all_horizontal_all_same_column():
    """Test all_horizontal where all inputs reference same column."""
    combined = pl.all_horizontal(pl.col("x") > 0, pl.col("x") < 100)

    result = restrict_expr_to_columns(combined, {"x"})
    assert_expr_equal(result, combined)


def test_any_horizontal_restriction():
    """Test restrict_visitor handles any_horizontal expressions."""
    pred1 = pl.col("date") > "2024-01-01"
    pred2 = pl.col("val") > 5
    combined = pl.any_horizontal(pred1, pred2)

    result_partial = restrict_expr_to_columns(combined, {"date"})
    assert result_partial is None

    result_both = restrict_expr_to_columns(combined, {"date", "val"})
    assert_expr_equal(result_both, combined)


def test_any_horizontal_all_same_column():
    """Test any_horizontal where all inputs reference same column."""
    combined = pl.any_horizontal(pl.col("x") == 1, pl.col("x") == 2)

    result = restrict_expr_to_columns(combined, {"x"})
    assert_expr_equal(result, combined)


def test_all_horizontal_nested():
    """Test nested all_horizontal expressions."""
    inner = pl.all_horizontal(pl.col("a") > 0, pl.col("b") > 0)
    outer = pl.all_horizontal(inner, pl.col("c") > 0)

    result = restrict_expr_to_columns(outer, {"a", "b"})
    expected = pl.all_horizontal(pl.col("a") > 0, pl.col("b") > 0)
    assert_expr_equal(result, expected)
