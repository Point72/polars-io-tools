import polars as pl

from polars_io_tools.io_sources.base import (
    ExprParser,
)
from polars_io_tools.io_sources.set_visitor import SetVisitor, convert_expr_to_valid_values


def test_set_visitor_visitor_direct():
    """Test using SetVisitor directly."""
    expr = pl.col("symbol").is_in(["US", "EU", "JP"])

    # Parse the expression
    parser = ExprParser()
    node = parser.parse(expr)

    # Apply the visitor
    visitor = SetVisitor("symbol")
    visitor.visit(node)
    result = visitor.process_results()

    # Verify the result
    assert result == {"US", "EU", "JP"}


def test_basic_equality():
    """Test a simple equality filter."""
    # Original: filters = [[("symbol", "=", "US")]]
    # Logic: symbol == "US"
    expr = pl.col("symbol") == "US"
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result == {"US"}


def test_multiple_equality_across_conjunctions():
    """Test DNF filter with multiple equalities in different conjunctions."""
    # Original: filters = [[("symbol", "=", "US"), ("price", ">", 100)], [("symbol", "=", "EU"), ("volume", ">", 1000)]]
    # Logic: (symbol == "US" & price > 100) | (symbol == "EU" & volume > 1000)
    expr = ((pl.col("symbol") == "US") & (pl.col("price") > 100)) | ((pl.col("symbol") == "EU") & (pl.col("volume") > 1000))
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result == {"US", "EU"}


def test_basic_in_operator():
    """Test the 'in' operator with a list of values."""
    # Original: filters = [[("symbol", "in", ["US", "EU", "JP"])]]
    # Logic: symbol.is_in(["US", "EU", "JP"])
    expr = pl.col("symbol").is_in(["US", "EU", "JP"])
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result == {"US", "EU", "JP"}


def test_multiple_in_across_conjunctions():
    """Test multiple 'in' operators across different conjunctions."""
    # Original: filters = [[("symbol", "in", ["US", "EU", "JP"]), ("price", ">", 100)], [("symbol", "in", ["EU", "UK", "CN"]), ("volume", ">", 1000)]]
    # Logic: (symbol.is_in(["US", "EU", "JP"]) & price > 100) | (symbol.is_in(["EU", "UK", "CN"]) & volume > 1000)
    expr = ((pl.col("symbol").is_in(["US", "EU", "JP"])) & (pl.col("price") > 100)) | (
        (pl.col("symbol").is_in(["EU", "UK", "CN"])) & (pl.col("volume") > 1000)
    )
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result == {"US", "EU", "JP", "UK", "CN"}


def test_combination_of_equal_and_in():
    """Test DNF with a mix of '=' and 'in' operators."""
    # Original: filters = [[("symbol", "=", "US"), ("price", ">", 100)], [("symbol", "in", ["EU", "UK", "CN"]), ("volume", ">", 1000)]]
    # Logic: (symbol == "US" & price > 100) | (symbol.is_in(["EU", "UK", "CN"]) & volume > 1000)
    expr = ((pl.col("symbol") == "US") & (pl.col("price") > 100)) | ((pl.col("symbol").is_in(["EU", "UK", "CN"])) & (pl.col("volume") > 1000))
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result == {"US", "EU", "UK", "CN"}


def test_single_conjunction_without_column():
    """Test when the column isn't present in a single conjunction."""
    # Original: filters = [[("price", ">", 100), ("volume", ">", 1000)]]
    # Logic: price > 100 & volume > 1000 (no reference to symbol)
    expr = (pl.col("price") > 100) & (pl.col("volume") > 1000)
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result is None


def test_multiple_conjunctions_one_without_column():
    """Test when the column isn't present in one of multiple conjunctions."""
    # Original: filters = [[("symbol", "=", "US"), ("price", ">", 100)], [("price", ">", 200), ("volume", ">", 1000)]]
    # Logic: (symbol == "US" & price > 100) | (price > 200 & volume > 1000)
    expr = ((pl.col("symbol") == "US") & (pl.col("price") > 100)) | ((pl.col("price") > 200) & (pl.col("volume") > 1000))
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result is None


def test_contradictions_within_conjunction():
    """Test when a conjunction contains a contradiction."""
    # Original: filters = [[("symbol", "=", "US"), ("symbol", "=", "EU")]]
    # Logic: symbol == "US" & symbol == "EU" (contradiction)
    expr = (pl.col("symbol") == "US") & (pl.col("symbol") == "EU")
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result == set()


def test_mixed_data_types():
    """Test handling of mixed data types in filters."""
    # Original: filters = [[("value", "=", 10)], [("value", "=", "abc")]]
    # Logic: value == 10 | value == "abc"
    expr = (pl.col("value") == 10) | (pl.col("value") == "abc")
    result = convert_expr_to_valid_values(expr, "value")
    assert result == {10, "abc"}


def test_complex_in_and_equal_combinations():
    """Test complex combinations of 'in' and '=' operators."""
    # Original: filters = [[("symbol", "in", ["US", "EU"]), ("symbol", "=", "US")], [("symbol", "in", ["JP", "CN"]), ("price", ">", 100)]]
    # Logic: (symbol.is_in(["US", "EU"]) & symbol == "US") | (symbol.is_in(["JP", "CN"]) & price > 100)
    expr = ((pl.col("symbol").is_in(["US", "EU"])) & (pl.col("symbol") == "US")) | ((pl.col("symbol").is_in(["JP", "CN"])) & (pl.col("price") > 100))
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result == {"US", "JP", "CN"}


def test_in_with_single_string_value():
    """Test 'in' with a single string value."""
    # Original: filters = [[("symbol", "in", "US")]]
    # Logic: symbol.is_in(["US"]) or symbol == "US"
    # Note: Single string value might need special handling depending on how is_in works in polars
    expr = pl.col("symbol") == "US"  # or pl.col("symbol").is_in(["US"])
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result == {"US"}


def test_in_with_empty_list():
    """Test 'in' with an empty list."""
    # Original: filters = [[("symbol", "in", [])]]
    # Logic: symbol.is_in([]) - empty list should always return False
    expr = pl.col("symbol").is_in([])
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result == set()


def test_single_equal_filter():
    """Test passing a single tuple as the filter."""
    # Original: filters = [[("symbol", "=", "US")]]
    # Logic: symbol == "US"
    expr = pl.col("symbol") == "US"
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result == {"US"}


def test_in_with_single_element_list():
    """Test 'in' with a list containing a single element."""
    # Original: filters = [[("symbol", "in", ["US"])]]
    # Logic: symbol.is_in(["US"])
    expr = pl.col("symbol").is_in(["US"])
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result == {"US"}


def test_multiple_complex_rules():
    """Test with multiple complex rules."""
    # Original: filters = [
    #     [("symbol", "in", ["US", "EU", "JP"]), ("price", ">", 100)],
    #     [("symbol", "=", "UK"), ("volume", ">", 1000)],
    #     [("symbol", "in", ["CN", "HK"]), ("symbol", "in", ["HK", "TW"])],
    # ]
    # Logic: (symbol.is_in(["US", "EU", "JP"]) & price > 100) |
    #        (symbol == "UK" & volume > 1000) |
    #        (symbol.is_in(["CN", "HK"]) & symbol.is_in(["HK", "TW"]))
    expr = (
        ((pl.col("symbol").is_in(["US", "EU", "JP"])) & (pl.col("price") > 100))
        | ((pl.col("symbol") == "UK") & (pl.col("volume") > 1000))
        | ((pl.col("symbol").is_in(["CN", "HK"])) & (pl.col("symbol").is_in(["HK", "TW"])))
    )
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result == {"US", "EU", "JP", "UK", "HK"}


def test_contradiction_in_one_conjunction_but_not_others():
    """Test when one conjunction has a contradiction but others don't."""
    # Original: filters = [
    #     [("symbol", "=", "US"), ("symbol", "=", "EU")],  # Contradiction
    #     [("symbol", "=", "JP")],  # Valid
    # ]
    # Logic: (symbol == "US" & symbol == "EU") | (symbol == "JP")
    expr = ((pl.col("symbol") == "US") & (pl.col("symbol") == "EU")) | (pl.col("symbol") == "JP")
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result == {"JP"}


def test_all_conjunctions_have_contradictions():
    """Test when all conjunctions have contradictions."""
    # Original: filters = [
    #     [("symbol", "=", "US"), ("symbol", "=", "EU")],  # Contradiction
    #     [("symbol", "in", []), ("price", ">", 100)],  # Empty 'in' is a contradiction
    # ]
    # Logic: (symbol == "US" & symbol == "EU") | (symbol.is_in([]) & price > 100)
    expr = ((pl.col("symbol") == "US") & (pl.col("symbol") == "EU")) | ((pl.col("symbol").is_in([])) & (pl.col("price") > 100))
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result == set()


def test_multiple_in_operators_within_single_conjunction():
    """Test multiple 'in' operators within a single conjunction."""
    # Original: filters = [[("symbol", "in", ["US", "EU", "JP"]), ("symbol", "in", ["JP", "CN", "UK"])]]
    # Logic: symbol.is_in(["US", "EU", "JP"]) & symbol.is_in(["JP", "CN", "UK"])
    expr = (pl.col("symbol").is_in(["US", "EU", "JP"])) & (pl.col("symbol").is_in(["JP", "CN", "UK"]))
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result == {"JP"}


def test_is_operator():
    """Test the 'is' operator for NULL checks."""
    # Original: filters = [[("symbol", "is", "US")]]
    # In polars, "is" would normally be is_null() or equality.
    # Here we're using equality since the test expects "US" to be in the result
    expr = pl.col("symbol") == "US"
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result == {"US"}


def test_not_equal_operator():
    """Test the '~' (not equal) operator."""
    # Original: filters = [[("symbol", "~", "US"), ("symbol", "in", ["US", "EU", "JP"])]]
    # Logic: symbol != "US" & symbol.is_in(["US", "EU", "JP"])
    expr = (pl.col("symbol") != "US") & (pl.col("symbol").is_in(["US", "EU", "JP"]))
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result == {"EU", "JP"}


def test_not_in_operator():
    """Test the '!in' (not in) operator."""
    # Original: filters = [[("symbol", "!in", ["US", "EU"]), ("symbol", "in", ["US", "EU", "JP", "CN"])]]
    # Logic: ~symbol.is_in(["US", "EU"]) & symbol.is_in(["US", "EU", "JP", "CN"])
    expr = (~pl.col("symbol").is_in(["US", "EU"])) & (pl.col("symbol").is_in(["US", "EU", "JP", "CN"]))
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result == {"JP", "CN"}


def test_different_operators_across_conjunctions():
    """Test different operators across conjunctions."""
    # Original: filters = [[("symbol", "=", "US")], [("symbol", "is", "AP")], [("symbol", "in", ["JP", "CN"])]]
    # Logic: (symbol == "US") | (symbol == "AP") | (symbol.is_in(["JP", "CN"]))
    expr = (pl.col("symbol") == "US") | (pl.col("symbol") == "AP") | (pl.col("symbol").is_in(["JP", "CN"]))
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result == {"US", "AP", "JP", "CN"}


def test_basic_boolean_cast():
    """Test that a simple boolean cast is properly traversed."""
    # Cast a simple equality expression to boolean
    expr = (pl.col("symbol") == "US").cast(pl.Boolean)
    result = convert_expr_to_valid_values(expr, "symbol")
    # The visitor should extract "US" from inside the cast
    assert result == {"US"}


def test_boolean_cast_in_and_expression():
    """Test boolean cast in an AND expression."""
    # Boolean cast on left side of AND
    expr = ((pl.col("symbol") == "US").cast(pl.Boolean)) & (pl.col("symbol") != "JP")
    result = convert_expr_to_valid_values(expr, "symbol")
    # Should extract "US" from the cast and apply the exclusion of "JP"
    assert result == {"US"}


def test_boolean_cast_in_or_expression():
    """Test boolean cast in an OR expression."""
    # Boolean cast on left side of OR
    expr = ((pl.col("symbol") == "US").cast(pl.Boolean)) | (pl.col("symbol") == "JP")
    result = convert_expr_to_valid_values(expr, "symbol")
    # Should extract values from both sides of OR
    assert result == {"US", "JP"}


def test_multiple_boolean_casts():
    """Test multiple boolean casts in an expression."""
    # Boolean casts on both sides of OR
    expr = ((pl.col("symbol") == "US").cast(pl.Boolean)) | ((pl.col("symbol") == "JP").cast(pl.Boolean))
    result = convert_expr_to_valid_values(expr, "symbol")
    # Should extract values from both casts
    assert result == {"US", "JP"}


def test_boolean_cast_with_in():
    """Test boolean cast with IS_IN function."""
    # Cast an IS_IN expression to boolean
    expr = (pl.col("symbol").is_in(["US", "JP"])).cast(pl.Boolean)
    result = convert_expr_to_valid_values(expr, "symbol")
    # Should extract values from inside the cast
    assert result == {"US", "JP"}


def test_nested_boolean_casts():
    """Test nested boolean casts."""
    # Expression with nested boolean casts
    expr = ((pl.col("symbol") == "US").cast(pl.Boolean)).cast(pl.Boolean)
    result = convert_expr_to_valid_values(expr, "symbol")
    # Should traverse through both casts
    assert result == {"US"}


def test_boolean_cast_of_complex_expression():
    """Test boolean cast of a complex expression."""
    # Cast a complex expression to boolean
    expr = (((pl.col("symbol") == "US") & (pl.col("price") > 100)) | ((pl.col("symbol") == "JP") & (pl.col("price") > 200))).cast(pl.Boolean)
    result = convert_expr_to_valid_values(expr, "symbol")
    # Should extract both symbol values from inside the cast
    assert result == {"US", "JP"}


def test_complex_nested_expressions_with_boolean_casts():
    """Test complex nested expressions with boolean casts."""
    # Multiple complex expressions with casts
    expr = (((pl.col("symbol") == "US") & (pl.col("price") > 100)).cast(pl.Boolean)) | (
        (pl.col("symbol").is_in(["JP", "EU"]) & (pl.col("volume") > 1000)).cast(pl.Boolean)
    )
    result = convert_expr_to_valid_values(expr, "symbol")
    # Should extract all valid values from both parts
    assert result == {"US", "JP", "EU"}


def test_boolean_cast_with_contradictions():
    """Test boolean cast with contradictions in the underlying expression."""
    # Cast an expression with contradictory requirements
    expr = ((pl.col("symbol") == "US") & (pl.col("symbol") == "JP")).cast(pl.Boolean)
    result = convert_expr_to_valid_values(expr, "symbol")
    # Contradiction should result in empty set
    assert result == set()


def test_boolean_cast_with_mixed_data_types():
    """Test boolean cast with mixed data types in the underlying expression."""
    # Cast an expression with different data types
    expr = ((pl.col("value") == 10) | (pl.col("value") == "abc")).cast(pl.Boolean)
    result = convert_expr_to_valid_values(expr, "value")
    # Should extract both values regardless of type
    assert result == {10, "abc"}


def test_non_boolean_cast():
    """Test that non-boolean casts are not specially handled."""
    # Cast to a non-boolean type
    expr = (pl.col("symbol") == "US").cast(pl.Utf8)
    # The visitor only handles boolean casts specially
    # For non-boolean casts, we don't extract valid values
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result is None  # Non-boolean casts should not extract valid values


def test_boolean_cast_with_not():
    """Test boolean cast with NOT function."""
    # Cast a negated expression to boolean
    expr = (~(pl.col("symbol") == "US")).cast(pl.Boolean)
    result = convert_expr_to_valid_values(expr, "symbol")
    # The visitor can handle negation but can't determine a concrete set of values
    # without knowing the entire universe of possible values
    assert result is None


def test_set_visitor_consistent_set_creation():
    """Test that SetVisitor creates new sets consistently instead of modifying existing ones."""
    from polars_io_tools.io_sources.base import get_parsed_expr
    from polars_io_tools.io_sources.set_visitor import SetVisitor

    # Test multiple equality operations create new sets consistently
    visitor = SetVisitor("symbol")

    # First equality - creates initial set
    expr1 = pl.col("symbol") == "US"
    node1 = get_parsed_expr(expr1)
    assert node1 is not None
    visitor.visit(node1)
    assert visitor.inclusions is not None
    first_set_id = id(visitor.inclusions)
    first_set_contents = visitor.inclusions.copy()

    # Second equality - should create new set, not modify existing
    expr2 = pl.col("symbol") == "EU"
    node2 = get_parsed_expr(expr2)
    assert node2 is not None
    visitor.visit(node2)
    assert visitor.inclusions is not None
    second_set_id = id(visitor.inclusions)

    # Verify consistent behavior: new sets are created, not modified in place
    assert first_set_id != second_set_id, "SetVisitor should create new sets consistently"
    assert visitor.inclusions == {"US", "EU"}, "Both values should be included in final set"
    assert first_set_contents == {"US"}, "Original set should remain unchanged"


def test_set_visitor_null_handling_consistent():
    """Test that NULL value handling also follows consistent set creation pattern."""
    from polars_io_tools.io_sources.base import get_parsed_expr
    from polars_io_tools.io_sources.set_visitor import SetVisitor

    visitor = SetVisitor("nullable_col")

    # First IS_NULL - creates initial set
    expr1 = pl.col("nullable_col").is_null()
    node1 = get_parsed_expr(expr1)
    assert node1 is not None
    visitor.visit(node1)
    first_set_id = id(visitor.inclusions) if visitor.inclusions else None

    # Second equality - should create new set consistently
    expr2 = pl.col("nullable_col") == "value"
    node2 = get_parsed_expr(expr2)
    assert node2 is not None
    visitor.visit(node2)
    second_set_id = id(visitor.inclusions) if visitor.inclusions else None

    # Verify consistent set creation behavior
    if first_set_id and second_set_id:
        assert first_set_id != second_set_id, "Consistent set creation for NULL handling"
    assert visitor.inclusions == {None, "value"}, "Both NULL and value should be included"


def test_logical_and_operator_handling():
    """
    Regression test for LOGICAL_AND vs AND operator bug.

    This test ensures that both LOGICAL_AND and AND operators are handled correctly
    when extracting valid values from expressions. The bug occurred because
    is_bitwise() only returned True for AND/OR/XOR but not LOGICAL_AND/LOGICAL_OR.

    Before the fix:
    - LOGICAL_AND expressions would return None (unhandled operator)
    - AND expressions would return correct symbol sets

    After the fix:
    - Both should return the same correct results
    """
    from datetime import date

    from polars_io_tools.io_sources.base import BinaryExprNode, ExprParser
    from polars_io_tools.io_sources.enum import OperatorType

    # Create expressions that should produce LOGICAL_AND
    # Based on the debug output, the issue occurs with this specific pattern
    date_expr = pl.col("date_partition") == date(2009, 10, 2)
    symbol_expr = pl.col("symbol") == "AAPL US"

    # Create the combined filter - this might get LOGICAL_AND operator
    combined_filter = date_expr & symbol_expr

    # Parse the expression to check what operator we actually get
    parser = ExprParser()
    parsed_node = parser.parse(combined_filter)

    # Manually create a BinaryExprNode with LOGICAL_AND to force the bug
    # This simulates what happens before Polars optimization (collect_schema)
    if isinstance(parsed_node, BinaryExprNode):
        # Create a copy but force it to be LOGICAL_AND
        from polars_io_tools.io_sources.base import BinaryExprNode

        logical_and_node = BinaryExprNode(
            expr=parsed_node.expr,
            can_extract_literal=False,
            left=parsed_node.left,
            right=parsed_node.right,
            op=OperatorType.LOGICAL_AND,  # Force this to be LOGICAL_AND
        )

        # Test symbol extraction with forced LOGICAL_AND
        result = convert_expr_to_valid_values(logical_and_node, "symbol")

        # This should extract the symbol value even with LOGICAL_AND
        # If the fix is commented out, this will return None
        assert result == {"AAPL US"}, f"LOGICAL_AND should extract symbol, got {result}"
        assert result is not None, "LOGICAL_AND should be handled, not return None"


# Test cases for alias expressions
def test_alias_simple_equality():
    """Test that aliases in simple equality expressions are handled properly."""
    expr = pl.col("symbol").alias("symbol_alias") == "US"
    result = convert_expr_to_valid_values(expr, "symbol")
    # This should extract the valid values from the aliased column expression
    assert result == {"US"}


def test_alias_in_operator():
    """Test aliases with is_in operator."""
    expr = pl.col("symbol").alias("symbol_alias").is_in(["US", "EU", "JP"])
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result == {"US", "EU", "JP"}


def test_alias_logical_operations():
    """Test aliases in logical operations."""
    expr = (pl.col("symbol").alias("symbol_alias") == "US") & (pl.col("price").alias("price_alias") > 100)
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result == {"US"}


def test_alias_nested_expression():
    """Test aliases on complex nested expressions."""
    expr = ((pl.col("symbol") == "US") | (pl.col("symbol") == "EU")).alias("symbol_filter")
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result == {"US", "EU"}


def test_alias_with_exclusions():
    """Test aliases with exclusion operations."""
    expr = (pl.col("symbol").alias("symbol_alias") != "US") & (pl.col("symbol").is_in(["US", "EU", "JP"]))
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result == {"EU", "JP"}


def test_alias_complex_mixed_operations():
    """Test complex expressions with mixed alias operations."""
    expr = ((pl.col("symbol").alias("s1") == "US") | (pl.col("symbol").alias("s2").is_in(["EU", "JP", "UK"]))) & (
        pl.col("symbol").alias("s3") != "UK"
    )
    result = convert_expr_to_valid_values(expr, "symbol")
    assert result == {"US", "EU", "JP"}
