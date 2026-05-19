from datetime import date, datetime

import polars as pl
import pytest

from polars_io_tools.io_sources.base import (
    ExprParser,
)
from polars_io_tools.io_sources.dnf_visitor import (
    DNFClause,
    DNFTuple,
    DNFVisitor,
    _is_contradiction,
    combine_and_dnf,
    convert_expr_to_dnf,
    is_contradiction,
    negate_dnf,
)
from polars_io_tools.tests.io_sources.conftest import assert_dnf_equal


# Fixtures for common expressions
@pytest.fixture
def basic_comparison_exprs():
    """Basic comparison expressions for testing."""
    return {
        "eq": (pl.col("x") == 10, [[("x", "=", 10)]]),
        "neq": (pl.col("y") != "test", [[("y", "!=", "test")]]),
        "gt": (pl.col("z") > 5, [[("z", ">", 5)]]),
        "lt": (pl.col("a") < 100, [[("a", "<", 100)]]),
        "gte": (pl.col("b") >= 20, [[("b", ">=", 20)]]),
        "lte": (pl.col("c") <= 50, [[("c", "<=", 50)]]),
        "reversed": (5 == pl.col("x"), [[("x", "=", 5)]]),  # Test reversed operands
    }


@pytest.fixture
def logical_operation_exprs():
    """Logical operation expressions for testing."""
    return {
        "simple_and": ((pl.col("x") == 10) & (pl.col("y") == "test"), [[("x", "=", 10), ("y", "=", "test")]]),
        "nested_and": ((pl.col("x") == 10) & (pl.col("y") == "test") & (pl.col("z") > 5), [[("x", "=", 10), ("y", "=", "test"), ("z", ">", 5)]]),
        "simple_or": ((pl.col("x") == 10) | (pl.col("y") == "test"), [[("x", "=", 10)], [("y", "=", "test")]]),
        "nested_or": ((pl.col("x") == 10) | (pl.col("y") == "test") | (pl.col("z") > 5), [[("x", "=", 10)], [("y", "=", "test")], [("z", ">", 5)]]),
        "and_or_mixed": (
            (pl.col("x") == 10) & ((pl.col("y") == "test") | (pl.col("z") > 5)),
            [[("x", "=", 10), ("y", "=", "test")], [("x", "=", 10), ("z", ">", 5)]],
        ),
        "or_and_mixed": (
            (pl.col("x") == 10) | ((pl.col("y") == "test") & (pl.col("z") > 5)),
            [[("x", "=", 10)], [("y", "=", "test"), ("z", ">", 5)]],
        ),
    }


@pytest.fixture
def function_exprs():
    """Function expressions for testing."""
    return {
        "is_in": (pl.col("x").is_in((1, 2, 3)), [[("x", "in", [1, 2, 3])]]),
        "is_null": (pl.col("y").is_null(), [[("y", "is", None)]]),
        "is_not_null": (pl.col("z").is_not_null(), [[("z", "is not", None)]]),
        "starts_with": (pl.col("s").str.starts_with("prefix"), [[("s", "~", "^prefix")]]),
        "ends_with": (pl.col("s").str.ends_with("suffix"), [[("s", "~", "suffix$")]]),
        "contains": (pl.col("s").str.contains("substr"), [[("s", "~", ".*substr.*")]]),
    }


@pytest.fixture
def complex_exprs():
    """Complex expressions for testing."""
    return {
        "complex_1": (
            ((pl.col("a") > 5) & (pl.col("b") <= 10)) | ((pl.col("c") == "test") & (pl.col("d").is_not_null())),
            [[("a", ">", 5), ("b", "<=", 10)], [("c", "=", "test"), ("d", "is not", None)]],
        ),
        "complex_2": (
            ((pl.col("a") > 5) & (pl.col("b") <= 10)) | (((pl.col("c") == "test") | (pl.col("d").ne(True))) & pl.col("e").is_null()),
            [[("a", ">", 5), ("b", "<=", 10)], [("c", "=", "test"), ("e", "is", None)], [("d", "!=", True), ("e", "is", None)]],
        ),
        "complex_3": ((pl.col("e").is_in([1, 2, 3]) & pl.col("f").str.contains("xyz")), [[("e", "in", [1, 2, 3]), ("f", "~", ".*xyz.*")]]),
    }


def test_single_predicate():
    """Test negation of a DNF with a single predicate."""
    assert negate_dnf([[("x", "=", 0)]]) == [[("x", "!=", 0)]]


def test_single_conjunction():
    """Test negation of a DNF with a single conjunction of multiple predicates."""
    dnf = [[("x", "=", 0), ("y", ">=", 5)]]
    expected = [[("x", "!=", 0)], [("y", "<", 5)]]
    assert negate_dnf(dnf) == expected


def test_two_clauses():
    """Test negation of a DNF with two clauses."""
    dnf = [[("x", "=", 0), ("y", ">=", 5)], [("z", "<", 10)]]
    expected = [[("x", "!=", 0), ("z", ">=", 10)], [("y", "<", 5), ("z", ">=", 10)]]
    assert negate_dnf(dnf) == expected


def test_complex_dnf():
    """Test negation of a complex DNF with multiple clauses."""
    dnf = [[("a", "=", 1), ("b", ">", 2)], [("c", "<=", 3)], [("d", "in", [4, 5])]]
    expected = [[("a", "!=", 1), ("c", ">", 3), ("d", "!in", [4, 5])], [("b", "<=", 2), ("c", ">", 3), ("d", "!in", [4, 5])]]
    assert negate_dnf(dnf) == expected


def test_equal_date():
    """Test equality with date objects."""
    # Create a date object
    date_obj = date(2023, 1, 1)

    expr = pl.col("date") == date_obj
    dnf = convert_expr_to_dnf(expr)

    # Verify correct operator and value are preserved
    assert dnf == [[("date", "=", date_obj)]]


def test_is_between_date():
    """Test date range filtering with is_between"""
    # Create date objects for range
    start_date = date(2023, 1, 1)
    end_date = date(2023, 12, 31)

    expr = pl.col("date").is_between(start_date, end_date)
    dnf = convert_expr_to_dnf(expr)

    # Verify correct operators and values are preserved
    assert_dnf_equal(dnf, [[("date", ">=", start_date), ("date", "<=", end_date)]])


def test_is_between_datetime():
    """Test datetime range filtering with is_between"""
    # Create datetime objects with time components
    start_dt = datetime(2023, 1, 1, 8, 30, 0)
    end_dt = datetime(2023, 12, 31, 17, 45, 30)

    expr = pl.col("timestamp").is_between(start_dt, end_dt)
    dnf = convert_expr_to_dnf(expr)

    # Verify correct operators and datetime objects are preserved
    assert dnf == [[("timestamp", ">=", start_dt), ("timestamp", "<=", end_dt)]]


def test_is_between_date_closed_parameters():
    """Test all closed parameter options with dates"""
    start_date = date(2023, 1, 1)
    end_date = date(2023, 12, 31)

    test_cases = [("both", ">=", "<="), ("left", ">=", "<"), ("right", ">", "<="), ("none", ">", "<")]

    for closed, lower_op, upper_op in test_cases:
        expr = pl.col("date").is_between(start_date, end_date, closed=closed)
        dnf = convert_expr_to_dnf(expr)
        expected = [[("date", lower_op, start_date), ("date", upper_op, end_date)]]
        assert dnf == expected, f"Failed with closed={closed}"


def test_is_between_same_day():
    """Test edge case with same date for both bounds"""
    same_date = date(2023, 6, 15)

    # Both inclusive should be equivalent to equals
    expr = pl.col("date").is_between(same_date, same_date)
    dnf = convert_expr_to_dnf(expr)
    assert dnf == [[("date", ">=", same_date), ("date", "<=", same_date)]]

    # With "none" closed parameter, this creates an impossible condition
    expr = pl.col("date").is_between(same_date, same_date, closed="none")
    dnf = convert_expr_to_dnf(expr)
    assert dnf == [[("date", ">", same_date), ("date", "<", same_date)]]


def test_is_between_mixed_date_types():
    """Test mixing datetime and date objects"""
    # Mix datetime and date objects (this is valid in Polars)
    start = date(2023, 1, 1)
    end = datetime(2023, 12, 31, 23, 59, 59)

    expr = pl.col("dt").is_between(start, end)
    dnf = convert_expr_to_dnf(expr)

    # Should preserve the original types
    assert dnf == [[("dt", ">=", start), ("dt", "<=", end)]]
    assert isinstance(dnf[0][0][2], date)
    assert isinstance(dnf[0][1][2], datetime)


def test_is_between_with_polars_date_literals():
    """Test with Polars date literals"""
    # Using Polars literals instead of Python date objects
    expr = pl.col("date").is_between(pl.lit(date(2023, 1, 1)), pl.lit(date(2023, 12, 31)))
    dnf = convert_expr_to_dnf(expr)

    # The get_literal_value function should extract Python date objects
    assert dnf[0][0][0] == "date"
    assert dnf[0][0][1] == ">="
    assert isinstance(dnf[0][0][2], date)
    assert dnf[0][0][2] == date(2023, 1, 1)

    assert dnf[0][1][0] == "date"
    assert dnf[0][1][1] == "<="
    assert isinstance(dnf[0][1][2], date)
    assert dnf[0][1][2] == date(2023, 12, 31)


@pytest.mark.parametrize(
    "dnf,expected",
    [
        ([[("x", "=", 0)], [("y", "<>", 5)], [("z", "in", [1, 2, 3])]], [[("x", "!=", 0), ("y", "=", 5), ("z", "!in", [1, 2, 3])]]),
        ([[("x", "!=", 0)], [("y", "not in", [1, 2])], [("z", "is not", None)]], [[("x", "=", 0), ("y", "in", [1, 2]), ("z", "is", None)]]),
        ([[("x", "=", 0), ("y", "=", 0)], [("z", "=", 0)]], [[("x", "!=", 0), ("z", "!=", 0)], [("y", "!=", 0), ("z", "!=", 0)]]),
    ],
)
def test_various_dnf_forms(dnf, expected):
    """Test negation of various DNF forms with different operators and structures."""
    assert negate_dnf(dnf) == expected


@pytest.mark.parametrize("key", ["eq", "neq", "gt", "lt", "gte", "lte", "reversed"])
def test_basic_comparisons(basic_comparison_exprs, key):
    """Test conversion of basic comparison expressions."""
    expr, expected = basic_comparison_exprs[key]
    result = convert_expr_to_dnf(expr)
    assert_dnf_equal(result, expected)


@pytest.mark.parametrize("key", ["simple_and", "nested_and", "simple_or", "nested_or", "and_or_mixed", "or_and_mixed"])
def test_logical_operations(logical_operation_exprs, key):
    """Test conversion of logical operations."""
    expr, expected = logical_operation_exprs[key]
    result = convert_expr_to_dnf(expr)
    assert_dnf_equal(result, expected)


@pytest.mark.parametrize("key", ["is_in", "is_null", "is_not_null", "starts_with", "ends_with", "contains"])
def test_function_expressions(function_exprs, key):
    """Test conversion of function expressions."""
    expr, expected = function_exprs[key]
    result = convert_expr_to_dnf(expr)
    assert_dnf_equal(result, expected)


@pytest.mark.parametrize("key", ["complex_1", "complex_2", "complex_3"])
def test_complex_expressions(complex_exprs, key):
    """Test conversion of complex nested expressions."""
    expr, expected = complex_exprs[key]
    result = convert_expr_to_dnf(expr)
    assert_dnf_equal(result, expected, f"Failed for {key}")


def test_collections_in_is_in():
    """Test is_in with different collection types."""
    # List
    expr = pl.col("x").is_in([1, 2, 3])
    expected = [[("x", "in", [1, 2, 3])]]
    result = convert_expr_to_dnf(expr)
    assert_dnf_equal(result, expected)

    # Tuple
    expr = pl.col("x").is_in((4, 5, 6))
    # The result might be a list or tuple depending on Polars implementation
    result = convert_expr_to_dnf(expr)
    if result[0][0][2] == (4, 5, 6):
        assert_dnf_equal(result, [[("x", "in", (4, 5, 6))]])
    else:
        assert_dnf_equal(result, [[("x", "in", [4, 5, 6])]])

    # Empty collection
    expr = pl.col("x").is_in([])
    expected = [[("x", "in", [])]]
    result = convert_expr_to_dnf(expr)
    assert_dnf_equal(result, expected)


def test_combine_and_dnf():
    """Test the combine_and_dnf function."""
    # Basic AND combination
    left = [[("x", "=", 10)]]
    right = [[("y", "=", "test")]]
    expected = [[("x", "=", 10), ("y", "=", "test")]]
    result = combine_and_dnf(left, right)
    assert_dnf_equal(result, expected)

    # AND with multiple clauses
    left = [[("x", "=", 10)], [("y", "=", "test")]]
    right = [[("z", ">", 5)]]
    expected = [[("x", "=", 10), ("z", ">", 5)], [("y", "=", "test"), ("z", ">", 5)]]
    result = combine_and_dnf(left, right)
    assert_dnf_equal(result, expected)

    # Handling None values
    assert combine_and_dnf(None, right) == right
    assert combine_and_dnf(left, None) == left
    assert combine_and_dnf(None, None) is None


def test_unsupported_expressions():
    """Test behavior with unsupported expression types."""
    # A complex expression that isn't fully supported
    expr = pl.col("x").cast(pl.Int64) + 5
    try:
        convert_expr_to_dnf(expr)
    except Exception as e:
        pytest.fail(f"convert_expr_to_dnf raised {e} on unsupported expression")


def test_literal_comparisons():
    """Test comparisons between literals."""
    expr = pl.lit(5) > pl.lit(3)
    try:
        convert_expr_to_dnf(expr)
    except Exception as e:
        pytest.fail(f"convert_expr_to_dnf raised {e} on literal comparison")


def test_is_in_single_item():
    """Test negated operators (prefixed with !)."""
    # This tests a feature that might not be implemented yet
    expr = pl.col("x").is_in(["US"])
    expected = [[("x", "in", ["US"])]]
    result = convert_expr_to_dnf(expr)
    assert_dnf_equal(result, expected)


def test_negated_operators():
    """Test negated operators (prefixed with !)."""
    # This tests a feature that might not be implemented yet
    expr = ~(pl.col("x").is_in([1, 2, 3]))
    expected = [[("x", "!in", [1, 2, 3])]]
    result = convert_expr_to_dnf(expr)
    assert_dnf_equal(result, expected)


def test_dnf_visitor_direct():
    """Test using DNFVisitor directly."""
    expr = (pl.col("x") == 10) & (pl.col("y") > 5)

    # Parse the expression
    parser = ExprParser()
    node = parser.parse(expr)

    # Apply the visitor
    visitor = DNFVisitor()
    visitor.visit(node)
    result = visitor.process_results()
    # Verify the result
    assert result == [[("x", "=", 10), ("y", ">", 5)]]


def test_all_horizontal():
    expr = pl.all_horizontal(
        pl.col("x") > 5,
        pl.col("y") == 10,
        pl.col("z") < 7.2,
    )
    expected = [
        [("x", ">", 5), ("y", "=", 10), ("z", "<", 7.2)],
    ]

    result = convert_expr_to_dnf(expr)
    assert_dnf_equal(result, expected)


def test_any_horizontal():
    expr = pl.any_horizontal(
        pl.col("x") > 5,
        pl.col("y") == 10,
        pl.col("z") < 7.2,
    )
    expected = [
        [("x", ">", 5)],
        [("y", "=", 10)],
        [("z", "<", 7.2)],
    ]

    result = convert_expr_to_dnf(expr)
    assert_dnf_equal(result, expected)


def test_any_horizontal_nested_with_and():
    expr = pl.any_horizontal(
        pl.col("x") > 5,
        (pl.col("y") == 10) & (pl.col("z") < 3),
    )
    expected = [
        [("x", ">", 5)],
        [("y", "=", 10), ("z", "<", 3)],
    ]

    result = convert_expr_to_dnf(expr)
    assert_dnf_equal(result, expected)


def test_ternary_basic():
    """Test basic ternary expression conversion to DNF."""
    expr = pl.when(pl.col("x") > 5).then(pl.col("y") == 10).otherwise(pl.col("z") < 3)

    # Expected:
    # ((x > 5) AND (y == 10)) OR ((NOT(x > 5)) AND (z < 3))
    expected = [
        [("x", ">", 5), ("y", "=", 10)],  # predicate and truthy
        [("x", "<=", 5), ("z", "<", 3)],  # negated predicate and falsy
    ]

    result = convert_expr_to_dnf(expr)
    assert_dnf_equal(result, expected)


def test_ternary_with_complex_predicate():
    """Test ternary expression with a complex predicate."""
    expr = pl.when((pl.col("x") > 5) & (pl.col("y") < 10)).then(pl.col("z") == 15).otherwise(pl.col("a") != 0)

    # Expected:
    # ((x > 5) AND (y < 10) AND (z == 15)) OR
    # ((NOT(x > 5) OR NOT(y < 10)) AND (a != 0))
    # which simplifies to:
    # ((x > 5) AND (y < 10) AND (z == 15)) OR
    # ((x <= 5) AND (a != 0)) OR ((y >= 10) AND (a != 0))
    expected = [
        [("x", ">", 5), ("y", "<", 10), ("z", "=", 15)],  # predicate and truthy
        [("x", "<=", 5), ("a", "!=", 0)],  # first negated predicate clause and falsy
        [("y", ">=", 10), ("a", "!=", 0)],  # second negated predicate clause and falsy
    ]

    result = convert_expr_to_dnf(expr)
    assert_dnf_equal(result, expected)


def test_ternary_with_function_expressions():
    """Test ternary with function expressions."""
    expr = pl.when(pl.col("x").is_in([1, 2, 3])).then((pl.col("y") > 5) & (pl.col("z") == 0)).otherwise(pl.col("w").is_null())

    # Expected:
    # ((x in [1,2,3]) AND (y > 5) AND (z == 0)) OR
    # ((x NOT in [1,2,3]) AND (w IS NULL))
    expected = [
        [("x", "in", [1, 2, 3]), ("y", ">", 5), ("z", "=", 0)],  # predicate and truthy
        [("x", "!in", [1, 2, 3]), ("w", "is", None)],  # negated predicate and falsy
    ]

    result = convert_expr_to_dnf(expr)
    assert_dnf_equal(result, expected)


def test_ternary_nested():
    """Test nested ternary expressions."""
    # Create a nested ternary expression
    inner_ternary = pl.when(pl.col("y") > 10).then(pl.col("z") == 5).otherwise(pl.col("z") == 0)
    outer_ternary = pl.when(pl.col("x") < 0).then(inner_ternary).otherwise(pl.col("w").is_not_null())

    # Expected structure:
    # ((x < 0) AND ((y > 10) AND (z == 5) OR (y <= 10) AND (z == 0))) OR
    # ((x >= 0) AND (w IS NOT NULL))
    #
    # Which expands to:
    # ((x < 0) AND (y > 10) AND (z == 5)) OR
    # ((x < 0) AND (y <= 10) AND (z == 0)) OR
    # ((x >= 0) AND (w IS NOT NULL))
    expected = [
        [("x", "<", 0), ("y", ">", 10), ("z", "=", 5)],  # outer predicate, inner predicate, inner truthy
        [("x", "<", 0), ("y", "<=", 10), ("z", "=", 0)],  # outer predicate, negated inner predicate, inner falsy
        [("x", ">=", 0), ("w", "is not", None)],  # negated outer predicate, outer falsy
    ]

    result = convert_expr_to_dnf(outer_ternary)
    assert_dnf_equal(result, expected)


def test_ternary_with_empty_clauses():
    """Test ternary expressions with empty clauses."""
    # When no conditions in truthy or falsy branch
    # Using pl.lit(True) and pl.lit(False) for simplicity
    expr = pl.when(pl.col("x") > 5).then(pl.lit(True)).otherwise(pl.lit(False))
    result = convert_expr_to_dnf(expr)
    expected = [[("x", ">", 5)]]
    # Just verify we got a result without asserting exact structure
    assert_dnf_equal(result, expected)


def test_ternary_chained_when():
    """Test chained when-then-when expressions in Polars."""
    # Create a chained when expression like:
    # WHEN x > 10 THEN
    #    WHEN y < 5 THEN z = 1
    #    ELSE z = 2
    # ELSE z = 3
    expr = pl.when(pl.col("x") > 10).then(pl.when(pl.col("y") < 5).then(pl.col("z") == 1).otherwise(pl.col("z") == 2)).otherwise(pl.col("z") == 3)

    # Expected DNF:
    # ((x > 10) AND (y < 5) AND (z == 1)) OR
    # ((x > 10) AND (y >= 5) AND (z == 2)) OR
    # ((x <= 10) AND (z == 3))
    expected = [
        [("x", ">", 10), ("y", "<", 5), ("z", "=", 1)],  # outer condition true, inner condition true
        [("x", ">", 10), ("y", ">=", 5), ("z", "=", 2)],  # outer condition true, inner condition false
        [("x", "<=", 10), ("z", "=", 3)],  # outer condition false
    ]

    result = convert_expr_to_dnf(expr)
    assert_dnf_equal(result, expected)


def test_ternary_multiple_chained_when():
    """Test multiple chained when expressions (if-elseif-else pattern)."""
    # Create a chain of when expressions like:
    # IF x > 100 THEN "high"
    # ELSE IF x > 50 THEN "medium"
    # ELSE IF x > 10 THEN "low"
    # ELSE "very low"
    expr = (
        pl.when(pl.col("x") > 100)
        .then(pl.col("result") == "high")
        .when(pl.col("x") > 50)
        .then(pl.col("result") == "medium")
        .when(pl.col("x") > 10)
        .then(pl.col("result") == "low")
        .otherwise(pl.col("result") == "very low")
    )

    # Expected DNF:
    # ((x > 100) AND (result == "high")) OR
    # ((x <= 100) AND (x > 50) AND (result == "medium")) OR
    # ((x <= 100) AND (x <= 50) AND (x > 10) AND (result == "low")) OR
    # ((x <= 100) AND (x <= 50) AND (x <= 10) AND (result == "very low"))
    expected = [
        [("x", ">", 100), ("result", "=", "high")],
        [("x", "<=", 100), ("x", ">", 50), ("result", "=", "medium")],
        [("x", "<=", 100), ("x", "<=", 50), ("x", ">", 10), ("result", "=", "low")],
        [("x", "<=", 100), ("x", "<=", 50), ("x", "<=", 10), ("result", "=", "very low")],
    ]

    result = convert_expr_to_dnf(expr)
    assert_dnf_equal(result, expected)


def test_ternary_multiple_conditions_in_when():
    """Test when expressions with multiple comma-separated conditions."""
    # When with multiple conditions (implicitly AND-ed together)
    expr = (
        pl.when(
            pl.col("x") > 10,  # First condition
            pl.col("y") < 5,  # Second condition (implicitly AND with first)
        )
        .then(pl.col("z") == 1)
        .otherwise(pl.col("z") == 2)
    )

    # Expected DNF:
    # ((x > 10) AND (y < 5) AND (z == 1)) OR
    # ((NOT(x > 10) OR NOT(y < 5)) AND (z == 2))
    # Which simplifies to:
    # ((x > 10) AND (y < 5) AND (z == 1)) OR
    # ((x <= 10) AND (z == 2)) OR
    # ((y >= 5) AND (z == 2))
    expected = [
        [("x", ">", 10), ("y", "<", 5), ("z", "=", 1)],  # All conditions true
        [("x", "<=", 10), ("z", "=", 2)],  # First condition false
        [("y", ">=", 5), ("z", "=", 2)],  # Second condition false
    ]

    result = convert_expr_to_dnf(expr)
    assert_dnf_equal(result, expected)


def test_ternary_multiple_when_with_multiple_conditions():
    """Test multiple when clauses each with multiple conditions."""
    # Chain of when expressions with multiple conditions in each
    expr = (
        pl.when(pl.col("x") > 100, pl.col("y") == "A")
        .then(pl.col("result") == "category_1")
        .when(pl.col("x") > 50, pl.col("y") == "B")
        .then(pl.col("result") == "category_2")
        .otherwise(pl.col("result") == "category_3")
    )

    # Expected DNF:
    # ((x > 100) AND (y == "A") AND (result == "category_1")) OR
    # ((NOT(x > 100) OR NOT(y == "A")) AND (x > 50) AND (y == "B") AND (result == "category_2")) OR
    # ((NOT(x > 100) OR NOT(y == "A")) AND (NOT(x > 50) OR NOT(y == "B")) AND (result == "category_3"))
    #
    # Which simplifies to multiple clauses:
    expected = [
        [("x", ">", 100), ("y", "=", "A"), ("result", "=", "category_1")],
        [("x", "<=", 100), ("x", ">", 50), ("y", "=", "B"), ("result", "=", "category_2")],
        [("y", "!=", "A"), ("x", ">", 50), ("y", "=", "B"), ("result", "=", "category_2")],
        [("x", "<=", 100), ("x", "<=", 50), ("result", "=", "category_3")],
        [("x", "<=", 100), ("y", "!=", "B"), ("result", "=", "category_3")],
        [("y", "!=", "A"), ("x", "<=", 50), ("result", "=", "category_3")],
        [("y", "!=", "A"), ("y", "!=", "B"), ("result", "=", "category_3")],
    ]

    result = convert_expr_to_dnf(expr)
    assert_dnf_equal(result, expected)


def test_ternary_complex_with_function_conditions():
    """Test when expressions with multiple conditions including functions."""
    # Using function calls within the multiple conditions
    expr = (
        pl.when(pl.col("x").is_in([1, 2, 3]), pl.col("y").str.contains("test"), pl.col("z").is_not_null())
        .then(pl.col("result") == "match")
        .otherwise(pl.col("result") == "no_match")
    )

    # Expected DNF:
    # ((x in [1,2,3]) AND (y ~ .*test.*) AND (z is not null) AND (result == "match")) OR
    # ((x not in [1,2,3]) AND (result == "no_match")) OR
    # ((y !~ .*test.*) AND (result == "no_match")) OR
    # ((z is null) AND (result == "no_match"))
    expected = [
        [("x", "in", [1, 2, 3]), ("y", "~", ".*test.*"), ("z", "is not", None), ("result", "=", "match")],
        [("x", "!in", [1, 2, 3]), ("result", "=", "no_match")],
        [("y", "!~", ".*test.*"), ("result", "=", "no_match")],
        [("z", "is", None), ("result", "=", "no_match")],
    ]

    result = convert_expr_to_dnf(expr)
    assert_dnf_equal(result, expected)


def test_ternary_nested_with_multiple_conditions():
    """Test nested ternary expressions with multiple conditions."""
    # Nested ternary with multiple conditions at each level
    expr = (
        pl.when(pl.col("category") == "A", pl.col("value") > 100)
        .then(
            pl.when(pl.col("subcategory") == "X", pl.col("priority") > 3)
            .then(pl.col("result") == "high_priority_A")
            .otherwise(pl.col("result") == "normal_A")
        )
        .otherwise(pl.when(pl.col("category") == "B").then(pl.col("result") == "any_B").otherwise(pl.col("result") == "other"))
    )

    # This creates a complex decision tree with many branches
    # The expected DNF would have multiple clauses for each possible path
    expected = [
        # Path: category=A, value>100, subcategory=X, priority>3 -> high_priority_A
        [("category", "=", "A"), ("value", ">", 100), ("subcategory", "=", "X"), ("priority", ">", 3), ("result", "=", "high_priority_A")],
        # Path: category=A, value>100, (subcategory!=X OR priority<=3) -> normal_A
        [("category", "=", "A"), ("value", ">", 100), ("subcategory", "!=", "X"), ("result", "=", "normal_A")],
        [("category", "=", "A"), ("value", ">", 100), ("priority", "<=", 3), ("result", "=", "normal_A")],
        # Path: (category!=A OR value<=100), category=B -> any_B
        [("category", "!=", "A"), ("category", "=", "B"), ("result", "=", "any_B")],
        [("value", "<=", 100), ("category", "=", "B"), ("result", "=", "any_B")],
        # Path: (category!=A OR value<=100), category!=B -> other
        [("category", "!=", "A"), ("category", "!=", "B"), ("result", "=", "other")],
        [("value", "<=", 100), ("category", "!=", "B"), ("result", "=", "other")],
    ]

    result = convert_expr_to_dnf(expr)
    assert_dnf_equal(result, expected)


def test_expr_with_col_comparisons():
    expr = (pl.col("a") > pl.col("b")) & (pl.col("c") == 12)
    result = convert_expr_to_dnf(expr)
    expected = [[("c", "=", 12)]]
    assert_dnf_equal(result, expected)

    # we cant get a restriction here
    expr = (pl.col("a") > pl.col("b")) | (pl.col("c") == 12)
    assert convert_expr_to_dnf(expr) is None


def test_supported_cast():
    expr = (pl.col.x == 0).cast(pl.Boolean)
    result = convert_expr_to_dnf(expr)
    expected = [[("x", "=", 0)]]
    assert_dnf_equal(result, expected)


# Helper to create DNFTuples easily for testing, and we get type hints
def T(column: str, op: str, value) -> DNFTuple:
    return (column, op, value)


# Test cases: List of (description, dnf_clause, expected_is_contradiction)
contradiction_test_cases = [
    # Empty Clause
    ("Empty clause (always true)", [], False),  # this is a degenerate case
    # Exact Value Contradictions
    ("Exact: col = 5 AND col = 6", [T("colA", "=", 5), T("colA", "=", 6)], True),
    ("Exact: col = 5 AND col != 5", [T("colA", "=", 5), T("colA", "!=", 5)], True),
    ("Exact: col = 5 AND col = 5", [T("colA", "=", 5), T("colA", "=", 5)], False),  # Redundant but not a contradiction
    ("Exact: col == 5 AND col <> 5", [T("colA", "==", 5), T("colA", "<>", 5)], True),
    # Range Contradictions
    ("Range: col > 10 AND col < 5", [T("colA", ">", 10), T("colA", "<", 5)], True),
    ("Range: col > 5 AND col < 5", [T("colA", ">", 5), T("colA", "<", 5)], True),
    ("Range: col >= 5 AND col <= 5", [T("colA", ">=", 5), T("colA", "<=", 5)], False),
    ("Range: col > 5 AND col <= 5", [T("colA", ">", 5), T("colA", "<=", 5)], True),
    ("Range: col >= 5 AND col < 5", [T("colA", ">=", 5), T("colA", "<", 5)], True),
    ("Range: col > 5 AND col > 10", [T("colA", ">", 5), T("colA", ">", 10)], False),  # min_bound becomes 10
    ("Range: col < 10 AND col < 5", [T("colA", "<", 10), T("colA", "<", 5)], False),  # max_bound becomes 5
    (
        "Range: col < datetime(2024, 1, 1) AND col >= datetime(2024, 1, 2)",
        [T("colA", "<", datetime(2024, 1, 1)), T("colA", ">=", datetime(2024, 1, 2))],
        True,
    ),  # datetime bounds contradict
    (
        "Range: col < datetime(2024, 1, 1) AND col >= datetime(2022, 1, 2)",
        [T("colA", "<", datetime(2024, 1, 1)), T("colA", ">=", datetime(2022, 1, 2))],
        False,
    ),  # datetime bounds don't contradict
    # Exact Value vs. Range
    ("Exact vs Range: col = 3 AND col > 5", [T("colA", "=", 3), T("colA", ">", 5)], True),
    ("Exact vs Range: col = 7 AND col > 5 AND col < 10", [T("colA", "=", 7), T("colA", ">", 5), T("colA", "<", 10)], False),
    ("Exact vs Range: col = 5 AND col >= 5", [T("colA", "=", 5), T("colA", ">=", 5)], False),
    ("Exact vs Range: col = 5 AND col > 5", [T("colA", "=", 5), T("colA", ">", 5)], True),
    ("Exact vs Range: col = 5 AND col <= 5", [T("colA", "=", 5), T("colA", "<=", 5)], False),
    ("Exact vs Range: col = 5 AND col < 5", [T("colA", "=", 5), T("colA", "<", 5)], True),
    # Inclusion/Exclusion Contradictions
    ("In/Ex: col IN (1,2) AND col = 3", [T("colA", "in", [1, 2]), T("colA", "=", 3)], True),
    ("In/Ex: col IN (1,2) AND col = 1", [T("colA", "in", [1, 2]), T("colA", "=", 1)], False),
    ("In/Ex: col IN (1,2) AND col NOT IN (1,2,3)", [T("colA", "in", [1, 2]), T("colA", "!in", [1, 2, 3])], True),
    ("In/Ex: col IN (1,2) AND col NOT IN (3,4)", [T("colA", "in", [1, 2]), T("colA", "!in", [3, 4])], False),
    ("In/Ex: col IN (1,2) AND col IN (3,4)", [T("colA", "in", [1, 2]), T("colA", "in", [3, 4])], True),
    ("In/Ex: col IN (1,2,3) AND col IN (3,4,5)", [T("colA", "in", [1, 2, 3]), T("colA", "in", [3, 4, 5])], False),  # Valid values: {3}
    ("In/Ex: col IN (1,2,3) AND col IN (3,4,5) AND col = 3", [T("colA", "in", [1, 2, 3]), T("colA", "in", [3, 4, 5]), T("colA", "=", 3)], False),
    ("In/Ex: col IN (1,2,3) AND col IN (3,4,5) AND col = 1", [T("colA", "in", [1, 2, 3]), T("colA", "in", [3, 4, 5]), T("colA", "=", 1)], True),
    (
        "In/Ex: col IN (1,2,3) AND col IN (3,4,5) AND col NOT IN (3)",
        [T("colA", "in", [1, 2, 3]), T("colA", "in", [3, 4, 5]), T("colA", "!in", [3])],
        True,
    ),
    # NULL Contradictions
    ("NULL: col IS NULL AND col = 5", [T("colA", "is", None), T("colA", "=", 5)], True),
    ("NULL: col IS NULL AND col > 0", [T("colA", "is", None), T("colA", ">", 0)], True),
    ("NULL: col IS NULL AND col IN (1,2)", [T("colA", "is", None), T("colA", "in", [1, 2])], True),
    ("NULL: col IS NULL AND col IS NOT NULL", [T("colA", "is", None), T("colA", "is not", None)], True),
    ("NULL: col IS NOT NULL AND col = None (exact value is None)", [T("colA", "is not", None), T("colA", "=", None)], True),
    ("NULL: col IS NOT NULL AND col IN (None,)", [T("colA", "is not", None), T("colA", "in", [None])], True),
    ("NULL: col IS NOT NULL AND col IN (None, 1)", [T("colA", "is not", None), T("colA", "in", [None, 1])], False),
    ("NULL: col IS NULL (no contradiction)", [T("colA", "is", None)], False),
    ("NULL: col IS NOT NULL (no contradiction)", [T("colA", "is not", None)], False),
    ("NULL: col IS NULL AND col IN (None,)", [T("colA", "is", None), T("colA", "in", [None])], False),  # Exact value None is compatible with IS NULL
    # Complex (Single Column)
    ("Complex: col > 0 AND col < 10 AND col IN (20,30)", [T("colA", ">", 0), T("colA", "<", 10), T("colA", "in", [20, 30])], True),
    ("Complex: col > 0 AND col < 10 AND col IN (5, 20)", [T("colA", ">", 0), T("colA", "<", 10), T("colA", "in", [5, 20])], False),  # 5 is valid
    ("Complex: col = 5 AND col != 5 AND col > 0", [T("colA", "=", 5), T("colA", "!=", 5), T("colA", ">", 0)], True),
    # Multiple Columns
    ("Multi-col: colA = 1 AND colA = 2 AND colB = 10", [T("colA", "=", 1), T("colA", "=", 2), T("colB", "=", 10)], True),
    ("Multi-col: colA = 1 AND colB > 5", [T("colA", "=", 1), T("colB", ">", 5)], False),
    (
        "Multi-col: colA > 0 AND colA < 1, colB = 1 (colA contradiction only for int)",
        [T("colA", ">", 0), T("colA", "<", 1), T("colB", "=", 1)],
        False,
    ),
    (
        "Multi-col: colA > 0 AND colA < 12, colB = 1, colA < -1 (colA contradiction)",
        [T("colA", ">", 0), T("colA", "<", 1), T("colA", "<", -1), T("colB", "=", 1)],
        True,
    ),
    # Non-Contradictions
    ("Non-C: col = 5 AND col > 0", [T("colA", "=", 5), T("colA", ">", 0)], False),
    ("Non-C: col > 0 AND col < 10", [T("colA", ">", 0), T("colA", "<", 10)], False),
    ("Non-C: col IN (1,2,3) AND col = 1", [T("colA", "in", [1, 2, 3]), T("colA", "=", 1)], False),
    ("Non-C: col IS NOT NULL AND col > 5", [T("colA", "is not", None), T("colA", ">", 5)], False),
    ("Non-C: colA = 1, colB = 'test', colC > 3.0", [T("colA", "=", 1), T("colB", "=", "test"), T("colC", ">", 3.0)], False),
    # Type Issues (False due to conservative TypeError handling)
    # Edge cases for IN operator value types
    ("Edge IN: col IN 'abc' (scalar string)", [T("colA", "in", "abc")], False),  # Analyzer wraps scalar in a list
    ("Edge IN: col IN 1 (scalar int)", [T("colA", "in", 1)], False),
    # IS with non-None values (current behavior test)
    ("IS non-None: colA IS TRUE", [T("colA", "is", True)], False),  # Sets is_null = False, no contradiction alone
    ("IS non-None: colA IS TRUE AND colA IS NULL", [T("colA", "is", True), T("colA", "is", None)], True),  # is_null=False vs is_null=True
    ("IS NOT non-None: colA IS NOT TRUE", [T("colA", "is not", True)], False),  # Does not set is_null, no contradiction
]


@pytest.mark.parametrize("description, dnf_clause, expected", contradiction_test_cases)
def test_is_contradiction(description: str, dnf_clause: DNFClause, expected: bool):
    assert is_contradiction(dnf_clause) == expected, f"Test failed for: {description}"


_is_contradiction_test_cases = [
    (
        "Multi option is in check",
        (pl.col("y").is_in(["a", "b"]) & pl.col("a").eq(5) & (pl.col("a").ne(5) | pl.col("y").is_in(["b", "a"]).not_())),
        pl.Schema({"a": pl.Boolean, "y": pl.String}),
        True,
    ),
    (
        "Compare with True (python boolean)",
        (pl.col("y").is_in(["a", "b"]) & pl.col("p").eq(True) & (pl.col("p").ne(True) | pl.col("y").is_in(["b", "a"]).not_())),
        pl.Schema({"x": pl.Int64, "y": pl.String, "p": pl.Boolean, "x2": pl.Int64, "y2": pl.String, "p2": pl.Boolean}),
        True,
    ),
    (
        "Compare with pl.lit(True)",
        (pl.col("y").is_in(["a", "b"]) & pl.col("p").eq(pl.lit(True)) & pl.Expr.or_(pl.col("p").ne(pl.lit(True)), ~pl.col("y").is_in(["b", "a"]))),
        pl.Schema({"x": pl.Int64, "y": pl.String, "p": pl.Boolean, "x2": pl.Int64, "y2": pl.String, "p2": pl.Boolean}),
        True,
    ),
]


@pytest.mark.parametrize("description, pl_expr, schema, expected", _is_contradiction_test_cases)
def test_is_contradiction_pl(description: str, pl_expr: pl.Expr, schema: pl.Schema, expected: bool):
    """Test is_contradiction with Polars expressions."""
    _is_contradiction(pl_expr, schema)
    assert _is_contradiction(pl_expr, schema) == expected, f"Test failed for: {description}"


def test_complex_expression_with_boolean_cast():
    """Test a more complex expression with a cast to Boolean."""
    expr = ((pl.col("x") > 5) & (pl.col("y") == "test")).cast(pl.Boolean)
    result = convert_expr_to_dnf(expr)
    expected = [[("x", ">", 5), ("y", "=", "test")]]
    assert_dnf_equal(result, expected)


def test_nested_boolean_casts():
    """Test expression with nested casts to Boolean."""
    expr = (pl.col("x") == 0).cast(pl.Boolean).cast(pl.Boolean)
    result = convert_expr_to_dnf(expr)
    expected = [[("x", "=", 0)]]
    assert_dnf_equal(result, expected)


def test_cast_other_than_boolean():
    """Test that casts to types other than Boolean are not processed."""
    expr = (pl.col("x") == 0).cast(pl.Int32)
    # This should fail to convert to DNF since the casting to non-Boolean
    # is not handled by visit_cast
    result = convert_expr_to_dnf(expr)
    assert result is None


def test_boolean_cast_in_logical_operation():
    """Test a cast to Boolean as part of a larger logical operation."""
    expr = ((pl.col("x") > 5).cast(pl.Boolean)) & (pl.col("y") == "test")
    result = convert_expr_to_dnf(expr)
    expected = [[("x", ">", 5), ("y", "=", "test")]]
    assert_dnf_equal(result, expected)


def test_boolean_cast_in_ternary_expression():
    """Test a cast to Boolean in a ternary expression."""
    expr = pl.when((pl.col("x") > 5).cast(pl.Boolean)).then(pl.col("y") == 10).otherwise(pl.col("z") < 3)
    result = convert_expr_to_dnf(expr)
    expected = [
        [("x", ">", 5), ("y", "=", 10)],  # predicate and truthy
        [("x", "<=", 5), ("z", "<", 3)],  # negated predicate and falsy
    ]
    assert_dnf_equal(result, expected)


# Test cases for alias expressions that should fail before the fix
def test_alias_simple_equality():
    """Test that aliases in simple equality expressions are handled properly."""
    expr = pl.col("x").alias("x_alias") == 10
    result = convert_expr_to_dnf(expr)
    # This should extract the DNF from the aliased column expression
    expected = [[("x", "=", 10)]]
    assert_dnf_equal(result, expected)


def test_alias_logical_operations():
    """Test aliases in logical operations."""
    expr = (pl.col("x").alias("x_alias") == 10) & (pl.col("y").alias("y_alias") > 5)
    result = convert_expr_to_dnf(expr)
    expected = [[("x", "=", 10), ("y", ">", 5)]]
    assert_dnf_equal(result, expected)


def test_alias_nested_expression():
    """Test aliases on complex nested expressions."""
    expr = ((pl.col("x") == 10) | (pl.col("y") == "test")).alias("complex_expr")
    result = convert_expr_to_dnf(expr)
    expected = [[("x", "=", 10)], [("y", "=", "test")]]
    assert_dnf_equal(result, expected)


def test_alias_with_functions():
    """Test aliases with function calls like is_in."""
    expr = pl.col("symbol").is_in(["US", "EU"]).alias("symbol_filter")
    result = convert_expr_to_dnf(expr)
    expected = [[("symbol", "in", ["US", "EU"])]]
    assert_dnf_equal(result, expected)


def test_is_contradiction_with_column_not_in_schema():
    """Test that is_contradiction returns False when column is not in schema.

    This tests the fix for the case where schema.get(column_name) returns None.
    Before the fix, trying to access col_typ.is_integer() would fail.
    The fix returns False (conservative - not proven to be a contradiction).
    """
    # Create a clause with valid bounds that might need enumeration to check.
    # The bounds are valid (3 < x < 10), so we need schema info to check further.
    clause = [("unknown_col", ">", 3), ("unknown_col", "<", 10), ("unknown_col", "!=", 5)]

    # Schema doesn't contain 'unknown_col'
    schema = pl.Schema({"other_col": pl.Int64})

    # Should return False (not proven to be a contradiction) because we don't know the column type
    # Without the fix, this would crash when trying to call methods on None
    result = is_contradiction(clause, schema)
    assert result is False


def test_is_contradiction_with_column_in_schema():
    """Test that is_contradiction properly detects contradiction when column IS in schema."""
    # Create a clause that is a contradiction for integer columns:
    # 3 < x < 10 AND x != all valid values in that range
    # This is equivalent to: x in {4, 5, 6, 7, 8, 9} but none of those are valid
    clause = [
        ("x", ">", 3),
        ("x", "<", 10),
        ("x", "!=", 4),
        ("x", "!=", 5),
        ("x", "!=", 6),
        ("x", "!=", 7),
        ("x", "!=", 8),
        ("x", "!=", 9),
    ]

    # Schema contains 'x' as Int64
    schema = pl.Schema({"x": pl.Int64})

    # Should return True (is a contradiction) because all valid integers are excluded
    result = is_contradiction(clause, schema)
    assert result is True


def test_is_contradiction_obvious_bound_conflict():
    """Test that obvious bound conflicts (min > max) are detected without schema."""
    # x > 5 AND x < 3 is an obvious contradiction regardless of type
    clause = [("x", ">", 5), ("x", "<", 3)]

    # Even with no schema, this should be detected as a contradiction
    result = is_contradiction(clause, schema=None)
    assert result is True
