import math
from datetime import date, datetime, time, timedelta, timezone
from typing import Dict, List, Type

import orjson
import polars as pl
import pytest
from packaging import version
from polars.testing import assert_frame_equal

from polars_io_tools.io_sources.base import (
    FAILED_LITERAL_RESULT,
    AliasNode,
    AnonymousFunctionNode,
    BaseExprNode,
    BinaryExprNode,
    CastNode,
    ColumnNode,
    ColumnsNode,
    ErrorNode,
    ExplodeNode,
    ExprVisitor,
    FunctionNode,
    LiteralNode,
    OperatorType,
    SelectorNode,
    TernaryNode,
    convert_datetime_to_polars,
    extract_column_name,
    get_literal_value,
    get_parsed_expr,
)
from polars_io_tools.io_sources.enum import (
    BooleanFunctionType,
    GenericFunctionType,
    ListFunctionType,
    StringFunctionType,
    StructFunctionType,
    TemporalFunctionType,
)

from .conftest import PredicateTracker


def assert_node_type(node: BaseExprNode, expected_type: Type[BaseExprNode]) -> None:
    """Assert that a node is of the expected type."""
    assert isinstance(node, expected_type), f"Expected {expected_type.__name__}, got {type(node).__name__}"


def check_binary_expression(node: BinaryExprNode, left_type: Type[BaseExprNode], op: OperatorType, right_type: Type[BaseExprNode]) -> None:
    """Check the structure of a binary expression."""
    assert isinstance(node.left, left_type), f"Expected left to be {left_type.__name__}, got {type(node.left).__name__}"
    assert node.op == op, f"Expected op to be {op}, got {node.op}"
    assert isinstance(node.right, right_type), f"Expected right to be {right_type.__name__}, got {type(node.right).__name__}"


def test_extract_column_name():
    """Test column name extraction from expressions with casts and aliases."""
    expr = pl.col("test_column1").cast(pl.Int32)
    assert "test_column1" == extract_column_name(get_parsed_expr(expr))

    expr = pl.col("test_column2").cast(pl.Datetime)
    assert "test_column2" == extract_column_name(get_parsed_expr(expr))

    expr = pl.col("test_column3").cast(pl.Datetime(time_unit="ns"))
    assert "test_column3" == extract_column_name(get_parsed_expr(expr))

    expr = pl.struct(pl.col("a"), pl.col("b"))
    assert None is extract_column_name(get_parsed_expr(expr))

    expr = pl.col("b").mul(2).cast(pl.Float64)
    assert None is extract_column_name(get_parsed_expr(expr))

    # Aliases should be handled correctly by looking through to the underlying column
    expr = pl.col("b").alias("big_b")
    assert "b" == extract_column_name(get_parsed_expr(expr))

    # Nested aliases and casts should work
    expr = pl.col("b").cast(pl.Int64).alias("first").alias("second")
    assert "b" == extract_column_name(get_parsed_expr(expr))

    # .dt.date() floors a Datetime column to day granularity (order-preserving, like cast(pl.Date)) and should
    # resolve to the underlying column so temporal predicates push down.
    expr = pl.col("ts").dt.date()
    assert "ts" == extract_column_name(get_parsed_expr(expr))
    expr = pl.col("ts").dt.date().alias("d")
    assert "ts" == extract_column_name(get_parsed_expr(expr))


def test_parse_simple_column():
    """Test parsing a simple column reference."""
    expr = pl.col("test_column")
    node = get_parsed_expr(expr)

    assert_node_type(node, ColumnNode)
    assert node.name == "test_column"


def test_parse_multiple_columns():
    """Test parsing expressions with multiple column references."""
    expr = pl.col(["a", "b", "c"])
    node = get_parsed_expr(expr)

    # Depending on the polars version, this might be parsed differently
    # but we should get a valid node
    if version.parse(pl.__version__) > version.parse("1.31.0"):
        # TODO: Handle selectors
        assert isinstance(node, SelectorNode)
    else:
        assert isinstance(node, ColumnsNode)
    assert node.names == ["a", "b", "c"]


def test_parse_literal():
    """Test parsing literal values of different types."""
    # Test integer
    expr = pl.lit(42)
    node = get_parsed_expr(expr)

    assert_node_type(node, LiteralNode)
    assert node.value == 42

    # Test string
    expr = pl.lit("hello")
    node = get_parsed_expr(expr)

    assert_node_type(node, LiteralNode)
    assert node.value == "hello"

    # Test boolean
    expr = pl.lit(True)
    node = get_parsed_expr(expr)

    assert_node_type(node, LiteralNode)
    assert node.value is True

    # Test None
    expr = pl.lit(None)
    node = get_parsed_expr(expr)

    assert_node_type(node, LiteralNode)
    assert node.value is None

    # Test list
    expr = pl.lit([1, 2, 3])
    node = get_parsed_expr(expr)

    assert_node_type(node, LiteralNode)
    assert node.value == [1, 2, 3]


def test_special_literals():
    """Test parsing special literals like NaN and Infinity."""
    # Test NaN
    expr = pl.lit(float("nan"))
    node = get_parsed_expr(expr)
    assert_node_type(node, LiteralNode)
    assert math.isnan(node.value)

    # Test Infinity
    expr = pl.lit(float("inf"))
    node = get_parsed_expr(expr)
    assert_node_type(node, LiteralNode)
    assert node.value == float("inf")

    # Test -Infinity
    expr = pl.lit(float("-inf"))
    node = get_parsed_expr(expr)
    assert_node_type(node, LiteralNode)
    assert node.value == float("-inf")


def test_date_time_literals_parsed():
    """Test parsing date and time literals."""
    # Test datetime
    dt = datetime(2023, 1, 1, 12, 0, 0)
    expr = pl.lit(dt)
    node = get_parsed_expr(expr)

    if version.parse(pl.__version__) >= version.parse("1.36.0-beta.1"):
        # In polars 1.36+, datetime literals are parsed directly as LiteralNode
        assert_node_type(node, LiteralNode)
        assert node.value == dt
    else:
        assert_node_type(node, CastNode)
        # Polars casts this to UTC automatically
        assert node.input.value == dt.replace(tzinfo=timezone.utc)

    # Test date
    d = date(2023, 1, 1)
    expr = pl.lit(d)
    node = get_parsed_expr(expr)

    assert_node_type(node, LiteralNode)
    assert node.value == d

    # Test time
    t = time(12, 30, 45)
    expr = pl.lit(t)
    node = get_parsed_expr(expr)

    assert_node_type(node, LiteralNode)
    assert node.value == t


def test_complex_literals():
    """Test parsing complex data structure literals."""
    # Test dictionary
    expr = pl.lit({"a": 1, "b": 2})
    node = get_parsed_expr(expr)

    # We cant use dictionaries as literal values
    if version.parse(pl.__version__) > version.parse("1.31.0"):
        assert_node_type(node, LiteralNode)
        assert node.value == {"a": 1, "b": 2}
    else:
        assert_node_type(node, ErrorNode)

    # Test nested list
    expr = pl.lit([[1, 2], [3, 4]])
    node = get_parsed_expr(expr)

    assert_node_type(node, LiteralNode)
    assert node.value == [[1, 2], [3, 4]]


def test_basic_literals():
    """Test extraction of basic literal values."""
    assert get_literal_value(pl.lit(5)) == 5
    assert get_literal_value(pl.lit("test")) == "test"
    assert get_literal_value(pl.lit(True)) is True
    assert get_literal_value(pl.lit(None)) is None


def test_list_literals():
    """Test extraction of list literals."""
    assert get_literal_value(pl.lit([1, 2, 3])) == [1, 2, 3]
    # Note: Tuples might be converted to lists by Polars
    result = get_literal_value(pl.lit((4, 5, 6)))
    assert result == (4, 5, 6) or result == [4, 5, 6]


def test_non_literal_expressions():
    """Test behavior with non-literal expressions."""
    # Should return None without raising exceptions
    # but we still get an error logged
    assert get_literal_value(pl.col("non_existent")) is FAILED_LITERAL_RESULT


def test_numeric_literals():
    """Test extraction of various numeric literal types."""
    assert get_literal_value(pl.lit(42)) == 42
    assert get_literal_value(pl.lit(3.14)) == 3.14
    assert get_literal_value(pl.lit(-10)) == -10
    assert get_literal_value(pl.lit(0)) == 0
    assert get_literal_value(pl.lit(float("inf"))) == float("inf")
    assert math.isnan(get_literal_value(pl.lit(float("nan"))))


def test_string_literals_special_cases():
    """Test extraction of string literals with special cases."""
    assert get_literal_value(pl.lit("")) == ""  # Empty string
    assert get_literal_value(pl.lit(" ")) == " "  # Space
    assert get_literal_value(pl.lit("Special chars: !@#$%^&*()")) == "Special chars: !@#$%^&*()"
    assert get_literal_value(pl.lit("Line1\nLine2")) == "Line1\nLine2"  # With newline


def test_boolean_literals():
    """Test extraction of boolean literals."""
    assert get_literal_value(pl.lit(True)) is True
    assert get_literal_value(pl.lit(False)) is False


def test_date_time_literals():
    """Test extraction of date and time literals."""

    today = date.today()
    assert get_literal_value(pl.lit(today)) == today

    now = datetime.now()
    assert get_literal_value(pl.lit(now)) == now

    t = time(12, 30, 45)
    assert get_literal_value(pl.lit(t)) == t

    delta = timedelta(days=1, hours=2)
    assert get_literal_value(pl.lit(delta)) == delta


def test_complex_nested_structures():
    """Test extraction of complex nested data structures."""
    # Dictionary
    assert get_literal_value(pl.lit({"a": 1, "b": 2})) == {"a": 1, "b": 2}

    # Nested list
    assert get_literal_value(pl.lit([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]

    # Mixed structure
    complex_struct = {"name": "test", "values": [1, 2, 3], "metadata": {"active": True}}
    assert get_literal_value(pl.lit(complex_struct)) == complex_struct


def test_edge_cases():
    """Test edge cases and potential failure scenarios."""
    # Very large integer
    large_int = 10**20
    assert get_literal_value(pl.lit(large_int)) == large_int

    # Very long string
    long_str = "a" * 10000
    assert get_literal_value(pl.lit(long_str)) == long_str


def test_compound_expressions():
    """Test behavior with compound expressions that aren't pure literals."""
    # Expression combining literals
    assert get_literal_value(pl.lit(5) + pl.lit(3)) == 8

    # Expression with function call
    assert get_literal_value(pl.lit("test").str.to_uppercase()) == "TEST"


def test_binary_data():
    """Test extraction of binary data literals."""
    binary_data = b"binary\x00data"
    assert get_literal_value(pl.lit(binary_data)) == binary_data


def test_parse_binary_operation():
    """Test parsing binary operations."""
    # Test equality
    expr = pl.col("a") == pl.col("b")
    node = get_parsed_expr(expr)

    assert_node_type(node, BinaryExprNode)
    check_binary_expression(node, ColumnNode, OperatorType.EQ, ColumnNode)

    # Test addition
    expr = pl.col("a") + pl.lit(5)
    node = get_parsed_expr(expr)

    assert_node_type(node, BinaryExprNode)
    check_binary_expression(node, ColumnNode, OperatorType.PLUS, LiteralNode)


def test_parse_all_comparison_operators():
    """Test parsing all comparison operators."""
    col_a = pl.col("a")
    value = pl.lit(10)

    operations = [
        (col_a == value, OperatorType.EQ),
        (col_a != value, OperatorType.NOT_EQ),
        (col_a < value, OperatorType.LT),
        (col_a <= value, OperatorType.LT_EQ),
        (col_a > value, OperatorType.GT),
        (col_a >= value, OperatorType.GT_EQ),
    ]

    for expr, expected_op in operations:
        node = get_parsed_expr(expr)
        assert_node_type(node, BinaryExprNode)
        assert node.op == expected_op


def test_parse_all_arithmetic_operators():
    """Test parsing all arithmetic operators."""
    col_a = pl.col("a")
    value = pl.lit(10)

    operations = [
        (col_a + value, OperatorType.PLUS),
        (col_a - value, OperatorType.MINUS),
        (col_a * value, OperatorType.MULTIPLY),
        (col_a / value, OperatorType.TRUE_DIVIDE),
    ]

    for expr, expected_op in operations:
        node = get_parsed_expr(expr)
        assert_node_type(node, BinaryExprNode)
        assert node.op == expected_op


def test_parse_all_logical_operators():
    """Test parsing all logical operators."""
    expr_a = pl.col("a") > 5
    expr_b = pl.col("b") < 10

    operations = [
        (expr_a & expr_b, OperatorType.AND),
        (expr_a | expr_b, OperatorType.OR),
    ]

    for expr, expected_op in operations:
        node = get_parsed_expr(expr)
        assert_node_type(node, BinaryExprNode)
        assert node.op == expected_op


def test_binary_expr_node_contents():
    """Test the contents of a BinaryExprNode in detail."""
    # Create a simple binary expression
    expr = pl.col("a") + pl.col("b")
    node = get_parsed_expr(expr)

    assert_node_type(node, BinaryExprNode)
    binary_node = node

    # Check operation
    assert binary_node.op == OperatorType.PLUS

    # Check left side
    assert_node_type(binary_node.left, ColumnNode)
    assert binary_node.left.name == "a"

    # Check right side
    assert_node_type(binary_node.right, ColumnNode)
    assert binary_node.right.name == "b"


def test_parse_alias():
    """Test parsing aliases."""
    expr = pl.col("test_column").alias("renamed")
    node = get_parsed_expr(expr)

    assert_node_type(node, AliasNode)
    alias_node = node
    assert alias_node.name == "renamed"
    assert_node_type(alias_node.input, ColumnNode)
    assert alias_node.input.name == "test_column"


def test_nested_alias():
    """Test parsing nested aliases."""
    if version.parse(pl.__version__) > version.parse("1.30.0"):
        # Nested aliases parsing is not correct in 1.31.0-beta.1
        # Since `.meta.pop()` returns an empty list improperly.
        return
    expr = pl.col("a").alias("b").alias("c")
    node = get_parsed_expr(expr)

    assert_node_type(node, AliasNode)
    assert node.name == "c"
    assert_node_type(node.input, AliasNode)
    assert node.input.name == "b"
    assert_node_type(node.input.input, ColumnNode)
    assert node.input.input.name == "a"


def test_parse_string_function():
    """Test parsing string functions."""
    # Test string contains function
    expr = pl.col("text").str.contains("pattern")
    node = get_parsed_expr(expr)
    assert isinstance(node, FunctionNode)
    assert node.function_type == StringFunctionType.CONTAINS

    assert len(node.inputs) == 2
    col_input = node.inputs[0]
    assert isinstance(col_input, ColumnNode)
    assert col_input.name == "text"

    lit_input = node.inputs[1]
    assert isinstance(lit_input, LiteralNode)
    assert lit_input.value == "pattern"

    # Test string uppercase function
    expr = pl.col("text").str.to_uppercase()
    node = get_parsed_expr(expr)

    assert isinstance(node, FunctionNode)
    assert node.function_type == StringFunctionType.UPPERCASE

    assert len(node.inputs) == 1
    col_input = node.inputs[0]
    assert isinstance(col_input, ColumnNode)
    assert col_input.name == "text"


def test_parse_not():
    """Test parsing NOT expressions."""
    expr = ~pl.col("flag")
    node = get_parsed_expr(expr)

    assert_node_type(node, FunctionNode)
    assert len(node.inputs) == 1
    assert_node_type(node.inputs[0], ColumnNode)
    assert node.inputs[0].name == "flag"


def test_parse_boolean_function():
    """Test parsing boolean functions."""
    # Test is_null
    expr = pl.col("value").is_null()
    node = get_parsed_expr(expr)

    assert isinstance(node, FunctionNode)
    assert node.function_type == BooleanFunctionType.IS_NULL

    assert len(node.inputs) == 1
    assert_node_type(node.inputs[0], ColumnNode)
    assert node.inputs[0].name == "value"

    # Test is_not_null
    expr = pl.col("value").is_not_null()
    node = get_parsed_expr(expr)

    assert isinstance(node, FunctionNode)
    assert node.function_type == BooleanFunctionType.IS_NOT_NULL

    assert len(node.inputs) == 1
    assert_node_type(node.inputs[0], ColumnNode)
    assert node.inputs[0].name == "value"


def test_parse_is_in_function():
    """Test parsing is_in function."""
    # Test with list of values
    expr = pl.col("value").is_in([1, 2, 3])
    node = get_parsed_expr(expr)

    assert isinstance(node, FunctionNode)
    assert node.function_type == BooleanFunctionType.IS_IN

    assert len(node.inputs) == 2
    col_input = node.inputs[0]
    assert isinstance(col_input, ColumnNode)
    assert col_input.name == "value"

    lit_input = node.inputs[1]
    assert isinstance(lit_input, LiteralNode)
    assert lit_input.value == [1, 2, 3]

    # Test with column
    expr = pl.col("value").is_in(pl.col("other_list"))
    node = get_parsed_expr(expr)

    assert isinstance(node, FunctionNode)
    assert node.function_type == BooleanFunctionType.IS_IN

    assert len(node.inputs) == 2
    col_input = node.inputs[0]
    assert isinstance(col_input, ColumnNode)
    assert col_input.name == "value"

    col_input = node.inputs[1]
    assert isinstance(col_input, ColumnNode)
    assert col_input.name == "other_list"


def test_parse_is_between():
    """Test parsing is_between function."""
    # Basic is_between
    for closed in [None, "both", "left", "right", "none"]:
        if closed is None:
            expr = pl.col("value").is_between(1, 10)
        else:
            expr = pl.col("value").is_between(1, 10, closed=closed)
        node = get_parsed_expr(expr)

        assert isinstance(node, FunctionNode)
        assert node.function_type == BooleanFunctionType.IS_BETWEEN
        # Note the capitalization switch
        if closed in [None, "both"]:
            assert node.options == {"closed": "Both"}
        elif closed == "left":
            assert node.options == {"closed": "Left"}
        elif closed == "right":
            assert node.options == {"closed": "Right"}
        elif closed == "none":
            assert node.options == {"closed": "None"}
        else:
            # This should never be hit
            assert False

        assert len(node.inputs) == 3
        col_input = node.inputs[0]
        assert isinstance(col_input, ColumnNode)
        assert col_input.name == "value"

        lit_input = node.inputs[1]
        assert isinstance(lit_input, LiteralNode)
        assert lit_input.value == 1

        lit_input = node.inputs[2]
        assert isinstance(lit_input, LiteralNode)
        assert lit_input.value == 10


def test_parse_cast():
    """Test parsing cast operations."""
    # Test casting to different types
    dtypes = [pl.Int32, pl.Float64, pl.Utf8, pl.Boolean, pl.Date, pl.Datetime]

    for dtype in dtypes:
        expr = pl.col("value").cast(dtype)
        node = get_parsed_expr(expr)
        assert isinstance(node, CastNode)


def test_parse_complex_expression():
    """Test parsing a complex expression with multiple operations."""
    expr = (pl.col("age") >= 18) & (pl.col("country").is_in(["USA", "Canada", "Mexico"]))
    node = get_parsed_expr(expr)

    assert_node_type(node, BinaryExprNode)
    binary_node = node

    # Check the AND operation
    assert binary_node.op == OperatorType.AND

    # Check left side (age >= 18)
    assert_node_type(binary_node.left, BinaryExprNode)
    left_binary = binary_node.left
    assert left_binary.op == OperatorType.GT_EQ
    assert_node_type(left_binary.left, ColumnNode)
    assert left_binary.left.name == "age"
    assert_node_type(left_binary.right, LiteralNode)
    assert left_binary.right.value == 18

    # Check right side (country.is_in(...))
    assert_node_type(binary_node.right, BaseExprNode)
    assert isinstance(binary_node.right, FunctionNode)
    assert binary_node.right.function_type == BooleanFunctionType.IS_IN
    assert binary_node.right.options == {"nulls_equal": False}
    assert len(binary_node.right.inputs) == 2
    assert binary_node.right.inputs[0].name == "country"
    assert binary_node.right.inputs[1].value == ["USA", "Canada", "Mexico"]


def test_parse_complex_nested_expressions():
    """Test parsing complex nested expressions."""
    # Build a complex expression with multiple levels of nesting
    expr = (
        pl.when((pl.col("a") > 10) & ((pl.col("b") < 5) | (pl.col("c").is_in([1, 2, 3]))))
        .then(pl.col("d") * 2 + pl.col("e").cast(pl.Float64))
        .otherwise(pl.col("f").str.to_uppercase().str.contains("X") | pl.col("g").is_null())
    )

    node = get_parsed_expr(expr)

    # Verify it's a TernaryNode (when-then-otherwise)
    assert_node_type(node, TernaryNode)
    ternary_node = node

    # Check predicate part: (a > 10) & ((b < 5) | (c.is_in([1, 2, 3])))
    assert_node_type(ternary_node.predicate, BinaryExprNode)
    predicate = ternary_node.predicate
    assert predicate.op == OperatorType.AND

    # Check left side of AND: (a > 10)
    assert_node_type(predicate.left, BinaryExprNode)
    left_pred = predicate.left
    assert left_pred.op == OperatorType.GT
    assert_node_type(left_pred.left, ColumnNode)
    assert left_pred.left.name == "a"
    assert_node_type(left_pred.right, LiteralNode)
    assert left_pred.right.value == 10

    # Check right side of AND: ((b < 5) | (c.is_in([1, 2, 3])))
    assert_node_type(predicate.right, BinaryExprNode)
    right_pred = predicate.right
    assert right_pred.op == OperatorType.OR

    # Check left part of OR: (b < 5)
    assert_node_type(right_pred.left, BinaryExprNode)
    left_or = right_pred.left
    assert left_or.op == OperatorType.LT
    assert_node_type(left_or.left, ColumnNode)
    assert left_or.left.name == "b"
    assert_node_type(left_or.right, LiteralNode)
    assert left_or.right.value == 5

    # Check right part of OR: (c.is_in([1, 2, 3]))
    assert_node_type(right_pred.right, FunctionNode)
    right_or = right_pred.right
    assert right_or.function_type == BooleanFunctionType.IS_IN
    assert len(right_or.inputs) == 2
    assert_node_type(right_or.inputs[0], ColumnNode)
    assert right_or.inputs[0].name == "c"
    assert_node_type(right_or.inputs[1], LiteralNode)
    assert right_or.inputs[1].value == [1, 2, 3]

    # Check "then" branch: pl.col("d") * 2 + pl.col("e").cast(pl.Float64)
    assert_node_type(ternary_node.truthy, BinaryExprNode)
    then_expr = ternary_node.truthy
    assert then_expr.op == OperatorType.PLUS

    # Left side of addition: d * 2
    assert_node_type(then_expr.left, BinaryExprNode)
    left_add = then_expr.left
    assert left_add.op == OperatorType.MULTIPLY
    assert_node_type(left_add.left, ColumnNode)
    assert left_add.left.name == "d"
    assert_node_type(left_add.right, LiteralNode)
    assert left_add.right.value == 2

    # Right side of addition: e.cast(pl.Float64)
    assert_node_type(then_expr.right, CastNode)
    right_add = then_expr.right
    assert_node_type(right_add.input, ColumnNode)
    assert right_add.input.name == "e"
    assert isinstance(right_add.dtype, pl.Float64)

    # Check "otherwise" branch: f.str.to_uppercase().str.contains("X") | g.is_null()
    assert_node_type(ternary_node.falsy, BinaryExprNode)
    else_expr = ternary_node.falsy
    assert else_expr.op == OperatorType.OR

    # Left side of OR: string operations
    assert_node_type(else_expr.left, FunctionNode)
    left_or_else = else_expr.left
    assert left_or_else.function_type == StringFunctionType.CONTAINS
    assert len(left_or_else.inputs) == 2

    # Check input to contains is uppercase function
    assert_node_type(left_or_else.inputs[0], FunctionNode)
    upper_fn = left_or_else.inputs[0]
    assert upper_fn.function_type == StringFunctionType.UPPERCASE
    assert len(upper_fn.inputs) == 1
    assert_node_type(upper_fn.inputs[0], ColumnNode)
    assert upper_fn.inputs[0].name == "f"

    # Check literal parameter to contains
    assert_node_type(left_or_else.inputs[1], LiteralNode)
    assert left_or_else.inputs[1].value == "X"

    # Right side of OR: g.is_null()
    assert_node_type(else_expr.right, FunctionNode)
    right_or_else = else_expr.right
    assert right_or_else.function_type == BooleanFunctionType.IS_NULL
    assert len(right_or_else.inputs) == 1
    assert_node_type(right_or_else.inputs[0], ColumnNode)
    assert right_or_else.inputs[0].name == "g"


def test_large_expression():
    """Test parsing a large expression with many operations."""
    # Build a large expression with many operations
    expr = pl.lit(0)
    for i in range(50):
        expr = expr + pl.lit(i)

    node = get_parsed_expr(expr)
    assert isinstance(node, BinaryExprNode)
    cur_node = node
    # We now unwrap the expression tree
    # We work backwards, as every right node is a literal
    for i in range(49, -1, -1):
        assert isinstance(cur_node, BinaryExprNode)
        right_node = cur_node.right
        assert isinstance(right_node, LiteralNode)
        assert right_node.value == i
        cur_node = cur_node.left


def test_parse_nested_expressions():
    """Test parsing deeply nested expressions."""

    # Build a recursive expression
    def build_recursive_expr(depth: int, col_name: str = "val") -> pl.Expr:
        if depth <= 0:
            return pl.col(col_name)
        return build_recursive_expr(depth - 1, col_name) + pl.lit(depth)

    expr = build_recursive_expr(10)
    node = get_parsed_expr(expr)

    # Verify it's parsed without errors
    assert isinstance(node, BinaryExprNode)
    for i in range(10, -1, -1):
        if i == 0:
            # base case
            assert isinstance(node, ColumnNode)
            assert node.name == "val"
            break
        assert isinstance(node, BinaryExprNode)
        right_node = node.right
        assert isinstance(right_node, LiteralNode)
        assert right_node.value == i
        node = node.left


def test_parse_datetime_functions():
    """Test parsing datetime functions."""
    date_col = pl.col("date")

    # Test datetime component extraction functions
    expressions = [
        date_col.dt.year(),
        date_col.dt.month(),
        date_col.dt.day(),
        date_col.dt.hour(),
        date_col.dt.minute(),
        date_col.dt.second(),
        date_col.dt.strftime("%Y-%m-%d"),
    ]

    for expr in expressions:
        node = get_parsed_expr(expr)
        # Verify it's parsed without errors
        assert isinstance(node, FunctionNode)
        assert isinstance(node.function_type, TemporalFunctionType)


def test_parse_string_manipulation():
    """Test parsing string manipulation functions."""
    text_col = pl.col("text")

    # Test various string functions
    expressions = [
        text_col.str.to_uppercase(),
        text_col.str.to_lowercase(),
        text_col.str.contains("pattern"),
        text_col.str.starts_with("prefix"),
        text_col.str.ends_with("suffix"),
        text_col.str.replace("old", "new"),
        text_col.str.split("delimiter"),
        text_col.str.join("other"),
    ]

    for expr in expressions:
        node = get_parsed_expr(expr)
        # Verify it's parsed without errors
        assert isinstance(node, FunctionNode)
        assert isinstance(node.function_type, StringFunctionType)


def test_multiple_string_operations():
    """Test parsing multiple chained string operations."""
    expr = pl.col("text").str.to_lowercase().str.replace(" ", "_").str.replace_all("[^a-z0-9_]", "")

    node = get_parsed_expr(expr)

    assert isinstance(node, FunctionNode)
    # assert isinstance(node.function_type, StringFunctionType)
    assert node.function_type == StringFunctionType.REPLACE
    assert node.options["n"] == -1  # this means we replace all

    assert len(node.inputs) == 3
    assert isinstance(node.inputs[1], LiteralNode)
    assert node.inputs[1].value == "[^a-z0-9_]"

    assert isinstance(node.inputs[2], LiteralNode)
    assert node.inputs[2].value == ""

    first_input = node.inputs[0]
    assert isinstance(first_input, FunctionNode)
    assert first_input.function_type == StringFunctionType.REPLACE
    assert first_input.options["n"] == 1  # this means we replace all


def test_parse_list_functions():
    """Test parsing list functions."""
    list_col = pl.col("list")

    # Test various list functions
    expressions = [
        list_col.list.len(),
        list_col.list.sum(),
        list_col.list.first(),
        list_col.list.last(),
        list_col.list.get(0),
        list_col.list.join("-"),
        list_col.list.slice(0, 5),
        list_col.list.gather([0, 2, 4]),
    ]

    for expr in expressions:
        node = get_parsed_expr(expr)
        # Verify it's parsed without errors
        assert isinstance(node, FunctionNode)
        assert isinstance(node.function_type, ListFunctionType)


class ExampleTestVisitor(ExprVisitor[List[str]]):
    """Visitor that collects node type names during traversal."""

    def __init__(self):
        self.visited = []

    def default_visit(self, node: BaseExprNode):
        self.visited.append(node.__class__.__name__)

    def visit_column(self, node: ColumnNode):
        self.visited.append(f"Column({node.name})")

    def visit_binary_expr(self, node: BinaryExprNode):
        self.visited.append(f"BinaryExpr({node.op})")
        self.visit(node.left)
        self.visit(node.right)

    def process_results(self):
        return self.visited


def test_visitor_pattern():
    """Test the visitor pattern implementation."""
    # Create an expression
    expr = (pl.col("a") == 5) & (pl.col("b") > 10)
    node = get_parsed_expr(expr)

    # Apply the visitor
    visitor = ExampleTestVisitor()
    visitor.visit(node)
    result = visitor.process_results()

    # Check results
    assert result[0] == f"BinaryExpr({OperatorType.AND})"
    assert "Column(a)" in result
    assert "Column(b)" in result


class CountingVisitor(ExprVisitor[Dict[str, int]]):
    """Visitor that counts node types."""

    def __init__(self):
        self.counts = {}

    def default_visit(self, node: BaseExprNode):
        node_type = node.__class__.__name__
        self.counts[node_type] = self.counts.get(node_type, 0) + 1
        node.visit_children(self)

    def process_results(self):
        return self.counts


def test_counting_visitor():
    """Test a visitor that counts node types."""
    # Create a complex expression
    expr = (pl.col("a") > 5) & (pl.col("b").is_in(["x", "y", "z"]))
    node = get_parsed_expr(expr)

    # Count node types
    visitor = CountingVisitor()
    visitor.visit(node)
    counts = visitor.process_results()

    # Check that we have the expected nodes
    # > and & comparisons
    assert counts.get("BinaryExprNode", 0) == 2
    assert counts.get("ColumnNode", 0) == 2
    # 5 and ["x", "y", "z"] literals
    assert counts.get("LiteralNode", 0) == 2
    # is_in function
    assert counts.get("FunctionNode", 0) == 1


def test_list_eval():
    """Test parsing filter operations."""
    # Filter a list column by a condition
    expr = pl.col("list_column").list.eval(pl.element() > 0)
    # In `1.31.0` we added a new "Eval" node type
    if version.parse(pl.__version__) <= version.parse("1.30.0"):
        node = get_parsed_expr(expr)
        assert isinstance(node, ErrorNode)
        assert "serialization not supported for this 'opaque' function" in node.error
    else:
        val = expr.meta.serialize(format="json")
        val_dict = orjson.loads(val)
        assert len(val_dict) == 1 and "Eval" in val_dict
        # TODO: Support parsing this node type


def test_ternary_operations():
    """Test parsing ternary operations (if-then-else)."""
    # When-then-otherwise expression
    expr = pl.when(pl.col("a") > 0).then(1).otherwise(0)
    node = get_parsed_expr(expr)

    # Verify it's parsed without errors
    assert isinstance(node, TernaryNode)
    assert isinstance(node.predicate, BinaryExprNode)
    assert node.predicate.op == OperatorType.GT

    assert node.truthy.value == 1
    assert node.falsy.value == 0


def test_explode_operations():
    """Test parsing explode operations."""
    expr = pl.col("list_column").explode()
    node = get_parsed_expr(expr)

    assert isinstance(node, ExplodeNode)
    assert node.input.name == "list_column"


def test_struct_operations():
    """Test parsing struct operations."""
    expr = pl.col("struct_col").struct.field("our_field_name")
    node = get_parsed_expr(expr)

    assert isinstance(node, FunctionNode)
    assert node.function_type == StructFunctionType.FIELD_BY_NAME
    assert len(node.inputs) == 1
    assert node.inputs[0].name == "struct_col"
    assert node.options == {node.function_type.value: "our_field_name"}

    expr = pl.col("struct_col").struct.field("our_field_name") > 5
    node = get_parsed_expr(expr)
    assert isinstance(node, BinaryExprNode)
    assert node.op == OperatorType.GT
    assert isinstance(node.left, FunctionNode)
    assert node.left.function_type == StructFunctionType.FIELD_BY_NAME
    assert len(node.left.inputs) == 1
    assert node.left.inputs[0].name == "struct_col"
    assert node.left.options == {node.left.function_type.value: "our_field_name"}
    assert node.right.can_extract_literal
    assert node.right.value == 5


def test_unknown_expressions():
    """Test an unmapped expression (coalesce)"""
    # Create a COALESCE like operation
    expr = pl.coalesce(pl.col("a"), pl.col("b"), pl.col("c"))
    node = get_parsed_expr(expr)

    # Verify it's parsed without errors
    assert isinstance(node, FunctionNode)
    # We don't map this specific function, so it gets mapped to unknown
    assert node.function_type == GenericFunctionType.UNKNOWN


def test_get_literal_value():
    """Test the get_literal_value function."""
    # Test basic types
    assert get_literal_value(pl.lit(5)) == 5
    assert get_literal_value(pl.lit("test")) == "test"
    assert get_literal_value(pl.lit(True)) is True
    assert get_literal_value(pl.lit(None)) is None

    # Test lists
    assert get_literal_value(pl.lit([1, 2, 3])) == [1, 2, 3]

    # Test dictionaries
    assert get_literal_value(pl.lit({"a": 1, "b": 2})) == {"a": 1, "b": 2}

    # Test date/time
    dt = datetime(2023, 1, 1)
    assert get_literal_value(pl.lit(dt)) == dt

    # Test with non-literal expressions
    assert get_literal_value(pl.col("non_existent")) is FAILED_LITERAL_RESULT


def test_numeric_predicates(tester):
    """Test that numeric predicates are pushed down and give correct results"""
    # Simple comparisons
    expr1 = pl.col("id") > 5
    tester.assert_predicate_pushed_down(expr1)

    expr2 = pl.col("float_val") <= 3.0
    tester.assert_predicate_pushed_down(expr2)

    # Compound conditions
    expr3 = (pl.col("id") > 3) & (pl.col("int_val") < 70)
    tester.assert_predicate_pushed_down(expr3)

    # Math expressions
    expr4 = pl.col("float_val") * 2 > 5
    tester.assert_predicate_pushed_down(expr4)


def test_string_predicates(tester):
    """Test that string predicates are pushed down and give correct results"""
    # Exact match
    expr1 = pl.col("string_val") == "A"
    tester.assert_predicate_pushed_down(expr1)

    # String contains - version dependent
    expr2 = pl.col("long_string").str.contains("text_")
    if version.parse(pl.__version__) >= version.parse("1.25.2"):
        # Should be pushed down in newer versions
        tester.assert_predicate_pushed_down(expr2)
    tester.assert_results_match(expr2)

    # String patterns
    expr3 = pl.col("string_val").is_in(["A", "B"])
    tester.assert_predicate_pushed_down(expr3)


def test_fill_null_predicates(tester):
    for val in [0, 1, 9]:
        expr = pl.col("nullable").fill_null(val) > 5
        tester.assert_predicate_pushed_down(expr)


def test_same_dtype_column_comparison(tester):
    expr = pl.col("id") < pl.col("int_val")
    tester.assert_predicate_pushed_down(expr)


def test_struct_field_pushed_down(tester):
    expr = pl.col("struct").struct.field("value") > 5
    tester.assert_predicate_pushed_down(expr)


def test_fill_ternary_predicates(tester):
    expr = (
        pl.when(
            pl.col("nullable") > 3,
            pl.col("int_val") > 5,
        )
        .then(pl.col("int_val") > 2)
        .when(pl.col("float_val") > 0)
        .then(pl.col("string_val").is_in(["a", "b"]))
        .when(pl.col("float_val") < 1)
        .then(pl.lit(True))
        .otherwise(pl.col("float_val") < 5)
    )
    tester.assert_predicate_pushed_down(expr)


def test_temporal_predicates(tester):
    """Test that temporal predicates are pushed down and give correct results"""
    # Date comparisons
    expr1 = pl.col("date_val") > date(2023, 1, 5)
    tester.assert_predicate_pushed_down(expr1)

    # DateTime operations - we have two cases
    datetime_val = datetime(2023, 1, 1, 18, 0, 0)
    datetime_val_end = datetime(2023, 1, 1, 23, 59, 59)

    # Case 1: Standard Python datetime
    expr2 = pl.col("datetime_val") < datetime_val
    tester.assert_results_match(expr2)
    if version.parse(pl.__version__) >= version.parse("1.26.0"):
        # Fixed in polars 1.26
        tester.assert_predicate_pushed_down(expr2)

    # Case 2: Converted datetime (should push down)
    expr3 = pl.col("datetime_val") < convert_datetime_to_polars(datetime_val)
    tester.assert_predicate_pushed_down(expr3)

    # Case 3: IsBetween check
    expr3 = pl.col("datetime_val").is_between(convert_datetime_to_polars(datetime_val), convert_datetime_to_polars(datetime_val_end))
    tester.assert_predicate_pushed_down(expr3)

    expr4 = pl.col("datetime_val").is_between(datetime_val, datetime_val_end)
    if version.parse(pl.__version__) >= version.parse("1.26.0"):
        # Fixed in polars 1.26
        tester.assert_predicate_pushed_down(expr4)

    # Extract components
    expr5 = pl.col("date_val").dt.day() == 3
    if version.parse(pl.__version__) >= version.parse("1.25.2"):
        # Should be pushed down in newer versions
        tester.assert_predicate_pushed_down(expr5)


def test_temporal_schema_edge_case():
    # Create a tracker with a custom dataframe where datetime values have
    # schema of nanoseconds.
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
            ],
            "value": [10, 20, 30],
        },
        schema={"timestamp": pl.Datetime(time_unit="ns"), "value": pl.Int64},
    )
    tracker = PredicateTracker(df)
    datetime_val = datetime(2023, 1, 1, 18, 0, 0)
    datetime_val_end = datetime(2023, 1, 1, 23, 59, 59)
    expr = pl.col("timestamp").is_between(datetime_val, datetime_val_end)
    expected_pushed_down = False
    if version.parse(pl.__version__) >= version.parse("1.28.0"):
        expected_pushed_down = True
    tracker.assert_predicate_pushed_down(expr, expected_pushed_down=expected_pushed_down)

    expr_converted = pl.col("timestamp").is_between(convert_datetime_to_polars(datetime_val), convert_datetime_to_polars(datetime_val_end))
    if version.parse(pl.__version__) >= version.parse("1.26.0"):
        tracker.assert_predicate_pushed_down(expr_converted)


def test_boolean_predicates(tester):
    """Test that boolean predicates are pushed down and give correct results"""
    # Direct boolean filtering
    expr1 = pl.col("bool_val")
    tester.assert_predicate_pushed_down(expr1)

    # Boolean operators
    expr2 = ~pl.col("bool_val")
    tester.assert_predicate_pushed_down(expr2)


def test_null_predicates(tester):
    """Test that null predicates are pushed down and give correct results"""
    # Is null checks
    expr1 = pl.col("nullable").is_null()
    tester.assert_predicate_pushed_down(expr1)

    # Is not null checks
    expr2 = pl.col("nullable").is_not_null()
    tester.assert_predicate_pushed_down(expr2)


def test_categorical_predicates(tester):
    """Test that categorical predicates are pushed down and give correct results"""
    # Categorical comparisons
    expr = pl.col("cat_val") == "A"
    tester.assert_predicate_pushed_down(expr)


def test_complex_predicates(tester):
    """Test that complex predicates are pushed down and give correct results"""
    # Complex nested expressions
    complex_expr = (pl.col("id") > 3) & ((pl.col("string_val") == "A") | (pl.col("float_val") > 3.0)) & ~pl.col("bool_val") | pl.col("cat_val").is_in(
        ["A", "C"]
    )
    tester.assert_predicate_pushed_down(complex_expr)

    # Multiple column types in the same expression
    mixed_expr = (pl.col("date_val") > date(2023, 1, 3)) & (pl.col("string_val").is_in(["A", "B"])) & (pl.col("float_val") > 2.0)
    tester.assert_predicate_pushed_down(mixed_expr)


def test_predicates_with_implicit_casts(tester):
    if version.parse(pl.__version__) < version.parse("1.28.0"):
        # Implicit casts prevent any predicate pushdown in
        # previouse versions of polars
        return
    datetime_expr = pl.col("datetime_val") > date(2023, 1, 2)
    datetime_expr_is_between = pl.col("datetime_val").is_between(date(2023, 1, 2), date(2023, 1, 3))
    tester.assert_predicate_pushed_down(datetime_expr)
    tester.assert_predicate_pushed_down(datetime_expr_is_between)

    float_int_col = pl.col("int_val") == pl.col("float_val")
    tester.assert_predicate_pushed_down(float_int_col)

    date_expr = pl.col("date_val") > datetime(2023, 1, 2, 12)
    tester.assert_predicate_pushed_down(date_expr)


@pytest.mark.parametrize("strict", [True, False])
def test_predicate_with_explicit_cast_on_column(tester, strict):
    """A filter with an explicit (redundant) cast on a column is pushed down.

    Parametrized to verify both strict=True and strict=False cast behavior.
    """
    # Use a no-op cast (Int -> Int64) so both strict modes succeed
    expr = pl.col("int_val").cast(pl.Int64, strict=strict) > 30
    tester.assert_predicate_pushed_down(expr)


@pytest.mark.parametrize("strict", [True, False])
def test_compare_float_gt_int_with_int_to_float_cast_pushed_down(tester, strict):
    """Ensure pushdown when casting int_val to Float64 in a float vs int comparison."""
    expr = pl.col("float_val") > pl.col("int_val").cast(pl.Float64, strict=strict)
    expected_pushed_down = version.parse(pl.__version__) >= version.parse("1.28.0")
    tester.assert_predicate_pushed_down(expr, expected_pushed_down=expected_pushed_down)


@pytest.mark.parametrize("strict", [True, False])
def test_compare_float_gt_int_with_float_to_int_cast_pushed_down(tester, strict):
    """Ensure pushdown when casting float_val to Int64 in a float vs int comparison."""
    expr = pl.col("float_val").cast(pl.Int64, strict=strict) > pl.col("int_val")
    expected_pushed_down = version.parse(pl.__version__) >= version.parse("1.28.0")
    tester.assert_predicate_pushed_down(expr, expected_pushed_down=expected_pushed_down)


def test_unsimplified_not_predicates(tester):
    # NOTE: Here we test that the polars optimization process
    # applies DeMorgan's law for us here
    expr = ~((pl.col("string_val") == "A") | (pl.col("id") > 3))

    def assert_func(original_expr, pushed_expr):
        # Check that the original expression is equivalent to the pushed expression
        # expected_pushed = (pl.col("string_val") != "A") & (pl.col("id") <= 3)

        pushed_expr_node = get_parsed_expr(pushed_expr)
        assert isinstance(pushed_expr_node, BinaryExprNode)
        assert pushed_expr_node.op == OperatorType.AND

        original_expr_node = get_parsed_expr(original_expr)
        assert isinstance(original_expr_node, FunctionNode)
        assert original_expr_node.function_type == BooleanFunctionType.NOT
        assert len(original_expr_node.inputs) == 1
        inner_node = original_expr_node.inputs[0]
        assert isinstance(inner_node, BinaryExprNode)
        assert inner_node.op == OperatorType.OR

    tester.assert_predicate_pushed_down(expr, assert_func)

    double_nested_expr = ~((pl.col("string_val") == "A") | (~(pl.col("date_val") == date(2023, 1, 3)) & (pl.col("float_val") > 2.0)))
    tester.assert_predicate_pushed_down(double_nested_expr)


def test_map_batches(tester):
    expr = pl.col("int_val").map_batches(lambda x: x > 5, return_dtype=pl.Boolean, is_elementwise=True)

    def assert_func(original_expr, pushed_expr):
        pushed_expr_node = get_parsed_expr(pushed_expr)
        assert isinstance(pushed_expr_node, AnonymousFunctionNode)

    # TODO: Check the actual shit
    if version.parse(pl.__version__) > version.parse("1.31.0"):
        pytest.xfail("This is not pushed down properly in polars 1.32.0 pre-release")
    tester.assert_predicate_pushed_down(expr, assert_func)


def test_fill_null_constant(tester):
    expr = pl.col("nullable").fill_null(pl.when(pl.col("id") > 5).then(100).otherwise(0)) > 50

    def assert_func(original_expr, pushed_expr):
        node = get_parsed_expr(pushed_expr)
        # TODO: The function is FillNull, add support for this in parsing
        assert node.left.function_type == GenericFunctionType.FILL_NULL

    if version.parse(pl.__version__) > version.parse("1.30.0"):
        tester.assert_predicate_pushed_down(expr, assert_func)
    else:
        # In versions 1.30.0 and before, this is not pushed down
        tester.assert_predicate_pushed_down(expr, expected_pushed_down=False)


def test_column_selection_with_predicates(tester):
    """Test that column selection works together with predicates"""
    lf = tester.lazy_frame
    expr = pl.col("id") > 5

    # Reset the tracker
    tester.reset()

    # Apply filter and select only certain columns
    filtered_result = lf.filter(expr).select(["id", "string_val", "date_val"]).collect()

    # Direct equivalent operation
    direct_result = tester.df.filter(expr).select(["id", "string_val", "date_val"])

    # Verify the results match
    assert_frame_equal(filtered_result, direct_result)

    # Verify both projection and predicate were pushed down
    assert tester.last_predicate is not None, "Predicate was not pushed down"
    assert tester.last_with_columns is not None, "Column projection was not pushed down"
    assert set(tester.last_with_columns) == {"id", "string_val", "date_val"}


def test_all_results_match_regardless_of_pushdown(tester):
    """
    Test that results match the direct filter even for expressions
    that might not be pushed down
    """
    # A mix of expressions that may or may not be pushed down
    expressions = [
        # Basic expressions (likely pushed down)
        (pl.col("id") > 5, True),
        (pl.col("string_val") == "A", True),
        # Complex expressions
        (pl.col("float_val").sin() > 0, False),
        (pl.col("string_val").str.to_uppercase() == "A", False),
        (pl.col("date_val").dt.strftime("%Y-%m-%d") == "2023-01-01", False),
        # In 1.30.0, this is not pushed down. Uncomment when this issue is fixed
        # https://github.com/pola-rs/polars/issues/22860
        # (pl.col("float_val").map_elements(lambda x: x > 2.0), False),
    ]

    # For each expression, ensure results match regardless of pushdown
    for expr, expected_pushdown in expressions:
        expected_pushdown = expected_pushdown or version.parse(pl.__version__) >= version.parse("1.25.2")
        tester.assert_predicate_pushed_down(expr, expected_pushed_down=expected_pushdown)
        if expected_pushdown:
            tester.assert_results_match(expr)


def test_expected_not_pushed_down(tester):
    filter_expressions = [
        # Filters with Window Functions
        pl.col("float_val") > pl.col("float_val").mean().over("string_val"),
        (pl.col("float_val") - pl.col("float_val").mean().over("string_val")) / pl.col("float_val").std().over("string_val") > 1.0,
        pl.col("float_val").diff().over([pl.col("string_val")]) > 0.5,
        # Filters with Aggregations
        pl.col("int_val") > pl.col("int_val").mean(),
        (pl.col("float_val") > pl.col("float_val").mean()) & (pl.col("int_val") < pl.col("int_val").median()),
        (pl.col("float_val") - pl.col("float_val").min()) / (pl.col("float_val").max() - pl.col("float_val").min()) > 0.5,
        # Filters with Group By Operations Combined with Other Complex Logic
        (pl.col("float_val") > pl.col("float_val").mean()) & (pl.col("float_val") > pl.col("float_val").mean().over("string_val")),
        (pl.col("id").count().over("string_val") > 1) & (pl.col("nullable").null_count() < pl.len()).over("string_val"),
        pl.col("float_val") > (pl.col("float_val").mean().over("string_val") + pl.col("float_val").std().over("string_val")),
        # Mixed Complex Expressions
        pl.col("nullable").fill_null(pl.col("nullable").mean().over("string_val")) > pl.col("nullable").fill_null(0).mean(),
        pl.when(pl.col("string_val") == "A")
        .then(pl.col("float_val") > pl.col("float_val").mean().over("string_val"))
        .otherwise(pl.col("int_val") > pl.col("int_val").median()),
        pl.col("float_val") > pl.col("float_val").first().over("string_val"),
        # Filters with fill_null that are not elementwise are not pushed down
        pl.col("nullable").fill_null(strategy="min") > 3,
        pl.col("float_val").diff() > 0.5,
        pl.col("int_val").map_batches(lambda x: x > 5, return_dtype=pl.Boolean, is_elementwise=False),
    ]
    for expr in filter_expressions:
        # Check that the predicate is not pushed down
        tester.assert_predicate_pushed_down(expr, expected_pushed_down=False)


def test_fill_null_function_parsing():
    """Test that FillNull functions are parsed correctly with proper function type."""
    # Test simple FillNull with literal
    expr = pl.col("nullable").fill_null(42) > 50
    node = get_parsed_expr(expr)

    assert isinstance(node, BinaryExprNode)
    assert node.op == OperatorType.GT
    assert isinstance(node.left, FunctionNode)
    assert node.left.function_type == GenericFunctionType.FILL_NULL
    assert len(node.left.inputs) == 2

    # Check inputs
    assert isinstance(node.left.inputs[0], ColumnNode)
    assert node.left.inputs[0].name == "nullable"
    assert isinstance(node.left.inputs[1], LiteralNode)
    assert node.left.inputs[1].value == 42

    # Check right side of comparison
    assert isinstance(node.right, LiteralNode)
    assert node.right.value == 50


def test_fill_null_with_expression():
    """Test that FillNull with complex expressions is parsed correctly."""
    # Test FillNull with ternary expression
    expr = pl.col("nullable").fill_null(pl.when(pl.col("id") > 5).then(100).otherwise(0)) > 50
    node = get_parsed_expr(expr)

    assert isinstance(node, BinaryExprNode)
    assert node.op == OperatorType.GT
    assert isinstance(node.left, FunctionNode)
    assert node.left.function_type == GenericFunctionType.FILL_NULL
    assert len(node.left.inputs) == 2

    # Check first input is column
    assert isinstance(node.left.inputs[0], ColumnNode)
    assert node.left.inputs[0].name == "nullable"

    # Check second input is ternary expression
    assert isinstance(node.left.inputs[1], TernaryNode)
    fill_expr = node.left.inputs[1]
    assert isinstance(fill_expr.predicate, BinaryExprNode)
    assert fill_expr.predicate.op == OperatorType.GT


def test_value_property_on_different_node_types():
    """Test that the value property works correctly on different node types.

    This tests that:
    - LiteralNode.value returns the literal value
    - ColumnNode.value returns None (uses BaseExprNode default)
    - Non-extractable nodes return None
    """
    # Test LiteralNode - should return the literal value
    literal_expr = pl.lit(42)
    literal_node = get_parsed_expr(literal_expr)
    assert isinstance(literal_node, LiteralNode)
    assert literal_node.value == 42
    assert literal_node.can_extract_literal is True

    # Test ColumnNode - should return None (from BaseExprNode default)
    col_expr = pl.col("x")
    col_node = get_parsed_expr(col_expr)
    assert isinstance(col_node, ColumnNode)
    assert col_node.value is None
    assert col_node.can_extract_literal is False

    # Test FunctionNode with column input - should not be extractable
    func_expr = pl.col("x").str.to_uppercase()
    func_node = get_parsed_expr(func_expr)
    assert isinstance(func_node, FunctionNode)
    assert func_node.value is None  # Because input is column, not literal
    assert func_node.can_extract_literal is False

    # Test nested literal - LiteralNode inside CastNode
    cast_lit_expr = pl.lit(42).cast(pl.Float64)
    cast_node = get_parsed_expr(cast_lit_expr)
    assert isinstance(cast_node, CastNode)
    # CastNode with literal input should be extractable
    assert cast_node.can_extract_literal is True
    # The value should be the casted literal
    assert cast_node.value == 42.0


class TestLimitNotPushedWithFilter:
    """
    Polars does NOT push head()/limit() (n_rows) to IO sources when a filter is present.

    This is critical - if the limit were pushed along with a filter that the source
    can't fully handle, you'd get fewer rows than expected.
    """

    @pytest.mark.parametrize("limit_method", ["head", "limit"])
    def test_limit_not_pushed_when_filter_present(self, limit_method):
        """
        Verify that Polars passes n_rows=None when there's a predicate.

        This is the fundamental safety behavior that prevents incorrect results
        when filters can't be fully pushed to the data source.
        """
        from polars.io.plugins import register_io_source

        received_params = {}

        def tracking_source(with_columns, predicate, n_rows, batch_size):
            received_params["predicate"] = predicate
            received_params["n_rows"] = n_rows
            yield pl.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})

        schema = {"x": pl.Int64, "y": pl.Int64}

        lf = register_io_source(tracking_source, schema=schema)
        lf_filtered = lf.filter(pl.col("x") > 2)

        if limit_method == "head":
            lf_filtered.head(2).collect()
        else:
            lf_filtered.limit(2).collect()

        assert received_params["predicate"] is not None, "Predicate should be passed"
        assert received_params["n_rows"] is None, (
            f"n_rows should NOT be pushed when predicate is present (using {limit_method}), got {received_params['n_rows']}"
        )

    @pytest.mark.parametrize("limit_method", ["head", "limit"])
    def test_limit_pushed_when_no_filter(self, limit_method):
        """Verify that head()/limit() IS pushed when there's no filter."""
        from polars.io.plugins import register_io_source

        received_params = {}

        def tracking_source(with_columns, predicate, n_rows, batch_size):
            received_params["predicate"] = predicate
            received_params["n_rows"] = n_rows
            yield pl.DataFrame({"x": [1, 2, 3, 4, 5]})

        schema = {"x": pl.Int64}

        lf = register_io_source(tracking_source, schema=schema)

        if limit_method == "head":
            lf.head(3).collect()
        else:
            lf.limit(3).collect()

        assert received_params["predicate"] is None, "No predicate should be passed"
        assert received_params["n_rows"] == 3, f"n_rows should be 3 (using {limit_method}), got {received_params['n_rows']}"
