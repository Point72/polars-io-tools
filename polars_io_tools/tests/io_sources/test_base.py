"""Parsing Polars expressions into nodes (``get_parsed_expr``), and reading literal values and column names from them.

A parsed tree is compared with a ``NodeShape`` (``helpers.node_shapes``): the node class expected at that place, plus
the attributes to check. Which filters Polars hands to an IO source is recorded in ``tests/pushdown/test_filter.py``.
"""

import json
import math
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from datetime import UTC, date, datetime, time, timedelta

import polars as pl
import pytest

from polars_io_tools.io_sources.base import (
    FAILED_LITERAL_RESULT,
    AliasNode,
    BaseExprNode,
    BinaryExprNode,
    CastNode,
    ColumnNode,
    ErrorNode,
    ExprVisitor,
    FunctionNode,
    KeepNameNode,
    LiteralNode,
    NodeTraverserParser,
    OperatorType,
    TernaryNode,
    UnknownNode,
    extract_column_name,
    extract_value_preserving_column_name,
    get_literal_value,
    get_parsed_expr,
    has_exact_value,
    is_value_order_preserving_cast,
)
from polars_io_tools.io_sources.enum import (
    BooleanFunctionType,
    GenericFunctionType,
    StringFunctionType,
    TemporalFunctionType,
)
from polars_io_tools.tests.helpers.cases import as_params
from polars_io_tools.tests.helpers.node_shapes import NodeShape, match_shape
from polars_io_tools.tests.helpers.polars_versions import polars_has


def member_of(enum: type) -> callable:
    return lambda function_type: isinstance(function_type, enum)


def column_shape(name: str) -> NodeShape:
    return NodeShape(ColumnNode, name=name)


def literal_shape(value) -> NodeShape:
    return NodeShape(LiteralNode, value=value)


def binary_shape(op: OperatorType, left: NodeShape, right: NodeShape) -> NodeShape:
    return NodeShape(BinaryExprNode, op=op, left=left, right=right)


def function_shape(function_type, *inputs: NodeShape, **attrs) -> NodeShape:
    """A function node of ``function_type``; ``inputs`` are checked only if given."""
    return NodeShape(FunctionNode, function_type=function_type, **({"inputs": list(inputs)} if inputs else {}), **attrs)


# --- get_parsed_expr --------------------------------------------------------------------------------------------------

_NOON_JAN_1_2023 = datetime(2023, 1, 1, 12)
_IS_BETWEEN_INPUTS = [column_shape("value"), literal_shape(1), literal_shape(10)]


def _datetime_literal_shape() -> NodeShape:
    if polars_has("datetime_literal_unwrapped"):
        return literal_shape(_NOON_JAN_1_2023)
    # Older Polars wraps the literal in a cast to the naive dtype.
    return NodeShape(CastNode, input=NodeShape(BaseExprNode, value=_NOON_JAN_1_2023.replace(tzinfo=UTC)))


@dataclass(frozen=True)
class ParseCase:
    """An expression and the tree it parses into."""

    id: str
    _: KW_ONLY
    expr: pl.Expr
    shape: NodeShape
    marks: tuple = ()


PARSE_CASES = [
    ParseCase("parse_simple_column", expr=pl.col("test_column"), shape=column_shape("test_column")),
    # literals
    ParseCase("value_property_on_different_node_types-literal", expr=pl.lit(42), shape=NodeShape(LiteralNode, value=42, can_extract_literal=True)),
    ParseCase("parse_literal-string", expr=pl.lit("hello"), shape=literal_shape("hello")),
    ParseCase("parse_literal-boolean", expr=pl.lit(True), shape=literal_shape(True)),
    ParseCase("parse_literal-none", expr=pl.lit(None), shape=literal_shape(None)),
    ParseCase("parse_literal-list", expr=pl.lit([1, 2, 3]), shape=literal_shape([1, 2, 3])),
    ParseCase("special_literals-nan", expr=pl.lit(float("nan")), shape=literal_shape(math.isnan)),
    ParseCase("special_literals-inf", expr=pl.lit(float("inf")), shape=literal_shape(float("inf"))),
    ParseCase("special_literals-neg_inf", expr=pl.lit(float("-inf")), shape=literal_shape(float("-inf"))),
    ParseCase("date_time_literals_parsed-datetime", expr=pl.lit(_NOON_JAN_1_2023), shape=_datetime_literal_shape()),
    ParseCase("date_time_literals_parsed-date", expr=pl.lit(date(2023, 1, 1)), shape=literal_shape(date(2023, 1, 1))),
    ParseCase("date_time_literals_parsed-time", expr=pl.lit(time(12, 30, 45)), shape=literal_shape(time(12, 30, 45))),
    ParseCase(
        "complex_literals-dict",
        expr=pl.lit({"a": 1, "b": 2}),
        shape=literal_shape({"a": 1, "b": 2}) if polars_has("struct_literal_parsed") else NodeShape(ErrorNode),
    ),
    ParseCase("complex_literals-nested_list", expr=pl.lit([[1, 2], [3, 4]]), shape=literal_shape([[1, 2], [3, 4]])),
    ParseCase("value_property_on_different_node_types-column", expr=pl.col("x"), shape=NodeShape(ColumnNode, value=None, can_extract_literal=False)),
    ParseCase(
        "value_property_on_different_node_types-function",
        expr=pl.col("x").str.to_uppercase(),
        shape=NodeShape(FunctionNode, value=None, can_extract_literal=False),
    ),
    ParseCase(
        "value_property_on_different_node_types-cast_literal",
        expr=pl.lit(42).cast(pl.Float64),
        shape=NodeShape(CastNode, value=42.0, can_extract_literal=True),
    ),
    # binary operators
    ParseCase(
        "parse_binary_operation-eq",
        expr=pl.col("a") == pl.col("b"),
        shape=binary_shape(OperatorType.EQ, NodeShape(ColumnNode), NodeShape(ColumnNode)),
    ),
    ParseCase(
        "parse_binary_operation-plus",
        expr=pl.col("a") + pl.lit(5),
        shape=binary_shape(OperatorType.PLUS, NodeShape(ColumnNode), NodeShape(LiteralNode)),
    ),
    ParseCase(
        "binary_expr_node_contents", expr=pl.col("a") + pl.col("b"), shape=binary_shape(OperatorType.PLUS, column_shape("a"), column_shape("b"))
    ),
    *[
        ParseCase(f"parse_all_comparison_operators-{op.name.lower()}", expr=fn(pl.col("a"), pl.lit(10)), shape=NodeShape(BinaryExprNode, op=op))
        for fn, op in [
            (lambda a, b: a == b, OperatorType.EQ),
            (lambda a, b: a != b, OperatorType.NOT_EQ),
            (lambda a, b: a < b, OperatorType.LT),
            (lambda a, b: a <= b, OperatorType.LT_EQ),
            (lambda a, b: a > b, OperatorType.GT),
            (lambda a, b: a >= b, OperatorType.GT_EQ),
        ]
    ],
    *[
        ParseCase(f"parse_all_arithmetic_operators-{op.name.lower()}", expr=fn(pl.col("a"), pl.lit(10)), shape=NodeShape(BinaryExprNode, op=op))
        for fn, op in [
            (lambda a, b: a + b, OperatorType.PLUS),
            (lambda a, b: a - b, OperatorType.MINUS),
            (lambda a, b: a * b, OperatorType.MULTIPLY),
            (lambda a, b: a / b, OperatorType.TRUE_DIVIDE),
        ]
    ],
    ParseCase("parse_all_logical_operators-and", expr=(pl.col("a") > 5) & (pl.col("b") < 10), shape=NodeShape(BinaryExprNode, op=OperatorType.AND)),
    ParseCase("parse_all_logical_operators-or", expr=(pl.col("a") > 5) | (pl.col("b") < 10), shape=NodeShape(BinaryExprNode, op=OperatorType.OR)),
    ParseCase(
        "nested_alias",
        expr=pl.col("a").alias("b").alias("c"),
        shape=NodeShape(AliasNode, name="c", input=NodeShape(AliasNode, name="b", input=column_shape("a"))),
        marks=(pytest.mark.skipif(polars_has("nested_alias_dropped"), reason="meta.pop() drops a nested alias"),),
    ),
    # functions
    ParseCase(
        "parse_string_function-contains",
        expr=pl.col("text").str.contains("pattern"),
        shape=function_shape(StringFunctionType.CONTAINS, column_shape("text"), literal_shape("pattern")),
    ),
    ParseCase(
        "parse_string_function-uppercase",
        expr=pl.col("text").str.to_uppercase(),
        shape=function_shape(StringFunctionType.UPPERCASE, column_shape("text")),
    ),
    ParseCase("parse_not", expr=~pl.col("flag"), shape=NodeShape(FunctionNode, inputs=[column_shape("flag")])),
    ParseCase(
        "parse_boolean_function-is_null", expr=pl.col("value").is_null(), shape=function_shape(BooleanFunctionType.IS_NULL, column_shape("value"))
    ),
    ParseCase(
        "parse_boolean_function-is_not_null",
        expr=pl.col("value").is_not_null(),
        shape=function_shape(BooleanFunctionType.IS_NOT_NULL, column_shape("value")),
    ),
    ParseCase(
        "parse_is_in_function-list",
        expr=pl.col("value").is_in([1, 2, 3]),
        shape=function_shape(BooleanFunctionType.IS_IN, column_shape("value"), literal_shape([1, 2, 3])),
    ),
    ParseCase(
        "parse_is_in_function-column",
        expr=pl.col("value").is_in(pl.col("other_list")),
        shape=function_shape(BooleanFunctionType.IS_IN, column_shape("value"), column_shape("other_list")),
    ),
    ParseCase(
        "parse_is_between-default",
        expr=pl.col("value").is_between(1, 10),
        shape=function_shape(BooleanFunctionType.IS_BETWEEN, *_IS_BETWEEN_INPUTS, options={"closed": "Both"}),
    ),
    *[
        ParseCase(
            f"parse_is_between-{closed}",
            expr=pl.col("value").is_between(1, 10, closed=closed),
            shape=function_shape(BooleanFunctionType.IS_BETWEEN, *_IS_BETWEEN_INPUTS, options={"closed": closed.capitalize()}),
        )
        for closed in ("both", "left", "right", "none")
    ],
    *[
        ParseCase(f"parse_cast-{dtype.__name__}", expr=pl.col("value").cast(dtype), shape=NodeShape(CastNode))
        for dtype in [pl.Int32, pl.Float64, pl.Utf8, pl.Boolean, pl.Date, pl.Datetime]
    ],
    *[
        ParseCase(f"parse_datetime_functions-{name}", expr=expr, shape=NodeShape(FunctionNode, function_type=member_of(TemporalFunctionType)))
        for name, expr in [
            ("year", pl.col("date").dt.year()),
            ("month", pl.col("date").dt.month()),
            ("day", pl.col("date").dt.day()),
            ("hour", pl.col("date").dt.hour()),
            ("minute", pl.col("date").dt.minute()),
            ("second", pl.col("date").dt.second()),
            ("strftime", pl.col("date").dt.strftime("%Y-%m-%d")),
        ]
    ],
    *[
        ParseCase(f"parse_string_manipulation-{name}", expr=expr, shape=NodeShape(FunctionNode, function_type=member_of(StringFunctionType)))
        for name, expr in [
            ("to_lowercase", pl.col("text").str.to_lowercase()),
            ("starts_with", pl.col("text").str.starts_with("prefix")),
            ("ends_with", pl.col("text").str.ends_with("suffix")),
            ("replace", pl.col("text").str.replace("old", "new")),
            ("split", pl.col("text").str.split("delimiter")),
            ("join", pl.col("text").str.join("other")),
        ]
    ],
    ParseCase(
        "unknown_expressions",
        expr=pl.coalesce(pl.col("a"), pl.col("b"), pl.col("c")),
        shape=NodeShape(FunctionNode, function_type=GenericFunctionType.UNKNOWN),
    ),
    ParseCase(
        "ternary_operations",
        expr=pl.when(pl.col("a") > 0).then(1).otherwise(0),
        shape=NodeShape(TernaryNode, predicate=NodeShape(BinaryExprNode, op=OperatorType.GT), truthy=literal_shape(1), falsy=literal_shape(0)),
    ),
    ParseCase(
        "fill_null_function_parsing",
        expr=pl.col("nullable").fill_null(42) > 50,
        shape=binary_shape(
            OperatorType.GT, function_shape(GenericFunctionType.FILL_NULL, column_shape("nullable"), literal_shape(42)), literal_shape(50)
        ),
    ),
    ParseCase(
        "fill_null_with_expression",
        expr=pl.col("nullable").fill_null(pl.when(pl.col("id") > 5).then(100).otherwise(0)) > 50,
        shape=NodeShape(
            BinaryExprNode,
            op=OperatorType.GT,
            left=function_shape(
                GenericFunctionType.FILL_NULL,
                column_shape("nullable"),
                NodeShape(TernaryNode, predicate=NodeShape(BinaryExprNode, op=OperatorType.GT)),
            ),
        ),
    ),
    ParseCase(
        "parse_complex_expression",
        expr=(pl.col("age") >= 18) & pl.col("country").is_in(["USA", "Canada", "Mexico"]),
        shape=binary_shape(
            OperatorType.AND,
            binary_shape(OperatorType.GT_EQ, column_shape("age"), literal_shape(18)),
            function_shape(
                BooleanFunctionType.IS_IN, column_shape("country"), literal_shape(["USA", "Canada", "Mexico"]), options={"nulls_equal": False}
            ),
        ),
    ),
    ParseCase(
        "parse_complex_nested_expressions",
        expr=pl.when((pl.col("a") > 10) & ((pl.col("b") < 5) | pl.col("c").is_in([1, 2, 3])))
        .then(pl.col("d") * 2 + pl.col("e").cast(pl.Float64))
        .otherwise(pl.col("f").str.to_uppercase().str.contains("X") | pl.col("g").is_null()),
        shape=NodeShape(
            TernaryNode,
            predicate=binary_shape(
                OperatorType.AND,
                binary_shape(OperatorType.GT, column_shape("a"), literal_shape(10)),
                binary_shape(
                    OperatorType.OR,
                    binary_shape(OperatorType.LT, column_shape("b"), literal_shape(5)),
                    function_shape(BooleanFunctionType.IS_IN, column_shape("c"), literal_shape([1, 2, 3])),
                ),
            ),
            truthy=binary_shape(
                OperatorType.PLUS,
                binary_shape(OperatorType.MULTIPLY, column_shape("d"), literal_shape(2)),
                NodeShape(CastNode, input=column_shape("e"), dtype=pl.Float64),
            ),
            falsy=binary_shape(
                OperatorType.OR,
                function_shape(StringFunctionType.CONTAINS, function_shape(StringFunctionType.UPPERCASE, column_shape("f")), literal_shape("X")),
                function_shape(BooleanFunctionType.IS_NULL, column_shape("g")),
            ),
        ),
    ),
]


@pytest.mark.parametrize("case", as_params(PARSE_CASES))
def test_parse(case: ParseCase):
    case.shape.match(get_parsed_expr(case.expr))


@dataclass(frozen=True)
class AdditionChainCase:
    """A long chain of additions of literals, which must parse into a left-deep tree."""

    id: str
    _: KW_ONLY
    # the left-most operand
    start: pl.Expr
    # the first literal added
    first: int
    # how many literals are added
    count: int


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            AdditionChainCase("large_expression", start=pl.lit(0), first=0, count=50),
            AdditionChainCase("parse_nested_expressions", start=pl.col("val"), first=1, count=10),
        ]
    ),
)
def test_parse_long_addition_chain(case: AdditionChainCase):
    """``start + lit(first) + ... + lit(first + count - 1)`` parses into a left-deep chain with each literal on the right."""
    expr = case.start
    for i in range(case.first, case.first + case.count):
        expr = expr + pl.lit(i)
    node = get_parsed_expr(expr)
    for i in reversed(range(case.first, case.first + case.count)):
        binary_shape(OperatorType.PLUS, NodeShape(BaseExprNode), literal_shape(i)).match(node)
        node = node.left
    (literal_shape(0) if case.first == 0 else column_shape("val")).match(node)


def test_list_eval_is_an_error_node_until_polars_serializes_it():
    expr = pl.col("list_column").list.eval(pl.element() > 0)
    if not polars_has("list_eval_serializes"):
        node = get_parsed_expr(expr)
        assert isinstance(node, ErrorNode)
        assert "serialization not supported for this 'opaque' function" in node.error
    else:
        val_dict = json.loads(expr.meta.serialize(format="json"))
        assert len(val_dict) == 1 and "Eval" in val_dict


class ExampleTestVisitor(ExprVisitor[list[str]]):
    """Collects node type names during traversal."""

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
    visitor = ExampleTestVisitor()
    visitor.visit(get_parsed_expr((pl.col("a") == 5) & (pl.col("b") > 10)))
    result = visitor.process_results()
    assert result[0] == f"BinaryExpr({OperatorType.AND})"
    assert "Column(a)" in result
    assert "Column(b)" in result


class CountingVisitor(ExprVisitor[dict[str, int]]):
    """Counts node types, visiting children through ``visit_children``."""

    def __init__(self):
        self.counts = {}

    def default_visit(self, node: BaseExprNode):
        node_type = node.__class__.__name__
        self.counts[node_type] = self.counts.get(node_type, 0) + 1
        node.visit_children(self)

    def process_results(self):
        return self.counts


def test_counting_visitor():
    visitor = CountingVisitor()
    visitor.visit(get_parsed_expr((pl.col("a") > 5) & pl.col("b").is_in(["x", "y", "z"])))
    assert visitor.process_results() == {"BinaryExprNode": 2, "ColumnNode": 2, "LiteralNode": 2, "FunctionNode": 1}


def test_node_traverser_parser_isolates_subtree_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure in one ``_parse_<kind>`` collapses only that subtree to ``ErrorNode``; the rest still parses."""

    def boom(self: NodeTraverserParser, expr: pl.Expr, obj: object, visitor: object) -> BaseExprNode:
        raise RuntimeError("simulated subtree failure")

    monkeypatch.setattr(NodeTraverserParser, "_parse_function_expr", boom)
    node = NodeTraverserParser().parse((pl.col("x") > 5) & pl.col("y").is_null())
    NodeShape(BinaryExprNode, left=NodeShape(BinaryExprNode), right=NodeShape(ErrorNode, error=lambda e: "simulated subtree failure" in e)).match(
        node
    )


def test_node_traverser_parser_unsupported_expression_degrades_to_unknown() -> None:
    """A Python UDF, which Polars' typed view refuses, degrades to ``UnknownNode`` rather than raising."""
    assert isinstance(NodeTraverserParser().parse(pl.col("x").map_batches(lambda s: s, return_dtype=pl.Boolean)), UnknownNode)


@pytest.mark.parametrize(
    "expr",
    [pl.col("x").is_null().alias("p"), pl.col("x").is_null().name.keep(), (~pl.col("x").is_null()).alias("p").alias("q")],
    ids=["alias", "keep_name", "nested_alias"],
)
def test_naming_wrapper_does_not_repeat_function(expr):
    df = pl.DataFrame({"x": [None, 1]})
    node = get_parsed_expr(expr)
    while isinstance(node, AliasNode | KeepNameNode):
        node = node.input
    assert isinstance(node, FunctionNode)
    assert df.select(node.expr).equals(df.select(expr.meta.undo_aliases()))
    assert not isinstance(node.inputs[0], FunctionNode) or node.inputs[0].function_type != node.function_type


@dataclass(frozen=True)
class CastOptionsCase:
    """Keyword arguments to ``cast``, and the options the parsed cast node holds."""

    id: str
    _: KW_ONLY
    kwargs: dict
    options: str


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            CastOptionsCase("strict", kwargs={"strict": True}, options="Strict"),
            CastOptionsCase("non_strict", kwargs={"strict": False}, options="NonStrict"),
            CastOptionsCase("wrap_numerical", kwargs={"wrap_numerical": True}, options="Overflowing"),
        ]
    ),
)
def test_cast_options_parsed(case: CastOptionsCase):
    NodeShape(CastNode, options=case.options).match(get_parsed_expr(pl.col("x").cast(pl.Int8, **case.kwargs)))


@dataclass(frozen=True)
class ExactValueCase:
    """A literal compared with a column, and whether ``has_exact_value`` finds its value exact."""

    id: str
    _: KW_ONLY
    value: pl.Expr
    # read the literal as a collection of values
    collection: bool
    expected: bool


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            ExactValueCase("int", value=pl.lit(1), collection=False, expected=True),
            ExactValueCase("list_as_scalar", value=pl.lit([1, 2]), collection=False, expected=False),
            ExactValueCase("list_as_collection", value=pl.lit([1, 2]), collection=True, expected=True),
            ExactValueCase("series", value=pl.lit(pl.Series(["a", "b"])), collection=False, expected=False),
            ExactValueCase("nan", value=pl.lit(float("nan")), collection=False, expected=False),
            ExactValueCase("nan_in_collection", value=pl.lit([float("nan"), 1.0]), collection=True, expected=False),
            # Valid in Polars, but outside Python's date range.
            ExactValueCase("date_past_python_max", value=pl.lit(4_000_000).cast(pl.Date), collection=False, expected=False),
            ExactValueCase("datetime_past_python_max", value=pl.lit(10**18).cast(pl.Datetime("us")), collection=False, expected=False),
        ]
    ),
)
def test_has_exact_value(case: ExactValueCase):
    node = get_parsed_expr(pl.col("x") == case.value).right
    assert has_exact_value(node, collection=case.collection) is case.expected


# --- get_literal_value ------------------------------------------------------------------------------------------------

_NOW, _TODAY = datetime.now(), date.today()
_NESTED_STRUCT = {"name": "test", "values": [1, 2, 3], "metadata": {"active": True}}


@dataclass(frozen=True)
class LiteralValueCase:
    """A literal expression and the Python value ``get_literal_value`` reads from it."""

    id: str
    _: KW_ONLY
    expr: pl.Expr
    # the value, or a function the value must pass
    expected: object


LITERAL_CASES = [
    LiteralValueCase("basic_literals-int", expr=pl.lit(5), expected=5),
    LiteralValueCase("basic_literals-string", expr=pl.lit("test"), expected="test"),
    LiteralValueCase("basic_literals-true", expr=pl.lit(True), expected=True),
    LiteralValueCase("basic_literals-none", expr=pl.lit(None), expected=None),
    LiteralValueCase("boolean_literals-false", expr=pl.lit(False), expected=False),
    LiteralValueCase("list_literals-list", expr=pl.lit([1, 2, 3]), expected=[1, 2, 3]),
    LiteralValueCase("list_literals-tuple", expr=pl.lit((4, 5, 6)), expected=lambda v: v in ((4, 5, 6), [4, 5, 6])),
    LiteralValueCase("non_literal_expressions", expr=pl.col("non_existent"), expected=lambda v: v is FAILED_LITERAL_RESULT),
    LiteralValueCase("numeric_literals-int", expr=pl.lit(42), expected=42),
    LiteralValueCase("numeric_literals-float", expr=pl.lit(3.14), expected=3.14),
    LiteralValueCase("numeric_literals-negative", expr=pl.lit(-10), expected=-10),
    LiteralValueCase("numeric_literals-zero", expr=pl.lit(0), expected=0),
    LiteralValueCase("numeric_literals-inf", expr=pl.lit(float("inf")), expected=float("inf")),
    LiteralValueCase("numeric_literals-nan", expr=pl.lit(float("nan")), expected=math.isnan),
    LiteralValueCase("string_literals_special_cases-empty", expr=pl.lit(""), expected=""),
    LiteralValueCase("string_literals_special_cases-space", expr=pl.lit(" "), expected=" "),
    LiteralValueCase("string_literals_special_cases-punctuation", expr=pl.lit("Special chars: !@#$%^&*()"), expected="Special chars: !@#$%^&*()"),
    LiteralValueCase("string_literals_special_cases-newline", expr=pl.lit("Line1\nLine2"), expected="Line1\nLine2"),
    LiteralValueCase("date_time_literals-date", expr=pl.lit(_TODAY), expected=_TODAY),
    LiteralValueCase("date_time_literals-datetime", expr=pl.lit(_NOW), expected=_NOW),
    LiteralValueCase("get_literal_value-datetime", expr=pl.lit(datetime(2023, 1, 1)), expected=datetime(2023, 1, 1)),
    LiteralValueCase("date_time_literals-time", expr=pl.lit(time(12, 30, 45)), expected=time(12, 30, 45)),
    LiteralValueCase("date_time_literals-timedelta", expr=pl.lit(timedelta(days=1, hours=2)), expected=timedelta(days=1, hours=2)),
    LiteralValueCase("complex_nested_structures-dict", expr=pl.lit({"a": 1, "b": 2}), expected={"a": 1, "b": 2}),
    LiteralValueCase("complex_nested_structures-nested_list", expr=pl.lit([[1, 2], [3, 4]]), expected=[[1, 2], [3, 4]]),
    LiteralValueCase("complex_nested_structures-mixed", expr=pl.lit(_NESTED_STRUCT), expected=_NESTED_STRUCT),
    LiteralValueCase("edge_cases-large_int", expr=pl.lit(10**20), expected=10**20),
    LiteralValueCase("edge_cases-long_string", expr=pl.lit("a" * 10000), expected="a" * 10000),
    LiteralValueCase("compound_expressions-sum", expr=pl.lit(5) + pl.lit(3), expected=8),
    LiteralValueCase("compound_expressions-function", expr=pl.lit("test").str.to_uppercase(), expected="TEST"),
    LiteralValueCase("binary_data", expr=pl.lit(b"binary\x00data"), expected=b"binary\x00data"),
]


@pytest.mark.parametrize("case", as_params(LITERAL_CASES))
def test_get_literal_value(case: LiteralValueCase):
    match_shape(get_literal_value(case.expr), case.expected, "value")


# --- column extraction ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnNameCase:
    """An expression, a column-name extractor and the column name it reads from the expression."""

    id: str
    _: KW_ONLY
    expr: pl.Expr
    # the column name, or None when the expression has no single column
    expected: str | None
    extractor: Callable[..., str | None]
    # column dtypes, for extractors that need them
    schema: dict[str, pl.DataType] | None


COLUMN_CASES = [
    ColumnNameCase(
        "extract_column_name-cast_int",
        expr=pl.col("test_column1").cast(pl.Int32),
        expected="test_column1",
        extractor=extract_column_name,
        schema=None,
    ),
    ColumnNameCase(
        "extract_column_name-cast_datetime",
        expr=pl.col("test_column2").cast(pl.Datetime),
        expected="test_column2",
        extractor=extract_column_name,
        schema=None,
    ),
    ColumnNameCase(
        "extract_column_name-cast_datetime_ns",
        expr=pl.col("test_column3").cast(pl.Datetime("ns")),
        expected="test_column3",
        extractor=extract_column_name,
        schema=None,
    ),
    ColumnNameCase("extract_column_name-struct", expr=pl.struct(pl.col("a"), pl.col("b")), expected=None, extractor=extract_column_name, schema=None),
    ColumnNameCase(
        "extract_column_name-arithmetic", expr=pl.col("b").mul(2).cast(pl.Float64), expected=None, extractor=extract_column_name, schema=None
    ),
    ColumnNameCase("extract_column_name-alias", expr=pl.col("b").alias("big_b"), expected="b", extractor=extract_column_name, schema=None),
    ColumnNameCase(
        "extract_column_name-cast_and_aliases",
        expr=pl.col("b").cast(pl.Int64).alias("first").alias("second"),
        expected="b",
        extractor=extract_column_name,
        schema=None,
    ),
    # ``dt.date()`` floors to day granularity, order-preserving like ``cast(pl.Date)``.
    ColumnNameCase("extract_column_name-dt_date", expr=pl.col("ts").dt.date(), expected="ts", extractor=extract_column_name, schema=None),
    ColumnNameCase(
        "extract_column_name-dt_date_alias", expr=pl.col("ts").dt.date().alias("d"), expected="ts", extractor=extract_column_name, schema=None
    ),
    # The strict extractor unwraps a cast only when the schema proves it value-preserving.
    ColumnNameCase(
        "extract_value_preserving_column_name-alias",
        expr=pl.col("b").alias("big_b"),
        expected="b",
        extractor=extract_value_preserving_column_name,
        schema=None,
    ),
    ColumnNameCase(
        "extract_value_preserving_column_name-cast_no_schema",
        expr=pl.col("a").cast(pl.Int32),
        expected=None,
        extractor=extract_value_preserving_column_name,
        schema=None,
    ),
    ColumnNameCase(
        "extract_value_preserving_column_name-narrowing",
        expr=pl.col("a").cast(pl.Int8, wrap_numerical=True),
        expected=None,
        extractor=extract_value_preserving_column_name,
        schema={"a": pl.Int64},
    ),
    ColumnNameCase(
        "extract_value_preserving_column_name-widening",
        expr=pl.col("a").cast(pl.Int64),
        expected="a",
        extractor=extract_value_preserving_column_name,
        schema={"a": pl.Int8},
    ),
    ColumnNameCase(
        "extract_value_preserving_column_name-decimal_widening",
        expr=pl.col("x").cast(pl.Decimal(38, 2)),
        expected="x",
        extractor=extract_value_preserving_column_name,
        schema={"x": pl.Decimal(10, 2)},
    ),
]


@pytest.mark.parametrize("case", as_params(COLUMN_CASES))
def test_extract_column(case: ColumnNameCase):
    node = get_parsed_expr(case.expr)
    assert (case.extractor(node) if case.schema is None else case.extractor(node, case.schema)) == case.expected


@dataclass(frozen=True)
class OrderPreservingCastCase:
    """A cast from one dtype to another, and whether it keeps every value and their order."""

    id: str
    _: KW_ONLY
    source: pl.DataType | type[pl.DataType]
    target: pl.DataType | type[pl.DataType]
    expect: bool


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            OrderPreservingCastCase("identity", source=pl.Int64, target=pl.Int64, expect=True),
            OrderPreservingCastCase("int_widening", source=pl.Int8, target=pl.Int64, expect=True),
            OrderPreservingCastCase("int_narrowing", source=pl.Int64, target=pl.Int8, expect=False),
            OrderPreservingCastCase("unsigned_to_wider_signed", source=pl.UInt8, target=pl.Int16, expect=True),
            OrderPreservingCastCase("signed_to_unsigned", source=pl.Int8, target=pl.UInt64, expect=False),
            OrderPreservingCastCase("decimal_widening", source=pl.Decimal(10, 2), target=pl.Decimal(38, 2), expect=True),
            OrderPreservingCastCase("decimal_rescaling", source=pl.Decimal(10, 2), target=pl.Decimal(38, 4), expect=False),
            OrderPreservingCastCase("cross_family", source=pl.Int64, target=pl.Float64, expect=False),
            OrderPreservingCastCase("unparameterized", source=pl.Datetime, target=pl.Datetime("ms"), expect=False),
        ]
    ),
)
def test_is_value_order_preserving_cast(case: OrderPreservingCastCase):
    assert is_value_order_preserving_cast(case.source, case.target) is case.expect


def test_total_days_parses_to_function_node():
    node = get_parsed_expr((pl.col("end_dt") - pl.col("start_dt")).dt.total_days())
    assert isinstance(node, FunctionNode) and node.function_type == TemporalFunctionType.TOTAL_DAYS
