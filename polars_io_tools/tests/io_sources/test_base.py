"""Expression parsing (``get_parsed_expr``), literal extraction, column extraction, and predicate pushdown through an IO
source. Parsed trees are compared with ``N`` shapes: a node class plus the attributes to check.
"""

import json
import math
from datetime import UTC, date, datetime, time, timedelta

import polars as pl
import pytest
from polars.io.plugins import register_io_source
from polars.testing import assert_frame_equal

from polars_io_tools.io_sources.base import (
    FAILED_LITERAL_RESULT,
    AliasNode,
    AnonymousFunctionNode,
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
    convert_datetime_to_polars,
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
from polars_io_tools.tests.helpers.cases import Case, params
from polars_io_tools.tests.helpers.polars_versions import polars_has

from .conftest import PredicateTracker

c = pl.col
Op = OperatorType
Bool = BooleanFunctionType
Str = StringFunctionType


class N:
    """The expected shape of a parsed node: its class and the attributes to compare.

    An attribute value that is an ``N`` or a list of them is matched recursively, a function is a predicate on the
    attribute, ``None`` and booleans are compared by identity, and anything else by equality.
    """

    def __init__(self, cls: type[BaseExprNode], **attrs):
        self.cls, self.attrs = cls, attrs

    def match(self, node, path: str = "node") -> None:
        assert isinstance(node, self.cls), f"{path}: expected {self.cls.__name__}, got {type(node).__name__}"
        for name, want in self.attrs.items():
            _match(getattr(node, name), want, f"{path}.{name}")


def _match(got, want, path: str) -> None:
    if isinstance(want, N):
        want.match(got, path)
    elif isinstance(want, list) and want and isinstance(want[0], N):
        assert len(got) == len(want), f"{path}: {len(got)} items, expected {len(want)}"
        for i, (g, w) in enumerate(zip(got, want)):
            _match(g, w, f"{path}[{i}]")
    elif callable(want) and not isinstance(want, type):
        assert want(got), f"{path}: {got!r} fails {want}"
    elif want is None or isinstance(want, bool):
        assert got is want, f"{path}: {got!r} is not {want!r}"
    else:
        assert got == want, f"{path}: {got!r} != {want!r}"


def kind_of(enum: type) -> callable:
    return lambda function_type: isinstance(function_type, enum)


def Col(name: str) -> N:
    return N(ColumnNode, name=name)


def Lit(value) -> N:
    return N(LiteralNode, value=value)


def Bin(op: OperatorType, left: N, right: N) -> N:
    return N(BinaryExprNode, op=op, left=left, right=right)


def Fn(function_type, *inputs: N, **attrs) -> N:
    return N(FunctionNode, function_type=function_type, **({"inputs": list(inputs)} if inputs else {}), **attrs)


# --- get_parsed_expr --------------------------------------------------------------------------------------------------

_DT = datetime(2023, 1, 1, 12)
_IS_BETWEEN_INPUTS = [Col("value"), Lit(1), Lit(10)]


def _datetime_literal_shape() -> N:
    if polars_has("datetime_literal_unwrapped"):
        return Lit(_DT)
    # Older Polars wraps the literal in a cast to the naive dtype.
    return N(CastNode, input=N(BaseExprNode, value=_DT.replace(tzinfo=UTC)))


PARSE_CASES = [
    Case("parse_simple_column", expr=c("test_column"), shape=Col("test_column")),
    # literals
    Case("value_property_on_different_node_types-literal", expr=pl.lit(42), shape=N(LiteralNode, value=42, can_extract_literal=True)),
    Case("parse_literal-string", expr=pl.lit("hello"), shape=Lit("hello")),
    Case("parse_literal-boolean", expr=pl.lit(True), shape=Lit(True)),
    Case("parse_literal-none", expr=pl.lit(None), shape=Lit(None)),
    Case("parse_literal-list", expr=pl.lit([1, 2, 3]), shape=Lit([1, 2, 3])),
    Case("special_literals-nan", expr=pl.lit(float("nan")), shape=Lit(math.isnan)),
    Case("special_literals-inf", expr=pl.lit(float("inf")), shape=Lit(float("inf"))),
    Case("special_literals-neg_inf", expr=pl.lit(float("-inf")), shape=Lit(float("-inf"))),
    Case("date_time_literals_parsed-datetime", expr=pl.lit(_DT), shape=_datetime_literal_shape()),
    Case("date_time_literals_parsed-date", expr=pl.lit(date(2023, 1, 1)), shape=Lit(date(2023, 1, 1))),
    Case("date_time_literals_parsed-time", expr=pl.lit(time(12, 30, 45)), shape=Lit(time(12, 30, 45))),
    Case(
        "complex_literals-dict", expr=pl.lit({"a": 1, "b": 2}), shape=Lit({"a": 1, "b": 2}) if polars_has("struct_literal_parsed") else N(ErrorNode)
    ),
    Case("complex_literals-nested_list", expr=pl.lit([[1, 2], [3, 4]]), shape=Lit([[1, 2], [3, 4]])),
    Case("value_property_on_different_node_types-column", expr=c("x"), shape=N(ColumnNode, value=None, can_extract_literal=False)),
    Case(
        "value_property_on_different_node_types-function",
        expr=c("x").str.to_uppercase(),
        shape=N(FunctionNode, value=None, can_extract_literal=False),
    ),
    Case(
        "value_property_on_different_node_types-cast_literal",
        expr=pl.lit(42).cast(pl.Float64),
        shape=N(CastNode, value=42.0, can_extract_literal=True),
    ),
    # binary operators
    Case("parse_binary_operation-eq", expr=c("a") == c("b"), shape=Bin(Op.EQ, N(ColumnNode), N(ColumnNode))),
    Case("parse_binary_operation-plus", expr=c("a") + pl.lit(5), shape=Bin(Op.PLUS, N(ColumnNode), N(LiteralNode))),
    Case("binary_expr_node_contents", expr=c("a") + c("b"), shape=Bin(Op.PLUS, Col("a"), Col("b"))),
    *[
        Case(f"parse_all_comparison_operators-{op.name.lower()}", expr=fn(c("a"), pl.lit(10)), shape=N(BinaryExprNode, op=op))
        for fn, op in [
            (lambda a, b: a == b, Op.EQ),
            (lambda a, b: a != b, Op.NOT_EQ),
            (lambda a, b: a < b, Op.LT),
            (lambda a, b: a <= b, Op.LT_EQ),
            (lambda a, b: a > b, Op.GT),
            (lambda a, b: a >= b, Op.GT_EQ),
        ]
    ],
    *[
        Case(f"parse_all_arithmetic_operators-{op.name.lower()}", expr=fn(c("a"), pl.lit(10)), shape=N(BinaryExprNode, op=op))
        for fn, op in [
            (lambda a, b: a + b, Op.PLUS),
            (lambda a, b: a - b, Op.MINUS),
            (lambda a, b: a * b, Op.MULTIPLY),
            (lambda a, b: a / b, Op.TRUE_DIVIDE),
        ]
    ],
    Case("parse_all_logical_operators-and", expr=(c("a") > 5) & (c("b") < 10), shape=N(BinaryExprNode, op=Op.AND)),
    Case("parse_all_logical_operators-or", expr=(c("a") > 5) | (c("b") < 10), shape=N(BinaryExprNode, op=Op.OR)),
    Case(
        "nested_alias",
        expr=c("a").alias("b").alias("c"),
        shape=N(AliasNode, name="c", input=N(AliasNode, name="b", input=Col("a"))),
        marks=(pytest.mark.skipif(polars_has("nested_alias_dropped"), reason="meta.pop() drops a nested alias"),),
    ),
    # functions
    Case("parse_string_function-contains", expr=c("text").str.contains("pattern"), shape=Fn(Str.CONTAINS, Col("text"), Lit("pattern"))),
    Case("parse_string_function-uppercase", expr=c("text").str.to_uppercase(), shape=Fn(Str.UPPERCASE, Col("text"))),
    Case("parse_not", expr=~c("flag"), shape=N(FunctionNode, inputs=[Col("flag")])),
    Case("parse_boolean_function-is_null", expr=c("value").is_null(), shape=Fn(Bool.IS_NULL, Col("value"))),
    Case("parse_boolean_function-is_not_null", expr=c("value").is_not_null(), shape=Fn(Bool.IS_NOT_NULL, Col("value"))),
    Case("parse_is_in_function-list", expr=c("value").is_in([1, 2, 3]), shape=Fn(Bool.IS_IN, Col("value"), Lit([1, 2, 3]))),
    Case("parse_is_in_function-column", expr=c("value").is_in(c("other_list")), shape=Fn(Bool.IS_IN, Col("value"), Col("other_list"))),
    Case("parse_is_between-default", expr=c("value").is_between(1, 10), shape=Fn(Bool.IS_BETWEEN, *_IS_BETWEEN_INPUTS, options={"closed": "Both"})),
    *[
        Case(
            f"parse_is_between-{closed}",
            expr=c("value").is_between(1, 10, closed=closed),
            shape=Fn(Bool.IS_BETWEEN, *_IS_BETWEEN_INPUTS, options={"closed": closed.capitalize()}),
        )
        for closed in ("both", "left", "right", "none")
    ],
    *[
        Case(f"parse_cast-{dtype.__name__}", expr=c("value").cast(dtype), shape=N(CastNode))
        for dtype in [pl.Int32, pl.Float64, pl.Utf8, pl.Boolean, pl.Date, pl.Datetime]
    ],
    *[
        Case(f"parse_datetime_functions-{name}", expr=expr, shape=N(FunctionNode, function_type=kind_of(TemporalFunctionType)))
        for name, expr in [
            ("year", c("date").dt.year()),
            ("month", c("date").dt.month()),
            ("day", c("date").dt.day()),
            ("hour", c("date").dt.hour()),
            ("minute", c("date").dt.minute()),
            ("second", c("date").dt.second()),
            ("strftime", c("date").dt.strftime("%Y-%m-%d")),
        ]
    ],
    *[
        Case(f"parse_string_manipulation-{name}", expr=expr, shape=N(FunctionNode, function_type=kind_of(StringFunctionType)))
        for name, expr in [
            ("to_lowercase", c("text").str.to_lowercase()),
            ("starts_with", c("text").str.starts_with("prefix")),
            ("ends_with", c("text").str.ends_with("suffix")),
            ("replace", c("text").str.replace("old", "new")),
            ("split", c("text").str.split("delimiter")),
            ("join", c("text").str.join("other")),
        ]
    ],
    Case("unknown_expressions", expr=pl.coalesce(c("a"), c("b"), c("c")), shape=N(FunctionNode, function_type=GenericFunctionType.UNKNOWN)),
    Case(
        "ternary_operations",
        expr=pl.when(c("a") > 0).then(1).otherwise(0),
        shape=N(TernaryNode, predicate=N(BinaryExprNode, op=Op.GT), truthy=Lit(1), falsy=Lit(0)),
    ),
    Case(
        "fill_null_function_parsing",
        expr=c("nullable").fill_null(42) > 50,
        shape=Bin(Op.GT, Fn(GenericFunctionType.FILL_NULL, Col("nullable"), Lit(42)), Lit(50)),
    ),
    Case(
        "fill_null_with_expression",
        expr=c("nullable").fill_null(pl.when(c("id") > 5).then(100).otherwise(0)) > 50,
        shape=N(
            BinaryExprNode, op=Op.GT, left=Fn(GenericFunctionType.FILL_NULL, Col("nullable"), N(TernaryNode, predicate=N(BinaryExprNode, op=Op.GT)))
        ),
    ),
    Case(
        "parse_complex_expression",
        expr=(c("age") >= 18) & c("country").is_in(["USA", "Canada", "Mexico"]),
        shape=Bin(
            Op.AND,
            Bin(Op.GT_EQ, Col("age"), Lit(18)),
            Fn(Bool.IS_IN, Col("country"), Lit(["USA", "Canada", "Mexico"]), options={"nulls_equal": False}),
        ),
    ),
    Case(
        "parse_complex_nested_expressions",
        expr=pl.when((c("a") > 10) & ((c("b") < 5) | c("c").is_in([1, 2, 3])))
        .then(c("d") * 2 + c("e").cast(pl.Float64))
        .otherwise(c("f").str.to_uppercase().str.contains("X") | c("g").is_null()),
        shape=N(
            TernaryNode,
            predicate=Bin(Op.AND, Bin(Op.GT, Col("a"), Lit(10)), Bin(Op.OR, Bin(Op.LT, Col("b"), Lit(5)), Fn(Bool.IS_IN, Col("c"), Lit([1, 2, 3])))),
            truthy=Bin(Op.PLUS, Bin(Op.MULTIPLY, Col("d"), Lit(2)), N(CastNode, input=Col("e"), dtype=pl.Float64)),
            falsy=Bin(Op.OR, Fn(Str.CONTAINS, Fn(Str.UPPERCASE, Col("f")), Lit("X")), Fn(Bool.IS_NULL, Col("g"))),
        ),
    ),
]


@pytest.mark.parametrize(("expr", "shape"), params(PARSE_CASES, "expr", "shape"))
def test_parse(expr, shape):
    shape.match(get_parsed_expr(expr))


@pytest.mark.parametrize(("start", "first", "count"), [(pl.lit(0), 0, 50), (c("val"), 1, 10)], ids=["large_expression", "parse_nested_expressions"])
def test_parse_long_addition_chain(start, first, count):
    """``start + lit(first) + ... + lit(first + count - 1)`` parses into a left-deep chain with each literal on the right."""
    expr = start
    for i in range(first, first + count):
        expr = expr + pl.lit(i)
    node = get_parsed_expr(expr)
    for i in reversed(range(first, first + count)):
        Bin(Op.PLUS, N(BaseExprNode), Lit(i)).match(node)
        node = node.left
    (Lit(0) if first == 0 else Col("val")).match(node)


def test_list_eval():
    expr = c("list_column").list.eval(pl.element() > 0)
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
    visitor.visit(get_parsed_expr((c("a") == 5) & (c("b") > 10)))
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
    visitor.visit(get_parsed_expr((c("a") > 5) & c("b").is_in(["x", "y", "z"])))
    assert visitor.process_results() == {"BinaryExprNode": 2, "ColumnNode": 2, "LiteralNode": 2, "FunctionNode": 1}


def test_node_traverser_parser_isolates_subtree_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure in one ``_parse_<kind>`` collapses only that subtree to ``ErrorNode``; the rest still parses."""

    def boom(self: NodeTraverserParser, expr: pl.Expr, obj: object, visitor: object) -> BaseExprNode:
        raise RuntimeError("simulated subtree failure")

    monkeypatch.setattr(NodeTraverserParser, "_parse_function_expr", boom)
    node = NodeTraverserParser().parse((c("x") > 5) & c("y").is_null())
    N(BinaryExprNode, left=N(BinaryExprNode), right=N(ErrorNode, error=lambda e: "simulated subtree failure" in e)).match(node)


def test_node_traverser_parser_unsupported_expression_degrades_to_unknown() -> None:
    """A Python UDF, which Polars' typed view refuses, degrades to ``UnknownNode`` rather than raising."""
    assert isinstance(NodeTraverserParser().parse(c("x").map_batches(lambda s: s, return_dtype=pl.Boolean)), UnknownNode)


@pytest.mark.parametrize(
    "expr",
    [c("x").is_null().alias("p"), c("x").is_null().name.keep(), (~c("x").is_null()).alias("p").alias("q")],
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


@pytest.mark.parametrize(
    ("kwargs", "options"),
    [({"strict": True}, "Strict"), ({"strict": False}, "NonStrict"), ({"wrap_numerical": True}, "Overflowing")],
    ids=["strict", "non_strict", "wrap_numerical"],
)
def test_cast_options_parsed(kwargs, options):
    N(CastNode, options=options).match(get_parsed_expr(c("x").cast(pl.Int8, **kwargs)))


@pytest.mark.parametrize(
    ("value", "collection", "expected"),
    [
        pytest.param(pl.lit(1), False, True, id="int"),
        pytest.param(pl.lit([1, 2]), False, False, id="list_as_scalar"),
        pytest.param(pl.lit([1, 2]), True, True, id="list_as_collection"),
        pytest.param(pl.lit(pl.Series(["a", "b"])), False, False, id="series"),
        pytest.param(pl.lit(float("nan")), False, False, id="nan"),
        pytest.param(pl.lit([float("nan"), 1.0]), True, False, id="nan_in_collection"),
        # Valid in Polars, but outside Python's date range.
        pytest.param(pl.lit(4_000_000).cast(pl.Date), False, False, id="date_past_python_max"),
        pytest.param(pl.lit(10**18).cast(pl.Datetime("us")), False, False, id="datetime_past_python_max"),
    ],
)
def test_has_exact_value(value, collection, expected):
    node = get_parsed_expr(c("x") == value).right
    assert has_exact_value(node, collection=collection) is expected


# --- get_literal_value ------------------------------------------------------------------------------------------------

_NOW, _TODAY = datetime.now(), date.today()
_STRUCT = {"name": "test", "values": [1, 2, 3], "metadata": {"active": True}}

LITERAL_CASES = [
    Case("basic_literals-int", expr=pl.lit(5), expect=5),
    Case("basic_literals-string", expr=pl.lit("test"), expect="test"),
    Case("basic_literals-true", expr=pl.lit(True), expect=True),
    Case("basic_literals-none", expr=pl.lit(None), expect=None),
    Case("boolean_literals-false", expr=pl.lit(False), expect=False),
    Case("list_literals-list", expr=pl.lit([1, 2, 3]), expect=[1, 2, 3]),
    Case("list_literals-tuple", expr=pl.lit((4, 5, 6)), expect=lambda v: v in ((4, 5, 6), [4, 5, 6])),
    Case("non_literal_expressions", expr=c("non_existent"), expect=lambda v: v is FAILED_LITERAL_RESULT),
    Case("numeric_literals-int", expr=pl.lit(42), expect=42),
    Case("numeric_literals-float", expr=pl.lit(3.14), expect=3.14),
    Case("numeric_literals-negative", expr=pl.lit(-10), expect=-10),
    Case("numeric_literals-zero", expr=pl.lit(0), expect=0),
    Case("numeric_literals-inf", expr=pl.lit(float("inf")), expect=float("inf")),
    Case("numeric_literals-nan", expr=pl.lit(float("nan")), expect=math.isnan),
    Case("string_literals_special_cases-empty", expr=pl.lit(""), expect=""),
    Case("string_literals_special_cases-space", expr=pl.lit(" "), expect=" "),
    Case("string_literals_special_cases-punctuation", expr=pl.lit("Special chars: !@#$%^&*()"), expect="Special chars: !@#$%^&*()"),
    Case("string_literals_special_cases-newline", expr=pl.lit("Line1\nLine2"), expect="Line1\nLine2"),
    Case("date_time_literals-date", expr=pl.lit(_TODAY), expect=_TODAY),
    Case("date_time_literals-datetime", expr=pl.lit(_NOW), expect=_NOW),
    Case("get_literal_value-datetime", expr=pl.lit(datetime(2023, 1, 1)), expect=datetime(2023, 1, 1)),
    Case("date_time_literals-time", expr=pl.lit(time(12, 30, 45)), expect=time(12, 30, 45)),
    Case("date_time_literals-timedelta", expr=pl.lit(timedelta(days=1, hours=2)), expect=timedelta(days=1, hours=2)),
    Case("complex_nested_structures-dict", expr=pl.lit({"a": 1, "b": 2}), expect={"a": 1, "b": 2}),
    Case("complex_nested_structures-nested_list", expr=pl.lit([[1, 2], [3, 4]]), expect=[[1, 2], [3, 4]]),
    Case("complex_nested_structures-mixed", expr=pl.lit(_STRUCT), expect=_STRUCT),
    Case("edge_cases-large_int", expr=pl.lit(10**20), expect=10**20),
    Case("edge_cases-long_string", expr=pl.lit("a" * 10000), expect="a" * 10000),
    Case("compound_expressions-sum", expr=pl.lit(5) + pl.lit(3), expect=8),
    Case("compound_expressions-function", expr=pl.lit("test").str.to_uppercase(), expect="TEST"),
    Case("binary_data", expr=pl.lit(b"binary\x00data"), expect=b"binary\x00data"),
]


@pytest.mark.parametrize(("expr", "expect"), params(LITERAL_CASES, "expr", "expect"))
def test_get_literal_value(expr, expect):
    _match(get_literal_value(expr), expect, "value")


# --- column extraction ------------------------------------------------------------------------------------------------

COLUMN_CASES = [
    Case("extract_column_name-cast_int", expr=c("test_column1").cast(pl.Int32), expect="test_column1", fn=extract_column_name, schema=None),
    Case("extract_column_name-cast_datetime", expr=c("test_column2").cast(pl.Datetime), expect="test_column2", fn=extract_column_name, schema=None),
    Case(
        "extract_column_name-cast_datetime_ns",
        expr=c("test_column3").cast(pl.Datetime("ns")),
        expect="test_column3",
        fn=extract_column_name,
        schema=None,
    ),
    Case("extract_column_name-struct", expr=pl.struct(c("a"), c("b")), expect=None, fn=extract_column_name, schema=None),
    Case("extract_column_name-arithmetic", expr=c("b").mul(2).cast(pl.Float64), expect=None, fn=extract_column_name, schema=None),
    Case("extract_column_name-alias", expr=c("b").alias("big_b"), expect="b", fn=extract_column_name, schema=None),
    Case(
        "extract_column_name-cast_and_aliases",
        expr=c("b").cast(pl.Int64).alias("first").alias("second"),
        expect="b",
        fn=extract_column_name,
        schema=None,
    ),
    # ``dt.date()`` floors to day granularity, order-preserving like ``cast(pl.Date)``.
    Case("extract_column_name-dt_date", expr=c("ts").dt.date(), expect="ts", fn=extract_column_name, schema=None),
    Case("extract_column_name-dt_date_alias", expr=c("ts").dt.date().alias("d"), expect="ts", fn=extract_column_name, schema=None),
    # The strict extractor unwraps a cast only when the schema proves it value-preserving.
    Case("extract_value_preserving_column_name-alias", expr=c("b").alias("big_b"), expect="b", fn=extract_value_preserving_column_name, schema=None),
    Case(
        "extract_value_preserving_column_name-cast_no_schema",
        expr=c("a").cast(pl.Int32),
        expect=None,
        fn=extract_value_preserving_column_name,
        schema=None,
    ),
    Case(
        "extract_value_preserving_column_name-narrowing",
        expr=c("a").cast(pl.Int8, wrap_numerical=True),
        expect=None,
        fn=extract_value_preserving_column_name,
        schema={"a": pl.Int64},
    ),
    Case(
        "extract_value_preserving_column_name-widening",
        expr=c("a").cast(pl.Int64),
        expect="a",
        fn=extract_value_preserving_column_name,
        schema={"a": pl.Int8},
    ),
    Case(
        "extract_value_preserving_column_name-decimal_widening",
        expr=c("x").cast(pl.Decimal(38, 2)),
        expect="x",
        fn=extract_value_preserving_column_name,
        schema={"x": pl.Decimal(10, 2)},
    ),
]


@pytest.mark.parametrize(("fn", "expr", "schema", "expect"), params(COLUMN_CASES, "fn", "expr", "schema", "expect"))
def test_extract_column(fn, expr, schema, expect):
    node = get_parsed_expr(expr)
    assert (fn(node) if schema is None else fn(node, schema)) == expect


@pytest.mark.parametrize(
    ("source", "target", "expect"),
    [
        pytest.param(pl.Int64, pl.Int64, True, id="identity"),
        pytest.param(pl.Int8, pl.Int64, True, id="int_widening"),
        pytest.param(pl.Int64, pl.Int8, False, id="int_narrowing"),
        pytest.param(pl.UInt8, pl.Int16, True, id="unsigned_to_wider_signed"),
        pytest.param(pl.Int8, pl.UInt64, False, id="signed_to_unsigned"),
        pytest.param(pl.Decimal(10, 2), pl.Decimal(38, 2), True, id="decimal_widening"),
        pytest.param(pl.Decimal(10, 2), pl.Decimal(38, 4), False, id="decimal_rescaling"),
        pytest.param(pl.Int64, pl.Float64, False, id="cross_family"),
        pytest.param(pl.Datetime, pl.Datetime("ms"), False, id="unparameterized"),
    ],
)
def test_is_value_order_preserving_cast(source, target, expect):
    assert is_value_order_preserving_cast(source, target) is expect


# --- predicate pushdown through an IO source --------------------------------------------------------------------------


def P(id, expr, pushed=True, results=False, check=None, df=None, marks=()):
    """``pushed``: whether the source receives a predicate (``None``: not checked); ``results``: also compare a plain
    filtered read; ``check(original, pushed)``: inspect the pushed predicate; ``df``: the source frame, else ``tester``.
    """
    return Case(id, expr=expr, pushed=pushed, results=results, check=check, df=df, marks=marks)


def _pushed_on(name: str):
    """Pushed from the Polars version with behavior ``name``; unchecked before it."""
    return True if polars_has(name) else None


def _pushed_is_and_of_negations(original, pushed):
    """Polars applies De Morgan before pushing: ``~(a | b)`` arrives as an AND; the original stays a NOT of an OR."""
    N(BinaryExprNode, op=Op.AND).match(get_parsed_expr(pushed))
    N(FunctionNode, function_type=Bool.NOT, inputs=[N(BinaryExprNode, op=Op.OR)]).match(get_parsed_expr(original))


def _pushed_fill_null(original, pushed):
    N(BinaryExprNode, left=N(FunctionNode, function_type=GenericFunctionType.FILL_NULL)).match(get_parsed_expr(pushed))


_DT_18, _DT_END = datetime(2023, 1, 1, 18), datetime(2023, 1, 1, 23, 59, 59)
_NS_FRAME = pl.DataFrame(
    {"timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)], "value": [10, 20, 30]},
    schema={"timestamp": pl.Datetime("ns"), "value": pl.Int64},
)
_PADDED = pl.DataFrame({"padded": ["  A  ", "  B  ", "  C  ", " D ", "E"]})
_NAMES = pl.DataFrame({"first": ["A", "B", "C"], "last": ["X", "Y", "Z"]})
_needs_implicit_casts = pytest.mark.skipif(not polars_has("pushes_implicit_casts"), reason="implicit casts block pushdown before Polars 1.28")

PUSHDOWN_CASES = [
    P("numeric_predicates-gt", c("id") > 5),
    P("numeric_predicates-le_float", c("float_val") <= 3.0),
    P("numeric_predicates-and", (c("id") > 3) & (c("int_val") < 70)),
    P("numeric_predicates-arithmetic", c("float_val") * 2 > 5),
    P("string_predicates-eq", c("string_val") == "A"),
    P("string_predicates-contains", c("long_string").str.contains("text_"), pushed=_pushed_on("pushes_function_predicates"), results=True),
    P("string_predicates-is_in", c("string_val").is_in(["A", "B"])),
    # str.strip_chars / strip_chars_start / strip_chars_end translate to TRIM / LTRIM / RTRIM
    P("strip_chars_predicate_pushed_down-both", c("padded").str.strip_chars() == "A", results=True, df=_PADDED),
    P("strip_chars_predicate_pushed_down-start", c("padded").str.strip_chars_start() == "A  ", results=True, df=_PADDED),
    P("strip_chars_predicate_pushed_down-end", c("padded").str.strip_chars_end() == "  A", results=True, df=_PADDED),
    P("concat_horizontal_predicate_pushed_down", pl.concat_str([c("first"), pl.lit("-"), c("last")]) == "A-X", results=True, df=_NAMES),
    *[P(f"fill_null_predicates-{v}", c("nullable").fill_null(v) > 5) for v in (0, 1, 9)],
    P("same_dtype_column_comparison", c("id") < c("int_val")),
    P("struct_field_pushed_down", c("struct").struct.field("value") > 5),
    P(
        "fill_ternary_predicates",
        pl.when(c("nullable") > 3, c("int_val") > 5)
        .then(c("int_val") > 2)
        .when(c("float_val") > 0)
        .then(c("string_val").is_in(["a", "b"]))
        .when(c("float_val") < 1)
        .then(pl.lit(True))
        .otherwise(c("float_val") < 5),
    ),
    P("temporal_predicates-date_gt", c("date_val") > date(2023, 1, 5)),
    P("temporal_predicates-python_datetime_lt", c("datetime_val") < _DT_18, pushed=_pushed_on("pushes_python_datetime_comparison"), results=True),
    P("temporal_predicates-polars_datetime_lt", c("datetime_val") < convert_datetime_to_polars(_DT_18)),
    P(
        "temporal_predicates-polars_datetime_is_between",
        c("datetime_val").is_between(convert_datetime_to_polars(_DT_18), convert_datetime_to_polars(_DT_END)),
    ),
    P(
        "temporal_predicates-python_datetime_is_between",
        c("datetime_val").is_between(_DT_18, _DT_END),
        pushed=_pushed_on("pushes_python_datetime_comparison"),
    ),
    P("temporal_predicates-dt_day", c("date_val").dt.day() == 3, pushed=_pushed_on("pushes_function_predicates")),
    # a nanosecond column compared with microsecond Python datetimes
    P(
        "temporal_schema_edge_case-python_datetimes",
        c("timestamp").is_between(_DT_18, _DT_END),
        pushed=polars_has("pushes_implicit_casts"),
        df=_NS_FRAME,
    ),
    P(
        "temporal_schema_edge_case-polars_datetimes",
        c("timestamp").is_between(convert_datetime_to_polars(_DT_18), convert_datetime_to_polars(_DT_END)),
        pushed=_pushed_on("pushes_python_datetime_comparison"),
        df=_NS_FRAME,
    ),
    P("boolean_predicates-column", c("bool_val")),
    P("boolean_predicates-not", ~c("bool_val")),
    P("null_predicates-is_null", c("nullable").is_null()),
    P("null_predicates-is_not_null", c("nullable").is_not_null()),
    P("categorical_predicates", c("cat_val") == "A"),
    P(
        "complex_predicates-nested",
        (c("id") > 3) & ((c("string_val") == "A") | (c("float_val") > 3.0)) & ~c("bool_val") | c("cat_val").is_in(["A", "C"]),
    ),
    P("complex_predicates-mixed_types", (c("date_val") > date(2023, 1, 3)) & c("string_val").is_in(["A", "B"]) & (c("float_val") > 2.0)),
    P("predicates_with_implicit_casts-datetime_gt_date", c("datetime_val") > date(2023, 1, 2), marks=(_needs_implicit_casts,)),
    P(
        "predicates_with_implicit_casts-datetime_is_between_dates",
        c("datetime_val").is_between(date(2023, 1, 2), date(2023, 1, 3)),
        marks=(_needs_implicit_casts,),
    ),
    P("predicates_with_implicit_casts-int_eq_float", c("int_val") == c("float_val"), marks=(_needs_implicit_casts,)),
    P("predicates_with_implicit_casts-date_gt_datetime", c("date_val") > datetime(2023, 1, 2, 12), marks=(_needs_implicit_casts,)),
    # a redundant Int -> Int64 cast succeeds in both strict modes
    *[P(f"predicate_with_explicit_cast_on_column-strict={s}", c("int_val").cast(pl.Int64, strict=s) > 30) for s in (True, False)],
    *[
        P(
            f"compare_float_gt_int_with_int_to_float_cast_pushed_down-strict={s}",
            c("float_val") > c("int_val").cast(pl.Float64, strict=s),
            pushed=polars_has("pushes_implicit_casts"),
        )
        for s in (True, False)
    ],
    *[
        P(
            f"compare_float_gt_int_with_float_to_int_cast_pushed_down-strict={s}",
            c("float_val").cast(pl.Int64, strict=s) > c("int_val"),
            pushed=polars_has("pushes_implicit_casts"),
        )
        for s in (True, False)
    ],
    P("unsimplified_not_predicates-de_morgan", ~((c("string_val") == "A") | (c("id") > 3)), check=_pushed_is_and_of_negations),
    P("unsimplified_not_predicates-double_nested", ~((c("string_val") == "A") | (~(c("date_val") == date(2023, 1, 3)) & (c("float_val") > 2.0)))),
    P(
        "map_batches",
        c("int_val").map_batches(lambda x: x > 5, return_dtype=pl.Boolean, is_elementwise=True),
        check=lambda original, pushed: N(AnonymousFunctionNode).match(get_parsed_expr(pushed)),
        marks=(pytest.mark.xfail(polars_has("anonymous_function_not_pushed"), reason="not pushed down properly from Polars 1.32.0", run=False),),
    ),
    P(
        "fill_null_constant",
        c("nullable").fill_null(pl.when(c("id") > 5).then(100).otherwise(0)) > 50,
        pushed=polars_has("pushes_fill_null_predicates"),
        check=_pushed_fill_null if polars_has("pushes_fill_null_predicates") else None,
    ),
    # results match whether or not the predicate is pushed
    P("all_results_match_regardless_of_pushdown-id_gt", c("id") > 5, results=True),
    P("all_results_match_regardless_of_pushdown-string_eq", c("string_val") == "A", results=True),
    *[
        P(
            f"all_results_match_regardless_of_pushdown-{name}",
            expr,
            pushed=polars_has("pushes_function_predicates"),
            results=polars_has("pushes_function_predicates"),
        )
        for name, expr in [
            ("sin", c("float_val").sin() > 0),
            ("to_uppercase", c("string_val").str.to_uppercase() == "A"),
            ("strftime", c("date_val").dt.strftime("%Y-%m-%d") == "2023-01-01"),
        ]
    ],
    # window functions, aggregations and non-elementwise functions are not pushed
    *[
        P(f"expected_not_pushed_down-{name}", expr, pushed=False)
        for name, expr in [
            ("window_mean", c("float_val") > c("float_val").mean().over("string_val")),
            ("window_zscore", (c("float_val") - c("float_val").mean().over("string_val")) / c("float_val").std().over("string_val") > 1.0),
            ("window_diff", c("float_val").diff().over([c("string_val")]) > 0.5),
            ("aggregate_mean", c("int_val") > c("int_val").mean()),
            ("aggregate_mean_and_median", (c("float_val") > c("float_val").mean()) & (c("int_val") < c("int_val").median())),
            ("aggregate_min_max", (c("float_val") - c("float_val").min()) / (c("float_val").max() - c("float_val").min()) > 0.5),
            ("aggregate_and_window", (c("float_val") > c("float_val").mean()) & (c("float_val") > c("float_val").mean().over("string_val"))),
            ("window_count_and_null_count", (c("id").count().over("string_val") > 1) & (c("nullable").null_count() < pl.len()).over("string_val")),
            ("window_mean_plus_std", c("float_val") > (c("float_val").mean().over("string_val") + c("float_val").std().over("string_val"))),
            ("fill_null_window_mean", c("nullable").fill_null(c("nullable").mean().over("string_val")) > c("nullable").fill_null(0).mean()),
            (
                "ternary_window_and_median",
                pl.when(c("string_val") == "A")
                .then(c("float_val") > c("float_val").mean().over("string_val"))
                .otherwise(c("int_val") > c("int_val").median()),
            ),
            ("window_first", c("float_val") > c("float_val").first().over("string_val")),
            ("fill_null_strategy", c("nullable").fill_null(strategy="min") > 3),
            ("diff", c("float_val").diff() > 0.5),
            ("map_batches_not_elementwise", c("int_val").map_batches(lambda x: x > 5, return_dtype=pl.Boolean, is_elementwise=False)),
        ]
    ],
]


@pytest.mark.parametrize(("expr", "pushed", "results", "check", "df"), params(PUSHDOWN_CASES, "expr", "pushed", "results", "check", "df"))
def test_pushdown(request, expr, pushed, results, check, df):
    tracker = request.getfixturevalue("tester") if df is None else PredicateTracker(df)
    if pushed is not None:
        tracker.assert_predicate_pushed_down(expr, check, expected_pushed_down=pushed)
    if results:
        tracker.assert_results_match(expr)


def test_column_selection_with_predicates(tester):
    """A projection and a predicate are both pushed."""
    columns = ["id", "string_val", "date_val"]
    tester.reset()
    result = tester.lazy_frame.filter(c("id") > 5).select(columns).collect()
    assert_frame_equal(result, tester.df.filter(c("id") > 5).select(columns))
    assert tester.last_predicate is not None, "Predicate was not pushed down"
    assert tester.last_with_columns is not None and set(tester.last_with_columns) == set(columns)


@pytest.mark.parametrize("limit_method", ["head", "limit"])
@pytest.mark.parametrize("with_filter", [True, False], ids=["limit_not_pushed_when_filter_present", "limit_pushed_when_no_filter"])
def test_limit_pushdown(with_filter, limit_method):
    """Polars passes ``n_rows`` to a source only without a filter; with one, a source that cannot apply the whole
    filter would otherwise return too few rows."""
    received = {}

    def tracking_source(with_columns, predicate, n_rows, batch_size):
        received.update(predicate=predicate, n_rows=n_rows)
        yield pl.DataFrame({"x": [1, 2, 3, 4, 5]})

    lf = register_io_source(tracking_source, schema={"x": pl.Int64})
    if with_filter:
        lf = lf.filter(c("x") > 2)
    getattr(lf, limit_method)(3).collect()
    assert (received["predicate"] is not None) is with_filter
    assert received["n_rows"] == (None if with_filter else 3)
