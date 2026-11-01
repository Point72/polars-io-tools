"""convert_expr_to_valid_values: a table of (predicate, column, schema) -> the set of values the column can hold.

``None`` means the column cannot be restricted to a finite set of values.
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from polars_io_tools.io_sources.base import BinaryExprNode, get_parsed_expr
from polars_io_tools.io_sources.enum import OperatorType
from polars_io_tools.io_sources.set_visitor import SetVisitor, convert_expr_to_valid_values
from polars_io_tools.tests.helpers.cases import Case, params

s, p, v, x = pl.col("symbol"), pl.col("price"), pl.col("volume"), pl.col("x")
B = pl.Boolean


def C(id, pred, expect, col="symbol", schema=None):
    return Case(id, pred, expect, {"col": col, "schema": schema})


CASES = [
    # equality and is_in
    C("basic_equality", s == "US", {"US"}),
    C("basic_in_operator", s.is_in(["US", "EU", "JP"]), {"US", "EU", "JP"}),
    C("in_with_single_element_list", s.is_in(["US"]), {"US"}),
    C("in_with_empty_list", s.is_in([]), set()),
    C("mixed_data_types", (pl.col("value") == 10) | (pl.col("value") == "abc"), {10, "abc"}, col="value"),
    # DNF shapes
    C("multiple_equality_across_conjunctions", ((s == "US") & (p > 100)) | ((s == "EU") & (v > 1000)), {"US", "EU"}),
    C(
        "multiple_in_across_conjunctions",
        (s.is_in(["US", "EU", "JP"]) & (p > 100)) | (s.is_in(["EU", "UK", "CN"]) & (v > 1000)),
        {"US", "EU", "JP", "UK", "CN"},
    ),
    C("combination_of_equal_and_in", ((s == "US") & (p > 100)) | (s.is_in(["EU", "UK", "CN"]) & (v > 1000)), {"US", "EU", "UK", "CN"}),
    C("single_conjunction_without_column", (p > 100) & (v > 1000), None),
    C("multiple_conjunctions_one_without_column", ((s == "US") & (p > 100)) | ((p > 200) & (v > 1000)), None),
    C("contradictions_within_conjunction", (s == "US") & (s == "EU"), set()),
    C("complex_in_and_equal_combinations", (s.is_in(["US", "EU"]) & (s == "US")) | (s.is_in(["JP", "CN"]) & (p > 100)), {"US", "JP", "CN"}),
    C(
        "multiple_complex_rules",
        (s.is_in(["US", "EU", "JP"]) & (p > 100)) | ((s == "UK") & (v > 1000)) | (s.is_in(["CN", "HK"]) & s.is_in(["HK", "TW"])),
        {"US", "EU", "JP", "UK", "HK"},
    ),
    C("contradiction_in_one_conjunction_but_not_others", ((s == "US") & (s == "EU")) | (s == "JP"), {"JP"}),
    C("all_conjunctions_have_contradictions", ((s == "US") & (s == "EU")) | (s.is_in([]) & (p > 100)), set()),
    C("multiple_in_operators_within_single_conjunction", s.is_in(["US", "EU", "JP"]) & s.is_in(["JP", "CN", "UK"]), {"JP"}),
    C("not_equal_operator", (s != "US") & s.is_in(["US", "EU", "JP"]), {"EU", "JP"}),
    C("not_in_operator", ~s.is_in(["US", "EU"]) & s.is_in(["US", "EU", "JP", "CN"]), {"JP", "CN"}),
    C("different_operators_across_conjunctions", (s == "US") | (s == "AP") | s.is_in(["JP", "CN"]), {"US", "AP", "JP", "CN"}),
    # Boolean casts are transparent, other casts are not
    C("basic_boolean_cast", (s == "US").cast(B), {"US"}),
    C("boolean_cast_in_and_expression", (s == "US").cast(B) & (s != "JP"), {"US"}),
    C("boolean_cast_in_or_expression", (s == "US").cast(B) | (s == "JP"), {"US", "JP"}),
    C("multiple_boolean_casts", (s == "US").cast(B) | (s == "JP").cast(B), {"US", "JP"}),
    C("boolean_cast_with_in", s.is_in(["US", "JP"]).cast(B), {"US", "JP"}),
    C("nested_boolean_casts", (s == "US").cast(B).cast(B), {"US"}),
    C("boolean_cast_of_complex_expression", (((s == "US") & (p > 100)) | ((s == "JP") & (p > 200))).cast(B), {"US", "JP"}),
    C(
        "complex_nested_expressions_with_boolean_casts",
        ((s == "US") & (p > 100)).cast(B) | (s.is_in(["JP", "EU"]) & (v > 1000)).cast(B),
        {"US", "JP", "EU"},
    ),
    C("boolean_cast_with_contradictions", ((s == "US") & (s == "JP")).cast(B), set()),
    C("boolean_cast_with_mixed_data_types", ((pl.col("value") == 10) | (pl.col("value") == "abc")).cast(B), {10, "abc"}, col="value"),
    C("non_boolean_cast", (s == "US").cast(pl.Utf8), None),
    C("boolean_cast_with_not", (~(s == "US")).cast(B), None),
    # aliases are transparent
    C("alias_simple_equality", s.alias("symbol_alias") == "US", {"US"}),
    C("alias_in_operator", s.alias("symbol_alias").is_in(["US", "EU", "JP"]), {"US", "EU", "JP"}),
    C("alias_logical_operations", (s.alias("symbol_alias") == "US") & (p.alias("price_alias") > 100), {"US"}),
    C("alias_with_exclusions", (s.alias("symbol_alias") != "US") & s.is_in(["US", "EU", "JP"]), {"EU", "JP"}),
    C(
        "alias_complex_mixed_operations",
        ((s.alias("s1") == "US") | s.alias("s2").is_in(["EU", "JP", "UK"])) & (s.alias("s3") != "UK"),
        {"US", "EU", "JP"},
    ),
    # a cast is unwrapped only when the schema proves it value-preserving
    C("narrowing_cast_on_column_is_not_stripped-no_schema", pl.col("a").cast(pl.Int8, wrap_numerical=True) == 1, None, col="a"),
    C("narrowing_cast_on_column_is_not_stripped-schema", pl.col("a").cast(pl.Int8, wrap_numerical=True) == 1, None, col="a", schema={"a": pl.Int64}),
    C("value_preserving_widening_cast_still_extracts_values-no_schema", pl.col("a").cast(pl.Int64) == 1, None, col="a"),
    C("value_preserving_widening_cast_still_extracts_values-schema", pl.col("a").cast(pl.Int64) == 1, {1}, col="a", schema={"a": pl.Int8}),
    # NOT must not exclude values when the inner AND ignored a side it could not convert
    C("not_over_relaxed_and_does_not_exclude_values-cast", x.is_in([1, 2]) & ~((x == 1) & (x.cast(pl.Int64) == 2)), {1, 2}, col="x"),
    C("not_over_relaxed_and_does_not_exclude_values-other_col", x.is_in([1, 2]) & ~((x == 1) & (pl.col("y") == 2)), {1, 2}, col="x"),
    C("not_over_relaxed_and_does_not_exclude_values-exact", x.is_in([1, 2]) & ~(x == 1), {2}, col="x"),
    C(
        "sub_microsecond_literal_is_not_pushed",
        pl.col("t") == pl.lit(1704198600000000500).cast(pl.Datetime("ns")),
        None,
        col="t",
        schema={"t": pl.Datetime("ns")},
    ),
]


@pytest.mark.parametrize(("pred", "col", "schema", "expect"), params(CASES, "pred", "col", "schema", "expect"))
def test_valid_values(pred, col, schema, expect):
    assert convert_expr_to_valid_values(pred, col, schema) == expect


_EARLY = datetime(2024, 11, 3, 1, 30, tzinfo=ZoneInfo("America/New_York"))


@pytest.mark.parametrize(
    ("df", "pred"),
    [
        (pl.DataFrame({"x": [float("nan"), 1.0, 2.0]}), x.is_in([float("nan"), 1.0]) & x.is_in([float("nan"), 2.0])),
        (pl.DataFrame({"x": [_EARLY, _EARLY.replace(fold=1)]}), x.is_in([_EARLY, _EARLY.replace(fold=1)])),
        (pl.DataFrame({"x": pl.Series([0.1], dtype=pl.Float32)}), x == 0.1),
        (pl.DataFrame({"x": [date(2024, 1, 1)]}), (x == date(2024, 1, 1)) & (x == datetime(2024, 1, 1))),
    ],
    ids=["nan", "dst_fold", "float32", "date_vs_datetime"],
)
def test_valid_values_cover_matching_rows(df, pred):
    """Every kept row's value must be in the extracted set, both by Python and by Polars comparison."""
    kept = df.filter(pred)["x"]
    assert not kept.is_empty()
    values = convert_expr_to_valid_values(pred, "x", df.schema)
    if values is not None:
        assert all(v in values for v in kept.to_list())
        assert kept.is_in(pl.Series(list(values), dtype=df.schema["x"])).all()


def test_set_visitor_visitor_direct():
    visitor = SetVisitor("symbol")
    visitor.visit(get_parsed_expr(s.is_in(["US", "EU", "JP"])))
    assert visitor.process_results() == {"US", "EU", "JP"}


@pytest.mark.parametrize(
    ("col", "first", "second", "first_expect", "expect"),
    [
        pytest.param("symbol", s == "US", s == "EU", {"US"}, {"US", "EU"}, id="consistent_set_creation"),
        pytest.param(
            "nullable_col", pl.col("nullable_col").is_null(), pl.col("nullable_col") == "value", {None}, {None, "value"}, id="null_handling"
        ),
    ],
)
def test_set_visitor_creates_new_sets(col, first, second, first_expect, expect):
    """Visiting a second node replaces the inclusions set rather than mutating it in place."""
    visitor = SetVisitor(col)
    visitor.visit(get_parsed_expr(first))
    first_set = visitor.inclusions
    assert first_set == first_expect
    visitor.visit(get_parsed_expr(second))
    assert visitor.inclusions is not first_set
    assert visitor.inclusions == expect
    assert first_set == first_expect


def test_logical_and_operator_handling():
    """Regression: LOGICAL_AND, the spelling before Polars optimizes a plan, is handled like AND."""
    node = get_parsed_expr((pl.col("date_partition") == date(2009, 10, 2)) & (s == "AAPL US"))
    assert isinstance(node, BinaryExprNode)
    forced = BinaryExprNode(expr=node.expr, can_extract_literal=False, left=node.left, right=node.right, op=OperatorType.LOGICAL_AND)
    assert convert_expr_to_valid_values(forced, "symbol") == {"AAPL US"}
