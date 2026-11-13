"""convert_expr_to_valid_values: the set of values a filter allows in one column.

``None`` means the filter does not limit the column to a known list of values.
"""

from dataclasses import KW_ONLY, dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from polars_io_tools.io_sources.base import BinaryExprNode, get_parsed_expr
from polars_io_tools.io_sources.enum import OperatorType
from polars_io_tools.io_sources.set_visitor import SetVisitor, convert_expr_to_valid_values
from polars_io_tools.tests.helpers.cases import as_params, epoch_ns


@dataclass(frozen=True)
class ValidValuesCase:
    """A filter, and the set of values it allows in one column."""

    id: str
    _: KW_ONLY
    predicate: pl.Expr
    # None: the filter does not limit the column to a known list of values
    expected_values: set | None
    column: str = "symbol"
    # column dtypes passed along with the filter
    schema: dict | None = None


# Filters on ``symbol`` (and sometimes ``price`` and ``volume``), and the symbols each one allows.
CASES = [
    # equality and is_in
    ValidValuesCase("basic_equality", predicate=pl.col("symbol") == "US", expected_values={"US"}),
    ValidValuesCase("basic_in_operator", predicate=pl.col("symbol").is_in(["US", "EU", "JP"]), expected_values={"US", "EU", "JP"}),
    ValidValuesCase("in_with_single_element_list", predicate=pl.col("symbol").is_in(["US"]), expected_values={"US"}),
    ValidValuesCase("in_with_empty_list", predicate=pl.col("symbol").is_in([]), expected_values=set()),
    ValidValuesCase("mixed_data_types", predicate=(pl.col("value") == 10) | (pl.col("value") == "abc"), expected_values={10, "abc"}, column="value"),
    # DNF shapes
    ValidValuesCase(
        "multiple_equality_across_conjunctions",
        predicate=((pl.col("symbol") == "US") & (pl.col("price") > 100)) | ((pl.col("symbol") == "EU") & (pl.col("volume") > 1000)),
        expected_values={"US", "EU"},
    ),
    ValidValuesCase(
        "multiple_in_across_conjunctions",
        predicate=(pl.col("symbol").is_in(["US", "EU", "JP"]) & (pl.col("price") > 100))
        | (pl.col("symbol").is_in(["EU", "UK", "CN"]) & (pl.col("volume") > 1000)),
        expected_values={"US", "EU", "JP", "UK", "CN"},
    ),
    ValidValuesCase(
        "combination_of_equal_and_in",
        predicate=((pl.col("symbol") == "US") & (pl.col("price") > 100)) | (pl.col("symbol").is_in(["EU", "UK", "CN"]) & (pl.col("volume") > 1000)),
        expected_values={"US", "EU", "UK", "CN"},
    ),
    ValidValuesCase("single_conjunction_without_column", predicate=(pl.col("price") > 100) & (pl.col("volume") > 1000), expected_values=None),
    ValidValuesCase(
        "multiple_conjunctions_one_without_column",
        predicate=((pl.col("symbol") == "US") & (pl.col("price") > 100)) | ((pl.col("price") > 200) & (pl.col("volume") > 1000)),
        expected_values=None,
    ),
    ValidValuesCase("contradictions_within_conjunction", predicate=(pl.col("symbol") == "US") & (pl.col("symbol") == "EU"), expected_values=set()),
    ValidValuesCase(
        "complex_in_and_equal_combinations",
        predicate=(pl.col("symbol").is_in(["US", "EU"]) & (pl.col("symbol") == "US"))
        | (pl.col("symbol").is_in(["JP", "CN"]) & (pl.col("price") > 100)),
        expected_values={"US", "JP", "CN"},
    ),
    ValidValuesCase(
        "multiple_complex_rules",
        predicate=(pl.col("symbol").is_in(["US", "EU", "JP"]) & (pl.col("price") > 100))
        | ((pl.col("symbol") == "UK") & (pl.col("volume") > 1000))
        | (pl.col("symbol").is_in(["CN", "HK"]) & pl.col("symbol").is_in(["HK", "TW"])),
        expected_values={"US", "EU", "JP", "UK", "HK"},
    ),
    ValidValuesCase(
        "contradiction_in_one_conjunction_but_not_others",
        predicate=((pl.col("symbol") == "US") & (pl.col("symbol") == "EU")) | (pl.col("symbol") == "JP"),
        expected_values={"JP"},
    ),
    ValidValuesCase(
        "all_conjunctions_have_contradictions",
        predicate=((pl.col("symbol") == "US") & (pl.col("symbol") == "EU")) | (pl.col("symbol").is_in([]) & (pl.col("price") > 100)),
        expected_values=set(),
    ),
    ValidValuesCase(
        "multiple_in_operators_within_single_conjunction",
        predicate=pl.col("symbol").is_in(["US", "EU", "JP"]) & pl.col("symbol").is_in(["JP", "CN", "UK"]),
        expected_values={"JP"},
    ),
    ValidValuesCase(
        "not_equal_operator", predicate=(pl.col("symbol") != "US") & pl.col("symbol").is_in(["US", "EU", "JP"]), expected_values={"EU", "JP"}
    ),
    ValidValuesCase(
        "not_in_operator",
        predicate=~pl.col("symbol").is_in(["US", "EU"]) & pl.col("symbol").is_in(["US", "EU", "JP", "CN"]),
        expected_values={"JP", "CN"},
    ),
    ValidValuesCase(
        "different_operators_across_conjunctions",
        predicate=(pl.col("symbol") == "US") | (pl.col("symbol") == "AP") | pl.col("symbol").is_in(["JP", "CN"]),
        expected_values={"US", "AP", "JP", "CN"},
    ),
    # Boolean casts are transparent, other casts are not
    ValidValuesCase("basic_boolean_cast", predicate=(pl.col("symbol") == "US").cast(pl.Boolean), expected_values={"US"}),
    ValidValuesCase(
        "boolean_cast_in_and_expression", predicate=(pl.col("symbol") == "US").cast(pl.Boolean) & (pl.col("symbol") != "JP"), expected_values={"US"}
    ),
    ValidValuesCase(
        "boolean_cast_in_or_expression",
        predicate=(pl.col("symbol") == "US").cast(pl.Boolean) | (pl.col("symbol") == "JP"),
        expected_values={"US", "JP"},
    ),
    ValidValuesCase(
        "multiple_boolean_casts",
        predicate=(pl.col("symbol") == "US").cast(pl.Boolean) | (pl.col("symbol") == "JP").cast(pl.Boolean),
        expected_values={"US", "JP"},
    ),
    ValidValuesCase("boolean_cast_with_in", predicate=pl.col("symbol").is_in(["US", "JP"]).cast(pl.Boolean), expected_values={"US", "JP"}),
    ValidValuesCase("nested_boolean_casts", predicate=(pl.col("symbol") == "US").cast(pl.Boolean).cast(pl.Boolean), expected_values={"US"}),
    ValidValuesCase(
        "boolean_cast_of_complex_expression",
        predicate=(((pl.col("symbol") == "US") & (pl.col("price") > 100)) | ((pl.col("symbol") == "JP") & (pl.col("price") > 200))).cast(pl.Boolean),
        expected_values={"US", "JP"},
    ),
    ValidValuesCase(
        "complex_nested_expressions_with_boolean_casts",
        predicate=((pl.col("symbol") == "US") & (pl.col("price") > 100)).cast(pl.Boolean)
        | (pl.col("symbol").is_in(["JP", "EU"]) & (pl.col("volume") > 1000)).cast(pl.Boolean),
        expected_values={"US", "JP", "EU"},
    ),
    ValidValuesCase(
        "boolean_cast_with_contradictions",
        predicate=((pl.col("symbol") == "US") & (pl.col("symbol") == "JP")).cast(pl.Boolean),
        expected_values=set(),
    ),
    ValidValuesCase(
        "boolean_cast_with_mixed_data_types",
        predicate=((pl.col("value") == 10) | (pl.col("value") == "abc")).cast(pl.Boolean),
        expected_values={10, "abc"},
        column="value",
    ),
    ValidValuesCase("non_boolean_cast", predicate=(pl.col("symbol") == "US").cast(pl.Utf8), expected_values=None),
    ValidValuesCase("boolean_cast_with_not", predicate=(~(pl.col("symbol") == "US")).cast(pl.Boolean), expected_values=None),
    # aliases are transparent
    ValidValuesCase("alias_simple_equality", predicate=pl.col("symbol").alias("symbol_alias") == "US", expected_values={"US"}),
    ValidValuesCase(
        "alias_in_operator", predicate=pl.col("symbol").alias("symbol_alias").is_in(["US", "EU", "JP"]), expected_values={"US", "EU", "JP"}
    ),
    ValidValuesCase(
        "alias_logical_operations",
        predicate=(pl.col("symbol").alias("symbol_alias") == "US") & (pl.col("price").alias("price_alias") > 100),
        expected_values={"US"},
    ),
    ValidValuesCase(
        "alias_with_exclusions",
        predicate=(pl.col("symbol").alias("symbol_alias") != "US") & pl.col("symbol").is_in(["US", "EU", "JP"]),
        expected_values={"EU", "JP"},
    ),
    ValidValuesCase(
        "alias_complex_mixed_operations",
        predicate=((pl.col("symbol").alias("s1") == "US") | pl.col("symbol").alias("s2").is_in(["EU", "JP", "UK"]))
        & (pl.col("symbol").alias("s3") != "UK"),
        expected_values={"US", "EU", "JP"},
    ),
    # a cast is unwrapped only when the schema proves it value-preserving
    ValidValuesCase(
        "narrowing_cast_on_column_is_not_stripped-no_schema",
        predicate=pl.col("a").cast(pl.Int8, wrap_numerical=True) == 1,
        expected_values=None,
        column="a",
    ),
    ValidValuesCase(
        "narrowing_cast_on_column_is_not_stripped-schema",
        predicate=pl.col("a").cast(pl.Int8, wrap_numerical=True) == 1,
        expected_values=None,
        column="a",
        schema={"a": pl.Int64},
    ),
    ValidValuesCase(
        "value_preserving_widening_cast_still_extracts_values-no_schema", predicate=pl.col("a").cast(pl.Int64) == 1, expected_values=None, column="a"
    ),
    ValidValuesCase(
        "value_preserving_widening_cast_still_extracts_values-schema",
        predicate=pl.col("a").cast(pl.Int64) == 1,
        expected_values={1},
        column="a",
        schema={"a": pl.Int8},
    ),
    # NOT must not exclude values when the inner AND ignored a side it could not convert
    ValidValuesCase(
        "not_over_relaxed_and_does_not_exclude_values-cast",
        predicate=pl.col("x").is_in([1, 2]) & ~((pl.col("x") == 1) & (pl.col("x").cast(pl.Int64) == 2)),
        expected_values={1, 2},
        column="x",
    ),
    ValidValuesCase(
        "not_over_relaxed_and_does_not_exclude_values-other_col",
        predicate=pl.col("x").is_in([1, 2]) & ~((pl.col("x") == 1) & (pl.col("y") == 2)),
        expected_values={1, 2},
        column="x",
    ),
    ValidValuesCase(
        "not_over_relaxed_and_does_not_exclude_values-exact",
        predicate=pl.col("x").is_in([1, 2]) & ~(pl.col("x") == 1),
        expected_values={2},
        column="x",
    ),
    ValidValuesCase(
        "sub_microsecond_literal_is_not_pushed",
        predicate=pl.col("t") == pl.lit(epoch_ns(datetime(2024, 1, 2, 12, 30)) + 500).cast(pl.Datetime("ns")),
        expected_values=None,
        column="t",
        schema={"t": pl.Datetime("ns")},
    ),
]


@pytest.mark.parametrize("case", as_params(CASES))
def test_valid_values(case: ValidValuesCase):
    assert convert_expr_to_valid_values(case.predicate, case.column, case.schema) == case.expected_values


# 01:30 on the day New York's clocks go back, which happens twice; Python's ``fold=1`` picks the second time.
NEW_YORK_REPEATED_HOUR = datetime(2024, 11, 3, 1, 30, tzinfo=ZoneInfo("America/New_York"))


@dataclass(frozen=True)
class ValidValuesCoverRowsCase:
    """A frame and a filter; the extracted values must cover every row the filter keeps."""

    id: str
    _: KW_ONLY
    df: pl.DataFrame
    pred: pl.Expr


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            ValidValuesCoverRowsCase(
                "nan",
                df=pl.DataFrame({"x": [float("nan"), 1.0, 2.0]}),
                pred=pl.col("x").is_in([float("nan"), 1.0]) & pl.col("x").is_in([float("nan"), 2.0]),
            ),
            ValidValuesCoverRowsCase(
                "dst_fold",
                df=pl.DataFrame({"x": [NEW_YORK_REPEATED_HOUR, NEW_YORK_REPEATED_HOUR.replace(fold=1)]}),
                pred=pl.col("x").is_in([NEW_YORK_REPEATED_HOUR, NEW_YORK_REPEATED_HOUR.replace(fold=1)]),
            ),
            ValidValuesCoverRowsCase("float32", df=pl.DataFrame({"x": pl.Series([0.1], dtype=pl.Float32)}), pred=pl.col("x") == 0.1),
            ValidValuesCoverRowsCase(
                "date_vs_datetime",
                df=pl.DataFrame({"x": [date(2024, 1, 1)]}),
                pred=(pl.col("x") == date(2024, 1, 1)) & (pl.col("x") == datetime(2024, 1, 1)),
            ),
        ]
    ),
)
def test_valid_values_cover_matching_rows(case: ValidValuesCoverRowsCase):
    """Every kept row's value must be in the extracted set, both by Python and by Polars comparison."""
    kept = case.df.filter(case.pred)["x"]
    assert not kept.is_empty()
    values = convert_expr_to_valid_values(case.pred, "x", case.df.schema)
    if values is not None:
        assert all(value in values for value in kept.to_list())
        assert kept.is_in(pl.Series(list(values), dtype=case.df.schema["x"])).all()


def test_set_visitor_visitor_direct():
    visitor = SetVisitor("symbol")
    visitor.visit(get_parsed_expr(pl.col("symbol").is_in(["US", "EU", "JP"])))
    assert visitor.process_results() == {"US", "EU", "JP"}


@dataclass(frozen=True)
class NewSetsCase:
    """A column and two filters visited in turn, and the values allowed after each."""

    id: str
    _: KW_ONLY
    col: str
    first: pl.Expr
    second: pl.Expr
    first_expect: set
    expect: set


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            NewSetsCase(
                "consistent_set_creation",
                col="symbol",
                first=pl.col("symbol") == "US",
                second=pl.col("symbol") == "EU",
                first_expect={"US"},
                expect={"US", "EU"},
            ),
            NewSetsCase(
                "null_handling",
                col="nullable_col",
                first=pl.col("nullable_col").is_null(),
                second=pl.col("nullable_col") == "value",
                first_expect={None},
                expect={None, "value"},
            ),
        ]
    ),
)
def test_set_visitor_creates_new_sets(case: NewSetsCase):
    """Visiting a second node replaces the inclusions set rather than mutating it in place."""
    visitor = SetVisitor(case.col)
    visitor.visit(get_parsed_expr(case.first))
    first_set = visitor.inclusions
    assert first_set == case.first_expect
    visitor.visit(get_parsed_expr(case.second))
    assert visitor.inclusions is not first_set
    assert visitor.inclusions == case.expect
    assert first_set == case.first_expect


def test_logical_and_operator_handling():
    """Regression: LOGICAL_AND, the spelling before Polars optimizes a plan, is handled like AND."""
    node = get_parsed_expr((pl.col("date_partition") == date(2009, 10, 2)) & (pl.col("symbol") == "AAPL US"))
    assert isinstance(node, BinaryExprNode)
    forced = BinaryExprNode(expr=node.expr, can_extract_literal=False, left=node.left, right=node.right, op=OperatorType.LOGICAL_AND)
    assert convert_expr_to_valid_values(forced, "symbol") == {"AAPL US"}
