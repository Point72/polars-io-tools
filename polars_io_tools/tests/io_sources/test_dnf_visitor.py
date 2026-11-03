"""convert_expr_to_dnf and the checks built on it: negating, combining, and finding filters no row can pass.

``convert_expr_to_dnf`` rewrites a filter as an OR of ANDs (disjunctive normal form, DNF): a list of groups, each group
a list of ``(column, operator, value)`` conditions. A row passes if every condition of at least one group holds.
``None`` means the filter cannot be written this way. A group whose conditions can never all hold, such as
``x > 5`` and ``x < 3``, is a contradiction.

The property tests at the end (using the Hypothesis library) check every function that reads a filter, not only the
DNF: on random frames and filters, the DNF, the contradiction check, the value sets (``set_visitor``), the ranges
(``range_visitor``) and the restriction to some columns (``restrict_visitor``) must keep every row plain Polars keeps.
"""

import functools
import itertools
import math
import operator
import time
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from types import EllipsisType
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import portion
import pytest
from hypothesis import assume, given

from polars_io_tools.io_sources.base import get_parsed_expr
from polars_io_tools.io_sources.dnf_visitor import (
    DNFVisitor,
    _is_contradiction,
    combine_and_dnf,
    convert_expr_to_dnf,
    is_contradiction,
    negate_dnf,
)
from polars_io_tools.io_sources.range_visitor import convert_expr_to_datetime_range, convert_expr_to_range
from polars_io_tools.io_sources.restrict_visitor import restrict_expr_to_columns
from polars_io_tools.io_sources.set_visitor import convert_expr_to_valid_values
from polars_io_tools.tests.helpers.assertions import assert_pushed_filter_keeps_rows
from polars_io_tools.tests.helpers.cases import HOUR_US, NEW_YORK, NEW_YORK_FIRST_0130_US, as_params, epoch_ns
from polars_io_tools.tests.helpers.property_checks import (
    SETTINGS,
    filter_a_source_receives,
    mask,
    no_panic,
    note_case,
    rows_plain_polars_keeps,
)
from polars_io_tools.tests.helpers.strategies import (
    POLARS_ERRORS,
    frames_and_predicates,
)


@dataclass(frozen=True)
class DnfCase:
    """A filter, and the DNF it converts to."""

    id: str
    _: KW_ONLY
    predicate: pl.Expr
    # ``...`` skips the comparison
    expected_dnf: list[list[tuple]] | None | EllipsisType = ...
    # the column dtypes passed to the converter
    schema: dict | pl.Schema | None = None
    # the DNF must keep every row of this frame that the filter keeps
    rows_to_keep: pl.DataFrame | None = None
    # compare the order of conditions inside a group too
    compare_exactly: bool = False


# A null, and 257, which becomes 1 when cast to Int8 with wrapping.
_A_WITH_NULL_AND_257 = pl.DataFrame({"a": [1, 1, 2, None, 257], "b": ["xx", "xyz", "xx", "xx", "xx"]})
_FLAG_SCHEMA = pl.Schema({"flag": pl.Boolean})
_YEAR_2023 = date(2023, 1, 1), date(2023, 12, 31)
# 01:30 on the day New York's clocks go back, which happens twice; Python's ``fold=1`` picks the second time.
_NEW_YORK_REPEATED_HOUR = datetime(2024, 11, 3, 1, 30, tzinfo=ZoneInfo("America/New_York"))
_NAN = float("nan")


DNF_CASES = [
    # comparisons
    DnfCase("basic_comparisons-eq", predicate=pl.col("x") == 10, expected_dnf=[[("x", "=", 10)]]),
    DnfCase("basic_comparisons-neq", predicate=pl.col("y") != "test", expected_dnf=[[("y", "!=", "test")]]),
    DnfCase("basic_comparisons-gt", predicate=pl.col("z") > 5, expected_dnf=[[("z", ">", 5)]]),
    DnfCase("basic_comparisons-lt", predicate=pl.col("a") < 100, expected_dnf=[[("a", "<", 100)]]),
    DnfCase("basic_comparisons-gte", predicate=pl.col("b") >= 20, expected_dnf=[[("b", ">=", 20)]]),
    DnfCase("basic_comparisons-lte", predicate=pl.col("c") <= 50, expected_dnf=[[("c", "<=", 50)]]),
    DnfCase("basic_comparisons-reversed", predicate=5 == pl.col("x"), expected_dnf=[[("x", "=", 5)]]),
    DnfCase("literal_on_left_is_flipped-lt", predicate=pl.lit(1) < pl.col("a"), expected_dnf=[[("a", ">", 1)]]),
    DnfCase("literal_on_left_is_flipped-ge", predicate=pl.lit(5) >= pl.col("a"), expected_dnf=[[("a", "<=", 5)]]),
    DnfCase("equal_date", predicate=pl.col("date") == date(2023, 1, 1), expected_dnf=[[("date", "=", date(2023, 1, 1))]], compare_exactly=True),
    DnfCase("expr_with_col_comparisons-and", predicate=(pl.col("a") > pl.col("b")) & (pl.col("c") == 12), expected_dnf=[[("c", "=", 12)]]),
    DnfCase("expr_with_col_comparisons-or", predicate=(pl.col("a") > pl.col("b")) | (pl.col("c") == 12), expected_dnf=None),
    DnfCase("unsupported_expressions", predicate=pl.col("x").cast(pl.Int64) + 5, expected_dnf=None),
    DnfCase("literal_comparisons", predicate=pl.lit(5) > pl.lit(3), expected_dnf=None),
    # Boolean columns: a bare column is ``== True`` and its negation ``== False``; Polars >= 2.1 lowers
    # ``col == True`` to the bare column and ``col == False`` to ``col.not()``.
    DnfCase(
        "boolean_column_predicate-bare", predicate=pl.col("flag"), expected_dnf=[[("flag", "=", True)]], schema=_FLAG_SCHEMA, compare_exactly=True
    ),
    DnfCase(
        "boolean_column_predicate-not",
        predicate=pl.col("flag").not_(),
        expected_dnf=[[("flag", "=", False)]],
        schema=_FLAG_SCHEMA,
        compare_exactly=True,
    ),
    DnfCase(
        "boolean_column_predicate-invert", predicate=~pl.col("flag"), expected_dnf=[[("flag", "=", False)]], schema=_FLAG_SCHEMA, compare_exactly=True
    ),
    DnfCase(
        "boolean_column_predicate-eq_true",
        predicate=pl.col("flag") == True,
        expected_dnf=[[("flag", "=", True)]],
        schema=_FLAG_SCHEMA,
        compare_exactly=True,
    ),
    DnfCase(
        "boolean_column_predicate-eq_false",
        predicate=pl.col("flag") == False,
        expected_dnf=[[("flag", "=", False)]],
        schema=_FLAG_SCHEMA,
        compare_exactly=True,
    ),
    DnfCase("boolean_column_predicate-no_schema", predicate=pl.col("flag"), expected_dnf=[[("flag", "=", True)]], compare_exactly=True),
    # ``cast(Boolean)`` on an integer column is ``x != 0``, not ``x == True``.
    DnfCase(
        "non_boolean_cast_to_boolean_not_misread-schema", predicate=pl.col("x").cast(pl.Boolean), expected_dnf=None, schema=pl.Schema({"x": pl.Int64})
    ),
    DnfCase("non_boolean_cast_to_boolean_not_misread-no_schema", predicate=pl.col("x").cast(pl.Boolean), expected_dnf=None),
    DnfCase(
        "non_boolean_cast_to_boolean_not_misread-and_eq",
        predicate=pl.col("x").cast(pl.Boolean) & (pl.col("x") == 2),
        expected_dnf=[[("x", "=", 2)]],
        compare_exactly=True,
    ),
    # is_between
    DnfCase(
        "is_between_date",
        predicate=pl.col("date").is_between(*_YEAR_2023),
        expected_dnf=[[("date", ">=", _YEAR_2023[0]), ("date", "<=", _YEAR_2023[1])]],
    ),
    DnfCase(
        "is_between_datetime",
        predicate=pl.col("timestamp").is_between(datetime(2023, 1, 1, 8, 30), datetime(2023, 12, 31, 17, 45, 30)),
        expected_dnf=[[("timestamp", ">=", datetime(2023, 1, 1, 8, 30)), ("timestamp", "<=", datetime(2023, 12, 31, 17, 45, 30))]],
        compare_exactly=True,
    ),
    *[
        DnfCase(
            f"is_between_date_closed_parameters-{closed}",
            predicate=pl.col("date").is_between(*_YEAR_2023, closed=closed),
            expected_dnf=[[("date", lo, _YEAR_2023[0]), ("date", hi, _YEAR_2023[1])]],
            compare_exactly=True,
        )
        for closed, lo, hi in [("both", ">=", "<="), ("left", ">=", "<"), ("right", ">", "<="), ("none", ">", "<")]
    ],
    DnfCase(
        "is_between_same_day-both",
        predicate=pl.col("date").is_between(date(2023, 6, 15), date(2023, 6, 15)),
        expected_dnf=[[("date", ">=", date(2023, 6, 15)), ("date", "<=", date(2023, 6, 15))]],
        compare_exactly=True,
    ),
    DnfCase(
        "is_between_same_day-none",
        predicate=pl.col("date").is_between(date(2023, 6, 15), date(2023, 6, 15), closed="none"),
        expected_dnf=[[("date", ">", date(2023, 6, 15)), ("date", "<", date(2023, 6, 15))]],
        compare_exactly=True,
    ),
    # literal types are kept: a date never equals a datetime
    DnfCase(
        "is_between_mixed_date_types",
        predicate=pl.col("dt").is_between(date(2023, 1, 1), datetime(2023, 12, 31, 23, 59, 59)),
        expected_dnf=[[("dt", ">=", date(2023, 1, 1)), ("dt", "<=", datetime(2023, 12, 31, 23, 59, 59))]],
        compare_exactly=True,
    ),
    DnfCase(
        "is_between_with_polars_date_literals",
        predicate=pl.col("date").is_between(pl.lit(_YEAR_2023[0]), pl.lit(_YEAR_2023[1])),
        expected_dnf=[[("date", ">=", _YEAR_2023[0]), ("date", "<=", _YEAR_2023[1])]],
        compare_exactly=True,
    ),
    # logical operations
    DnfCase(
        "logical_operations-simple_and", predicate=(pl.col("x") == 10) & (pl.col("y") == "test"), expected_dnf=[[("x", "=", 10), ("y", "=", "test")]]
    ),
    DnfCase(
        "logical_operations-nested_and",
        predicate=(pl.col("x") == 10) & (pl.col("y") == "test") & (pl.col("z") > 5),
        expected_dnf=[[("x", "=", 10), ("y", "=", "test"), ("z", ">", 5)]],
    ),
    DnfCase(
        "logical_operations-simple_or", predicate=(pl.col("x") == 10) | (pl.col("y") == "test"), expected_dnf=[[("x", "=", 10)], [("y", "=", "test")]]
    ),
    DnfCase(
        "logical_operations-nested_or",
        predicate=(pl.col("x") == 10) | (pl.col("y") == "test") | (pl.col("z") > 5),
        expected_dnf=[[("x", "=", 10)], [("y", "=", "test")], [("z", ">", 5)]],
    ),
    DnfCase(
        "logical_operations-and_or_mixed",
        predicate=(pl.col("x") == 10) & ((pl.col("y") == "test") | (pl.col("z") > 5)),
        expected_dnf=[[("x", "=", 10), ("y", "=", "test")], [("x", "=", 10), ("z", ">", 5)]],
    ),
    DnfCase(
        "logical_operations-or_and_mixed",
        predicate=(pl.col("x") == 10) | ((pl.col("y") == "test") & (pl.col("z") > 5)),
        expected_dnf=[[("x", "=", 10)], [("y", "=", "test"), ("z", ">", 5)]],
    ),
    DnfCase(
        "any_horizontal_nested_with_and",
        predicate=pl.any_horizontal(pl.col("x") > 5, (pl.col("y") == 10) & (pl.col("z") < 3)),
        expected_dnf=[[("x", ">", 5)], [("y", "=", 10), ("z", "<", 3)]],
    ),
    # functions
    DnfCase("function_expressions-is_in", predicate=pl.col("x").is_in((1, 2, 3)), expected_dnf=[[("x", "in", [1, 2, 3])]]),
    DnfCase("function_expressions-is_null", predicate=pl.col("y").is_null(), expected_dnf=[[("y", "is", None)]]),
    DnfCase("function_expressions-is_not_null", predicate=pl.col("z").is_not_null(), expected_dnf=[[("z", "is not", None)]]),
    DnfCase("function_expressions-starts_with", predicate=pl.col("s").str.starts_with("prefix"), expected_dnf=[[("s", "~", "^prefix")]]),
    DnfCase("function_expressions-ends_with", predicate=pl.col("s").str.ends_with("suffix"), expected_dnf=[[("s", "~", "suffix$")]]),
    DnfCase("function_expressions-contains", predicate=pl.col("s").str.contains("substr"), expected_dnf=[[("s", "~", "substr")]]),
    DnfCase("collections_in_is_in-list", predicate=pl.col("x").is_in([1, 2, 3]), expected_dnf=[[("x", "in", [1, 2, 3])]]),
    DnfCase("collections_in_is_in-tuple", predicate=pl.col("x").is_in((4, 5, 6)), expected_dnf=[[("x", "in", [4, 5, 6])]]),
    DnfCase("collections_in_is_in-empty", predicate=pl.col("x").is_in([]), expected_dnf=[[("x", "in", [])]]),
    DnfCase("is_in_single_item", predicate=pl.col("x").is_in(["US"]), expected_dnf=[[("x", "in", ["US"])]]),
    DnfCase("negated_operators", predicate=~pl.col("x").is_in([1, 2, 3]), expected_dnf=[[("x", "!in", [1, 2, 3])]]),
    DnfCase("eq_missing_and_ne_missing_with_null_literal-eq", predicate=pl.col("a").eq_missing(None), expected_dnf=[[("a", "is", None)]]),
    DnfCase("eq_missing_and_ne_missing_with_null_literal-ne", predicate=pl.col("a").ne_missing(None), expected_dnf=[[("a", "is not", None)]]),
    DnfCase(
        "complex_expressions-complex_1",
        predicate=((pl.col("a") > 5) & (pl.col("b") <= 10)) | ((pl.col("c") == "test") & pl.col("d").is_not_null()),
        expected_dnf=[[("a", ">", 5), ("b", "<=", 10)], [("c", "=", "test"), ("d", "is not", None)]],
    ),
    DnfCase(
        "complex_expressions-complex_2",
        predicate=((pl.col("a") > 5) & (pl.col("b") <= 10)) | (((pl.col("c") == "test") | pl.col("d").ne(True)) & pl.col("e").is_null()),
        expected_dnf=[[("a", ">", 5), ("b", "<=", 10)], [("c", "=", "test"), ("e", "is", None)], [("d", "!=", True), ("e", "is", None)]],
    ),
    DnfCase(
        "complex_expressions-complex_3",
        predicate=pl.col("e").is_in([1, 2, 3]) & pl.col("f").str.contains("xyz"),
        expected_dnf=[[("e", "in", [1, 2, 3]), ("f", "~", "xyz")]],
    ),
    # A cast to Boolean is ignored; other casts stop the conversion.
    DnfCase("supported_cast", predicate=(pl.col("x") == 0).cast(pl.Boolean), expected_dnf=[[("x", "=", 0)]]),
    DnfCase(
        "complex_expression_with_boolean_cast",
        predicate=((pl.col("x") > 5) & (pl.col("y") == "test")).cast(pl.Boolean),
        expected_dnf=[[("x", ">", 5), ("y", "=", "test")]],
    ),
    DnfCase("nested_boolean_casts", predicate=(pl.col("x") == 0).cast(pl.Boolean).cast(pl.Boolean), expected_dnf=[[("x", "=", 0)]]),
    DnfCase("cast_other_than_boolean", predicate=(pl.col("x") == 0).cast(pl.Int32), expected_dnf=None),
    DnfCase(
        "boolean_cast_in_logical_operation",
        predicate=(pl.col("x") > 5).cast(pl.Boolean) & (pl.col("y") == "test"),
        expected_dnf=[[("x", ">", 5), ("y", "=", "test")]],
    ),
    DnfCase("value_preserving_widening_cast_still_pushes_down-no_schema", predicate=pl.col("a").cast(pl.Int64) == 1, expected_dnf=None),
    DnfCase(
        "value_preserving_widening_cast_still_pushes_down-schema",
        predicate=pl.col("a").cast(pl.Int64) == 1,
        expected_dnf=[[("a", "=", 1)]],
        schema={"a": pl.Int8},
    ),
    DnfCase(
        "sub_microsecond_is_in_literal_is_not_pushed",
        predicate=pl.col("t").is_in(pl.concat_list(pl.lit(epoch_ns(datetime(2024, 1, 2, 12, 30)) + 500).cast(pl.Datetime("ns")))),
        expected_dnf=None,
        schema={"t": pl.Datetime("ns")},
    ),
    # Aliases are ignored.
    DnfCase("alias_simple_equality", predicate=pl.col("x").alias("x_alias") == 10, expected_dnf=[[("x", "=", 10)]]),
    DnfCase(
        "alias_logical_operations",
        predicate=(pl.col("x").alias("x_alias") == 10) & (pl.col("y").alias("y_alias") > 5),
        expected_dnf=[[("x", "=", 10), ("y", ">", 5)]],
    ),
    # when/then/otherwise: a row whose condition is null takes the otherwise branch
    DnfCase(
        "ternary_basic",
        predicate=pl.when(pl.col("x") > 5).then(pl.col("y") == 10).otherwise(pl.col("z") < 3),
        expected_dnf=[[("x", ">", 5), ("y", "=", 10)], [("x", "<=", 5), ("z", "<", 3)], [("x", "is", None), ("z", "<", 3)]],
    ),
    DnfCase(
        "boolean_cast_in_ternary_expression",
        predicate=pl.when((pl.col("x") > 5).cast(pl.Boolean)).then(pl.col("y") == 10).otherwise(pl.col("z") < 3),
        expected_dnf=[[("x", ">", 5), ("y", "=", 10)], [("x", "<=", 5), ("z", "<", 3)], [("x", "is", None), ("z", "<", 3)]],
    ),
    DnfCase(
        "ternary_with_complex_predicate",
        predicate=pl.when((pl.col("x") > 5) & (pl.col("y") < 10)).then(pl.col("z") == 15).otherwise(pl.col("a") != 0),
        expected_dnf=[
            [("x", ">", 5), ("y", "<", 10), ("z", "=", 15)],
            [("x", "<=", 5), ("a", "!=", 0)],
            [("y", ">=", 10), ("a", "!=", 0)],
            [("a", "!=", 0), ("x", "is", None)],
            [("a", "!=", 0), ("y", "is", None)],
        ],
    ),
    DnfCase(
        "ternary_with_function_expressions",
        predicate=pl.when(pl.col("x").is_in([1, 2, 3])).then((pl.col("y") > 5) & (pl.col("z") == 0)).otherwise(pl.col("w").is_null()),
        expected_dnf=[
            [("x", "in", [1, 2, 3]), ("y", ">", 5), ("z", "=", 0)],
            [("x", "!in", [1, 2, 3]), ("w", "is", None)],
            [("w", "is", None), ("x", "is", None)],
        ],
    ),
    DnfCase(
        "ternary_nested",
        predicate=pl.when(pl.col("x") < 0)
        .then(pl.when(pl.col("y") > 10).then(pl.col("z") == 5).otherwise(pl.col("z") == 0))
        .otherwise(pl.col("w").is_not_null()),
        expected_dnf=[
            [("x", "<", 0), ("y", ">", 10), ("z", "=", 5)],
            [("x", "<", 0), ("y", "<=", 10), ("z", "=", 0)],
            [("x", ">=", 0), ("w", "is not", None)],
            [("x", "<", 0), ("y", "is", None), ("z", "=", 0)],
            [("w", "is not", None), ("x", "is", None)],
        ],
    ),
    DnfCase(
        "ternary_with_empty_clauses", predicate=pl.when(pl.col("x") > 5).then(pl.lit(True)).otherwise(pl.lit(False)), expected_dnf=[[("x", ">", 5)]]
    ),
    DnfCase(
        "ternary_chained_when",
        predicate=pl.when(pl.col("x") > 10)
        .then(pl.when(pl.col("y") < 5).then(pl.col("z") == 1).otherwise(pl.col("z") == 2))
        .otherwise(pl.col("z") == 3),
        expected_dnf=[
            [("x", ">", 10), ("y", "<", 5), ("z", "=", 1)],
            [("x", ">", 10), ("y", ">=", 5), ("z", "=", 2)],
            [("x", "<=", 10), ("z", "=", 3)],
            [("x", ">", 10), ("y", "is", None), ("z", "=", 2)],
            [("x", "is", None), ("z", "=", 3)],
        ],
    ),
    DnfCase(
        "ternary_multiple_chained_when",
        predicate=pl.when(pl.col("x") > 100)
        .then(pl.col("result") == "high")
        .when(pl.col("x") > 50)
        .then(pl.col("result") == "medium")
        .when(pl.col("x") > 10)
        .then(pl.col("result") == "low")
        .otherwise(pl.col("result") == "very low"),
        expected_dnf=[
            [("x", ">", 100), ("result", "=", "high")],
            [("x", "<=", 100), ("x", ">", 50), ("result", "=", "medium")],
            [("x", "<=", 100), ("x", "<=", 50), ("x", ">", 10), ("result", "=", "low")],
            [("x", "<=", 100), ("x", "<=", 50), ("x", "<=", 10), ("result", "=", "very low")],
            [("result", "=", "very low"), ("x", "is", None), ("x", "is", None), ("x", "is", None)],
        ],
    ),
    DnfCase(
        "ternary_multiple_conditions_in_when",
        predicate=pl.when(pl.col("x") > 10, pl.col("y") < 5).then(pl.col("z") == 1).otherwise(pl.col("z") == 2),
        expected_dnf=[
            [("x", ">", 10), ("y", "<", 5), ("z", "=", 1)],
            [("x", "<=", 10), ("z", "=", 2)],
            [("y", ">=", 5), ("z", "=", 2)],
            [("x", "is", None), ("z", "=", 2)],
            [("y", "is", None), ("z", "=", 2)],
        ],
    ),
    DnfCase(
        "ternary_multiple_when_with_multiple_conditions",
        predicate=pl.when(pl.col("x") > 100, pl.col("y") == "A")
        .then(pl.col("result") == "category_1")
        .when(pl.col("x") > 50, pl.col("y") == "B")
        .then(pl.col("result") == "category_2")
        .otherwise(pl.col("result") == "category_3"),
        expected_dnf=[
            [("x", ">", 100), ("y", "=", "A"), ("result", "=", "category_1")],
            [("x", "<=", 100), ("x", ">", 50), ("y", "=", "B"), ("result", "=", "category_2")],
            [("y", "!=", "A"), ("x", ">", 50), ("y", "=", "B"), ("result", "=", "category_2")],
            [("x", "<=", 100), ("x", "<=", 50), ("result", "=", "category_3")],
            [("x", "<=", 100), ("y", "!=", "B"), ("result", "=", "category_3")],
            [("y", "!=", "A"), ("x", "<=", 50), ("result", "=", "category_3")],
            [("y", "!=", "A"), ("y", "!=", "B"), ("result", "=", "category_3")],
            [("result", "=", "category_3"), ("x", "<=", 100), ("y", "is", None)],
            [("result", "=", "category_3"), ("x", "is", None), ("y", "!=", "A")],
            [("result", "=", "category_3"), ("y", "!=", "A"), ("y", "is", None)],
            [("result", "=", "category_3"), ("x", "is", None), ("y", "!=", "B")],
            [("result", "=", "category_3"), ("x", "is", None), ("x", "is", None)],
            [("result", "=", "category_3"), ("x", "is", None), ("y", "is", None)],
            [("result", "=", "category_3"), ("x", "<=", 50), ("y", "is", None)],
            [("result", "=", "category_3"), ("y", "!=", "B"), ("y", "is", None)],
            [("result", "=", "category_3"), ("x", "is", None), ("y", "is", None)],
            [("result", "=", "category_3"), ("y", "is", None), ("y", "is", None)],
        ],
    ),
    DnfCase(
        "ternary_nested_with_multiple_conditions",
        predicate=pl.when(pl.col("category") == "A", pl.col("value") > 100)
        .then(
            pl.when(pl.col("subcategory") == "X", pl.col("priority") > 3)
            .then(pl.col("result") == "high_priority_A")
            .otherwise(pl.col("result") == "normal_A")
        )
        .otherwise(pl.when(pl.col("category") == "B").then(pl.col("result") == "any_B").otherwise(pl.col("result") == "other")),
        expected_dnf=[
            [("category", "=", "A"), ("value", ">", 100), ("subcategory", "=", "X"), ("priority", ">", 3), ("result", "=", "high_priority_A")],
            [("category", "=", "A"), ("value", ">", 100), ("subcategory", "!=", "X"), ("result", "=", "normal_A")],
            [("category", "=", "A"), ("value", ">", 100), ("priority", "<=", 3), ("result", "=", "normal_A")],
            [("category", "!=", "A"), ("category", "=", "B"), ("result", "=", "any_B")],
            [("value", "<=", 100), ("category", "=", "B"), ("result", "=", "any_B")],
            [("category", "!=", "A"), ("category", "!=", "B"), ("result", "=", "other")],
            [("value", "<=", 100), ("category", "!=", "B"), ("result", "=", "other")],
            [("category", "=", "A"), ("result", "=", "normal_A"), ("subcategory", "is", None), ("value", ">", 100)],
            [("category", "=", "A"), ("priority", "is", None), ("result", "=", "normal_A"), ("value", ">", 100)],
            [("category", "!=", "A"), ("category", "is", None), ("result", "=", "other")],
            [("category", "is", None), ("result", "=", "other"), ("value", "<=", 100)],
            [("category", "!=", "B"), ("category", "is", None), ("result", "=", "other")],
            [("category", "is", None), ("category", "is", None), ("result", "=", "other")],
            [("category", "=", "B"), ("result", "=", "any_B"), ("value", "is", None)],
            [("category", "!=", "B"), ("result", "=", "other"), ("value", "is", None)],
            [("category", "is", None), ("result", "=", "other"), ("value", "is", None)],
        ],
    ),
    # Nulls, and filters only partly converted: the DNF must keep every row of ``rows_to_keep`` that the filter keeps.
    DnfCase(
        "not_over_partially_converted_and_does_not_drop_rows",
        predicate=~((pl.col("a") == 1) & (pl.col("b").str.len_chars() == 2)),
        expected_dnf=None,
        rows_to_keep=_A_WITH_NULL_AND_257,
    ),
    DnfCase(
        "ne_missing_keeps_null_rows",
        predicate=pl.col("a").ne_missing(1),
        expected_dnf=[[("a", "!=", 1)], [("a", "is", None)]],
        rows_to_keep=_A_WITH_NULL_AND_257,
    ),
    DnfCase(
        "negated_is_in_nulls_equal_keeps_nulls",
        predicate=~pl.col("a").is_in([1], nulls_equal=True),
        expected_dnf=None,
        rows_to_keep=_A_WITH_NULL_AND_257,
    ),
    DnfCase(
        "is_in_nulls_equal_with_null_value_keeps_nulls",
        predicate=pl.col("a").is_in([1, None], nulls_equal=True),
        expected_dnf=[[("a", "in", [1])], [("a", "is", None)]],
        rows_to_keep=pl.DataFrame({"a": [1, 2, None]}, schema={"a": pl.Int64}),
    ),
    # A narrowing cast is refused even with a schema: 257 wraps to 1 under Int8.
    DnfCase(
        "narrowing_cast_on_column_is_not_stripped-no_schema",
        predicate=pl.col("a").cast(pl.Int8, wrap_numerical=True) == 1,
        expected_dnf=None,
        rows_to_keep=_A_WITH_NULL_AND_257,
    ),
    DnfCase(
        "narrowing_cast_on_column_is_not_stripped-schema",
        predicate=pl.col("a").cast(pl.Int8, wrap_numerical=True) == 1,
        expected_dnf=None,
        schema={"a": pl.Int64},
        rows_to_keep=_A_WITH_NULL_AND_257,
    ),
    DnfCase(
        "ternary_otherwise_keeps_rows_where_condition_is_null",
        predicate=pl.when(pl.col("a") >= 5).then(False).otherwise(True),
        rows_to_keep=pl.DataFrame({"a": [None, 1, 5]}),
    ),
    DnfCase(
        "regex_contains_is_not_escaped-regex", predicate=pl.col("s").str.contains("A|B"), rows_to_keep=pl.DataFrame({"s": ["A", "B", "A|B", "C"]})
    ),
    DnfCase(
        "regex_contains_is_not_escaped-literal",
        predicate=pl.col("s").str.contains("A|B", literal=True),
        rows_to_keep=pl.DataFrame({"s": ["A", "B", "A|B", "C"]}),
    ),
]


def _dnf_as_polars_filter(dnf: list[list[tuple]]) -> pl.Expr:
    """The DNF as a Polars filter; a row where it is null is dropped, as a source would drop it."""

    def condition(col, op, value):
        col = pl.col(col)
        if op == "in":
            return col.is_in(list(value))
        if op in ("not in", "!in"):
            return ~col.is_in(list(value))
        if op == "is":
            return col.is_null()
        if op in ("is not", "!is"):
            return col.is_not_null()
        return _COMPARISON_OPERATORS[op](col, value)

    return pl.any_horizontal([pl.all_horizontal([condition(*parts) for parts in group]) for group in dnf]).fill_null(False)


_COMPARISON_OPERATORS = {
    "=": lambda col, v: col == v,
    "==": lambda col, v: col == v,
    "!=": lambda col, v: col != v,
    "<": lambda col, v: col < v,
    "<=": lambda col, v: col <= v,
    ">": lambda col, v: col > v,
    ">=": lambda col, v: col >= v,
    "~": lambda col, v: col.str.contains(v),
}


def _assert_dnf_keeps_rows(df: pl.DataFrame, pred: pl.Expr, dnf) -> None:
    assert_pushed_filter_keeps_rows(df, pred, None if dnf is None else _dnf_as_polars_filter(dnf))


@pytest.mark.parametrize("case", as_params(DNF_CASES))
def test_dnf(case: DnfCase):
    dnf = convert_expr_to_dnf(case.predicate, case.schema)
    if case.compare_exactly:
        assert dnf == case.expected_dnf
    elif case.expected_dnf is not ...:
        assert_dnf_equal(dnf, case.expected_dnf)
    if case.rows_to_keep is not None:
        _assert_dnf_keeps_rows(case.rows_to_keep, case.predicate, dnf)


@dataclass(frozen=True)
class KeepsMatchingRowsCase:
    """A frame and a filter; the DNF and the contradiction check must not drop a row the filter keeps."""

    id: str
    _: KW_ONLY
    df: pl.DataFrame
    pred: pl.Expr


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            KeepsMatchingRowsCase("nan_eq", df=pl.DataFrame({"x": [_NAN, 1.0, 2.0]}), pred=(pl.col("x") == _NAN) & (pl.col("x") == _NAN)),
            KeepsMatchingRowsCase(
                "nan_is_in", df=pl.DataFrame({"x": [_NAN, 1.0, 2.0]}), pred=pl.col("x").is_in([_NAN, 1.0]) & pl.col("x").is_in([_NAN, 2.0])
            ),
            KeepsMatchingRowsCase(
                "dst_fold",
                df=pl.DataFrame({"x": [_NEW_YORK_REPEATED_HOUR, _NEW_YORK_REPEATED_HOUR.replace(fold=1)]}),
                pred=(pl.col("x") == _NEW_YORK_REPEATED_HOUR) & (pl.col("x") != _NEW_YORK_REPEATED_HOUR.replace(fold=1)),
            ),
            KeepsMatchingRowsCase(
                "enum_order",
                df=pl.DataFrame({"x": pl.Series(["z", "a", "m"], dtype=pl.Enum(["z", "a", "m"]))}),
                pred=pl.col("x").is_in(["z", "m"]) & (pl.col("x") < "a"),
            ),
            KeepsMatchingRowsCase(
                "is_between_null_bound",
                df=pl.DataFrame({"x": [-1, 0, 1, None]}),
                pred=pl.when(pl.col("x").is_between(None, 0)).then(False).otherwise(True),
            ),
            KeepsMatchingRowsCase(
                "date_vs_datetime_bound",
                df=pl.DataFrame({"x": [date(2024, 1, 1)]}),
                pred=pl.when(pl.col("x") == date(2024, 1, 1)).then(pl.col("x") >= datetime(2024, 1, 1)).otherwise(False),
            ),
            KeepsMatchingRowsCase(
                "date_vs_datetime_eq",
                df=pl.DataFrame({"x": [date(2024, 1, 1)]}),
                pred=(pl.col("x") == date(2024, 1, 1)) & (pl.col("x") == datetime(2024, 1, 1)),
            ),
            KeepsMatchingRowsCase(
                "float32_eq",
                df=pl.DataFrame({"x": pl.Series([0.1], dtype=pl.Float32)}),
                pred=(pl.col("x") == 0.1) & (pl.col("x") == float(np.float32(0.1))),
            ),
            KeepsMatchingRowsCase(
                "float32_ternary",
                df=pl.DataFrame({"x": pl.Series([0.1], dtype=pl.Float32)}),
                pred=pl.when(pl.col("x") == 0.1).then(pl.col("x") == float(np.float32(0.1))).otherwise(False),
            ),
            KeepsMatchingRowsCase("fractional_int_bounds", df=pl.DataFrame({"x": [2]}), pred=(pl.col("x") > 1.5) & (pl.col("x") < 2.5)),
            KeepsMatchingRowsCase(
                "infinite_int_bounds", df=pl.DataFrame({"x": [2]}), pred=(pl.col("x") > float("-inf")) & (pl.col("x") < float("inf"))
            ),
            KeepsMatchingRowsCase("verbose_regex", df=pl.DataFrame({"x": ["a", "b"]}), pred=pl.col("x").str.contains("(?x)a # comment")),
            KeepsMatchingRowsCase("non_strict_regex", df=pl.DataFrame({"x": ["a", "["]}), pred=pl.col("x").str.contains("[", strict=False)),
            KeepsMatchingRowsCase(
                "literal_finer_than_unit",
                df=pl.DataFrame({"x": pl.Series([datetime(2024, 1, 1)], dtype=pl.Datetime("ms"))}),
                pred=(pl.col("x") == datetime(2024, 1, 1, microsecond=1)) & (pl.col("x") == datetime(2024, 1, 1, microsecond=2)),
            ),
            KeepsMatchingRowsCase(
                "int_vs_float_beyond_2_53", df=pl.DataFrame({"x": [float(2**53)]}), pred=(pl.col("x") == 2**53 + 1) & (pl.col("x") == float(2**53))
            ),
        ]
    ),
)
def test_dnf_and_contradiction_keep_matching_rows(case: KeepsMatchingRowsCase):
    """Python value comparisons must not prove a contradiction or drop rows that Polars keeps."""
    assert case.df.filter(case.pred).is_empty() or _is_contradiction(case.pred, case.df.schema) is False
    _assert_dnf_keeps_rows(case.df, case.pred, convert_expr_to_dnf(case.pred, case.df.schema))


@pytest.mark.parametrize("dtype", [pl.Time, pl.Datetime("ns"), pl.Duration("ns")], ids=["Time", "datetime_ns", "duration_ns"])
@pytest.mark.parametrize("literal_left", [False, True], ids=["column_left", "literal_left"])
@pytest.mark.parametrize("operation", ["=", "!=", "<", "<=", ">", ">="])
@pytest.mark.parametrize("tick", [1, 1000], ids=["submicrosecond", "microsecond"])
def test_temporal_comparison_dnf_preserves_nanoseconds(dtype, literal_left, operation, tick):
    frame = pl.DataFrame({"value": pl.Series([0, 1, 2, 1000, 1001], dtype=pl.Int64).cast(dtype)})
    literal = pl.lit(tick).cast(dtype)
    left, right = (literal, pl.col("value")) if literal_left else (pl.col("value"), literal)
    predicate = _COMPARISON_OPERATORS[operation](left, right)
    dnf = convert_expr_to_dnf(predicate)
    if tick == 1000:
        assert dnf is not None
    if dnf is not None:
        dnf = [[(name, comparison, pl.lit(value, dtype=frame.schema[name])) for name, comparison, value in clause] for clause in dnf]
    _assert_dnf_keeps_rows(frame, predicate, dnf)


def test_dnf_visitor_direct():
    visitor = DNFVisitor()
    visitor.visit(get_parsed_expr((pl.col("x") == 10) & (pl.col("y") > 5)))
    assert visitor.process_results() == [[("x", "=", 10), ("y", ">", 5)]]


def test_combine_and_dnf():
    one, two = [[("x", "=", 10)]], [[("y", "=", "test")]]
    assert_dnf_equal(combine_and_dnf(one, two), [[("x", "=", 10), ("y", "=", "test")]])
    either, right = [[("x", "=", 10)], [("y", "=", "test")]], [[("z", ">", 5)]]
    assert_dnf_equal(combine_and_dnf(either, right), [[("x", "=", 10), ("z", ">", 5)], [("y", "=", "test"), ("z", ">", 5)]])
    assert combine_and_dnf(None, right) == right
    assert combine_and_dnf(either, None) == either
    assert combine_and_dnf(None, None) is None


@dataclass(frozen=True)
class NegateDnfCase:
    """A DNF, and its negation."""

    id: str
    _: KW_ONLY
    dnf: list[list[tuple]]
    expect: list[list[tuple]]


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            NegateDnfCase("single_predicate", dnf=[[("x", "=", 0)]], expect=[[("x", "!=", 0)]]),
            NegateDnfCase("single_conjunction", dnf=[[("x", "=", 0), ("y", ">=", 5)]], expect=[[("x", "!=", 0)], [("y", "<", 5)]]),
            NegateDnfCase(
                "two_clauses",
                dnf=[[("x", "=", 0), ("y", ">=", 5)], [("z", "<", 10)]],
                expect=[[("x", "!=", 0), ("z", ">=", 10)], [("y", "<", 5), ("z", ">=", 10)]],
            ),
            NegateDnfCase(
                "complex_dnf",
                dnf=[[("a", "=", 1), ("b", ">", 2)], [("c", "<=", 3)], [("d", "in", [4, 5])]],
                expect=[[("a", "!=", 1), ("c", ">", 3), ("d", "!in", [4, 5])], [("b", "<=", 2), ("c", ">", 3), ("d", "!in", [4, 5])]],
            ),
            NegateDnfCase(
                "various_dnf_forms-eq_ne_in",
                dnf=[[("x", "=", 0)], [("y", "<>", 5)], [("z", "in", [1, 2, 3])]],
                expect=[[("x", "!=", 0), ("y", "=", 5), ("z", "!in", [1, 2, 3])]],
            ),
            NegateDnfCase(
                "various_dnf_forms-ne_not_in_is_not",
                dnf=[[("x", "!=", 0)], [("y", "not in", [1, 2])], [("z", "is not", None)]],
                expect=[[("x", "=", 0), ("y", "in", [1, 2]), ("z", "is", None)]],
            ),
            NegateDnfCase(
                "various_dnf_forms-and_or",
                dnf=[[("x", "=", 0), ("y", "=", 0)], [("z", "=", 0)]],
                expect=[[("x", "!=", 0), ("z", "!=", 0)], [("y", "!=", 0), ("z", "!=", 0)]],
            ),
        ]
    ),
)
def test_negate_dnf(case: NegateDnfCase):
    assert negate_dnf(case.dnf) == case.expect


_X_IS_INT64 = pl.Schema({"x": pl.Int64})


@dataclass(frozen=True)
class ClauseContradictionCase:
    """One group of conditions, and whether ``is_contradiction`` finds that no row can pass them all."""

    id: str
    _: KW_ONLY
    # (column, operator, value) conditions
    clause: list[tuple]
    # the column dtypes, if any
    schema: pl.Schema | None
    expect: bool


CONTRADICTIONS = [
    ClauseContradictionCase("Empty clause (always true)", clause=[], schema=None, expect=False),
    # Exact values
    ClauseContradictionCase("Exact: col = 5 AND col = 6", clause=[("colA", "=", 5), ("colA", "=", 6)], schema=None, expect=True),
    ClauseContradictionCase("Exact: col = 5 AND col != 5", clause=[("colA", "=", 5), ("colA", "!=", 5)], schema=None, expect=True),
    ClauseContradictionCase("Exact: col = 5 AND col = 5", clause=[("colA", "=", 5), ("colA", "=", 5)], schema=None, expect=False),
    ClauseContradictionCase("Exact: col == 5 AND col <> 5", clause=[("colA", "==", 5), ("colA", "<>", 5)], schema=None, expect=True),
    ClauseContradictionCase("Exact: x = 2", clause=[("x", "=", 2)], schema=None, expect=False),
    # Ranges
    ClauseContradictionCase("Range: col > 10 AND col < 5", clause=[("colA", ">", 10), ("colA", "<", 5)], schema=None, expect=True),
    ClauseContradictionCase("Range: col > 5 AND col < 5", clause=[("colA", ">", 5), ("colA", "<", 5)], schema=None, expect=True),
    ClauseContradictionCase("Range: col >= 5 AND col <= 5", clause=[("colA", ">=", 5), ("colA", "<=", 5)], schema=None, expect=False),
    ClauseContradictionCase("Range: col > 5 AND col <= 5", clause=[("colA", ">", 5), ("colA", "<=", 5)], schema=None, expect=True),
    ClauseContradictionCase("Range: col >= 5 AND col < 5", clause=[("colA", ">=", 5), ("colA", "<", 5)], schema=None, expect=True),
    ClauseContradictionCase("Range: col > 5 AND col > 10", clause=[("colA", ">", 5), ("colA", ">", 10)], schema=None, expect=False),
    ClauseContradictionCase("Range: col < 10 AND col < 5", clause=[("colA", "<", 10), ("colA", "<", 5)], schema=None, expect=False),
    ClauseContradictionCase(
        "Range: col < datetime(2024, 1, 1) AND col >= datetime(2024, 1, 2)",
        clause=[("colA", "<", datetime(2024, 1, 1)), ("colA", ">=", datetime(2024, 1, 2))],
        schema=None,
        expect=True,
    ),
    ClauseContradictionCase(
        "Range: col < datetime(2024, 1, 1) AND col >= datetime(2022, 1, 2)",
        clause=[("colA", "<", datetime(2024, 1, 1)), ("colA", ">=", datetime(2022, 1, 2))],
        schema=None,
        expect=False,
    ),
    ClauseContradictionCase(
        "is_contradiction_obvious_bound_conflict: x > 5 AND x < 3", clause=[("x", ">", 5), ("x", "<", 3)], schema=None, expect=True
    ),
    # Exact value vs range
    ClauseContradictionCase("Exact vs Range: col = 3 AND col > 5", clause=[("colA", "=", 3), ("colA", ">", 5)], schema=None, expect=True),
    ClauseContradictionCase(
        "Exact vs Range: col = 7 AND col > 5 AND col < 10", clause=[("colA", "=", 7), ("colA", ">", 5), ("colA", "<", 10)], schema=None, expect=False
    ),
    ClauseContradictionCase("Exact vs Range: col = 5 AND col >= 5", clause=[("colA", "=", 5), ("colA", ">=", 5)], schema=None, expect=False),
    ClauseContradictionCase("Exact vs Range: col = 5 AND col > 5", clause=[("colA", "=", 5), ("colA", ">", 5)], schema=None, expect=True),
    ClauseContradictionCase("Exact vs Range: col = 5 AND col <= 5", clause=[("colA", "=", 5), ("colA", "<=", 5)], schema=None, expect=False),
    ClauseContradictionCase("Exact vs Range: col = 5 AND col < 5", clause=[("colA", "=", 5), ("colA", "<", 5)], schema=None, expect=True),
    # Inclusion and exclusion
    ClauseContradictionCase("In/Ex: col IN (1,2) AND col = 3", clause=[("colA", "in", [1, 2]), ("colA", "=", 3)], schema=None, expect=True),
    ClauseContradictionCase("In/Ex: col IN (1,2) AND col = 1", clause=[("colA", "in", [1, 2]), ("colA", "=", 1)], schema=None, expect=False),
    ClauseContradictionCase(
        "In/Ex: col IN (1,2) AND col NOT IN (1,2,3)", clause=[("colA", "in", [1, 2]), ("colA", "!in", [1, 2, 3])], schema=None, expect=True
    ),
    ClauseContradictionCase(
        "In/Ex: col IN (1,2) AND col NOT IN (3,4)", clause=[("colA", "in", [1, 2]), ("colA", "!in", [3, 4])], schema=None, expect=False
    ),
    ClauseContradictionCase(
        "In/Ex: col IN (1,2) AND col IN (3,4)", clause=[("colA", "in", [1, 2]), ("colA", "in", [3, 4])], schema=None, expect=True
    ),
    ClauseContradictionCase(
        "In/Ex: col IN (1,2,3) AND col IN (3,4,5)", clause=[("colA", "in", [1, 2, 3]), ("colA", "in", [3, 4, 5])], schema=None, expect=False
    ),
    ClauseContradictionCase(
        "In/Ex: col IN (1,2,3) AND col IN (3,4,5) AND col = 3",
        clause=[("colA", "in", [1, 2, 3]), ("colA", "in", [3, 4, 5]), ("colA", "=", 3)],
        schema=None,
        expect=False,
    ),
    ClauseContradictionCase(
        "In/Ex: col IN (1,2,3) AND col IN (3,4,5) AND col = 1",
        clause=[("colA", "in", [1, 2, 3]), ("colA", "in", [3, 4, 5]), ("colA", "=", 1)],
        schema=None,
        expect=True,
    ),
    ClauseContradictionCase(
        "In/Ex: col IN (1,2,3) AND col IN (3,4,5) AND col NOT IN (3)",
        clause=[("colA", "in", [1, 2, 3]), ("colA", "in", [3, 4, 5]), ("colA", "!in", [3])],
        schema=None,
        expect=True,
    ),
    # NULL
    ClauseContradictionCase("NULL: col IS NULL AND col = 5", clause=[("colA", "is", None), ("colA", "=", 5)], schema=None, expect=True),
    ClauseContradictionCase("NULL: col IS NULL AND col > 0", clause=[("colA", "is", None), ("colA", ">", 0)], schema=None, expect=True),
    ClauseContradictionCase("NULL: col IS NULL AND col IN (1,2)", clause=[("colA", "is", None), ("colA", "in", [1, 2])], schema=None, expect=True),
    ClauseContradictionCase(
        "NULL: col IS NULL AND col IS NOT NULL", clause=[("colA", "is", None), ("colA", "is not", None)], schema=None, expect=True
    ),
    ClauseContradictionCase(
        "NULL: col IS NOT NULL AND col = None (exact value is None)", clause=[("colA", "is not", None), ("colA", "=", None)], schema=None, expect=True
    ),
    ClauseContradictionCase(
        "NULL: col IS NOT NULL AND col IN (None,)", clause=[("colA", "is not", None), ("colA", "in", [None])], schema=None, expect=True
    ),
    ClauseContradictionCase(
        "NULL: col IS NOT NULL AND col IN (None, 1)", clause=[("colA", "is not", None), ("colA", "in", [None, 1])], schema=None, expect=False
    ),
    ClauseContradictionCase("NULL: col IS NULL (no contradiction)", clause=[("colA", "is", None)], schema=None, expect=False),
    ClauseContradictionCase("NULL: col IS NOT NULL (no contradiction)", clause=[("colA", "is not", None)], schema=None, expect=False),
    ClauseContradictionCase("NULL: col IS NULL AND col IN (None,)", clause=[("colA", "is", None), ("colA", "in", [None])], schema=None, expect=False),
    # Complex, single column
    ClauseContradictionCase(
        "Complex: col > 0 AND col < 10 AND col IN (20,30)",
        clause=[("colA", ">", 0), ("colA", "<", 10), ("colA", "in", [20, 30])],
        schema=None,
        expect=True,
    ),
    ClauseContradictionCase(
        "Complex: col > 0 AND col < 10 AND col IN (5, 20)",
        clause=[("colA", ">", 0), ("colA", "<", 10), ("colA", "in", [5, 20])],
        schema=None,
        expect=False,
    ),
    ClauseContradictionCase(
        "Complex: col = 5 AND col != 5 AND col > 0", clause=[("colA", "=", 5), ("colA", "!=", 5), ("colA", ">", 0)], schema=None, expect=True
    ),
    # Multiple columns
    ClauseContradictionCase(
        "Multi-col: colA = 1 AND colA = 2 AND colB = 10", clause=[("colA", "=", 1), ("colA", "=", 2), ("colB", "=", 10)], schema=None, expect=True
    ),
    ClauseContradictionCase("Multi-col: colA = 1 AND colB > 5", clause=[("colA", "=", 1), ("colB", ">", 5)], schema=None, expect=False),
    ClauseContradictionCase(
        "Multi-col: colA > 0 AND colA < 1, colB = 1 (colA contradiction only for int)",
        clause=[("colA", ">", 0), ("colA", "<", 1), ("colB", "=", 1)],
        schema=None,
        expect=False,
    ),
    ClauseContradictionCase(
        "Multi-col: colA > 0 AND colA < 12, colB = 1, colA < -1 (colA contradiction)",
        clause=[("colA", ">", 0), ("colA", "<", 1), ("colA", "<", -1), ("colB", "=", 1)],
        schema=None,
        expect=True,
    ),
    # Non-contradictions
    ClauseContradictionCase("Non-C: col = 5 AND col > 0", clause=[("colA", "=", 5), ("colA", ">", 0)], schema=None, expect=False),
    ClauseContradictionCase("Non-C: col > 0 AND col < 10", clause=[("colA", ">", 0), ("colA", "<", 10)], schema=None, expect=False),
    ClauseContradictionCase("Non-C: col IN (1,2,3) AND col = 1", clause=[("colA", "in", [1, 2, 3]), ("colA", "=", 1)], schema=None, expect=False),
    ClauseContradictionCase("Non-C: col IS NOT NULL AND col > 5", clause=[("colA", "is not", None), ("colA", ">", 5)], schema=None, expect=False),
    ClauseContradictionCase(
        "Non-C: colA = 1, colB = 'test', colC > 3.0", clause=[("colA", "=", 1), ("colB", "=", "test"), ("colC", ">", 3.0)], schema=None, expect=False
    ),
    # A scalar IN value is wrapped in a list
    ClauseContradictionCase("Edge IN: col IN 'abc' (scalar string)", clause=[("colA", "in", "abc")], schema=None, expect=False),
    ClauseContradictionCase("Edge IN: col IN 1 (scalar int)", clause=[("colA", "in", 1)], schema=None, expect=False),
    # IS with a value other than None
    ClauseContradictionCase("IS non-None: colA IS TRUE", clause=[("colA", "is", True)], schema=None, expect=False),
    ClauseContradictionCase(
        "IS non-None: colA IS TRUE AND colA IS NULL", clause=[("colA", "is", True), ("colA", "is", None)], schema=None, expect=True
    ),
    ClauseContradictionCase("IS NOT non-None: colA IS NOT TRUE", clause=[("colA", "is not", True)], schema=None, expect=False),
    # Enumerating an integer range needs the column's dtype: unknown -> not proven a contradiction.
    ClauseContradictionCase(
        "is_contradiction_with_column_not_in_schema",
        clause=[("unknown_col", ">", 3), ("unknown_col", "<", 10), ("unknown_col", "!=", 5)],
        schema=pl.Schema({"other_col": pl.Int64}),
        expect=False,
    ),
    ClauseContradictionCase(
        "is_contradiction_with_column_in_schema",
        clause=[("x", ">", 3), ("x", "<", 10), *(("x", "!=", v) for v in range(4, 10))],
        schema=_X_IS_INT64,
        expect=True,
    ),
]


@pytest.mark.parametrize("case", as_params(CONTRADICTIONS))
def test_is_contradiction(case: ClauseContradictionCase):
    assert is_contradiction(case.clause, case.schema) is case.expect


# Columns of three dtypes, two of each.
_MIXED_SCHEMA = pl.Schema({"x": pl.Int64, "y": pl.String, "p": pl.Boolean, "x2": pl.Int64, "y2": pl.String, "p2": pl.Boolean})


@dataclass(frozen=True)
class FilterContradictionCase:
    """A whole filter, and whether ``_is_contradiction`` finds that no row can pass it."""

    id: str
    _: KW_ONLY
    pred: pl.Expr
    schema: pl.Schema | None
    expect: bool


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            FilterContradictionCase(
                "Multi option is in check",
                pred=pl.col("y").is_in(["a", "b"]) & pl.col("a").eq(5) & (pl.col("a").ne(5) | pl.col("y").is_in(["b", "a"]).not_()),
                schema=pl.Schema({"a": pl.Boolean, "y": pl.String}),
                expect=True,
            ),
            FilterContradictionCase(
                "Compare with True (python boolean)",
                pred=pl.col("y").is_in(["a", "b"]) & pl.col("p").eq(True) & (pl.col("p").ne(True) | pl.col("y").is_in(["b", "a"]).not_()),
                schema=_MIXED_SCHEMA,
                expect=True,
            ),
            FilterContradictionCase(
                "Compare with pl.lit(True)",
                pred=pl.col("y").is_in(["a", "b"])
                & pl.col("p").eq(pl.lit(True))
                & pl.Expr.or_(pl.col("p").ne(pl.lit(True)), ~pl.col("y").is_in(["b", "a"])),
                schema=_MIXED_SCHEMA,
                expect=True,
            ),
        ]
    ),
)
def test_is_contradiction_pl(case: FilterContradictionCase):
    assert _is_contradiction(case.pred, case.schema) == case.expect


_COMPARISONS = {
    "=": operator.eq,
    "==": operator.eq,
    "!=": operator.ne,
    "<>": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "!>": operator.le,
    "!>=": operator.lt,
    "!<": operator.ge,
    "!<=": operator.gt,
}


def dnf_expr(dnf: list, schema: dict | None = None) -> pl.Expr:
    """A DNF of ``(column, op, value)`` terms as a Polars predicate with SQL null semantics.

    Values are compared as dynamic Python literals, as a consumer without the schema does, or typed with the column's
    dtype when ``schema`` is given.
    """

    def lit(c, value):
        return pl.lit(value) if schema is None else pl.lit(value, dtype=schema[c])

    def term(c: str, op: str, value) -> pl.Expr:
        e = pl.col(c)
        if op in ("in", "!in", "not in"):
            values = list(value) if isinstance(value, (list, tuple, set, frozenset)) else [value]
            members = pl.any_horizontal([e == lit(c, v) for v in values if v is not None] or [pl.lit(False)])
            return members if op == "in" else ~members
        if op == "is" or (op in ("=", "==") and value is None):
            return e.is_null() if value is None else e == lit(c, value)
        if op in ("!is", "is not"):
            return e.is_not_null() if value is None else e != lit(c, value)
        if op in ("~", "=~", "!~"):
            match = e.str.contains(value)
            return ~match if op == "!~" else match
        return _COMPARISONS[op](e, lit(c, value))

    clauses = [pl.all_horizontal([term(*t) for t in clause] or [pl.lit(True)]).fill_null(False) for clause in dnf]
    return pl.any_horizontal(clauses) if clauses else pl.lit(False)


def _naive_utc(v):
    if isinstance(v, datetime):
        return v.astimezone(UTC).replace(tzinfo=None) if v.tzinfo is not None else v
    return datetime.combine(v, datetime.min.time())


def _midnight(v):
    return datetime.combine(v, datetime.min.time()) if isinstance(v, date) and not isinstance(v, datetime) else v


def _within(value, interval) -> bool:
    """``value in interval``, reading a date as its midnight against datetime bounds, and a naive value as UTC against
    zoned bounds."""
    try:
        return value in interval
    except TypeError:
        pass
    try:
        return _naive_utc(value) in interval
    except TypeError:
        pass
    try:
        return _naive_utc(value).replace(tzinfo=UTC) in interval
    except TypeError:  # date bounds: their midnights
        return _naive_utc(value) in interval.apply(lambda i: i.replace(lower=_midnight(i.lower), upper=_midnight(i.upper)))


def _members(series: pl.Series, values: set) -> pl.Series:
    """Whether each value of ``series`` is one of ``values`` as a consumer filtering on ``col == value`` sees it."""
    frame = series.to_frame("c")
    try:
        return mask(frame, pl.any_horizontal([pl.col("c") == v for v in values if v is not None] + [pl.col("c").is_null() & (None in values)]))
    except POLARS_ERRORS:  # values a consumer cannot compare with the column
        return pl.Series([v in values for v in series.to_list()])


def _inside(series: pl.Series, interval) -> pl.Series:
    """Whether each non-null, non-NaN value of ``series`` lies in ``interval`` as a consumer filtering on its bounds
    sees it."""
    c = pl.col("c")

    def bound(atomic) -> pl.Expr:
        lower = pl.lit(True) if atomic.lower == -portion.inf else (c >= atomic.lower if atomic.left == portion.CLOSED else c > atomic.lower)
        upper = pl.lit(True) if atomic.upper == portion.inf else (c <= atomic.upper if atomic.right == portion.CLOSED else c < atomic.upper)
        return lower & upper

    skip = c.is_null() | (c.is_nan() if series.dtype.is_float() else pl.lit(False))
    try:
        return mask(series.to_frame("c"), skip | pl.any_horizontal([bound(a) for a in interval] or [pl.lit(False)]))
    except POLARS_ERRORS:  # bounds a consumer cannot compare with the column
        return pl.Series([v is None or (isinstance(v, float) and math.isnan(v)) or v in interval for v in series.to_list()])


def _dnf_mask(frame: pl.DataFrame, dnf: list) -> pl.Series | None:
    """The rows a consumer of ``dnf`` keeps with dynamic or typed literals (either counts); None if Polars rejects both.

    Polars compares a column with a dynamic literal of another time unit in the coarser unit on some versions.
    """
    masks = []
    for schema in (None, dict(frame.schema)):
        try:
            masks.append(mask(frame, dnf_expr(dnf, schema)))
        except POLARS_ERRORS:
            continue
    return functools.reduce(operator.or_, masks) if masks else None


def check_filter_readers(frame: pl.DataFrame, pred: pl.Expr, keep: pl.Series, schema: dict | None) -> None:
    """Assert that every function reading ``pred`` (DNF, contradiction check, value sets, ranges, restriction to some
    columns), given ``schema`` (or none), keeps the rows of ``frame`` where ``keep`` is true."""
    full = dict(frame.schema)
    kept = frame.filter(keep)
    # Every caller of the DNF and the contradiction check passes the schema; without it, an integer column used as a
    # Boolean reads as one.
    if kept.height and schema is not None:
        assert not _is_contradiction(pred, schema), pred
    dnf = convert_expr_to_dnf(pred, schema) if schema is not None else None
    if dnf is not None:
        matched = _dnf_mask(frame, dnf)
        assert matched is None or matched.filter(keep).all(), (pred, dnf)
    for col in frame.columns:
        dtype = full[col]
        try:
            kept_values = kept[col].to_list()
        except POLARS_ERRORS:  # values Python cannot hold
            continue
        values = convert_expr_to_valid_values(pred, col, schema)
        if values is not None:
            assert _members(kept[col], values).all(), (pred, col, values)
        if schema is None:
            continue
        if isinstance(dtype, (pl.Date, pl.Datetime)):
            for enclosure, preserve in ((True, False), (False, False), (False, True)):
                try:
                    interval = convert_expr_to_datetime_range(pred, col, get_enclosure=enclosure, preserve_dates=preserve)
                except TypeError:  # date and datetime bounds that do not compare; callers fall back without preserve_dates
                    assert preserve
                    continue
                as_bound = (lambda v: v) if preserve and dtype == pl.Date else _naive_utc
                assert all(_within(as_bound(v), interval) for v in kept_values if v is not None), (pred, col, interval)
        elif dtype.is_numeric():
            interval = convert_expr_to_range(pred, col)
            assert _inside(kept[col], interval).all(), (pred, col, interval)
    if schema is None:
        return
    for size in range(1, len(frame.columns) + 1):
        for cols in itertools.combinations(frame.columns, size):
            restricted = restrict_expr_to_columns(pred, cols)
            if restricted is not None:
                assert set(restricted.meta.root_names()) <= set(cols), restricted
                try:
                    matched = mask(frame, restricted)
                except POLARS_ERRORS:  # Polars 1.37 rejects some sub-expressions it accepts within the predicate
                    continue
                assert matched.filter(keep).all(), (pred, cols, restricted)


@no_panic
def check_predicate(frame: pl.DataFrame, pred: pl.Expr) -> None:
    """Assert every filter reader, with and without the schema, keeps the rows ``pred`` keeps on ``frame``, and likewise
    for the filter Polars hands to an IO source; skip filters Polars rejects."""
    keep = rows_plain_polars_keeps(frame, pred)
    assume(keep is not None)
    for known in (dict(frame.schema), None):
        check_filter_readers(frame, pred, keep, known)
    pushed = filter_a_source_receives(frame, pred)
    if pushed is not None:
        # Polars may rewrite the filter it hands on (``is_in([])`` becomes ``false``), so it is checked against its own result.
        pushed_keep = rows_plain_polars_keeps(frame, pushed)
        if pushed_keep is not None:
            for known in (dict(frame.schema), None):
                check_filter_readers(frame, pushed, pushed_keep, known)


@SETTINGS
@given(case=frames_and_predicates())
def test_random_predicates_are_covered(case):
    """Every function that reads a filter keeps the rows a random filter keeps (see ``check_filter_readers``)."""
    note_case(*case)
    check_predicate(*case)


# 01:30 in New York on 2024-11-03, before and after the clocks go back.
_NEW_YORK_FIRST_0130, _NEW_YORK_SECOND_0130 = (
    pl.Series([t], dtype=pl.Int64).cast(pl.Datetime("ns", NEW_YORK))
    for t in (NEW_YORK_FIRST_0130_US * 1000, (NEW_YORK_FIRST_0130_US + HOUR_US) * 1000)
)


@dataclass(frozen=True)
class NamedPredicateCase:
    """A frame and a filter that once broke one of the functions that read a filter."""

    id: str
    _: KW_ONLY
    frame: pl.DataFrame
    pred: pl.Expr
    marks: tuple = ()


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            NamedPredicateCase("false", frame=pl.DataFrame({"x": [1, None]}), pred=pl.lit(False)),
            NamedPredicateCase(
                "decimal_float",
                frame=pl.DataFrame({"x": pl.Series([Decimal("0.1")], dtype=pl.Decimal(10, 1))}),
                pred=(pl.col("x") == 0.1) & (pl.col("x") == Decimal("0.1")),
                marks=(pytest.mark.xfail(strict=True, reason="a float and a Decimal literal of equal value are taken as a contradiction"),),
            ),
            NamedPredicateCase(
                "dst_fold",
                frame=pl.DataFrame({"x": pl.concat([_NEW_YORK_FIRST_0130, _NEW_YORK_SECOND_0130])}),
                pred=(pl.col("x") == _NEW_YORK_FIRST_0130[0]) & (pl.col("x") != _NEW_YORK_SECOND_0130[0]),
            ),
            NamedPredicateCase("float32", frame=pl.DataFrame({"x": pl.Series([0.1], dtype=pl.Float32)}), pred=pl.col("x") == 0.1),
            NamedPredicateCase("int_cast_bool", frame=pl.DataFrame({"x": [2, 0, None]}), pred=pl.col("x").cast(pl.Boolean) & (pl.col("x") == 2)),
            NamedPredicateCase(
                "bitwise", frame=pl.DataFrame({"x": [1, 2, 3], "y": [2, 2, 1]}), pred=((pl.col("x") & pl.col("y")) > 0) & (pl.col("x") > 1)
            ),
            NamedPredicateCase(
                "null_condition",
                frame=pl.DataFrame({"k": [None, 5, 0], "x": [1, 300, 4]}),
                pred=pl.when(pl.col("k") > 1).then(pl.col("x") > 100).otherwise(pl.col("k").is_null()),
            ),
            NamedPredicateCase(
                "ns_floor",
                frame=pl.DataFrame({"t": pl.Series([epoch_ns(datetime(2024, 1, 1)) + 500], dtype=pl.Int64).cast(pl.Datetime("ns"))}),
                pred=pl.col("t") == datetime(2024, 1, 1),
            ),
            NamedPredicateCase("cast_source", frame=pl.DataFrame({"x": [1]}), pred=pl.col("x").cast(pl.String) == "1"),
            NamedPredicateCase(
                "float_rounds_int", frame=pl.DataFrame({"x": [2**53 + 1]}), pred=(pl.col("x") == float(2**53)) & (pl.col("x") > 2**53)
            ),
            # Float32 cannot hold 20240101, so the frame holds the nearest Float32, 20240100.0.
            NamedPredicateCase("float32_int", frame=pl.DataFrame({"x": pl.Series([20240100.0], dtype=pl.Float32)}), pred=pl.col("x") == 20240101),
            NamedPredicateCase(
                "zoned_fold_bound",
                frame=pl.DataFrame({"x": _NEW_YORK_FIRST_0130.dt.offset_by("29m59s999999us")}),
                pred=pl.col("x") <= _NEW_YORK_SECOND_0130[0],
            ),
            NamedPredicateCase(
                "decimal_overflow",
                frame=pl.DataFrame({"x": pl.Series([Decimal(-(10**27))], dtype=pl.Decimal(38, 2))}),
                pred=pl.col("x") <= pl.lit(Decimal("0.01"), dtype=pl.Decimal(38, 2)),
            ),
            NamedPredicateCase("beyond_nanoseconds", frame=pl.DataFrame({"x": _NEW_YORK_SECOND_0130}), pred=pl.col("x") > date(1677, 9, 20)),
            NamedPredicateCase("nan", frame=pl.DataFrame({"x": [float("nan"), 1.0, None]}), pred=~(pl.col("x") <= 5.0)),
            NamedPredicateCase("bool_int", frame=pl.DataFrame({"a": [True, False, None]}), pred=(pl.col("a") > 0) | (pl.col("a") == 5)),
        ]
    ),
)
def test_named_predicates_are_covered(case: NamedPredicateCase):
    """Filters that once broke one of the functions that read a filter, checked as in the random test above."""
    check_predicate(case.frame, case.pred)


_TERMS = 10_000


_ROWS = {"x": [0, 1, 5, _TERMS, None, 3], "y": [1, 0, None, 3, _TERMS, -1]}


@dataclass(frozen=True)
class LongChainCase:
    """A filter of ten thousand terms, and which rows it keeps."""

    id: str
    _: KW_ONLY
    # builds the filter
    build: Callable[[], pl.Expr]
    # whether the filter keeps the row with values ``x`` and ``y``
    keeps: Callable[[object, object], bool]


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            LongChainCase(
                "or_chain",
                build=lambda: functools.reduce(operator.or_, [pl.col("x") == i for i in range(_TERMS)]),
                keeps=lambda x, y: x is not None and 0 <= x < _TERMS,
            ),
            LongChainCase(
                "and_chain",
                build=lambda: functools.reduce(operator.and_, [pl.col("x") != i for i in range(_TERMS)]),
                keeps=lambda x, y: x is not None and not 0 <= x < _TERMS,
            ),
            LongChainCase(
                "and_of_or",
                build=lambda: functools.reduce(operator.and_, [(pl.col("x") == i) | (pl.col("y") < i + 1) for i in range(_TERMS // 2)]),
                keeps=lambda x, y: all((x is not None and x == i) or (y is not None and y < i + 1) for i in range(_TERMS // 2)),
            ),
            LongChainCase(
                "negated_or_of_and",
                build=lambda: ~functools.reduce(operator.or_, [(pl.col("x") > i) & (pl.col("y") < i) for i in range(_TERMS // 2)]),
                keeps=lambda x, y: all((x is not None and x <= i) or (y is not None and y >= i) for i in range(_TERMS // 2)),
            ),
        ]
    ),
)
def test_ten_thousand_terms_complete_and_stay_sound(case: LongChainCase):
    """The DNF, the contradiction check, the value sets and the restriction to one column finish on a long chain and
    keep its rows. Polars itself needs about a minute to evaluate such a chain, so the kept rows come from Python."""
    frame = pl.DataFrame(_ROWS)
    schema = dict(frame.schema)
    kept = pl.Series([case.keeps(x, y) for x, y in frame.iter_rows()])
    assert kept.any()
    start = time.monotonic()
    expr = case.build()
    node = get_parsed_expr(expr)
    assert not _is_contradiction(node, schema)
    dnf = convert_expr_to_dnf(node, schema)
    assert dnf is None or _dnf_mask(frame, dnf).filter(kept).all()
    for col in ("x", "y"):
        values = convert_expr_to_valid_values(node, col, schema)
        assert values is None or _members(frame.filter(kept)[col], values).all()
        restricted = restrict_expr_to_columns(node, [col])
        if restricted is not None:
            assert set(restricted.meta.root_names()) == {col}
            # Polars needs about a minute to evaluate the unbalanced original, so an unchanged predicate is not evaluated.
            assert restricted.meta.eq(expr) or mask(frame, restricted).filter(kept).all(), (col, restricted)
    assert time.monotonic() - start < 600


def test_dnf_of_many_disjunctions_stays_within_budget():
    pred = functools.reduce(operator.and_, [(pl.col(f"a{i}") == 1) | (pl.col(f"b{i}") == 1) for i in range(16)])
    dnf = convert_expr_to_dnf(pred, {c: pl.Int64 for i in range(16) for c in (f"a{i}", f"b{i}")})
    assert dnf is None or len(dnf) <= 10_000


def test_dnf_of_wide_clauses_stays_within_budget():
    """Conditions within the size limit are kept, and the looser filter that results is not negated."""
    from polars_io_tools.io_sources.dnf_visitor import MAX_DNF_TUPLES

    pred = pl.all_horizontal([pl.col(f"c{i}") == 1 for i in range(1_000)] + [(pl.col(f"a{i}") == 1) | (pl.col(f"b{i}") == 1) for i in range(13)])
    schema = {c: pl.Int64 for c in pred.meta.root_names()}
    dnf = convert_expr_to_dnf(pred, schema)
    assert dnf is not None and sum(len(clause) for clause in dnf) <= MAX_DNF_TUPLES
    assert all(("c999", "=", 1) in clause for clause in dnf)
    assert convert_expr_to_dnf(~pred, schema) is None


def _any_of(col: str, n: int) -> pl.Expr:
    return functools.reduce(operator.or_, [pl.col(col) == i for i in range(n)])


@dataclass(frozen=True)
class OverBudgetAndCase:
    """The last disjunction of an AND over the DNF budget, and the bounds each clause must keep from it."""

    id: str
    _: KW_ONLY
    last: pl.Expr
    bounds: list[tuple]


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            OverBudgetAndCase("members", last=_any_of("c", 50), bounds=[("c", "in", list(range(50)))]),
            OverBudgetAndCase(
                "range",
                last=functools.reduce(operator.or_, [pl.col("c").is_between(i, i + 1) for i in range(50)]),
                bounds=[("c", ">=", 0), ("c", "<=", 50)],
            ),
        ]
    ),
)
def test_an_and_over_the_budget_keeps_the_bounds_of_the_side_it_drops(case: OverBudgetAndCase):
    from polars_io_tools.io_sources.dnf_visitor import MAX_DNF_TUPLES

    pred = _any_of("a", 50) & _any_of("b", 50) & case.last
    values = pl.DataFrame({"a": [0, 1, 49, 50, 51, -1, None]}, schema={"a": pl.Int64})
    frame = values.join(values.rename({"a": "b"}), how="cross").join(values.rename({"a": "c"}), how="cross")
    schema = dict(frame.schema)
    dnf = convert_expr_to_dnf(pred, schema)
    assert dnf is not None and sum(len(clause) for clause in dnf) <= MAX_DNF_TUPLES
    assert all(clause[-len(case.bounds) :] == case.bounds for clause in dnf)
    # The other tuples of each clause are those of an exact AND of the first two disjunctions.
    assert _dnf_mask(frame, [case.bounds]).filter(mask(frame, case.last)).all()
    assert convert_expr_to_dnf(~pred, schema) is None


@dataclass(frozen=True)
class ContradictionCase:
    """A filter, and whether ``_is_contradiction`` finds that no row can pass it."""

    id: str
    _: KW_ONLY
    predicate: pl.Expr
    expected: bool
    # the column dtypes passed along, if any
    schema: dict | pl.Schema | None = None


_X_5_TO_6_BUT_NOT_5_OR_6 = pl.col("x").is_between(5, 6, closed="both") & pl.col("x").ne(5) & pl.col("x").ne(6)


_JAN_1_TO_2_2024_BUT_NOT_THOSE_DATETIMES = (
    pl.col("x").is_between(datetime(2024, 1, 1), datetime(2024, 1, 2), closed="both")
    & pl.col("x").ne(datetime(2024, 1, 1))
    & pl.col("x").ne(datetime(2024, 1, 2))
)


_JAN_1_TO_2_2024_BUT_NOT_THOSE_DATES = (
    pl.col("x").is_between(date(2024, 1, 1), date(2024, 1, 2), closed="both") & pl.col("x").ne(date(2024, 1, 1)) & pl.col("x").ne(date(2024, 1, 2))
)


_NOT_JAN_1_2_OR_3_2024 = pl.col("x").ne(date(2024, 1, 1)) & pl.col("x").ne(date(2024, 1, 2)) & pl.col("x").ne(date(2024, 1, 3))


PARSED_FILTER_CONTRADICTIONS = [
    ContradictionCase("single_clause_contradiction", predicate=(pl.col("a") == 5) & (pl.col("a") == 6), expected=True),
    ContradictionCase("single_clause_not_contradiction", predicate=(pl.col("a") == 5) & (pl.col("a") > 0), expected=False),
    ContradictionCase("single_clause_eq_and_gt", predicate=(pl.col("a") == 5) & (pl.col("a") > 5), expected=True),
    ContradictionCase(
        "all_clauses_contradictions", predicate=((pl.col("a") == 5) & (pl.col("a") == 6)) | ((pl.col("b") < 0) & (pl.col("b") > 10)), expected=True
    ),
    ContradictionCase(
        "one_clause_not_contradiction", predicate=((pl.col("a") == 5) & (pl.col("a") == 6)) | ((pl.col("b") > 0) & (pl.col("b") < 10)), expected=False
    ),
    ContradictionCase(
        "no_clause_contradiction", predicate=((pl.col("a") == 5) & (pl.col("a") > 0)) | ((pl.col("b") > 0) & (pl.col("b") < 10)), expected=False
    ),
    ContradictionCase("literal_true", predicate=pl.lit(True), expected=False),
    ContradictionCase("literal_false", predicate=pl.lit(False), expected=True),
    ContradictionCase(
        "nested_all_contradictions",
        predicate=(((pl.col("a") == 1) & (pl.col("a") == 2)) | ((pl.col("b") == 3) & (pl.col("b") == 4))) & ((pl.col("c") > 5) & (pl.col("c") < 0)),
        expected=True,
    ),
    ContradictionCase(
        "nested_one_path_not_contradiction",
        predicate=(((pl.col("a") == 1) & (pl.col("a") > 0)) | ((pl.col("b") == 3) & (pl.col("b") == 4))) & ((pl.col("c") > 0) & (pl.col("c") < 10)),
        expected=False,
    ),
    ContradictionCase("is_null_and_value", predicate=pl.col("x").is_null() & (pl.col("x") == 5), expected=True),
    ContradictionCase("is_null_or_value", predicate=pl.col("x").is_null() | (pl.col("x") == 5), expected=False),
    ContradictionCase("is_not_null_and_is_null", predicate=pl.col("x").is_not_null() & pl.col("x").is_null(), expected=True),
    ContradictionCase("is_not_null_and_value", predicate=pl.col("x").is_not_null() & (pl.col("x") == 5), expected=False),
    ContradictionCase("not_literal_false", predicate=~pl.lit(False), expected=False),
    ContradictionCase("always_true_or", predicate=(pl.col("a") > 0) | (pl.col("a") <= 0), expected=False),
    ContradictionCase("is_between_disjoint", predicate=pl.col("x").is_between(1, 5) & pl.col("x").is_between(10, 15), expected=True),
    ContradictionCase("is_between_and_not_disjoint", predicate=pl.col("x").is_between(1, 5) & ~pl.col("x").is_between(10, 15), expected=False),
    ContradictionCase("is_between_and_not_same", predicate=pl.col("x").is_between(1, 5) & ~pl.col("x").is_between(1, 5), expected=True),
    ContradictionCase("is_between_overlapping", predicate=pl.col("x").is_between(1, 10) & pl.col("x").is_between(5, 15), expected=False),
    ContradictionCase("is_between_value_outside", predicate=pl.col("x").is_between(1, 5) & (pl.col("x") == 10), expected=True),
    ContradictionCase("is_between_value_inside", predicate=pl.col("x").is_between(1, 5) & (pl.col("x") == 3), expected=False),
    ContradictionCase("is_between_reversed_bounds", predicate=pl.col("x").is_between(5, 1) & (pl.col("x") == 3), expected=True),
    ContradictionCase("is_between_inclusive_bound", predicate=pl.col("x").is_between(1, 5, closed="both") & (pl.col("x") == 5), expected=False),
    ContradictionCase("is_between_exclusive_upper_bound", predicate=pl.col("x").is_between(1, 5, closed="left") & (pl.col("x") == 5), expected=True),
    ContradictionCase(
        "is_between_open_bounds_or_either_bound",
        predicate=pl.col("x").is_between(5, 6, closed="none")
        & ((pl.col("x") == 5) | (pl.col("x") == 6))
        & pl.col("x").is_between(4, 7, closed="none"),
        expected=True,
    ),
    ContradictionCase("is_in_disjoint", predicate=pl.col("x").is_in([1, 2, 3]) & pl.col("x").is_in([4, 5, 6]), expected=True),
    ContradictionCase("is_in_overlapping", predicate=pl.col("x").is_in([1, 2, 3]) & pl.col("x").is_in([3, 4, 5]), expected=False),
    ContradictionCase("is_in_value_outside", predicate=pl.col("x").is_in([1, 2, 3]) & (pl.col("x") == 4), expected=True),
    ContradictionCase("is_in_value_inside", predicate=pl.col("x").is_in([1, 2, 3]) & (pl.col("x") == 2), expected=False),
    ContradictionCase("is_in_and_not_same", predicate=pl.col("x").is_in([1, 2, 3]) & pl.col("x").is_in([1, 2, 3]).not_(), expected=True),
    ContradictionCase("is_in_and_not_other", predicate=pl.col("x").is_in([1, 2, 3]) & pl.col("x").is_in([4, 5, 6]).not_(), expected=False),
    ContradictionCase(
        "is_in_excluded_value", predicate=pl.col("x").is_in([1, 2, 3]) & ~pl.col("x").is_in([3, 4, 5]) & (pl.col("x") == 3), expected=True
    ),
    ContradictionCase("is_in_empty", predicate=pl.col("x").is_in([]) & (pl.col("x") == 5), expected=True),
    ContradictionCase("is_in_value_gt_exclusive", predicate=pl.col("x").is_in([5]) & (pl.col("x") > 5), expected=True),
    ContradictionCase("is_in_value_eq", predicate=pl.col("x").is_in([5]) & (pl.col("x") == 5), expected=False),
    # Without a schema the bounds cannot be enumerated.
    ContradictionCase("excluded_bounds_no_schema", predicate=_X_5_TO_6_BUT_NOT_5_OR_6, expected=False),
    ContradictionCase("excluded_bounds_int", predicate=_X_5_TO_6_BUT_NOT_5_OR_6, expected=True, schema=pl.Schema({"x": pl.Int32})),
    ContradictionCase("excluded_bounds_float", predicate=_X_5_TO_6_BUT_NOT_5_OR_6, expected=False, schema=pl.Schema({"x": pl.Float64})),
    ContradictionCase(
        "excluded_bounds_datetime", predicate=_JAN_1_TO_2_2024_BUT_NOT_THOSE_DATETIMES, expected=False, schema=pl.Schema({"x": pl.Datetime()})
    ),
    ContradictionCase("excluded_bounds_date", predicate=_JAN_1_TO_2_2024_BUT_NOT_THOSE_DATES, expected=True, schema=pl.Schema({"x": pl.Date()})),
    ContradictionCase(
        "excluded_dates_closed_left",
        predicate=pl.col("x").is_between(date(2024, 1, 1), date(2024, 1, 4), closed="left") & _NOT_JAN_1_2_OR_3_2024,
        expected=True,
        schema=pl.Schema({"x": pl.Date()}),
    ),
    ContradictionCase(
        "excluded_dates_closed_right",
        predicate=pl.col("x").is_between(date(2024, 1, 1), date(2024, 1, 4), closed="right") & _NOT_JAN_1_2_OR_3_2024,
        expected=False,
        schema=pl.Schema({"x": pl.Date()}),
    ),
]


@pytest.mark.parametrize("case", as_params(PARSED_FILTER_CONTRADICTIONS))
def test_lazy_cache_is_contradiction(case: ContradictionCase):
    assert (_is_contradiction(case.predicate) if case.schema is None else _is_contradiction(case.predicate, case.schema)) == case.expected


def assert_dnf_equal(result: list[list[tuple[str, str, Any]]] | None, expected: list[list[tuple[str, str, Any]]] | None, msg: str | None = None):
    """
    Assert that two DNF expressions are equivalent regardless of predicate order.
    This ignores ordering within each clause (inner list) but preserves clause order.
    """
    if result is None and expected is None:
        return True

    if result is None or expected is None:
        pytest.fail(f"One result is None but the other is not: result={result}, expected={expected}")

    # Convert each clause to a comparable representation
    result_clauses = [sort_clause(clause) for clause in result]
    expected_clauses = [sort_clause(clause) for clause in expected]

    # Check that each clause in result appears in expected
    missing_in_expected = [clause for clause in result_clauses if not clause_in_list(clause, expected_clauses)]
    # Check that each clause in expected appears in result
    missing_in_result = [clause for clause in expected_clauses if not clause_in_list(clause, result_clauses)]

    if missing_in_expected or missing_in_result:
        error_msg = "\nDNF structures don't match!"
        if missing_in_expected:
            error_msg += f"\nOnly in result: {missing_in_expected}"
        if missing_in_result:
            error_msg += f"\nOnly in expected: {missing_in_result}"

        if msg:
            error_msg += f"\n{msg}"

        pytest.fail(error_msg)


def sort_clause(clause: list[tuple[str, str, Any]]) -> list[tuple[str, str, Any]]:
    """
    Sort a clause by its column names and operators for consistent comparison.
    Since we can't hash the entire tuple if some values are unhashable,
    we'll sort by the first two elements (column and operator).
    """
    return sorted(clause, key=lambda x: (x[0], x[1]))


def clause_in_list(clause: list[tuple[str, str, Any]], clause_list: list[list[tuple[str, str, Any]]]) -> bool:
    """
    Check if a clause appears in a list of clauses, with deep comparison for values.
    """
    for other_clause in clause_list:
        if len(clause) != len(other_clause):
            continue

        # Check if all tuples in clause match with tuples in other_clause
        if all(tuples_equal(a, b) for a, b in zip(clause, other_clause)):
            return True

    return False


def tuples_equal(a: tuple[str, str, Any], b: tuple[str, str, Any]) -> bool:
    """
    Compare two predicate tuples, handling special cases for collections and None values.
    """
    if a[0] != b[0] or a[1] != b[1]:  # Column and operator must match exactly
        return False

    # Handle None values
    if a[2] is None and b[2] is None:
        return True

    # Handle collection types (list, tuple, etc.)
    if isinstance(a[2], (list, tuple)) and isinstance(b[2], (list, tuple)):
        if len(a[2]) != len(b[2]):
            return False
        return all(x == y for x, y in zip(a[2], b[2]))

    # For all other types, use regular equality
    return a[2] == b[2]
