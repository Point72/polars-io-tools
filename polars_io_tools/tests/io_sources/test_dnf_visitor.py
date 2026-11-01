"""DNF conversion: tables of predicate -> DNF, DNF negation, contradiction detection, and checks that a DNF keeps every
row its predicate keeps.

A DNF row's ``expect`` is compared ignoring the order of terms inside a clause, or exactly when the row sets ``exact``;
``...`` skips the comparison for rows that only check kept rows.
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from polars_io_tools.io_sources.base import get_parsed_expr
from polars_io_tools.io_sources.dnf_visitor import (
    DNFVisitor,
    _is_contradiction,
    combine_and_dnf,
    convert_expr_to_dnf,
    is_contradiction,
    negate_dnf,
)
from polars_io_tools.tests.helpers.assertions import assert_keeps_rows
from polars_io_tools.tests.helpers.cases import Case, params
from polars_io_tools.tests.io_sources.conftest import assert_dnf_equal

c = pl.col
x, y, z, a, s = c("x"), c("y"), c("z"), c("a"), c("s")
B = pl.Boolean


def D(id, pred, expect=..., schema=None, df=None, exact=False):
    return Case(id, pred, expect, {"schema": schema, "df": df, "exact": exact})


_NULL_MIX = pl.DataFrame({"a": [1, 1, 2, None, 257], "b": ["xx", "xyz", "xx", "xx", "xx"]})
_FLAG = pl.Schema({"flag": pl.Boolean})
_Y2 = date(2023, 1, 1), date(2023, 12, 31)
_NY = ZoneInfo("America/New_York")
_EARLY = datetime(2024, 11, 3, 1, 30, tzinfo=_NY)
_NAN = float("nan")


DNF_CASES = [
    # comparisons
    D("basic_comparisons-eq", x == 10, [[("x", "=", 10)]]),
    D("basic_comparisons-neq", y != "test", [[("y", "!=", "test")]]),
    D("basic_comparisons-gt", z > 5, [[("z", ">", 5)]]),
    D("basic_comparisons-lt", a < 100, [[("a", "<", 100)]]),
    D("basic_comparisons-gte", c("b") >= 20, [[("b", ">=", 20)]]),
    D("basic_comparisons-lte", c("c") <= 50, [[("c", "<=", 50)]]),
    D("basic_comparisons-reversed", 5 == x, [[("x", "=", 5)]]),
    D("literal_on_left_is_flipped-lt", pl.lit(1) < a, [[("a", ">", 1)]]),
    D("literal_on_left_is_flipped-ge", pl.lit(5) >= a, [[("a", "<=", 5)]]),
    D("equal_date", c("date") == date(2023, 1, 1), [[("date", "=", date(2023, 1, 1))]], exact=True),
    D("expr_with_col_comparisons-and", (a > c("b")) & (c("c") == 12), [[("c", "=", 12)]]),
    D("expr_with_col_comparisons-or", (a > c("b")) | (c("c") == 12), None),
    D("unsupported_expressions", x.cast(pl.Int64) + 5, None),
    D("literal_comparisons", pl.lit(5) > pl.lit(3), None),
    # Boolean columns: a bare column is ``== True`` and its negation ``== False``; Polars >= 2.1 lowers
    # ``col == True`` to the bare column and ``col == False`` to ``col.not()``.
    D("boolean_column_predicate-bare", c("flag"), [[("flag", "=", True)]], schema=_FLAG, exact=True),
    D("boolean_column_predicate-not", c("flag").not_(), [[("flag", "=", False)]], schema=_FLAG, exact=True),
    D("boolean_column_predicate-invert", ~c("flag"), [[("flag", "=", False)]], schema=_FLAG, exact=True),
    D("boolean_column_predicate-eq_true", c("flag") == True, [[("flag", "=", True)]], schema=_FLAG, exact=True),
    D("boolean_column_predicate-eq_false", c("flag") == False, [[("flag", "=", False)]], schema=_FLAG, exact=True),
    D("boolean_column_predicate-no_schema", c("flag"), [[("flag", "=", True)]], exact=True),
    # ``cast(Boolean)`` on an integer column is ``x != 0``, not ``x == True``.
    D("non_boolean_cast_to_boolean_not_misread-schema", x.cast(B), None, schema=pl.Schema({"x": pl.Int64})),
    D("non_boolean_cast_to_boolean_not_misread-no_schema", x.cast(B), None),
    D("non_boolean_cast_to_boolean_not_misread-and_eq", x.cast(B) & (x == 2), [[("x", "=", 2)]], exact=True),
    # is_between
    D("is_between_date", c("date").is_between(*_Y2), [[("date", ">=", _Y2[0]), ("date", "<=", _Y2[1])]]),
    D(
        "is_between_datetime",
        c("timestamp").is_between(datetime(2023, 1, 1, 8, 30), datetime(2023, 12, 31, 17, 45, 30)),
        [[("timestamp", ">=", datetime(2023, 1, 1, 8, 30)), ("timestamp", "<=", datetime(2023, 12, 31, 17, 45, 30))]],
        exact=True,
    ),
    *[
        D(
            f"is_between_date_closed_parameters-{closed}",
            c("date").is_between(*_Y2, closed=closed),
            [[("date", lo, _Y2[0]), ("date", hi, _Y2[1])]],
            exact=True,
        )
        for closed, lo, hi in [("both", ">=", "<="), ("left", ">=", "<"), ("right", ">", "<="), ("none", ">", "<")]
    ],
    D(
        "is_between_same_day-both",
        c("date").is_between(date(2023, 6, 15), date(2023, 6, 15)),
        [[("date", ">=", date(2023, 6, 15)), ("date", "<=", date(2023, 6, 15))]],
        exact=True,
    ),
    D(
        "is_between_same_day-none",
        c("date").is_between(date(2023, 6, 15), date(2023, 6, 15), closed="none"),
        [[("date", ">", date(2023, 6, 15)), ("date", "<", date(2023, 6, 15))]],
        exact=True,
    ),
    # literal types are kept: a date never equals a datetime
    D(
        "is_between_mixed_date_types",
        c("dt").is_between(date(2023, 1, 1), datetime(2023, 12, 31, 23, 59, 59)),
        [[("dt", ">=", date(2023, 1, 1)), ("dt", "<=", datetime(2023, 12, 31, 23, 59, 59))]],
        exact=True,
    ),
    D(
        "is_between_with_polars_date_literals",
        c("date").is_between(pl.lit(_Y2[0]), pl.lit(_Y2[1])),
        [[("date", ">=", _Y2[0]), ("date", "<=", _Y2[1])]],
        exact=True,
    ),
    # logical operations
    D("logical_operations-simple_and", (x == 10) & (y == "test"), [[("x", "=", 10), ("y", "=", "test")]]),
    D("logical_operations-nested_and", (x == 10) & (y == "test") & (z > 5), [[("x", "=", 10), ("y", "=", "test"), ("z", ">", 5)]]),
    D("logical_operations-simple_or", (x == 10) | (y == "test"), [[("x", "=", 10)], [("y", "=", "test")]]),
    D("logical_operations-nested_or", (x == 10) | (y == "test") | (z > 5), [[("x", "=", 10)], [("y", "=", "test")], [("z", ">", 5)]]),
    D(
        "logical_operations-and_or_mixed",
        (x == 10) & ((y == "test") | (z > 5)),
        [[("x", "=", 10), ("y", "=", "test")], [("x", "=", 10), ("z", ">", 5)]],
    ),
    D("logical_operations-or_and_mixed", (x == 10) | ((y == "test") & (z > 5)), [[("x", "=", 10)], [("y", "=", "test"), ("z", ">", 5)]]),
    D("any_horizontal_nested_with_and", pl.any_horizontal(x > 5, (y == 10) & (z < 3)), [[("x", ">", 5)], [("y", "=", 10), ("z", "<", 3)]]),
    # functions
    D("function_expressions-is_in", x.is_in((1, 2, 3)), [[("x", "in", [1, 2, 3])]]),
    D("function_expressions-is_null", y.is_null(), [[("y", "is", None)]]),
    D("function_expressions-is_not_null", z.is_not_null(), [[("z", "is not", None)]]),
    D("function_expressions-starts_with", s.str.starts_with("prefix"), [[("s", "~", "^prefix")]]),
    D("function_expressions-ends_with", s.str.ends_with("suffix"), [[("s", "~", "suffix$")]]),
    D("function_expressions-contains", s.str.contains("substr"), [[("s", "~", "substr")]]),
    D("collections_in_is_in-list", x.is_in([1, 2, 3]), [[("x", "in", [1, 2, 3])]]),
    D("collections_in_is_in-tuple", x.is_in((4, 5, 6)), [[("x", "in", [4, 5, 6])]]),
    D("collections_in_is_in-empty", x.is_in([]), [[("x", "in", [])]]),
    D("is_in_single_item", x.is_in(["US"]), [[("x", "in", ["US"])]]),
    D("negated_operators", ~x.is_in([1, 2, 3]), [[("x", "!in", [1, 2, 3])]]),
    D("eq_missing_and_ne_missing_with_null_literal-eq", a.eq_missing(None), [[("a", "is", None)]]),
    D("eq_missing_and_ne_missing_with_null_literal-ne", a.ne_missing(None), [[("a", "is not", None)]]),
    D(
        "complex_expressions-complex_1",
        ((a > 5) & (c("b") <= 10)) | ((c("c") == "test") & c("d").is_not_null()),
        [[("a", ">", 5), ("b", "<=", 10)], [("c", "=", "test"), ("d", "is not", None)]],
    ),
    D(
        "complex_expressions-complex_2",
        ((a > 5) & (c("b") <= 10)) | (((c("c") == "test") | c("d").ne(True)) & c("e").is_null()),
        [[("a", ">", 5), ("b", "<=", 10)], [("c", "=", "test"), ("e", "is", None)], [("d", "!=", True), ("e", "is", None)]],
    ),
    D("complex_expressions-complex_3", c("e").is_in([1, 2, 3]) & c("f").str.contains("xyz"), [[("e", "in", [1, 2, 3]), ("f", "~", "xyz")]]),
    # Boolean casts are transparent, other casts are not
    D("supported_cast", (x == 0).cast(B), [[("x", "=", 0)]]),
    D("complex_expression_with_boolean_cast", ((x > 5) & (y == "test")).cast(B), [[("x", ">", 5), ("y", "=", "test")]]),
    D("nested_boolean_casts", (x == 0).cast(B).cast(B), [[("x", "=", 0)]]),
    D("cast_other_than_boolean", (x == 0).cast(pl.Int32), None),
    D("boolean_cast_in_logical_operation", (x > 5).cast(B) & (y == "test"), [[("x", ">", 5), ("y", "=", "test")]]),
    D("value_preserving_widening_cast_still_pushes_down-no_schema", a.cast(pl.Int64) == 1, None),
    D("value_preserving_widening_cast_still_pushes_down-schema", a.cast(pl.Int64) == 1, [[("a", "=", 1)]], schema={"a": pl.Int8}),
    D(
        "sub_microsecond_is_in_literal_is_not_pushed",
        c("t").is_in(pl.concat_list(pl.lit(1704198600000000500).cast(pl.Datetime("ns")))),
        None,
        schema={"t": pl.Datetime("ns")},
    ),
    # aliases are transparent
    D("alias_simple_equality", x.alias("x_alias") == 10, [[("x", "=", 10)]]),
    D("alias_logical_operations", (x.alias("x_alias") == 10) & (y.alias("y_alias") > 5), [[("x", "=", 10), ("y", ">", 5)]]),
    # ternaries: a null condition takes the otherwise branch
    D(
        "ternary_basic",
        pl.when(x > 5).then(y == 10).otherwise(z < 3),
        [[("x", ">", 5), ("y", "=", 10)], [("x", "<=", 5), ("z", "<", 3)], [("x", "is", None), ("z", "<", 3)]],
    ),
    D(
        "boolean_cast_in_ternary_expression",
        pl.when((x > 5).cast(B)).then(y == 10).otherwise(z < 3),
        [[("x", ">", 5), ("y", "=", 10)], [("x", "<=", 5), ("z", "<", 3)], [("x", "is", None), ("z", "<", 3)]],
    ),
    D(
        "ternary_with_complex_predicate",
        pl.when((x > 5) & (y < 10)).then(z == 15).otherwise(a != 0),
        [
            [("x", ">", 5), ("y", "<", 10), ("z", "=", 15)],
            [("x", "<=", 5), ("a", "!=", 0)],
            [("y", ">=", 10), ("a", "!=", 0)],
            [("a", "!=", 0), ("x", "is", None)],
            [("a", "!=", 0), ("y", "is", None)],
        ],
    ),
    D(
        "ternary_with_function_expressions",
        pl.when(x.is_in([1, 2, 3])).then((y > 5) & (z == 0)).otherwise(c("w").is_null()),
        [
            [("x", "in", [1, 2, 3]), ("y", ">", 5), ("z", "=", 0)],
            [("x", "!in", [1, 2, 3]), ("w", "is", None)],
            [("w", "is", None), ("x", "is", None)],
        ],
    ),
    D(
        "ternary_nested",
        pl.when(x < 0).then(pl.when(y > 10).then(z == 5).otherwise(z == 0)).otherwise(c("w").is_not_null()),
        [
            [("x", "<", 0), ("y", ">", 10), ("z", "=", 5)],
            [("x", "<", 0), ("y", "<=", 10), ("z", "=", 0)],
            [("x", ">=", 0), ("w", "is not", None)],
            [("x", "<", 0), ("y", "is", None), ("z", "=", 0)],
            [("w", "is not", None), ("x", "is", None)],
        ],
    ),
    D("ternary_with_empty_clauses", pl.when(x > 5).then(pl.lit(True)).otherwise(pl.lit(False)), [[("x", ">", 5)]]),
    D(
        "ternary_chained_when",
        pl.when(x > 10).then(pl.when(y < 5).then(z == 1).otherwise(z == 2)).otherwise(z == 3),
        [
            [("x", ">", 10), ("y", "<", 5), ("z", "=", 1)],
            [("x", ">", 10), ("y", ">=", 5), ("z", "=", 2)],
            [("x", "<=", 10), ("z", "=", 3)],
            [("x", ">", 10), ("y", "is", None), ("z", "=", 2)],
            [("x", "is", None), ("z", "=", 3)],
        ],
    ),
    D(
        "ternary_multiple_chained_when",
        pl.when(x > 100)
        .then(c("result") == "high")
        .when(x > 50)
        .then(c("result") == "medium")
        .when(x > 10)
        .then(c("result") == "low")
        .otherwise(c("result") == "very low"),
        [
            [("x", ">", 100), ("result", "=", "high")],
            [("x", "<=", 100), ("x", ">", 50), ("result", "=", "medium")],
            [("x", "<=", 100), ("x", "<=", 50), ("x", ">", 10), ("result", "=", "low")],
            [("x", "<=", 100), ("x", "<=", 50), ("x", "<=", 10), ("result", "=", "very low")],
            [("result", "=", "very low"), ("x", "is", None), ("x", "is", None), ("x", "is", None)],
        ],
    ),
    D(
        "ternary_multiple_conditions_in_when",
        pl.when(x > 10, y < 5).then(z == 1).otherwise(z == 2),
        [
            [("x", ">", 10), ("y", "<", 5), ("z", "=", 1)],
            [("x", "<=", 10), ("z", "=", 2)],
            [("y", ">=", 5), ("z", "=", 2)],
            [("x", "is", None), ("z", "=", 2)],
            [("y", "is", None), ("z", "=", 2)],
        ],
    ),
    D(
        "ternary_multiple_when_with_multiple_conditions",
        pl.when(x > 100, y == "A")
        .then(c("result") == "category_1")
        .when(x > 50, y == "B")
        .then(c("result") == "category_2")
        .otherwise(c("result") == "category_3"),
        [
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
    D(
        "ternary_nested_with_multiple_conditions",
        pl.when(c("category") == "A", c("value") > 100)
        .then(pl.when(c("subcategory") == "X", c("priority") > 3).then(c("result") == "high_priority_A").otherwise(c("result") == "normal_A"))
        .otherwise(pl.when(c("category") == "B").then(c("result") == "any_B").otherwise(c("result") == "other")),
        [
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
    # Nulls and partial conversions: the DNF must keep every row of ``df`` that the predicate keeps.
    D("not_over_partially_converted_and_does_not_drop_rows", ~((a == 1) & (c("b").str.len_chars() == 2)), None, df=_NULL_MIX),
    D("ne_missing_keeps_null_rows", a.ne_missing(1), [[("a", "!=", 1)], [("a", "is", None)]], df=_NULL_MIX),
    D("negated_is_in_nulls_equal_keeps_nulls", ~a.is_in([1], nulls_equal=True), None, df=_NULL_MIX),
    D(
        "is_in_nulls_equal_with_null_value_keeps_nulls",
        a.is_in([1, None], nulls_equal=True),
        [[("a", "in", [1])], [("a", "is", None)]],
        df=pl.DataFrame({"a": [1, 2, None]}, schema={"a": pl.Int64}),
    ),
    # A narrowing cast is refused even with a schema: 257 wraps to 1 under Int8.
    D("narrowing_cast_on_column_is_not_stripped-no_schema", a.cast(pl.Int8, wrap_numerical=True) == 1, None, df=_NULL_MIX),
    D("narrowing_cast_on_column_is_not_stripped-schema", a.cast(pl.Int8, wrap_numerical=True) == 1, None, schema={"a": pl.Int64}, df=_NULL_MIX),
    D("ternary_otherwise_keeps_rows_where_condition_is_null", pl.when(a >= 5).then(False).otherwise(True), df=pl.DataFrame({"a": [None, 1, 5]})),
    D("regex_contains_is_not_escaped-regex", s.str.contains("A|B"), df=pl.DataFrame({"s": ["A", "B", "A|B", "C"]})),
    D("regex_contains_is_not_escaped-literal", s.str.contains("A|B", literal=True), df=pl.DataFrame({"s": ["A", "B", "A|B", "C"]})),
]


def _dnf_to_expr(dnf: list[list[tuple]]) -> pl.Expr:
    """A DNF materialized back into a Polars predicate (null -> drop, like a scan filter)."""

    def term(col, op, value):
        col = pl.col(col)
        if op == "in":
            return col.is_in(list(value))
        if op in ("not in", "!in"):
            return ~col.is_in(list(value))
        if op == "is":
            return col.is_null()
        if op in ("is not", "!is"):
            return col.is_not_null()
        return _DNF_OPS[op](col, value)

    return pl.any_horizontal([pl.all_horizontal([term(*t) for t in clause]) for clause in dnf]).fill_null(False)


_DNF_OPS = {
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
    assert_keeps_rows(df, pred, None if dnf is None else _dnf_to_expr(dnf))


@pytest.mark.parametrize(("pred", "schema", "df", "exact", "expect"), params(DNF_CASES, "pred", "schema", "df", "exact", "expect"))
def test_dnf(pred, schema, df, exact, expect):
    dnf = convert_expr_to_dnf(pred, schema)
    if exact:
        assert dnf == expect
    elif expect is not ...:
        assert_dnf_equal(dnf, expect)
    if df is not None:
        _assert_dnf_keeps_rows(df, pred, dnf)


_X = pl.col("x")


@pytest.mark.parametrize(
    ("df", "pred"),
    [
        (pl.DataFrame({"x": [_NAN, 1.0, 2.0]}), (_X == _NAN) & (_X == _NAN)),
        (pl.DataFrame({"x": [_NAN, 1.0, 2.0]}), _X.is_in([_NAN, 1.0]) & _X.is_in([_NAN, 2.0])),
        (pl.DataFrame({"x": [_EARLY, _EARLY.replace(fold=1)]}), (_X == _EARLY) & (_X != _EARLY.replace(fold=1))),
        (pl.DataFrame({"x": pl.Series(["z", "a", "m"], dtype=pl.Enum(["z", "a", "m"]))}), _X.is_in(["z", "m"]) & (_X < "a")),
        (pl.DataFrame({"x": [-1, 0, 1, None]}), pl.when(_X.is_between(None, 0)).then(False).otherwise(True)),
        (pl.DataFrame({"x": [date(2024, 1, 1)]}), pl.when(_X == date(2024, 1, 1)).then(_X >= datetime(2024, 1, 1)).otherwise(False)),
        (pl.DataFrame({"x": [date(2024, 1, 1)]}), (_X == date(2024, 1, 1)) & (_X == datetime(2024, 1, 1))),
        (pl.DataFrame({"x": pl.Series([0.1], dtype=pl.Float32)}), (_X == 0.1) & (_X == 0.10000000149011612)),
        (pl.DataFrame({"x": pl.Series([0.1], dtype=pl.Float32)}), pl.when(_X == 0.1).then(_X == 0.10000000149011612).otherwise(False)),
        (pl.DataFrame({"x": [2]}), (_X > 1.5) & (_X < 2.5)),
        (pl.DataFrame({"x": [2]}), (_X > float("-inf")) & (_X < float("inf"))),
        (pl.DataFrame({"x": ["a", "b"]}), _X.str.contains("(?x)a # comment")),
        (pl.DataFrame({"x": ["a", "["]}), _X.str.contains("[", strict=False)),
        (
            pl.DataFrame({"x": pl.Series([datetime(2024, 1, 1)], dtype=pl.Datetime("ms"))}),
            (_X == datetime(2024, 1, 1, microsecond=1)) & (_X == datetime(2024, 1, 1, microsecond=2)),
        ),
        (pl.DataFrame({"x": [float(2**53)]}), (_X == 2**53 + 1) & (_X == float(2**53))),
    ],
    ids=[
        "nan_eq",
        "nan_is_in",
        "dst_fold",
        "enum_order",
        "is_between_null_bound",
        "date_vs_datetime_bound",
        "date_vs_datetime_eq",
        "float32_eq",
        "float32_ternary",
        "fractional_int_bounds",
        "infinite_int_bounds",
        "verbose_regex",
        "non_strict_regex",
        "literal_finer_than_unit",
        "int_vs_float_beyond_2_53",
    ],
)
def test_dnf_and_contradiction_keep_matching_rows(df, pred):
    """Python value comparisons must not prove a contradiction or drop rows that Polars keeps."""
    assert df.filter(pred).is_empty() or _is_contradiction(pred, df.schema) is False
    _assert_dnf_keeps_rows(df, pred, convert_expr_to_dnf(pred, df.schema))


@pytest.mark.parametrize("dtype", [pl.Time, pl.Datetime("ns"), pl.Duration("ns")])
@pytest.mark.parametrize("literal_left", [False, True])
@pytest.mark.parametrize("operation", ["=", "!=", "<", "<=", ">", ">="])
@pytest.mark.parametrize("tick", [1, 1000], ids=["submicrosecond", "microsecond"])
def test_temporal_comparison_dnf_preserves_nanoseconds(dtype, literal_left, operation, tick):
    frame = pl.DataFrame({"value": pl.Series([0, 1, 2, 1000, 1001], dtype=pl.Int64).cast(dtype)})
    literal = pl.lit(tick).cast(dtype)
    left, right = (literal, pl.col("value")) if literal_left else (pl.col("value"), literal)
    predicate = _DNF_OPS[operation](left, right)
    dnf = convert_expr_to_dnf(predicate)
    if tick == 1000:
        assert dnf is not None
    if dnf is not None:
        dnf = [[(name, comparison, pl.lit(value, dtype=frame.schema[name])) for name, comparison, value in clause] for clause in dnf]
    _assert_dnf_keeps_rows(frame, predicate, dnf)


def test_dnf_visitor_direct():
    visitor = DNFVisitor()
    visitor.visit(get_parsed_expr((x == 10) & (y > 5)))
    assert visitor.process_results() == [[("x", "=", 10), ("y", ">", 5)]]


def test_combine_and_dnf():
    one, two = [[("x", "=", 10)]], [[("y", "=", "test")]]
    assert_dnf_equal(combine_and_dnf(one, two), [[("x", "=", 10), ("y", "=", "test")]])
    either, right = [[("x", "=", 10)], [("y", "=", "test")]], [[("z", ">", 5)]]
    assert_dnf_equal(combine_and_dnf(either, right), [[("x", "=", 10), ("z", ">", 5)], [("y", "=", "test"), ("z", ">", 5)]])
    assert combine_and_dnf(None, right) == right
    assert combine_and_dnf(either, None) == either
    assert combine_and_dnf(None, None) is None


@pytest.mark.parametrize(
    ("dnf", "expect"),
    [
        pytest.param([[("x", "=", 0)]], [[("x", "!=", 0)]], id="single_predicate"),
        pytest.param([[("x", "=", 0), ("y", ">=", 5)]], [[("x", "!=", 0)], [("y", "<", 5)]], id="single_conjunction"),
        pytest.param(
            [[("x", "=", 0), ("y", ">=", 5)], [("z", "<", 10)]],
            [[("x", "!=", 0), ("z", ">=", 10)], [("y", "<", 5), ("z", ">=", 10)]],
            id="two_clauses",
        ),
        pytest.param(
            [[("a", "=", 1), ("b", ">", 2)], [("c", "<=", 3)], [("d", "in", [4, 5])]],
            [[("a", "!=", 1), ("c", ">", 3), ("d", "!in", [4, 5])], [("b", "<=", 2), ("c", ">", 3), ("d", "!in", [4, 5])]],
            id="complex_dnf",
        ),
        pytest.param(
            [[("x", "=", 0)], [("y", "<>", 5)], [("z", "in", [1, 2, 3])]],
            [[("x", "!=", 0), ("y", "=", 5), ("z", "!in", [1, 2, 3])]],
            id="various_dnf_forms-eq_ne_in",
        ),
        pytest.param(
            [[("x", "!=", 0)], [("y", "not in", [1, 2])], [("z", "is not", None)]],
            [[("x", "=", 0), ("y", "in", [1, 2]), ("z", "is", None)]],
            id="various_dnf_forms-ne_not_in_is_not",
        ),
        pytest.param(
            [[("x", "=", 0), ("y", "=", 0)], [("z", "=", 0)]],
            [[("x", "!=", 0), ("z", "!=", 0)], [("y", "!=", 0), ("z", "!=", 0)]],
            id="various_dnf_forms-and_or",
        ),
    ],
)
def test_negate_dnf(dnf, expect):
    assert negate_dnf(dnf) == expect


def T(column: str, op: str, value) -> tuple:
    return (column, op, value)


_INT_X = pl.Schema({"x": pl.Int64})

# (description, clause, schema, is a contradiction)
CONTRADICTIONS = [
    ("Empty clause (always true)", [], None, False),
    # Exact values
    ("Exact: col = 5 AND col = 6", [T("colA", "=", 5), T("colA", "=", 6)], None, True),
    ("Exact: col = 5 AND col != 5", [T("colA", "=", 5), T("colA", "!=", 5)], None, True),
    ("Exact: col = 5 AND col = 5", [T("colA", "=", 5), T("colA", "=", 5)], None, False),
    ("Exact: col == 5 AND col <> 5", [T("colA", "==", 5), T("colA", "<>", 5)], None, True),
    ("Exact: x = 2", [T("x", "=", 2)], None, False),
    # Ranges
    ("Range: col > 10 AND col < 5", [T("colA", ">", 10), T("colA", "<", 5)], None, True),
    ("Range: col > 5 AND col < 5", [T("colA", ">", 5), T("colA", "<", 5)], None, True),
    ("Range: col >= 5 AND col <= 5", [T("colA", ">=", 5), T("colA", "<=", 5)], None, False),
    ("Range: col > 5 AND col <= 5", [T("colA", ">", 5), T("colA", "<=", 5)], None, True),
    ("Range: col >= 5 AND col < 5", [T("colA", ">=", 5), T("colA", "<", 5)], None, True),
    ("Range: col > 5 AND col > 10", [T("colA", ">", 5), T("colA", ">", 10)], None, False),
    ("Range: col < 10 AND col < 5", [T("colA", "<", 10), T("colA", "<", 5)], None, False),
    (
        "Range: col < datetime(2024, 1, 1) AND col >= datetime(2024, 1, 2)",
        [T("colA", "<", datetime(2024, 1, 1)), T("colA", ">=", datetime(2024, 1, 2))],
        None,
        True,
    ),
    (
        "Range: col < datetime(2024, 1, 1) AND col >= datetime(2022, 1, 2)",
        [T("colA", "<", datetime(2024, 1, 1)), T("colA", ">=", datetime(2022, 1, 2))],
        None,
        False,
    ),
    ("is_contradiction_obvious_bound_conflict: x > 5 AND x < 3", [T("x", ">", 5), T("x", "<", 3)], None, True),
    # Exact value vs range
    ("Exact vs Range: col = 3 AND col > 5", [T("colA", "=", 3), T("colA", ">", 5)], None, True),
    ("Exact vs Range: col = 7 AND col > 5 AND col < 10", [T("colA", "=", 7), T("colA", ">", 5), T("colA", "<", 10)], None, False),
    ("Exact vs Range: col = 5 AND col >= 5", [T("colA", "=", 5), T("colA", ">=", 5)], None, False),
    ("Exact vs Range: col = 5 AND col > 5", [T("colA", "=", 5), T("colA", ">", 5)], None, True),
    ("Exact vs Range: col = 5 AND col <= 5", [T("colA", "=", 5), T("colA", "<=", 5)], None, False),
    ("Exact vs Range: col = 5 AND col < 5", [T("colA", "=", 5), T("colA", "<", 5)], None, True),
    # Inclusion and exclusion
    ("In/Ex: col IN (1,2) AND col = 3", [T("colA", "in", [1, 2]), T("colA", "=", 3)], None, True),
    ("In/Ex: col IN (1,2) AND col = 1", [T("colA", "in", [1, 2]), T("colA", "=", 1)], None, False),
    ("In/Ex: col IN (1,2) AND col NOT IN (1,2,3)", [T("colA", "in", [1, 2]), T("colA", "!in", [1, 2, 3])], None, True),
    ("In/Ex: col IN (1,2) AND col NOT IN (3,4)", [T("colA", "in", [1, 2]), T("colA", "!in", [3, 4])], None, False),
    ("In/Ex: col IN (1,2) AND col IN (3,4)", [T("colA", "in", [1, 2]), T("colA", "in", [3, 4])], None, True),
    ("In/Ex: col IN (1,2,3) AND col IN (3,4,5)", [T("colA", "in", [1, 2, 3]), T("colA", "in", [3, 4, 5])], None, False),
    (
        "In/Ex: col IN (1,2,3) AND col IN (3,4,5) AND col = 3",
        [T("colA", "in", [1, 2, 3]), T("colA", "in", [3, 4, 5]), T("colA", "=", 3)],
        None,
        False,
    ),
    ("In/Ex: col IN (1,2,3) AND col IN (3,4,5) AND col = 1", [T("colA", "in", [1, 2, 3]), T("colA", "in", [3, 4, 5]), T("colA", "=", 1)], None, True),
    (
        "In/Ex: col IN (1,2,3) AND col IN (3,4,5) AND col NOT IN (3)",
        [T("colA", "in", [1, 2, 3]), T("colA", "in", [3, 4, 5]), T("colA", "!in", [3])],
        None,
        True,
    ),
    # NULL
    ("NULL: col IS NULL AND col = 5", [T("colA", "is", None), T("colA", "=", 5)], None, True),
    ("NULL: col IS NULL AND col > 0", [T("colA", "is", None), T("colA", ">", 0)], None, True),
    ("NULL: col IS NULL AND col IN (1,2)", [T("colA", "is", None), T("colA", "in", [1, 2])], None, True),
    ("NULL: col IS NULL AND col IS NOT NULL", [T("colA", "is", None), T("colA", "is not", None)], None, True),
    ("NULL: col IS NOT NULL AND col = None (exact value is None)", [T("colA", "is not", None), T("colA", "=", None)], None, True),
    ("NULL: col IS NOT NULL AND col IN (None,)", [T("colA", "is not", None), T("colA", "in", [None])], None, True),
    ("NULL: col IS NOT NULL AND col IN (None, 1)", [T("colA", "is not", None), T("colA", "in", [None, 1])], None, False),
    ("NULL: col IS NULL (no contradiction)", [T("colA", "is", None)], None, False),
    ("NULL: col IS NOT NULL (no contradiction)", [T("colA", "is not", None)], None, False),
    ("NULL: col IS NULL AND col IN (None,)", [T("colA", "is", None), T("colA", "in", [None])], None, False),
    # Complex, single column
    ("Complex: col > 0 AND col < 10 AND col IN (20,30)", [T("colA", ">", 0), T("colA", "<", 10), T("colA", "in", [20, 30])], None, True),
    ("Complex: col > 0 AND col < 10 AND col IN (5, 20)", [T("colA", ">", 0), T("colA", "<", 10), T("colA", "in", [5, 20])], None, False),
    ("Complex: col = 5 AND col != 5 AND col > 0", [T("colA", "=", 5), T("colA", "!=", 5), T("colA", ">", 0)], None, True),
    # Multiple columns
    ("Multi-col: colA = 1 AND colA = 2 AND colB = 10", [T("colA", "=", 1), T("colA", "=", 2), T("colB", "=", 10)], None, True),
    ("Multi-col: colA = 1 AND colB > 5", [T("colA", "=", 1), T("colB", ">", 5)], None, False),
    (
        "Multi-col: colA > 0 AND colA < 1, colB = 1 (colA contradiction only for int)",
        [T("colA", ">", 0), T("colA", "<", 1), T("colB", "=", 1)],
        None,
        False,
    ),
    (
        "Multi-col: colA > 0 AND colA < 12, colB = 1, colA < -1 (colA contradiction)",
        [T("colA", ">", 0), T("colA", "<", 1), T("colA", "<", -1), T("colB", "=", 1)],
        None,
        True,
    ),
    # Non-contradictions
    ("Non-C: col = 5 AND col > 0", [T("colA", "=", 5), T("colA", ">", 0)], None, False),
    ("Non-C: col > 0 AND col < 10", [T("colA", ">", 0), T("colA", "<", 10)], None, False),
    ("Non-C: col IN (1,2,3) AND col = 1", [T("colA", "in", [1, 2, 3]), T("colA", "=", 1)], None, False),
    ("Non-C: col IS NOT NULL AND col > 5", [T("colA", "is not", None), T("colA", ">", 5)], None, False),
    ("Non-C: colA = 1, colB = 'test', colC > 3.0", [T("colA", "=", 1), T("colB", "=", "test"), T("colC", ">", 3.0)], None, False),
    # A scalar IN value is wrapped in a list
    ("Edge IN: col IN 'abc' (scalar string)", [T("colA", "in", "abc")], None, False),
    ("Edge IN: col IN 1 (scalar int)", [T("colA", "in", 1)], None, False),
    # IS with a value other than None
    ("IS non-None: colA IS TRUE", [T("colA", "is", True)], None, False),
    ("IS non-None: colA IS TRUE AND colA IS NULL", [T("colA", "is", True), T("colA", "is", None)], None, True),
    ("IS NOT non-None: colA IS NOT TRUE", [T("colA", "is not", True)], None, False),
    # Enumerating an integer range needs the column's dtype: unknown -> not proven a contradiction.
    (
        "is_contradiction_with_column_not_in_schema",
        [T("unknown_col", ">", 3), T("unknown_col", "<", 10), T("unknown_col", "!=", 5)],
        pl.Schema({"other_col": pl.Int64}),
        False,
    ),
    ("is_contradiction_with_column_in_schema", [T("x", ">", 3), T("x", "<", 10), *(T("x", "!=", v) for v in range(4, 10))], _INT_X, True),
]


@pytest.mark.parametrize(("clause", "schema", "expect"), [pytest.param(*row[1:], id=row[0]) for row in CONTRADICTIONS])
def test_is_contradiction(clause, schema, expect):
    assert is_contradiction(clause, schema) is expect


_YP = pl.Schema({"x": pl.Int64, "y": pl.String, "p": pl.Boolean, "x2": pl.Int64, "y2": pl.String, "p2": pl.Boolean})


@pytest.mark.parametrize(
    ("pred", "schema", "expect"),
    [
        pytest.param(
            c("y").is_in(["a", "b"]) & c("a").eq(5) & (c("a").ne(5) | c("y").is_in(["b", "a"]).not_()),
            pl.Schema({"a": pl.Boolean, "y": pl.String}),
            True,
            id="Multi option is in check",
        ),
        pytest.param(
            c("y").is_in(["a", "b"]) & c("p").eq(True) & (c("p").ne(True) | c("y").is_in(["b", "a"]).not_()),
            _YP,
            True,
            id="Compare with True (python boolean)",
        ),
        pytest.param(
            c("y").is_in(["a", "b"]) & c("p").eq(pl.lit(True)) & pl.Expr.or_(c("p").ne(pl.lit(True)), ~c("y").is_in(["b", "a"])),
            _YP,
            True,
            id="Compare with pl.lit(True)",
        ),
    ],
)
def test_is_contradiction_pl(pred, schema, expect):
    assert _is_contradiction(pred, schema) == expect
