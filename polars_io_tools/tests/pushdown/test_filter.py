"""Filters on a source: which ones Polars hands to the source, and in what form.

The table holds filters on a sample source (the ``tracked_sample_data`` fixture) or on a small frame, whether Polars
hands each one to the source and, for some, the shape the filter it hands over parses into. Where Polars versions
differ, the row names the change from ``helpers.polars_versions``.

The last test runs ``all_horizontal`` and ``any_horizontal`` over integer columns cast to Boolean through four of the
library's sources.
"""

import logging
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from datetime import date, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_io_tools.io_sources.base import (
    AnonymousFunctionNode,
    BinaryExprNode,
    FunctionNode,
    OperatorType,
    convert_datetime_to_polars,
    get_parsed_expr,
)
from polars_io_tools.io_sources.enum import BooleanFunctionType, GenericFunctionType
from polars_io_tools.io_sources.lazy_cache_memory import cache_memory
from polars_io_tools.io_sources.lazy_cache_parquet import cache_parquet
from polars_io_tools.io_sources.pushdown_combine import FilterSpec, pushdown_combine
from polars_io_tools.testing import PredicateTracker
from polars_io_tools.tests.helpers.cases import as_params
from polars_io_tools.tests.helpers.node_shapes import NodeShape
from polars_io_tools.tests.helpers.polars_versions import polars_has, xfail_if_polars_has


@dataclass(frozen=True)
class SourceFilterCase:
    """A filter on a source, and what the source should receive when a query applies it."""

    id: str
    _: KW_ONLY
    predicate: pl.Expr
    # whether a filter reaches the source; None checks nothing
    expect_pushed: bool | None = True
    # also compare the rows with a plain filtered read
    compare_results: bool = False
    # receives the original and the received filter, and asserts on them
    check_pushed: Callable[[pl.Expr, pl.Expr], None] | None = None
    # the data the source serves; None uses the ``tracked_sample_data`` fixture
    source_frame: pl.DataFrame | None = None
    marks: tuple = ()


def _pushed_if_polars_has(name: str):
    """``True`` on Polars versions with the behavior ``name``; ``None`` (not checked) before it."""
    return True if polars_has(name) else None


def _pushed_is_and_of_negations(original, pushed):
    """Polars rewrites ``~(a | b)`` as ``~a & ~b`` before handing it to the source; the original stays a NOT of an OR."""
    NodeShape(BinaryExprNode, op=OperatorType.AND).match(get_parsed_expr(pushed))
    NodeShape(FunctionNode, function_type=BooleanFunctionType.NOT, inputs=[NodeShape(BinaryExprNode, op=OperatorType.OR)]).match(
        get_parsed_expr(original)
    )


def _pushed_is_fill_null_comparison(original, pushed):
    NodeShape(BinaryExprNode, left=NodeShape(FunctionNode, function_type=GenericFunctionType.FILL_NULL)).match(get_parsed_expr(pushed))


_SIX_PM_JAN_1_2023, _END_OF_JAN_1_2023 = datetime(2023, 1, 1, 18), datetime(2023, 1, 1, 23, 59, 59)


_NANOSECOND_TIMESTAMPS = pl.DataFrame(
    {"timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)], "value": [10, 20, 30]},
    schema={"timestamp": pl.Datetime("ns"), "value": pl.Int64},
)


_PADDED_STRINGS = pl.DataFrame({"padded": ["  A  ", "  B  ", "  C  ", " D ", "E"]})


_FIRST_AND_LAST_NAMES = pl.DataFrame({"first": ["A", "B", "C"], "last": ["X", "Y", "Z"]})


skip_before_implicit_casts_are_pushed = pytest.mark.skipif(
    not polars_has("pushes_implicit_casts"), reason="implicit casts block pushdown before Polars 1.28"
)


# Filters on the ``tracked_sample_data`` source (or on ``source_frame``), and whether Polars hands each one to the source.
PUSHDOWN_CASES = [
    SourceFilterCase("numeric_predicates-gt", predicate=pl.col("id") > 5),
    SourceFilterCase("numeric_predicates-le_float", predicate=pl.col("float_val") <= 3.0),
    SourceFilterCase("numeric_predicates-and", predicate=(pl.col("id") > 3) & (pl.col("int_val") < 70)),
    SourceFilterCase("numeric_predicates-arithmetic", predicate=pl.col("float_val") * 2 > 5),
    SourceFilterCase("string_predicates-eq", predicate=pl.col("string_val") == "A"),
    SourceFilterCase(
        "string_predicates-contains",
        predicate=pl.col("long_string").str.contains("text_"),
        expect_pushed=_pushed_if_polars_has("pushes_function_predicates"),
        compare_results=True,
    ),
    SourceFilterCase("string_predicates-is_in", predicate=pl.col("string_val").is_in(["A", "B"])),
    # str.strip_chars / strip_chars_start / strip_chars_end translate to TRIM / LTRIM / RTRIM
    SourceFilterCase(
        "strip_chars_predicate_pushed_down-both",
        predicate=pl.col("padded").str.strip_chars() == "A",
        compare_results=True,
        source_frame=_PADDED_STRINGS,
    ),
    SourceFilterCase(
        "strip_chars_predicate_pushed_down-start",
        predicate=pl.col("padded").str.strip_chars_start() == "A  ",
        compare_results=True,
        source_frame=_PADDED_STRINGS,
    ),
    SourceFilterCase(
        "strip_chars_predicate_pushed_down-end",
        predicate=pl.col("padded").str.strip_chars_end() == "  A",
        compare_results=True,
        source_frame=_PADDED_STRINGS,
    ),
    SourceFilterCase(
        "concat_horizontal_predicate_pushed_down",
        predicate=pl.concat_str([pl.col("first"), pl.lit("-"), pl.col("last")]) == "A-X",
        compare_results=True,
        source_frame=_FIRST_AND_LAST_NAMES,
    ),
    *[SourceFilterCase(f"fill_null_predicates-{v}", predicate=pl.col("nullable").fill_null(v) > 5) for v in (0, 1, 9)],
    SourceFilterCase("same_dtype_column_comparison", predicate=pl.col("id") < pl.col("int_val")),
    SourceFilterCase("struct_field_pushed_down", predicate=pl.col("struct").struct.field("value") > 5),
    SourceFilterCase(
        "fill_ternary_predicates",
        predicate=pl.when(pl.col("nullable") > 3, pl.col("int_val") > 5)
        .then(pl.col("int_val") > 2)
        .when(pl.col("float_val") > 0)
        .then(pl.col("string_val").is_in(["a", "b"]))
        .when(pl.col("float_val") < 1)
        .then(pl.lit(True))
        .otherwise(pl.col("float_val") < 5),
    ),
    SourceFilterCase("temporal_predicates-date_gt", predicate=pl.col("date_val") > date(2023, 1, 5)),
    SourceFilterCase(
        "temporal_predicates-python_datetime_lt",
        predicate=pl.col("datetime_val") < _SIX_PM_JAN_1_2023,
        expect_pushed=_pushed_if_polars_has("pushes_python_datetime_comparison"),
        compare_results=True,
    ),
    SourceFilterCase("temporal_predicates-polars_datetime_lt", predicate=pl.col("datetime_val") < convert_datetime_to_polars(_SIX_PM_JAN_1_2023)),
    SourceFilterCase(
        "temporal_predicates-polars_datetime_is_between",
        predicate=pl.col("datetime_val").is_between(convert_datetime_to_polars(_SIX_PM_JAN_1_2023), convert_datetime_to_polars(_END_OF_JAN_1_2023)),
    ),
    SourceFilterCase(
        "temporal_predicates-python_datetime_is_between",
        predicate=pl.col("datetime_val").is_between(_SIX_PM_JAN_1_2023, _END_OF_JAN_1_2023),
        expect_pushed=_pushed_if_polars_has("pushes_python_datetime_comparison"),
    ),
    SourceFilterCase(
        "temporal_predicates-dt_day", predicate=pl.col("date_val").dt.day() == 3, expect_pushed=_pushed_if_polars_has("pushes_function_predicates")
    ),
    # a nanosecond column compared with microsecond Python datetimes
    SourceFilterCase(
        "temporal_schema_edge_case-python_datetimes",
        predicate=pl.col("timestamp").is_between(_SIX_PM_JAN_1_2023, _END_OF_JAN_1_2023),
        expect_pushed=polars_has("pushes_implicit_casts"),
        source_frame=_NANOSECOND_TIMESTAMPS,
    ),
    SourceFilterCase(
        "temporal_schema_edge_case-polars_datetimes",
        predicate=pl.col("timestamp").is_between(convert_datetime_to_polars(_SIX_PM_JAN_1_2023), convert_datetime_to_polars(_END_OF_JAN_1_2023)),
        expect_pushed=_pushed_if_polars_has("pushes_python_datetime_comparison"),
        source_frame=_NANOSECOND_TIMESTAMPS,
    ),
    SourceFilterCase("boolean_predicates-column", predicate=pl.col("bool_val")),
    SourceFilterCase("boolean_predicates-not", predicate=~pl.col("bool_val")),
    SourceFilterCase("null_predicates-is_null", predicate=pl.col("nullable").is_null()),
    SourceFilterCase("null_predicates-is_not_null", predicate=pl.col("nullable").is_not_null()),
    SourceFilterCase("categorical_predicates", predicate=pl.col("cat_val") == "A"),
    SourceFilterCase(
        "complex_predicates-nested",
        predicate=(pl.col("id") > 3) & ((pl.col("string_val") == "A") | (pl.col("float_val") > 3.0)) & ~pl.col("bool_val")
        | pl.col("cat_val").is_in(["A", "C"]),
    ),
    SourceFilterCase(
        "complex_predicates-mixed_types",
        predicate=(pl.col("date_val") > date(2023, 1, 3)) & pl.col("string_val").is_in(["A", "B"]) & (pl.col("float_val") > 2.0),
    ),
    SourceFilterCase(
        "predicates_with_implicit_casts-datetime_gt_date",
        predicate=pl.col("datetime_val") > date(2023, 1, 2),
        marks=(skip_before_implicit_casts_are_pushed,),
    ),
    SourceFilterCase(
        "predicates_with_implicit_casts-datetime_is_between_dates",
        predicate=pl.col("datetime_val").is_between(date(2023, 1, 2), date(2023, 1, 3)),
        marks=(skip_before_implicit_casts_are_pushed,),
    ),
    SourceFilterCase(
        "predicates_with_implicit_casts-int_eq_float",
        predicate=pl.col("int_val") == pl.col("float_val"),
        marks=(skip_before_implicit_casts_are_pushed,),
    ),
    SourceFilterCase(
        "predicates_with_implicit_casts-date_gt_datetime",
        predicate=pl.col("date_val") > datetime(2023, 1, 2, 12),
        marks=(skip_before_implicit_casts_are_pushed,),
    ),
    # A cast of an Int64 column to Int64 changes nothing, with or without strict.
    *[
        SourceFilterCase(f"predicate_with_explicit_cast_on_column-strict={s}", predicate=pl.col("int_val").cast(pl.Int64, strict=s) > 30)
        for s in (True, False)
    ],
    *[
        SourceFilterCase(
            f"compare_float_gt_int_with_int_to_float_cast_pushed_down-strict={s}",
            predicate=pl.col("float_val") > pl.col("int_val").cast(pl.Float64, strict=s),
            expect_pushed=polars_has("pushes_implicit_casts"),
        )
        for s in (True, False)
    ],
    *[
        SourceFilterCase(
            f"compare_float_gt_int_with_float_to_int_cast_pushed_down-strict={s}",
            predicate=pl.col("float_val").cast(pl.Int64, strict=s) > pl.col("int_val"),
            expect_pushed=polars_has("pushes_implicit_casts"),
        )
        for s in (True, False)
    ],
    SourceFilterCase(
        "unsimplified_not_predicates-de_morgan",
        predicate=~((pl.col("string_val") == "A") | (pl.col("id") > 3)),
        check_pushed=_pushed_is_and_of_negations,
    ),
    SourceFilterCase(
        "unsimplified_not_predicates-double_nested",
        predicate=~((pl.col("string_val") == "A") | (~(pl.col("date_val") == date(2023, 1, 3)) & (pl.col("float_val") > 2.0))),
    ),
    SourceFilterCase(
        "map_batches",
        predicate=pl.col("int_val").map_batches(lambda x: x > 5, return_dtype=pl.Boolean, is_elementwise=True),
        check_pushed=lambda original, pushed: NodeShape(AnonymousFunctionNode).match(get_parsed_expr(pushed)),
        marks=(xfail_if_polars_has("anonymous_function_not_pushed"),),
    ),
    SourceFilterCase(
        "fill_null_constant",
        predicate=pl.col("nullable").fill_null(pl.when(pl.col("id") > 5).then(100).otherwise(0)) > 50,
        expect_pushed=polars_has("pushes_fill_null_predicates"),
        check_pushed=_pushed_is_fill_null_comparison if polars_has("pushes_fill_null_predicates") else None,
    ),
    # results match whether or not the predicate is pushed
    SourceFilterCase("all_results_match_regardless_of_pushdown-id_gt", predicate=pl.col("id") > 5, compare_results=True),
    SourceFilterCase("all_results_match_regardless_of_pushdown-string_eq", predicate=pl.col("string_val") == "A", compare_results=True),
    *[
        SourceFilterCase(
            f"all_results_match_regardless_of_pushdown-{name}",
            predicate=expr,
            expect_pushed=polars_has("pushes_function_predicates"),
            compare_results=polars_has("pushes_function_predicates"),
        )
        for name, expr in [
            ("sin", pl.col("float_val").sin() > 0),
            ("to_uppercase", pl.col("string_val").str.to_uppercase() == "A"),
            ("strftime", pl.col("date_val").dt.strftime("%Y-%m-%d") == "2023-01-01"),
        ]
    ],
    # window functions, aggregations and non-elementwise functions are not pushed
    *[
        SourceFilterCase(f"expected_not_pushed_down-{name}", predicate=expr, expect_pushed=False)
        for name, expr in [
            ("window_mean", pl.col("float_val") > pl.col("float_val").mean().over("string_val")),
            (
                "window_zscore",
                (pl.col("float_val") - pl.col("float_val").mean().over("string_val")) / pl.col("float_val").std().over("string_val") > 1.0,
            ),
            ("window_diff", pl.col("float_val").diff().over([pl.col("string_val")]) > 0.5),
            ("aggregate_mean", pl.col("int_val") > pl.col("int_val").mean()),
            ("aggregate_mean_and_median", (pl.col("float_val") > pl.col("float_val").mean()) & (pl.col("int_val") < pl.col("int_val").median())),
            ("aggregate_min_max", (pl.col("float_val") - pl.col("float_val").min()) / (pl.col("float_val").max() - pl.col("float_val").min()) > 0.5),
            (
                "aggregate_and_window",
                (pl.col("float_val") > pl.col("float_val").mean()) & (pl.col("float_val") > pl.col("float_val").mean().over("string_val")),
            ),
            (
                "window_count_and_null_count",
                (pl.col("id").count().over("string_val") > 1) & (pl.col("nullable").null_count() < pl.len()).over("string_val"),
            ),
            (
                "window_mean_plus_std",
                pl.col("float_val") > (pl.col("float_val").mean().over("string_val") + pl.col("float_val").std().over("string_val")),
            ),
            (
                "fill_null_window_mean",
                pl.col("nullable").fill_null(pl.col("nullable").mean().over("string_val")) > pl.col("nullable").fill_null(0).mean(),
            ),
            (
                "ternary_window_and_median",
                pl.when(pl.col("string_val") == "A")
                .then(pl.col("float_val") > pl.col("float_val").mean().over("string_val"))
                .otherwise(pl.col("int_val") > pl.col("int_val").median()),
            ),
            ("window_first", pl.col("float_val") > pl.col("float_val").first().over("string_val")),
            ("fill_null_strategy", pl.col("nullable").fill_null(strategy="min") > 3),
            ("diff", pl.col("float_val").diff() > 0.5),
            ("map_batches_not_elementwise", pl.col("int_val").map_batches(lambda x: x > 5, return_dtype=pl.Boolean, is_elementwise=False)),
        ]
    ],
]


@pytest.mark.parametrize("case", as_params(PUSHDOWN_CASES))
def test_pushdown(request, case: SourceFilterCase):
    tracker = request.getfixturevalue("tracked_sample_data") if case.source_frame is None else PredicateTracker(case.source_frame)
    if case.expect_pushed is not None:
        tracker.assert_predicate_pushed_down(case.predicate, case.check_pushed, expected_pushed_down=case.expect_pushed)
    if case.compare_results:
        tracker.assert_results_match(case.predicate)


def test_column_selection_with_predicates(tracked_sample_data):
    """A projection and a predicate are both pushed."""
    columns = ["id", "string_val", "date_val"]
    tracked_sample_data.reset()
    result = tracked_sample_data.lazy_frame.filter(pl.col("id") > 5).select(columns).collect()
    assert_frame_equal(result, tracked_sample_data.df.filter(pl.col("id") > 5).select(columns))
    assert tracked_sample_data.last_predicate is not None, "Predicate was not pushed down"
    assert tracked_sample_data.last_with_columns is not None and set(tracked_sample_data.last_with_columns) == set(columns)


_ENTRY_POINTS = {
    "cache_memory": lambda lf, path: cache_memory(lf, schema=lf.collect_schema(), partition_cols="c0"),
    "cache_parquet": lambda lf, path: cache_parquet(lf, path, "date", extra_partition_cols="c0"),
    "debug": lambda lf, path: lf.piot.debug(log_level=logging.DEBUG),
    "pushdown_combine": lambda lf, path: pushdown_combine({"s": (lf, {"c0": FilterSpec()})}, lambda s: s["s"]),
}


# Polars rejects non-Boolean inputs to any/all_horizontal (https://github.com/pola-rs/polars/pull/30574), so each
# integer input is cast to Boolean.
HORIZONTAL_OVER_INTEGERS = {
    "all": pl.all_horizontal(pl.col("c0").cast(pl.Boolean), pl.col("c0") > 0),
    "not_all": ~pl.all_horizontal(pl.col("c0").cast(pl.Boolean), pl.col("c0") > 0),
    "any_of_functions": pl.any_horizontal(pl.col("c0").abs().cast(pl.Boolean), pl.col("i").cast(pl.Boolean)),
}


@pytest.mark.parametrize("pred_name", sorted(HORIZONTAL_OVER_INTEGERS))
@pytest.mark.parametrize("entry_point", sorted(_ENTRY_POINTS))
def test_horizontal_over_integers_cast_to_boolean(entry_point, pred_name, tmp_path):
    frame = pl.DataFrame({"date": [date(2024, 1, 1)] * 6, "i": range(6), "c0": pl.Series([0, 1, 2, 3, -1, None], dtype=pl.Int8)})
    lf = _ENTRY_POINTS[entry_point](frame.lazy(), tmp_path)
    pred = HORIZONTAL_OVER_INTEGERS[pred_name]
    assert_frame_equal(lf.filter(pred).collect(), frame.filter(pred), check_row_order=False)
