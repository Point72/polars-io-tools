"""The helpers in ``io_sources.util``: filtering without pushdown, wrapping a source so its errors say where they
came from, adding columns in dependency order, S3 storage options, and excluding known partitions."""

import functools
import operator
import subprocess
import textwrap
import threading
from collections.abc import Callable, Iterator
from dataclasses import KW_ONLY, dataclass, field
from datetime import UTC, date, datetime, timedelta, timezone
from graphlib import CycleError

import polars as pl
import pytest
from polars.exceptions import ColumnNotFoundError, SchemaError
from polars.testing import assert_frame_equal

from polars_io_tools._compat import POLARS_GE_2
from polars_io_tools.io_sources.util import (
    _resolve_endpoint_hostname,
    _storage_options_for,
    extract_description_block,
    inject_description_block,
    partition_exclusion_predicate,
    register_io_source_with_is_pure,
    with_columns_topo,
    wrap_io_source_with_error_catching,
)
from polars_io_tools.testing import PredicateTracker, io_source_assert
from polars_io_tools.tests.helpers.assertions import assert_matches_plain_polars, assert_translated_text
from polars_io_tools.tests.helpers.cases import as_params
from polars_io_tools.tests.helpers.pickling import pickle_round_trip
from polars_io_tools.tests.helpers.polars_versions import polars_has
from polars_io_tools.tests.helpers.subprocesses import SUBPROCESS_TIMEOUT_S, run_in_subprocess

_SMALL_FRAME = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50], "c": ["x", "y", "x", "y", "x"]})
_EASTERN_STANDARD_TIME = timezone(timedelta(hours=-5))
_MANY_DTYPES = pl.DataFrame(
    {
        "int_col": [1, 2, 3, 4, 5],
        "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
        "str_col": ["a", "b", "c", "d", "e"],
        "bool_col": [True, False, True, False, True],
        "date_col": [date(2023, m, 1) for m in range(1, 6)],
        "datetime_col": [datetime(2023, m, 1, 9 + m) for m in range(1, 6)],
        "datetime_utc": [datetime(2023, m, 1, 9 + m, tzinfo=UTC) for m in range(1, 6)],
        "datetime_est": [datetime(2023, m, 1, 9 + m, tzinfo=_EASTERN_STANDARD_TIME) for m in range(1, 6)],
        "nullable_int": [1, None, 3, None, 5],
        "list_col": [[1, 2], [3, 4, 5], [6], [7, 8, 9, 10], []],
    }
)


@dataclass(frozen=True)
class NoPushdownCase:
    """Filters that ``filter_no_pushdown`` must apply after reading, giving the same rows as a normal filter."""

    id: str
    _: KW_ONLY
    frame: pl.DataFrame
    # one expression or a list
    filters: pl.Expr | list[pl.Expr]
    # check the query plans too: a normal filter, and ``filter_no_pushdown`` with optimizations turned off, reach the
    # source (``SELECTION`` and no ``FILTER``), while ``filter_no_pushdown`` keeps a ``FILTER``
    check_plans: bool = False


_LIST_LENGTH = pl.col("list_col").list.len()
NO_PUSHDOWN = [
    *[
        NoPushdownCase(f"filter_no_pushdown_via_pipe_print_explain-{name}", frame=_SMALL_FRAME, filters=e, check_plans=True)
        for name, e in [("two_filters", [pl.col("c") == "x", pl.col("a") > 2]), ("one_filter", [pl.col("c") == "x"])]
    ],
    *[
        NoPushdownCase(f"filter_no_pushdown_comprehensive-{name}", frame=_MANY_DTYPES, filters=e)
        for name, e in [
            ("float_cast_gt", pl.col("int_col").cast(pl.Float64) > 2.5),
            ("aliased_string_eq", pl.col("str_col").alias("renamed_str") == "c"),
            ("is_not_null", pl.col("nullable_int").is_not_null()),
            ("aliased_utc_hour", pl.col("datetime_utc").dt.hour().alias("utc_hour") >= 12),
            ("converted_time_zone_hour", pl.col("datetime_utc").dt.convert_time_zone("US/Eastern").alias("utc_to_est").dt.hour() >= 5),
            ("when_then_otherwise", pl.when(pl.col("int_col") > 3).then(pl.col("str_col")).otherwise(pl.lit("default")).alias("conditional") == "d"),
            (
                "list_length_category",
                pl.when(_LIST_LENGTH == 0)
                .then(pl.lit("empty"))
                .when(_LIST_LENGTH == 1)
                .then(pl.lit("single"))
                .otherwise(pl.lit("multiple"))
                .alias("list_category")
                == "multiple",
            ),
            ("list_length_gt_2", _LIST_LENGTH > 2),
            ("list_length_eq_0", _LIST_LENGTH == 0),
            ("list_sum_fill_null", pl.col("list_col").list.sum().alias("list_sum").fill_null(0) > 10),
            ("two_expressions", [pl.col("int_col") > 2, pl.col("str_col") != "b"]),
            ("three_expressions", [pl.col("bool_col"), pl.col("nullable_int").is_not_null(), pl.col("date_col").dt.year() == 2023]),
            (
                "three_expressions_with_any_horizontal",
                [
                    pl.col("bool_col") | pl.col("nullable_int").is_not_null(),
                    pl.col("date_col").dt.year() == 2023,
                    pl.any_horizontal(pl.col("int_col") > 3, pl.col("float_col") < 5.0),
                ],
            ),
        ]
    ],
    NoPushdownCase(
        "filter_no_pushdown_edge_cases-mixed",
        frame=_MANY_DTYPES,
        filters=[
            (pl.col("int_col").cast(pl.Float64) > 2.0)
            & (pl.col("str_col").alias("renamed") != "x")
            & pl.col("nullable_int").is_not_null()
            & (pl.col("date_col").dt.year() == 2023)
            & (pl.col("datetime_utc").dt.hour().alias("utc_hour") >= 10)
        ],
    ),
]


@pytest.mark.parametrize("case", as_params(NO_PUSHDOWN))
def test_filter_no_pushdown(case: NoPushdownCase):
    """filter_no_pushdown gives the rows a normal filter gives, without pushing the filter into the source."""
    lf = io_source_assert(case.frame, lambda _: True)
    normal = lf.filter(case.filters)
    variants = [lf.piot.filter_no_pushdown(case.filters), lf.piot.filter_no_pushdown(case.filters, _disable_optimizations=True)]
    for variant in variants:
        assert_frame_equal(variant.collect(), normal.collect())
    assert_translated_text(variants[0].explain(), "FILTER", omits="SELECTION" if case.check_plans else ())
    if case.check_plans:
        for pushed in (normal, variants[1]):
            assert_translated_text(pushed.explain(), "SELECTION: ", omits="FILTER")


def test_filter_no_pushdown_accepts_an_empty_list():
    # an empty list is only accepted with optimizations on, so ``_disable_optimizations`` is not tried
    lf = io_source_assert(_MANY_DTYPES, lambda _: True)
    assert_frame_equal(lf.piot.filter_no_pushdown([]).collect(), lf.collect())


class TestRegisterIoSourceWithIsPure:
    def test_cse_optimization_with_is_pure(self):
        """is_pure=True lets a self-join read the source once."""
        tracker = PredicateTracker(pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]}))
        lf = register_io_source_with_is_pure(_tracking_source(tracker), schema=tracker.df.schema)
        result = lf.join(lf, on="id", how="inner").collect(engine="in-memory" if POLARS_GE_2 else "auto")
        assert_frame_equal(result, pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30], "value_right": [10, 20, 30]}), check_row_order=False)
        if polars_has("pure_udf_cse"):
            assert tracker.call_count == 1

    @pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
    @pytest.mark.parametrize(
        "engine",
        [
            "auto",
            "streaming",
            pytest.param("in-memory", marks=pytest.mark.skipif(not POLARS_GE_2, reason="Polars 1 uses auto for in-memory execution")),
        ],
    )
    def test_source_exceptions_surface_as_compute_error(self, error_type, engine):
        def failing_source(with_columns, predicate, n_rows, batch_size):
            yield pl.DataFrame({"value": [1]})
            raise error_type("reader failed")

        lf = register_io_source_with_is_pure(failing_source, schema={"value": pl.Int64})
        with pytest.raises(pl.exceptions.ComputeError, match="reader failed") as error:
            lf.collect(engine=engine)
        if POLARS_GE_2:
            assert isinstance(error.value.__cause__, error_type)

    @pytest.mark.parametrize("project", [False, True], ids=["no_projection", "projection"])
    def test_head_then_filter(self, project):
        df = pl.DataFrame({"a": range(10), "b": range(10, 20)})
        lf = register_io_source_with_is_pure(_tracking_source(PredicateTracker(df)), schema=df.schema)

        def query(lf):
            lf = lf.head(5).filter(pl.col("a") > 2)
            return lf.select("b") if project else lf

        assert_matches_plain_polars(query(lf), query(df.lazy()), None, check_row_order=True)

    def test_kwargs_passthrough(self):
        """validate_schema reaches register_io_source and rejects a frame with another schema."""

        def wrong_schema(with_columns, predicate, n_rows, batch_size):
            yield pl.DataFrame({"y": [1, 2, 3]})

        result = register_io_source_with_is_pure(wrong_schema, schema={"x": pl.Int64}, validate_schema=True)
        assert isinstance(result, pl.LazyFrame)
        with pytest.raises(SchemaError):
            result.collect()


_COUNT_LOCK = threading.Lock()


def _tracking_source(tracker: PredicateTracker):
    """An IO source callable over ``tracker.df`` that counts its calls; it filters before taking n_rows."""

    def source(with_columns, predicate, n_rows, batch_size) -> Iterator[pl.DataFrame]:
        with _COUNT_LOCK:
            tracker.call_count += 1
        df = tracker.df if predicate is None else tracker.df.filter(predicate)
        df = df if with_columns is None else df.select(with_columns)
        yield df if n_rows is None else df.head(n_rows)

    return source


def test_error_catching_preserves_function_metadata():
    def original_io_source(with_columns, predicate, n_rows, batch_size):
        """Original docstring for testing."""
        yield pl.DataFrame({"x": [1, 2, 3]})

    wrapped = wrap_io_source_with_error_catching(original_io_source, identifier="test")
    assert (wrapped.__name__, wrapped.__doc__, wrapped.__wrapped__) == ("original_io_source", "Original docstring for testing.", original_io_source)


def test_error_catching_normal_operation():
    df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
    wrapped = wrap_io_source_with_error_catching(_tracking_source(PredicateTracker(df)), identifier="test")
    (result,) = wrapped(with_columns=["x"], predicate=pl.col("x") > 1, n_rows=None, batch_size=None)
    assert_frame_equal(result, pl.DataFrame({"x": [2, 3]}))


class CustomError(Exception):
    pass


def _failing_source(error, yields_first=False):
    def failing_io_source(*args, **kwargs):
        if yields_first:
            yield pl.DataFrame({"x": [1]})
        raise error

    return failing_io_source


_X_AND_Y_FILTER = (pl.col("x") > 5) & (pl.col("y") < 11)
_LONG_FILTER = pl.col("data_date").is_between(pl.lit("2024-01-01"), pl.lit("2024-12-31")) & (pl.col("region") == "US") & (pl.col("value") > 100)
_ALL_ARGUMENTS_NONE = {"with_columns": None, "predicate": None, "n_rows": None, "batch_size": None}


@dataclass(frozen=True)
class ErrorCatchingCase:
    """A failing source wrapped with error catching, the batches it yields before the error, and the error that comes
    out: a ``RuntimeError`` caused by an exception of type ``cause``."""

    id: str
    _: KW_ONLY
    source: Callable
    kwargs: dict
    cause: type[Exception]
    # the error message holds every item of ``message_contains`` (a tuple item: any of its strings) and none of
    # ``message_omits``
    message_contains: tuple
    message_omits: str | tuple = ()
    args: tuple = ()
    # as dicts
    batches_before_error: list[dict] = field(default_factory=list)


ERRORS = [
    ErrorCatchingCase(
        "error_catching_catches_exceptions",
        source=_failing_source(CustomError("Something went wrong!")),
        kwargs=_ALL_ARGUMENTS_NONE,
        cause=CustomError,
        message_contains=(
            "IO Source 'failing_io_source' failed",
            "=== IO SOURCE ERROR ===",
            "Function: failing_io_source",
            "Identifier: test",
            "Error Type: CustomError",
            "Error Message: Something went wrong!",
        ),
    ),
    # the first batch comes through; the error surfaces on the next
    ErrorCatchingCase(
        "error_catching_with_generator_exception",
        source=_failing_source(ValueError("Generator failed"), yields_first=True),
        kwargs=_ALL_ARGUMENTS_NONE,
        batches_before_error=[{"x": [1]}],
        cause=ValueError,
        message_contains=("IO Source 'failing_io_source' failed", "Generator failed"),
    ),
    ErrorCatchingCase(
        "error_catching_preserves_arguments_in_error",
        source=_failing_source(Exception("Test error")),
        kwargs={"with_columns": ["x", "y"], "predicate": _X_AND_Y_FILTER, "n_rows": 10, "batch_size": 100},
        cause=Exception,
        message_contains=("Call Arguments:", "with_columns", "predicate", "n_rows", "batch_size", str(_X_AND_Y_FILTER)),
    ),
    # expressions in arguments appear in full, not as a truncated repr
    ErrorCatchingCase(
        "error_catching_converts_expr_in_args_to_string",
        source=_failing_source(Exception("e")),
        args=(None, _LONG_FILTER, None, 100000),
        kwargs={},
        cause=Exception,
        message_contains=("data_date", "is_between", "region", "value"),
        message_omits="<Expr [",
    ),
    ErrorCatchingCase(
        "error_catching_handles_mixed_args_and_kwargs",
        source=_failing_source(Exception("e")),
        args=(pl.col("x").cast(pl.Float64) + pl.col("y"), "normal_arg"),
        kwargs={"predicate": pl.col("z").is_in([1, 2, 3]), "other": "value"},
        cause=Exception,
        message_contains=(("cast", "Float64"), "is_in"),
        message_omits="<Expr [",
    ),
]


@pytest.mark.parametrize("case", as_params(ERRORS))
def test_error_catching_raises_with_context(case: ErrorCatchingCase):
    batches = wrap_io_source_with_error_catching(case.source, identifier="test")(*case.args, **case.kwargs)
    got = []
    with pytest.raises(RuntimeError) as exc_info:
        got.extend(b.to_dict(as_series=False) for b in batches)
    assert got == case.batches_before_error
    assert type(exc_info.value.__cause__) is case.cause
    assert_translated_text(str(exc_info.value), contains=case.message_contains, omits=case.message_omits)


def test_catching_then_debug_logging_no_error(caplog):
    """The wrapper logs nothing itself; a caller's debug log of the error carries the full context."""
    import logging

    wrapped = wrap_io_source_with_error_catching(_failing_source(CustomError("kaboom!")), identifier="debug_logging_test")
    caplog.set_level(logging.DEBUG)
    try:
        list(wrapped(**_ALL_ARGUMENTS_NONE))
    except RuntimeError:
        logging.debug("Captured wrapped IO error", exc_info=True)
    assert all(record.levelno < logging.ERROR for record in caplog.records)
    assert_translated_text(
        caplog.text,
        [
            "=== IO SOURCE ERROR ===",
            "Function: failing_io_source",
            "Identifier: debug_logging_test",
            "Error Type: CustomError",
            "Error Message: kaboom!",
        ],
    )


def _with_columns_topo_through_piot(lf, exprs):
    return lf.piot.with_columns_topo(exprs)


@dataclass(frozen=True)
class TopoCase:
    """Expressions added to a frame with ``with_columns_topo``, and the columns they make."""

    id: str
    _: KW_ONLY
    # the input frame's data, by column
    input_columns: dict[str, list]
    exprs: list[pl.Expr]
    # the expected columns of the result; None: the input frame unchanged
    expected_columns: dict[str, list] | None
    # receives the input frame and ``exprs``
    run: Callable[[pl.LazyFrame, list[pl.Expr]], pl.LazyFrame] = with_columns_topo


_X_1_2_3 = {"x": [1, 2, 3]}
_B_IS_X_PLUS_1, _C_IS_X_TIMES_2 = (pl.col("x") + 1).alias("b"), (pl.col("x") * 2).alias("c")
_B3_IS_B_PLUS_1_ALIASED_TWICE = (pl.col("b") + 1).alias("b2").alias("b3")

# Expressions that depend on each other's output, in an order that needs sorting, and the columns they make.
TOPO_CASES = [
    TopoCase(
        "simple_dependency_two_columns",
        input_columns={"b": [1, 2]},
        exprs=[(pl.col("b") + 1).name.suffix("_2"), (pl.col("b_2") + 1).name.suffix("_3")],
        expected_columns={"b_2": [2, 3], "b_2_3": [3, 4]},
    ),
    TopoCase(
        "chain_dependencies",
        input_columns={"c": [1, 2, 3]},
        exprs=[(pl.col("c") * 2).alias("b"), (pl.col("b") + 1).alias("a")],
        expected_columns={"c": [1, 2, 3], "b": [2, 4, 6], "a": [3, 5, 7]},
    ),
    TopoCase(
        "diamond_dependencies",
        input_columns=_X_1_2_3,
        exprs=[(pl.col("b") + pl.col("c")).alias("d"), _B_IS_X_PLUS_1, _C_IS_X_TIMES_2],
        expected_columns={"b": [2, 3, 4], "c": [2, 4, 6], "d": [4, 7, 10]},
    ),
    TopoCase(
        "diamond_dependencies_multiple_input",
        input_columns=_X_1_2_3,
        exprs=[(pl.col("b") + pl.col("c")).alias("e"), _B_IS_X_PLUS_1, _C_IS_X_TIMES_2, pl.sum_horizontal(pl.col("b"), pl.col("c")).alias("d")],
        expected_columns={"b": [2, 3, 4], "c": [2, 4, 6], "e": [4, 7, 10], "d": [4, 7, 10]},
    ),
    TopoCase(
        "diamond_dependencies_multiple_input_boolean",
        input_columns=_X_1_2_3,
        exprs=[
            _B_IS_X_PLUS_1,
            _C_IS_X_TIMES_2,
            pl.sum_horizontal(pl.col("b"), pl.col("c")).alias("d_sum"),
            pl.any_horizontal(pl.col("b") > 2, pl.col("c") > 3).alias("d_any"),
        ],
        expected_columns={"b": [2, 3, 4], "c": [2, 4, 6], "d_sum": [4, 7, 10], "d_any": [False, True, True]},
    ),
    TopoCase(
        "mixed_interleaved_dependencies",
        input_columns={"x": [1, 2]},
        exprs=[(pl.col("x") + 1).alias("a"), (pl.col("x") * 3).alias("b"), (pl.col("a") * 2).alias("d"), (pl.col("b") + pl.col("d")).alias("e")],
        expected_columns={"a": [2, 3], "b": [3, 6], "d": [4, 6], "e": [7, 12]},
        run=_with_columns_topo_through_piot,
    ),
    TopoCase(
        "double_alias_chaining",
        input_columns={"b": [1, 2]},
        exprs=[_B3_IS_B_PLUS_1_ALIASED_TWICE],
        expected_columns={"b3": [2, 3]},
        run=_with_columns_topo_through_piot,
    ),
    TopoCase(
        "double_alias_chain_dependency_on_final",
        input_columns={"b": [1, 2]},
        exprs=[_B3_IS_B_PLUS_1_ALIASED_TWICE, (pl.col("b3") + 1).alias("b4")],
        expected_columns={"b3": [2, 3], "b4": [3, 4]},
        run=_with_columns_topo_through_piot,
    ),
    TopoCase(
        "empty_exprs_returns_original",
        input_columns=_SMALL_FRAME.to_dict(as_series=False),
        exprs=[],
        expected_columns=None,
        run=_with_columns_topo_through_piot,
    ),
]


@pytest.mark.parametrize("case", as_params(TOPO_CASES))
def test_with_columns_topo(case: TopoCase):
    lf = pl.LazyFrame(case.input_columns)
    out = case.run(lf, case.exprs)
    if case.expected_columns is None:
        assert_frame_equal(out.collect(), lf.collect())
    else:
        assert_frame_equal(out.select(list(case.expected_columns)).collect(), pl.DataFrame(case.expected_columns))


@dataclass(frozen=True)
class TopoErrorCase:
    """Expressions ``with_columns_topo`` rejects, and when the error comes."""

    id: str
    _: KW_ONLY
    # the input frame's data
    base: dict
    exprs: list[pl.Expr]
    error: type[Exception]
    # raised on collect rather than when the query is built
    on_collect: bool


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            # only the final alias of a chain exists
            TopoErrorCase(
                "double_alias_chain_dependency_on_intermediate_errors",
                base={"b": [1, 2]},
                exprs=[_B3_IS_B_PLUS_1_ALIASED_TWICE, (pl.col("b2") + 1).alias("b4")],
                error=ColumnNotFoundError,
                on_collect=True,
            ),
            TopoErrorCase(
                "cycle_detection",
                base={"x": [1, 2]},
                exprs=[(pl.col("b") + 1).alias("a"), (pl.col("a") + 1).alias("b")],
                error=CycleError,
                on_collect=False,
            ),
        ]
    ),
)
def test_with_columns_topo_errors(case: TopoErrorCase):
    lf = pl.LazyFrame(case.base)
    if case.on_collect:
        out = lf.piot.with_columns_topo(case.exprs)
        with pytest.raises(case.error):
            out.collect()
    else:
        with pytest.raises(case.error):
            lf.piot.with_columns_topo(case.exprs)


def test_with_columns_topo_independent_batched():
    """Independent expressions share a single ``with_columns`` stage."""
    out = with_columns_topo(pl.LazyFrame(_X_1_2_3), [(pl.col("x") + 1).alias("a"), (pl.col("x") * 2).alias("b"), (pl.col("x") - 3).alias("c")])
    plan = out.explain()
    assert plan.count("WITH_COLUMNS") == 1
    assert all(f'alias("{name}")' in plan for name in "abc")
    assert_frame_equal(out.select(["a", "b", "c"]).collect(), pl.DataFrame({"a": [2, 3, 4], "b": [2, 4, 6], "c": [-2, -1, 0]}))


def test_with_columns_topo_large_scale():
    # Massive DAG: ~10k columns, max depth 15, varying sizes per layer
    rows = 3
    lf = pl.LazyFrame({"x": list(range(rows))})

    total_cols = 10_000
    num_layers = 15
    base = total_cols // num_layers
    rem = total_cols % num_layers
    sizes = [base + (1 if i < rem else 0) for i in range(num_layers)]

    exprs: list[pl.Expr] = []
    # Layer 0: depend on base column 'x'
    for k in range(sizes[0]):
        name = f"c0_{k}"
        exprs.append((pl.col("x") + pl.lit(k)).alias(name))

    # Subsequent layers: each column depends on same-index column from previous layer
    for layer in range(1, num_layers):
        prev_size = sizes[layer - 1]
        for k in range(sizes[layer]):
            upstream = f"c{layer - 1}_{k % prev_size}"
            name = f"c{layer}_{k}"
            # Add layer index to propagate deterministic values
            exprs.append((pl.col(upstream) + pl.lit(layer)).alias(name))

    out = lf.piot.with_columns_topo(exprs)

    # Select a representative subset across layers to keep memory modest
    sample_cols = [
        "c0_0",
        f"c1_{sizes[1] - 1}",
        f"c5_{sizes[5] // 2}",
        f"c14_{(sizes[14] - 1)}",
    ]

    df = out.select(sample_cols).collect()

    # Expected values: c0_k = x + k; c_l_k = x + (k' from prev) + sum_{t=1..l} t
    def expected_for_layer(layer: int, k: int) -> list[int]:
        # Resolve k back to initial index through modulo chain; since we map k -> k % prev_size,
        # the base index effectively becomes k modulo sizes[l0] iteratively. For our construction,
        # taking modulo only once at each step preserves the original k for same-index mapping.
        base_k = k
        s = 0
        for t in range(1, layer + 1):
            s += t
        return [x + base_k + s for x in range(rows)]

    expected = pl.DataFrame(
        {
            sample_cols[0]: [x + 0 for x in range(rows)],
            sample_cols[1]: expected_for_layer(1, sizes[1] - 1),
            sample_cols[2]: expected_for_layer(5, sizes[5] // 2),
            sample_cols[3]: expected_for_layer(14, sizes[14] - 1),
        }
    )

    assert_frame_equal(df, expected)


@pytest.mark.xfail(reason="Selectors expansion not yet supported by with_columns_topo", strict=True)
def test_with_columns_topo_with_selector():
    import polars.selectors as cs

    lf = pl.LazyFrame(
        {
            "a": [1.123, 2.456, 3.789],
            "b": [4.987, 5.654, 6.321],
            "c": ["foo", "bar", "baz"],
        }
    )
    exprs = [cs.numeric().as_expr().round(2)]
    out = with_columns_topo(lf, exprs)
    df = out.collect()
    expected = pl.DataFrame(
        {
            "a": [1.12, 2.46, 3.79],
            "b": [4.99, 5.65, 6.32],
            "c": ["foo", "bar", "baz"],
        }
    )
    assert_frame_equal(df, expected)


@pytest.mark.xfail(reason="Implicit selectors expansion not yet supported by with_columns_topo", strict=True)
def test_with_columns_topo_with_implicit_selector_and_dependency():
    # Implicit selector via multi-column expr and .name.suffix; plus dependent expr
    lf = pl.LazyFrame(
        {
            "a": [1, 2, 3],
            "b": [10, 20, 30],
        }
    )

    exprs = [
        (pl.col("a", "b") + 1).name.suffix("_2"),  # creates a_2, b_2 implicitly
        (pl.col("a_2") + pl.col("b_2")).alias("sum2"),  # depends on outputs above
    ]

    out = with_columns_topo(lf, exprs)
    df = out.collect()

    expected = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [10, 20, 30],
            "a_2": [2, 3, 4],
            "b_2": [11, 21, 31],
            "sum2": [13, 24, 35],
        }
    )

    # Order may differ; compare by selecting expected columns
    assert_frame_equal(df.select(expected.columns), expected)


@dataclass(frozen=True)
class StorageOptionsCase:
    """A URI, what boto3 and the environment provide, and the storage options ``_storage_options_for`` returns."""

    id: str
    _: KW_ONLY
    uri: str
    # the (pyarrow, polars) pair of dicts
    expected_options: tuple[dict, dict]
    # what a fake boto3 session reports
    boto3_region: str | None = None
    # (access key, secret, token)
    boto3_credentials: tuple | None = None
    # environment variables to set
    environment: dict[str, str] = field(default_factory=dict)
    aws_profile: str | None = None
    # text that must appear in the log
    expected_log: str | None = None


_BOTH_ENDPOINT_VARIABLES = {"AWS_ENDPOINT_URL": "https://endpoint-url", "AWS_S3_ENDPOINT": "https://s3-endpoint"}


def _expected_options(access, secret, token=None, region=None, endpoint=None):
    """The (pyarrow, polars) storage options for these credentials, region and endpoint."""
    pyarrow = {"access_key": access, "secret_key": secret} if access else {}
    polars = {"aws_access_key_id": access, "aws_secret_access_key": secret} if access else {}
    if token:
        pyarrow["session_token"], polars["aws_session_token"] = token, token
    if region:
        pyarrow["region"] = region
    if endpoint:
        pyarrow["endpoint_override"], polars["endpoint_url"] = endpoint, endpoint
        if endpoint.startswith("http://"):
            polars["allow_http"] = "true"
    return pyarrow, polars


STORAGE_OPTIONS = [
    *[
        StorageOptionsCase(f"non_s3_uri_returns_empty-{n}", uri=uri, expected_options=({}, {}))
        for n, uri in [("file", "file:///tmp/cache"), ("local", "/local/path/cache"), ("http", "http://example.com/cache")]
    ],
    # an endpoint in the query wins over the environment
    StorageOptionsCase(
        "s3_with_credentials_and_token",
        uri="s3://bucket/path?endpoint_override=https://query-endpoint",
        expected_options=_expected_options("AKIA_TEST", "SECRET_TEST", "TOKEN_TEST", "us-west-2", "https://query-endpoint"),
        boto3_region="us-west-2",
        boto3_credentials=("AKIA_TEST", "SECRET_TEST", "TOKEN_TEST"),
        environment={"AWS_ENDPOINT_URL": "https://env-endpoint", "AWS_S3_ENDPOINT": "https://env-s3-endpoint"},
        aws_profile="test-profile",
    ),
    # region is a pyarrow option only
    StorageOptionsCase(
        "s3_with_credentials_no_token",
        uri="s3://bucket/path",
        expected_options=_expected_options("AKIA_NO_TOKEN", "SECRET_NO_TOKEN", None, "eu-central-1", "https://env-only-endpoint"),
        boto3_region="eu-central-1",
        boto3_credentials=("AKIA_NO_TOKEN", "SECRET_NO_TOKEN", None),
        environment={"AWS_ENDPOINT_URL": "https://env-only-endpoint"},
    ),
    StorageOptionsCase(
        "s3a_scheme_supported",
        uri="s3a://bucket/path",
        expected_options=_expected_options("AKIA_S3A", "SECRET_S3A"),
        boto3_credentials=("AKIA_S3A", "SECRET_S3A", None),
    ),
    # AWS_ENDPOINT_URL wins over AWS_S3_ENDPOINT, a query endpoint over both
    StorageOptionsCase(
        "endpoint_env_precedence-env",
        uri="s3://bucket/path",
        expected_options=_expected_options("AKIA_ENV", "SECRET_ENV", None, "ap-southeast-1", "https://endpoint-url"),
        boto3_region="ap-southeast-1",
        boto3_credentials=("AKIA_ENV", "SECRET_ENV", None),
        environment=_BOTH_ENDPOINT_VARIABLES,
    ),
    StorageOptionsCase(
        "endpoint_env_precedence-query",
        uri="s3://bucket/path?endpoint_override=https://query",
        expected_options=_expected_options("AKIA_ENV", "SECRET_ENV", None, "ap-southeast-1", "https://query"),
        boto3_region="ap-southeast-1",
        boto3_credentials=("AKIA_ENV", "SECRET_ENV", None),
        environment=_BOTH_ENDPOINT_VARIABLES,
    ),
    StorageOptionsCase(
        "no_credentials_only_region_and_endpoint",
        uri="s3://bucket/path",
        expected_options=_expected_options(None, None, None, "us-east-1", "https://only-env-endpoint"),
        boto3_region="us-east-1",
        environment={"AWS_S3_ENDPOINT": "https://only-env-endpoint"},
    ),
    StorageOptionsCase(
        "profile_name_forwarded_to_boto3_session",
        uri="s3://bucket/path",
        expected_options=_expected_options("AKIA_PROFILE", "SECRET_PROFILE"),
        boto3_credentials=("AKIA_PROFILE", "SECRET_PROFILE", None),
        aws_profile="dev-profile",
    ),
    # an unresolvable endpoint host is kept as configured
    StorageOptionsCase(
        "storage_options_resolution_failure_keeps_original",
        uri="s3://bucket/path",
        expected_options=_expected_options("AKIA_TEST", "SECRET_TEST", None, "us-east-1", "http://grid:9020"),
        boto3_region="us-east-1",
        boto3_credentials=("AKIA_TEST", "SECRET_TEST", None),
        environment={"AWS_ENDPOINT_URL": "http://grid:9020"},
        expected_log="Failed to resolve hostname",
    ),
]


@pytest.mark.parametrize("case", as_params(STORAGE_OPTIONS))
def test_storage_options_for(monkeypatch, caplog, case: StorageOptionsCase):
    import socket
    from types import SimpleNamespace

    sessions = []

    class FakeSession:
        def __init__(self, profile_name=None):
            sessions.append(profile_name)
            self.region_name = case.boto3_region

        def get_credentials(self):
            if case.boto3_credentials is None:
                return None
            access_key, secret_key, token = case.boto3_credentials
            return SimpleNamespace(access_key=access_key, secret_key=secret_key, token=token)

    class FakeCredentialProvider:
        def __init__(self, **kwargs):
            pass

        def _storage_update_options(self):
            return {}

    def unresolvable(hostname):
        raise socket.gaierror("Name or service not known")

    monkeypatch.setattr("boto3.Session", FakeSession)
    monkeypatch.setattr("polars_io_tools.io_sources.util.pl.CredentialProviderAWS", FakeCredentialProvider, raising=False)
    monkeypatch.setattr("socket.gethostbyname", unresolvable)
    for name in ("AWS_ENDPOINT_URL", "AWS_S3_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in case.environment.items():
        monkeypatch.setenv(name, value)
    _resolve_endpoint_hostname.cache_clear()
    try:
        pyarrow, polars, provider = opts = _storage_options_for(case.uri, aws_profile=case.aws_profile)
    finally:
        _resolve_endpoint_hostname.cache_clear()
    assert (opts.pyarrow, opts.polars) == (pyarrow, polars) == case.expected_options
    s3 = case.uri.startswith("s3")
    assert (provider is not None) == s3
    assert sessions == ([case.aws_profile] if s3 else [])
    assert case.expected_log is None or case.expected_log in caplog.text


class _HostnameLookups(list):
    ip: str | None = None


@pytest.fixture
def lookups(monkeypatch):
    """Hostnames passed to ``socket.gethostbyname``, which resolves each to ``lookups.ip``."""
    _resolve_endpoint_hostname.cache_clear()
    calls = _HostnameLookups()

    def fake(hostname):
        calls.append(hostname)
        return calls.ip

    monkeypatch.setattr("socket.gethostbyname", fake)
    yield calls
    _resolve_endpoint_hostname.cache_clear()


@dataclass(frozen=True)
class EndpointHostnameCase:
    """An endpoint URL, the IP its host resolves to, the URL after resolving, and the host looked up."""

    id: str
    _: KW_ONLY
    endpoint: str
    ip: str
    expect: str
    # None: no lookup
    hostname: str | None


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            EndpointHostnameCase(
                "resolve_hostname_with_port", endpoint="http://grid:9020", ip="10.1.2.3", expect="http://10.1.2.3:9020", hostname="grid"
            ),
            EndpointHostnameCase(
                "resolve_hostname_with_fqdn_and_port",
                endpoint="http://.myhost.int.:9020",
                ip="10.4.5.6",
                expect="http://10.4.5.6:9020",
                hostname=".myhost.int.",
            ),
            EndpointHostnameCase(
                "resolve_hostname_without_port",
                endpoint="http://example.com",
                ip="192.168.1.100",
                expect="http://192.168.1.100",
                hostname="example.com",
            ),
            EndpointHostnameCase(
                "resolve_https_endpoint",
                endpoint="https://secure.example.com:443",
                ip="10.0.0.1",
                expect="https://10.0.0.1:443",
                hostname="secure.example.com",
            ),
            EndpointHostnameCase(
                "resolve_hostname_with_path",
                endpoint="http://grid:9020/some/path",
                ip="10.1.2.3",
                expect="http://10.1.2.3:9020/some/path",
                hostname="grid",
            ),
            EndpointHostnameCase(
                "resolve_no_hostname_returns_original", endpoint="file:///path/to/file", ip="10.1.2.3", expect="file:///path/to/file", hostname=None
            ),
        ]
    ),
)
def test_resolve_endpoint_hostname(lookups, case: EndpointHostnameCase):
    lookups.ip = case.ip
    assert _resolve_endpoint_hostname(case.endpoint) == case.expect
    assert lookups == ([case.hostname] if case.hostname else [])


def test_resolve_failure_returns_original(monkeypatch, caplog):
    """A failed lookup keeps the endpoint as given and logs a warning."""
    import socket as socket_module

    def failing_gethostbyname(hostname):
        raise socket_module.gaierror("Name or service not known")

    _resolve_endpoint_hostname.cache_clear()
    monkeypatch.setattr("socket.gethostbyname", failing_gethostbyname)
    assert _resolve_endpoint_hostname("http://nonexistent.host:9020") == "http://nonexistent.host:9020"
    assert "Failed to resolve hostname" in caplog.text
    _resolve_endpoint_hostname.cache_clear()


@dataclass(frozen=True)
class ExclusionPredicateCase:
    """Known partition keys, and the rows of a frame the exclusion filter keeps."""

    id: str
    _: KW_ONLY
    # the frame's data
    df: dict
    known: pl.DataFrame
    columns: list[str]
    # the kept rows; None: no filter is made
    expect: dict | None


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            ExclusionPredicateCase(
                "none_when_nothing_to_exclude-no_known_rows",
                df={"a": [1]},
                known=pl.DataFrame({"a": []}, schema={"a": pl.Int64}),
                columns=["a"],
                expect=None,
            ),
            ExclusionPredicateCase("none_when_nothing_to_exclude-no_columns", df={"a": [1]}, known=pl.DataFrame({"a": [1]}), columns=[], expect=None),
            ExclusionPredicateCase(
                "single_column_excludes_known",
                df={"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]},
                known=pl.DataFrame({"a": [1, 3]}),
                columns=["a"],
                expect={"a": [2, 4], "b": [20, 40]},
            ),
            # excludes (1, 10) and (2, 20) only
            ExclusionPredicateCase(
                "multi_column_excludes_only_exact_combos",
                df={"a": [1, 1, 2, 2], "b": [10, 20, 10, 20]},
                known=pl.DataFrame({"a": [1, 2], "b": [10, 20]}),
                columns=["a", "b"],
                expect={"a": [1, 2], "b": [20, 10]},
            ),
            ExclusionPredicateCase(
                "duplicates_in_known_are_deduplicated", df={"a": [1, 2, 3]}, known=pl.DataFrame({"a": [1, 1, 1, 2]}), columns=["a"], expect={"a": [3]}
            ),
        ]
    ),
)
def test_partition_exclusion_predicate(case: ExclusionPredicateCase):
    pred = partition_exclusion_predicate(case.known, case.columns)
    if case.expect is None:
        assert pred is None
    else:
        assert pl.DataFrame(case.df).filter(pred).sort(list(case.df)).to_dict(as_series=False) == case.expect


@dataclass(frozen=True)
class NullSafeExclusionCase:
    """Known values of ``a``, and the values of ``[1, None, 3, 5]`` the exclusion filter keeps."""

    id: str
    _: KW_ONLY
    known_vals: list
    expected: list


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            NullSafeExclusionCase("null_not_known_is_kept", known_vals=[1, 3], expected=[None, 5]),
            NullSafeExclusionCase("null_known_is_excluded", known_vals=[1, None], expected=[3, 5]),
            NullSafeExclusionCase("single_value", known_vals=[3], expected=[1, None, 5]),
            NullSafeExclusionCase("only_null", known_vals=[None], expected=[1, 3, 5]),
        ]
    ),
)
def test_partition_exclusion_predicate_null_safe_matches_eq_missing(case: NullSafeExclusionCase):
    df = pl.DataFrame({"a": [1, None, 3, 5]}, schema={"a": pl.Int64})
    known = pl.DataFrame({"a": case.known_vals}, schema={"a": pl.Int64})
    assert df.filter(partition_exclusion_predicate(known, ["a"]))["a"].to_list() == case.expected
    reference = functools.reduce(operator.and_, [~pl.col("a").eq_missing(v) for v in case.known_vals])
    assert df.filter(reference)["a"].to_list() == case.expected


def test_partition_exclusion_predicate_remains_analyzable_for_contradiction_detection():
    """Querying for a in {1, 3} while excluding {1, 3} is a contradiction the DNF analysis can prove."""
    from polars_io_tools.io_sources.dnf_visitor import _is_contradiction

    excl = partition_exclusion_predicate(pl.DataFrame({"a": [1, 3]}, schema={"a": pl.Int64}), ["a"])
    assert _is_contradiction(pl.col("a").is_in([1, 3]) & excl) is True


@dataclass(frozen=True)
class DescriptionBlockCase:
    """A description and a tag, and the block ``extract_description_block`` returns."""

    id: str
    _: KW_ONLY
    desc: str | None
    tag: str
    want: str | None


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            DescriptionBlockCase(
                "inject_and_extract_roundtrip_block_helpers",
                desc=inject_description_block(None, "foo.block", "aGVsbG8="),
                tag="foo.block",
                want="aGVsbG8=",
            ),
            DescriptionBlockCase("extract_missing_returns_none_block_helpers-text", desc="some text without blocks", tag="foo", want=None),
            DescriptionBlockCase("extract_missing_returns_none_block_helpers-none", desc=None, tag="foo", want=None),
            DescriptionBlockCase(
                "extract_ignores_whitespace_and_takes_first_line_block_helpers", desc="\n[foo:begin]\n  abc\n  def\n[foo:end]", tag="foo", want="abc"
            ),
        ]
    ),
)
def test_extract_description_block(case: DescriptionBlockCase):
    assert extract_description_block(case.desc, case.tag) == case.want


@pytest.mark.parametrize(
    "otel_setup",
    [
        "",
        """
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(InMemorySpanExporter()))
        trace.set_tracer_provider(provider)
        """,
    ],
    ids=["otel-no-op", "otel-batch-processor"],
)
def test_iosource_collect_batches_does_not_deadlock_at_threads_one(otel_setup):
    """Reproduces the polars-io-tools / polars deadlock and asserts the fix holds.

    Without the helper's thread-pool gate this script hangs forever at
    ``POLARS_MAX_THREADS=1`` due to a Rayon scheduling deadlock when the
    outer engine reaches an io_source whose callback re-enters polars via
    ``collect_batches`` and the outer plan contains a join with a
    pushed-down predicate.
    """
    script = textwrap.dedent(
        """
        import polars as pl
        from polars_io_tools.io_sources.util import (
            collect_lf_in_io_source,
            register_io_source_with_is_pure,
        )

        assert pl.thread_pool_size() == 1, pl.thread_pool_size()

        __OTEL_SETUP__

        # This is the minimal pattern that deadlocks polars 1.39.3+1.40.1
        # without the fix in collect_lf_in_io_source. Removing any one of:
        # the join, the pushed-down predicate, the inner filter+collect_batches,
        # or POLARS_MAX_THREADS=1 avoids the hang.
        SCHEMA = {"key": pl.Utf8, "value": pl.Int64}

        def reader(with_columns, predicate, n_rows, batch_size):
            df = pl.DataFrame(
                {
                    "key": ["a", "b", "c", "d", "e"] * 20_000,
                    "value": list(range(100_000)),
                },
                schema=SCHEMA,
            )
            lf = df.lazy()
            if predicate is not None:
                lf = lf.filter(predicate)
            yield from collect_lf_in_io_source(lf, batch_size or 100_000)

        src = register_io_source_with_is_pure(reader, schema=SCHEMA)
        rhs = pl.LazyFrame(
            {"key": ["a", "b", "c", "d", "e"], "label": ["A", "B", "C", "D", "E"]}
        )

        plan = (
            src.filter(pl.col("key").is_in(["a", "b", "c", "d", "e"]))
               .join(rhs, on="key", how="left")
        )
        out = plan.collect()
        assert out.height == 100_000, out.height
        print("OK", out.height)
        """
    ).replace("__OTEL_SETUP__", textwrap.dedent(otel_setup))

    try:
        result = run_in_subprocess(script)
    except subprocess.TimeoutExpired as e:
        pytest.fail(f"deadlock: subprocess did not finish in {SUBPROCESS_TIMEOUT_S}s\nstdout so far: {e.stdout!r}\nstderr so far: {e.stderr!r}")

    assert result.returncode == 0, f"subprocess failed: returncode={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    assert "OK 100000" in result.stdout, result.stdout


def test_helper_collect_method_at_threads_one():
    """``collect_lf_in_io_source`` at a single worker: stream via ``collect_batches`` on fixed
    runtimes (``POLARS_STREAMING_SUPPORTS_NESTED_IO_PLUGIN``), else fall back to full-collect +
    ``iter_slices`` to dodge the (old) collect_batches re-entrancy deadlock. Either way the
    results are correct and the call does not hang."""
    script = textwrap.dedent(
        """
        import polars as pl
        from polars_io_tools.io_sources.util import collect_lf_in_io_source
        from polars_io_tools._compat import (
            POLARS_HAS_COLLECT_BATCHES,
            POLARS_STREAMING_SUPPORTS_NESTED_IO_PLUGIN,
        )

        assert pl.thread_pool_size() == 1, pl.thread_pool_size()

        used = {"collect_batches": 0}
        _orig_cb = getattr(pl.LazyFrame, "collect_batches", None)
        if _orig_cb is not None:
            def _spy(self, *a, **k):
                used["collect_batches"] += 1
                return _orig_cb(self, *a, **k)
            pl.LazyFrame.collect_batches = _spy

        lf = pl.LazyFrame({"a": list(range(1000))}).filter(pl.col("a") < 500)

        # batch_size=None -> single DataFrame (always a plain collect, never collect_batches)
        chunks = list(collect_lf_in_io_source(lf, None))
        assert len(chunks) == 1, len(chunks)
        assert chunks[0].height == 500, chunks[0].height
        assert used["collect_batches"] == 0, used

        # batch_size set -> multiple DataFrames summing to expected
        chunks = list(collect_lf_in_io_source(lf, 100))
        assert sum(c.height for c in chunks) == 500

        expect_batches = POLARS_HAS_COLLECT_BATCHES and POLARS_STREAMING_SUPPORTS_NESTED_IO_PLUGIN
        if expect_batches:
            assert used["collect_batches"] == 1, ("expected collect_batches on fixed runtime", used)
        else:
            assert used["collect_batches"] == 0, ("expected iter_slices fallback on old runtime", used)
        print("OK")
        """
    )
    result = run_in_subprocess(script)
    assert result.returncode == 0, f"subprocess failed: returncode={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    assert "OK" in result.stdout, result.stdout


class TestFilterNoPushdownPickle:
    """Tests for filter_no_pushdown pickle support."""

    def test_filter_no_pushdown_pickle_basic(self):
        """filter_no_pushdown LazyFrames can be pickled and unpickled."""
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["a", "b", "c", "d", "e"]})

        lf = df.lazy().piot.filter_no_pushdown(pl.col("a") > 2)
        lf_unpickled = pickle_round_trip(lf)

        expected = df.filter(pl.col("a") > 2)
        result = lf_unpickled.collect()
        assert_frame_equal(result, expected)


class TestWithColumnsTopoPickle:
    """Tests for with_columns_topo pickle support."""

    def test_with_columns_topo_pickle_basic(self):
        """with_columns_topo LazyFrames can be pickled and unpickled."""
        df = pl.DataFrame({"a": [1, 2, 3]})

        lf = df.lazy().piot.with_columns_topo(
            [
                (pl.col("a") + 1).alias("b"),
                (pl.col("b") + 1).alias("c"),
            ]
        )
        lf_unpickled = pickle_round_trip(lf)

        # Verify results match exactly
        expected = lf.collect()
        result = lf_unpickled.collect()
        assert_frame_equal(result, expected)
