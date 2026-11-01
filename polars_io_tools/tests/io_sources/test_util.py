import functools
import operator
import threading
from collections.abc import Iterator
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
    partition_exclusion_predicate,
    register_io_source_with_is_pure,
    with_columns_topo,
    wrap_io_source_with_error_catching,
)
from polars_io_tools.testing import PredicateTracker
from polars_io_tools.tests.helpers.catalogue import Case, params
from polars_io_tools.tests.helpers.oracles import assert_matches_native, assert_text
from polars_io_tools.tests.helpers.quirks import quirk

from .conftest import io_source_assert

c = pl.col


_SAMPLE = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50], "c": ["x", "y", "x", "y", "x"]})
_UTC, _EST = UTC, timezone(timedelta(hours=-5))
_COMPLEX = pl.DataFrame(
    {
        "int_col": [1, 2, 3, 4, 5],
        "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
        "str_col": ["a", "b", "c", "d", "e"],
        "bool_col": [True, False, True, False, True],
        "date_col": [date(2023, m, 1) for m in range(1, 6)],
        "datetime_col": [datetime(2023, m, 1, 9 + m) for m in range(1, 6)],
        "datetime_utc": [datetime(2023, m, 1, 9 + m, tzinfo=_UTC) for m in range(1, 6)],
        "datetime_est": [datetime(2023, m, 1, 9 + m, tzinfo=_EST) for m in range(1, 6)],
        "nullable_int": [1, None, 3, None, 5],
        "list_col": [[1, 2], [3, 4, 5], [6], [7, 8, 9, 10], []],
    }
)


def F(id, df, expr, pushed_plans=False):
    """``pushed_plans``: also check the plans: a normal filter and the disabled variant push into the source
    (SELECTION, no FILTER); filter_no_pushdown keeps a FILTER and pushes nothing."""
    return Case(id, expr, kw={"df": df, "pushed_plans": pushed_plans})


_LIST_LEN = pl.col("list_col").list.len()
NO_PUSHDOWN = [
    *[
        F(f"filter_no_pushdown_via_pipe_print_explain-{i}", _SAMPLE, e, True)
        for i, e in enumerate([[pl.col("c") == "x", pl.col("a") > 2], [pl.col("c") == "x"]])
    ],
    *[
        F(f"filter_no_pushdown_comprehensive-{i}", _COMPLEX, e)
        for i, e in enumerate(
            [
                pl.col("int_col").cast(pl.Float64) > 2.5,
                pl.col("str_col").alias("renamed_str") == "c",
                pl.col("nullable_int").is_not_null(),
                pl.col("datetime_utc").dt.hour().alias("utc_hour") >= 12,
                pl.col("datetime_utc").dt.convert_time_zone("US/Eastern").alias("utc_to_est").dt.hour() >= 5,
                pl.when(pl.col("int_col") > 3).then(pl.col("str_col")).otherwise(pl.lit("default")).alias("conditional") == "d",
                pl.when(_LIST_LEN == 0)
                .then(pl.lit("empty"))
                .when(_LIST_LEN == 1)
                .then(pl.lit("single"))
                .otherwise(pl.lit("multiple"))
                .alias("list_category")
                == "multiple",
                _LIST_LEN > 2,
                _LIST_LEN == 0,
                pl.col("list_col").list.sum().alias("list_sum").fill_null(0) > 10,
                # several expressions are ANDed into one temporary column
                [pl.col("int_col") > 2, pl.col("str_col") != "b"],
                [pl.col("bool_col"), pl.col("nullable_int").is_not_null(), pl.col("date_col").dt.year() == 2023],
                [
                    pl.col("bool_col") | pl.col("nullable_int").is_not_null(),
                    pl.col("date_col").dt.year() == 2023,
                    pl.any_horizontal(pl.col("int_col") > 3, pl.col("float_col") < 5.0),
                ],
            ]
        )
    ],
    F("filter_no_pushdown_edge_cases-empty", _COMPLEX, []),
    F(
        "filter_no_pushdown_edge_cases-mixed",
        _COMPLEX,
        [
            (pl.col("int_col").cast(pl.Float64) > 2.0)
            & (pl.col("str_col").alias("renamed") != "x")
            & pl.col("nullable_int").is_not_null()
            & (pl.col("date_col").dt.year() == 2023)
            & (pl.col("datetime_utc").dt.hour().alias("utc_hour") >= 10)
        ],
    ),
]


@pytest.mark.parametrize(("expr", "df", "pushed_plans"), params(NO_PUSHDOWN, "pred", "df", "pushed_plans"))
def test_filter_no_pushdown(expr, df, pushed_plans):
    """filter_no_pushdown gives the rows a normal filter gives, without pushing the filter into the source."""
    lf = io_source_assert(df, lambda _: True)
    empty = isinstance(expr, list) and not expr
    normal = lf if empty else lf.filter(expr)
    # an empty list is only accepted with optimizations on
    variants = [lf.piot.filter_no_pushdown(expr)] + ([] if empty else [lf.piot.filter_no_pushdown(expr, _disable_optimizations=True)])
    for variant in variants:
        assert_frame_equal(variant.collect(), normal.collect())
    if not empty:
        assert_text(variants[0].explain(), "FILTER", omits="SELECTION" if pushed_plans else ())
    if pushed_plans:
        for pushed in (normal, variants[1]):
            assert_text(pushed.explain(), "SELECTION: ", omits="FILTER")


class TestRegisterIoSourceWithIsPure:
    def test_cse_optimization_with_is_pure(self):
        """is_pure=True lets a self-join read the source once."""
        tracker = PredicateTracker(pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]}))
        lf = register_io_source_with_is_pure(_tracking_source(tracker), schema=tracker.df.schema)
        result = lf.join(lf, on="id", how="inner").collect(engine="in-memory" if POLARS_GE_2 else "auto")
        assert_frame_equal(result, pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30], "value_right": [10, 20, 30]}), check_row_order=False)
        if quirk("pure_udf_cse"):
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

    @pytest.mark.parametrize("project", [False, True])
    def test_head_then_filter(self, project):
        df = pl.DataFrame({"a": range(10), "b": range(10, 20)})
        lf = register_io_source_with_is_pure(_tracking_source(PredicateTracker(df)), schema=df.schema)

        def query(lf):
            lf = lf.head(5).filter(pl.col("a") > 2)
            return lf.select("b") if project else lf

        assert_matches_native(query(lf), query(df.lazy()), None, check_row_order=True)

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


def _raises(error, yields_first=False):
    def failing_io_source(*args, **kwargs):
        if yields_first:
            yield pl.DataFrame({"x": [1]})
        raise error

    return failing_io_source


_PRED = (pl.col("x") > 5) & (pl.col("y") < 11)
_LONG = pl.col("data_date").is_between(pl.lit("2024-01-01"), pl.lit("2024-12-31")) & (pl.col("region") == "US") & (pl.col("value") > 100)
_STD = {"with_columns": None, "predicate": None, "n_rows": None, "batch_size": None}


def X(*contains, omits=()):
    return {"contains": contains, "omits": omits}


# (failing source, call args) -> the batches before the error, and the RuntimeError: its cause and message
ERRORS = [
    Case(
        "error_catching_catches_exceptions",
        (_raises(CustomError("Something went wrong!")), (), _STD),
        (
            [],
            CustomError,
            X(
                "IO Source 'failing_io_source' failed",
                "=== IO SOURCE ERROR ===",
                "Function: failing_io_source",
                "Identifier: test",
                "Error Type: CustomError",
                "Error Message: Something went wrong!",
            ),
        ),
    ),
    # the first batch comes through; the error surfaces on the next
    Case(
        "error_catching_with_generator_exception",
        (_raises(ValueError("Generator failed"), yields_first=True), (), _STD),
        ([{"x": [1]}], ValueError, X("IO Source 'failing_io_source' failed", "Generator failed")),
    ),
    Case(
        "error_catching_preserves_arguments_in_error",
        (_raises(Exception("Test error")), (), {"with_columns": ["x", "y"], "predicate": _PRED, "n_rows": 10, "batch_size": 100}),
        ([], Exception, X("Call Arguments:", "with_columns", "predicate", "n_rows", "batch_size", str(_PRED))),
    ),
    # expressions in arguments appear in full, not as a truncated repr
    Case(
        "error_catching_converts_expr_in_args_to_string",
        (_raises(Exception("e")), (None, _LONG, None, 100000), {}),
        ([], Exception, X("data_date", "is_between", "region", "value", omits="<Expr [")),
    ),
    Case(
        "error_catching_handles_mixed_args_and_kwargs",
        (
            _raises(Exception("e")),
            (pl.col("x").cast(pl.Float64) + pl.col("y"), "normal_arg"),
            {"predicate": pl.col("z").is_in([1, 2, 3]), "other": "value"},
        ),
        ([], Exception, X(("cast", "Float64"), "is_in", omits="<Expr [")),
    ),
]


@pytest.mark.parametrize(("call", "expect"), params(ERRORS, "pred", "expect"))
def test_error_catching_raises_with_context(call, expect):
    source, args, kwargs = call
    batches = wrap_io_source_with_error_catching(source, identifier="test")(*args, **kwargs)
    first, cause, text = expect
    got = []
    with pytest.raises(RuntimeError) as exc_info:
        got.extend(b.to_dict(as_series=False) for b in batches)
    assert got == first
    assert type(exc_info.value.__cause__) is cause
    assert_text(str(exc_info.value), **text)


def test_catching_then_debug_logging_no_error(caplog):
    """The wrapper logs nothing itself; a caller's debug log of the error carries the full context."""
    import logging

    wrapped = wrap_io_source_with_error_catching(_raises(CustomError("kaboom!")), identifier="debug_logging_test")
    caplog.set_level(logging.DEBUG)
    try:
        list(wrapped(**_STD))
    except RuntimeError:
        logging.debug("Captured wrapped IO error", exc_info=True)
    assert all(record.levelno < logging.ERROR for record in caplog.records)
    assert_text(
        caplog.text,
        [
            "=== IO SOURCE ERROR ===",
            "Function: failing_io_source",
            "Identifier: debug_logging_test",
            "Error Type: CustomError",
            "Error Message: kaboom!",
        ],
    )


def _topo(lf, exprs):
    return with_columns_topo(lf, exprs)


def _topo_accessor(lf, exprs):
    return lf.piot.with_columns_topo(exprs)


def T(id, base, exprs, expect, call=_topo):
    """``expect`` holds the expected output columns; ``None`` means the input frame unchanged."""
    return Case(id, exprs, expect, {"base": base, "call": call})


_X = {"x": [1, 2, 3]}
_X_PLUS_1, _X_TIMES_2 = (c("x") + 1).alias("b"), (c("x") * 2).alias("c")
_DOUBLE_ALIAS = (c("b") + 1).alias("b2").alias("b3")

TOPO_CASES = [
    T(
        "simple_dependency_two_columns",
        {"b": [1, 2]},
        [(c("b") + 1).name.suffix("_2"), (c("b_2") + 1).name.suffix("_3")],
        {"b_2": [2, 3], "b_2_3": [3, 4]},
    ),
    T("chain_dependencies", {"c": [1, 2, 3]}, [(c("c") * 2).alias("b"), (c("b") + 1).alias("a")], {"c": [1, 2, 3], "b": [2, 4, 6], "a": [3, 5, 7]}),
    T("diamond_dependencies", _X, [(c("b") + c("c")).alias("d"), _X_PLUS_1, _X_TIMES_2], {"b": [2, 3, 4], "c": [2, 4, 6], "d": [4, 7, 10]}),
    T(
        "diamond_dependencies_multiple_input",
        _X,
        [(c("b") + c("c")).alias("e"), _X_PLUS_1, _X_TIMES_2, pl.sum_horizontal(c("b"), c("c")).alias("d")],
        {"b": [2, 3, 4], "c": [2, 4, 6], "e": [4, 7, 10], "d": [4, 7, 10]},
    ),
    T(
        "diamond_dependencies_multiple_input_boolean",
        _X,
        [_X_PLUS_1, _X_TIMES_2, pl.sum_horizontal(c("b"), c("c")).alias("d_sum"), pl.any_horizontal(c("b") > 2, c("c") > 3).alias("d_any")],
        {"b": [2, 3, 4], "c": [2, 4, 6], "d_sum": [4, 7, 10], "d_any": [False, True, True]},
    ),
    T(
        "mixed_interleaved_dependencies",
        {"x": [1, 2]},
        [(c("x") + 1).alias("a"), (c("x") * 3).alias("b"), (c("a") * 2).alias("d"), (c("b") + c("d")).alias("e")],
        {"a": [2, 3], "b": [3, 6], "d": [4, 6], "e": [7, 12]},
        call=_topo_accessor,
    ),
    T("double_alias_chaining", {"b": [1, 2]}, [_DOUBLE_ALIAS], {"b3": [2, 3]}, call=_topo_accessor),
    T(
        "double_alias_chain_dependency_on_final",
        {"b": [1, 2]},
        [_DOUBLE_ALIAS, (c("b3") + 1).alias("b4")],
        {"b3": [2, 3], "b4": [3, 4]},
        call=_topo_accessor,
    ),
    T("empty_exprs_returns_original", _SAMPLE.to_dict(as_series=False), [], None, call=_topo_accessor),
]


@pytest.mark.parametrize(("base", "exprs", "call", "expect"), params(TOPO_CASES, "base", "pred", "call", "expect"))
def test_with_columns_topo(base, exprs, call, expect):
    lf = pl.LazyFrame(base)
    out = call(lf, exprs)
    if expect is None:
        assert_frame_equal(out.collect(), lf.collect())
    else:
        assert_frame_equal(out.select(list(expect)).collect(), pl.DataFrame(expect))


@pytest.mark.parametrize(
    ("base", "exprs", "error", "on_collect"),
    [
        # only the final alias of a chain exists
        pytest.param(
            {"b": [1, 2]},
            [_DOUBLE_ALIAS, (c("b2") + 1).alias("b4")],
            ColumnNotFoundError,
            True,
            id="double_alias_chain_dependency_on_intermediate_errors",
        ),
        pytest.param({"x": [1, 2]}, [(c("b") + 1).alias("a"), (c("a") + 1).alias("b")], CycleError, False, id="cycle_detection"),
    ],
)
def test_with_columns_topo_errors(base, exprs, error, on_collect):
    lf = pl.LazyFrame(base)
    if on_collect:
        out = lf.piot.with_columns_topo(exprs)
        with pytest.raises(error):
            out.collect()
    else:
        with pytest.raises(error):
            lf.piot.with_columns_topo(exprs)


def test_with_columns_topo_independent_batched():
    """Independent expressions share a single ``with_columns`` stage."""
    out = with_columns_topo(pl.LazyFrame(_X), [(c("x") + 1).alias("a"), (c("x") * 2).alias("b"), (c("x") - 3).alias("c")])
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


@pytest.mark.xfail(reason="Selectors expansion not yet supported by with_columns_topo")
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


@pytest.mark.xfail(reason="Implicit selectors expansion not yet supported by with_columns_topo")
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


def S(id, uri, pyarrow, polars, *, region=None, creds=None, env=None, profile=None, log=None):
    """A ``_storage_options_for`` row: boto3 has ``region`` and ``creds`` (access key, secret, token) and the
    environment holds ``env``; the expected option dicts follow."""
    return Case(id, uri, (pyarrow, polars), {"region": region, "creds": creds, "env": env or {}, "profile": profile, "log": log})


_ENV_BOTH = {"AWS_ENDPOINT_URL": "https://endpoint-url", "AWS_S3_ENDPOINT": "https://s3-endpoint"}


def _keys(access, secret, token=None, region=None, endpoint=None):
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
        S(f"non_s3_uri_returns_empty-{n}", uri, {}, {})
        for n, uri in [("file", "file:///tmp/cache"), ("local", "/local/path/cache"), ("http", "http://example.com/cache")]
    ],
    # an endpoint in the query wins over the environment
    S(
        "s3_with_credentials_and_token",
        "s3://bucket/path?endpoint_override=https://query-endpoint",
        *_keys("AKIA_TEST", "SECRET_TEST", "TOKEN_TEST", "us-west-2", "https://query-endpoint"),
        region="us-west-2",
        creds=("AKIA_TEST", "SECRET_TEST", "TOKEN_TEST"),
        env={"AWS_ENDPOINT_URL": "https://env-endpoint", "AWS_S3_ENDPOINT": "https://env-s3-endpoint"},
        profile="test-profile",
    ),
    # region is a pyarrow option only
    S(
        "s3_with_credentials_no_token",
        "s3://bucket/path",
        *_keys("AKIA_NO_TOKEN", "SECRET_NO_TOKEN", None, "eu-central-1", "https://env-only-endpoint"),
        region="eu-central-1",
        creds=("AKIA_NO_TOKEN", "SECRET_NO_TOKEN", None),
        env={"AWS_ENDPOINT_URL": "https://env-only-endpoint"},
    ),
    S("s3a_scheme_supported", "s3a://bucket/path", *_keys("AKIA_S3A", "SECRET_S3A"), creds=("AKIA_S3A", "SECRET_S3A", None)),
    # AWS_ENDPOINT_URL wins over AWS_S3_ENDPOINT, a query endpoint over both
    S(
        "endpoint_env_precedence-env",
        "s3://bucket/path",
        *_keys("AKIA_ENV", "SECRET_ENV", None, "ap-southeast-1", "https://endpoint-url"),
        region="ap-southeast-1",
        creds=("AKIA_ENV", "SECRET_ENV", None),
        env=_ENV_BOTH,
    ),
    S(
        "endpoint_env_precedence-query",
        "s3://bucket/path?endpoint_override=https://query",
        *_keys("AKIA_ENV", "SECRET_ENV", None, "ap-southeast-1", "https://query"),
        region="ap-southeast-1",
        creds=("AKIA_ENV", "SECRET_ENV", None),
        env=_ENV_BOTH,
    ),
    S(
        "no_credentials_only_region_and_endpoint",
        "s3://bucket/path",
        *_keys(None, None, None, "us-east-1", "https://only-env-endpoint"),
        region="us-east-1",
        env={"AWS_S3_ENDPOINT": "https://only-env-endpoint"},
    ),
    S(
        "profile_name_forwarded_to_boto3_session",
        "s3://bucket/path",
        *_keys("AKIA_PROFILE", "SECRET_PROFILE"),
        creds=("AKIA_PROFILE", "SECRET_PROFILE", None),
        profile="dev-profile",
    ),
    # an unresolvable endpoint host is kept as configured
    S(
        "storage_options_resolution_failure_keeps_original",
        "s3://bucket/path",
        *_keys("AKIA_TEST", "SECRET_TEST", None, "us-east-1", "http://grid:9020"),
        region="us-east-1",
        creds=("AKIA_TEST", "SECRET_TEST", None),
        env={"AWS_ENDPOINT_URL": "http://grid:9020"},
        log="Failed to resolve hostname",
    ),
]


@pytest.mark.parametrize(
    ("uri", "expect", "region", "creds", "env", "profile", "log"),
    params(STORAGE_OPTIONS, "pred", "expect", "region", "creds", "env", "profile", "log"),
)
def test_storage_options_for(monkeypatch, caplog, uri, expect, region, creds, env, profile, log):
    import socket
    from types import SimpleNamespace

    sessions = []

    class FakeSession:
        def __init__(self, profile_name=None):
            sessions.append(profile_name)
            self.region_name = region

        def get_credentials(self):
            return None if creds is None else SimpleNamespace(access_key=creds[0], secret_key=creds[1], token=creds[2])

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
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    _resolve_endpoint_hostname.cache_clear()
    try:
        pyarrow, polars, provider = opts = _storage_options_for(uri, aws_profile=profile)
    finally:
        _resolve_endpoint_hostname.cache_clear()
    assert (opts.pyarrow, opts.polars) == (pyarrow, polars) == expect
    s3 = uri.startswith("s3")
    assert (provider is not None) == s3
    assert sessions == ([profile] if s3 else [])
    assert log is None or log in caplog.text


class _Lookups(list):
    ip: str | None = None


@pytest.fixture
def lookups(monkeypatch):
    """Hostnames passed to ``socket.gethostbyname``, which resolves each to ``lookups.ip``."""
    _resolve_endpoint_hostname.cache_clear()
    calls = _Lookups()

    def fake(hostname):
        calls.append(hostname)
        return calls.ip

    monkeypatch.setattr("socket.gethostbyname", fake)
    yield calls
    _resolve_endpoint_hostname.cache_clear()


@pytest.mark.parametrize(
    ("endpoint", "ip", "expect", "hostname"),
    [
        pytest.param("http://grid:9020", "10.1.2.3", "http://10.1.2.3:9020", "grid", id="resolve_hostname_with_port"),
        pytest.param("http://.myhost.int.:9020", "10.4.5.6", "http://10.4.5.6:9020", ".myhost.int.", id="resolve_hostname_with_fqdn_and_port"),
        pytest.param("http://example.com", "192.168.1.100", "http://192.168.1.100", "example.com", id="resolve_hostname_without_port"),
        pytest.param("https://secure.example.com:443", "10.0.0.1", "https://10.0.0.1:443", "secure.example.com", id="resolve_https_endpoint"),
        pytest.param("http://grid:9020/some/path", "10.1.2.3", "http://10.1.2.3:9020/some/path", "grid", id="resolve_hostname_with_path"),
        pytest.param("file:///path/to/file", "10.1.2.3", "file:///path/to/file", None, id="resolve_no_hostname_returns_original"),
    ],
)
def test_resolve_endpoint_hostname(lookups, endpoint, ip, expect, hostname):
    lookups.ip = ip
    assert _resolve_endpoint_hostname(endpoint) == expect
    assert lookups == ([hostname] if hostname else [])


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


@pytest.mark.parametrize(
    ("df", "known", "columns", "expect"),
    [
        pytest.param({"a": [1]}, pl.DataFrame({"a": []}, schema={"a": pl.Int64}), ["a"], None, id="none_when_nothing_to_exclude-no_known_rows"),
        pytest.param({"a": [1]}, pl.DataFrame({"a": [1]}), [], None, id="none_when_nothing_to_exclude-no_columns"),
        pytest.param(
            {"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]},
            pl.DataFrame({"a": [1, 3]}),
            ["a"],
            {"a": [2, 4], "b": [20, 40]},
            id="single_column_excludes_known",
        ),
        # excludes (1, 10) and (2, 20) only
        pytest.param(
            {"a": [1, 1, 2, 2], "b": [10, 20, 10, 20]},
            pl.DataFrame({"a": [1, 2], "b": [10, 20]}),
            ["a", "b"],
            {"a": [1, 2], "b": [20, 10]},
            id="multi_column_excludes_only_exact_combos",
        ),
        pytest.param({"a": [1, 2, 3]}, pl.DataFrame({"a": [1, 1, 1, 2]}), ["a"], {"a": [3]}, id="duplicates_in_known_are_deduplicated"),
    ],
)
def test_partition_exclusion_predicate(df, known, columns, expect):
    pred = partition_exclusion_predicate(known, columns)
    if expect is None:
        assert pred is None
    else:
        assert pl.DataFrame(df).filter(pred).sort(list(df)).to_dict(as_series=False) == expect


@pytest.mark.parametrize(
    ("known_vals", "expected"),
    [
        pytest.param([1, 3], [None, 5], id="null_not_known_is_kept"),
        pytest.param([1, None], [3, 5], id="null_known_is_excluded"),
        pytest.param([3], [1, None, 5], id="single_value"),
        pytest.param([None], [1, 3, 5], id="only_null"),
    ],
)
def test_partition_exclusion_predicate_null_safe_matches_eq_missing(known_vals, expected):
    df = pl.DataFrame({"a": [1, None, 3, 5]}, schema={"a": pl.Int64})
    known = pl.DataFrame({"a": known_vals}, schema={"a": pl.Int64})
    assert df.filter(partition_exclusion_predicate(known, ["a"]))["a"].to_list() == expected
    reference = functools.reduce(operator.and_, [~c("a").eq_missing(v) for v in known_vals])
    assert df.filter(reference)["a"].to_list() == expected


def test_partition_exclusion_predicate_remains_analyzable_for_contradiction_detection():
    """Querying for a in {1, 3} while excluding {1, 3} is a contradiction the DNF analysis can prove."""
    from polars_io_tools.io_sources.dnf_visitor import _is_contradiction

    excl = partition_exclusion_predicate(pl.DataFrame({"a": [1, 3]}, schema={"a": pl.Int64}), ["a"])
    assert _is_contradiction(c("a").is_in([1, 3]) & excl) is True
