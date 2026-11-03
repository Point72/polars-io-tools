"""execute_on_ray: distributed reads equal the same query on one frame, with the expected number of Ray tasks.

``DISTRIBUTED`` rows give a frame, the partitions, a downstream query and the task count; the test compares the
distributed result with the query on the undistributed frame. Scheduling, ordering and failure tests patch the
worker and stay functions.
"""

import datetime
import gc
import random
import sys
import time
from datetime import date, timedelta

import polars as pl
import polars.selectors as cs
import pytest
from polars.testing import assert_frame_equal

ray = pytest.importorskip("ray", exc_type=ImportError)

import polars_io_tools as cpl
import polars_io_tools.io_sources.lazy_ray  # explicit import needed if using pytest-xdist
from polars_io_tools.io_sources.lazy_ray import RayPartition, cartesian_partitions, discrete_partitions
from polars_io_tools.io_sources.partitions import by_key, by_time, by_value
from polars_io_tools.io_sources.pushdown_combine import FilterSpec, pushdown_combine
from polars_io_tools.tests.helpers.cases import Case, params

# Ray's task cancellation is unstable on Windows (its Windows support is beta): cancelling an
# in-flight task while the Polars source generator is being closed early (n_rows satisfied, or a
# failing partition) can segfault the interpreter. The behavior under test is platform-agnostic
# and fully covered on Linux/macOS, so these early-termination cases are skipped on Windows.
_ray_cancel_flaky_on_windows = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Ray task cancellation on early source termination can crash the interpreter on Windows (Ray Windows support is beta).",
)

# Cap Ray's plasma object store and tolerate re-init. CI containers default to a tiny
# /dev/shm (~64MB), so an uncapped object store over-allocates and OOM-kills workers,
# crashing the suite mid-run.
_RAY_INIT_KWARGS = {"num_cpus": 2, "object_store_memory": 200 * 1024 * 1024, "ignore_reinit_error": True}


@pytest.fixture(scope="session", autouse=True)
def shared_ray_cluster():
    """Start a tiny Ray cluster once for all tests and shut it down at the end of the session."""
    if not ray.is_initialized():
        ray.init(**_RAY_INIT_KWARGS)
    yield
    ray.shutdown()


@pytest.fixture(autouse=True)
def _gc_after_test():
    """Force garbage collection after each test.

    Ray task/actor references linger via cyclic references and traceback locals
    (``sys.last_traceback`` and friends), so their processes and object-store memory
    accumulate across this file's many Ray tests until a worker is OOM-killed in CI.
    Reclaiming eagerly between tests keeps the footprint bounded.
    """
    yield
    for attr in ("last_traceback", "last_value", "last_type"):
        try:
            delattr(sys, attr)
        except AttributeError:
            pass
    gc.collect()


def generate_sample_lazyframe(start: datetime.datetime, end: datetime.datetime) -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "date": (dates := pl.datetime_range(start, end, interval="1d", eager=True)),
            "quantity": range(len(dates)),
            "price": [(100 + i) for i in range(len(dates))],
        }
    )


N_IDS = 6


def generate_panel(n_ids: int = N_IDS, n_days: int = 10) -> pl.LazyFrame:
    """A small (id x date) cross-section with a deterministic value column."""
    dates = pl.date_range(date(2023, 1, 1), date(2023, 1, n_days), interval="1d", eager=True)
    panel = pl.DataFrame({"id": list(range(n_ids))}).join(pl.DataFrame({"date": dates}), how="cross")
    return panel.with_columns(val=(pl.col("id") * 100 + pl.col("date").dt.day())).lazy()


def _category_lazyframe(start, end):
    dates = pl.datetime_range(start, end, interval="1d", eager=True)
    return pl.LazyFrame({"date": dates, "category": [("A", "B", "C")[i % 3] for i in range(len(dates))], "val": range(len(dates))})


def _patch_options(monkeypatch, wrap_remote):
    """Route every task submission ``_execute_partition.options(**kw).remote(...)`` through ``wrap_remote(kw, remote)``."""
    original_options = cpl.io_sources.lazy_ray._execute_partition.options

    def options(**kw):
        stub = original_options(**kw)
        stub.remote = wrap_remote(kw, stub.remote)  # type: ignore[attr-defined]
        return stub

    monkeypatch.setattr(cpl.io_sources.lazy_ray._execute_partition, "options", options)


def _count_tasks(monkeypatch):
    calls = {"n": 0}

    def wrap(_kw, remote):
        def counting_remote(*args, **kwargs):
            calls["n"] += 1
            return remote(*args, **kwargs)

        return counting_remote

    _patch_options(monkeypatch, wrap)
    return calls


# Distributed reads equal the undistributed query.

_Y23 = (datetime.datetime(2023, 1, 1), datetime.datetime(2023, 12, 31))
_JAN10 = (datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 10))
_WINDOWS = [(date(2023, 1, 1), date(2023, 1, 6)), (date(2023, 1, 6), date(2023, 1, 11))]
_CARTESIAN = {"date_windows": _WINDOWS, "buckets": [[0, 1, 2], [3, 4, 5]], "date_column": "date", "bucket": "id"}
_MID = datetime.datetime(2023, 2, 15)


def _sorted_by(*cols):
    return lambda lf: lf.sort(*cols)


_by_id_date = _sorted_by("id", "date")


def R(id, frame, parts, query, tasks=None, native=None):
    """``frame()`` distributed by ``parts`` (keyword arguments if a dict) then ``query`` equals ``native`` (default: ``query``) on ``frame()``."""
    return Case(id, frame=frame, parts=parts, query=query, tasks=tasks, native=native)


def _f(*a, **k):
    return lambda: generate_panel(*a, **k)


_NAN = float("nan")
DISTRIBUTED = [
    R(
        "actually_partitioned",
        lambda: generate_sample_lazyframe(*_Y23),
        {"date_column": "date", "time_unit": "daily"},
        lambda lf: lf.filter(pl.col("date").is_between(*_Y23)).sort("date"),
        365,
    ),
    R(
        "closed_both_ray_tasks",
        lambda: generate_sample_lazyframe(*_JAN10),
        {"date_column": "date", "time_unit": "daily"},
        lambda lf: lf.filter(pl.col("date").is_between(*_JAN10)).sort("date"),
        10,
    ),
    # The upper-bound partition is pruned.
    R(
        "closed_left_produces_fewer_ray_tasks",
        lambda: generate_sample_lazyframe(*_JAN10),
        {"date_column": "date", "time_unit": "daily"},
        lambda lf: lf.filter(pl.col("date").is_between(*_JAN10, closed="left")).sort("date"),
        9,
    ),
    R("by_column_scalar_keys", _f(), by_key(range(N_IDS), "id"), _by_id_date, N_IDS),
    R("by_expr_hash_bucket", _f(), by_key(range(3), pl.col("id").hash() % 3), _by_id_date, 3),
    R("discrete_partitions_is_in", _f(), discrete_partitions("id", [[0, 1], [2, 3], [4, 5]]), _by_id_date, 3),
    R("cartesian_date_x_bucket", _f(), cartesian_partitions(**_CARTESIAN), _by_id_date, 4),
    # The key column feeds the worker predicate, then is dropped.
    R("column_retention_under_projection", _f(), by_key(range(N_IDS), "id"), lambda lf: lf.select("val").sort("val")),
    R(
        "by_selector_multi_column",
        lambda: generate_panel(3, 4).with_columns(grp=pl.col("id") % 2),
        by_key(pl.DataFrame({"id": [0, 1, 2], "grp": [0, 1, 0]}), cs.by_name("id", "grp")),
        _by_id_date,
    ),
    R(
        "selector_expanded_predicate_retains_columns",
        lambda: pl.LazyFrame({"a": [1, -1, 1], "b": [1, 1, -1], "v": [10, 20, 30]}),
        [RayPartition(pl.all_horizontal(pl.col("a", "b") > 0), key="pos"), RayPartition(~pl.all_horizontal(pl.col("a", "b") > 0), key="neg")],
        lambda lf: lf.select("v").sort("v"),
    ),
    R(
        "by_expr_null_key",
        lambda: pl.LazyFrame({"id": [None, 1, 2], "v": [10, 20, 30]}, schema={"id": pl.Int64, "v": pl.Int64}),
        by_key([None, 1, 2], pl.col("id")),
        _sorted_by("v"),
    ),
    R("by_series_name_mismatch", _f(3, 4), by_key(pl.Series("whatever", [0, 1, 2]), "id"), _by_id_date),
    # Pruning: a downstream key predicate launches only matching partitions, and never drops a matching one.
    R("pruning_equality_key", _f(), by_key(range(N_IDS), "id"), lambda lf: lf.filter(pl.col("id") == 3).sort("date"), 1),
    R(
        "partitions_primitive_no_pruning_correct_under_filter",
        _f(),
        discrete_partitions("id", [[0, 1], [2, 3], [4, 5]]),
        lambda lf: lf.filter(pl.col("id").is_in([0, 5])).sort("id", "date"),
        3,
    ),
    R(
        "cartesian_no_pruning_correct_under_filter",
        _f(),
        cartesian_partitions(**_CARTESIAN),
        lambda lf: lf.filter(pl.col("date") < date(2023, 1, 6)).filter(pl.col("id").is_in([0, 1, 2])).sort("id", "date"),
        4,
    ),
    R("pruning_fail_open_unrelated_predicate", _f(), by_key(range(N_IDS), "id"), lambda lf: lf.filter(pl.col("val") > 500).sort("id", "date"), N_IDS),
    R("pruning_never_drops_matching", _f(), by_key(range(N_IDS), "id"), lambda lf: lf.filter(pl.col("id") >= 2).sort("id", "date"), 4),
    R(
        "pruning_no_false_prune_nan_in_is_in",
        lambda: pl.LazyFrame({"x": [2.0, _NAN], "v": [1, 2]}),
        discrete_partitions("x", [[2.0, _NAN]]),
        lambda lf: lf.filter(pl.col("x").is_in([1.0, _NAN])).sort("v"),
    ),
    R(
        "pruning_no_false_prune_float_coercion",
        lambda: pl.LazyFrame({"id": [2**53 + 1], "w": [7]}, schema={"id": pl.Int64, "w": pl.Int64}),
        by_key([2**53 + 1], "id"),
        lambda lf: lf.filter(pl.col("id") == float(2**53)),
    ),
    R(
        "by_expr_cast_no_false_prune",
        lambda: pl.LazyFrame({"id": [2]}, schema={"id": pl.Int64}),
        by_key([True], pl.col("id").cast(pl.Boolean)),
        lambda lf: lf.filter(pl.col("id") == 2),
    ),
    R(
        "partitions_multicolumn_null_no_false_prune",
        lambda: pl.LazyFrame({"c": [None], "d": [1]}, schema={"c": pl.String, "d": pl.Int64}),
        [RayPartition(pl.col("c").is_in(["b", None], nulls_equal=True) & (pl.col("d") == 1), key="p")],
        lambda lf: lf.filter(pl.col("c").is_in(["a", None], nulls_equal=True) & (pl.col("d") == 1)),
    ),
    R(
        "pruning_by_nan_key_exact",
        lambda: pl.LazyFrame({"x": [1.0, _NAN], "v": [1, 2]}),
        by_key([1.0, _NAN], "x"),
        lambda lf: lf.filter(pl.col("x").is_in([_NAN])).sort("v"),
    ),
    # Float keys are not observational under == (-0.0 vs +0.0), so pruning fails open.
    R(
        "pruning_float_key_no_false_prune",
        lambda: pl.LazyFrame({"x": [-0.0], "v": [1]}),
        by_key([0.0], "x"),
        lambda lf: lf.filter((pl.lit(1.0) / pl.col("x")) < 0),
    ),
    R(
        "cartesian_none_bucket_matches_nulls",
        lambda: pl.LazyFrame(
            {"date": [date(2026, 1, 1)] * 2, "desk": [None, "rates"], "value": [100, 200]},
            schema={"date": pl.Date, "desk": pl.String, "value": pl.Int64},
        ),
        cartesian_partitions(
            date_windows=[(date(2026, 1, 1), date(2026, 1, 2))], buckets={"missing-desk": None, "rates": "rates"}, date_column="date", bucket="desk"
        ),
        _sorted_by("value"),
    ),
    R(
        "discrete_partitions_none_member_matches_nulls",
        lambda: pl.LazyFrame({"id": [None, 1, 2], "v": [10, 20, 30]}, schema={"id": pl.Int64, "v": pl.Int64}),
        discrete_partitions("id", [[1, None], [2]]),
        _sorted_by("v"),
    ),
    # A pushed selector predicate (no root names) must see every column.
    R(
        "selector_downstream_predicate_retains_columns",
        lambda: pl.LazyFrame({"a": [1, None], "b": [None, 2]}, schema={"a": pl.Int64, "b": pl.Int64}),
        discrete_partitions("a", [[1], [2]]),
        lambda lf: lf.filter(pl.all_horizontal(pl.all().is_not_null())).select("a").sort("a"),
    ),
    # Unified ``partitions=`` entry point
    R(
        "unified_by_time_matches_calendar_and_single_frame",
        lambda: (
            generate_sample_lazyframe(datetime.datetime(2023, 1, 1), datetime.datetime(2023, 6, 30))
            .filter(pl.col("quantity") > 10)
            .with_columns(pl.col("price") * 2)
        ),
        cpl.by_time("date", "1mo"),
        lambda lf: lf.filter(pl.col("date").is_between(datetime.datetime(2023, 1, 1), datetime.datetime(2023, 6, 30))).sort("date"),
    ),
    R(
        "unified_by_time_matches_calendar_and_single_frame-legacy",
        lambda: (
            generate_sample_lazyframe(datetime.datetime(2023, 1, 1), datetime.datetime(2023, 6, 30))
            .filter(pl.col("quantity") > 10)
            .with_columns(pl.col("price") * 2)
        ),
        {"date_column": "date", "time_unit": "monthly"},
        lambda lf: lf.filter(pl.col("date").is_between(datetime.datetime(2023, 1, 1), datetime.datetime(2023, 6, 30))).sort("date"),
    ),
    R(
        "unified_by_value_derived_from_predicate",
        lambda: _category_lazyframe(datetime.datetime(2023, 1, 1), datetime.datetime(2023, 3, 31)),
        cpl.by_value("category"),
        lambda lf: lf.filter(pl.col("category").is_in(["A", "B"])).sort("date"),
    ),
    R(
        "unified_explicit_read_partition_list",
        lambda: generate_sample_lazyframe(datetime.datetime(2023, 1, 1), datetime.datetime(2023, 3, 31)),
        [cpl.ReadPartition(pl.col("date") < _MID, key="lo"), cpl.ReadPartition(pl.col("date") >= _MID, key="hi")],
        _sorted_by("date"),
    ),
    # An explicit value list needs no pushed-down predicate.
    R(
        "unified_by_value_explicit_needs_no_predicate",
        lambda: _category_lazyframe(datetime.datetime(2023, 1, 1), datetime.datetime(2023, 2, 28)),
        by_value("category", ["A", "B"]),
        _sorted_by("date"),
        native=lambda lf: lf.filter(pl.col("category").is_in(["A", "B"])).sort("date"),
    ),
    # A contradictory date filter yields no partitions and an empty frame.
    R(
        "unified_by_time_empty_range_returns_empty",
        lambda: generate_sample_lazyframe(*_Y23),
        by_time("date", "1mo"),
        lambda lf: lf.filter((pl.col("date") >= datetime.datetime(2023, 6, 1)) & (pl.col("date") < datetime.datetime(2023, 1, 1))),
        0,
    ),
]


@pytest.mark.parametrize(("frame", "parts", "query", "tasks", "native"), params(DISTRIBUTED, "frame", "parts", "query", "tasks", "native"))
def test_distributed_matches_single_frame(monkeypatch, frame, parts, query, tasks, native):
    calls = _count_tasks(monkeypatch)
    lf = frame()
    distributed = lf.piot.execute_on_ray(**parts) if isinstance(parts, dict) else lf.piot.execute_on_ray(parts)
    result = query(distributed).collect()
    assert_frame_equal(result, (native or query)(lf).collect())
    if tasks is not None:
        assert calls["n"] == tasks


def test_basic():
    """A pipeline distributed by month still takes further filters."""
    lf = generate_sample_lazyframe(*_Y23)
    result = (
        lf.filter(pl.col("quantity") > 100)
        .with_columns(pl.col("price") * 2)
        .piot.execute_on_ray(date_column="date", time_unit="monthly")
        .filter(pl.col("date").is_between(*_Y23))
        .sort("date")
    )
    assert isinstance(result, pl.LazyFrame)
    assert result.collect().shape == (264, 3)
    assert result.filter(pl.col("price") % 2 != 0).collect().shape == (0, 3)
    assert result.filter(pl.col("price") % 4 == 0).filter(pl.col("quantity") % 7 != 0).collect().shape == (113, 3)


def test_results_are_in_chronological_order(monkeypatch):
    """Rows come back in date order even when Ray tasks finish in random order."""
    lf = generate_sample_lazyframe(s := datetime.datetime(2023, 1, 1), e := datetime.datetime(2023, 1, 15))
    original_remote = cpl.io_sources.lazy_ray._execute_partition.remote

    @ray.remote
    def _slow_execute_partition(*args, **kwargs):
        time.sleep(random.uniform(0.01, 0.2))
        return ray.get(original_remote(*args, **kwargs))

    monkeypatch.setattr(cpl.io_sources.lazy_ray._execute_partition, "remote", lambda *a, **k: _slow_execute_partition.remote(*a, **k))
    dates = lf.piot.execute_on_ray(date_column="date", time_unit="daily").filter(pl.col("date").is_between(s, e)).collect()["date"].to_list()
    assert dates == sorted(dates)
    assert len(dates) == 15


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "date_column": "date",
                "time_unit": "daily",
                "remote_options": {"num_cpus": 0.25, "runtime_env": {"env_vars": {"POLARS_MAX_THREADS": "2"}}},
            },
            None,
        ),
        (
            {"partitions": by_key(pl.DataFrame({"id": [0, 1, 2], "ray": [{"num_cpus": 1}] * 3}), "id", partition_remote_options="ray")},
            {"num_cpus": 1},
        ),
        ({"partitions": by_key([0, 1, 2], "id", partition_remote_options={0: {"num_cpus": 1}})}, None),
    ],
    ids=["remote_options_forwarded", "per_partition_remote_options_struct_column", "per_partition_remote_options_mapping"],
)
def test_remote_options_forwarded(monkeypatch, kwargs, expected):
    """Ray receives the given remote options, globally or per partition."""
    seen = []
    _patch_options(monkeypatch, lambda kw, remote: seen.append(kw) or remote)
    lf = generate_panel(n_ids=3) if "partitions" in kwargs else generate_sample_lazyframe(*_Y23)
    if "partitions" in kwargs:
        assert_frame_equal(lf.piot.execute_on_ray(kwargs["partitions"]).sort("id", "date").collect(), lf.sort("id", "date").collect())
    else:
        lf.piot.execute_on_ray(**kwargs).filter(pl.col("date").is_between(*_Y23)).collect()
    if expected is not None:
        assert seen and all(kw.get("num_cpus") == 1 for kw in seen)
    elif "remote_options" in kwargs:
        assert seen and all(kw == kwargs["remote_options"] for kw in seen)


def test_max_concurrency(monkeypatch):
    """No more than ``max_concurrency`` tasks run at once."""
    max_conc = 2
    active = [0]

    @ray.remote
    def fake_part(*_a, **_k):
        active[0] += 1
        assert active[0] <= max_conc
        time.sleep(0.05)
        active[0] -= 1
        return pl.DataFrame().to_arrow()

    monkeypatch.setattr(cpl.io_sources.lazy_ray._execute_partition, "remote", fake_part.remote)
    s, e = datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 10)
    lf = pl.LazyFrame({"date": pl.datetime_range(s, e, "1d", eager=True)})
    lf.piot.execute_on_ray(date_column="date", time_unit="daily", max_concurrency=max_conc, remote_options={"num_cpus": 0}).filter(
        pl.col("date").is_between(s, e)
    ).collect()


def test_no_ray():
    """Without a Ray cluster the read raises."""
    ray.shutdown()
    lf = generate_sample_lazyframe(*_Y23)
    try:
        with pytest.raises(RuntimeError):
            lf.piot.execute_on_ray(date_column="date", time_unit="monthly").filter(pl.col("date").is_between(*_Y23)).collect()
    finally:
        ray.init(**_RAY_INIT_KWARGS)


@pytest.mark.parametrize("upstream_filter", [None, pl.col("quantity") > 10], ids=["execute_on_ray_pickle_basic", "execute_on_ray_pickle_with_filter"])
def test_execute_on_ray_pickle(shared_ray_cluster, upstream_filter):
    """An execute_on_ray frame survives a cloudpickle round trip, with its schema and results."""
    import cloudpickle

    s, e = datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 31)
    lf = generate_sample_lazyframe(s, e)
    if upstream_filter is not None:
        lf = lf.filter(upstream_filter)
    ray_lf = lf.piot.execute_on_ray(date_column="date", time_unit="daily").filter(pl.col("date").is_between(s, e))
    unpickled = cloudpickle.loads(cloudpickle.dumps(ray_lf))
    assert ray_lf.collect_schema() == unpickled.collect_schema()
    assert_frame_equal(ray_lf.sort("date").collect(), unpickled.sort("date").collect())


def test_empty_partition_result_schema():
    lf = generate_panel()
    result = lf.piot.execute_on_ray(by_key([999], "id")).collect()
    assert result.height == 0
    assert result.schema == lf.collect_schema()


@pytest.mark.parametrize(
    ("lf", "keys"),
    [
        (generate_panel(), [1, 1]),
        (pl.LazyFrame({"x": [1.0, _NAN], "v": [1, 2]}), [_NAN, _NAN]),
        # Both keys match the microsecond 1970-01-01 00:00:00.000001.
        (pl.LazyFrame({"t": pl.Series([1], dtype=pl.Datetime("us")), "v": [1]}), pl.Series([1000, 1500], dtype=pl.Datetime("ns"))),
    ],
    ids=["duplicate_key_rejected", "duplicate_nan_keys_rejected", "keys_equal_in_the_column_dtype_rejected"],
)
def test_duplicate_key_rejected(lf, keys):
    with pytest.raises(ValueError, match="Duplicate partition key"):
        lf.piot.execute_on_ray(by_key(keys, lf.collect_schema().names()[0])).collect()


def test_completion_order_and_preserve_order():
    lf = generate_panel()
    specs = discrete_partitions("id", [[i] for i in range(N_IDS)])
    assert_frame_equal(lf.piot.execute_on_ray(specs).sort("id", "date").collect(), lf.sort("id", "date").collect())
    ids = lf.piot.execute_on_ray(specs, preserve_partition_order=True).collect()["id"].to_list()
    assert ids == sorted(ids)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"partitions": discrete_partitions("id", [[0]]), "return_as": "bogus"}, "return_as"),
        ({"partitions": discrete_partitions("id", [[0]]), "max_concurrency": 0}, "max_concurrency"),
        ({"partitions": cpl.by_time("date", "1mo"), "date_column": "date", "time_unit": "monthly"}, "either"),
    ],
    ids=["validation_errors-return_as", "validation_errors-max_concurrency", "unified_partitions_and_legacy_are_mutually_exclusive"],
)
def test_validation_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        generate_panel().piot.execute_on_ray(**kwargs)


def test_invariant1_lookback_self_padding():
    """Byte-identical single-frame vs distributed collect through a FilterSpec(lookback).

    Each date-window partition must self-pad its lookback: the partition predicate must
    reach the inner source so it expands the read window, computes the rolling value, and
    trims after combine. If pruning consumed the predicate, boundary rows would be wrong.
    """
    data = pl.LazyFrame({"date": [date(2024, 1, i) for i in range(1, 13)], "val": [float(i) for i in range(1, 13)]})
    lf = pushdown_combine(
        sources={"data": (data, {"date": FilterSpec(lookback=timedelta(days=3))})},
        combine=lambda s: s["data"].sort("date").with_columns(pl.col("val").rolling_sum(window_size=3, min_samples=1).alias("rs")),
    )
    windows = [(date(2024, 1, 1), date(2024, 1, 5)), (date(2024, 1, 5), date(2024, 1, 9)), (date(2024, 1, 9), date(2024, 1, 13))]
    specs = [RayPartition((pl.col("date") >= lo) & (pl.col("date") < hi), key=i) for i, (lo, hi) in enumerate(windows)]
    assert_frame_equal(lf.sort("date").collect(), lf.piot.execute_on_ray(specs).sort("date").collect())


@pytest.mark.parametrize(
    "keys",
    [
        pl.Series("x", [1_730_611_800_000_000, 1_730_615_400_000_000]).cast(pl.Datetime("us", "America/New_York")),
        pl.Series("x", [1_000, 1_001]).cast(pl.Datetime("ns")),
        pl.Series("x", [1_000, 1_001]).cast(pl.Time),
    ],
    ids=["dst_fold", "ns", "time"],
)
def test_keys_equal_as_python_objects_are_distinct(keys):
    lf = pl.LazyFrame({"x": keys, "v": [1, 2]}).piot.execute_on_ray(by_key(keys.to_frame(), "x"))
    assert lf.collect().sort("v")["v"].to_list() == [1, 2]
    assert lf.filter(pl.col("x") == pl.lit(keys[1:])).collect()["v"].to_list() == [2]


def test_pruning_python_udf_no_false_prune():
    """A Python UDF predicate must not be pruned via single-key evaluation."""

    class EverySecond:
        def __init__(self):
            self.n = 0

        def __call__(self, _):
            self.n += 1
            return self.n % 2 == 0

    lf = pl.DataFrame({"id": [1, 1], "v": [10, 11]}, schema={"id": pl.Int64, "v": pl.Int64}).lazy()
    udf = EverySecond()
    predicate = pl.col("id").map_elements(udf, return_dtype=pl.Boolean)
    result = polars_io_tools.io_sources.lazy_ray._prune_by_key([RayPartition(pl.col("id") == 1, key=1)], ["id"], predicate, lf.collect_schema())
    assert [p.key for p in result] == [1]
    assert udf.n == 0


def test_explicit_generator_partitions_are_reusable():
    """A generator of partitions is materialized once, so a second collect is not empty."""
    lf = generate_sample_lazyframe(datetime.datetime(2023, 1, 1), datetime.datetime(2023, 3, 31))

    def gen():
        yield cpl.ReadPartition(pl.col("date") < _MID, key="lo")
        yield cpl.ReadPartition(pl.col("date") >= _MID, key="hi")

    ray_lf = lf.piot.execute_on_ray(gen())
    first = ray_lf.sort("date").collect()
    assert first.height == lf.collect().height
    assert_frame_equal(first, ray_lf.sort("date").collect())


# ---------------------------------------------------------------------------
# Bounded, ordered fan-out (deque window). Tasks are always submitted in spec
# order, so the k-th submission corresponds to spec index k; the helpers below
# exploit that to force deterministic completion / failure schedules. They patch
# ``.options`` (not ``.remote``) because the executor submits via
# ``_execute_partition.options(...).remote(...)``.
# ---------------------------------------------------------------------------


@ray.remote(num_cpus=0)
def _sleep_then(delay, boxed_ref):
    time.sleep(delay)
    return ray.get(boxed_ref[0])


@ray.remote(num_cpus=0)
def _raise_or(fail, boxed_ref):
    if fail:
        raise RuntimeError("worker boom")
    return ray.get(boxed_ref[0])


def _patch_worker(monkeypatch, wrap):
    """Route each real task's ObjectRef through ``wrap(k, submit_real)``.

    ``k`` is the 0-based submission index (== spec index); ``submit_real()`` submits the genuine
    partition task and returns its ObjectRef. Refs are boxed in a list so Ray does not eagerly
    resolve them as task arguments.
    """
    counter = {"n": 0}

    def wrap_remote(_kw, remote):
        def wrapped_remote(*args, **kwargs):
            k = counter["n"]
            counter["n"] += 1
            return wrap(k, lambda: remote(*args, **kwargs))

        return wrapped_remote

    _patch_options(monkeypatch, wrap_remote)


def _patch_delays(monkeypatch, delays):
    """Make the k-th submitted task complete after ``delays[k]`` extra seconds."""

    def wrap(k, submit_real):
        delay = delays[k] if k < len(delays) else 0.0
        return _sleep_then.remote(delay, [submit_real()])

    _patch_worker(monkeypatch, wrap)


def _patch_failures(monkeypatch, fail_indices):
    """Make tasks at the given spec indices raise inside the worker."""

    def wrap(k, submit_real):
        return _raise_or.remote(k in fail_indices, [submit_real()])

    _patch_worker(monkeypatch, wrap)


@pytest.mark.parametrize(
    "delays",
    [
        [0.20, 0.15, 0.10, 0.05, 0.02, 0.0],  # reverse: head is the straggler
        [0.20, 0.0, 0.0, 0.0, 0.0, 0.0],  # head straggler only
        [0.0, 0.0, 0.20, 0.0, 0.0, 0.0],  # middle straggler
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # all equal
    ],
    ids=["reverse", "head", "middle", "equal"],
)
def test_ordered_preserves_order_under_reorder(monkeypatch, delays):
    lf = generate_panel()
    specs = discrete_partitions("id", [[i] for i in range(N_IDS)])
    _patch_delays(monkeypatch, delays)

    ordered = lf.piot.execute_on_ray(specs, preserve_partition_order=True, remote_options={"num_cpus": 0}).collect()
    ids = ordered["id"].to_list()
    assert ids == sorted(ids)
    assert_frame_equal(ordered.sort("id", "date"), lf.sort("id", "date").collect())


@pytest.mark.parametrize("max_concurrency", [1, 2, 100])
def test_ordered_correct_for_window_sizes(monkeypatch, max_concurrency):
    lf = generate_panel()
    specs = discrete_partitions("id", [[i] for i in range(N_IDS)])
    # Reverse completion order stresses the window at every size.
    _patch_delays(monkeypatch, [0.15, 0.12, 0.09, 0.06, 0.03, 0.0])

    ordered = lf.piot.execute_on_ray(specs, preserve_partition_order=True, max_concurrency=max_concurrency, remote_options={"num_cpus": 0}).collect()
    assert ordered["id"].to_list() == sorted(ordered["id"].to_list())
    assert_frame_equal(ordered.sort("id", "date"), lf.sort("id", "date").collect())


def test_ordered_single_partition():
    lf = generate_panel(n_ids=1)
    got = lf.piot.execute_on_ray(discrete_partitions("id", [[0]]), preserve_partition_order=True).sort("date").collect()
    assert_frame_equal(got, lf.sort("date").collect())


def test_ordered_window_never_exceeds_max_concurrency(monkeypatch):
    """The deque must never keep more than ``max_concurrency`` tasks running at once."""
    max_conc = 3

    @ray.remote
    class _Gauge:
        def __init__(self):
            self.active = 0
            self.peak = 0

        def enter(self):
            self.active += 1
            self.peak = max(self.peak, self.active)
            return self.active

        def leave(self):
            self.active -= 1

        def peak_value(self):
            return self.peak

    gauge = _Gauge.remote()

    @ray.remote(num_cpus=0)
    def fake_part(*_a, **_k):
        active = ray.get(gauge.enter.remote())
        assert active <= max_conc
        time.sleep(0.05)
        ray.get(gauge.leave.remote())
        return pl.DataFrame({"id": [0], "date": [datetime.date(2023, 1, 1)], "val": [0]}).to_arrow()

    _patch_options(monkeypatch, lambda _kw, _remote: fake_part.remote)

    specs = discrete_partitions("id", [[i] for i in range(12)])
    got = (
        generate_panel(n_ids=12)
        .piot.execute_on_ray(specs, preserve_partition_order=True, max_concurrency=max_conc, remote_options={"num_cpus": 0})
        .collect()
    )
    assert got.height == 12  # every partition was consumed
    assert ray.get(gauge.peak_value.remote()) == max_conc


@_ray_cancel_flaky_on_windows
def test_ordered_n_rows_stops_submitting(monkeypatch):
    """With n_rows satisfied by the first partition, only the initial window is submitted."""
    lf = generate_panel(n_ids=8)  # each id block has 10 rows
    specs = discrete_partitions("id", [[i] for i in range(8)])
    calls = _count_tasks(monkeypatch)

    got = lf.piot.execute_on_ray(specs, preserve_partition_order=True, max_concurrency=2).head(1).collect()
    assert got.height == 1
    assert got["id"].to_list() == [0]
    assert calls["n"] == 2, f"expected only the initial window submitted, got {calls['n']}"


@_ray_cancel_flaky_on_windows
def test_ordered_required_partition_failure_raises(monkeypatch):
    lf = generate_panel()
    specs = discrete_partitions("id", [[i] for i in range(N_IDS)])
    _patch_failures(monkeypatch, {0})  # the head is required -> must surface
    with pytest.raises((RuntimeError, pl.exceptions.ComputeError), match="partition 0"):
        lf.piot.execute_on_ray(specs, preserve_partition_order=True, remote_options={"num_cpus": 0}).collect()


@_ray_cancel_flaky_on_windows
def test_ordered_failure_beyond_satisfied_n_rows_is_ignored(monkeypatch):
    """A later partition that fails is never observed once an ordered prefix satisfies n_rows."""
    lf = generate_panel(n_ids=6)
    specs = discrete_partitions("id", [[i] for i in range(6)])
    # All submitted at once (max_concurrency defaults high); the last partition fails on the
    # worker, but we stop after partition 0 and drop the rest without ever calling ray.get.
    _patch_failures(monkeypatch, {5})
    got = lf.piot.execute_on_ray(specs, preserve_partition_order=True, remote_options={"num_cpus": 0}).head(1).collect()
    assert got.height == 1
    assert got["id"].to_list() == [0]


@_ray_cancel_flaky_on_windows
def test_ordered_exact_n_rows_does_not_fetch_next(monkeypatch):
    """When the first partition supplies *exactly* n_rows, the next partition must not be
    fetched -- a failure in it would otherwise surface incorrectly."""
    lf = generate_panel(n_ids=2)  # each id block has exactly 10 rows
    specs = discrete_partitions("id", [[0], [1]])
    _patch_failures(monkeypatch, {1})  # partition 1 fails if we ever fetch it
    got = lf.piot.execute_on_ray(specs, preserve_partition_order=True, remote_options={"num_cpus": 0}).head(10).collect()
    assert got.height == 10
    assert got["id"].to_list() == [0] * 10


@_ray_cancel_flaky_on_windows
def test_completion_order_failure_raises(monkeypatch):
    lf = generate_panel()
    specs = discrete_partitions("id", [[i] for i in range(N_IDS)])
    _patch_failures(monkeypatch, {3})
    with pytest.raises((RuntimeError, pl.exceptions.ComputeError), match="partition 3"):
        lf.piot.execute_on_ray(specs, preserve_partition_order=False, remote_options={"num_cpus": 0}).collect()
