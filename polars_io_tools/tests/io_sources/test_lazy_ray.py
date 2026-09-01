import datetime
import random
import sys
import time

import polars as pl
import pytest
from polars.testing import assert_frame_equal

ray = pytest.importorskip("ray", exc_type=ImportError)

import polars_io_tools as cpl
import polars_io_tools.io_sources.lazy_ray  # explicit import needed if using pytest-xdist

# Ray's task cancellation is unstable on Windows (its Windows support is beta): cancelling an
# in-flight task while the Polars source generator is being closed early (n_rows satisfied, or a
# failing partition) can segfault the interpreter. The behaviour under test is platform-agnostic
# and fully covered on Linux/macOS, so these early-termination cases are skipped on Windows.
_ray_cancel_flaky_on_windows = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Ray task cancellation on early source termination can crash the interpreter on Windows (Ray Windows support is beta).",
)


@pytest.fixture(scope="session", autouse=True)
def shared_ray_cluster():
    """
    Start a tiny local-mode Ray cluster once for all
    tests and shut it down at the end of the session.
    """
    if not ray.is_initialized():
        ray.init(num_cpus=2)
    yield
    ray.shutdown()


def generate_sample_lazyframe(start: datetime.datetime, end: datetime.datetime) -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "date": (dates := pl.datetime_range(start, end, interval="1d", eager=True)),
            "quantity": range(len(dates)),
            "price": [(100 + i) for i in range(len(dates))],
        }
    )


def test_basic():
    """
    Test that the explicit range functionality works.
    """
    lf = generate_sample_lazyframe((s := datetime.datetime(2023, 1, 1)), (e := datetime.datetime(2023, 12, 31)))
    result = (
        lf.filter(pl.col("quantity") > 100)
        .with_columns(pl.col("price") * 2)
        .piot.execute_on_ray(date_column="date", time_unit="monthly")
        .filter(pl.col("date").is_between(s, e))
        .sort("date")
    )
    assert isinstance(result, pl.LazyFrame)
    assert result.collect().shape == (264, 3)

    # Further filters still work
    assert result.filter(pl.col("price") % 2 != 0).collect().shape == (0, 3)
    assert result.filter(pl.col("price") % 4 == 0).filter(pl.col("quantity") % 7 != 0).collect().shape == (113, 3)


def test_actually_partitioned(monkeypatch):
    """
    Test that the chunks are actually getting sent to the cluster.
    """
    lf = generate_sample_lazyframe((s := datetime.datetime(2023, 1, 1)), (e := datetime.datetime(2023, 12, 31)))

    expected_partitions = 365

    calls = {"n": 0}
    original_options = cpl.io_sources.lazy_ray._execute_partition.options

    def counting_options(**kw):
        stub = original_options(**kw)
        orig_remote = stub.remote

        def counting_remote(*args, **kwargs):
            calls["n"] += 1
            return orig_remote(*args, **kwargs)

        stub.remote = counting_remote  # type: ignore[attr-defined]
        return stub

    monkeypatch.setattr(
        cpl.io_sources.lazy_ray._execute_partition,
        "options",
        counting_options,
    )

    (lf.piot.execute_on_ray(date_column="date", time_unit="daily").filter(pl.col("date").is_between(s, e)).collect())

    assert calls["n"] == expected_partitions, f"expected {expected_partitions} Ray tasks, got {calls['n']}"


def test_results_are_in_chronological_order(monkeypatch):
    """
    Test that the results are in order, even if the
    Ray tasks are forced to finish in a random order.
    """
    lf = generate_sample_lazyframe((s := datetime.datetime(2023, 1, 1)), (e := datetime.datetime(2023, 1, 15)))

    # build a delayed wrapper
    original_remote = cpl.io_sources.lazy_ray._execute_partition.remote

    # Remote helper that injects a random delay *inside* the worker
    @ray.remote
    def _slow_execute_partition(*args, **kwargs):
        time.sleep(random.uniform(0.01, 0.2))  # random delay
        return ray.get(original_remote(*args, **kwargs))

    # Replace `.remote`
    def delayed_remote(*args, **kwargs):
        return _slow_execute_partition.remote(*args, **kwargs)

    monkeypatch.setattr(cpl.io_sources.lazy_ray._execute_partition, "remote", delayed_remote)

    df = (
        lf.piot.execute_on_ray(date_column="date", time_unit="daily")
        .filter(pl.col("date").is_between(s, e))  # push-down predicate
        .collect()
    )

    dates = df["date"].to_list()
    assert dates == sorted(dates), "rows are not in chronological order"
    assert len(dates) == 15, "unexpected number of rows"


def test_remote_options_forwarded(monkeypatch):
    """
    Ensure execute_on_ray passes the given remote_options dict to Ray.
    """
    import polars_io_tools as cpl

    lf = generate_sample_lazyframe((s := datetime.datetime(2023, 1, 1)), (e := datetime.datetime(2023, 12, 31)))

    opts = {"num_cpus": 0.25, "runtime_env": {"env_vars": {"POLARS_MAX_THREADS": "2"}}}
    seen = None

    # intercept .options(**kw) to capture the kwargs
    original_options = cpl.io_sources.lazy_ray._execute_partition.options

    def capture_options(**kwargs):
        nonlocal seen
        seen = kwargs
        return original_options(**kwargs)

    monkeypatch.setattr(
        cpl.io_sources.lazy_ray._execute_partition,
        "options",
        capture_options,
    )

    (lf.piot.execute_on_ray(date_column="date", time_unit="daily", remote_options=opts).filter(pl.col("date").is_between(s, e)).collect())

    assert seen == opts


def test_max_concurrency(monkeypatch):
    """
    Test that `execute_on_ray` never runs more than `max_concurrency` tasks at the same time.
    """

    max_conc = 2
    active = [0]  # mutable counter in closure

    # stub that asserts concurrency is \leq `max_conc``
    @ray.remote
    def fake_part(*_a, **_k):
        active[0] += 1
        assert active[0] <= max_conc
        time.sleep(0.05)  # keep the task alive briefly
        active[0] -= 1
        return pl.DataFrame().to_arrow()

    monkeypatch.setattr(cpl.io_sources.lazy_ray._execute_partition, "remote", fake_part.remote)

    s, e = datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 10)
    lf = pl.LazyFrame({"date": pl.datetime_range(s, e, "1d", eager=True)})

    # This will raise if the inner assert is violated
    lf.piot.execute_on_ray(date_column="date", time_unit="daily", max_concurrency=max_conc, remote_options={"num_cpus": 0}).filter(
        pl.col("date").is_between(s, e)
    ).collect()


def test_no_ray():
    """
    Test that an error is raised when no cluster is available.
    """
    ray.shutdown()
    lf = generate_sample_lazyframe((s := datetime.datetime(2023, 1, 1)), (e := datetime.datetime(2023, 12, 31)))

    with pytest.raises(RuntimeError):
        (
            lf.filter(pl.col("quantity") > 100)
            .with_columns(pl.col("price") * 2)
            .piot.execute_on_ray(date_column="date", time_unit="monthly")
            .filter(pl.col("date").is_between(s, e))
            .sort("date")
            .collect()
        )

    # Re-initialize Ray for subsequent tests
    ray.init(num_cpus=2)


class TestPartitionPruning:
    """
    Integration test: verify that execute_on_ray prunes partitions with
    exclusive temporal bounds, resulting in fewer Ray tasks being submitted.

    Unit tests for the underlying pruning logic (interval extraction,
    intersection) live in test_range_visitor.py::TestPartitionPruningLogic.
    """

    def test_closed_left_produces_fewer_ray_tasks(self, monkeypatch):
        """
        closed='left' should submit one fewer Ray task than closed='both'
        for the same date range (the upper-bound partition is pruned).
        """
        s = datetime.datetime(2024, 1, 1)
        e = datetime.datetime(2024, 1, 10)
        lf = generate_sample_lazyframe(s, e)

        calls = {"n": 0}
        original_options = cpl.io_sources.lazy_ray._execute_partition.options

        def counting_options(**kw):
            stub = original_options(**kw)
            orig_remote = stub.remote

            def counting_remote(*args, **kwargs):
                calls["n"] += 1
                return orig_remote(*args, **kwargs)

            stub.remote = counting_remote
            return stub

        monkeypatch.setattr(
            cpl.io_sources.lazy_ray._execute_partition,
            "options",
            counting_options,
        )

        # closed="both" (default)
        calls["n"] = 0
        lf.piot.execute_on_ray(date_column="date", time_unit="daily").filter(pl.col("date").is_between(s, e)).collect()
        both_count = calls["n"]

        # closed="left" — one fewer partition
        calls["n"] = 0
        lf.piot.execute_on_ray(date_column="date", time_unit="daily").filter(pl.col("date").is_between(s, e, closed="left")).collect()
        left_count = calls["n"]

        assert left_count == both_count - 1, (
            f"closed='left' should submit one fewer Ray task than closed='both': expected {both_count - 1}, got {left_count}"
        )


# These tests verify that execute_on_ray LazyFrames can be serialized with cloudpickle,
# which is required for distributed computing. Tests are located here rather than in
# test_pickle.py because they require Ray cluster infrastructure defined in this file.


class TestExecuteOnRayPickle:
    """Tests for execute_on_ray cloudpickle serialization support."""

    def test_execute_on_ray_pickle_basic(self, shared_ray_cluster):
        """execute_on_ray LazyFrames can be pickled and unpickled."""
        import cloudpickle

        s, e = datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 31)
        lf = generate_sample_lazyframe(s, e)

        # Create ray-distributed LazyFrame
        ray_lf = lf.piot.execute_on_ray(date_column="date", time_unit="daily").filter(pl.col("date").is_between(s, e))

        # Pickle roundtrip
        pickled = cloudpickle.dumps(ray_lf)
        lf_unpickled = cloudpickle.loads(pickled)

        # Verify schema is preserved
        assert ray_lf.collect_schema() == lf_unpickled.collect_schema()

        # Verify results match exactly
        expected = ray_lf.sort("date").collect()
        result = lf_unpickled.sort("date").collect()
        assert_frame_equal(expected, result)

    def test_execute_on_ray_pickle_with_filter(self, shared_ray_cluster):
        """execute_on_ray with additional filters can be pickled."""
        import cloudpickle

        s, e = datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 31)
        lf = generate_sample_lazyframe(s, e)

        # Create ray-distributed LazyFrame with filter
        ray_lf = lf.filter(pl.col("quantity") > 10).piot.execute_on_ray(date_column="date", time_unit="daily").filter(pl.col("date").is_between(s, e))

        pickled = cloudpickle.dumps(ray_lf)
        lf_unpickled = cloudpickle.loads(pickled)

        # Verify results match exactly
        expected = ray_lf.sort("date").collect()
        result = lf_unpickled.sort("date").collect()
        assert_frame_equal(expected, result)


# ---------------------------------------------------------------------------
# Generic partitioning: execute_on_ray(partitions=...) with by_key / explicit specs / builders
# ---------------------------------------------------------------------------

from datetime import date, timedelta

from polars_io_tools.io_sources.lazy_ray import (
    RayPartition,
    cartesian_partitions,
    discrete_partitions,
)
from polars_io_tools.io_sources.partitions import by_key, by_time, by_value
from polars_io_tools.io_sources.pushdown_combine import FilterSpec, pushdown_combine

N_IDS = 6


def generate_panel(n_ids: int = N_IDS, n_days: int = 10) -> pl.LazyFrame:
    """A small (id x date) cross-section with a deterministic value column."""
    dates = pl.date_range(date(2023, 1, 1), date(2023, 1, n_days), interval="1d", eager=True)
    ids = pl.DataFrame({"id": list(range(n_ids))})
    panel = ids.join(pl.DataFrame({"date": dates}), how="cross")
    panel = panel.with_columns(val=(pl.col("id") * 100 + pl.col("date").dt.day()))
    return panel.lazy()


def _count_tasks(monkeypatch):
    calls = {"n": 0}
    original_options = cpl.io_sources.lazy_ray._execute_partition.options

    def counting_options(**kw):
        stub = original_options(**kw)
        orig_remote = stub.remote

        def counting_remote(*args, **kwargs):
            calls["n"] += 1
            return orig_remote(*args, **kwargs)

        stub.remote = counting_remote  # type: ignore[attr-defined]
        return stub

    monkeypatch.setattr(cpl.io_sources.lazy_ray._execute_partition, "options", counting_options)
    return calls


def test_by_column_scalar_keys(monkeypatch):
    lf = generate_panel()
    calls = _count_tasks(monkeypatch)

    result = lf.piot.execute_on_ray(by_key(range(N_IDS), "id")).sort("id", "date").collect()
    expected = lf.sort("id", "date").collect()

    assert_frame_equal(result, expected)
    assert calls["n"] == N_IDS


def test_by_expr_hash_bucket(monkeypatch):
    lf = generate_panel()
    calls = _count_tasks(monkeypatch)
    n_buckets = 3

    result = lf.piot.execute_on_ray(by_key(range(n_buckets), pl.col("id").hash() % n_buckets)).sort("id", "date").collect()
    expected = lf.sort("id", "date").collect()

    assert_frame_equal(result, expected)
    assert calls["n"] == n_buckets


def test_discrete_partitions_is_in():
    lf = generate_panel()
    specs = discrete_partitions("id", [[0, 1], [2, 3], [4, 5]])
    result = lf.piot.execute_on_ray(specs).sort("id", "date").collect()
    expected = lf.sort("id", "date").collect()
    assert_frame_equal(result, expected)


def test_cartesian_date_x_bucket(monkeypatch):
    lf = generate_panel(n_days=10)
    calls = _count_tasks(monkeypatch)
    windows = [(date(2023, 1, 1), date(2023, 1, 6)), (date(2023, 1, 6), date(2023, 1, 11))]
    specs = cartesian_partitions(
        date_windows=windows,
        buckets=[[0, 1, 2], [3, 4, 5]],
        date_column="date",
        bucket="id",
    )
    result = lf.piot.execute_on_ray(specs).sort("id", "date").collect()
    expected = lf.sort("id", "date").collect()
    assert_frame_equal(result, expected)
    assert calls["n"] == 4  # 2 windows x 2 buckets


def test_column_retention_under_projection():
    """The partition key column must be retained for the worker predicate, then dropped."""
    lf = generate_panel()
    # `id` is the partition key but is NOT selected downstream.
    result = lf.piot.execute_on_ray(by_key(range(N_IDS), "id")).select("val").sort("val").collect()
    expected = lf.select("val").sort("val").collect()
    assert result.columns == ["val"]
    assert_frame_equal(result, expected)


def test_empty_partition_result_schema():
    lf = generate_panel()
    # A key that matches nothing.
    result = lf.piot.execute_on_ray(by_key([999], "id")).collect()
    assert result.height == 0
    assert result.schema == lf.collect_schema()


def test_duplicate_key_rejected():
    lf = generate_panel()
    with pytest.raises(ValueError, match="Duplicate partition key"):
        lf.piot.execute_on_ray(by_key([1, 1], "id")).collect()


def test_completion_order_and_preserve_order():
    lf = generate_panel()
    specs = discrete_partitions("id", [[i] for i in range(N_IDS)])

    # completion order (default): correct rows regardless of arrival order
    got = lf.piot.execute_on_ray(specs).sort("id", "date").collect()
    expected = lf.sort("id", "date").collect()
    assert_frame_equal(got, expected)

    # preserve order: id blocks appear in spec order
    ordered = lf.piot.execute_on_ray(specs, preserve_partition_order=True).collect()
    ids = ordered["id"].to_list()
    assert ids == sorted(ids)


def test_per_partition_remote_options_struct_column(monkeypatch):
    lf = generate_panel(n_ids=3)
    seen = []
    original_options = cpl.io_sources.lazy_ray._execute_partition.options

    def capturing_options(**kw):
        seen.append(kw)
        return original_options(**kw)

    monkeypatch.setattr(cpl.io_sources.lazy_ray._execute_partition, "options", capturing_options)

    parts = pl.DataFrame({"id": [0, 1, 2], "ray": [{"num_cpus": 1}, {"num_cpus": 1}, {"num_cpus": 1}]})
    result = lf.piot.execute_on_ray(by_key(parts, "id", partition_remote_options="ray")).sort("id", "date").collect()
    expected = lf.sort("id", "date").collect()
    assert_frame_equal(result, expected)
    assert all(kw.get("num_cpus") == 1 for kw in seen)


def test_per_partition_remote_options_mapping(monkeypatch):
    lf = generate_panel(n_ids=3)
    seen = []
    original_options = cpl.io_sources.lazy_ray._execute_partition.options

    def capturing_options(**kw):
        seen.append(kw)
        return original_options(**kw)

    monkeypatch.setattr(cpl.io_sources.lazy_ray._execute_partition, "options", capturing_options)

    result = lf.piot.execute_on_ray(by_key([0, 1, 2], "id", partition_remote_options={0: {"num_cpus": 1}})).sort("id", "date").collect()
    expected = lf.sort("id", "date").collect()
    assert_frame_equal(result, expected)


def test_validation_errors():
    lf = generate_panel()
    with pytest.raises(ValueError, match="return_as"):
        lf.piot.execute_on_ray(discrete_partitions("id", [[0]]), return_as="bogus")
    with pytest.raises(ValueError, match="max_concurrency"):
        lf.piot.execute_on_ray(discrete_partitions("id", [[0]]), max_concurrency=0)


def test_invariant1_lookback_self_padding():
    """Byte-identical single-frame vs distributed collect through a FilterSpec(lookback).

    Each date-window partition must self-pad its lookback: the partition predicate must
    reach the inner source so it expands the read window, computes the rolling value, and
    trims after combine. If pruning consumed the predicate, boundary rows would be wrong.
    """
    dates = [date(2024, 1, i) for i in range(1, 13)]
    vals = [float(i) for i in range(1, 13)]
    data = pl.LazyFrame({"date": dates, "val": vals})

    def combine_with_rolling(s):
        return s["data"].sort("date").with_columns(pl.col("val").rolling_sum(window_size=3, min_samples=1).alias("rs"))

    lf = pushdown_combine(
        sources={"data": (data, {"date": FilterSpec(lookback=timedelta(days=3))})},
        combine=combine_with_rolling,
    )

    single = lf.sort("date").collect()

    windows = [
        (date(2024, 1, 1), date(2024, 1, 5)),
        (date(2024, 1, 5), date(2024, 1, 9)),
        (date(2024, 1, 9), date(2024, 1, 13)),
    ]
    specs = [RayPartition((pl.col("date") >= lo) & (pl.col("date") < hi), key=i) for i, (lo, hi) in enumerate(windows)]
    distributed = lf.piot.execute_on_ray(specs).sort("date").collect()

    assert_frame_equal(single, distributed)


# ---------------------------------------------------------------------------
# Regression tests for review findings (selectors, null/expr keys, series name, NaN dup)
# ---------------------------------------------------------------------------

import polars.selectors as cs


def test_by_selector_multi_column():
    lf = generate_panel(n_ids=3, n_days=4)
    lf2 = lf.with_columns(grp=(pl.col("id") % 2))
    parts = pl.DataFrame({"id": [0, 1, 2], "grp": [0, 1, 0]})
    result = lf2.piot.execute_on_ray(by_key(parts, cs.by_name("id", "grp"))).sort("id", "date").collect()
    expected = lf2.sort("id", "date").collect()
    assert_frame_equal(result, expected)


def test_selector_expanded_predicate_retains_columns():
    """A predicate whose root_names cannot be resolved must not lose columns under projection."""
    lf = pl.LazyFrame({"a": [1, -1, 1], "b": [1, 1, -1], "v": [10, 20, 30]})
    specs = [
        RayPartition(pl.all_horizontal(pl.col("a", "b") > 0), key="pos"),
        RayPartition(~pl.all_horizontal(pl.col("a", "b") > 0), key="neg"),
    ]
    result = lf.piot.execute_on_ray(specs).select("v").sort("v").collect()
    expected = lf.select("v").sort("v").collect()
    assert result.columns == ["v"]
    assert_frame_equal(result, expected)


def test_by_expr_null_key():
    lf = pl.LazyFrame({"id": [None, 1, 2], "v": [10, 20, 30]}, schema={"id": pl.Int64, "v": pl.Int64})
    result = lf.piot.execute_on_ray(by_key([None, 1, 2], pl.col("id"))).sort("v").collect()
    expected = lf.sort("v").collect()
    assert_frame_equal(result, expected)


def test_by_series_name_mismatch():
    lf = generate_panel(n_ids=3, n_days=4)
    # Series named differently from `by`; should be aligned to the key column.
    result = lf.piot.execute_on_ray(by_key(pl.Series("whatever", [0, 1, 2]), "id")).sort("id", "date").collect()
    expected = lf.sort("id", "date").collect()
    assert_frame_equal(result, expected)


def test_duplicate_nan_keys_rejected():
    lf = pl.LazyFrame({"x": [1.0, float("nan")], "v": [1, 2]})
    nan = float("nan")
    with pytest.raises(ValueError, match="Duplicate partition key"):
        lf.piot.execute_on_ray(by_key([nan, nan], "x")).collect()


# ---------------------------------------------------------------------------
# Predicate-pushdown key pruning (fail-open)
# ---------------------------------------------------------------------------


def test_pruning_equality_key(monkeypatch):
    """A downstream `id == 3` should launch only the matching partition."""
    lf = generate_panel()
    calls = _count_tasks(monkeypatch)
    result = lf.piot.execute_on_ray(by_key(range(N_IDS), "id")).filter(pl.col("id") == 3).sort("date").collect()
    expected = lf.filter(pl.col("id") == 3).sort("date").collect()
    assert_frame_equal(result, expected)
    assert calls["n"] == 1


def test_partitions_primitive_no_pruning_correct_under_filter(monkeypatch):
    """execute_on_ray does not prune caller-supplied explicit specs; results stay correct
    under a downstream filter and every partition runs."""
    lf = generate_panel()
    calls = _count_tasks(monkeypatch)
    specs = discrete_partitions("id", [[0, 1], [2, 3], [4, 5]])
    result = lf.piot.execute_on_ray(specs).filter(pl.col("id").is_in([0, 5])).sort("id", "date").collect()
    expected = lf.filter(pl.col("id").is_in([0, 5])).sort("id", "date").collect()
    assert_frame_equal(result, expected)
    assert calls["n"] == 3  # caller controls the spec set; no pruning


def test_cartesian_no_pruning_correct_under_filter(monkeypatch):
    lf = generate_panel(n_days=10)
    calls = _count_tasks(monkeypatch)
    windows = [(date(2023, 1, 1), date(2023, 1, 6)), (date(2023, 1, 6), date(2023, 1, 11))]
    specs = cartesian_partitions(date_windows=windows, buckets=[[0, 1, 2], [3, 4, 5]], date_column="date", bucket="id")
    result = (
        lf.piot.execute_on_ray(specs).filter(pl.col("date") < date(2023, 1, 6)).filter(pl.col("id").is_in([0, 1, 2])).sort("id", "date").collect()
    )
    expected = lf.filter(pl.col("date") < date(2023, 1, 6)).filter(pl.col("id").is_in([0, 1, 2])).sort("id", "date").collect()
    assert_frame_equal(result, expected)
    assert calls["n"] == 4  # all cells run; caller controls the spec set


def test_pruning_fail_open_unrelated_predicate(monkeypatch):
    """A downstream predicate on a non-key column must NOT prune any partition."""
    lf = generate_panel()
    calls = _count_tasks(monkeypatch)
    result = lf.piot.execute_on_ray(by_key(range(N_IDS), "id")).filter(pl.col("val") > 500).sort("id", "date").collect()
    expected = lf.filter(pl.col("val") > 500).sort("id", "date").collect()
    assert_frame_equal(result, expected)
    assert calls["n"] == N_IDS  # nothing pruned


def test_pruning_never_drops_matching(monkeypatch):
    """Pruning must not drop a partition that can still contribute rows (correctness)."""
    lf = generate_panel()
    result = lf.piot.execute_on_ray(by_key(range(N_IDS), "id")).filter(pl.col("id") >= 2).sort("id", "date").collect()
    expected = lf.filter(pl.col("id") >= 2).sort("id", "date").collect()
    assert_frame_equal(result, expected)


def test_pruning_no_false_prune_nan_in_is_in():
    """NaN member in an is_in must not be dropped into a false prune."""
    lf = pl.DataFrame({"x": [2.0, float("nan")], "v": [1, 2]}).lazy()
    specs = discrete_partitions("x", [[2.0, float("nan")]])
    result = lf.piot.execute_on_ray(specs).filter(pl.col("x").is_in([1.0, float("nan")])).collect()
    expected = lf.filter(pl.col("x").is_in([1.0, float("nan")])).collect()
    assert_frame_equal(result.sort("v"), expected.sort("v"))
    assert result.height == 1  # the NaN row must survive


def test_pruning_no_false_prune_float_coercion():
    """Int/float coercion must not cause a false prune."""
    v = 2**53 + 1
    lf = pl.DataFrame({"id": [v], "w": [7]}, schema={"id": pl.Int64, "w": pl.Int64}).lazy()
    result = lf.piot.execute_on_ray(by_key([v], "id")).filter(pl.col("id") == float(2**53)).collect()
    expected = lf.filter(pl.col("id") == float(2**53)).collect()
    assert_frame_equal(result, expected)
    assert result.height == expected.height


def test_by_expr_cast_no_false_prune():
    """A cast expression key must not false-prune (expression keys are never pruned)."""
    lf = pl.DataFrame({"id": [2]}, schema={"id": pl.Int64}).lazy()
    result = lf.piot.execute_on_ray(by_key([True], pl.col("id").cast(pl.Boolean))).filter(pl.col("id") == 2).collect()
    expected = lf.filter(pl.col("id") == 2).collect()
    assert_frame_equal(result, expected)
    assert result.height == 1


def test_partitions_multicolumn_null_no_false_prune():
    """Arbitrary specs with null-matching predicates must not be pruned."""
    schema = pl.Schema({"c": pl.String, "d": pl.Int64})
    lf = pl.DataFrame({"c": [None], "d": [1]}, schema=schema).lazy()
    specs = [RayPartition(pl.col("c").is_in(["b", None], nulls_equal=True) & (pl.col("d") == 1), key="p")]
    downstream = pl.col("c").is_in(["a", None], nulls_equal=True) & (pl.col("d") == 1)
    result = lf.piot.execute_on_ray(specs).filter(downstream).collect()
    expected = lf.filter(downstream).collect()
    assert_frame_equal(result, expected)
    assert result.height == 1  # the null row survives


def test_pruning_by_nan_key_exact():
    """Exact key evaluation keeps a NaN partition matched by the downstream predicate."""
    lf = pl.DataFrame({"x": [1.0, float("nan")], "v": [1, 2]}).lazy()
    result = lf.piot.execute_on_ray(by_key([1.0, float("nan")], "x")).filter(pl.col("x").is_in([float("nan")])).collect()
    expected = lf.filter(pl.col("x").is_in([float("nan")])).collect()
    assert_frame_equal(result.sort("v"), expected.sort("v"))


def test_pruning_float_key_no_false_prune():
    """Float keys are not observational under == (-0.0 vs +0.0); must fail open."""
    lf = pl.DataFrame({"x": [-0.0], "v": [1]}).lazy()
    predicate = (pl.lit(1.0) / pl.col("x")) < 0
    result = lf.piot.execute_on_ray(by_key([0.0], "x")).filter(predicate).collect()
    expected = lf.filter(predicate).collect()
    assert_frame_equal(result, expected)
    assert result.height == 1


def test_pruning_python_udf_no_false_prune():
    """A Python UDF predicate must not be pruned via single-key evaluation."""

    class EverySecond:
        def __init__(self):
            self.n = 0

        def __call__(self, _):
            self.n += 1
            return self.n % 2 == 0

    lf = pl.DataFrame({"id": [1, 1], "v": [10, 11]}, schema={"id": pl.Int64, "v": pl.Int64}).lazy()
    predicate = pl.col("id").map_elements(EverySecond(), return_dtype=pl.Boolean)
    result = lf.piot.execute_on_ray(by_key([1], "id")).filter(predicate).collect()
    expected = lf.filter(predicate).collect()
    assert_frame_equal(result.sort("v"), expected.sort("v"))


def test_cartesian_none_bucket_matches_nulls():
    """A None bucket value must match null rows via is_null(), not `== None`."""
    lf = pl.DataFrame(
        {
            "date": [date(2026, 1, 1), date(2026, 1, 1)],
            "desk": [None, "rates"],
            "value": [100, 200],
        },
        schema={"date": pl.Date, "desk": pl.String, "value": pl.Int64},
    ).lazy()
    specs = cartesian_partitions(
        date_windows=[(date(2026, 1, 1), date(2026, 1, 2))],
        buckets={"missing-desk": None, "rates": "rates"},
        date_column="date",
        bucket="desk",
    )
    result = lf.piot.execute_on_ray(specs).sort("value").collect()
    expected = lf.sort("value").collect()
    assert_frame_equal(result, expected)
    assert result.height == 2  # both the null-desk and the rates row


def test_discrete_partitions_none_member_matches_nulls():
    """A None member in a discrete group must include null rows."""
    lf = pl.DataFrame({"id": [None, 1, 2], "v": [10, 20, 30]}, schema={"id": pl.Int64, "v": pl.Int64}).lazy()
    specs = discrete_partitions("id", [[1, None], [2]])
    result = lf.piot.execute_on_ray(specs).sort("v").collect()
    expected = lf.sort("v").collect()
    assert_frame_equal(result, expected)
    assert result.height == 3


def test_selector_downstream_predicate_retains_columns():
    """A pushed-down selector-expanded predicate (empty root_names) must see all columns."""
    lf = pl.DataFrame({"a": [1, None], "b": [None, 2]}, schema={"a": pl.Int64, "b": pl.Int64}).lazy()
    specs = discrete_partitions("a", [[1], [2]])  # non-selector partition preds
    selector_predicate = pl.all_horizontal(pl.all().is_not_null())
    # Downstream selector predicate + projection to only "a": must evaluate over full schema.
    result = lf.piot.execute_on_ray(specs).filter(selector_predicate).select("a").collect()
    expected = lf.filter(selector_predicate).select("a").collect()
    assert_frame_equal(result.sort("a"), expected.sort("a"))
    assert result.height == 0  # no row has both a and b non-null


# --- Unified `partitions=` entry point (shared ReadPartition vocabulary) --------------------


def _category_lazyframe(start, end):
    dates = pl.datetime_range(start, end, interval="1d", eager=True)
    return pl.LazyFrame(
        {
            "date": dates,
            "category": [("A", "B", "C")[i % 3] for i in range(len(dates))],
            "val": range(len(dates)),
        }
    )


def test_unified_by_time_matches_calendar_and_single_frame():
    lf = generate_sample_lazyframe((s := datetime.datetime(2023, 1, 1)), (e := datetime.datetime(2023, 6, 30)))
    pipeline = lf.filter(pl.col("quantity") > 10).with_columns(pl.col("price") * 2)
    flt = pl.col("date").is_between(s, e)

    expected = pipeline.filter(flt).sort("date").collect()
    unified = pipeline.piot.execute_on_ray(cpl.by_time("date", "1mo")).filter(flt).sort("date").collect()
    legacy = pipeline.piot.execute_on_ray(date_column="date", time_unit="monthly").filter(flt).sort("date").collect()

    assert_frame_equal(unified, expected)
    assert_frame_equal(unified, legacy)


def test_unified_by_value_derived_from_predicate():
    lf = _category_lazyframe(datetime.datetime(2023, 1, 1), datetime.datetime(2023, 3, 31))
    flt = pl.col("category").is_in(["A", "B"])

    expected = lf.filter(flt).sort("date").collect()
    got = lf.piot.execute_on_ray(cpl.by_value("category")).filter(flt).sort("date").collect()

    assert_frame_equal(got, expected)


def test_unified_explicit_read_partition_list():
    lf = generate_sample_lazyframe(datetime.datetime(2023, 1, 1), datetime.datetime(2023, 3, 31))
    mid = datetime.datetime(2023, 2, 15)
    parts = [
        cpl.ReadPartition(pl.col("date") < mid, key="lo"),
        cpl.ReadPartition(pl.col("date") >= mid, key="hi"),
    ]

    expected = lf.sort("date").collect()
    got = lf.piot.execute_on_ray(parts).sort("date").collect()

    assert_frame_equal(got, expected)


def test_unified_partitions_and_legacy_are_mutually_exclusive():
    lf = generate_sample_lazyframe(datetime.datetime(2023, 1, 1), datetime.datetime(2023, 3, 31))
    with pytest.raises(ValueError, match="either"):
        lf.piot.execute_on_ray(cpl.by_time("date", "1mo"), date_column="date", time_unit="monthly")


def test_unified_by_value_explicit_needs_no_predicate():
    # An explicit value list is fully specified: it must run without any pushed-down predicate.
    lf = _category_lazyframe(datetime.datetime(2023, 1, 1), datetime.datetime(2023, 2, 28))
    expected = lf.filter(pl.col("category").is_in(["A", "B"])).sort("date").collect()
    got = lf.piot.execute_on_ray(by_value("category", ["A", "B"])).sort("date").collect()
    assert_frame_equal(got, expected)


def test_unified_by_time_empty_range_returns_empty():
    # A contradictory date filter yields no partitions -> an empty frame, not an error.
    lf = generate_sample_lazyframe(datetime.datetime(2023, 1, 1), datetime.datetime(2023, 12, 31))
    flt = (pl.col("date") >= datetime.datetime(2023, 6, 1)) & (pl.col("date") < datetime.datetime(2023, 1, 1))
    got = lf.piot.execute_on_ray(by_time("date", "1mo")).filter(flt).collect()
    assert got.height == 0


def test_explicit_generator_partitions_are_reusable():
    # A generator of ReadPartitions must be materialised once, so re-collecting the same
    # LazyFrame does not silently yield an empty frame the second time.
    lf = generate_sample_lazyframe(datetime.datetime(2023, 1, 1), datetime.datetime(2023, 3, 31))
    mid = datetime.datetime(2023, 2, 15)

    def gen():
        yield cpl.ReadPartition(pl.col("date") < mid, key="lo")
        yield cpl.ReadPartition(pl.col("date") >= mid, key="hi")

    ray_lf = lf.piot.execute_on_ray(gen())
    first = ray_lf.sort("date").collect()
    second = ray_lf.sort("date").collect()
    assert first.height == lf.collect().height
    assert_frame_equal(first, second)


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
    original_options = cpl.io_sources.lazy_ray._execute_partition.options
    counter = {"n": 0}

    def counting_options(**kw):
        stub = original_options(**kw)
        orig_remote = stub.remote

        def wrapped_remote(*args, **kwargs):
            k = counter["n"]
            counter["n"] += 1
            return wrap(k, lambda: orig_remote(*args, **kwargs))

        stub.remote = wrapped_remote  # type: ignore[attr-defined]
        return stub

    monkeypatch.setattr(cpl.io_sources.lazy_ray._execute_partition, "options", counting_options)


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

    original_options = cpl.io_sources.lazy_ray._execute_partition.options

    def counting_options(**kw):
        stub = original_options(**kw)
        stub.remote = fake_part.remote  # type: ignore[attr-defined]
        return stub

    monkeypatch.setattr(cpl.io_sources.lazy_ray._execute_partition, "options", counting_options)

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
    calls = {"n": 0}
    original_options = cpl.io_sources.lazy_ray._execute_partition.options

    def counting_options(**kw):
        stub = original_options(**kw)
        orig_remote = stub.remote

        def counting_remote(*args, **kwargs):
            calls["n"] += 1
            return orig_remote(*args, **kwargs)

        stub.remote = counting_remote  # type: ignore[attr-defined]
        return stub

    monkeypatch.setattr(cpl.io_sources.lazy_ray._execute_partition, "options", counting_options)

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
