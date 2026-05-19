import datetime  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402

import polars as pl  # noqa: E402
import pytest  # noqa: E402
from polars.testing import assert_frame_equal  # noqa: E402

ray = pytest.importorskip("ray", exc_type=ImportError)

import polars_io_tools as cpl  # noqa: E402,F401
import polars_io_tools.io_sources.lazy_ray  # noqa: E402  # explicit import needed if using pytest-xdist


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
