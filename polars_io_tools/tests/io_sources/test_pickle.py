"""
Tests for cloudpickle support of LazyFrame io sources.

These tests ensure that LazyFrames created by polars-io-tools io sources can be
serialized with cloudpickle (required for distributed computing with Ray, etc.).
"""

from datetime import date, datetime, timedelta

import cloudpickle
import polars as pl
from polars.testing import assert_frame_equal

import polars_io_tools as cpl


def _pickle_roundtrip(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Helper to pickle and unpickle a LazyFrame."""
    pickled = cloudpickle.dumps(lf)
    return cloudpickle.loads(pickled)


class TestScanDeltaPickle:
    """Tests for cpl.scan_delta pickle support."""

    def test_scan_delta_pickle_basic(self, tmp_path):
        """scan_delta LazyFrames can be pickled and unpickled."""
        path = str(tmp_path / "delta_table")

        # Create a simple delta table
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        df.write_delta(path)

        # Create LazyFrame and pickle roundtrip
        lf = cpl.scan_delta(path)
        lf_unpickled = _pickle_roundtrip(lf)

        # Verify results match
        expected = df
        result = lf_unpickled.collect()
        assert_frame_equal(result, expected)

    def test_scan_delta_pickle_with_mapping(self, tmp_path):
        """scan_delta with type mappings can be pickled."""
        path = str(tmp_path / "delta_table")

        # Create delta table with Datetime[ns] (requires mapping)
        df = pl.DataFrame(
            {
                "ts": pl.Series([datetime(2025, 1, 1)]).cast(pl.Datetime("ns")),
                "value": [42],
            }
        )
        cpl.sink_delta(df.lazy(), path, mode="overwrite")

        lf = cpl.scan_delta(path)
        lf_unpickled = _pickle_roundtrip(lf)

        result = lf_unpickled.collect()
        assert result.schema["ts"] == pl.Datetime("ns")
        assert result["ts"].to_list() == [datetime(2025, 1, 1)]

    def test_scan_delta_pickle_with_filter(self, tmp_path):
        """scan_delta with filters can be pickled."""
        path = str(tmp_path / "delta_table")

        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["a", "b", "c", "d", "e"]})
        df.write_delta(path)

        lf = cpl.scan_delta(path).filter(pl.col("a") > 2)
        lf_unpickled = _pickle_roundtrip(lf)

        expected = df.filter(pl.col("a") > 2)
        result = lf_unpickled.collect()
        assert_frame_equal(result, expected)

    def test_scan_delta_pickle_with_select(self, tmp_path):
        """scan_delta with column selection can be pickled."""
        path = str(tmp_path / "delta_table")

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [10, 20, 30]})
        df.write_delta(path)

        lf = cpl.scan_delta(path).select(["a", "b"])
        lf_unpickled = _pickle_roundtrip(lf)

        expected = df.select(["a", "b"])
        result = lf_unpickled.collect()
        assert_frame_equal(result, expected)


class TestCacheParquetPickle:
    """Tests for cache_parquet pickle support."""

    def test_cache_parquet_pickle_basic(self, tmp_path):
        """cache_parquet LazyFrames can be pickled and unpickled."""
        cache_path = str(tmp_path / "cache")

        df = pl.DataFrame(
            {
                "date": [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)],
                "value": [10, 20, 30],
            }
        )
        source_lf = df.lazy()

        # First collect to populate the cache
        lf = source_lf.piot.cache_parquet(
            cache_path=cache_path,
            date_column="date",
        )
        lf.collect()

        # Now create a new LazyFrame from the cache and test pickling
        lf2 = source_lf.piot.cache_parquet(
            cache_path=cache_path,
            date_column="date",
        )
        lf_unpickled = _pickle_roundtrip(lf2)

        result = lf_unpickled.collect()
        assert_frame_equal(result, df)


class TestpiotDebugPickle:
    """Tests for debug pickle support."""

    def test_debug_pickle_basic(self):
        """debug LazyFrames can be pickled and unpickled."""
        import logging

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        source_lf = df.lazy()

        # log_level should be None or an integer (e.g., logging.DEBUG)
        lf = source_lf.piot.debug(log_level=logging.DEBUG)
        lf_unpickled = _pickle_roundtrip(lf)

        result = lf_unpickled.collect()
        assert_frame_equal(result, df)


class TestpiotCachePickle:
    """Tests for cachepickle support."""

    def test_piot_cache_pickle_basic(self, tmp_path):
        """cacheLazyFrames can be pickled and unpickled."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        source_lf = df.lazy()

        lf = source_lf.piot.cache(order_by="a")
        lf_unpickled = _pickle_roundtrip(lf)

        result = lf_unpickled.collect()
        assert_frame_equal(result, df)


class TestFilteredJoinPickle:
    """Tests for filtered_join pickle support."""

    def test_filtered_join_pickle_basic(self):
        """filtered_join LazyFrames can be pickled and unpickled."""
        left_df = pl.DataFrame({"id": [1, 2, 3], "value_left": ["a", "b", "c"]})
        right_df = pl.DataFrame({"id": [2, 3, 4], "value_right": ["x", "y", "z"]})

        lf = left_df.lazy().piot.filtered_join(right_df.lazy(), on="id", how="inner")
        lf_unpickled = _pickle_roundtrip(lf)

        expected = left_df.join(right_df, on="id", how="inner")
        result = lf_unpickled.collect()
        assert_frame_equal(result, expected)


class TestConcatNamedPickle:
    """Tests for concat_named pickle support."""

    def test_concat_named_pickle_basic(self):
        """concat_named LazyFrames can be pickled and unpickled."""
        df1 = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        df2 = pl.DataFrame({"a": [3, 4], "b": ["z", "w"]})

        # concat_named requires identifier_cols which must match the key tuple structure
        lf = cpl.concat_named(
            {("first",): df1.lazy(), ("second",): df2.lazy()},
            identifier_cols=["source"],
        )
        lf_unpickled = _pickle_roundtrip(lf)

        # Verify results match exactly (sort for deterministic order)
        expected = lf.sort("a").collect()
        result = lf_unpickled.sort("a").collect()
        assert_frame_equal(result, expected)


class TestMultiSourcePickle:
    """Tests for multi_source pickle support."""

    def test_multi_source_pickle_basic(self):
        """multi_source LazyFrames can be pickled and unpickled."""
        left_df = pl.DataFrame(
            {
                "date": [date(2025, 1, 1), date(2025, 1, 2)],
                "id": [1, 2],
                "value_left": [10, 20],
            }
        )
        right_df = pl.DataFrame(
            {
                "date": [date(2025, 1, 1), date(2025, 1, 2)],
                "id": [1, 2],
                "value_right": [100, 200],
            }
        )

        lf = cpl.multi_source(
            sources={
                "left": (left_df.lazy(), {"date": cpl.FilterSpec(), "id": cpl.FilterSpec()}),
                "right": (right_df.lazy(), {"date": cpl.FilterSpec(), "id": cpl.FilterSpec()}),
            },
            combine=lambda s: s["left"].join(s["right"], on=["date", "id"]),
        )
        lf_unpickled = _pickle_roundtrip(lf)

        expected = left_df.join(right_df, on=["date", "id"])
        result = lf_unpickled.collect()
        assert_frame_equal(result, expected)


class TestTsWithColumnsPickle:
    """Tests for ts_with_columns pickle support."""

    def test_ts_with_columns_pickle_basic(self):
        """ts_with_columns LazyFrames can be pickled and unpickled."""
        df = pl.DataFrame(
            {
                "date": [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)],
                "value": [10, 20, 30],
            }
        )

        # ts_with_columns uses index_col parameter (not ts_col)
        lf = df.lazy().piot.ts_with_columns(
            [pl.col("value").shift(1).alias("prev_value")],
            index_col="date",
            lookback=timedelta(days=1),
        )
        lf_unpickled = _pickle_roundtrip(lf)

        # Verify results match exactly
        expected = lf.collect()
        result = lf_unpickled.collect()
        assert_frame_equal(result, expected)


class TestFilterNoPushdownPickle:
    """Tests for filter_no_pushdown pickle support."""

    def test_filter_no_pushdown_pickle_basic(self):
        """filter_no_pushdown LazyFrames can be pickled and unpickled."""
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["a", "b", "c", "d", "e"]})

        lf = df.lazy().piot.filter_no_pushdown(pl.col("a") > 2)
        lf_unpickled = _pickle_roundtrip(lf)

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
        lf_unpickled = _pickle_roundtrip(lf)

        # Verify results match exactly
        expected = lf.collect()
        result = lf_unpickled.collect()
        assert_frame_equal(result, expected)


class TestScanNarwhalsPickle:
    """Tests for scan_narwhals / from_narwhals pickle support."""

    def test_scan_narwhals_lazy_pickle_basic(self):
        """scan_narwhals LazyFrames (lazy backend) can be pickled and unpickled."""
        import narwhals as nw

        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        nw_lf = nw.from_native(df.lazy())

        lf = cpl.from_narwhals(nw_lf)
        assert isinstance(lf, pl.LazyFrame)
        lf_unpickled = _pickle_roundtrip(lf)

        expected = lf.collect()
        result = lf_unpickled.collect()
        assert_frame_equal(result, expected)

    def test_scan_narwhals_pickle_with_filter_and_select(self):
        """scan_narwhals LazyFrames with downstream ops can be pickled and unpickled."""
        import narwhals as nw

        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50], "c": ["x", "y", "x", "y", "x"]})
        nw_lf = nw.from_native(df.lazy())

        lf = cpl.from_narwhals(nw_lf).filter(pl.col("a") > 2).select(["a", "b"])
        lf_unpickled = _pickle_roundtrip(lf)

        expected = df.filter(pl.col("a") > 2).select(["a", "b"])
        result = lf_unpickled.collect()
        assert_frame_equal(result, expected)


# The following io sources have pickle tests in their respective test files
# because they require specific mocking infrastructure or external dependencies:
#
# - scan_db: test_lazy_sql_reader.py::TestScanDbPickle
# - scan_datadog: test_lazy_datadog_reader.py::TestScanDatadogPickle
# - from_narwhals: test_lazy_narwhals_reader.py::TestFromNarwhalsPickle
# - execute_on_ray: test_lazy_ray.py::TestExecuteOnRayPickle
