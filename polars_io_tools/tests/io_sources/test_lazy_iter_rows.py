import datetime

import polars as pl
import pytest

from polars_io_tools._compat import POLARS_HAS_COLLECT_BATCHES


class TestIterRowsBasic:
    def test_iter_rows_returns_all_rows(self):
        """Test that iter_rows returns all rows from LazyFrame."""
        n_rows = 100
        lf = pl.LazyFrame({"a": range(n_rows), "b": range(100, 100 + n_rows)})

        rows = list(lf.piot.iter_rows(named=False))

        assert len(rows) == n_rows
        assert rows[0] == (0, 100)
        assert rows[-1] == (99, 199)

    def test_iter_rows_named_true_returns_dicts(self):
        """Test that named=True returns dictionaries."""
        lf = pl.LazyFrame({"a": [1, 2, 3], "b": [10, 20, 30]})

        rows = list(lf.piot.iter_rows(named=True))

        assert len(rows) == 3
        assert rows[0] == {"a": 1, "b": 10}
        assert rows[1] == {"a": 2, "b": 20}
        assert rows[2] == {"a": 3, "b": 30}
        assert all(isinstance(row, dict) for row in rows)

    def test_iter_rows_named_false_returns_tuples(self):
        """Test that named=False (default) returns tuples."""
        lf = pl.LazyFrame({"a": [1, 2, 3], "b": [10, 20, 30]})

        rows = list(lf.piot.iter_rows(named=False))

        assert len(rows) == 3
        assert rows[0] == (1, 10)
        assert rows[1] == (2, 20)
        assert rows[2] == (3, 30)
        assert all(isinstance(row, tuple) for row in rows)

    def test_iter_rows_default_named_is_false(self):
        """Test that default named parameter is False (returns tuples)."""
        lf = pl.LazyFrame({"a": [1, 2], "b": [10, 20]})

        rows = list(lf.piot.iter_rows())

        assert all(isinstance(row, tuple) for row in rows)
        assert rows[0] == (1, 10)


class TestIterRowsBufferSize:
    def test_iter_rows_with_small_buffer(self):
        """Test iteration with buffer_size smaller than total rows."""
        n_rows = 10
        lf = pl.LazyFrame({"id": range(n_rows)})

        rows = list(lf.piot.iter_rows(buffer_size=3, named=True))

        assert len(rows) == n_rows
        assert [row["id"] for row in rows] == list(range(n_rows))

    def test_iter_rows_buffer_larger_than_total(self):
        """Test that buffer_size larger than total rows works correctly."""
        n_rows = 10
        lf = pl.LazyFrame({"id": range(n_rows), "value": range(100, 100 + n_rows)})

        rows = list(lf.piot.iter_rows(buffer_size=1000, named=True))

        assert len(rows) == n_rows
        assert rows[0] == {"id": 0, "value": 100}
        assert rows[-1] == {"id": 9, "value": 109}

    def test_iter_rows_buffer_equals_total(self):
        """Test that buffer_size equal to total rows works correctly."""
        n_rows = 50
        lf = pl.LazyFrame({"id": range(n_rows)})

        rows = list(lf.piot.iter_rows(buffer_size=n_rows, named=True))

        assert len(rows) == n_rows
        assert [row["id"] for row in rows] == list(range(n_rows))

    def test_iter_rows_not_evenly_divisible(self):
        """Test with n_rows not evenly divisible by buffer_size."""
        # 2125 rows with buffer_size=1000 creates batches: [1000, 1000, 125]
        n_rows = 2125
        lf = pl.LazyFrame({"id": range(n_rows)})

        rows = list(lf.piot.iter_rows(buffer_size=1000, named=True))

        assert len(rows) == n_rows
        assert rows[0]["id"] == 0
        assert rows[999]["id"] == 999  # Last of first batch
        assert rows[1000]["id"] == 1000  # First of second batch
        assert rows[1999]["id"] == 1999  # Last of second batch
        assert rows[2000]["id"] == 2000  # First of third (partial) batch
        assert rows[-1]["id"] == 2124  # Last row

    def test_iter_rows_various_buffer_sizes(self):
        """Test that different buffer sizes all return the same rows."""
        n_rows = 1000
        lf = pl.LazyFrame({"id": range(n_rows), "value": range(n_rows, 2 * n_rows)})

        buffer_sizes = [1, 10, 100, 333, 500, 999, 1000, 2000]
        results = []

        for buffer_size in buffer_sizes:
            rows = list(lf.piot.iter_rows(buffer_size=buffer_size, named=True))
            results.append(rows)

        for i in range(1, len(results)):
            assert results[i] == results[0], f"buffer_size={buffer_sizes[i]} produced different results"


class TestIterRowsEdgeCases:
    def test_iter_rows_empty_lazyframe(self):
        """Test iteration over empty LazyFrame."""
        lf = pl.LazyFrame({"a": [], "b": []}, schema={"a": pl.Int64, "b": pl.Int64})

        rows = list(lf.piot.iter_rows(named=True))

        assert len(rows) == 0

    def test_iter_rows_single_row(self):
        """Test iteration over single row LazyFrame."""
        lf = pl.LazyFrame({"a": [42], "b": [100]})

        rows = list(lf.piot.iter_rows(named=True))

        assert len(rows) == 1
        assert rows[0] == {"a": 42, "b": 100}

    def test_iter_rows_single_column(self):
        """Test iteration over LazyFrame with single column."""
        lf = pl.LazyFrame({"value": [1, 2, 3, 4, 5]})

        rows = list(lf.piot.iter_rows(named=True))

        assert len(rows) == 5
        assert rows[0] == {"value": 1}
        assert rows[-1] == {"value": 5}

    def test_iter_rows_many_columns(self):
        """Test iteration over LazyFrame with many columns."""
        n_cols = 20
        n_rows = 100
        data = {f"col_{i}": range(i * 100, i * 100 + n_rows) for i in range(n_cols)}
        lf = pl.LazyFrame(data)

        rows = list(lf.piot.iter_rows(buffer_size=25, named=True))

        assert len(rows) == n_rows
        assert len(rows[0]) == n_cols
        assert rows[0]["col_0"] == 0
        assert rows[0]["col_19"] == 1900


class TestIterRowsNoMissingRows:
    def test_iter_rows_no_missing_rows(self):
        """Verify no rows are missing across batch boundaries."""
        n_rows = 2500
        lf = pl.LazyFrame({"id": range(n_rows)})

        collected_ids = [row["id"] for row in lf.piot.iter_rows(buffer_size=512, named=True)]

        assert len(collected_ids) == n_rows
        assert sorted(collected_ids) == list(range(n_rows))

    def test_iter_rows_no_duplicates(self):
        """Verify no rows are duplicated."""
        n_rows = 1000
        lf = pl.LazyFrame({"id": range(n_rows)})

        collected_ids = [row["id"] for row in lf.piot.iter_rows(buffer_size=333, named=True)]
        assert len(collected_ids) == len(set(collected_ids))

    def test_iter_rows_sequential_ids(self):
        """Verify IDs are sequential (no gaps)."""
        n_rows = 5000
        lf = pl.LazyFrame({"id": range(n_rows)})

        collected_ids = [row["id"] for row in lf.piot.iter_rows(buffer_size=777, named=True)]
        assert collected_ids == list(range(n_rows))

    def test_iter_rows_batch_boundaries(self):
        """Test rows at batch boundaries are correct."""
        n_rows = 1005
        buffer_size = 100
        lf = pl.LazyFrame({"id": range(n_rows)})

        rows = list(lf.piot.iter_rows(buffer_size=buffer_size, named=True))
        assert rows[0]["id"] == 0  # First
        assert rows[99]["id"] == 99  # End of batch 1
        assert rows[100]["id"] == 100  # Start of batch 2
        assert rows[999]["id"] == 999  # End of batch 10
        assert rows[1000]["id"] == 1000  # Start of batch 11 (partial)
        assert rows[1004]["id"] == 1004  # Last


class TestIterRowsMaintainOrder:
    def test_iter_rows_maintain_order_true(self):
        """Test that maintain_order=True preserves row order."""
        n_rows = 1000
        lf = pl.LazyFrame({"id": range(n_rows), "value": range(n_rows, 2 * n_rows)})

        rows = list(lf.piot.iter_rows(buffer_size=250, maintain_order=True, named=True))

        ids = [row["id"] for row in rows]
        assert ids == list(range(n_rows)), "Order should be preserved"

    @pytest.mark.skipif(not POLARS_HAS_COLLECT_BATCHES, reason="maintain_order requires Polars >= 1.34.0")
    def test_iter_rows_maintain_order_false(self):
        """Test that maintain_order=False still returns all rows (order may vary)."""
        n_rows = 500
        lf = pl.LazyFrame({"id": range(n_rows)})

        rows = list(lf.piot.iter_rows(buffer_size=100, maintain_order=False, named=True))

        ids = sorted([row["id"] for row in rows])
        assert ids == list(range(n_rows))


class TestIterRowsWithOperations:
    def test_iter_rows_with_filter(self):
        """Test iter_rows on filtered LazyFrame."""
        lf = pl.LazyFrame({"id": range(100), "value": range(100)})
        filtered_lf = lf.filter(pl.col("value") % 2 == 0)

        rows = list(filtered_lf.piot.iter_rows(buffer_size=20, named=True))

        assert len(rows) == 50
        assert all(row["value"] % 2 == 0 for row in rows)
        assert [row["id"] for row in rows] == list(range(0, 100, 2))

    def test_iter_rows_with_select(self):
        """Test iter_rows with column selection."""
        lf = pl.LazyFrame({"a": range(50), "b": range(50, 100), "c": range(100, 150)})
        selected_lf = lf.select(["a", "c"])

        rows = list(selected_lf.piot.iter_rows(buffer_size=15, named=True))

        assert len(rows) == 50
        assert all("b" not in row for row in rows)
        assert rows[0] == {"a": 0, "c": 100}

    def test_iter_rows_with_sort(self):
        """Test iter_rows with sorted LazyFrame."""
        lf = pl.LazyFrame({"id": [5, 2, 8, 1, 9, 3]})
        sorted_lf = lf.sort("id")

        rows = list(sorted_lf.piot.iter_rows(buffer_size=2, named=True))

        ids = [row["id"] for row in rows]
        assert ids == [1, 2, 3, 5, 8, 9]

    def test_iter_rows_with_with_columns(self):
        """Test iter_rows with computed columns."""
        lf = pl.LazyFrame({"a": range(20)})
        computed_lf = lf.with_columns((pl.col("a") * 2).alias("doubled"))

        rows = list(computed_lf.piot.iter_rows(buffer_size=7, named=True))

        assert len(rows) == 20
        assert rows[0] == {"a": 0, "doubled": 0}
        assert rows[10] == {"a": 10, "doubled": 20}


class TestIterRowsDataTypes:
    def test_iter_rows_with_dates(self):
        """Test iter_rows with date columns."""
        dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2), datetime.date(2024, 1, 3)]
        lf = pl.LazyFrame({"date": dates, "value": [1, 2, 3]})

        rows = list(lf.piot.iter_rows(named=True))

        assert len(rows) == 3
        assert rows[0]["date"] == datetime.date(2024, 1, 1)
        assert rows[-1]["date"] == datetime.date(2024, 1, 3)

    def test_iter_rows_with_strings(self):
        """Test iter_rows with string columns."""
        lf = pl.LazyFrame({"name": ["Alice", "Bob", "Charlie"], "age": [30, 25, 35]})

        rows = list(lf.piot.iter_rows(named=True))

        assert len(rows) == 3
        assert rows[0] == {"name": "Alice", "age": 30}
        assert rows[1] == {"name": "Bob", "age": 25}

    def test_iter_rows_with_nulls(self):
        """Test iter_rows with null values."""
        lf = pl.LazyFrame({"a": [1, None, 3], "b": [None, 2, 3]})

        rows = list(lf.piot.iter_rows(named=True))

        assert len(rows) == 3
        assert rows[0] == {"a": 1, "b": None}
        assert rows[1] == {"a": None, "b": 2}
        assert rows[2] == {"a": 3, "b": 3}

    def test_iter_rows_with_mixed_types(self):
        """Test iter_rows with mixed data types."""
        lf = pl.LazyFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
                "date_col": [datetime.date(2024, 1, i) for i in [1, 2, 3]],
            }
        )

        rows = list(lf.piot.iter_rows(buffer_size=2, named=True))

        assert len(rows) == 3
        assert rows[0]["int_col"] == 1
        assert rows[0]["float_col"] == 1.1
        assert rows[0]["str_col"] == "a"
        assert rows[0]["bool_col"] is True
        assert rows[0]["date_col"] == datetime.date(2024, 1, 1)


class TestIterRowsPerformance:
    def test_iter_rows_matches_collect(self):
        """Verify iter_rows returns same data as collect()."""
        n_rows = 1000
        lf = pl.LazyFrame({"a": range(n_rows), "b": range(1000, 1000 + n_rows)})

        rows_iter = list(lf.piot.iter_rows(buffer_size=250, named=True))

        df = lf.collect()
        rows_collect = df.to_dicts()

        assert rows_iter == rows_collect

    def test_iter_rows_large_dataset(self):
        """Test iter_rows on larger dataset."""
        n_rows = 50_000
        lf = pl.LazyFrame({"id": range(n_rows), "value": range(n_rows, 2 * n_rows)})

        count = 0
        first_id = None
        last_id = None

        for row in lf.piot.iter_rows(buffer_size=5000, named=True):
            if first_id is None:
                first_id = row["id"]
            last_id = row["id"]
            count += 1

        assert count == n_rows
        assert first_id == 0
        assert last_id == n_rows - 1
