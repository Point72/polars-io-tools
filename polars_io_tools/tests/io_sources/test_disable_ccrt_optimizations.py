from datetime import date

import polars as pl

import polars_io_tools as cpl


def test_filtered_join():
    df1 = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).lazy()
    df2 = pl.DataFrame({"a": [1, 2, 3], "c": [7, 8, 9]}).lazy()
    with cpl.disable_optimizations():
        explain = df1.piot.filtered_join(df2, "a", maintain_order="left").filter(pl.col("b") == 5).explain()

    assert explain == df1.join(df2, "a", maintain_order="left").filter(pl.col("b") == 5).explain()


def test_join_asof():
    df1 = pl.DataFrame({"a": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)], "b": [4, 5, 6]}).lazy()
    df2 = pl.DataFrame({"a": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)], "c": [7, 8, 9]}).lazy()
    with cpl.disable_optimizations():
        explain = df1.piot.filtered_join_asof(df2, on="a").filter(pl.col("a") == 2).explain()

    assert explain == df1.join_asof(df2, on="a").filter(pl.col("a") == 2).explain()


def test_ts_with_columns():
    df1 = pl.DataFrame({"a": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)], "b": [4, 5, 6]}).lazy()
    with cpl.disable_optimizations():
        explain = df1.piot.ts_with_columns([pl.lit("x").alias("x")], index_col="a").explain()

    assert explain == df1.with_columns([pl.lit("x").alias("x")]).explain()


def test_ts_with_columns_callable():
    df1 = pl.DataFrame({"a": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)], "b": [4, 5, 6]}).lazy()

    def expr_fn(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(pl.lit("y").alias("y"))

    with cpl.disable_optimizations():
        explain = df1.piot.ts_with_columns(expr_fn, index_col="a").explain()

    assert explain == df1.pipe(expr_fn).explain()


def test_cache():
    df1 = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).lazy()
    with cpl.disable_optimizations():
        explain = df1.piot.cache({}).filter(pl.col("b") == 5).explain()

    assert explain == df1.filter(pl.col("b") == 5).explain()
