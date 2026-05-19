import logging

import polars as pl
from polars.testing import assert_frame_equal

import polars_io_tools.io_sources  # noqa: F401


def test_debug_print(capsys):
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).lazy()
    df_debug = df.piot.debug()
    assert_frame_equal(df_debug.collect(), df.collect())
    assert "debug called with `with_columns=None`, `predicate=None`, `n_rows=None`, `batch_size=None` on lazy frame" in capsys.readouterr().out


def test_debug_info(caplog):
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).lazy()
    df_debug = df.piot.debug(log_level=logging.INFO)
    caplog.set_level(logging.INFO)
    assert_frame_equal(df_debug.collect(), df.collect())
    assert "debug called with `with_columns=None`, `predicate=None`, `n_rows=None`, `batch_size=None` on lazy frame" in caplog.text


def test_debug_head(caplog):
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).lazy()
    df_debug = df.piot.debug(log_level=logging.INFO)
    caplog.set_level(logging.INFO)
    df2 = df.head(5).filter(pl.col("a") > 1).select(["a", "b"])
    df_debug2 = df_debug.head(5).filter(pl.col("a") > 1).select(["a", "b"])
    assert_frame_equal(df_debug2.collect(), df2.collect())
    assert "debug called with" in caplog.text
    assert "`with_columns=None`" in caplog.text
    assert "`n_rows=5`" in caplog.text


def test_debug_filter(caplog):
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).lazy()
    df_debug = df.piot.debug(log_level=logging.INFO)
    caplog.set_level(logging.INFO)
    df2 = df.filter(pl.col("a") > 1).select(["a", "b"])
    df_debug2 = df_debug.filter(pl.col("a") > 1).select(["a", "b"])
    assert_frame_equal(df_debug2.collect(), df2.collect())
    assert "debug called with" in caplog.text
    assert "`with_columns=None`" in caplog.text
    assert '`predicate=[(col("a")) > (1)]`' in caplog.text
