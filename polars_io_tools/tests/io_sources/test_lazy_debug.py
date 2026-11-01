"""LazyFrame.piot.debug: a pass-through that prints or logs the arguments Polars pushes to it."""

import logging

import polars as pl

import polars_io_tools.io_sources  # noqa: F401
from polars_io_tools.tests.helpers.oracles import assert_matches_native, assert_text

DF = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
NONE_PUSHED = "debug called with `with_columns=None`, `predicate=None`, `n_rows=None`"


def _debug(caplog, capsys, op, log_level, *fragments):
    """``op`` through debug gives what it gives on the frame, and the debug output holds ``fragments``."""
    caplog.set_level(logging.INFO)
    kw = {} if log_level is None else {"log_level": log_level}
    assert_matches_native(op(DF.lazy().piot.debug(**kw)), op(DF.lazy()), None, check_row_order=True)
    assert_text(capsys.readouterr().out if log_level is None else caplog.text, fragments)


def test_debug_print(caplog, capsys):
    _debug(caplog, capsys, lambda lf: lf, None, NONE_PUSHED)


def test_debug_info(caplog, capsys):
    _debug(caplog, capsys, lambda lf: lf, logging.INFO, NONE_PUSHED)


def test_debug_head(caplog, capsys):
    _debug(
        caplog,
        capsys,
        lambda lf: lf.head(5).filter(pl.col("a") > 1).select(["a", "b"]),
        logging.INFO,
        "debug called with",
        "`with_columns=None`",
        "`n_rows=5`",
    )


def test_debug_filter(caplog, capsys):
    _debug(
        caplog,
        capsys,
        lambda lf: lf.filter(pl.col("a") > 1).select(["a", "b"]),
        logging.INFO,
        "debug called with",
        "`with_columns=None`",
        '`predicate=[(col("a")) > (1)]`',
    )


def test_debug_logs_per_use_by_default(caplog):
    """is_pure defaults to False, so a reused debug logs once per use."""
    debug = pl.DataFrame({"a": [1]}).lazy().piot.debug(log_level=logging.INFO)
    caplog.set_level(logging.INFO)
    pl.concat([debug, debug]).collect(engine="streaming")
    assert caplog.text.count("debug called with") == 2
