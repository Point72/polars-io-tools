"""LazyFrame.piot.debug: a pass-through that prints or logs the arguments Polars pushes to it."""

import logging

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_io_tools.io_sources  # noqa: F401
from polars_io_tools._compat import POLARS_HAS_COLLECT_BATCHES
from polars_io_tools.tests.helpers.assertions import assert_matches_plain_polars, assert_translated_text
from polars_io_tools.tests.helpers.pickling import pickle_round_trip

TWO_COLUMN_FRAME = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
NOTHING_PUSHED_MESSAGE = "debug called with `with_columns=None`, `predicate=None`, `n_rows=None`"


def _assert_debug_output(caplog, capsys, op, log_level, *fragments):
    """``op`` through debug gives what it gives without debug, and the printed or logged output holds ``fragments``."""
    caplog.set_level(logging.INFO)
    kw = {} if log_level is None else {"log_level": log_level}
    assert_matches_plain_polars(op(TWO_COLUMN_FRAME.lazy().piot.debug(**kw)), op(TWO_COLUMN_FRAME.lazy()), None, check_row_order=True)
    assert_translated_text(capsys.readouterr().out if log_level is None else caplog.text, fragments)


def test_debug_print(caplog, capsys):
    _assert_debug_output(caplog, capsys, lambda lf: lf, None, NOTHING_PUSHED_MESSAGE)


def test_debug_info(caplog, capsys):
    _assert_debug_output(caplog, capsys, lambda lf: lf, logging.INFO, NOTHING_PUSHED_MESSAGE)


def test_debug_head(caplog, capsys):
    _assert_debug_output(
        caplog,
        capsys,
        lambda lf: lf.head(5).filter(pl.col("a") > 1).select(["a", "b"]),
        logging.INFO,
        "debug called with",
        "`with_columns=None`",
        "`n_rows=5`",
    )


def test_debug_filter(caplog, capsys):
    _assert_debug_output(
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


@pytest.mark.skipif(not POLARS_HAS_COLLECT_BATCHES, reason="collect_batches requires Polars >= 1.34.0")
def test_collect_batches_lazy_debug_yields_equal():
    base = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
    lf = base.piot.debug()
    chunks = list(lf.collect_batches(chunk_size=2))
    combined = pl.concat(chunks, how="vertical").sort("a")
    expected = lf.collect().sort("a")

    assert_frame_equal(combined, expected)


class TestpiotDebugPickle:
    """Tests for debug pickle support."""

    def test_debug_pickle_basic(self):
        """debug LazyFrames can be pickled and unpickled."""
        import logging

        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        source_lf = df.lazy()

        # log_level should be None or an integer (e.g., logging.DEBUG)
        lf = source_lf.piot.debug(log_level=logging.DEBUG)
        lf_unpickled = pickle_round_trip(lf)

        result = lf_unpickled.collect()
        assert_frame_equal(result, df)
