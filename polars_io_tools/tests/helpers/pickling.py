"""Pickle round trips, as Ray and other distributed runners do with a LazyFrame they send to a worker."""

import cloudpickle
import polars as pl


def pickle_round_trip(lf: pl.LazyFrame) -> pl.LazyFrame:
    """``lf`` pickled with cloudpickle and loaded back."""
    return cloudpickle.loads(cloudpickle.dumps(lf))
