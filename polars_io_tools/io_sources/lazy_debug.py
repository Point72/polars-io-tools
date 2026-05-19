import logging
from typing import Iterator, List, Optional

import polars as pl

from .util import collect_lf_in_io_source, register_io_source_with_is_pure

log = logging.getLogger(__name__)


__all__ = ("debug",)


def debug(
    self: pl.LazyFrame,
    log_level: Optional[int] = None,
) -> pl.LazyFrame:
    """
    A very simple pass-through lazy frame source to help with debugging experimentation of polars io sources and lazy frame behavior.

    Args:
        self: The input data frame to cache columns of.
        log_level: If provided, will log at the given level. If None, will print. Defaults to None.
    """
    schema = self.collect_schema()

    def source_generator(
        with_columns: Optional[List[str]],
        predicate: Optional[pl.Expr],
        n_rows: Optional[int],
        batch_size: Optional[int],
    ) -> Iterator[pl.DataFrame]:
        """A generator that returns a dataframe from the cache."""
        msg = f"debug called with `with_columns={with_columns}`, `predicate={predicate}`, `n_rows={n_rows}`, `batch_size={batch_size}` on lazy frame {repr(self)} with optimized plan:\n{self.explain()}."
        if log_level is not None:
            log.log(log_level, msg)
        else:
            print(msg)

        df = self
        if predicate is not None:
            df = df.filter(predicate)
        if with_columns is not None:
            df = df.select(with_columns)
        if n_rows is not None:
            df = df.head(n_rows)

        # Apply batch_size with version-guarded streaming
        try:
            yield from collect_lf_in_io_source(df, batch_size)
        except Exception as e:
            err_msg = f"Failed during collection in debug.\nPolars plan:\n{self.explain()}\nError: {e.__class__.__name__}:{e}"
            raise RuntimeError(err_msg) from e

    # TODO: Turn on validate_schema when this is solved: https://github.com/pola-rs/polars/issues/22110
    return register_io_source_with_is_pure(source_generator, schema=schema, validate_schema=False)
