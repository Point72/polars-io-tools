from typing import Any, Iterator

import polars as pl

from .._compat import POLARS_HAS_COLLECT_BATCHES

__all__ = ("iter_rows",)


def iter_rows(
    lf: pl.LazyFrame, *, named: bool = False, buffer_size: int = 512, maintain_order: bool = True
) -> Iterator[tuple[Any, ...]] | Iterator[dict[str, Any]]:
    """
    Iterate over rows efficiently by collecting in batches.

    This function wraps efficient batch collection with row-by-row iteration,
    allowing you to process large LazyFrames without loading everything into memory.
    Internally uses `collect_batches` when available (Polars >= 1.34.0) or falls back
    to slicing for older versions.

    Args:
        lf (pl.LazyFrame): The LazyFrame to iterate over
        named (bool, default=False): If True, yield dictionaries. If False, yield tuples
        buffer_size (int, default=512): Number of rows to collect per batch
        maintain_order (bool, default=True): If True, maintain row order from the query plan

    Yields:
        dict[str, Any] | tuple[Any, ...]: Individual rows as dictionaries (if named=True) or tuples (if named=False)

    Examples:
        Simple usage example:
            >>> lf = pl.LazyFrame({"a": [1, 2], "b": [10, 20]})
            >>> for row in lf.piot.iter_rows(named=True):
            ...     print(row)
            {'a': 1, 'b': 10}
            {'a': 2, 'b': 20}

        With ``named=False`` (default), rows are returned as tuples:
            >>> for row in lf.piot.iter_rows(maintain_order=True):
            ...     print(row)
            (1, 10)
            (2, 20)
    """

    if POLARS_HAS_COLLECT_BATCHES:
        for batch in lf.collect_batches(chunk_size=buffer_size, maintain_order=maintain_order):
            for row in batch.iter_rows(named=named, buffer_size=buffer_size):
                yield row
    else:
        for row in lf.collect().iter_rows(named=named, buffer_size=buffer_size):
            yield row
