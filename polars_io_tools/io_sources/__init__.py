import functools
from contextlib import contextmanager
from typing import TYPE_CHECKING

import polars as pl

from .base import *
from .concat_named import *
from .delta_io import *
from .join import *
from .lazy_cache import cache, cache as _lazy_cache
from .lazy_cache_parquet import *
from .lazy_clickhouse_reader import *
from .lazy_clickhouse_writer import *
from .lazy_datadog_reader import *
from .lazy_debug import debug, debug as _lazy_debug
from .lazy_iter_rows import *
from .lazy_narwhals_reader import *
from .lazy_sql_reader import *
from .multi_source import *
from .sql_dialects import *
from .translated_source import *
from .ts import *
from .util import *

if TYPE_CHECKING:
    from .delta_io import sink_delta as sink_delta
    from .join import filtered_join, filtered_join_asof
    from .lazy_clickhouse_reader import scan_clickhouse
    from .lazy_clickhouse_writer import sink_clickhouse
    from .lazy_iter_rows import iter_rows
    from .multi_source import FilterSpec, multi_source
    from .ts import ts_with_columns
    from .util import filter_no_pushdown, with_columns_topo

if TYPE_CHECKING:
    # We don't want to import `execute_on_ray` at the top level; however
    # we also can't import it *inside* the execute_on_ray method of the
    # PIOTOperations class, because thit needs to be defined at the module level
    # for the `functools.wraps` decorator to work. That's why we use a stub here.
    from .lazy_ray import execute_on_ray as _execute_on_ray_proto
else:

    def _execute_on_ray_proto(*_a, **_kw): ...


@pl.api.register_lazyframe_namespace("piot")
class PIOTOperations:
    # If this flag is true, will replace optimized operations with their standard polars equivalents so that
    # explain can be run on the dataframe. Note that these modified explain plans will not reflect the additional
    # optimizations we are performing, but should return the same answers.
    _DISABLE_OPTIMIZATIONS: bool = False

    def __init__(self, lf: pl.LazyFrame) -> None:
        self._lf = lf

    @functools.wraps(_lazy_debug)
    def debug(self, *args, **kwargs) -> pl.LazyFrame:
        return _lazy_debug(self._lf, *args, **kwargs)

    @functools.wraps(_lazy_cache)
    def cache(self, *args, **kwargs) -> pl.LazyFrame:
        if self._DISABLE_OPTIMIZATIONS:
            return self._lf
        return _lazy_cache(self._lf, *args, **kwargs)

    @functools.wraps(filtered_join)
    def filtered_join(self, *args, **kwargs) -> pl.LazyFrame:
        if self._DISABLE_OPTIMIZATIONS:
            return self._lf.join(*args, **kwargs)
        return filtered_join(self._lf, *args, **kwargs)

    @functools.wraps(cache_parquet)  # noqa: F405
    def cache_parquet(self, *args, **kwargs) -> pl.LazyFrame:
        return cache_parquet(self._lf, *args, **kwargs)  # noqa: F405

    @functools.wraps(_execute_on_ray_proto)
    def execute_on_ray(self, *args, **kwargs) -> pl.LazyFrame:
        # heavy import happens only when the user calls the method
        from .lazy_ray import execute_on_ray as _execute_on_ray

        return _execute_on_ray(self._lf, *args, **kwargs)

    @functools.wraps(filtered_join_asof)
    def filtered_join_asof(self, *args, **kwargs) -> pl.LazyFrame:
        if self._DISABLE_OPTIMIZATIONS:
            return self._lf.join_asof(*args, **kwargs)
        return filtered_join_asof(self._lf, *args, **kwargs)

    @functools.wraps(ts_with_columns)
    def ts_with_columns(self, *args, **kwargs) -> pl.LazyFrame:
        if self._DISABLE_OPTIMIZATIONS:
            kwargs["_disable_optimizations"] = True
        return ts_with_columns(self._lf, *args, **kwargs)

    @functools.wraps(filter_no_pushdown)
    def filter_no_pushdown(self, *args, **kwargs) -> pl.LazyFrame:
        if self._DISABLE_OPTIMIZATIONS:
            kwargs["_disable_optimizations"] = True
        return filter_no_pushdown(self._lf, *args, **kwargs)

    @functools.wraps(with_columns_topo)
    def with_columns_topo(self, *args, **kwargs) -> pl.LazyFrame:
        return with_columns_topo(self._lf, *args, **kwargs)

    @functools.wraps(sink_delta)
    def sink_delta(self, *args, **kwargs):
        return sink_delta(self._lf, *args, **kwargs)

    @functools.wraps(sink_clickhouse)
    def sink_clickhouse(self, *args, **kwargs):
        return sink_clickhouse(self._lf, *args, **kwargs)

    @functools.wraps(iter_rows)
    def iter_rows(self, *args, **kwargs):
        return iter_rows(self._lf, *args, **kwargs)


@contextmanager
def disable_optimizations():
    """Context manager for disabling optimizations in operations.

    This is useful to get a proper polars "explain" plan for a LazyFrame that would otherwise be obscured
    because polars will list all operations as "PYTHON_SCAN".

    It can also be used to test that the results with and without optimizations are the same.
    """
    try:
        PIOTOperations._DISABLE_OPTIMIZATIONS = True
        yield
    finally:
        PIOTOperations._DISABLE_OPTIMIZATIONS = False
