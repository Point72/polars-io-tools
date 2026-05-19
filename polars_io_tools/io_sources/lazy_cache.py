import hashlib
import logging
import warnings
from collections.abc import MutableMapping
from typing import Any, Dict, Hashable, Iterator, List, Literal, NamedTuple, Optional, Tuple

import polars as pl

from .dnf_visitor import _is_contradiction
from .restrict_visitor import restrict_expr_to_columns
from .util import register_io_source_with_is_pure

log = logging.getLogger(__name__)


__all__ = ["cache"]


_PartitionKey = Tuple[Tuple[str, Any], ...]


class _CacheKey(NamedTuple):
    """A key for the cache, consisting of the column name, dataframe key, and partition key."""

    df_key: str
    col: str
    partition_key: _PartitionKey


_CACHE: Dict[_CacheKey, pl.Series] = {}


def _df_key(df: pl.LazyFrame) -> str:
    """Return a unique key for the given dataframe."""
    return hashlib.md5(df.serialize()).hexdigest()


def _partition_key(partition_values: Dict[str, Hashable]) -> _PartitionKey:
    return tuple(sorted(partition_values.items()))


def _generate_expr(row: dict, schema: pl.Schema) -> pl.Expr:
    # Given a row of data, generate a Polars expression that represents
    # NOT matching that row
    col_exprs = []
    for col, value in row.items():
        if schema[col] == pl.List:
            if len(value) == 0:
                continue
            elif len(value) == 1:
                # If the value is a single item, we can use `ne` directly
                col_exprs.append(pl.col(col).ne(value[0]))
            else:
                # If the value is a list, we can use `is_in` to check if the column is not in the list
                col_exprs.append(pl.col(col).is_in(value).not_())
        else:
            target = value
            col_exprs.append(pl.col(col).ne(target))
    if len(col_exprs) == 1:
        return col_exprs[0]
    return pl.Expr.or_(*col_exprs)


def _repeated_grouping(df: pl.DataFrame) -> pl.DataFrame:
    # A performance optimization, is_in is much cheaper than OR operations
    # we perform group_by's to create a smaller expression. These are expensive
    # so we should only do this when we have few columns
    if df.is_empty():
        return df

    columns = df.columns
    last_df = None
    for idx, col in enumerate(columns):
        if df.schema[col] == pl.List:
            agg_func = pl.col(col).explode().unique().sort()
        else:
            agg_func = pl.col(col).unique().sort()
        new_df = df.group_by(
            columns[:idx] + columns[idx + 1 :],
        ).agg(agg_func)
        if last_df is not None and new_df.height == last_df.height:
            # If the height hasn't changed, we stop grouping
            return df
        last_df = df
        df = new_df
    return df


def _extract_filter_from_df(df: pl.DataFrame) -> Optional[pl.Expr]:
    """Extract filters from a DataFrame, returning a list of expressions."""
    if df.is_empty():
        return None
    schema = df.schema
    if len(schema) == 1:
        col_name, dtyp = next(iter(schema.items()))
        vals = df.select(col_name).to_series().to_list()
        if len(vals) == 1:
            inner_vals = vals[0]
            if isinstance(inner_vals, list):
                if len(inner_vals) == 0:
                    return None
                elif len(inner_vals) == 1:
                    return pl.col(col_name).ne(inner_vals[0])
                else:
                    return pl.col(col_name).is_in(inner_vals).not_()

    row_exprs = []
    for row in df.iter_rows(named=True):
        row_exprs.append(_generate_expr(row, schema))

    if len(row_exprs) == 1:
        return row_exprs[0]

    return pl.Expr.and_(*row_exprs)


def cache(
    self: pl.LazyFrame,
    cache: Optional[MutableMapping[_CacheKey, pl.Series]] = None,
    *,
    partition_cols: Tuple[str, ...] = (),
    cache_mode: Literal["cache", "ignore", "rebuild"] = "cache",
    log_explain: bool = False,
    **kwargs,
) -> pl.LazyFrame:
    """
    Create an intermediate cache for the columns of a LazyFrame, optionally partitioned by a set of columns.

    Predicates on the LazyFrame that operate on the partition columns will be used to restrict the set of partitions that are cached.
    Data is cached at the column/partition level, so all other predicates are only applied after the cache blocks are generated.

    Motivation: When iterating on data in a Lazy Frame, the scope of columns and partitions that need to be collected for further iteration may not be known upfront by the researcher.
    If there are many columns/partitions and evaluation is slow, then collecting all the data can be expensive and unnecessary.
    However, collecting too-few columns/partitions means that expanding the set of data requires either re-evaluating columns that were already collected,
    or user-level manipulation to combine previously collected data with new data so that everything is available for the next step.
    The  cacher solves this problem by lazily collecting columns and partitions as needed by the user.

    Depending on the cache implementation, the cache can persist across sessions.
    This provides checkpointing capabilities for heavy pipelines that may fail in the middle and need to be restarted.

    Args:
        self: The input data frame to cache columns of.
        cache: An optional implementation of a cache backend. Defaults to a global in-memory cache.
        partition_cols: An optional set of columns to partition the cache by. It is recommended that queries to the underlying frame for the partition cols are fast,
            i.e. they correspond to the parquet partition columns. Ordering of the result is not guaranteed when using partition columns.
        cache_mode: The caching mode; use "cache" for regular caching, "rebuild" to overwrite existing elements of the cache (i.e. to force a refresh), or "ignore" to not use the cache at all.
        log_explain: If True, logs the query plan when defining the function.
        **kwargs: Arguments to pass to the collect() method of the input data frame (i.e. to use a different engine)

    Notes:
        - The cache key is formed based on a serialized version of the input LazyFrame, the column name and the partitions.
          It means that if a persistent cache implementation is provided, the cache can remain valid between sessions.
        - The cache will be invalidated if the input LazyFrame is changed in any way (i.e. by adding a new column, or changing the underlying data source).
          It also means that the act of generating the column_cache will change the key for downstream caches,
          i.e. `df.piot.cache().select(expr_1).piot.cache()` will have a different cache from `df.select(expr_1).piot.cache()`,
          even though they return the same result.
        - Turn on debug level logging for more info about the cache hits and misses.

        .. warning::

           **Ordering Requirement**: This function relies on the source LazyFrame producing
           consistent row ordering across multiple collects. Since columns are cached independently,
           if the source LazyFrame does not guarantee deterministic ordering, different columns
           may be cached with different row orderings, leading to misaligned data when combined.

    Examples:
        Simple usage example:
            >>> import polars_io_tools.io_sources  # registers .piot namespace
            >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}).lazy()
            >>> cache = {}  # or use a persistent cache like diskcache.Cache("./polars_cache")
            >>> _ = df.piot.cache(cache).select("x").head(1).collect()
            >>> len(cache) > 0
            True
            >>> _ = df.piot.cache(cache).select(["x", "y"]).collect()  # x will be pulled from the cache
            >>> len(cache) > 1  # y was added to the cache
            True

        Speed up iteration with slow operations by caching the results of previous operations:
            >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}).lazy()
            >>> df = df.with_columns([
            ...     (pl.col("x") * 2).alias("slow"),
            ...     (pl.col("x") * 3).alias("very_slow"),
            ... ])
            >>> df = df.piot.cache()

        This first call evaluates the "slow" column (and stores it in the cache):
            >>> result = df.select(pl.col("slow").max()).collect()
            >>> result["slow"][0]
            6

        Now pull from the cache (rather than having to re-evaluate):
            >>> result = df.select([pl.col("slow").min().alias("slow_min"), pl.col("slow").max().alias("slow_max")]).collect()
            >>> result["slow_min"][0], result["slow_max"][0]
            (2, 6)

        Until now, we have been able to work with the data without ever having to materialize the "very_slow" column.
    """
    if cache_mode not in ("cache", "ignore", "rebuild"):
        raise ValueError(f"Invalid cache mode: {cache_mode}")
    if cache_mode == "ignore":
        return self

    warnings.warn(
        "cacherelies on the source LazyFrame having consistent row ordering. "
        "If the source uses non-deterministic operations (e.g., joins without maintain_order), "
        "cached columns may have misaligned rows."
    )

    if cache is None:
        cache = _CACHE

    partition_cols = tuple(sorted(partition_cols))
    # Collect the schema of the lazy frame, so we know what the output should look like
    # Note: This may be slow. Also, make sure to call this *before* generating _df_key.
    schema = self.collect_schema()

    # Generate the schema of the partition columns, as we'll need to apply the partition predicate to a frame with this schema
    partition_schema = {p_col: schema[p_col] for p_col in partition_cols}

    # Create a key for the dataframe as part of the cache keys
    df_key = _df_key(self)
    if log_explain:
        log.debug(str(self.explain()))

    def source_generator(
        with_columns: Optional[List[str]],
        predicate: Optional[pl.Expr],
        n_rows: Optional[int],
        batch_size: Optional[int],
    ) -> Iterator[pl.DataFrame]:
        """A generator that returns a dataframe from the cache."""
        # Get the set of columns to select
        if with_columns is None:
            columns_to_select = list(schema)
        else:
            columns_to_select = with_columns

        partition_predicate = None if predicate is None else restrict_expr_to_columns(predicate, set(partition_cols))

        # For each column, define a list of partitions we can find in the cache
        # Later, we will filter these partitions based on relevancy - for now we grab everything
        cached_partitions: Dict[str, List[Dict[str, Any]]] = {col: [] for col in columns_to_select}
        if cache_mode == "cache":
            if partition_cols:
                # Traverse all cache keys once (might be slow, as we don't index the cache keys by (col, df_key)
                # The goal is to find all partitions for which we have data for the given df, col
                for cache_key in cache:
                    if cache_key.df_key == df_key:
                        for col in columns_to_select:
                            if cache_key.col == col:
                                cached_partitions[col].append(dict(cache_key.partition_key))
            else:
                # When there are no partition columns, do not need to traverse all cache keys,
                # can look up the existence of the cache key directly
                for col in columns_to_select:
                    cached_partitions[col] = []
                    cache_key = _CacheKey(col=col, df_key=df_key, partition_key=_partition_key({}))
                    if cache_key in cache:
                        cached_partitions[col].append(dict(cache_key.partition_key))
        elif cache_mode == "rebuild":
            pass
        else:
            raise NotImplementedError

        # Set up a variable to store the data that goes into the final result, keyed by partition
        data: Dict[_PartitionKey, Dict[str, pl.Series]] = {}

        # Build a frame of the partition values we have, and apply the partition_predicate to select relevant partitions
        filtered_partition_dfs = []
        for col, partitions in cached_partitions.items():
            if partition_cols:
                partition_df = pl.DataFrame(partitions, schema=partition_schema)
                partition_df = partition_df if partition_predicate is None else partition_df.filter(partition_predicate)
                filtered_partition_dfs.append(partition_df)

                # Partition values in this frame are needed for the return value
                for row in partition_df.iter_rows(named=True):
                    partition_key = _partition_key(row)
                    cache_key = _CacheKey(col=col, df_key=df_key, partition_key=partition_key)
                    data.setdefault(partition_key, {})[col] = cache[cache_key]
                    log.debug("Using cached partition: %s", cache_key)

            elif partitions:  # Partitions will be an empty dict if the col was in the cache
                partition_key = _partition_key({})
                cache_key = _CacheKey(col=col, df_key=df_key, partition_key=partition_key)
                data.setdefault(partition_key, {})[col] = cache[cache_key]
                log.debug("Using cached partition: %s", cache_key)

        # For each partition identified in the existing cache, check if we need to query the lazy frame for more columns
        frames_to_collect = {}
        # query_predicate will represent the query for the partitions we do not have in the cache yet
        query_predicate = partition_predicate

        for partition_key, partition_data in data.items():
            cols_to_collect = [c for c in columns_to_select if c not in partition_data]
            # If we have no columns to collect, then we will receive an empty frame
            if not cols_to_collect:
                frames_to_collect[partition_key] = pl.LazyFrame()
                continue
            if partition_cols:
                expr_list = []
                for p_col, p_val in partition_key:
                    expr_list.append(pl.col(p_col) == p_val)
                selected_predicate = pl.Expr.and_(*expr_list) if len(expr_list) > 1 else expr_list[0]
                filtered_df = self.filter(selected_predicate)
            else:
                filtered_df = self
            # This frame corresponds to more columns needed for a known partition key
            frames_to_collect[partition_key] = filtered_df.select(cols_to_collect)

        # Lastly, query all columns for all partitions that are not in the cache.
        # We don't know the partition key, so need to include the partition columns in the query,
        # and will need to partition this frame when saving the results
        # TODO: If all the partition cols are enums/bools, then could detect when all partitions are present and skip the step
        if partition_cols or not data:
            cols_to_collect = columns_to_select.copy()
            # Make sure to include the partition columns in the query
            for p in set(partition_cols).difference(cols_to_collect):
                cols_to_collect.append(p)
            can_skip_query = False
            if query_predicate is not None and filtered_partition_dfs:
                # We already filtered the frame containing our partition information with our predicate.
                # However, now we could have a large number of combinations to check.
                filtered_tot_df = pl.concat(filtered_partition_dfs, how="vertical")
                if not filtered_tot_df.is_empty():
                    if len(partition_cols) > 1:
                        # If we have multiple partition columns, we need to group by all but one of them,
                        # so that we can check if the partition key is in the cache.
                        grouped_df = _repeated_grouping(filtered_tot_df)
                    else:
                        # If we have only one partition column, we can just select it
                        grouped_df = filtered_tot_df.select(pl.implode(partition_cols[0]))
                    not_in_cache_expr = _extract_filter_from_df(grouped_df)
                    if not_in_cache_expr is not None:
                        query_predicate = query_predicate & not_in_cache_expr
                    # We can skip the query if our filters form a contradiction
                    try:
                        can_skip_query = _is_contradiction(query_predicate, schema=schema)
                    except Exception as e:
                        log.warning(
                            f"Failed to check if the query predicate is a contradiction, this may be due to a large number of partition columns: {e}",
                        )
                        can_skip_query = False

            # This frame doesn't correspond to a fixed partition key
            if not can_skip_query:
                log.debug("Querying for data without a fixed partition key... %s", query_predicate)
                frames_to_collect[None] = (
                    self.filter(query_predicate).select(cols_to_collect) if query_predicate is not None else self.select(cols_to_collect)
                )

        # Collect all the frames together in a single call for efficiency
        if frames_to_collect:
            try:
                collected_frames = pl.collect_all(frames_to_collect.values(), **kwargs)
            except Exception as e:
                err_msg = f"Failed to collect lazy frame in piot_cache.\nPolars plan:\n{pl.explain_all(frames_to_collect.values())}"
                err_msg += f"\n\nWhile running the above, received error: {e.__class__.__name__}:{e}"
                raise RuntimeError(err_msg) from e
        else:
            collected_frames = []

        # Add the data from the frames to the cache
        for partition_key, frame in zip(frames_to_collect, collected_frames):
            if partition_key is None:
                if partition_cols:
                    frame_iter = frame.partition_by(partition_cols, as_dict=True).items()
                else:
                    frame_iter = [((), frame)]

                for partition_values, partition_frame in frame_iter:
                    partition_key = _partition_key(dict(zip(partition_cols, partition_values)))
                    for col in partition_frame.columns:
                        cache_key = _CacheKey(col=col, df_key=df_key, partition_key=partition_key)
                        log.debug("Caching new partition:  %s", cache_key)
                        cache[cache_key] = partition_frame[col]
                        data.setdefault(partition_key, {})[col] = partition_frame[col]
            else:
                for col in frame.columns:
                    cache_key = _CacheKey(col=col, df_key=df_key, partition_key=partition_key)
                    log.debug("Caching new partition:  %s", cache_key)
                    cache[cache_key] = frame[col]
                    data.setdefault(partition_key, {})[col] = frame[col]

        # Return the data as a dataframe
        out_schema = {col: schema[col] for col in columns_to_select}
        frames = []
        for partition_key, col_values in data.items():
            # We might have pulled in more columns than requested for populating the cache,
            # We need to filter them out appropriately.
            df = pl.DataFrame(col_values, schema_overrides=out_schema).select(columns_to_select)
            if predicate is not None:
                df = df.filter(predicate)
            frames.append(df)

        if frames:
            df = pl.concat(frames, how="vertical")
        else:
            df = pl.DataFrame(schema=out_schema)

        # Apply n_rows
        df = df.head(n_rows) if n_rows is not None else df
        # Apply batch_size
        if batch_size is None:
            yield df
        else:
            yield from df.iter_slices(n_rows=batch_size)

    return register_io_source_with_is_pure(source_generator, schema=schema, validate_schema=True)
