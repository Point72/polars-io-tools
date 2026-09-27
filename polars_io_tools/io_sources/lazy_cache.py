import hashlib
import logging
import threading
from collections.abc import Iterator, MutableMapping, Sequence
from typing import Any, Literal, NamedTuple

import polars as pl

from .dnf_visitor import _is_contradiction
from .restrict_visitor import restrict_expr_to_columns
from .util import (
    PartitionKey as _PartitionKey,
    partition_exclusion_predicate,
    partition_key as _partition_key,
    register_io_source_with_is_pure,
)

log = logging.getLogger(__name__)


__all__ = ("cache",)


class _CacheKey(NamedTuple):
    """A key for the cache, consisting of the column name, dataframe key, and partition key."""

    df_key: str
    col: str
    partition_key: _PartitionKey


_CACHE: dict[_CacheKey, pl.Series] = {}

# Guards cache discovery and publication so concurrent collections sharing a cache cannot mutate
# the mapping mid-iteration or read a half-published partition. collect_all runs outside the
# lock, so independent collections still parallelize.
_CACHE_LOCK = threading.Lock()


def _cache_lock() -> threading.Lock:
    """Return the shared cache lock, via a function so the io_source stays picklable.

    A ``threading.Lock`` cannot be pickled, so the generator closes over this module-level
    function (referenced by name) rather than the lock object itself.
    """
    return _CACHE_LOCK


def _df_key(df: pl.LazyFrame, order_by: tuple[str, ...] = (), partition_cols: tuple[str, ...] = ()) -> str:
    """Return a unique key for the given dataframe, ordering key and partition layout."""
    payload = df.serialize() + repr((tuple(order_by), tuple(partition_cols))).encode()
    return hashlib.md5(payload).hexdigest()


def _validate_order_by_unique(frame: pl.DataFrame, order_by: tuple[str, ...]) -> None:
    """Raise if ``order_by`` does not uniquely identify the rows of ``frame``.

    Operates on a frame already collected for the cache fill, so it adds no extra pass
    over the source.
    """
    if frame.height and frame.select(list(order_by)).is_duplicated().any():
        raise ValueError(
            f"order_by={order_by} does not uniquely identify rows (duplicate keys found). "
            "A unique total order is required so that independently cached columns remain "
            "aligned. When partition_cols is used, order_by must be unique within each "
            "partition. Pass validate=False to skip this check if you are certain the "
            "ordering is unique."
        )


def cache(
    self: pl.LazyFrame,
    cache: MutableMapping[_CacheKey, pl.Series] | None = None,
    *,
    order_by: str | Sequence[str],
    partition_cols: tuple[str, ...] = (),
    column_granularity: bool = True,
    cache_mode: Literal["cache", "ignore", "rebuild"] = "cache",
    validate: bool = True,
    log_explain: bool = False,
    description: str | None = None,
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
        order_by: One or more columns that uniquely identify each row (a unique total order) —
            typically the frame's natural row identity, such as a primary key or ``date`` + ``symbol``.
            Every cached column is collected and sorted by these columns so that independently cached
            columns share one row order and stay aligned when recombined. ``partition_cols`` is not a
            substitute: a partition normally holds many rows, so ``order_by`` must be unique *within*
            each partition. Uniqueness is verified (see ``validate``). Output rows are returned
            sorted by ``order_by`` (within each partition when ``partition_cols`` is set).
        partition_cols: An optional set of columns to partition the cache by. It is recommended that queries to the underlying frame for the partition cols are fast,
            i.e. they correspond to the parquet partition columns.
        column_granularity: If True (default), each column is cached independently, so selecting a
            not-yet-cached column re-queries the source for it. If False, the first touch of a partition
            collects and stores its whole schema (under the same per-column keys), so later selects of any
            column are served from the cache rather than re-evaluated — at the cost of evaluating every
            column up front. Row partitioning is preserved either way, and a ``False`` fill is readable by a
            ``True`` reader.
        cache_mode: The caching mode; use "cache" for regular caching, "rebuild" to overwrite existing elements of the cache (i.e. to force a refresh), or "ignore" to not use the cache at all.
        validate: If True (default), verify that ``order_by`` uniquely identifies rows of each
            collected block, raising at ``collect`` time otherwise (Polars surfaces this as a
            ``ComputeError`` wrapping the validation ``ValueError``). The check runs on data already
            collected for the cache fill, so it adds no extra pass over the source. Set to False to
            skip it on hot paths where uniqueness is already guaranteed.
        log_explain: If True, logs the query plan when defining the function.
        description: Optional free-form description of this source instance, attached to its OpenTelemetry span (``explain_detail``).
        **kwargs: Arguments to pass to the collect() method of the input data frame (i.e. to use a different engine)

    Notes:
        - The cache key is formed based on a serialized version of the input LazyFrame, the column name and the partitions.
          It means that if a persistent cache implementation is provided, the cache can remain valid between sessions.
        - The cache will be invalidated if the input LazyFrame is changed in any way (i.e. by adding a new column, or changing the underlying data source).
          It also means that the act of generating the column_cache will change the key for downstream caches,
          i.e. `df.piot.cache(order_by="id").select(expr_1).piot.cache(order_by="id")` will have a different cache from `df.select(expr_1).piot.cache(order_by="id")`,
          even though they return the same result.
        - Turn on debug level logging for more info about the cache hits and misses.

    Examples:
        Simple usage example:
            >>> import polars_io_tools.io_sources  # registers .piot namespace
            >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}).lazy()
            >>> cache = {}  # or use a persistent cache like diskcache.Cache("./polars_cache")
            >>> _ = df.piot.cache(cache, order_by="x").select("x").head(1).collect()
            >>> len(cache) > 0
            True
            >>> _ = df.piot.cache(cache, order_by="x").select(["x", "y"]).collect()  # x will be pulled from the cache
            >>> len(cache) > 1  # y was added to the cache
            True

        Speed up iteration with slow operations by caching the results of previous operations:
            >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}).lazy()
            >>> df = df.with_columns([
            ...     (pl.col("x") * 2).alias("slow"),
            ...     (pl.col("x") * 3).alias("very_slow"),
            ... ])
            >>> df = df.piot.cache(order_by="x")

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

    order_by = (order_by,) if isinstance(order_by, str) else tuple(order_by)
    if not order_by:
        raise ValueError("order_by must specify at least one column")

    if cache_mode == "ignore":
        return self

    if cache is None:
        cache = _CACHE

    partition_cols = tuple(sorted(partition_cols))
    # Collect the schema of the lazy frame, so we know what the output should look like
    # Note: This may be slow. Also, make sure to call this *before* generating _df_key.
    schema = self.collect_schema()

    missing = [col for col in order_by if col not in schema]
    if missing:
        raise ValueError(f"order_by columns not found in frame schema: {missing}")

    # Generate the schema of the partition columns, as we'll need to apply the partition predicate to a frame with this schema
    partition_schema = {p_col: schema[p_col] for p_col in partition_cols}

    # Create a key for the dataframe as part of the cache keys. The ordering key and partition
    # layout are folded in so that caches built with a different order_by or different
    # partition_cols never collide.
    df_key = _df_key(self, order_by, partition_cols)
    if log_explain:
        log.debug(str(self.explain()))

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        """A generator that returns a dataframe from the cache."""
        # Columns the caller actually wants back.
        if with_columns is None:
            return_columns = list(schema)
        else:
            return_columns = with_columns
        # Columns to look up and fill in the cache. When column_granularity is False we collect
        # and store the whole schema on any miss (under the existing per-column keys), so a
        # partition's first touch pulls every column and later selects are pure hits.
        columns_to_select = list(schema) if not column_granularity else return_columns

        partition_predicate = None if predicate is None else restrict_expr_to_columns(predicate, set(partition_cols))

        # For each column, define a list of partitions we can find in the cache
        # Later, we will filter these partitions based on relevancy - for now we grab everything
        cached_partitions: dict[str, list[dict[str, Any]]] = {col: [] for col in columns_to_select}
        # Read under the lock so a concurrent publish can't resize the mapping mid-scan.
        with _cache_lock():
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
        data: dict[_PartitionKey, dict[str, pl.Series]] = {}

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
                    expr_list.append(pl.col(p_col).eq_missing(p_val))
                selected_predicate = pl.Expr.and_(*expr_list) if len(expr_list) > 1 else expr_list[0]
                filtered_df = self.filter(selected_predicate)
            else:
                filtered_df = self
            # Include the ordering key so the block can be sorted into the canonical order.
            cols_with_order = cols_to_collect + [c for c in order_by if c not in cols_to_collect]
            # This frame corresponds to more columns needed for a known partition key
            frames_to_collect[partition_key] = filtered_df.select(cols_with_order).sort(list(order_by))

        # Lastly, query all columns for all partitions that are not in the cache.
        # We don't know the partition key, so need to include the partition columns in the query,
        # and will need to partition this frame when saving the results
        # TODO: If all the partition cols are enums/bools, then could detect when all partitions are present and skip the step
        if partition_cols or not data:
            cols_to_collect = columns_to_select.copy()
            # Make sure to include the partition columns in the query
            for p in set(partition_cols).difference(cols_to_collect):
                cols_to_collect.append(p)
            # Make sure to include the ordering key so the block can be sorted canonically.
            for c in order_by:
                if c not in cols_to_collect:
                    cols_to_collect.append(c)
            can_skip_query = False
            if filtered_partition_dfs:
                # Restrict the query to only the partitions we do not already have cached, by
                # excluding every known partition. This runs even without an incoming predicate
                # (an unfiltered read): otherwise an unfiltered select over a fully-cached
                # partitioned frame would re-scan the whole source instead of just the missing
                # partitions. Mirrors the in-memory cache's build planner.
                filtered_tot_df = pl.concat(filtered_partition_dfs, how="vertical")
                not_in_cache_expr = partition_exclusion_predicate(filtered_tot_df, partition_cols)
                if not_in_cache_expr is not None:
                    if query_predicate is None:
                        # No incoming predicate: the query is just the exclusion. A bare
                        # exclusion of already-seen partitions is never itself a contradiction
                        # (short of enumerating the column's full domain, see the TODO above),
                        # so skip the contradiction check, which can be expensive with many
                        # partitions, and simply restrict the scan.
                        query_predicate = not_in_cache_expr
                    else:
                        query_predicate = query_predicate & not_in_cache_expr
                        # With an incoming predicate the combination may be impossible (all
                        # matching partitions already cached); if so we can skip the query.
                        try:
                            can_skip_query = _is_contradiction(query_predicate, schema=schema)
                        except Exception as e:  # noqa: BLE001 -- intentional broad catch (defensive fallback)
                            log.warning(
                                f"Failed to check if the query predicate is a contradiction, this may be due to a large number of partition columns: {e}",
                            )
                            can_skip_query = False

            # This frame doesn't correspond to a fixed partition key
            if not can_skip_query:
                log.debug("Querying for data without a fixed partition key... %s", query_predicate)
                frames_to_collect[None] = (
                    self.filter(query_predicate).select(cols_to_collect).sort(list(order_by))
                    if query_predicate is not None
                    else self.select(cols_to_collect).sort(list(order_by))
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

        # Publish under the lock so a partition's columns appear to other collections together.
        with _cache_lock():
            for partition_key, frame in zip(frames_to_collect, collected_frames):
                if partition_key is None:
                    if partition_cols:
                        frame_iter = frame.partition_by(partition_cols, as_dict=True).items()
                    else:
                        frame_iter = [((), frame)]

                    for partition_values, partition_frame in frame_iter:
                        if validate and all(c in partition_frame.columns for c in order_by):
                            _validate_order_by_unique(partition_frame, order_by)
                        partition_key = _partition_key(dict(zip(partition_cols, partition_values)))
                        for col in partition_frame.columns:
                            cache_key = _CacheKey(col=col, df_key=df_key, partition_key=partition_key)
                            log.debug("Caching new partition:  %s", cache_key)
                            cache[cache_key] = partition_frame[col]
                            data.setdefault(partition_key, {})[col] = partition_frame[col]
                else:
                    if validate and all(c in frame.columns for c in order_by):
                        _validate_order_by_unique(frame, order_by)
                    for col in frame.columns:
                        cache_key = _CacheKey(col=col, df_key=df_key, partition_key=partition_key)
                        log.debug("Caching new partition:  %s", cache_key)
                        cache[cache_key] = frame[col]
                        data.setdefault(partition_key, {})[col] = frame[col]

        # Return the data as a dataframe
        out_schema = {col: schema[col] for col in return_columns}
        frames = []
        for partition_key, col_values in data.items():
            # We might have pulled in more columns than requested for populating the cache,
            # We need to filter them out appropriately.
            df = pl.DataFrame(col_values, schema_overrides=out_schema).select(return_columns)
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

    return register_io_source_with_is_pure(source_generator, schema=schema, validate_schema=True, explain_detail=description)
