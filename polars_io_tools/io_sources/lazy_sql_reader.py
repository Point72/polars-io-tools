import concurrent.futures
import logging
import os
import queue
import threading
from collections.abc import Iterable
from functools import lru_cache
from typing import Any

import polars as pl
from sqlglot import exp, parse_one
from sqlglot.dialects.dialect import Dialect

from .partitions import Partitioner, ReadPartition, as_partition_list, retained_columns
from .sql_dialects import MSSQL
from .sql_utils import (
    apply_polars_io_source_exprs,
    fix_three_part_identifiers,
    wrap_query_with_casts,
)
from .util import register_io_source_with_is_pure

__all__ = ("scan_db",)


# Configure logging
log = logging.getLogger(__name__)

# Process-wide budget for concurrent SQL connections. Partition reads are IO-bound (each worker
# is mostly blocked on the database), so the right cap is an IO connection budget, not the CPU
# compute pool. Tune with `POLARS_IO_TOOLS_MAX_SQL_CONNECTIONS`; when unset the default is
# `min(pl.thread_pool_size(), 8)` -- a modest 8 on a normal machine, but self-throttling to 1 in
# fan-out clusters that pin `POLARS_MAX_THREADS=1` per worker (where the real risk is N workers x
# N connections). This treats `POLARS_MAX_THREADS` only as a downward ceiling / "be gentle"
# signal, not as a compute-sizing driver. The shared executor is the global governor so
# concurrent scans cannot collectively oversubscribe. The budget is parsed lazily (on first
# parallel use) so a malformed value fails there, naming the variable, rather than breaking
# `import polars_io_tools` for everyone -- including users who never touch `scan_db`.
_SQL_CONNECTIONS_ENV = "POLARS_IO_TOOLS_MAX_SQL_CONNECTIONS"
_SQL_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None
_SQL_CONNECTION_BUDGET: int | None = None
_SQL_EXECUTOR_LOCK = threading.Lock()


def _resolve_sql_connection_budget() -> int:
    raw = os.environ.get(_SQL_CONNECTIONS_ENV)
    if raw is None or not raw.strip():
        return min(pl.thread_pool_size(), 8)
    try:
        value = int(raw.strip())
    except ValueError:
        raise ValueError(f"{_SQL_CONNECTIONS_ENV} must be a positive integer, got {raw!r}.") from None
    if value < 1:
        raise ValueError(f"{_SQL_CONNECTIONS_ENV} must be a positive integer (>= 1), got {raw!r}.")
    return value


def _get_sql_executor() -> tuple[concurrent.futures.ThreadPoolExecutor, int]:
    """Return the shared SQL thread pool and its connection budget (``k``).

    Both are created together under the lock so the pool size and the per-scan concurrency cap
    are always the same authoritative number.
    """
    global _SQL_EXECUTOR, _SQL_CONNECTION_BUDGET
    with _SQL_EXECUTOR_LOCK:
        if _SQL_EXECUTOR is None:
            _SQL_CONNECTION_BUDGET = _resolve_sql_connection_budget()
            _SQL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=_SQL_CONNECTION_BUDGET)
        return _SQL_EXECUTOR, _SQL_CONNECTION_BUDGET


# How long a producer/consumer blocks on a full/empty queue before re-checking the ``stop`` event
# (cancellation) or a worker's future (abnormal termination). Only affects cleanup/failure latency,
# never steady-state throughput (an unblocked put/get returns immediately).
_QUEUE_POLL_SECONDS = 0.1


class _SliceDone:
    """Terminal marker a worker puts on its queue when it has streamed a slice cleanly."""

    __slots__ = ("idx",)

    def __init__(self, idx: int) -> None:
        self.idx = idx


def _unwrap_top(node: exp.Expression) -> exp.Expression:
    """Unwrap parenthesized / subquery roots to the effective top-level query node."""
    while isinstance(node, (exp.Subquery, exp.Paren)) and node.this is not None:
        node = node.this
    return node


def _top_candidates(parsed: exp.Expression) -> list[exp.Expression]:
    """Query node(s) whose clauses apply to the final result.

    For a set operation (UNION/INTERSECT/EXCEPT) this includes the last branch, since dialects
    differ on whether a trailing ORDER BY/LIMIT/OFFSET/FETCH hangs off the set node or its final
    SELECT. A clause nested inside a FROM subquery is the user's own semantics and is not included.
    """
    top = _unwrap_top(parsed)
    candidates = [top]
    if isinstance(top, exp.SetOperation) and top.expression is not None:
        candidates.append(_unwrap_top(top.expression))
    return candidates


def _partition_unsafe_clause(parsed: exp.Expression) -> str | None:
    """Name a top-level clause that makes a *partitioned* read return wrong rows, else ``None``.

    Each slice runs ``SELECT * FROM (<query>) AS t WHERE <bound>``, so a top-level
    ``LIMIT``/``TOP``/``FETCH``/``OFFSET`` is applied *before* the slice bound (each slice returns a
    different subset of the same prefix), and ``QUALIFY``/window functions are recomputed over the
    whole input per slice. ``ORDER BY`` is handled separately (a warning -- it only loses global
    order, the row set is still correct). Windows/QUALIFY are matched anywhere; the row-limiting
    clauses are matched at the (unwrapped) top level only, since a nested subquery's own ``LIMIT``
    is part of the user's query semantics.
    """
    if parsed.find(exp.Window) is not None:
        return "a window function"
    if parsed.find(exp.Qualify) is not None:
        return "QUALIFY"
    for node in _top_candidates(parsed):
        if node.args.get("limit") is not None:
            return "LIMIT/TOP"
        if node.args.get("fetch") is not None:
            return "FETCH"
        if node.args.get("offset") is not None:
            return "OFFSET"
    return None


def _has_top_level_order(parsed: exp.Expression) -> bool:
    return any(node.args.get("order") is not None for node in _top_candidates(parsed))


@lru_cache(None)
def get_sqlglot_dialect_odbc(conn_string: str) -> str | type[Dialect] | None:
    import pyodbc

    DIALECT_MAP: dict[str, str | type[Dialect]] = {
        "microsoft sql server": MSSQL,
        "postgresql": "postgres",
        "oracle": "oracle",
        "mysql": "mysql",
        "snowflake": "snowflake",
        "sqlite": "sqlite",
        "amazon redshift": "redshift",
    }
    with pyodbc.connect(conn_string) as conn:
        try:
            return DIALECT_MAP[conn.getinfo(pyodbc.SQL_DBMS_NAME).lower()]
        except Exception as e:  # noqa: BLE001 -- intentional broad catch (defensive fallback)
            log.warning(f"Got exception when trying to find dialect: {e}")
            return None


def get_schema_from_query_odbc(
    query: exp.Expression,
    connection: str | Any,
    dialect: str | type[Dialect] | None,
    **kwargs: Any,
) -> dict[str, pl.DataType]:
    """
    Get the schema for a SQL query using arrow-odbc.

    Args:
        query (str): The SQL query
        connection (Union[str, Any]): Database connection or connection string
        dialect (str): SQL dialect
        **kwargs: Additional arguments to pass to arrow_odbc's read_arrow_batches_from_odbc.
            These are passed through to ensure consistency between schema detection
            and data fetching (e.g., query_timeout_sec, schema overrides, etc.).

    Returns:
        Dict[str, pl.DataType]: Schema mapping column names to Polars data types
    """

    try:
        from arrow_odbc import read_arrow_batches_from_odbc

        # Create connection string if not already a string
        conn_string = connection if isinstance(connection, str) else str(connection)
        schema_query_parsed = query.limit(0, dialect=dialect)  # type: ignore[union-attr]
        identifier_parsed = schema_query_parsed.transform(fix_three_part_identifiers)
        schema_query = identifier_parsed.sql(dialect=dialect)

        # Use arrow_odbc to get schema information directly
        # The batch reader provides schema information even for empty result sets
        batch_reader = read_arrow_batches_from_odbc(
            query=schema_query,
            batch_size=1,
            connection_string=conn_string,
            **kwargs,
        )

        # We can access the PyArrow schema directly from the batch reader
        import pyarrow as pa

        arrow_schema = batch_reader.schema
        df = pl.DataFrame(pa.Table.from_pylist([], schema=arrow_schema))
        return dict(df.schema)
    except Exception as e:
        raise ValueError(f"Could not determine schema for query: {query}, with error: {e}") from e


def scan_db(
    query: str,
    connection: str,
    fetch_size: int = 10000,
    cast_map: dict[str, Any] | None = None,
    *,
    partitions: "Partitioner | Iterable[ReadPartition] | None" = None,
    max_partitions: int = 512,
    max_concurrency: int | None = None,
    description: str | None = None,
    **kwargs,
) -> pl.LazyFrame:
    """
    Create a LazyFrame from a SQL query with predicate pushdown support.

    This is the primary user-facing function in this module.

    This function creates a LazyFrame that will execute SQL queries against the provided
    connection with optimized predicate pushdown. Filters applied to the LazyFrame will
    be translated back to SQL and pushed to the database.

    When ``partitions`` is set, the reader splits the query into independent slices and pulls
    them over parallel connections, streaming each slice's batches as they arrive. This can
    dramatically speed up large, scan-like extracts whose single-cursor transfer is the
    bottleneck. It is fully opt-in: with ``partitions=None`` the behaviour is identical to a
    plain single scan.

    Rows are yielded in completion order, which is not deterministic; apply a Polars ``.sort()``
    after the read if you need a specific order. A top-level ``ORDER BY`` in ``query`` is honoured
    only for an unpartitioned read (a single connection) -- under partitioning it is dropped with a
    warning, since each slice is read independently.

    Args:
        query (str): The SQL query to execute
        connection (str): A connection string (*not* a database connection object)
        fetch_size (int, default 10000): Number of rows to fetch at a time. This is a default needed by the \
            source generator function that scan_db wraps (because it is required \
            by the Polars IO plugins API). This value will only be used if Polars \
            does not pass a value for batch size; if it does, that will be used instead.
        cast_map (dict[str, pl.DataType] | None, default None): Optional mapping of output column \
            name to a Polars dtype to cast that column to *server-side*. The narrowing is emitted \
            as a SQL ``CAST`` inside the query, and the reported schema reflects the target dtype, \
            so filters on the cast column push down to the database. Use this to correct a \
            mis-declared source type (e.g. a column stored as ``datetime`` that should be ``date``, \
            or a numeric id delivered as ``float`` that should be an integer) without abandoning \
            ``select *`` — remaining columns pass through untouched. \
            Caveat: a predicate on a cast column is pushed down as ``CAST(col AS ...) <op> value``. \
            Wrapping the column in a function can prevent the database from using an index on it \
            (SARGability), so the effect on the query plan is backend dependent — some engines \
            optimize specific conversions (for example SQL Server seeks on ``CAST(datetime AS date)``) \
            while others fall back to a scan. For a hot path on a large indexed table, prefer a \
            filter expressed directly on the physical column instead of the cast one.
        partitions (Partitioner | Iterable[ReadPartition] | None, default None): How to split the read. \
            Pass a partitioner from :mod:`polars_io_tools.io_sources.partitions` (``by_time``, ``by_value``, \
            ``by_range``) to derive slices from the filter pushed down at scan time, or an explicit iterable \
            of :class:`ReadPartition` for hand-built slices. Each slice becomes one query on its own \
            connection; any part of a slice's predicate that cannot be pushed to SQL is enforced \
            client-side. When a partitioner cannot derive a bounded split, the query runs unpartitioned.
        max_partitions (int, default 512): Guardrail -- if partitioning would produce more than this \
            many slices, raise (raise this limit or coarsen the partitions).
        max_concurrency (int | None, default None): Optional cap on the number of partitions pulled \
            simultaneously. Hard-capped at the process-wide SQL connection budget \
            (``POLARS_IO_TOOLS_MAX_SQL_CONNECTIONS``; default ``min(pl.thread_pool_size(), 8)``, which \
            self-throttles in fan-out clusters that pin ``POLARS_MAX_THREADS=1``). Use it to throttle \
            *below* that on a shared server. None means use the full budget.
        description: Optional free-form description of this source instance, attached to its OpenTelemetry span (``explain_detail``).
        **kwargs: Additional arguments for the database connector

    Returns:
        pl.LazyFrame: A Polars LazyFrame with predicate pushdown support
    """

    conn_string = connection if isinstance(connection, str) else str(connection)

    def _fetch_info_needing_connection() -> tuple[
        dict[str, pl.DataType],
        exp.Expression,
        str | type[Dialect] | None,
    ]:
        # Figure out what dialect of SQL we're using
        dialect = get_sqlglot_dialect_odbc(conn_string=connection)

        # Parse the original query
        parsed_query = parse_one(query, dialect=dialect)
        return (
            # Pass kwargs to schema query for consistency (e.g., query_timeout_sec, schema overrides)
            get_schema_from_query_odbc(parsed_query.copy(), connection, dialect=dialect, **kwargs),
            parsed_query,
            dialect,
        )

    schema, parsed_query, dialect = _fetch_info_needing_connection()

    # Keep the raw user AST (before any cast_map wrapping) for the partition-safety guard: a
    # top-level ORDER BY/LIMIT/window is nested away once we wrap, so it must be checked here.
    original_parsed_query = parsed_query.copy()

    if cast_map:
        unknown = [name for name in cast_map if name not in schema]
        if unknown:
            raise ValueError(f"cast_map references column(s) not in the query schema {list(schema)}: {unknown}")
        # Narrow the columns server-side and report the target dtypes, so predicates on
        # a cast column push down to the database instead of stalling above a client cast.
        parsed_query = wrap_query_with_casts(parsed_query, dialect, list(schema), cast_map)
        schema = {name: cast_map.get(name, dtype) for name, dtype in schema.items()}

    if isinstance(partitions, Partitioner):
        if partitions.on not in schema:
            raise ValueError(f"partition column {partitions.on!r} is not in the query schema {list(schema)}")
    elif partitions is not None:
        # Materialise an explicit iterable once so re-collecting the LazyFrame (whose source
        # generator runs per collect) does not exhaust a bare generator on the first pass.
        partitions = list(partitions)

    def _select_cols(df: pl.DataFrame, with_columns: list[str] | None) -> pl.DataFrame:
        if with_columns is not None:
            wanted = set(with_columns)
            return df.select(col for col in schema if col in wanted)
        return df

    def _build_sql(
        predicate: pl.Expr | None,
        with_columns: list[str] | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> str:
        # Reuse the shared subquery machinery (MSSQL ORDER BY / OPTION hoisting, identifier
        # quoting) for both the pushed predicate and any partition bound folded into it.
        final_query_expr = apply_polars_io_source_exprs(parsed_query.copy(), dialect, with_columns, predicate, n_rows, batch_size)
        return final_query_expr.transform(fix_three_part_identifiers).sql(dialect=dialect)

    def _read(sql: str, batch_size: int | None):
        from arrow_odbc import read_arrow_batches_from_odbc

        return read_arrow_batches_from_odbc(
            query=sql,
            batch_size=fetch_size if batch_size is None else batch_size,
            connection_string=conn_string,
            **kwargs,
        )

    def _process_batch(record_batch, client_predicate: pl.Expr | None, keep_columns: list[str] | None) -> pl.DataFrame:
        """Turn one arrow batch into an exact, projected DataFrame.

        ``client_predicate`` (the pushed predicate ANDed with any partition bound) is reapplied
        client-side so the slice is exact even if only part of it translated to SQL -- this is
        what keeps partitions disjoint. ``keep_columns`` are the caller's requested columns; any
        extra columns retained only to evaluate the predicate are then dropped.
        """
        df = pl.DataFrame(record_batch)
        if client_predicate is not None:
            df = df.filter(client_predicate)
        return _select_cols(df, keep_columns)

    def _stream_slice(
        idx: int, sql: str, client_predicate: pl.Expr | None, batch_size: int | None, keep_columns: list[str] | None, put, stop: threading.Event
    ) -> None:
        """Worker body: stream one slice's batches into a bounded queue via ``put``.

        Batches are pushed one at a time (never accumulated into a whole-slice list), so peak
        memory stays ``O(k x batch_size)`` regardless of how many rows a slice has. ``put`` blocks
        when the queue is full (backpressure) and returns ``False`` once ``stop`` is set, at which
        point the worker exits promptly. The worker is stop-checked at batch boundaries only:
        an already-issued ODBC fetch cannot be interrupted mid-flight (rely on ``query_timeout_sec``
        for a hung query). On error the exception propagates and is carried by the worker's future.
        """
        try:
            if stop.is_set():
                return
            for record_batch in _read(sql, batch_size):
                if stop.is_set():
                    return
                if not put(_process_batch(record_batch, client_predicate, keep_columns)):
                    return
            put(_SliceDone(idx))
        except Exception as e:
            raise RuntimeError(f"scan_db failed executing partition slice {idx}:\n{sql}") from e

    # Create the generator function for our custom IO source
    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ):
        # Short-circuit: if the caller already knows zero rows are needed
        # (e.g. from head(0) on a contradictory filter), skip the query entirely.
        if n_rows == 0:
            yield _select_cols(pl.DataFrame({}, schema=schema), with_columns)
            return

        # Resolve the partition slices. ``as_partition_list`` distinguishes three cases:
        #   None -> a partitioner could not derive a bounded split (fall back to one query),
        #   []   -> a known-empty partition set (yield nothing),
        #   list -> concrete slices.
        partition_list = as_partition_list(partitions, predicate) if partitions is not None else None

        if partition_list is not None and len(partition_list) > max_partitions:
            # A concrete partition set may intentionally select a subset (e.g. by_value with an
            # explicit value list), so it cannot be silently replaced by a single unpartitioned
            # query -- raise rather than risk returning extra rows.
            raise ValueError(
                f"Partition count {len(partition_list)} exceeds max_partitions={max_partitions}; raise max_partitions or coarsen the partitions."
            )

        if partition_list is not None and len(partition_list) == 0:
            yield _select_cols(pl.DataFrame({}, schema=schema), with_columns)
            return

        # Build (sql, client_predicate) per slice. For a partitioned read we AND the partition
        # bound into the pushed predicate (reused by both the server-side SQL and the client-side
        # safety filter) and retain the columns those predicates need through projection so the
        # filter can be evaluated; n_rows is applied client-side across the stream.
        if partition_list:
            preds = [part.predicate for part in partition_list]
            if predicate is not None:
                preds.append(predicate)
            effective_wc = retained_columns(preds, with_columns)
            work: list[tuple[str, pl.Expr | None]] = []
            for part in partition_list:
                combined = part.predicate if predicate is None else (predicate & part.predicate)
                work.append((_build_sql(combined, effective_wc, None, batch_size), combined))
        else:
            work = [(_build_sql(predicate, with_columns, n_rows, batch_size), predicate)]

        # Safety guard: only when we actually partition (>1 slice). Each slice wraps the query as a
        # subquery and filters outside it, so a top-level LIMIT/OFFSET/TOP/FETCH/QUALIFY/window
        # would return wrong rows -- raise. A top-level ORDER BY only loses global order (row set is
        # still correct), so warn and proceed; the user can .sort() in Polars after the read.
        if len(work) > 1:
            unsafe = _partition_unsafe_clause(original_parsed_query)
            if unsafe is not None:
                raise ValueError(
                    f"scan_db cannot partition a query containing {unsafe}: partitioning wraps the query in a "
                    f"subquery and filters each slice, which changes the result. Apply it in Polars after the read "
                    f"(e.g. .head()/.sort()), or read without `partitions=`."
                )
            if _has_top_level_order(original_parsed_query):
                log.warning(
                    "scan_db: the query's top-level ORDER BY is not applied across partitions -- each slice is read "
                    "independently, so the rows are correct but not globally ordered. Use a Polars .sort() after the "
                    "read, or read without `partitions=`."
                )

        for sql, _ in work:
            log.debug("Executing SQL with pushdown: %s", sql)
        log.debug("scan_db running %d partition(s)", len(work))

        yielded_rows = 0

        def _emit(df: pl.DataFrame):
            """Yield a frame honouring the global n_rows cap across the stream."""
            nonlocal yielded_rows
            if n_rows is not None:
                remaining = n_rows - yielded_rows
                if remaining <= 0:
                    return
                if df.height > remaining:
                    df = df.head(remaining)
            yielded_rows += df.height
            yield df

        try:
            if len(work) == 1:
                # Single-connection streaming path (preserves original behaviour incl. empty result).
                sql, client_predicate = work[0]
                count = 0
                for record_batch in _read(sql, batch_size):
                    yield from _emit(_process_batch(record_batch, client_predicate, with_columns))
                    count += 1
                    if n_rows is not None and yielded_rows >= n_rows:
                        break
                if count == 0:
                    yield _select_cols(pl.DataFrame({}, schema=schema), with_columns)
                return

            # Partitioned path: bounded-concurrency *streaming* fan-out. Each worker streams its
            # slice's batches into a bounded queue (never a whole-slice list), so peak memory is a
            # small constant x k x batch_size, independent of row count. `k` and the pool size come
            # from one authoritative budget.
            executor, budget = _get_sql_executor()
            k = budget if max_concurrency is None else max(1, min(max_concurrency, budget))
            stop = threading.Event()

            def _put(q: queue.Queue, item) -> bool:
                # Blocking put with periodic `stop` re-checks, so a worker parked on a full queue
                # wakes and exits during cleanup instead of leaking. Only affects cleanup latency.
                while not stop.is_set():
                    try:
                        q.put(item, timeout=_QUEUE_POLL_SECONDS)
                        return True
                    except queue.Full:
                        continue
                return False

            # All workers stream into one shared bounded queue; yield whatever arrives, keeping `k`
            # slices running. Output is in completion order (not deterministic) -- callers that need
            # a specific order .sort() in Polars after the read.
            shared: queue.Queue = queue.Queue(maxsize=k)
            futures: dict[int, concurrent.futures.Future] = {}
            next_submit = 0

            def _launch():
                nonlocal next_submit
                idx = next_submit
                sql, client_predicate = work[idx]
                futures[idx] = executor.submit(_stream_slice, idx, sql, client_predicate, batch_size, with_columns, lambda it: _put(shared, it), stop)
                next_submit += 1

            try:
                for _ in range(min(k, len(work))):
                    _launch()
                while futures:
                    try:
                        item = shared.get(timeout=_QUEUE_POLL_SECONDS)
                    except queue.Empty:
                        # Surface a worker that errored (its future carries the exception); leave
                        # clean futures alone -- their `_SliceDone` marker (still queued) is the
                        # sole authority for reconciling and relaunching, so we can't drop one.
                        for fut in list(futures.values()):
                            if fut.done() and fut.exception() is not None:
                                fut.result()
                        continue
                    if isinstance(item, _SliceDone):
                        futures.pop(item.idx).result()  # reconcile
                        if next_submit < len(work):
                            _launch()
                        continue
                    yield from _emit(item)
                    if n_rows is not None and yielded_rows >= n_rows:
                        return
            finally:
                stop.set()
                for fut in futures.values():
                    fut.cancel()
                # cancel() cannot stop a worker that has already started, so wait for the running
                # ones to observe `stop` and exit (bounded to one batch fetch) before unwinding.
                # Otherwise we could return -- on n_rows early-stop or a sibling's error -- while a
                # worker still holds a connection and fetches, leaking work onto the shared executor
                # (and racing any DB state the caller tears down). wait() never re-raises, so a
                # cleanup-time worker failure cannot mask the primary early-stop result or error.
                concurrent.futures.wait(futures.values())

            if yielded_rows == 0:
                yield _select_cols(pl.DataFrame({}, schema=schema), with_columns)

        except Exception as e:
            err_msg = f"Failed to execute SQL query.\nPredicate:\n{predicate}\n The `with_columns` used: {with_columns}\n"
            err_msg += f"\n\nWhile running the above, received error: {e.__class__.__name__}:{e}"
            raise RuntimeError(err_msg) from e

    return register_io_source_with_is_pure(source_generator, schema=schema, explain_detail=description)
