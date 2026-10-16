import logging

import polars as pl

from .._compat import POLARS_GE_2
from .util import optional_deps_error, register_io_source_with_is_pure

try:
    import pyarrow as pa
    import requests
    from sqlglot import parse_one
except ImportError as exc:
    raise optional_deps_error("ClickHouse support") from exc

from .sql_utils import (
    apply_polars_io_source_exprs,
    fix_three_part_identifiers,
)

__all__ = ("scan_clickhouse",)


# Configure logging
log = logging.getLogger(__name__)

# Default (connect, read) timeout for the HTTP request. A connect timeout guards against an
# unreachable host; ``None`` for the read timeout so a long-running query that keeps streaming
# rows is never interrupted. Overridable via the ``timeout`` argument of ``scan_clickhouse``.
_HTTP_TIMEOUT = (10, None)


def get_batch_reader_http(query: str, url: str, params: dict, timeout=_HTTP_TIMEOUT):
    query = f"{query} FORMAT ArrowStream"
    r = requests.post(url, params=(params | {"query": query}), stream=True, timeout=timeout)
    r.raise_for_status()
    # Requests leaves raw responses encoded; decode HTTP compression as Arrow reads.
    r.raw.decode_content = True
    return pa.ipc.open_stream(r.raw)


def scan_clickhouse(query: str, url: str, params: dict, fetch_size: int = 10000, description: str | None = None, *, timeout=_HTTP_TIMEOUT):
    dialect = "clickhouse"
    parsed_query = parse_one(query, dialect=dialect)
    schema_query_parsed = parsed_query.copy().limit(0, dialect=dialect)
    identifier_parsed = schema_query_parsed.transform(fix_three_part_identifiers)
    schema_query = identifier_parsed.sql(dialect=dialect)
    try:
        reader = get_batch_reader_http(schema_query, url, params, timeout=timeout)
        arrow_schema = reader.schema
        df = pl.DataFrame(pa.Table.from_pylist([], schema=arrow_schema))
        schema = dict(df.schema)
        reader.close()
    except Exception as e:
        raise ValueError(f"Could not determine schema for query: {query}, with error: {e}") from e

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
            empty = pl.DataFrame({}, schema=schema)
            if with_columns is not None:
                empty = empty.select(col for col in schema if col in set(with_columns))
            yield empty
            return

        # Generate a new SQL query by combining the original query with the predicate
        query_copy = parsed_query.copy()
        final_query_expr = apply_polars_io_source_exprs(query_copy, dialect, with_columns, predicate, n_rows, batch_size)
        # Convert back to SQL string
        final_sql = final_query_expr.sql(dialect=dialect)
        log.debug(f"Executing SQL with pushdown: {final_sql}")

        try:
            # Map the Polars batch size (or the fetch_size default) to ClickHouse's block size
            # so streamed Arrow batches are sized accordingly. max_block_size is a plain HTTP
            # setting and a hint, not a hard guarantee on record-batch size.
            block_size = batch_size if batch_size is not None else fetch_size
            if POLARS_GE_2 and fetch_size > 0:
                block_size = min(block_size, fetch_size)
            ch_params = dict(params)
            if block_size > 0:
                ch_params.setdefault("max_block_size", block_size)
            reader = get_batch_reader_http(final_sql, url, ch_params, timeout=timeout)

            # Track if we've yielded any batches yet
            # This is necessary in case the query yields
            # no records
            count = 0

            def select_cols(df) -> pl.DataFrame:
                if with_columns is not None:
                    with_cols_set = set(with_columns)
                    return df.select(col for col in schema if col in with_cols_set)
                return df

            try:
                while True:
                    batch = reader.read_next_batch()
                    df = pl.DataFrame(batch)
                    if predicate is not None:
                        df = df.filter(predicate)
                    yield select_cols(df)
                    count += 1
            except StopIteration:
                pass
            finally:
                reader.close()

            if count == 0:
                yield select_cols(pl.DataFrame({}, schema=schema))

        except Exception as e:
            err_msg = f"Failed to execute SQL query: {final_sql}\nPredicate:\n{predicate}\n The `with_columns` used: {with_columns}\n"
            err_msg += f"\n\nWhile running the above, received error: {e.__class__.__name__}:{e}"
            raise RuntimeError(err_msg) from e

    return register_io_source_with_is_pure(source_generator, schema=schema, explain_detail=description)
