import logging
from typing import List, Optional

import polars as pl
import pyarrow as pa
import requests
from sqlglot import parse_one

from .sql_utils import (
    apply_polars_io_source_exprs,
    fix_three_part_identifiers,
)
from .util import register_io_source_with_is_pure

__all__ = ["scan_clickhouse"]


# Configure logging
log = logging.getLogger(__name__)


def get_batch_reader_http(query: str, url: str, params: dict):
    query = f"{query} FORMAT ArrowStream"
    r = requests.post(url, params=(params | {"query": query}), stream=True)
    r.raise_for_status()
    return pa.ipc.open_stream(r.raw)


def scan_clickhouse(query: str, url: str, params: dict, fetch_size: int = 10000):
    # TODO: fetch_size param needs to be properly handled
    log.warning("fetch_size=%d is currently ignored and has no effect. Proper fetch_size support will be added in a future release.", fetch_size)
    dialect = "clickhouse"
    parsed_query = parse_one(query, dialect=dialect)
    schema_query_parsed = parsed_query.copy().limit(0, dialect=dialect)
    identifier_parsed = schema_query_parsed.transform(fix_three_part_identifiers)
    schema_query = identifier_parsed.sql(dialect=dialect)
    try:
        reader = get_batch_reader_http(schema_query, url, params)
        arrow_schema = reader.schema
        df = pl.DataFrame(pa.Table.from_pylist([], schema=arrow_schema))
        schema = dict(df.schema)
        reader.close()
    except Exception as e:
        raise ValueError(f"Could not determine schema for query: {query}, with error: {e}") from e

    # Create the generator function for our custom IO source
    def source_generator(
        with_columns: Optional[List[str]],
        predicate: Optional[pl.Expr],
        n_rows: Optional[int],
        batch_size: Optional[int],
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
            # TODO: Support batch_size
            # TODO: Support kwargs and whatever they might mean in clickhouse's context
            reader = get_batch_reader_http(final_sql, url, params)

            # Track if we've yielded any batches yet
            # This is necessary in case the query yields
            # no records
            count = 0

            def select_cols(df) -> pl.DataFrame:
                if with_columns is not None:
                    with_cols_set = set(with_columns)
                    return df.select(col for col in schema.keys() if col in with_cols_set)
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

    return register_io_source_with_is_pure(source_generator, schema=schema)
