import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

import polars as pl
from sqlglot import exp, parse_one
from sqlglot.dialects.dialect import Dialect

from .sql_dialects import MSSQL
from .sql_utils import (
    apply_polars_io_source_exprs,
    fix_three_part_identifiers,
)
from .util import register_io_source_with_is_pure

__all__ = ["scan_db"]


# Configure logging
log = logging.getLogger(__name__)


@lru_cache(None)
def get_sqlglot_dialect_odbc(conn_string: str) -> Optional[Union[str, type[Dialect]]]:
    import pyodbc

    DIALECT_MAP: dict[str, Union[str, type[Dialect]]] = {
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
        except Exception as e:
            log.warning(f"Got exception when trying to find dialect: {e}")
            return None


def get_schema_from_query_odbc(
    query: exp.Expression,
    connection: Union[str, Any],
    dialect: Optional[Union[str, type[Dialect]]],
    **kwargs: Any,
) -> Dict[str, pl.DataType]:
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


def scan_db(query: str, connection: str, fetch_size: int = 10000, **kwargs) -> pl.LazyFrame:
    """
    Create a LazyFrame from a SQL query with predicate pushdown support.

    This is the primary user-facing function in this module.

    This function creates a LazyFrame that will execute SQL queries against the provided
    connection with optimized predicate pushdown. Filters applied to the LazyFrame will
    be translated back to SQL and pushed to the database.

    Args:
        query (str): The SQL query to execute
        connection (str): A connection string (*not* a database connection object)
        fetch_size (int, default 10000): Number of rows to fetch at a time. This is a default needed by the \
            source generator function that scan_db wraps (because it is required \
            by the Polars IO plugins API). This value will only be used if Polars \
            does not pass a value for batch size; if it does, that will be used instead.
        **kwargs: Additional arguments for the database connector

    Returns:
        pl.LazyFrame: A Polars LazyFrame with predicate pushdown support
    """

    def _fetch_info_needing_connection() -> tuple[
        dict[str, pl.DataType],
        exp.Expression,
        Optional[Union[str, type[Dialect]]],
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

        # Create a connection string if needed
        conn_string = connection if isinstance(connection, str) else str(connection)
        try:
            from arrow_odbc import read_arrow_batches_from_odbc

            # Use arrow_odbc directly to fetch results
            batch_reader = read_arrow_batches_from_odbc(
                query=final_sql,
                batch_size=fetch_size if batch_size is None else batch_size,
                connection_string=conn_string,
                # Pass through additional connection options
                # that the user specified in the parent function
                **kwargs,
            )

            # Track if we've yielded any batches yet
            # This is necessary in case the query yields
            # no records
            count = 0

            def select_cols(df) -> pl.DataFrame:
                if with_columns is not None:
                    with_cols_set = set(with_columns)
                    return df.select(col for col in schema.keys() if col in with_cols_set)
                return df

            for record_batch in batch_reader:
                df = pl.DataFrame(record_batch)
                if predicate is not None:
                    df = df.filter(predicate)
                yield select_cols(df)
                count += 1

            if count == 0:
                yield select_cols(pl.DataFrame({}, schema=schema))

        except Exception as e:
            err_msg = f"Failed to execute SQL query: {final_sql}\nPredicate:\n{predicate}\n The `with_columns` used: {with_columns}\n"
            err_msg += f"\n\nWhile running the above, received error: {e.__class__.__name__}:{e}"
            raise RuntimeError(err_msg) from e

    return register_io_source_with_is_pure(source_generator, schema=schema)
