import io
import logging

import polars as pl
import pyarrow as pa
import requests

from .._compat import POLARS_HAS_COLLECT_BATCHES

__all__ = ["sink_clickhouse"]


# Configure logging
log = logging.getLogger(__name__)


def _write_arrow_to_clickhouse(table: str, arrow_table: pa.Table, url: str, params: dict) -> None:
    """Serialize a PyArrow Table to Arrow IPC and POST it to ClickHouse."""

    # Serialize the PyArrow Table to Arrow IPC Streaming format bytes.
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, arrow_table.schema) as writer:
        writer.write_table(arrow_table)
    payload = sink.getvalue()

    # POST the bytes to ClickHouse.
    query = f"INSERT INTO {table} FORMAT ArrowStream"
    try:
        r = requests.post(
            url,
            params=(params | {"query": query}),
            data=payload,
            headers={"Content-Type": "application/octet-stream"},
        )
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Failed to write Arrow data to ClickHouse table {table!r} "
            f"({arrow_table.num_rows} rows, {arrow_table.num_columns} columns).\n  Error: {e}"
        ) from e


def _prepare_for_clickhouse(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Cast unsupported types to ClickHouse-compatible equivalents.

    Duration/Time -> Int64, Categorical/Enum -> String.
    """
    return lf.with_columns(
        pl.col(pl.Duration).cast(pl.Int64),
        pl.col(pl.Time).cast(pl.Int64),
        pl.col(pl.Categorical).cast(pl.String),
        pl.col(pl.Enum).cast(pl.String),
    )


def sink_clickhouse(
    lf: pl.LazyFrame,
    table: str,
    url: str,
    params: dict,
    *,
    chunk_size: int | None = None,
) -> None:
    """Write a Polars LazyFrame to an existing ClickHouse table via HTTP Arrow IPC streaming.

    The target table must already exist in ClickHouse. This function does not
    create tables automatically because ClickHouse table creation requires
    deployment-specific settings (engine type, ORDER BY key, partitioning,
    TTL, replication, etc.) that cannot be reliably inferred from a DataFrame
    schema. Create tables via SQL before calling this function.

    The following Polars types are automatically cast before writing because
    ClickHouse has no native equivalent:

    - ``Duration`` -> ``Int64`` (raw tick count)
    - ``Time`` -> ``Int64`` (nanoseconds since midnight)
    - ``Categorical`` / ``Enum`` -> ``String``

    Args:
        lf (pl.LazyFrame): The LazyFrame to write.
        table (str): Target ClickHouse table name (e.g. ``"db.my_table"``).
        url (str): ClickHouse HTTP endpoint URL (e.g. ``"https://host:8443"``).
        params (dict): HTTP query parameters forwarded to every request. Typically includes
            ``user``, ``password``, and ``database``.
        chunk_size (int or None, default None): When set, collect and POST data in batches of this many rows instead
            of materializing the entire LazyFrame at once. Requires Polars >= 1.34.0.
            Batched writes are **not** transactional; a failure mid-stream may leave
            partial data in the table.

    Raises:
        RuntimeError: If the table does not exist or the ClickHouse server returns an error.
    """
    # Cast unsupported Polars types (Duration, Time, Categorical, Enum)
    lf = _prepare_for_clickhouse(lf)

    #  Write data
    if POLARS_HAS_COLLECT_BATCHES and chunk_size is not None and chunk_size != -1:
        for batch_df in lf.collect_batches(chunk_size=chunk_size, maintain_order=True):
            arrow_table = batch_df.to_arrow()
            _write_arrow_to_clickhouse(table, arrow_table, url, params)
    else:
        df = lf.collect()
        arrow_table = df.to_arrow()
        _write_arrow_to_clickhouse(table, arrow_table, url, params)
