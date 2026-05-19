from __future__ import annotations

import base64
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union

import polars as pl

from .._compat import POLARS_HAS_COLLECT_BATCHES
from .base import get_parsed_expr
from .dnf_visitor import convert_expr_to_dnf
from .enum import DataType
from .translated_source import mapping_to_metadata, metadata_to_mapping, translate_polars_predicate
from .util import _storage_options_for, collect_lf_in_io_source, extract_description_block, inject_description_block, register_io_source_with_is_pure

if TYPE_CHECKING:
    from deltalake import DeltaTable

# Mapping block tag used in Delta table metadata description.
# The mapping is stored as base64-encoded JSON between [tag:begin] and [tag:end] markers.
_MAPPING_BLOCK_TAG = "cpl.mapping"

__all__ = [
    "scan_delta",
    "sink_delta",
]

log = logging.getLogger(__name__)


def _read_mapping_from_delta_configuration(
    source: Union[str, Path, Any],
    *,
    storage_options: Optional[Dict[str, Any]] = None,
    version: Optional[Union[int, str, datetime]] = None,
) -> Optional[Dict[str, pl.DataType]]:
    from deltalake import DeltaTable  # type: ignore

    # Accept pre-constructed DeltaTable
    if isinstance(source, DeltaTable):
        dt = source
    else:
        # DeltaTable only accepts int version; str/datetime handled via load_as_of
        int_version = version if isinstance(version, int) or version is None else None
        dt = DeltaTable(source, storage_options=storage_options, version=int_version)
    meta = dt.metadata()
    if meta is None:
        return None
    # Prefer description-embedded mapping
    desc = getattr(meta, "description", None) or getattr(meta, "configuration", dict()).get("description")
    meta_bytes = None
    if isinstance(desc, str):
        # Extensible block syntax only: [MAPPING_BLOCK_TAG:begin] ... [MAPPING_BLOCK_TAG:end]
        payload = extract_description_block(desc, _MAPPING_BLOCK_TAG)
        if payload is not None:
            meta_bytes = base64.b64decode(payload, validate=True)
    if meta_bytes is None:
        return None
    mapping = metadata_to_mapping(meta_bytes)
    return mapping


def _compute_exposed_schema_from_dt(
    dt: Any,
    mapping: Optional[Dict[str, pl.DataType]],
) -> Dict[str, pl.DataType]:
    """Derive exposed schema from a `DeltaTable` using optional logical-type mapping."""
    exposed: dict[str, pl.DataType] = {}
    base_schema = pl.Schema(dt.schema())
    for name, dtype in base_schema.items():
        exposed[name] = mapping.get(name, dtype) if mapping is not None else dtype
    return exposed


def _scan_parquet_with_delta_uris(
    dt: "DeltaTable",
    file_uris: list[str],
    *,
    storage_options: Dict[str, Any],
    credential_provider: Optional[Union[str, Any]],
    rechunk: Optional[bool],
) -> pl.LazyFrame:
    """Build a partition-aware Parquet scan using Delta table metadata and provided URIs.

    Mirrors Polars' Delta scan path when reading Parquet files directly:
    - Split schema into main vs hive (partition) columns; we rely on
      `hive_partitioning=True` to project partition columns.
    - Return `pl.scan_parquet(...)` configured with provided options.
    """
    # Gather schema and partition columns to mirror Polars' split
    delta_schema = dt.schema()
    polars_schema = pl.Schema(delta_schema)
    meta = dt.metadata()
    partition_columns = list(getattr(meta, "partition_columns", []) or [])

    # Split schema names into main vs hive (partition) columns
    # Split into main schema (non-partition columns) and hive schema (partition columns)
    main_schema: Dict[str, pl.DataType] = {}
    hive_schema: Dict[str, pl.DataType] = {}
    if partition_columns:
        for name, dtype in polars_schema.items():
            if name in partition_columns:
                hive_schema[name] = dtype
            else:
                main_schema[name] = dtype
    else:
        # No partition columns; all columns belong to main schema
        for name, dtype in polars_schema.items():
            main_schema[name] = dtype

    return pl.scan_parquet(
        file_uris,
        schema=main_schema or None,
        hive_schema=(hive_schema or None) if partition_columns else None,
        missing_columns="insert",
        extra_columns="ignore",
        hive_partitioning=bool(partition_columns),
        storage_options=storage_options,
        credential_provider=credential_provider,
        rechunk=rechunk or False,
        cast_options=pl.ScanCastOptions._default_iceberg(),
    )


def _convert_literal_for_partition(value: Any, dtype: Optional[pl.DataType]) -> Any:
    """Convert a Python literal to the underlying partition value representation.

    - None → None (Delta `file_uris` expects None to match null partitions)
    - Datetime → integer epoch counts in declared unit
    - Duration → integer counts in declared unit
    - Time → integer nanoseconds since midnight
    - Otherwise passthrough
    """
    if value is None:
        return None
    if dtype is None:
        return value
    try:
        if isinstance(dtype, pl.Datetime):
            # Use Polars to compute epoch counts in the target time unit
            unit = getattr(dtype, "time_unit", None) or "ns"
            s = pl.Series([value]).cast(pl.Datetime(unit), strict=False)
            return s.dt.timestamp(unit).cast(pl.Int64).to_list()[0]
        if isinstance(dtype, pl.Duration):
            unit = getattr(dtype, "time_unit", None) or "ms"
            s = pl.Series([value], dtype=pl.Duration(unit))
            return s.cast(pl.Int64).to_list()[0]
        if (dtype == pl.Time) or isinstance(dtype, pl.Time):
            s = pl.Series([value], dtype=pl.Time)
            return s.cast(pl.Int64).to_list()[0]
    except Exception:
        # Fallback: return original if conversion fails
        return value
    return value


def _restrict_and_normalize_partition_filters(
    dt: Any,
    predicate: Optional[pl.Expr],
    mapping: Optional[Dict[str, pl.DataType]],
) -> list[list[tuple[str, str, Any]]] | None:
    """Restrict DNF to Delta partition columns and normalize operators/values.

    - Normalize operators: '=='→'=', '!in'→'not in', 'is None'→ '=' ''
    - Convert literal values for mapped temporal types to underlying ints

    Delta partition filter rules (delta-rs Table.file_uris):
    - Predicates are expressed in DNF: inner lists are AND, outer list is OR.
    - Supported ops: '=', '!=', 'in', 'not in'. Range ops are not supported.
    - Values must be strings; use empty string '' to represent NULL partition values.
    See: https://github.com/delta-io/delta-rs/blob/python-v1.2.1/python/deltalake/table.py#L341

    Note that while these docs state DNF is accepted, we can actually only pass in conjunctions, so we manually call file_uri's multiple times if we have a disjunction.
    """
    if predicate is None:
        return None
    dnf = convert_expr_to_dnf(get_parsed_expr(predicate))
    if not dnf:
        return None
    # We do not restrict to partition columns; Delta may ignore non-partition filters.
    normalized: list[list[tuple[str, str, Any]]] = []
    unsafe_prune = False  # if any clause becomes empty (all unsupported), we must not prune at all
    # Represent null partitions as empty string for both `file_uris` and
    # hive-style paths. The deltalake `encode_partition_value` does not accept
    # None, so we must use "" here.
    null_tokens = [""]
    supported_ops = {"=", "!=", "in", "not in"}
    for clause in dnf:
        out_clause: list[tuple[str, str, Any]] = []
        for col, op, val in clause:
            # Map unsupported/negated ops to supported ones
            op2 = op
            if op2 == "==":
                op2 = "="
            elif op2 == "!in":
                op2 = "not in"
            # Normalize null comparisons
            if (op2 in {"is", "="}) and val is None:
                # Represent null partitions; include both None and empty-string tokens
                for tok in null_tokens:
                    normalized.append([(col, "=", tok)])
                continue
            elif (op2 in {"is not", "!="}) and val is None:
                for tok in null_tokens:
                    out_clause.append((col, "!=", tok))
                val2 = None
            else:
                val2 = val
            # Convert values for temporal mapped types
            dtype = mapping.get(col) if mapping is not None else None
            if op2 in {"in", "not in"}:
                seq = list(val2) if isinstance(val2, (list, tuple, set)) else [val2]
                out_vals = [_convert_literal_for_partition(v, dtype) for v in seq]
                out_clause.append((col, op2, out_vals))
                if op2 == "not in":
                    # We make sure we include null tokens
                    for tok in null_tokens:
                        if tok not in out_vals:
                            normalized.append([(col, "=", tok)])
            else:
                out_val = _convert_literal_for_partition(val2, dtype)
                # Ensure we only keep supported ops to avoid over-pruning
                if op2 in supported_ops:
                    out_clause.append((col, op2, out_val))
                # else: skip this tuple
        if len(out_clause) > 0:
            normalized.append(out_clause)
        else:
            # Entire clause had no supported tuples; with OR semantics, we cannot safely prune
            unsafe_prune = True
    if unsafe_prune:
        return None
    return normalized or None


def _get_partition_uris(dt: Any, predicate: Optional[pl.Expr], mapping: Optional[Dict[str, pl.DataType]]) -> list[str]:
    """Retrieve data file URIs from Delta by translating a predicate into DNF,
    restricting to partition columns and converting literal types as needed.

    Always perform a per-clause union because `DeltaTable.file_uris` expects a
    single clause (list of tuples). This yields exact pruning without widening
    filters and avoids version-dependent errors from nested DNF.
    """
    pf = _restrict_and_normalize_partition_filters(dt, predicate, mapping)
    log.debug("Restricted Delta partition_filters: %s", pf)
    # If no filters, return all URIs directly
    if not pf:
        return dt.file_uris()

    # Union results across DNF clauses
    seen: set[str] = set()
    fallback_all = False
    for idx, clause in enumerate(pf):
        # Convert None literal tokens to empty string for deltalake
        clause_adj = []
        for c, o, v in clause:
            if isinstance(v, list):
                v2 = ["" if vv is None else vv for vv in v]
            else:
                v2 = "" if v is None else v
            # Detect explicit null token usage which deltalake may not prune reliably
            if (o in {"=", "in"}) and ((v2 == "") or (isinstance(v2, list) and "" in v2)):
                # Cannot reliably prune null partitions via file_uris;
                log.debug("Clause %d contains explicit null; skipping file_uris prune", idx)
                continue
            clause_adj.append((c, o, v2))

        log.debug("Querying file_uris for clause %d: %s", idx, clause_adj)
        res = []
        try:
            res = dt.file_uris(partition_filters=clause_adj)
        except Exception as e_clause:
            log.debug("Falling back to getting all files, collecting file_uris failed for clause %d: %r", idx, e_clause, exc_info=True)
            fallback_all = True
        if fallback_all:
            # No need to continue, we will fallback to all anyways
            break
        for u in res:
            seen.add(u)

    if fallback_all:
        # Union with all URIs to ensure null partitions are included when needed
        seen = dt.file_uris()
    res = sorted(seen)
    log.debug(f"Returning file_uris: {res}")
    return res


def infer_logical_mapping(schema: Dict[str, pl.DataType]) -> Dict[str, pl.DataType]:
    """Construct a logical-type mapping from a Polars schema.

    For temporal types (Datetime, Duration, Time) in the provided schema,
    returns a mapping from column name to the logical `pl.DataType` that should
    be exposed by readers/writers.

    Non-temporal types are omitted.
    """
    mapping: Dict[str, pl.DataType] = {}
    for name, dt in schema.items():
        base = DataType.from_polars_dtype(dt)
        if base == DataType.DATETIME:
            unit: Literal["ns", "us", "ms"] | None = getattr(dt, "time_unit", None)
            # Delta supports microseconds natively; skip mapping for us
            if unit != "us" and unit is not None:
                mapping[name] = pl.Datetime(unit)
        elif base == DataType.DURATION:
            unit = getattr(dt, "time_unit", None)
            # Always map durations to integer counts for storage (including 'us')
            if unit in ("ns", "us", "ms"):
                mapping[name] = pl.Duration(unit)  # type: ignore[arg-type]
        elif base == DataType.TIME:
            mapping[name] = pl.Time()
    return mapping


def build_delta_write_exprs(
    schema: Dict[str, pl.DataType],
    mapping: Optional[Dict[str, pl.DataType]] = None,
) -> list[pl.Expr]:
    """Build per-column expressions to convert logical temporal types to
    Delta-compatible underlying integers for writing.

    - Datetime → epoch counts in declared unit (Int64)
    - Duration → integer counts in declared unit (Int64)
    - Time → nanoseconds since midnight (Int64)
    - Other columns → passthrough or cast to provided dtype
    """
    exprs: list[pl.Expr] = []
    for name_ in schema.keys():
        if not mapping:
            exprs.append(pl.col(name_))
            continue
        dt = mapping.get(name_)
        if dt is None:
            exprs.append(pl.col(name_))
            continue
        if (dt == pl.Datetime) or isinstance(dt, pl.Datetime):
            unit: Literal["ns", "us", "ms"] | None = getattr(dt, "time_unit", None)
            # If microseconds, store as native Delta datetime(us) without mapping
            if unit == "us" or unit is None:
                exprs.append(pl.col(name_).cast(dt, strict=False).alias(name_))
            else:
                expr = pl.col(name_).cast(dt, strict=False).dt.timestamp(unit).cast(pl.Int64).alias(name_)
                exprs.append(expr)
        elif (dt == pl.Duration) or isinstance(dt, pl.Duration):
            # Store as integer counts regardless of unit (including 'us')
            expr = pl.col(name_).cast(dt, strict=False).cast(pl.Int64).alias(name_)
            exprs.append(expr)
        elif (dt == pl.Time) or isinstance(dt, pl.Time):
            expr = pl.col(name_).cast(pl.Time, strict=False).cast(pl.Int64).alias(name_)
            exprs.append(expr)
        else:
            exprs.append(pl.col(name_).cast(dt, strict=False).alias(name_))
    return exprs


def with_mapping_description(opts: Dict[str, Any], mapping: Optional[Dict[str, pl.DataType]]) -> Dict[str, Any]:
    """Return a copy of delta write options with our logical mapping description injected.

    If mapping is None or empty, returns the original options unchanged.
    """
    if not mapping:
        return opts
    new_opts = dict(opts)
    payload = base64.b64encode(mapping_to_metadata(mapping)).decode("ascii")
    new_opts["description"] = inject_description_block(new_opts.get("description"), _MAPPING_BLOCK_TAG, payload)
    return new_opts


def sink_delta(
    lf: pl.LazyFrame,
    target: str | Path,
    *,
    mode: Literal["error", "append", "overwrite", "ignore", "merge"] = "error",
    overwrite_schema: bool | None = None,
    storage_options: Optional[Dict[str, str]] = None,
    credential_provider: Optional[str] = "auto",
    delta_write_options: Optional[Dict[str, Any]] = None,
    delta_merge_options: Optional[Dict[str, Any]] = None,
    translate_logical_types: bool = True,
    chunk_size: int | None = None,
    aws_profile: Optional[str] = None,
) -> Any:
    """
    Write a LazyFrame to a Delta table with logical type translation.

    Polars does not have a ``sink_delta`` function for LazyFrames - only
    ``DataFrame.write_delta`` for eager DataFrames. This function provides
    LazyFrame support and adds transparent handling for logical types that
    Delta Lake does not natively support (``Datetime[ns/ms]``, ``Duration``,
    ``Time``). These types are converted to integers for storage and a mapping
    is embedded in the Delta table metadata for recovery on read via
    ``cpl.scan_delta``.

    Note: ``Datetime[us]`` is natively supported by Delta Lake and is written
    without translation.

    Args:
        lf: The LazyFrame to write.
        target: Path or URI to the Delta table root directory.

            Note: For local filesystem, absolute and relative paths are supported but
            for supported object storages (GCS, Azure, S3) a full URI must be provided.
        mode ({'error', 'append', 'overwrite', 'ignore', 'merge'}): How to handle existing data.

            - ``'error'``: Raise an error if the table exists (default).
            - ``'append'``: Append data to the existing table.
            - ``'overwrite'``: Overwrite the existing table.
            - ``'ignore'``: Do nothing if the table exists.
            - ``'merge'``: Merge data into the existing table (requires ``delta_merge_options``).
        overwrite_schema: If True, allows overwriting the table schema when mode is ``'overwrite'``.

            .. deprecated::
                Use the parameter ``delta_write_options`` instead and pass
                ``{"schema_mode": "overwrite"}``.
        storage_options: Extra options for the storage backends supported by ``deltalake``.
            For cloud storages, this may include configurations for authentication etc.

            - See a list of supported storage options for S3 `here <https://docs.rs/object_store/latest/object_store/aws/enum.AmazonS3ConfigKey.html#variants>`__.
            - See a list of supported storage options for GCS `here <https://docs.rs/object_store/latest/object_store/gcp/enum.GoogleConfigKey.html#variants>`__.
            - See a list of supported storage options for Azure `here <https://docs.rs/object_store/latest/object_store/azure/enum.AzureConfigKey.html#variants>`__.
        credential_provider: Provide a function that can be called to provide cloud storage
            credentials. The function is expected to return a dictionary of
            credential keys along with an optional credential expiry time.
        delta_write_options: Additional keyword arguments passed to ``deltalake.write_deltalake``.
            See supported options `here <https://delta-io.github.io/delta-rs/api/delta_writer/#deltalake.write_deltalake>`__.
        delta_merge_options: Keyword arguments required for ``MERGE`` operations when ``mode='merge'``.
            See supported options `here <https://delta-io.github.io/delta-rs/api/delta_table/#deltalake.DeltaTable.merge>`__.
        translate_logical_types: **polars-io-tools extension.** When ``True`` (default), converts unsupported
            logical types (``Datetime[ns/ms]``, ``Duration``, ``Time``) to integers
            for storage and embeds a mapping in the Delta table metadata. When ``False``,
            writes data as-is without translation (may fail for unsupported types).
        chunk_size: **polars-io-tools extension.** When set, uses ``collect_batches`` to write
            data in chunks of the specified size. This enables streaming writes for
            large datasets. Set to ``-1`` to disable chunking and collect the entire
            frame before writing. Note: chunking is only used for ``mode='append'``,
            ``'error'``, and ``'ignore'``; ``'overwrite'`` and ``'merge'`` always
            collect the full frame first.
        aws_profile: **polars-io-tools extension.** AWS profile name to use for S3 storage options
            when a credential_provider is not explicitly provided. Helps populate
            endpoint URLs and other S3-specific configuration.

    Returns:
        deltalake.TableMerger or None: Returns a ``TableMerger`` for ``mode='merge'`` (to chain merge operations),
            ``None`` otherwise.

    Notes:
        **Why sink_delta?** Polars only provides ``DataFrame.write_delta`` which
        requires collecting the entire LazyFrame into memory first. This function
        accepts LazyFrames directly and can use ``collect_batches`` for streaming
        writes when ``chunk_size`` is set.

        **Logical type mapping**: A mapping is embedded in the Delta table metadata
        description (delimited by ``[cpl.mapping:begin]`` and ``[cpl.mapping:end]``).
        This mapping records which columns were originally ``Datetime[ns/ms]``,
        ``Duration``, or ``Time`` types, allowing ``cpl.scan_delta`` to restore them.

        **Unsupported types in Delta**: The Polars data types ``Null`` and ``Time``
        are not supported by the Delta protocol specification. This function handles
        ``Time`` by converting to nanoseconds since midnight (Int64). ``Null`` columns
        will still raise an error.

    Examples:
        Write a LazyFrame with Datetime[ns] column:

        >>> import polars as pl
        >>> import polars_io_tools as cpl
        >>> from datetime import datetime
        >>> lf = pl.LazyFrame({
        ...     "ts": pl.Series([datetime(2025, 1, 1)]).cast(pl.Datetime("ns")),
        ...     "value": [42],
        ... })
        >>> cpl.sink_delta(lf, "/path/to/delta-table/", mode="overwrite")  # doctest: +SKIP

        Streaming write with chunking:

        >>> cpl.sink_delta(lf, "/path/to/delta-table/", mode="append", chunk_size=10000)  # doctest: +SKIP

        Write to S3 with AWS profile:

        >>> cpl.sink_delta(lf, "s3://bucket/delta-table/", mode="overwrite", aws_profile="my-profile")  # doctest: +SKIP

    See Also:
        polars.DataFrame.write_delta : Polars' native Delta writer for DataFrames
            (https://docs.pola.rs/api/python/stable/reference/api/polars.DataFrame.write_delta.html).
        scan_delta : Read a Delta table with logical type translation.
    """
    # Use Polars logical dtypes only for mapping; no adapters
    mapping: Optional[Dict[str, pl.DataType]]
    schema = lf.collect_schema()
    mapping = infer_logical_mapping(schema) if translate_logical_types else None

    exprs = build_delta_write_exprs(schema, mapping)

    # Build delta write options; we'll use Polars DataFrame.write_delta
    target_str = str(target)

    # Compose storage options (similar to scan_delta)
    if credential_provider not in ["auto", None] and hasattr(credential_provider, "profile_name"):
        aws_profile = getattr(credential_provider, "profile_name", aws_profile)
    discovered = _storage_options_for(target_str, aws_profile=aws_profile).polars
    polars_opts = {**discovered, **(storage_options or {})}

    # Ensure table directory exists if local; write_delta will manage logs/files for local paths
    if not (target_str.startswith("s3://") or target_str.startswith("s3a://")):
        os.makedirs(target_str, exist_ok=True)

    prepared_lf = lf.select(exprs)

    # Overwrite writes as a single batch to avoid stream issues
    if mode == "merge":
        df = prepared_lf.collect()
        # Include description mapping for merge operations as well
        delta_opts = with_mapping_description(delta_write_options or {}, mapping if translate_logical_types else None)
        return df.write_delta(
            target,
            mode=mode,
            storage_options=polars_opts,
            credential_provider=credential_provider,
            delta_write_options=delta_opts,
            delta_merge_options=delta_merge_options or {},
        )
    elif mode == "overwrite":
        batch_df = prepared_lf.collect()
        delta_opts = dict(delta_write_options or {})
        if overwrite_schema:
            delta_opts.setdefault("schema_mode", "overwrite")
        if translate_logical_types:
            delta_opts = with_mapping_description(delta_opts, mapping)
        batch_df.write_delta(
            target,
            mode=mode,
            storage_options=polars_opts,
            credential_provider=credential_provider,
            delta_write_options=delta_opts,
        )
    else:
        base_opts = dict(delta_write_options or {})
        if overwrite_schema:
            base_opts.setdefault("schema_mode", "overwrite")
        first_opts = with_mapping_description(dict(base_opts), mapping if translate_logical_types else None)

        first = True
        if POLARS_HAS_COLLECT_BATCHES and chunk_size != -1:
            for batch_df in prepared_lf.collect_batches(chunk_size=chunk_size, maintain_order=True):
                batch_df.write_delta(
                    target,
                    mode=(mode if first else "append"),
                    storage_options=polars_opts,
                    credential_provider=credential_provider,
                    delta_write_options=(first_opts if first else base_opts),
                )
                first = False
        else:
            # Single-batch non-streaming path; use base_opts prepared above
            df = prepared_lf.collect()
            return df.write_delta(
                target,
                mode=mode,
                storage_options=polars_opts,
                credential_provider=credential_provider,
                delta_write_options=first_opts,
            )


def scan_delta(
    source: Union[str, Path, Any],
    *,
    version: Optional[Union[int, str, datetime]] = None,
    storage_options: Optional[Dict[str, Any]] = None,
    credential_provider: Optional[Union[Literal["auto"], Any]] = "auto",
    delta_table_options: Optional[Dict[str, Any]] = None,
    use_pyarrow: bool = False,
    pyarrow_options: Optional[Dict[str, Any]] = None,
    rechunk: Optional[bool] = None,
    aws_profile: Optional[str] = None,
    pushdown_predicate_deltalake: bool = True,
) -> pl.LazyFrame:
    """
    Lazily read from a Delta lake table with logical type translation.

    This function wraps Polars' ``pl.scan_delta`` and adds support for logical
    types that Delta Lake does not natively support (``Datetime[ns/ms]``,
    ``Duration``, ``Time``). These types are stored as integers and automatically
    cast back to their logical types on read using mapping metadata embedded in
    the Delta table description.

    Args:
        source: DeltaTable or a Path or URI to the root of the Delta lake table.

            Note: For local filesystem, absolute and relative paths are supported but
            for supported object storages (GCS, Azure, S3) a full URI must be provided.

            See: ``pl.scan_delta`` in the Polars documentation.
        version: Numerical version or timestamp version of the Delta lake table.
            If not provided, the latest version is read.

            See: ``pl.scan_delta`` in the Polars documentation.
        storage_options: Extra options for the storage backends supported by ``deltalake``.
            For cloud storages, this may include configurations for authentication etc.

            More info: https://delta-io.github.io/delta-rs/usage/loading-table/

            See: ``pl.scan_delta`` in the Polars documentation.
        credential_provider: Provide a function that can be called to provide cloud storage
            credentials. The function is expected to return a dictionary of
            credential keys along with an optional credential expiry time.

            See: ``pl.scan_delta`` in the Polars documentation.
        delta_table_options: Additional keyword arguments while reading a Delta lake Table.
            Only used when ``pushdown_predicate_deltalake=False``.

            See: ``pl.scan_delta`` in the Polars documentation.
        use_pyarrow: Flag to enable pyarrow dataset reads.
            Only used when ``pushdown_predicate_deltalake=False``.

            See: ``pl.scan_delta`` in the Polars documentation.
        pyarrow_options: Keyword arguments while converting a Delta lake Table to pyarrow table.
            Only used when ``pushdown_predicate_deltalake=False`` and ``use_pyarrow=True``.

            See: ``pl.scan_delta`` in the Polars documentation.
        rechunk: Make sure that all columns are contiguous in memory by
            aggregating the chunks into a single array.

            See: ``pl.scan_delta`` in the Polars documentation.
        aws_profile: **polars-io-tools extension.** AWS profile name to use for S3 storage options
            when a credential_provider is not explicitly provided. Helps populate
            endpoint URLs and other S3-specific configuration.
        pushdown_predicate_deltalake: **polars-io-tools extension.** When ``True`` (default), attempts predicate
            pushdown via deltalake ``file_uris(partition_filters=...)`` to skip
            irrelevant Parquet files based on partition column predicates without
            reading file metadata. This can significantly reduce I/O on partitioned
            tables. When ``False``, falls back to standard ``pl.scan_delta`` behavior
            without partition pruning.

    Returns:
        LazyFrame

    Notes:
        **Logical type mapping**: Tables written with ``cpl.sink_delta`` embed a
        mapping in the Delta table metadata description (delimited by
        ``[cpl.mapping:begin]`` and ``[cpl.mapping:end]``). This mapping records which
        columns were originally ``Datetime[ns/ms]``, ``Duration``, or ``Time`` types.
        On read, these columns are automatically cast from their integer storage back
        to the original logical types.

        If no mapping is present, this function behaves identically to ``pl.scan_delta``.

        **Predicate pushdown**: Predicates on mapped columns are automatically rewritten
        to operate on the underlying integer storage, enabling full predicate pushdown
        through the Parquet reader.

    Examples:
        Basic scan of a Delta table with logical types:

        >>> import polars_io_tools as cpl
        >>> lf = cpl.scan_delta("/path/to/delta-table/")  # doctest: +SKIP
        >>> lf.collect()  # doctest: +SKIP

        Scan with filtering on a mapped Datetime[ns] column:

        >>> from datetime import datetime
        >>> lf = cpl.scan_delta("/path/to/delta-table/")  # doctest: +SKIP
        >>> lf.filter(pl.col("ts") >= datetime(2025, 1, 1)).collect()  # doctest: +SKIP

        Scan from S3 with AWS profile:

        >>> lf = cpl.scan_delta("s3://bucket/delta-table/", aws_profile="my-profile")  # doctest: +SKIP
        >>> lf.collect()  # doctest: +SKIP

    See Also:
        polars.scan_delta : Polars' native Delta scanner
            (https://docs.pola.rs/api/python/stable/reference/api/polars.scan_delta.html).
        sink_delta : Write a LazyFrame to a Delta table with logical type translation.
    """
    # Compose storage options once
    if credential_provider not in ["auto", None] and hasattr(credential_provider, "profile_name"):
        aws_profile = getattr(credential_provider, "profile_name", aws_profile)
    discovered = _storage_options_for(str(source), aws_profile=aws_profile).polars
    polars_opts = {**discovered, **(storage_options or {})}

    # Build DeltaTable to derive mapping + schema, but don't capture it in the closure
    # (DeltaTable contains RawDeltaTable which is not pickleable)
    mapping: Optional[Dict[str, pl.DataType]]
    exposed_schema: Dict[str, pl.DataType]
    from deltalake import DeltaTable  # type: ignore

    # DeltaTable only accepts int version; str/datetime handled via load_as_of
    int_version = version if isinstance(version, int) or version is None else None

    # If source is already a DeltaTable, extract the URI for pickling purposes
    if isinstance(source, DeltaTable):
        source_uri = source.table_uri
        dt = source
    else:
        source_uri = str(source)
        dt = DeltaTable(source_uri, storage_options=polars_opts, version=int_version)

    mapping = _read_mapping_from_delta_configuration(dt, storage_options=None, version=None)
    exposed_schema = _compute_exposed_schema_from_dt(dt, mapping)

    # Don't capture 'dt' in closure - it's not pickleable. Instead capture only
    # pickleable parameters and recreate DeltaTable inside the generator.
    def source_generator(
        with_columns: Optional[list[str]],
        predicate: Optional[pl.Expr],
        n_rows: Optional[int],
        batch_size: Optional[int],
    ):
        # Try Delta partition pruning path
        # There is an open issue for polars to support this natively:
        # https://github.com/pola-rs/polars/issues/23780
        # Namely, utilizing the file level statistics for predicate pushdown.
        # We convert polars predicates to deltalake DNF to filter the file uris
        # to restrict the files polars looks at. When this issue is closed
        # upstream and polars implements this behavior, we can remove this code.

        # Recreate DeltaTable inside generator to avoid pickling issues
        inner_dt = DeltaTable(source_uri, storage_options=polars_opts, version=int_version)

        inner_lf: pl.LazyFrame
        if pushdown_predicate_deltalake:
            # Build partition filters and prune URIs with fallback union if needed
            uris = _get_partition_uris(inner_dt, predicate, mapping)
            inner_lf = _scan_parquet_with_delta_uris(
                inner_dt,
                uris,
                storage_options=polars_opts,
                credential_provider=credential_provider,
                rechunk=rechunk,
            )
        else:
            log.debug("Partition pruning path failed not attempted, utilizing raw deltalake")
            # Fallback: glob parquet files under root, else native delta scan
            inner_lf = pl.scan_delta(
                source_uri,
                version=version,
                storage_options=polars_opts,
                credential_provider=credential_provider,
                delta_table_options=delta_table_options,
                use_pyarrow=use_pyarrow,
                pyarrow_options=pyarrow_options,
                rechunk=rechunk,
            )
        lf = inner_lf
        if not mapping:
            # Simple case, no mapping
            if with_columns:
                lf = lf.select(with_columns)
        else:
            # Rewrite predicate for logical mapping
            pred_rewritten = translate_polars_predicate(predicate, mapping)
            if pred_rewritten is not None:
                log.debug(f"Applying translated predicate: {str(pred_rewritten)}")
                lf = lf.filter(pred_rewritten)

            # Projection and casting of mapped columns
            output_cols = with_columns if with_columns is not None else list(exposed_schema.keys())
            exprs = [(pl.col(name).cast(mapping[name], strict=False).alias(name) if name in mapping else pl.col(name)) for name in output_cols]
            lf = lf.select(exprs)
        # We apply the original predicate in either case
        # to ensure we don't accidentally drop a filter
        if predicate is not None:
            lf = lf.filter(predicate)

        if n_rows is not None:
            lf = lf.limit(n_rows)

        yield from collect_lf_in_io_source(lf, batch_size)

    return register_io_source_with_is_pure(source_generator, schema=exposed_schema)
