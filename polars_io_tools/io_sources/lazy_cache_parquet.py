import datetime
import functools
import logging
import operator
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum, auto
from pathlib import Path, PureWindowsPath
from string import Template
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

if TYPE_CHECKING:
    import pyarrow.fs as pa_fs

import polars as pl
import portion
from polars.exceptions import ComputeError
from portion import Interval
from pydantic import GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

from .._compat import POLARS_HAS_PARTITION_BY
from .dnf_visitor import ColumnConstraintAnalyzer, DNFClause, convert_expr_to_dnf
from .range_visitor import _convert_atomic_interval_to_polars_expr, convert_expr_to_datetime_range
from .restrict_visitor import restrict_expr_to_columns
from .util import _is_in_collection, _storage_options_for, collect_lf_in_io_source, partition_exclusion_predicate, register_io_source_with_is_pure


@dataclass(frozen=True)
class WritePlan:
    should_write: bool
    filter_predicate: pl.Expr | None
    write_empty_missing: bool


@dataclass(frozen=True)
class ReadPlan:
    """Plan for reading from cache after writes complete.

    use_paths: Explicit list of cache file paths to read, or None to use glob pattern.
    """

    use_paths: list[str] | None


log = logging.getLogger(__name__)

__all__ = ("CacheMode", "cache_parquet")


def _path_as_file_uri(path: Path | PureWindowsPath) -> str:
    path_str = str(path)
    if len(path_str) >= 3 and path_str[1] == ":" and path_str[2] in {"/", "\\"}:
        return PureWindowsPath(path_str).as_uri()
    return Path(path_str).as_uri()


def _prepare_lf_for_sink_from_io_source(lf: pl.LazyFrame) -> pl.LazyFrame:
    plan = lf.explain(optimized=False)
    has_python_scan = "PYTHON SCAN" in plan or "PYTHON[" in plan
    if pl.thread_pool_size() > 1 and not has_python_scan:
        return lf
    # Avoid re-entering partitioned sink execution from an io_source callback when
    # Polars has only one Rayon worker, or when the source plan itself contains a
    # Python io_source. This path is less memory efficient, but it prevents a
    # scheduler deadlock for chained cache_parquet calls.
    return lf.collect().lazy()


# Constants

# Polars' default cap on how many partition sinks may be open at once (the
# POLARS_MAX_OPEN_SINKS default). Writes are checked against this cap.
_DEFAULT_MAX_OPEN_SINKS = 128
_MAX_ENUMERATED_SCAN_PATHS = 64


def _max_open_sinks() -> int:
    """Return the open-sink cap polars will enforce (POLARS_MAX_OPEN_SINKS, or the default)."""
    raw = os.environ.get("POLARS_MAX_OPEN_SINKS")
    if raw is None:
        return _DEFAULT_MAX_OPEN_SINKS
    try:
        return int(raw)
    except ValueError:
        return _DEFAULT_MAX_OPEN_SINKS


def _dataframe_write_parquet_kwargs(
    *,
    write_kwargs: dict,
    metadata: dict[str, str],
    storage_options: dict,
    credential_provider: pl.CredentialProviderAWS | None,
) -> dict:
    parquet_kwargs = dict(write_kwargs)
    parquet_kwargs.pop("metadata", None)
    parquet_kwargs.pop("lazy", None)
    parquet_kwargs.pop("maintain_order", None)
    parquet_kwargs["metadata"] = metadata
    parquet_kwargs["mkdir"] = True
    if storage_options:
        parquet_kwargs["storage_options"] = storage_options
    if credential_provider:
        parquet_kwargs["credential_provider"] = credential_provider
    return parquet_kwargs


def _write_empty_parquet_files_sequentially(
    paths: list[str],
    *,
    schema: pl.Schema,
    metadata: dict[str, str],
    storage_options: dict,
    credential_provider: pl.CredentialProviderAWS | None,
    write_kwargs: dict,
) -> None:
    empty_df = pl.DataFrame(schema=schema)
    parquet_kwargs = _dataframe_write_parquet_kwargs(
        write_kwargs=write_kwargs,
        metadata=metadata,
        storage_options=storage_options,
        credential_provider=credential_provider,
    )
    for path in paths:
        empty_df.write_parquet(path, **parquet_kwargs)


class CacheMode(Enum):
    """Enum to represent handling of caches.

    Supports instantiation from string names (e.g., "CACHE", "REBUILD") or integer values
    when used as a pydantic field, in addition to the standard enum member access.
    """

    CACHE = auto()  # Normal caching behavior - read/write, keep existing objects
    IGNORE = auto()  # Bypass cache entirely, always generate fresh data
    REBUILD = auto()  # Ignore cache on reads, query upstream, and write fresh data (overwriting existing)

    @classmethod
    def validate(cls, v) -> "CacheMode":
        """Validate and convert input to CacheMode enum.

        Accepts:
        - CacheMode instances (returned as-is)
        - Strings matching enum member names (e.g., "CACHE", "REBUILD")
        - Integers matching enum values
        """
        if isinstance(v, cls):
            return v
        elif isinstance(v, str):
            return cls[v]
        elif isinstance(v, int):
            return cls(v)
        raise ValueError(f"Cannot convert value to CacheMode: {v}")

    @staticmethod
    def _serialize(value: "CacheMode") -> str:
        """Serialize CacheMode to its string name for JSON output."""
        return value.name

    @classmethod
    def __get_pydantic_json_schema__(cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        """Generate JSON schema for CacheMode enum."""
        field_schema = handler(core_schema.str_schema())
        field_schema.update(
            type="string",
            title=cls.__name__,
            description=cls.__doc__ or "An enumeration of cache modes",
            enum=list(cls.__members__.keys()),
        )
        return field_schema

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: Callable[[Any], core_schema.CoreSchema],
    ) -> core_schema.CoreSchema:
        """Generate pydantic core schema for CacheMode validation and serialization."""
        return core_schema.no_info_before_validator_function(
            cls.validate,
            core_schema.any_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize, info_arg=False, return_schema=core_schema.str_schema(), when_used="json"
            ),
        )


class Enumerability(Enum):
    """Classification of partition expectation enumerability.

    - FINITE: finite set of concrete partition keys can be enumerated.
    - UNBOUNDED: cannot enumerate (e.g., one-sided/unbounded date predicate).
    - UNCONSTRAINED: exactly one row with all-null partition keys (no restrictions).
    """

    FINITE = auto()
    UNBOUNDED = auto()
    UNCONSTRAINED = auto()


@dataclass(frozen=True)
class PartitionInfo:
    """Summarizes partition planning inputs for read/write decisions."""

    join_cols: list[str]
    expected_parts_df: pl.DataFrame
    existing_parts_df: pl.DataFrame
    enumerability: Enumerability
    # For unbounded queries, this holds the expected partitions clipped to existing data bounds
    clipped_expected_parts_df: pl.DataFrame | None = None


def _is_all_null_row(df: pl.DataFrame) -> bool:
    """Return True if DataFrame is exactly one row and all values are None."""
    if df.height != 1:
        return False
    row = df.row(0, named=True)
    return all(v is None for v in row.values())


def _classify_enumerability(expected_parts_df: pl.DataFrame, date_column: str | None) -> Enumerability:
    """Classify the expected partitions enumerability given the date column context."""
    if date_column and expected_parts_df.is_empty():
        # With a date column, empty expected set implies unbounded/one-sided
        return Enumerability.UNBOUNDED
    if _is_all_null_row(expected_parts_df):
        return Enumerability.UNCONSTRAINED
    return Enumerability.FINITE


def _build_partition_info(
    *,
    expected_parts_df: pl.DataFrame,
    existing_parts_df: pl.DataFrame,
    date_column: str | None,
    extra_cols: list[str],
    predicate: pl.Expr | None = None,
    time_unit: Literal["daily", "monthly", "yearly"] | None = None,
    schema: pl.Schema | None = None,
) -> PartitionInfo:
    """Create a PartitionInfo snapshot to drive downstream planning decisions."""
    join_cols = extra_cols + ([date_column] if date_column else [])
    enumerability = _classify_enumerability(expected_parts_df, date_column)

    # For unbounded queries with a date column, compute clipped expected partitions
    clipped_expected_parts_df = None

    if enumerability == Enumerability.UNBOUNDED and date_column and time_unit and schema and predicate is not None:
        # Use the range visitor to extract the date interval from the predicate
        date_interval = convert_expr_to_datetime_range(predicate, date_column, get_enclosure=True)
        clipped_expected_parts_df = _compute_clipped_expected_partitions(
            predicate_interval=date_interval,
            existing_parts_df=existing_parts_df,
            date_column=date_column,
            extra_cols=extra_cols,
            time_unit=time_unit,
            schema=schema,
        )
        if clipped_expected_parts_df is not None:
            log.debug(
                "Computed %d clipped expected partitions for unbounded query (interval: %s)",
                len(clipped_expected_parts_df),
                date_interval,
            )

    return PartitionInfo(
        join_cols=join_cols,
        expected_parts_df=expected_parts_df,
        existing_parts_df=existing_parts_df,
        enumerability=enumerability,
        clipped_expected_parts_df=clipped_expected_parts_df,
    )


# Partition format specifications
_PARTITION_FORMATS: dict[str, dict[str, Any]] = {
    "daily": {"default": "$year-$month-$day", "required": {"year", "month", "day"}},
    "monthly": {"default": "$year-$month", "required": {"year", "month"}},
    "yearly": {"default": "$year", "required": {"year"}},
}

_STRFTIME_MAP = {"year": "%Y", "month": "%m", "day": "%d"}


def _strftime_from_template(tmpl: str) -> str:
    """Convert a partition template with $year/$month/$day placeholders to strftime format."""
    fmt = tmpl
    for placeholder, code in _STRFTIME_MAP.items():
        fmt = fmt.replace(f"${placeholder}", code)
    return fmt


def _build_scan_paths(
    parts_df: pl.DataFrame,
    time_unit_dir: str,
    template_for_metadata: str | None,
    effective_time_unit: Literal["daily", "monthly", "yearly", "null"],
    date_column: str | None,
    extra_cols: list[str],
) -> list[str]:
    """Build explicit cache file paths from an expected partitions DataFrame.

    - Returns [] if `parts_df` is empty or any join column has None (cannot enumerate).
    - Sorts rows by join columns for deterministic order.
    - Uses custom template or default format to render date strings.
    """
    if parts_df.is_empty():
        return []

    join_cols = extra_cols + ([date_column] if date_column else [])

    # If any join column has None, we cannot enumerate finite paths
    if join_cols and parts_df.select(join_cols).null_count().sum_horizontal().item() > 0:
        return []

    date_fmt = None
    if date_column:
        date_fmt = _strftime_from_template(template_for_metadata or _PARTITION_FORMATS[effective_time_unit]["default"])

    keys = _partition_keys(parts_df, extra_cols, date_column, date_fmt)
    return [f"{time_unit_dir}/{key}.parquet" for key in sorted(keys.unique())]


def _build_not_existing_partitions_pred(
    existing_parts_df: pl.DataFrame,
    join_cols: list[str],
    date_column: str | None = None,
    time_unit: Literal["daily", "monthly", "yearly", "null"] | None = None,
    date_dtype: pl.DataType | None = None,
) -> pl.Expr | None:
    """Keep rows outside cached partitions, or return None when there is nothing to exclude.

    Date keys exclude their whole calendar period, including for timezone-aware columns.
    """
    if (
        date_column is None
        or date_column not in join_cols
        or time_unit not in _PARTITION_FORMATS
        or date_dtype is None
        or existing_parts_df.is_empty()
    ):
        return partition_exclusion_predicate(existing_parts_df, join_cols)

    extra_cols = [c for c in join_cols if c != date_column]
    date_col = pl.col(date_column)
    row_key = _partition_date_expr(date_column, date_dtype, time_unit)
    keyed = existing_parts_df.with_columns(_stored_key_date_expr(date_column, existing_parts_df.schema[date_column]))

    # A row is cached when its (combination, partition date) pair is. Group the pairs by combination (one `is_in` per distinct date set) or by
    # date (one term per date), whichever gives fewer terms. Grouping by combination alone degrades when combinations cover different dates.
    combos_by_dates: dict[frozenset[date], list[tuple]] = {}
    combos_by_date: dict[date, list[tuple]] = {}
    null_date_combos: list[tuple] = []
    groups = keyed.group_by(extra_cols, maintain_order=True) if extra_cols else [((), keyed)]
    for key, grp in groups:
        combo = tuple(key) if extra_cols else ()
        dates = frozenset(grp[date_column].drop_nulls().to_list())
        if dates:
            combos_by_dates.setdefault(dates, []).append(combo)
        for d in dates:
            combos_by_date.setdefault(d, []).append(combo)
        if grp[date_column].null_count():
            null_date_combos.append(combo)

    def outside_combos(combos: list[tuple]) -> pl.Expr | None:
        if not extra_cols:
            return None
        return partition_exclusion_predicate(pl.DataFrame(combos, schema=existing_parts_df.select(extra_cols).schema, orient="row"), extra_cols)

    if len(combos_by_dates) <= len(combos_by_date):
        keep_groups = [(~row_key.is_in(sorted(dates)) | date_col.is_null(), combos) for dates, combos in combos_by_dates.items()]
    else:
        keep_groups = [((row_key != d) | date_col.is_null(), combos) for d, combos in combos_by_date.items()]
    if null_date_combos:
        keep_groups.append((date_col.is_not_null(), null_date_combos))
    keep_exprs: list[pl.Expr] = []
    for keep, combos in keep_groups:
        others = outside_combos(combos)
        keep_exprs.append(keep if others is None else others | keep)
    if not keep_exprs:
        return None
    return keep_exprs[0] if len(keep_exprs) == 1 else pl.Expr.and_(*keep_exprs)


def _partition_date_expr(date_column: str, dtype: pl.DataType, time_unit: Literal["daily", "monthly", "yearly"]) -> pl.Expr:
    """Expression for the calendar date (period start) of the partition holding each row.

    Matches the writer, which formats the column with ``dt.strftime`` (local time for timezone-aware columns).
    """
    expr = pl.col(date_column)
    if dtype != pl.Date:
        expr = expr.dt.date()
    if time_unit == "monthly":
        expr = expr.dt.truncate("1mo")
    elif time_unit == "yearly":
        expr = expr.dt.truncate("1y")
    return expr


def _stored_key_date_expr(date_column: str, dtype: pl.DataType) -> pl.Expr:
    """Calendar date of a partition key held in a partitions frame.

    Partition frames hold key dates cast to the column dtype, which places them at midnight UTC for timezone-aware columns.
    """
    expr = pl.col(date_column)
    if isinstance(dtype, pl.Datetime):
        if dtype.time_zone is not None:
            expr = expr.dt.convert_time_zone("UTC")
        expr = expr.dt.date()
    return expr


def _partition_key_component(value: Any) -> str:
    """Serialize one partition key value as the ``PartitionBy`` writer does for its path segment."""
    return unquote(str(value))


def _serialize_key_values(values: pl.Series) -> pl.Series:
    """Serialize non-null partition values as the writer of the installed polars does."""
    if POLARS_HAS_PARTITION_BY:
        return pl.Series([_partition_key_component(v) for v in values.to_list()], dtype=pl.String)
    # PartitionByKey (polars < 1.38) names partitions by the value cast to String
    return values.cast(pl.String)


def _partition_keys(parts_df: pl.DataFrame, extra_cols: list[str], date_column: str | None, date_fmt: str | None) -> pl.Series:
    """Cache keys (``a/b/2024-01``, no extension) for each row of a partitions frame.

    Uses the writer's serialization so keys compare equal to listed file keys. Rows with a null (unconstrained) partition value get a null key.
    """
    exprs = []
    for c in extra_cols:
        values = parts_df[c].drop_nulls().unique()
        exprs.append(pl.col(c).replace_strict(values, _serialize_key_values(values), return_dtype=pl.String))
    if date_column is not None and date_fmt is not None:
        exprs.append(_stored_key_date_expr(date_column, parts_df.schema[date_column]).dt.strftime(date_fmt))
    if not exprs:
        return pl.Series("key", [None] * parts_df.height, dtype=pl.String)
    return parts_df.select(pl.concat_str(exprs, separator="/").alias("key")).to_series()


def _extract_existing_files(fs: "pa_fs.FileSystem", fs_path_prefix: str) -> set[str]:
    """List partition files under a prefix and return their relative keys.

    - For local filesystems, walks `fs_path_prefix` and collects `*.parquet` paths.
    - For S3, creates the directory if missing and lists files recursively.
    - Returns partition keys without the file extension and without the prefix.

    This is used by the cache planner to understand what partitions already exist
    without reading all data.
    """
    import pyarrow.fs as pa_fs

    existing_parts: set[str] = set()

    if isinstance(fs, pa_fs.LocalFileSystem):
        path = Path(fs_path_prefix)
        if path.exists():
            for p in path.glob("**/*.parquet"):
                existing_parts.add(p.relative_to(path).with_suffix("").as_posix())
    else:
        # S3FileSystem is optional in pyarrow; use getattr for safe access
        S3FileSystem = getattr(pa_fs, "S3FileSystem", None)
        if S3FileSystem is not None and isinstance(fs, S3FileSystem):
            try:
                fs.create_dir(fs_path_prefix + "/")
            except Exception as e:  # noqa: BLE001 -- intentional broad catch (defensive fallback)
                log.debug(f"Failed to create S3 directory {fs_path_prefix}: {e}")

        try:
            for info in fs.get_file_info(pa_fs.FileSelector(fs_path_prefix, recursive=True)):
                if info.type == pa_fs.FileType.File and info.path.endswith(".parquet"):
                    rel = Path(info.path[len(fs_path_prefix) + 1 :])
                    existing_parts.add(rel.with_suffix("").as_posix())
        except Exception:  # noqa: BLE001,S110 -- intentional broad catch; best-effort cleanup, failure is non-fatal
            # Directory doesn't exist yet
            pass

    return existing_parts


def _split_custom_partition_keys(
    existing_files: set[str],
    all_partition_cols: list[str],
    date_column: str,
) -> pl.DataFrame:
    """Return raw extra-column and date keys from custom-format paths.

    Extra keys may be plain or ``key=value`` segments. The date key contains all remaining segments.
    """
    extra_cols = [col for col in all_partition_cols if col != date_column]
    n_extra = len(extra_cols)

    rows = []
    for file_path in existing_files:
        path_parts = file_path.split("/")
        if len(path_parts) <= n_extra:
            continue
        extras = [part.split("=", 1)[1] if "=" in part else part for part in path_parts[:n_extra]]
        rows.append([*extras, "/".join(path_parts[n_extra:])])

    return pl.DataFrame(rows, schema={**{c: pl.Utf8 for c in extra_cols}, date_column: pl.Utf8}, orient="row")


# Null partition values: "None" from PartitionBy (polars >= 1.38), "null" from PartitionByKey
_NULL_KEY_TOKENS = ("None", "null")
_BOOL_KEYS = {"True": True, "true": True, "False": False, "false": False}


def _key_decoder(dtype: pl.DataType) -> Callable[[str], Any] | None:
    """Parser for a partition key of ``dtype``, or None to fall back to a polars cast."""
    if dtype == pl.Boolean:
        return _BOOL_KEYS.__getitem__
    if dtype.is_integer():
        return int
    if dtype.is_float():
        return float
    if dtype == pl.Date:
        return datetime.date.fromisoformat
    if isinstance(dtype, pl.Datetime):
        return datetime.datetime.fromisoformat
    if dtype == pl.Time:
        return datetime.time.fromisoformat
    return None


def _parse_date_keys(df: pl.DataFrame, date_column: str, template: str) -> pl.DataFrame:
    """Parse the date key column with ``template``; drop rows whose key does not match it."""
    raw = pl.col(date_column)
    parsed = pl.when(raw.is_in(_NULL_KEY_TOKENS)).then(None).otherwise(raw.str.strptime(pl.Date, _strftime_from_template(template), strict=False))
    df = df.with_columns(parsed.alias("__piot_parsed__"))
    malformed = pl.col("__piot_parsed__").is_null() & pl.col(date_column).is_not_null() & ~pl.col(date_column).is_in(_NULL_KEY_TOKENS)
    n_malformed = df.select(malformed.sum()).item()
    if n_malformed:
        log.debug("Ignoring %d cached file(s) not matching partition format %r", n_malformed, template)
        df = df.filter(~malformed)
    return df.with_columns(pl.col("__piot_parsed__").alias(date_column)).drop("__piot_parsed__")


def _cast_partition_keys(df: pl.DataFrame, partition_schema_dict: dict) -> pl.DataFrame:
    """Decode string partition keys to the partition schema.

    Accepts the key spellings of both polars writers. Rows whose key does not decode are dropped, so an unrecognized file is treated as not
    cached rather than as the null partition.
    """
    undecoded = pl.Series(values=[False] * df.height, dtype=pl.Boolean)
    for c, dtype in partition_schema_dict.items():
        if df.schema[c] != pl.Utf8:
            continue
        if dtype == pl.Utf8 or dtype == pl.Categorical:
            df = df.with_columns(pl.when(~pl.col(c).is_in(_NULL_KEY_TOKENS)).then(pl.col(c)).cast(dtype).alias(c))
            continue
        raw = df[c]
        decode = _key_decoder(dtype)
        if decode is None:
            decoded = raw.cast(dtype, strict=False)
        else:
            mapping: dict[str, Any] = {}
            for key in raw.drop_nulls().unique().to_list():
                if key in _NULL_KEY_TOKENS:
                    continue
                try:
                    mapping[key] = decode(key)
                except (ValueError, KeyError, ArithmeticError):
                    pass
            decoded = pl.Series(c, [mapping.get(key) for key in raw.to_list()], dtype=dtype, strict=False)
        undecoded = undecoded | (raw.is_not_null() & ~raw.is_in(_NULL_KEY_TOKENS) & decoded.is_null())
        df = df.with_columns(decoded.alias(c))
    if undecoded.any():
        log.debug("Ignoring %d cached file(s) whose partition keys do not match the schema", undecoded.sum())
        df = df.filter(~undecoded)
    return df


def _construct_partitions_df_from_existing(
    existing_files: set[str],
    schema: pl.Schema,
    all_partition_cols: list[str],
    template_for_metadata: str | None,
    date_column: str | None = None,
    time_unit: Literal["daily", "monthly", "yearly"] | None = None,
) -> pl.DataFrame:
    """Decode existing file keys to the partition schema, omitting malformed keys."""
    partition_schema_dict = {k: schema[k] for k in all_partition_cols}

    if not existing_files:
        return pl.DataFrame(schema=partition_schema_dict)

    is_custom_format = (
        date_column
        and template_for_metadata
        and time_unit in _PARTITION_FORMATS
        and template_for_metadata != _PARTITION_FORMATS[time_unit]["default"]
    )
    if is_custom_format:
        df = _split_custom_partition_keys(existing_files, all_partition_cols, date_column)
    else:
        raw_data = [part.split("/") for part in existing_files]
        df = pl.DataFrame(raw_data, schema={k: pl.Utf8 for k in all_partition_cols}, orient="row")

    if date_column and template_for_metadata and (is_custom_format or date_column in all_partition_cols):
        df = _parse_date_keys(df, date_column, template_for_metadata)
    if is_custom_format:
        df = df.select(all_partition_cols)
    return _cast_partition_keys(df, partition_schema_dict)


def _get_expected_partitions_df(
    pred: pl.Expr | None,
    date_column: str | None,
    extra_cols: list[str],
    time_unit: Literal["daily", "monthly", "yearly", "null"],
    schema: pl.Schema,
) -> pl.DataFrame:
    """Compute expected partitions that match a predicate, using DNF enumeration.

    Restricts the predicate to partition columns, enumerates combinations using DNF,
    and returns a DataFrame of candidate partitions to read from cache.
    """
    all_partition_cols = extra_cols + ([date_column] if date_column else [])

    # Handle single partition case (no partition columns)
    # Represent as a single-row, all-null placeholder to indicate
    # an unconstrained single partition.
    if not all_partition_cols:
        return pl.DataFrame([{}])

    partition_schema = {col: schema[col] for col in all_partition_cols}
    # Restrict predicate to partition columns only
    partition_cols_set = set(all_partition_cols)
    restricted_pred = restrict_expr_to_columns(pred, partition_cols_set) if pred is not None else None

    if restricted_pred is None:
        # Predicate does not involve partition columns. Same as no predicate.
        return pl.DataFrame([dict.fromkeys(all_partition_cols)], schema=partition_schema)

    # Always use DNF enumeration, but allow None placeholders for unconstrained columns
    # time_unit "null" is only used when date_column is None, in which case we don't enumerate dates
    effective_time_unit: Literal["daily", "monthly", "yearly"] = time_unit if time_unit != "null" else "daily"
    result_df = _enumerate_partitions_from_dnf(restricted_pred, date_column, extra_cols, effective_time_unit, schema)

    log.debug("Enumerated %d partition combinations", len(result_df))
    return result_df


def _enumerate_partitions_from_dnf(
    pred: pl.Expr,
    date_column: str | None,
    extra_cols: list[str],
    time_unit: Literal["daily", "monthly", "yearly"],
    schema: pl.Schema,
) -> pl.DataFrame:
    """Unified DNF enumeration that handles partial constraints with None placeholders."""
    all_partition_cols = extra_cols + ([date_column] if date_column else [])
    partition_schema = {col: schema[col] for col in all_partition_cols}

    try:
        dnf_clauses = convert_expr_to_dnf(pred)
        if not dnf_clauses:
            return pl.DataFrame([dict.fromkeys(all_partition_cols)], schema=partition_schema)

        clause_dfs = []
        for i, clause in enumerate(dnf_clauses):
            clause_df = _process_dnf_clause_to_partitions(clause, date_column, extra_cols, time_unit, schema)

            if clause_df is not None and not clause_df.is_empty():
                clause_dfs.append(clause_df)

        if not clause_dfs:
            # All clauses were contradictions - return empty result
            return pl.DataFrame(schema=partition_schema)

        # Union all clause results and deduplicate
        return pl.concat(clause_dfs).unique()

    except Exception as e:  # noqa: BLE001 -- intentional broad catch (defensive fallback)
        log.debug(f"DNF enumeration failed: {e}, returning unconstrained placeholder")
        # Even on failure, return a DataFrame with None placeholders rather than failing
        return pl.DataFrame([dict.fromkeys(all_partition_cols)], schema=partition_schema)


def _process_dnf_clause_to_partitions(
    clause: DNFClause,
    date_column: str | None,
    extra_cols: list[str],
    time_unit: Literal["daily", "monthly", "yearly"],
    schema: pl.Schema,
) -> pl.DataFrame | None:
    """Process a single DNF clause, using None for unconstrained columns."""
    all_partition_cols = extra_cols + ([date_column] if date_column else [])
    partition_schema = {c: schema[c] for c in all_partition_cols}

    # Group constraints by column
    constraints_by_col = {}
    for col, op, val in clause:
        if col in all_partition_cols:
            if col not in constraints_by_col:
                constraints_by_col[col] = []
            constraints_by_col[col].append((op, val))

    # Process each partition column - use None if unconstrained
    column_value_sets = {}

    for col in all_partition_cols:
        constraints = constraints_by_col.get(col, [])

        matches_null = any(
            op in ("is", "==", "in") and (val is None or (isinstance(val, (list, set, tuple)) and None in val)) for op, val in constraints
        )
        if not constraints or matches_null:
            # Unconstrained, or matching null (which the None placeholder cannot distinguish from unconstrained)
            column_value_sets[col] = {None}
        elif date_column and col == date_column:
            values = _get_date_partitions_from_constraints(constraints, date_column, time_unit, schema)
            if not values:
                # Contradiction in date constraints
                return pl.DataFrame(schema=partition_schema)
            column_value_sets[col] = values
        else:
            values = _get_column_values_from_constraints(constraints, col, schema)
            if values is None:
                # Non-enumerable extra-column constraints use wildcard reads without a date column,
                # or the unbounded/date-clipping fallback when a date column is present.
                if date_column is None:
                    column_value_sets[col] = {None}
                else:
                    return pl.DataFrame(schema=partition_schema)
            elif not values:
                # Contradiction in column constraints
                return pl.DataFrame(schema=partition_schema)
            else:
                column_value_sets[col] = values

    # Generate cross product of all column values (including None placeholders)
    import itertools

    all_combinations = itertools.product(*[column_value_sets[col] for col in all_partition_cols])

    return pl.DataFrame(list(all_combinations), schema=partition_schema, orient="row")


def _get_date_partitions_from_constraints(
    constraints: list[tuple[str, Any]],
    date_column: str,
    time_unit: Literal["daily", "monthly", "yearly"],
    schema: pl.Schema,
) -> set[date]:
    """Return widened date keys, or an empty set when no finite dates can be enumerated."""
    if not constraints:
        return set()

    analyzer = ColumnConstraintAnalyzer(date_column)
    for op, val in constraints:
        analyzer.update_from_predicate(op, val)

    if analyzer.has_contradiction(schema=schema):
        return set()

    # `!=` only rules out a whole partition when a partition is exactly one Date value.
    exact_exclusions = time_unit == "daily" and schema[date_column] == pl.Date
    dates = _extract_and_widen_dates(analyzer, time_unit, exact_exclusions=exact_exclusions)
    return set(pl.Series(date_column, dates, dtype=schema[date_column]).to_list())


def _get_column_values_from_constraints(
    constraints: list[tuple[str, Any]],
    column: str,
    schema: pl.Schema,
) -> set[Any] | None:
    """Return finite matching values, an empty set for a contradiction, or None for a non-enumerable constraint."""
    if not constraints:
        return set()

    analyzer = ColumnConstraintAnalyzer(column)
    for op, val in constraints:
        analyzer.update_from_predicate(op, val)

    if analyzer.has_contradiction(schema=schema):
        return set()

    values = _extract_finite_values(analyzer, column, schema)
    return None if values is None else set(pl.Series(column, list(values), dtype=schema[column]).to_list())


def _extract_and_widen_dates(
    analyzer: ColumnConstraintAnalyzer,
    time_unit: Literal["daily", "monthly", "yearly"],
    *,
    exact_exclusions: bool = False,
) -> list[date]:
    """Extract valid dates from analyzer and apply time unit widening.

    Excluded (``!=``) values remove a partition only when ``exact_exclusions`` is set, i.e. when each partition holds a single date value;
    otherwise the partition still has other matching rows.
    """
    dates = set()

    # Handle exact values
    if analyzer.exact_values:
        for val in analyzer.exact_values:
            date_val = _convert_to_date(val)
            if date_val:
                dates.add(_widen_date(date_val, time_unit))

    # Handle inclusion set
    if analyzer.inclusion_set is not None:
        for val in analyzer.inclusion_set:
            date_val = _convert_to_date(val)
            if date_val:
                dates.add(_widen_date(date_val, time_unit))

    # Handle date ranges
    if analyzer.min_bound or analyzer.max_bound:
        range_dates = _enumerate_date_range(analyzer, time_unit)
        dates.update(range_dates)

    # Remove excluded dates
    if exact_exclusions and analyzer.exclusion_values:
        for val in analyzer.exclusion_values:
            if type(val) is date:
                dates.discard(val)

    return sorted(dates)


def _convert_to_date(val: Any) -> date | None:
    """Convert various date-like values to date objects."""
    if isinstance(val, datetime.datetime):
        return val.date()
    elif isinstance(val, date):
        return val
    return None


def _widen_date(date_val: date, time_unit: Literal["daily", "monthly", "yearly"]) -> date:
    """Apply widening to a date based on the partitioning time unit."""
    if time_unit == "daily":
        return date_val
    elif time_unit == "monthly":
        return date_val.replace(day=1)
    elif time_unit == "yearly":
        return date_val.replace(month=1, day=1)
    else:
        return date_val


def _enumerate_date_range(
    analyzer: ColumnConstraintAnalyzer,
    time_unit: Literal["daily", "monthly", "yearly"],
) -> set[date]:
    """Enumerate date range from bounds, applying widening."""
    dates = set()

    min_val = max_val = None
    if analyzer.min_bound:
        min_val, min_inclusive = analyzer.min_bound
        min_date = _convert_to_date(min_val)
        if min_date and not min_inclusive:
            min_date = min_date + timedelta(days=1)
        min_val = min_date

    if analyzer.max_bound:
        max_val, max_inclusive = analyzer.max_bound
        max_date = _convert_to_date(max_val)
        if max_date and not max_inclusive:
            max_date = max_date - timedelta(days=1)
        max_val = max_date

    if not min_val or not max_val:
        log.warning(f"One of: Maximumum value: {max_val} and Minimum value: {min_val} is unbounded.")
        return set()

    # Generate partition keys with widening
    current = _widen_date(min_val, time_unit)
    max_widened = _widen_date(max_val, time_unit)

    while current <= max_widened:
        dates.add(current)
        current = _step_date(current, time_unit)

    return dates


def _step_date(d: date, time_unit: Literal["daily", "monthly", "yearly"]) -> date:
    """Step to next partition boundary."""
    if time_unit == "daily":
        return d + timedelta(days=1)
    elif time_unit == "monthly":
        if d.month == 12:
            return d.replace(year=d.year + 1, month=1)
        else:
            return d.replace(month=d.month + 1)
    else:  # yearly
        return d.replace(year=d.year + 1)


def _extract_finite_values(
    analyzer: ColumnConstraintAnalyzer,
    column: str,
    schema: pl.Schema,
) -> set[Any] | None:
    """Extract finite set of valid values, or None if not possible."""
    if analyzer.exact_values:
        return analyzer.exact_values - analyzer.exclusion_values

    if analyzer.inclusion_set is not None:
        return analyzer.inclusion_set - analyzer.exclusion_values

    # Bounded integer ranges can be enumerated.
    if analyzer.min_bound and analyzer.max_bound:
        col_type = schema.get(column)
        if col_type and col_type.is_integer():
            return _enumerate_integer_range(analyzer)

    return None


def _enumerate_integer_range(analyzer: ColumnConstraintAnalyzer) -> set[int] | None:
    """Enumerate a bounded integer range, omitting excluded values."""
    if not analyzer.min_bound or not analyzer.max_bound:
        return None
    min_val, min_inclusive = analyzer.min_bound
    max_val, max_inclusive = analyzer.max_bound

    if not min_inclusive:
        min_val += 1
    if not max_inclusive:
        max_val -= 1

    values = set(range(min_val, max_val + 1))
    return values - analyzer.exclusion_values


def _compute_clipped_expected_partitions(
    predicate_interval: Interval,
    existing_parts_df: pl.DataFrame,
    date_column: str,
    extra_cols: list[str],
    time_unit: Literal["daily", "monthly", "yearly"],
    schema: pl.Schema,
) -> pl.DataFrame | None:
    """Return partitions for a one-sided date interval clipped to existing cache bounds.

    Return None when the interval is not one-sided, no cached dates exist, or the clipped range is empty.
    """
    if predicate_interval.empty:
        return None

    # Check if fully bounded (both sides finite) or fully unbounded (both sides infinite)
    is_unbounded_below = predicate_interval.lower == -portion.inf
    is_unbounded_above = predicate_interval.upper == portion.inf

    if not is_unbounded_below and not is_unbounded_above:
        # Fully bounded - use normal enumeration
        return None
    if is_unbounded_below and is_unbounded_above:
        # Fully unbounded - can't clip
        return None

    if existing_parts_df.is_empty() or date_column not in existing_parts_df.columns:
        return None

    existing_dates = existing_parts_df[date_column].drop_nulls()
    if existing_dates.is_empty():
        return None

    existing_bound = _convert_to_date(existing_dates.max() if is_unbounded_above else existing_dates.min())
    if existing_bound is None:
        return None
    predicate_bound = predicate_interval.lower if is_unbounded_above else predicate_interval.upper
    if isinstance(predicate_bound, datetime.datetime):
        predicate_bound = predicate_bound.date()
    lower, upper = (predicate_bound, existing_bound) if is_unbounded_above else (existing_bound, predicate_bound)
    clipped_min = _widen_date(lower, time_unit)
    clipped_max = _widen_date(upper, time_unit)

    # Don't enumerate if clipped range is invalid
    if clipped_min > clipped_max:
        return None

    # Enumerate the clipped date range
    dates: set[date] = set()
    current = clipped_min
    while current <= clipped_max:
        dates.add(current)
        current = _step_date(current, time_unit)

    dates_df = pl.DataFrame({date_column: sorted(dates)}, schema={date_column: schema[date_column]})
    if not extra_cols:
        return dates_df
    all_partition_cols = extra_cols + [date_column]
    partition_schema = {col: schema[col] for col in all_partition_cols}
    return existing_parts_df.select(extra_cols).unique().join(dates_df, how="cross").select(all_partition_cols).cast(partition_schema)


def _generate_write_predicate_from_partitions_df(
    partitions_df: pl.DataFrame,
    date_column: str | None,
    extra_cols: list[str],
    time_unit: Literal["daily", "monthly", "yearly", "null"],
) -> pl.Expr | None:
    """Return a predicate covering the target partitions, or None when unrestricted.

    Within merged date ranges, extra-column values may cover more combinations than the target partitions.
    """
    if partitions_df.is_empty():
        return None

    # A single row with all-nulls for partition keys means "no predicate".
    if _is_all_null_row(partitions_df):
        return None

    if not date_column:
        all_extra_predicates = []
        for row_dict in partitions_df.to_dicts():
            row_predicates = [pl.col(col) == row_dict[col] for col in extra_cols if row_dict.get(col) is not None]
            if row_predicates:
                all_extra_predicates.append(pl.Expr.and_(*row_predicates))
        return pl.Expr.or_(*all_extra_predicates) if all_extra_predicates else None

    partition_intervals: list[tuple[Interval, dict[str, Any]]] = []
    for row_dict in partitions_df.to_dicts():
        date_val = row_dict.get(date_column)
        if date_val is None:
            interval = portion.open(-portion.inf, portion.inf)

        else:
            if time_unit not in _PARTITION_FORMATS:
                raise ValueError(f"Unsupported time_unit: {time_unit}")
            start_date = _widen_date(_convert_to_date(date_val), time_unit)
            interval = portion.closedopen(start_date, _step_date(start_date, time_unit))
        extra_values = {col: row_dict[col] for col in extra_cols if row_dict.get(col) is not None}
        partition_intervals.append((interval, extra_values))

    merged_groups: list[tuple[Interval, list[dict[str, Any]]]] = []
    partition_intervals.sort(key=lambda x: (x[0].lower, x[0].upper))
    current_interval = partition_intervals[0][0]
    current_partitions = [partition_intervals[0][1]]

    for interval, partition_values in partition_intervals[1:]:
        if current_interval.overlaps(interval) or current_interval.adjacent(interval):
            current_interval = current_interval | interval
            current_partitions.append(partition_values)
        else:
            merged_groups.append((current_interval, current_partitions))
            current_interval = interval
            current_partitions = [partition_values]

    merged_groups.append((current_interval, current_partitions))
    group_predicates = []

    for merged_interval, partition_list in merged_groups:
        cur_date_expr = _convert_atomic_interval_to_polars_expr(merged_interval, date_column)
        group_preds = [cur_date_expr] if cur_date_expr is not None else []
        for col in extra_cols:
            if any(col not in partition_values for partition_values in partition_list):
                continue
            values = {partition_values[col] for partition_values in partition_list}
            if len(values) == 1:
                group_preds.append(pl.col(col) == next(iter(values)))
            else:
                group_preds.append(pl.col(col).is_in(_is_in_collection(list(values), partitions_df.schema[col])))
        if group_preds:
            group_predicates.append(pl.Expr.and_(*group_preds))
    return pl.Expr.or_(*group_predicates) if group_predicates else None


def _schema_from_cache(
    cache_path: str,
    storage_opts: dict | None,
    credential_provider: pl.CredentialProviderAWS | None,
) -> pl.Schema | None:
    """Get the schema from an existing cache."""
    uri = f"{cache_path.rstrip('/')}/**/*.parquet"
    try:
        kwargs = {"storage_options": storage_opts} if storage_opts else {}
        if credential_provider:
            kwargs["credential_provider"] = credential_provider

        lf = pl.scan_parquet(uri, n_rows=0, **kwargs)
        return lf.collect_schema()
    except Exception:  # noqa: BLE001 -- intentional broad catch (defensive fallback)
        log.debug("Could not infer schema from cache; this is the first write")
        return None


def _normalise_partition_format(time_unit: Literal["daily", "monthly", "yearly"], partition_format: str | None) -> tuple[str, str]:
    """Return the normalized strftime format and template for metadata."""
    spec = _PARTITION_FORMATS[time_unit]

    if partition_format is None:
        partition_format = spec["default"]

    tmpl = Template(partition_format)
    try:
        placeholders = set(tmpl.get_identifiers())
    except ValueError as exc:
        raise ValueError(f"Invalid partition_format: {exc}") from None

    if placeholders != spec["required"]:
        missing = spec["required"] - placeholders
        extra = placeholders - spec["required"]
        msgs: list[str] = []
        if missing:
            msgs.append("missing " + ", ".join(f"${m}" for m in sorted(missing)))
        if extra:
            msgs.append("unexpected " + ", ".join(f"${e}" for e in sorted(extra)))
        raise ValueError("Partition format error - " + "; ".join(msgs))

    return _strftime_from_template(partition_format), tmpl.template


def _build_write_plan(
    *,
    cache_mode: CacheMode,
    partition_info: PartitionInfo,
    date_column: str | None,
    predicate: pl.Expr | None,
    effective_time_unit: Literal["daily", "monthly", "yearly", "null"],
    date_dtype: pl.DataType | None = None,
) -> WritePlan:
    """Construct a write plan based on cache mode and enumerability.

    Consolidates special-case logic for one-sided/unbounded queries and unconstrained cases.
    """
    need_write = cache_mode in {CacheMode.CACHE, CacheMode.REBUILD}

    if not need_write:
        return WritePlan(False, None, False)

    if cache_mode == CacheMode.REBUILD:
        return WritePlan(True, None, True)

    enum = partition_info.enumerability

    if enum == Enumerability.UNCONSTRAINED:
        # If we have partition columns, always query upstream (new partitions might exist)
        # Filter out rows that would fall into already-cached partitions
        if partition_info.join_cols:
            not_existing = _build_not_existing_partitions_pred(
                partition_info.existing_parts_df, partition_info.join_cols, date_column, effective_time_unit, date_dtype
            )
            return WritePlan(True, not_existing, False)
        # No partition columns - single global partition, trust cache once written
        if partition_info.existing_parts_df.is_empty():
            return WritePlan(True, None, False)
        return WritePlan(False, None, False)

    if enum == Enumerability.UNBOUNDED:
        restricted = (
            restrict_expr_to_columns(predicate, set(partition_info.join_cols)) if predicate is not None and partition_info.join_cols else None
        )
        not_existing = _build_not_existing_partitions_pred(
            partition_info.existing_parts_df, partition_info.join_cols, date_column, effective_time_unit, date_dtype
        )
        if restricted is not None and not_existing is not None:
            combined = restricted & not_existing
        else:
            combined = restricted if restricted is not None else not_existing
        return WritePlan(True, combined, False)

    return WritePlan(True, None, True)


def _cache_scan_from_read_plan(
    read_plan: ReadPlan,
    time_unit_dir: str,
    scan_kwargs: dict,
    schema: pl.Schema,
) -> pl.LazyFrame:
    """Create a cache scan LazyFrame according to the read plan, with logging.

    - If `use_paths` is provided, log and scan enumerated paths (or return empty LF if list is empty).
    - Otherwise, log and scan using a glob.
    """
    if read_plan.use_paths is not None:
        log.debug("Cache scan uses %d enumerated path(s): %s", len(read_plan.use_paths), read_plan.use_paths)
        if len(read_plan.use_paths) == 0:
            return pl.LazyFrame(schema=schema)
        return pl.scan_parquet(read_plan.use_paths, **scan_kwargs)
    else:
        log.debug("Cache scan uses glob: %s/**/*.parquet", time_unit_dir)
        return pl.scan_parquet(f"{time_unit_dir}/**/*.parquet", **scan_kwargs)


def _get_fs_path_directory_info(
    cache_uri: str,
    time_unit: Literal["daily", "monthly", "yearly", "null"],
) -> tuple[str, str]:
    """Returns the filesystem path prefix and the time unit directory for the given cache URI.

    returns:
        tuple(str, str): (fs_path_prefix, time_unit_dir)
    """
    parsed = urlparse(cache_uri)
    is_s3 = parsed.scheme in {"s3", "s3a"}

    if not is_s3:
        # Handle file:// URIs - extract the path component
        if parsed.scheme == "file":
            local_path = url2pathname(unquote(parsed.path))
        else:
            local_path = cache_uri
        local_time_unit_path = Path(local_path).expanduser().resolve() / time_unit
        fs_path_prefix = str(local_time_unit_path)
        # Use file:// prefix to route local paths through object-store,
        # which provides atomic writes automatically (polars >= 1.33.1)
        time_unit_dir = _path_as_file_uri(local_time_unit_path)
        return fs_path_prefix, time_unit_dir

    bucket = parsed.netloc
    # Normalize prefix to avoid accidental double slashes when composing keys
    # Only strip trailing slashes; preserve leading slash as-is
    prefix = parsed.path.rstrip("/")
    base_key = f"{prefix}/{time_unit}" if prefix else time_unit

    # Compose without introducing an extra slash when prefix has a single leading '/'
    # Preserve multiple leading slashes if present in prefix
    if base_key.startswith("/"):
        fs_path_prefix = f"{bucket}{base_key}"
    else:
        fs_path_prefix = f"{bucket}/{base_key}"
    query = f"?{parsed.query}" if parsed.query else ""
    time_unit_dir = f"s3://{fs_path_prefix}{query}"

    log.debug("Using S3 cache path: %s", time_unit_dir)
    return fs_path_prefix, time_unit_dir


def _build_read_plan(
    *,
    partition_info: PartitionInfo,
    date_column: str | None,
    predicate: pl.Expr | None,
    time_unit_dir: str,
    template_for_metadata: str | None,
    effective_time_unit: Literal["daily", "monthly", "yearly", "null"],
    has_cached_files: bool,
) -> ReadPlan:
    """Construct a read plan for reading from cache.

    After writes complete, we just read from cache - no upstream merge needed.

    Returns a ReadPlan with:
    - use_paths: explicit cache file paths (None = use glob pattern)
    """
    enum = partition_info.enumerability
    join_cols = partition_info.join_cols
    extra_cols = [c for c in join_cols if c != (date_column or "")]

    # No cached files exist yet: read nothing rather than globbing an empty directory (which
    # raises "expanded paths were empty"). This happens when a wildcard/unconstrained read wrote
    # no partitions, e.g. its predicate matched no upstream rows.
    if not has_cached_files:
        return ReadPlan(use_paths=[])

    def build_cache_paths(parts_df: pl.DataFrame) -> list[str] | None:
        return _build_scan_paths(
            parts_df=parts_df,
            time_unit_dir=time_unit_dir,
            template_for_metadata=template_for_metadata,
            effective_time_unit=effective_time_unit,
            date_column=date_column,
            extra_cols=extra_cols,
        )

    # For unbounded/unconstrained queries, use glob pattern (can't enumerate all paths)
    if enum in (Enumerability.UNBOUNDED, Enumerability.UNCONSTRAINED):
        return ReadPlan(use_paths=None)

    # For finite queries, enumerate paths from expected partitions
    expected_df = partition_info.expected_parts_df
    if expected_df.is_empty():
        return ReadPlan(use_paths=[])

    # If expected_df has NULL values in partition columns, we can't enumerate paths
    # Fall back to glob in that case
    has_null_partition_values = expected_df.select(join_cols).null_count().sum_horizontal().item() > 0
    if has_null_partition_values:
        return ReadPlan(use_paths=None)

    # Build explicit paths when we have a predicate (for partition pruning)
    use_paths = build_cache_paths(expected_df) if predicate is not None else None
    if use_paths is not None and len(use_paths) > _MAX_ENUMERATED_SCAN_PATHS:
        log.debug(
            "Cache scan has %d enumerated path(s), exceeding limit %d; using glob",
            len(use_paths),
            _MAX_ENUMERATED_SCAN_PATHS,
        )
        use_paths = None
    return ReadPlan(use_paths=use_paths)


def _compute_partitions_to_write(part_info: PartitionInfo, existing_keys: set[str], date_column: str | None, date_fmt: str | None) -> pl.DataFrame:
    """Expected partitions whose cache file is not already present.

    Rows with an unconstrained (null) partition value are always kept.
    """
    expected = part_info.expected_parts_df
    if not existing_keys or not part_info.join_cols or expected.is_empty():
        return expected
    extra_cols = [c for c in part_info.join_cols if c != date_column]
    keys = _partition_keys(expected, extra_cols, date_column, date_fmt)
    return expected.filter(pl.Series([k is None or k not in existing_keys for k in keys], dtype=pl.Boolean))


def _build_final_scan(
    cache_scan: pl.LazyFrame,
    predicate: pl.Expr | None,
    with_columns: list[str] | None,
    n_rows: int | None,
) -> pl.LazyFrame:
    """Build the final scan LazyFrame from cache."""
    scan = cache_scan
    if predicate is not None:
        scan = scan.filter(predicate)
    if with_columns is not None:
        scan = scan.select(with_columns)
    if n_rows is not None:
        scan = scan.limit(n_rows)
    return scan


def _build_cache_metadata(
    effective_time_unit: str,
    template_for_metadata: str | None,
    extra_cols: list[str],
    user_metadata: dict | None = None,
) -> dict[str, str]:
    """Build metadata dictionary for cache parquet files."""
    metadata = {
        "__piot__time_unit": effective_time_unit,
    }

    # Add partition format if we have a date column with formatting
    metadata["__piot__partition_format"] = template_for_metadata if template_for_metadata else ""

    # Add extra partition columns info
    metadata["__piot__partitioned_by"] = ",".join(extra_cols) if extra_cols else ""

    # Merge user-provided metadata
    if user_metadata:
        metadata.update(user_metadata)

    return metadata


def cache_parquet(
    self_or_fn: pl.LazyFrame | Callable[[], pl.LazyFrame],
    cache_path: str | Path,
    date_column: str | None = None,
    *,
    time_unit: Literal["daily", "monthly", "yearly"] = "monthly",
    partition_format: str | None = None,
    cache_mode: CacheMode = CacheMode.CACHE,
    aws_profile: str | None = None,
    write_kwargs: dict | None = None,
    read_kwargs: dict | None = None,
    extra_partition_cols: str | list[str] | None = None,
    schema: pl.Schema | None = None,
    write_bounding_columns: list[str] | None = None,
    description: str | None = None,
) -> pl.LazyFrame:
    """
    Cache a LazyFrame to Parquet files with optional date-based partitioning. Supports daily, monthly, or yearly
    date partitioning with customizable formats, or simple partitioning by other columns.

    **When date_column is provided:**

    - You can customize the partition format using ``partition_format`` with ``$year``, ``$month``, and ``$day`` placeholders
    - The placeholders must match the granularity of the time unit:

      - For daily partitioning, use ``$year``, ``$month``, and ``$day``.
      - For monthly partitioning, use ``$year`` and ``$month``.
      - For yearly partitioning, use only ``$year``.

    - On-disk Parquet filenames inherit the highest-resolution temporal unit from the provided partition format

    **When date_column is None:**

    - Only extra_partition_cols are used for partitioning
    - time_unit and partition_format are ignored
    - Files are organized by the values in extra_partition_cols
    - To match behavior with date_column specified, when no date_column is specified,
      the files will live in a subdirectory entitled "null"

    .. warning::

       You are **strongly** discouraged from mixing multiple time units in the same cache directory,
       although it is technically permissible.

    Args:
        self (pl.LazyFrame): The LazyFrame to cache.
        cache_path (str | pathlib.Path): Root directory where Parquet files are stored.
        date_column (str, optional): Name of the column that contains dates or datetimes. If None, only extra_partition_cols
            will be used for partitioning (no date-based organization).
        extra_partition_cols (str | list[str], optional): Additional columns to partition by.
            Duration and nanosecond Datetime columns are not supported.
        time_unit ({"daily", "monthly", "yearly"}, default "monthly"): Partitioning granularity.
        partition_format (str, optional): Format using ``$year``, ``$month``, ``$day`` placeholders. If None, defaults based on time_unit:

            - ``$year-$month-$day`` for daily
            - ``$year-$month`` for monthly
            - ``$year`` for yearly
        cache_mode (CacheMode, default CacheMode.CACHE): Determines the caching behavior:

            - ``CacheMode.CACHE``: Normal caching behavior (read from cache if available, write missing partitions)
            - ``CacheMode.IGNORE``: Bypass the cache completely (always query upstream, don't read or write cache)
            - ``CacheMode.REBUILD``: Refresh the cache - ignores existing cache on reads, queries upstream for all
              data matching the predicate, and writes fresh data to cache (overwriting existing partition files).
              Unlike deleting and rebuilding, this preserves partitions outside the current query scope.
        aws_profile (str, optional): AWS profile name for S3 access. If None, uses the AWS_PROFILE environment variable.
            This is only relevant if the cache_path is an S3 URI.
        write_kwargs (dict, optional): Additional keyword arguments for writing Parquet files. Internally, this is passed
            to the ``.sink_parquet()`` function; see the Polars documentation for valid kwargs.
        read_kwargs (dict, optional): Additional keyword arguments for reading Parquet files. Internally, this is passed
            to the ``.scan_parquet()`` function; see the Polars documentation for valid kwargs.
        write_bounding_columns (list[str], optional): Opt-in predicate-aware write. When set, the whitelisted subset of
            the pushed predicate (restricted to these columns) is applied to the data written to cache, so the cache is
            bounded to that predicate. Such caches are **predicate-scoped**: a partition holds only the rows matching the
            predicate that wrote it. Because ``CacheMode.CACHE`` skips partitions that already exist on disk (keyed on
            partition columns, not content), a bounded cache is typically regenerated with ``cache_mode=CacheMode.REBUILD``
            so each run overwrites with the current predicate's rows; under ``CacheMode.CACHE`` a partition first written
            under one predicate is not re-written for a wider one. Default None leaves write behavior unchanged.
        description: Optional free-form description of this source instance, attached to its OpenTelemetry span (``explain_detail``).

    Returns:
        pl.LazyFrame: If the cache has all data: a LazyFrame reading from the cache.
            Otherwise: the original LazyFrame with data written to cache.

    Notes:
        Null partition values are cached in their own partition. String partition columns must not contain the values ``"None"`` or
        ``"null"``, which are read back as null.

        **Query Classification (Enumerability)**

        Queries are classified based on whether partition keys can be enumerated from the predicate:

        - **FINITE**: Partition keys can be fully enumerated (e.g., ``date.is_between(start, end)``).
          Writes missing partitions, creates empty files for gaps, reads from enumerated paths.
          A predicate that constrains a partition column but cannot be enumerated to a finite set of
          values (e.g. ``!=`` or an unbounded non-date range) leaves that column unconstrained: the
          read falls back to a glob and applies the full predicate, and (under ``CacheMode.CACHE``)
          the write queries upstream by the predicate while excluding already-cached partitions.
        - **UNBOUNDED**: One-sided date bounds (e.g., ``date >= X`` or ``date <= Y``).
          Always queries upstream filtering out existing partitions. Gap-filling is clipped to
          existing cache bounds (see below).
        - **UNCONSTRAINED**: No partition-constraining predicate. Behavior depends on partition columns:

          +-------------------------------+----------------------------------------------------------+
          | Scenario                      | Behavior                                                 |
          +===============================+==========================================================+
          | With partition columns        | Always queries upstream, filters out existing partitions |
          | (date_column or extra cols)   | (new partition values might exist)                       |
          +-------------------------------+----------------------------------------------------------+
          | No partition columns          | Trusts cache once written (single global partition)      |
          +-------------------------------+----------------------------------------------------------+

        **Unbounded Range Gap-Filling**

        For one-sided queries, empty partition files are written only within the bounds of existing
        cached data:

        - ``date >= X`` (unbounded above): Fills gaps from X up to max(existing_partitions), not beyond.
        - ``date <= Y`` (unbounded below): Fills gaps from min(existing_partitions) down to Y, not before.

        This prevents creating empty files for future dates that may have data arriving later.

        **Data Arrival Assumption**: This assumes data arrives monotonically - once data exists for
        date X, all dates before X are complete. Gaps within existing bounds are treated as truly
        empty (no data existed), not as "data arriving later." If your data source can have late-
        arriving data for dates within the existing range, use ``cache_mode=CacheMode.REBUILD`` to
        refresh the cache.

        **Write-Then-Read Architecture**

        ``cache_parquet`` follows a simple pattern: write missing partitions to cache, then read
        entirely from cache. There is no merging of upstream and cache data during reads.

        **Other Notes**

        - Extra partition columns: filters on extra columns are honored alongside the date predicate.
          Upstream filtering is restricted to partition columns, fetching only missing partition keys.
        - Path selection optimization: when partitions can be enumerated, explicit paths are read
          instead of a glob, reducing metadata scans. Falls back to glob when enumeration isn't possible.
        - Partition completeness assumption: if a partition has any data in cache, it's assumed complete
          for the purpose of unbounded queries.
        - Open-sink cap: a single write opens one streaming sink per partition. Polars caps the number
          of simultaneously open sinks (``POLARS_MAX_OPEN_SINKS``, default 128); exceeding it would
          silently drop rows, so ``cache_parquet`` raises instead. To write more partitions in one call,
          set ``POLARS_MAX_OPEN_SINKS`` above the partition count *before the process starts*, or reduce
          the partition count per call (e.g. a coarser ``time_unit`` or chunked date ranges).
    """
    if cache_mode == CacheMode.IGNORE:
        return self_or_fn() if callable(self_or_fn) else self_or_fn

    # Validate arguments
    write_kwargs = write_kwargs or {}
    read_kwargs = read_kwargs or {}

    for kwarg_dict, name in [(read_kwargs, "read_kwargs"), (write_kwargs, "write_kwargs")]:
        if "storage_options" in kwarg_dict:
            raise ValueError(f"`{name}` should not contain 'storage_options'; we infer these for you")

    import pyarrow.fs as pa_fs

    cache_uri = str(cache_path)
    # Use "null" as time_unit when no date column to maintain consistent structure
    effective_time_unit = time_unit if date_column else "null"

    fs_path_prefix, time_unit_dir = _get_fs_path_directory_info(cache_uri, effective_time_unit)

    if time_unit_dir.startswith("s3://"):
        pyarrow_opts, polars_opts, credential_provider = _storage_options_for(cache_uri, aws_profile=aws_profile)
        # S3FileSystem is optional in pyarrow; use getattr for safe access
        S3FileSystem = getattr(pa_fs, "S3FileSystem", None)
        if S3FileSystem is None:
            raise ImportError("S3 support requires pyarrow to be built with S3 filesystem support")
        fs = S3FileSystem(**pyarrow_opts)
    else:
        pyarrow_opts, polars_opts, credential_provider = {}, {}, None
        fs = pa_fs.LocalFileSystem()

    # Lazily materialize source once (if callable) and reuse
    source_lf: pl.LazyFrame | None
    if isinstance(self_or_fn, pl.LazyFrame):
        source_lf = self_or_fn
    else:
        source_lf = None

    def get_source() -> pl.LazyFrame:
        nonlocal source_lf
        if source_lf is None:
            # self_or_fn is a Callable when source_lf is None (checked above)
            source_lf = self_or_fn()  # type: ignore[operator]
        return source_lf

    # Resolve schema with minimal cost: prefer provided, then cache, then source (callable only once)
    schema = schema or _schema_from_cache(cache_uri, polars_opts, credential_provider)
    if schema is None:
        source_lf = get_source()
        schema = source_lf.collect_schema()

    # Validate that date_column exists in schema if provided
    if date_column is not None and date_column not in schema.names():
        raise ValueError(f"date_column '{date_column}' not present in dataframe")

    # Process extra partition columns
    if extra_partition_cols is None:
        _extra_cols: list[str] = []
    elif isinstance(extra_partition_cols, str):
        _extra_cols = [extra_partition_cols]
    else:
        _extra_cols = list(extra_partition_cols)

    _extra_cols = [c for c in dict.fromkeys(_extra_cols) if c != date_column]
    for c in _extra_cols:
        if c not in schema.names():
            raise ValueError(f"extra_partition_cols: column {c} not present in dataframe")
        if schema[c].base_type() == pl.Duration or (isinstance(schema[c], pl.Datetime) and schema[c].time_unit == "ns"):
            raise ValueError(f"extra_partition_cols: cannot partition on Duration or nanosecond Datetime column {c}")

    if time_unit not in {"daily", "monthly", "yearly"}:
        raise ValueError(f"Invalid time unit {time_unit}; must be one of {{'daily','monthly','yearly'}}")

    # Only normalize partition format if we have a date column
    if date_column is not None:
        partition_format, template_for_metadata = _normalise_partition_format(time_unit, partition_format)
    else:
        partition_format, template_for_metadata = None, None

    # Virtual columns for partitioning
    key_exprs: list[pl.Expr] = []
    for col in _extra_cols:
        key_exprs.append(pl.col(col).alias(f"__piot_key_{col}__"))

    # Only add date key if date_column is provided
    if date_column is not None and partition_format is not None:
        DATE_KEY_ALIAS = "__piot_key_date__"
        key_exprs.append(pl.col(date_column).dt.strftime(partition_format).alias(DATE_KEY_ALIAS))
    else:
        DATE_KEY_ALIAS = None

    # Handle degenerate case: no partition columns at all
    if not key_exprs:
        # Create a single partition with a constant key
        key_exprs = [pl.lit("data").alias("__piot_key_single__")]

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        log.debug("Running with %s against location %s with columns '%s' and predicate: '%s'", cache_mode, time_unit_dir, with_columns, predicate)

        existing_parts = _extract_existing_files(fs, fs_path_prefix)
        all_partition_cols = _extra_cols + ([date_column] if date_column else [])

        existing_parts_df = _construct_partitions_df_from_existing(
            existing_files=existing_parts,
            schema=schema,
            all_partition_cols=all_partition_cols,
            template_for_metadata=template_for_metadata,
            date_column=date_column,
            time_unit=time_unit,
        )

        expected_parts_df = _get_expected_partitions_df(predicate, date_column, _extra_cols, effective_time_unit, schema)
        part_info = _build_partition_info(
            expected_parts_df=expected_parts_df,
            existing_parts_df=existing_parts_df,
            date_column=date_column,
            extra_cols=_extra_cols,
            predicate=predicate,
            time_unit=time_unit if date_column else None,
            schema=schema,
        )

        write_plan = _build_write_plan(
            cache_mode=cache_mode,
            partition_info=part_info,
            date_column=date_column,
            predicate=predicate,
            effective_time_unit=effective_time_unit,
            date_dtype=schema[date_column] if date_column else None,
        )

        written_parts: set[str] = set()
        is_finite = part_info.enumerability == Enumerability.FINITE
        # For REBUILD mode, always write all expected partitions (overwriting existing)
        # For CACHE mode, only write partitions that don't already exist
        if cache_mode == CacheMode.REBUILD:
            partitions_to_write_df = part_info.expected_parts_df
        elif is_finite:
            partitions_to_write_df = _compute_partitions_to_write(part_info, existing_parts, date_column, partition_format)
        else:
            partitions_to_write_df = part_info.expected_parts_df
        do_write = write_plan.should_write and not (is_finite and partitions_to_write_df.is_empty())

        log.debug(
            "Expected %d partitions, %d existing, writing %d (do_write=%s)",
            len(part_info.expected_parts_df),
            len(existing_parts_df),
            len(partitions_to_write_df),
            do_write,
        )

        if do_write:
            lf_to_write = source_lf if source_lf is not None else get_source()
            join_cols = part_info.join_cols
            has_wildcard = (
                bool(join_cols)
                and not partitions_to_write_df.is_empty()
                and partitions_to_write_df.select(join_cols).null_count().sum_horizontal().item() > 0
            )
            if date_column is None and has_wildcard and cache_mode in (CacheMode.CACHE, CacheMode.REBUILD):
                # Without date partitioning, wildcard targets use the partition-restricted query.
                # CACHE skips existing partitions; REBUILD refreshes only partitions in scope.
                restricted = restrict_expr_to_columns(predicate, set(join_cols)) if predicate is not None else None
                write_parts = [restricted] if restricted is not None else []
                if cache_mode == CacheMode.CACHE:
                    not_existing = _build_not_existing_partitions_pred(existing_parts_df, join_cols)
                    if not_existing is not None:
                        write_parts.append(not_existing)
                if write_parts:
                    lf_to_write = lf_to_write.filter(functools.reduce(operator.and_, write_parts))
            elif not partitions_to_write_df.is_empty():
                # Narrow the write to only the partitions that are actually missing
                extra_cols = [c for c in join_cols if c != (date_column or "")]
                write_filter = _generate_write_predicate_from_partitions_df(
                    partitions_df=partitions_to_write_df,
                    date_column=date_column,
                    extra_cols=extra_cols,
                    time_unit=effective_time_unit,
                )
                if cache_mode == CacheMode.CACHE:
                    # The filter above can cover cached partitions (merged date ranges, unconstrained placeholder columns), so exclude them
                    not_existing = _build_not_existing_partitions_pred(
                        existing_parts_df, join_cols, date_column, effective_time_unit, schema[date_column] if date_column else None
                    )
                    if not_existing is not None:
                        write_filter = not_existing if write_filter is None else write_filter & not_existing
                if write_filter is not None:
                    lf_to_write = lf_to_write.filter(write_filter)
            else:
                if write_plan.filter_predicate is not None:
                    lf_to_write = lf_to_write.filter(write_plan.filter_predicate)

            if write_bounding_columns and predicate is not None:
                bounding_pred = restrict_expr_to_columns(predicate, set(write_bounding_columns))
                if bounding_pred is not None:
                    lf_to_write = lf_to_write.filter(bounding_pred)

            # Track written partition keys via file_path callback
            tracked_keys: set[str] = set()

            def _file_path_callback(ctx):
                """Generate file path for each partition and track written keys."""
                # Handle API differences between PartitionBy (new) and PartitionByKey (old)
                if hasattr(ctx, "partition_keys"):
                    # index_in_partition > 0 means polars reopened this sink after
                    # eviction; the reopened sink would overwrite the partition's
                    # earlier file, so fail here instead of losing rows. Raise (not
                    # assert) so it survives `python -O`.
                    if ctx.index_in_partition != 0:
                        raise RuntimeError(
                            "cache_parquet: polars reopened a partition sink "
                            f"(index_in_partition={ctx.index_in_partition}) for key "
                            f"{ctx.partition_keys.row(0)!r}. The open-sink cap "
                            f"({_max_open_sinks()}) was exceeded and rows would be "
                            "silently dropped. Set POLARS_MAX_OPEN_SINKS above the "
                            "partition count before the process starts, or write fewer "
                            "partitions per call."
                        )
                    key_values = [_partition_key_component(v) for v in ctx.partition_keys.row(0)]
                else:
                    # PartitionByKey: ctx.keys is a list of key objects with .str_value
                    key_values = [unquote(k.str_value) for k in ctx.keys]

                # Track the key path
                key_path = "/".join(key_values)
                tracked_keys.add(key_path)

                # Build file path: all but last as directories, last with .parquet
                res_parts = key_values[:-1]
                res_parts.append(f"{key_values[-1]}.parquet")
                return "/".join(res_parts)

            # Write to cache using appropriate partition API
            # Note: Local filesystem paths use file:// prefix for atomic writes via object-store
            if POLARS_HAS_PARTITION_BY:
                partition_obj = pl.PartitionBy(
                    base_path=time_unit_dir,
                    key=key_exprs,
                    file_path_provider=_file_path_callback,
                    include_key=False,
                    # Disable size-based splitting so each partition is one file
                    # (index_in_partition stays 0); otherwise a large partition could be
                    # split, breaking the one-file-per-partition layout the reader expects.
                    approximate_bytes_per_file=None,
                )
            else:
                partition_obj = pl.PartitionByKey(
                    base_path=time_unit_dir,
                    by=key_exprs,
                    include_key=False,
                    file_path=_file_path_callback,
                )

            cache_metadata = _build_cache_metadata(
                effective_time_unit=effective_time_unit,
                template_for_metadata=template_for_metadata,
                extra_cols=_extra_cols,
                user_metadata=write_kwargs.get("metadata"),
            )
            # Above the open-sink cap, polars evicts and reopens partition sinks, and the
            # custom file_path_provider's reopened sink overwrites the partition's earlier
            # file, silently dropping rows. Reject here when the partition count is known
            # up front; the callback's index_in_partition check is the runtime backstop for
            # cases (e.g. unbounded queries) where it isn't.
            n_partitions = len(partitions_to_write_df)
            max_open_sinks = _max_open_sinks()
            if n_partitions > max_open_sinks:
                raise ValueError(
                    f"cache_parquet would write {n_partitions} partitions in a single "
                    f"streaming sink, exceeding the polars open-sink cap of "
                    f"{max_open_sinks}. Exceeding the cap silently drops rows. Set the "
                    "POLARS_MAX_OPEN_SINKS environment variable above the partition "
                    "count BEFORE the process starts, or write fewer partitions per "
                    "call (e.g. a coarser time_unit or chunked date ranges)."
                )

            sink_write_kwargs = dict(write_kwargs)
            sink_write_kwargs.pop("metadata", None)
            _prepare_lf_for_sink_from_io_source(lf_to_write).sink_parquet(  # type: ignore[call-overload]
                partition_obj,
                maintain_order=True,
                mkdir=True,
                metadata=cache_metadata,
                storage_options=polars_opts,
                **({"credential_provider": credential_provider} if credential_provider else {}),
                **sink_write_kwargs,
            )

            # Update written_parts from tracked keys
            written_parts.update(tracked_keys)
            log.debug("Wrote %i new partition(s): %s", len(written_parts), sorted(written_parts))

            # REBUILD refreshes every in-scope partition, so a partition that existed but was not
            # re-emitted upstream is now stale; remove its file so the wildcard glob scan does not
            # keep returning it. Only the wildcard path needs this: the enumerable path writes
            # empty files for missing partitions below, and CACHE never removes anything.
            if cache_mode == CacheMode.REBUILD and has_wildcard and date_column is None and not existing_parts_df.is_empty():
                restricted = restrict_expr_to_columns(predicate, set(join_cols)) if predicate is not None else None
                in_scope = existing_parts_df.drop_nulls()
                if restricted is not None:
                    in_scope = in_scope.filter(restricted)
                if not in_scope.is_empty():
                    in_scope_keys = set(_partition_keys(in_scope, _extra_cols, date_column, partition_format).drop_nulls())
                    for stale_key in sorted(in_scope_keys - written_parts):
                        try:
                            fs.delete_file(f"{fs_path_prefix}/{stale_key}.parquet")
                            log.debug("Removed stale partition on REBUILD: %s", stale_key)
                        except Exception as e:  # noqa: BLE001 -- best-effort cleanup, non-fatal
                            log.warning("Failed to remove stale partition %s on REBUILD: %s", stale_key, e)

            missing = None
            # For bounded ranges (FINITE), optionally write empty missing partitions
            # For unbounded ranges with clipped bounds, also write empty partitions within the clipped range
            should_write_empty = (write_plan.write_empty_missing and part_info.enumerability == Enumerability.FINITE) or (
                part_info.enumerability == Enumerability.UNBOUNDED and part_info.clipped_expected_parts_df is not None
            )
            if should_write_empty:
                # Choose the appropriate expected partitions DataFrame
                if part_info.enumerability == Enumerability.UNBOUNDED and part_info.clipped_expected_parts_df is not None:
                    # Use clipped partitions for unbounded queries
                    source_expected_df = part_info.clipped_expected_parts_df
                    log.debug(
                        "Using %d clipped expected partitions for empty file writing (unbounded query)",
                        len(source_expected_df),
                    )
                else:
                    # Use normal expected partitions for finite queries
                    source_expected_df = part_info.expected_parts_df

                # write missing partitions
                # Null expected keys are wildcard placeholders, so their empty paths cannot be enumerated.
                nulls_dropped_expected_df = source_expected_df.drop_nulls()
                if not nulls_dropped_expected_df.is_empty():
                    expected_keys = set(_partition_keys(nulls_dropped_expected_df, _extra_cols, date_column, partition_format).drop_nulls())
                    # For REBUILD mode, don't subtract existing_parts - we want to overwrite
                    # For CACHE mode, skip partitions that already exist
                    if cache_mode == CacheMode.REBUILD:
                        missing = expected_keys - written_parts
                    else:
                        missing = expected_keys - existing_parts - written_parts
                if missing:
                    sorted_keys = sorted(missing)

                    # Precompute metadata and common kwargs once
                    built_metadata = _build_cache_metadata(
                        effective_time_unit=effective_time_unit,
                        template_for_metadata=template_for_metadata,
                        extra_cols=_extra_cols,
                        user_metadata=write_kwargs.get("metadata"),
                    )
                    paths = [f"{time_unit_dir}/{key}.parquet" for key in sorted_keys]
                    log.debug("Writing %d empty partition(s) sequentially", len(paths))
                    _write_empty_parquet_files_sequentially(
                        paths,
                        schema=schema,
                        metadata=built_metadata,
                        storage_options=polars_opts,
                        credential_provider=credential_provider,
                        write_kwargs=write_kwargs,
                    )
                    log.debug("Wrote %i empty partition(s): %s", len(missing), sorted(missing))

        # Refresh the file listing to avoid scanning an empty cache after writes.
        existing_parts = _extract_existing_files(fs, fs_path_prefix)
        read_plan = _build_read_plan(
            partition_info=part_info,
            date_column=date_column,
            predicate=predicate,
            time_unit_dir=time_unit_dir,
            template_for_metadata=template_for_metadata,
            effective_time_unit=effective_time_unit,
            has_cached_files=bool(existing_parts),
        )

        scan_kwargs = {"storage_options": polars_opts} if polars_opts else {}
        if credential_provider:
            scan_kwargs["credential_provider"] = credential_provider
        scan_kwargs.update(read_kwargs)

        cache_scan = _cache_scan_from_read_plan(read_plan, time_unit_dir, scan_kwargs, schema)
        scan = _build_final_scan(cache_scan, predicate, with_columns, n_rows)

        def error_wrapper(e):
            return f"Failed to collect lazy frame in cache_parquet.\nPlan:\n{scan.explain()}\n\nReceived error: {e.__class__.__name__}:{e}"

        log.debug("Start: Loading data from cache at %s", time_unit_dir)
        start = time.time()
        try:
            yield from collect_lf_in_io_source(scan, batch_size)
        except ComputeError as e:
            if "expected at least 1 source" in str(e):
                log.debug("Predicate eliminated all row groups - returning empty dataframe with schema %s", schema)
                out_df = pl.DataFrame(schema=schema).select(with_columns) if with_columns else pl.DataFrame(schema=schema)
                yield out_df
            else:
                raise RuntimeError(error_wrapper(e)) from e
        except Exception as e:
            raise RuntimeError(error_wrapper(e)) from e
        finally:
            end = time.time()
            log.debug("End: Loading data from cache at %s took %s seconds", time_unit_dir, end - start)

    return register_io_source_with_is_pure(io_source=source_generator, schema=schema, validate_schema=False, explain_detail=description)
