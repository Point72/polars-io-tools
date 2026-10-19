import datetime
import functools
import itertools
import logging
import operator
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from datetime import date, timedelta
from enum import Enum, auto
from pathlib import Path, PureWindowsPath
from string import Template
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote, unquote, urlparse
from urllib.request import url2pathname
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    import pyarrow.fs as pa_fs

import polars as pl
import portion
from polars.exceptions import ComputeError, PanicException
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
    """Explicit cache paths, or None for a glob scan."""

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


# Polars' default cap on how many partition sinks may be open at once (the
# POLARS_MAX_OPEN_SINKS default). Writes are checked against this cap.
_DEFAULT_MAX_OPEN_SINKS = 128
_MAX_ENUMERATED_SCAN_PATHS = 64


def _max_open_sinks() -> int:
    """Return the open-sink cap polars will enforce (POLARS_MAX_OPEN_SINKS, or the default)."""
    try:
        return int(os.environ.get("POLARS_MAX_OPEN_SINKS", _DEFAULT_MAX_OPEN_SINKS))
    except ValueError:
        return _DEFAULT_MAX_OPEN_SINKS


def _cache_read_retries() -> int:
    try:
        return max(0, int(os.environ.get("POLARS_IO_TOOLS_CACHE_READ_RETRIES", "1")))
    except ValueError:
        return 1


def _storage_kwargs(storage_options: dict | None, credential_provider: pl.CredentialProviderAWS | None) -> dict:
    kwargs = {"storage_options": storage_options} if storage_options else {}
    if credential_provider:
        kwargs["credential_provider"] = credential_provider
    return kwargs


def _dataframe_write_parquet_kwargs(
    *,
    write_kwargs: dict,
    metadata: dict[str, str],
    storage_options: dict,
    credential_provider: pl.CredentialProviderAWS | None,
) -> dict:
    parquet_kwargs = dict(write_kwargs)
    for key in ("metadata", "lazy", "maintain_order"):
        parquet_kwargs.pop(key, None)
    parquet_kwargs["metadata"] = metadata
    parquet_kwargs["mkdir"] = True
    parquet_kwargs.update(_storage_kwargs(storage_options, credential_provider))
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
    READ_ONLY = auto()  # Read cached partitions, query upstream for uncached partitions, never write
    CACHE_ONLY = auto()  # Read cached partitions only, never query upstream or write

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
    return df.height == 1 and all(v is None for v in df.row(0))


def _has_null_keys(df: pl.DataFrame, columns: list[str]) -> bool:
    return bool(columns) and not df.is_empty() and df.select(columns).null_count().sum_horizontal().item() > 0


def _restrict_predicate(predicate: pl.Expr | None, columns: list[str]) -> pl.Expr | None:
    return restrict_expr_to_columns(predicate, set(columns)) if predicate is not None and columns else None


def _and_predicates(*predicates: pl.Expr | None) -> pl.Expr | None:
    filters = [p for p in predicates if p is not None]
    return functools.reduce(operator.and_, filters) if filters else None


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
    bounds_parts_df: pl.DataFrame | None = None,
) -> PartitionInfo:
    """Create a PartitionInfo snapshot to drive downstream planning decisions.

    ``bounds_parts_df`` sets the cached date bounds that clip one-sided queries; it defaults to ``existing_parts_df``.
    """
    join_cols = extra_cols + ([date_column] if date_column else [])
    enumerability = _classify_enumerability(expected_parts_df, date_column)

    clipped_expected_parts_df = None

    if enumerability == Enumerability.UNBOUNDED and date_column and time_unit and schema and predicate is not None:
        date_interval = _local_date_interval(predicate, date_column, schema[date_column])
        clipped_expected_parts_df = _compute_clipped_expected_partitions(
            predicate_interval=date_interval,
            existing_parts_df=existing_parts_df if bounds_parts_df is None else bounds_parts_df,
            date_column=date_column,
            extra_cols=extra_cols,
            time_unit=time_unit,
            schema=schema,
        )
        if clipped_expected_parts_df is not None and extra_cols:
            restricted = _restrict_predicate(predicate, extra_cols)
            if restricted is not None:
                clipped_expected_parts_df = clipped_expected_parts_df.filter(restricted)
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


def _cache_file_uri(time_unit_dir: str, key: str) -> str:
    path = quote(key, safe="/") if POLARS_HAS_PARTITION_BY and urlparse(time_unit_dir).scheme in {"file", "s3", "s3a"} else key
    return f"{time_unit_dir}/{path}.parquet"


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

    if _has_null_keys(parts_df, join_cols):
        return []

    date_fmt = None
    if date_column:
        date_fmt = _strftime_from_template(template_for_metadata or _PARTITION_FORMATS[effective_time_unit]["default"])

    keys = _partition_keys(parts_df, extra_cols, date_column, date_fmt)
    return [_cache_file_uri(time_unit_dir, key) for key in sorted(keys.unique())]


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
    return _and_predicates(*keep_exprs)


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
    return str(value)


def _encode_partition_file_key(key: str, windows: bool) -> str:
    if not windows:
        return key
    return "".join(f"%{ord(char):02X}" if char in '%<>:"\\|?*' else char for char in key)


def _serialize_key_values(values: pl.Series) -> pl.Series:
    """Serialize non-null partition values as the writer of the installed polars does."""
    if POLARS_HAS_PARTITION_BY:
        return pl.Series([_partition_key_component(v) for v in values.to_list()], dtype=pl.String)
    # PartitionByKey (polars < 1.38) names partitions by the value cast to String
    return values.cast(pl.String)


def _partition_keys(
    parts_df: pl.DataFrame, extra_cols: list[str], date_column: str | None, date_fmt: str | None, null_key_token: str | None = None
) -> pl.Series:
    """Cache keys (``a/b/2024-01``, no extension) for each row of a partitions frame.

    Uses the writer's serialization so keys compare equal to listed file keys. Rows with a null (unconstrained) partition value get a null key.
    """
    exprs = []
    for c in extra_cols:
        values = parts_df[c].drop_nulls().unique()
        expr = pl.col(c).replace_strict(values, _serialize_key_values(values), return_dtype=pl.String)
        exprs.append(expr.fill_null(null_key_token) if null_key_token is not None else expr)
    if date_column is not None and date_fmt is not None:
        expr = _stored_key_date_expr(date_column, parts_df.schema[date_column]).dt.strftime(date_fmt)
        exprs.append(expr.fill_null(null_key_token) if null_key_token is not None else expr)
    if not exprs:
        return pl.Series("key", [None] * parts_df.height, dtype=pl.String)
    return parts_df.select(pl.concat_str(exprs, separator="/").alias("key")).to_series()


def _extract_existing_files(
    fs: "pa_fs.FileSystem", fs_path_prefix: str, *, create_s3_dir: bool = True, mtimes: dict[str, datetime.datetime] | None = None
) -> set[str]:
    """Return relative file keys under a prefix, without reading cached data.

    When ``mtimes`` is given, it is filled with each key's UTC modification time.
    """
    import pyarrow.fs as pa_fs

    existing_parts: set[str] = set()

    if isinstance(fs, pa_fs.LocalFileSystem):
        path = Path(fs_path_prefix)
        if path.exists():
            for p in path.glob("**/*.parquet"):
                key = p.relative_to(path).with_suffix("").as_posix()
                if mtimes is not None:
                    try:
                        mtimes[key] = datetime.datetime.fromtimestamp(p.stat().st_mtime, datetime.UTC)
                    except FileNotFoundError:
                        # Deleted by another process since the glob.
                        continue
                existing_parts.add(key)
    else:
        # S3FileSystem is optional in pyarrow; use getattr for safe access
        S3FileSystem = getattr(pa_fs, "S3FileSystem", None)
        if create_s3_dir and S3FileSystem is not None and isinstance(fs, S3FileSystem):
            try:
                fs.create_dir(fs_path_prefix + "/")
            except Exception as e:  # noqa: BLE001 -- intentional broad catch (defensive fallback)
                log.debug(f"Failed to create S3 directory {fs_path_prefix}: {e}")

        try:
            for info in fs.get_file_info(pa_fs.FileSelector(fs_path_prefix, recursive=True)):
                if info.type == pa_fs.FileType.File and info.path.endswith(".parquet"):
                    key = Path(info.path[len(fs_path_prefix) + 1 :]).with_suffix("").as_posix()
                    existing_parts.add(key)
                    if mtimes is not None and info.mtime is not None:
                        mtime = info.mtime
                        mtimes[key] = mtime.replace(tzinfo=datetime.UTC) if mtime.tzinfo is None else mtime.astimezone(datetime.UTC)
        except Exception:  # noqa: BLE001,S110 -- intentional broad catch; best-effort cleanup, failure is non-fatal
            # Directory doesn't exist yet
            pass

    return existing_parts


def _split_custom_partition_keys(
    existing_files: set[str],
    all_partition_cols: list[str],
    date_column: str,
    file_key_column: str | None = None,
    decode_file_keys: bool = False,
) -> pl.DataFrame:
    """Return raw extra-column and date keys from custom-format paths; the date key contains all segments after the extra-column ones."""
    extra_cols = [col for col in all_partition_cols if col != date_column]
    n_extra = len(extra_cols)

    rows = []
    for file_path in existing_files:
        path_parts = (unquote(file_path) if decode_file_keys else file_path).split("/")
        if len(path_parts) <= n_extra:
            continue
        row = [*path_parts[:n_extra], "/".join(path_parts[n_extra:])]
        rows.append([*row, file_path] if file_key_column else row)

    key_schema = {**{c: pl.Utf8 for c in extra_cols}, date_column: pl.Utf8}
    if file_key_column:
        key_schema[file_key_column] = pl.String
    return pl.DataFrame(rows, schema=key_schema, orient="row")


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
    file_key_column: str | None = None,
    decode_file_keys: bool = False,
) -> pl.DataFrame:
    """Decode existing file keys to the partition schema, omitting malformed keys."""
    partition_schema_dict = {k: schema[k] for k in all_partition_cols}

    if not existing_files:
        return pl.DataFrame(schema={**partition_schema_dict, **({file_key_column: pl.String} if file_key_column else {})})

    is_custom_format = bool(date_column and template_for_metadata)
    if is_custom_format:
        df = _split_custom_partition_keys(existing_files, all_partition_cols, date_column, file_key_column, decode_file_keys)
    else:
        raw_schema = {k: pl.Utf8 for k in all_partition_cols}
        logical_keys = [(unquote(part) if decode_file_keys else part, part) for part in existing_files]
        if file_key_column:
            raw_schema[file_key_column] = pl.String
            raw_data = [[*logical.split("/"), physical] if all_partition_cols else [physical] for logical, physical in logical_keys]
        else:
            raw_data = [logical.split("/") for logical, _ in logical_keys]
        df = pl.DataFrame(raw_data, schema=raw_schema, orient="row")

    if date_column and template_for_metadata and (is_custom_format or date_column in all_partition_cols):
        df = _parse_date_keys(df, date_column, template_for_metadata)
    if is_custom_format:
        df = df.select([*all_partition_cols, *([file_key_column] if file_key_column else [])])
    return _cast_partition_keys(df, partition_schema_dict).cast(partition_schema_dict)


def _get_expected_partitions_df(
    pred: pl.Expr | None,
    date_column: str | None,
    extra_cols: list[str],
    time_unit: Literal["daily", "monthly", "yearly", "null"],
    schema: pl.Schema,
) -> pl.DataFrame:
    """Enumerate candidate partition keys, using None for unconstrained columns."""
    all_partition_cols = extra_cols + ([date_column] if date_column else [])

    if not all_partition_cols:
        return pl.DataFrame([{}])

    partition_schema = {col: schema[col] for col in all_partition_cols}
    restricted_pred = _restrict_predicate(pred, all_partition_cols)

    if restricted_pred is None:
        return pl.DataFrame([dict.fromkeys(all_partition_cols)], schema=partition_schema)

    result_df = _enumerate_partitions_from_dnf(restricted_pred, date_column, extra_cols, time_unit if time_unit != "null" else "daily", schema)

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
        for clause in dnf_clauses:
            clause_df = _process_dnf_clause_to_partitions(clause, date_column, extra_cols, time_unit, schema)
            if clause_df is None:
                # One non-enumerable clause makes the whole disjunction unbounded.
                return pl.DataFrame(schema=partition_schema)
            if not clause_df.is_empty():
                clause_dfs.append(clause_df)

        if not clause_dfs:
            return pl.DataFrame(schema=partition_schema)

        return pl.concat(clause_dfs).unique()

    except Exception as e:  # noqa: BLE001 -- intentional broad catch (defensive fallback)
        log.debug(f"DNF enumeration failed: {e}, returning unconstrained placeholder")
        return pl.DataFrame([dict.fromkeys(all_partition_cols)], schema=partition_schema)


def _process_dnf_clause_to_partitions(
    clause: DNFClause,
    date_column: str | None,
    extra_cols: list[str],
    time_unit: Literal["daily", "monthly", "yearly"],
    schema: pl.Schema,
) -> pl.DataFrame | None:
    """Process a single DNF clause, using None for unconstrained columns.

    Return an empty frame when the clause matches nothing, or None when its partitions cannot be enumerated.
    """
    all_partition_cols = extra_cols + ([date_column] if date_column else [])
    partition_schema = {c: schema[c] for c in all_partition_cols}

    constraints_by_col = {}
    for col, op, val in clause:
        if col in all_partition_cols:
            constraints_by_col.setdefault(col, []).append((op, val))

    column_value_sets = {}

    for col in all_partition_cols:
        constraints = constraints_by_col.get(col, [])

        matches_null = any(
            op in ("is", "==", "in") and (val is None or (isinstance(val, (list, set, tuple)) and None in val)) for op, val in constraints
        )
        if not constraints or matches_null:
            # Unconstrained, or matching null (which the None placeholder cannot distinguish from unconstrained)
            column_value_sets[col] = {None}
        else:
            analyzer = ColumnConstraintAnalyzer(col)
            for op, val in constraints:
                analyzer.update_from_predicate(op, val)
            if analyzer.has_contradiction(schema=schema):
                return pl.DataFrame(schema=partition_schema)
            if col == date_column:
                if not (analyzer.exact_values or analyzer.inclusion_set or (analyzer.min_bound and analyzer.max_bound)):
                    return None
                # A date exclusion removes a whole partition only when that partition holds one Date value.
                values = _extract_and_widen_dates(analyzer, time_unit, exact_exclusions=time_unit == "daily" and schema[col] == pl.Date)
            else:
                values = _extract_finite_values(analyzer, col, schema)
            if values is not None:
                values = set(pl.Series(col, list(values), dtype=schema[col]).to_list())
            if values is None:
                # Non-enumerable extra-column constraints use wildcard reads without a date column,
                # or the unbounded/date-clipping fallback when a date column is present.
                if date_column is None:
                    column_value_sets[col] = {None}
                else:
                    return None
            elif not values:
                return pl.DataFrame(schema=partition_schema)
            else:
                column_value_sets[col] = values

    return pl.DataFrame(list(itertools.product(*column_value_sets.values())), schema=partition_schema, orient="row")


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

    value_sets = [set(values) for values in (analyzer.exact_values, analyzer.inclusion_set) if values]
    if value_sets:
        # Listed values and range bounds all constrain this clause, so keep only listed values within the bounds.
        lower, upper = _bound_date(analyzer.min_bound, step=1), _bound_date(analyzer.max_bound, step=-1)
        for val in set.intersection(*value_sets):
            date_val = _convert_to_date(val)
            if date_val and (lower is None or date_val >= lower) and (upper is None or date_val <= upper):
                dates.add(_widen_date(date_val, time_unit))
    elif analyzer.min_bound or analyzer.max_bound:
        dates.update(_enumerate_date_range(analyzer, time_unit))

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


def _bound_date(bound: tuple[Any, bool] | None, step: int) -> date | None:
    """First (``step=1``) or last (``step=-1``) date a lower or upper bound admits.

    An exclusive bound excludes its own date only when no time of that date passes it: a Date, or a datetime upper bound at midnight.
    """
    if not bound:
        return None
    val, inclusive = bound
    d = _convert_to_date(val)
    if d is None or inclusive:
        return d
    excludes_date = type(val) is date or (step < 0 and val.time() == datetime.time())
    return d + timedelta(days=step) if excludes_date else d


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
    min_val = _bound_date(analyzer.min_bound, step=1)
    max_val = _bound_date(analyzer.max_bound, step=-1)
    if not min_val or not max_val:
        log.warning(f"One of: Maximumum value: {max_val} and Minimum value: {min_val} is unbounded.")
        return set()

    return set(_partition_dates(min_val, max_val, time_unit))


def _partition_dates(lower: date, upper: date, time_unit: Literal["daily", "monthly", "yearly"]) -> Iterator[date]:
    """Yield partition start dates overlapping inclusive calendar bounds."""
    current, last = _widen_date(lower, time_unit), _widen_date(upper, time_unit)
    while current <= last:
        yield current
        current = _step_date(current, time_unit)


def _step_date(d: date, time_unit: Literal["daily", "monthly", "yearly"]) -> date:
    """Step to next partition boundary."""
    if time_unit == "daily":
        return d + timedelta(days=1)
    elif time_unit == "monthly":
        if d.month == 12:
            return d.replace(year=d.year + 1, month=1)
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

    is_unbounded_below = predicate_interval.lower == -portion.inf
    is_unbounded_above = predicate_interval.upper == portion.inf

    if is_unbounded_below == is_unbounded_above:
        return None

    if existing_parts_df.is_empty() or date_column not in existing_parts_df.columns:
        return None

    existing_dates = existing_parts_df.select(_stored_key_date_expr(date_column, existing_parts_df.schema[date_column])).to_series().drop_nulls()
    if existing_dates.is_empty():
        return None

    existing_bound = existing_dates.max() if is_unbounded_above else existing_dates.min()
    if is_unbounded_above:
        predicate_bound = _bound_date((predicate_interval.lower, predicate_interval.left == portion.CLOSED), 1)
    else:
        predicate_bound = _bound_date((predicate_interval.upper, predicate_interval.right == portion.CLOSED), -1)
    lower, upper = (predicate_bound, existing_bound) if is_unbounded_above else (existing_bound, predicate_bound)
    dates = list(_partition_dates(lower, upper, time_unit))
    if not dates:
        return None
    dates_df = pl.DataFrame({date_column: dates}, schema={date_column: schema[date_column]})
    if not extra_cols:
        return dates_df
    all_partition_cols = extra_cols + [date_column]
    partition_schema = {col: schema[col] for col in all_partition_cols}
    return existing_parts_df.select(extra_cols).unique().join(dates_df, how="cross").select(all_partition_cols).cast(partition_schema)


def _local_date_interval(predicate: pl.Expr, date_column: str, dtype: pl.DataType) -> Interval:
    """Enclosing range of ``predicate`` on the date column, with datetime bounds in the column's time zone."""
    interval = convert_expr_to_datetime_range(predicate, date_column, get_enclosure=True)
    time_zone = getattr(dtype, "time_zone", None)
    if time_zone is None or interval.empty:
        return interval

    # The range visitor returns timezone-aware literals as naive UTC
    to_local = lambda v: v.replace(tzinfo=datetime.UTC).astimezone(ZoneInfo(time_zone)) if isinstance(v, datetime.datetime) else v
    return interval.replace(lower=to_local(interval.lower), upper=to_local(interval.upper))


def _partition_dates_expr(interval: Interval, date_column: str, dtype: pl.DataType) -> pl.Expr | None:
    """Filter for an atomic interval of partition start dates; timezone-aware columns compare at local midnight."""
    time_zone = getattr(dtype, "time_zone", None)
    if time_zone is not None:
        to_local = lambda d: datetime.datetime.combine(d, datetime.time(), ZoneInfo(time_zone)) if isinstance(d, date) else d
        interval = interval.replace(lower=to_local(interval.lower), upper=to_local(interval.upper))
    return _convert_atomic_interval_to_polars_expr(interval, date_column)


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

    if _is_all_null_row(partitions_df):
        return None

    if not date_column:
        all_extra_predicates = []
        for row_dict in partitions_df.to_dicts():
            row_predicates = [pl.col(col) == row_dict[col] for col in extra_cols if row_dict.get(col) is not None]
            if row_predicates:
                all_extra_predicates.append(pl.Expr.and_(*row_predicates))
        return pl.Expr.or_(*all_extra_predicates) if all_extra_predicates else None

    date_dtype = partitions_df.schema[date_column]
    partition_intervals: list[tuple[Interval, dict[str, Any]]] = []
    for row_dict in partitions_df.with_columns(_stored_key_date_expr(date_column, date_dtype)).to_dicts():
        date_val = row_dict.get(date_column)
        interval = portion.open(-portion.inf, portion.inf)
        if date_val is not None:
            if time_unit not in _PARTITION_FORMATS:
                raise ValueError(f"Unsupported time_unit: {time_unit}")
            start_date = _widen_date(date_val, time_unit)
            interval = portion.closedopen(start_date, _step_date(start_date, time_unit))
        extra_values = {col: row_dict[col] for col in extra_cols if row_dict.get(col) is not None}
        partition_intervals.append((interval, extra_values))

    merged_groups: list[tuple[Interval, list[dict[str, Any]]]] = []
    partition_intervals.sort(key=lambda x: (x[0].lower, x[0].upper))
    for interval, partition_values in partition_intervals:
        if merged_groups and (merged_groups[-1][0].overlaps(interval) or merged_groups[-1][0].adjacent(interval)):
            previous, values = merged_groups[-1]
            values.append(partition_values)
            merged_groups[-1] = (previous | interval, values)
        else:
            merged_groups.append((interval, [partition_values]))
    group_predicates = []

    for merged_interval, partition_list in merged_groups:
        cur_date_expr = _partition_dates_expr(merged_interval, date_column, date_dtype)
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
        lf = pl.scan_parquet(uri, n_rows=0, **_storage_kwargs(storage_opts, credential_provider))
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
    """Decide whether to write and how to filter unbounded partitions."""
    if cache_mode not in {CacheMode.CACHE, CacheMode.REBUILD}:
        return WritePlan(False, None, False)

    if cache_mode == CacheMode.REBUILD or partition_info.enumerability == Enumerability.FINITE:
        return WritePlan(True, None, True)

    enum = partition_info.enumerability

    # No partition columns - single global partition, trust cache once written.
    if enum == Enumerability.UNCONSTRAINED and not partition_info.join_cols:
        return WritePlan(partition_info.existing_parts_df.is_empty(), None, False)

    filters = [
        _build_not_existing_partitions_pred(partition_info.existing_parts_df, partition_info.join_cols, date_column, effective_time_unit, date_dtype)
    ]
    if enum == Enumerability.UNBOUNDED and predicate is not None:
        # Restrict writes only by whole partitions so each cache file remains complete.
        filters.append(_restrict_predicate(predicate, [c for c in partition_info.join_cols if c != date_column]))
        if date_column is not None and date_dtype is not None and effective_time_unit in _PARTITION_FORMATS:
            interval = _local_date_interval(predicate, date_column, date_dtype)
            if interval.empty:
                filters.append(pl.lit(False))
            else:
                lower, upper = -portion.inf, portion.inf
                if interval.lower != -portion.inf:
                    lower = _widen_date(_bound_date((interval.lower, interval.left == portion.CLOSED), 1), effective_time_unit)
                if interval.upper != portion.inf:
                    upper = _widen_date(_bound_date((interval.upper, interval.right == portion.CLOSED), -1), effective_time_unit)
                    upper = _step_date(upper, effective_time_unit)
                filters.append(_partition_dates_expr(portion.closedopen(lower, upper), date_column, date_dtype))
    return WritePlan(True, _and_predicates(*filters), False)


def _cache_scan_from_read_plan(
    read_plan: ReadPlan,
    time_unit_dir: str,
    scan_kwargs: dict,
    schema: pl.Schema,
) -> pl.LazyFrame:
    """Scan explicit cache paths or the cache glob, returning an empty frame for no paths."""
    if read_plan.use_paths is not None:
        log.debug("Cache scan uses %d enumerated path(s): %s", len(read_plan.use_paths), read_plan.use_paths)
        if not read_plan.use_paths:
            return pl.LazyFrame(schema=schema)
        return pl.scan_parquet(read_plan.use_paths, **{**scan_kwargs, "glob": False})
    log.debug("Cache scan uses glob: %s/**/*.parquet", time_unit_dir)
    return pl.scan_parquet(f"{time_unit_dir}/**/*.parquet", **scan_kwargs)


def _get_fs_path_directory_info(
    cache_uri: str,
    time_unit: Literal["daily", "monthly", "yearly", "null"],
) -> tuple[str, str]:
    """Return the filesystem listing prefix and sink URI for a cache's time unit directory."""
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

    fs_path_prefix = f"{bucket}{base_key}" if base_key.startswith("/") else f"{bucket}/{base_key}"
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
    file_key_column: str | None = None,
) -> ReadPlan:
    """Choose explicit paths for finite queries, or a glob for wildcard queries."""
    join_cols = partition_info.join_cols
    extra_cols = [c for c in join_cols if c != (date_column or "")]

    # Globbing an empty cache raises "expanded paths were empty".
    if not has_cached_files:
        return ReadPlan(use_paths=[])

    if partition_info.enumerability != Enumerability.FINITE:
        return ReadPlan(use_paths=None)

    expected_df = partition_info.expected_parts_df
    if expected_df.is_empty():
        return ReadPlan(use_paths=[])

    if _has_null_keys(expected_df, join_cols) or predicate is None:
        return ReadPlan(use_paths=None)

    if file_key_column:
        known = partition_info.existing_parts_df.select([*join_cols, file_key_column]).cast(expected_df.schema)
        matches = expected_df.join(known, on=join_cols, how="inner")
        use_paths = [_cache_file_uri(time_unit_dir, key) for key in sorted(matches[file_key_column].unique())]
    else:
        use_paths = _build_scan_paths(expected_df, time_unit_dir, template_for_metadata, effective_time_unit, date_column, extra_cols)
    if len(use_paths) > _MAX_ENUMERATED_SCAN_PATHS:
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
    existing = part_info.existing_parts_df
    if existing.is_empty():
        existing = _construct_partitions_df_from_existing(existing_keys, expected.schema, part_info.join_cols, date_fmt, date_column)
    known = existing.select(part_info.join_cols).cast(expected.schema)
    return expected.join(known, on=part_info.join_cols, how="anti")


def _conform_to_schema(lf: pl.LazyFrame, schema: pl.Schema) -> pl.LazyFrame:
    """Select ``schema``'s columns, casting only those whose dtype differs so predicates on the rest still push down."""
    actual = lf.collect_schema()
    return lf.select([pl.col(name) if actual.get(name) == dtype else pl.col(name).cast(dtype) for name, dtype in schema.items()])


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


def _collect_cache_scan(
    make_scan: Callable[[], pl.LazyFrame],
    schema: pl.Schema,
    with_columns: list[str] | None,
    batch_size: int | None,
    time_unit_dir: str,
    retry: bool = True,
) -> Iterator[pl.DataFrame]:
    """Collect a cache scan, handling predicates that eliminate every row group.

    Set ``retry`` to False when the scan includes an upstream query, which a retry would run again.
    """
    log.debug("Start: Loading data from cache at %s", time_unit_dir)
    start = time.time()
    scan = make_scan()
    try:
        # A concurrent refresh can replace a file mid-read. A read that fails before yielding any rows is retried with a new scan,
        # since a scan keeps the Parquet metadata it has already read.
        attempts = 1 + _cache_read_retries() if retry else 1
        for attempt in range(attempts):
            if attempt:
                scan = make_scan()
            yielded = False
            try:
                for batch in collect_lf_in_io_source(scan, batch_size):
                    yielded = True
                    yield batch
                return
            except (ComputeError, OSError, PanicException) as e:
                if yielded or attempt + 1 == attempts or "expected at least 1 source" in str(e):
                    raise
                log.warning("Cache read failed (attempt %d/%d), retrying; a concurrent refresh can cause this: %s", attempt + 1, attempts, e)
    except Exception as e:
        if isinstance(e, ComputeError) and "expected at least 1 source" in str(e):
            log.debug("Predicate eliminated all row groups - returning empty dataframe with schema %s", schema)
            yield pl.DataFrame(schema=schema).select(with_columns) if with_columns else pl.DataFrame(schema=schema)
        else:
            raise RuntimeError(
                f"Failed to collect lazy frame in cache_parquet.\nPlan:\n{scan.explain()}\n\nReceived error: {e.__class__.__name__}:{e}"
            ) from e
    finally:
        log.debug("End: Loading data from cache at %s took %s seconds", time_unit_dir, time.time() - start)


def _build_cache_metadata(
    effective_time_unit: str,
    template_for_metadata: str | None,
    extra_cols: list[str],
    user_metadata: dict | None = None,
) -> dict[str, str]:
    """Build metadata dictionary for cache parquet files."""
    metadata = {
        "__piot__time_unit": effective_time_unit,
        "__piot__partition_format": template_for_metadata or "",
        "__piot__partitioned_by": ",".join(extra_cols),
    }
    if user_metadata:
        metadata.update(user_metadata)

    return metadata


def _rebuild_partition_scope(
    predicate: pl.Expr | None,
    date_column: str | None,
    extra_cols: list[str],
    time_unit: Literal["daily", "monthly", "yearly", "null"],
    schema: pl.Schema,
    *,
    partition_keys: bool = False,
) -> pl.Expr | None:
    restricted = _restrict_predicate(predicate, extra_cols + ([date_column] if date_column else []))
    if restricted is None:
        return None
    if date_column is None or date_column not in restricted.meta.root_names():
        return restricted
    extra_predicate = _restrict_predicate(predicate, extra_cols)
    date_predicate = _restrict_predicate(predicate, [date_column])
    null_row = pl.DataFrame({date_column: [None]}, schema={date_column: schema[date_column]})
    includes_null_date = date_predicate is not None and null_row.select(date_predicate).item() is True
    clauses = convert_expr_to_dnf(restricted)
    if clauses is None:
        return _restrict_predicate(predicate, extra_cols)
    comparisons = {
        "=": operator.eq,
        "==": operator.eq,
        "!=": operator.ne,
        "<>": operator.ne,
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "!>": operator.le,
        "!>=": operator.lt,
        "!<": operator.ge,
        "!<=": operator.gt,
    }
    scopes = []
    for clause in clauses:
        filters = []
        date_constraints = []
        for col, op, value in clause:
            if col == date_column:
                date_constraints.append((op, value))
                continue
            expr = pl.col(col)
            if op == "is" or (op in ("=", "==") and value is None):
                filters.append(expr.is_null())
            elif op in ("!is", "is not"):
                filters.append(expr.is_not_null())
            elif op in ("in", "!in", "not in"):
                values = list(value) if isinstance(value, (list, tuple, set)) else [value]
                expr = expr.is_in(_is_in_collection(values, schema[col]))
                filters.append((expr.not_() | pl.col(col).is_null()) if op != "in" else expr | pl.col(col).is_null())
            elif op in ("~", "=~", "!~"):
                expr = expr.str.contains(value)
                filters.append(expr.not_() if op == "!~" else expr)
            else:
                expr = comparisons[op](expr, value)
                filters.append(expr | pl.col(col).is_null() if op in ("!=", "<>") else expr)
        if date_constraints:
            analyzer = ColumnConstraintAnalyzer(date_column)
            for op, value in date_constraints:
                analyzer.update_from_predicate(op, value)
            if analyzer.has_contradiction(schema=schema):
                continue
            date_expr = pl.col(date_column)
            row_key = (
                _stored_key_date_expr(date_column, schema[date_column])
                if partition_keys
                else _partition_date_expr(date_column, schema[date_column], time_unit)
            )
            values = analyzer.exact_values or analyzer.inclusion_set
            if analyzer.is_null is True:
                filters.append(date_expr.is_null())
            elif values is not None:
                dates = _extract_and_widen_dates(analyzer, time_unit, exact_exclusions=time_unit == "daily" and schema[date_column] == pl.Date)
                selected = row_key.is_in(dates).fill_null(False)
                filters.append(selected | date_expr.is_null() if None in values and includes_null_date else selected)
            else:
                filters.append(pl.lit(True) if includes_null_date else date_expr.is_not_null())
                if analyzer.min_bound:
                    filters.append(row_key >= _widen_date(_bound_date(analyzer.min_bound, 1), time_unit))
                if analyzer.max_bound:
                    filters.append(row_key <= _widen_date(_bound_date(analyzer.max_bound, -1), time_unit))
                if time_unit == "daily" and schema[date_column] == pl.Date:
                    filters.extend(date_expr.ne_missing(value) for value in analyzer.exclusion_values)
        if not filters:
            return None
        scopes.append(pl.all_horizontal(filters))
    return _and_predicates(pl.any_horizontal(scopes) if scopes else pl.lit(False), extra_predicate)


def _write_partition_filter(
    partitions: pl.DataFrame,
    part_info: PartitionInfo,
    write_plan: WritePlan,
    cache_mode: CacheMode,
    predicate: pl.Expr | None,
    date_column: str | None,
    time_unit: Literal["daily", "monthly", "yearly", "null"],
    schema: pl.Schema,
) -> pl.Expr | None:
    """Select complete target partitions, excluding existing ones in CACHE mode."""
    join_cols = part_info.join_cols
    if cache_mode == CacheMode.REBUILD:
        return _rebuild_partition_scope(predicate, date_column, [c for c in join_cols if c != date_column], time_unit, schema)
    if date_column is None and _has_null_keys(partitions, join_cols) and cache_mode in (CacheMode.CACHE, CacheMode.REBUILD):
        write_filter = _restrict_predicate(predicate, join_cols)
    elif not partitions.is_empty():
        write_filter = _generate_write_predicate_from_partitions_df(
            partitions, date_column, [c for c in join_cols if c != (date_column or "")], time_unit
        )
    else:
        return write_plan.filter_predicate
    if cache_mode == CacheMode.CACHE:
        write_filter = _and_predicates(
            write_filter,
            _build_not_existing_partitions_pred(
                part_info.existing_parts_df, join_cols, date_column, time_unit, schema[date_column] if date_column else None
            ),
        )
    return write_filter


def _sink_cache_partitions(
    lf: pl.LazyFrame,
    time_unit_dir: str,
    key_exprs: list[pl.Expr],
    n_partitions: int,
    metadata: dict[str, str],
    storage_options: dict,
    credential_provider: pl.CredentialProviderAWS | None,
    write_kwargs: dict,
    key_aliases: dict[str, str],
    encode_file_keys: bool,
    fs: "pa_fs.FileSystem",
    fs_path_prefix: str,
) -> tuple[set[str], set[str]]:
    """Sink one file per partition and return its physical and logical keys.

    If the sink fails, every file it opened is deleted (best effort), so a partially written partition is refetched rather than read as complete.
    """
    written: set[str] = set()
    written_logical: set[str] = set()

    def reopened(key: object) -> RuntimeError:
        return RuntimeError(
            f"cache_parquet: polars reopened a partition sink for key {key!r}. "
            f"The open-sink cap ({_max_open_sinks()}) was exceeded and rows would be "
            "silently dropped. Set POLARS_MAX_OPEN_SINKS above the "
            "partition count before the process starts, or write fewer "
            "partitions per call."
        )

    def file_path(ctx):
        if hasattr(ctx, "partition_keys"):
            # A reopened sink overwrites its earlier file; fail rather than silently dropping rows.
            if ctx.index_in_partition != 0:
                raise reopened(ctx.partition_keys.row(0))
            values = [_partition_key_component(v) for v in ctx.partition_keys.row(0)]
        else:
            values = [unquote(k.str_value) for k in ctx.keys]
        logical_key = "/".join(values)
        key = key_aliases.get(logical_key, _encode_partition_file_key(logical_key, encode_file_keys))
        if key in written:
            raise reopened(logical_key)
        written_logical.add(logical_key)
        written.add(key)
        return f"{key}.parquet"

    if POLARS_HAS_PARTITION_BY:
        partition_obj = pl.PartitionBy(
            base_path=time_unit_dir,
            key=key_exprs,
            file_path_provider=file_path,
            include_key=False,
            # Size-based splitting violates the one-file-per-partition layout.
            approximate_bytes_per_file=None,
        )
    else:
        partition_obj = pl.PartitionByKey(base_path=time_unit_dir, by=key_exprs, include_key=False, file_path=file_path)

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

    sink_kwargs = dict(write_kwargs)
    sink_kwargs.pop("metadata", None)
    try:
        _prepare_lf_for_sink_from_io_source(lf).sink_parquet(  # type: ignore[call-overload]
            partition_obj,
            maintain_order=True,
            mkdir=True,
            metadata=metadata,
            storage_options=storage_options,
            **({"credential_provider": credential_provider} if credential_provider else {}),
            **sink_kwargs,
        )
    except BaseException:  # PanicException derives from BaseException.
        if written:
            log.warning("Cache write failed; deleting %d partition file(s) it opened: %s", len(written), sorted(written))
        for key in sorted(written):
            try:
                fs.delete_file(f"{fs_path_prefix}/{key}.parquet")
            except FileNotFoundError:
                pass
            except Exception as e:  # noqa: BLE001 -- keep the original error; this file may hold incomplete data
                log.warning("Failed to delete partition %s after a failed cache write: %s", key, e)
        raise
    log.debug("Wrote %i new partition(s): %s", len(written), sorted(written))
    return written, written_logical


def _missing_empty_keys(
    part_info: PartitionInfo,
    write_plan: WritePlan,
    cache_mode: CacheMode,
    written: set[str],
    existing: set[str],
    extra_cols: list[str],
    date_column: str | None,
    date_fmt: str | None,
    scope: pl.Expr | None = None,
) -> set[str]:
    """Return enumerable gap keys, clipping unbounded queries to existing cache bounds."""
    if part_info.enumerability == Enumerability.UNBOUNDED:
        expected = part_info.clipped_expected_parts_df
    elif part_info.enumerability == Enumerability.FINITE and write_plan.write_empty_missing:
        expected = part_info.expected_parts_df
    else:
        return set()
    if expected is None:
        return set()
    if cache_mode == CacheMode.REBUILD and scope is not None:
        expected = expected.filter(scope)
    expected = expected.drop_nulls()
    if expected.is_empty():
        return set()
    known = _construct_partitions_df_from_existing(written, expected.schema, part_info.join_cols, date_fmt, date_column).cast(expected.schema)
    if cache_mode != CacheMode.REBUILD:
        existing_rows = part_info.existing_parts_df.select(part_info.join_cols).cast(expected.schema)
        known = pl.concat([known, existing_rows], how="vertical")
    expected = expected.join(known, on=part_info.join_cols, how="anti")
    keys = set(_partition_keys(expected, extra_cols, date_column, date_fmt).drop_nulls())
    return keys


def _stale_partition_keys(existing: pl.DataFrame, written: set[str], scope: pl.Expr | None, file_key_column: str) -> set[str]:
    """Return cached wildcard targets absent from the rebuilt data."""
    in_scope = existing if scope is None else existing.filter(scope)
    return set(in_scope[file_key_column].to_list()) - written


ExpiryPolicy = Callable[[datetime.datetime, datetime.datetime | None, datetime.datetime], bool]


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _default_expired(
    written_at: datetime.datetime,
    period_end: datetime.datetime | None,
    now: datetime.datetime,
    *,
    ttl: timedelta | None,
    final_after: timedelta | None,
) -> bool:
    if final_after is not None and period_end is not None and written_at >= period_end + final_after:
        return False
    return ttl is None or now - written_at >= ttl


def _expired_file_keys(
    parts_df: pl.DataFrame,
    mtimes: dict[str, datetime.datetime],
    *,
    file_key_column: str,
    date_column: str | None,
    date_dtype: pl.DataType | None,
    time_unit: Literal["daily", "monthly", "yearly"],
    is_expired: ExpiryPolicy,
    now: datetime.datetime,
) -> set[str]:
    """Physical keys of cached files that ``is_expired`` reports as expired."""
    keys = parts_df[file_key_column].to_list()
    starts = [None] * len(keys)
    if date_column is not None and date_dtype is not None:
        starts = parts_df.select(_stored_key_date_expr(date_column, date_dtype)).to_series().to_list()
    # Periods end at local midnight for timezone-aware columns and at UTC midnight otherwise.
    zone = ZoneInfo(getattr(date_dtype, "time_zone", None) or "UTC")
    expired = set()
    for key, start in zip(keys, starts, strict=True):
        written_at = mtimes.get(key)
        if written_at is None:
            continue
        period_end = None
        if start is not None:
            period_end = datetime.datetime.combine(_step_date(_widen_date(start, time_unit), time_unit), datetime.time(), zone).astimezone(
                datetime.UTC
            )
        if is_expired(written_at, period_end, now):
            expired.add(key)
    return expired


def cache_parquet(
    self_or_fn: pl.LazyFrame | Callable[[], pl.LazyFrame],
    cache_path: str | Path,
    date_column: str | None = None,
    *,
    time_unit: Literal["daily", "monthly", "yearly"] = "monthly",
    partition_format: str | None = None,
    cache_mode: CacheMode = CacheMode.CACHE,
    ttl: timedelta | ExpiryPolicy | None = None,
    final_after: timedelta | None = None,
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
            - ``CacheMode.READ_ONLY``: Read the partitions that are already cached and query upstream for the
              partitions that are not, without writing anything to the cache. A cached partition is always served
              from the cache, even if it is empty.
            - ``CacheMode.CACHE_ONLY``: Read only the partitions that are already cached; never query upstream or
              write. Returns an empty frame if nothing matching is cached. With a callable source and an empty
              cache, ``schema`` is required.
        ttl (timedelta | Callable, optional): Expire cached files written at least this long ago. An expired file is
            refetched and overwritten under ``CacheMode.CACHE``, and read from upstream instead of the cache under
            ``CacheMode.READ_ONLY``. ``CacheMode.CACHE_ONLY`` serves expired files and ``CacheMode.REBUILD`` ignores
            expiry. Only files within the query's partition scope are refetched; an expired file for which upstream
            returns no rows is replaced by an empty file. Expiry uses the file modification time, which copying or
            syncing a cache may reset. A callable ``ttl(written_at, period_end, now) -> bool`` replaces the built-in
            policy and returns True for an expired file. Its arguments are timezone-aware UTC datetimes, and
            ``period_end`` is the exclusive end of the file's date partition, or None without one. The callable must
            depend only on its arguments. A read that fails before returning any rows, for example because another
            process refreshed a file mid-read, is retried ``POLARS_IO_TOOLS_CACHE_READ_RETRIES`` times (default 1),
            unless the read also queries upstream.
        final_after (timedelta, optional): Never expire a partition once its file was written at least this long after
            the partition's period ended (local midnight for timezone-aware columns, UTC midnight otherwise). ``ttl``
            then applies only to the remaining partitions; without ``ttl`` they are refetched on every query. Requires
            ``date_column`` and a non-callable ``ttl``.
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

        Queries are classified based on whether partition keys can be enumerated from the predicate. The writes
        described below happen in ``CacheMode.CACHE`` and ``CacheMode.REBUILD``. ``CacheMode.READ_ONLY`` returns the
        upstream rows it would have written, and ``CacheMode.CACHE_ONLY`` never queries upstream.

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
        arriving data for dates within the existing range, set ``ttl`` and ``final_after`` so recent
        partitions are refetched, or use ``cache_mode=CacheMode.REBUILD`` to refresh the cache.

        **Write-Then-Read Architecture**

        ``cache_parquet`` follows a simple pattern: write missing partitions to cache, then read
        entirely from cache. There is no merging of upstream and cache data during reads, except in
        ``CacheMode.READ_ONLY``, which cannot write and so reads uncached partitions from upstream.

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
        - Failed writes: if a write raises, the partition files it opened are deleted so the next query
          refetches them. Deletion is best effort; failures are logged, and a file that survives may need
          manual removal. A killed process deletes nothing. Concurrent readers may fail or read incomplete
          partitions.
    """
    if cache_mode == CacheMode.IGNORE:
        return self_or_fn() if callable(self_or_fn) else self_or_fn

    # Validate arguments
    if ttl is not None and not isinstance(ttl, timedelta) and not callable(ttl):
        raise TypeError(f"ttl must be a timedelta or a callable, got {type(ttl).__name__}")
    if final_after is not None:
        if not isinstance(final_after, timedelta):
            raise TypeError(f"final_after must be a timedelta, got {type(final_after).__name__}")
        if date_column is None:
            raise ValueError("final_after requires date_column")
        if callable(ttl):
            raise ValueError("final_after cannot be combined with a callable ttl")
    for name, value in (("ttl", ttl), ("final_after", final_after)):
        if isinstance(value, timedelta) and value < timedelta(0):
            raise ValueError(f"{name} must not be negative")
    is_expired: ExpiryPolicy | None = None
    if callable(ttl):
        is_expired = ttl
    elif ttl is not None or final_after is not None:
        is_expired = functools.partial(_default_expired, ttl=ttl, final_after=final_after)
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

    windows_file_keys = os.name == "nt" and isinstance(fs, pa_fs.LocalFileSystem)
    source_lf = self_or_fn if isinstance(self_or_fn, pl.LazyFrame) else None

    def get_source() -> pl.LazyFrame:
        nonlocal source_lf
        if source_lf is None:
            source_lf = self_or_fn()  # type: ignore[operator]
        return source_lf

    # Resolve schema with minimal cost: prefer provided, then cache, then source (callable only once)
    schema = schema or _schema_from_cache(cache_uri, polars_opts, credential_provider)
    if schema is None:
        if cache_mode == CacheMode.CACHE_ONLY and source_lf is None:
            raise ValueError("CacheMode.CACHE_ONLY with a callable source requires `schema` when the cache is empty")
        schema = get_source().collect_schema()

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

    all_partition_cols = _extra_cols + ([date_column] if date_column else [])
    file_key_column = "__piot_file_key__"
    while file_key_column in all_partition_cols:
        file_key_column += "_"
    key_exprs = [pl.col(col).alias(f"__piot_key_{col}__") for col in _extra_cols]
    if date_column is not None and partition_format is not None:
        key_exprs.append(pl.col(date_column).dt.strftime(partition_format).alias("__piot_key_date__"))
    if not key_exprs:
        key_exprs = [pl.lit("data").alias("__piot_key_single__")]

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        log.debug("Running with %s against location %s with columns '%s' and predicate: '%s'", cache_mode, time_unit_dir, with_columns, predicate)

        no_write = cache_mode in (CacheMode.READ_ONLY, CacheMode.CACHE_ONLY)
        # READ_ONLY plans like CACHE, but reads the missing partitions from upstream instead of writing them.
        plan_mode = CacheMode.CACHE if cache_mode == CacheMode.READ_ONLY else cache_mode
        read_schema = schema
        if cache_mode == CacheMode.READ_ONLY and with_columns is not None:
            required_columns = set(with_columns) | set(all_partition_cols)
            if predicate is not None:
                required_columns.update(predicate.meta.root_names())
            read_schema = pl.Schema({name: dtype for name, dtype in schema.items() if name in required_columns})
        mtimes: dict[str, datetime.datetime] | None = {} if is_expired is not None and cache_mode in (CacheMode.CACHE, CacheMode.READ_ONLY) else None
        existing_parts = _extract_existing_files(fs, fs_path_prefix, create_s3_dir=not no_write, mtimes=mtimes)
        existing_parts_df = _construct_partitions_df_from_existing(
            existing_files=existing_parts,
            schema=schema,
            all_partition_cols=all_partition_cols,
            template_for_metadata=template_for_metadata,
            date_column=date_column,
            time_unit=time_unit,
            file_key_column=file_key_column,
            decode_file_keys=windows_file_keys,
        )

        # Expired files in the query's partition scope are planned as missing; all other files stay cached.
        refresh_keys: set[str] = set()
        if mtimes:
            expired = _expired_file_keys(
                existing_parts_df,
                mtimes,
                file_key_column=file_key_column,
                date_column=date_column,
                date_dtype=schema[date_column] if date_column else None,
                time_unit=time_unit,
                is_expired=is_expired,  # type: ignore[arg-type]
                now=_utcnow(),
            )
            if expired:
                expired_df = existing_parts_df.filter(pl.col(file_key_column).is_in(sorted(expired)))
                scope = _rebuild_partition_scope(predicate, date_column, _extra_cols, effective_time_unit, schema, partition_keys=True)
                refresh_keys = set((expired_df if scope is None else expired_df.filter(scope))[file_key_column].to_list())
                log.debug("Refreshing %d expired partition(s) in scope: %s", len(refresh_keys), sorted(refresh_keys))
        plan_parts = existing_parts - refresh_keys
        plan_parts_df = existing_parts_df.filter(~pl.col(file_key_column).is_in(sorted(refresh_keys))) if refresh_keys else existing_parts_df

        expected_parts_df = _get_expected_partitions_df(predicate, date_column, _extra_cols, effective_time_unit, schema)
        part_info = _build_partition_info(
            expected_parts_df=expected_parts_df,
            existing_parts_df=plan_parts_df,
            date_column=date_column,
            extra_cols=_extra_cols,
            predicate=predicate,
            time_unit=time_unit if date_column else None,
            schema=schema,
            bounds_parts_df=existing_parts_df,
        )

        write_plan = _build_write_plan(
            cache_mode=plan_mode,
            partition_info=part_info,
            date_column=date_column,
            predicate=predicate,
            effective_time_unit=effective_time_unit,
            date_dtype=schema[date_column] if date_column else None,
        )

        is_finite = part_info.enumerability == Enumerability.FINITE
        if is_finite and plan_mode != CacheMode.REBUILD:
            partitions_to_write_df = _compute_partitions_to_write(part_info, plan_parts, date_column, partition_format)
        else:
            partitions_to_write_df = part_info.expected_parts_df
        fetch_upstream = write_plan.should_write and not (is_finite and partitions_to_write_df.is_empty())

        log.debug(
            "Expected %d partitions, %d existing, %d missing (fetch_upstream=%s, write=%s)",
            len(part_info.expected_parts_df),
            len(existing_parts_df),
            len(partitions_to_write_df),
            fetch_upstream,
            fetch_upstream and not no_write,
        )

        upstream_lf = None
        if fetch_upstream:
            upstream_lf = get_source()
            if cache_mode == CacheMode.READ_ONLY:
                # Rows are returned rather than written, so partition filters must see the cached dtypes.
                upstream_lf = _conform_to_schema(upstream_lf, read_schema)
            write_filter = _write_partition_filter(
                partitions_to_write_df, part_info, write_plan, plan_mode, predicate, date_column, effective_time_unit, schema
            )
            if write_filter is not None:
                upstream_lf = upstream_lf.filter(write_filter)
            bounding_pred = _restrict_predicate(predicate, write_bounding_columns or [])
            if bounding_pred is not None:
                upstream_lf = upstream_lf.filter(bounding_pred)

        if upstream_lf is not None and not no_write:
            cache_metadata = _build_cache_metadata(
                effective_time_unit=effective_time_unit,
                template_for_metadata=template_for_metadata,
                extra_cols=_extra_cols,
                user_metadata=write_kwargs.get("metadata"),
            )
            key_aliases = {}
            if all_partition_cols and not existing_parts_df.is_empty():
                logical_keys = _partition_keys(
                    existing_parts_df,
                    _extra_cols,
                    date_column,
                    partition_format,
                    null_key_token="None" if POLARS_HAS_PARTITION_BY else "null",
                )
                key_aliases = dict(zip(logical_keys, existing_parts_df[file_key_column]))
            written_parts, written_logical_parts = _sink_cache_partitions(
                upstream_lf,
                time_unit_dir,
                key_exprs,
                len(partitions_to_write_df),
                cache_metadata,
                polars_opts,
                credential_provider,
                write_kwargs,
                key_aliases,
                windows_file_keys,
                fs,
                fs_path_prefix,
            )

            # Enumerated rebuild targets receive empty files; wildcard and unbounded targets require stale-file removal.
            wildcard = not is_finite or _has_null_keys(partitions_to_write_df, part_info.join_cols)
            if cache_mode == CacheMode.REBUILD and wildcard and not existing_parts_df.is_empty():
                key_scope = _rebuild_partition_scope(predicate, date_column, _extra_cols, effective_time_unit, schema, partition_keys=True)
                stale = _stale_partition_keys(existing_parts_df, written_parts, key_scope, file_key_column)
                for key in sorted(stale):
                    try:
                        fs.delete_file(f"{fs_path_prefix}/{key}.parquet")
                        log.debug("Removed stale partition on REBUILD: %s", key)
                    except Exception as e:  # noqa: BLE001 -- best-effort cleanup, non-fatal
                        log.warning("Failed to remove stale partition %s on REBUILD: %s", key, e)

            key_scope = (
                _rebuild_partition_scope(predicate, date_column, _extra_cols, effective_time_unit, schema, partition_keys=True)
                if cache_mode == CacheMode.REBUILD
                else None
            )
            missing = _missing_empty_keys(
                part_info, write_plan, cache_mode, written_logical_parts, plan_parts, _extra_cols, date_column, partition_format, key_scope
            )
            empty_paths = {f"{time_unit_dir}/{key_aliases.get(key, _encode_partition_file_key(key, windows_file_keys))}.parquet" for key in missing}
            # Expired files that upstream no longer fills are emptied by physical key so their stale rows are not read.
            emptied_expired = refresh_keys - written_parts
            if emptied_expired:
                log.warning("Replacing %d expired cache file(s) with empty files: %s", len(emptied_expired), sorted(emptied_expired))
                empty_paths.update(f"{time_unit_dir}/{key}.parquet" for key in emptied_expired)
            if empty_paths:
                paths = sorted(empty_paths)
                log.debug("Writing %d empty partition(s) sequentially", len(paths))
                _write_empty_parquet_files_sequentially(
                    paths,
                    schema=schema,
                    metadata=cache_metadata,
                    storage_options=polars_opts,
                    credential_provider=credential_provider,
                    write_kwargs=write_kwargs,
                )
                log.debug("Wrote %i empty partition(s): %s", len(paths), paths)

        if upstream_lf is not None and not no_write:
            # Refresh the file listing to avoid scanning an empty cache after writes.
            existing_parts = _extract_existing_files(fs, fs_path_prefix)
            part_info = replace(
                part_info,
                existing_parts_df=_construct_partitions_df_from_existing(
                    existing_parts,
                    schema,
                    all_partition_cols,
                    template_for_metadata,
                    date_column,
                    time_unit,
                    file_key_column=file_key_column,
                    decode_file_keys=windows_file_keys,
                ),
            )
        elif refresh_keys and cache_mode == CacheMode.CACHE:
            # Nothing was written, so read every cached file.
            part_info = replace(part_info, existing_parts_df=existing_parts_df)
        read_plan = _build_read_plan(
            partition_info=part_info,
            date_column=date_column,
            predicate=predicate,
            time_unit_dir=time_unit_dir,
            template_for_metadata=template_for_metadata,
            effective_time_unit=effective_time_unit,
            has_cached_files=bool(existing_parts),
            file_key_column=file_key_column,
        )

        if refresh_keys and no_write and read_plan.use_paths is None:
            # Upstream supplies the expired partitions, so scan only the other cached files.
            read_plan = ReadPlan(use_paths=[_cache_file_uri(time_unit_dir, key) for key in sorted(plan_parts)])

        scan_kwargs = _storage_kwargs(polars_opts, credential_provider)
        scan_kwargs.update(read_kwargs)

        def make_scan() -> pl.LazyFrame:
            cache_scan = _cache_scan_from_read_plan(read_plan, time_unit_dir, scan_kwargs, schema)
            if no_write and upstream_lf is not None:
                cache_scan = pl.concat([_conform_to_schema(cache_scan, read_schema), upstream_lf], how="vertical")
                if "n_rows" in read_kwargs:
                    cache_scan = cache_scan.head(read_kwargs["n_rows"])
            return _build_final_scan(cache_scan, predicate, with_columns, n_rows)

        yield from _collect_cache_scan(make_scan, schema, with_columns, batch_size, time_unit_dir, retry=not (no_write and upstream_lf is not None))

    return register_io_source_with_is_pure(io_source=source_generator, schema=schema, validate_schema=False, explain_detail=description)
