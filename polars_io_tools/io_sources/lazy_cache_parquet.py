import datetime
import functools
import logging
import operator
import time
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum, auto
from pathlib import Path, PureWindowsPath
from string import Template
from typing import TYPE_CHECKING, Any, Callable, Iterator, List, Literal, Optional, Set, Union
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
from .util import _storage_options_for, collect_lf_in_io_source, register_io_source_with_is_pure


@dataclass(frozen=True)
class WritePlan:
    should_write: bool
    filter_predicate: Optional[pl.Expr]
    write_empty_missing: bool


@dataclass(frozen=True)
class ReadPlan:
    """Plan for reading from cache after writes complete.

    use_paths: Explicit list of cache file paths to read, or None to use glob pattern.
    """

    use_paths: Optional[list[str]]


log = logging.getLogger(__name__)

__all__ = ["CacheMode", "cache_parquet"]


def _path_as_file_uri(path: Union[Path, PureWindowsPath]) -> str:
    path_str = str(path)
    if len(path_str) >= 3 and path_str[1] == ":" and path_str[2] in {"/", "\\"}:
        return PureWindowsPath(path_str).as_uri()
    return Path(path_str).as_uri()


def _prepare_lf_for_sink_from_io_source(lf: pl.LazyFrame) -> pl.LazyFrame:
    has_python_scan = "PYTHON SCAN" in lf.explain(optimized=False)
    if pl.thread_pool_size() > 1 and not has_python_scan:
        return lf
    # Avoid re-entering partitioned sink execution from an io_source callback when
    # Polars has only one Rayon worker, or when the source plan itself contains a
    # Python io_source. This path is less memory efficient, but it prevents a
    # scheduler deadlock for chained cache_parquet calls.
    return lf.collect().lazy()


# Constants

_MAX_PARTITIONED_SINK_PARTITIONS = 64
_MAX_ENUMERATED_SCAN_PATHS = 64


def _dataframe_write_parquet_kwargs(
    *,
    write_kwargs: dict,
    metadata: dict[str, str],
    storage_options: dict,
    credential_provider: Optional[pl.CredentialProviderAWS],
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


def _write_partitioned_lf_sequentially(
    lf: pl.LazyFrame,
    *,
    key_exprs: List[pl.Expr],
    key_names: List[str],
    time_unit_dir: str,
    metadata: dict[str, str],
    storage_options: dict,
    credential_provider: Optional[pl.CredentialProviderAWS],
    write_kwargs: dict,
) -> set[str]:
    df = lf.with_columns(key_exprs).collect()
    if df.is_empty():
        return set()

    parquet_kwargs = _dataframe_write_parquet_kwargs(
        write_kwargs=write_kwargs,
        metadata=metadata,
        storage_options=storage_options,
        credential_provider=credential_provider,
    )

    written_keys: set[str] = set()
    groups = df.partition_by(key_names, maintain_order=True, include_key=True, as_dict=True)
    for key_tuple, group in groups.items():
        key_values = [unquote(str(v)) for v in key_tuple]
        key_path = "/".join(key_values)
        written_keys.add(key_path)

        path_parts = key_values[:-1]
        path_parts.append(f"{key_values[-1]}.parquet")
        group.drop(key_names).write_parquet(f"{time_unit_dir}/{'/'.join(path_parts)}", **parquet_kwargs)

    return written_keys


def _write_empty_parquet_files_sequentially(
    paths: List[str],
    *,
    schema: pl.Schema,
    metadata: dict[str, str],
    storage_options: dict,
    credential_provider: Optional[pl.CredentialProviderAWS],
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
    clipped_expected_parts_df: Optional[pl.DataFrame] = None


def _is_all_null_row(df: pl.DataFrame) -> bool:
    """Return True if DataFrame is exactly one row and all values are None."""
    if df.height != 1:
        return False
    row = df.row(0, named=True)
    return all(v is None for v in row.values())


def _classify_enumerability(expected_parts_df: pl.DataFrame, date_column: Optional[str]) -> Enumerability:
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
    date_column: Optional[str],
    extra_cols: list[str],
    predicate: Optional[pl.Expr] = None,
    time_unit: Optional[Literal["daily", "monthly", "yearly"]] = None,
    schema: Optional[pl.Schema] = None,
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
    for placeholder, code in [("$year", "%Y"), ("$month", "%m"), ("$day", "%d")]:
        fmt = fmt.replace(placeholder, code)
    return fmt


def _build_scan_paths(
    parts_df: pl.DataFrame,
    time_unit_dir: str,
    template_for_metadata: Optional[str],
    effective_time_unit: Literal["daily", "monthly", "yearly", "null"],
    date_column: Optional[str],
    extra_cols: List[str],
) -> List[str]:
    """Build explicit cache file paths from an expected partitions DataFrame.

    - Returns [] if `parts_df` is empty or any join column has None (cannot enumerate).
    - Sorts rows by join columns for deterministic order.
    - Uses custom template or default format to render date strings.
    """
    if parts_df.is_empty():
        return []

    join_cols = extra_cols + ([date_column] if date_column else [])

    # If any join column has None, we cannot enumerate finite paths
    for row in parts_df.to_dicts():
        if any(row.get(c) is None for c in join_cols):
            return []

    # Determine date formatting
    date_fmt = None
    if date_column:
        if template_for_metadata:
            date_fmt = _strftime_from_template(template_for_metadata)
        else:
            default_tmpl = _PARTITION_FORMATS[effective_time_unit]["default"]
            date_fmt = _strftime_from_template(default_tmpl)

    # Ensure deterministic path order
    parts_sorted = parts_df.sort(join_cols) if join_cols else parts_df

    paths: List[str] = []
    for row in parts_sorted.to_dicts():
        components: List[str] = []
        for c in extra_cols:
            components.append(str(row[c]))
        if date_column:
            dval = row[date_column]
            components.append(dval.strftime(date_fmt))
        key = "/".join(components)
        paths.append(f"{time_unit_dir}/{key}.parquet")
    return paths


def _build_not_existing_partitions_pred(
    existing_parts_df: pl.DataFrame,
    join_cols: list[str],
) -> Optional[pl.Expr]:
    """Build a predicate that excludes existing partition combinations.

    Returns an expression equivalent to NOT(OR(AND(col==val ...))) across rows in `existing_parts_df[join_cols]`.
    If no rows or no join columns, returns None.
    """
    if not join_cols or existing_parts_df.is_empty():
        return None
    rows = existing_parts_df.select(join_cols).unique().to_dicts()
    if not rows:
        return None
    ors: list[pl.Expr] = []
    for row in rows:
        ands: list[pl.Expr] = []
        for col in join_cols:
            val = row[col]
            ands.append(pl.col(col) == pl.lit(val))
        if ands:
            conj = functools.reduce(operator.and_, ands)
            ors.append(conj)
    if not ors:
        return None
    disj = functools.reduce(operator.or_, ors)
    return ~disj


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
            except Exception as e:
                log.debug(f"Failed to create S3 directory {fs_path_prefix}: {e}")

        try:
            for info in fs.get_file_info(pa_fs.FileSelector(fs_path_prefix, recursive=True)):
                if info.type == pa_fs.FileType.File and info.path.endswith(".parquet"):
                    rel = Path(info.path[len(fs_path_prefix) + 1 :])
                    existing_parts.add(rel.with_suffix("").as_posix())
        except Exception:
            # Directory doesn't exist yet
            pass

    return existing_parts


def _is_custom_partition_format(
    template_for_metadata: Optional[str],
    time_unit: Optional[Literal["daily", "monthly", "yearly"]],
    date_column: Optional[str],
) -> bool:
    """Return True if a non-default partition template is used.

    A template is considered custom when it is provided and does not match the
    default for the given `time_unit` (e.g., `$year-$month-$day` for `daily`).
    """
    if not (date_column and template_for_metadata and time_unit and time_unit in _PARTITION_FORMATS):
        return False

    default_format = _PARTITION_FORMATS[time_unit]["default"]
    return template_for_metadata != default_format


def _parse_custom_partition_files(
    existing_files: set[str],
    all_partition_cols: List[str],
    date_column: str,
    partition_schema_dict: dict,
) -> pl.DataFrame:
    """Parse existing partition file keys using a custom template and reconstruct dates.

    - Handles extra partition columns in `key=value` or plain formats.
    - Extracts date components from path segments and builds a Date column via Polars.
    - Returns a DataFrame with the expected partition schema.
    """
    extra_cols = [col for col in all_partition_cols if col != date_column]

    # Build data for all columns including separate date components
    all_data = []
    date_years = []
    date_months = []
    date_days = []

    for file_path in existing_files:
        path_parts = file_path.split("/")
        row = {}

        # Handle extra columns (assume they come first)
        for i, col in enumerate(extra_cols):
            if i < len(path_parts):
                part = path_parts[i]
                # Handle key=value format
                if "=" in part:
                    row[col] = part.split("=", 1)[1]
                else:
                    row[col] = part

        # Extract date components from remaining path parts
        remaining_parts = path_parts[len(extra_cols) :]
        date_components = {"year": 1970, "month": 1, "day": 1}  # Defaults

        for part in remaining_parts:
            if "=" in part:
                key, value = part.split("=", 1)
                key_lower = key.lower()
                try:
                    if any(pattern in key_lower for pattern in ["year", "y"]):
                        date_components["year"] = int(value)
                    elif any(pattern in key_lower for pattern in ["month", "m"]):
                        date_components["month"] = int(value)
                    elif any(pattern in key_lower for pattern in ["day", "d"]):
                        date_components["day"] = int(value)
                except ValueError:
                    # Skip invalid numeric values
                    continue

        # Store row data and date components
        all_data.append(row)
        date_years.append(date_components["year"])
        date_months.append(date_components["month"])
        date_days.append(date_components["day"])

    if not all_data:
        return pl.DataFrame(schema=partition_schema_dict)

    # Create DataFrame with extra columns
    if extra_cols:
        extra_data = {col: [row.get(col) for row in all_data] for col in extra_cols}
        df = pl.DataFrame(extra_data)
    else:
        df = pl.DataFrame([{}] * len(all_data))

    # Use Polars to construct the date column from components
    df = df.with_columns([pl.Series("__year", date_years), pl.Series("__month", date_months), pl.Series("__day", date_days)])

    # Construct date using Polars date function
    df = df.with_columns(pl.date("__year", "__month", "__day").alias(date_column)).drop(["__year", "__month", "__day"])

    return df.cast(partition_schema_dict, strict=False)


def _construct_partitions_df_from_existing(
    existing_files: set[str],
    schema: pl.Schema,
    all_partition_cols: List[str],
    template_for_metadata: Optional[str],
    date_column: Optional[str] = None,
    time_unit: Optional[Literal["daily", "monthly", "yearly"]] = None,
) -> pl.DataFrame:
    """Construct a partitions DataFrame from existing keys and optional templates.

    If a custom template is provided, delegates to `_parse_custom_partition_files`.
    Otherwise, assumes partition columns map to path segments directly.
    """
    partition_schema_dict = {k: schema[k] for k in all_partition_cols}

    if not existing_files:
        return pl.DataFrame(schema=partition_schema_dict)

    # Check if we have a custom partition format
    is_custom_format = _is_custom_partition_format(template_for_metadata, time_unit, date_column)

    if is_custom_format and date_column is not None:
        # For custom formats, delegate to specialized parsing function
        # date_column is guaranteed to be non-None when is_custom_format is True
        return _parse_custom_partition_files(existing_files, all_partition_cols, date_column, partition_schema_dict)

    else:
        # Default case: partition structure matches column names directly
        raw_data = [part.split("/") for part in existing_files]
        df = pl.DataFrame(raw_data, schema={k: pl.Utf8 for k in all_partition_cols}, orient="row")

        # Parse date column if needed for default formats
        if date_column and template_for_metadata and date_column in all_partition_cols:
            # Convert `$year/$month/$day` placeholders to strftime using helper
            strftime_fmt = _strftime_from_template(template_for_metadata)
            df = df.with_columns(pl.col(date_column).str.strptime(pl.Date, strftime_fmt))

        return df.cast(partition_schema_dict, strict=False)


def _get_expected_partitions_df(
    pred: Optional[pl.Expr],
    date_column: Optional[str],
    extra_cols: List[str],
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
    date_column: Optional[str],
    extra_cols: List[str],
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

    except Exception as e:
        log.debug(f"DNF enumeration failed: {e}, returning unconstrained placeholder")
        # Even on failure, return a DataFrame with None placeholders rather than failing
        return pl.DataFrame([dict.fromkeys(all_partition_cols)], schema=partition_schema)


def _process_dnf_clause_to_partitions(
    clause: DNFClause,
    date_column: Optional[str],
    extra_cols: List[str],
    time_unit: Literal["daily", "monthly", "yearly"],
    schema: pl.Schema,
) -> Optional[pl.DataFrame]:
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

        if not constraints:
            # Unconstrained column - use None as placeholder
            column_value_sets[col] = {None}
        elif date_column and col == date_column:
            col_df = _get_date_partitions_from_constraints(constraints, date_column, time_unit, schema)
            if col_df.is_empty():
                # Contradiction in date constraints
                return pl.DataFrame(schema=partition_schema)
            column_value_sets[col] = set(col_df[col].to_list())
        else:
            col_df = _get_column_values_from_constraints(constraints, col, schema)
            if col_df.is_empty():
                # Contradiction in column constraints
                return pl.DataFrame(schema=partition_schema)
            column_value_sets[col] = set(col_df[col].to_list())

    # Generate cross product of all column values (including None placeholders)
    import itertools

    all_combinations = itertools.product(*[column_value_sets[col] for col in all_partition_cols])

    # Convert combinations to DataFrame rows
    rows = []
    for combination in all_combinations:
        row = dict(zip(all_partition_cols, combination))
        rows.append(row)

    return pl.DataFrame(rows, schema=partition_schema)


def _get_date_partitions_from_constraints(
    constraints: List[tuple[str, Any]],
    date_column: str,
    time_unit: Literal["daily", "monthly", "yearly"],
    schema: pl.Schema,
) -> pl.DataFrame:
    """Convert date constraints to partitions with appropriate time unit widening."""
    df_schema = {date_column: schema[date_column]}
    if not constraints:
        return pl.DataFrame(schema=df_schema)

    analyzer = ColumnConstraintAnalyzer(date_column)
    for op, val in constraints:
        analyzer.update_from_predicate(op, val)

    if analyzer.has_contradiction(schema=schema):
        return pl.DataFrame(schema=df_schema)

    valid_dates = _extract_and_widen_dates(analyzer, time_unit)
    if not valid_dates:
        # If no finite dates can be enumerated (e.g., one-sided bound), return empty.
        # Upstream fallback is handled in source_generator.
        return pl.DataFrame(schema=df_schema)

    return pl.DataFrame({date_column: valid_dates}, schema=df_schema)


def _get_column_values_from_constraints(
    constraints: List[tuple[str, Any]],
    column: str,
    schema: pl.Schema,
) -> pl.DataFrame:
    """Convert column constraints to valid values."""
    if not constraints:
        return pl.DataFrame(schema={column: schema[column]})

    analyzer = ColumnConstraintAnalyzer(column)
    for op, val in constraints:
        analyzer.update_from_predicate(op, val)

    if analyzer.has_contradiction(schema=schema):
        return pl.DataFrame(schema={column: schema[column]})

    valid_values = _extract_finite_values(analyzer, column, schema)
    if not valid_values:
        return pl.DataFrame(schema={column: schema[column]})

    return pl.DataFrame({column: list(valid_values)}, schema={column: schema[column]})


def _extract_and_widen_dates(
    analyzer: ColumnConstraintAnalyzer,
    time_unit: Literal["daily", "monthly", "yearly"],
) -> List[date]:
    """Extract valid dates from analyzer and apply time unit widening."""
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
    if analyzer.exclusion_values:
        for val in analyzer.exclusion_values:
            date_val = _convert_to_date(val)
            if date_val:
                dates.discard(_widen_date(date_val, time_unit))

    return sorted(dates)


def _convert_to_date(val: Any) -> Optional[date]:
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
    max_partitions: Optional[int] = None,
) -> Set[date]:
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

    while current <= max_widened and (max_partitions is None or len(dates) < max_partitions):
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
) -> Optional[Set[Any]]:
    """Extract finite set of valid values, or None if not possible."""
    if analyzer.exact_values:
        return analyzer.exact_values - analyzer.exclusion_values

    if analyzer.inclusion_set is not None:
        return analyzer.inclusion_set - analyzer.exclusion_values

    # For bounded integer ranges, enumerate small ranges
    if analyzer.min_bound and analyzer.max_bound:
        col_type = schema.get(column)
        if col_type and col_type.is_integer():
            return _enumerate_integer_range(analyzer)

    return None


def _enumerate_integer_range(analyzer: ColumnConstraintAnalyzer, max_size: Optional[int] = None) -> Optional[Set[int]]:
    """Enumerate integer range if it's small enough."""
    if not analyzer.min_bound or not analyzer.max_bound:
        return None
    min_val, min_inclusive = analyzer.min_bound
    max_val, max_inclusive = analyzer.max_bound

    if not min_inclusive:
        min_val += 1
    if not max_inclusive:
        max_val -= 1

    if (max_size is not None) and (max_val - min_val + 1 > max_size):
        return None

    values = set(range(min_val, max_val + 1))
    return values - analyzer.exclusion_values


def _compute_clipped_expected_partitions(
    predicate_interval: Interval,
    existing_parts_df: pl.DataFrame,
    date_column: str,
    extra_cols: List[str],
    time_unit: Literal["daily", "monthly", "yearly"],
    schema: pl.Schema,
) -> Optional[pl.DataFrame]:
    """Compute expected partitions for unbounded queries, clipped to existing data bounds.

    Uses the predicate's interval (from convert_expr_to_datetime_range) to determine bounds.
    - For unbounded above (lower=X, upper=+inf): Enumerate from X to max(existing)
    - For unbounded below (lower=-inf, upper=Y): Enumerate from min(existing) to Y

    Returns None if:
    - The interval is empty or fully bounded (use normal enumeration)
    - The interval is fully unbounded on both sides
    - No existing partitions to clip to
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

    # Get existing date bounds
    existing_dates = existing_parts_df[date_column].drop_nulls()
    if existing_dates.is_empty():
        return None

    existing_min = existing_dates.min()
    existing_max = existing_dates.max()

    # Compute clipped bounds
    if is_unbounded_above:
        # date >= X: use predicate's lower bound, clip upper to existing_max
        lower_val = predicate_interval.lower
        # Convert datetime to date if needed
        predicate_min = lower_val.date() if isinstance(lower_val, datetime.datetime) else lower_val
        existing_max_date = existing_max.date() if isinstance(existing_max, datetime.datetime) else existing_max
        # existing_max_date must be a date at this point
        if not isinstance(existing_max_date, date):
            return None
        clipped_min = _widen_date(predicate_min, time_unit)
        clipped_max = _widen_date(existing_max_date, time_unit)
    else:
        # date <= Y: clip lower to existing_min, use predicate's upper bound
        upper_val = predicate_interval.upper
        # Convert datetime to date if needed
        predicate_max = upper_val.date() if isinstance(upper_val, datetime.datetime) else upper_val
        existing_min_date = existing_min.date() if isinstance(existing_min, datetime.datetime) else existing_min
        # existing_min_date must be a date at this point
        if not isinstance(existing_min_date, date):
            return None
        clipped_min = _widen_date(existing_min_date, time_unit)
        clipped_max = _widen_date(predicate_max, time_unit)

    # Don't enumerate if clipped range is invalid
    if clipped_min > clipped_max:
        return None

    # Enumerate the clipped date range
    dates: Set[date] = set()
    current = clipped_min
    while current <= clipped_max:
        dates.add(current)
        current = _step_date(current, time_unit)

    if not dates:
        return None

    # Build partitions DataFrame
    all_partition_cols = extra_cols + [date_column]
    partition_schema = {col: schema[col] for col in all_partition_cols}

    # For extra columns, cross product with the clipped dates
    if extra_cols:
        extra_combos = existing_parts_df.select(extra_cols).unique().to_dicts()
        if not extra_combos:
            return pl.DataFrame({date_column: sorted(dates)}, schema={date_column: schema[date_column]})

        rows = []
        for combo in extra_combos:
            for d in dates:
                row = combo.copy()
                row[date_column] = d
                rows.append(row)
        return pl.DataFrame(rows, schema=partition_schema)
    else:
        return pl.DataFrame({date_column: sorted(dates)}, schema={date_column: schema[date_column]})


def _generate_write_predicate_from_partitions_df(
    partitions_df: pl.DataFrame,
    date_column: Optional[str],
    extra_cols: List[str],
    time_unit: Literal["daily", "monthly", "yearly", "null"],
) -> Optional[pl.Expr]:
    """Generate a predicate that efficiently covers all partitions in the DataFrame.
    There is an inherent trade-off here:
    - If we list the partitions explicitly and we have many, we may end up with a very predicate that might segfault or cause poor performance.
    - If we perform a cross-product of all partitions, we may be querying for much more data than necessary.

    Currently, we:
    1. Groups partitions by contiguous date ranges using portion.Interval (if date_column exists)
    2. Creates separate OR conditions for each contiguous range
    3. Combines them to minimize the filter overhead
    """
    if partitions_df.is_empty():
        return None

    # A single row with all-nulls for partition keys means "no predicate".
    if partitions_df.height == 1 and all(v is None for v in partitions_df.row(0)):
        return None

    # Handle case where there's no date column - just use extra columns
    if not date_column:
        # Without date column, create simple predicates for extra columns
        all_extra_predicates = []

        for row_dict in partitions_df.to_dicts():
            row_predicates = []
            for col in extra_cols:
                val = row_dict.get(col)
                if val is not None:
                    row_predicates.append(pl.col(col) == val)

            if row_predicates:
                if len(row_predicates) == 1:
                    all_extra_predicates.append(row_predicates[0])
                else:
                    all_extra_predicates.append(functools.reduce(operator.and_, row_predicates))

        if not all_extra_predicates:
            return None
        elif len(all_extra_predicates) == 1:
            return all_extra_predicates[0]
        else:
            return functools.reduce(operator.or_, all_extra_predicates)

    # Step 1: Convert each partition row to an interval and collect associated partition values
    partition_intervals: List[tuple[Interval, dict[str, Any]]] = []
    for row_dict in partitions_df.to_dicts():
        date_val = row_dict.get(date_column)
        if date_val is None:
            interval = portion.open(-portion.inf, portion.inf)

        else:
            # Create interval based on time_unit
            if time_unit == "daily":
                start_date = date_val
                end_date = date_val + timedelta(days=1)  # End exclusive
            elif time_unit == "monthly":
                start_date = date_val.replace(day=1)
                end_date = _next_month_start(start_date)
            elif time_unit == "yearly":
                start_date = date(date_val.year, 1, 1)
                end_date = date(date_val.year + 1, 1, 1)
            else:
                raise ValueError(f"Unsupported time_unit: {time_unit}")

            # Create interval (left-closed, right-open to match the logic)
            start_date = _convert_to_date(start_date)
            end_date = _convert_to_date(end_date)
            if end_date is None:
                end_date = portion.inf
            interval = portion.closedopen(start_date, end_date)

        # Collect non-null extra column values for this partition
        extra_values = {col: row_dict[col] for col in extra_cols if row_dict.get(col) is not None}

        partition_intervals.append((interval, extra_values))

    if not partition_intervals:
        return None

    # Step 2: Group partitions by overlapping/adjacent intervals
    # We'll merge intervals and collect all partition values that fall within each merged interval
    merged_groups: List[tuple[Interval, List[dict[str, Any]]]] = []

    # Sort by interval start for merging
    partition_intervals.sort(key=lambda x: (x[0].lower, x[0].upper))

    current_interval = partition_intervals[0][0]
    current_partitions = [partition_intervals[0][1]]

    for interval, partition_values in partition_intervals[1:]:
        # Check if this interval overlaps or is adjacent to current_interval
        if current_interval.overlaps(interval) or current_interval.adjacent(interval):
            # Merge intervals and combine partition values
            current_interval = current_interval | interval
            current_partitions.append(partition_values)
        else:
            # Save current group and start new one
            merged_groups.append((current_interval, current_partitions))
            current_interval = interval
            current_partitions = [partition_values]

    # Don't forget the last group
    merged_groups.append((current_interval, current_partitions))

    # Step 3: Create OR conditions for each merged group
    group_predicates = []

    for merged_interval, partition_list in merged_groups:
        # Create date range predicate for this interval
        # Convert portion interval bounds back to date objects
        start_date = merged_interval.lower
        end_date = merged_interval.upper
        cur_date_expr = _convert_atomic_interval_to_polars_expr(merged_interval, date_column)

        # Collect all unique values for extra columns within this interval
        values_per_column: dict[str, set[Any]] = {col: set() for col in extra_cols}
        for partition_values in partition_list:
            for col, val in partition_values.items():
                if val is not None:
                    values_per_column[col].add(val)

        # Create extra column predicates
        extra_preds = []
        for col, values in values_per_column.items():
            if len(values) == 0:
                continue
            elif len(values) == 1:
                extra_preds.append(pl.col(col) == next(iter(values)))
            else:
                extra_preds.append(pl.col(col).is_in(list(values)))

        # Combine date predicate with extra column predicates
        if extra_preds:
            group_pred = cur_date_expr
            for extra_pred in extra_preds:
                group_pred = group_pred & extra_pred if group_pred is not None else extra_pred
        else:
            group_pred = cur_date_expr
        if group_pred is not None:
            group_predicates.append(group_pred)

    # Step 4: Combine all group predicates with OR
    if len(group_predicates) == 0:
        return None
    elif len(group_predicates) == 1:
        return group_predicates[0]
    else:
        return functools.reduce(operator.or_, group_predicates)


def _schema_from_cache(
    cache_path: str,
    storage_opts: Optional[dict],
    credential_provider: Optional[pl.CredentialProviderAWS],
) -> Optional[pl.Schema]:
    """Get the schema from an existing cache."""
    uri = f"{cache_path.rstrip('/')}/**/*.parquet"
    try:
        kwargs = {"storage_options": storage_opts} if storage_opts else {}
        if credential_provider:
            kwargs["credential_provider"] = credential_provider

        lf = pl.scan_parquet(uri, n_rows=0, **kwargs)
        return lf.collect_schema()
    except Exception:
        log.debug("Could not infer schema from cache; this is the first write")
        return None


def _next_month_start(d: datetime.date) -> datetime.date:
    """Get the first day of the next month."""
    if d.month == 12:
        return datetime.date(d.year + 1, 1, 1)
    return datetime.date(d.year, d.month + 1, 1)


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

    # Replace placeholders with strftime codes
    strftime_fmt = partition_format
    for ph, sfmt in _STRFTIME_MAP.items():
        strftime_fmt = strftime_fmt.replace(f"${ph}", sfmt)

    return strftime_fmt, tmpl.template


def _build_write_plan(
    *,
    cache_mode: CacheMode,
    partition_info: PartitionInfo,
    date_column: Optional[str],
    predicate: Optional[pl.Expr],
    effective_time_unit: Literal["daily", "monthly", "yearly", "null"],
) -> WritePlan:
    """Construct a write plan based on cache mode and enumerability.

    Consolidates special-case logic for one-sided/unbounded queries and unconstrained cases.
    """
    need_write = cache_mode in {CacheMode.CACHE, CacheMode.REBUILD}
    filter_predicate: Optional[pl.Expr] = None

    if not need_write:
        return WritePlan(False, None, False)

    if cache_mode == CacheMode.REBUILD:
        return WritePlan(True, None, True)

    enum = partition_info.enumerability

    if enum == Enumerability.UNCONSTRAINED:
        # If we have partition columns, always query upstream (new partitions might exist)
        # Filter out rows that would fall into already-cached partitions
        if partition_info.join_cols:
            not_existing = _build_not_existing_partitions_pred(partition_info.existing_parts_df, partition_info.join_cols)
            return WritePlan(True, not_existing, False)
        # No partition columns - single global partition, trust cache once written
        if partition_info.existing_parts_df.is_empty():
            return WritePlan(True, None, False)
        return WritePlan(False, None, False)

    if enum == Enumerability.UNBOUNDED:
        restricted = (
            restrict_expr_to_columns(predicate, set(partition_info.join_cols)) if predicate is not None and partition_info.join_cols else None
        )
        not_existing = _build_not_existing_partitions_pred(partition_info.existing_parts_df, partition_info.join_cols)
        combined = restricted & not_existing if restricted is not None and not_existing is not None else (restricted or not_existing)
        return WritePlan(True, combined, False)

    if not partition_info.expected_parts_df.is_empty():
        extra_cols = [c for c in partition_info.join_cols if c != (date_column or "")]
        filter_predicate = _generate_write_predicate_from_partitions_df(
            partitions_df=partition_info.expected_parts_df,
            date_column=date_column,
            extra_cols=extra_cols,
            time_unit=effective_time_unit,
        )
    return WritePlan(True, filter_predicate, True)


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
    date_column: Optional[str],
    predicate: Optional[pl.Expr],
    time_unit_dir: str,
    template_for_metadata: Optional[str],
    effective_time_unit: Literal["daily", "monthly", "yearly", "null"],
) -> ReadPlan:
    """Construct a read plan for reading from cache.

    After writes complete, we just read from cache - no upstream merge needed.

    Returns a ReadPlan with:
    - use_paths: explicit cache file paths (None = use glob pattern)
    """
    enum = partition_info.enumerability
    join_cols = partition_info.join_cols
    extra_cols = [c for c in join_cols if c != (date_column or "")]

    def build_cache_paths(parts_df: pl.DataFrame) -> Optional[list[str]]:
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


def _compute_partitions_to_write(part_info: PartitionInfo, existing_parts_df: pl.DataFrame) -> pl.DataFrame:
    """Compute which partitions need to be written (expected - existing)."""
    if existing_parts_df.is_empty() or not part_info.join_cols:
        return part_info.expected_parts_df
    return part_info.expected_parts_df.join(existing_parts_df, on=part_info.join_cols, how="anti")


def _build_final_scan(
    cache_scan: pl.LazyFrame,
    predicate: Optional[pl.Expr],
    with_columns: Optional[List[str]],
    n_rows: Optional[int],
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
    template_for_metadata: Optional[str],
    extra_cols: List[str],
    user_metadata: Optional[dict] = None,
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
    self_or_fn: Union[pl.LazyFrame, Callable[[], pl.LazyFrame]],
    cache_path: Union[str, Path],
    date_column: Optional[str] = None,
    *,
    time_unit: Literal["daily", "monthly", "yearly"] = "monthly",
    partition_format: Optional[str] = None,
    cache_mode: CacheMode = CacheMode.CACHE,
    aws_profile: Optional[str] = None,
    write_kwargs: Optional[dict] = None,
    read_kwargs: Optional[dict] = None,
    extra_partition_cols: Optional[Union[str, List[str]]] = None,
    schema: Optional[pl.Schema] = None,
    write_bounding_columns: Optional[List[str]] = None,
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

    Returns:
        pl.LazyFrame: If the cache has all data: a LazyFrame reading from the cache.
            Otherwise: the original LazyFrame with data written to cache.

    Notes:
        **Query Classification (Enumerability)**

        Queries are classified based on whether partition keys can be enumerated from the predicate:

        - **FINITE**: Partition keys can be fully enumerated (e.g., ``date.is_between(start, end)``).
          Writes missing partitions, creates empty files for gaps, reads from enumerated paths.
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
    source_lf: Optional[pl.LazyFrame]
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
        with_columns: Optional[List[str]],
        predicate: Optional[pl.Expr],
        n_rows: Optional[int],
        batch_size: Optional[int],
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
        )

        written_parts: set[str] = set()
        is_finite = part_info.enumerability == Enumerability.FINITE
        # For REBUILD mode, always write all expected partitions (overwriting existing)
        # For CACHE mode, only write partitions that don't already exist
        if cache_mode == CacheMode.REBUILD:
            partitions_to_write_df = part_info.expected_parts_df
        elif is_finite:
            partitions_to_write_df = _compute_partitions_to_write(part_info, existing_parts_df)
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
            if not partitions_to_write_df.is_empty():
                # Narrow the write to only the partitions that are actually missing
                extra_cols = [c for c in part_info.join_cols if c != (date_column or "")]
                write_filter = _generate_write_predicate_from_partitions_df(
                    partitions_df=partitions_to_write_df,
                    date_column=date_column,
                    extra_cols=extra_cols,
                    time_unit=effective_time_unit,
                )
                if write_filter is not None:
                    lf_to_write = lf_to_write.filter(write_filter)
            else:
                if write_plan.filter_predicate is not None:
                    lf_to_write = lf_to_write.filter(write_plan.filter_predicate)

            if write_bounding_columns and predicate is not None:
                bounding_pred = restrict_expr_to_columns(predicate, set(write_bounding_columns))
                if bounding_pred is not None:
                    lf_to_write = lf_to_write.filter(bounding_pred)

            key_parts = [pl.col("keys").struct.field(f"__piot_key_{c}__").cast(pl.Utf8) for c in _extra_cols]
            if DATE_KEY_ALIAS is not None:
                key_parts.append(pl.col("keys").struct.field(DATE_KEY_ALIAS).cast(pl.Utf8))

            key_names = [f"__piot_key_{c}__" for c in _extra_cols]
            if DATE_KEY_ALIAS is not None:
                key_names.append(DATE_KEY_ALIAS)

            # Handle single partition case - add the dummy key we created earlier
            if not key_parts and not _extra_cols and date_column is None:
                key_parts.append(pl.col("keys").struct.field("__piot_key_single__").cast(pl.Utf8))
                key_names.append("__piot_key_single__")

            # Track written partition keys via file_path callback
            tracked_keys: set[str] = set()

            def _file_path_callback(ctx):
                """Generate file path for each partition and track written keys."""
                # Handle API differences between PartitionBy (new) and PartitionByKey (old)
                if hasattr(ctx, "partition_keys"):
                    # PartitionBy: ctx.partition_keys is a single-row DataFrame
                    key_values = [unquote(str(v)) for v in ctx.partition_keys.row(0)]
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
            if len(partitions_to_write_df) > _MAX_PARTITIONED_SINK_PARTITIONS:
                log.debug(
                    "Writing partitions sequentially because %d partition(s) exceed limit %d",
                    len(partitions_to_write_df),
                    _MAX_PARTITIONED_SINK_PARTITIONS,
                )
                tracked_keys.update(
                    _write_partitioned_lf_sequentially(
                        lf_to_write,
                        key_exprs=key_exprs,
                        key_names=key_names,
                        time_unit_dir=time_unit_dir,
                        metadata=cache_metadata,
                        storage_options=polars_opts,
                        credential_provider=credential_provider,
                        write_kwargs=write_kwargs,
                    )
                )
            else:
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
                # We get rid of rows with any nulls
                # since we do not accept Nulls in partition keys
                nulls_dropped_expected_df = source_expected_df.drop_nulls()
                if not nulls_dropped_expected_df.is_empty():
                    expected_keys = set([unquote(p) for p in nulls_dropped_expected_df.select(pl.concat_str(key_exprs, separator="/")).to_series()])
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

        # Read results via centralized plan
        # Refresh existing partitions snapshot after writes for accurate read planning
        existing_parts = _extract_existing_files(fs, fs_path_prefix)
        existing_parts_df = _construct_partitions_df_from_existing(
            existing_files=existing_parts,
            schema=schema,
            all_partition_cols=all_partition_cols,
            template_for_metadata=template_for_metadata,
            date_column=date_column,
            time_unit=time_unit,
        )
        part_info_read = _build_partition_info(
            expected_parts_df=expected_parts_df,
            existing_parts_df=existing_parts_df,
            date_column=date_column,
            extra_cols=_extra_cols,
            predicate=predicate,
            time_unit=time_unit if date_column else None,
            schema=schema,
        )
        read_plan = _build_read_plan(
            partition_info=part_info_read,
            date_column=date_column,
            predicate=predicate,
            time_unit_dir=time_unit_dir,
            template_for_metadata=template_for_metadata,
            effective_time_unit=effective_time_unit,
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

    return register_io_source_with_is_pure(io_source=source_generator, schema=schema, validate_schema=False)
