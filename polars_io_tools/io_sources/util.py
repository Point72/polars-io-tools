import functools
import logging
import os
import socket
from datetime import date, datetime, timedelta
from graphlib import TopologicalSorter
from typing import Iterator, List, NamedTuple, Optional, Tuple, Union
from urllib.parse import parse_qs, urlparse

import polars as pl
import portion
from packaging import version

from .._compat import POLARS_HAS_COLLECT_BATCHES

log = logging.getLogger(__name__)

__all__ = (
    "collect_lf_in_io_source",
    "filter_no_pushdown",
    "extract_description_block",
    "inject_description_block",
    "register_io_source_with_is_pure",
    "wrap_io_source_with_error_catching",
    "with_columns_topo",
)


def collect_lf_in_io_source(
    lf: pl.LazyFrame,
    batch_size: Optional[int],
) -> Iterator[pl.DataFrame]:
    """Collect a LazyFrame from inside a ``register_io_source`` callback.

    Yields a single DataFrame when ``batch_size`` is ``None`` and one or more DataFrames of approximately ``batch_size`` rows otherwise.

    This helper centralizes the streaming-vs-non-streaming choice for io_source callbacks. It avoids ``LazyFrame.collect_batches`` whenever the polars
    thread pool only has a single worker, because calling ``collect_batches`` from inside an io_source callback under ``POLARS_MAX_THREADS=1`` deadlocks
    polars' Rayon scheduler when the outer plan involves a join with a pushed-down predicate. See
    ``polars_io_tools/tests/io_sources/test_single_thread_deadlock.py`` for the regression test that pins this behavior.

    When the fallback path is taken, ``LazyFrame.collect`` materializes the whole frame and ``DataFrame.iter_slices`` yields zero-copy views. Callers
    that need each yielded DataFrame to be a single contiguous chunk should apply ``.rechunk()`` themselves on the yielded values.

    The fallback is *not* memory-equivalent to ``collect_batches``: it materializes the inner LazyFrame in full before yielding the first batch,
    whereas ``collect_batches`` only holds one batch live at a time. The fallback's peak extra memory is therefore the size of the fully materialized
    inner frame; ``collect_batches``' peak is roughly ``batch_size`` rows. The regression is only meaningful when the inner frame is much larger than
    ``batch_size``; for typical io_sources whose upstream already streams in row-bounded chunks (e.g. tickstore returns ~100k-row gRPC messages) the
    inner frame is comparable to ``batch_size`` and the difference is negligible.

    Note that the ``thread_pool_size() > 1`` gate is necessary but not provably sufficient: with multiple concurrent io_source callbacks the pool can
    still be saturated. It does, however, eliminate the only deadlock we have actually observed in production.
    """
    if batch_size is None:
        yield lf.collect()
        return
    if POLARS_HAS_COLLECT_BATCHES and pl.thread_pool_size() > 1:
        yield from lf.collect_batches(chunk_size=batch_size, maintain_order=True)
    else:
        yield from lf.collect().iter_slices(n_rows=batch_size)


class StorageOptions(NamedTuple):
    """Structured storage options for clarity between backends."""

    pyarrow: dict
    polars: dict
    credential_provider: Optional[pl.CredentialProviderAWS]


# On-prem object store endpoints that should have their hostnames resolved to IP addresses.
# This is done to avoid overloading DNS servers at scale when many workers connect simultaneously.
# When running large-scale distributed workloads, the DNS servers can become a bottleneck,
# so we resolve the hostname once and use the IP address directly.
_ONPREM_ENDPOINTS_TO_RESOLVE = frozenset(
    {
        "http://gridprodobs.saccap.int.:9020",
        "http://gridprodobs:9020",
    }
)


@functools.lru_cache(maxsize=16)
def _resolve_endpoint_hostname(endpoint: str) -> str:
    """Resolve the hostname in an endpoint URL to an IP address.

    This is used for on-prem object stores to avoid overloading DNS servers
    when many workers connect simultaneously at scale. Results are cached
    to avoid repeated DNS lookups.

    Args:
        endpoint (str): The endpoint URL, e.g. "http://gridprodobs:9020"

    Returns:
        str: The endpoint URL with the hostname replaced by its IP address,
            e.g. "http://10.1.2.3:9020"
    """
    parsed = urlparse(endpoint)
    hostname = parsed.hostname
    if hostname is None:
        return endpoint

    try:
        ip_address = socket.gethostbyname(hostname)
        # Reconstruct the URL with the IP address instead of the hostname
        # parsed.netloc includes host:port, so we need to replace just the host part
        if parsed.port:
            new_netloc = f"{ip_address}:{parsed.port}"
        else:
            new_netloc = ip_address

        # Rebuild the URL with the new netloc
        return f"{parsed.scheme}://{new_netloc}{parsed.path}"
    except socket.gaierror:
        log.warning("Failed to resolve hostname %s to IP address, using original endpoint", hostname)
        return endpoint


def _storage_options_for(cache_uri: str, aws_profile: str | None = None) -> StorageOptions:
    """Build storage options and credential provider for S3 access.

    Args:
        cache_uri (str): The URI of the cache, e.g. "s3://my-bucket/path/to/cache"
        aws_profile (str, optional): AWS profile name to use for credentials. If None, uses AWS_PROFILE env var or default.

    Returns:
        StorageOptions: A named tuple containing:
            - pyarrow: dict of options for PyArrow S3FileSystem (uses access_key, secret_key, endpoint_override)
            - polars: dict of options for Polars/Deltalake (uses aws_access_key_id, aws_secret_access_key, endpoint_url)
            - credential_provider: CredentialProviderAWS instance or None for local filesystems
    """
    parsed = urlparse(cache_uri)
    if parsed.scheme not in {"s3", "s3a"}:
        return StorageOptions({}, {}, None)  # local filesystem

    import boto3

    session = boto3.Session(profile_name=aws_profile)
    creds = session.get_credentials()

    # Check for endpoint in query string first, then environment variables
    endpoint = parse_qs(parsed.query).get("endpoint_override", [None])[0]
    if endpoint is None:
        endpoint = os.getenv("AWS_ENDPOINT_URL") or os.getenv("AWS_S3_ENDPOINT")

    pyarrow_opts, polars_opts = {}, {}

    if creds is not None:
        pyarrow_opts.update(
            {
                "access_key": creds.access_key,
                "secret_key": str(creds.secret_key),
                **({"session_token": str(creds.token)} if creds.token else {}),
            }
        )
        polars_opts.update(
            {
                "aws_access_key_id": creds.access_key,
                "aws_secret_access_key": str(creds.secret_key),
                **({"aws_session_token": str(creds.token)} if creds.token else {}),
            }
        )
    if session.region_name:
        # Only add this for pyarrow, as polars/deltalake does not use this key
        pyarrow_opts["region"] = session.region_name

    # Create credential provider - it may have additional endpoint info
    credential_provider = pl.CredentialProviderAWS(
        profile_name=(aws_profile or os.getenv("AWS_PROFILE")),
        _storage_options_has_endpoint_url=(endpoint is not None),
    )

    # If endpoint not already set, try to get it from the credential provider
    if endpoint is None and hasattr(credential_provider, "_storage_update_options"):
        cred_opts = credential_provider._storage_update_options()
        endpoint = cred_opts.get("endpoint_url")

    # For on-prem object stores, resolve hostname to IP address to avoid overloading DNS
    # when many workers connect simultaneously at scale
    if endpoint in _ONPREM_ENDPOINTS_TO_RESOLVE:
        endpoint = _resolve_endpoint_hostname(endpoint)

    # Set endpoint in both option dicts with appropriate keys
    if endpoint:
        pyarrow_opts["endpoint_override"] = endpoint
        polars_opts["endpoint_url"] = endpoint
        # delta-rs/object_store requires allow_http for non-HTTPS endpoints
        if endpoint.startswith("http://"):
            polars_opts["allow_http"] = "true"

    return StorageOptions(pyarrow_opts, polars_opts, credential_provider)


def extract_description_block(description: str | None, name: str) -> str | None:
    """Extract a named block from a Delta metadata description.

    Looks for a section delimited by `[name:begin]` and `[name:end]`, and
    returns the first non-empty line inside the block (trimmed). Returns None
    if the block is not found.
    """
    if not description:
        return None
    begin_tag = f"[{name}:begin]"
    end_tag = f"[{name}:end]"
    if begin_tag not in description or end_tag not in description:
        return None
    block = description.split(begin_tag, 1)[1].split(end_tag, 1)[0]
    payload = block.strip().splitlines()[0] if block.strip() else ""
    return payload or None


def inject_description_block(description: str | None, name: str, payload: str) -> str:
    """Append a named block to a Delta metadata description.

    Produces a block in the form:

    [name:begin]\n
    payload\n
    [name:end]

    and appends it to the existing description (or starts a new string).
    """
    desc = description or ""
    block = f"\n[{name}:begin]\n{payload}\n[{name}:end]"
    return desc + block


def filter_no_pushdown(
    self: pl.LazyFrame, expressions: list[pl.Expr] | pl.Expr | tuple[pl.Expr] | set[pl.Expr], _disable_optimizations: bool = False
) -> pl.LazyFrame:
    """
    Pipe to apply one or more filters without allowing predicate pushdown.

    This materializes each provided filter expression as a temporary boolean
    column, applies a combined filter on those booleans, then drops the
    temporary columns. Because the filter is expressed over derived columns,
    the optimizer will not push it below the projection that creates them,
    avoiding interference with CSE (common sub-plan elimination) and other pushdown-sensitive operations. CSE is sensitive to pushdown because as of
    polars 1.33.0, predicate pushdown is preferred by the optimizer, so pushed down
    predicates could prevent CSE (such as in a self-join with a filtered version of our LazyFrame)

    NOTE: In most cases, predicate pushdown is good and unless there's a specific reason to disable it, it should be left enabled. Thus, generally the native Polars `.filter` function on LazyFrames is preferred over this function.

    Args:
        frame: A `pl.DataFrame` or `pl.LazyFrame`.
        *filters: One or more `pl.Expr` filter expressions, or a single list/tuple of
            such expressions.

    Returns:
        Same type as `frame` (eager or lazy), with the filters applied but not
        pushed down.
    """
    if _disable_optimizations:
        return self.filter(expressions)

    if not isinstance(expressions, (list, tuple, set)):
        expressions = [expressions]
    elif isinstance(expressions, set):
        expressions = list(expressions)

    if not expressions:
        return self

    # Combine all expressions with AND into a single temporary column
    if len(expressions) == 1:
        combined_expr = expressions[0]
    else:
        combined_expr = pl.all_horizontal(expressions)

    tmp_col = "__cpl_np_combined__"
    out = self.with_columns(combined_expr.alias(tmp_col))
    out = out.filter(pl.col(tmp_col).eq(pl.lit(True)))
    out = out.drop(tmp_col)
    return out


def _format_arg_for_error(arg):
    """Format an argument for error messages, converting pl.Expr to full string representation."""
    if isinstance(arg, pl.Expr):
        return str(arg)
    return arg


def _format_args_for_error(args: tuple) -> tuple:
    """Format args tuple for error messages, converting pl.Expr instances to strings."""
    return tuple(_format_arg_for_error(arg) for arg in args)


def _format_kwargs_for_error(kwargs: dict) -> dict:
    """Format kwargs dict for error messages, converting pl.Expr instances to strings."""
    return {key: _format_arg_for_error(value) for key, value in kwargs.items()}


def wrap_io_source_with_error_catching(io_source, identifier: str = ""):
    """
    Wrap an IO source function with comprehensive error catching and reporting.

    This wrapper catches any exceptions thrown by the io_source function and
    provides detailed error information including stack traces, arguments, and
    context. Useful when working with `register_io_source` where optimizer
    rewrites can obscure the provenance of failures.

    Args:
        io_source (callable): The IO source function to wrap

    Returns:
        callable: A wrapped version of the IO source function with error catching
    """
    import functools
    import traceback

    @functools.wraps(io_source)
    def error_catching_io_source(*args, **source_kwargs):
        """Wrapper that catches and reports detailed errors from the io_source."""
        try:
            yield from io_source(*args, **source_kwargs)
        except Exception as e:
            # Build comprehensive error information
            io_source_name = getattr(io_source, "__name__", str(io_source))
            # Convert all pl.Expr instances to their full string representation
            # so error messages show complete expressions instead of truncated repr
            args_str = _format_args_for_error(args)
            kwargs_str_dict = _format_kwargs_for_error(source_kwargs)

            error_msg = f"""
=== IO SOURCE ERROR ===
Function: {io_source_name}
Identifier: {identifier}
Error Type: {type(e).__name__}
Error Message: {str(e)}

Call Arguments:
  args: {args_str}
  kwargs: {kwargs_str_dict}

Full Stack Trace:
{traceback.format_exc()}
========================
"""

            # Re-raise with enhanced context (no logging to avoid unsuppressible error logs)
            raise RuntimeError(f"IO Source '{io_source_name}' failed: {str(e)} with detailed error information:\n{error_msg}") from e

    return error_catching_io_source


def register_io_source_with_is_pure(io_source, schema, wrap_with_error_catching: bool = True, **kwargs):
    """
    Register an io source with is_pure=True if Polars version >= 1.33.1.

    Wraps `polars.io.plugins.register_io_source`, adding `is_pure=True` for
    Polars versions that support it (1.33.1+). Optionally wraps the source
    with `wrap_io_source_with_error_catching` for better diagnostics.

    Args:
        io_source (callable): The IO source function
        schema (dict, pl.Schema, or callable returning either): The schema for the IO source. May be passed as an eagerly-resolved
            ``dict`` / ``pl.Schema``, or as a zero-argument callable returning
            one of those. Passing a callable defers schema resolution until
            Polars actually needs it (typically at collect time), which avoids
            forcing ``collect_schema()`` on input LazyFrames at construction.
            Callable schemas require Polars >= 1.22.0.
        **kwargs: Additional keyword arguments to pass to register_io_source

    Returns:
        pl.LazyFrame: A LazyFrame with the registered IO source
    """
    from polars.io.plugins import register_io_source

    # Check if Polars version supports is_pure parameter (1.33.1+)
    if version.parse(pl.__version__) >= version.parse("1.33.1"):
        kwargs.setdefault("is_pure", True)

    if wrap_with_error_catching:
        io_source = wrap_io_source_with_error_catching(io_source)
    return register_io_source(io_source, schema=schema, **kwargs)


def with_columns_topo(lf: pl.LazyFrame, exprs: list[pl.Expr]) -> pl.LazyFrame:
    """
    Apply expressions to a LazyFrame in topological order, batching
    independent expressions into the same `with_columns` call to encourage
    parallel execution.

    NOTE: `with_columns_topo` currently only supports expressions with 1 output per expression (but optionally many input columns). Particularly, this means implicit or explicit selector usage is not supported.
    Implicit selector example (unsupported):
        - `pl.col("a", "b") + 1`
    Explicit selector example (unsupported):
        - `polars.selectors.numeric() + 1`

    Example:
        Simple dependency using alias:

        >>> lf = pl.LazyFrame({"b": [1, 2]})
        >>> out = with_columns_topo(
        ...     lf,
        ...     [
        ...         (pl.col("b2") + 1).alias("a2"),   # depends on b2
        ...         (pl.col("b") + 1).alias("b2"),    # base expression
        ...     ],
        ... )
        >>> out.select(["b2", "a2"]).collect()
        shape: (2, 2)
        ┌─────┬─────┐
        │ b2  ┆ a2  │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 2   ┆ 3   │
        │ 3   ┆ 4   │
        └─────┴─────┘

        Multiple-input, single-output dependency:

        >>> lf = pl.LazyFrame({"x": [1, 2, 3]})
        >>> exprs = [
        ...     (pl.col("x") + 1).alias("b"),
        ...     (pl.col("x") * 2).alias("c"),
        ...     pl.sum_horizontal(pl.col("b"), pl.col("c")).alias("d_sum"),
        ... ]
        >>> out = with_columns_topo(lf, exprs)
        >>> out.select(["b", "c", "d_sum"]).collect()
        shape: (3, 3)
        ┌─────┬─────┬───────┐
        │ b   ┆ c   ┆ d_sum │
        │ --- ┆ --- ┆ ---   │
        │ i64 ┆ i64 ┆ i64   │
        ╞═════╪═════╪═══════╡
        │ 2   ┆ 2   ┆ 4     │
        │ 3   ┆ 4   ┆ 7     │
        │ 4   ┆ 6   ┆ 10    │
        └─────┴─────┴───────┘

    Args:
        lf (pl.LazyFrame): The input LazyFrame.
        exprs (list[pl.Expr]): Expressions to add; later expressions may depend on aliases defined by earlier ones.

    Returns:
        pl.LazyFrame: The LazyFrame with expressions applied in a dependency-safe order.
    """
    # Map output name -> expression
    name_to_expr: dict[str, pl.Expr] = {}
    feature_names: set[str] = set()

    for expr in exprs:
        if expr.meta.has_multiple_outputs():
            raise ValueError(f"with_columns_topo does not support multi-output expressions, received: {str(expr)}")
        name = expr.meta.output_name()
        name_to_expr[name] = expr
        feature_names.add(name)

    # Build graph as: feature -> set(dependencies that are also features)
    # Only include intra-expression dependencies so we don't schedule base columns.
    dep_graph: dict[str, set[str]] = {}
    for name, expr in name_to_expr.items():
        upstreams = set(expr.meta.root_names())
        internal_upstreams = upstreams.intersection(feature_names)
        dep_graph[name] = internal_upstreams

    ts = TopologicalSorter(dep_graph)
    ts.prepare()

    while ts:
        ready = ts.get_ready()
        lf = lf.with_columns([name_to_expr[name] for name in ready])
        ts.done(*ready)
    return lf


def _convert_interval_to_slices(
    interval: portion.Interval, max_duration: Optional[timedelta] = None
) -> List[Union[date, datetime, Tuple[datetime, datetime]]]:
    """Convert a portion interval to a list of slices each spanning at most a day

    Args:
        interval (portion.Interval): The interval to convert.
        max_duration (Optional[timedelta]):
            The maximum duration of each slice. We will break continuous intervals into chunks of at most this duration.
            If None, then continuous intervals will not be modified.
    """
    true_dates = []
    # The intervals should already be non-overlapping
    for atomic_interval in interval:
        # # Handle singleton intervals
        upper, lower = atomic_interval.upper, atomic_interval.lower
        if lower == upper:
            true_dates.append(lower)
            continue
        if max_duration is None:
            # If no max duration, we just return the interval as is
            true_dates.append((lower, upper))
            continue

        current = lower
        # We do not have to extend again, since we already handled that earlier.
        # Now break the interval into chunks
        while current < upper:
            next_day = current + max_duration
            end = min(next_day, upper)
            true_dates.append((current, end))
            current = end

    # Sort the results
    true_dates.sort(key=lambda x: x[0] if isinstance(x, tuple) else x)
    return true_dates


def _extend_interval(interval: portion.Interval, column_type: pl.DataType) -> portion.Interval:
    """Extend closed intervals based on the polars DataType. Since TickStore slice bounds are exclusive, if we want to be inclusive of the end, we must over-subscribe the smallest interval we are able to."""

    def _internal_apply(atomic_interval: portion.Interval) -> portion.Interval:
        if atomic_interval.right != portion.CLOSED or atomic_interval.upper == portion.inf:
            return atomic_interval  # return as unchanged
        upper_bound = atomic_interval.upper
        if column_type == pl.Date:
            upper_bound += timedelta(days=1)
        else:
            time_unit = getattr(column_type, "time_unit", None)
            if time_unit is None or time_unit == "ms":
                # If None, we use the largest unit pl.Datetime supports
                upper_bound += timedelta(milliseconds=1)
            elif time_unit in ["us", "ns"]:
                # Python datetimes can't handle ns directly
                upper_bound += timedelta(microseconds=1)
        return atomic_interval.replace(upper=upper_bound)

    return interval.apply(_internal_apply)
