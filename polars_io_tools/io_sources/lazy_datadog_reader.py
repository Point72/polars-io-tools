import datetime
import warnings
from typing import Iterator, List, Optional

import polars as pl
import portion

from .range_visitor import convert_expr_to_datetime_range
from .util import _convert_interval_to_slices, register_io_source_with_is_pure

__all__ = ["scan_datadog", "metric_query"]


def metric_query(query: str, start: int, end: int, api_key: str, app_key: str, interval: Optional[int] = None) -> dict:
    """
    This functions queries Datadog metrics. This function replicates
    the call to `api.Metric.query`, where `api` is an initalized Python
    Datadog object from the `datadog` Python library. You may be
    interested in using this function if you want to query Datadog,
    but don't want to depend on the Datadog Python library.

    Args:
        query: Datadog query string.
        start: Unix timestamp for the start of the query window (seconds).
        end: Unix timestamp for the end of the query window (seconds).
        api_key: Datadog API key.
        app_key: Datadog application key.
        interval: Roll-up interval in seconds. If None, Datadog determines it automatically.
    """
    import requests

    base = "https://api.datadoghq.com/api/v1/query"
    params = {
        "from": start,
        "to": end,
        "query": query,
        "api_key": api_key,
        "application_key": app_key,
    }
    if interval is not None:  # Only add interval if it's explicitly provided
        params["interval"] = interval

    resp = requests.get(base, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def scan_datadog(
    query: str,
    api_key: str,
    app_key: str,
    max_chunk_duration_seconds: Optional[int] = 60 * 60 * 24,  # Default to 1 day chunks
    dd_interval: Optional[int] = None,
    additional_schema: Optional[dict] = {},
    overwrite_schema: bool = False,
) -> pl.LazyFrame:
    """
    Return a Polars `LazyFrame` that holds the result of a Datadog
    metric query. You can utilize predicate pushdown by filtering on
    the `timestamp` column (e.g., if a filter such as
    `pl.col("timestamp") < pl.datetime(2025, 1, 1)`
    is present, only the necessary timerange is requested from Datadog).
    Columns that are not present in the response are inserted as null-filled
    `Series` so the schema never changes.

    Args:
        query: Datadog query expression.
        api_key, app_key: Datadog credentials.
        max_chunk_duration_seconds: Maximum duration for each individual API request to Datadog, in seconds.
            Longer time ranges will be split into multiple requests.
            If None or 0, the entire time range is fetched in a single request (respecting Datadog limits).
            Defaults to 86400 (1 day).
        dd_interval: Datadog query roll-up interval in seconds. If None, Datadog determines
            the interval automatically. Please note that because the interval is
            influenced by the number of days being queried, it is recommended to carefully
            examine the sampling rate/interval of the data you recieve from this function.
            This will not affect your results for queries that are less than 1 day,
            but for longer queries, the interval may be different than you expect (depending
            on what Datadog chooses).

            See the following links for more details: https://docs.datadoghq.com/api/latest/metrics/?code-lang=curl#:~:text=Defaults%20to%20a%20reasonable%20interval%20for%20the%20given%20timeframe

            If you notice that the returned data is not sampled at the expected rate, you may
            want to resample the data after loading it into Polars using the built-in functions
            for downsampling/upsampling: https://docs.pola.rs/user-guide/transformations/time-series/resampling/

            Please also note that the Datadog query language's `.rollup()` function may also
            be used to specify the sampling rate of the data you are querying. See this link
            for more details: https://docs.datadoghq.com/dashboards/functions/rollup/
        additional_schema: Additional columns to add to the schema of the resulting `LazyFrame`. We try
            to be reasonably exhaustive with the default schema, but you can add more
            columns that you expect to be present in the response.
        overwrite_schema: If True, the `additional_schema` will overwrite the default schema (not
            add to it). You should probably avoid using this for safety reasons.

    Returns:
        pl.LazyFrame: A Polars LazyFrame

    Raises:
        ValueError: If the time range for the query cannot be determined from predicates
            on the 'timestamp' column.
    """
    schema = {
        "timestamp": pl.Datetime,
        "value": pl.Float64,
        "unit": pl.String,
        "query_index": pl.Int64,
        "aggr": pl.String,
        "metric": pl.String,
        "expression": pl.String,
        "scope": pl.String,
        "interval": pl.Int64,
        "length": pl.Int64,
        "start": pl.Int64,
        "end": pl.Int64,
        "display_name": pl.String,
        "user": pl.String,
        "module": pl.String,
        "function": pl.String,
        "type": pl.String,
        "kube_namespace": pl.String,
        "pod_name": pl.String,
    }

    if overwrite_schema and additional_schema is not None:
        schema = additional_schema
    elif additional_schema is not None:
        schema = schema | additional_schema

    def source_generator(
        with_columns: Optional[List[str]],
        predicate: Optional[pl.Expr],
        n_rows: Optional[int] = 100_000,
        batch_size: Optional[int] = 100_000,
    ) -> Iterator[pl.DataFrame]:
        import requests

        if n_rows is not None and n_rows <= 0:
            yield pl.DataFrame(schema=schema)
            return

        # Extract time interval from predicate
        specified_schema = {k: v for k, v in schema.items() if k in (with_columns if with_columns else schema.keys())}

        if predicate is not None:
            time_interval_from_predicate = convert_expr_to_datetime_range(predicate, "timestamp")
            if time_interval_from_predicate.empty:
                # Return empty DataFrame for contradictory ranges
                yield pl.DataFrame(schema=specified_schema)
                return
            elif time_interval_from_predicate.lower == -portion.inf:
                raise ValueError(
                    "Query start time could not be determined. "
                    "Please apply a filter with a lower bound on the 'timestamp' column "
                    "(e.g., pl.col('timestamp') > my_datetime_object)."
                )
        else:
            raise ValueError(
                "Query start time could not be determined. "
                "Please apply a filter with a lower bound on the 'timestamp' column "
                "(e.g., pl.col('timestamp') > my_datetime_object)."
            )

        # Convert interval to time slices
        max_duration = datetime.timedelta(seconds=max_chunk_duration_seconds) if max_chunk_duration_seconds else None
        time_slices = _convert_interval_to_slices(time_interval_from_predicate, max_duration)

        if not time_slices:
            yield pl.DataFrame(schema=specified_schema)
            return

        must_have_cols = set(schema.keys()) if with_columns is None else set(with_columns)
        must_have_cols.add("timestamp")
        must_have_cols.add("value")

        all_rows = []

        # Process each time slice
        for time_slice in time_slices:
            # Handle both single points and ranges
            if isinstance(time_slice, tuple):
                start_time, end_time = time_slice
            else:
                # Single point in time - use same time for start and end
                start_time = end_time = time_slice

            # Convert to timestamps (handle both date and datetime)
            if isinstance(start_time, datetime.datetime):
                start_ts_seconds = int(start_time.timestamp())
            elif isinstance(start_time, datetime.date):
                # Convert date to datetime at midnight
                start_ts_seconds = int(datetime.datetime.combine(start_time, datetime.time.min).timestamp())
            else:
                # Fallback for other types with timestamp() method
                start_ts_seconds = int(start_time.timestamp())  # type: ignore[union-attr]
            if isinstance(end_time, datetime.datetime):
                end_ts_seconds = int(end_time.timestamp())
            elif isinstance(end_time, datetime.date):
                # Convert date to datetime at midnight
                end_ts_seconds = int(datetime.datetime.combine(end_time, datetime.time.min).timestamp())
            else:
                # Fallback for other types with timestamp() method
                end_ts_seconds = int(end_time.timestamp())  # type: ignore[union-attr]

            if start_ts_seconds >= end_ts_seconds:
                continue

            try:
                results = metric_query(
                    query=query,
                    start=start_ts_seconds,
                    end=end_ts_seconds,
                    api_key=api_key,
                    app_key=app_key,
                    interval=dd_interval,
                )

            except requests.exceptions.RequestException as e:
                warnings.warn(f"API request failed for slice {start_time}-{end_time}: {e}")
                continue

            if results.get("status") == "error":
                raise ValueError(f"Datadog API returned an error: {results.get('error', 'Unknown error')}")

            for series_data in results.get("series", []):
                series_base_info = {}
                for key in ("query_index", "aggr", "metric", "expression", "scope", "interval", "length", "start", "end", "display_name"):
                    if key in must_have_cols:
                        series_base_info[key] = series_data.get(key)

                if "unit" in must_have_cols:
                    series_base_info["unit"] = str(series_data.get("unit"))

                for tag_str in series_data.get("tag_set", []):
                    if ":" in tag_str:
                        tag_key, tag_value = tag_str.split(":", 1)
                        if tag_key in must_have_cols and tag_key in schema:
                            series_base_info[tag_key] = tag_value

                for ts_milliseconds, point_value in series_data.get("pointlist", []):
                    if point_value is None:
                        continue

                    row_data = series_base_info.copy()
                    if "timestamp" in must_have_cols:
                        row_data["timestamp"] = datetime.datetime.fromtimestamp(ts_milliseconds / 1000.0, tz=datetime.timezone.utc)
                    if "value" in must_have_cols:
                        row_data["value"] = point_value

                    all_rows.append(row_data)

                    if n_rows is not None and len(all_rows) >= n_rows:
                        break
                if n_rows is not None and len(all_rows) >= n_rows:
                    break
            if n_rows is not None and len(all_rows) >= n_rows:
                break

        if not all_rows:
            yield pl.DataFrame(schema=specified_schema)
            return

        # Truncate to n_rows if needed
        rows_to_return = all_rows[:n_rows] if n_rows is not None else all_rows

        # Yield in batches of batch_size
        effective_batch_size = batch_size if batch_size is not None else len(rows_to_return)
        for idx in range(0, len(rows_to_return), effective_batch_size):
            chunk_rows = rows_to_return[idx : idx + effective_batch_size]
            df = pl.DataFrame(chunk_rows, schema=specified_schema)

            # Apply predicate if provided
            if predicate is not None:
                df = df.filter(predicate)

            yield df

    return register_io_source_with_is_pure(source_generator, schema=schema)
