import io
from datetime import datetime, timedelta
from typing import Iterator, List, Literal, Optional, Tuple, Union

import cloudpickle
import polars as pl
import portion
from tqdm import tqdm

from .range_visitor import convert_expr_to_datetime_range
from .util import _extend_interval

try:
    import ray
except ImportError:
    raise ImportError("The `execute_on_ray` function requires the `ray` package. Please install it with `pip install ray`.")


from .util import register_io_source_with_is_pure

# As we did in the `lazy_parquet_cache` module, we expose the function to users, in case
# they want to use it directly or though Polars' `.pipe()` syntax; however, the canonical
# useage is to call the function as a method on the `LazyFrame`'s `piot` namespace.
__all__ = ["execute_on_ray"]


def _partition_specs(
    date_min: datetime,
    date_max: datetime,
    unit: Literal["daily", "monthly", "yearly"],
) -> List[Tuple[datetime, datetime]]:
    """
    Returns a list of (start, end) tuples that cover the
    range. Each `end` is exclusive, i.e. [start, end).
    """
    if date_min is None or date_max is None:
        return []

    specs: List[Tuple[datetime, datetime]] = []

    if unit == "daily":
        cur = date_min.date()
        end = date_max.date()
        while cur <= end:
            nxt = cur + timedelta(days=1)
            specs.append((datetime.combine(cur, datetime.min.time()), datetime.combine(nxt, datetime.min.time())))
            cur = nxt

    elif unit == "monthly":
        cur = datetime(date_min.year, date_min.month, 1)
        end = datetime(date_max.year, date_max.month, 1)
        while cur <= end:
            nxt = datetime(cur.year + 1, 1, 1) if cur.month == 12 else datetime(cur.year, cur.month + 1, 1)
            specs.append((cur, nxt))
            cur = nxt

    elif unit == "yearly":
        for yr in range(date_min.year, date_max.year + 1):
            start = datetime(yr, 1, 1)
            end = datetime(yr + 1, 1, 1)
            specs.append((start, end))

    return specs


def _trim_partition_specs(
    specs: List[Tuple[datetime, datetime]],
    date_interval: "portion.Interval",
    column_type: pl.DataType,
) -> List[Tuple[datetime, datetime]]:
    """
    Intersect each [start, end) partition with the user's temporal interval
    and return only those with non-empty overlap.

    Both the start and end of each partition are tightened to the
    intersection.  Since ``_execute_partition`` uses ``col < end`` (open
    upper), any closed upper bound is converted to open via
    ``_extend_interval`` (the same logic used by the tickstore reader).
    """
    trimmed: List[Tuple[datetime, datetime]] = []
    for start, end in specs:
        intersection = portion.closedopen(start, end) & date_interval
        if not intersection.empty:
            extended = _extend_interval(intersection, column_type)
            trimmed.append((extended.lower, extended.upper))
    return trimmed


@ray.remote
def _execute_partition(
    plan_bytes_or_ref: Union[bytes, ray.ObjectRef],
    date_col: str,
    start: datetime,
    end: datetime,
    return_as: Literal["arrow", "ipc", "parquet"] = "arrow",
) -> bytes:
    """
    Deserialise plan, apply `[start, end)` filter, collect, return IPC bytes.
    Works wheter or not Ray has already dereferenced `plan_bytes_or_ref`.
    """
    # Resolve the reference if we have it; don't worry
    # otherwise, since we've got the raw bytes
    if isinstance(plan_bytes_or_ref, ray.ObjectRef):
        plan_bytes = ray.get(plan_bytes_or_ref)
    else:
        plan_bytes = plan_bytes_or_ref

    lf = cloudpickle.loads(plan_bytes)

    predicate = (pl.col(date_col) >= start) & (pl.col(date_col) < end)
    out = lf.filter(predicate).collect()

    if return_as == "arrow":
        return out.to_arrow()
    elif return_as == "parquet":
        buf = io.BytesIO()
        out.write_parquet(buf, use_pyarrow=True)
        return buf.getvalue()
    elif return_as == "ipc":
        buf = io.BytesIO()
        out.write_ipc(buf)
        return buf.getvalue()


def execute_on_ray(
    self: pl.LazyFrame,
    *,
    date_column: str,
    time_unit: Literal["daily", "monthly", "yearly"],
    return_as: Literal["arrow", "ipc", "parquet"] = "arrow",
    remote_options: Optional[dict] = None,
    max_concurrency: Optional[int] = 100,
) -> pl.LazyFrame:
    """
    Execute a Polars LazyFrame on an *already initialised* Ray cluster,
    distributing the work by calendar periods.

    The function returns **another** LazyFrame whose scan node is a
    custom I/O source.  No computation happens immediately; evaluation
    is triggered only when the user calls `.collect()`.

    Args:
        date_column (str): Name of the datetime column that defines the partitioning axis.
            The column must be of type `pl.Datetime` or `pl.Date`.
        time_unit ({"daily", "monthly", "yearly"}): Granularity of the split.
        return_as ({"arrow", "ipc", "parquet"}, default "arrow"): The format in which the Ray worker returns the data.
            - "arrow" returns zero-copy Arrow buffers,
            - "ipc" returns Arrow IPC buffers,
            - "parquet" returns Parquet buffers.
        remote_options (Optional[dict]): A dictionary of options for each Ray task. Please see the Ray
            documentation for details: https://docs.ray.io/en/latest/_modules/ray/remote_function.html#RemoteFunction.options
            You may wish to specify keys such as ``num_cpus``, ``num_gpus``, or
            pass environment variables like POLARS_MAX_THREADS and POLARS_ENGINE_AFFINITY
            in the runtime environment. If `None`, the task launches with standard defaults.
        max_concurrency (Optional[int], default 100): The maximum number of concurrent tasks to run. We follow this pattern from the Ray documentation:
            https://docs.ray.io/en/latest/ray-core/patterns/limit-pending-tasks.html. You may also wish to experiment
            with using resource hints to manage concurrency, although we do not recommend doing so; please see the
            following page for this alternative pattern: https://docs.ray.io/en/latest/ray-core/patterns/limit-running-tasks.html

    Returns:
        pl.LazyFrame: A new LazyFrame whose execution plan includes information about
            running execution on a Ray cluster when the user calls `.collect()`

    **NOTE:** This function requires a bounded predicate on the chosen `date_column`
    """

    if not ray.is_initialized():
        raise RuntimeError("Ray is not initialised. Please call `ray.init()` before using `execute_on_ray`.")

    remote_options = remote_options or {}
    if not isinstance(remote_options, dict):
        raise TypeError("`remote_options` must be a dict or None.")

    original_lf = self

    def source_generator(
        with_columns: Optional[List[str]],
        predicate: Optional[pl.Expr],
        n_rows: Optional[int],
        batch_size: Optional[int],
    ) -> Iterator[pl.DataFrame]:
        lf = original_lf

        requested_cols = with_columns[:] if with_columns is not None else None

        # push-downs from the optimizer
        if with_columns is not None:
            if date_column not in with_columns:
                with_columns = with_columns + [date_column]
            lf = lf.select(with_columns)

        if predicate is not None:
            lf = lf.filter(predicate)

        # derive temporal bounds from the pushed-down predicate
        if predicate is None:
            raise ValueError(f"`execute_on_ray` requires a bounded temporal predicate on column '{date_column}'.")
        date_interval = convert_expr_to_datetime_range(predicate, date_column, get_enclosure=False)

        if date_interval.empty:
            yield pl.DataFrame(schema=self.collect_schema())
            return

        if (date_interval.lower is portion.inf) or (date_interval.upper is -portion.inf):
            raise ValueError("Un-bounded temporal predicate - unable to split the work.")

        _min = date_interval.lower
        _max = date_interval.upper

        specs = _partition_specs(_min, _max, time_unit)

        date_col_type = self.collect_schema()[date_column]
        specs = _trim_partition_specs(specs, date_interval, date_col_type)

        if not specs:
            yield pl.DataFrame(schema=self.collect_schema())
            return

        plan_bytes = cloudpickle.dumps(lf)

        # create iterator of (idx, start, end) triples
        spec_iter = iter(enumerate(specs))

        future_to_idx: dict[ray.ObjectRef, int] = {}
        pending_futures: set[ray.ObjectRef] = set()

        # helper to launch one task
        def submit(idx: int, span: tuple[datetime, datetime]):
            f = _execute_partition.options(**remote_options).remote(plan_bytes, date_column, span[0], span[1], return_as)
            future_to_idx[f] = idx
            pending_futures.add(f)

        # prime up to `max_concurrency` tasks
        effective_concurrency = max_concurrency if max_concurrency is not None else len(specs)
        for _ in range(min(effective_concurrency, len(specs))):
            idx, span = next(spec_iter)
            submit(idx, span)

        pbar = tqdm(total=len(specs), desc="execute_on_ray")

        ready_parts: dict[int, pl.DataFrame] = {}
        next_to_yield = 0
        rows_yielded = 0  # needed for global `n_rows` limit

        while pending_futures or ready_parts:
            if pending_futures:
                ready_refs, _ = ray.wait(list(pending_futures), num_returns=1)
                for ref in ready_refs:
                    pending_futures.remove(ref)
                    idx = future_to_idx[ref]
                    try:
                        blob = ray.get(ref)
                    except Exception as e:
                        err_msg = (
                            f"Ray worker failed while executing partition {idx} of lazy frame.\nPolars plan for this lazy frame:\n{lf.explain()}"
                        )
                        err_msg += f"\n\nWhile running the above, received error: {e.__class__.__name__}:{e}"
                        raise RuntimeError(err_msg) from e

                    # immediately submit a new task if specs remain
                    try:
                        new_idx, new_span = next(spec_iter)
                        submit(new_idx, new_span)
                    except StopIteration:
                        pass

                    pbar.update()

                    blob = ray.get(ref)

                    df: pl.DataFrame
                    if return_as == "arrow":
                        result = pl.DataFrame(blob)
                        df = result
                    elif return_as == "parquet":
                        df = pl.read_parquet(io.BytesIO(blob))
                    elif return_as == "ipc":
                        df = pl.read_ipc(io.BytesIO(blob))
                    else:
                        raise ValueError(f"Unsupported return format: {return_as}")
                    ready_parts[idx] = df

            while next_to_yield in ready_parts:
                df = ready_parts.pop(next_to_yield)
                next_to_yield += 1

                # We need to drop the date column if the user did not originally request it.
                if requested_cols is not None and date_column not in requested_cols:
                    df = df.drop(date_column)

                if n_rows is not None:
                    remaining = n_rows - rows_yielded
                    if remaining <= 0:
                        return  # limit already met
                    if len(df) > remaining:
                        df = df.head(remaining)
                    rows_yielded += len(df)

                if batch_size is not None and len(df) > batch_size:
                    for i in range(0, len(df), batch_size):
                        yield df.slice(i, batch_size)
                else:
                    yield df

        if next_to_yield == 0:  # nothing ever yielded
            yield pl.DataFrame(schema=self.collect_schema())

        pbar.close()

    return register_io_source_with_is_pure(source_generator, schema=lambda: self.collect_schema())
