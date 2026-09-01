import logging
from collections.abc import Callable, Iterator
from datetime import date, timedelta
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

from .util import register_io_source_with_is_pure

__all__ = ("scan_synthetic_panel", "scan_synthetic_regression")

log = logging.getLogger(__name__)


# (aux_rng, group_rng, x_batch, chunk_idx, row_in_chunk) -> (mean_batch, extras_dict).
# aux_rng / group_rng are per-scan streams distinct from x/eps/w so extras stay
# batch-independent; they are also distinct from each other so that adding a decorative
# category does not shift the group draw for the same seed. row_in_chunk is the offset
# within the current chunk (0 when chunk_sizes is None), used by callers that need
# positional layout within a chunk (e.g. the panel's ``symbol = row % n_symbols``).
MeanComputer = Callable[[np.random.Generator, np.random.Generator, np.ndarray, int, int], tuple[np.ndarray, dict]]


def _parallel_standard_normal(rngs: list[np.random.Generator], shape: tuple[int, int], pool) -> np.ndarray:
    """Fill a ``shape`` array with standard-normal draws, one contiguous row-slice per generator.

    Rows are split across ``rngs`` and each slice is filled in place via ``out=`` on ``pool``;
    ``standard_normal`` releases the GIL, so the fill runs across cores. Following NumPy's
    documented multithreaded-generation pattern; slices are disjoint, so the writes never race.
    """
    out = np.empty(shape, dtype=np.float64)
    chunks = np.array_split(out, len(rngs), axis=0)
    list(pool.map(lambda rng, chunk: rng.standard_normal(out=chunk), rngs, chunks))
    return out


def _check_reserved(name: str, kind: str, *, n_features: int, n_responses: int, use_weights: bool, extra: tuple[str, ...] = ()) -> None:
    """Raise ValueError if ``name`` collides with a generated column name."""
    reserved = {f"x{i}" for i in range(n_features)} | {f"y{i}" for i in range(n_responses)} | ({"weight"} if use_weights else set()) | set(extra)
    if name in reserved:
        raise ValueError(f"{kind}={name!r} collides with a generated column name; choose another.")


def _register_source(
    *,
    n_samples: int,
    n_features: int,
    n_responses: int,
    use_weights: bool,
    weights_low: float,
    weights_high: float,
    epsilon_loc: float,
    epsilon_scale: float,
    seed: int | None,
    fetch_size: int,
    mean_computer: MeanComputer,
    extras_schema: dict,
    chunk_sizes: np.ndarray | None = None,
    n_workers: int = 1,
    explain_name: str,
    explain_detail: str | None = None,
) -> pl.LazyFrame:
    feature_cols = [f"x{i}" for i in range(n_features)]
    response_cols = [f"y{i}" for i in range(n_responses)]
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    schema_fields: dict[str, pl.DataType] = {name: pl.Float64() for name in feature_cols + response_cols}
    schema_fields.update(extras_schema)
    if use_weights:
        schema_fields["weight"] = pl.Float64()
    schema = pl.Schema(schema_fields)

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        bs = batch_size or fetch_size
        x_ss, eps_ss, w_ss, aux_ss, group_ss = np.random.SeedSequence(seed).spawn(5)
        w_rng = np.random.default_rng(w_ss)
        aux_rng = np.random.default_rng(aux_ss)
        group_rng = np.random.default_rng(group_ss)

        # x/eps are the dominant cost; optionally fill them in parallel. w/aux/group
        # stay single-stream (cheap, and already batch_size-independent). n_workers==1
        # keeps the exact serial streams, so default output is byte-for-byte unchanged.
        threaded = n_workers > 1 and n_samples > 0
        pool = None
        if threaded:
            from concurrent.futures import ThreadPoolExecutor

            # Fixed-size blocks (independent of Polars' batch_size) filled in parallel by one
            # persistent Generator per worker, so row i depends only on (seed, n_workers, fetch_size).
            block_rows = min(n_workers * fetch_size, n_samples)
            x_rngs = [np.random.default_rng(s) for s in x_ss.spawn(n_workers)]
            eps_rngs = [np.random.default_rng(s) for s in eps_ss.spawn(n_workers)]
            pool = ThreadPoolExecutor(max_workers=n_workers)
            x_block = eps_block = None
            block_pos = block_rows  # trigger a fill on the first iteration
        else:
            x_rng = np.random.default_rng(x_ss)
            eps_rng = np.random.default_rng(eps_ss)

        remaining_gen = n_samples
        remaining_deliver = n_rows if n_rows is not None else n_samples

        chunk_idx = 0
        rows_left_in_chunk = int(chunk_sizes[0]) if chunk_sizes is not None and len(chunk_sizes) > 0 else 0

        try:
            while remaining_gen > 0 and remaining_deliver > 0:
                if chunk_sizes is not None:
                    k = min(bs, rows_left_in_chunk, remaining_gen)
                    row_in_chunk = int(chunk_sizes[chunk_idx]) - rows_left_in_chunk
                else:
                    k = min(bs, remaining_gen)
                    row_in_chunk = 0
                if threaded:
                    if block_pos == block_rows:
                        x_block = _parallel_standard_normal(x_rngs, (block_rows, n_features), pool)
                        eps_block = _parallel_standard_normal(eps_rngs, (block_rows, n_responses), pool)
                        block_pos = 0
                    k = min(k, block_rows - block_pos)  # keep each batch within one fixed block
                    x_batch = x_block[block_pos : block_pos + k]
                    eps_raw = eps_block[block_pos : block_pos + k]
                    block_pos += k
                else:
                    x_batch = x_rng.standard_normal(size=(k, n_features))
                    eps_raw = eps_rng.standard_normal(size=(k, n_responses))
                if use_weights:
                    w_batch = w_rng.uniform(weights_low, weights_high, size=k)
                    eps_scale = epsilon_scale / np.sqrt(w_batch)[:, None]
                else:
                    w_batch = None
                    eps_scale = epsilon_scale
                eps_batch = eps_raw * eps_scale + epsilon_loc
                mean_batch, extras = mean_computer(aux_rng, group_rng, x_batch, chunk_idx, row_in_chunk)
                y_batch = mean_batch + eps_batch

                data: dict = {name: x_batch[:, i] for i, name in enumerate(feature_cols)}
                for i, name in enumerate(response_cols):
                    data[name] = y_batch[:, i]
                data.update(extras)
                if use_weights:
                    data["weight"] = w_batch

                df = pl.DataFrame(data)
                if predicate is not None:
                    df = df.filter(predicate)
                if with_columns is not None:
                    df = df.select(with_columns)

                if df.height > remaining_deliver:
                    df = df.head(remaining_deliver)
                remaining_deliver -= df.height
                remaining_gen -= k  # Always k, not df.height: RNG must advance by the full generated batch to keep row-i values batch-independent.

                if chunk_sizes is not None:
                    rows_left_in_chunk -= k
                    if rows_left_in_chunk == 0 and chunk_idx + 1 < len(chunk_sizes):
                        chunk_idx += 1
                        rows_left_in_chunk = int(chunk_sizes[chunk_idx])

                log.debug("scan_synthetic: yielded %d rows; remaining_gen=%d, remaining_deliver=%d", df.height, remaining_gen, remaining_deliver)
                yield df
        finally:
            if pool is not None:
                pool.shutdown(wait=False)

    return register_io_source_with_is_pure(
        source_generator, schema=schema, is_pure=seed is not None, explain_name=explain_name, explain_detail=explain_detail
    )


def _validate_common(
    *,
    n_features: int,
    n_responses: int,
    epsilon_scale: float,
    fetch_size: int,
    use_weights: bool,
    weights_low: float,
    weights_high: float,
) -> None:
    if n_features < 1:
        raise ValueError(f"n_features must be >= 1, got {n_features}")
    if n_responses < 1:
        raise ValueError(f"n_responses must be >= 1, got {n_responses}")
    if epsilon_scale < 0:
        raise ValueError(f"epsilon_scale must be >= 0, got {epsilon_scale}")
    if fetch_size < 1:
        raise ValueError(f"fetch_size must be >= 1, got {fetch_size}")
    if use_weights and not (0 < weights_low < weights_high):
        raise ValueError(f"require 0 < weights_low < weights_high, got weights_low={weights_low}, weights_high={weights_high}")


def scan_synthetic_regression(
    *,
    n_samples: int,
    n_features: int,
    n_responses: int = 1,
    use_weights: bool = False,
    weights_low: float = 0.5,
    weights_high: float = 1.5,
    betas: npt.ArrayLike | None = None,
    epsilon_loc: float = 0.0,
    epsilon_scale: float = 1.0,
    chunk_key: str | None = None,
    n_chunks: int | None = None,
    seed: int | None = None,
    n_workers: int = 1,
    fetch_size: int = 10_000,
    description: str | None = None,
) -> pl.LazyFrame:
    """
    A lazy source of synthetic linear-regression data ``Y = X @ B + E`` with Gaussian noise.

    Columns are ``x0..x{n_features-1}`` and ``y0..y{n_responses-1}``, all ``Float64``.
    When ``use_weights=True``, a ``weight`` column drawn from ``Uniform(weights_low, weights_high)`` is emitted and the noise is scaled to ``eps ~ N(loc, scale²/w)`` so a WLS fit with these weights recovers ``betas``.

    Data is batch-independent: for a fixed ``seed``, row ``i`` has the same RNG-drawn values regardless of the ``batch_size`` Polars chooses (or of any ``n_rows`` pushdown). ``y`` values are reproducible up to floating-point reassociation in the ``X @ betas`` matmul (≤1 ULP). This is what makes ``is_pure=True`` sound when a seed is provided.

    Note: With no pushdowns, the source generates ``n_samples`` rows. When a ``.head(k)`` is pushed down as ``n_rows``, the source generates as many batches as needed to deliver ``k`` post-filter rows — under a tight filter this may draw many more than ``n_samples`` internal rows before returning. A filter that never passes (e.g. ``.filter(False)``) will loop indefinitely; make sure your filter has a positive pass rate. In ``chunk_key`` mode the ceiling is the ``n_samples``-row K-chunk partition, so a tight filter can still under-deliver.

    Chunked mode: when ``chunk_key`` and ``n_chunks`` are provided, the source emits a monotonic non-decreasing ``Int64`` column with values ``0, 1, ..., n_chunks - 1``. Chunks are contiguous and rows are yielded in chunk-key order; each yielded ``pl.DataFrame`` contains rows from exactly one chunk key. This lets a downstream ``.set_sorted(chunk_key).group_by(chunk_key)`` finalize each group as the source walks past it (with ``POLARS_FORCE_SORTED_GROUP_BY=1``), keeping peak memory to one chunk. Rows are split as evenly as possible: the first ``n_samples % n_chunks`` chunks get ``ceil(n_samples/n_chunks)`` rows, the rest get ``floor(n_samples/n_chunks)`` rows. The single global ``betas`` is used for every row regardless of chunk.

    Args:
        n_samples: Total number of rows to generate. Must be >= 0.
        n_features: Number of feature columns. Must be >= 1.
        n_responses: Number of target columns. Must be >= 1. Defaults to 1.
        use_weights: If True, emits a ``weight`` column drawn from ``Uniform(weights_low, weights_high)`` and generates ``y`` under the WLS model ``Var(eps) = epsilon_scale² / w``.
        weights_low: Lower bound of the uniform draw. Must satisfy ``0 < weights_low < weights_high``. Only used when ``use_weights=True``. Defaults to 0.5.
        weights_high: Upper bound of the uniform draw. Must satisfy ``weights_low < weights_high``. Only used when ``use_weights=True``. Defaults to 1.5.
        betas: Coefficient matrix of shape ``(n_features, n_responses)``. When ``n_responses=1``, a 1-D array of shape ``(n_features,)`` is also accepted. If None, drawn from ``rng.standard_normal`` and frozen for all batches.
        epsilon_loc: Mean of the Gaussian noise. Defaults to 0.0.
        epsilon_scale: When ``use_weights=False``, the noise stddev for every row. When ``use_weights=True``, the *reference* stddev at ``w=1``; actual per-row noise is ``N(loc, (epsilon_scale/√w)²)``. Must be >= 0. Defaults to 1.0.
        chunk_key: If provided together with ``n_chunks``, emits a monotonic ``Int64`` column with this name whose values are ``0, 1, ..., n_chunks - 1``. Must not collide with a generated column name.
        n_chunks: Number of contiguous chunks to split ``n_samples`` into. Required when ``chunk_key`` is set. Must satisfy ``1 <= n_chunks <= n_samples``.
        seed: Seed for ``np.random.default_rng``. If None, uses fresh entropy per call (and the source is registered with ``is_pure=False``).
        n_workers: Number of threads used to fill the ``x``/``y`` Gaussian draws (the dominant cost) in parallel. ``1`` (default) keeps the fully serial path and its exact output. With ``n_workers > 1`` the draws are generated in fixed-size blocks split across threads, so reproducibility is keyed on ``(seed, n_workers, fetch_size)`` and remains independent of the ``batch_size`` Polars chooses; values differ from the serial path. Peak generation memory grows with ``n_workers`` (a block holds ``n_workers * fetch_size`` rows). Most useful when ``n_samples`` is large.
        fetch_size: Default number of rows generated per batch when Polars does not provide a ``batch_size``. Must be >= 1. Defaults to 10_000.
        description: Optional free-form description of this source instance, attached to its OpenTelemetry span (``explain_detail``).
    """
    _validate_common(
        n_features=n_features,
        n_responses=n_responses,
        epsilon_scale=epsilon_scale,
        fetch_size=fetch_size,
        use_weights=use_weights,
        weights_low=weights_low,
        weights_high=weights_high,
    )
    if n_samples < 0:
        raise ValueError(f"n_samples must be >= 0, got {n_samples}")
    if (chunk_key is None) != (n_chunks is None):
        raise ValueError("chunk_key and n_chunks must be provided together")
    chunk_sizes = None
    if chunk_key is not None:
        assert n_chunks is not None  # enforced by the mutual-presence check above
        if n_samples < 1 or n_chunks < 1 or n_chunks > n_samples:
            raise ValueError(f"n_chunks must satisfy 1 <= n_chunks <= n_samples (n_samples={n_samples}, n_chunks={n_chunks})")
        _check_reserved(chunk_key, "chunk_key", n_features=n_features, n_responses=n_responses, use_weights=use_weights)
        base, rem = divmod(n_samples, n_chunks)
        chunk_sizes = np.full(n_chunks, base, dtype=np.int64)
        chunk_sizes[:rem] += 1

    if betas is None:
        # Dedicated child of the root SeedSequence so β does not share bits with the
        # x/eps/w/aux/group streams spawned inside _register_source.
        betas_arr = np.random.default_rng(np.random.SeedSequence(seed).spawn(6)[5]).standard_normal(size=(n_features, n_responses))
    else:
        betas_arr = np.array(betas, dtype=np.float64)
        if betas_arr.ndim == 1 and n_responses == 1:
            betas_arr = betas_arr[:, None]
        if betas_arr.shape != (n_features, n_responses):
            raise ValueError(f"betas must have shape (n_features={n_features}, n_responses={n_responses}), got {betas_arr.shape}")

    if chunk_key is not None:

        def mean_computer(
            aux_rng: np.random.Generator, group_rng: np.random.Generator, x_batch: np.ndarray, chunk_idx: int, row_in_chunk: int
        ) -> tuple[np.ndarray, dict]:
            k = x_batch.shape[0]
            return x_batch @ betas_arr, {chunk_key: np.full(k, chunk_idx, dtype=np.int64)}

        extras_schema = {chunk_key: pl.Int64()}
    else:

        def mean_computer(
            aux_rng: np.random.Generator, group_rng: np.random.Generator, x_batch: np.ndarray, chunk_idx: int, row_in_chunk: int
        ) -> tuple[np.ndarray, dict]:
            return x_batch @ betas_arr, {}

        extras_schema = {}

    return _register_source(
        n_samples=n_samples,
        n_features=n_features,
        n_responses=n_responses,
        use_weights=use_weights,
        weights_low=weights_low,
        weights_high=weights_high,
        epsilon_loc=epsilon_loc,
        epsilon_scale=epsilon_scale,
        seed=seed,
        fetch_size=fetch_size,
        mean_computer=mean_computer,
        extras_schema=extras_schema,
        chunk_sizes=chunk_sizes,
        n_workers=n_workers,
        explain_name="scan_synthetic_regression",
        explain_detail=description,
    )


def scan_synthetic_panel(
    *,
    start_date: date,
    end_date: date,
    freq: str = "1D",
    n_symbols: int = 1,
    n_features: int,
    n_responses: int = 1,
    betas: npt.ArrayLike | None = None,
    use_weights: bool = False,
    weights_low: float = 0.5,
    weights_high: float = 1.5,
    categories: list[list[Any]] | None = None,
    group_by: tuple[str, list[Any]] | None = None,
    epsilon_loc: float = 0.0,
    epsilon_scale: float = 1.0,
    seed: int | None = None,
    n_workers: int = 1,
    fetch_size: int = 10_000,
    description: str | None = None,
) -> pl.LazyFrame:
    """
    A lazy source of synthetic panel data on a ``(date, symbol)`` grid. This generator draws independent per-row noise, uses ``x_i``/``y_i`` column names, and treats weights as a WLS variance model. Rows are yielded date-by-date so ``.set_sorted("date").group_by("date")`` streams cleanly under ``engine="streaming"``.

    Emits one row per ``(date, symbol)`` pair for each date in ``pd.bdate_range(start_date, end_date + 1 day, freq=freq, inclusive="left")``.

    Response is ``y[i] = X[i] @ betas + eps[i]`` with per-row Gaussian noise. Categories, if provided, are drawn independently per row from the given value sets — decorative dimensions, not correlated with ``y``. When ``use_weights=True``, a ``weight`` column drawn from ``Uniform(weights_low, weights_high)`` is emitted and the noise is scaled to ``eps ~ N(loc, scale²/w)`` so a WLS fit with these weights recovers ``betas``.

    Per-group coefficients: passing ``group_by=(name, values)`` promotes a column named ``name`` with per-row draws from ``values`` into a *β-driver* — ``betas`` is then interpreted as ``(n_groups, n_features, n_responses)`` where ``n_groups == len(values)``, and each row's ``y = X @ betas[group_idx]``. A ``group_by(name).agg(...ols...)`` on the collected frame recovers each group's β. The group column is separate from ``categories`` (which remain decorative) and its draw is independent of the category draws for the same seed. Values are indexed positionally, so ``values`` may contain duplicates or non-hashable entries — mapping is by index, not by value. Streaming caveat: because groups are interleaved within each date-chunk, a streaming ``group_by(name)`` requires ``.sort(name)`` first; streaming still composes cleanly when grouping by ``date``.

    Args:
        start_date: First business day emitted (inclusive).
        end_date: Last business day emitted (inclusive).
        freq: The frequency string for ``pd.bdate_range`` (e.g. ``"1D"``, ``"W"``, ``"H"``). Defaults to ``"1D"``.
        n_symbols: Number of symbols per date. Must be >= 1. Defaults to 1.
        n_features: Number of feature columns. Must be >= 1.
        n_responses: Number of target columns. Must be >= 1. Defaults to 1.
        betas: When ``group_by`` is None: coefficient matrix of shape ``(n_features, n_responses)`` (or ``(n_features,)`` if ``n_responses=1``). When ``group_by`` is set: per-group coefficient tensor of shape ``(n_groups, n_features, n_responses)`` (or ``(n_groups, n_features)`` if ``n_responses=1``), where ``n_groups == len(group_by[1])``. If None, drawn from ``rng.standard_normal`` and frozen for all rows.
        use_weights: If True, emits a ``weight`` column drawn from ``Uniform(weights_low, weights_high)`` and generates ``y`` under the WLS model ``Var(eps) = epsilon_scale² / w``.
        weights_low: Lower bound of the uniform draw. Must satisfy ``0 < weights_low < weights_high``. Only used when ``use_weights=True``. Defaults to 0.5.
        weights_high: Upper bound of the uniform draw. Must satisfy ``weights_low < weights_high``. Only used when ``use_weights=True``. Defaults to 1.5.
        categories: A list of value sets, one per emitted ``category_i`` column. e.g. ``[["A","B","C"], ["X","Y"]]`` emits ``category_0`` and ``category_1`` drawn independently per row. Categories are decorative — not correlated with ``y``. Defaults to None (no category columns).
        group_by: ``(name, values)`` pair. When set, emits a column named ``name`` whose per-row values are drawn uniformly from ``values`` (indexed positionally, so duplicates are allowed), and reinterprets ``betas`` as per-group coefficients (see above). The column name must not collide with a generated column name or a category column, and its draw uses a dedicated seed stream so adding/removing decorative categories does not shift group assignment for the same seed. Defaults to None (single global β).
        epsilon_loc: Mean of the Gaussian noise. Defaults to 0.0.
        epsilon_scale: When ``use_weights=False``, the noise stddev for every row. When ``use_weights=True``, the *reference* stddev at ``w=1``; actual per-row noise is ``N(loc, (epsilon_scale/√w)²)``. Must be >= 0. Defaults to 1.0.
        seed: Seed for ``np.random.default_rng``. If None, uses fresh entropy per call (and the source is registered with ``is_pure=False``).
        n_workers: Number of threads used to fill the ``x``/``y`` Gaussian draws (the dominant cost) in parallel. ``1`` (default) keeps the fully serial path and its exact output. With ``n_workers > 1`` the draws are generated in fixed-size blocks split across threads, so reproducibility is keyed on ``(seed, n_workers, fetch_size)`` and remains independent of the ``batch_size`` Polars chooses; values differ from the serial path. Peak generation memory grows with ``n_workers`` (a block holds ``n_workers * fetch_size`` rows). For panels the per-date batch is ``n_symbols`` rows, so parallelism only helps when ``n_symbols`` is large.
        fetch_size: Default number of rows generated per batch when Polars does not provide a ``batch_size``. Must be >= 1. Defaults to 10_000.
        description: Optional free-form description of this source instance, attached to its OpenTelemetry span (``explain_detail``).
    """
    _validate_common(
        n_features=n_features,
        n_responses=n_responses,
        epsilon_scale=epsilon_scale,
        fetch_size=fetch_size,
        use_weights=use_weights,
        weights_low=weights_low,
        weights_high=weights_high,
    )
    if n_symbols < 1:
        raise ValueError(f"n_symbols must be >= 1, got {n_symbols}")

    date_idx = pd.bdate_range(
        start_date,
        end_date + timedelta(1),
        freq=freq,
        inclusive="left",
        name="date",
    )
    n_dates = len(date_idx)
    if n_dates < 1:
        raise ValueError(f"start_date={start_date} and end_date={end_date} with freq={freq!r} yields no dates")

    # Pre-convert to numpy once: index-based draws avoid ``choice``'s platform-dependent int
    # width (int32 on Windows vs Int64 in the polars schema), and (name, values) pairs keep
    # closure and schema dtypes in lockstep.
    category_arrays: list[tuple[str, np.ndarray]] = []
    if categories is not None:
        for i, cats in enumerate(categories):
            if not isinstance(cats, (list, tuple)) or len(cats) == 0:
                raise ValueError(f"categories[{i}] must be a non-empty list, got {cats!r}")
            name = f"category_{i}"
            _check_reserved(
                name,
                "category column",
                n_features=n_features,
                n_responses=n_responses,
                use_weights=use_weights,
                extra=("date", "symbol", "timestamp"),
            )
            category_arrays.append((name, np.array(cats)))

    group_name: str | None = None
    group_values_arr: np.ndarray | None = None
    n_groups = 0
    if group_by is not None:
        if not isinstance(group_by, tuple) or len(group_by) != 2:
            raise ValueError(f"group_by must be a (name, values) tuple, got {group_by!r}")
        group_name, group_values = group_by
        if not isinstance(group_name, str) or not group_name:
            raise ValueError(f"group_by name must be a non-empty string, got {group_name!r}")
        if not isinstance(group_values, (list, tuple)) or len(group_values) == 0:
            raise ValueError(f"group_by values must be a non-empty list, got {group_values!r}")
        _check_reserved(
            group_name,
            "group_by name",
            n_features=n_features,
            n_responses=n_responses,
            use_weights=use_weights,
            extra=("date", "symbol", "timestamp", *(n for n, _ in category_arrays)),
        )
        n_groups = len(group_values)
        group_values_arr = np.array(group_values)

    # Dedicated child of the root SeedSequence for the β fallback draw, so β does not
    # share bits with the x/eps/w/aux/group streams spawned inside _register_source.
    betas_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(6)[5])
    if group_by is None:
        if betas is None:
            betas_arr = betas_rng.standard_normal(size=(n_features, n_responses))
        else:
            betas_arr = np.array(betas, dtype=np.float64)
            if betas_arr.ndim == 1 and n_responses == 1:
                betas_arr = betas_arr[:, None]
            if betas_arr.shape != (n_features, n_responses):
                raise ValueError(f"betas must have shape (n_features={n_features}, n_responses={n_responses}), got {betas_arr.shape}")
    else:
        if betas is None:
            betas_arr = betas_rng.standard_normal(size=(n_groups, n_features, n_responses))
        else:
            betas_arr = np.array(betas, dtype=np.float64)
            if betas_arr.ndim == 2 and n_responses == 1:
                betas_arr = betas_arr[:, :, None]
            if betas_arr.shape != (n_groups, n_features, n_responses):
                raise ValueError(
                    f"betas must have shape (n_groups={n_groups}, n_features={n_features}, n_responses={n_responses}), got {betas_arr.shape}"
                )

    symbols_arr = np.array([f"sym_{i}" for i in range(n_symbols)])
    dates_arr = date_idx.to_numpy(dtype="datetime64[D]")
    timestamps_arr = date_idx.to_numpy().astype("datetime64[us]")

    chunk_sizes = np.full(n_dates, n_symbols, dtype=np.int64)

    def mean_computer(
        aux_rng: np.random.Generator, group_rng: np.random.Generator, x_batch: np.ndarray, chunk_idx: int, row_in_chunk: int
    ) -> tuple[np.ndarray, dict]:
        k = x_batch.shape[0]
        extra: dict = {
            "symbol": symbols_arr[np.arange(row_in_chunk, row_in_chunk + k) % n_symbols],
            "date": np.full(k, dates_arr[chunk_idx], dtype="datetime64[D]"),
            "timestamp": np.full(k, timestamps_arr[chunk_idx], dtype="datetime64[us]"),
        }
        for name, values in category_arrays:
            extra[name] = values[aux_rng.integers(0, len(values), size=k)]
        if group_by is None:
            return x_batch @ betas_arr, extra
        group_idx = group_rng.integers(0, n_groups, size=k)
        extra[group_name] = group_values_arr[group_idx]
        row_betas = betas_arr[group_idx]
        mean_batch = np.einsum("ij,ijk->ik", x_batch, row_betas)
        return mean_batch, extra

    extras_schema: dict = {"symbol": pl.String(), "date": pl.Date(), "timestamp": pl.Datetime("us")}
    for name, values in category_arrays:
        extras_schema[name] = pl.Series(values).dtype
    if group_by is not None:
        extras_schema[group_name] = pl.Series(group_values_arr).dtype

    return _register_source(
        n_samples=n_symbols * n_dates,
        n_features=n_features,
        n_responses=n_responses,
        use_weights=use_weights,
        weights_low=weights_low,
        weights_high=weights_high,
        epsilon_loc=epsilon_loc,
        epsilon_scale=epsilon_scale,
        seed=seed,
        fetch_size=fetch_size,
        mean_computer=mean_computer,
        extras_schema=extras_schema,
        chunk_sizes=chunk_sizes,
        n_workers=n_workers,
        explain_name="scan_synthetic_panel",
        explain_detail=description,
    )
