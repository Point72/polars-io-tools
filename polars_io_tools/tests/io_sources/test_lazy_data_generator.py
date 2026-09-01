from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import pytest

from polars_io_tools.io_sources import scan_synthetic_panel, scan_synthetic_regression


class LinearRegression:
    """Minimal OLS via ``numpy.linalg.lstsq``, mirroring the subset of
    ``sklearn.linear_model.LinearRegression`` these tests use (``coef_`` only),
    so the suite needs no scikit-learn dependency."""

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        A = np.column_stack([X, np.ones(len(X))]) if self.fit_intercept else X
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        if self.fit_intercept:
            sol = sol[:-1]
        self.coef_ = sol.T if sol.ndim > 1 else sol
        return self


def test_beta_fitting():
    betas = np.array([[0.45], [0.23], [-0.42], [0.22], [-0.88]])
    df = scan_synthetic_regression(n_samples=10_000, n_features=5, n_responses=1, betas=betas, seed=1234).collect()
    features = df.select("x0", "x1", "x2", "x3", "x4").to_numpy()
    labels = df.select("y0").to_numpy()
    fitted_betas = LinearRegression().fit(features, labels).coef_
    assert np.max(np.abs(fitted_betas.reshape(-1) - betas.reshape(-1))) < 0.05


def test_wls_beta_recovery():
    # use_weights=True scales noise as Var(eps) = epsilon_scale^2 / w, so a WLS fit
    # (equivalently, sqrt(w)-scaled OLS on X, y) should recover betas.
    betas = np.array([[0.45], [0.23], [-0.42], [0.22], [-0.88]])
    df = scan_synthetic_regression(
        n_samples=10_000, n_features=5, betas=betas, use_weights=True, weights_low=0.2, weights_high=2.5, seed=1234
    ).collect()
    X = df.select("x0", "x1", "x2", "x3", "x4").to_numpy()
    y = df["y0"].to_numpy()
    sqrt_w = np.sqrt(df["weight"].to_numpy())[:, None]
    fitted, _, _, _ = np.linalg.lstsq(sqrt_w * X, sqrt_w[:, 0] * y, rcond=None)
    assert np.max(np.abs(fitted - betas.reshape(-1))) < 0.05


def test_multi_response_beta_recovery():
    # n_responses=2 must produce y0 and y1 columns with independent beta columns recovered from each.
    betas = np.array([[0.5, -0.3], [0.2, 0.7], [-0.4, 0.1]])  # (n_features=3, n_responses=2)
    df = scan_synthetic_regression(n_samples=10_000, n_features=3, n_responses=2, betas=betas, seed=1).collect()
    assert "y0" in df.columns and "y1" in df.columns
    X = df.select("x0", "x1", "x2").to_numpy()
    for j in range(2):
        fit = LinearRegression(fit_intercept=False).fit(X, df[f"y{j}"].to_numpy()).coef_
        assert np.max(np.abs(fit - betas[:, j])) < 0.05


def test_n_rows_pushdown_head():
    # .head(k) is pushed down as n_rows; must deliver exactly k rows and match the prefix of the full scan.
    kwargs = {"n_samples": 1000, "n_features": 3, "seed": 42}
    full = scan_synthetic_regression(**kwargs).collect().to_numpy()
    head = scan_synthetic_regression(**kwargs).head(137).collect().to_numpy()
    assert head.shape[0] == 137
    assert np.array_equal(head, full[:137])


def test_predicate_and_projection_pushdown():
    # Projection + predicate pushed through the IO source: filter applied inside _source_generator,
    # selected columns only in the emitted DataFrame.
    lf = scan_synthetic_regression(n_samples=2000, n_features=3, seed=0)
    df = lf.select("x0", "y0").filter(pl.col("x0") > 0).collect()
    assert set(df.columns) == {"x0", "y0"}
    assert (df["x0"] > 0).all()
    assert df.height > 0


def test_panel_per_group_betas_recovered():
    n_features = 3
    group_values = ["alpha", "beta", "gamma", "delta"]
    betas = np.array(
        [
            [[0.5], [-0.3], [0.8]],
            [[1.2], [0.1], [-0.5]],
            [[-0.9], [0.4], [0.2]],
            [[0.0], [0.7], [-0.6]],
        ]
    )
    df = scan_synthetic_panel(
        start_date=date(2020, 1, 6),
        end_date=date(2020, 10, 13),  # ~200 business days
        n_symbols=50,
        n_features=n_features,
        betas=betas,
        group_by=("g", group_values),
        seed=42,
    ).collect()
    assert set(df["g"].unique().to_list()) == set(group_values)
    for g_idx, g_val in enumerate(group_values):
        sub = df.filter(pl.col("g") == g_val)
        X = sub.select("x0", "x1", "x2").to_numpy()
        y = sub["y0"].to_numpy()
        fitted = LinearRegression(fit_intercept=False).fit(X, y).coef_
        assert np.max(np.abs(fitted - betas[g_idx, :, 0])) < 0.1


def test_panel_group_by_seed_independent_of_categories():
    # Group draw uses its own seed stream, so adding/removing decorative categories
    # must not shift group assignments for the same seed.
    common = {
        "start_date": _PANEL_START,
        "end_date": _PANEL_END_10,
        "n_symbols": 4,
        "n_features": 2,
        "group_by": ("g", ["A", "B", "C"]),
        "seed": 7,
    }
    a = scan_synthetic_panel(**common).collect()
    b = scan_synthetic_panel(**common, categories=[["u", "v"]]).collect()
    assert a["g"].to_list() == b["g"].to_list()


def test_seed_reproducibility():
    kwargs = {"n_samples": 500, "n_features": 3, "seed": 42}
    assert scan_synthetic_regression(**kwargs).collect().equals(scan_synthetic_regression(**kwargs).collect())


def test_batch_size_independence():
    # Same seed + different fetch_size => identical rows (deterministic row-indexed RNG streams).
    kwargs = {"n_samples": 1000, "n_features": 3, "seed": 42}
    a = scan_synthetic_regression(**kwargs, fetch_size=1000).collect().to_numpy()
    b = scan_synthetic_regression(**kwargs, fetch_size=37).collect().to_numpy()
    assert np.array_equal(a, b)


def test_panel_group_by_batch_size_independence():
    kwargs = {
        "start_date": date(2020, 1, 6),
        "end_date": date(2020, 1, 13),
        "n_symbols": 8,
        "n_features": 3,
        "group_by": ("g", ["A", "B", "C", "D", "E"]),
        "seed": 42,
    }
    a = scan_synthetic_panel(**kwargs, fetch_size=1000).collect()
    b = scan_synthetic_panel(**kwargs, fetch_size=3).collect()
    assert a["g"].equals(b["g"])
    assert np.array_equal(
        a.select("x0", "x1", "x2", "y0").to_numpy(),
        b.select("x0", "x1", "x2", "y0").to_numpy(),
    )


def test_invalid_args():
    with pytest.raises(ValueError, match="n_features"):
        scan_synthetic_regression(n_samples=10, n_features=0)
    with pytest.raises(ValueError, match="epsilon_scale"):
        scan_synthetic_regression(n_samples=10, n_features=2, epsilon_scale=-1.0)
    with pytest.raises(ValueError, match="betas must have shape"):
        scan_synthetic_regression(n_samples=10, n_features=2, betas=np.zeros((3, 1)))
    with pytest.raises(ValueError, match="weights_low"):
        scan_synthetic_regression(n_samples=10, n_features=2, use_weights=True, weights_low=1.0, weights_high=0.5)


def test_panel_group_by_invalid_args():
    common = {"start_date": _PANEL_START, "end_date": _PANEL_END_4, "n_symbols": 3, "n_features": 2}
    with pytest.raises(ValueError, match="group_by values"):
        scan_synthetic_panel(**common, group_by=("g", []))
    with pytest.raises(ValueError, match="betas must have shape"):
        # (n_groups=3, n_features=2, n_responses=1) required, providing wrong shape
        scan_synthetic_panel(**common, group_by=("g", ["A", "B", "C"]), betas=np.zeros((2, 2, 1)))
    with pytest.raises(ValueError, match="group_by name"):
        scan_synthetic_panel(**common, group_by=("x0", ["A", "B"]))
    with pytest.raises(ValueError, match="group_by name"):
        # collision with a decorative category column
        scan_synthetic_panel(**common, categories=[["u", "v"]], group_by=("category_0", ["A", "B"]))


def test_chunked_key_sorted_and_sized():
    n_samples, n_chunks = 1003, 7
    df = scan_synthetic_regression(n_samples=n_samples, n_features=3, chunk_key="chunk", n_chunks=n_chunks, seed=0).collect()
    assert df.schema["chunk"] == pl.Int64
    keys = df["chunk"].to_numpy()
    assert np.all(np.diff(keys) >= 0)
    counts = df.group_by("chunk", maintain_order=True).len()["len"].to_numpy()
    base, rem = divmod(n_samples, n_chunks)
    assert len(counts) == n_chunks
    assert np.array_equal(counts, np.array([base + 1] * rem + [base] * (n_chunks - rem)))
    assert np.array_equal(df.group_by("chunk", maintain_order=True).len()["chunk"].to_numpy(), np.arange(n_chunks))


def test_chunked_single_global_beta():
    # A fit on any per-chunk subset recovers the same global betas.
    betas = np.array([[0.7], [-0.3], [0.5]])
    df = scan_synthetic_regression(n_samples=20_000, n_features=3, betas=betas, chunk_key="chunk", n_chunks=10, seed=1234).collect()
    # Global fit recovers betas.
    X = df.select("x0", "x1", "x2").to_numpy()
    y = df["y0"].to_numpy()
    global_fit = LinearRegression(fit_intercept=False).fit(X, y).coef_
    assert np.max(np.abs(global_fit - betas.reshape(-1))) < 0.05
    unique_keys = df["chunk"].unique(maintain_order=True).to_list()
    for k in unique_keys:
        sub = df.filter(pl.col("chunk") == k)
        Xk = sub.select("x0", "x1", "x2").to_numpy()
        yk = sub["y0"].to_numpy()
        fit_k = LinearRegression(fit_intercept=False).fit(Xk, yk).coef_
        assert np.max(np.abs(fit_k - betas.reshape(-1))) < 0.15


def test_chunked_batch_size_independence():
    # fetch_size smaller than a chunk must subdivide, not straddle chunk boundaries.
    kwargs = {"n_samples": 1000, "n_features": 3, "chunk_key": "chunk", "n_chunks": 8, "seed": 42}
    a = scan_synthetic_regression(**kwargs, fetch_size=1000).collect()
    b = scan_synthetic_regression(**kwargs, fetch_size=37).collect()
    assert np.array_equal(a.drop("chunk").to_numpy(), b.drop("chunk").to_numpy())
    assert a["chunk"].equals(b["chunk"])
    assert np.all(np.diff(b["chunk"].to_numpy()) >= 0)


_PANEL_START = date(2020, 1, 6)  # Monday
_PANEL_END_4 = date(2020, 1, 9)  # 4 business days
_PANEL_END_10 = date(2020, 1, 15)  # 10 calendar days (default freq="1D" includes weekends)
_PANEL_END_MONTH = date(2020, 1, 31)


def test_panel_schema_and_shape():
    df = scan_synthetic_panel(
        start_date=_PANEL_START,
        end_date=_PANEL_END_4,
        n_symbols=5,
        n_features=3,
        categories=[["A", "B"], ["X", "Y", "Z"]],
        use_weights=True,
        weights_low=0.2,
        weights_high=2.5,
        seed=0,
    ).collect()
    assert df.height == 5 * 4
    assert df.schema["date"] == pl.Date
    assert df.schema["timestamp"] == pl.Datetime("us")
    assert df.schema["symbol"] == pl.String
    assert df.schema["category_0"] == pl.String
    assert df.schema["category_1"] == pl.String
    assert df.schema["weight"] == pl.Float64
    assert set(df["category_0"].unique().to_list()).issubset({"A", "B"})
    assert set(df["category_1"].unique().to_list()).issubset({"X", "Y", "Z"})
    w = df["weight"].to_numpy()
    assert np.all(w >= 0.2) and np.all(w <= 2.5)
    assert df["weight"].n_unique() > 1


def test_panel_sorted_by_date_and_symbol_coverage():
    df = scan_synthetic_panel(start_date=_PANEL_START, end_date=_PANEL_END_10, n_symbols=7, n_features=2, seed=1).collect()
    assert df["date"].is_sorted()
    n_dates = df["date"].n_unique()
    assert n_dates == 10
    per_date = df.group_by("date", maintain_order=True).len()["len"].to_numpy()
    assert np.array_equal(per_date, np.full(n_dates, 7))
    per_symbol = df.group_by("symbol").len()["len"].to_numpy()
    assert np.all(per_symbol == n_dates)
    assert df["symbol"].n_unique() == 7
    assert (df["timestamp"].dt.date() == df["date"]).all()
    df_w = scan_synthetic_panel(start_date=_PANEL_START, end_date=_PANEL_END_MONTH, freq="W", n_symbols=2, n_features=2, seed=0).collect()
    expected_dates = pd.bdate_range(_PANEL_START, _PANEL_END_MONTH + pd.Timedelta(days=1), freq="W", inclusive="left").date
    assert df_w["date"].unique(maintain_order=True).to_list() == list(expected_dates)


def test_panel_beta_recovery():
    betas = np.array([[1.5], [-2.0], [0.5], [0.8]])
    df = scan_synthetic_panel(
        start_date=date(2020, 1, 6),
        end_date=date(2020, 10, 13),  # ~200 business days
        n_symbols=50,
        n_features=4,
        betas=betas,
        epsilon_scale=0.3,
        seed=42,
    ).collect()
    X = df.select("x0", "x1", "x2", "x3").to_numpy()
    fit = LinearRegression(fit_intercept=False).fit(X, df["y0"].to_numpy()).coef_
    assert np.max(np.abs(fit - betas.reshape(-1))) < 0.01


def test_panel_batch_size_independence():
    # fetch_size smaller than n_symbols must not straddle date boundaries or shuffle symbols.
    # Also covers seed reproducibility (same seed => same rows regardless of fetch_size).
    kwargs = {"start_date": date(2020, 1, 6), "end_date": date(2020, 1, 13), "n_symbols": 8, "n_features": 3, "seed": 42}
    a = scan_synthetic_panel(**kwargs, fetch_size=1000).collect()
    b = scan_synthetic_panel(**kwargs, fetch_size=3).collect()
    assert a["date"].equals(b["date"])
    assert a["symbol"].equals(b["symbol"])
    assert np.array_equal(
        a.select("x0", "x1", "x2", "y0").to_numpy(),
        b.select("x0", "x1", "x2", "y0").to_numpy(),
    )
    assert scan_synthetic_panel(**kwargs, fetch_size=1000).collect().equals(a)


def test_panel_streaming_group_by():
    lf = scan_synthetic_panel(start_date=_PANEL_START, end_date=_PANEL_END_MONTH, n_symbols=10, n_features=3, seed=7).set_sorted("date")
    counts = lf.group_by("date", maintain_order=True).agg(pl.len().alias("n")).collect(engine="streaming")
    assert (counts["n"] == 10).all()


def test_panel_invalid_args():
    with pytest.raises(ValueError, match="n_symbols"):
        scan_synthetic_panel(start_date=_PANEL_START, end_date=_PANEL_END_4, n_symbols=0, n_features=2)
    with pytest.raises(ValueError, match="n_features"):
        scan_synthetic_panel(start_date=_PANEL_START, end_date=_PANEL_END_4, n_symbols=5, n_features=0)
    with pytest.raises(ValueError, match="betas must have shape"):
        scan_synthetic_panel(start_date=_PANEL_START, end_date=_PANEL_END_4, n_symbols=5, n_features=2, betas=np.zeros((3, 1)))
    with pytest.raises(ValueError, match="categories"):
        scan_synthetic_panel(start_date=_PANEL_START, end_date=_PANEL_END_4, n_symbols=5, n_features=2, categories=[[]])
    with pytest.raises(ValueError, match="weights_low"):
        scan_synthetic_panel(
            start_date=_PANEL_START,
            end_date=_PANEL_END_4,
            n_symbols=3,
            n_features=2,
            use_weights=True,
            weights_low=1.0,
            weights_high=0.5,
        )


def test_panel_betas_1d_promotion():
    # 1-D betas accepted when n_responses=1
    df = scan_synthetic_panel(
        start_date=_PANEL_START,
        end_date=_PANEL_END_4,
        n_symbols=5,
        n_features=3,
        betas=np.array([1.0, 2.0, 3.0]),
        epsilon_scale=0.0,
        seed=0,
    ).collect()
    X = df.select("x0", "x1", "x2").to_numpy()
    y_expected = X @ np.array([1.0, 2.0, 3.0])
    assert np.allclose(df["y0"].to_numpy(), y_expected, atol=1e-12)


def test_n_workers_determinism_and_schema_parity():
    # Fixed (seed, n_workers) reproduces exactly; schema/shape match the serial path.
    kwargs = {"n_samples": 12_345, "n_features": 6, "n_responses": 2, "use_weights": True, "seed": 7}
    serial = scan_synthetic_regression(**kwargs, n_workers=1).collect()
    a = scan_synthetic_regression(**kwargs, n_workers=4).collect()
    b = scan_synthetic_regression(**kwargs, n_workers=4).collect()
    assert a.equals(b)
    assert a.schema == serial.schema
    assert a.shape == serial.shape
    # Threading changes the RNG stream layout, so values differ from the serial path.
    assert not a.equals(serial)


def test_n_workers_batch_size_independence():
    # With n_workers > 1, reproducibility is keyed on (seed, n_workers, fetch_size): blocks are
    # sized n_workers * fetch_size, so fetch_size is part of the contract (like seed). What must
    # still hold for is_pure is independence from the *runtime* batch_size Polars chooses at a
    # fixed fetch_size -- verified here by comparing the in-memory and streaming engines.
    kwargs = {"n_samples": 4000, "n_features": 4, "seed": 42, "n_workers": 8, "fetch_size": 512}
    a = scan_synthetic_regression(**kwargs).collect()
    b = scan_synthetic_regression(**kwargs).collect(engine="streaming")
    assert a.equals(b)


def test_n_workers_beta_recovery():
    betas = np.array([1.0, -2.0, 0.5, 3.0])
    df = scan_synthetic_regression(n_samples=200_000, n_features=4, n_responses=1, betas=betas, epsilon_scale=0.1, seed=3, n_workers=8).collect()
    features = df.select("x0", "x1", "x2", "x3").to_numpy()
    labels = df.select("y0").to_numpy()
    fitted = LinearRegression().fit(features, labels).coef_
    assert np.max(np.abs(fitted.reshape(-1) - betas)) < 0.02


def test_n_workers_pushdowns():
    kwargs = {"n_samples": 5000, "n_features": 5, "n_responses": 2, "seed": 1, "n_workers": 8}
    assert scan_synthetic_regression(**kwargs).head(321).collect().height == 321
    assert scan_synthetic_regression(**kwargs).select(["x0", "y1"]).collect().columns == ["x0", "y1"]


def test_n_workers_panel_parity():
    kwargs = {
        "start_date": date(2020, 1, 1),
        "end_date": date(2020, 2, 1),
        "n_symbols": 500,
        "n_features": 4,
        "seed": 5,
    }
    serial = scan_synthetic_panel(**kwargs, n_workers=1).collect()
    a = scan_synthetic_panel(**kwargs, n_workers=4).collect()
    b = scan_synthetic_panel(**kwargs, n_workers=4).collect()
    assert a.equals(b)
    assert a.schema == serial.schema
    assert a.shape == serial.shape


def test_n_workers_invalid():
    with pytest.raises(ValueError, match="n_workers must be >= 1"):
        scan_synthetic_regression(n_samples=10, n_features=2, seed=1, n_workers=0)
    with pytest.raises(ValueError, match="n_workers must be >= 1"):
        scan_synthetic_panel(start_date=date(2020, 1, 1), end_date=date(2020, 1, 3), n_features=2, seed=1, n_workers=0)


def test_n_workers_multi_block_batch_size_independence():
    # n_samples spans several fixed blocks (block_rows = n_workers * fetch_size = 4 * 250 = 1000),
    # exercising the block-cursor refill and boundary cap. Output must be identical whether Polars
    # drives it in memory or via the streaming engine (different runtime batch sizes) at fixed
    # fetch_size, and must not depend on that batch size.
    kwargs = {"n_samples": 3300, "n_features": 3, "n_responses": 2, "use_weights": True, "seed": 9, "n_workers": 4, "fetch_size": 250}
    a = scan_synthetic_regression(**kwargs).collect()
    b = scan_synthetic_regression(**kwargs).collect(engine="streaming")
    assert a.equals(b)
    assert a.height == 3300


def test_n_workers_panel_batch_size_independence_full():
    # Threaded panel with categories, group_by, and weights: x/eps come from the block filler
    # while aux/group/weight stay per-batch serial streams -- verify they stay row-aligned and
    # the whole frame is independent of the runtime batch_size (in-memory vs streaming engine).
    kwargs = {
        "start_date": date(2020, 1, 1),
        "end_date": date(2020, 3, 1),
        "n_symbols": 137,  # not a divisor of block_rows, so date chunks straddle blocks
        "n_features": 4,
        "categories": [["A", "B", "C"]],
        "group_by": ("g", ["x", "y"]),
        "use_weights": True,
        "seed": 21,
        "n_workers": 8,
        "fetch_size": 64,
    }
    a = scan_synthetic_panel(**kwargs).collect()
    b = scan_synthetic_panel(**kwargs).collect(engine="streaming")
    assert a.equals(b)


def test_n_workers_head_pushdown_terminates_early():
    # An early n_rows pushdown must return promptly and shut its pool down via the generator's
    # finally, without generating the full declared dataset.
    df = scan_synthetic_regression(n_samples=50_000_000, n_features=4, seed=1, n_workers=4).head(5).collect()
    assert df.height == 5
