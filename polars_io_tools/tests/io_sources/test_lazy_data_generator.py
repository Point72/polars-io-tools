"""scan_synthetic_regression and scan_synthetic_panel.

Tables: generated data -> least-squares fits recover the betas; two ways of generating the same data -> identical
frames; Polars steps on the reader -> what they give on the collected data; invalid arguments -> errors.
"""

from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_io_tools.io_sources import scan_synthetic_panel, scan_synthetic_regression
from polars_io_tools.tests.helpers.assertions import assert_matches_native
from polars_io_tools.tests.helpers.cases import Case, params

REG, PANEL = scan_synthetic_regression, scan_synthetic_panel
_PANEL_START = date(2020, 1, 6)  # Monday
_PANEL_END_4 = date(2020, 1, 9)  # 4 business days
_PANEL_END_10 = date(2020, 1, 15)  # 10 calendar days (default freq="1D" includes weekends)
_PANEL_END_MONTH = date(2020, 1, 31)
_LONG_PANEL = {"start_date": date(2020, 1, 6), "end_date": date(2020, 10, 13), "n_symbols": 50}  # ~200 business days


def _lstsq(X, y, weights=None, intercept=False):
    """Least-squares coefficients of y on X; ``weights`` gives WLS, ``intercept`` fits and drops a constant."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if weights is not None:
        sqrt_w = np.sqrt(weights)
        X, y = X * sqrt_w[:, None], y * sqrt_w
    A = np.column_stack([X, np.ones(len(X))]) if intercept else X
    sol = np.linalg.lstsq(A, y, rcond=None)[0]
    return sol[:-1] if intercept else sol


def B(id, scan, kwargs, fits, n_features):
    """``fits``: (rows to fit on or None, response, true betas, tolerance, weighted, intercept)."""
    return Case(id, scan=(scan, kwargs), fits=fits, features=[f"x{i}" for i in range(n_features)])


_BETAS5 = np.array([0.45, 0.23, -0.42, 0.22, -0.88])
_GROUP_BETAS = np.array([[0.5, -0.3, 0.8], [1.2, 0.1, -0.5], [-0.9, 0.4, 0.2], [0.0, 0.7, -0.6]])
_GROUPS = ["alpha", "beta", "gamma", "delta"]
_MULTI = np.array([[0.5, -0.3], [0.2, 0.7], [-0.4, 0.1]])
_GLOBAL = np.array([0.7, -0.3, 0.5])

BETA_FITS = [
    B(
        "beta_fitting",
        REG,
        {"n_samples": 10_000, "n_features": 5, "n_responses": 1, "betas": _BETAS5[:, None], "seed": 1234},
        [(None, "y0", _BETAS5, 0.05, False, True)],
        5,
    ),
    # use_weights scales the noise variance by 1 / w, so a WLS fit recovers the betas
    B(
        "wls_beta_recovery",
        REG,
        {"n_samples": 10_000, "n_features": 5, "betas": _BETAS5[:, None], "use_weights": True, "weights_low": 0.2, "weights_high": 2.5, "seed": 1234},
        [(None, "y0", _BETAS5, 0.05, True, False)],
        5,
    ),
    B(
        "multi_response_beta_recovery",
        REG,
        {"n_samples": 10_000, "n_features": 3, "n_responses": 2, "betas": _MULTI, "seed": 1},
        [(None, f"y{j}", _MULTI[:, j], 0.05, False, False) for j in range(2)],
        3,
    ),
    B(
        "panel_per_group_betas_recovered",
        PANEL,
        {**_LONG_PANEL, "n_features": 3, "betas": _GROUP_BETAS[:, :, None], "group_by": ("g", _GROUPS), "seed": 42},
        [(pl.col("g") == g, "y0", _GROUP_BETAS[i], 0.1, False, False) for i, g in enumerate(_GROUPS)],
        3,
    ),
    # one global beta: the whole frame and every chunk recover it
    B(
        "chunked_single_global_beta",
        REG,
        {"n_samples": 20_000, "n_features": 3, "betas": _GLOBAL[:, None], "chunk_key": "chunk", "n_chunks": 10, "seed": 1234},
        [(None, "y0", _GLOBAL, 0.05, False, False), *[(pl.col("chunk") == k, "y0", _GLOBAL, 0.15, False, False) for k in range(10)]],
        3,
    ),
    B(
        "panel_beta_recovery",
        PANEL,
        {**_LONG_PANEL, "n_features": 4, "betas": np.array([[1.5], [-2.0], [0.5], [0.8]]), "epsilon_scale": 0.3, "seed": 42},
        [(None, "y0", np.array([1.5, -2.0, 0.5, 0.8]), 0.01, False, False)],
        4,
    ),
    B(
        "n_workers_beta_recovery",
        REG,
        {
            "n_samples": 200_000,
            "n_features": 4,
            "n_responses": 1,
            "betas": np.array([1.0, -2.0, 0.5, 3.0]),
            "epsilon_scale": 0.1,
            "seed": 3,
            "n_workers": 8,
        },
        [(None, "y0", np.array([1.0, -2.0, 0.5, 3.0]), 0.02, False, True)],
        4,
    ),
]


@pytest.mark.parametrize(("scan", "fits", "features"), params(BETA_FITS, "scan", "fits", "features"))
def test_beta_recovery(scan, fits, features):
    scanner, kwargs = scan
    df = scanner(**kwargs).collect()
    assert {f"y{j}" for j in range(kwargs.get("n_responses", 1))} <= set(df.columns)
    for rows, response, betas, tol, weighted, intercept in fits:
        sub = df if rows is None else df.filter(rows)
        assert sub.height > 0
        fitted = _lstsq(sub.select(features).to_numpy(), sub[response].to_numpy(), sub["weight"].to_numpy() if weighted else None, intercept)
        assert np.max(np.abs(fitted - betas)) < tol


def _engine(name):
    return lambda lf: lf.collect(engine=name)


def _collect(lf):
    return lf.collect()


def D(id, scan, kwargs, a, b, columns=None, same=True):
    """Generating with extra arguments ``a`` and ``b`` (a dict, or a collect function) gives the same ``columns``.
    ``same=False``: the frames share schema and shape but differ."""
    return Case(id, scan=(scan, kwargs), same=same, a=a, b=b, columns=columns)


_REG_1000 = {"n_samples": 1000, "n_features": 3, "seed": 42}
_PANEL_WEEK = {"start_date": date(2020, 1, 6), "end_date": date(2020, 1, 13), "n_symbols": 8, "n_features": 3, "seed": 42}
_FETCH = ({"fetch_size": 1000}, {"fetch_size": 3})
_PANEL_PARITY = {"start_date": date(2020, 1, 1), "end_date": date(2020, 2, 1), "n_symbols": 500, "n_features": 4, "seed": 5}
_REG_WORKERS = {"n_samples": 12_345, "n_features": 6, "n_responses": 2, "use_weights": True, "seed": 7}

DETERMINISM = [
    D("seed_reproducibility", REG, {"n_samples": 500, "n_features": 3, "seed": 42}, {}, {}),
    # row-indexed RNG streams: the batch size does not change the rows
    D("batch_size_independence", REG, _REG_1000, {"fetch_size": 1000}, {"fetch_size": 37}),
    # a fetch size smaller than a chunk subdivides it
    D("chunked_batch_size_independence", REG, {**_REG_1000, "chunk_key": "chunk", "n_chunks": 8}, {"fetch_size": 1000}, {"fetch_size": 37}),
    # a fetch size smaller than n_symbols must not straddle dates or shuffle symbols
    D("panel_batch_size_independence", PANEL, _PANEL_WEEK, *_FETCH),
    D("panel_group_by_batch_size_independence", PANEL, {**_PANEL_WEEK, "group_by": ("g", ["A", "B", "C", "D", "E"])}, *_FETCH),
    # the group draw has its own seed stream, so decorative categories do not shift it
    D(
        "panel_group_by_seed_independent_of_categories",
        PANEL,
        {"start_date": _PANEL_START, "end_date": _PANEL_END_10, "n_symbols": 4, "n_features": 2, "group_by": ("g", ["A", "B", "C"]), "seed": 7},
        {},
        {"categories": [["u", "v"]]},
        ["g"],
    ),
    # with n_workers, a fixed (seed, n_workers, fetch_size) reproduces, whatever batch size the engine asks for
    D("n_workers_determinism_and_schema_parity-repeat", REG, _REG_WORKERS, {"n_workers": 4}, {"n_workers": 4}),
    # threading changes the RNG stream layout
    D("n_workers_determinism_and_schema_parity-serial", REG, _REG_WORKERS, {"n_workers": 4}, {"n_workers": 1}, same=False),
    D(
        "n_workers_batch_size_independence",
        REG,
        {"n_samples": 4000, "n_features": 4, "seed": 42, "n_workers": 8, "fetch_size": 512},
        _collect,
        _engine("streaming"),
    ),
    # several fixed blocks of n_workers * fetch_size rows
    D(
        "n_workers_multi_block_batch_size_independence",
        REG,
        {"n_samples": 3300, "n_features": 3, "n_responses": 2, "use_weights": True, "seed": 9, "n_workers": 4, "fetch_size": 250},
        _collect,
        _engine("streaming"),
    ),
    D("n_workers_panel_parity-repeat", PANEL, _PANEL_PARITY, {"n_workers": 4}, {"n_workers": 4}),
    D("n_workers_panel_parity-serial", PANEL, _PANEL_PARITY, {"n_workers": 4}, {"n_workers": 1}, same=None),
    # x and eps come from the block filler, the other columns from per-batch streams; they stay aligned
    D(
        "n_workers_panel_batch_size_independence_full",
        PANEL,
        {
            "start_date": date(2020, 1, 1),
            "end_date": date(2020, 3, 1),
            "n_symbols": 137,  # not a divisor of the block size, so dates straddle blocks
            "n_features": 4,
            "categories": [["A", "B", "C"]],
            "group_by": ("g", ["x", "y"]),
            "use_weights": True,
            "seed": 21,
            "n_workers": 8,
            "fetch_size": 64,
        },
        _collect,
        _engine("streaming"),
    ),
]


@pytest.mark.parametrize(("scan", "same", "a", "b", "columns"), params(DETERMINISM, "scan", "same", "a", "b", "columns"))
def test_same_data(scan, same, a, b, columns):
    """``same`` True: equal frames; False: same schema and shape, different values; None: same schema and shape."""
    scanner, kwargs = scan

    def make(variant):
        return variant(scanner(**kwargs)) if callable(variant) else scanner(**kwargs, **variant).collect()

    first, second = make(a), make(b)
    if columns:
        first, second = first.select(columns), second.select(columns)
    if same:
        assert_frame_equal(first, second)
    else:
        assert (first.schema, first.shape) == (second.schema, second.shape)
        assert same is None or not first.equals(second)


_REG_2000 = {"n_samples": 2000, "n_features": 3, "seed": 0}
_REG_5000_W = {"n_samples": 5000, "n_features": 5, "n_responses": 2, "seed": 1, "n_workers": 8}


@pytest.mark.parametrize(
    ("kwargs", "op"),
    [
        # head is pushed down as n_rows and gives the prefix of the full scan
        pytest.param({"n_samples": 1000, "n_features": 3, "seed": 42}, lambda lf: lf.head(137), id="n_rows_pushdown_head"),
        pytest.param(_REG_2000, lambda lf: lf.select("x0", "y0").filter(pl.col("x0") > 0), id="predicate_and_projection_pushdown"),
        pytest.param(_REG_5000_W, lambda lf: lf.head(321), id="n_workers_pushdowns-head"),
        pytest.param(_REG_5000_W, lambda lf: lf.select(["x0", "y1"]), id="n_workers_pushdowns-select"),
    ],
)
def test_pushdown_matches_collected(kwargs, op):
    got = assert_matches_native(op(REG(**kwargs)), op(REG(**kwargs).collect().lazy()), None, check_row_order=True)
    assert got.height > 0


def test_n_workers_head_pushdown_terminates_early():
    """An early n_rows stops the pool without generating the declared rows."""
    assert REG(n_samples=50_000_000, n_features=4, seed=1, n_workers=4).head(5).collect().height == 5


_P4 = {"start_date": _PANEL_START, "end_date": _PANEL_END_4}

INVALID = [
    Case("invalid_args-n_features", make=lambda: REG(n_samples=10, n_features=0), match="n_features"),
    Case("invalid_args-epsilon_scale", make=lambda: REG(n_samples=10, n_features=2, epsilon_scale=-1.0), match="epsilon_scale"),
    Case("invalid_args-betas", make=lambda: REG(n_samples=10, n_features=2, betas=np.zeros((3, 1))), match="betas must have shape"),
    Case(
        "invalid_args-weights", make=lambda: REG(n_samples=10, n_features=2, use_weights=True, weights_low=1.0, weights_high=0.5), match="weights_low"
    ),
    Case("panel_group_by_invalid_args-no_values", make=lambda: PANEL(**_P4, n_symbols=3, n_features=2, group_by=("g", [])), match="group_by values"),
    # betas must be (n_groups, n_features, n_responses)
    Case(
        "panel_group_by_invalid_args-betas",
        make=lambda: PANEL(**_P4, n_symbols=3, n_features=2, group_by=("g", ["A", "B", "C"]), betas=np.zeros((2, 2, 1))),
        match="betas must have shape",
    ),
    Case(
        "panel_group_by_invalid_args-feature_name",
        make=lambda: PANEL(**_P4, n_symbols=3, n_features=2, group_by=("x0", ["A", "B"])),
        match="group_by name",
    ),
    Case(
        "panel_group_by_invalid_args-category_name",
        make=lambda: PANEL(**_P4, n_symbols=3, n_features=2, categories=[["u", "v"]], group_by=("category_0", ["A", "B"])),
        match="group_by name",
    ),
    Case("panel_invalid_args-n_symbols", make=lambda: PANEL(**_P4, n_symbols=0, n_features=2), match="n_symbols"),
    Case("panel_invalid_args-n_features", make=lambda: PANEL(**_P4, n_symbols=5, n_features=0), match="n_features"),
    Case("panel_invalid_args-betas", make=lambda: PANEL(**_P4, n_symbols=5, n_features=2, betas=np.zeros((3, 1))), match="betas must have shape"),
    Case("panel_invalid_args-categories", make=lambda: PANEL(**_P4, n_symbols=5, n_features=2, categories=[[]]), match="categories"),
    Case(
        "panel_invalid_args-weights",
        make=lambda: PANEL(**_P4, n_symbols=3, n_features=2, use_weights=True, weights_low=1.0, weights_high=0.5),
        match="weights_low",
    ),
    Case("n_workers_invalid-regression", make=lambda: REG(n_samples=10, n_features=2, seed=1, n_workers=0), match="n_workers must be >= 1"),
    Case(
        "n_workers_invalid-panel",
        make=lambda: PANEL(start_date=date(2020, 1, 1), end_date=date(2020, 1, 3), n_features=2, seed=1, n_workers=0),
        match="n_workers must be >= 1",
    ),
]


@pytest.mark.parametrize(("make", "match"), params(INVALID, "make", "match"))
def test_invalid_args(make, match):
    with pytest.raises(ValueError, match=match):
        make()


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


def test_panel_streaming_group_by():
    lf = scan_synthetic_panel(start_date=_PANEL_START, end_date=_PANEL_END_MONTH, n_symbols=10, n_features=3, seed=7).set_sorted("date")
    counts = lf.group_by("date", maintain_order=True).agg(pl.len().alias("n")).collect(engine="streaming")
    assert (counts["n"] == 10).all()


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
