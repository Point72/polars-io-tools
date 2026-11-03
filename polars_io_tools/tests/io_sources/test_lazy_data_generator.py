"""scan_synthetic_regression and scan_synthetic_panel.

Tables: least-squares fits on generated data that must recover the coefficients ("betas") used to generate it; pairs
of ways to generate data that must give the same frames; query steps on the reader that must give what they give on
the collected data; and invalid arguments that must raise.
"""

from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from datetime import date

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_io_tools.io_sources import scan_synthetic_panel, scan_synthetic_regression
from polars_io_tools.tests.helpers.assertions import assert_matches_plain_polars
from polars_io_tools.tests.helpers.cases import as_params

_PANEL_START = date(2020, 1, 6)  # Monday
_PANEL_END_4 = date(2020, 1, 9)  # 4 business days
_PANEL_END_10 = date(2020, 1, 15)  # 10 calendar days (default freq="1D" includes weekends)
_PANEL_END_MONTH = date(2020, 1, 31)
_PANEL_200_BUSINESS_DAYS = {"start_date": date(2020, 1, 6), "end_date": date(2020, 10, 13), "n_symbols": 50}  # ~200 business days


def _least_squares_coefficients(X, y, weights=None, intercept=False):
    """Least-squares coefficients of y on X; ``weights`` gives WLS, ``intercept`` fits and drops a constant."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if weights is not None:
        sqrt_w = np.sqrt(weights)
        X, y = X * sqrt_w[:, None], y * sqrt_w
    A = np.column_stack([X, np.ones(len(X))]) if intercept else X
    sol = np.linalg.lstsq(A, y, rcond=None)[0]
    return sol[:-1] if intercept else sol


@dataclass(frozen=True, kw_only=True)
class BetaFit:
    """A least-squares fit of a response on the features; each fitted coefficient must be within ``tolerance`` of
    ``betas``."""

    betas: np.ndarray
    tolerance: float
    # the rows to fit on; None fits every row
    rows: pl.Expr | None = None
    response: str = "y0"
    # weight the fit by the ``weight`` column
    weighted: bool = False
    # fit a constant term too
    intercept: bool = False


_FIVE_BETAS = np.array([0.45, 0.23, -0.42, 0.22, -0.88])
_BETAS_PER_GROUP = np.array([[0.5, -0.3, 0.8], [1.2, 0.1, -0.5], [-0.9, 0.4, 0.2], [0.0, 0.7, -0.6]])
_GROUPS = ["alpha", "beta", "gamma", "delta"]
_BETAS_FOR_TWO_RESPONSES = np.array([[0.5, -0.3], [0.2, 0.7], [-0.4, 0.1]])
_ONE_SET_OF_BETAS = np.array([0.7, -0.3, 0.5])


@dataclass(frozen=True)
class BetaFitCase:
    """Generated data, and fits that must recover the betas it was generated with."""

    id: str
    _: KW_ONLY
    # the generator, called with ``scan_arguments``
    scan: Callable[..., pl.LazyFrame]
    scan_arguments: dict
    fits: list[BetaFit]
    # the fits use the features ``x0``, ``x1``, ... ``x{n_features - 1}``
    n_features: int


BETA_FITS = [
    BetaFitCase(
        "beta_fitting",
        scan=scan_synthetic_regression,
        scan_arguments={"n_samples": 10_000, "n_features": 5, "n_responses": 1, "betas": _FIVE_BETAS[:, None], "seed": 1234},
        fits=[BetaFit(betas=_FIVE_BETAS, tolerance=0.05, intercept=True)],
        n_features=5,
    ),
    # use_weights scales the noise variance by 1 / w, so a WLS fit recovers the betas
    BetaFitCase(
        "wls_beta_recovery",
        scan=scan_synthetic_regression,
        scan_arguments={
            "n_samples": 10_000,
            "n_features": 5,
            "betas": _FIVE_BETAS[:, None],
            "use_weights": True,
            "weights_low": 0.2,
            "weights_high": 2.5,
            "seed": 1234,
        },
        fits=[BetaFit(betas=_FIVE_BETAS, tolerance=0.05, weighted=True)],
        n_features=5,
    ),
    BetaFitCase(
        "multi_response_beta_recovery",
        scan=scan_synthetic_regression,
        scan_arguments={"n_samples": 10_000, "n_features": 3, "n_responses": 2, "betas": _BETAS_FOR_TWO_RESPONSES, "seed": 1},
        fits=[BetaFit(response=f"y{j}", betas=_BETAS_FOR_TWO_RESPONSES[:, j], tolerance=0.05) for j in range(2)],
        n_features=3,
    ),
    BetaFitCase(
        "panel_per_group_betas_recovered",
        scan=scan_synthetic_panel,
        scan_arguments={**_PANEL_200_BUSINESS_DAYS, "n_features": 3, "betas": _BETAS_PER_GROUP[:, :, None], "group_by": ("g", _GROUPS), "seed": 42},
        fits=[BetaFit(rows=pl.col("g") == g, betas=_BETAS_PER_GROUP[i], tolerance=0.1) for i, g in enumerate(_GROUPS)],
        n_features=3,
    ),
    # one global beta: the whole frame and every chunk recover it
    BetaFitCase(
        "chunked_single_global_beta",
        scan=scan_synthetic_regression,
        scan_arguments={
            "n_samples": 20_000,
            "n_features": 3,
            "betas": _ONE_SET_OF_BETAS[:, None],
            "chunk_key": "chunk",
            "n_chunks": 10,
            "seed": 1234,
        },
        fits=[
            BetaFit(betas=_ONE_SET_OF_BETAS, tolerance=0.05),
            *[BetaFit(rows=pl.col("chunk") == k, betas=_ONE_SET_OF_BETAS, tolerance=0.15) for k in range(10)],
        ],
        n_features=3,
    ),
    BetaFitCase(
        "panel_beta_recovery",
        scan=scan_synthetic_panel,
        scan_arguments={
            **_PANEL_200_BUSINESS_DAYS,
            "n_features": 4,
            "betas": np.array([[1.5], [-2.0], [0.5], [0.8]]),
            "epsilon_scale": 0.3,
            "seed": 42,
        },
        fits=[BetaFit(betas=np.array([1.5, -2.0, 0.5, 0.8]), tolerance=0.01)],
        n_features=4,
    ),
    BetaFitCase(
        "n_workers_beta_recovery",
        scan=scan_synthetic_regression,
        scan_arguments={
            "n_samples": 200_000,
            "n_features": 4,
            "n_responses": 1,
            "betas": np.array([1.0, -2.0, 0.5, 3.0]),
            "epsilon_scale": 0.1,
            "seed": 3,
            "n_workers": 8,
        },
        fits=[BetaFit(betas=np.array([1.0, -2.0, 0.5, 3.0]), tolerance=0.02, intercept=True)],
        n_features=4,
    ),
]


@pytest.mark.parametrize("case", as_params(BETA_FITS))
def test_beta_recovery(case: BetaFitCase):
    df = case.scan(**case.scan_arguments).collect()
    assert {f"y{j}" for j in range(case.scan_arguments.get("n_responses", 1))} <= set(df.columns)
    features = [f"x{i}" for i in range(case.n_features)]
    for fit in case.fits:
        sub = df if fit.rows is None else df.filter(fit.rows)
        assert sub.height > 0
        weights = sub["weight"].to_numpy() if fit.weighted else None
        fitted = _least_squares_coefficients(sub.select(features).to_numpy(), sub[fit.response].to_numpy(), weights, fit.intercept)
        assert np.max(np.abs(fitted - fit.betas)) < fit.tolerance


def _collect_with_engine(name):
    return lambda lf: lf.collect(engine=name)


def _collect_default(lf):
    return lf.collect()


_REGRESSION_1000_ROWS = {"n_samples": 1000, "n_features": 3, "seed": 42}
_PANEL_ONE_WEEK = {"start_date": date(2020, 1, 6), "end_date": date(2020, 1, 13), "n_symbols": 8, "n_features": 3, "seed": 42}
_PANEL_500_SYMBOLS = {"start_date": date(2020, 1, 1), "end_date": date(2020, 2, 1), "n_symbols": 500, "n_features": 4, "seed": 5}
_REGRESSION_WITH_WEIGHTS = {"n_samples": 12_345, "n_features": 6, "n_responses": 2, "use_weights": True, "seed": 7}


@dataclass(frozen=True)
class SameDataCase:
    """Two runs of a generator, and how their frames must compare."""

    id: str
    _: KW_ONLY
    # the generator, called with ``scan_arguments``
    scan: Callable[..., pl.LazyFrame]
    scan_arguments: dict
    # each run is extra scan arguments, or a function that collects the lazy frame
    first_run: dict | Callable[[pl.LazyFrame], pl.DataFrame]
    second_run: dict | Callable[[pl.LazyFrame], pl.DataFrame]
    # compare only these columns; None compares every column
    compare_columns: list[str] | None = None
    # True: the frames are equal; False: same schema and shape but different values; None: same schema and shape
    frames_equal: bool | None = True


DETERMINISM = [
    SameDataCase(
        "seed_reproducibility",
        scan=scan_synthetic_regression,
        scan_arguments={"n_samples": 500, "n_features": 3, "seed": 42},
        first_run={},
        second_run={},
    ),
    # row-indexed RNG streams: the batch size does not change the rows
    SameDataCase(
        "batch_size_independence",
        scan=scan_synthetic_regression,
        scan_arguments=_REGRESSION_1000_ROWS,
        first_run={"fetch_size": 1000},
        second_run={"fetch_size": 37},
    ),
    # a fetch size smaller than a chunk subdivides it
    SameDataCase(
        "chunked_batch_size_independence",
        scan=scan_synthetic_regression,
        scan_arguments={**_REGRESSION_1000_ROWS, "chunk_key": "chunk", "n_chunks": 8},
        first_run={"fetch_size": 1000},
        second_run={"fetch_size": 37},
    ),
    # a fetch size smaller than n_symbols must not straddle dates or shuffle symbols
    SameDataCase(
        "panel_batch_size_independence",
        scan=scan_synthetic_panel,
        scan_arguments=_PANEL_ONE_WEEK,
        first_run={"fetch_size": 1000},
        second_run={"fetch_size": 3},
    ),
    SameDataCase(
        "panel_group_by_batch_size_independence",
        scan=scan_synthetic_panel,
        scan_arguments={**_PANEL_ONE_WEEK, "group_by": ("g", ["A", "B", "C", "D", "E"])},
        first_run={"fetch_size": 1000},
        second_run={"fetch_size": 3},
    ),
    # the group draw has its own seed stream, so decorative categories do not shift it
    SameDataCase(
        "panel_group_by_seed_independent_of_categories",
        scan=scan_synthetic_panel,
        scan_arguments={
            "start_date": _PANEL_START,
            "end_date": _PANEL_END_10,
            "n_symbols": 4,
            "n_features": 2,
            "group_by": ("g", ["A", "B", "C"]),
            "seed": 7,
        },
        first_run={},
        second_run={"categories": [["u", "v"]]},
        compare_columns=["g"],
    ),
    # with n_workers, a fixed (seed, n_workers, fetch_size) reproduces, whatever batch size the engine asks for
    SameDataCase(
        "n_workers_determinism_and_schema_parity-repeat",
        scan=scan_synthetic_regression,
        scan_arguments=_REGRESSION_WITH_WEIGHTS,
        first_run={"n_workers": 4},
        second_run={"n_workers": 4},
    ),
    # threading changes the RNG stream layout
    SameDataCase(
        "n_workers_determinism_and_schema_parity-serial",
        scan=scan_synthetic_regression,
        scan_arguments=_REGRESSION_WITH_WEIGHTS,
        first_run={"n_workers": 4},
        second_run={"n_workers": 1},
        frames_equal=False,
    ),
    SameDataCase(
        "n_workers_batch_size_independence",
        scan=scan_synthetic_regression,
        scan_arguments={"n_samples": 4000, "n_features": 4, "seed": 42, "n_workers": 8, "fetch_size": 512},
        first_run=_collect_default,
        second_run=_collect_with_engine("streaming"),
    ),
    # several fixed blocks of n_workers * fetch_size rows
    SameDataCase(
        "n_workers_multi_block_batch_size_independence",
        scan=scan_synthetic_regression,
        scan_arguments={"n_samples": 3300, "n_features": 3, "n_responses": 2, "use_weights": True, "seed": 9, "n_workers": 4, "fetch_size": 250},
        first_run=_collect_default,
        second_run=_collect_with_engine("streaming"),
    ),
    SameDataCase(
        "n_workers_panel_parity-repeat",
        scan=scan_synthetic_panel,
        scan_arguments=_PANEL_500_SYMBOLS,
        first_run={"n_workers": 4},
        second_run={"n_workers": 4},
    ),
    SameDataCase(
        "n_workers_panel_parity-serial",
        scan=scan_synthetic_panel,
        scan_arguments=_PANEL_500_SYMBOLS,
        first_run={"n_workers": 4},
        second_run={"n_workers": 1},
        frames_equal=None,
    ),
    # x and eps come from the block filler, the other columns from per-batch streams; they stay aligned
    SameDataCase(
        "n_workers_panel_batch_size_independence_full",
        scan=scan_synthetic_panel,
        scan_arguments={
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
        first_run=_collect_default,
        second_run=_collect_with_engine("streaming"),
    ),
]


@pytest.mark.parametrize("case", as_params(DETERMINISM))
def test_same_data(case: SameDataCase):
    def generate(run):
        return run(case.scan(**case.scan_arguments)) if callable(run) else case.scan(**case.scan_arguments, **run).collect()

    first, second = generate(case.first_run), generate(case.second_run)
    if case.compare_columns:
        first, second = first.select(case.compare_columns), second.select(case.compare_columns)
    if case.frames_equal:
        assert_frame_equal(first, second)
    else:
        assert (first.schema, first.shape) == (second.schema, second.shape)
        assert case.frames_equal is None or not first.equals(second)


_REGRESSION_2000_ROWS = {"n_samples": 2000, "n_features": 3, "seed": 0}
_REGRESSION_5000_ROWS_8_WORKERS = {"n_samples": 5000, "n_features": 5, "n_responses": 2, "seed": 1, "n_workers": 8}


@dataclass(frozen=True)
class GeneratorPushdownCase:
    """Generator arguments and a query; the lazy query must return what the query on the collected frame returns."""

    id: str
    _: KW_ONLY
    # for ``scan_synthetic_regression``
    kwargs: dict
    op: Callable[[pl.LazyFrame], pl.LazyFrame]


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            # head is pushed down as n_rows and gives the prefix of the full scan
            GeneratorPushdownCase("n_rows_pushdown_head", kwargs={"n_samples": 1000, "n_features": 3, "seed": 42}, op=lambda lf: lf.head(137)),
            GeneratorPushdownCase(
                "predicate_and_projection_pushdown", kwargs=_REGRESSION_2000_ROWS, op=lambda lf: lf.select("x0", "y0").filter(pl.col("x0") > 0)
            ),
            GeneratorPushdownCase("n_workers_pushdowns-head", kwargs=_REGRESSION_5000_ROWS_8_WORKERS, op=lambda lf: lf.head(321)),
            GeneratorPushdownCase("n_workers_pushdowns-select", kwargs=_REGRESSION_5000_ROWS_8_WORKERS, op=lambda lf: lf.select(["x0", "y1"])),
        ]
    ),
)
def test_pushdown_matches_collected(case: GeneratorPushdownCase):
    got = assert_matches_plain_polars(
        case.op(scan_synthetic_regression(**case.kwargs)),
        case.op(scan_synthetic_regression(**case.kwargs).collect().lazy()),
        None,
        check_row_order=True,
    )
    assert got.height > 0


def test_n_workers_head_pushdown_terminates_early():
    """An early n_rows stops the pool without generating the declared rows."""
    assert scan_synthetic_regression(n_samples=50_000_000, n_features=4, seed=1, n_workers=4).head(5).collect().height == 5


_FOUR_BUSINESS_DAYS = {"start_date": _PANEL_START, "end_date": _PANEL_END_4}


@dataclass(frozen=True)
class InvalidArgumentsCase:
    """A generator call with invalid arguments, and the text its ValueError must contain."""

    id: str
    _: KW_ONLY
    make: Callable[[], object]
    match: str


INVALID = [
    InvalidArgumentsCase("invalid_args-n_features", make=lambda: scan_synthetic_regression(n_samples=10, n_features=0), match="n_features"),
    InvalidArgumentsCase(
        "invalid_args-epsilon_scale", make=lambda: scan_synthetic_regression(n_samples=10, n_features=2, epsilon_scale=-1.0), match="epsilon_scale"
    ),
    InvalidArgumentsCase(
        "invalid_args-betas",
        make=lambda: scan_synthetic_regression(n_samples=10, n_features=2, betas=np.zeros((3, 1))),
        match="betas must have shape",
    ),
    InvalidArgumentsCase(
        "invalid_args-weights",
        make=lambda: scan_synthetic_regression(n_samples=10, n_features=2, use_weights=True, weights_low=1.0, weights_high=0.5),
        match="weights_low",
    ),
    InvalidArgumentsCase(
        "panel_group_by_invalid_args-no_values",
        make=lambda: scan_synthetic_panel(**_FOUR_BUSINESS_DAYS, n_symbols=3, n_features=2, group_by=("g", [])),
        match="group_by values",
    ),
    # betas must be (n_groups, n_features, n_responses)
    InvalidArgumentsCase(
        "panel_group_by_invalid_args-betas",
        make=lambda: scan_synthetic_panel(
            **_FOUR_BUSINESS_DAYS, n_symbols=3, n_features=2, group_by=("g", ["A", "B", "C"]), betas=np.zeros((2, 2, 1))
        ),
        match="betas must have shape",
    ),
    InvalidArgumentsCase(
        "panel_group_by_invalid_args-feature_name",
        make=lambda: scan_synthetic_panel(**_FOUR_BUSINESS_DAYS, n_symbols=3, n_features=2, group_by=("x0", ["A", "B"])),
        match="group_by name",
    ),
    InvalidArgumentsCase(
        "panel_group_by_invalid_args-category_name",
        make=lambda: scan_synthetic_panel(
            **_FOUR_BUSINESS_DAYS, n_symbols=3, n_features=2, categories=[["u", "v"]], group_by=("category_0", ["A", "B"])
        ),
        match="group_by name",
    ),
    InvalidArgumentsCase(
        "panel_invalid_args-n_symbols", make=lambda: scan_synthetic_panel(**_FOUR_BUSINESS_DAYS, n_symbols=0, n_features=2), match="n_symbols"
    ),
    InvalidArgumentsCase(
        "panel_invalid_args-n_features", make=lambda: scan_synthetic_panel(**_FOUR_BUSINESS_DAYS, n_symbols=5, n_features=0), match="n_features"
    ),
    InvalidArgumentsCase(
        "panel_invalid_args-betas",
        make=lambda: scan_synthetic_panel(**_FOUR_BUSINESS_DAYS, n_symbols=5, n_features=2, betas=np.zeros((3, 1))),
        match="betas must have shape",
    ),
    InvalidArgumentsCase(
        "panel_invalid_args-categories",
        make=lambda: scan_synthetic_panel(**_FOUR_BUSINESS_DAYS, n_symbols=5, n_features=2, categories=[[]]),
        match="categories",
    ),
    InvalidArgumentsCase(
        "panel_invalid_args-weights",
        make=lambda: scan_synthetic_panel(**_FOUR_BUSINESS_DAYS, n_symbols=3, n_features=2, use_weights=True, weights_low=1.0, weights_high=0.5),
        match="weights_low",
    ),
    InvalidArgumentsCase(
        "n_workers_invalid-regression",
        make=lambda: scan_synthetic_regression(n_samples=10, n_features=2, seed=1, n_workers=0),
        match="n_workers must be >= 1",
    ),
    InvalidArgumentsCase(
        "n_workers_invalid-panel",
        make=lambda: scan_synthetic_panel(start_date=date(2020, 1, 1), end_date=date(2020, 1, 3), n_features=2, seed=1, n_workers=0),
        match="n_workers must be >= 1",
    ),
]


@pytest.mark.parametrize("case", as_params(INVALID))
def test_invalid_args(case: InvalidArgumentsCase):
    with pytest.raises(ValueError, match=case.match):
        case.make()


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
