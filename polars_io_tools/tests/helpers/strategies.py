"""Hypothesis strategies for property tests: typed values and frames with nulls, and random predicate shapes.

Frames hold nulls and a wide mix of dtypes. Values are drawn in a physical form (integer ticks for temporal dtypes,
unscaled integers for Decimal, strings for String, Enum and Categorical) and built into Series by Polars itself.
"""

import functools
import math
import operator
from decimal import Context, Decimal

import polars as pl
from hypothesis import strategies as st

from polars_io_tools.tests.helpers.cases import COLUMN_DTYPES, DST_US, INT64, TEXT

POLARS_ERRORS = (Exception, pl.exceptions.PanicException)
"""What evaluating an expression with Polars can raise; a Rust panic is not an ``Exception``."""

_SCALE = {"ns": 1000, "us": 1, "ms": 0.001}
_PYTHON_SECONDS = (-62_135_596_800, 253_402_300_799)  # 0001-01-01 and 9999-12-31 23:59:59
_NS_EXTREMES = (INT64[0] + 2 * 86_400 * 10**9, INT64[1] - 2 * 86_400 * 10**9)

OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "eq_missing": lambda a, b: a.eq_missing(b),
    "ne_missing": lambda a, b: a.ne_missing(b),
}


def _detect_implode() -> bool:
    """Whether ``is_in`` takes its collection as a list scalar (``implode``) rather than as a column."""
    try:
        out = pl.DataFrame({"a": [1, 2]}).select(pl.col("a").is_in(pl.lit(pl.Series([1, 2])).implode()))
        return out.to_series().to_list() == [True, True]
    except POLARS_ERRORS:
        return False


IMPLODE = _detect_implode()


def collection(series: pl.Series) -> pl.Expr:
    """An ``is_in`` collection literal holding ``series`` in its own dtype."""
    return pl.lit(series.implode()) if IMPLODE else pl.lit(series)


def physical(dtype: pl.DataType) -> st.SearchStrategy:
    """Non-null physical values of ``dtype``."""
    if dtype.is_integer():
        lo, hi = pl.select(lo=dtype.min(), hi=dtype.max()).row(0)
        return st.one_of(st.integers(lo, hi), st.sampled_from([lo, hi, 0, -1 if lo < 0 else 1]))
    if dtype == pl.Boolean:
        return st.booleans()
    if dtype.is_float():
        width = 32 if dtype == pl.Float32 else 64
        return st.one_of(st.floats(width=width), st.sampled_from([math.nan, -0.0, 0.0, 1.0, math.inf]))
    if dtype == pl.Date:
        return st.one_of(st.integers(19_700, 20_100), st.integers(-200_000, 200_000))
    if isinstance(dtype, pl.Datetime):
        scale = _SCALE[dtype.time_unit]
        near = st.builds(lambda b, d: int((b + d) * scale), st.sampled_from(DST_US), st.integers(-7_200_000_000, 7_200_000_000))
        wide = st.integers(-2_000_000_000_000_000, 4_000_000_000_000_000).map(lambda x: int(x * scale))
        if dtype.time_zone is not None:
            return st.one_of(near, wide)
        # The extremes stay inside the range of Python's datetime, and nanosecond values a day away from the Int64
        # limits: beyond them are known crashes (see test_properties), where a Polars panic leaves the expression parser
        # unusable, caches cannot key a partition, or day and microsecond bounds overflow.
        per_second = int(10**6 * scale)
        extremes = _NS_EXTREMES if dtype.time_unit == "ns" else (_PYTHON_SECONDS[0] * per_second, _PYTHON_SECONDS[1] * per_second)
        return st.one_of(near, wide, st.sampled_from(extremes))
    if isinstance(dtype, pl.Duration):
        return st.one_of(st.integers(-(10**12), 10**12), st.sampled_from([*INT64, 0, 500]))
    if dtype == pl.Time:
        return st.one_of(st.integers(0, 86_400 * 10**9 - 1), st.sampled_from([0, 500, 1000, 86_400 * 10**9 - 1]))
    if isinstance(dtype, pl.Decimal):
        bound = 10 ** (dtype.precision or 38) - 1
        return st.one_of(st.integers(-bound, bound), st.integers(-1000, 1000), st.sampled_from([-bound, bound, 0]))
    if isinstance(dtype, pl.Enum):
        return st.sampled_from(dtype.categories.to_list())
    if dtype == pl.Categorical:
        return st.sampled_from(["x", "y", "zz", "None", ""])
    if dtype == pl.String:
        return st.one_of(st.sampled_from(TEXT), st.text(alphabet="ab/é", max_size=3))
    raise AssertionError(dtype)


def native(dtype: pl.DataType, vals: list) -> pl.Series:
    """A Series of ``dtype`` holding physical values ``vals`` (``None`` is null), built by Polars."""
    if dtype.is_float():
        return pl.Series(vals, dtype=pl.Float64).cast(dtype)
    if dtype == pl.Date:
        return pl.Series(vals, dtype=pl.Int32).cast(dtype)
    if isinstance(dtype, (pl.Datetime, pl.Duration)) or dtype == pl.Time:
        return pl.Series(vals, dtype=pl.Int64).cast(dtype)
    if isinstance(dtype, pl.Decimal):
        ctx = Context(prec=80)
        return pl.Series([None if v is None else Decimal(v).scaleb(-dtype.scale, ctx) for v in vals], dtype=dtype)
    if isinstance(dtype, pl.Enum):
        return pl.Series(vals, dtype=pl.String).cast(dtype)
    return pl.Series(vals, dtype=dtype)


def python_value(value, dtype: pl.DataType):
    """``value`` as the Python object Polars gives for it; ``ValueError`` if Python cannot hold it."""
    try:
        return native(dtype, [value]).to_list()[0]
    except POLARS_ERRORS as e:
        raise ValueError(value) from e


def typed_lit(value, dtype: pl.DataType) -> pl.Expr:
    """A literal of ``dtype`` holding the physical ``value`` exactly."""
    if value is None:
        return pl.lit(None, dtype=dtype)
    if dtype == pl.Date:
        return pl.lit(value, dtype=pl.Int32).cast(dtype)
    if isinstance(dtype, (pl.Datetime, pl.Duration)) or dtype == pl.Time:
        return pl.lit(value, dtype=pl.Int64).cast(dtype)
    if isinstance(dtype, pl.Decimal):
        return pl.lit(python_value(value, dtype), dtype=dtype)
    return pl.lit(value, dtype=dtype)


def _literal(draw, dtype: pl.DataType, pool: list) -> pl.Expr:
    """A literal of ``dtype``, sometimes as the dynamic Python object a user would write."""
    value = draw(st.one_of(*([st.sampled_from(pool)] if pool else []), physical(dtype), st.none()))
    if value is not None and draw(st.booleans()):
        try:
            return pl.lit(python_value(value, dtype))
        except ValueError:
            pass
    return typed_lit(value, dtype)


def _leaf(draw, schema: dict, pools: dict, negated: bool = False) -> pl.Expr:
    name = draw(st.sampled_from(sorted(schema)))
    dtype, pool, col = schema[name], pools[name], pl.col(name)
    kinds = ["cmp", "cmp", "in", "between", "null", "const"]
    # Polars issue "IO plugin receives all_horizontal(int, bool) without the cast": negated, it is passed as a bitwise NOT.
    kinds += ["bitwise", *([] if negated else ["horizontal"])] if dtype.is_integer() else []
    kinds += ["bool"] if dtype == pl.Boolean else []
    kinds += ["pattern"] if dtype == pl.String else []
    kinds += ["date"] if isinstance(dtype, pl.Datetime) else []
    kinds += ["cast"] if dtype.is_numeric() else []
    kinds += ["temporal_cast"] if isinstance(dtype, (pl.Date, pl.Datetime)) else []
    kind = draw(st.sampled_from(kinds))
    if kind == "cmp":
        op = OPS[draw(st.sampled_from(sorted(OPS)))]
        lit = _literal(draw, dtype, pool)
        return op(lit, col) if draw(st.booleans()) and op in (operator.lt, operator.le, operator.eq) else op(col, lit)
    if kind == "in":
        members = draw(st.lists(st.one_of(st.sampled_from(pool), st.none()), max_size=3))
        series = native(dtype, members)
        # Polars 1.x compares a list of Python datetimes or timedeltas with a column of another time unit by raw ticks,
        # or rejects it; library code then fails on the misread membership, so such lists are typed.
        python_list_ok = not (isinstance(dtype, (pl.Datetime, pl.Duration)) and dtype.time_unit != "us") and dtype != pl.Time
        if draw(st.booleans()) or not python_list_ok:
            return col.is_in(collection(series), nulls_equal=draw(st.booleans()))
        try:
            return col.is_in([v for v in series.to_list() if v is not None])
        except POLARS_ERRORS:
            return col.is_in(collection(series))
    if kind == "between":
        closed = draw(st.sampled_from(["both", "left", "right", "none"]))
        return col.is_between(_literal(draw, dtype, pool), _literal(draw, dtype, pool), closed=closed)
    if kind == "null":
        return col.is_null() if draw(st.booleans()) else col.is_not_null()
    if kind == "const":
        return pl.lit(draw(st.sampled_from([True, False, None])), dtype=pl.Boolean)
    if kind == "bitwise":
        other = pl.col(draw(st.sampled_from([n for n, d in schema.items() if d.is_integer()])))
        return [(col & other) > 0, (col | other).cast(pl.Boolean), (~col) == -1][draw(st.integers(0, 2))]
    if kind == "horizontal":
        return pl.all_horizontal(col, col > 0) if draw(st.booleans()) else pl.any_horizontal(col)
    if kind == "bool":
        return col if draw(st.booleans()) else ~col
    if kind == "pattern":
        text = draw(st.sampled_from([p for p in pool if p] or ["a"]))[:1]
        return draw(st.sampled_from([col.str.starts_with(text), col.str.ends_with(text), col.str.contains(text, literal=True)]))
    if kind == "date":
        day = draw(st.integers(19_700, 20_100))
        return OPS[draw(st.sampled_from(["==", "<", ">=", "!="]))](col.dt.date(), typed_lit(day, pl.Date()))
    if kind == "temporal_cast":
        target = draw(st.sampled_from([pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("ns"), pl.Datetime("us", "UTC"), pl.Date()]))
        return OPS[draw(st.sampled_from(sorted(OPS)))](col.cast(target, strict=False), _literal(draw, target, []))
    target = draw(st.sampled_from([pl.Float64(), pl.Int16()]))
    cast = col.cast(target, strict=False) if draw(st.booleans()) else col.cast(target, wrap_numerical=True)
    return OPS[draw(st.sampled_from(["==", "<", ">="]))](cast, typed_lit(draw(physical(target)), target))


def predicate(draw, schema: dict, pools: dict, depth: int, negated: bool = False) -> pl.Expr:
    """A random Boolean predicate over the columns of ``schema``, comparing with values drawn mostly from ``pools``."""
    if depth == 0 or draw(st.integers(0, 2)) == 0:
        return _leaf(draw, schema, pools, negated)
    sub = functools.partial(predicate, draw, schema, pools, depth - 1, negated)
    shape = draw(st.sampled_from(["and", "and", "or", "or", "xor", "not", "when", "all", "any", "is_null", "is_not_null", "cast"]))
    if shape in ("and", "or", "xor"):
        return {"and": operator.and_, "or": operator.or_, "xor": operator.xor}[shape](sub(), sub())
    if shape == "not":
        return ~predicate(draw, schema, pools, depth - 1, negated=True)
    if shape == "when":
        return pl.when(sub()).then(sub()).otherwise(sub())
    if shape in ("all", "any"):
        return (pl.all_horizontal if shape == "all" else pl.any_horizontal)(sub(), sub(), sub())
    if shape == "cast":
        return sub().cast(pl.Boolean)
    return sub().is_null() if shape == "is_null" else sub().is_not_null()


def column(draw, dtype: pl.DataType, height: int) -> tuple[pl.Series, list]:
    """A column of ``dtype`` with nulls, and the pool of physical values it draws from."""
    pool = draw(st.lists(physical(dtype), min_size=1, max_size=4))
    series = native(dtype, draw(st.lists(st.sampled_from([*pool, None]), min_size=height, max_size=height)))
    if dtype.is_float() and draw(st.booleans()):
        series = series.set(series == 0.0, -0.0)
    return series, pool


@st.composite
def frames_and_predicates(draw, dtypes=COLUMN_DTYPES, max_columns: int = 3, height: int = 4, depth: int = 3):
    """A frame of up to ``max_columns`` columns named ``c0``, ``c1``... and a random predicate on it."""
    chosen = draw(st.lists(st.sampled_from(dtypes), min_size=1, max_size=max_columns))
    schema = {f"c{i}": d for i, d in enumerate(chosen)}
    columns, pools = {}, {}
    for name, dtype in schema.items():
        columns[name], pools[name] = column(draw, dtype, height)
    return pl.DataFrame(columns), predicate(draw, schema, pools, depth)
