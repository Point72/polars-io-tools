"""Random frames and random filters for the property tests, which use the Hypothesis library to generate inputs.

A frame has a few columns of mixed dtypes with nulls; a filter combines comparisons, ``is_in``, ``is_between`` and
other expressions on those columns. Values are first drawn as raw numbers or strings, the form Polars stores them in
(a count of time units for dates and times, an integer without the decimal point for Decimal), and Polars itself turns
them into Series, so every value is one Polars can hold.
"""

import functools
import math
import operator
from datetime import date, datetime
from decimal import Context, Decimal

import polars as pl
from hypothesis import strategies as st

from polars_io_tools.tests.helpers.cases import AWKWARD_STRINGS, COLUMN_DTYPES, INT64_MIN_MAX, NEW_YORK, NEW_YORK_2024_CLOCK_CHANGES_US
from polars_io_tools.tests.helpers.polars_versions import polars_has

POLARS_ERRORS = (Exception, pl.exceptions.PanicException)
"""What evaluating an expression with Polars can raise; a Rust panic is not an ``Exception``."""

_UNITS_PER_MICROSECOND = {"ns": 1000, "us": 1, "ms": 0.001}
_PYTHON_DATETIME_SECONDS = (-62_135_596_800, 253_402_300_799)  # 0001-01-01 and 9999-12-31 23:59:59
_NANOSECOND_EXTREMES = (INT64_MIN_MAX[0] + 2 * 86_400 * 10**9, INT64_MIN_MAX[1] - 2 * 86_400 * 10**9)

COMPARISONS = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "eq_missing": lambda a, b: a.eq_missing(b),
    "ne_missing": lambda a, b: a.ne_missing(b),
}


def is_in_values(series: pl.Series) -> pl.Expr:
    """``series`` as the argument of ``is_in``, keeping its dtype."""
    return pl.lit(series.implode()) if polars_has("is_in_takes_a_list_value") else pl.lit(series)


def raw_values(dtype: pl.DataType) -> st.SearchStrategy:
    """Draws non-null values of ``dtype`` in the raw form Polars stores them in, including extremes and edge values."""
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
        scale = _UNITS_PER_MICROSECOND[dtype.time_unit]
        near_clock_change = st.builds(
            lambda base, offset: int((base + offset) * scale),
            st.sampled_from(NEW_YORK_2024_CLOCK_CHANGES_US),
            st.integers(-7_200_000_000, 7_200_000_000),
        )
        wide = st.integers(-2_000_000_000_000_000, 4_000_000_000_000_000).map(lambda x: int(x * scale))
        if dtype.time_zone is not None:
            return st.one_of(near_clock_change, wide)
        # The extremes stay inside the range of Python's datetime, and nanosecond values a day away from the Int64
        # limits: beyond them are known crashes, where a Polars panic leaves the expression parser unusable, caches
        # cannot key a partition, or day and microsecond bounds overflow (strict xfails in test_pushdown_combine.py,
        # test_lazy_cache_memory.py, test_lazy_cache_parquet.py and test_partitions.py).
        per_second = int(10**6 * scale)
        extremes = (
            _NANOSECOND_EXTREMES if dtype.time_unit == "ns" else (_PYTHON_DATETIME_SECONDS[0] * per_second, _PYTHON_DATETIME_SECONDS[1] * per_second)
        )
        return st.one_of(near_clock_change, wide, st.sampled_from(extremes))
    if isinstance(dtype, pl.Duration):
        return st.one_of(st.integers(-(10**12), 10**12), st.sampled_from([*INT64_MIN_MAX, 0, 500]))
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
        return st.one_of(st.sampled_from(AWKWARD_STRINGS), st.text(alphabet="ab/é", max_size=3))
    raise AssertionError(dtype)


def series_from_raw(dtype: pl.DataType, vals: list) -> pl.Series:
    """A Series of ``dtype`` built by Polars from raw values as ``raw_values`` draws them (``None`` is null)."""
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


def python_value_from_raw(value, dtype: pl.DataType):
    """The Python object Polars returns for the raw ``value`` of ``dtype``; ``ValueError`` if Python cannot hold it."""
    try:
        return series_from_raw(dtype, [value]).to_list()[0]
    except POLARS_ERRORS as e:
        raise ValueError(value) from e


def literal_from_raw(value, dtype: pl.DataType) -> pl.Expr:
    """A ``pl.lit`` of ``dtype`` holding the raw ``value`` exactly."""
    if value is None:
        return pl.lit(None, dtype=dtype)
    if dtype == pl.Date:
        return pl.lit(value, dtype=pl.Int32).cast(dtype)
    if isinstance(dtype, (pl.Datetime, pl.Duration)) or dtype == pl.Time:
        return pl.lit(value, dtype=pl.Int64).cast(dtype)
    if isinstance(dtype, pl.Decimal):
        return pl.lit(python_value_from_raw(value, dtype), dtype=dtype)
    return pl.lit(value, dtype=dtype)


def _literal(draw, dtype: pl.DataType, column_values: list) -> pl.Expr:
    """A literal of ``dtype``, usually one of ``column_values``; sometimes written as a plain Python value without a
    dtype, as a user would write it."""
    value = draw(st.one_of(*([st.sampled_from(column_values)] if column_values else []), raw_values(dtype), st.none()))
    if value is not None and draw(st.booleans()):
        try:
            return pl.lit(python_value_from_raw(value, dtype))
        except ValueError:
            pass
    return literal_from_raw(value, dtype)


def _single_condition(draw, schema: dict, values_by_column: dict) -> pl.Expr:
    """A Boolean expression on one column (sometimes two), with no ``&``, ``|`` or ``~`` joining other conditions."""
    name = draw(st.sampled_from(sorted(schema)))
    dtype, column_values, col = schema[name], values_by_column[name], pl.col(name)
    kinds = ["compare", "compare", "in", "between", "null", "const"]
    # No horizontal function over integers: Polars issue "IO plugin receives `all_horizontal(int, bool).cast(Boolean)`
    # without the cast, and a lazy filter by it fails".
    kinds += ["bitwise"] if dtype.is_integer() else []
    kinds += ["bool"] if dtype == pl.Boolean else []
    kinds += ["pattern"] if dtype == pl.String else []
    kinds += ["date"] if isinstance(dtype, pl.Datetime) else []
    kinds += ["cast"] if dtype.is_numeric() else []
    kinds += ["temporal_cast"] if isinstance(dtype, (pl.Date, pl.Datetime)) else []
    kind = draw(st.sampled_from(kinds))
    if kind == "compare":
        op = COMPARISONS[draw(st.sampled_from(sorted(COMPARISONS)))]
        lit = _literal(draw, dtype, column_values)
        return op(lit, col) if draw(st.booleans()) and op in (operator.lt, operator.le, operator.eq) else op(col, lit)
    if kind == "in":
        members = draw(st.lists(st.one_of(st.sampled_from(column_values), st.none()), max_size=3))
        series = series_from_raw(dtype, members)
        # Polars 1.x compares a list of Python datetimes or timedeltas with a column of another time unit by raw ticks,
        # or rejects it; library code then fails on the misread membership, so such lists are typed.
        python_list_ok = not (isinstance(dtype, (pl.Datetime, pl.Duration)) and dtype.time_unit != "us") and dtype != pl.Time
        if draw(st.booleans()) or not python_list_ok:
            return col.is_in(is_in_values(series), nulls_equal=draw(st.booleans()))
        try:
            return col.is_in([v for v in series.to_list() if v is not None])
        except POLARS_ERRORS:
            return col.is_in(is_in_values(series))
    if kind == "between":
        closed = draw(st.sampled_from(["both", "left", "right", "none"]))
        return col.is_between(_literal(draw, dtype, column_values), _literal(draw, dtype, column_values), closed=closed)
    if kind == "null":
        return col.is_null() if draw(st.booleans()) else col.is_not_null()
    if kind == "const":
        return pl.lit(draw(st.sampled_from([True, False, None])), dtype=pl.Boolean)
    if kind == "bitwise":
        other = pl.col(draw(st.sampled_from([n for n, d in schema.items() if d.is_integer()])))
        return [(col & other) > 0, (col | other).cast(pl.Boolean), (~col) == -1][draw(st.integers(0, 2))]
    if kind == "bool":
        return col if draw(st.booleans()) else ~col
    if kind == "pattern":
        text = draw(st.sampled_from([value for value in column_values if value] or ["a"]))[:1]
        return draw(st.sampled_from([col.str.starts_with(text), col.str.ends_with(text), col.str.contains(text, literal=True)]))
    if kind == "date":
        day = draw(st.integers(19_700, 20_100))
        return COMPARISONS[draw(st.sampled_from(["==", "<", ">=", "!="]))](col.dt.date(), literal_from_raw(day, pl.Date()))
    if kind == "temporal_cast":
        target = draw(st.sampled_from([pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("ns"), pl.Datetime("us", "UTC"), pl.Date()]))
        return COMPARISONS[draw(st.sampled_from(sorted(COMPARISONS)))](col.cast(target, strict=False), _literal(draw, target, []))
    target = draw(st.sampled_from([pl.Float64(), pl.Int16()]))
    cast = col.cast(target, strict=False) if draw(st.booleans()) else col.cast(target, wrap_numerical=True)
    return COMPARISONS[draw(st.sampled_from(["==", "<", ">="]))](cast, literal_from_raw(draw(raw_values(target)), target))


def random_filter(draw, schema: dict, values_by_column: dict, depth: int) -> pl.Expr:
    """A random Boolean expression over the columns of ``schema``, nested up to ``depth`` levels, comparing columns
    mostly with values from ``values_by_column`` so that some rows match."""
    if depth == 0 or draw(st.integers(0, 2)) == 0:
        return _single_condition(draw, schema, values_by_column)
    sub = functools.partial(random_filter, draw, schema, values_by_column, depth - 1)
    shape = draw(st.sampled_from(["and", "and", "or", "or", "xor", "not", "when", "all", "any", "is_null", "is_not_null", "cast"]))
    if shape in ("and", "or", "xor"):
        return {"and": operator.and_, "or": operator.or_, "xor": operator.xor}[shape](sub(), sub())
    if shape == "not":
        return ~sub()
    if shape == "when":
        return pl.when(sub()).then(sub()).otherwise(sub())
    if shape in ("all", "any"):
        return (pl.all_horizontal if shape == "all" else pl.any_horizontal)(sub(), sub(), sub())
    if shape == "cast":
        return sub().cast(pl.Boolean)
    return sub().is_null() if shape == "is_null" else sub().is_not_null()


def _random_column(draw, dtype: pl.DataType, height: int) -> tuple[pl.Series, list]:
    """A column of ``dtype`` with nulls, and the few raw values its non-null rows are drawn from."""
    values = draw(st.lists(raw_values(dtype), min_size=1, max_size=4))
    series = series_from_raw(dtype, draw(st.lists(st.sampled_from([*values, None]), min_size=height, max_size=height)))
    if dtype.is_float() and draw(st.booleans()):
        series = series.set(series == 0.0, -0.0)
    return series, values


@st.composite
def frames_and_predicates(draw, dtypes=COLUMN_DTYPES, max_columns: int = 3, height: int = 4, depth: int = 3):
    """Draws a frame of ``height`` rows and up to ``max_columns`` columns named ``c0``, ``c1``..., and a random filter on it."""
    chosen = draw(st.lists(st.sampled_from(dtypes), min_size=1, max_size=max_columns))
    schema = {f"c{i}": dtype for i, dtype in enumerate(chosen)}
    columns, values_by_column = {}, {}
    for name, dtype in schema.items():
        columns[name], values_by_column[name] = _random_column(draw, dtype, height)
    return pl.DataFrame(columns), random_filter(draw, schema, values_by_column, depth)


PARTITION_KEY_POOLS = {
    "int": pl.Series([1, 2, 3, None]),
    "uint64": pl.Series([0, 2**64 - 1, 2**63, None], dtype=pl.UInt64),
    "string": pl.Series(["a", "a/b", "None", "~2F", "", None]),
    "categorical": pl.Series(["a", "a/b", "", None], dtype=pl.Categorical),
    "enum": pl.Series(["c", "a", None], dtype=pl.Enum(["c", "a", "b"])),
    "decimal": pl.Series([Decimal("0.10"), Decimal("-1.25"), None], dtype=pl.Decimal(38, 2)),
    "datetime_us_ny": pl.Series([*NEW_YORK_2024_CLOCK_CHANGES_US, None]).cast(pl.Datetime("us", NEW_YORK)),
    "datetime_ms": pl.Series([1_704_067_200_000, 1_704_067_200_001, None]).cast(pl.Datetime("ms")),
    "date": pl.Series([19_723, 19_724, None], dtype=pl.Int32).cast(pl.Date),
    "time": pl.Series([0, 1, 1000, None]).cast(pl.Time),
    "float": pl.Series([0.0, -0.0, float("nan"), 1.5, None]),
    "bool": pl.Series([True, False, None]),
}


_DAYS = [date(2024, 1, 31), date(2024, 2, 1), date(2024, 2, 29), date(2024, 3, 1), None]


@st.composite
def partitioned_frames_and_predicates(draw, pools=PARTITION_KEY_POOLS):
    """A frame with a date column, a partition column ``x`` of values from one of ``pools`` and a row id, and three
    predicates on them."""
    kind = draw(st.sampled_from(sorted(pools)))
    pool = pools[kind]
    zoned = draw(st.booleans())
    n = draw(st.integers(1, 12))
    xs = draw(st.lists(st.integers(0, len(pool) - 1), min_size=n, max_size=n))
    ds = draw(st.lists(st.integers(0, len(_DAYS) - 1), min_size=n, max_size=n))
    day = pl.Series([_DAYS[i] for i in ds], dtype=pl.Date)
    if zoned:
        day = day.cast(pl.Datetime("us")).dt.offset_by("23h30m").dt.replace_time_zone(NEW_YORK)
    frame = pl.DataFrame({"date": day, "x": pool.gather(xs), "i": list(range(n))})

    def x_atom():
        literal = pl.lit(pool.slice(draw(st.integers(0, len(pool) - 2)), 1)).first()
        op = draw(st.sampled_from(["==", "!=", "in", "null", ">=", "ne_missing"]))
        if op == "==":
            return pl.col("x") == literal
        if op == "!=":
            return pl.col("x") != literal
        if op == "ne_missing":
            return pl.col("x").ne_missing(literal)
        if op == "null":
            return pl.col("x").is_null()
        if op == ">=" and kind not in ("string", "categorical", "bool"):
            return pl.col("x") >= literal
        members = pool.gather(draw(st.lists(st.integers(0, len(pool) - 2), min_size=1, max_size=3)))
        return pl.col("x").is_in(members.implode())

    def date_atom():
        d = _DAYS[draw(st.integers(0, len(_DAYS) - 2))]
        literal = pl.lit(d) if not zoned else pl.lit(datetime.combine(d, datetime.min.time())).dt.replace_time_zone(NEW_YORK).dt.offset_by("12h")
        op = draw(st.sampled_from([">=", "<", "==", "between", "null"]))
        col = pl.col("date") if op != "==" or not zoned else pl.col("date").dt.date()
        if op == "==":
            return col == pl.lit(d)
        if op == "null":
            return col.is_null()
        if op == "between":
            return col.is_between(literal, literal.dt.offset_by(f"{draw(st.integers(0, 40))}d"))
        return col >= literal if op == ">=" else col < literal

    def predicate(depth=0):
        choice = draw(st.integers(0, 4 if depth < 2 else 1))
        if choice == 0:
            return x_atom()
        if choice == 1:
            return date_atom()
        if choice == 2:
            return predicate(depth + 1) & predicate(depth + 1)
        if choice == 3:
            return predicate(depth + 1) | predicate(depth + 1)
        return ~predicate(depth + 1)

    return frame, [predicate() for _ in range(3)]
