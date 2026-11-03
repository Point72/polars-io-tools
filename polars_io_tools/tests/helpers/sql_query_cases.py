"""Reader queries that ``scan_db`` and ``scan_clickhouse`` must both answer as the database does, and the check.

A ``QueryCase`` is a query, the Polars steps run on the reader's frame, and what to check. ``SHARED_QUERIES`` holds the
queries both readers run: ``test_lazy_sql_reader.py`` runs them with ``scan_db`` and ``test_lazy_clickhouse_reader.py``
with ``scan_clickhouse``, each against its own ``FakeDatabase`` (see ``sql_backends``).
"""

import logging
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_io_tools.tests.helpers.assertions import assert_matches_plain_polars, assert_translated_text
from polars_io_tools.tests.helpers.sql_backends import SQL_READERS


@dataclass(frozen=True)
class SentSql:
    """What the one data query the reader sends must contain, omit or start with, checked case-insensitively.

    To add filters, column selections or a row limit to the user's query, the SQL reader can wrap it in an outer query:
    ``SELECT <columns> FROM (<user's query>) AS __cpl_subq WHERE ...``. ``__cpl_subq`` is the alias the reader gives
    the user's query when it wraps it in an outer query. The inner part is the SQL text before ``__cpl_subq``, which
    holds the user's query (after the outer column list), and the outer part is the text after it, which holds the
    outer query's ``WHERE`` and row limit.

    ``contains``, ``omits`` and ``startswith`` check the whole query, the ``inner_*`` and ``outer_*`` fields its inner
    and outer parts. A tuple item of a ``contains`` field is satisfied by any of its alternatives.
    """

    contains: tuple = ()
    omits: tuple = ()
    startswith: str | None = None
    inner_contains: tuple = ()
    outer_contains: tuple = ()
    outer_omits: tuple = ()


@dataclass(frozen=True)
class QueryCase:
    """A reader ``query`` and the Polars ``steps`` run on it; the result must equal the database's answer to ``query``
    put through the same ``steps``.

    ``native_transform`` turns the database's own answer to ``query`` into what the reader should return before
    ``steps`` run, for a reader option that changes the data, such as ``cast_map`` changing a column's dtype or a
    partition that filters rows on the client.
    """

    id: str
    _: KW_ONLY
    query: str
    steps: Callable[[pl.LazyFrame], pl.LazyFrame] = lambda lf: lf
    # names of tables the test file creates first
    tables: tuple[str, ...] = ()
    # reader keyword arguments; a callable value is called for each test
    scan_kwargs: dict = field(default_factory=dict)
    # applied to the database's answer before ``steps``; see the class docstring
    native_transform: Callable[[pl.DataFrame], pl.DataFrame] | None = None
    check_row_order: bool = False
    expected_height: int | None = None
    expected_query_count: int | None = None
    expected_log: str | None = None
    log_level: int = logging.WARNING
    # the query with this WHERE clause, then selecting ``expected_where_columns``, gives the same frame
    expected_where: str | None = None
    expected_where_columns: list[str] | None = None
    sent_sql: SentSql | None = None


def filtered(*preds, select=None, after=None):
    """Steps that select ``select`` (if given), filter by ``preds``, then select ``after`` (if given)."""

    def op(lf):
        lf = lf.select(select) if select else lf
        lf = lf.filter(*preds)
        return lf.select(after) if after else lf

    return op


BOND_HOLIDAY_COLUMNS = ["item_id", "item_code", "country_code", "item_name", "year", "event_date", "day_of_week", "event_name", "file_type"]
EVENT_NAME_HAS_DAY = pl.col("event_name").str.contains("Day")
YEAR_WEEKDAY_AND_DAY_FILTER = (
    pl.col("year").is_between(2015, 2050) & ((pl.col("day_of_week") == "Fri") | (pl.col("day_of_week") == "Sat")) & EVENT_NAME_HAS_DAY
)

# Queries on the bond holiday table (SampleA) and the employee test score table (EmployeeTbl), with filters that have
# an equivalent SQL WHERE clause, and heads.
SHARED_QUERIES = [
    QueryCase(
        "basic_query",
        query="SELECT item_id, item_name, year FROM SampleA",
        steps=filtered(pl.col("year") > 2015),
        expected_where="year > 2015",
        check_row_order=True,
    ),
    QueryCase(
        "column_selection",
        query="SELECT * FROM SampleA",
        steps=filtered(pl.col("item_name").str.contains("Da"), select=["item_id", "item_name", "event_date"]),
        expected_where="item_name LIKE '%Da%'",
        expected_where_columns=["item_id", "item_name", "event_date"],
    ),
    QueryCase(
        "complex_query",
        query=f"SELECT {', '.join(BOND_HOLIDAY_COLUMNS)} FROM SampleA",
        steps=filtered(YEAR_WEEKDAY_AND_DAY_FILTER, select=BOND_HOLIDAY_COLUMNS),
        expected_where="year BETWEEN 2015 AND 2050 AND (day_of_week = 'Fri' OR day_of_week = 'Sat') AND event_name LIKE '%Day%'",
        expected_where_columns=BOND_HOLIDAY_COLUMNS,
    ),
    QueryCase(
        "not_null",
        query="SELECT item_id, item_name, event_date FROM SampleA",
        steps=filtered(pl.col("event_date").is_not_null()),
        expected_where="event_date IS NOT NULL",
        check_row_order=True,
    ),
    # filter on a column the selection drops
    QueryCase(
        "empty_df",
        query="SELECT item_id, item_name, event_date, year FROM SampleA",
        steps=filtered(pl.col("year") < 1900, after=["item_id", "item_name", "event_date"]),
        expected_where="year < 1900",
        expected_where_columns=["item_id", "item_name", "event_date"],
        check_row_order=True,
        expected_height=0,
    ),
    QueryCase(
        "in_operator",
        query="SELECT item_id, item_name, year, event_date, day_of_week FROM SampleA",
        steps=filtered(pl.col("day_of_week").is_in(["Fri", "Sat", "Sun"])),
        expected_where="day_of_week IN ('Fri', 'Sat', 'Sun')",
    ),
    QueryCase(
        "not_like",
        query="SELECT item_id, item_name, year, event_name FROM SampleA",
        steps=filtered(~EVENT_NAME_HAS_DAY),
        expected_where="event_name NOT LIKE '%Day%'",
    ),
    QueryCase(
        "polars_pushes_n_rows_without_predicate",
        query="SELECT * FROM EmployeeTbl",
        steps=lambda lf: lf.head(5),
        sent_sql=SentSql(contains=("LIMIT",)),
        check_row_order=True,
    ),
    QueryCase(
        "head_zero_skips_query",
        query="SELECT item_id, item_name, year FROM SampleA",
        steps=lambda lf: lf.head(0),
        expected_query_count=0,
    ),
    QueryCase(
        "head_zero_with_column_selection",
        query="SELECT item_id, item_name, year FROM SampleA",
        steps=lambda lf: lf.select(["item_id", "item_name"]).head(0),
        expected_query_count=0,
    ),
]


def shared_query_params(reader: str) -> list:
    """One ``(reader, case)`` param per row of ``SHARED_QUERIES``; the test id ends with the reader's name."""
    return [pytest.param(reader, case, id=f"{case.id}-{reader}") for case in SHARED_QUERIES]


def check_scan_matches_database(fake_database, caplog, reader: str, case: QueryCase) -> None:
    """Run ``case`` with ``reader`` (a key of ``SQL_READERS``) on ``fake_database`` and assert everything it states."""
    lf = SQL_READERS[reader](case.query, **{k: v() if callable(v) else v for k, v in case.scan_kwargs.items()})
    start = len(fake_database.calls)
    answer = fake_database.run_directly(case.query)
    want = case.steps((case.native_transform(answer) if case.native_transform else answer).lazy())
    with caplog.at_level(case.log_level):
        got = assert_matches_plain_polars(case.steps(lf), want, None, check_row_order=case.check_row_order)
    sent = fake_database.queries(start)
    calls = len(fake_database.calls) - start

    assert got.columns == want.collect_schema().names(), f"columns {got.columns}"
    assert case.steps(lf).collect_schema() == got.schema, "the lazy schema differs from the collected schema"
    if case.expected_height is not None:
        assert got.height == case.expected_height, f"{got.height} rows"
    if case.expected_where is not None:
        filtered_read = SQL_READERS[reader](f"{case.query} WHERE {case.expected_where}")
        if case.expected_where_columns is not None:
            filtered_read = filtered_read.select(case.expected_where_columns)
        assert_frame_equal(filtered_read.collect(), got, check_row_order=case.check_row_order)

    if case.expected_log is not None:
        assert any(case.expected_log in r.message for r in caplog.records), [r.message for r in caplog.records]
    if case.expected_query_count is not None:
        assert calls == case.expected_query_count, fake_database.calls[start:]
    if case.sent_sql is not None:
        (sql,) = sent
        inner, outer = sql.split("__cpl_subq")[0], sql.split("__cpl_subq")[-1]
        want_sql = case.sent_sql
        assert_translated_text(sql, want_sql.contains, want_sql.omits, startswith=want_sql.startswith, ignore_case=True)
        assert_translated_text(inner, want_sql.inner_contains, ignore_case=True)
        assert_translated_text(outer, want_sql.outer_contains, want_sql.outer_omits, ignore_case=True)
