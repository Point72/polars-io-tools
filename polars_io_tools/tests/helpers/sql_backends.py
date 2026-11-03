"""A fake database for the ``scan_db`` and ``scan_clickhouse`` tests, so they run without a database server.

``FakeDatabase`` runs each query the readers send on an in-memory DuckDB and records it, so a test can check both the
rows a reader returns and the SQL it sent. ``route_readers_to`` points both readers at a ``FakeDatabase``, and
``SQL_READERS`` opens a query with either reader.
"""

import io
import sys
import threading
from datetime import date
from types import SimpleNamespace

import duckdb
import polars as pl
import pyarrow as pa

import polars_io_tools as cpl

# A few rows of a bond market holiday calendar: one holiday per row, for a financial centre.
BOND_HOLIDAYS_CSV = (
    "CenterID,CentreCode,ISOCountryCode,Centre,EventYear,EventDate,EventDayOfWeek,EventName,FileType\n"
    "103,GTQ,GT,Guatemala Quetzal,2021,2021-01-01T00:00:00.000,Fri,New Year's Day,G\n"
    "146,PHP,PH,Philippine Peso,2045,2045-04-06T00:00:00.000,Thu,Holy Thursday,G\n"
    "158,KPW,KP,Pyongyang,2053,2053-02-19T00:00:00.000,Wed,Lunar New Year 1,G\n"
    "67,BND,BN,Bandar Seri Begawan,2046,2046-06-30T00:00:00.000,Sat,Mid-year Bank Holiday,G\n"
    "148,BDT,BD,Dhaka,2015,2015-03-17T00:00:00.000,Tue,Birthday of Father of the Nation,G\n"
    "56,ZMW,ZM,Lusaka,2013,2013-10-24T00:00:00.000,Thu,Independence Day,G\n"
    "77,ERN,ER,Asmara,2016,2016-01-07T00:00:00.000,Thu,Christmas (Ge'ez),G\n"
    "47,RON,RO,Bucharest,2021,2021-12-25T00:00:00.000,Sat,Christmas Day,G\n"
    "126,ETB,ET,Addis Ababa,2054,2054-10-14T00:00:00.000,Wed,Birth of Prophet Mohammed*,G\n"
    "148,BDT,BD,Dhaka,2053,2053-05-20T00:00:00.000,Tue,Eid-ul Fitr 2*,G\n"
)


def bond_holidays_df() -> pl.DataFrame:
    return pl.read_csv(io.StringIO(BOND_HOLIDAYS_CSV), try_parse_dates=True)


def employee_test_scores_df() -> pl.DataFrame:
    """Test scores of a few employees, with repeated names and dates so that filters and group-bys have work to do."""
    names = [
        "Alice",
        "Bob",
        "Alice",
        "Bob",
        "Charlie",
        "Alice",
        "David",
        "Eve",
        "Bob",
        "Charlie",
        "Alice",
        "Bartholomew",
        "David",
        "Eve",
        "Bob",
        "Charlie",
        "Frank",
    ]
    days = [1, 1, 2, 2, 2, 3, 2, 4, 5, 5, 6, 7, 6, 8, 9, 9, 10]
    return pl.DataFrame(
        {
            "id": list(range(1, 18)),
            "name": names,
            "score": [85.4, 92.1, 34.1, 51.6, 78.3, 88.2, 90.5, 87.9, 89.7, 82.1, 91.3, 67.1, 85.8, 90.2, 94.5, 75.6, 88.9],
            "passed": [True, True, False, False, False, True, True, True, True, True, True, False, True, True, True, False, True],
            "test_date": [date(2022, 1, d) for d in days],
        }
    )


def duckdb_with_tables(**tables: pl.DataFrame) -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB holding each keyword's frame as a table named by the keyword."""
    conn = duckdb.connect(database=":memory:")
    for name, df in tables.items():
        conn.register("__frame", df.to_arrow())
        conn.execute(f"CREATE TABLE {name} AS SELECT * FROM __frame")
        conn.unregister("__frame")
    return conn


class FakeBatchReader:
    """Stands in for the batch reader ``arrow_odbc`` returns: yields the record batches of ``table``."""

    def __init__(self, table: pa.Table, batch_size: int):
        self.table = table
        self.batch_size = batch_size
        self.schema = table.schema

    def __iter__(self):
        return iter(self.table.to_batches())


class FakeDatabase:
    """Runs the queries the SQL readers send on the DuckDB ``conn``.

    Each query is recorded in ``calls`` as a dict with the SQL text (``query``), the reader's other keyword arguments
    (``kwargs``) and, for ClickHouse, the HTTP parameters (``params``).
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn
        self.calls: list[dict] = []
        self.lock = threading.Lock()

    def table(self, query: str) -> pa.Table:
        with self.lock:
            return self.conn.sql(query).to_arrow_table()

    def run_directly(self, query: str) -> pl.DataFrame:
        """What the database returns for ``query``, without going through a reader and without recording it."""
        return pl.from_arrow(self.table(query))

    def read_through_odbc(self, query: str, batch_size: int, connection_string: str, **kwargs):
        """Stands in for ``arrow_odbc.read_arrow_batches_from_odbc``."""
        self.calls.append({"query": query, "kwargs": dict(kwargs), "params": {}})
        return FakeBatchReader(self.table(query), batch_size)

    def read_through_http(self, query: str, url: str, params: dict, **kwargs):
        """Stands in for the HTTP request ``scan_clickhouse`` sends; answers in Arrow's streaming format."""
        self.calls.append({"query": query, "kwargs": dict(kwargs), "params": dict(params)})
        table = self.table(query)
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            for batch in table.to_batches():
                writer.write_batch(batch)
        return pa.ipc.open_stream(sink.getvalue())

    def queries(self, since: int = 0, data: bool = True) -> list[str]:
        """The recorded queries from call number ``since`` on: those that read rows, or with ``data=False`` those that
        only ask for the column names and types (``LIMIT 0``)."""
        return [c["query"] for c in self.calls[since:] if ("LIMIT 0" not in c["query"]) == data]


def route_readers_to(monkeypatch, database: FakeDatabase) -> None:
    """Make ``scan_db`` and ``scan_clickhouse`` send their queries to ``database`` for the rest of the test."""
    from polars_io_tools.io_sources import lazy_clickhouse_reader, lazy_sql_reader

    monkeypatch.setitem(sys.modules, "arrow_odbc", SimpleNamespace(read_arrow_batches_from_odbc=database.read_through_odbc))
    monkeypatch.setattr(lazy_sql_reader, "get_sqlglot_dialect_odbc", lambda conn_string: "duckdb")
    monkeypatch.setattr(lazy_clickhouse_reader, "get_batch_reader_http", database.read_through_http)


CLICKHOUSE_URL = "http://localhost:8123"

# Open a query with each reader: reader name -> function(query, **reader keyword arguments) -> LazyFrame.
SQL_READERS = {
    "scan_db": lambda query, **kw: cpl.scan_db(query, "fake_connection_string", **kw),
    "scan_clickhouse": lambda query, **kw: cpl.scan_clickhouse(query, CLICKHOUSE_URL, kw.pop("params", {}), **kw),
}
