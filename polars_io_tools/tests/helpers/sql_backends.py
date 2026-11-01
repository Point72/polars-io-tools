"""An in-memory DuckDB behind the SQL readers, so their tests run without a database server.

``Backend`` answers the queries ``scan_db`` sends through ``arrow_odbc`` and the queries ``scan_clickhouse`` sends over
HTTP, and logs each one. ``ENTRY_POINTS`` builds a reader over a query for either entry point.
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

CC_BOND_CSV = (
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


def cc_bond_df() -> pl.DataFrame:
    return pl.read_csv(io.StringIO(CC_BOND_CSV), try_parse_dates=True)


def employee_df() -> pl.DataFrame:
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


def connect(**tables: pl.DataFrame) -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB holding each frame as a table of that name."""
    conn = duckdb.connect(database=":memory:")
    for name, df in tables.items():
        conn.register("__frame", df.to_arrow())
        conn.execute(f"CREATE TABLE {name} AS SELECT * FROM __frame")
        conn.unregister("__frame")
    return conn


class FakeBatchReader:
    def __init__(self, table: pa.Table, batch_size: int):
        self.table = table
        self.batch_size = batch_size
        self.schema = table.schema

    def __iter__(self):
        return iter(self.table.to_batches())


class Backend:
    """Runs reader queries on ``conn`` and logs each as ``{"query", "kwargs", "params"}`` in ``calls``."""

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn
        self.calls: list[dict] = []
        self.lock = threading.Lock()

    def table(self, query: str) -> pa.Table:
        with self.lock:
            return self.conn.sql(query).to_arrow_table()

    def native(self, query: str) -> pl.DataFrame:
        """What the database returns for ``query``."""
        return pl.from_arrow(self.table(query))

    def odbc(self, query: str, batch_size: int, connection_string: str, **kwargs):
        self.calls.append({"query": query, "kwargs": dict(kwargs), "params": {}})
        return FakeBatchReader(self.table(query), batch_size)

    def http(self, query: str, url: str, params: dict, **kwargs):
        self.calls.append({"query": query, "kwargs": dict(kwargs), "params": dict(params)})
        table = self.table(query)
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            for batch in table.to_batches():
                writer.write_batch(batch)
        return pa.ipc.open_stream(sink.getvalue())

    def queries(self, since: int = 0, data: bool = True) -> list[str]:
        """Logged data queries (or schema probes, with ``data=False``) from call number ``since`` on."""
        return [c["query"] for c in self.calls[since:] if ("LIMIT 0" not in c["query"]) == data]


def install(monkeypatch, backend: Backend) -> None:
    """Route ``scan_db`` and ``scan_clickhouse`` to ``backend``."""
    from polars_io_tools.io_sources import lazy_clickhouse_reader, lazy_sql_reader

    monkeypatch.setitem(sys.modules, "arrow_odbc", SimpleNamespace(read_arrow_batches_from_odbc=backend.odbc))
    monkeypatch.setattr(lazy_sql_reader, "get_sqlglot_dialect_odbc", lambda conn_string: "duckdb")
    monkeypatch.setattr(lazy_clickhouse_reader, "get_batch_reader_http", backend.http)


CLICKHOUSE_URL = "http://localhost:8123"

ENTRY_POINTS = {
    "scan_db": lambda query, **kw: cpl.scan_db(query, "fake_connection_string", **kw),
    "scan_clickhouse": lambda query, **kw: cpl.scan_clickhouse(query, CLICKHOUSE_URL, kw.pop("params", {}), **kw),
}
