"""scan_clickhouse: HTTP decoding and the parameters each request carries.

Queries run on an in-memory DuckDB behind a fake ``get_batch_reader_http`` (see ``helpers.sql_backends``); the HTTP
test mocks ``requests.post`` and runs the real batch reader. ``test_scan_matches_native`` runs the queries
scan_clickhouse shares with scan_db (``helpers.sql_query_cases``).
"""

import gzip
import io
import zlib
from dataclasses import KW_ONLY, dataclass
from datetime import date

import polars as pl
import pyarrow as pa
import pytest
import requests
from polars.testing import assert_frame_equal
from urllib3.response import HTTPResponse

from polars_io_tools._compat import POLARS_HAS_COLLECT_BATCHES
from polars_io_tools.io_sources import lazy_clickhouse_reader
from polars_io_tools.io_sources.lazy_clickhouse_reader import get_batch_reader_http
from polars_io_tools.tests.helpers.assertions import assert_translated_text
from polars_io_tools.tests.helpers.cases import as_params
from polars_io_tools.tests.helpers.sql_backends import (
    SQL_READERS,
    FakeDatabase,
    bond_holidays_df,
    duckdb_with_tables,
    employee_test_scores_df,
    route_readers_to,
)
from polars_io_tools.tests.helpers.sql_query_cases import check_scan_matches_database, shared_query_params

scan_clickhouse = SQL_READERS["scan_clickhouse"]


@pytest.fixture(scope="module")
def duckdb_connection():
    conn = duckdb_with_tables(CC_Bond=bond_holidays_df(), EmployeeTbl=employee_test_scores_df())
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def fake_database(monkeypatch, duckdb_connection):
    """The fake database scan_clickhouse queries."""
    backend = FakeDatabase(duckdb_connection)
    route_readers_to(monkeypatch, backend)
    return backend


@pytest.mark.parametrize(("entry", "case"), shared_query_params("scan_clickhouse"))
def test_scan_matches_native(fake_database, caplog, entry, case):
    check_scan_matches_database(fake_database, caplog, entry, case)


@pytest.mark.parametrize("content_encoding", [None, "gzip", "deflate"], ids=["no_encoding", "gzip", "deflate"])
@pytest.mark.parametrize("empty", [False, True], ids=["with_rows", "empty_response"])
def test_scan_clickhouse_decodes_http_response(monkeypatch, content_encoding, empty):
    """Schema and data reads decode plain, gzip and deflate bodies, including empty results."""
    expected = pl.DataFrame({"id": [1, 2, 3, 4], "label": ["alpha", "β", None, ""]})
    if empty:
        expected = expected.head(0)
    queries, responses = [], []

    def fake_post(url, *, params, stream, timeout=None):
        assert stream is True
        queries.append(params["query"])
        table = expected.head(0).to_arrow() if "LIMIT 0" in params["query"] else expected.to_arrow()
        sink = io.BytesIO()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table, max_chunksize=2)
        payload = {None: bytes, "gzip": gzip.compress, "deflate": zlib.compress}[content_encoding](sink.getvalue())
        headers = {} if content_encoding is None else {"Content-Encoding": content_encoding}
        response = requests.Response()
        response.status_code = 200
        response.headers.update(headers)
        # Requests' streaming transport does not decode raw reads.
        response.raw = HTTPResponse(body=io.BytesIO(payload), headers=headers, preload_content=False, decode_content=False)
        responses.append(response)
        return response

    monkeypatch.setattr(lazy_clickhouse_reader, "get_batch_reader_http", get_batch_reader_http)
    monkeypatch.setattr(lazy_clickhouse_reader.requests, "post", fake_post)
    try:
        assert_frame_equal(scan_clickhouse("SELECT id, label FROM events").collect(), expected)
        assert len(queries) == 2
        assert "LIMIT 0" in queries[0]
        assert all(q.endswith("FORMAT ArrowStream") for q in queries)
    finally:
        for response in responses:
            response.close()


_CENTER_IDS = "SELECT CenterID FROM CC_Bond"


@dataclass(frozen=True)
class RequestOptionCase:
    """A ``scan_clickhouse`` option, and the value the HTTP requests carry for it."""

    id: str
    _: KW_ONLY
    scan_options: dict
    # only the data query carries it, not the schema query
    data_only: bool
    field: str
    # the requests carry ``want`` under ``field[key]``
    key: str
    want: object


@pytest.mark.parametrize(
    "case",
    as_params(
        [
            RequestOptionCase(
                "scan_clickhouse_forwards_timeout_to_schema_and_data",
                scan_options={"timeout": (2, 30)},
                data_only=False,
                field="kwargs",
                key="timeout",
                want=(2, 30),
            ),
            RequestOptionCase(
                "scan_clickhouse_uses_fetch_size_as_block_size",
                scan_options={"fetch_size": 321},
                data_only=True,
                field="params",
                key="max_block_size",
                want=321,
            ),
            RequestOptionCase(
                "scan_clickhouse_preserves_explicit_max_block_size",
                scan_options={"fetch_size": 321, "params": {"max_block_size": 777}},
                data_only=False,
                field="params",
                key="max_block_size",
                want=777,
            ),
        ]
    ),
)
def test_scan_clickhouse_request_carries(fake_database, case: RequestOptionCase):
    """The schema query and the data query (or only the latter) carry ``want`` under ``field[key]``."""
    given = {k: dict(v) if isinstance(v, dict) else v for k, v in case.scan_options.items()}
    scan_clickhouse(_CENTER_IDS, **given).collect()
    calls = [c for c in fake_database.calls if not case.data_only or "LIMIT 0" not in c["query"]]
    assert len(calls) == (1 if case.data_only else 2)
    assert [c[case.field][case.key] for c in calls] == [case.want] * len(calls)
    assert given.get("params", {}) == case.scan_options.get("params", {}), "the caller's params were modified"


@pytest.mark.skipif(not POLARS_HAS_COLLECT_BATCHES, reason="collect_batches requires Polars >= 1.34.0")
def test_scan_clickhouse_runtime_batch_size_overrides_fetch_size(fake_database, monkeypatch):
    original_register = lazy_clickhouse_reader.register_io_source_with_is_pure
    runtime_batch_sizes = []

    def capturing_register(source, *args, **kwargs):
        def capturing_source(with_columns, predicate, n_rows, batch_size):
            runtime_batch_sizes.append(batch_size)
            yield from source(with_columns, predicate, n_rows, batch_size)

        return original_register(capturing_source, *args, **kwargs)

    monkeypatch.setattr(lazy_clickhouse_reader, "register_io_source_with_is_pure", capturing_register)
    list(scan_clickhouse(_CENTER_IDS, fetch_size=200000).collect_batches(chunk_size=2))
    assert runtime_batch_sizes
    assert [c["params"]["max_block_size"] for c in fake_database.calls if "LIMIT 0" not in c["query"]] == runtime_batch_sizes


def test_dt_date_filter_pushdown(fake_database, monkeypatch):
    """``.dt.date()`` pushes down as a ``Date32`` cast rather than being dropped (issue #51).

    DuckDB cannot parse ``Date32``, so the data query returns an empty frame with the full schema.
    """

    sent = []

    def reader(query, url, params, **kwargs):
        sent.append(query)
        return fake_database.read_through_http(query if "LIMIT 0" in query else "SELECT * FROM CC_Bond LIMIT 0", url, params, **kwargs)

    monkeypatch.setattr(lazy_clickhouse_reader, "get_batch_reader_http", reader)
    scan_clickhouse("SELECT * FROM CC_Bond").filter(pl.col("EventDate").dt.date() == date(2016, 1, 1)).collect()
    (sql,) = [q for q in sent if "LIMIT 0" not in q]
    assert_translated_text(sql, ["Date32", "EventDate", "2016-01-01"])
