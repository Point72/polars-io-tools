"""scan_clickhouse: HTTP decoding and the parameters each request carries.

Queries run on an in-memory DuckDB behind a fake ``get_batch_reader_http`` (see ``helpers.sql_backends``); the HTTP
test mocks ``requests.post`` and runs the real batch reader. The query tests scan_clickhouse shares with scan_db are
rows of ``test_scan_matches_native`` in test_lazy_sql_reader.
"""

import gzip
import io
import zlib
from datetime import date

import polars as pl
import pyarrow as pa
import pytest
import requests
from polars.testing import assert_frame_equal
from urllib3.response import HTTPResponse

from polars_io_tools._compat import POLARS_HAS_COLLECT_BATCHES
from polars_io_tools.io_sources import lazy_clickhouse_reader as ch_mod
from polars_io_tools.io_sources.lazy_clickhouse_reader import get_batch_reader_http
from polars_io_tools.tests.helpers.assertions import assert_text
from polars_io_tools.tests.helpers.sql_backends import ENTRY_POINTS, Backend, cc_bond_df, connect, install

scan_clickhouse = ENTRY_POINTS["scan_clickhouse"]


@pytest.fixture(scope="module")
def duckdb_connection():
    conn = connect(CC_Bond=cc_bond_df())
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def db(monkeypatch, duckdb_connection):
    backend = Backend(duckdb_connection)
    install(monkeypatch, backend)
    return backend


@pytest.mark.parametrize("content_encoding", [None, "gzip", "deflate"])
@pytest.mark.parametrize("empty", [False, True])
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

    monkeypatch.setattr(ch_mod, "get_batch_reader_http", get_batch_reader_http)
    monkeypatch.setattr(ch_mod.requests, "post", fake_post)
    try:
        assert_frame_equal(scan_clickhouse("SELECT id, label FROM events").collect(), expected)
        assert len(queries) == 2
        assert "LIMIT 0" in queries[0]
        assert all(q.endswith("FORMAT ArrowStream") for q in queries)
    finally:
        for response in responses:
            response.close()


_ONE_COL = "SELECT CenterID FROM CC_Bond"


@pytest.mark.parametrize(
    ("kw", "data_only", "field", "key", "want"),
    [
        pytest.param({"timeout": (2, 30)}, False, "kwargs", "timeout", (2, 30), id="scan_clickhouse_forwards_timeout_to_schema_and_data"),
        pytest.param({"fetch_size": 321}, True, "params", "max_block_size", 321, id="scan_clickhouse_uses_fetch_size_as_block_size"),
        pytest.param(
            {"fetch_size": 321, "params": {"max_block_size": 777}},
            False,
            "params",
            "max_block_size",
            777,
            id="scan_clickhouse_preserves_explicit_max_block_size",
        ),
    ],
)
def test_scan_clickhouse_request_carries(db, kw, data_only, field, key, want):
    """The schema probe and the data query (or only the latter) carry ``want`` under ``field[key]``."""
    given = {k: dict(v) if isinstance(v, dict) else v for k, v in kw.items()}
    scan_clickhouse(_ONE_COL, **given).collect()
    calls = [c for c in db.calls if not data_only or "LIMIT 0" not in c["query"]]
    assert len(calls) == (1 if data_only else 2)
    assert [c[field][key] for c in calls] == [want] * len(calls)
    assert given.get("params", {}) == kw.get("params", {}), "the caller's params were modified"


@pytest.mark.skipif(not POLARS_HAS_COLLECT_BATCHES, reason="collect_batches requires Polars >= 1.34.0")
def test_scan_clickhouse_runtime_batch_size_overrides_fetch_size(db, monkeypatch):
    original_register = ch_mod.register_io_source_with_is_pure
    runtime_batch_sizes = []

    def capturing_register(source, *args, **kwargs):
        def capturing_source(with_columns, predicate, n_rows, batch_size):
            runtime_batch_sizes.append(batch_size)
            yield from source(with_columns, predicate, n_rows, batch_size)

        return original_register(capturing_source, *args, **kwargs)

    monkeypatch.setattr(ch_mod, "register_io_source_with_is_pure", capturing_register)
    list(scan_clickhouse(_ONE_COL, fetch_size=200000).collect_batches(chunk_size=2))
    assert runtime_batch_sizes
    assert [c["params"]["max_block_size"] for c in db.calls if "LIMIT 0" not in c["query"]] == runtime_batch_sizes


def test_dt_date_filter_pushdown(db, monkeypatch):
    """``.dt.date()`` pushes down as a ``Date32`` cast rather than being dropped (issue #51).

    DuckDB cannot parse ``Date32``, so the data query returns an empty frame with the full schema.
    """

    sent = []

    def reader(query, url, params, **kwargs):
        sent.append(query)
        return db.http(query if "LIMIT 0" in query else "SELECT * FROM CC_Bond LIMIT 0", url, params, **kwargs)

    monkeypatch.setattr(ch_mod, "get_batch_reader_http", reader)
    scan_clickhouse("SELECT * FROM CC_Bond").filter(pl.col("EventDate").dt.date() == date(2016, 1, 1)).collect()
    (sql,) = [q for q in sent if "LIMIT 0" not in q]
    assert_text(sql, ["Date32", "EventDate", "2016-01-01"])
