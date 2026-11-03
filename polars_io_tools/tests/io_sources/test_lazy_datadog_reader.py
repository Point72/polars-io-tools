"""scan_datadog against a mocked ``metric_query``.

One table: a mocked API response, scan arguments and Polars steps -> the result's height (or the error) and the number
of API calls. Pickle rows also check the unpickled frame gives the same result (it calls a pickled copy of the mock).
"""

import datetime
import time
from unittest.mock import patch

import cloudpickle
import polars as pl
import pytest
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal

import polars_io_tools as cpl
from polars_io_tools.tests.helpers.cases import Case, params

API_KEY = "dummy_api_key_for_testing"
APP_KEY = "dummy_app_key_for_testing"
UTC = datetime.UTC


@pytest.fixture
def mock_metric_query():
    with patch("polars_io_tools.io_sources.lazy_datadog_reader.metric_query") as mock:
        yield mock


# A response in the shape of one to "sum:piot_telemetry.test{type:call} by {user,module,function}.as_count()".
BASIC_RESPONSE = {
    "status": "ok",
    "res_type": "time_series",
    "resp_version": 1,
    "query": "sum:piot_telemetry.test{type:call} by {user,module,function}.as_count()",
    "from_date": 1748635900000,
    "to_date": 1748635960000,
    "series": [
        {
            "unit": None,
            "query_index": 0,
            "aggr": "sum",
            "metric": "piot_telemetry.test",
            "tag_set": ["function:feed._init", "module:example.adapters.feed", "user:service_user"],
            "expression": "sum:piot_telemetry.test{function:feed._init,module:example.adapters.feed,user:service_user,type:call}.as_count()",
            "scope": "function:feed._init,module:example.adapters.feed,type:call,user:service_user",
            "interval": 1,
            "length": 5,
            "start": 1748635902000,
            "end": 1748635940000,
            "pointlist": [
                [1748635902000.0, 1.0],
                [1748635914000.0, None],
                [1748635915000.0, None],
                [1748635937000.0, None],
                [1748635940000.0, None],
            ],
            "display_name": "piot_telemetry.test",
            "attributes": {},
        },
        {
            "unit": None,
            "query_index": 0,
            "aggr": "sum",
            "metric": "piot_telemetry.test",
            "tag_set": ["function:jobs", "module:example.cli.jobs", "user:user1"],
            "expression": "sum:piot_telemetry.test{function:jobs,module:example.cli.jobs,user:user1,type:call}.as_count()",
            "scope": "function:jobs,module:example.cli.jobs,type:call,user:user1",
            "interval": 1,
            "length": 5,
            "start": 1748635902000,
            "end": 1748635940000,
            "pointlist": [
                [1748635902000.0, None],
                [1748635914000.0, None],
                [1748635915000.0, None],
                [1748635937000.0, 1.0],
                [1748635940000.0, None],
            ],
            "display_name": "piot_telemetry.test",
            "attributes": {},
        },
    ],
    "values": [],
    "times": [],
    "message": "",
    "group_by": ["function", "module", "user"],
}


def _window(start, end):
    return lambda lf: lf.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") < end))


def _series(start: datetime.datetime, values: list[float], step_sec: float, tags=()) -> dict:
    return {
        "series": [
            {"metric": "m", "pointlist": [[(start.timestamp() + i * step_sec) * 1000, v] for i, v in enumerate(values)], "tag_set": list(tags)}
        ]
    }


_BASIC = _window(*(datetime.datetime.fromtimestamp(ms // 1000, tz=UTC).replace(tzinfo=None) for ms in (1748635900000, 1748635960000)))
_DAY = (datetime.datetime(2023, 1, 1, tzinfo=UTC), datetime.datetime(2023, 1, 2, tzinfo=UTC))
_HOUR = (datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 1, 1))
_HOUR_POINTS = _series(_HOUR[0].replace(tzinfo=UTC), [1.0, 5.0], 1800, ["tag:value"])


def D(id, op, expect, calls, response=None, pickle=False, **scan):
    """``expect`` is the result's height or the exception collecting raises."""
    return Case(id, op=op, expect=expect, response=response, calls=calls, pickle=pickle, scan=scan)


RUNS = [
    D("basic-all_columns", _BASIC, 2, 1, BASIC_RESPONSE),
    D("basic-tag_columns", lambda lf: _BASIC(lf).select(["user", "module", "function"]), 2, 1, BASIC_RESPONSE),
    # a lower time bound is required
    D("no_lower_bound-upper_only", lambda lf: lf.filter(pl.col("timestamp") < datetime.datetime(2025, 1, 1)), ComputeError, 0),
    D("no_lower_bound-unfiltered", lambda lf: lf, ComputeError, 0),
    D("start_after_end-all_columns", _window(_DAY[1], _DAY[0]), 0, 0),
    D("start_after_end-selected", lambda lf: _window(_DAY[1], _DAY[0])(lf).select(["timestamp", "value"]), 0, 0),
    D("empty_response", _window(*_DAY), 0, 1, {"series": []}),
    # two hours in one-hour chunks
    D(
        "chunking",
        _window(*(d.replace(tzinfo=None) for d in (_DAY[0], _DAY[0] + datetime.timedelta(hours=2)))),
        4,
        2,
        _series(_DAY[0], [0.0, 1.0], 1),
        max_chunk_duration_seconds=3600,
    ),
    D("scan_datadog_pickle_basic", _window(*_HOUR), 2, 1, _HOUR_POINTS, pickle=True),
    D("scan_datadog_pickle_with_filter", lambda lf: _window(*_HOUR)(lf).filter(pl.col("value") > 3.0), 1, 1, _HOUR_POINTS, pickle=True),
]


@pytest.mark.parametrize(("op", "expect", "response", "calls", "pickle", "scan"), params(RUNS, "op", "expect", "response", "calls", "pickle", "scan"))
def test_scan_datadog(mock_metric_query, op, expect, response, calls, pickle, scan):
    mock_metric_query.return_value = response
    lf = op(cpl.scan_datadog(query="avg:my.metric{tag:value} by {user}", api_key=API_KEY, app_key=APP_KEY, **scan))
    if isinstance(expect, type):
        with pytest.raises(expect):
            lf.collect()
    else:
        unpickled = cloudpickle.loads(cloudpickle.dumps(lf)) if pickle else None
        got = lf.sort("timestamp").collect() if "timestamp" in lf.collect_schema() else lf.collect()
        assert got.height == expect, got
        if pickle:
            assert unpickled.collect_schema() == lf.collect_schema()
            assert_frame_equal(unpickled.sort("timestamp").collect(), got)
    assert mock_metric_query.call_count == calls


def test_interval_parameter(mock_metric_query):
    """dd_interval reaches metric_query; naive bounds are UTC seconds."""
    mock_metric_query.return_value = {"series": []}
    start, end = datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 2)
    _window(start, end)(cpl.scan_datadog(query="some:query", api_key=API_KEY, app_key=APP_KEY, dd_interval=300)).collect()
    mock_metric_query.assert_called_once_with(
        query="some:query",
        start=int(start.replace(tzinfo=UTC).timestamp()),
        end=int(end.replace(tzinfo=UTC).timestamp()),
        api_key=API_KEY,
        app_key=APP_KEY,
        interval=300,
    )


_POINT = datetime.datetime(2024, 1, 1, 12)


@pytest.fixture
def new_york_tz(monkeypatch):
    """Run with a non-UTC host timezone; predicate bounds are naive UTC regardless."""
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is unavailable")
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.mark.parametrize(
    "pred",
    [
        pl.col("timestamp").is_between(_POINT - datetime.timedelta(minutes=1), _POINT + datetime.timedelta(minutes=1)),
        pl.col("timestamp") == _POINT,
        pl.col("timestamp").is_between(_POINT, _POINT + datetime.timedelta(milliseconds=500)),
        pl.col("timestamp") >= _POINT,
    ],
)
def test_points_matching_predicate_are_fetched(new_york_tz, pred):
    point_sec = int(_POINT.replace(tzinfo=UTC).timestamp())

    def api_stub(**kwargs):
        return {"series": [{"pointlist": [[point_sec * 1000, 7.0]] if kwargs["start"] <= point_sec <= kwargs["end"] else []}]}

    df = pl.DataFrame({"timestamp": [_POINT], "value": [7.0]})
    with patch("polars_io_tools.io_sources.lazy_datadog_reader.metric_query", side_effect=api_stub):
        lf = cpl.scan_datadog("q", API_KEY, APP_KEY, additional_schema=dict(df.schema), overwrite_schema=True)
        assert_frame_equal(lf.filter(pred).collect(), df.filter(pred))
