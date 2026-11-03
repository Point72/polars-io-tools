"""scan_datadog against a mocked ``metric_query``.

One table: a mocked API response, scan arguments and a query on the reader, with the number of rows the query returns
(or the error it raises) and the number of API calls. Rows with ``check_pickle`` also check that the frame after a
pickle round trip gives the same result (it calls a pickled copy of the mock).
"""

import datetime
import time
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field
from unittest.mock import patch

import cloudpickle
import polars as pl
import pytest
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal

import polars_io_tools as cpl
from polars_io_tools.tests.helpers.cases import as_params

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


def _timestamps_between(start, end):
    return lambda lf: lf.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") < end))


def _response_with_one_series(start: datetime.datetime, values: list[float], step_sec: float, tags=()) -> dict:
    return {
        "series": [
            {"metric": "m", "pointlist": [[(start.timestamp() + i * step_sec) * 1000, v] for i, v in enumerate(values)], "tag_set": list(tags)}
        ]
    }


_WITHIN_BASIC_RESPONSE = _timestamps_between(
    *(datetime.datetime.fromtimestamp(ms // 1000, tz=UTC).replace(tzinfo=None) for ms in (1748635900000, 1748635960000))
)
_JAN_1_2023_UTC = (datetime.datetime(2023, 1, 1, tzinfo=UTC), datetime.datetime(2023, 1, 2, tzinfo=UTC))
_FIRST_HOUR_OF_2023 = (datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 1, 1))
_TWO_POINTS_IN_FIRST_HOUR = _response_with_one_series(_FIRST_HOUR_OF_2023[0].replace(tzinfo=UTC), [1.0, 5.0], 1800, ["tag:value"])


@dataclass(frozen=True)
class DatadogScanCase:
    """A query on ``scan_datadog`` with ``metric_query`` mocked, how many rows come back and how many API calls are
    made."""

    id: str
    _: KW_ONLY
    # receives the scanned frame and returns the query to collect
    query: Callable[[pl.LazyFrame], pl.LazyFrame]
    # the number of rows the query returns, unless collecting it ``raises``
    expected_height: int | None = None
    raises: type[Exception] | None = None
    # the number of calls to ``metric_query``
    api_calls: int
    # what the mocked ``metric_query`` returns
    response: dict | None = None
    # also check that the frame survives pickling
    check_pickle: bool = False
    # more keyword arguments for ``scan_datadog``
    scan_arguments: dict = field(default_factory=dict)


RUNS = [
    DatadogScanCase("basic-all_columns", query=_WITHIN_BASIC_RESPONSE, expected_height=2, api_calls=1, response=BASIC_RESPONSE),
    DatadogScanCase(
        "basic-tag_columns",
        query=lambda lf: _WITHIN_BASIC_RESPONSE(lf).select(["user", "module", "function"]),
        expected_height=2,
        api_calls=1,
        response=BASIC_RESPONSE,
    ),
    # a lower time bound is required
    DatadogScanCase(
        "no_lower_bound-upper_only",
        query=lambda lf: lf.filter(pl.col("timestamp") < datetime.datetime(2025, 1, 1)),
        raises=ComputeError,
        api_calls=0,
    ),
    DatadogScanCase("no_lower_bound-unfiltered", query=lambda lf: lf, raises=ComputeError, api_calls=0),
    DatadogScanCase("start_after_end-all_columns", query=_timestamps_between(_JAN_1_2023_UTC[1], _JAN_1_2023_UTC[0]), expected_height=0, api_calls=0),
    DatadogScanCase(
        "start_after_end-selected",
        query=lambda lf: _timestamps_between(_JAN_1_2023_UTC[1], _JAN_1_2023_UTC[0])(lf).select(["timestamp", "value"]),
        expected_height=0,
        api_calls=0,
    ),
    DatadogScanCase("empty_response", query=_timestamps_between(*_JAN_1_2023_UTC), expected_height=0, api_calls=1, response={"series": []}),
    # two hours in one-hour chunks
    DatadogScanCase(
        "chunking",
        query=_timestamps_between(*(d.replace(tzinfo=None) for d in (_JAN_1_2023_UTC[0], _JAN_1_2023_UTC[0] + datetime.timedelta(hours=2)))),
        expected_height=4,
        api_calls=2,
        response=_response_with_one_series(_JAN_1_2023_UTC[0], [0.0, 1.0], 1),
        scan_arguments={"max_chunk_duration_seconds": 3600},
    ),
    DatadogScanCase(
        "scan_datadog_pickle_basic",
        query=_timestamps_between(*_FIRST_HOUR_OF_2023),
        expected_height=2,
        api_calls=1,
        response=_TWO_POINTS_IN_FIRST_HOUR,
        check_pickle=True,
    ),
    DatadogScanCase(
        "scan_datadog_pickle_with_filter",
        query=lambda lf: _timestamps_between(*_FIRST_HOUR_OF_2023)(lf).filter(pl.col("value") > 3.0),
        expected_height=1,
        api_calls=1,
        response=_TWO_POINTS_IN_FIRST_HOUR,
        check_pickle=True,
    ),
]


@pytest.mark.parametrize("case", as_params(RUNS))
def test_scan_datadog(mock_metric_query, case: DatadogScanCase):
    mock_metric_query.return_value = case.response
    lf = case.query(cpl.scan_datadog(query="avg:my.metric{tag:value} by {user}", api_key=API_KEY, app_key=APP_KEY, **case.scan_arguments))
    if case.raises is not None:
        with pytest.raises(case.raises):
            lf.collect()
    else:
        unpickled = cloudpickle.loads(cloudpickle.dumps(lf)) if case.check_pickle else None
        got = lf.sort("timestamp").collect() if "timestamp" in lf.collect_schema() else lf.collect()
        assert got.height == case.expected_height, got
        if case.check_pickle:
            assert unpickled.collect_schema() == lf.collect_schema()
            assert_frame_equal(unpickled.sort("timestamp").collect(), got)
    assert mock_metric_query.call_count == case.api_calls


def test_interval_parameter(mock_metric_query):
    """dd_interval reaches metric_query; naive bounds are UTC seconds."""
    mock_metric_query.return_value = {"series": []}
    start, end = datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 2)
    _timestamps_between(start, end)(cpl.scan_datadog(query="some:query", api_key=API_KEY, app_key=APP_KEY, dd_interval=300)).collect()
    mock_metric_query.assert_called_once_with(
        query="some:query",
        start=int(start.replace(tzinfo=UTC).timestamp()),
        end=int(end.replace(tzinfo=UTC).timestamp()),
        api_key=API_KEY,
        app_key=APP_KEY,
        interval=300,
    )


_NOON_JAN_1_2024 = datetime.datetime(2024, 1, 1, 12)


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
        pl.col("timestamp").is_between(_NOON_JAN_1_2024 - datetime.timedelta(minutes=1), _NOON_JAN_1_2024 + datetime.timedelta(minutes=1)),
        pl.col("timestamp") == _NOON_JAN_1_2024,
        pl.col("timestamp").is_between(_NOON_JAN_1_2024, _NOON_JAN_1_2024 + datetime.timedelta(milliseconds=500)),
        pl.col("timestamp") >= _NOON_JAN_1_2024,
    ],
    ids=["within_a_minute_of_noon", "at_noon", "within_half_a_second_after_noon", "from_noon"],
)
def test_points_matching_predicate_are_fetched(new_york_tz, pred):
    point_sec = int(_NOON_JAN_1_2024.replace(tzinfo=UTC).timestamp())

    def api_stub(**kwargs):
        return {"series": [{"pointlist": [[point_sec * 1000, 7.0]] if kwargs["start"] <= point_sec <= kwargs["end"] else []}]}

    df = pl.DataFrame({"timestamp": [_NOON_JAN_1_2024], "value": [7.0]})
    with patch("polars_io_tools.io_sources.lazy_datadog_reader.metric_query", side_effect=api_stub):
        lf = cpl.scan_datadog("q", API_KEY, APP_KEY, additional_schema=dict(df.schema), overwrite_schema=True)
        assert_frame_equal(lf.filter(pred).collect(), df.filter(pred))
