"""The table row and the check shared by the pushdown test files."""

from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY, dataclass, field

import polars as pl

from polars_io_tools.testing import PredicateTracker
from polars_io_tools.tests.helpers.assertions import ExpectedPushdown, assert_pushdown, failure_note


@dataclass(frozen=True)
class PushdownCase:
    """A query over sources that record what they receive, and what each source should receive when it is collected."""

    id: str
    _: KW_ONLY
    # the data each source serves, by source name
    sources: dict[str, pl.DataFrame]
    # builds the query from the sources' lazy frames, looked up by source name
    pipeline: Callable[[Mapping[str, pl.LazyFrame]], pl.LazyFrame]
    # what each source should receive, by source name; a source not listed is not checked
    expected_pushdown: dict[str, ExpectedPushdown] = field(default_factory=dict)
    # receives the query result and the sources' trackers by name, and returns True if they are right
    result_check: Callable[[pl.DataFrame, dict[str, PredicateTracker]], bool] | None = None
    marks: tuple = ()


class _LazyFramesByName(dict):
    """Trackers by source name; looking a name up gives the tracker's lazy frame, so a query may read a source twice."""

    def __getitem__(self, name):
        return super().__getitem__(name).lazy_frame


def check_pushdown(case: PushdownCase) -> None:
    """Collect the row's query and assert what each source received, and ``result_check`` if given."""
    trackers = {name: PredicateTracker(df) for name, df in case.sources.items()}
    result = case.pipeline(_LazyFramesByName(trackers)).collect()
    for name, expected in case.expected_pushdown.items():
        with failure_note(f"source {name!r}"):
            assert_pushdown(trackers[name], expected)
    if case.result_check:
        assert case.result_check(result, trackers)
