import os

import pytest

from polars_io_tools.testing import PredicateTracker
from polars_io_tools.tests.helpers.cases import create_sample_data

try:
    from hypothesis import settings
except ImportError:
    pass
else:
    # A quick default; select a longer run with ``HYPOTHESIS_PROFILE=hunt`` or ``--hypothesis-profile=hunt``. CI runs the
    # quick examples in a fixed order and prints how to reproduce a failure.
    settings.register_profile("quick", max_examples=50)
    settings.register_profile("ci", settings.get_profile("quick"), derandomize=True, print_blob=True)
    settings.register_profile("hunt", max_examples=2000)
    settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quick"))


def pytest_addoption(parser):
    parser.addoption("--run-benchmarks", action="store_true", default=False, help="run benchmarks")


def pytest_configure(config):
    config.addinivalue_line("markers", "benchmark: mark test as benchmark")
    config.addinivalue_line("markers", "gap: a pushdown Polars could perform but does not (locked in so a change is noticed)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-benchmarks"):
        return
    skip_benchmark = pytest.mark.skip(reason="need --run-benchmarks option to run")
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip_benchmark)


@pytest.fixture
def tracked_sample_data():
    """Create a fresh PredicateTracker for each test with sample data."""
    return PredicateTracker(create_sample_data())
