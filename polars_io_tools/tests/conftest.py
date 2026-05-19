import pytest


def pytest_addoption(parser):
    parser.addoption("--run-benchmarks", action="store_true", default=False, help="run benchmarks")


def pytest_configure(config):
    config.addinivalue_line("markers", "benchmark: mark test as benchmark")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-benchmarks"):
        return
    skip_benchmark = pytest.mark.skip(reason="need --run-benchmarks option to run")
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip_benchmark)
