"""Pytest configuration for the pushdown characterization suite."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gap: pushdown that polars could logically perform but currently does not (characterization lock-in).",
    )
