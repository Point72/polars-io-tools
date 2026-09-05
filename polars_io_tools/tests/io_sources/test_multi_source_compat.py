"""Backwards-compat: ``multi_source`` must alias ``pushdown_combine``.

``multi_source`` was public in v0.1.0–v0.1.2 and renamed to ``pushdown_combine``
in v0.2.0. The ``multi_source`` name is kept as a silent alias so the historical
import paths keep resolving.
"""

import importlib

import polars_io_tools as piot


def test_multi_source_is_pushdown_combine():
    assert piot.multi_source is piot.pushdown_combine


def test_multi_source_import_paths_resolve():
    from polars_io_tools import FilterSpec, multi_source
    from polars_io_tools.io_sources import (
        FilterSpec as FilterSpec_ios,
        multi_source as multi_source_ios,
    )
    from polars_io_tools.io_sources.multi_source import (
        FilterSpec as FilterSpec_mod,
        multi_source as multi_source_mod,
    )

    assert multi_source is multi_source_ios is multi_source_mod is piot.pushdown_combine
    assert FilterSpec is FilterSpec_ios is FilterSpec_mod is piot.FilterSpec


def test_multi_source_module_importable():
    mod = importlib.import_module("polars_io_tools.io_sources.multi_source")
    assert mod.multi_source is piot.pushdown_combine
    assert mod.__all__ == ("FilterSpec", "multi_source")
