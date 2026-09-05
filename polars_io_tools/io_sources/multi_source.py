"""Backwards-compatibility alias for the former ``multi_source`` entry point.

``multi_source`` was renamed to :func:`pushdown_combine` in v0.2.0. This module
re-exports the current implementation under the historical name so that imports
from the pre-0.2.0 surface keep working::

    from polars_io_tools import multi_source
    from polars_io_tools.io_sources import multi_source
    from polars_io_tools.io_sources.multi_source import FilterSpec, multi_source

``multi_source`` is the same object as :func:`pushdown_combine`.
"""

from .pushdown_combine import FilterSpec, pushdown_combine

multi_source = pushdown_combine

__all__ = ("FilterSpec", "multi_source")
