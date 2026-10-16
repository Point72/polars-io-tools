"""Internal Polars feature flags.

This module centralizes version-based feature toggles for Polars.
It is intentionally internal (underscore-prefixed) and not re-exported.
"""

import polars as pl
from packaging import version

# Polars 1.34.0 introduced LazyFrame.collect_batches with maintain_order
# Use this flag to branch streaming collection behavior.
POLARS_HAS_COLLECT_BATCHES = version.parse(pl.__version__) >= version.parse("1.34.0")

# Polars 1.38.0 unified the partition interface
POLARS_HAS_PARTITION_BY = version.parse(pl.__version__) >= version.parse("1.38.0")

# Nested streaming IO-source collection is enabled only on verified runtimes.
# Earlier streaming engines can deadlock when nested caches share a small worker pool.
POLARS_STREAMING_SUPPORTS_NESTED_IO_PLUGIN = version.parse(pl.__version__) >= version.parse("1.43.2")

__all__ = ()
