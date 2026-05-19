"""Run docstring examples in polars_io_tools.io_sources as pytest tests.

Automatically discovers all importable modules that contain ``>>>`` examples
and parametrises one test-case per module.  Modules that fail to import
(e.g. due to missing optional dependencies) are silently skipped.
"""

import doctest
import importlib
import pkgutil

import pytest

import polars_io_tools.io_sources as _pkg


def _discover_doctest_modules() -> list[str]:
    """Return fully-qualified names of modules that contain at least one doctest."""
    finder = doctest.DocTestFinder()
    modules: list[str] = []
    for info in pkgutil.walk_packages(_pkg.__path__, prefix=_pkg.__name__ + "."):
        try:
            mod = importlib.import_module(info.name)
        except Exception:
            continue
        if any(test.examples for test in finder.find(mod)):
            modules.append(info.name)
    return modules


@pytest.mark.parametrize("module_name", _discover_doctest_modules())
def test_docstring_examples(module_name: str) -> None:
    """Verify that all ``>>>`` examples in *module_name* execute without error."""
    mod = importlib.import_module(module_name)
    results = doctest.testmod(mod, verbose=False, optionflags=doctest.ELLIPSIS)
    assert results.failed == 0, f"{results.failed} doctest failure(s) in {module_name}"
