"""Run a Python script in a new process with a set number of Polars threads.

Tests of deadlocks at one Polars thread need a new process: Polars reads ``POLARS_MAX_THREADS`` once, when it is
imported, and its thread pool cannot be resized afterwards.
"""

import os
import subprocess
import sys

# Long enough for the small queries these scripts run; a deadlocked script never finishes, so any timeout catches it.
SUBPROCESS_TIMEOUT_S = 60


def run_in_subprocess(script: str, *, threads: int = 1) -> subprocess.CompletedProcess:
    """Run ``script`` with ``POLARS_MAX_THREADS=threads`` and return the finished process, with its output as text.

    Raises ``subprocess.TimeoutExpired`` if the script runs longer than ``SUBPROCESS_TIMEOUT_S``.
    """
    # The rest of the environment is inherited, so the new process finds the same Python packages.
    env = {**os.environ, "POLARS_MAX_THREADS": str(threads)}
    return subprocess.run([sys.executable, "-P", "-c", script], env=env, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_S)
