# Tests

## Where tests go

Each library module has one test file, at the same place under `tests/`:

| Library module                                 | Test file                                 |
| ---------------------------------------------- | ----------------------------------------- |
| `polars_io_tools/io_sources/foo.py`            | `tests/io_sources/test_foo.py`            |
| `polars_io_tools/io_sources/__init__.py`       | `tests/io_sources/test_init.py`           |
| `polars_io_tools/testing/predicate_tracker.py` | `tests/testing/test_predicate_tracker.py` |

A test goes in the file of the module whose function it calls, including its property tests,
regression tests and pickling tests. A test that calls functions of several modules goes in the
file of the main one, and its docstring names the others.

So the tests of a function are mostly in its module's file, but not all of them: when another
module uses the function, the tests of that use live in the caller's file. For example, `by_time`
from `partitions.py` is tested in `test_partitions.py`, and its use by the Ray and SQL readers in
`test_lazy_ray.py` and `test_lazy_sql_reader.py`. Search the tests for the function's name to find
them all.

Three places hold other tests:

- `tests/pushdown/`: what plain Polars hands to a data source (filters, column selections, row
  counts), one file per Polars operation. They record how Polars behaves, so that a change in a new
  Polars version shows up as a failing test.
- `tests/test_doctests.py`: runs the `>>>` examples in the docstrings of `io_sources`.
- `tests/integration/`: tests against real S3 and databases, which need credentials. See
  `tests/integration/README.md`.

## Shared helpers

Code that more than one test file uses lives in `tests/helpers/`. Code used by one file stays in
that file.

- `cases.py`: `as_params` for writing a test as a table of rows (see "A table-driven test" below),
  and data several files use (dtypes, awkward values, `daily_frame`, `create_sample_data`).
- `assertions.py`: compare a result with plain Polars (`assert_matches_plain_polars`), check what a
  source received (`assert_pushdown` with an `ExpectedPushdown`, `assert_pushed_filter_keeps_rows`,
  `assert_pushed_time_bounds`) and check SQL text (`assert_translated_text`, `sql_text`).
  `failure_note` adds the step of a test (which read, query or source) to a failing assertion.
- `polars_versions.py`: `polars_has` and the xfail marks, for expectations that depend on the
  Polars version.
- `strategies.py`: Hypothesis strategies for property tests (`frames_and_predicates`,
  `random_filter`, `partitioned_frames_and_predicates`).
- `property_checks.py`: what property tests check against: `SETTINGS`, the rows plain Polars keeps
  for a filter, the filter Polars hands to a source, and `check_cache_keeps_filtered_rows`.
- `node_shapes.py`: `NodeShape`, to check the tree `get_parsed_expr` returns.
- `sql_backends.py`: `FakeDatabase`, an in-memory DuckDB that `scan_db` and `scan_clickhouse`
  query, so the SQL readers run without a server.
- `sql_query_cases.py`: the queries both SQL readers must answer as the database does, and their
  check. Add a query here when it applies to both readers.
- `cache_helpers.py`: for `cache_parquet` and `cache_memory` tests: compare a cached read
  with plain Polars, list the files a cache wrote, and record the reads of the data behind it.
- `pickling.py`: `pickle_round_trip`, to check that a frame survives cloudpickle (as Ray needs).
- `subprocesses.py`: `run_in_subprocess`, to run a script with one Polars thread when testing
  for a deadlock.

`tests/pushdown/helpers.py` holds the row type (`PushdownCase`) and the check of the
`tests/pushdown/` tables.
The root `conftest.py` registers the Hypothesis profiles and the `gap` and `benchmark` marks, and
provides the `tracked_sample_data` fixture.

The `gap` mark labels a pushdown test row where Polars could hand a filter or column selection to
the source but does not. The row states what Polars does today, so it passes; when a new Polars
version closes the gap, the row fails and should be updated. `-m gap` runs only these rows.

Many tests call the library through `lf.piot`, for example `lf.piot.filtered_join(...)`. `piot` is
the namespace polars-io-tools registers on `LazyFrame` when it is imported, so its functions can be
called as methods of a frame.

## Adding a test

### A regression test for a bug

Put it in the test file of the module that had the bug. If the bug fits a table in that file, add
a row. Otherwise write a test function whose name says what must hold. Check that the test fails
without the fix.

For a `cache_parquet` bug in `test_lazy_cache_parquet.py`: if the bug comes from a partition value
that cannot be a file name (too long, a reserved Windows name, a trailing dot or space), add the
value to the parametrized `test_values_no_file_can_be_named_are_read_from_upstream`. Add other cache
bugs as a row of the `REGRESSIONS` table, or of `NO_REWRITE_REGRESSIONS` if no read may rewrite the
files the seed reads wrote.

### Checking a fix for new bugs

A fix often adds a code path that the bug's own test is the only one to reach. Before opening a
pull request:

1. List what the new path skips or adds: options it no longer applies (for example `read_kwargs`
   applied to one scan of two) and values it now computes (for example every day of a range).
2. Add those options and values to the strategy of a property test of the module, and run it with
   `HYPOTHESIS_PROFILE=hunt`.
3. Run inputs that reach the new path on the main branch too (`git worktree add ../base main`). A
   result that changed is a new bug unless plain Polars gives the new result.

### A pushdown expectation

For a new IO source, test what it receives and returns in its own test file. If the source calls a
backend (a database, an HTTP API), see the next paragraph. Otherwise, wrap the data in a
`PredicateTracker` (from `polars_io_tools.testing`) to record the filter and columns a source
receives, check them with `assert_pushdown`, and compare results with
`assert_matches_plain_polars`.

If the source sends filters on to a backend, the test cannot reach the real backend, so replace the
call to it:

1. Replace the backend call. `test_lazy_datadog_reader.py` patches the API function with
   `unittest.mock.patch` (or use pytest's `monkeypatch.setattr`) and records the arguments. For the
   SQL readers, the `fake_database` fixture of `test_lazy_sql_reader.py` and
   `test_lazy_clickhouse_reader.py` routes the queries to a `FakeDatabase` and records them.
1. Check what was sent. For text (SQL, a query string) use `assert_translated_text`. For a Polars
   expression use `assert_pushed_filter_keeps_rows`, which checks that the sent filter drops no row
   the user's filter keeps.
1. Compare the result with plain Polars reading the same data, with `assert_matches_plain_polars`.

For example, in `test_lazy_sql_reader.py`, which defines the `fake_database` fixture:

```python
from datetime import date

import polars as pl

from polars_io_tools.tests.helpers.assertions import assert_matches_plain_polars, assert_translated_text
from polars_io_tools.tests.helpers.sql_backends import SQL_READERS


def test_scan_db_sends_the_date_filter(fake_database):
    query = "SELECT * FROM SampleA"
    pred = pl.col("event_date") >= date(2016, 1, 1)
    start = len(fake_database.calls)
    reference = fake_database.run_directly(query).lazy()  # the database's own answer
    assert_matches_plain_polars(SQL_READERS["scan_db"](query), reference, pred)
    (sql,) = fake_database.queries(start)  # the one query that read rows
    assert_translated_text(sql, ["WHERE", "event_date", "2016-01-01"])
```

`fake_database.calls` records every query a reader sends. `fake_database.queries(start)` returns the
queries sent since call number `start`, leaving out the `LIMIT 0` query the readers send to fetch
the column names and types.

For a Polars operation, add a `PushdownCase` row to `tests/pushdown/test_<operation>.py`, or create
that file with its own data, a `ROWS` table and the same short `test_pushdown` function as the
other files. `expected_pushdown` maps a source name to an `ExpectedPushdown`: `predicate` is
whether a filter reached the source, `projection` the columns it was asked for (or whether it was
asked for a subset) and `calls` how many times it was scanned. If Polars could hand a filter or
column selection to the source but does not, add `marks=(pytest.mark.gap,)` to the row and cite
the Polars issue in a comment above it.

### A table-driven test

Most tests are tables: a list of rows and one test function that runs each row. Each table has its
own small dataclass, named after what a row is (`SqlTranslationCase`, `RangeCase`), with the test
id first and the other fields keyword-only:

```python
from dataclasses import KW_ONLY, dataclass

import polars as pl
import portion
import pytest

from polars_io_tools.io_sources.range_visitor import convert_expr_to_range
from polars_io_tools.tests.helpers.cases import as_params


@dataclass(frozen=True)
class RangeCase:
    """A filter, and the range of values it allows in one column."""

    id: str
    _: KW_ONLY
    predicate: pl.Expr
    expected_range: portion.Interval
    column: str = "value_col"


RANGE_CASES = [
    RangeCase("equality", predicate=pl.col("value_col") == 42, expected_range=portion.singleton(42)),
    RangeCase("other_column", predicate=pl.col("x") > 1, expected_range=portion.open(1, portion.inf), column="x"),
]


@pytest.mark.parametrize("case", as_params(RANGE_CASES))
def test_range(case: RangeCase):
    assert convert_expr_to_range(case.predicate, case.column) == case.expected_range
```

To add a row, add an instance to the table; its `id` is the name pytest shows for it. A field
whose meaning is not obvious from its name gets a one-line comment above it. A row that needs a
pytest mark (a skip, an expected failure, `gap`) has a `marks: tuple = ()` field and sets
`marks=(...,)`; `as_params` passes it to `pytest.param`. Keep any computation the test needs, such
as a default that depends on another field, in the test function.

### A Polars-version-dependent expectation

Add the behavior to `POLARS_BEHAVIORS` in `helpers/polars_versions.py`, with the first Polars
version that has it and a one-line description. Then use `polars_has("name")` for the expected
value, or mark the test with `xfail_unless_polars_has("name")` or `xfail_if_polars_has("name")`.
Do not compare version strings in a test.

To find the tests that depend on a behavior, search for its name (for example
`polars_has("pushes_forward_asof_filter")`) and for the name of the Polars operation.

### A property test

Put it in the test file of the module it checks. Draw inputs with the strategies in
`helpers/strategies.py` and check them with `helpers/property_checks.py`. Decorate the test with
`@SETTINGS` and `@given(...)`. When the test finds a failure, add the failing input as an
`@example(...)`, or as a named test, so it runs every time.

## Running the tests

```bash
make test-py                                                    # the whole suite
python -m pytest polars_io_tools/tests/io_sources/test_util.py  # one file
python -m pytest polars_io_tools/tests -m gap                   # the pushdown gaps of Polars
python -m pytest polars_io_tools/tests --run-benchmarks         # with the benchmarks
```

The property tests use [Hypothesis](https://hypothesis.readthedocs.io). Choose a profile with
`HYPOTHESIS_PROFILE=<name>` or `--hypothesis-profile=<name>`:

- `quick` (the default): 50 examples per test.
- `ci`: `quick` with examples in a fixed order, printing how to reproduce a failure.
- `hunt`: 2,000 examples per test, for a longer search for bugs.

Hypothesis draws different examples on each run. To repeat a run, pass `--hypothesis-seed=<n>`.
