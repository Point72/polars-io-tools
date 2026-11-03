"""What plain Polars pushes to a source through each operation, one row per pipeline.

A row builds a pipeline over ``PredicateTracker`` sources and names, per source, whether a predicate or a projection
reached it (``assert_pushdown``). Rows marked ``gap`` lock in a pushdown Polars could perform but does not; their build
is a starting point for a reproducer of an upstream issue, which still needs a source that does not depend on this
package. Pivot and unpivot rows live with their wrappers in
``io_sources/test_pushdown_pivot.py`` and ``io_sources/test_pushdown_unpivot.py``.
"""

import polars as pl
import pytest

from polars_io_tools.testing import PredicateTracker
from polars_io_tools.tests.helpers.assertions import assert_pushdown
from polars_io_tools.tests.helpers.cases import Case, params
from polars_io_tools.tests.helpers.polars_versions import polars_has

c = pl.col
gap = pytest.mark.gap

X32 = {"x": pl.DataFrame({"x": pl.Series([1, 2, 3, 4, 5], dtype=pl.Int32)})}
KA = pl.DataFrame({"k": [1, 2, 3], "a": [10, 20, 30]})
CONCAT = {"a": KA, "b": pl.DataFrame({"k": [1, 2, 3], "b": [100, 200, 300]})}
LABELLED = {"a": pl.DataFrame({"k": [1, 2, 3], "v": [10, 20, 30]}), "b": pl.DataFrame({"k": [4, 5], "v": [40, 50]})}
CSE = {"t": pl.DataFrame({"k": [1, 2, 3, 4, 5], "v": [10, 20, 30, 40, 50]})}
GROUPED = {"t": pl.DataFrame({"g": ["a", "a", "b", "b"], "h": [1, 1, 2, 2], "k": [1, 10, 2, 20], "v": [10, 20, 30, 40], "w": [1, 2, 3, 4]})}
ASOF = {
    "a": pl.DataFrame({"t": [1, 2, 3, 4], "g": ["a", "a", "b", "b"], "va": [10, 20, 30, 40]}),
    "b": pl.DataFrame({"t": [1, 2, 3, 4], "g": ["a", "a", "b", "b"], "vb": [100, 200, 300, 400]}),
}
MODELS = {"t": pl.DataFrame({"model": ["A", "B", "C", "D"], "val": [1, 2, 3, 4]})}
LOOKUP = pl.DataFrame({"model": ["A", "B", "C", "D"], "family": ["X", "X", "Y", "Y"]}).lazy()
FAMILY = {"A": "X", "B": "X", "C": "Y", "D": "Y"}
JOINED = {"a": KA, "b": pl.DataFrame({"k": [1, 2, 3], "b": [100, 200, 300], "c": [1, 2, 3]})}
WINDOWED = {"t": pl.DataFrame({"k": [1, 1, 2, 2, 3], "v": [10, None, 30, None, 50]})}


def _asof(t):
    return t["a"].join_asof(t["b"], on="t", by="g")


unsorted_by = pytest.mark.filterwarnings("ignore:Sortedness of columns cannot be checked")


def _x_is_3_4_5(result, _trackers):
    assert result["x"].to_list() == [3, 4, 5]


def _vb1_not_in_right_scan(_result, trackers):
    assert "vb1" not in (trackers["b"].last_with_columns or []), trackers["b"].last_with_columns


def P(id, frames, build, marks=(), check=None, **expect):
    """A row: ``build`` maps lazy frames by name to the pipeline; ``expect`` maps a source name to ``assert_pushdown`` arguments."""
    return Case(id, build=build, expect=expect, kw={"frames": frames, "check": check}, marks=marks)


ROWS = [
    # cast: a widening cast is lossless, so a filter above it is equivalent on the source (polars#23369). Polars < 2.1
    # leaves the filter above the cast. A narrowing cast can wrap, so it is not a candidate.
    P(
        "cast-widening_cast_then_filter_pushdown",
        X32,
        lambda t: t["x"].with_columns(c("x").cast(pl.Int64)).filter(c("x") > 2),
        gap,
        _x_is_3_4_5,
        x={"predicate": polars_has("pushes_widening_cast")},
    ),
    # concat
    P(
        "concat-vertical_concat_filter_on_shared_col_pushed",
        CONCAT,
        lambda t: pl.concat([t["a"].select("k", "a"), t["b"].select("k", c("b").alias("a"))]).filter(c("k") == 2),
        a={"predicate": True},
        b={"predicate": True},
    ),
    # Horizontal concat is positional: a sound rewrite applies one row mask to every input. Polars pushes nothing (polars#26552).
    P(
        "concat-horizontal_concat_filter_on_left_col_NOT_pushed",
        CONCAT,
        lambda t: pl.concat([t["a"], t["b"].select("b")], how="horizontal").filter(c("k") > 1),
        gap,
        a={"predicate": False},
    ),
    # The side tagged with the other label could be dropped from the plan; both sides are scanned (polars#23345).
    P(
        "concat-vertical_concat_literal_side_NOT_pruned",
        LABELLED,
        lambda t: pl.concat([t["a"].with_columns(pl.lit("real").alias("kind")), t["b"].with_columns(pl.lit("synthetic").alias("kind"))]).filter(
            c("kind") == "real"
        ),
        gap,
        a={"calls": 1},
        b={"calls": 1},
    ),
    # CSE: two filters of one frame scan it twice; one scan with p1 | p2 would do (polars#26502).
    P(
        "cse-concat_of_two_filters_scans_source_twice",
        CSE,
        lambda t: pl.concat([t["t"].filter(c("k") < 3), t["t"].filter(c("k") > 3)]),
        gap,
        t={"calls": 2},
    ),
    # group_by
    P("group_by-filter_on_group_key_pushed", GROUPED, lambda t: t["t"].group_by("g").agg(c("v").sum()).filter(c("g") == "a"), t={"predicate": True}),
    P(
        "group_by-filter_on_secondary_group_key_pushed",
        GROUPED,
        lambda t: t["t"].group_by("g", "h").agg(c("v").sum()).filter(c("h") == 1),
        t={"predicate": True},
    ),
    P(
        "group_by-select_subset_of_aggs_prunes_unused_source_cols",
        GROUPED,
        lambda t: t["t"].group_by("g").agg(c("v").sum().alias("vs"), c("w").sum().alias("ws")).select("g", "vs"),
        t={"projection": {"g", "v"}},
    ),
    # polars#22623: grouping by a literal still prunes the source columns the aggregations do not use.
    P("group_by-group_by_literal_key_pushes_projection", GROUPED, lambda t: t["t"].group_by(pl.lit(1)).agg(c("v").sum()), t={"projection": {"v"}}),
    # Rows the filter drops add nothing to the sum, so the filter could reach the source (polars#27207).
    P(
        "group_by-expr_filter_inside_agg_NOT_pushed",
        GROUPED,
        lambda t: t["t"].group_by("g").agg(c("v").filter(c("k") > 5).sum()),
        gap,
        t={"predicate": False},
    ),
    # join_asof: a ``by`` value either matches on both sides or is null after the join, so a filter on it reaches both. A
    # filter on the ``on`` key or a right-only column would change the match candidates and is not a candidate.
    P(
        "join_asof-filter_on_by_column_pushed",
        ASOF,
        lambda t: _asof(t).filter(c("g") == "a"),
        unsorted_by,
        a={"predicate": True},
        b={"predicate": True},
    ),
    # A null-preserving row-wise expression on a right-only column could move into the right scan (polars#25867).
    P(
        "join_asof-post_join_elementwise_expr_on_right_col_NOT_moved_into_right_scan",
        ASOF,
        lambda t: _asof(t).with_columns((c("vb") + 1).alias("vb1")),
        (gap, unsorted_by),
        _vb1_not_in_right_scan,
    ),
    # A filter on a column from a small lookup could become an is_in on the source key (polars#21710, polars#23345).
    P(
        "join_lookup-lookup_join_filter_on_derived_col_NOT_pushed_as_isin",
        MODELS,
        lambda t: t["t"].join(LOOKUP, on="model").filter(c("family") == "X"),
        gap,
        t={"predicate": False},
    ),
    P("join_lookup-manual_isin_filter_pushed", MODELS, lambda t: t["t"].filter(c("model").is_in(["A", "B"])), t={"predicate": True}),
    # join
    P(
        "join-inner_join_filter_on_key_pushed",
        JOINED,
        lambda t: t["a"].join(t["b"], on="k").filter(c("k") == 2),
        a={"predicate": True},
        b={"predicate": True},
    ),
    P(
        "join-right_join_filter_on_key_pushed",
        JOINED,
        lambda t: t["a"].join(t["b"], on="k", how="right").filter(c("k") == 2),
        a={"predicate": True},
        b={"predicate": True},
    ),
    P(
        "join-inner_join_projection_prunes_unused_right_cols",
        JOINED,
        lambda t: t["a"].join(t["b"], on="k").select("k", "a", "b"),
        b={"projection": {"k", "b"}},
    ),
    # replace_strict maps every source value through a literal dict, so a filter on its output could become an is_in on
    # the source (polars#23345). Non-strict replace passes unmapped values through and has no such simple inverse.
    P(
        "replace-replace_strict_filter_on_derived_col_NOT_pushed",
        MODELS,
        lambda t: t["t"].with_columns(c("model").replace_strict(FAMILY).alias("family")).filter(c("family") == "X"),
        gap,
        t={"predicate": False},
    ),
    P(
        "replace-filter_on_source_col_after_replace_strict_pushed",
        MODELS,
        lambda t: t["t"].with_columns(c("model").replace_strict(FAMILY).alias("family")).filter(c("model") == "B"),
        t={"predicate": True},
    ),
    # window: dropping other partitions leaves a partition's window values unchanged, so a filter on the partition key
    # reaches the source. A filter on another column would change them and is not a candidate.
    P(
        "window-filter_on_partition_col_of_shift_over_pushed",
        WINDOWED,
        lambda t: t["t"].with_columns(c("v").shift(1).over("k").alias("vs")).filter(c("k") == 1),
        t={"predicate": True},
    ),
    P(
        "window-filter_on_partition_col_of_forward_fill_over_pushed",
        WINDOWED,
        lambda t: t["t"].with_columns(c("v").forward_fill().over("k").alias("vs")).filter(c("k") == 1),
        t={"predicate": True},
    ),
]


class _Frames(dict):
    """Trackers by name; a lookup returns the tracker's frame, so a pipeline may read one source twice."""

    def __getitem__(self, name):
        return super().__getitem__(name).lazy_frame


@pytest.mark.parametrize(("build", "expect", "kw"), params(ROWS, "build", "expect", "kw"))
def test_pushdown(build, expect, kw):
    trackers = {name: PredicateTracker(df) for name, df in kw["frames"].items()}
    result = build(_Frames(trackers)).collect()
    for name, observed in expect.items():
        assert_pushdown(trackers[name], **observed)
    if kw["check"]:
        kw["check"](result, trackers)
