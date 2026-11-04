"""Polars behavior that changed between Polars versions, listed in one table.

A test that depends on such a behavior calls ``polars_has("name")`` instead of comparing version strings, so each
change is written down once, with the first Polars version that has it.
"""

import polars as pl
import pytest
from packaging.version import Version

POLARS_VERSION = Version(pl.__version__)

# behavior name -> (first Polars version with the behavior, what the behavior is)
POLARS_BEHAVIORS: dict[str, tuple[str, str]] = {
    "logical_operands_reach_source_uncast": (
        "1.8.0",
        "all_horizontal(i8, i8).cast(Boolean) reaches an IO source as i8 && i8, whose non-Boolean operands a lazy filter rejects",
    ),
    "pushes_function_predicates": ("1.25.2", "string and temporal-component predicates such as str.contains and dt.day reach the source"),
    "pushes_python_datetime_comparison": ("1.26.0", "a comparison with a Python datetime literal reaches the source"),
    "is_in_takes_a_list_value": ("1.28.0", "is_in takes its values as one list value (an imploded Series) instead of a column"),
    "pushes_implicit_casts": ("1.28.0", "a predicate holding an implicit or explicit numeric/temporal cast reaches the source"),
    "nested_alias_dropped": ("1.31.0b1", "meta.pop() returns no input for a nested alias, so the parser loses it"),
    "list_eval_serializes": ("1.31.0b1", "list.eval serializes as an Eval node instead of an opaque function"),
    "pushes_fill_null_predicates": ("1.31.0b1", "a fill_null predicate reaches the source"),
    "struct_literal_parsed": ("1.32.0b1", "a dict literal parses as a struct LiteralNode instead of an ErrorNode"),
    "pushes_join_filters": ("1.32.0b1", "Polars pushes a filter after a join to the sources of the join itself"),
    "rewrites_filtered_left_join_as_inner": ("1.32.0b1", "a left join followed by a filter that drops null right rows becomes an inner join"),
    "anonymous_function_not_pushed": ("1.32.0b1", "a predicate on an anonymous function reaches the source as an expression the parser cannot read"),
    "pure_udf_cse": ("1.33.1", "common-subexpression elimination dedupes a pure map_batches UDF"),
    "datetime_literal_unwrapped": ("1.36.0b1", "pl.lit(datetime) parses as a LiteralNode instead of a CastNode"),
    "time_partition_subsecond": ("1.38.0", "a Time partition key keeps sub-second precision in its path"),
    "join_asof_null_by_unmatched": ("1.44.0", "join_asof no longer matches a null by key"),
    "cross_joins_filtered_sources": ("1.44.0", "a cross join of two IO sources filtered on a null key returns the right rows"),
    "is_in_coerces_temporal_units": ("2.0.0", "is_in compares a temporal list in the column's unit, as equality does (polars#22824)"),
    "ns_literal_min_overflows": ("2.0.0", "re-reading a nanosecond literal near the Int64 minimum overflows"),
    "pushes_widening_cast": ("2.1.0", "a filter above a widening cast reaches the source (polars#23369)"),
    "pushes_forward_asof_filter": ("2.1.0", "a filter after a forward join_asof reaches the right source"),
    "simplifies_eq_true": ("2.1.0", "a filter col == True reaches the source as col"),
}


def polars_has(name: str) -> bool:
    """Whether the installed Polars has the behavior ``name``."""
    return POLARS_VERSION >= Version(POLARS_BEHAVIORS[name][0])


def xfail_unless_polars_has(name: str, reason: str = "") -> pytest.MarkDecorator:
    """A pytest mark: the test must fail on Polars versions without the behavior ``name`` and pass on the others."""
    since, what = POLARS_BEHAVIORS[name]
    return pytest.mark.xfail(not polars_has(name), reason=reason or f"needs Polars >= {since}: {what}", strict=True)


def xfail_if_polars_has(name: str, reason: str = "") -> pytest.MarkDecorator:
    """A pytest mark: the test must fail on Polars versions with the behavior ``name`` and pass on the others."""
    since, what = POLARS_BEHAVIORS[name]
    return pytest.mark.xfail(polars_has(name), reason=reason or f"fails from Polars {since}: {what}", strict=True)
