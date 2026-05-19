import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import polars as pl

from polars_io_tools.io_sources.restrict_visitor import restrict_expr_to_columns

from .util import collect_lf_in_io_source, register_io_source_with_is_pure

__all__ = ("concat_named",)

log = logging.getLogger(__name__)


def concat_named(
    lf_dict: Dict[Any, pl.LazyFrame],
    identifier_cols: List[Union[str, Tuple[str, pl.DataType]]],
    *,
    log_explain: bool = False,
    **kwargs: Any,
) -> pl.LazyFrame:
    """
    Concatenate multiple LazyFrames into a single LazyFrame with added identifier columns.

    This function vertically concatenates LazyFrames (horizontal and other concatenation types are not supported yet)
    from a dictionary mapping identifier tuples to LazyFrames.
    It adds the identifier values as columns to each LazyFrame, allowing downstream filtering
    on these identifier columns. The function optimizes filter operations by only materializing
    LazyFrames whose identifiers match the filter conditions, significantly improving performance
    when working with large collections of LazyFrames where queries typically target specific subsets.

    This addresses a limitation in Polars where filters on constant-valued columns in unions
    do not prune branches. See https://github.com/pola-rs/polars/issues/24782 for context.
    With native ``pl.concat``, filtering on a literal column still executes all branches;
    ``concat_named`` intercepts the predicate and only materializes matching LazyFrames.

    Args:
        lf_dict (Dict[Any, pl.LazyFrame]): Dictionary mapping identifier tuples to LazyFrames. Each key should be a tuple whose length
            matches the length of `identifier_cols`. The values in these tuples will be added as columns
            to identify the source LazyFrame for each row.

        identifier_cols (List[Union[str, Tuple[str, pl.DataType]]]): A list specifying the identifier columns to add. Each element can be either:
            - A string: the column name (data type will be inferred)
            - A tuple: (column_name, data_type) to explicitly set the column type
            The order of elements in this list must match the order of values in the dictionary keys.

        log_explain (bool, default False): If True, logs the LazyFrame execution plan for debugging purposes.

        **kwargs (Any): Additional arguments passed to `pl.concat()` for concatenation.

    Returns:
        pl.LazyFrame: A LazyFrame representing the concatenated data with added identifier columns.

    Notes:
        - Filter operations on the identifier columns are optimized to only load the LazyFrames
          that match the filter conditions.
        - The LazyFrames are concatenated in the same order as they appear in the input dictionary.

    Examples:
        Basic usage with single identifier column:
            >>> lf1 = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            >>> lf2 = pl.LazyFrame({"a": [7, 8, 9], "b": [10, 11, 12]})
            >>> result = concat_named({("foo",): lf1, ("bar",): lf2}, ["source"]).collect()
            >>> print(result)
            shape: (6, 3)
            ┌─────┬─────┬────────┐
            │ a   ┆ b   ┆ source │
            │ --- ┆ --- ┆ ---    │
            │ i64 ┆ i64 ┆ str    │
            ╞═════╪═════╪════════╡
            │ 1   ┆ 4   ┆ foo    │
            │ 2   ┆ 5   ┆ foo    │
            │ 3   ┆ 6   ┆ foo    │
            │ 7   ┆ 10  ┆ bar    │
            │ 8   ┆ 11  ┆ bar    │
            │ 9   ┆ 12  ┆ bar    │
            └─────┴─────┴────────┘

        With filter optimization (only lf1 will be materialized):
            >>> result = concat_named({("foo",): lf1, ("bar",): lf2}, ["source"]).filter(pl.col("source") == "foo").collect()
            >>> print(result)
            shape: (3, 3)
            ┌─────┬─────┬────────┐
            │ a   ┆ b   ┆ source │
            │ --- ┆ --- ┆ ---    │
            │ i64 ┆ i64 ┆ str    │
            ╞═════╪═════╪════════╡
            │ 1   ┆ 4   ┆ foo    │
            │ 2   ┆ 5   ┆ foo    │
            │ 3   ┆ 6   ┆ foo    │
            └─────┴─────┴────────┘

        With multiple identifier columns and explicit types:
            >>> lf1 = pl.LazyFrame({"data": [1, 2]})
            >>> lf2 = pl.LazyFrame({"data": [3, 4]})
            >>> result = concat_named(
            ...     {("east", "2023-01-01"): lf1, ("west", "2023-01-02"): lf2},
            ...     [("region", pl.Utf8), ("date", pl.Date)]
            ... ).collect()
            >>> print(result)
            shape: (4, 3)
            ┌──────┬────────┬────────────┐
            │ data ┆ region ┆ date       │
            │ ---  ┆ ---    ┆ ---        │
            │ i64  ┆ str    ┆ date       │
            ╞══════╪════════╪════════════╡
            │ 1    ┆ east   ┆ 2023-01-01 │
            │ 2    ┆ east   ┆ 2023-01-01 │
            │ 3    ┆ west   ┆ 2023-01-02 │
            │ 4    ┆ west   ┆ 2023-01-02 │
            └──────┴────────┴────────────┘
    """
    if not lf_dict:
        raise ValueError("Cannot concatenate an empty dictionary of LazyFrames.")
    # We concatenate these, and make sure we have a list of unique columns that
    # will extract the correct lazyframe from.
    data_dict = {}
    col_names = [col if isinstance(col, str) else col[0] for col in identifier_cols]
    index_lf = {col: [] for col in col_names}
    lf_id_col = "__lf_id"
    index_lf[lf_id_col] = []
    schema = None

    for keys, lf in lf_dict.items():
        expr_list = []
        if len(identifier_cols) != len(keys):
            raise ValueError(f"Number of unique columns {len(identifier_cols)} does not match number of keys {len(keys)}")
        for col_info, value in zip(identifier_cols, keys):
            if isinstance(col_info, tuple):
                col_name, dtype = col_info
            else:
                col_name = col_info
                dtype = None  # inferred
            index_lf[col_name].append(value)
            expr = pl.lit(value)
            if dtype is not None:
                expr = expr.cast(dtype)
            expr = expr.alias(col_name)
            expr_list.append(expr)
        lf = lf.with_columns(expr_list)
        if schema is None:
            schema = lf.collect_schema()
        id_ = id(lf)
        index_lf[lf_id_col].append(id_)
        data_dict[id_] = lf

    index_df = pl.DataFrame(index_lf)

    def source_gen(
        with_columns: Optional[List[str]],
        predicate: Optional[pl.Expr],
        n_rows: Optional[int],
        batch_size: Optional[int],
    ) -> Iterator[pl.DataFrame]:
        restricted_predicate = restrict_expr_to_columns(predicate, col_names) if predicate is not None else None
        true_lf: pl.LazyFrame
        if restricted_predicate is not None:
            lf_ids = set(index_df.filter(restricted_predicate).select(lf_id_col).to_series().to_list())
            lf_to_concat = [lf for lf_id, lf in data_dict.items() if lf_id in lf_ids]
            true_lf = pl.concat(lf_to_concat, **kwargs)  # type: ignore[assignment]
        else:
            true_lf = pl.concat(data_dict.values(), **kwargs)  # type: ignore[assignment]
        if predicate is not None:
            true_lf = true_lf.filter(predicate)
        if with_columns is not None:
            true_lf = true_lf.select(with_columns)

        if n_rows is not None:
            true_lf = true_lf.limit(n_rows)
        if log_explain:
            log.debug(f"concat_named: LazyFrame plan:\n{str(true_lf.explain())}")
        try:
            yield from collect_lf_in_io_source(true_lf, batch_size)
        except Exception as e:
            err_msg = f"Failed to collect lazy frame in concat_named.\nPolars plan:\n{true_lf.explain()}"
            err_msg += f"\n\nWhile running the above, received error: {e.__class__.__name__}:{e}"
            raise RuntimeError(err_msg) from e

    return register_io_source_with_is_pure(source_gen, schema=schema)
