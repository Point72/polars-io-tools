"""The MSSQL sqlglot dialect: bracket quoting, qualified stars, and the case of table-valued function names."""

from dataclasses import KW_ONLY, dataclass

import pytest
from sqlglot import exp, parse_one

from polars_io_tools.io_sources.sql_dialects import MSSQL
from polars_io_tools.tests.helpers.assertions import assert_translated_text, sql_text
from polars_io_tools.tests.helpers.cases import as_params


@dataclass(frozen=True)
class MssqlRenderCase:
    """A SQL node or query, and what its MSSQL rendering must hold."""

    id: str
    _: KW_ONLY
    # a sqlglot node, or SQL text parsed with the MSSQL dialect
    source: exp.Expression | str
    # keyword arguments for ``assert_translated_text``, from ``sql_text``
    expected_text: dict


# MSSQL rendering: bracket quoting, qualified stars, and case kept for anonymous nodes (table-valued functions).
MSSQL_RENDER = [
    MssqlRenderCase("mssql_bracket_quoting-plain", source=exp.Column(this="CenterID"), expected_text=sql_text(equals="[CenterID]")),
    MssqlRenderCase("mssql_bracket_quoting-reserved", source=exp.Column(this="GROUP"), expected_text=sql_text(equals="[GROUP]")),
    MssqlRenderCase("mssql_bracket_quoting-bracket", source=exp.Column(this="Some ] Column"), expected_text=sql_text(equals="[Some ]] Column]")),
    MssqlRenderCase(
        "mssql_bracket_quoting-qualified", source=exp.Column(this="Amount", table="Sales"), expected_text=sql_text(equals="[Sales].[Amount]")
    ),
    MssqlRenderCase(
        "qualified_star_in_join",
        source="SELECT a.*, b.SomeCol FROM ServerA.dbo.vTableOne a LEFT JOIN ServerB.dbo.vTableTwo b"
        " ON a.JoinKey = b.JoinKey AND a.AsOfDate BETWEEN b.StartDate AND b.EndDate",
        expected_text=sql_text("[a].*", omits="[*]"),
    ),
    MssqlRenderCase("qualified_star_without_join", source="SELECT t.* FROM SomeDB.dbo.SomeTable t", expected_text=sql_text("[t].*", omits="[*]")),
    MssqlRenderCase("unqualified_star", source="SELECT * FROM SomeTable", expected_text=sql_text("SELECT *", omits="[*]")),
    MssqlRenderCase(
        "tvf_with_literal_args_preserves_case",
        source="SELECT * FROM [FuturesData].[dbo].f_GetFuturesMaxVolAsOfDate('2026-02-06')",
        expected_text=sql_text("f_GetFuturesMaxVolAsOfDate", omits="F_GETFUTURESMAXVOLASOFDATE"),
    ),
    MssqlRenderCase(
        "table_with_nolock_hint_preserves_case", source="SELECT * FROM MyTable (nolock)", expected_text=sql_text("MyTable", omits="MYTABLE")
    ),
    MssqlRenderCase(
        "tvf_with_multiple_args_preserves_case",
        source="SELECT * FROM [db].[dbo].fn_GetData('2026-01-01', 42)",
        expected_text=sql_text("fn_GetData", omits="FN_GETDATA"),
    ),
    MssqlRenderCase(
        "tvf_roundtrip_preserves_full_query",
        source="SELECT * FROM [FuturesData].[dbo].f_GetFuturesMaxVolAsOfDate('2026-02-06')",
        expected_text=sql_text("[FuturesData]", "[dbo]", "f_GetFuturesMaxVolAsOfDate('2026-02-06')"),
    ),
]


@pytest.mark.parametrize("case", as_params(MSSQL_RENDER))
def test_mssql_render(case: MssqlRenderCase):
    node = parse_one(case.source, dialect=MSSQL) if isinstance(case.source, str) else case.source
    assert_translated_text(node.sql(dialect=MSSQL), **case.expected_text)
