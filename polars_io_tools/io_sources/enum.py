from enum import Enum
from typing import Optional

import polars as pl

__all__ = [
    "get_function_enum",
    "ArrayFunctionType",
    "BinaryFunctionType",
    "BitwiseFunctionType",
    "BooleanFunctionType",
    "DataType",
    "Expr",
    "FunctionType",  # Include the base class
    "ListFunctionType",
    "OperatorType",
    "StringFunctionType",
    "StructFunctionType",
    "TemporalFunctionType",
    "TimeUnit",
    "TrigonometricFunctionType",
]


class Expr(str, Enum):
    """Enum representing expression types, subclassing str for string compatibility."""

    Alias = "Alias"
    Column = "Column"
    Columns = "Columns"
    DtypeColumn = "DtypeColumn"
    IndexColumn = "IndexColumn"
    Literal = "Literal"
    BinaryExpr = "BinaryExpr"
    Cast = "Cast"
    Sort = "Sort"
    Gather = "Gather"
    SortBy = "SortBy"
    Agg = "Agg"
    Ternary = "Ternary"
    Function = "Function"
    Explode = "Explode"
    Filter = "Filter"
    Window = "Window"
    Wildcard = "Wildcard"
    Slice = "Slice"
    Exclude = "Exclude"
    KeepName = "KeepName"
    Len = "Len"
    Nth = "Nth"
    RenameAlias = "RenameAlias"
    Field = "Field"
    AnonymousFunction = "AnonymousFunction"
    SubPlan = "SubPlan"
    Selector = "Selector"


class OperatorType(str, Enum):
    """Binary operators that can be used in expressions."""

    # Comparison operators
    EQ = "Eq"
    EQ_VALIDITY = "EqValidity"
    NOT_EQ = "NotEq"
    NOT_EQ_VALIDITY = "NotEqValidity"
    LT = "Lt"
    LT_EQ = "LtEq"
    GT = "Gt"
    GT_EQ = "GtEq"

    # Arithmetic operators
    PLUS = "Plus"
    MINUS = "Minus"
    MULTIPLY = "Multiply"
    DIVIDE = "Divide"
    TRUE_DIVIDE = "TrueDivide"
    FLOOR_DIVIDE = "FloorDivide"
    MODULUS = "Modulus"

    # Logical operators
    AND = "And"
    OR = "Or"
    XOR = "Xor"
    LOGICAL_AND = "LogicalAnd"
    LOGICAL_OR = "LogicalOr"

    # Map internal representation to display string
    @property
    def display_value(self) -> str:
        display_map = {
            "Eq": "=",
            "EqValidity": "=_validity",
            "NotEq": "!=",
            "NotEqValidity": "!=_validity",
            "Lt": "<",
            "LtEq": "<=",
            "Gt": ">",
            "GtEq": ">=",
            "Plus": "+",
            "Minus": "-",
            "Multiply": "*",
            "Divide": "/",
            "TrueDivide": "true_div",
            "FloorDivide": "//",
            "Modulus": "%",
            "And": "and",
            "Or": "or",
            "Xor": "xor",
            "LogicalAnd": "&&",
            "LogicalOr": "||",
        }
        return display_map.get(self.value, self.value)

    def is_comparison(self) -> bool:
        """Check if the operator is a comparison operator."""
        # For this enum, we check the string value against comparison operators
        return self in {
            OperatorType.EQ,
            OperatorType.NOT_EQ,
            OperatorType.LT,
            OperatorType.LT_EQ,
            OperatorType.GT,
            OperatorType.GT_EQ,
            OperatorType.EQ_VALIDITY,
            OperatorType.NOT_EQ_VALIDITY,
        }

    def is_bitwise(self) -> bool:
        """Check if the operator is a bitwise operator."""
        # In this enum, AND, OR, XOR are considered bitwise operations
        return self in {OperatorType.AND, OperatorType.OR, OperatorType.XOR}

    def is_bitwise_or_logical(self) -> bool:
        """Check if the operator is either a bitwise or a logical operator."""
        return self.is_bitwise() or self in {OperatorType.LOGICAL_AND, OperatorType.LOGICAL_OR}

    def is_comparison_or_bitwise(self) -> bool:
        """Check if the operator is either a comparison or a bitwise operator."""
        return self.is_comparison() or self.is_bitwise()

    def swap_operands(self) -> "OperatorType":
        """Return the operator that results from swapping the operands.
        For commutative operations, this returns the same operator.
        For non-commutative operations, returns the inverse operator.
        """
        swap_map = {
            OperatorType.EQ: OperatorType.EQ,  # commutative
            OperatorType.GT: OperatorType.LT,  # swaps
            OperatorType.GT_EQ: OperatorType.LT_EQ,  # swaps
            OperatorType.LT_EQ: OperatorType.GT_EQ,  # swaps
            OperatorType.OR: OperatorType.OR,  # commutative
            OperatorType.LOGICAL_AND: OperatorType.LOGICAL_AND,  # commutative
            OperatorType.LOGICAL_OR: OperatorType.LOGICAL_OR,  # commutative
            OperatorType.XOR: OperatorType.XOR,  # commutative
            OperatorType.NOT_EQ: OperatorType.NOT_EQ,  # commutative
            OperatorType.EQ_VALIDITY: OperatorType.EQ_VALIDITY,  # commutative
            OperatorType.NOT_EQ_VALIDITY: OperatorType.NOT_EQ_VALIDITY,  # commutative
            OperatorType.DIVIDE: OperatorType.MULTIPLY,  # swaps with multiplication
            OperatorType.MULTIPLY: OperatorType.DIVIDE,  # swaps with division
            OperatorType.AND: OperatorType.AND,  # commutative
            OperatorType.PLUS: OperatorType.MINUS,  # swaps with subtraction
            OperatorType.MINUS: OperatorType.PLUS,  # swaps with addition
            OperatorType.LT: OperatorType.GT,  # swaps
            OperatorType.TRUE_DIVIDE: OperatorType.MULTIPLY,  # swaps with multiplication
            OperatorType.FLOOR_DIVIDE: OperatorType.MULTIPLY,  # swaps with multiplication
            OperatorType.MODULUS: OperatorType.MODULUS,  # no obvious swap, keep as is
        }

        if self in swap_map:
            return swap_map[self]
        else:
            raise NotImplementedError(f"Swap operation not implemented for {self}")

    def is_arithmetic(self) -> bool:
        """Check if the operator is an arithmetic operator.
        Defined as any operator that is neither comparison nor bitwise."""
        return not self.is_comparison_or_bitwise()


class FunctionType(str, Enum):
    """Base class for function type enums."""

    ...


class GenericFunctionType(FunctionType):
    UNKNOWN = "Unknown"  # to signify unknown functions, not in Rust
    FILL_NULL = "FillNull"  # Fill null values with specified value


class StringFunctionType(FunctionType):
    """Types of string functions."""

    CONCAT_HORIZONTAL = "ConcatHorizontal"
    CONCAT_VERTICAL = "ConcatVertical"
    CONTAINS = "Contains"
    COUNT_MATCHES = "CountMatches"
    ENDS_WITH = "EndsWith"
    EXTRACT = "Extract"
    EXTRACT_ALL = "ExtractAll"
    EXTRACT_GROUPS = "ExtractGroups"
    FIND = "Find"
    TO_INTEGER = "ToInteger"
    LEN_BYTES = "LenBytes"
    LEN_CHARS = "LenChars"
    LOWERCASE = "Lowercase"
    JSON_DECODE = "JsonDecode"
    JSON_PATH_MATCH = "JsonPathMatch"
    REPLACE = "Replace"
    NORMALIZE = "Normalize"
    REVERSE = "Reverse"
    PAD_START = "PadStart"
    PAD_END = "PadEnd"
    SLICE = "Slice"
    HEAD = "Head"
    TAIL = "Tail"
    HEX_ENCODE = "HexEncode"
    HEX_DECODE = "HexDecode"
    BASE64_ENCODE = "Base64Encode"
    BASE64_DECODE = "Base64Decode"
    STARTS_WITH = "StartsWith"
    STRIP_CHARS = "StripChars"
    STRIP_CHARS_START = "StripCharsStart"
    STRIP_CHARS_END = "StripCharsEnd"
    STRIP_PREFIX = "StripPrefix"
    STRIP_SUFFIX = "StripSuffix"
    SPLIT_EXACT = "SplitExact"
    SPLIT_N = "SplitN"
    STRPTIME = "Strptime"
    SPLIT = "Split"
    TO_DECIMAL = "ToDecimal"
    TITLECASE = "Titlecase"
    UPPERCASE = "Uppercase"
    ZFILL = "ZFill"
    CONTAINS_ANY = "ContainsAny"
    REPLACE_MANY = "ReplaceMany"
    EXTRACT_MANY = "ExtractMany"
    FIND_MANY = "FindMany"
    ESCAPE_REGEX = "EscapeRegex"


class ArrayFunctionType(FunctionType):
    """Types of array functions."""

    LENGTH = "Length"
    MIN = "Min"
    MAX = "Max"
    SUM = "Sum"
    TO_LIST = "ToList"
    UNIQUE = "Unique"
    N_UNIQUE = "NUnique"
    STD = "Std"
    VAR = "Var"
    MEDIAN = "Median"
    ANY = "Any"
    ALL = "All"
    SORT = "Sort"
    REVERSE = "Reverse"
    ARG_MIN = "ArgMin"
    ARG_MAX = "ArgMax"
    GET = "Get"
    JOIN = "Join"
    CONTAINS = "Contains"
    COUNT_MATCHES = "CountMatches"
    SHIFT = "Shift"
    EXPLODE = "Explode"
    CONCAT = "Concat"


class ListFunctionType(FunctionType):
    """Types of list functions."""

    CONCAT = "Concat"
    CONTAINS = "Contains"
    DROP_NULLS = "DropNulls"
    SAMPLE = "Sample"
    SLICE = "Slice"
    SHIFT = "Shift"
    GET = "Get"
    GATHER = "Gather"
    GATHER_EVERY = "GatherEvery"
    COUNT_MATCHES = "CountMatches"
    SUM = "Sum"
    LENGTH = "Length"
    MAX = "Max"
    MIN = "Min"
    MEAN = "Mean"
    MEDIAN = "Median"
    STD = "Std"
    VAR = "Var"
    ARG_MIN = "ArgMin"
    ARG_MAX = "ArgMax"
    DIFF = "Diff"
    SORT = "Sort"
    REVERSE = "Reverse"
    UNIQUE = "Unique"
    N_UNIQUE = "NUnique"
    SET_OPERATION = "SetOperation"
    ANY = "Any"
    ALL = "All"
    JOIN = "Join"
    TO_ARRAY = "ToArray"
    TO_STRUCT = "ToStruct"


class TemporalFunctionType(FunctionType):
    """Types of temporal functions."""

    MILLENNIUM = "Millennium"
    CENTURY = "Century"
    YEAR = "Year"
    IS_LEAP_YEAR = "IsLeapYear"
    ISO_YEAR = "IsoYear"
    QUARTER = "Quarter"
    MONTH = "Month"
    WEEK = "Week"
    WEEK_DAY = "WeekDay"
    DAY = "Day"
    ORDINAL_DAY = "OrdinalDay"
    TIME = "Time"
    DATE = "Date"
    DATETIME = "Datetime"
    DURATION = "Duration"
    HOUR = "Hour"
    MINUTE = "Minute"
    SECOND = "Second"
    MILLISECOND = "Millisecond"
    MICROSECOND = "Microsecond"
    NANOSECOND = "Nanosecond"
    TOTAL_DAYS = "TotalDays"
    TOTAL_HOURS = "TotalHours"
    TOTAL_MINUTES = "TotalMinutes"
    TOTAL_SECONDS = "TotalSeconds"
    TOTAL_MILLISECONDS = "TotalMilliseconds"
    TOTAL_MICROSECONDS = "TotalMicroseconds"
    TOTAL_NANOSECONDS = "TotalNanoseconds"
    TO_STRING = "ToString"
    CAST_TIME_UNIT = "CastTimeUnit"
    WITH_TIME_UNIT = "WithTimeUnit"
    CONVERT_TIME_ZONE = "ConvertTimeZone"
    TIMESTAMP = "TimeStamp"
    TRUNCATE = "Truncate"
    OFFSET_BY = "OffsetBy"
    MONTH_START = "MonthStart"
    MONTH_END = "MonthEnd"
    BASE_UTC_OFFSET = "BaseUtcOffset"
    DST_OFFSET = "DSTOffset"
    ROUND = "Round"
    REPLACE = "Replace"
    REPLACE_TIME_ZONE = "ReplaceTimeZone"
    COMBINE = "Combine"
    DATETIME_FUNCTION = "DatetimeFunction"


class StructFunctionType(FunctionType):
    """Types of struct functions."""

    FIELD_BY_INDEX = "FieldByIndex"
    FIELD_BY_NAME = "FieldByName"
    RENAME_FIELDS = "RenameFields"
    PREFIX_FIELDS = "PrefixFields"
    SUFFIX_FIELDS = "SuffixFields"
    JSON_ENCODE = "JsonEncode"
    WITH_FIELDS = "WithFields"
    MULTIPLE_FIELDS = "MultipleFields"


class BinaryFunctionType(FunctionType):
    """Types of binary functions."""

    CONTAINS = "Contains"
    STARTS_WITH = "StartsWith"
    ENDS_WITH = "EndsWith"
    HEX_DECODE = "HexDecode"
    HEX_ENCODE = "HexEncode"
    BASE64_DECODE = "Base64Decode"
    BASE64_ENCODE = "Base64Encode"
    SIZE = "Size"
    FROM_BUFFER = "FromBuffer"


class BitwiseFunctionType(FunctionType):
    """Types of bitwise functions."""

    COUNT_ONES = "CountOnes"
    COUNT_ZEROS = "CountZeros"
    LEADING_ONES = "LeadingOnes"
    LEADING_ZEROS = "LeadingZeros"
    TRAILING_ONES = "TrailingOnes"
    TRAILING_ZEROS = "TrailingZeros"
    AND = "And"
    OR = "Or"
    XOR = "Xor"


class BooleanFunctionType(FunctionType):
    """Types of boolean functions."""

    ANY = "Any"
    ALL = "All"
    IS_NULL = "IsNull"
    IS_NOT_NULL = "IsNotNull"
    IS_FINITE = "IsFinite"
    IS_INFINITE = "IsInfinite"
    IS_NAN = "IsNan"
    IS_NOT_NAN = "IsNotNan"
    IS_FIRST_DISTINCT = "IsFirstDistinct"
    IS_LAST_DISTINCT = "IsLastDistinct"
    IS_UNIQUE = "IsUnique"
    IS_DUPLICATED = "IsDuplicated"
    IS_BETWEEN = "IsBetween"
    IS_IN = "IsIn"
    ALL_HORIZONTAL = "AllHorizontal"
    ANY_HORIZONTAL = "AnyHorizontal"
    NOT = "Not"


class TrigonometricFunctionType(FunctionType):
    """Types of trigonometric functions."""

    COS = "Cos"
    COT = "Cot"
    SIN = "Sin"
    TAN = "Tan"
    ARCCOS = "ArcCos"
    ARCSIN = "ArcSin"
    ARCTAN = "ArcTan"
    COSH = "Cosh"
    SINH = "Sinh"
    TANH = "Tanh"
    ARCCOSH = "ArcCosh"
    ARCSINH = "ArcSinh"
    ARCTANH = "ArcTanh"
    DEGREES = "Degrees"
    RADIANS = "Radians"


# Note these cannot be pushed down
# class AggFunctionType(FunctionType):
#     """Types of aggregation functions."""
#     MIN = "Min"
#     MAX = "Max"
#     MEDIAN = "Median"
#     N_UNIQUE = "NUnique"
#     FIRST = "First"
#     LAST = "Last"
#     MEAN = "Mean"
#     IMPLODE = "Implode"
#     COUNT = "Count"
#     QUANTILE = "Quantile"
#     SUM = "Sum"
#     AGG_GROUPS = "AggGroups"
#     STD = "Std"
#     VAR = "Var"


# class WindowType(str, Enum):
#     """Types of window functions."""
#     OVER = "Over"
#     ROLLING = "Rolling"


class DataType(str, Enum):
    """
    Python representation of Rust's DataType enum.

    This is a string enum where each variant corresponds to a Rust DataType variant.
    Feature-gated variants from Rust are included but commented as such.
    """

    # Basic primitive types
    BOOLEAN = "Boolean"
    UINT8 = "UInt8"
    UINT16 = "UInt16"
    UINT32 = "UInt32"
    UINT64 = "UInt64"
    INT8 = "Int8"
    INT16 = "Int16"
    INT32 = "Int32"
    INT64 = "Int64"
    INT128 = "Int128"
    FLOAT32 = "Float32"
    FLOAT64 = "Float64"

    # Feature: dtype-decimal
    # Original: Decimal(Option<usize>, Option<usize>)
    DECIMAL = "Decimal"

    # String types
    STRING = "String"
    BINARY = "Binary"
    BINARY_OFFSET = "BinaryOffset"  # this doesnt seem to map to anything?

    # Date and time types
    DATE = "Date"
    # Original: Datetime(TimeUnit, Option<TimeZone>)
    DATETIME = "Datetime"
    # Original: Duration(TimeUnit)
    DURATION = "Duration"
    TIME = "Time"
    # Feature: dtype-array
    # Original: Array(Box<DataType>, usize)
    ARRAY = "Array"

    # Nested types
    # Original: List(Box<DataType>)
    LIST = "List"

    # Feature: object
    # Original: Object(&'static str)
    OBJECT = "Object"

    NULL = "Null"

    # Feature: dtype-categorical
    # Original: Categorical(Option<RevMapping>, CategoricalOrdering)
    CATEGORICAL = "Categorical"
    # Original: Enum(Option<RevMapping>, CategoricalOrdering)
    ENUM = "Enum"

    # Feature: dtype-struct
    # Original: Struct(Vec<Field>)
    STRUCT = "Struct"

    # Original: Unknown(UnknownKind)
    UNKNOWN = "Unknown"

    def get_class(self):
        dtype_map = {
            "Int8": pl.Int8,
            "Int16": pl.Int16,
            "Int32": pl.Int32,
            "Int64": pl.Int64,
            "UInt8": pl.UInt8,
            "UInt16": pl.UInt16,
            "UInt32": pl.UInt32,
            "UInt64": pl.UInt64,
            "Float32": pl.Float32,
            "Float64": pl.Float64,
            "Boolean": pl.Boolean,
            "Utf8": pl.Utf8,
            "String": pl.Utf8,
            "Date": pl.Date,
            "Datetime": pl.Datetime,
            "Time": pl.Time,
            "Categorical": pl.Categorical,
            "Enum": pl.Enum,
            "Duration": pl.Duration,
            "Null": pl.Null,
            "Object": pl.Object,
            "Decimal": pl.Decimal,
            "List": pl.List,
            "Struct": pl.Struct,
            "Unknown": pl.Unknown,
            "Array": pl.Array,
            "Binary": pl.Binary,
        }
        return dtype_map.get(self.value, None)

    @classmethod
    def from_polars_dtype(cls, dt: pl.DataType) -> "DataType":
        """Map a Polars dtype to our DataType enum using `dt.base_type()`."""
        bt = dt.base_type()
        match bt:
            case pl.Datetime:
                return cls.DATETIME
            case pl.Duration:
                return cls.DURATION
            case pl.Time:
                return cls.TIME
            case pl.Int64:
                return cls.INT64
            case pl.Int32:
                return cls.INT32
            case pl.UInt64:
                return cls.UINT64
            case pl.UInt32:
                return cls.UINT32
            case pl.Utf8:
                return cls.STRING
            case pl.Boolean:
                return cls.BOOLEAN
            case pl.Float64:
                return cls.FLOAT64
            case pl.Float32:
                return cls.FLOAT32
            case _:
                return cls.UNKNOWN


class TimeUnit(str, Enum):
    """Types of time units."""

    NANOSECONDS = "Nanoseconds"
    MICROSECONDS = "Microseconds"
    MILLISECONDS = "Milliseconds"

    def to_datetime_conversion(self) -> str:
        """Get the string representation of the time unit for datetime/duration construction."""
        match self:
            case TimeUnit.NANOSECONDS:
                return "ns"
            case TimeUnit.MICROSECONDS:
                return "us"
            case TimeUnit.MILLISECONDS:
                return "ms"

    @classmethod
    def from_string(cls, unit: Optional[str]) -> Optional["TimeUnit"]:
        """
        Map a Polars time unit string ("ns", "us", "ms") to a TimeUnit enum.

        Returns None if unit is None.
        """
        match unit:
            case "ns":
                return cls.NANOSECONDS
            case "us":
                return cls.MICROSECONDS
            case "ms":
                return cls.MILLISECONDS
            case _:
                return None


def get_function_enum(category_key: str, function_name: str) -> Optional[FunctionType]:
    """
    Get the appropriate function enum instance based on a category key and function name.

    Args:
        category_key: The category identifier (e.g., "Boolean", "StringExpr", "ArrayExpr")
        function_name: The name of the function to look up in PascalCase (e.g., "Contains")

    Returns:
        The matching enum instance if found, None otherwise

    Examples:
        >>> get_function_enum("StringExpr", "Contains")
        <StringFunctionType.CONTAINS: 'Contains'>
    """
    # Map from category keys in function_info to their corresponding enum classes
    category_mapping = {
        "Boolean": BooleanFunctionType,
        "StringExpr": StringFunctionType,
        "ArrayExpr": ArrayFunctionType,
        "ListExpr": ListFunctionType,
        "TemporalExpr": TemporalFunctionType,
        "StructExpr": StructFunctionType,
        "BinaryExpr": BinaryFunctionType,
        "Bitwise": BitwiseFunctionType,
        "Trigonometry": TrigonometricFunctionType,
    }

    # Get the enum class for the given category
    enum_class = category_mapping.get(category_key)
    if not enum_class:
        return None

    try:
        # Direct lookup of the function name in the enum class
        return enum_class(function_name)
    except ValueError:
        # Function name not found in the enum class
        return None
