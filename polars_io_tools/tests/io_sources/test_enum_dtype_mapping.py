import polars as pl

from polars_io_tools.io_sources.enum import DataType, TimeUnit


def test_edata_type_from_polars_primitives():
    assert DataType.from_polars_dtype(pl.Int64) == DataType.INT64
    assert DataType.from_polars_dtype(pl.Int32) == DataType.INT32
    assert DataType.from_polars_dtype(pl.UInt64) == DataType.UINT64
    assert DataType.from_polars_dtype(pl.UInt32) == DataType.UINT32
    assert DataType.from_polars_dtype(pl.Float64) == DataType.FLOAT64
    assert DataType.from_polars_dtype(pl.Float32) == DataType.FLOAT32
    assert DataType.from_polars_dtype(pl.Boolean) == DataType.BOOLEAN
    assert DataType.from_polars_dtype(pl.Utf8) == DataType.STRING


def test_edata_type_from_polars_temporal():
    assert DataType.from_polars_dtype(pl.Time) == DataType.TIME
    assert DataType.from_polars_dtype(pl.Datetime("ns")) == DataType.DATETIME
    assert DataType.from_polars_dtype(pl.Duration("ms")) == DataType.DURATION


def test_polars_base_class_from_edata_type_primitives():
    assert DataType.INT64.get_class() == pl.Int64
    assert DataType.INT32.get_class() == pl.Int32
    assert DataType.UINT64.get_class() == pl.UInt64
    assert DataType.UINT32.get_class() == pl.UInt32
    assert DataType.FLOAT64.get_class() == pl.Float64
    assert DataType.FLOAT32.get_class() == pl.Float32
    assert DataType.BOOLEAN.get_class() == pl.Boolean
    assert DataType.STRING.get_class() == pl.Utf8


def test_polars_unit_application_for_temporal_classes():
    # Check base classes
    assert DataType.DATETIME.get_class() == pl.Datetime
    assert DataType.DURATION.get_class() == pl.Duration
    assert DataType.TIME.get_class() == pl.Time
    # Units are handled by TimeUnit conversion strings when constructing concrete dtypes
    assert str(pl.Datetime(TimeUnit.NANOSECONDS.to_datetime_conversion())) == str(pl.Datetime("ns"))
    assert str(pl.Datetime(TimeUnit.MICROSECONDS.to_datetime_conversion())) == str(pl.Datetime("us"))
    assert str(pl.Datetime(TimeUnit.MILLISECONDS.to_datetime_conversion())) == str(pl.Datetime("ms"))
    assert str(pl.Duration(TimeUnit.NANOSECONDS.to_datetime_conversion())) == str(pl.Duration("ns"))
    assert str(pl.Duration(TimeUnit.MICROSECONDS.to_datetime_conversion())) == str(pl.Duration("us"))
    assert str(pl.Duration(TimeUnit.MILLISECONDS.to_datetime_conversion())) == str(pl.Duration("ms"))


def test_timeunit_from_string_roundtrip():
    assert TimeUnit.from_string("ns") == TimeUnit.NANOSECONDS
    assert TimeUnit.from_string("us") == TimeUnit.MICROSECONDS
    assert TimeUnit.from_string("ms") == TimeUnit.MILLISECONDS
    assert TimeUnit.from_string(None) is None
