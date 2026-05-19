"""
Integration tests for the lazy Polars SQL reader.

This module contains integration tests that require actual database connections
to SQL Server databases. These tests are based on the examples in lazy_polars_sql.py
and are intended to verify that the SQL reader works correctly against real databases.

Note: These tests are excluded from the regular test suite by default and must be
run explicitly when database access is available.
"""
