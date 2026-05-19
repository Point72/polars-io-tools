"""
Integration test for Delta IO on S3.

Requires passing --aws-profile and optionally --bucket via pytest CLI.
Bucket defaults to "polars-io-tools-tests". Root path derives from file name.
"""

import pytest

import polars_io_tools as cpl  # noqa
from polars_io_tools.tests.io_sources.test_delta_io import _run_delta_io_roundtrip


@pytest.mark.integration
@pytest.mark.aws_profile_required
def test_s3_delta_io_roundtrip(s3_root, aws_profile):
    _run_delta_io_roundtrip(s3_root, aws_profile)


@pytest.mark.integration
@pytest.mark.aws_profile_required
def test_s3_delta_io_with_env_var(s3_root, aws_profile_via_env):
    """Test that delta IO works when AWS_PROFILE is set via env var.

    This verifies that _storage_options_for correctly uses the AWS_PROFILE
    environment variable when aws_profile parameter is None. This single test
    covers the env var fallback for all S3 operations (delta, cache_parquet, etc.)
    since they all use the same _storage_options_for function.
    """
    _run_delta_io_roundtrip(s3_root + "_env_var", aws_profile_via_env)
