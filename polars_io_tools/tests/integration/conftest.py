from pathlib import Path

import pyarrow.fs as pa_fs
import pytest

from polars_io_tools.io_sources.util import _storage_options_for


def pytest_addoption(parser):
    parser.addoption(
        "--aws-profile",
        action="store",
        default=None,
        help="AWS profile to use for S3 integration tests",
    )
    parser.addoption(
        "--bucket",
        action="store",
        default="piot-polars-io-tools-tests",
        help="S3 bucket name to use for integration tests",
    )
    parser.addoption(
        "--databricks-token",
        action="store",
        default=None,
        help="Databricks token for integration tests",
    )
    parser.addoption(
        "--tickstore-group",
        action="store",
        default=None,
        help="TickStore group name for integration tests (e.g., 'Trading_piot')",
    )
    parser.addoption(
        "--tickstore-conn-str",
        action="store",
        default="DRIVER={ODBC Driver 17 for SQL Server};SERVER=RESEARCHSQL;Trusted_connection=yes;DATABASE=OneFeed",
        help="ODBC connection string for TickStore credential lookup",
    )
    parser.addoption("--clickhouse-url", action="store", default=None, help="URL for the ClickhouseDB http connection")
    parser.addoption("--clickhouse-user", action="store", default=None, help="Username for ClickhouseDB connection")
    parser.addoption("--clickhouse-password", action="store", default=None, help="Password for ClickhouseDB connection")


def pytest_configure(config):
    # Register marker to avoid PyTest warnings
    config.addinivalue_line("markers", "integration: mark test as integration")
    config.addinivalue_line(
        "markers",
        "aws_profile_required: mark test that requires --aws-profile",
    )
    config.addinivalue_line(
        "markers",
        "tickstore_required: mark test that requires --tickstore-group",
    )
    config.addinivalue_line(
        "markers",
        "clickhouse_required: mark test that requires --clickhouse-url, --clickhouse-user, and --clickhouse-password",
    )


@pytest.fixture(scope="session")
def aws_profile(pytestconfig):
    # May be None; only tests marked as requiring it will enforce presence
    return pytestconfig.getoption("--aws-profile") or None


@pytest.fixture(scope="session")
def databricks_token(pytestconfig):
    # May be None; only tests marked as requiring it will enforce presence
    return pytestconfig.getoption("--databricks-token") or None


@pytest.fixture(scope="session")
def tickstore_group(pytestconfig):
    # May be None; only tests marked as requiring it will enforce presence
    return pytestconfig.getoption("--tickstore-group") or None


@pytest.fixture(scope="session")
def tickstore_conn_str(pytestconfig):
    return pytestconfig.getoption("--tickstore-conn-str")


@pytest.fixture(scope="session")
def clickhouse_url(pytestconfig):
    return pytestconfig.getoption("--clickhouse-url") or None


@pytest.fixture(scope="session")
def clickhouse_user(pytestconfig):
    return pytestconfig.getoption("--clickhouse-user") or None


@pytest.fixture(scope="session")
def clickhouse_password(pytestconfig):
    return pytestconfig.getoption("--clickhouse-password") or None


@pytest.fixture
def aws_profile_via_env(aws_profile, monkeypatch):
    """Set AWS_PROFILE env var instead of passing it directly.

    This fixture sets the AWS_PROFILE environment variable and returns None,
    so tests can verify that _storage_options_for correctly falls back to
    the env var when aws_profile parameter is not provided.
    """
    if aws_profile:
        monkeypatch.setenv("AWS_PROFILE", aws_profile)
    return None


@pytest.fixture(scope="session")
def s3_bucket(pytestconfig):
    return pytestconfig.getoption("--bucket")


def _cleanup_s3_path(s3_path: str, aws_profile: str | None) -> None:
    """Clean up files at an S3 path."""
    pyarrow_opts = _storage_options_for(s3_path, aws_profile=aws_profile).pyarrow
    if not pyarrow_opts:
        return  # No credentials available
    fs = pa_fs.S3FileSystem(**pyarrow_opts)
    fs_path = s3_path.replace("s3://", "", 1)
    try:
        # Delete files individually instead of using delete_dir.
        # Some S3-compatible stores (e.g., on-prem) require Content-MD5 header
        # for batch DeleteObjects, which pyarrow doesn't provide. Individual
        # DeleteObject calls don't require this header.
        file_infos = fs.get_file_info(pa_fs.FileSelector(fs_path, recursive=True))
        for file_info in file_infos:
            if file_info.type == pa_fs.FileType.File:
                fs.delete_file(file_info.path)
    except FileNotFoundError:
        pass  # Directory doesn't exist yet, nothing to clean up


@pytest.fixture(scope="function")
def s3_root(s3_bucket, aws_profile, request):
    # Derive a unique root per-test-file, using the module filename
    test_file_stem = Path(request.fspath).stem
    s3_path = f"s3://{s3_bucket}/{test_file_stem}"

    # Clean up any existing data at the start of each test to ensure a clean slate.
    # This is necessary because CacheMode.REBUILD no longer deletes the entire cache.
    if aws_profile:
        _cleanup_s3_path(s3_path, aws_profile)
        # Also clean up the _env_var path used by env var tests
        _cleanup_s3_path(s3_path + "_env_var", aws_profile)

    return s3_path


def pytest_collection_modifyitems(config, items):
    # Mark all tests in this directory as integration
    for item in items:
        item.add_marker(pytest.mark.integration)

    # Skip tests marked as aws_profile_required if no profile provided
    profile = config.getoption("--aws-profile")
    if not profile:
        skip_marker = pytest.mark.skip(reason="requires --aws-profile for this test")
        for item in items:
            if "aws_profile_required" in item.keywords:
                item.add_marker(skip_marker)

    # Skip tests marked at databricks_auth_required if a user did not provide a host and token
    databricks_token = config.getoption("--databricks-token")
    if not databricks_token:
        skip_marker = pytest.mark.skip(reason="requires --databricks-token for this test")
        for item in items:
            if "databricks_auth_required" in item.keywords:
                item.add_marker(skip_marker)

    # Skip tests marked as tickstore_required if no group provided
    tickstore_group = config.getoption("--tickstore-group")
    if not tickstore_group:
        skip_marker = pytest.mark.skip(reason="requires --tickstore-group for this test")
        for item in items:
            if "tickstore_required" in item.keywords:
                item.add_marker(skip_marker)

    # Skip tests marked as clickhouse_required if url, user, or password not provided
    clickhouse_url = config.getoption("--clickhouse-url")
    clickhouse_user = config.getoption("--clickhouse-user")
    clickhouse_password = config.getoption("--clickhouse-password")
    if not (clickhouse_url and clickhouse_user and clickhouse_password):
        skip_marker = pytest.mark.skip(reason="requires --clickhouse-url, --clickhouse-user, and --clickhouse-password for this test")
        for item in items:
            if "clickhouse_required" in item.keywords:
                item.add_marker(skip_marker)
