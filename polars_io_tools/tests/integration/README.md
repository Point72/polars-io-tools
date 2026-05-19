**Overview**
- Integration tests validate real S3 interactions for `cache_parquet` and `delta_io` (since `polars`, `pyarrow` and `deltalake` involve similar, but slightly different, access patterns to hit s3. )
- Tests are opt-in and marked `integration`; some require an AWS profile.

**Requirements**
- AWS credentials configured for the selected profile (e.g., in `~/.aws/config` and `~/.aws/credentials`).
- An S3 bucket; defaults to `polars-io-tools-tests` but can be overridden.
- PyTest CLI options:
  - `--aws-profile <name>` for tests marked `aws_profile_required`.
  - `--bucket <bucket>` to set the target bucket (optional; default provided).

**Behavior**
- Each test module derives its S3 root prefix from the test file name (e.g., `test_s3_cache_parquet`), ensuring isolation.
- Cache Parquet writes to `s3://<bucket>/<file_stem>/daily/...` using a custom partition format.
- Delta IO writes to `s3://<bucket>/<file_stem>/delta_table` and scans with logical type translation.

**Run All Integration Tests**
- `pytest polars_io_tools/tests/integration -m integration --aws-profile myprofile --bucket polars-io-tools-tests`

**Run a Single Test**
- `pytest polars_io_tools/tests/integration/test_s3_cache_parquet.py -m integration --aws-profile myprofile --bucket polars-io-tools-tests`
- `pytest polars_io_tools/tests/integration/test_s3_delta_io.py -m integration --aws-profile myprofile --bucket polars-io-tools-tests`

**Notes**
- Tests marked `aws_profile_required` are skipped if `--aws-profile` is not provided.
- If you use a non-AWS S3-compatible endpoint, set the appropriate environment variables (e.g., `AWS_ENDPOINT_URL`) in addition to providing `--aws-profile`.
