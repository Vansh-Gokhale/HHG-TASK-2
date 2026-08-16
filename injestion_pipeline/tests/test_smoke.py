"""Smoke test for the ingestion pipeline.

Runs main.py --smoke-test against the dataset with --limit 20,
then validates:
  - Output LanceDB directory exists
  - Table has rows
  - ingestion_report.json and manifest.json were written
  - Every row's vector field has exactly 384 floats with no NaNs
"""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest


# Use a temporary output directory for tests
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "test_lancedb_output")


@pytest.fixture(scope="module")
def run_smoke_test():
    """Run the pipeline in smoke-test mode and return the output dir."""
    # Clean up any previous test output
    if os.path.exists(TEST_OUTPUT_DIR):
        shutil.rmtree(TEST_OUTPUT_DIR)

    # Run the pipeline
    result = subprocess.run(
        [
            sys.executable, "-m", "ingestion.main",
            "--smoke-test",
            "--output-dir", TEST_OUTPUT_DIR,
            "--fresh",
        ],
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        capture_output=True,
        text=True,
        timeout=600,  # 10 minute timeout
    )

    print("STDOUT:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)

    assert result.returncode == 0, (
        f"Pipeline failed with return code {result.returncode}.\n"
        f"STDERR: {result.stderr[-1000:]}"
    )

    yield TEST_OUTPUT_DIR

    # Cleanup after tests
    # (Commented out to allow inspection of test output)
    # if os.path.exists(TEST_OUTPUT_DIR):
    #     shutil.rmtree(TEST_OUTPUT_DIR)


def test_output_dir_exists(run_smoke_test):
    """Output LanceDB directory should exist."""
    assert os.path.isdir(run_smoke_test), (
        f"Output directory {run_smoke_test} does not exist"
    )


def test_table_has_rows(run_smoke_test):
    """The LanceDB table should have at least one row."""
    import lancedb

    db = lancedb.connect(run_smoke_test)
    table_names = db.table_names()
    assert "passages" in table_names, (
        f"Table 'passages' not found. Available tables: {table_names}"
    )

    table = db.open_table("passages")
    df = table.to_pandas()
    assert len(df) > 0, "Table 'passages' has no rows"
    print(f"Table has {len(df)} rows")


def test_ingestion_report_exists(run_smoke_test):
    """ingestion_report.json should be written."""
    report_path = os.path.join(run_smoke_test, "ingestion_report.json")
    assert os.path.isfile(report_path), f"ingestion_report.json not found at {report_path}"

    with open(report_path, "r") as f:
        report = json.load(f)

    # Check expected keys
    assert "embedding_model" in report
    assert report["embedding_model"] == "intfloat/multilingual-e5-small"
    assert "embedding_dim" in report
    assert report["embedding_dim"] == 384
    assert "languages_processed" in report
    assert "stage_counts" in report
    assert "wall_clock_seconds_by_stage" in report
    print(f"Report: {json.dumps(report, indent=2)}")


def test_manifest_exists(run_smoke_test):
    """manifest.json should be written."""
    manifest_path = os.path.join(run_smoke_test, "manifest.json")
    assert os.path.isfile(manifest_path), f"manifest.json not found at {manifest_path}"

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    assert manifest["embedding_model"] == "intfloat/multilingual-e5-small"
    assert manifest["embedding_dim"] == 384
    assert manifest["passage_prefix"] == "passage: "
    assert manifest["query_prefix"] == "query: "
    print(f"Manifest: {json.dumps(manifest, indent=2)}")


def test_vectors_are_valid(run_smoke_test):
    """Every row's vector field should have exactly 384 floats with no NaNs."""
    import lancedb

    db = lancedb.connect(run_smoke_test)
    table = db.open_table("passages")
    df = table.to_pandas()

    for idx, row in df.iterrows():
        vector = row["vector"]
        vec_array = np.array(vector, dtype=np.float32)

        assert vec_array.shape == (384,), (
            f"Row {idx}: vector has shape {vec_array.shape}, expected (384,)"
        )

        assert not np.any(np.isnan(vec_array)), (
            f"Row {idx}: vector contains NaN values"
        )

        norm = np.linalg.norm(vec_array)
        assert norm > 0, f"Row {idx}: vector has zero L2 norm"

    print(f"All {len(df)} vectors validated: 384 dims, no NaNs, non-zero norm")
