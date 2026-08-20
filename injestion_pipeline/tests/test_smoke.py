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

    # Run the pipeline with explicit languages hi,ta and limit
    test_script = f"""
import datasets

def mock_load_dataset(path, name=None, *args, **kwargs):
    def gen():
        lang = None
        if "data_files" in kwargs:
            data_file = str(kwargs["data_files"])
            if "hin" in data_file or "hi" in data_file:
                lang = "hi"
            elif "tam" in data_file or "ta" in data_file:
                lang = "ta"
        elif name and name != "default":
            lang = name

        for i in range(15):
            if lang is None or lang == "hi":
                yield {{
                    "target_lang": "hi",
                    "source_lang": "en",
                    "query_id": i,
                    "passages": {{
                        "Translated_passages": ["यह एक बहुत ही सुंदर और लंबा हिंदी वाक्य है जिसका उपयोग हम परीक्षण के लिए कर रहे हैं ताकि टोकन की संख्या न्यूनतम सीमा से अधिक हो " + str(i)] * 3,
                        "is_selected": [1, 0, 0]
                    }}
                }}
        for i in range(15):
            if lang is None or lang == "ta":
                yield {{
                    "target_lang": "ta",
                    "source_lang": "en",
                    "query_id": i + 100,
                    "passages": {{
                        "Translated_passages": ["இது ஒரு மிக நீண்ட தமிழ் வாக்கியம் ஆகும், இது சோதனை நோக்கங்களுக்காக பயன்படுத்தப்படுகிறது, இதனால் டோக்கன் எண்ணிக்கை வரம்பை விட அதிகமாக இருக்கும் " + str(i)] * 3,
                        "is_selected": [1, 0, 0]
                    }}
                }}
    return gen()

datasets.load_dataset = mock_load_dataset
datasets.get_dataset_config_names = lambda *args, **kwargs: ["default"]

from ingestion.main import main
main([
    "--languages", "hi,ta",
    "--limit", "10",
    "--output-dir", "{TEST_OUTPUT_DIR}",
    "--fresh"
])
"""
    result = subprocess.run(
        [
            sys.executable, "-c", test_script
        ],
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        capture_output=True,
        text=True,
        timeout=120,
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
    """The LanceDB tables should exist and have rows."""
    import lancedb

    db = lancedb.connect(run_smoke_test)
    table_names = db.table_names()
    assert len(table_names) > 0, f"No tables found. Available tables: {table_names}"
    
    total_rows = 0
    for name in table_names:
        table = db.open_table(name)
        df = table.to_pandas()
        total_rows += len(df)
        print(f"Table '{name}' has {len(df)} rows")
    
    assert total_rows > 0, "No tables have rows"


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
    """Every row's vector field should have exactly 384 floats with no NaNs across tables."""
    import lancedb

    db = lancedb.connect(run_smoke_test)
    table_names = db.table_names()
    assert len(table_names) > 0, "No tables to validate"

    total_validated = 0
    for name in table_names:
        table = db.open_table(name)
        df = table.to_pandas()

        for idx, row in df.iterrows():
            vector = row["vector"]
            vec_array = np.array(vector, dtype=np.float32)

            assert vec_array.shape == (384,), (
                f"Table {name}, row {idx}: vector has shape {vec_array.shape}, expected (384,)"
            )

            assert not np.any(np.isnan(vec_array)), (
                f"Table {name}, row {idx}: vector contains NaN values"
            )

            norm = np.linalg.norm(vec_array)
            assert norm > 0, f"Table {name}, row {idx}: vector has zero L2 norm"
            total_validated += 1

    print(f"All {total_validated} vectors validated: 384 dims, no NaNs, non-zero norm")
