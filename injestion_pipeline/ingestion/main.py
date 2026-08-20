"""Entry point for the ingestion pipeline.

Usage:
    python -m ingestion.main [--flags]
    python ingestion/main.py [--flags]

Examples:
    # Full ingestion (all languages)
    python -m ingestion.main --output-dir ./lancedb_data

    # Smoke test (20 rows, no index)
    python -m ingestion.main --smoke-test

    # Specific languages with limit
    python -m ingestion.main --languages hi,ta --limit 100

    # Fresh start (clear output dir first)
    python -m ingestion.main --fresh --output-dir ./lancedb_data
"""

import os
# Bypass TensorFlow/Keras import errors in transformers library
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

# Set HuggingFace Cache to current directory to avoid filling C drive
local_cache = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "hf_cache"))
os.environ["HF_DATASETS_CACHE"] = local_cache
os.environ["HUGGINGFACE_HUB_CACHE"] = local_cache

import logging
import sys

from .config import parse_args
from .pipeline import run_pipeline


def setup_logging():
    """Configure logging with timestamps at INFO level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main(argv=None):
    """Parse CLI args and run the ingestion pipeline."""
    setup_logging()
    logger = logging.getLogger(__name__)

    args = parse_args(argv)
    logger.info("Starting ingestion pipeline with args: %s", vars(args))

    try:
        run_pipeline(args)
        logger.info("Pipeline completed successfully.")
    except Exception:
        logger.exception("Pipeline failed with an error.")
        sys.exit(1)


if __name__ == "__main__":
    main()
