"""CLI argument parsing and default configuration.

All pipeline parameters are controlled via argparse flags with sensible defaults.
No extra config-file dependency required.
"""

import argparse
import os


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline batch ingestion pipeline for multilingual RAG (LanceDB).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset loading
    parser.add_argument(
        "--languages",
        type=str,
        default=None,
        help="Comma-separated list of language configs to process (e.g. 'hi,ta'). "
             "Default: all available configs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max rows to load per language (for fast smoke testing).",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./lancedb_data",
        help="Directory to write the LanceDB table to.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clear the output directory before writing (idempotent fresh start).",
    )

    # Batching
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of dataset rows to process per batch.",
    )

    # Chunking — fixed-size
    parser.add_argument(
        "--fixed-chunk-tokens",
        type=int,
        default=256,
        help="Token count per chunk for fixed-size chunking (strategy A & C).",
    )
    parser.add_argument(
        "--fixed-chunk-overlap",
        type=int,
        default=51,
        help="Overlap tokens between consecutive fixed-size chunks (~20%% of 256).",
    )

    # Chunking — semantic
    parser.add_argument(
        "--semantic-similarity-threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold for semantic chunking (strategy B).",
    )

    # Chunk quality gate
    parser.add_argument(
        "--min-chunk-tokens",
        type=int,
        default=10,
        help="Minimum tokens per chunk; shorter chunks are dropped.",
    )
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=512,
        help="Maximum tokens per chunk; longer chunks are dropped.",
    )
    parser.add_argument(
        "--near-dup-threshold",
        type=float,
        default=0.95,
        help="Cosine similarity threshold for near-duplicate chunk removal.",
    )

    # Content safety
    parser.add_argument(
        "--denylist-path",
        type=str,
        default=os.path.join("config", "denylist.txt"),
        help="Path to the denylist file (one phrase per line).",
    )

    # Smoke test shortcut
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Shortcut: sets --limit 20 and skips ANN index building.",
    )

    return parser


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments and apply smoke-test overrides."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Smoke-test shortcut overrides
    if args.smoke_test:
        args.limit = 20

    # Parse languages list
    if args.languages:
        args.languages = [lang.strip() for lang in args.languages.split(",")]

    return args
