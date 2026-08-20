"""LanceDB storage layer.

Handles connecting to LanceDB, creating/overwriting tables with explicit schema,
incremental batch writes, and conditional ANN index creation.
"""

import logging
import os
import shutil
from typing import List, Dict, Any, Optional

import lancedb
import numpy as np
import pyarrow as pa

from .schema import TABLE_SCHEMA, TABLE_NAME, EMBEDDING_DIM

logger = logging.getLogger(__name__)


class LanceDBStore:
    """Manages LanceDB connection, table lifecycle, and writes."""

    def __init__(self, output_dir: str, fresh: bool = False):
        self.output_dir = output_dir
        self.fresh = fresh
        self.db = None
        self.tables = {}
        self.total_rows_written = {}

    def connect(self):
        """Connect to LanceDB, optionally clearing the output directory first."""
        if self.fresh and os.path.exists(self.output_dir):
            logger.info("--fresh flag set: clearing output directory %s", self.output_dir)
            shutil.rmtree(self.output_dir)

        os.makedirs(self.output_dir, exist_ok=True)
        self.db = lancedb.connect(self.output_dir)
        logger.info("Connected to LanceDB at %s", self.output_dir)

    def create_table(self, table_name: str):
        """Create (or overwrite) the passages table with explicit PyArrow schema."""
        self.tables[table_name] = self.db.create_table(
            table_name,
            schema=TABLE_SCHEMA,
            mode="overwrite",
        )
        self.total_rows_written[table_name] = 0
        logger.info("Created table '%s' with explicit schema (mode=overwrite).", table_name)

    def add_batch(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray, table_name: str):
        """Write a batch of chunks + embeddings to the table.

        Args:
            chunks: List of chunk dicts with keys matching TABLE_SCHEMA field names.
            embeddings: Embedding matrix (N x 384) corresponding to chunks.
            table_name: The table to write to.
        """
        if not chunks:
            return

        table = self.tables.get(table_name)
        if table is None:
            raise RuntimeError(f"Table {table_name} not created. Call create_table() first.")

        # Build PyArrow arrays
        records = []
        for chunk, embedding in zip(chunks, embeddings):
            records.append({
                "chunk_id": chunk["chunk_id"],
                "source_doc_id": chunk["source_doc_id"],
                "chunk_strategy": chunk["chunk_strategy"],
                "language": chunk["language"],
                "text": chunk["text"],
                "vector": embedding.tolist(),
            })

        table.add(records)
        self.total_rows_written[table_name] += len(records)
        logger.debug("Wrote %d records to LanceDB table %s (total: %d).",
                      len(records), table_name, self.total_rows_written[table_name])

    def maybe_build_index(self, skip_index: bool = False):
        """Build ANN index if row count > 5000 and not in smoke-test mode."""
        built = {}
        for table_name, table in self.tables.items():
            if skip_index:
                logger.info("Index build skipped (smoke-test mode) for table %s.", table_name)
                built[table_name] = False
                continue

            if self.total_rows_written[table_name] <= 5000:
                logger.info(
                    "Skipping ANN index for %s: only %d rows (threshold: 5000). "
                    "Flat brute-force search is fast enough at this scale.",
                    table_name, self.total_rows_written[table_name],
                )
                built[table_name] = False
                continue

            logger.info("Building ANN index on %d rows for %s...", self.total_rows_written[table_name], table_name)
            table.create_index(
                metric="cosine",
                vector_column_name="vector",
            )
            logger.info("ANN index built successfully for %s.", table_name)
            built[table_name] = True
        return built

    def get_row_count(self, table_name: str) -> int:
        """Return the total rows written to a table."""
        return self.total_rows_written.get(table_name, 0)
