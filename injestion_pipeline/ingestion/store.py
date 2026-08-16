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
        self.table = None
        self.total_rows_written = 0

    def connect(self):
        """Connect to LanceDB, optionally clearing the output directory first."""
        if self.fresh and os.path.exists(self.output_dir):
            logger.info("--fresh flag set: clearing output directory %s", self.output_dir)
            shutil.rmtree(self.output_dir)

        os.makedirs(self.output_dir, exist_ok=True)
        self.db = lancedb.connect(self.output_dir)
        logger.info("Connected to LanceDB at %s", self.output_dir)

    def create_table(self):
        """Create (or overwrite) the passages table with explicit PyArrow schema."""
        self.table = self.db.create_table(
            TABLE_NAME,
            schema=TABLE_SCHEMA,
            mode="overwrite",
        )
        self.total_rows_written = 0
        logger.info("Created table '%s' with explicit schema (mode=overwrite).", TABLE_NAME)

    def add_batch(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """Write a batch of chunks + embeddings to the table.

        Args:
            chunks: List of chunk dicts with keys matching TABLE_SCHEMA field names.
            embeddings: Embedding matrix (N x 384) corresponding to chunks.
        """
        if not chunks:
            return

        if self.table is None:
            raise RuntimeError("Table not created. Call create_table() first.")

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

        self.table.add(records)
        self.total_rows_written += len(records)
        logger.debug("Wrote %d records to LanceDB (total: %d).",
                      len(records), self.total_rows_written)

    def maybe_build_index(self, skip_index: bool = False):
        """Build ANN index if row count > 5000 and not in smoke-test mode.

        LanceDB's IVF_PQ index needs meaningful row counts per partition;
        below ~5000 rows, flat brute-force search is fast enough and avoids
        a common failure mode at hackathon scale.
        """
        if skip_index:
            logger.info("Index build skipped (smoke-test mode).")
            return False

        if self.total_rows_written <= 5000:
            logger.info(
                "Skipping ANN index: only %d rows (threshold: 5000). "
                "Flat brute-force search is fast enough at this scale.",
                self.total_rows_written,
            )
            return False

        logger.info("Building ANN index on %d rows...", self.total_rows_written)
        self.table.create_index(
            metric="cosine",
            vector_column_name="vector",
        )
        logger.info("ANN index built successfully.")
        return True

    def get_row_count(self) -> int:
        """Return the total rows written."""
        return self.total_rows_written
