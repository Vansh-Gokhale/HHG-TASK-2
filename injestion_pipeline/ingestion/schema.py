"""LanceDB / PyArrow schema definition for the ingestion pipeline.

This schema is used for both table creation and validation.
The vector dimension (384) matches intfloat/multilingual-e5-small output.
"""

import pyarrow as pa

# E5-small embedding dimension
EMBEDDING_DIM = 384

# Valid chunk strategy labels
CHUNK_STRATEGIES = ("fixed", "semantic", "metadata_aware")

# Explicit PyArrow schema — never let LanceDB infer it
TABLE_SCHEMA = pa.schema([
    pa.field("chunk_id", pa.string()),
    pa.field("source_doc_id", pa.string()),
    pa.field("chunk_strategy", pa.string()),   # "fixed" | "semantic" | "metadata_aware"
    pa.field("language", pa.string()),
    pa.field("text", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
])

TABLE_NAME = "passages"
