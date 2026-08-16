"""Embedding module for the ingestion pipeline.

Uses intfloat/multilingual-e5-small via SentenceTransformers with ONNX backend.
All passage texts are prefixed with "passage: " before embedding, per E5 convention.

IMPORTANT — runtime query embedding MUST use this exact model
(intfloat/multilingual-e5-small) + "query: " prefix to stay in the same vector space.
"""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# E5 prefix conventions
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "  # documented here for manifest; not used in ingestion

MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384


class E5Embedder:
    """Wrapper around SentenceTransformer for e5-small ONNX embedding."""

    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load the model and tokenizer. Call once before embedding."""
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer

        logger.info("Loading embedding model: %s (ONNX backend)", MODEL_NAME)
        # runtime query embedding MUST use this exact model + "query: " prefix
        # to stay in the same vector space
        self.model = SentenceTransformer(
            MODEL_NAME, 
            backend="onnx",
            model_kwargs={"providers": ["CPUExecutionProvider"]}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        logger.info("Model loaded successfully. Embedding dim: %d", EMBEDDING_DIM)

    def get_tokenizer(self):
        """Return the tokenizer (for use in chunking)."""
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        return self.tokenizer

    def embed_texts(self, texts: List[str], add_prefix: bool = True) -> np.ndarray:
        """Embed a list of texts with the E5 passage prefix.

        Args:
            texts: Raw text strings to embed.
            add_prefix: Whether to prepend "passage: " (default True for ingestion).

        Returns:
            Normalized embedding matrix of shape (N, 384).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        if add_prefix:
            texts = [PASSAGE_PREFIX + t for t in texts]

        # Batch the encoding
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # runtime query embedding MUST use this exact model + "query: " prefix
            # to stay in the same vector space
            emb = self.model.encode(
                batch,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_embeddings.append(emb)

        if not all_embeddings:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        return np.vstack(all_embeddings).astype(np.float32)

    def embed_sentences_for_semantic_chunking(self, sentences: List[str]) -> np.ndarray:
        """Embed individual sentences for semantic chunking.

        Uses the same model and prefix convention. This avoids loading a
        second embedding model — the single e5 model is reused.
        """
        return self.embed_texts(sentences, add_prefix=True)
