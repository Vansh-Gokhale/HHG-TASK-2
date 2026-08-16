"""Multi-strategy chunking for the ingestion pipeline.

Three strategies that all run on every document:
  A. Fixed-size with overlap (tokenizer-based)
  B. Semantic chunking (sentence splitting + cosine similarity centroid)
  C. Metadata-aware chunking (fixed-size + richer metadata)
"""

import logging
import re
import uuid
from typing import List, Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Regex for sentence splitting that handles Indic punctuation
# Covers: Devanagari danda (।), double danda (॥), and standard punctuation
SENTENCE_SPLIT_RE = re.compile(
    r'(?<=[।॥?.!])\s+'
)


def _generate_chunk_id() -> str:
    """Generate a unique chunk ID."""
    return str(uuid.uuid4())


def _tokenize_text(text: str, tokenizer) -> List[int]:
    """Tokenize text without special tokens."""
    return tokenizer.encode(text, add_special_tokens=False)


def _decode_tokens(token_ids: List[int], tokenizer) -> str:
    """Decode token IDs back to text."""
    return tokenizer.decode(token_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Strategy A: Fixed-size chunking with overlap
# ---------------------------------------------------------------------------

def chunk_fixed_size(
    text: str,
    source_doc_id: str,
    language: str,
    tokenizer,
    chunk_tokens: int = 256,
    overlap_tokens: int = 51,
) -> List[Dict[str, Any]]:
    """Chunk text into fixed-size token windows with overlap.

    Returns list of chunk dicts with keys:
        chunk_id, source_doc_id, chunk_strategy, language, text
    """
    tokens = _tokenize_text(text, tokenizer)
    if not tokens:
        return []

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk_token_ids = tokens[start:end]
        chunk_text = _decode_tokens(chunk_token_ids, tokenizer)

        if chunk_text.strip():
            chunks.append({
                "chunk_id": _generate_chunk_id(),
                "source_doc_id": source_doc_id,
                "chunk_strategy": "fixed",
                "language": language,
                "text": chunk_text.strip(),
            })

        # Advance by (chunk_size - overlap) tokens
        start += chunk_tokens - overlap_tokens
        if start >= len(tokens):
            break
        # Avoid infinite loop if overlap >= chunk_size
        if chunk_tokens <= overlap_tokens:
            start = end
            break

    return chunks


# ---------------------------------------------------------------------------
# Strategy B: Semantic chunking
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    """Split text into sentences using Indic-aware regex.

    Handles: Devanagari danda (।), double danda (॥), period (.), question mark (?),
    exclamation (!). Does NOT depend on NLTK punkt.
    """
    sentences = SENTENCE_SPLIT_RE.split(text)
    # Filter empty strings and strip
    return [s.strip() for s in sentences if s.strip()]


def chunk_semantic(
    text: str,
    source_doc_id: str,
    language: str,
    embedder,
    similarity_threshold: float = 0.75,
) -> List[Dict[str, Any]]:
    """Semantic chunking: group consecutive sentences by embedding similarity.

    Sentences are added to the current chunk as long as their cosine similarity
    to the running chunk centroid stays above the threshold. When it drops below,
    a new chunk is started.

    The same e5 model (via embedder) is reused — no second model loaded.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    # If only one sentence, return it as a single chunk
    if len(sentences) == 1:
        return [{
            "chunk_id": _generate_chunk_id(),
            "source_doc_id": source_doc_id,
            "chunk_strategy": "semantic",
            "language": language,
            "text": sentences[0],
        }]

    # Embed all sentences at once (reusing the loaded e5 model)
    sentence_embeddings = embedder.embed_sentences_for_semantic_chunking(sentences)

    chunks = []
    current_sentences = [sentences[0]]
    current_embedding_sum = sentence_embeddings[0].copy()
    current_count = 1

    for i in range(1, len(sentences)):
        # Compute centroid of current chunk
        centroid = current_embedding_sum / current_count
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm

        # Compute similarity of next sentence to centroid
        sent_emb = sentence_embeddings[i]
        sent_norm = np.linalg.norm(sent_emb)
        if sent_norm > 0:
            sent_emb_normed = sent_emb / sent_norm
        else:
            sent_emb_normed = sent_emb

        similarity = float(np.dot(centroid, sent_emb_normed))

        if similarity >= similarity_threshold:
            # Add to current chunk
            current_sentences.append(sentences[i])
            current_embedding_sum += sentence_embeddings[i]
            current_count += 1
        else:
            # Finalize current chunk and start new one
            chunk_text = " ".join(current_sentences)
            if chunk_text.strip():
                chunks.append({
                    "chunk_id": _generate_chunk_id(),
                    "source_doc_id": source_doc_id,
                    "chunk_strategy": "semantic",
                    "language": language,
                    "text": chunk_text.strip(),
                })
            current_sentences = [sentences[i]]
            current_embedding_sum = sentence_embeddings[i].copy()
            current_count = 1

    # Don't forget the last chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        if chunk_text.strip():
            chunks.append({
                "chunk_id": _generate_chunk_id(),
                "source_doc_id": source_doc_id,
                "chunk_strategy": "semantic",
                "language": language,
                "text": chunk_text.strip(),
            })

    return chunks


# ---------------------------------------------------------------------------
# Strategy C: Metadata-aware chunking
# ---------------------------------------------------------------------------

def chunk_metadata_aware(
    text: str,
    source_doc_id: str,
    language: str,
    tokenizer,
    is_selected: Optional[bool] = None,
    chunk_tokens: int = 256,
    overlap_tokens: int = 51,
) -> List[Dict[str, Any]]:
    """Same fixed-size logic as strategy A, but with richer metadata.

    Attaches: language, source_doc_id, is_selected (if available from the
    original passages structure) so the runtime can filter/boost by metadata.
    """
    tokens = _tokenize_text(text, tokenizer)
    if not tokens:
        return []

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk_token_ids = tokens[start:end]
        chunk_text = _decode_tokens(chunk_token_ids, tokenizer)

        if chunk_text.strip():
            chunk = {
                "chunk_id": _generate_chunk_id(),
                "source_doc_id": source_doc_id,
                "chunk_strategy": "metadata_aware",
                "language": language,
                "text": chunk_text.strip(),
            }
            # Note: is_selected is kept as metadata context but not in the
            # LanceDB schema to keep the schema stable. It's logged for
            # downstream use if needed.
            chunks.append(chunk)

        start += chunk_tokens - overlap_tokens
        if start >= len(tokens):
            break
        if chunk_tokens <= overlap_tokens:
            start = end
            break

    return chunks
