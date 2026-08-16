"""Guardrails for the ingestion pipeline.

Implements four guardrail stages:
1. Structural validation — UTF-8, non-empty, exact-duplicate removal via SHA-256
2. Content safety filter — regex/keyword denylist scan
3. Chunk quality gate — min/max token length, near-duplicate removal
4. Vector validation — NaN/zero-norm rejection
"""

import hashlib
import logging
import re
import unicodedata
from typing import List, Dict, Any, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Structural validation
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize text for deduplication: strip, lowercase, collapse whitespace."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def compute_text_hash(text: str) -> str:
    """SHA-256 hash of normalized text for exact deduplication."""
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_and_dedup_passages(
    passages: List[Dict[str, Any]],
    seen_hashes: Set[str],
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Validate passages structurally and deduplicate.

    Args:
        passages: List of passage dicts with at least 'text' key.
        seen_hashes: Mutable set of already-seen SHA-256 hashes (cross-language).

    Returns:
        (valid_passages, rejected_invalid_count, rejected_duplicate_count)
    """
    valid = []
    rejected_invalid = 0
    rejected_duplicate = 0

    for passage in passages:
        text = passage.get("text", "")

        # Check non-empty
        if not text or not text.strip():
            rejected_invalid += 1
            continue

        # Check valid UTF-8 (Python strings are always valid Unicode, but
        # guard against surrogate escapes from bad data)
        try:
            text.encode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            rejected_invalid += 1
            continue

        # Exact-duplicate check via SHA-256
        text_hash = compute_text_hash(text)
        if text_hash in seen_hashes:
            rejected_duplicate += 1
            continue

        seen_hashes.add(text_hash)
        valid.append(passage)

    return valid, rejected_invalid, rejected_duplicate


# ---------------------------------------------------------------------------
# 2. Content safety filter (denylist)
# ---------------------------------------------------------------------------

def load_denylist(path: str) -> List[re.Pattern]:
    """Load denylist phrases from a file and compile to regex patterns.

    Each non-empty, non-comment line becomes a case-insensitive regex pattern.
    """
    patterns = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Escape regex metacharacters, match as whole phrase
                pattern = re.compile(re.escape(line), re.IGNORECASE)
                patterns.append(pattern)
    except FileNotFoundError:
        logger.warning("Denylist file not found at %s — no content filtering applied.", path)
    return patterns


def filter_by_denylist(
    passages: List[Dict[str, Any]],
    denylist_patterns: List[re.Pattern],
) -> Tuple[List[Dict[str, Any]], int]:
    """Filter passages against denylist patterns.

    Returns:
        (surviving_passages, rejected_count)
    """
    if not denylist_patterns:
        return passages, 0

    surviving = []
    rejected = 0

    for passage in passages:
        text = passage.get("text", "")
        matched = False
        for pattern in denylist_patterns:
            if pattern.search(text):
                logger.debug("Denylist match in passage %s: pattern=%s",
                             passage.get("source_doc_id", "?"), pattern.pattern)
                matched = True
                break
        if matched:
            rejected += 1
        else:
            surviving.append(passage)

    return surviving, rejected


# ---------------------------------------------------------------------------
# 3. Chunk quality gate
# ---------------------------------------------------------------------------

def filter_chunks_by_length(
    chunks: List[Dict[str, Any]],
    tokenizer,
    min_tokens: int = 10,
    max_tokens: int = 512,
) -> Tuple[List[Dict[str, Any]], int]:
    """Drop chunks that are too short or too long.

    Returns:
        (surviving_chunks, rejected_count)
    """
    surviving = []
    rejected = 0

    for chunk in chunks:
        text = chunk.get("text", "")
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        if token_count < min_tokens or token_count > max_tokens:
            rejected += 1
        else:
            surviving.append(chunk)

    return surviving, rejected


def deduplicate_chunks_by_embedding(
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    threshold: float = 0.95,
) -> Tuple[List[Dict[str, Any]], np.ndarray, int]:
    """Remove near-duplicate chunks within the same source document via cosine similarity.

    Args:
        chunks: List of chunk dicts (must have 'source_doc_id').
        embeddings: Corresponding embedding matrix (N x dim).
        threshold: Cosine similarity threshold for near-duplicate detection.

    Returns:
        (unique_chunks, unique_embeddings, removed_count)
    """
    if len(chunks) <= 1:
        return chunks, embeddings, 0

    # Group by source_doc_id
    doc_groups: Dict[str, List[int]] = {}
    for i, chunk in enumerate(chunks):
        doc_id = chunk.get("source_doc_id", "")
        doc_groups.setdefault(doc_id, []).append(i)

    keep_indices = set()
    removed = 0

    for doc_id, indices in doc_groups.items():
        if len(indices) <= 1:
            keep_indices.update(indices)
            continue

        # Extract embeddings for this document's chunks
        doc_embeddings = embeddings[indices]

        # Normalize for cosine similarity
        norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # avoid division by zero
        normed = doc_embeddings / norms

        # Greedy dedup: keep first, skip if too similar to any kept
        kept_in_doc = [0]  # index within doc_embeddings
        for j in range(1, len(indices)):
            sims = normed[j] @ normed[kept_in_doc].T
            if np.max(sims) < threshold:
                kept_in_doc.append(j)
            else:
                removed += 1

        for local_idx in kept_in_doc:
            keep_indices.add(indices[local_idx])

    # Build filtered results maintaining order
    sorted_keep = sorted(keep_indices)
    unique_chunks = [chunks[i] for i in sorted_keep]
    unique_embeddings = embeddings[sorted_keep]

    return unique_chunks, unique_embeddings, removed


# ---------------------------------------------------------------------------
# 4. Vector validation
# ---------------------------------------------------------------------------

def validate_vectors(
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
) -> Tuple[List[Dict[str, Any]], np.ndarray, int]:
    """Reject embeddings containing NaN or with zero L2 norm.

    Returns:
        (valid_chunks, valid_embeddings, rejected_count)
    """
    valid_mask = np.ones(len(embeddings), dtype=bool)

    # Check for NaN
    nan_mask = np.any(np.isnan(embeddings), axis=1)
    valid_mask &= ~nan_mask

    # Check for zero L2 norm
    norms = np.linalg.norm(embeddings, axis=1)
    zero_mask = norms == 0.0
    valid_mask &= ~zero_mask

    rejected = int(np.sum(~valid_mask))
    if rejected > 0:
        logger.warning("Rejected %d vectors (NaN or zero norm).", rejected)

    valid_chunks = [chunks[i] for i in range(len(chunks)) if valid_mask[i]]
    valid_embeddings = embeddings[valid_mask]

    return valid_chunks, valid_embeddings, rejected
