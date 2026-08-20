"""Pipeline orchestrator — runs stages 1-8 of the ingestion pipeline.

Stages:
  1. Load dataset
  2. Extract & validate passages (structural guardrail)
  3. Content safety filter (denylist guardrail)
  4. Multi-strategy chunking (fixed, semantic, metadata-aware)
  5. Chunk quality gate guardrail
  6. Embed chunks
  7. Vector validation guardrail
  8. Write to LanceDB
"""

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

from .chunkers import chunk_fixed_size, chunk_metadata_aware, chunk_semantic
from .embedder import E5Embedder, EMBEDDING_DIM, MODEL_NAME, PASSAGE_PREFIX, QUERY_PREFIX
from .guardrails import (
    filter_by_denylist,
    filter_chunks_by_length,
    deduplicate_chunks_by_embedding,
    load_denylist,
    validate_and_dedup_passages,
    validate_vectors,
)
from .schema import LANGUAGE_MAP, to_locale_table_name, normalize_lang_code
from .store import LanceDBStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Passage extraction (defensive: dict-of-lists AND list-of-dicts)
# ---------------------------------------------------------------------------

def extract_passages_from_row(
    row: Dict[str, Any],
    language: str,
) -> List[Dict[str, Any]]:
    """Extract passage texts from a single dataset row.

    Handles both dict-of-lists and list-of-dicts shapes for the 'passages'
    field. Fails loudly with a clear error if neither shape matches.

    Returns a list of dicts with keys: text, source_doc_id, language, is_selected
    """
    passages_raw = row.get("passages")
    query_id = row.get("query_id", 0)
    locale_lang = to_locale_table_name(language)

    if passages_raw is None:
        raise ValueError(
            f"Row has no 'passages' field. Available keys: {list(row.keys())}"
        )

    extracted = []

    if isinstance(passages_raw, dict):
        # Dict-of-lists shape: {English_passages: [...], Translated_passages: [...], is_selected: [...]}
        # Also handle the MS MARCO original shape: {passage_text: [...], is_selected: [...], url: [...]}
        passage_texts = None
        is_selected_list = None

        # Try Translated_passages first (MSMARCO-XI specific)
        if "Translated_passages" in passages_raw:
            passage_texts = passages_raw["Translated_passages"]
        elif "passage_text" in passages_raw:
            passage_texts = passages_raw["passage_text"]
        else:
            raise ValueError(
                f"Dict-of-lists passages has no recognized text key. "
                f"Keys found: {list(passages_raw.keys())}. "
                f"Expected 'Translated_passages' or 'passage_text'."
            )

        if "is_selected" in passages_raw:
            is_selected_list = passages_raw["is_selected"]

        if not isinstance(passage_texts, list):
            raise ValueError(
                f"Expected passages text field to be a list, got {type(passage_texts).__name__}"
            )

        for idx, text in enumerate(passage_texts):
            is_sel = None
            if is_selected_list and idx < len(is_selected_list):
                is_sel = bool(is_selected_list[idx])

            extracted.append({
                "text": str(text) if text else "",
                "source_doc_id": f"{language}_{query_id}_{idx}",
                "language": locale_lang,
                "is_selected": is_sel,
            })

    elif isinstance(passages_raw, list):
        # List-of-dicts shape: [{passage_text: ..., is_selected: ..., url: ...}, ...]
        for idx, passage_dict in enumerate(passages_raw):
            if not isinstance(passage_dict, dict):
                raise ValueError(
                    f"List-of-dicts passages: element {idx} is {type(passage_dict).__name__}, "
                    f"expected dict."
                )

            text = (
                passage_dict.get("Translated_passages")
                or passage_dict.get("passage_text")
                or passage_dict.get("text")
                or ""
            )
            is_sel = passage_dict.get("is_selected")
            if is_sel is not None:
                is_sel = bool(is_sel)

            extracted.append({
                "text": str(text) if text else "",
                "source_doc_id": f"{language}_{query_id}_{idx}",
                "language": locale_lang,
                "is_selected": is_sel,
            })
    else:
        raise ValueError(
            f"Passages field is neither dict nor list — got {type(passages_raw).__name__}. "
            f"Cannot extract passage texts. This is a dataset format error."
        )

    return extracted


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args):
    """Run the full ingestion pipeline (stages 1-8)."""
    from datasets import load_dataset

    timings = {}
    stage_counts = {
        "total_rows_loaded": 0,
        "total_passages_extracted": 0,
        "after_structural_validation": 0,
        "after_safety_filter": 0,
        "total_chunks_produced": 0,
        "after_chunk_quality_gate": 0,
        "after_vector_validation": 0,
        "final_rows_written": 0,
    }
    chunk_counts_by_strategy = {"fixed": 0, "semantic": 0, "metadata_aware": 0}
    rejected_counts = {
        "structural_invalid": 0,
        "structural_duplicate": 0,
        "safety_filter": 0,
        "chunk_too_short_or_long": 0,
        "chunk_near_duplicate": 0,
        "vector_invalid": 0,
    }

    # -----------------------------------------------------------------------
    # Stage 1: Load dataset
    # -----------------------------------------------------------------------
    t0 = time.time()
    logger.info("STAGE 1: Loading dataset configurations")
    logger.info("=" * 60)

    from datasets import get_dataset_config_names, load_dataset
    try:
        all_languages = get_dataset_config_names("ai4bharat/MSMARCO-XI")
    except Exception as e:
        logger.error(f"Could not fetch configs: {e}")
        all_languages = list(LANGUAGE_MAP.keys())
    
    if all_languages == ["default"]:
        all_languages = list(LANGUAGE_MAP.keys())

    # Filter to requested languages (accepting both 'ta' and 'ta-IN')
    if args.languages:
        normalized_requested = [normalize_lang_code(l) for l in args.languages]
        languages_to_process = [
            lang for lang in normalized_requested if lang in all_languages or lang in LANGUAGE_MAP
        ]
        missing = set(normalized_requested) - set(languages_to_process)
        if missing:
            logger.warning("Requested languages not found in known configs: %s", missing)
    else:
        languages_to_process = [l for l in all_languages if l in LANGUAGE_MAP] or list(LANGUAGE_MAP.keys())

    logger.info("Languages to process: %s (table names: %s)",
                languages_to_process, [to_locale_table_name(l) for l in languages_to_process])
    timings["load"] = time.time() - t0

    # -----------------------------------------------------------------------
    # Initialize embedder and store
    # -----------------------------------------------------------------------
    t0 = time.time()
    logger.info("Initializing embedding model...")
    embedder = E5Embedder(batch_size=32)
    embedder.load()
    tokenizer = embedder.get_tokenizer()
    timings["model_load"] = time.time() - t0

    store = LanceDBStore(output_dir=args.output_dir, fresh=args.fresh)
    store.connect()
    # Load denylist
    denylist_patterns = load_denylist(args.denylist_path)
    logger.info("Loaded %d denylist patterns.", len(denylist_patterns))

    # Shared dedup hash set across all languages
    seen_hashes: Set[str] = set()

    # Initialize tables with standard locale format (e.g. 'ta-IN')
    for lang in languages_to_process:
        locale_name = to_locale_table_name(lang)
        store.create_table(locale_name)

    # -----------------------------------------------------------------------
    # Process dataset (Streaming)
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Loading dataset in streaming mode...")
    
    # 3-letter prefix mapping for ai4bharat/MSMARCO-XI train files
    PARQUET_FILE_MAP = {
        "as": "train/asmtrain.parquet",
        "bn": "train/bentrain.parquet",
        "gu": "train/gujtrain.parquet",
        "hi": "train/hintrain.parquet",
        "kn": "train/kantrain.parquet",
        "ml": "train/maltrain.parquet",
        "mr": "train/martrain.parquet",
        "ne": "train/neptrain.parquet",
        "or": "train/oritrain.parquet",
        "pa": "train/pantrain.parquet",
        "sa": "train/santrain.parquet",
        "ta": "train/tamtrain.parquet",
        "ur": "train/urdtrain.parquet",
    }

    # Helper to stream rows for each language
    def _stream_language_rows(lang_code):
        parquet_rel = PARQUET_FILE_MAP.get(lang_code)
        if parquet_rel:
            try:
                ds = load_dataset(
                    "parquet",
                    data_files={"train": f"https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/{parquet_rel}"},
                    split="train",
                    streaming=True,
                )
                yielded = 0
                for r in ds:
                    r["target_lang"] = lang_code
                    yield r
                    yielded += 1
                if yielded > 0:
                    return
            except Exception as ex:
                logger.debug("Direct parquet stream for %s failed (%s), falling back to default stream", lang_code, ex)
        # Fallback to default split
        try:
            ds_stream = load_dataset("ai4bharat/MSMARCO-XI", "default", split="train", streaming=True)
            for r in ds_stream:
                row_lang = r.get("target_lang") or r.get("source_lang")
                if row_lang == lang_code:
                    yield r
        except Exception as ex:
            logger.error("Failed to load stream for %s: %s", lang_code, ex)

    rows_processed = {lang: 0 for lang in languages_to_process}
    batch_rows_by_lang = {lang: [] for lang in languages_to_process}
    
    # We will use a shared progress bar for all requested languages
    pbar_total = sum(args.limit for lang in languages_to_process) if args.limit else None
    pbar = tqdm(total=pbar_total, desc="Rows processed")

    t_extract = 0.0
    t_validate = 0.0
    t_safety = 0.0
    t_chunk = 0.0
    t_quality = 0.0
    t_embed = 0.0
    t_vecval = 0.0
    t_write = 0.0

    def _process_batch(lang, batch_rows):
        nonlocal t_extract, t_validate, t_safety, t_chunk, t_quality, t_embed, t_vecval, t_write

        # -----------------------------------------------------------
        # Stage 2: Extract passages
        # -----------------------------------------------------------
        _t = time.time()
        all_passages = []
        for i in range(len(batch_rows)):
            row = batch_rows[i]
            try:
                passages = extract_passages_from_row(row, lang)
                all_passages.extend(passages)
            except ValueError as e:
                logger.error("Failed to extract passages: %s", e)
                continue

        stage_counts["total_passages_extracted"] += len(all_passages)
        t_extract += time.time() - _t

        if not all_passages:
            return

        # -----------------------------------------------------------
        # Stage 3: Structural validation guardrail
        # -----------------------------------------------------------
        _t = time.time()
        valid_passages, n_invalid, n_dup = validate_and_dedup_passages(
            all_passages, seen_hashes
        )
        rejected_counts["structural_invalid"] += n_invalid
        rejected_counts["structural_duplicate"] += n_dup
        stage_counts["after_structural_validation"] += len(valid_passages)
        t_validate += time.time() - _t

        if not valid_passages:
            return

        # -----------------------------------------------------------
        # Stage 4: Content safety filter guardrail
        # -----------------------------------------------------------
        _t = time.time()
        safe_passages, n_filtered = filter_by_denylist(
            valid_passages, denylist_patterns
        )
        rejected_counts["safety_filter"] += n_filtered
        stage_counts["after_safety_filter"] += len(safe_passages)
        t_safety += time.time() - _t

        if not safe_passages:
            return

        # -----------------------------------------------------------
        # Stage 5: Multi-strategy chunking
        # -----------------------------------------------------------
        _t = time.time()
        all_chunks = []
        locale_lang = to_locale_table_name(lang)
        for passage in safe_passages:
            text = passage["text"]
            doc_id = passage["source_doc_id"]
            is_sel = passage.get("is_selected")

            fixed_chunks = chunk_fixed_size(
                text, doc_id, locale_lang, tokenizer,
                chunk_tokens=args.fixed_chunk_tokens,
                overlap_tokens=args.fixed_chunk_overlap,
            )
            all_chunks.extend(fixed_chunks)
            chunk_counts_by_strategy["fixed"] += len(fixed_chunks)

            semantic_chunks = chunk_semantic(
                text, doc_id, locale_lang, embedder,
                similarity_threshold=args.semantic_similarity_threshold,
            )
            all_chunks.extend(semantic_chunks)
            chunk_counts_by_strategy["semantic"] += len(semantic_chunks)

            meta_chunks = chunk_metadata_aware(
                text, doc_id, locale_lang, tokenizer,
                is_selected=is_sel,
                chunk_tokens=args.fixed_chunk_tokens,
                overlap_tokens=args.fixed_chunk_overlap,
            )
            all_chunks.extend(meta_chunks)
            chunk_counts_by_strategy["metadata_aware"] += len(meta_chunks)

        stage_counts["total_chunks_produced"] += len(all_chunks)
        t_chunk += time.time() - _t

        if not all_chunks:
            return

        # -----------------------------------------------------------
        # Stage 6: Embed chunks
        # -----------------------------------------------------------
        _t = time.time()
        chunk_texts = [c["text"] for c in all_chunks]
        embeddings = embedder.embed_texts(chunk_texts)
        t_embed += time.time() - _t

        # -----------------------------------------------------------
        # Stage 5 (cont): Chunk quality gate guardrail
        # -----------------------------------------------------------
        _t = time.time()
        length_surviving, n_length_dropped = filter_chunks_by_length(
            all_chunks, tokenizer,
            min_tokens=args.min_chunk_tokens,
            max_tokens=args.max_chunk_tokens,
        )

        if n_length_dropped > 0:
            surviving_indices = []
            length_surviving_set = set(id(c) for c in length_surviving)
            for idx, chunk in enumerate(all_chunks):
                if id(chunk) in length_surviving_set:
                    surviving_indices.append(idx)
            embeddings = embeddings[surviving_indices]
            all_chunks = length_surviving

        rejected_counts["chunk_too_short_or_long"] += n_length_dropped

        n_near_dup = 0
        if len(all_chunks) > 0:
            all_chunks, embeddings, n_near_dup = deduplicate_chunks_by_embedding(
                all_chunks, embeddings,
                threshold=args.near_dup_threshold,
            )
            rejected_counts["chunk_near_duplicate"] += n_near_dup

        stage_counts["after_chunk_quality_gate"] += len(all_chunks)
        t_quality += time.time() - _t

        if not all_chunks:
            return

        # -----------------------------------------------------------
        # Stage 7: Vector validation guardrail
        # -----------------------------------------------------------
        _t = time.time()
        valid_chunks, valid_embeddings, n_vec_invalid = validate_vectors(
            all_chunks, embeddings
        )

        if n_vec_invalid > 0:
            failed_mask = np.any(np.isnan(embeddings), axis=1) | (
                np.linalg.norm(embeddings, axis=1) == 0.0
            )
            failed_indices = np.where(failed_mask)[0]
            failed_chunks = [all_chunks[i] for i in failed_indices]
            failed_texts = [c["text"] for c in failed_chunks]

            retry_embeddings = embedder.embed_texts(failed_texts)
            retry_valid, retry_emb, retry_invalid = validate_vectors(
                failed_chunks, retry_embeddings
            )

            if retry_valid:
                valid_chunks.extend(retry_valid)
                valid_embeddings = np.vstack([valid_embeddings, retry_emb]) if len(valid_embeddings) > 0 else retry_emb
                n_vec_invalid = retry_invalid

        rejected_counts["vector_invalid"] += n_vec_invalid
        stage_counts["after_vector_validation"] += len(valid_chunks)
        t_vecval += time.time() - _t

        if not valid_chunks:
            return

        # -----------------------------------------------------------
        # Stage 8: Write to LanceDB
        # -----------------------------------------------------------
        _t = time.time()
        locale_table = to_locale_table_name(lang)
        store.add_batch(valid_chunks, valid_embeddings, table_name=locale_table)
        stage_counts["final_rows_written"] += len(valid_chunks)
        t_write += time.time() - _t


    try:
        for lang in languages_to_process:
            for row in _stream_language_rows(lang):
                if args.limit and rows_processed[lang] >= args.limit:
                    break

                batch_rows_by_lang[lang].append(row)
                rows_processed[lang] += 1
                stage_counts["total_rows_loaded"] += 1
                if pbar is not None:
                    pbar.update(1)

                if len(batch_rows_by_lang[lang]) >= args.batch_size:
                    _process_batch(lang, batch_rows_by_lang[lang])
                    batch_rows_by_lang[lang] = []

            # Flush remaining for this language
            if batch_rows_by_lang[lang]:
                _process_batch(lang, batch_rows_by_lang[lang])
                batch_rows_by_lang[lang] = []

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Flushing current batches...")
        for lang in languages_to_process:
            if batch_rows_by_lang[lang]:
                _process_batch(lang, batch_rows_by_lang[lang])
                batch_rows_by_lang[lang] = []
    finally:
        pbar.close()

    # -----------------------------------------------------------------------
    # Timings
    # -----------------------------------------------------------------------
    timings["extract_passages"] = t_extract
    timings["structural_validation"] = t_validate
    timings["safety_filter"] = t_safety
    timings["chunking"] = t_chunk
    timings["embedding"] = t_embed
    timings["chunk_quality_gate"] = t_quality
    timings["vector_validation"] = t_vecval
    timings["write_lancedb"] = t_write

    # -----------------------------------------------------------------------
    # Maybe build ANN index
    # -----------------------------------------------------------------------
    t0 = time.time()
    skip_index = args.smoke_test
    index_built = store.maybe_build_index(skip_index=skip_index)
    timings["index_build"] = time.time() - t0

    # -----------------------------------------------------------------------
    # Write reports
    # -----------------------------------------------------------------------
    locales_processed = [to_locale_table_name(l) for l in languages_to_process]
    _write_reports(args, locales_processed, stage_counts, chunk_counts_by_strategy,
                   rejected_counts, index_built, timings)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Languages processed: %s (tables: %s)", languages_to_process, locales_processed)
    logger.info("Total rows loaded: %d", stage_counts["total_rows_loaded"])
    logger.info("Total passages extracted: %d", stage_counts["total_passages_extracted"])
    logger.info("Final rows written to LanceDB: %d", stage_counts["final_rows_written"])
    logger.info("Index built: %s", index_built)

    for stage, secs in timings.items():
        logger.info("  %s: %.2f seconds", stage, secs)


def _write_reports(
    args,
    languages_processed: List[str],
    stage_counts: Dict[str, int],
    chunk_counts_by_strategy: Dict[str, int],
    rejected_counts: Dict[str, int],
    index_built: bool,
    timings: Dict[str, float],
):
    """Write ingestion_report.json and manifest.json."""
    report = {
        "embedding_model": MODEL_NAME,
        "embedding_backend": "onnx",
        "embedding_dim": EMBEDDING_DIM,
        "passage_prefix": PASSAGE_PREFIX,
        "query_prefix_note": "runtime MUST prefix queries with 'query: ' to match this index",
        "languages_processed": languages_processed,
        "stage_counts": stage_counts,
        "chunk_counts_by_strategy": chunk_counts_by_strategy,
        "rejected_counts_by_guardrail": rejected_counts,
        "index_built": index_built,
        "wall_clock_seconds_by_stage": {k: round(v, 3) for k, v in timings.items()},
    }

    manifest = {
        "embedding_model": MODEL_NAME,
        "embedding_backend": "onnx",
        "embedding_dim": EMBEDDING_DIM,
        "passage_prefix": PASSAGE_PREFIX,
        "query_prefix": QUERY_PREFIX,
        "query_prefix_note": "runtime MUST prefix queries with 'query: ' to match this index",
    }

    os.makedirs(args.output_dir, exist_ok=True)

    report_path = os.path.join(args.output_dir, "ingestion_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Wrote ingestion report to %s", report_path)

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("Wrote manifest to %s", manifest_path)
