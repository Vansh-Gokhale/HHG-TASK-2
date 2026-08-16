# Claude Code prompt — Voice RAG data ingestion pipeline (LanceDB)

Paste everything below into Claude Code as a single prompt.

---

## Context

Build the **offline, batch data ingestion pipeline** for a voice-enabled multilingual
RAG system (hackathon project, "RAG in Goa"). This pipeline runs **once, locally or
on a free Colab CPU runtime — never inside the production container**. It reads the
raw dataset, validates and chunks it with multiple strategies, embeds the chunks, and
writes a persisted **LanceDB** table to disk. The production service (built separately,
out of scope here) will later mount that LanceDB directory **read-only** and only load
a lightweight query-time embedder — so nothing in this pipeline needs to fit a 500MB
runtime RAM budget. It does need to run comfortably on a free-tier machine (assume
~4-8GB RAM, no GPU guaranteed).

**Scope boundary — build ONLY the ingestion pipeline.** Do not write any FastAPI
service, STT/TTS integration, or LLM generation code. This is a standalone,
independently runnable Python project.

## Dataset

`ai4bharat/MSMARCO-XI` on Hugging Face. It is split into **per-language configs**
(e.g. `hi`, `ta`, `te`, `kn`, `bn`, `mr`, `gu`, `ml`, `pa`, `or`, `as`, `ne`, `ur` —
enumerate the actual available configs at runtime, don't hardcode a guessed list).

```python
from datasets import load_dataset
ds = load_dataset("ai4bharat/MSMARCO-XI", "hi", split="train")
```

**Important — do not assume the exact field layout.** Based on the dataset card,
examples are expected to include `query`, an answer field (possibly named `Answer`
or `answers`), a `passages` field (likely structured like the original MS MARCO
format — a dict of parallel lists such as `passage_text`, `is_selected`, `url`, OR
a list of dicts — this is genuinely unconfirmed), plus `source_lang`, `target_lang`,
and a `meta` dict. **Step 1 of the pipeline must print `ds.features` and a sample
row for at least 2 languages, and the extraction code must defensively handle both
a dict-of-lists and a list-of-dicts shape for `passages`** rather than assuming one.
Fail loudly with a clear error (not a silent skip) if neither shape matches.

The retrievable corpus for RAG is the **passage text**, not the queries — extract
and deduplicate passage texts across all examples/languages as the ingestion source
documents. Keep `query`/answer fields available in metadata only if useful for later
eval, not as index content.

## Hard constraints

- **Zero cost.** Every dependency must be open-source / free-tier. No paid APIs in
  this pipeline (no Groq/Sarvam here — those belong to the runtime service, out of
  scope).
- **Offline only.** This never runs on the 500MB Railway runtime container. Assume
  it runs locally or on Colab's free CPU tier.
- **Same embedder, both sides.** Ingestion embeds with `intfloat/multilingual-e5-small`
  and must use the **exact same model, precision, and prompt-prefix convention** the
  runtime query path will use later. Write this explicitly into a `manifest.json` (see
  Deliverables) so the runtime team can verify they match — a mismatch silently breaks
  retrieval.
- **E5 prefix convention is mandatory**: prefix every passage with `"passage: "`
  before embedding. (The runtime side will separately prefix queries with `"query: "`
  — document this in the manifest, don't implement the query side here.)
- **Idempotent.** Re-running the script against the same output directory must not
  duplicate data — use `mode="overwrite"` on table creation, or a `--fresh` flag that
  clears the output dir first.
- **Memory-safe batch processing.** Never load the full dataset + full embedding
  matrix into memory at once. Process in batches (default 500 rows), write to LanceDB
  incrementally.

## Pipeline stages (implement in this exact order)

1. **Load** — `datasets.load_dataset`, per language config, with a `--limit` flag
   (rows per language) for fast smoke-testing.
2. **Structural validation guardrail** — reject non-UTF-8 / empty passage text;
   drop exact-duplicate passages via SHA-256 hash of normalized text (strip,
   lowercase, collapse whitespace) within and across languages. Log a count of
   rows rejected here.
3. **Content safety filter guardrail** — a cheap, non-ML regex/keyword denylist
   scan against passage text. **Load the denylist from an external, user-editable
   file** (`config/denylist.txt`, one phrase per line, create it empty with a
   `# add banned terms, one per line` comment — do not populate it with actual
   terms). Any passage matching a denylist entry is excluded and logged, not
   crashed on.
4. **Multi-strategy chunking** — fan out each surviving passage into three
   parallel chunking strategies (all three run on every document, all three
   feed the same table with a `chunk_strategy` metadata column):
   - **A. Fixed-size with overlap**: tokenize with the e5 tokenizer
     (`AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")`), chunk
     at 256 tokens with 20% (~51 token) overlap. Configurable via CLI flags.
   - **B. Semantic chunking**: split passage into sentences using a lightweight
     regex sentence splitter that handles Indic sentence-final punctuation
     (`।`, `॥`, `?`, `!`, `.`) — do not depend on NLTK's `punkt`, its Indic-script
     support is unreliable. Embed each sentence with the same e5 model, then grow
     a chunk by adding consecutive sentences while cosine similarity to the
     running chunk centroid stays above a threshold (default 0.75); start a new
     chunk when it drops below. Reuse the single loaded e5 model for this — do
     not load a second embedding model.
   - **C. Metadata-aware chunking**: same fixed-size logic as strategy A, but
     attach richer metadata columns — `language` (from the dataset config),
     `source_doc_id`, and `is_selected` (if present in the original passages
     structure) — so the runtime can filter/boost by metadata at query time.
5. **Chunk quality gate guardrail** — drop chunks under 10 tokens or over 512
   tokens; deduplicate near-identical chunks via cosine similarity >0.95 on
   their (about-to-be-computed or already-computed) embeddings, scoped within
   the same source document. Log counts dropped.
6. **Embed** — `intfloat/multilingual-e5-small` loaded via
   `SentenceTransformer("intfloat/multilingual-e5-small", backend="onnx")`
   (requires `pip install "sentence-transformers[onnx]"`, which pulls in
   `optimum[onnxruntime]` — this is the officially supported path, prefer it
   over manual `optimum-cli` export). Prefix every chunk text with `"passage: "`
   before embedding, batch the calls (default batch size 32), normalize output
   vectors (`normalize_embeddings=True`).
7. **Vector validation guardrail** — reject any embedding containing NaN or with
   zero L2 norm; log and skip (do not silently insert). If a batch has failures,
   retry that batch once before skipping the failing rows individually.
8. **Write to LanceDB** — persist to a local directory (default `./lancedb_data`,
   configurable). Define an explicit PyArrow schema (don't let LanceDB infer it):

   ```python
   import pyarrow as pa
   schema = pa.schema([
       pa.field("chunk_id", pa.string()),
       pa.field("source_doc_id", pa.string()),
       pa.field("chunk_strategy", pa.string()),   # "fixed" | "semantic" | "metadata_aware"
       pa.field("language", pa.string()),
       pa.field("text", pa.string()),
       pa.field("vector", pa.list_(pa.float32(), 384)),  # e5-small dim = 384
   ])
   ```

   Connect with `lancedb.connect(output_dir)`, write incrementally per batch with
   `table.add(records)` after an initial `db.create_table(name, schema=schema, mode="overwrite")`
   on the first batch. **Only build an ANN index (`table.create_index(metric="cosine", vector_column_name="vector")`)
   if the final row count exceeds ~5,000** — LanceDB's IVF_PQ index needs a
   meaningful minimum row count per partition to be effective/valid; below that,
   flat brute-force search is already fast enough at hackathon corpus scale and
   skipping the index avoids a common failure mode. Print whichever path was
   taken.

## Project structure

```
ingestion/
  __init__.py
  config.py          # CLI args / defaults (argparse, no extra config-file dependency)
  schema.py           # LanceDB / PyArrow schema
  guardrails.py        # structural validation, denylist filter, chunk quality gate, vector validation
  chunkers.py          # strategies A, B, C
  embedder.py          # e5 ONNX load + batched embed with passage: prefix
  store.py             # LanceDB connect/write/index logic
  pipeline.py           # orchestrates stages 1-8, logging, batching
  main.py               # entrypoint, argparse wiring
config/
  denylist.txt          # empty, user-editable
requirements.txt
README.md
```

## CLI flags (argparse, all with sensible defaults)

`--languages` (comma list, default: all available configs), `--limit` (rows per
language, for smoke tests), `--output-dir` (default `./lancedb_data`), `--fresh`
(clear output dir first), `--batch-size` (default 500), `--fixed-chunk-tokens`
(default 256), `--fixed-chunk-overlap` (default 51), `--semantic-similarity-threshold`
(default 0.75), `--min-chunk-tokens` (default 10), `--max-chunk-tokens` (default 512),
`--near-dup-threshold` (default 0.95), `--denylist-path` (default `config/denylist.txt`),
`--smoke-test` (shortcut: `--limit 20` and skips index build).

## Logging & reporting

Use the standard `logging` module (INFO level, timestamps). Track and print counts
at every guardrail gate: rows in → after dedup → after safety filter → chunks
produced per strategy → chunks after quality gate → vectors after validation →
final rows written. At the end, write `ingestion_report.json` with:

```json
{
  "embedding_model": "intfloat/multilingual-e5-small",
  "embedding_backend": "onnx",
  "embedding_dim": 384,
  "passage_prefix": "passage: ",
  "query_prefix_note": "runtime MUST prefix queries with 'query: ' to match this index",
  "languages_processed": [...],
  "stage_counts": {...},
  "chunk_counts_by_strategy": {...},
  "rejected_counts_by_guardrail": {...},
  "index_built": true/false,
  "wall_clock_seconds_by_stage": {...}
}
```

Also write a separate `manifest.json` with just the embedding model / dim / prefix
fields — this is the file the runtime service should read to self-verify it matches.

## requirements.txt (pin these)

```
datasets
lancedb
pyarrow
sentence-transformers[onnx]
numpy
tqdm
```

## Testing

Include a `pytest` smoke test that runs `main.py --smoke-test` against one language
with `--limit 20`, asserts the output LanceDB directory exists, the table has rows,
`ingestion_report.json` and `manifest.json` were written, and every row's `vector`
field has exactly 384 floats with no NaNs.

## Deliverables checklist

- [ ] Full `ingestion/` package as structured above
- [ ] `config/denylist.txt` (empty, with usage comment)
- [ ] `requirements.txt`
- [ ] `README.md` — how to run full ingestion vs. `--smoke-test`, what each flag does
- [ ] `ingestion_report.json` and `manifest.json` produced on every run
- [ ] pytest smoke test
- [ ] Inline comment at the embedding step explicitly flagging: "runtime query
      embedding MUST use this exact model + `query: ` prefix to stay in the same
      vector space"

Ask me before installing anything beyond `requirements.txt`, and print `ds.features`
for at least two language configs before writing any extraction code, since the
exact passage-field shape is unconfirmed — verify it against real data first.
