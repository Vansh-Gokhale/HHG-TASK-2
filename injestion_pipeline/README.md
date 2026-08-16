# Multilingual Voice RAG Ingestion Pipeline

An offline, batch data ingestion pipeline designed for multilingual voice-enabled Retrieval-Augmented Generation (RAG) systems.

---

## 🚀 Key Features

* **Data Loader**: Auto-fetches and processes the `ai4bharat/MSMARCO-XI` dataset.
* **Content Filters**: Built-in UTF-8 validation, SHA-256 deduplication, and a customizable safety denylist.
* **Multi-Strategy Chunking**: Fixed-size windowing, semantic (similarity-based) splitting, and metadata-aware chunking.
* **Fast Embeddings**: Uses ONNX-optimized `intfloat/multilingual-e5-small` model.
* **LanceDB Storage**: Direct high-performance batch writes to database tables on disk.

---

## 🛠️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Smoke Test (Fast check, 20 rows)
```bash
python -m ingestion.main --smoke-test
```

### 3. Run Ingestion (Full dataset)
```bash
python -m ingestion.main --output-dir ./lancedb_data
```

### 4. Run Specific Languages
```bash
python -m ingestion.main --languages hi,ta --limit 100
```

---

## 📁 CLI Configuration

| CLI Flag | Default | Description |
| :--- | :--- | :--- |
| `--languages` | all | Comma-separated list of language codes (e.g. `hi,ta`) |
| `--limit` | None | Maximum rows to process per language |
| `--output-dir` | `./lancedb_data` | Directory where LanceDB tables are saved |
| `--fresh` | False | If true, wipes output directory before writing |
