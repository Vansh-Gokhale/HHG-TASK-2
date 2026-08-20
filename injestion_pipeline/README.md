# Multilingual Voice RAG Ingestion Pipeline

An offline, batch data ingestion pipeline designed for multilingual voice-enabled Retrieval-Augmented Generation (RAG) systems using LanceDB.

---

## 🚀 Key Features

* **Data Loader**: Auto-fetches and processes the `ai4bharat/MSMARCO-XI` dataset using streaming mode.
* **Content Filters**: Built-in UTF-8 validation, SHA-256 deduplication, and a customizable safety denylist.
* **Multi-Strategy Chunking**: Fixed-size windowing, semantic (similarity-based) splitting, and metadata-aware chunking.
* **Fast Embeddings**: Uses ONNX-optimized `intfloat/multilingual-e5-small` model (~120MB, 384 dimensions).
* **LanceDB Storage**: Serverless, high-performance vector database written directly on disk.

---

## 🛠️ Step-by-Step Execution Guide

### 1. Setup Environment
To keep your disk usage isolated, the code directs all Hugging Face caches to `../hf_cache` (one level above the project folder).
Make sure you have dependencies installed:
```bash
pip install -r requirements.txt
```

### 2. Testing the Pipeline with a Partial Dataset
The entire `ai4bharat/MSMARCO-XI` dataset is **55.6 GB**. If you have limited disk space, **DO NOT run a full ingestion**. 

Because the pipeline uses **Streaming Mode** (`streaming=True`), it reads the dataset records sequentially over the network rather than downloading the entire 55.6 GB dataset upfront. 

To test the pipeline locally on a small subset (e.g., 100 rows per language):
```bash
python -m ingestion.main --languages hi,ta --limit 100 --output-dir ./lancedb_data
```
Or use the smoke test shortcut (20 rows, skips ANN index building):
```bash
python -m ingestion.main --smoke-test --output-dir ./lancedb_data
```

### 3. Generate the DB on a Friend's Laptop
If you want to run the full or larger ingestion on your friend's laptop (which has more disk space/RAM):
1. **Push your code** to GitHub.
2. On your friend's laptop, clone the repository:
   ```bash
   git clone <your-repo-url>
   cd injestion_pipeline
   pip install -r requirements.txt
   ```
3. Run the ingestion pipeline (either full or with a larger limit):
   ```bash
   # Ingest 50,000 rows for Hindi and Tamil
   python -m ingestion.main --languages hi,ta --limit 50000 --output-dir ./lancedb_data
   ```
4. Once completed, copy the generated output folder (`lancedb_data/`) back to your laptop.

---

## 📊 Database Size Math & Railway (500MB Limit) Constraints

### Vector DB Size Estimations
The pipeline uses `multilingual-e5-small` which produces **384-dimensional** embeddings.
* **Vector storage:** $384 \text{ dimensions} \times 4 \text{ bytes (float32)} \approx 1.5 \text{ KB}$ per chunk.
* **Metadata & Raw Text storage:** LanceDB stores the original passage text, language, and source document ID alongside the vector. This adds roughly $\approx 1.0 \text{ KB}$ to $1.5 \text{ KB}$ per chunk.
* **Total storage:** $\approx 2.5 \text{ KB}$ to $3.0 \text{ KB}$ per chunk.

| Number of Chunks | Estimated LanceDB Size | Can run on Railway? |
| :--- | :--- | :--- |
| **10,000** | ~25 MB - 30 MB | ✅ Yes |
| **50,000** | ~125 MB - 150 MB | ✅ Yes |
| **100,000** | ~250 MB - 300 MB | ✅ Yes (Safe limit) |
| **200,000** | ~500 MB - 600 MB | ⚠️ Exceeds Railway disk limits |

### Railway Efficiency Analysis
* **Serverless Overhead:** LanceDB is **extremely efficient** for Railway because it is an embedded serverless DB (like SQLite). It has **zero idle memory and CPU overhead**, unlike managed databases (e.g. Pinecone, pgvector on RDS) which consume RAM 24/7.
* **Hosting constraints:** Under Railway's 500MB limit, your API image and application space will take around 150MB-200MB, leaving **~300MB for the LanceDB folder**. This limits your total storage to **~100,000 chunks**. We recommend targeting a specific sub-slice of the dataset (e.g., `--limit 20000`) for the production Railway deploy.

---

## ☁️ Azure Student Plan Cost Analysis

If you decide to deploy or run the database generation on your Azure Student account (which contains a $100 free credit):

### 1. Ingestion / Generation Cost (Virtual Machine)
Do not leave a VM running 24/7. Spin it up, run ingestion, download the database folder, and delete it:
* **Recommended VM:** `Standard_B2s` (2 vCPUs, 4 GB RAM)
* **Cost:** **~$0.0416 per hour** (~$30/month if run constantly).
* Running the ingestion pipeline for a couple of hours will cost **less than $0.10**.

### 2. Database Storage Cost
* **Azure Blob Storage (Hot LRS):** **~$0.018 per GB per month**. Storing a 5 GB vector DB on Blob Storage costs less than **$0.10/month**.
* **Azure Files (Standard):** **~$0.06 per GB per month**. Useful if mounting the DB directly to a serverless Azure Container App.
* **Avoid managed databases** like Azure Database for PostgreSQL (with pgvector) or Azure Cosmos DB — they cost a minimum of **$15 - $50/month** and will exhaust your student credits quickly. Keep it serverless using LanceDB stored on Azure Files or Blob Storage.
