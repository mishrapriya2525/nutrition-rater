# 🧠 Brain Health Nutrition Rater

A **production-grade RAG system** that rates food products and supplements for brain health impact using **Azure OpenAI GPT-4o mini + Qdrant hybrid vector search**, based on Dr. Daniel Amen's *Bright Minds* framework.

> **Current status:** End-to-end pipeline validated on 50 products · 206 knowledge chunks ingested · Batch architecture designed for 2.5M+ product scale

---

## 🎯 What It Does

- Rates any food product or supplement from **0–100** for brain health impact
- Returns a clear **Grade: ✅ Good / 🟡 Neutral / ❌ Bad / ⚪ NOT FOUND**
- Provides **2–4 sentence expert advice** per product grounded in the Bright Minds framework
- Supports **single, batch, and async job** processing via REST API
- Batch pipeline with **resume support** — safely restarts after interruption without reprocessing cached results

---

## 🏗️ Architecture

```
User Query → FastAPI → HybridRetriever (Qdrant) → Azure OpenAI GPT-4o mini → Rated Output
                              ↓
                    Dense Embeddings (all-MiniLM-L6-v2)
                  + Sparse BM25 (Reciprocal Rank Fusion)
```

### Why Hybrid Search?
Pure semantic search misses exact ingredient and brand name matches. Pure keyword search misses intent. This system combines both using **Reciprocal Rank Fusion (RRF)** — giving better retrieval precision than either approach alone.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Azure OpenAI GPT-4o mini |
| Vector DB | Qdrant (hybrid search) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| API | FastAPI + Uvicorn |
| Async Queue | Celery + Redis |
| Deploy | Docker + Docker Compose |
| UI | HTML/CSS/JS Dashboard |
| Logging | structlog (JSON in prod, pretty in dev) |
| Config | pydantic-settings + .env |

---

## 📊 Sample Results

| Product | Score | Grade |
|---|---|---|
| Wild Alaskan Salmon Omega-3 | 90 | ✅ Good |
| Turmeric Curcumin BioPerine | 85 | ✅ Good |
| Magnesium Glycinate 400mg | 70 | ✅ Good |
| Probiotic 50 Billion CFU | 65 | 🟡 Neutral |
| Diet Soda with Aspartame | 10 | ❌ Bad |
| Frosted Sugar Flakes Cereal | 10 | ❌ Bad |
| Unknown Supplement XYZ | null | ⚪ NOT FOUND |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- Docker Desktop
- Azure OpenAI API Key (GPT-4o mini deployment)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/mishrapriya2525/nutrition-rag-rater.git
cd nutrition-rag-rater

# 2. Start infrastructure
docker-compose up -d qdrant redis

# 3. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 4. Install dependencies
pip install -r requirements.txt

# 5. Setup environment
cp config/.env.example config/.env
# Add these in config/.env:
# AZURE_OPENAI_KEY=your_key
# AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
# AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini-2

# 6. Download knowledgebase data
python scripts/download_demo_data.py

# 7. Ingest knowledgebase into Qdrant
python ingestion/ingest.py --source data/knowledgebase/ --collection brain_health_kb

# 8. Start API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 9. Open UI
start ui/index.html    # Windows
open ui/index.html     # Mac
```

### Quick Smoke Test

```bash
# Test retriever
python test_retrieval.py

# Test rater (single product)
python test_rater.py

# Run 50-product batch validation
python simple_batch.py
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/rate/single` | POST | Rate a single product |
| `/rate/batch` | POST | Rate up to 1000 products synchronously |
| `/rate/async-job` | POST | Submit large batch as async Celery job |
| `/job/status/{id}` | GET | Poll async job progress |
| `/job/result/{id}` | GET | Download completed job as CSV |
| `/health` | GET | Liveness check |
| `/docs` | GET | Swagger UI |

### Example Request

```json
POST /rate/single
{
  "db_id": "001",
  "product": "Wild Alaskan Salmon Omega-3 1000mg",
  "brand": "Nordic Naturals"
}
```

### Example Response

```json
{
  "db_id": "001",
  "product": "Wild Alaskan Salmon Omega-3 1000mg",
  "brand": "Nordic Naturals",
  "score": 90,
  "grade": "Good",
  "advice": "Rich in omega-3 fatty acids DHA and EPA essential for brain health; supports neurotransmitter function and reduces inflammation; high-quality source with no artificial additives."
}
```

---

## 🧠 How RAG Works Here

1. **Ingestion** — Knowledgebase documents (TXT, PDF, MD, DOCX) chunked into 512-token segments with 64-token overlap using `RecursiveCharacterTextSplitter`
2. **Embedding** — Each chunk embedded with `all-MiniLM-L6-v2` (384-dim), stored as dense vectors in Qdrant
3. **Sparse Indexing** — BM25-style sparse vectors built from a shared vocabulary and stored alongside dense vectors
4. **Hybrid Retrieval** — Dense cosine similarity + BM25 sparse search, fused via **Reciprocal Rank Fusion (RRF)** — top-5 chunks returned
5. **Generation** — Retrieved chunks passed to **Azure OpenAI GPT-4o mini** with `temperature=0.0` and a strict CSV output contract
6. **Validation** — Output parsed and validated for schema compliance (column count, grade values, score range) before returning — falls back to NOT FOUND on any violation

---

## ⚙️ Batch Processing

For large-scale product rating:

```bash
# Rate products from a CSV file
python scripts/run_batch.py \
    --input data/products/sample_products.csv \
    --output data/output/rated_products.csv \
    --batch-size 500 \
    --resume
```

**Batch features:**
- Celery `group()` for parallel processing across workers
- `--resume` flag skips already-cached products (Redis) — safe to restart mid-run
- Checkpoint logging every 5,000 products with ETA calculation
- Automatic fallback to NOT FOUND on worker timeout

**To run at full scale with Open Food Facts (~2.5M products):**
```bash
wget https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz
# Then run the batch pipeline above
# Note: full run requires Azure OpenAI API budget and rate limit planning
```

---

## 🔬 Evaluation

```bash
# Format compliance check on batch output
python evaluation/run_eval.py --output-csv data/output/rated_products.csv

# Consistency check (same product rated twice, tolerance ±5 points)
python evaluation/run_eval.py --consistency

# Golden set evaluation (requires golden_set.csv)
python evaluation/run_eval.py --golden data/evaluation/golden_set.csv
```

Evaluation checks: column order compliance, valid grade values, score range (0–100), NOT FOUND handling, and grade accuracy against human-labeled golden set (80% accuracy threshold).

---

## 📁 Project Structure

```
nutrition-rater/
├── api/            → FastAPI endpoints + rate limiting + Redis cache
├── rag/            → HybridRetriever (RRF) + NutritionRater (LLM)
├── ingestion/      → Document loader, chunker, BM25 vocab builder, Qdrant upsert
├── workers/        → Celery async tasks
├── scripts/        → run_batch.py (production), download_demo_data.py, simple_batch.py
├── evaluation/     → run_eval.py — golden set eval, consistency checks, format compliance
├── config/         → pydantic-settings, .env
├── ui/             → HTML/CSS/JS dashboard
└── docker/         → Dockerfiles
```

---

## 📈 Scale Design

| Layer | Technology | Notes |
|---|---|---|
| Vector Search | Qdrant hybrid | Current: 206 chunks · Designed for millions |
| Async Processing | Celery + Redis | Parallel batch jobs with resume support |
| LLM | Azure OpenAI GPT-4o mini | Rate limits apply at scale |
| Caching | Redis (24hr TTL) | Avoids re-rating identical products |
| API | FastAPI + Uvicorn | Rate limited: 100 req/min per IP |
| Infrastructure | Docker Compose | Single-node · Kubernetes-ready architecture |

---

## 👩‍💻 Author

**Priya Mishra** — Senior AI Engineer
[LinkedIn](https://www.linkedin.com/in/priyamishra-886892141) · [GitHub](https://github.com/mishrapriya2525)
