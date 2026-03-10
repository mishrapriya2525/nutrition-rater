# 🧠 Brain Health Nutrition Rater

A production-grade RAG system that rates food products and supplements 
for brain health impact using GPT-4 + Qdrant vector search, 
based on Dr. Daniel Amen's Bright Minds framework.

## 🎯 What It Does

- Rates any food product or supplement from 0-100 for brain health
- Returns Grade: Good / Neutral / Bad / NOT FOUND
- Provides 2-4 sentence expert advice per product
- Handles 2.5M+ products via batch processing

## 🏗️ Architecture
```
User Query → FastAPI → HybridRetriever (Qdrant) → GPT-4 → CSV Output
                              ↓
                    Dense Embeddings (all-MiniLM-L6-v2)
                  + Sparse BM25 (Reciprocal Rank Fusion)
```

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | GPT-4o-mini (OpenAI) |
| Vector DB | Qdrant (hybrid search) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| API | FastAPI + Uvicorn |
| Queue | Celery + Redis |
| Deploy | Docker + Docker Compose |
| UI | HTML/CSS/JS Dashboard |

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

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- Docker Desktop
- OpenAI API Key

### Setup
```bash
# 1. Clone the repo
git clone https://github.com/mishrapriya2525/nutrition-rag-rater.git
cd nutrition-rag-rater

# 2. Start infrastructure
docker-compose up -d qdrant redis

# 3. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Setup environment
cp config/.env.example config/.env
# Add your OPENAI_API_KEY in config/.env

# 6. Ingest knowledgebase
python ingestion/ingest.py --source data/knowledgebase/ --collection brain_health_kb

# 7. Start API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 8. Open UI
open ui/index.html
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/rate/single` | POST | Rate a single product |
| `/rate/batch` | POST | Rate multiple products |
| `/rate/async-job` | POST | Submit async batch job |
| `/job/status/{id}` | GET | Check job progress |
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

## 🧠 How RAG Works Here

1. **Ingestion**: Nutrition knowledgebase chunked into 512-token segments
2. **Embedding**: Each chunk embedded using all-MiniLM-L6-v2
3. **Hybrid Search**: Dense semantic + BM25 sparse search with RRF fusion
4. **Generation**: Top-5 chunks passed to GPT-4 with strict CSV contract
5. **Validation**: Output validated for schema compliance before returning

## 📁 Project Structure
```
nutrition-rater/
├── api/          → FastAPI endpoints
├── rag/          → Retriever + Rater (core RAG logic)
├── ingestion/    → Chunking + embedding + Qdrant upsert
├── workers/      → Celery async tasks
├── scripts/      → Batch runner + evaluation
├── config/       → Settings + environment
├── ui/           → HTML dashboard
└── docker/       → Dockerfiles
```

## 👩‍💻 Author

**Priya Mishra**  
Power BI Developer + AI/RAG Engineer  
[GitHub](https://github.com/mishrapriya2525) • [Upwork](#)