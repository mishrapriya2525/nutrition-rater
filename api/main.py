"""
api/main.py
-----------
FastAPI service with three rating modes:
  POST /rate/single      — rate one product, return CSV row
  POST /rate/batch       — rate up to 1000 products
  POST /rate/async-job   — submit large batch as async job
  GET  /job/status/{id}  — poll job progress
  GET  /job/result/{id}  — download result CSV
  GET  /health           — liveness check
"""

import csv
import io
import time
import uuid
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

import redis.asyncio as aioredis
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config.settings import get_settings
from ingestion.logger import get_logger
from workers.tasks import rate_async_batch

logger = get_logger(__name__)
settings = get_settings()

app = FastAPI(
    title="Brain-Health Nutrition Rater API",
    description="RAG-powered nutrition scoring for 2.5M+ products (Dr. Amen / Bright Minds framework)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared Redis client ───────────────────────────────────────────────────────

async def get_redis():
    client = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


# ── Pydantic Models ───────────────────────────────────────────────────────────

class ProductInput(BaseModel):
    db_id: Optional[str] = Field(default="", description="Source database ID")
    product: str = Field(..., min_length=1, description="Product name")
    brand: Optional[str] = Field(default=None, description="Brand name")


class BatchInput(BaseModel):
    products: list[ProductInput] = Field(..., max_length=1000)


class RatingResult(BaseModel):
    db_id: str
    product: str
    brand: str
    score: Optional[int]
    grade: str
    advice: str


class JobSubmitted(BaseModel):
    job_id: str
    status: str
    product_count: int
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str          # pending | running | complete | failed
    progress: float      # 0–100
    processed: int
    total: int
    error: Optional[str] = None


# ── CSV helpers ───────────────────────────────────────────────────────────────

CSV_COLUMNS = ["db_id", "product", "brand", "Score", "Grade", "Advice"]

def results_to_csv(results: list[dict]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=CSV_COLUMNS,
        extrasaction="ignore",
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    for r in results:
        writer.writerow({
            "db_id": r.get("db_id", ""),
            "product": r.get("product", ""),
            "brand": r.get("brand", ""),
            "Score": r.get("score", ""),
            "Grade": r.get("grade", ""),
            "Advice": r.get("advice", ""),
        })
    return output.getvalue()


def result_to_pydantic(r: dict) -> RatingResult:
    return RatingResult(
        db_id=r.get("db_id", ""),
        product=r.get("product", ""),
        brand=r.get("brand", ""),
        score=r.get("score"),
        grade=r.get("grade", "NOT FOUND"),
        advice=r.get("advice", ""),
    )


# ── Rate limiter (simple Redis sliding window) ────────────────────────────────

async def check_rate_limit(request: Request, redis: aioredis.Redis):
    """100 requests/minute per IP."""
    ip = request.client.host
    key = f"rl:{ip}:{int(time.time() // 60)}"
    count = await redis.incr(key)
    await redis.expire(key, 120)
    if count > 100:
        raise HTTPException(status_code=429, detail="Rate limit exceeded (100 req/min)")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


#@app.post("/rate/single", response_model=RatingResult, summary="Rate a single product")
#async def rate_single(
    #product_input: ProductInput,
    #request: Request,
    #redis: aioredis.Redis = Depends(get_redis),
#):
    #await check_rate_limit(request, redis)
    
    
@app.post("/rate/single", response_model=RatingResult, summary="Rate a single product")
async def rate_single(product_input: ProductInput):
    from rag.rater import NutritionRater
    rater = NutritionRater()
    result = rater.rate(
        product=product_input.product,
        db_id=product_input.db_id or "",
        brand=product_input.brand,
    )
    return result_to_pydantic(result)

    # Cache lookup
    cache_key = f"rating:{product_input.product.lower().strip()}:{(product_input.brand or '').lower().strip()}"
    cached = await redis.get(cache_key)
    if cached:
        import json
        logger.debug("cache_hit", product=product_input.product)
        return result_to_pydantic(json.loads(cached))

    # Rate product
    from rag.rater import NutritionRater
    rater = NutritionRater()
    result = rater.rate(
        product=product_input.product,
        db_id=product_input.db_id or "",
        brand=product_input.brand,
    )

    # Cache result
    import json
    await redis.setex(cache_key, settings.cache_ttl_seconds, json.dumps(result))

    return result_to_pydantic(result)


@app.post("/rate/batch", summary="Rate up to 1000 products synchronously")
async def rate_batch(
    batch_input: BatchInput,
    request: Request,
    redis: aioredis.Redis = Depends(get_redis),
):
    await check_rate_limit(request, redis)

    from rag.rater import NutritionRater
    rater = NutritionRater()

    results = []
    for item in batch_input.products:
        # Cache check per item
        cache_key = f"rating:{item.product.lower().strip()}:{(item.brand or '').lower().strip()}"
        cached = await redis.get(cache_key)
        if cached:
            import json
            results.append(json.loads(cached))
            continue

        result = rater.rate(
            product=item.product,
            db_id=item.db_id or "",
            brand=item.brand,
        )
        import json
        await redis.setex(cache_key, settings.cache_ttl_seconds, json.dumps(result))
        results.append(result)

    csv_content = results_to_csv(results)
    return StreamingResponse(
        io.BytesIO(csv_content.encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ratings.csv"},
    )


@app.post("/rate/async-job", response_model=JobSubmitted, summary="Submit large batch as async job")
async def submit_async_job(
    batch_input: BatchInput,
    redis: aioredis.Redis = Depends(get_redis),
):
    job_id = str(uuid.uuid4())
    products = [p.model_dump() for p in batch_input.products]

    # Store job metadata in Redis
    import json
    await redis.setex(
        f"job:{job_id}:meta",
        3600 * 24,
        json.dumps({
            "status": "pending",
            "total": len(products),
            "processed": 0,
            "progress": 0.0,
            "error": None,
        }),
    )

    # Dispatch Celery task
    rate_async_batch.apply_async(
        args=[job_id, products],
        task_id=job_id,
    )

    logger.info("async_job_submitted", job_id=job_id, count=len(products))
    return JobSubmitted(
        job_id=job_id,
        status="pending",
        product_count=len(products),
        message=f"Job submitted. Poll /job/status/{job_id} for progress.",
    )


@app.get("/job/status/{job_id}", response_model=JobStatus, summary="Poll async job status")
async def job_status(
    job_id: str,
    redis: aioredis.Redis = Depends(get_redis),
):
    import json
    raw = await redis.get(f"job:{job_id}:meta")
    if not raw:
        raise HTTPException(status_code=404, detail="Job not found")
    meta = json.loads(raw)
    return JobStatus(job_id=job_id, **meta)


@app.get("/job/result/{job_id}", summary="Download completed job CSV")
async def job_result(
    job_id: str,
    redis: aioredis.Redis = Depends(get_redis),
):
    import json
    raw = await redis.get(f"job:{job_id}:meta")
    if not raw:
        raise HTTPException(status_code=404, detail="Job not found")
    meta = json.loads(raw)
    if meta["status"] != "complete":
        raise HTTPException(status_code=202, detail=f"Job not complete. Status: {meta['status']}")

    csv_data = await redis.get(f"job:{job_id}:result")
    if not csv_data:
        raise HTTPException(status_code=404, detail="Result not found")

    return StreamingResponse(
        io.BytesIO(csv_data.encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=job_{job_id}.csv"},
    )
