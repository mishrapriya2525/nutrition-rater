"""
workers/tasks.py
----------------
Celery tasks for async batch processing.
Handles queue, retries, progress tracking, caching, and result storage.
"""

import json
import time
from typing import Optional

import redis as sync_redis
from celery import Celery
from celery.utils.log import get_task_logger

from config.settings import get_settings

settings = get_settings()
logger = get_task_logger(__name__)

# ── Celery App ────────────────────────────────────────────────────────────────

celery_app = Celery(
    "nutrition_rater",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    worker_concurrency=settings.celery_concurrency,
    task_acks_late=True,
    worker_prefetch_multiplier=1,      # fair distribution
    task_track_started=True,
    result_expires=86400,              # 24h result TTL
    task_soft_time_limit=600,          # 10 min soft limit per task
    task_time_limit=660,               # 11 min hard kill
)

# ── Redis client for job metadata ─────────────────────────────────────────────

def get_sync_redis():
    return sync_redis.from_url(settings.redis_url, decode_responses=True)


def update_job_meta(r, job_id: str, **kwargs):
    raw = r.get(f"job:{job_id}:meta")
    meta = json.loads(raw) if raw else {}
    meta.update(kwargs)
    r.setex(f"job:{job_id}:meta", 86400, json.dumps(meta))


# ── Core Rating Task ──────────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="workers.tasks.rate_async_batch",
    max_retries=settings.max_retries,
    default_retry_delay=settings.retry_delay_seconds,
)
def rate_async_batch(self, job_id: str, products: list[dict]):
    """
    Process a batch of products asynchronously.
    Updates Redis job metadata with progress.
    Stores final CSV in Redis on completion.
    """
    from rag.rater import NutritionRater

    r = get_sync_redis()
    total = len(products)
    results = []
    failed = []

    update_job_meta(r, job_id, status="running", total=total, processed=0, progress=0.0)
    logger.info(f"[{job_id}] Starting batch — {total} products")

    try:
        rater = NutritionRater()

        for i, item in enumerate(products):
            product = item.get("product", "")
            brand = item.get("brand") or ""
            db_id = item.get("db_id") or ""

            # Deduplication: check cache first
            cache_key = f"rating:{product.lower().strip()}:{brand.lower().strip()}"
            cached = r.get(cache_key)

            if cached:
                result = json.loads(cached)
                logger.debug(f"[{job_id}] Cache hit: {product}")
            else:
                try:
                    result = rater.rate(product=product, db_id=db_id, brand=brand)
                    r.setex(cache_key, settings.cache_ttl_seconds, json.dumps(result))
                except Exception as e:
                    logger.warning(f"[{job_id}] Item failed: {product} — {e}")
                    failed.append({"item": item, "error": str(e)})
                    result = {
                        "db_id": db_id,
                        "product": product,
                        "brand": brand,
                        "score": None,
                        "grade": "NOT FOUND",
                        "advice": "",
                    }

            results.append(result)

            # Update progress every 50 items or on last item
            if (i + 1) % 50 == 0 or (i + 1) == total:
                progress = round(((i + 1) / total) * 100, 1)
                update_job_meta(
                    r, job_id,
                    processed=i + 1,
                    progress=progress,
                    status="running",
                )

            # Small sleep to avoid hammering the LLM API
            time.sleep(0.05)

        # ── Serialize results to CSV
        import csv, io
        CSV_COLUMNS = ["db_id", "product", "brand", "Score", "Grade", "Advice"]
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for res in results:
            writer.writerow({
                "db_id": res.get("db_id", ""),
                "product": res.get("product", ""),
                "brand": res.get("brand", ""),
                "Score": res.get("score", ""),
                "Grade": res.get("grade", ""),
                "Advice": res.get("advice", ""),
            })
        csv_content = output.getvalue()

        # Store result CSV in Redis
        r.setex(f"job:{job_id}:result", 86400, csv_content)

        # Store failed items log
        if failed:
            r.setex(f"job:{job_id}:failed", 86400, json.dumps(failed))

        update_job_meta(
            r, job_id,
            status="complete",
            processed=total,
            progress=100.0,
            failed_count=len(failed),
        )
        logger.info(f"[{job_id}] Complete — {total} processed, {len(failed)} failed")

    except Exception as exc:
        logger.error(f"[{job_id}] Batch failed: {exc}")
        update_job_meta(r, job_id, status="failed", error=str(exc))
        raise self.retry(exc=exc)


# ── Sub-task for massive scale (2.5M products) ────────────────────────────────

@celery_app.task(name="workers.tasks.rate_single_product")
def rate_single_product(product: str, db_id: str, brand: str) -> dict:
    """
    Atomic single-product rating task.
    Used by the massive batch pipeline (scripts/run_batch.py).
    """
    from rag.rater import NutritionRater
    r = get_sync_redis()

    cache_key = f"rating:{product.lower().strip()}:{brand.lower().strip()}"
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)

    rater = NutritionRater()
    result = rater.rate(product=product, db_id=db_id, brand=brand or None)
    r.setex(cache_key, settings.cache_ttl_seconds, json.dumps(result))
    return result
