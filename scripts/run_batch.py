"""
scripts/run_batch.py
--------------------
Production batch runner for 2.5M+ products.
Uses Celery chord/group for parallel processing with backpressure control.

Usage:
    python scripts/run_batch.py \
        --input data/products.csv \
        --output data/output/rated_products.csv \
        --workers 16 \
        --batch-size 500 \
        --resume                     # skip already-cached products
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import redis
from celery import group
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from ingestion.logger import get_logger
from workers.tasks import celery_app, rate_single_product

logger = get_logger(__name__)
settings = get_settings()

CSV_COLUMNS = ["db_id", "product", "brand", "Score", "Grade", "Advice"]
CHECKPOINT_INTERVAL = 5000   # write checkpoint every N products


def load_products(input_path: str) -> list[dict]:
    """Load product CSV. Expected columns: product (required), db_id, brand (optional)."""
    products = []
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            product = row.get("product") or row.get("name") or row.get("product_name", "")
            if not product.strip():
                continue
            products.append({
                "product": product.strip(),
                "db_id": (row.get("db_id") or row.get("id") or "").strip(),
                "brand": (row.get("brand") or row.get("brand_name") or "").strip(),
            })
    return products


def get_cached_ids(r: redis.Redis, products: list[dict]) -> set[str]:
    """Return set of product keys already cached — for resume support."""
    cached = set()
    for p in products:
        key = f"rating:{p['product'].lower().strip()}:{p['brand'].lower().strip()}"
        if r.exists(key):
            cached.add(key)
    return cached


def write_csv_row(writer, result: dict):
    writer.writerow({
        "db_id": result.get("db_id", ""),
        "product": result.get("product", ""),
        "brand": result.get("brand", ""),
        "Score": result.get("score", ""),
        "Grade": result.get("grade", ""),
        "Advice": result.get("advice", ""),
    })


def run_batch(
    input_path: str,
    output_path: str,
    batch_size: int = 500,
    resume: bool = False,
):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Load products
    logger.info("loading_products", path=input_path)
    products = load_products(input_path)
    total = len(products)
    logger.info("products_loaded", total=total)

    r = redis.from_url(settings.redis_url, decode_responses=True)

    # ── Resume: filter already-processed
    if resume:
        to_process = []
        for p in products:
            cache_key = f"rating:{p['product'].lower().strip()}:{p['brand'].lower().strip()}"
            if not r.exists(cache_key):
                to_process.append(p)
        skipped = total - len(to_process)
        logger.info("resume_mode", skipped=skipped, remaining=len(to_process))
        products = to_process

    if not products:
        logger.info("nothing_to_process")
        print("✅ All products already cached. Nothing to process.")
        return

    # ── Open output CSV
    write_mode = "a" if resume else "w"
    out_file = open(output_path, write_mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=CSV_COLUMNS)
    if not resume:
        writer.writeheader()

    # ── Process in batches with Celery group
    processed = 0
    failed = 0
    start_time = time.time()

    batches = [products[i:i+batch_size] for i in range(0, len(products), batch_size)]
    logger.info("batches_created", count=len(batches), batch_size=batch_size)

    with tqdm(total=len(products), desc="Rating products", unit="products") as pbar:
        for batch_num, batch in enumerate(batches):
            # Submit batch as Celery group
            job = group(
                rate_single_product.s(
                    p["product"], p["db_id"], p["brand"]
                )
                for p in batch
            )

            # Execute with backpressure: max 1000 queued tasks
            result = job.apply_async()

            # Collect results with timeout
            try:
                batch_results = result.get(timeout=300, disable_sync_subtasks=False)
            except Exception as e:
                logger.error("batch_failed", batch=batch_num, error=str(e))
                # Fallback: mark all as NOT FOUND
                batch_results = [
                    {"db_id": p["db_id"], "product": p["product"], "brand": p["brand"],
                     "score": None, "grade": "NOT FOUND", "advice": ""}
                    for p in batch
                ]
                failed += len(batch)

            # Write results
            for res in batch_results:
                write_csv_row(writer, res)
                processed += 1

            out_file.flush()
            pbar.update(len(batch))

            # Checkpoint log
            if (batch_num + 1) % (CHECKPOINT_INTERVAL // batch_size) == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (len(products) - processed) / rate if rate > 0 else 0
                logger.info(
                    "checkpoint",
                    processed=processed,
                    total=len(products),
                    rate_per_sec=round(rate, 1),
                    eta_hours=round(eta / 3600, 2),
                    failed=failed,
                )

    out_file.close()

    elapsed = time.time() - start_time
    logger.info(
        "batch_complete",
        total_processed=processed,
        failed=failed,
        elapsed_seconds=round(elapsed),
        output=output_path,
    )
    print(f"\n✅ Batch complete!")
    print(f"   Processed : {processed:,}")
    print(f"   Failed    : {failed:,}")
    print(f"   Elapsed   : {elapsed/3600:.2f} hours")
    print(f"   Output    : {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch rating pipeline")
    parser.add_argument("--input", required=True, help="Input products CSV")
    parser.add_argument("--output", required=True, help="Output rated CSV")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--resume", action="store_true", help="Skip cached products")
    args = parser.parse_args()

    run_batch(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
        resume=args.resume,
    )
