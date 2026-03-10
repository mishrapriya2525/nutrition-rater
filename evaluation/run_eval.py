"""
evaluation/run_eval.py
-----------------------
Evaluation harness for the nutrition rater.
Tests:
  1. CSV format compliance (column order, types, valid grades)
  2. NOT FOUND handling
  3. Score consistency (same product → same score)
  4. Spot-check scoring against golden set

Usage:
    python evaluation/run_eval.py --golden data/evaluation/golden_set.csv
    python evaluation/run_eval.py --output-csv data/output/ratings.csv   # validate existing output
"""

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.logger import get_logger

logger = get_logger(__name__)

CSV_COLUMNS = ["db_id", "product", "brand", "Score", "Grade", "Advice"]
VALID_GRADES = {"Good", "Neutral", "Bad", "NOT FOUND"}


# ── Format Compliance Checks ──────────────────────────────────────────────────

def validate_csv_file(filepath: str) -> dict:
    """
    Full format compliance check on a CSV output file.
    Returns report dict with pass/fail counts and error details.
    """
    errors = []
    warnings = []
    total = 0
    not_found_count = 0

    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Check column names
        if reader.fieldnames != CSV_COLUMNS:
            errors.append(f"COLUMN ORDER MISMATCH: got {reader.fieldnames}, expected {CSV_COLUMNS}")
            return {"passed": False, "errors": errors, "total": 0}

        for line_num, row in enumerate(reader, start=2):
            total += 1
            grade = row.get("Grade", "")
            score_raw = row.get("Score", "")
            advice = row.get("Advice", "")
            product = row.get("product", "")

            # Grade must be valid
            if grade not in VALID_GRADES:
                errors.append(f"Line {line_num}: Invalid Grade '{grade}' for product '{product}'")

            if grade == "NOT FOUND":
                not_found_count += 1
                if score_raw.strip():
                    errors.append(f"Line {line_num}: NOT FOUND row should have blank Score, got '{score_raw}'")
                if advice.strip():
                    warnings.append(f"Line {line_num}: NOT FOUND row has non-blank Advice")
            else:
                # Score must be integer 0–100
                try:
                    score = int(score_raw.strip())
                    if not (0 <= score <= 100):
                        errors.append(f"Line {line_num}: Score {score} out of range [0,100]")
                except ValueError:
                    errors.append(f"Line {line_num}: Non-integer Score '{score_raw}' for '{product}'")

                # Advice should be present
                if not advice.strip():
                    warnings.append(f"Line {line_num}: Missing advice for rated product '{product}'")

            # Product must be present
            if not product.strip():
                errors.append(f"Line {line_num}: Empty product name")

    passed = len(errors) == 0
    return {
        "passed": passed,
        "total_rows": total,
        "not_found_count": not_found_count,
        "not_found_pct": round(not_found_count / total * 100, 1) if total else 0,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors[:50],      # first 50 errors
        "warnings": warnings[:20],  # first 20 warnings
    }


# ── Consistency Check ─────────────────────────────────────────────────────────

def consistency_check(n: int = 10) -> dict:
    """
    Rate the same N products twice and check for consistent scores.
    Tolerance: ±5 points.
    """
    from rag.rater import NutritionRater

    test_products = [
        {"product": "Wild Alaskan Salmon Omega-3 1000mg", "brand": "Nordic Naturals"},
        {"product": "Frosted Sugar Flakes Cereal", "brand": "Kellogg's"},
        {"product": "Turmeric Curcumin with BioPerine", "brand": "BioSchwartz"},
        {"product": "Magnesium Glycinate 400mg", "brand": "Doctor's Best"},
        {"product": "Diet Soda with Aspartame", "brand": "Generic"},
    ][:n]

    rater = NutritionRater()
    inconsistent = []

    for p in test_products:
        r1 = rater.rate(p["product"], brand=p["brand"])
        r2 = rater.rate(p["product"], brand=p["brand"])

        s1 = r1.get("score") or 0
        s2 = r2.get("score") or 0
        g1 = r1.get("grade")
        g2 = r2.get("grade")

        if g1 != g2 or abs(s1 - s2) > 5:
            inconsistent.append({
                "product": p["product"],
                "run1": {"score": s1, "grade": g1},
                "run2": {"score": s2, "grade": g2},
            })

    return {
        "tested": len(test_products),
        "inconsistent": len(inconsistent),
        "passed": len(inconsistent) == 0,
        "details": inconsistent,
    }


# ── Golden Set Evaluation ─────────────────────────────────────────────────────

def golden_set_eval(golden_csv: str) -> dict:
    """
    Compare system output against a human-labeled golden set.
    Golden CSV columns: product, brand, expected_grade, expected_score_min, expected_score_max
    """
    from rag.rater import NutritionRater
    rater = NutritionRater()

    results = []
    with open(golden_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            product = row["product"]
            brand = row.get("brand", "")
            expected_grade = row["expected_grade"]
            score_min = int(row.get("expected_score_min", 0))
            score_max = int(row.get("expected_score_max", 100))

            result = rater.rate(product, brand=brand)
            actual_grade = result.get("grade")
            actual_score = result.get("score") or 0

            grade_match = actual_grade == expected_grade
            score_in_range = score_min <= actual_score <= score_max

            results.append({
                "product": product,
                "expected_grade": expected_grade,
                "actual_grade": actual_grade,
                "grade_match": grade_match,
                "score_in_range": score_in_range,
                "actual_score": actual_score,
                "expected_range": f"{score_min}–{score_max}",
            })

    total = len(results)
    grade_accuracy = sum(1 for r in results if r["grade_match"]) / total if total else 0
    score_accuracy = sum(1 for r in results if r["score_in_range"]) / total if total else 0

    return {
        "total": total,
        "grade_accuracy": round(grade_accuracy * 100, 1),
        "score_accuracy": round(score_accuracy * 100, 1),
        "passed": grade_accuracy >= 0.80,   # 80% grade accuracy threshold
        "details": results,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def print_report(title: str, report: dict):
    status = "✅ PASS" if report.get("passed") else "❌ FAIL"
    print(f"\n{'='*60}")
    print(f"  {title} — {status}")
    print(f"{'='*60}")
    for k, v in report.items():
        if k not in ("errors", "warnings", "details"):
            print(f"  {k:25s}: {v}")
    if report.get("errors"):
        print(f"\n  Top Errors:")
        for e in report["errors"][:10]:
            print(f"    ✗ {e}")
    if report.get("warnings"):
        print(f"\n  Warnings:")
        for w in report["warnings"][:5]:
            print(f"    ⚠ {w}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", help="Path to golden set CSV")
    parser.add_argument("--output-csv", help="Path to output CSV to validate format")
    parser.add_argument("--consistency", action="store_true", help="Run consistency check")
    parser.add_argument("--results-json", default="data/evaluation/results.json")
    args = parser.parse_args()

    all_results = {}

    if args.output_csv:
        print(f"\n🔍 Validating CSV format: {args.output_csv}")
        report = validate_csv_file(args.output_csv)
        print_report("CSV Format Compliance", report)
        all_results["format"] = report

    if args.consistency:
        print("\n🔁 Running consistency check...")
        report = consistency_check(n=5)
        print_report("Consistency Check", report)
        all_results["consistency"] = report

    if args.golden:
        print(f"\n🎯 Running golden set evaluation: {args.golden}")
        report = golden_set_eval(args.golden)
        print_report("Golden Set Evaluation", report)
        all_results["golden"] = report

    if not any([args.output_csv, args.consistency, args.golden]):
        print("Usage: python evaluation/run_eval.py --help")
        sys.exit(1)

    # Save results
    Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results_json).write_text(json.dumps(all_results, indent=2))
    print(f"\n📊 Results saved → {args.results_json}")
