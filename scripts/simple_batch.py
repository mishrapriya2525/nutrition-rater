import sys, csv
sys.path.insert(0, '.')
from pathlib import Path
from rag.rater import NutritionRater

Path("data/output").mkdir(parents=True, exist_ok=True)
rater = NutritionRater()

products = list(csv.DictReader(open("data/products/sample_products.csv")))[:50]

with open("data/output/rated_products.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["db_id","product","brand","Score","Grade","Advice"])
    writer.writeheader()
    for i, p in enumerate(products):
        result = rater.rate(p["product"], p["db_id"], p.get("brand",""))
        writer.writerow({
            "db_id": result["db_id"],
            "product": result["product"],
            "brand": result["brand"],
            "Score": result["score"] or "",
            "Grade": result["grade"],
            "Advice": result["advice"]
        })
        print(f"{i+1}/50 — {result['product'][:30]} → {result['grade']} {result['score'] or ''}")

print("\n✅ Done! Output: data/output/rated_products.csv")