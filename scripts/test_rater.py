import sys
sys.path.insert(0, '.')

from rag.rater import NutritionRater

rater = NutritionRater()
result = rater.rate(
    product="Wild Alaskan Salmon Omega-3 1000mg",
    db_id="001",
    brand="Nordic Naturals"
)
print(result)