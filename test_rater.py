# test_rater.py banao
from rag.rater import NutritionRater

rater = NutritionRater()
result = rater.rate(product="Omega-3 Fish Oil", db_id="001", brand="Nordic Naturals")
print(result)
