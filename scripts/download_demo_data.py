"""
scripts/download_demo_data.py
-----------------------------
Downloads USDA FoodData Central as knowledgebase
and a sample of Open Food Facts as the product dataset.

Usage:
    python scripts/download_demo_data.py
"""

import csv
import json
import os
import sys
import zipfile
from pathlib import Path

import httpx
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

KB_DIR = Path("data/knowledgebase")
PRODUCTS_DIR = Path("data/products")
KB_DIR.mkdir(parents=True, exist_ok=True)
PRODUCTS_DIR.mkdir(parents=True, exist_ok=True)


# ── USDA FoodData Central — knowledgebase ─────────────────────────────────────

USDA_API_BASE = "https://api.nal.usda.gov/fdc/v1"
USDA_API_KEY = os.getenv("USDA_API_KEY", "DEMO_KEY")  # free key at api.nal.usda.gov

BRAIN_HEALTH_QUERIES = [
    "omega-3 fatty acids DHA EPA brain",
    "antioxidants blueberries brain health",
    "sugar refined glucose cognitive",
    "artificial sweeteners aspartame brain",
    "trans fat inflammation cognitive",
    "B vitamins brain neurotransmitter",
    "magnesium brain mental health",
    "zinc brain cognitive function",
    "probiotics gut brain connection",
    "curcumin turmeric anti-inflammatory brain",
    "dark chocolate flavonoids brain",
    "salmon fish brain DHA",
    "green tea EGCG neuroprotection",
    "processed food additives brain harm",
    "MSG excitotoxin brain",
]


def download_usda_knowledgebase():
    """Download USDA food nutrients data as structured text for RAG ingestion."""
    print("📥 Downloading USDA FoodData knowledge...")

    all_text = []
    client = httpx.Client(timeout=30)

    for query in tqdm(BRAIN_HEALTH_QUERIES, desc="USDA queries"):
        try:
            resp = client.get(
                f"{USDA_API_BASE}/foods/search",
                params={
                    "query": query,
                    "pageSize": 25,
                    "api_key": USDA_API_KEY,
                    "dataType": ["Foundation", "SR Legacy"],
                },
            )
            if resp.status_code != 200:
                continue

            data = resp.json()
            for food in data.get("foods", []):
                name = food.get("description", "")
                nutrients = food.get("foodNutrients", [])
                nutrient_text = "; ".join(
                    f"{n.get('nutrientName', '')}: {n.get('value', '')}{n.get('unitName', '')}"
                    for n in nutrients[:15]
                    if n.get("value")
                )
                category = food.get("foodCategory", "")
                text_chunk = (
                    f"Food: {name}\n"
                    f"Category: {category}\n"
                    f"Key Nutrients: {nutrient_text}\n"
                    f"Brain Health Context: {query}\n"
                )
                all_text.append(text_chunk)

        except Exception as e:
            print(f"  ⚠ USDA query failed: {query} — {e}")

    # Save as a single knowledge text file
    output = KB_DIR / "usda_brain_health_nutrients.txt"
    output.write_text("\n\n---\n\n".join(all_text), encoding="utf-8")
    print(f"✅ USDA knowledge saved → {output} ({len(all_text)} entries)")

    # Also write Dr. Amen-style nutrition rules as a static KB file
    write_amen_rules()


def write_amen_rules():
    """
    Write static Bright Minds / Dr. Amen nutrition principles as a knowledgebase file.
    In production, replace this with the actual book content provided by the client.
    """
    rules = """
# Bright Minds Nutrition Philosophy — Dr. Daniel Amen

## Core Principle
Food is medicine. Every bite either helps or hurts your brain. The goal is to choose
foods that heal, support, and protect the brain over the long term.

## Foods That HELP the Brain (Score 70–100)
- Omega-3 fatty acids (DHA, EPA): Found in wild salmon, sardines, mackerel, omega-3 supplements.
  Critical for brain cell membrane integrity, reduces inflammation, improves mood and cognition.
- Antioxidants: Blueberries, dark leafy greens, dark chocolate (>70% cacao), green tea.
  Protect neurons from oxidative stress and free radical damage.
- B Vitamins: B6, B9 (folate), B12 — essential for neurotransmitter synthesis.
  Deficiency linked to depression, cognitive decline, dementia.
- Magnesium: Almonds, spinach, avocado. Regulates NMDA receptors, reduces anxiety.
- Zinc: Pumpkin seeds, beef, chickpeas. Critical for hippocampal neurogenesis.
- Probiotics/Prebiotics: Gut-brain axis; supports serotonin production (90% made in gut).
- Turmeric/Curcumin: Potent anti-inflammatory; crosses blood-brain barrier; neuroprotective.
- Low-glycemic foods: Berries, vegetables, legumes. Stable blood sugar = stable brain.

## Foods That HURT the Brain (Score 0–39)
- Refined sugar and high-fructose corn syrup: Spikes insulin, causes neuroinflammation,
  impairs hippocampal function and memory. Found in sodas, candy, white bread, pastries.
- Artificial food dyes (Red 40, Yellow 5, Yellow 6, Blue 1): Linked to ADHD, behavioral issues.
- Artificial sweeteners (aspartame, sucralose): Aspartame is an excitotoxin; disrupts gut microbiome.
- Trans fats / partially hydrogenated oils: Cause vascular inflammation, block omega-3 uptake.
- MSG (monosodium glutamate): Excitotoxin — overstimulates neurons to the point of damage.
- Ultra-processed foods (NOVA Group 4): Linked to 22–28% increased risk of depression and anxiety.
- Alcohol: Neurotoxin; shrinks brain volume; disrupts sleep and memory consolidation.
- Gluten (for sensitive individuals): Causes systemic inflammation, brain fog in celiac/sensitivity.
- Vegetable/seed oils high in omega-6 (corn, soybean, canola): Promote neuroinflammation.

## Neutral Foods (Score 40–69)
- Whole grains (moderate): Some B vitamins, but can spike blood sugar in excess.
- Dairy (moderate): Contains protein and B12, but can be inflammatory for some.
- Chicken/turkey: Lean protein for neurotransmitter precursors; lower omega-3 than fish.
- Eggs: Excellent choline source (essential for acetylcholine), but limit if cardiovascular risk.

## Supplement Rating Guidance
- Fish oil / Krill oil: Excellent — DHA/EPA direct brain support (score 80–95)
- Vitamin D3: Excellent — neuroprotective, anti-inflammatory (score 80–90)
- Magnesium glycinate/threonate: Excellent for sleep, anxiety, cognition (score 80–90)
- Multivitamins with synthetic fillers/dyes: Moderate — nutrition OK but fillers concerning
- Protein powders with sucralose/aspartame: Bad — sweetener risk negates protein benefit
- Pre-workouts with artificial dyes + caffeine + aspartame: Bad

## Scoring Conservatism Rule
When uncertain, score lower. A product with unknown ingredients defaults to NOT FOUND.
Partial information warrants a conservative score. Bright Minds principle: "When in doubt, leave it out."

## NOT FOUND Rule
If the product name gives insufficient information about ingredients/composition, output NOT FOUND.
Do not guess or hallucinate ingredient profiles for unknown products.
"""
    rules_file = KB_DIR / "bright_minds_nutrition_rules.txt"
    rules_file.write_text(rules, encoding="utf-8")
    print(f"✅ Bright Minds rules written → {rules_file}")


# ── Open Food Facts — sample product dataset ──────────────────────────────────

OFF_SAMPLE_URL = "https://huggingface.co/datasets/openfoodfacts/product-database/resolve/main/off.parquet"


def create_sample_products_csv(n: int = 1000):
    """
    Creates a demo products CSV from a small hardcoded sample.
    In production, download the full Open Food Facts parquet (4M products).
    """
    print(f"\n📦 Creating sample product dataset ({n} items)...")

    sample_products = [
        {"db_id": "001", "product": "Wild Alaskan Salmon Omega-3 1000mg", "brand": "Nordic Naturals"},
        {"db_id": "002", "product": "Frosted Sugar Flakes Cereal", "brand": "Kellogg's"},
        {"db_id": "003", "product": "Organic Blueberry Antioxidant Blend", "brand": "Navitas Organics"},
        {"db_id": "004", "product": "Diet Soda with Aspartame", "brand": "Coca-Cola"},
        {"db_id": "005", "product": "Turmeric Curcumin with BioPerine", "brand": "BioSchwartz"},
        {"db_id": "006", "product": "Instant Ramen Noodles MSG", "brand": "Maruchan"},
        {"db_id": "007", "product": "Magnesium Glycinate 400mg", "brand": "Doctor's Best"},
        {"db_id": "008", "product": "Skittles Rainbow Candy", "brand": "Mars"},
        {"db_id": "009", "product": "Vitamin B-Complex with B12", "brand": "Thorne"},
        {"db_id": "010", "product": "Protein Bar with Sucralose", "brand": "Quest Nutrition"},
        {"db_id": "011", "product": "Extra Virgin Olive Oil", "brand": "California Olive Ranch"},
        {"db_id": "012", "product": "Gummy Bears with Red 40", "brand": "Haribo"},
        {"db_id": "013", "product": "Probiotic 50 Billion CFU", "brand": "Garden of Life"},
        {"db_id": "014", "product": "Doritos Nacho Cheese", "brand": "Frito-Lay"},
        {"db_id": "015", "product": "Vitamin D3 5000 IU", "brand": "NOW Foods"},
        {"db_id": "016", "product": "Energy Drink with Artificial Colors", "brand": "Monster"},
        {"db_id": "017", "product": "Walnuts Raw Unsalted", "brand": "Kirkland"},
        {"db_id": "018", "product": "Chocolate Frosting with Trans Fat", "brand": "Betty Crocker"},
        {"db_id": "019", "product": "Wild Blueberry Powder", "brand": "Wilderness Poets"},
        {"db_id": "020", "product": "Xyz Unknown Supplement 3000", "brand": ""},
    ]

    # Repeat to reach n products for demo
    products = []
    for i in range(n):
        base = sample_products[i % len(sample_products)].copy()
        base["db_id"] = str(i + 1).zfill(6)
        products.append(base)

    output = PRODUCTS_DIR / "sample_products.csv"
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["db_id", "product", "brand"])
        writer.writeheader()
        writer.writerows(products)

    print(f"✅ Sample products saved → {output} ({len(products)} rows)")
    print(f"\n💡 For full 2.5M product run, download Open Food Facts:")
    print(f"   wget https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz")
    print(f"   Then: python scripts/run_batch.py --input data/off_products.csv --output data/output/ratings.csv --resume")


if __name__ == "__main__":
    download_usda_knowledgebase()
    create_sample_products_csv(n=1000)
    print("\n🎉 Demo data ready!")
    print("   Next step: python ingestion/ingest.py --source data/knowledgebase/ --collection brain_health_kb")
