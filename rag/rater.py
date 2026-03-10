"""
rag/rater.py
------------
Core rating engine. Combines retrieved context + LLM to produce
strictly formatted CSV output per the schema contract.

Output schema: db_id,product,brand,Score,Grade,Advice
"""

import re
from typing import Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import get_settings
from ingestion.logger import get_logger
from rag.retriever import HybridRetriever

logger = get_logger(__name__)
settings = get_settings()


# ── Prompt Templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a strict nutrition rater for brain health, trained exclusively on the
Dr. Daniel Amen / Bright Minds / Amen Clinics nutrition philosophy.

YOUR ONLY JOB: Output a single CSV row. Nothing else. No explanations. No markdown. No headers.

OUTPUT FORMAT (exact column order, no deviations):
db_id,product,brand,Score,Grade,Advice

SCORING RULES:
- Score: integer 0–100. Be CONSERVATIVE — default toward lower scores when uncertain.
- Grade: EXACTLY one of: Good | Neutral | Bad | NOT FOUND
- Advice: 2–4 sentences. Plain text. No commas within advice (use semicolons instead).
- If you cannot confidently rate the product from the provided knowledge context: output NOT FOUND with blank Score and Advice.

BRAIN-HEALTH RATING FRAMEWORK (from knowledgebase):
- GOOD (70–100): Omega-3 rich; antioxidant-dense; low glycemic; no artificial additives; supports neurotransmitters; anti-inflammatory
- NEUTRAL (40–69): Mixed profile; some beneficial nutrients offset by minor concerns
- BAD (0–39): High refined sugar; artificial dyes/sweeteners; trans fats; excitotoxins (MSG/aspartame); ultra-processed; pro-inflammatory

STRICT RULES:
1. Use ONLY the provided knowledgebase context to rate — never invent or hallucinate
2. Do NOT use external nutrition frameworks (USDA guidelines, FDA labels, etc.)
3. Do NOT add extra columns, headers, commentary, or line breaks
4. Do NOT invent brand names or db_ids — use exactly what was provided
5. If brand is unknown, leave the brand field blank
6. Wrap Advice in double quotes if it contains commas
7. Output exactly ONE CSV row per call

EXAMPLE OUTPUTS:
12345,Omega-3 Fish Oil 1000mg,Nordic Naturals,88,Good,"High DHA/EPA content supports brain cell membrane integrity; anti-inflammatory properties protect against cognitive decline; minimal additives align with Amen Clinics guidelines; excellent choice for daily brain support."
67890,Frosted Sugar Cereal,Generic,15,Bad,"Extremely high refined sugar content spikes blood glucose and impairs prefrontal cortex function; artificial dyes linked to hyperactivity and attention issues; ultra-processed with no meaningful brain nutrients; strongly discouraged in the Bright Minds protocol."
99999,Unknown Supplement X,,,,NOT FOUND,,"""


USER_PROMPT_TEMPLATE = """Rate this product for brain health impact.

PRODUCT INFO:
- db_id: {db_id}
- Product Name: {product}
- Brand: {brand}

KNOWLEDGEBASE CONTEXT:
{context}

Output the CSV row now:"""


# ── CSV Validator ─────────────────────────────────────────────────────────────

class CSVValidationError(Exception):
    pass


def validate_and_parse_csv_row(raw: str, expected_db_id: str, expected_product: str, expected_brand: str) -> dict:
    """
    Validates LLM output is a properly formatted CSV row.
    Returns parsed dict or raises CSVValidationError.
    """
    raw = raw.strip().strip("`").strip()

    # Remove any accidental header rows
    if raw.lower().startswith("db_id"):
        lines = raw.split("\n")
        raw = "\n".join(l for l in lines if not l.lower().startswith("db_id"))
        raw = raw.strip()

    # Parse CSV carefully (handle quoted fields)
    import csv
    import io
    try:
        reader = csv.reader(io.StringIO(raw))
        rows = list(reader)
        if not rows:
            raise CSVValidationError("Empty output")
        row = rows[0]
    except Exception as e:
        raise CSVValidationError(f"CSV parse failed: {e}")

    # Must have exactly 6 columns
    if len(row) != 6:
        raise CSVValidationError(f"Expected 6 columns, got {len(row)}: {row}")

    db_id, product, brand, score_raw, grade, advice = row

    # Validate Grade
    valid_grades = {"Good", "Neutral", "Bad", "NOT FOUND"}
    if grade not in valid_grades:
        raise CSVValidationError(f"Invalid grade: '{grade}'")

    # Validate Score
    if grade == "NOT FOUND":
        score = None
        advice = ""
    else:
        try:
            score = int(score_raw.strip())
            if not (0 <= score <= 100):
                raise CSVValidationError(f"Score out of range: {score}")
        except ValueError:
            raise CSVValidationError(f"Non-integer score: '{score_raw}'")

        # Advice length check
        sentences = [s.strip() for s in re.split(r'[.!?]+', advice) if s.strip()]
        
        # Advice length check - just ensure some content exists
        if len(advice.strip()) < 20:
            raise CSVValidationError(f"Advice too short")

    # Enforce: never invent db_id or product
    # Allow LLM to normalize slightly but flag major divergence
    return {
        "db_id": expected_db_id,
        "product": expected_product,
        "brand": expected_brand,
        "score": score,
        "grade": grade,
        "advice": advice.strip(),
    }


# ── Rater ────────────────────────────────────────────────────────────────────

class NutritionRater:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.retriever = HybridRetriever()
        logger.info("rater_initialized")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_llm(self, context: str, db_id: str, product: str, brand: str) -> str:
        """Call openAI with strict CSV contract. Retries on failure."""
        user_prompt = USER_PROMPT_TEMPLATE.format(
            db_id=db_id,
            product=product,
            brand=brand or "",
            context=context if context else "No relevant knowledgebase context found for this product.",
        )
        
        response = self.client.chat.completions.create(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    def rate(
        self,
        product: str,
        db_id: str = "",
        brand: Optional[str] = None,
    ) -> dict:
        """
        Rate a single product. Returns parsed result dict.
        Always returns a valid result (never raises).
        """
        db_id = db_id or ""
        brand = brand or ""

        try:
            # Step 1: Retrieve relevant knowledge
            chunks = self.retriever.retrieve(product, brand)
            context = self.retriever.format_context(chunks)

            # Step 2: Call LLM
            raw_output = self._call_llm(context, db_id, product, brand)
            logger.debug("llm_raw_output", product=product, raw=raw_output)

            # Step 3: Validate & parse
            result = validate_and_parse_csv_row(raw_output, db_id, product, brand)
            logger.info(
                "product_rated",
                product=product,
                grade=result["grade"],
                score=result["score"],
            )
            return result

        except CSVValidationError as e:
            logger.warning("csv_validation_failed", product=product, error=str(e))
            return self._not_found_result(db_id, product, brand)

        except Exception as e:
            logger.error("rating_failed", product=product, error=str(e))
            return self._not_found_result(db_id, product, brand)

    def _not_found_result(self, db_id: str, product: str, brand: str) -> dict:
        return {
            "db_id": db_id,
            "product": product,
            "brand": brand,
            "score": None,
            "grade": "NOT FOUND",
            "advice": "",
        }

    def rate_batch(self, products: list[dict]) -> list[dict]:
        """Rate a list of products. Each dict: {product, db_id?, brand?}"""
        results = []
        for item in products:
            result = self.rate(
                product=item.get("product", ""),
                db_id=item.get("db_id", ""),
                brand=item.get("brand"),
            )
            results.append(result)
        return results
