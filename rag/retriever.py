"""
rag/retriever.py
----------------
Hybrid retriever combining dense (semantic) + sparse (BM25) search.
Returns top-K relevant knowledge chunks for a given product query.
"""

import json
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FusionQuery,
    NearestQuery,
    Prefetch,
    Query,
    SparseVector,
)
from sentence_transformers import SentenceTransformer

from config.settings import get_settings
from ingestion.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class HybridRetriever:
    """
    Retrieves relevant knowledge chunks using hybrid search:
    - Dense: cosine similarity on sentence-transformer embeddings
    - Sparse: BM25-style token overlap
    - Fusion: Reciprocal Rank Fusion (RRF) to merge results
    """

    def __init__(self, collection: Optional[str] = None):
        self.collection = collection or settings.qdrant_collection
        self.embedder = SentenceTransformer(settings.embedding_model)
        self.client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key or None,
        https=False,
        )
        self.vocab: dict[str, int] = self._load_vocab()
        logger.info("retriever_initialized", collection=self.collection)

    def _load_vocab(self) -> dict[str, int]:
        vocab_path = Path(f"data/{self.collection}_vocab.json")
        if vocab_path.exists():
            return json.loads(vocab_path.read_text())
        logger.warning("vocab_not_found", path=str(vocab_path))
        return {}

    def _build_sparse_vector(self, text: str) -> SparseVector:
        tokens = text.lower().split()
        freq: dict[int, float] = {}
        for tok in tokens:
            idx = self.vocab.get(tok)
            if idx is not None:
                freq[idx] = freq.get(idx, 0) + 1.0
        if not freq:
            return SparseVector(indices=[0], values=[0.0])
        total = sum(freq.values())
        return SparseVector(
            indices=list(freq.keys()),
            values=[v / total for v in freq.values()],
        )

    def _build_query(self, text: str) -> str:
        """Construct a rich query from product info for better retrieval."""
        return f"nutrition health impact ingredients: {text}"

    def retrieve(
        self,
        product_name: str,
        brand: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Retrieve top-K relevant knowledge chunks for a product.
        Returns list of {text, source, score} dicts.
        """
        k = top_k or settings.rag_top_k
        query_text = self._build_query(
            f"{product_name} {brand or ''}".strip()
        )

        # Dense embedding
        dense_vec = self.embedder.encode(
            query_text, normalize_embeddings=True
        ).tolist()

        # Sparse BM25 vector
        sparse_vec = self._build_sparse_vector(query_text)

        try:
            # Hybrid search with RRF fusion
            results = self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    Prefetch(
                        query=dense_vec,
                        using="dense",
                        limit=k * 2,
                    ),
                    Prefetch(
                        query=sparse_vec,
                        using="sparse",
                        limit=k * 2,
                    ),
                ],
                query=FusionQuery(fusion="rrf"),
                limit=k,
                with_payload=True,
            )

            chunks = []
            for r in results.points:
                score = r.score if hasattr(r, "score") else 0.0
                if score < settings.rag_score_threshold:
                    continue
                chunks.append({
                    "text": r.payload.get("text", ""),
                    "source": r.payload.get("source", ""),
                    "score": round(score, 4),
                })

            logger.debug(
                "retrieval_done",
                product=product_name,
                chunks_returned=len(chunks),
            )
            return chunks

        except Exception as e:
            logger.error("retrieval_failed", product=product_name, error=str(e))
            return []

    def format_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a single context string for the LLM."""
        if not chunks:
            return ""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[Source {i}: {chunk['source']}]\n{chunk['text']}"
            )
        return "\n\n---\n\n".join(parts)
