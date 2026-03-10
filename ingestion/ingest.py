"""
ingestion/ingest.py
-------------------
Ingests knowledgebase documents (PDF, TXT, MD, DOCX) into Qdrant.
Supports chunking, embedding, BM25 sparse vectors, and metadata tagging.

Usage:
    python ingestion/ingest.py --source data/knowledgebase/ --collection brain_health_kb
    python ingestion/ingest.py --source data/usda/ --collection brain_health_kb --append
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Generator

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
    VectorsConfig,
)
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config.settings import get_settings
from ingestion.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ── Constants ────────────────────────────────────────────────────────────────
CHUNK_SIZE = 512          # tokens approx
CHUNK_OVERLAP = 64
BATCH_UPSERT = 128        # points per Qdrant upsert call


# ── Document Loaders ─────────────────────────────────────────────────────────

def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: Path) -> str:
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(str(path))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception as e:
        logger.warning("pdf_load_failed", path=str(path), error=str(e))
        return ""


def load_docx(path: Path) -> str:
    try:
        from docx import Document
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        logger.warning("docx_load_failed", path=str(path), error=str(e))
        return ""


LOADERS = {
    ".txt": load_txt,
    ".md": load_txt,
    ".pdf": load_pdf,
    ".docx": load_docx,
}


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    loader = LOADERS.get(suffix)
    if not loader:
        logger.debug("unsupported_file_type", path=str(path))
        return ""
    return loader(path)


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, source: str) -> list[dict]:
    """Split text into overlapping chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [
        {
            "text": chunk.strip(),
            "source": source,
            "chunk_index": i,
            "chunk_id": hashlib.sha256(f"{source}::{i}::{chunk}".encode()).hexdigest()[:16],
        }
        for i, chunk in enumerate(chunks)
        if len(chunk.strip()) > 50  # skip tiny fragments
    ]


# ── BM25 Sparse Vectors ───────────────────────────────────────────────────────

def build_sparse_vector(text: str, vocab: dict[str, int]) -> SparseVector:
    """Convert text to BM25-style sparse vector using a shared vocabulary."""
    tokens = text.lower().split()
    freq: dict[int, float] = {}
    for tok in tokens:
        idx = vocab.get(tok)
        if idx is not None:
            freq[idx] = freq.get(idx, 0) + 1.0
    if not freq:
        return SparseVector(indices=[0], values=[0.0])
    total = sum(freq.values())
    return SparseVector(
        indices=list(freq.keys()),
        values=[v / total for v in freq.values()],
    )


def build_vocabulary(chunks: list[dict]) -> dict[str, int]:
    """Build token→index vocabulary from all chunks."""
    vocab: dict[str, int] = {}
    idx = 0
    for chunk in chunks:
        for tok in chunk["text"].lower().split():
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab


# ── Qdrant Setup ─────────────────────────────────────────────────────────────

def init_collection(client: QdrantClient, collection: str, recreate: bool = False):
    """Create collection with hybrid search support (dense + sparse)."""
    exists = any(c.name == collection for c in client.get_collections().collections)
    if exists and not recreate:
        logger.info("collection_exists", collection=collection)
        return
    if exists:
        client.delete_collection(collection)

    client.create_collection(
        collection_name=collection,
        vectors_config={
            "dense": VectorParams(size=settings.embedding_dim, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
        },
    )
    logger.info("collection_created", collection=collection)


# ── Main Ingestion ────────────────────────────────────────────────────────────

def ingest(source_dir: str, collection: str, append: bool = False):
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # ── 1. Load all documents
    all_chunks: list[dict] = []
    files = list(source_path.rglob("*"))
    logger.info("scanning_files", count=len(files), source=source_dir)

    for fpath in tqdm(files, desc="Loading documents"):
        if not fpath.is_file():
            continue
        text = load_document(fpath)
        if not text.strip():
            continue
        chunks = chunk_text(text, source=fpath.name)
        all_chunks.extend(chunks)
        logger.debug("file_chunked", file=fpath.name, chunks=len(chunks))

    logger.info("total_chunks", count=len(all_chunks))
    if not all_chunks:
        logger.error("no_chunks_found")
        return

    # ── 2. Build vocabulary for sparse vectors
    logger.info("building_vocabulary")
    vocab = build_vocabulary(all_chunks)
    vocab_path = Path(f"data/{collection}_vocab.json")
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_path.write_text(json.dumps(vocab))
    logger.info("vocabulary_saved", path=str(vocab_path), size=len(vocab))

    # ── 3. Embed all chunks (dense)
    logger.info("loading_embedding_model", model=settings.embedding_model)
    embedder = SentenceTransformer(settings.embedding_model)

    texts = [c["text"] for c in all_chunks]
    logger.info("embedding_chunks", count=len(texts))
    dense_vectors = embedder.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # ── 4. Connect to Qdrant
    client = QdrantClient(
    host=settings.qdrant_host,
    port=settings.qdrant_port,
    api_key=settings.qdrant_api_key or None,
    https=False,
    )
    init_collection(client, collection, recreate=not append)

    # ── 5. Upsert points in batches
    logger.info("upserting_to_qdrant", collection=collection)
    points = []
    for i, (chunk, dense_vec) in enumerate(zip(all_chunks, dense_vectors)):
        sparse_vec = build_sparse_vector(chunk["text"], vocab)
        points.append(
            PointStruct(
                id=i,
                vector={
                    "dense": dense_vec.tolist(),
                    "sparse": sparse_vec,
                },
                payload={
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    "chunk_id": chunk["chunk_id"],
                },
            )
        )

        if len(points) >= BATCH_UPSERT:
            client.upsert(collection_name=collection, points=points)
            points = []

    if points:
        client.upsert(collection_name=collection, points=points)

    count = client.count(collection_name=collection).count
    logger.info("ingestion_complete", collection=collection, total_points=count)
    print(f"\n✅ Ingestion complete — {count} chunks in collection '{collection}'")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest knowledgebase into Qdrant")
    parser.add_argument("--source", required=True, help="Path to knowledgebase directory")
    parser.add_argument("--collection", default=settings.qdrant_collection)
    parser.add_argument("--append", action="store_true", help="Append to existing collection")
    args = parser.parse_args()
    ingest(args.source, args.collection, args.append)
