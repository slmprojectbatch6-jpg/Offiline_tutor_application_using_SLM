"""
Phase 3: Embedding Generation + FAISS Vector Database
-------------------------------------------------------
Reads all chunk JSONs, generates sentence embeddings using
all-MiniLM-L6-v2, builds a FAISS index, and saves metadata.
"""

import json
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from backend.config import settings


def load_all_chunks() -> List[Dict[str, Any]]:
    """Load chunks from all subject JSON files."""
    all_chunks = []
    for subject in settings.SUBJECTS:
        chunk_file = settings.CHUNKS_DIR / f"{subject}_chunks.json"
        if not chunk_file.exists():
            logger.warning(f"Chunk file not found: {chunk_file} (run Phase 2 first)")
            continue
        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        all_chunks.extend(chunks)
        logger.info(f"Loaded {len(chunks)} chunks for {subject}")
    return all_chunks


def generate_embeddings(texts: List[str]) -> np.ndarray:
    """Generate embeddings for a list of texts using MiniLM."""
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
    model = SentenceTransformer(settings.EMBEDDING_MODEL)

    logger.info(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # Normalize for cosine similarity
        convert_to_numpy=True,
    )
    logger.success(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> Any:
    """Build FAISS index from embeddings."""
    import faiss

    dim = embeddings.shape[1]
    logger.info(f"Building FAISS index with dimension {dim}...")

    # Use IndexFlatIP (Inner Product) since embeddings are normalized
    # → equivalent to cosine similarity
    index = faiss.IndexFlatIP(dim)

    # For larger datasets, use IVF index for speed:
    # quantizer = faiss.IndexFlatIP(dim)
    # index = faiss.IndexIVFFlat(quantizer, dim, min(100, len(embeddings)//10))
    # index.train(embeddings)

    index.add(embeddings.astype(np.float32))
    logger.success(f"FAISS index built with {index.ntotal} vectors")
    return index


def save_index(index: Any, metadata: List[Dict]) -> None:
    """Save FAISS index and metadata to disk."""
    import faiss

    settings.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, str(settings.FAISS_INDEX_PATH))
    logger.success(f"FAISS index saved → {settings.FAISS_INDEX_PATH}")

    # Save metadata (chunk info without embeddings)
    with open(settings.FAISS_METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    logger.success(f"Metadata saved → {settings.FAISS_METADATA_PATH}")


def load_index():
    """Load FAISS index and metadata (used at runtime by RAG pipeline)."""
    import faiss

    if not settings.FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {settings.FAISS_INDEX_PATH}. "
            "Please run Phase 3 first: python scripts/generate_embeddings.py"
        )

    index = faiss.read_index(str(settings.FAISS_INDEX_PATH))
    with open(settings.FAISS_METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
    return index, metadata


def main():
    logger.info("=" * 60)
    logger.info("PHASE 3: EMBEDDING GENERATION + FAISS VECTOR DB")
    logger.info("=" * 60)

    # Load all chunks
    chunks = load_all_chunks()
    if not chunks:
        logger.error("No chunks found. Please run Phase 1 and Phase 2 first.")
        return

    logger.info(f"Total chunks to embed: {len(chunks)}")

    # Extract texts
    texts = [c["text"] for c in chunks]

    # Generate embeddings
    embeddings = generate_embeddings(texts)

    # Build FAISS index
    index = build_faiss_index(embeddings)

    # Prepare lean metadata (no text duplication in metadata)
    metadata = [
        {
            "chunk_id": c["chunk_id"],
            "subject": c["subject"],
            "chapter": c["chapter"],
            "chunk_index": c.get("chunk_index", 0),
            "text": c["text"],           # Keep text for retrieval display
            "word_count": c.get("word_count", 0),
        }
        for c in chunks
    ]

    # Save everything
    save_index(index, metadata)

    logger.success("\n✅ Phase 3 Complete!")
    logger.info(f"  Total vectors: {index.ntotal}")
    logger.info(f"  Index file: {settings.FAISS_INDEX_PATH}")
    logger.info(f"  Metadata file: {settings.FAISS_METADATA_PATH}")


if __name__ == "__main__":
    main()
