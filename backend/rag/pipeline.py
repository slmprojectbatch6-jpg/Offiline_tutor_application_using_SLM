"""
Phase 4 + 5: RAG Pipeline + Phi-3 Integration
-----------------------------------------------
Handles query embedding → FAISS retrieval → context building → Phi-3 generation
"""

import sys
import pickle
import warnings
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from loguru import logger
from backend.config import settings

warnings.filterwarnings("ignore")


# ── Singleton Embedding Model ─────────────────────────────────
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        import os
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model (first time)...")
        # Set local cache directory to avoid downloading from Hugging Face
        cache_dir = settings.MODELS_DIR / "sentence-transformers"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Force offline mode - don't try to reach Hugging Face
        os.environ["HF_HOME"] = str(settings.MODELS_DIR)
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        
        try:
            # Try loading with local-files-only first
            _embedding_model = SentenceTransformer(
                settings.EMBEDDING_MODEL, 
                cache_folder=str(cache_dir),
                trust_remote_code=False,
                local_files_only=True  # Only use cached files
            )
            logger.success("Embedding model loaded from cache.")
        except Exception as e:
            logger.warning(f"Local-only loading failed: {e}, trying with cache folder...")
            _embedding_model = SentenceTransformer(
                settings.EMBEDDING_MODEL, 
                cache_folder=str(cache_dir),
                trust_remote_code=False
            )
            logger.success("Embedding model loaded.")
    return _embedding_model


# ── Singleton FAISS Index ─────────────────────────────────────
_faiss_index = None
_metadata: List[Dict] = []

def get_faiss_index():
    global _faiss_index, _metadata
    if _faiss_index is None:
        import faiss
        if not settings.FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                "FAISS index not found. Please run:\n"
                "  python scripts/generate_embeddings.py"
            )
        _faiss_index = faiss.read_index(str(settings.FAISS_INDEX_PATH))
        with open(settings.FAISS_METADATA_PATH, "rb") as f:
            _metadata = pickle.load(f)
        logger.info(f"FAISS index loaded: {_faiss_index.ntotal} vectors")
    return _faiss_index, _metadata


# ── Retriever ─────────────────────────────────────────────────

def retrieve_chunks(
    query: str,
    subject_filter: Optional[str] = None,
    top_k: int = settings.TOP_K_CHUNKS,
) -> List[Dict[str, Any]]:
    """
    Embed query → search FAISS → return top-k relevant chunks.
    Optional subject_filter restricts results to one subject.
    """
    embed_model = get_embedding_model()
    faiss_index, metadata = get_faiss_index()

    # Embed query
    query_vec = embed_model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    # Search FAISS (returns more results if filtering)
    search_k = top_k * 5 if subject_filter else top_k
    scores, indices = faiss_index.search(query_vec, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        chunk = metadata[idx].copy()
        chunk["score"] = float(score)

        # Filter by subject if requested
        if subject_filter and chunk["subject"] != subject_filter.lower():
            continue

        results.append(chunk)
        if len(results) >= top_k:
            break

    return results


# ── Context Builder ───────────────────────────────────────────

def build_context(chunks: List[Dict[str, Any]]) -> str:
    """Build a context string from retrieved chunks."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        subject = chunk["subject"].capitalize()
        chapter = chunk["chapter"]
        text = chunk["text"]
        parts.append(f"[Source {i} | {subject} - {chapter}]\n{text}")
    return "\n\n".join(parts)


# ── Prompt Template ───────────────────────────────────────────

TUTOR_PROMPT_TEMPLATE = """<|system|>
You are a helpful and friendly NCERT tutor for school students (Classes 6-10).
Answer ONLY using the context provided below. Do not use outside knowledge.
If the answer is not in the context, say "I don't have information on this topic in my NCERT content."

CRITICAL RULE: The context may contain textbook artifacts like "Questions", "Answer in 100-150 words", or chapter names. Ignore them completely.
DO NOT output textbook questions or headers. Synthesize a clean, factual answer based ONLY on what the user asks.

Give a clear, structured answer in simple language suitable for school students.
Use bullet points or numbered lists where appropriate.
<|end|>
<|user|>
Context:
{context}

Question: {question}
<|end|>
<|assistant|>
<|assistant|>
Answer:"""


def build_prompt(query: str, chunks: List[Dict], history: str = "") -> str:
    context = build_context(chunks)
    
    question_text = f"Previous Context:\n{history}\n\nUser Question: {query}" if history else query
    
    return TUTOR_PROMPT_TEMPLATE.format(context=context, question=question_text)


# ── RAG Pipeline ──────────────────────────────────────────────

def run_rag_query(
    query: str,
    subject_filter: Optional[str] = None,
    top_k: int = settings.TOP_K_CHUNKS,
    stream: bool = False,
    history: str = "",
) -> Dict[str, Any]:
    """
    Full RAG pipeline:
    1. Initialize AgentRAG
    2. Query AgentRAG for answer and chunks
    """
    from backend.rag.agent_rag_fixed import AgentRAG
    
    embed_model = get_embedding_model()
    faiss_index, metadata = get_faiss_index()
    
    agent = AgentRAG(
        ollama_model=settings.OLLAMA_MODEL,
        faiss_index=faiss_index,
        embedder=embed_model,
        chunks=metadata,
        mode="simple", # Start with simple mode as recommended by prompt
        debug=False # Disabled by default
    )
    
    # We pass history inline here, AgentRAG is smart enough to handle generating queries from it
    user_question = f"Previous Context:\n{history}\n\nUser Question: {query}" if history else query
    
    result = agent.query(user_question, subject_filter=subject_filter)
    answer = result.get("answer", "I could not generate an answer.")
    chunks = result.get("chunks", [])

    return {
        "answer": answer,
        "chunks": chunks,
        "sources": [
            {
                "subject": c["subject"],
                "chapter": c["chapter"],
                "score": round(c.get("score", 0.0), 4),
                "preview": c.get("text", "")[:150] + "...",
            }
            for c in chunks
        ],
    }


def run_rag_stream(
    query: str,
    subject_filter: Optional[str] = None,
    top_k: int = settings.TOP_K_CHUNKS,
    history: str = "",
) -> Generator[str, None, None]:
    """Streaming version of RAG pipeline."""
    from backend.rag.phi3_model import get_phi3_model

    # Use strict query for FAISS search to get correct chunks
    chunks = retrieve_chunks(query, subject_filter=subject_filter, top_k=top_k)

    if not chunks:
        yield "I could not find relevant NCERT content for your question."
        return

    # Pass history into the LLM prompt so it understands reference words like 'it', 'he', 'they'
    prompt = build_prompt(query, chunks, history=history)
    model = get_phi3_model()

    yield from model.stream(prompt)
