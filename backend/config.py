"""
Central Configuration for Offline Tutor Application
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


# ── Base directory ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────────
    APP_NAME: str = "Offline Personalized Tutor"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # ── Secret (JWT) ─────────────────────────────────────────
    SECRET_KEY: str = "ncert-offline-tutor-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 43200  # 30 days

    # ── Paths ─────────────────────────────────────────────────
    DATA_DIR: Path = BASE_DIR / "data"
    EXTRACTED_TEXT_DIR: Path = BASE_DIR / "extracted_text"
    CHUNKS_DIR: Path = BASE_DIR / "chunks"
    VECTOR_DB_DIR: Path = BASE_DIR / "vector_db"
    MODELS_DIR: Path = BASE_DIR / "models"
    DATABASE_DIR: Path = BASE_DIR / "database"

    # ── Subjects ──────────────────────────────────────────────
    SUBJECTS: list = ["history", "polity", "geography", "economics", "gk"]

    # ── Embedding Model ───────────────────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # ── Qwen2 LLM (Ollama) ────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2:0.5b" # Reduced from phi3 to prevent memory errors
    PHI3_CONTEXT_LENGTH: int = 4096
    PHI3_MAX_TOKENS: int = 512
    PHI3_TEMPERATURE: float = 0.3

    # ── RAG ───────────────────────────────────────────────────
    TOP_K_CHUNKS: int = 4
    CHUNK_SIZE_WORDS: int = 400
    CHUNK_OVERLAP_WORDS: int = 50

    # ── SQLite Database ───────────────────────────────────────
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BASE_DIR}/database/tutor.db"

    # ── FAISS ─────────────────────────────────────────────────
    FAISS_INDEX_PATH: Path = BASE_DIR / "vector_db" / "faiss_index"
    FAISS_METADATA_PATH: Path = BASE_DIR / "vector_db" / "metadata.pkl"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure all directories exist
for _dir in [
    settings.DATA_DIR,
    settings.EXTRACTED_TEXT_DIR,
    settings.CHUNKS_DIR,
    settings.VECTOR_DB_DIR,
    settings.MODELS_DIR,
    settings.DATABASE_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)
