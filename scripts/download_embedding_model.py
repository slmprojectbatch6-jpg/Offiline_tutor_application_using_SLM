"""
Download embedding model for offline use.
Run this script once with internet connection to cache the model locally.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from loguru import logger

def download_model():
    """Download and cache the sentence-transformers embedding model."""
    try:
        logger.info("Starting embedding model download...")
        logger.info(f"Cache directory: {settings.MODELS_DIR}")
        
        from sentence_transformers import SentenceTransformer
        import os
        
        # Create cache directory
        cache_dir = settings.MODELS_DIR / "sentence-transformers"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables
        os.environ["HF_HOME"] = str(settings.MODELS_DIR)
        
        # Download model
        logger.info(f"Downloading {settings.EMBEDDING_MODEL}...")
        model = SentenceTransformer(
            settings.EMBEDDING_MODEL, 
            cache_folder=str(cache_dir)
        )
        
        logger.success("Embedding model downloaded successfully!")
        logger.info(f"Model cached in: {cache_dir}")
        logger.info("You can now run your application offline.")
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_model()
