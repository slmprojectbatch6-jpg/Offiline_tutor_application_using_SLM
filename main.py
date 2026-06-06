"""
Root entry point — starts the FastAPI server with uvicorn.
Run: python main.py
"""
import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # Prevent .pyc corruption on force-kill

import uvicorn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
