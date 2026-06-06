"""
FastAPI Main Application Entry Point
"""
import sys
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.database.models import init_db
from backend.api.auth import router as auth_router
from backend.api.chat import router as chat_router
from backend.api.dashboard import router as dashboard_router

from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# ── Lifespan ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 55)
    logger.info(f"  {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 55)

    # Init database
    logger.info("Initialising SQLite database...")
    await init_db()
    logger.success("Database ready.")

    # Run heavy loading in a background thread so the server starts instantly
    def preload_models():
        logger.info("Pre-loading AI models in the background...")
        try:
            from backend.rag.pipeline import get_embedding_model, get_faiss_index
            get_embedding_model()
            get_faiss_index()
            logger.success("RAG pipeline ready.")
        except FileNotFoundError:
            logger.warning(
                "FAISS index not found — RAG disabled until you run Phase 1-3 scripts"
            )
        except Exception as e:
            logger.warning(f"RAG pipeline init warning: {e}")

        try:
            from backend.rag.phi3_model import get_phi3_model
            get_phi3_model()
        except Exception as e:
            logger.warning(f"Phi-3 init: {e}")

    # Run synchronously. This prevents the GIL from freezing the async event loop for browser requests.
    preload_models()

    logger.success("Application startup complete! 🚀")
    logger.info(f"Open browser: http://localhost:8000")

    yield  # App runs here

    logger.info("Shutting down gracefully...")


# ── App ────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Offline AI Tutor powered by Phi-3 + RAG + NCERT",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Exception Handlers ─────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    body = await request.body()
    logger.error(f"Validation Error: {exc.errors()}\nBody: {body}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# ── CORS ───────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(dashboard_router)

# ── Static Files ───────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Health Check ───────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    faiss_ready = settings.FAISS_INDEX_PATH.exists()
    
    # Simple check for the Ollama integration based on the model class status
    phi3_ready = False
    try:
        from backend.rag.phi3_model import get_phi3_model
        ollama_model = get_phi3_model()
        phi3_ready = not ollama_model._mock_mode
    except Exception:
        pass
        
    return {
        "status": "online",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "faiss_index": "ready" if faiss_ready else "not built — run Phase 1-3 scripts",
        "phi3_model": "ready" if phi3_ready else f"not ready — start Ollama app and run `ollama run {settings.OLLAMA_MODEL}`",
    }

# ── Serve Frontend ─────────────────────────────────────────────
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Frontend not found. Build the frontend first."}
