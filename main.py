"""
RAGSmith – Main application entry point
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette import status

templates = Jinja2Templates(directory="templates")
from config import get_settings
from database import init_db
from services.llm import check_llm_available, list_available_models, LLMError

from pydantic import BaseModel

cfg = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RAGSmith starting up [env=%s, db=%s, llm=both (ollama+groq), storage=%s]",
                cfg.app_env, cfg.db_driver, cfg.storage_backend)
    init_db()
    os.makedirs(cfg.faiss_index_dir, exist_ok=True)
    os.makedirs(cfg.faiss_chunks_dir, exist_ok=True)
    if cfg.storage_backend == "local":
        os.makedirs(cfg.local_upload_dir, exist_ok=True)
    os.makedirs("exports", exist_ok=True)
    logger.info("RAGSmith ready ✓")
    yield
    logger.info("RAGSmith shutting down.")

logging.basicConfig(
    level=getattr(logging, cfg.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ragsmith")


app = FastAPI(
    title="RAGSmith",
    description="Fully Open-Source Multi-Project RAG Builder with Local Export",
    version="1.0.0",
    lifespan=lifespan,
    # Disable docs in production to reduce attack surface (optional)
    docs_url="/docs" if not cfg.is_production else None,
    redoc_url="/redoc" if not cfg.is_production else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
from routers import projects, documents, query, export, sessions, settings


app.include_router(projects.router,  prefix="/api/projects",  tags=["Projects"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(query.router,     prefix="/api/query",     tags=["Query"])
app.include_router(sessions.router,  prefix="/api/sessions",  tags=["Sessions"])
app.include_router(settings.router,  prefix="/api/settings",  tags=["Settings"])
app.include_router(export.router,    prefix="/api/export",    tags=["Export"])

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled Exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected internal server error occurred.", "code": "INTERNAL_SERVER_ERROR"},
    )


# --- Custom Exception Handler for LLMError ---
@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError):
    logger.error("LLMError: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc), "code": "LLM_ERROR"},
    )


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    llm_status = check_llm_available()   # now returns both
    return {
        "status": "ok",
        "app": "RAGSmith",
        "version": "1.0.0",
        "env": cfg.app_env,
        "db": cfg.db_driver,
        "llm_provider": "both",
        "llm_available": llm_status["ollama_available"] or llm_status["groq_available"],
        "llm_detail": llm_status.get("detail", ""),
        "ollama_available": llm_status.get("ollama_available", False),
        "groq_available": llm_status.get("groq_available", False),
        "storage": cfg.storage_backend,
    }


@app.get("/api/settings")
async def get_settings_api():
    """Legacy settings endpoint — provider is now 'both'."""
    return {
        "llm_provider": "both",
        "ollama_base_url": cfg.ollama_base_url,
        "ollama_default_model": cfg.ollama_default_model,
        "groq_default_model": cfg.groq_default_model,
        "embedding_model": cfg.embedding_model,
    }