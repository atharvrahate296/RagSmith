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

from config import get_settings
from database import init_db
from routers import projects, documents, query, export

cfg = get_settings()

logging.basicConfig(
    level=getattr(logging, cfg.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ragsmith")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RAGSmith starting up [env=%s, db=%s, llm=%s, storage=%s]",
                cfg.app_env, cfg.db_driver, cfg.llm_provider, cfg.storage_backend)
    init_db()
    os.makedirs(cfg.faiss_index_dir, exist_ok=True)
    os.makedirs(cfg.faiss_chunks_dir, exist_ok=True)
    if cfg.storage_backend == "local":
        os.makedirs(cfg.local_upload_dir, exist_ok=True)
    os.makedirs("exports", exist_ok=True)
    logger.info("RAGSmith ready ✓")
    yield
    logger.info("RAGSmith shutting down.")


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
templates = Jinja2Templates(directory="templates")

app.include_router(projects.router,  prefix="/api/projects",  tags=["Projects"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(query.router,     prefix="/api/query",     tags=["Query"])
app.include_router(export.router,    prefix="/api/export",    tags=["Export"])


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    from services.llm import check_llm_available
    llm_status = check_llm_available()
    return {
        "status": "ok",
        "app": "RAGSmith",
        "version": "1.0.0",
        "env": cfg.app_env,
        "db": cfg.db_driver,
        "llm_provider": llm_status["provider"],
        "llm_available": llm_status["available"],
        "llm_detail": llm_status["detail"],
        "storage": cfg.storage_backend,
    }
