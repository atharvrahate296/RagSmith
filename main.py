"""
RAGSmith - Fully Open-Source Multi-Project RAG Builder
Main application entry point
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from database import init_db
from routers import projects, documents, query, export

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ragsmith")


# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown logic."""
    logger.info("RAGSmith starting up …")
    init_db()
    os.makedirs("data/indexes", exist_ok=True)
    os.makedirs("data/chunks", exist_ok=True)
    os.makedirs("data/uploads", exist_ok=True)
    os.makedirs("exports", exist_ok=True)
    logger.info("RAGSmith ready ✓")
    yield
    logger.info("RAGSmith shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAGSmith",
    description="Fully Open-Source Multi-Project RAG Builder with Local Export",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files & templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Routers
app.include_router(projects.router, prefix="/api/projects", tags=["Projects"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(query.router, prefix="/api/query", tags=["Query"])
app.include_router(export.router, prefix="/api/export", tags=["Export"])


# ── UI root ───────────────────────────────────────────────────────────────────
from fastapi import Request
from fastapi.responses import HTMLResponse


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "ok", "app": "RAGSmith", "version": "1.0.0"}
