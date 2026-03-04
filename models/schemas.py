"""
RAGSmith – Pydantic request/response schemas
"""

from typing import Optional, List
from pydantic import BaseModel, Field


# ── Projects ──────────────────────────────────────────────────────────────────

class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Unique project name")
    description: Optional[str] = Field("", max_length=500)
    model: Optional[str] = Field("mistral", description="Ollama model to use for generation")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of chunks to retrieve")


class ProjectUpdate(BaseModel):
    description: Optional[str] = Field(None, max_length=500)
    model: Optional[str] = None
    top_k: Optional[int] = Field(None, ge=1, le=20)


class ProjectResponse(BaseModel):
    id: int
    name: str
    description: str
    model: str
    top_k: int
    created_at: str
    updated_at: str
    document_count: int = 0


# ── Documents ─────────────────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    id: int
    project_id: int
    filename: str
    num_chunks: int
    status: str
    error_msg: Optional[str]
    created_at: str


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class ChunkResult(BaseModel):
    text: str
    score: float
    doc_filename: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[ChunkResult]
    model: str


# ── Query Logs ────────────────────────────────────────────────────────────────

class QueryLogResponse(BaseModel):
    id: int
    project_id: int
    query_text: str
    response: str
    num_chunks: int
    created_at: str


# ── Export ────────────────────────────────────────────────────────────────────

class ExportResponse(BaseModel):
    project_name: str
    export_path: str
    message: str
