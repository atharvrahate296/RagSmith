"""
RAGSmith – Query router
Executes the full RAG pipeline: embed → search → generate.
"""

import logging
from typing import List

from fastapi import APIRouter, HTTPException

from database import get_connection, db_fetchone, db_fetchall, db_execute, db_insert, ph
from models.schemas import QueryRequest, QueryResponse, ChunkResult, QueryLogResponse
from services.processor import search_index
from services.llm import generate_answer

router = APIRouter()
logger = logging.getLogger("ragsmith.query")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _get_project_or_404(conn, project_id: int) -> dict:
    row = db_fetchone(conn, f"SELECT * FROM projects WHERE id={ph()}", (project_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")
    return row


@router.post("/{project_id}", response_model=QueryResponse)
def query_project(project_id: int, body: QueryRequest):
    conn = get_connection()
    try:
        project = _get_project_or_404(conn, project_id)

        ready_count = db_fetchone(conn,
            f"SELECT COUNT(*) as cnt FROM documents WHERE project_id={ph()} AND status='ready'",
            (project_id,))["cnt"]

        if not ready_count:
            raise HTTPException(status_code=422,
                detail="No indexed documents in this project. Upload and process a document first.")

        model = project["model"]
        top_k = project["top_k"]

        results = search_index(
            project_id=project_id,
            query_text=body.query,
            top_k=top_k,
            embedding_model=EMBEDDING_MODEL,
        )

        if not results:
            answer = ("No relevant information found in the knowledge base. "
                      "Try rephrasing or uploading more documents.")
            sources: List[ChunkResult] = []
        else:
            try:
                answer = generate_answer(query=body.query, context_chunks=results, model=model)
            except (ConnectionError, RuntimeError) as exc:
                # Surface LLM errors as a 503 with the actual reason
                raise HTTPException(status_code=503, detail=str(exc))
            sources = [
                ChunkResult(text=text[:500], score=round(score, 4), doc_filename=filename)
                for text, score, filename in results
            ]

        db_insert(conn,
            f"INSERT INTO query_logs (project_id, query_text, response, num_chunks) VALUES ({ph()},{ph()},{ph()},{ph()})",
            (project_id, body.query, answer, len(results)))

        return QueryResponse(query=body.query, answer=answer, sources=sources, model=model)
    finally:
        conn.close()


@router.get("/{project_id}/history", response_model=List[QueryLogResponse])
def query_history(project_id: int, limit: int = 20):
    conn = get_connection()
    try:
        _get_project_or_404(conn, project_id)
        rows = db_fetchall(conn,
            f"SELECT * FROM query_logs WHERE project_id={ph()} ORDER BY created_at DESC LIMIT {ph()}",
            (project_id, min(limit, 100)))
        return [QueryLogResponse(
            id=r["id"], project_id=r["project_id"],
            query_text=r["query_text"], response=r["response"],
            num_chunks=r["num_chunks"], created_at=str(r["created_at"]),
        ) for r in rows]
    finally:
        conn.close()
