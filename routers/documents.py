"""
RAGSmith – Documents router
Handles file upload, async processing pipeline, and retry.
Uses storage service for file persistence (local or S3).
Uses db helpers for SQLite/Postgres compatibility.
"""

import logging
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, status

from config import get_settings
from database import get_connection, db_execute, db_fetchone, db_fetchall, db_insert, ph
from models.schemas import DocumentResponse
from services.processor import extract_text, chunk_text, build_or_update_index
from services.storage import save_upload, load_upload, delete_upload

router = APIRouter()
logger = logging.getLogger("ragsmith.documents")

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".doc", ".rst", ".csv"}

# Embedding model is ALWAYS the sentence-transformer — never the LLM model name
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _get_project_or_404(conn, project_id: int) -> dict:
    row = db_fetchone(conn, f"SELECT * FROM projects WHERE id = {ph()}", (project_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")
    return row


def _to_response(row: dict) -> DocumentResponse:
    return DocumentResponse(
        id=row["id"],
        project_id=row["project_id"],
        filename=row["filename"],
        num_chunks=row["num_chunks"],
        status=row["status"],
        error_msg=row.get("error_msg"),
        created_at=str(row["created_at"]),
    )


def _process_document(doc_id: int, project_id: int, file_key: str, filename: str):
    """
    Background task: load file → extract → chunk → embed → FAISS index.
    file_key is either a local path or S3 URI depending on storage backend.
    """
    conn = get_connection()
    try:
        db_execute(conn, f"UPDATE documents SET status='processing' WHERE id={ph()}", (doc_id,), commit=True)

        file_bytes = load_upload(file_key)
        text = extract_text(file_bytes, filename)
        if not text:
            raise ValueError("No text could be extracted from the document.")

        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("Document produced no chunks after processing.")

        num_indexed = build_or_update_index(
            project_id=project_id,
            new_chunks=chunks,
            doc_id=doc_id,
            filename=filename,
            embedding_model=EMBEDDING_MODEL,
        )

        db_execute(conn,
            f"UPDATE documents SET status='ready', num_chunks={ph()} WHERE id={ph()}",
            (num_indexed, doc_id), commit=True)
        logger.info("Document %d processed: %d chunks", doc_id, num_indexed)

    except Exception as exc:
        logger.exception("Failed to process document %d: %s", doc_id, exc)
        db_execute(conn,
            f"UPDATE documents SET status='error', error_msg={ph()} WHERE id={ph()}",
            (str(exc)[:500], doc_id), commit=True)
    finally:
        conn.close()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/{project_id}", response_model=List[DocumentResponse])
def list_documents(project_id: int):
    conn = get_connection()
    try:
        _get_project_or_404(conn, project_id)
        rows = db_fetchall(conn,
            f"SELECT * FROM documents WHERE project_id={ph()} ORDER BY created_at DESC",
            (project_id,))
        return [_to_response(r) for r in rows]
    finally:
        conn.close()


@router.post("/{project_id}/upload", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    project_id: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    import pathlib
    cfg = get_settings()

    ext = pathlib.Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    file_bytes = await file.read()
    if len(file_bytes) > cfg.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {cfg.max_upload_mb} MB limit.")
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    conn = get_connection()
    try:
        project = _get_project_or_404(conn, project_id)
        safe_name = (file.filename or "upload").replace("/", "_").replace("\\", "_")

        # Persist file via storage service (local or S3)
        file_key = save_upload(file_bytes, project_id, safe_name)

        doc_id = db_insert(conn,
            f"INSERT INTO documents (project_id, filename, file_path, status) VALUES ({ph()},{ph()},{ph()},'pending')",
            (project_id, safe_name, file_key))

        background_tasks.add_task(
            _process_document,
            doc_id=doc_id, project_id=project_id,
            file_key=file_key, filename=safe_name,
        )

        row = db_fetchone(conn, f"SELECT * FROM documents WHERE id={ph()}", (doc_id,))
        return _to_response(row)
    finally:
        conn.close()


@router.get("/{project_id}/doc/{doc_id}", response_model=DocumentResponse)
def get_document_status(project_id: int, doc_id: int):
    conn = get_connection()
    try:
        _get_project_or_404(conn, project_id)
        row = db_fetchone(conn,
            f"SELECT * FROM documents WHERE id={ph()} AND project_id={ph()}",
            (doc_id, project_id))
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")
        return _to_response(row)
    finally:
        conn.close()


@router.delete("/{project_id}/doc/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(project_id: int, doc_id: int):
    conn = get_connection()
    try:
        _get_project_or_404(conn, project_id)
        row = db_fetchone(conn,
            f"SELECT * FROM documents WHERE id={ph()} AND project_id={ph()}",
            (doc_id, project_id))
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")
        if row.get("file_path"):
            try:
                delete_upload(row["file_path"])
            except Exception as e:
                logger.warning("Could not delete file %s: %s", row["file_path"], e)
        db_execute(conn, f"DELETE FROM documents WHERE id={ph()}", (doc_id,), commit=True)
        logger.info("Deleted document id=%d from project id=%d", doc_id, project_id)
    finally:
        conn.close()


@router.post("/{project_id}/doc/{doc_id}/retry", response_model=DocumentResponse)
def retry_document(project_id: int, doc_id: int, background_tasks: BackgroundTasks):
    """Re-process a document that previously errored without re-uploading."""
    conn = get_connection()
    try:
        _get_project_or_404(conn, project_id)
        row = db_fetchone(conn,
            f"SELECT * FROM documents WHERE id={ph()} AND project_id={ph()}",
            (doc_id, project_id))
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")
        if row["status"] not in ("error", "pending"):
            raise HTTPException(status_code=409,
                detail=f"Cannot retry a document with status '{row['status']}'.")

        file_key = row.get("file_path", "")
        if not file_key:
            raise HTTPException(status_code=404,
                detail="No file key stored. Please re-upload the document.")

        db_execute(conn,
            f"UPDATE documents SET status='pending', error_msg=NULL, num_chunks=0 WHERE id={ph()}",
            (doc_id,), commit=True)

        background_tasks.add_task(
            _process_document,
            doc_id=doc_id, project_id=project_id,
            file_key=file_key, filename=row["filename"],
        )

        updated = db_fetchone(conn, f"SELECT * FROM documents WHERE id={ph()}", (doc_id,))
        return _to_response(updated)
    finally:
        conn.close()
