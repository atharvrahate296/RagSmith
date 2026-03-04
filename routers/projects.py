"""
RAGSmith – Projects router
CRUD for RAG projects.
"""

import logging
from typing import List

from fastapi import APIRouter, HTTPException, status

from database import get_connection
from models.schemas import ProjectCreate, ProjectUpdate, ProjectResponse
from services.processor import delete_project_index

router = APIRouter()
logger = logging.getLogger("ragsmith.projects")


def _row_to_project(row, doc_count: int = 0) -> ProjectResponse:
    return ProjectResponse(
        id=row["id"],
        name=row["name"],
        description=row["description"] or "",
        model=row["model"],
        top_k=row["top_k"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        document_count=doc_count,
    )


@router.get("/", response_model=List[ProjectResponse])
def list_projects():
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT p.*, COUNT(d.id) as doc_count
            FROM projects p
            LEFT JOIN documents d ON d.project_id = p.id
            GROUP BY p.id
            ORDER BY p.created_at DESC
            """
        ).fetchall()
        return [_row_to_project(r, r["doc_count"]) for r in rows]
    finally:
        conn.close()


@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(body: ProjectCreate):
    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT id FROM projects WHERE name = ?", (body.name,)
        ).fetchone()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Project '{body.name}' already exists.",
            )
        cur = conn.execute(
            """
            INSERT INTO projects (name, description, model, top_k)
            VALUES (?, ?, ?, ?)
            """,
            (body.name, body.description or "", body.model or "mistral", body.top_k or 5),
        )
        conn.commit()
        row = conn.execute(
            "SELECT *, 0 as doc_count FROM projects WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
        logger.info("Created project '%s' (id=%d)", body.name, cur.lastrowid)
        return _row_to_project(row, 0)
    finally:
        conn.close()


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: int):
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT p.*, COUNT(d.id) as doc_count
            FROM projects p
            LEFT JOIN documents d ON d.project_id = p.id
            WHERE p.id = ?
            GROUP BY p.id
            """,
            (project_id,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Project not found")
        return _row_to_project(row, row["doc_count"])
    finally:
        conn.close()


@router.patch("/{project_id}", response_model=ProjectResponse)
def update_project(project_id: int, body: ProjectUpdate):
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Project not found")

        fields = []
        values = []
        if body.description is not None:
            fields.append("description = ?")
            values.append(body.description)
        if body.model is not None:
            fields.append("model = ?")
            values.append(body.model)
        if body.top_k is not None:
            fields.append("top_k = ?")
            values.append(body.top_k)

        if fields:
            fields.append("updated_at = datetime('now')")
            values.append(project_id)
            conn.execute(
                f"UPDATE projects SET {', '.join(fields)} WHERE id = ?", values
            )
            conn.commit()

        updated = conn.execute(
            """
            SELECT p.*, COUNT(d.id) as doc_count
            FROM projects p
            LEFT JOIN documents d ON d.project_id = p.id
            WHERE p.id = ?
            GROUP BY p.id
            """,
            (project_id,),
        ).fetchone()
        return _row_to_project(updated, updated["doc_count"])
    finally:
        conn.close()


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: int):
    conn = get_connection()
    try:
        row = conn.execute("SELECT id FROM projects WHERE id = ?", (project_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Project not found")
        conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        conn.commit()
        delete_project_index(project_id)
        logger.info("Deleted project id=%d", project_id)
    finally:
        conn.close()
