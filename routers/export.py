"""
RAGSmith – Export router
Packages a project as a self-contained runnable zip archive.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from database import get_connection
from services.exporter import export_project

router = APIRouter()
logger = logging.getLogger("ragsmith.export")


def _get_project_or_404(conn, project_id: int):
    row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")
    return row


@router.get("/{project_id}")
def export_and_download(project_id: int):
    """
    Export project as a self-contained zip archive and download it.
    The archive contains app.py, faiss.index, chunks.pkl, requirements.txt, README.md.
    """
    conn = get_connection()
    try:
        project = _get_project_or_404(conn, project_id)

        zip_path = export_project(
            project_id=project_id,
            project_name=project["name"],
            model=project["model"],
            top_k=project["top_k"],
        )

        if not Path(zip_path).exists():
            raise HTTPException(status_code=500, detail="Export failed: zip file not created.")

        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in project["name"]
        )
        return FileResponse(
            path=zip_path,
            media_type="application/zip",
            filename=f"{safe_name}_ragsmith_export.zip",
        )

    finally:
        conn.close()
