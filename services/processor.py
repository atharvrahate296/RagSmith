"""
RAGSmith – Document processing pipeline

Upload → Extract → Chunk → Embed → Index → Store

Supported file types: PDF, TXT, MD, DOCX
"""

import io
import os
import pickle
import logging
import pathlib
from typing import List, Tuple

import numpy as np

from services.embeddings import embed_texts

logger = logging.getLogger("ragsmith.processor")

CHUNK_SIZE = 500        # characters
CHUNK_OVERLAP = 100     # characters


# ── Text Extraction ───────────────────────────────────────────────────────────

def _extract_text_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF using PyMuPDF (fitz) or pdfminer fallback."""
    try:
        import fitz  # PyMuPDF  (AGPL)
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages)
    except ImportError:
        pass

    try:
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams

        output = io.StringIO()
        extract_text_to_fp(
            io.BytesIO(file_bytes),
            output,
            laparams=LAParams(),
            output_type="text",
            codec="utf-8",
        )
        return output.getvalue()
    except ImportError as exc:
        raise RuntimeError(
            "No PDF library found. Install PyMuPDF: pip install pymupdf"
        ) from exc


def _extract_text_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX using python-docx."""
    try:
        import docx  # python-docx (MIT)
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    except ImportError as exc:
        raise RuntimeError(
            "python-docx not installed. Run: pip install python-docx"
        ) from exc


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Route to the correct extractor based on file extension."""
    ext = pathlib.Path(filename).suffix.lower()
    if ext == ".pdf":
        text = _extract_text_pdf(file_bytes)
    elif ext in (".docx", ".doc"):
        text = _extract_text_docx(file_bytes)
    elif ext in (".txt", ".md", ".rst", ".csv"):
        text = file_bytes.decode("utf-8", errors="replace")
    else:
        # Attempt UTF-8 decode for unknown types
        try:
            text = file_bytes.decode("utf-8", errors="replace")
        except Exception as exc:
            raise ValueError(f"Unsupported file type: {ext}") from exc

    return text.strip()


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Recursive character-based chunking.
    Splits on paragraph boundaries first, then sentences, then characters.
    """
    if not text:
        return []

    chunks: List[str] = []

    # Try paragraph-level splitting first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append(current)
            # If a single paragraph exceeds chunk_size, split it further
            if len(para) > chunk_size:
                sub_chunks = _split_by_size(para, chunk_size, overlap)
                chunks.extend(sub_chunks[:-1])
                current = sub_chunks[-1] if sub_chunks else ""
            else:
                current = para

    if current:
        chunks.append(current)

    # Apply overlap: prepend tail of previous chunk
    overlapped: List[str] = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            overlapped.append(chunk)
        else:
            tail = chunks[i - 1][-overlap:] if overlap else ""
            overlapped.append((tail + " " + chunk).strip() if tail else chunk)

    return [c for c in overlapped if c]


def _split_by_size(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Hard character-level split for oversized segments."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ── FAISS Index ───────────────────────────────────────────────────────────────

def _index_path(project_id: int) -> str:
    return f"data/indexes/project_{project_id}.index"


def _chunks_path(project_id: int) -> str:
    return f"data/chunks/project_{project_id}.pkl"


def build_or_update_index(
    project_id: int,
    new_chunks: List[str],
    doc_id: int,
    filename: str,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> int:
    """
    Embed new_chunks and add them to the project's FAISS index.
    Returns the number of chunks successfully indexed.
    """
    try:
        import faiss  # MIT
    except ImportError as exc:
        raise RuntimeError(
            "faiss-cpu not installed. Run: pip install faiss-cpu"
        ) from exc

    if not new_chunks:
        return 0

    # Load existing index and chunk metadata if present
    idx_path = _index_path(project_id)
    pkl_path = _chunks_path(project_id)

    existing_chunks: List[dict] = []
    index = None

    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            existing_chunks = pickle.load(f)

    if os.path.exists(idx_path):
        index = faiss.read_index(idx_path)

    # Embed new chunks
    logger.info("Embedding %d chunks for doc '%s' …", len(new_chunks), filename)
    vectors = embed_texts(new_chunks, model_name=embedding_model)

    dim = vectors.shape[1]

    if index is None:
        # Inner product on L2-normalised vectors == cosine similarity
        index = faiss.IndexFlatIP(dim)

    index.add(vectors)

    # Append metadata
    for chunk_text_val in new_chunks:
        existing_chunks.append(
            {
                "text": chunk_text_val,
                "doc_id": doc_id,
                "filename": filename,
            }
        )

    # Persist
    os.makedirs(os.path.dirname(idx_path), exist_ok=True)
    faiss.write_index(index, idx_path)
    with open(pkl_path, "wb") as f:
        pickle.dump(existing_chunks, f)

    logger.info("Index updated: %d total vectors", index.ntotal)
    return len(new_chunks)


def search_index(
    project_id: int,
    query_text: str,
    top_k: int = 5,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> List[Tuple[str, float, str]]:
    """
    Search the project's FAISS index.

    Returns
    -------
    List of (chunk_text, score, filename) tuples, best-first.
    """
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss-cpu not installed.") from exc

    idx_path = _index_path(project_id)
    pkl_path = _chunks_path(project_id)

    if not os.path.exists(idx_path) or not os.path.exists(pkl_path):
        return []

    index = faiss.read_index(idx_path)
    with open(pkl_path, "rb") as f:
        chunk_meta: List[dict] = pickle.load(f)

    from services.embeddings import embed_query
    q_vec = embed_query(query_text, model_name=embedding_model)

    k = min(top_k, index.ntotal)
    if k == 0:
        return []

    scores, indices = index.search(q_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunk_meta):
            continue
        meta = chunk_meta[idx]
        results.append((meta["text"], float(score), meta["filename"]))

    return results


def delete_project_index(project_id: int) -> None:
    """Remove all index files for a project."""
    for path in [_index_path(project_id), _chunks_path(project_id)]:
        if os.path.exists(path):
            os.remove(path)
            logger.info("Deleted: %s", path)
