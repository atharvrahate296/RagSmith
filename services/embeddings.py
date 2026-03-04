"""
RAGSmith – Embedding service
Uses SentenceTransformers (Apache 2.0) with all-MiniLM-L6-v2 by default.
Models are downloaded once and cached locally; no internet needed afterwards.
"""

import logging
import numpy as np
from typing import List

logger = logging.getLogger("ragsmith.embeddings")

# Module-level singleton so the model is loaded only once per process
_model = None
_model_name: str = ""


def _get_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load and cache the embedding model."""
    global _model, _model_name
    if _model is None or _model_name != model_name:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", model_name)
            _model = SentenceTransformer(model_name)
            _model_name = model_name
            logger.info("Embedding model loaded ✓")
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc
    return _model


def embed_texts(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embed a list of texts.

    Returns
    -------
    np.ndarray  shape (N, D), dtype float32
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    model = _get_model(model_name)
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity via inner product
    )
    return vectors.astype(np.float32)


def embed_query(text: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed a single query string. Returns shape (1, D)."""
    return embed_texts([text], model_name=model_name)
