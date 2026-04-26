"""
RAGSmith – Settings router
Both Ollama and Groq are always available.
This router exposes model lists, key management, and status for both providers.
"""

import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from database import get_connection, db_fetchone, db_execute, db_insert, ph
from config import get_settings

router = APIRouter()
logger = logging.getLogger("ragsmith.settings")


# ── Pydantic models ───────────────────────────────────────────────────────────

class SettingResponse(BaseModel):
    groq_api_key_configured: bool
    groq_available: bool
    ollama_available: bool
    llm_provider: str           # kept for legacy clients — returns "both"
    available_models: List[str] # kept for legacy clients — returns groq models if key set, else ollama

class ModelListResponse(BaseModel):
    provider: str
    models: List[str]

class APIKeyValidationRequest(BaseModel):
    api_key: str = Field(..., description="API key to validate")

class APIKeyValidationResponse(BaseModel):
    valid: bool
    message: str
    provider: str


# ── DB helpers ────────────────────────────────────────────────────────────────

def _save_setting(key: str, value: str) -> None:
    conn = get_connection()
    try:
        existing = db_fetchone(conn, f"SELECT value FROM app_settings WHERE key={ph()}", (key,))
        if existing:
            db_execute(conn,
                f"UPDATE app_settings SET value={ph()}, updated_at=datetime('now') WHERE key={ph()}",
                (value, key), commit=True)
        else:
            db_insert(conn,
                f"INSERT INTO app_settings (key, value) VALUES ({ph()},{ph()})",
                (key, value), commit=True)
    finally:
        conn.close()


def _load_setting(key: str) -> Optional[str]:
    conn = get_connection()
    try:
        row = db_fetchone(conn, f"SELECT value FROM app_settings WHERE key={ph()}", (key,))
        return row["value"] if row else None
    finally:
        conn.close()


def _effective_groq_key(cfg) -> str:
    return _load_setting("groq_api_key") or cfg.groq_api_key


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/", response_model=SettingResponse)
def get_settings_info():
    """Overall settings snapshot — used by the App Settings modal."""
    cfg = get_settings()
    from services.llm import _check_ollama, _check_groq, LLMError

    groq_key     = _effective_groq_key(cfg)
    ollama_ok    = _check_ollama(cfg.ollama_base_url)
    groq_ok      = False
    if groq_key:
        try:
            groq_ok = _check_groq(groq_key)
        except LLMError:
            pass

    # legacy `available_models` — return groq list if key available, else ollama
    if groq_ok:
        av_models = [m.strip() for m in cfg.groq_available_models.split(",")]
    else:
        av_models = [m.strip() for m in cfg.ollama_available_models.split(",")]

    return SettingResponse(
        groq_api_key_configured=bool(groq_key),
        groq_available=groq_ok,
        ollama_available=ollama_ok,
        llm_provider="both",
        available_models=av_models,
    )


@router.get("/models", response_model=ModelListResponse)
def get_available_models(provider: Optional[str] = None):
    """
    Return model list for `provider` ('ollama' or 'groq').
    If no provider specified, tries groq first (if key configured), then ollama.
    """
    cfg = get_settings()
    from services.llm import _groq_list_models, _ollama_list_models, LLMError

    target = (provider or "").lower()

    if target == "groq" or (not target):
        groq_key = _effective_groq_key(cfg)
        if groq_key:
            try:
                models = _groq_list_models(groq_key)
                return ModelListResponse(provider="groq", models=models)
            except Exception as exc:
                logger.warning("Groq model list failed: %s", exc)
                if target == "groq":
                    # Fallback to configured list
                    return ModelListResponse(
                        provider="groq",
                        models=[m.strip() for m in cfg.groq_available_models.split(",") if m.strip()]
                    )
        elif target == "groq":
            # No key — return static list
            return ModelListResponse(
                provider="groq",
                models=[m.strip() for m in cfg.groq_available_models.split(",") if m.strip()]
            )

    # ollama
    try:
        models = _ollama_list_models(cfg.ollama_base_url)
    except Exception:
        models = [m.strip() for m in cfg.ollama_available_models.split(",") if m.strip()]
    return ModelListResponse(provider="ollama", models=models)


@router.post("/groq/validate", response_model=APIKeyValidationResponse)
def validate_groq_api_key(body: APIKeyValidationRequest):
    """Validate a Groq API key without saving it."""
    if not body.api_key or not body.api_key.strip():
        raise HTTPException(status_code=400, detail="API key cannot be empty")
    try:
        from services.llm import _check_groq, _groq_list_models
        is_valid = _check_groq(body.api_key)
        if is_valid:
            models = _groq_list_models(body.api_key)
            return APIKeyValidationResponse(
                valid=True,
                message=f"Valid key. {len(models)} models available.",
                provider="groq"
            )
        return APIKeyValidationResponse(valid=False, message="Key validation failed.", provider="groq")
    except Exception as exc:
        logger.error("Groq validation error: %s", exc)
        return APIKeyValidationResponse(valid=False, message=f"Error: {exc}", provider="groq")


@router.post("/groq/save")
def save_groq_api_key(body: APIKeyValidationRequest):
    """Validate then persist Groq API key to the database."""
    if not body.api_key or not body.api_key.strip():
        raise HTTPException(status_code=400, detail="API key cannot be empty")
    try:
        from services.llm import _check_groq
        if not _check_groq(body.api_key):
            raise HTTPException(status_code=400, detail="Invalid Groq API key")
        _save_setting("groq_api_key", body.api_key)
        logger.info("Groq API key saved")
        return {"success": True, "message": "Groq API key saved successfully", "provider": "groq"}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error saving Groq key: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to save key: {exc}")


@router.get("/groq/key-status")
def get_groq_key_status():
    """Check whether a Groq API key is configured."""
    cfg = get_settings()
    stored = _effective_groq_key(cfg)
    return {"configured": bool(stored), "provider": "groq"}


@router.get("/ollama/status")
def get_ollama_status():
    """Check whether local Ollama is reachable."""
    cfg = get_settings()
    from services.llm import _check_ollama, _ollama_list_models
    ok = _check_ollama(cfg.ollama_base_url)
    models = []
    if ok:
        try:
            models = _ollama_list_models(cfg.ollama_base_url)
        except Exception:
            pass
    return {
        "available": ok,
        "base_url": cfg.ollama_base_url,
        "models": models,
    }