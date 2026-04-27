"""
RAGSmith – LLM inference service
Both Ollama and Groq are available simultaneously.
The `provider` argument determines which backend is used per call.

  ollama → Local Ollama (MIT)  — always available if Ollama is running
  groq   → Groq Cloud API      — available when GROQ_API_KEY is set
"""

import json
import logging
import urllib.request
import urllib.error
from typing import List, Tuple, Optional

logger = logging.getLogger("ragsmith.llm")

SYSTEM_PROMPT = (
    "You are RAGSmith, a helpful AI assistant. "
    "Answer the user's question using ONLY the context provided below. "
    "If the context does not contain enough information, say so honestly. "
    "Do not hallucinate or invent facts. Be concise and accurate.\n"
)

GROQ_API_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"


class LLMError(Exception):
    pass


def _requests():
    try:
        import requests
        return requests
    except ImportError as exc:
        raise RuntimeError("requests not installed. Run: pip install requests") from exc


def _try_json_parse(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("JSON decode failed for: %s", text[:120])
        return {}


# ── Public API ────────────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    context_chunks: List[Tuple[str, float, str]],
    model: str = "",
    provider: str = "",
    history: List[Tuple[str, str]] = None,
) -> str:
    """
    Generate an answer using either Ollama or Groq.
    `provider` must be 'ollama' or 'groq'.  Defaults to 'ollama'.
    `model`    defaults to the configured default for that provider.
    """
    from config import get_settings
    cfg = get_settings()

    effective_provider = (provider or "ollama").lower()
    effective_model    = model or cfg.default_model_for(effective_provider)

    if effective_provider == "groq":
        groq_key = _get_effective_groq_key(cfg)
        return _groq_generate(query, context_chunks, effective_model, groq_key, history)

    return _ollama_generate(query, context_chunks, effective_model, cfg.ollama_base_url, history)


def check_llm_available(provider: Optional[str] = None) -> dict:
    """
    Check availability.
    If provider is None, checks BOTH and returns a combined status dict.
    If provider is 'ollama' or 'groq', returns single-provider status.
    """
    from config import get_settings
    cfg = get_settings()

    if provider is None:
        # Check both
        ollama_ok = _check_ollama(cfg.ollama_base_url)
        groq_key  = _get_effective_groq_key(cfg)
        groq_ok   = False
        groq_detail = "No Groq API key configured"
        if groq_key:
            try:
                groq_ok = _check_groq(groq_key)
                groq_detail = "Groq API reachable" if groq_ok else "Groq API unreachable"
            except LLMError as e:
                groq_detail = str(e)

        # For the legacy single-provider health field, prefer groq if available
        primary_provider = "groq" if groq_ok else "ollama"
        primary_available = groq_ok or ollama_ok

        return {
            "available": primary_available,
            "provider": primary_provider,
            "detail": f"ollama={'up' if ollama_ok else 'down'}, groq={'up' if groq_ok else 'down'}",
            "ollama_available": ollama_ok,
            "groq_available": groq_ok,
            "groq_detail": groq_detail,
            "ollama_detail": f"Ollama at {cfg.ollama_base_url}" if ollama_ok else "Ollama not running",
        }

    if provider == "groq":
        groq_key = _get_effective_groq_key(cfg)
        if not groq_key:
            return {"available": False, "provider": "groq", "detail": "No Groq API key configured"}
        try:
            ok = _check_groq(groq_key)
            return {"available": ok, "provider": "groq",
                    "detail": "Groq API reachable" if ok else "Groq API unreachable"}
        except LLMError as e:
            return {"available": False, "provider": "groq", "detail": str(e)}

    # ollama
    ok = _check_ollama(cfg.ollama_base_url)
    return {
        "available": ok, "provider": "ollama",
        "detail": f"Ollama at {cfg.ollama_base_url}" if ok else "Ollama not running",
    }


def list_available_models(provider: str = "ollama") -> List[str]:
    """List models for the given provider."""
    from config import get_settings
    cfg = get_settings()

    if provider == "groq":
        groq_key = _get_effective_groq_key(cfg)
        if not groq_key:
            return [m.strip() for m in cfg.groq_available_models.split(",") if m.strip()]
        try:
            return _groq_list_models(groq_key)
        except Exception:
            return [m.strip() for m in cfg.groq_available_models.split(",") if m.strip()]

    try:
        return _ollama_list_models(cfg.ollama_base_url)
    except Exception:
        return [m.strip() for m in cfg.ollama_available_models.split(",") if m.strip()]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_effective_groq_key(cfg) -> str:
    """Return Groq key from DB override first, then env/config."""
    try:
        from database import get_connection, db_fetchone, ph
        conn = get_connection()
        try:
            row = db_fetchone(conn, f"SELECT value FROM app_settings WHERE key={ph()}", ("groq_api_key",))
            if row and row["value"]:
                return row["value"]
        finally:
            conn.close()
    except Exception:
        pass
    return cfg.groq_api_key


def _build_messages(
    query: str,
    context_chunks: List[Tuple[str, float, str]],
    history: List[Tuple[str, str]] = None
) -> List[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        for q, a in history:
            messages.append({"role": "user",      "content": q})
            messages.append({"role": "assistant",  "content": a})

    if context_chunks:
        parts = [
            f"[Source {i}: {fn} | relevance: {s:.3f}]\n{t}"
            for i, (t, s, fn) in enumerate(context_chunks, 1)
        ]
        ctx = "\n\n---\n\n".join(parts)
    else:
        ctx = "No relevant context found."

    user_content = f"CONTEXT:\n{ctx}\n\nQUESTION: {query}\n\nANSWER:"
    messages.append({"role": "user", "content": user_content})
    return messages


# ── Ollama ───────────────────────────────────────────────────────────────────

def _ollama_generate(query, context_chunks, model, base_url, history=None):
    payload = json.dumps({
        "model": model,
        "messages": _build_messages(query, context_chunks, history),
        "stream": True,
        "options": {"temperature": 0.2, "top_p": 0.9, "num_ctx": 4096},
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            parts = []
            for line in resp.read().decode().splitlines():
                if not line.strip():
                    continue
                obj = _try_json_parse(line)
                parts.append(obj.get("message", {}).get("content", ""))
                if obj.get("done"):
                    break
            response_text = "".join(parts).strip()
            if not response_text:
                logger.warning("Ollama returned empty response for model %s", model)
                return "Ollama returned an empty response."
            return response_text

    except urllib.error.URLError as exc:
        logger.error("Cannot reach Ollama at %s: %s", base_url, exc)
        raise LLMError(f"Cannot reach Ollama at {base_url}. Ensure it is running.") from exc
    except Exception as exc:
        logger.error("Unexpected Ollama error: %s", exc)
        raise LLMError(f"Ollama error: {exc}") from exc


def _check_ollama(base_url: str) -> bool:
    try:
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=5)
        return True
    except Exception:
        return False


def _ollama_list_models(base_url: str) -> List[str]:
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=10) as r:
            data = _try_json_parse(r.read().decode())
            return [m["name"] for m in data.get("models", [])]
    except Exception as exc:
        raise LLMError(f"Ollama error: {exc}") from exc


# ── Groq ─────────────────────────────────────────────────────────────────────

def _groq_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _groq_generate(query, context_chunks, model, api_key, history=None):
    if not api_key:
        raise LLMError("Groq API key is not set. Add it in App Settings.")

    requests = _requests()
    payload = {
        "model": model,
        "messages": _build_messages(query, context_chunks, history),
        "temperature": 0.2,
        "max_tokens": 1024,
    }

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers=_groq_headers(api_key),
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    except requests.exceptions.HTTPError as exc:
        raise LLMError(f"Groq HTTP error: {exc}") from exc
    except requests.exceptions.ConnectionError as exc:
        raise LLMError(f"Groq connection error: {exc}") from exc
    except requests.exceptions.Timeout:
        raise LLMError("Groq timeout") 
    except Exception as exc:
        raise LLMError(f"Groq unexpected error: {exc}") from exc


def _check_groq(api_key: str) -> bool:
    if not api_key:
        raise LLMError("Groq API key not set")
    requests = _requests()
    resp = requests.get(GROQ_MODELS_URL, headers=_groq_headers(api_key), timeout=10)
    if resp.status_code == 200:
        return True
    if resp.status_code == 401:
        raise LLMError("Invalid Groq API key")
    return False


def _groq_list_models(api_key: str) -> List[str]:
    requests = _requests()
    resp = requests.get(GROQ_MODELS_URL, headers=_groq_headers(api_key), timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return sorted([m["id"] for m in data.get("data", [])])