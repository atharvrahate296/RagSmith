"""
RAGSmith – LLM inference service
Supports two providers, switchable via LLM_PROVIDER env var:

  ollama → Local Ollama (MIT)  — dev / self-hosted
  groq   → Groq Cloud API      — AWS deployment (free tier)

Uses `requests` for Groq calls — urllib gets blocked by Cloudflare (error 1010).
Ollama stays on urllib since it's local and has no Cloudflare.
"""

import json
import logging
import urllib.request
import urllib.error
from typing import List, Tuple

logger = logging.getLogger("ragsmith.llm")

SYSTEM_PROMPT = (
    "You are RAGSmith, a helpful AI assistant. "
    "Answer the user's question using ONLY the context provided below. "
    "If the context does not contain enough information, say so honestly. "
    "Do not hallucinate or invent facts. Be concise and accurate.\n"
)

GROQ_API_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"

# requests is already in requirements.txt (pulled in by sentence-transformers/huggingface)
# It has proper TLS + User-Agent headers that pass Cloudflare — urllib doesn't.
def _requests():
    try:
        import requests
        return requests
    except ImportError as exc:
        raise RuntimeError("requests not installed. Run: pip install requests") from exc


# ── Public API ────────────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    context_chunks: List[Tuple[str, float, str]],
    model: str = "",
) -> str:
    """Generate a RAG-grounded answer. model overrides config default if given."""
    from config import get_settings
    cfg = get_settings()
    effective_model = model or cfg.effective_llm_model
    if cfg.llm_provider == "groq":
        return _groq_generate(query, context_chunks, effective_model, cfg.groq_api_key)
    return _ollama_generate(query, context_chunks, effective_model, cfg.ollama_base_url)


def check_llm_available() -> dict:
    """Return availability status for the /health endpoint status indicator."""
    from config import get_settings
    cfg = get_settings()
    if cfg.llm_provider == "groq":
        ok = _check_groq(cfg.groq_api_key)
        return {"available": ok, "provider": "groq",
                "detail": "Groq API reachable" if ok else "Groq API unreachable or invalid key"}
    ok = _check_ollama(cfg.ollama_base_url)
    return {"available": ok, "provider": "ollama",
            "detail": f"Ollama at {cfg.ollama_base_url}" if ok else "Ollama not running"}


def list_available_models() -> List[str]:
    from config import get_settings
    cfg = get_settings()
    if cfg.llm_provider == "groq":
        return _groq_list_models(cfg.groq_api_key)
    return _ollama_list_models(cfg.ollama_base_url)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _build_user_message(query: str, context_chunks: List[Tuple[str, float, str]]) -> str:
    if context_chunks:
        parts = [f"[Source {i}: {fn} | relevance: {s:.3f}]\n{t}"
                 for i, (t, s, fn) in enumerate(context_chunks, 1)]
        ctx = "\n\n---\n\n".join(parts)
    else:
        ctx = "No relevant context found."
    return f"CONTEXT:\n{ctx}\n\nQUESTION: {query}\n\nANSWER:"


# ── Ollama (urllib is fine — local, no Cloudflare) ────────────────────────────

def _ollama_generate(query, context_chunks, model, base_url):
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_message(query, context_chunks)},
        ],
        "stream": True,
        "options": {"temperature": 0.2, "top_p": 0.9, "num_ctx": 4096},
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/api/chat", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            parts = []
            for line in resp.read().decode().splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    parts.append(obj.get("message", {}).get("content", ""))
                    if obj.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
            return "".join(parts).strip() or "Ollama returned an empty response."
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Cannot reach Ollama at {base_url}. "
            "Ensure Ollama is running: https://ollama.com"
        ) from exc


def _check_ollama(base_url: str) -> bool:
    try:
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=5)
        return True
    except Exception:
        return False


def _ollama_list_models(base_url: str) -> List[str]:
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=10) as r:
            return [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception:
        return []


# ── Groq (requests — passes Cloudflare, urllib gets 403/1010) ────────────────

def _groq_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _groq_generate(query, context_chunks, model, api_key):
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com"
        )

    requests = _requests()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_message(query, context_chunks)},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
        "stream": False,
    }

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers=_groq_headers(api_key),
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()
        logger.info("Groq: model=%s tokens=%s",
                    result.get("model", model),
                    result.get("usage", {}).get("total_tokens", "?"))
        return result["choices"][0]["message"]["content"].strip()

    except requests.exceptions.HTTPError as exc:
        body = exc.response.text if exc.response is not None else str(exc)
        raise RuntimeError(f"Groq API error {exc.response.status_code}: {body}") from exc
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(f"Cannot reach Groq API: {exc}") from exc
    except requests.exceptions.Timeout as exc:
        raise ConnectionError("Groq API request timed out after 60s") from exc


def _check_groq(api_key: str) -> bool:
    """
    Ping Groq /models to verify key + connectivity.
    Only used for /health indicator — does NOT gate queries.
    """
    if not api_key:
        return False
    requests = _requests()
    try:
        resp = requests.get(
            GROQ_MODELS_URL,
            headers=_groq_headers(api_key),
            timeout=10,
        )
        return resp.status_code == 200
    except Exception:
        return False


def _groq_list_models(api_key: str) -> List[str]:
    if not api_key:
        return []
    requests = _requests()
    try:
        resp = requests.get(
            GROQ_MODELS_URL,
            headers=_groq_headers(api_key),
            timeout=10,
        )
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]
    except Exception:
        return []
