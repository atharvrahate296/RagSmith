"""
RAGSmith – Local LLM inference via Ollama (MIT)

Ollama must be running locally: https://ollama.com
Default model: mistral (Apache 2.0)
Alternatives: llama3, phi3, gemma, etc.
"""

import json
import logging
import urllib.request
import urllib.error
from typing import List, Tuple

logger = logging.getLogger("ragsmith.llm")

OLLAMA_BASE_URL = "http://localhost:11434"
SYSTEM_PROMPT = (
    "You are RAGSmith, a helpful AI assistant. "
    "Answer the user's question using ONLY the context provided below. "
    "If the context does not contain enough information, say so honestly. "
    "Do not hallucinate or invent facts. Be concise and accurate.\n"
)


def _ollama_request(payload: dict) -> dict:
    """Send a POST request to the Ollama API."""
    url = f"{OLLAMA_BASE_URL}/api/chat"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
            # Ollama streams NDJSON; collect all content pieces
            content_parts = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "message" in obj and "content" in obj["message"]:
                        content_parts.append(obj["message"]["content"])
                    if obj.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
            return {"content": "".join(content_parts)}
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
            "Make sure Ollama is running: https://ollama.com"
        ) from exc


def generate_answer(
    query: str,
    context_chunks: List[Tuple[str, float, str]],
    model: str = "mistral",
) -> str:
    """
    Generate a RAG-grounded answer using the local Ollama model.

    Parameters
    ----------
    query          : User question
    context_chunks : List of (text, score, filename) retrieved chunks
    model          : Ollama model name

    Returns
    -------
    str  Generated answer
    """
    if not context_chunks:
        context_str = "No relevant context found."
    else:
        context_parts = []
        for i, (text, score, filename) in enumerate(context_chunks, 1):
            context_parts.append(
                f"[Source {i}: {filename} | relevance: {score:.3f}]\n{text}"
            )
        context_str = "\n\n---\n\n".join(context_parts)

    user_message = (
        f"CONTEXT:\n{context_str}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER:"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "stream": True,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_ctx": 4096,
        },
    }

    logger.info("Calling Ollama model '%s' …", model)
    result = _ollama_request(payload)
    answer = result.get("content", "").strip()

    if not answer:
        answer = "I was unable to generate a response. Please check your Ollama installation."

    return answer


def list_available_models() -> List[str]:
    """Return models currently available in the local Ollama instance."""
    url = f"{OLLAMA_BASE_URL}/api/tags"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def check_ollama_running() -> bool:
    """Check whether Ollama is reachable."""
    try:
        url = f"{OLLAMA_BASE_URL}/api/tags"
        urllib.request.urlopen(url, timeout=5)
        return True
    except Exception:
        return False
