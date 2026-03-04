#!/usr/bin/env bash
# RAGSmith startup script

set -euo pipefail

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
WORKERS="${WORKERS:-1}"

echo "╔══════════════════════════════════════════╗"
echo "║          ⬡  RAGSmith v1.0                ║"
echo "║   Open-Source Local RAG Builder           ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Check Python version
python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "→ Python $python_version"

# Check Ollama
if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "→ Ollama: running ✓"
else
  echo "→ Ollama: NOT running ⚠"
  echo "  Install Ollama from https://ollama.com and run: ollama pull mistral"
fi

echo ""
echo "→ Starting on http://$HOST:$PORT"
echo ""

python3 -m uvicorn main:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers "$WORKERS" \
  --reload \
  --log-level info
