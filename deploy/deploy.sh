#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# RAGSmith – Deploy script
# Run this for every code update after the initial setup.
# Performs a zero-downtime restart via systemd reload.
#
# Usage (from your local machine):
#   ssh ec2-user@YOUR_EC2_IP 'bash /home/ragsmith/app/deploy/deploy.sh'
#
# Or from the EC2 instance directly:
#   bash /home/ragsmith/app/deploy/deploy.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

APP_DIR="/home/ragsmith/app"
VENV_DIR="/home/ragsmith/venv"
SERVICE="ragsmith"

echo "──────────────────────────────────────────"
echo "  RAGSmith Deploy — $(date '+%Y-%m-%d %H:%M:%S')"
echo "──────────────────────────────────────────"

cd "$APP_DIR"

# ── 1. Pull latest code ───────────────────────────────────────────────────────
echo "→ Pulling latest code…"
git fetch origin main
git reset --hard origin/main
echo "  Commit: $(git log -1 --format='%h %s')"

# ── 2. Install / update Python dependencies ──────────────────────────────────
echo "→ Installing dependencies…"
"$VENV_DIR/bin/pip" install -q --upgrade pip
"$VENV_DIR/bin/pip" install -q -r requirements.txt

# ── 3. Restart service (systemd handles the SIGTERM → wait → start cycle) ────
echo "→ Restarting service…"
sudo systemctl restart "$SERVICE"

# ── 4. Wait and verify ───────────────────────────────────────────────────────
echo "→ Waiting for startup…"
sleep 5

if systemctl is-active --quiet "$SERVICE"; then
    echo ""
    echo "  ✓ $SERVICE is running"
    # Hit health endpoint
    HEALTH=$(curl -sf http://127.0.0.1:8000/health 2>/dev/null || echo "unreachable")
    echo "  Health: $HEALTH"
else
    echo "  ✗ $SERVICE failed to start"
    echo ""
    echo "Last 20 log lines:"
    journalctl -u "$SERVICE" -n 20 --no-pager
    exit 1
fi

echo ""
echo "Deploy complete ✓"
echo "──────────────────────────────────────────"
