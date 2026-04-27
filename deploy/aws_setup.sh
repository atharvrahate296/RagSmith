#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# RAGSmith – AWS EC2 Bootstrap Script
# Run this ONCE on a fresh Amazon Linux 2023 / Ubuntu 22.04 t2.micro instance.
#
# Usage:
#   chmod +x deploy/aws_setup.sh
#   sudo bash deploy/aws_setup.sh
#
# What it does:
#   1. Installs system dependencies (Python, Nginx, Git, etc.)
#   2. Creates a 'ragsmith' system user
#   3. Clones the repo and sets up a Python venv
#   4. Installs Python packages
#   5. Mounts the EBS data volume (if attached)
#   6. Configures Nginx reverse proxy
#   7. Installs and starts the systemd service
#   8. Opens firewall ports 80 and 443
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config — edit these ───────────────────────────────────────────────────────
REPO_URL="https://github.com/YOUR_USERNAME/ragsmith.git"   # <── change this
APP_DIR="/home/ragsmith/app"
VENV_DIR="/home/ragsmith/venv"
DATA_VOLUME_DEVICE="/dev/xvdf"   # EBS volume device name (check AWS console)
DATA_MOUNT="/home/ragsmith/app/data"
# ─────────────────────────────────────────────────────────────────────────────

echo "============================================"
echo "  RAGSmith – EC2 Bootstrap"
echo "============================================"

# Detect distro
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
else
    DISTRO="unknown"
fi

echo "→ Detected OS: $DISTRO"

# ── 1. System packages ────────────────────────────────────────────────────────
echo "→ Installing system packages…"
if [[ "$DISTRO" == "amzn" ]]; then
    # Amazon Linux 2023
    dnf update -y
    dnf install -y python3.11 python3.11-pip python3.11-devel \
        nginx git gcc postgresql-devel \
        libmupdf-devel wget curl htop
    # Symlink python3 → python3.11
    alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
elif [[ "$DISTRO" == "ubuntu" ]]; then
    apt-get update -y
    apt-get install -y python3.11 python3.11-pip python3.11-venv python3.11-dev \
        nginx git gcc libpq-dev libmupdf-dev wget curl htop
fi

# ── 2. Create ragsmith user ───────────────────────────────────────────────────
echo "→ Creating ragsmith user…"
if ! id "ragsmith" &>/dev/null; then
    useradd --system --create-home --shell /bin/bash ragsmith
    echo "  Created user: ragsmith"
else
    echo "  User ragsmith already exists"
fi

# ── 3. Clone repository ───────────────────────────────────────────────────────
echo "→ Cloning repository…"
if [ -d "$APP_DIR" ]; then
    echo "  App directory exists, pulling latest…"
    cd "$APP_DIR" && sudo -u ragsmith git pull origin main
else
    sudo -u ragsmith git clone "$REPO_URL" "$APP_DIR"
fi

# ── 4. Python virtual environment ────────────────────────────────────────────
echo "→ Setting up Python venv…"
sudo -u ragsmith python3.11 -m venv "$VENV_DIR"
sudo -u ragsmith "$VENV_DIR/bin/pip" install --upgrade pip wheel
sudo -u ragsmith "$VENV_DIR/bin/pip" install -r "$APP_DIR/requirements.txt"
sudo -u ragsmith "$VENV_DIR/bin/pip" install gunicorn psycopg2-binary boto3

# ── 5. EBS Data Volume ────────────────────────────────────────────────────────
echo "→ Checking EBS data volume…"
if [ -b "$DATA_VOLUME_DEVICE" ]; then
    # Check if already formatted
    FSTYPE=$(blkid -o value -s TYPE "$DATA_VOLUME_DEVICE" 2>/dev/null || echo "")
    if [ -z "$FSTYPE" ]; then
        echo "  Formatting EBS volume as ext4…"
        mkfs -t ext4 "$DATA_VOLUME_DEVICE"
    fi

    mkdir -p "$DATA_MOUNT"
    mount "$DATA_VOLUME_DEVICE" "$DATA_MOUNT"

    # Add to fstab for auto-mount on reboot
    DEVICE_UUID=$(blkid -o value -s UUID "$DATA_VOLUME_DEVICE")
    if ! grep -q "$DEVICE_UUID" /etc/fstab; then
        echo "UUID=$DEVICE_UUID $DATA_MOUNT ext4 defaults,nofail 0 2" >> /etc/fstab
        echo "  Added to /etc/fstab"
    fi

    chown -R ragsmith:ragsmith "$DATA_MOUNT"
    echo "  EBS volume mounted at $DATA_MOUNT"
else
    echo "  No EBS volume at $DATA_VOLUME_DEVICE — using EC2 instance store (data lost on stop)"
    mkdir -p "$DATA_MOUNT"
    chown -R ragsmith:ragsmith "$DATA_MOUNT"
fi

# ── 6. Create data subdirectories ────────────────────────────────────────────
sudo -u ragsmith mkdir -p \
    "$APP_DIR/data/indexes" \
    "$APP_DIR/data/chunks" \
    "$APP_DIR/data/uploads" \
    "$APP_DIR/exports"

# ── 7. .env file ─────────────────────────────────────────────────────────────
if [ ! -f "$APP_DIR/.env" ]; then
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    chown ragsmith:ragsmith "$APP_DIR/.env"
    chmod 600 "$APP_DIR/.env"
    echo ""
    echo "  ⚠  IMPORTANT: Edit $APP_DIR/.env before starting the service!"
    echo "     nano $APP_DIR/.env"
    echo ""
fi

# ── 8. Nginx ──────────────────────────────────────────────────────────────────
echo "→ Configuring Nginx…"

# Copy static files location into nginx config
cp "$APP_DIR/nginx/ragsmith.conf" /etc/nginx/sites-available/ragsmith 2>/dev/null || \
    cp "$APP_DIR/nginx/ragsmith.conf" /etc/nginx/conf.d/ragsmith.conf

# Enable site (Debian/Ubuntu style)
if [ -d /etc/nginx/sites-enabled ]; then
    ln -sf /etc/nginx/sites-available/ragsmith /etc/nginx/sites-enabled/ragsmith
    rm -f /etc/nginx/sites-enabled/default
fi

nginx -t
systemctl enable nginx
systemctl restart nginx
echo "  Nginx configured ✓"

# ── 9. Systemd service ────────────────────────────────────────────────────────
echo "→ Installing systemd service…"
cp "$APP_DIR/deploy/ragsmith.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable ragsmith
echo "  Service installed ✓"

# ── 10. Firewall (firewalld on Amazon Linux, ufw on Ubuntu) ──────────────────
echo "→ Opening firewall ports 80 and 443…"
if command -v firewall-cmd &>/dev/null; then
    firewall-cmd --permanent --add-service=http
    firewall-cmd --permanent --add-service=https
    firewall-cmd --reload
elif command -v ufw &>/dev/null; then
    ufw allow 'Nginx Full'
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Bootstrap complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Edit your .env file:"
echo "     nano $APP_DIR/.env"
echo ""
echo "     Required settings:"
echo "       APP_ENV=production"
echo "       DB_DRIVER=postgres"
echo "       DATABASE_URL=postgresql://user:pass@your-rds-endpoint:5432/ragsmith"
echo "       LLM_PROVIDER=groq"
echo "       GROQ_API_KEY=your_groq_key"
echo "       STORAGE_BACKEND=s3"
echo "       S3_BUCKET_NAME=your-bucket-name"
echo "       AWS_REGION=us-east-1"
echo ""
echo "  2. Start the service:"
echo "     sudo systemctl start ragsmith"
echo ""
echo "  3. Check status:"
echo "     sudo systemctl status ragsmith"
echo "     sudo journalctl -u ragsmith -f"
echo ""
echo "  4. Visit http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
echo ""
