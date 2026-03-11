#!/bin/bash
# Deploy LEDiff on a Vast.ai GPU instance
#
# === Vast.ai instance setup ===
#
# 1. Choose a GPU instance (RTX 3080+ recommended, 10+ GB VRAM)
# 2. Use a PyTorch template image (e.g. pytorch/pytorch:2.x-cuda12.x-runtime)
# 3. In "Docker options", add:  -p 8001:8001
# 4. Set disk space to at least 25 GB (both models)
#
# === On the instance ===
#
# SSH in, then:
#   git clone --recurse-submodules <repo-url>
#   cd lediff-service/service
#   ./deploy_vastai.sh
#
# === Access ===
#
# Option A: Click "Open" on the instance card (Cloudflare tunnel, HTTPS)
# Option B: Use direct IP:port from "IP Port Info" popup
#           (or use env var VAST_TCP_PORT_8001 for the external port)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== LEDiff Vast.ai Deployment ==="
echo ""

# 1. Install Python dependencies
echo "[1/3] Installing dependencies..."
pip install -q -r "$SCRIPT_DIR/backend/requirements.txt"
pip install -q -e "$REPO_DIR/vendor/LEDiff"

# 2. Check model weights — download both if missing
echo "[2/3] Checking model weights..."

HAS_MODEL=false
for MODEL_TYPE in highlight shadow; do
    if [ -d "$SCRIPT_DIR/weights/$MODEL_TYPE/sd_model" ] && [ -f "$SCRIPT_DIR/weights/$MODEL_TYPE/merge_model.pth" ]; then
        HAS_MODEL=true
        echo "  Found: $MODEL_TYPE"
    fi
done

if [ "$HAS_MODEL" = false ]; then
    echo "No models found. Downloading both models..."
    "$SCRIPT_DIR/download_weights.sh"

    # Verify at least one model was downloaded
    HAS_MODEL=false
    for MODEL_TYPE in highlight shadow; do
        if [ -d "$SCRIPT_DIR/weights/$MODEL_TYPE/sd_model" ] && [ -f "$SCRIPT_DIR/weights/$MODEL_TYPE/merge_model.pth" ]; then
            HAS_MODEL=true
        fi
    done

    if [ "$HAS_MODEL" = false ]; then
        echo ""
        echo "============================================================"
        echo "  Download failed. Please run manually:"
        echo "    ./download_weights.sh"
        echo ""
        echo "  Or see README.md for manual installation."
        echo "============================================================"
        echo ""
        exit 1
    fi
fi

# 3. Start the service
echo "[3/3] Starting service on port 8001..."
echo ""

# Show access info if running on Vast.ai
if [ -n "$VAST_TCP_PORT_8001" ]; then
    echo "Direct access: http://$(hostname -I | awk '{print $1}'):$VAST_TCP_PORT_8001"
fi
echo "Local: http://0.0.0.0:8001"
echo ""

cd "$SCRIPT_DIR/backend"
export LEDIFF_WEIGHTS_DIR="$SCRIPT_DIR/weights"
export MAX_MEGAPIXELS=50
export JOB_TTL_HOURS=24

exec uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 1
