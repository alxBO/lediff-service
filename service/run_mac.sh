#!/bin/bash
# Run LEDiff service natively on Mac (with MPS GPU acceleration)
#
# Prerequisites:
#   1. pip install -r backend/requirements.txt
#   2. pip install -e ../vendor/LEDiff
#   3. Place model weights in weights/ directory (see README)
#
# Usage: ./run_mac.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Install LEDiff custom diffusers if not already
if ! python -c "import diffusers; print(diffusers.__file__)" 2>/dev/null | grep -q "LEDiff"; then
    echo "Installing LEDiff custom diffusers..."
    pip install -e "$REPO_DIR/vendor/LEDiff" -q
fi

export LEDIFF_WEIGHTS_DIR="$SCRIPT_DIR/weights"
export MAX_MEGAPIXELS=50
export JOB_TTL_HOURS=24

# Check at least one model is present
HAS_MODEL=false
for MODEL_TYPE in highlight shadow; do
    if [ -d "$LEDIFF_WEIGHTS_DIR/$MODEL_TYPE/sd_model" ] && [ -f "$LEDIFF_WEIGHTS_DIR/$MODEL_TYPE/merge_model.pth" ]; then
        HAS_MODEL=true
        echo "Found model: $MODEL_TYPE"
    fi
done

if [ "$HAS_MODEL" = false ]; then
    echo ""
    echo "============================================================"
    echo "  No LEDiff models found!"
    echo ""
    echo "  Download pretrained weights:"
    echo "    ./download_weights.sh             # Both models"
    echo "    ./download_weights.sh highlight   # Highlight recovery (overexposed)"
    echo "    ./download_weights.sh shadow      # Shadow recovery (underexposed)"
    echo ""
    echo "  Or see README.md for manual installation."
    echo "============================================================"
    echo ""
    exit 1
fi

echo ""
echo "Starting LEDiff on http://localhost:8001"
echo "Backend: PyTorch (MPS/CPU auto-detect)"
echo ""

cd "$SCRIPT_DIR/backend"
exec uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 1
