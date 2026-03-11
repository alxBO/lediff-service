#!/bin/bash
# Download LEDiff pretrained model weights from Google Drive
#
# Two models available:
#   - Highlight Hallucination: for overexposed/highlight recovery
#   - Shadow Hallucination: for underexposed/shadow recovery
#
# Usage:
#   ./download_weights.sh                    # Downloads BOTH models
#   ./download_weights.sh highlight          # Downloads highlight model only
#   ./download_weights.sh shadow             # Downloads shadow model only

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEIGHTS_DIR="$SCRIPT_DIR/weights"

# Google Drive file IDs
HIGHLIGHT_ID="1gd9KNmOQ3RH4yvX_Fp4hu2Jko64nbt2j"
SHADOW_ID="1tMk0rovHSt93wSeIiQ6fxsxPTCiZ6Dmw"

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown -q
fi

download_model() {
    local MODEL_TYPE="$1"
    local FILE_ID="$2"
    local DEST_DIR="$WEIGHTS_DIR/$MODEL_TYPE"

    # Skip if already present
    if [ -d "$DEST_DIR/sd_model" ] && [ -f "$DEST_DIR/merge_model.pth" ]; then
        echo "[$MODEL_TYPE] Already installed at $DEST_DIR — skipping."
        return 0
    fi

    echo ""
    echo "=== Downloading $MODEL_TYPE model ==="

    local TMPZIP="/tmp/lediff_${MODEL_TYPE}.zip"
    echo "Downloading from Google Drive (this may take a few minutes)..."
    gdown --id "$FILE_ID" -O "$TMPZIP"

    # Extract to temp dir
    local RAW_DIR="$DEST_DIR/_raw"
    echo "Extracting..."
    mkdir -p "$DEST_DIR"
    rm -rf "$RAW_DIR"
    unzip -q -o "$TMPZIP" -d "$RAW_DIR"
    rm "$TMPZIP"

    # Find the actual model directory (handle nested zip structures)
    local MODEL_ROOT
    MODEL_ROOT=$(find "$RAW_DIR" -name "model_index.json" -print -quit | xargs dirname 2>/dev/null || true)

    if [ -z "$MODEL_ROOT" ]; then
        MODEL_ROOT="$RAW_DIR"
        if [ ! -d "$MODEL_ROOT/vae" ]; then
            MODEL_ROOT=$(find "$RAW_DIR" -name "vae" -type d -print -quit | xargs dirname 2>/dev/null || true)
        fi
    fi

    if [ -z "$MODEL_ROOT" ] || [ ! -d "$MODEL_ROOT" ]; then
        echo "WARNING: Could not find model structure in the $MODEL_TYPE archive."
        echo "Please manually organize the files in $DEST_DIR/sd_model/"
        rm -rf "$RAW_DIR"
        return 1
    fi

    # Move to final location
    rm -rf "$DEST_DIR/sd_model"
    mv "$MODEL_ROOT" "$DEST_DIR/sd_model"
    rm -rf "$RAW_DIR"

    # Copy merge_model.pth to expected location
    if [ -f "$DEST_DIR/sd_model/vae/merge_model.pth" ]; then
        cp "$DEST_DIR/sd_model/vae/merge_model.pth" "$DEST_DIR/merge_model.pth"
    elif [ -f "$DEST_DIR/sd_model/merge_model.pth" ]; then
        cp "$DEST_DIR/sd_model/merge_model.pth" "$DEST_DIR/merge_model.pth"
    else
        echo "WARNING: merge_model.pth not found in $MODEL_TYPE archive."
        echo "Please locate it manually and place it at: $DEST_DIR/merge_model.pth"
        return 1
    fi

    echo "[$MODEL_TYPE] Installed at $DEST_DIR"
    ls -la "$DEST_DIR/sd_model/" 2>/dev/null || true
    return 0
}

# Determine what to download
MODEL_ARG="${1:-all}"

case "$MODEL_ARG" in
    highlight)
        download_model "highlight" "$HIGHLIGHT_ID"
        ;;
    shadow)
        download_model "shadow" "$SHADOW_ID"
        ;;
    all|"")
        download_model "highlight" "$HIGHLIGHT_ID"
        download_model "shadow" "$SHADOW_ID"
        ;;
    *)
        echo "Usage: $0 [highlight|shadow|all]"
        echo "  highlight  - Highlight recovery (overexposed images)"
        echo "  shadow     - Shadow recovery (underexposed images)"
        echo "  all        - Both models (default)"
        exit 1
        ;;
esac

echo ""
echo "Done! Weights directory:"
ls -la "$WEIGHTS_DIR/" 2>/dev/null || true
echo ""
echo "You can now run: ./run_mac.sh"
