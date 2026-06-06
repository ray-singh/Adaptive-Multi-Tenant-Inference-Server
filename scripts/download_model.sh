#!/usr/bin/env bash
# Downloads SmolLM2-135M-Instruct (Q4_K_M GGUF, ~270 MB) into models/.
# SmolLM2-135M is fast enough for CPU inference and small enough for laptops.
set -euo pipefail

REPO="bartowski/SmolLM2-135M-Instruct-GGUF"
FILE="SmolLM2-135M-Instruct-Q4_K_M.gguf"
DEST="models/${FILE}"
URL="https://huggingface.co/${REPO}/resolve/main/${FILE}"

mkdir -p models

if [[ -f "${DEST}" ]]; then
    echo "Already downloaded: ${DEST}"
else
    echo "Downloading ${FILE} (~270 MB)..."
    curl -L --progress-bar "${URL}" -o "${DEST}"
    echo "Saved to ${DEST}"
fi

echo ""
echo "Start the server with:"
echo "  MODEL_PATH=$(pwd)/${DEST} ./build/inference_server"
