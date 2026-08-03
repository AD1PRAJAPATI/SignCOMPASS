#!/bin/bash
# setup_shubert.sh  —  Clone SHuBERT repo, download weights, install dependencies.
#
# Run ONCE on the cluster before extract_shubert_features.py.
#
# Usage:
#   bash setup_shubert.sh
#
# What it does:
#   1. Clones ShesterG/SHuBERT into $SHUBERT_DIR
#   2. Installs the fairseq fork (required by SHuBERT)
#   3. Downloads pretrained weights from Google Drive via gdown:
#        - shubert.pt          (the main SHuBERT encoder)
#        - dino_face.pt        (fine-tuned DINOv2 for face crops)
#        - dino_left_hand.pt   (fine-tuned DINOv2 for left-hand crops)
#        - dino_right_hand.pt  (fine-tuned DINOv2 for right-hand crops)
#   4. Verifies everything looks right.

set -euo pipefail

# ── Paths — edit if needed ─────────────────────────────────────────────────────
BASE_DIR="/project2/jessetho_1732/aditeya"
SHUBERT_DIR="${BASE_DIR}/SHuBERT"
WEIGHTS_DIR="${BASE_DIR}/data/shubert_weights"
CONDA_ENV="${BASE_DIR}/envs/cslr"    # reuse existing env if possible

# Google Drive folder ID from the SHuBERT README
# https://drive.google.com/drive/folders/1aOZEkENp2B-5sRq5F67dYsirnHwsFjKV
GDRIVE_FOLDER_ID="1aOZEkENp2B-5sRq5F67dYsirnHwsFjKV"

echo "=================================================="
echo "SHuBERT Setup"
echo "  SHuBERT repo:  ${SHUBERT_DIR}"
echo "  Weights:       ${WEIGHTS_DIR}"
echo "=================================================="

mkdir -p "${WEIGHTS_DIR}"

# ── Step 1: Clone SHuBERT ─────────────────────────────────────────────────────
if [[ -d "${SHUBERT_DIR}" ]]; then
    echo "[Step 1] SHuBERT already cloned at ${SHUBERT_DIR}"
else
    echo "[Step 1] Cloning SHuBERT..."
    git clone https://github.com/ShesterG/SHuBERT.git "${SHUBERT_DIR}"
fi

# ── Step 2: Install fairseq fork ──────────────────────────────────────────────
# The fairseq fork requires omegaconf 2.0.x which has invalid metadata in pip>=24.
# Fix: install omegaconf with an older pip via a temp venv, then install fairseq
# with --no-deps and manually add the handful of required packages.
echo ""
echo "[Step 2] Installing SHuBERT's fairseq fork (pip>=24 workaround)..."
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

# 2a: Install omegaconf using a pinned pip that accepts the old metadata
pip install "pip==23.3.2" --quiet                    # downgrade pip temporarily
pip install "omegaconf==2.0.6" --quiet               # now installs fine
pip install "pip>=24" --quiet                         # restore modern pip

# 2b: Install fairseq fork itself without touching deps (already handled above)
cd "${SHUBERT_DIR}/fairseq"
pip install -e . --no-deps --quiet
cd "${BASE_DIR}"

# 2c: Install the remaining fairseq deps that are safe on modern pip
pip install \
    "hydra-core==1.0.7" \
    "antlr4-python3-runtime==4.8" \
    "portalocker>=2.0" \
    "sacrebleu>=1.4.12" \
    "sentencepiece" \
    --quiet || true    # non-fatal: cslr env may already have some of these

# ── Step 3: Download weights via gdown ────────────────────────────────────────
echo ""
echo "[Step 3] Downloading SHuBERT weights from Google Drive..."
echo "  Folder: https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}"

# Install gdown if not available
pip install gdown --quiet --break-system-packages 2>/dev/null || true

# Download entire folder
gdown --folder "https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}" \
      --output "${WEIGHTS_DIR}" || {
    echo ""
    echo "[ERROR] gdown failed. Manual download instructions:"
    echo "  1. Open: https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}"
    echo "  2. Download all files to: ${WEIGHTS_DIR}/"
    echo "     Expected files: shubert.pt, dino_face.pt, dino_left_hand.pt, dino_right_hand.pt"
    echo "  3. Re-run this script or skip to extract_shubert_features.py"
    exit 1
}

# ── Step 4: Normalise filenames ───────────────────────────────────────────────
# The Drive folder uses different names than our code expects. Create symlinks.
echo ""
echo "[Step 4] Normalising weight filenames..."

cd "${WEIGHTS_DIR}"

# SHuBERT encoder: checkpoint_836_400000.pt → shubert.pt
[[ ! -f shubert.pt && -f checkpoint_836_400000.pt ]] && \
    ln -sf checkpoint_836_400000.pt shubert.pt && echo "  Linked shubert.pt"

# Face DINOv2: face_dinov2_checkpoint.pth → dino_face.pt
[[ ! -f dino_face.pt && -f face_dinov2_checkpoint.pth ]] && \
    ln -sf face_dinov2_checkpoint.pth dino_face.pt && echo "  Linked dino_face.pt"

# Hands DINOv2: ONE model used for BOTH hands
[[ ! -f dino_left_hand.pt && -f hands_dinov2_checkpoint.pth ]] && \
    ln -sf hands_dinov2_checkpoint.pth dino_left_hand.pt && echo "  Linked dino_left_hand.pt"
[[ ! -f dino_right_hand.pt && -f hands_dinov2_checkpoint.pth ]] && \
    ln -sf hands_dinov2_checkpoint.pth dino_right_hand.pt && echo "  Linked dino_right_hand.pt"

cd "${BASE_DIR}"

echo ""
echo "[Step 5] Verifying weights..."
for f in shubert.pt dino_face.pt dino_left_hand.pt dino_right_hand.pt; do
    path="${WEIGHTS_DIR}/${f}"
    if [[ -f "${path}" ]]; then
        size=$(du -sh "${path}" | cut -f1)
        echo "  ✅ ${f}  (${size})"
    else
        echo "  ❌ ${f} not found"
    fi
done

echo ""
echo "=================================================="
echo "Setup complete!"
echo ""
echo "Next:"
echo "  python extract_shubert_features.py"
echo "=================================================="
