#!/bin/bash
#SBATCH --job-name=islr_asl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nlp_hiprio
#SBATCH --time=24:00:00
#SBATCH --account=jessetho_1732

# ─────────────────────────────────────────────────────────────────────────────
# ISLR training pipeline on ASL-Citizen
#
# Stages:
#   0. Verify/download ASL-Citizen data
#   1. Extract MediaPipe pose features from MP4 videos (CPU, parallelised)
#   2. Train ISLRModel (GPU, ArcFace, 2731 classes)
#   3. Print gate verdict: top-1 > 63%? R@10 > 91%?
#
# Run with:
#   sbatch train_islr.sbatch
#   sbatch train_islr.sbatch --export=ONLY_EXTRACT=1   # pose extraction only
#   sbatch train_islr.sbatch --export=ONLY_TRAIN=1     # training only (assumes extracted)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
TS=$(date +"%Y%m%d_%H%M%S")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR="/project2/jessetho_1732/aditeya/islr_pipeline"
DATA_DIR="/project2/jessetho_1732/aditeya/data/asl_citizen"
mkdir -p "${BASE_DIR}/logs"

# Redirect stdout+stderr to timestamped log file
exec > "${BASE_DIR}/logs/islr_${TS}.out" 2> "${BASE_DIR}/logs/islr_${TS}.err"

echo "=================================================="
echo "Job: ISLR ASL-Citizen"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=================================================="
nvidia-smi || true

# ── Environment ───────────────────────────────────────────────────────────────
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
export PATH=/project2/jessetho_1732/aditeya/envs/cslr/bin:$PATH

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

cd "$BASE_DIR"

# ── Step 0: Verify / download ASL-Citizen ─────────────────────────────────────
if [[ "${ONLY_TRAIN:-0}" != "1" ]]; then
    echo ""
    echo "=== Step 0: Verify ASL-Citizen data ==="

    if [[ ! -f "${DATA_DIR}/metadata.csv" ]]; then
        echo "metadata.csv not found — downloading ASL-Citizen..."
        echo "NOTE: Download requires internet access (run on a login node if blocked)."
        python download_asl_citizen.py
    else
        echo "metadata.csv found. Verifying..."
        python download_asl_citizen.py --verify
    fi
fi

# ── Step 1: Extract pose features ─────────────────────────────────────────────
if [[ "${ONLY_TRAIN:-0}" != "1" ]]; then
    echo ""
    echo "=== Step 1: Extract MediaPipe pose features ==="
    FEAT_DIR="${DATA_DIR}/pose_features"

    if [[ -d "${FEAT_DIR}" ]]; then
        N_FEATS=$(find "${FEAT_DIR}" -maxdepth 1 -name "*.pt" | wc -l | xargs)
        N_VIDS=$(find "${DATA_DIR}/videos" -maxdepth 1 -name "*.mp4" | wc -l | xargs)
        echo "Pose features: ${N_FEATS} / ${N_VIDS} videos already extracted."
        if [[ "${N_FEATS}" -ge "${N_VIDS}" ]]; then
            echo "All features present — skipping extraction."
        else
            python extract_pose_islr.py --workers 8 --model_complexity 1
        fi
    else
        echo "No pose features found — extracting all..."
        python extract_pose_islr.py --workers 8 --model_complexity 1
    fi

    if [[ "${ONLY_EXTRACT:-0}" == "1" ]]; then
        echo "ONLY_EXTRACT=1 — stopping after extraction."
        exit 0
    fi
fi

# ── Step 2: Train ISLR model ──────────────────────────────────────────────────
echo ""
echo "=== Step 2: Train ISLRModel (base, seed=42) ==="

python train_islr.py \
    --size        base    \
    --epochs      120     \
    --warmup_epochs 5     \
    --batch_size  128     \
    --lr          5e-4    \
    --wd          1e-4    \
    --dropout     0.2     \
    --max_frames  64      \
    --patience    20      \
    --num_workers 8       \
    --seed        42      \
    --save_dir    "${BASE_DIR}/checkpoints_islr"

# ── Step 3: Optionally run a second seed for variance ─────────────────────────
echo ""
echo "=== Step 3: Train ISLRModel (base, seed=123) ==="

python train_islr.py \
    --size        base    \
    --epochs      120     \
    --warmup_epochs 5     \
    --batch_size  128     \
    --lr          5e-4    \
    --wd          1e-4    \
    --dropout     0.2     \
    --max_frames  64      \
    --patience    20      \
    --num_workers 8       \
    --seed        123     \
    --save_dir    "${BASE_DIR}/checkpoints_islr"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Summary ==="
grep -h "ISLR_RESULT" "${BASE_DIR}/logs/islr_${TS}.out" || true

echo ""
echo "=== Gate Check ==="
# Extract test top-1 from both seeds and compare vs 63% baseline
grep "test_top1" "${BASE_DIR}/logs/islr_${TS}.out" | awk -F'test_top1=' '{print $2}' | awk '{print "top-1:", $1}' || true

echo ""
echo "Job finished: $(date)"