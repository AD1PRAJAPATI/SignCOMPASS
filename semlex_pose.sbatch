#!/bin/bash
#SBATCH --job-name=sl_pose_islr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --partition=nlp
#SBATCH --time=24:00:00
#SBATCH --account=jessetho_1732

set -euo pipefail
source ~/semlex_paths.sh
BASE_DIR="/project2/jessetho_1732/aditeya/islr_pipeline"
mkdir -p "${BASE_DIR}/logs"
TS=$(date +"%Y%m%d_%H%M%S")
exec > "${BASE_DIR}/logs/sl_pose_${TS}.out" 2> "${BASE_DIR}/logs/sl_pose_${TS}.err"

echo "Host: $(hostname) | Date: $(date)"
echo "VID=$VID"
echo "OUT=$SEMLEX/pose_features"

# env (CPU only — MediaPipe needs no GPU)
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
export PATH=/project2/jessetho_1732/aditeya/envs/cslr/bin:$PATH
module purge 2>/dev/null || true
unset LD_LIBRARY_PATH || true

cd "$BASE_DIR"
python extract_pose_islr.py \
    --video_dir "$VID" \
    --out_dir   "$SEMLEX/pose_features" \
    --ext       .mp4 \
    --workers   16 \
    --model_complexity 1

echo "Done: $(date)"
echo "count: $(ls $SEMLEX/pose_features/*.pt 2>/dev/null | wc -l)"
