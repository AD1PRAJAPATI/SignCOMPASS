#!/bin/bash
#SBATCH --job-name=incl_pose
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64GB
#SBATCH --partition=nlp --time=12:00:00 --account=jessetho_1732 --requeue
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/include_pose_%j.out
set -uo pipefail
source ~/semlex_paths.sh
module purge 2>/dev/null || true; unset LD_LIBRARY_PATH || true
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
export PATH=/project2/jessetho_1732/aditeya/envs/cslr/bin:$PATH
ROOT=$(readlink -f /project2/jessetho_1732/aditeya/data/include)
cd "$ISLR"
# OpenCV reads .MOV; --ext match is case-insensitive
python extract_pose_islr.py \
  --video_dir "$ROOT/videos_mp4" \
  --out_dir "$ROOT/pose_features" \
  --ext .mov \
  --workers 8 --model_complexity 1
echo "pose_features count: $(ls "$ROOT/pose_features"/*.pt 2>/dev/null | wc -l)"
