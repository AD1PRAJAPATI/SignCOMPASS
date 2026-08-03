#!/bin/bash
#SBATCH --job-name=sl_pose_islr
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96GB
#SBATCH --partition=nlp --time=24:00:00 --account=jessetho_1732
set -euo pipefail
source ~/semlex_paths.sh
TS=$(date +"%Y%m%d_%H%M%S")
exec > "$ISLR/logs/sl_pose_${TS}.out" 2> "$ISLR/logs/sl_pose_${TS}.err"
echo "Host: $(hostname) | $(date)"
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
export PATH=/project2/jessetho_1732/aditeya/envs/cslr/bin:$PATH
module purge 2>/dev/null || true; unset LD_LIBRARY_PATH || true
cd "$ISLR"
python extract_pose_islr.py --video_dir "$VID" --out_dir "$SEMLEX/pose_features" \
  --ext .mp4 --workers 8 --model_complexity 1
echo "Done $(date) | count: $(ls $SEMLEX/pose_features/*.pt 2>/dev/null | wc -l)"
