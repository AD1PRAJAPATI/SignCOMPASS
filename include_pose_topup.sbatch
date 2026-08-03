#!/bin/bash
#SBATCH --job-name=incl_pose2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64GB
#SBATCH --partition=nlp --time=12:00:00 --account=jessetho_1732 --requeue
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/include_pose_%j.out
set -euo pipefail
source ~/semlex_paths.sh
source "$ISLR/include_env.sh"
module purge 2>/dev/null || true; unset LD_LIBRARY_PATH || true
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
cd "$ISLR"
for ext in .MOV .mov .mp4; do
  n=$(ls "$INCLUDE/_pose_todo"/*"$ext" 2>/dev/null | wc -l || true)
  [ "${n:-0}" -eq 0 ] && continue
  echo "extracting $n *$ext"
  python extract_pose_islr.py --video_dir "$INCLUDE/_pose_todo" \
    --out_dir "$INCLUDE/pose_features" --ext "$ext" --workers 8 --model_complexity 1
done
echo "pose_features: $(ls $INCLUDE/pose_features/*.pt 2>/dev/null | wc -l)"
