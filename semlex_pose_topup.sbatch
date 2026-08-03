#!/bin/bash
#SBATCH --job-name=sl_pose_tv
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=128GB
#SBATCH --partition=nlp --time=2-00:00:00 --account=jessetho_1732 --requeue
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/sl_pose_topup_%j.out
set -euo pipefail
source ~/semlex_paths.sh
module purge 2>/dev/null || true; unset LD_LIBRARY_PATH || true
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
cd "$ISLR"
N=$(ls "$WORK/trainval_pose_mp4" 2>/dev/null | wc -l)
echo "extracting pose for N=$N videos (workers=4, 48h wall)"
if [ "$N" -eq 0 ]; then echo "nothing to do"; exit 0; fi
# fewer workers → less RAM (previous job OOMed with 8)
python extract_pose_islr.py --video_dir "$WORK/trainval_pose_mp4" \
  --out_dir "$SEMLEX/pose_features" --ext .mp4 --workers 4 --model_complexity 1
echo "pose_features now: $(ls $SEMLEX/pose_features/*.pt 2>/dev/null | wc -l)"
