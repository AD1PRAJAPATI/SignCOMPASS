#!/bin/bash
#SBATCH --job-name=sl_pose_test
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96GB
#SBATCH --partition=nlp --time=12:00:00 --account=jessetho_1732 --requeue
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/sl_pose_test_%j.out
set -uo pipefail
source ~/semlex_paths.sh
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
export PATH=/project2/jessetho_1732/aditeya/envs/cslr/bin:$PATH
module purge 2>/dev/null || true; unset LD_LIBRARY_PATH || true
cd "$ISLR"
python extract_pose_islr.py --video_dir "$WORK/test_pose_mp4" --out_dir "$SEMLEX/pose_features" \
  --ext .mp4 --workers 8 --model_complexity 1
echo "pose_features now: $(ls $SEMLEX/pose_features | wc -l)"
