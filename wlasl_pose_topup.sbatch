#!/bin/bash
#SBATCH --job-name=wlasl_pose
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=nlp
#SBATCH --time=12:00:00
#SBATCH --account=jessetho_1732
#SBATCH --requeue
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/wlasl_pose_%j.out
set -euo pipefail
export ISLR=/project2/jessetho_1732/aditeya/islr_pipeline
source ~/semlex_paths.sh; source "$ISLR/wlasl_env.sh"
module purge 2>/dev/null||true; unset LD_LIBRARY_PATH||true
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
cd "$ISLR"
python extract_pose_islr.py --video_dir "$VID" --out_dir "$WLASL/pose_features" --ext .mp4 --workers 8 --model_complexity 1
echo "pose_features: $(ls $WLASL/pose_features/*.pt 2>/dev/null|wc -l)"
