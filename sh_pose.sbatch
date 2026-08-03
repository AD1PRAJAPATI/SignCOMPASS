#!/bin/bash
#SBATCH --job-name=sh_pose
#SBATCH --array=0-99%50          # 100 chunks, up to 50 running at once
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --partition=nlp_hiprio
#SBATCH --time=08:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/pose_%A_%a.out

set -uo pipefail
source ~/shubert_paths.sh
module purge 2>/dev/null
unset LD_LIBRARY_PATH
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/feature_extraction

# 83,399 videos / 100 chunks -> 834 per chunk. index = this array task.
python $REPO/dataset/kpe_mediapipe.py \
  --index $SLURM_ARRAY_TASK_ID --batch_size 834 \
  --files_list $WORK/videos.list --pose_path $WORK/pose --stats_path $WORK/stats \
  --problem_file_path $WORK/prob_pose_${SLURM_ARRAY_TASK_ID}.txt --time_limit 28000 \
  --face_model_path $WORK/mp_models/face_landmarker.task \
  --hand_model_path $WORK/mp_models/hand_landmarker.task
