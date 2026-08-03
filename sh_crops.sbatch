#!/bin/bash
#SBATCH --job-name=sh_crops
#SBATCH --array=0-99%50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --partition=nlp_hiprio
#SBATCH --time=08:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/crops_%A_%a.out

set -uo pipefail
source ~/shubert_paths.sh
module purge 2>/dev/null
unset LD_LIBRARY_PATH
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/feature_extraction

I=$SLURM_ARRAY_TASK_ID
cd $REPO/dataset
# face crops -> <id>_face.mp4
python crop_face.py  --index $I --batch_size 834 --files_list $WORK/videos.list \
  --pose_path $WORK/pose --face_path $WORK/face_crops \
  --problem_file_path $WORK/prob_face_${I}.txt --time_limit 28000
# hand crops -> <id>_hand1.mp4 / <id>_hand2.mp4
python crop_hands.py --index $I --batch_size 834 --files_list $WORK/videos.list \
  --pose_path $WORK/pose --hand_path $WORK/hand_crops \
  --problem_file_path $WORK/prob_hand_${I}.txt --time_limit 28000
