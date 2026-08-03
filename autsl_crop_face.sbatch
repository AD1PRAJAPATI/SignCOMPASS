#!/bin/bash
#SBATCH --job-name=incl_cropf
#SBATCH --array=0-99%40
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16GB
#SBATCH --partition=nlp --time=08:00:00 --account=jessetho_1732 --requeue
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/incl_cropf_%A_%a.out
set -uo pipefail
source ~/semlex_paths.sh
source "$ISLR/autsl_env.sh"
module purge 2>/dev/null; unset LD_LIBRARY_PATH
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/feature_extraction
LIST=$WORK/face_remaining.list
N=$(python -c "import gzip,pickle;print(len(pickle.loads(gzip.open('$LIST','rb').read())))")
CHUNKS=100; BS=$(( (N + CHUNKS - 1) / CHUNKS )); NB=$(( (N + BS - 1) / BS ))
if [ "$SLURM_ARRAY_TASK_ID" -ge "$NB" ]; then echo "no work"; exit 0; fi
echo "INCLUDE crop_face N=$N BS=$BS idx=$SLURM_ARRAY_TASK_ID"
cd $REPO/dataset
python crop_face.py --index $SLURM_ARRAY_TASK_ID --batch_size $BS --files_list $LIST \
  --pose_path $WORK/pose --face_path $WORK/face_crops \
  --problem_file_path $WORK/prob_face_${SLURM_ARRAY_TASK_ID}.txt --time_limit 28000
