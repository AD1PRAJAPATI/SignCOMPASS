#!/bin/bash
#SBATCH --job-name=sh_body
#SBATCH --array=0-99%50
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16GB
#SBATCH --partition=nlp_hiprio --time=08:00:00 --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/body_%A_%a.out
set -uo pipefail
source ~/shubert_paths.sh
module purge 2>/dev/null; unset LD_LIBRARY_PATH
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/feature_extraction
LIST=$WORK/pose.list
N=$(python -c "import gzip,pickle;print(len(pickle.loads(gzip.open('$LIST','rb').read())))")
CHUNKS=100; BS=$(( (N + CHUNKS - 1) / CHUNKS )); NB=$(( (N + BS - 1) / BS ))
if [ "$SLURM_ARRAY_TASK_ID" -ge "$NB" ]; then echo "no work (N=$N idx=$SLURM_ARRAY_TASK_ID)"; exit 0; fi
echo "N=$N BS=$BS NB=$NB idx=$SLURM_ARRAY_TASK_ID"
cd $REPO/features
python body_features.py --index $SLURM_ARRAY_TASK_ID --batch_size $BS \
  --files_list $LIST --pose_features_path $WORK/body_feats --time_limit 28000
