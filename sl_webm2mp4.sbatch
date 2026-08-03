#!/bin/bash
#SBATCH --job-name=sl_conv
#SBATCH --array=0-49%50
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8GB
#SBATCH --partition=nlp --time=12:00:00 --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/conv_%A_%a.out
set -uo pipefail
source ~/semlex_paths.sh
module purge 2>/dev/null; unset LD_LIBRARY_PATH
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/feature_extraction
SRC=$SEMLEX/videos; OUT=$SEMLEX/videos_mp4; mkdir -p $OUT
N=50; I=$SLURM_ARRAY_TASK_ID; i=0
for f in $SRC/*.webm; do
  if (( i % N == I )); then
    b=$(basename "$f" .webm)
    [ -f "$OUT/$b.mp4" ] || ffmpeg -y -loglevel error -i "$f" -an -c:v libx264 -preset veryfast "$OUT/$b.mp4" </dev/null || echo "FAIL $b"
  fi
  i=$((i+1))
done
echo "task $I done"
