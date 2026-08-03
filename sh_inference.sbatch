#!/bin/bash
#SBATCH --job-name=sh_infer
#SBATCH --array=0-99%30
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24GB --gres=gpu:a100:1
#SBATCH --partition=nlp_hiprio --time=08:00:00 --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/infer_%A_%a.out
set -uo pipefail
source ~/shubert_paths.sh
module purge 2>/dev/null; unset LD_LIBRARY_PATH
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
export PYTHONPATH="${PYTHONPATH:-}:$REPO/fairseq"
cd $REPO/features
python shubert_inference.py --index $SLURM_ARRAY_TASK_ID --batch_size 834 \
  --csv_path $WORK/infer.csv --checkpoint_path $W/shubert.pt --output_dir $WORK/shubert_npy
