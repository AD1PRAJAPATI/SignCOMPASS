#!/bin/bash
#SBATCH --job-name=alpha_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nlp
#SBATCH --time=04:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/alpha_sweep_%j.out
set -uo pipefail
export ISLR=/project2/jessetho_1732/aditeya/islr_pipeline
source ~/semlex_paths.sh
unset ISLR_DATASET
module purge 2>/dev/null || true; unset LD_LIBRARY_PATH || true
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
export PYTHONPATH="$REPO:$REPO/fairseq:${PYTHONPATH:-}"
cd "$ISLR"
python multi_ensemble.py
