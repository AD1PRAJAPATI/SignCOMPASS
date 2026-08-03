#!/bin/bash
#SBATCH --job-name=sl_ens
#SBATCH --account=jessetho_1732 --partition=nlp --gres=gpu:a100:1
#SBATCH --cpus-per-task=4 --mem=32GB --time=04:00:00
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/ensemble_%j.out
set -uo pipefail
source ~/semlex_paths.sh
module purge 2>/dev/null; unset LD_LIBRARY_PATH
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
export PYTHONPATH="$REPO:$REPO/fairseq:${PYTHONPATH:-}"
export ISLR_DATASET=sem_lex
cd "$ISLR"
python multi_ensemble.py --data_root /project2/jessetho_1732/aditeya \
  --pose_ckpt $ISLR/ckpt_sl_pose/fusion_pose_base_seed42/best.pt \
  --shft_ckpt $ISLR/ckpt_sl_shft/best.pt \
  --shubert_base $W/shubert.pt
