#!/bin/bash
#SBATCH --job-name=sh_ft
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64GB --gres=gpu:a100:1
#SBATCH --partition=nlp_hiprio --time=16:00:00 --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/shft_%j.out
set -uo pipefail
source ~/shubert_paths.sh
module purge 2>/dev/null; unset LD_LIBRARY_PATH
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
export PYTHONPATH=$PYTHONPATH:$REPO/fairseq
cd $ISLR
python train_shubert_ft.py --epochs 40 --batch_size 32 --lr 5e-4 --backbone_lr 1e-5
grep SHUBERT_FT_RESULT /project2/jessetho_1732/aditeya/islr_pipeline/logs/shft_${SLURM_JOB_ID}.out || true
