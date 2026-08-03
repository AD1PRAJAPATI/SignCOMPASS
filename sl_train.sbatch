#!/bin/bash
#SBATCH --job-name=sl_train
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64GB --gres=gpu:1
#SBATCH --partition=nlp_hiprio --time=24:00:00 --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/sltrain_%j.out
set -uo pipefail
source ~/shubert_paths.sh
module purge 2>/dev/null; unset LD_LIBRARY_PATH
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
export PYTHONPATH=${PYTHONPATH:-}:$REPO/fairseq
export ISLR_DATASET=sem_lex
cd $ISLR
echo "=== Sem-Lex pose stream ==="
python train_fusion.py --streams pose --seed 42 --save_dir ckpt_sl_pose --epochs 120
echo "=== Sem-Lex SHuBERT-FT stream ==="
python train_shubert_ft.py --epochs 40 --batch_size 32 --seed 42 --save_dir ckpt_sl_shft
