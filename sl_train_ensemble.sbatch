#!/bin/bash
#SBATCH --job-name=sl_train_ens
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nlp
#SBATCH --time=24:00:00
#SBATCH --account=jessetho_1732

set -euo pipefail
source ~/semlex_paths.sh
export ISLR_DATASET=sem_lex          # route dataset code to Sem-Lex
BASE=/project2/jessetho_1732/aditeya/islr_pipeline
mkdir -p "$BASE/logs"
TS=$(date +"%Y%m%d_%H%M%S")
exec > "$BASE/logs/sl_train_ens_${TS}.out" 2> "$BASE/logs/sl_train_ens_${TS}.err"

source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
export PATH=/project2/jessetho_1732/aditeya/envs/cslr/bin:$PATH
cd "$BASE"
python -c "import torch;print('CUDA',torch.cuda.is_available())"

echo "=== 1. Train pose stream (Sem-Lex) ==="
python train_fusion.py \
    --streams   pose \
    --data_root /project2/jessetho_1732/aditeya \
    --save_dir  "$BASE/ckpt_sl_pose" \
    --seed 42

echo "=== 2. Ensemble pose + SHuBERT-ft (Sem-Lex) ==="
python multi_ensemble.py \
    --data_root   /project2/jessetho_1732/aditeya \
    --pose_ckpt   "$BASE/ckpt_sl_pose/fusion_pose_base_seed42/best.pt" \
    --shft_ckpt   "$BASE/ckpt_sl_shft/best.pt" \
    --shubert_base "$SEMLEX/shubert_features"

echo "Done: $(date)"
