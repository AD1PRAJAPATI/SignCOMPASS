#!/bin/bash
#SBATCH --job-name=islr_fusion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nlp_hiprio
#SBATCH --time=24:00:00
#SBATCH --account=jessetho_1732

set -euo pipefail
TS=$(date +"%Y%m%d_%H%M%S")
ISLR_DIR="/project2/jessetho_1732/aditeya/islr_pipeline"   # <-- dir containing train_fusion.py
LOG_DIR="/project2/jessetho_1732/aditeya/islr_pipeline/logs"
mkdir -p "$LOG_DIR"
exec >"${LOG_DIR}/fusion_${TS}.out" 2>"${LOG_DIR}/fusion_${TS}.err"

echo "Start $(hostname) $(date)"; nvidia-smi || true
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
export PATH=/project2/jessetho_1732/aditeya/envs/cslr/bin:$PATH
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
cd "$ISLR_DIR"

# Ablations + full fusion (comparable to pose-only baseline)
python train_fusion.py --streams pose      --size base --seed 42 --rgb_dir $ASL/shubert_features
python train_fusion.py --streams rgb       --size base --seed 42 --rgb_dir $ASL/shubert_features
python train_fusion.py --streams pose+rgb  --size base --seed 42 --rgb_dir $ASL/shubert_features

echo "=== Summary ==="
grep -h "FUSION_RESULT" "${LOG_DIR}/fusion_${TS}.out" || true
echo "Done $(date)"
