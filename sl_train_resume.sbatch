#!/bin/bash
# Resume Sem-Lex after preemption: skip pose (done), warm-start SHuBERT-FT, then ensemble.
# Prefer hipri to avoid another kick mid-run.
#
#   sbatch $ISLR/sl_train_resume.sbatch
#SBATCH --job-name=sl_train_res
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64GB --gres=gpu:a100:1
#SBATCH --partition=nlp_hiprio --time=12:00:00 --account=jessetho_1732 --requeue
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/sltrain_resume_%j.out
set -euo pipefail
source ~/semlex_paths.sh
module purge 2>/dev/null || true
unset LD_LIBRARY_PATH || true
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
export PYTHONPATH="$REPO:$REPO/fairseq:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export ISLR_DATASET=sem_lex
cd "$ISLR"

POSE=$ISLR/ckpt_sl_pose/fusion_pose_base_seed42/best.pt
SHFT=$ISLR/ckpt_sl_shft/best.pt
[ -f "$POSE" ] || { echo "missing pose ckpt $POSE"; exit 1; }
[ -f "$SHFT" ] || { echo "missing shft ckpt $SHFT (need at least partial FT best.pt)"; exit 1; }

echo "=== skip pose (already done) ==="
ls -la "$POSE"

echo "=== resume SHuBERT-FT from $SHFT (32 more ep) ==="
python train_shubert_ft.py --epochs 32 --warmup_epochs 1 --batch_size 32 --seed 42 \
  --data_root /project2/jessetho_1732/aditeya \
  --save_dir ckpt_sl_shft \
  --ckpt "$W/shubert.pt" \
  --init_ft "$SHFT" \
  --num_workers 4 \
  --patience 10

echo "=== Ensemble ==="
python multi_ensemble.py --data_root /project2/jessetho_1732/aditeya \
  --pose_ckpt "$POSE" \
  --shft_ckpt "$SHFT" \
  --shubert_base "$W/shubert.pt"

echo "DONE RESUME $(date)"
