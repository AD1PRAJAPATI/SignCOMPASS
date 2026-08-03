#!/bin/bash
#SBATCH --job-name=sl_train_full
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64GB --gres=gpu:a100:1
#SBATCH --partition=nlp --time=36:00:00 --account=jessetho_1732 --requeue
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/sltrain_full_%j.out
set -euo pipefail
source ~/semlex_paths.sh
module purge 2>/dev/null; unset LD_LIBRARY_PATH
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
export PYTHONPATH="$REPO:$REPO/fairseq:${PYTHONPATH:-}"
export ISLR_DATASET=sem_lex
cd "$ISLR"

echo "=== coverage before train ==="
python semlex_coverage.py

TRAIN_OK=$(cat "$ISLR/logs/semlex_train_usable.txt" 2>/dev/null || echo 0)
echo "train SHuBERT-ready=$TRAIN_OK"
if [ "$TRAIN_OK" -lt 30000 ]; then
  echo "REFUSING train: need >=30000 train stream feats, got $TRAIN_OK"
  exit 1
fi

echo "=== scrub truncated pose_features .pt ==="
python - <<'PY'
import csv, os, os.path as P
S=os.environ["SEMLEX"]
pd=P.join(S,"pose_features")
n=bad=0
for fn in os.listdir(pd):
    if not fn.endswith(".pt"): continue
    n+=1
    p=P.join(pd,fn)
    try:
        if P.getsize(p)<64: raise OSError("tiny")
        import torch
        torch.load(p, map_location="cpu", weights_only=True)
    except Exception:
        os.remove(p); bad+=1
print(f"scanned {n} pose .pt, removed corrupt={bad}")
PY

echo "=== 1) Sem-Lex pose stream (120 ep) ==="
python train_fusion.py --streams pose --seed 42 \
  --data_root /project2/jessetho_1732/aditeya \
  --save_dir ckpt_sl_pose --epochs 120 --num_workers 4

echo "=== 2) Sem-Lex SHuBERT-FT (40 ep) ==="
python train_shubert_ft.py --epochs 40 --batch_size 32 --seed 42 \
  --data_root /project2/jessetho_1732/aditeya \
  --save_dir ckpt_sl_shft \
  --ckpt "$W/shubert.pt" --num_workers 4

echo "=== 3) Ensemble ==="
python multi_ensemble.py --data_root /project2/jessetho_1732/aditeya \
  --pose_ckpt $ISLR/ckpt_sl_pose/fusion_pose_base_seed42/best.pt \
  --shft_ckpt $ISLR/ckpt_sl_shft/best.pt \
  --shubert_base $W/shubert.pt

echo "DONE $(date)"
