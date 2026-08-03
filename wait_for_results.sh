#!/bin/bash
# Poll until Sem-Lex ensemble prints numbers. Run on cluster:
#   bash $ISLR/wait_for_results.sh
set -euo pipefail
source ~/semlex_paths.sh
echo "Waiting on crop→dino→ensemble chain (jobs 5205566..5205570)..."
echo "Ctrl-C is fine; chain keeps running in Slurm."
while true; do
  # newest ensemble log with the success line
  hit=$(ls -t "$ISLR"/logs/ensemble_*.out 2>/dev/null | while read -r f; do
    grep -q "ENSEMBLE TEST:" "$f" 2>/dev/null && echo "$f" && break
  done)
  if [ -n "${hit:-}" ]; then
    echo
    echo "========== RESULTS ($hit) =========="
    grep -E "usable |best alpha|ENSEMBLE TEST|pose=|shubert_ft" "$hit" || true
    echo "===================================="
    exit 0
  fi
  # progress crumbs
  nf=$(ls "$WORK"/face_feats/*_face.npy 2>/dev/null | wc -l)
  n1=$(ls "$WORK"/hand1_feats/*_hand1.npy 2>/dev/null | wc -l)
  n2=$(ls "$WORK"/hand2_feats/*_hand2.npy 2>/dev/null | wc -l)
  u=$(cat "$ISLR"/logs/usable_check.txt 2>/dev/null || echo "?")
  sq=$(squeue -u "$USER" -h -o "%i %T %j" 2>/dev/null | grep -E "5205566|5205567|5205568|5205569|5205570|sh_dino|sh_crop|sl_ens|ens_gate|g_dino" | head -8 | tr '\n' '|' || true)
  echo "$(date +%H:%M:%S) feats face/h1/h2=$nf/$n1/$n2  usable_check=$u  queue: ${sq:-idle/done}"
  sleep 120
done
