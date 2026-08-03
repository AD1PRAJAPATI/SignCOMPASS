#!/bin/bash
# Double-click or: open include_fix_pose.command
cd "$(dirname "$0")" || true
set -e
echo "Submitting INCLUDE official pose top-up + retrain via Endeavour..."
ssh -o BatchMode=yes -o ConnectTimeout=30 endeavour bash -lc '
  source ~/semlex_paths.sh
  cd "$ISLR"
  bash include_fix_pose_and_requeue.sh
  echo
  squeue -u aditeyak | head -25
'
echo
echo "Done. Press Enter to close."
read -r _
