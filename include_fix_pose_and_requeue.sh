#!/bin/bash
# Cancel hung pose jobs, resubmit fixed pose + official train.
#   source ~/semlex_paths.sh && bash $ISLR/include_fix_pose_and_requeue.sh
set -euo pipefail
source ~/semlex_paths.sh
source "$ISLR/include_env.sh"
cd "$ISLR"

echo "Cancelling old incl_pose_o / incl_train_hf ..."
scancel -u aditeyak -n incl_pose_o 2>/dev/null || true
scancel -u aditeyak -n incl_train_hf 2>/dev/null || true
sleep 2

JP=$(sbatch --parsable --requeue include_pose_official.sbatch)
echo "pose_official = $JP  -> logs/include_pose_official_${JP}.out"

JT=$(sbatch --parsable --requeue --dependency=afterok:$JP incl_train_official.sbatch)
echo "train_official = $JT -> logs/incl_train_hf_${JT}.out (waits for pose afterok)"
echo
squeue -u aditeyak | head -25
echo
echo "Watch:  tail -f \$ISLR/logs/include_pose_official_${JP}.out"
