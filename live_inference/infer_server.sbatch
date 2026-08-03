#!/bin/bash
#SBATCH --job-name=sign_server
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition=nlp
#SBATCH --time=04:00:00
#SBATCH --account=YOUR_SLURM_ACCOUNT
set -euo pipefail
export ISLR=${PROJECT_ROOT}/islr_pipeline
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate ${PROJECT_ROOT}/envs/cslr
cd "$(dirname "$0")"
CKPT=$ISLR/checkpoints_fusion/fusion_pose_base_seed42/best.pt
VOCAB=$ISLR/checkpoints_fusion/fusion_pose_base_seed42/vocab.json
PORT=8000
echo "=================================================="
echo " node: $(hostname)   port: $PORT"
echo " tunnel: ssh -N -L ${PORT}:$(hostname):${PORT} ${USER}@<login-node>"
echo "=================================================="
python infer_server.py --ckpt "$CKPT" --vocab "$VOCAB" --port $PORT \
       ${GALLERY:+--gallery "$GALLERY"}
