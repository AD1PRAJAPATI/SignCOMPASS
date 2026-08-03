#!/bin/bash
#SBATCH --job-name=sh_fusion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nlp_hiprio
#SBATCH --time=12:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/fusionbest_%j.out
set -uo pipefail
source ~/shubert_paths.sh
module purge 2>/dev/null; unset LD_LIBRARY_PATH
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
cd $ISLR

python train_fusion.py --streams pose+rgb --rgb_dir $ASL/shubert_L4 --size base --seed 42 --save_dir ckpt_fusion_L4

echo "=== RESULT ==="
grep "FUSION_RESULT" /project2/jessetho_1732/aditeya/islr_pipeline/logs/fusionbest_${SLURM_JOB_ID}.out || true
