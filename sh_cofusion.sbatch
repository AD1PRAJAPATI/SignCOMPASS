#!/bin/bash
#SBATCH --job-name=cofusion
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48GB --gres=gpu:a100:1
#SBATCH --partition=nlp_hiprio --time=12:00:00 --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/cofusion_%j.out
set -uo pipefail
source ~/shubert_paths.sh
module purge 2>/dev/null; unset LD_LIBRARY_PATH
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
cd $ISLR
python train_cofusion.py --rgb_dir $ASL/shubert_L4 --size base --seed 42
grep COFUSION_RESULT /project2/jessetho_1732/aditeya/islr_pipeline/logs/cofusion_${SLURM_JOB_ID}.out || true
