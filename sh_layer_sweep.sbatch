#!/bin/bash
#SBATCH --job-name=sh_layersweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nlp_hiprio
#SBATCH --time=12:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/layersweep_%j.out
set -uo pipefail
source ~/shubert_paths.sh
module purge 2>/dev/null; unset LD_LIBRARY_PATH
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
cd $ISLR

for L in 4 6 8; do
  OUT=$ASL/shubert_L$L
  if [ "$(ls "$OUT" 2>/dev/null | wc -l)" -lt 80000 ]; then
    echo "=== converting layer $L ==="
    python $ISLR/shubert_convert.py --in_dir $WORK/shubert_npy --out_dir $OUT --layer $L
  fi
  echo "=== training rgb-only on layer $L ==="
  python train_fusion.py --streams rgb --rgb_dir $OUT --size base --seed 42 --epochs 60 --save_dir ckpt_L$L
done

echo "=== LAYER SWEEP SUMMARY ==="
grep "FUSION_RESULT: streams=rgb" /project2/jessetho_1732/aditeya/islr_pipeline/logs/layersweep_${SLURM_JOB_ID}.out || true
