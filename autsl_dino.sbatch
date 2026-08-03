#!/bin/bash
#SBATCH --job-name=incl_dino
#SBATCH --array=0-99%10
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24GB --gres=gpu:a100:1
#SBATCH --partition=nlp --time=08:00:00 --account=jessetho_1732 --requeue
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/incl_dino_%A_%a.out
set -uo pipefail
source ~/semlex_paths.sh
source "$ISLR/autsl_env.sh"
module purge 2>/dev/null; unset LD_LIBRARY_PATH
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/dino
I=$SLURM_ARRAY_TASK_ID
N=$(python -c "import gzip,pickle;print(len(pickle.loads(gzip.open('$WORK/face.list','rb').read())))")
CHUNKS=100; BS=$(( (N + CHUNKS - 1) / CHUNKS )); NB=$(( (N + BS - 1) / BS ))
if [ "$I" -ge "$NB" ]; then echo "no work N=$N idx=$I"; exit 0; fi
echo "INCLUDE dino N=$N BS=$BS idx=$I WORK=$WORK"
cd $REPO/features
python dinov2_features.py --index $I --batch_size $BS --files_list $WORK/face.list \
  --output_folder $WORK/face_feats  --dino_path $W/dino_face.pt       --time_limit 28000
python dinov2_features.py --index $I --batch_size $BS --files_list $WORK/hand1.list \
  --output_folder $WORK/hand1_feats --dino_path $W/dino_left_hand.pt  --time_limit 28000
python dinov2_features.py --index $I --batch_size $BS --files_list $WORK/hand2.list \
  --output_folder $WORK/hand2_feats --dino_path $W/dino_right_hand.pt --time_limit 28000
