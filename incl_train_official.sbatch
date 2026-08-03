#!/bin/bash
#SBATCH --job-name=incl_train_hf
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64GB --gres=gpu:a100:1
#SBATCH --partition=nlp --time=24:00:00 --account=jessetho_1732 --requeue
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/incl_train_hf_%j.out
set -euo pipefail
source ~/semlex_paths.sh
source "$ISLR/include_env.sh"
module purge 2>/dev/null; unset LD_LIBRARY_PATH
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
export PYTHONPATH="$REPO:$REPO/fairseq:${PYTHONPATH:-}"
export ISLR_DATASET=include
cd "$ISLR"

python - <<'PY'
import csv, os, os.path as P
root=os.environ["INCLUDE"]; W=os.environ["WORK"]
rows=list(csv.DictReader(open(P.join(root,"metadata.csv"))))
def ok(*a):
    try: return P.getsize(P.join(*a))>100
    except: return False
for split in ("train","val","test"):
    ids=[r["video_id"] for r in rows if r["split"]==split]
    sh=sum(1 for v in ids if ok(W,"face_feats",v+"_face.npy") and ok(W,"hand1_feats",v+"_hand1.npy")
           and ok(W,"hand2_feats",v+"_hand2.npy") and ok(W,"body_feats",v+"_pose.npy"))
    pose=sum(1 for v in ids if ok(root,"pose_features",v+".pt"))
    print(f"[official] {split}: n={len(ids)} shubert={sh} pose={pose}")
train_n=sum(1 for r in rows if r["split"]=="train" and ok(W,"face_feats",r["video_id"]+"_face.npy")
            and ok(W,"hand1_feats",r["video_id"]+"_hand1.npy") and ok(W,"hand2_feats",r["video_id"]+"_hand2.npy")
            and ok(W,"body_feats",r["video_id"]+"_pose.npy"))
if train_n < 1500:
    raise SystemExit(f"REFUSING: train shubert ready={train_n}")
PY

echo "=== OFFICIAL INCLUDE pose train ==="
python train_fusion.py --streams pose --seed 42 \
  --data_root /project2/jessetho_1732/aditeya \
  --save_dir ckpt_include_pose_hf --epochs 80 --num_workers 4

echo "=== OFFICIAL INCLUDE SHuBERT-FT ==="
python train_shubert_ft.py --epochs 40 --batch_size 32 --seed 42 \
  --data_root /project2/jessetho_1732/aditeya \
  --save_dir ckpt_include_shft_hf \
  --ckpt "$W/shubert.pt" --num_workers 4

echo "=== OFFICIAL INCLUDE ensemble ==="
python multi_ensemble.py --data_root /project2/jessetho_1732/aditeya \
  --pose_ckpt $ISLR/ckpt_include_pose_hf/fusion_pose_base_seed42/best.pt \
  --shft_ckpt $ISLR/ckpt_include_shft_hf/best.pt \
  --shubert_base $W/shubert.pt

echo "DONE OFFICIAL INCLUDE $(date)"
