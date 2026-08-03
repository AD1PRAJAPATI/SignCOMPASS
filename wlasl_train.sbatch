#!/bin/bash
#SBATCH --job-name=incl_train
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64GB --gres=gpu:a100:1
#SBATCH --partition=nlp --time=24:00:00 --account=jessetho_1732 --requeue
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/incl_train_%j.out
set -euo pipefail
source ~/semlex_paths.sh
source "$ISLR/wlasl_env.sh"
module purge 2>/dev/null; unset LD_LIBRARY_PATH
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
export PYTHONPATH="$REPO:$REPO/fairseq:${PYTHONPATH:-}"
export ISLR_DATASET=wlasl
cd "$ISLR"

python - <<'PY'
import csv, os, os.path as P
root=os.environ["INCLUDE"]; W=os.environ["WORK"]
rows=list(csv.DictReader(open(P.join(root,"metadata.csv"))))
def ok(*a):
    try: return P.getsize(P.join(*a))>100
    except: return False
for split in ("train","val","test"):
    ids=[r["video_id"] for r in rows if str(r["split"]).strip().lower()==split]
    sh=sum(1 for v in ids if ok(W,"face_feats",v+"_face.npy") and ok(W,"hand1_feats",v+"_hand1.npy")
           and ok(W,"hand2_feats",v+"_hand2.npy") and ok(W,"body_feats",v+"_pose.npy"))
    pose=sum(1 for v in ids if ok(root,"pose_features",v+".pt"))
    print(f"{split}: n={len(ids)} shubert_ready={sh} pose_pt={pose}")
train_sh=sum(1 for r in rows if str(r["split"]).strip().lower()=="train"
             and ok(W,"face_feats",r["video_id"]+"_face.npy")
             and ok(W,"hand1_feats",r["video_id"]+"_hand1.npy")
             and ok(W,"hand2_feats",r["video_id"]+"_hand2.npy")
             and ok(W,"body_feats",r["video_id"]+"_pose.npy"))
open(P.join(os.environ["ISLR"],"logs","include_train_usable.txt"),"w").write(f"{train_sh}\n")
print("train_shubert_ready", train_sh)
if train_sh < 2000:
    raise SystemExit(f"REFUSING INCLUDE train: only {train_sh} ready")
PY

echo "=== INCLUDE pose stream ==="
python train_fusion.py --streams pose --seed 42 \
  --data_root /project2/jessetho_1732/aditeya \
  --save_dir ckpt_wlasl_pose --epochs 80 --num_workers 4

echo "=== INCLUDE SHuBERT-FT ==="
python train_shubert_ft.py --epochs 40 --batch_size 32 --seed 42 \
  --data_root /project2/jessetho_1732/aditeya \
  --save_dir ckpt_wlasl_shft \
  --ckpt "$W/shubert.pt" --num_workers 4

echo "=== INCLUDE ensemble ==="
python multi_ensemble.py --data_root /project2/jessetho_1732/aditeya \
  --pose_ckpt $ISLR/ckpt_wlasl_pose/fusion_pose_base_seed42/best.pt \
  --shft_ckpt $ISLR/ckpt_wlasl_shft/best.pt \
  --shubert_base $W/shubert.pt

echo "DONE INCLUDE $(date)"
