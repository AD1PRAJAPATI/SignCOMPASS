#!/bin/bash
#SBATCH --job-name=incl_ens_hf
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64GB --gres=gpu:a100:1
#SBATCH --partition=nlp_hiprio --time=02:00:00 --account=jessetho_1732 --requeue
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/incl_ens_hf_%j.out
set -euo pipefail
source ~/semlex_paths.sh
source "$ISLR/include_env.sh"
module purge 2>/dev/null || true
unset LD_LIBRARY_PATH || true
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
    pose=sum(1 for v in ids if ok(root,"pose_features",v+".pt"))
    sh=sum(1 for v in ids if ok(W,"face_feats",v+"_face.npy") and ok(W,"hand1_feats",v+"_hand1.npy")
           and ok(W,"hand2_feats",v+"_hand2.npy") and ok(W,"body_feats",v+"_pose.npy"))
    print(f"[official] {split}: n={len(ids)} shubert={sh} pose={pose}")
PY

python multi_ensemble.py --data_root /project2/jessetho_1732/aditeya \
  --pose_ckpt $ISLR/ckpt_include_pose_hf/fusion_pose_base_seed42/best.pt \
  --shft_ckpt $ISLR/ckpt_include_shft_hf/best.pt \
  --shubert_base $W/shubert.pt

echo "DONE OFFICIAL INCLUDE ENSEMBLE $(date)"
