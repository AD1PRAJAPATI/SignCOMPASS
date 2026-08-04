#!/bin/bash
# After dino finishes: only submit ensemble if test usable >= 1000.
set -euo pipefail
source ~/semlex_paths.sh
PY=${PROJECT_ROOT}/envs/cslr/bin/python
$PY - <<'PY'
import csv, os, os.path as P
S=os.environ["SEMLEX"]; W=os.environ["WORK"]; ISLR=os.environ["ISLR"]
rows=list(csv.DictReader(open(P.join(S,"metadata.csv"))))
test=[r["video_id"] for r in rows if str(r["split"]).strip().lower()=="test"]
ex=lambda *a: P.exists(P.join(*a))
usable=sum(1 for v in test if ex(S,"pose_features",v+".pt") and ex(W,"face_feats",v+"_face.npy")
           and ex(W,"hand1_feats",v+"_hand1.npy") and ex(W,"hand2_feats",v+"_hand2.npy")
           and ex(W,"body_feats",v+"_pose.npy"))
print("usable test for ensemble:", usable, "/", len(test))
open(P.join(ISLR,"logs","usable_check.txt"),"w").write(f"{usable}\n")
if usable < 1000:
    raise SystemExit(f"REFUSING ensemble: usable={usable} (need >=1000)")
PY
cd "$ISLR"
sbatch run_ensemble.sbatch
