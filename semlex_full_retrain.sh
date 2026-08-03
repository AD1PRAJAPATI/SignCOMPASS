#!/bin/bash
# Full Sem-Lex path to real numbers:
#   stub-clean → crop train/val/test → body → dino → retrain pose+SHuBERT-FT → ensemble
#
#   source ~/semlex_paths.sh && bash $ISLR/semlex_full_retrain.sh
set -euo pipefail
source ~/semlex_paths.sh
cd "$ISLR"
mkdir -p "$ISLR/logs"
PY=/project2/jessetho_1732/aditeya/envs/cslr/bin/python
GLUE="--parsable -A jessetho_1732 -p nlp -t 00:45:00 --mem=8G -c 2 -o $ISLR/logs/glue_%j.out"
chmod +x "$ISLR/semlex_full_retrain.sh" 2>/dev/null || true

echo "========== 0) coverage now =========="
$PY "$ISLR/semlex_coverage.py" || true

echo "========== 1) delete stub crops (ALL splits) + rebuild remaining lists =========="
$PY - <<'PY'
import csv, os, os.path as P, shutil, glob, gzip, pickle
S=os.environ["SEMLEX"]; W=os.environ["WORK"]; VID=os.environ["VID"]
MINB=1024
rows=list(csv.DictReader(open(P.join(S,"metadata.csv"))))
ids=[r["video_id"] for r in rows]
def sz(p):
    try: return P.getsize(p)
    except OSError: return -1
del_f=del_h=0
for v in ids:
    for name, sub in ((v+"_face.mp4","face_crops"),
                      (v+"_hand1.mp4","hand_crops"),
                      (v+"_hand2.mp4","hand_crops")):
        p=P.join(W,sub,name)
        if 0 <= sz(p) <= MINB:
            os.remove(p)
            if "face" in name: del_f+=1
            else: del_h+=1
print(f"deleted stubs: face={del_f} hands={del_h}")
arch=P.join(W, f"prob_archive_{os.getpid()}")
os.makedirs(arch, exist_ok=True)
n=0
for pat in ("prob_face_*.txt","prob_hand_*.txt","prob_face.txt","prob_hand.txt"):
    for p in glob.glob(P.join(W,pat)):
        shutil.move(p, P.join(arch, P.basename(p))); n+=1
print(f"archived {n} problem files -> {arch}")

def bad(path):
    try: return P.getsize(path) <= MINB
    except OSError: return True

face_rem=[]; hand_rem=[]; body_rem=[]; pose_json_miss=0
for r in rows:
    v=r["video_id"]
    mp4=P.join(VID,v+".mp4")
    if not P.exists(mp4): continue
    pj=P.join(W,"pose",v+"_pose.json")
    if not P.exists(pj):
        pose_json_miss+=1; continue
    if bad(P.join(W,"face_crops",v+"_face.mp4")):
        face_rem.append(P.abspath(mp4))
    if bad(P.join(W,"hand_crops",v+"_hand1.mp4")) or bad(P.join(W,"hand_crops",v+"_hand2.mp4")):
        hand_rem.append(P.abspath(mp4))
    if not P.exists(P.join(W,"body_feats",v+"_pose.npy")):
        body_rem.append(pj)

def dump(o,paths):
    with gzip.GzipFile(o,"wb") as f: f.write(pickle.dumps(paths,protocol=0))
    print(o, len(paths))

dump(P.join(W,"face_remaining.list"), face_rem)
dump(P.join(W,"hands_remaining.list"), hand_rem)
dump(P.join(W,"pose.list"), body_rem)
print(f"videos with mp4 but missing pose.json (skipped for crop): {pose_json_miss}")
if pose_json_miss > 500:
    print("WARNING: many missing pose.json — after this chain, consider re-running sl_sh_pose.sbatch")
PY

echo "========== 2) submit crops + body =========="
JCF=$(sbatch --parsable --requeue sl_sh_crop_face.sbatch);  echo "crop_face  = $JCF"
JCH=$(sbatch --parsable --requeue sl_sh_crop_hands.sbatch); echo "crop_hands = $JCH"
NBODY=$($PY -c "import gzip,pickle,os;print(len(pickle.loads(gzip.open(os.environ['WORK']+'/pose.list','rb').read())))")
JB=""
if [ "$NBODY" -gt 0 ]; then
  JB=$(sbatch --parsable --requeue sl_sh_body.sbatch); echo "body       = $JB  (N=$NBODY)"
else
  echo "body       = skip (pose.list empty — body feats already complete)"
fi

echo "========== 3) after crops: ALL-split dino lists → dino =========="
JL=$(sbatch $GLUE -J g_dino_all --dependency=afterany:$JCF:$JCH \
  --wrap "source ~/semlex_paths.sh; $PY $ISLR/build_dino_lists_good.py 1024 all")
echo "dino_lists = $JL"

JD=$(sbatch --parsable --requeue --dependency=afterok:$JL sl_sh_dino.sbatch)
echo "dino       = $JD"

echo "========== 4) coverage → retrain pose + SHuBERT-FT + ensemble =========="
DEP_AFTER="$JD"
[ -n "$JB" ] && DEP_AFTER="$DEP_AFTER:$JB"
JG=$(sbatch $GLUE -J g_cov --dependency=afterany:$DEP_AFTER \
  --wrap "source ~/semlex_paths.sh; $PY $ISLR/semlex_coverage.py")
echo "coverage   = $JG"

JT=$(sbatch --parsable --requeue --dependency=afterok:$JG sl_train_full.sbatch)
echo "train_full = $JT  -> logs/sltrain_full_${JT}.out"

echo
echo "Chain submitted."
echo "  crops:  cropface_${JCF}_* / crophands_${JCH}_*"
echo "  dino:   dino_${JD}_*"
echo "  train:  sltrain_full_${JT}.out   (prints ENSEMBLE TEST at end)"
echo "  Gate inside train: train SHuBERT-ready >= 30000 or job exits"
