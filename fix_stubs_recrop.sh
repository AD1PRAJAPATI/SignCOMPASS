#!/bin/bash
# Fix Sem-Lex test DINO: delete 257B stub crops, clear problem skip-lists,
# re-crop face/hands, rebuild size-filtered dino lists, then dino → ensemble.
# Run ON THE CLUSTER:  bash $ISLR/fix_stubs_recrop.sh
set -euo pipefail
source ~/semlex_paths.sh
cd "$ISLR"
mkdir -p "$ISLR/logs"
PY=${PROJECT_ROOT}/envs/cslr/bin/python
GLUE="--parsable -A YOUR_ACCOUNT -p nlp -t 00:30:00 --mem=8G -c 2 -o $ISLR/logs/glue_%j.out"
chmod +x "$ISLR/check_usable_then_ensemble.sh" "$ISLR/fix_stubs_recrop.sh" 2>/dev/null || true

echo "=== 1) delete stub crops (test ids) + archive problem files ==="
$PY - <<'PY'
import csv, os, os.path as P, shutil, glob
S=os.environ["SEMLEX"]; W=os.environ["WORK"]
MINB=1024
rows=list(csv.DictReader(open(P.join(S,"metadata.csv"))))
test=[r["video_id"] for r in rows if str(r["split"]).strip().lower()=="test"]
def sz(p):
    try: return P.getsize(p)
    except OSError: return -1
del_f=del_h=0
for v in test:
    for name, sub in ((v+"_face.mp4","face_crops"),
                      (v+"_hand1.mp4","hand_crops"),
                      (v+"_hand2.mp4","hand_crops")):
        p=P.join(W,sub,name)
        s=sz(p)
        if 0 <= s <= MINB:
            os.remove(p)
            if "face" in name: del_f+=1
            else: del_h+=1
print(f"deleted stubs: face={del_f} hands={del_h} (size<={MINB})")
arch=P.join(W, f"prob_archive_{os.getpid()}")
os.makedirs(arch, exist_ok=True)
n=0
for pat in ("prob_face_*.txt","prob_hand_*.txt","prob_face.txt","prob_hand.txt"):
    for p in glob.glob(P.join(W,pat)):
        shutil.move(p, P.join(arch, P.basename(p))); n+=1
print(f"archived {n} problem files -> {arch}")
need_f=need_h=0
for v in test:
    if not P.exists(P.join(S,"videos_mp4",v+".mp4")): continue
    if not P.exists(P.join(W,"pose",v+"_pose.json")): continue
    if sz(P.join(W,"face_crops",v+"_face.mp4"))<=MINB: need_f+=1
    if sz(P.join(W,"hand_crops",v+"_hand1.mp4"))<=MINB or sz(P.join(W,"hand_crops",v+"_hand2.mp4"))<=MINB:
        need_h+=1
print(f"test needing face crop: {need_f} | hand crops: {need_h}")
PY

echo "=== 2) rebuild remaining crop lists (tiny == missing) ==="
$PY - <<'PY'
import csv,os,os.path as P,gzip,pickle
S=os.environ["SEMLEX"]; W=os.environ["WORK"]; VID=os.environ["VID"]
MINB=1024
rows=list(csv.DictReader(open(P.join(S,"metadata.csv"))))
def bad(path):
    try: return P.getsize(path) <= MINB
    except OSError: return True
face_rem=[]; hand_rem=[]
for r in rows:
    v=r["video_id"]
    mp4=P.join(VID,v+".mp4")
    if not P.exists(mp4): continue
    if not P.exists(P.join(W,"pose",v+"_pose.json")): continue
    if bad(P.join(W,"face_crops",v+"_face.mp4")):
        face_rem.append(P.abspath(mp4))
    if bad(P.join(W,"hand_crops",v+"_hand1.mp4")) or bad(P.join(W,"hand_crops",v+"_hand2.mp4")):
        hand_rem.append(P.abspath(mp4))
def dump(o,paths):
    with gzip.GzipFile(o,"wb") as f: f.write(pickle.dumps(paths,protocol=0))
    print(o, len(paths))
dump(P.join(W,"face_remaining.list"), face_rem)
dump(P.join(W,"hands_remaining.list"), hand_rem)
PY

echo "=== 3) submit crop → dino lists → dino → usable-gated ensemble ==="
JCF=$(sbatch --parsable --requeue sl_sh_crop_face.sbatch);   echo "crop_face  = $JCF"
JCH=$(sbatch --parsable --requeue sl_sh_crop_hands.sbatch);  echo "crop_hands = $JCH"

JL=$(sbatch $GLUE -J g_dino_lists --dependency=afterany:$JCF:$JCH \
  --wrap "source ~/semlex_paths.sh; $PY $ISLR/build_dino_lists_good.py 1024")
echo "dino_lists = $JL"

JD=$(sbatch --parsable --requeue --dependency=afterok:$JL sl_sh_dino.sbatch)
echo "dino       = $JD"

JE=$(sbatch $GLUE -J ens_gate --dependency=afterany:$JD \
  --wrap "bash $ISLR/check_usable_then_ensemble.sh")
echo "ens_gate   = $JE"
echo
echo "Done submitting. Watch:"
echo "  crops:  \$ISLR/logs/cropface_${JCF}_0.out  /  crophands_${JCH}_0.out"
echo "  dino:   \$ISLR/logs/dino_${JD}_0.out"
echo "  usable: \$ISLR/logs/usable_check.txt"
echo "Re-running ensemble alone is useless until hand1/hand2 feats leave 0."
