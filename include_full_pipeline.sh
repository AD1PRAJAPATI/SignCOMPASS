#!/bin/bash
# INCLUDE (Indian Sign Language) full chain on scratch — parallel-safe with Sem-Lex.
# Does NOT touch data/sem_lex. Dino capped at 10 GPUs.
#
#   source ~/semlex_paths.sh && bash $ISLR/include_full_pipeline.sh
set -euo pipefail
source ~/semlex_paths.sh
source "$ISLR/include_env.sh"
cd "$ISLR"
mkdir -p "$ISLR/logs"
PY=/project2/jessetho_1732/aditeya/envs/cslr/bin/python
GLUE="--parsable -A jessetho_1732 -p nlp -t 00:45:00 --mem=8G -c 2 -o $ISLR/logs/glue_%j.out"
chmod +x "$ISLR"/incl_*.sbatch "$ISLR/include_full_pipeline.sh" 2>/dev/null || true

echo "INCLUDE=$INCLUDE"
echo "VID=$VID  WORK=$WORK"

if [ ! -d "$INCLUDE" ]; then
  echo "Missing $INCLUDE — run: ln -sfn /scratch1/aditeyak/include /project2/jessetho_1732/aditeya/data/include"
  exit 1
fi

echo "=== 1) metadata + flat video symlinks ==="
$PY "$ISLR/include_make_metadata.py" --root "$INCLUDE" --no-hf
$PY - <<'PY'
import csv, os
from pathlib import Path
root = Path(os.environ["INCLUDE"])
vid_dir = Path(os.environ["VID"]); vid_dir.mkdir(exist_ok=True)
rows = list(csv.DictReader(open(root / "metadata.csv")))
ok = skip = miss = 0
for r in rows:
    src = Path(r["video_path"])
    if not src.is_file():
        miss += 1; continue
    dst = vid_dir / f"{r['video_id']}{src.suffix}"
    if dst.exists() or dst.is_symlink():
        skip += 1; continue
    os.symlink(src.resolve(), dst); ok += 1
print(f"symlinks +{ok} already={skip} miss_src={miss} total={len(list(vid_dir.iterdir()))}")
PY

echo "=== 2) pose_features top-up list ==="
$PY - <<'PY'
import csv, os, os.path as P
root=os.environ["INCLUDE"]; vid=os.environ["VID"]
out=P.join(root,"_pose_todo"); os.makedirs(out, exist_ok=True)
for f in list(os.listdir(out)):
    p=P.join(out,f)
    if P.islink(p) or P.isfile(p): os.remove(p)
n=0
for r in csv.DictReader(open(P.join(root,"metadata.csv"))):
    v=r["video_id"]
    pt=P.join(root,"pose_features",v+".pt")
    if P.exists(pt) and P.getsize(pt)>64: continue
    hits=[P.join(vid,f) for f in os.listdir(vid) if f.startswith(v+".")]
    if not hits: continue
    ext=P.splitext(hits[0])[1]
    os.symlink(hits[0], P.join(out, v+ext)); n+=1
print(f"pose todo: {n}")
open(P.join(os.environ["ISLR"],"logs","include_pose_todo.txt"),"w").write(str(n)+"\n")
PY

NPOSE=$(cat "$ISLR/logs/include_pose_todo.txt")
JP=""
if [ "$NPOSE" -gt 0 ]; then
  cat > "$ISLR/include_pose_topup.sbatch" <<'SBEOF'
#!/bin/bash
#SBATCH --job-name=incl_pose2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64GB
#SBATCH --partition=nlp --time=12:00:00 --account=jessetho_1732 --requeue
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/include_pose_%j.out
set -euo pipefail
source ~/semlex_paths.sh
source "$ISLR/include_env.sh"
module purge 2>/dev/null || true; unset LD_LIBRARY_PATH || true
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
cd "$ISLR"
for ext in .MOV .mov .mp4; do
  n=$(ls "$INCLUDE/_pose_todo"/*"$ext" 2>/dev/null | wc -l || true)
  [ "${n:-0}" -eq 0 ] && continue
  echo "extracting $n *$ext"
  python extract_pose_islr.py --video_dir "$INCLUDE/_pose_todo" \
    --out_dir "$INCLUDE/pose_features" --ext "$ext" --workers 8 --model_complexity 1
done
echo "pose_features: $(ls $INCLUDE/pose_features/*.pt 2>/dev/null | wc -l)"
SBEOF
  JP=$(sbatch --parsable --requeue include_pose_topup.sbatch)
  echo "pose_topup = $JP"
else
  echo "pose_topup = skip"
fi

echo "=== 3) videos.list + initial remaining lists ==="
$PY - <<'PY'
import os, os.path as P, gzip, pickle
VID=os.environ["VID"]; W=os.environ["WORK"]
vids=sorted(P.abspath(P.join(VID,f)) for f in os.listdir(VID)
            if f.lower().endswith((".mov",".mp4",".avi")))
with gzip.GzipFile(P.join(W,"videos.list"),"wb") as f:
    f.write(pickle.dumps(vids, protocol=0))
print("videos.list", len(vids))
PY
# placeholder remaining (full list until kpe finishes — crops skip missing pose json)
$PY "$ISLR/include_rebuild_lists.py" remaining || true
# if no pose jsons yet, put all videos in remaining so crops wait on kpe rebuild
$PY - <<'PY'
import os, os.path as P, gzip, pickle
W=os.environ["WORK"]; VID=os.environ["VID"]
njson=len([f for f in os.listdir(P.join(W,"pose")) if f.endswith("_pose.json")]) if P.isdir(P.join(W,"pose")) else 0
print("existing pose.json", njson)
if njson < 100:
    vids=sorted(P.abspath(P.join(VID,f)) for f in os.listdir(VID) if f.lower().endswith((".mov",".mp4")))
    for name in ("face_remaining.list","hands_remaining.list"):
        with gzip.GzipFile(P.join(W,name),"wb") as f:
            f.write(pickle.dumps(vids, protocol=0))
        print(name, len(vids), "(pre-kpe placeholder)")
    with gzip.GzipFile(P.join(W,"pose.list"),"wb") as f:
        f.write(pickle.dumps([], protocol=0))
PY

echo "=== 4) submit chain ==="
KPE_DEP=()
if [ -n "$JP" ]; then KPE_DEP=(--dependency=afterany:$JP); fi
JK=$(sbatch --parsable --requeue "${KPE_DEP[@]}" incl_kpe.sbatch); echo "kpe = $JK"

JL1=$(sbatch $GLUE -J incl_lists1 --dependency=afterany:$JK \
  --wrap "source ~/semlex_paths.sh; source $ISLR/include_env.sh; $PY $ISLR/include_rebuild_lists.py remaining")
echo "lists1 = $JL1"

JCF=$(sbatch --parsable --requeue --dependency=afterok:$JL1 incl_crop_face.sbatch);  echo "crop_face  = $JCF"
JCH=$(sbatch --parsable --requeue --dependency=afterok:$JL1 incl_crop_hands.sbatch); echo "crop_hands = $JCH"
JB=$(sbatch --parsable --requeue --dependency=afterok:$JL1 incl_body.sbatch);        echo "body       = $JB"

JL2=$(sbatch $GLUE -J incl_dino_lists --dependency=afterany:$JCF:$JCH \
  --wrap "source ~/semlex_paths.sh; source $ISLR/include_env.sh; $PY $ISLR/include_rebuild_lists.py dino")
echo "dino_lists = $JL2"

JD=$(sbatch --parsable --requeue --dependency=afterok:$JL2 incl_dino.sbatch)
echo "dino = $JD"

# train after dino + body + pose topup (if any)
TRAIN_DEP="afterany:$JD:$JB"
[ -n "$JP" ] && TRAIN_DEP="${TRAIN_DEP}:$JP"
JT=$(sbatch --parsable --requeue --dependency=$TRAIN_DEP incl_train.sbatch)
echo "train = $JT -> logs/incl_train_${JT}.out"

echo
echo "INCLUDE pipeline submitted."
echo "  WORK=$WORK (scratch via symlink)"
echo "  Watch: logs/incl_kpe_${JK}_0.out  then  logs/incl_train_${JT}.out"
echo "  Sem-Lex: still run  bash \$ISLR/semlex_pose_topup_and_train.sh  if not submitted"
