#!/bin/bash
# Rebuild INCLUDE metadata from official HF splits, check coverage, retrain+ensemble.
# Does not touch Sem-Lex jobs.
#
#   source ~/semlex_paths.sh && bash $ISLR/include_official_eval.sh
set -euo pipefail
source ~/semlex_paths.sh
source "$ISLR/include_env.sh"
cd "$ISLR"
PY=${PROJECT_ROOT}/envs/cslr/bin/python
mkdir -p "$ISLR/logs"
chmod +x "$ISLR/include_official_eval.sh" 2>/dev/null || true

echo "INCLUDE=$INCLUDE"

echo "=== 1) official HF metadata (backup carved splits, then overwrite) ==="
if [ -f "$INCLUDE/metadata.csv" ]; then
  cp -a "$INCLUDE/metadata.csv" "$INCLUDE/metadata_carved_backup_$(date +%Y%m%d_%H%M%S).csv"
  echo "backed up carved metadata"
fi
# prefer parquet; fallback to datasets-server JSON in the python file
$PY "$ISLR/include_make_metadata.py" --root "$INCLUDE" --out "$INCLUDE/metadata.csv"

echo "=== 2) refresh flat videos_mp4 symlinks for metadata ids ==="
$PY - <<'PY'
import csv, os
from pathlib import Path
root = Path(os.environ["INCLUDE"])
vid = Path(os.environ["VID"]); vid.mkdir(exist_ok=True)
rows = list(csv.DictReader(open(root / "metadata.csv")))
ok = skip = miss = 0
for r in rows:
    src = Path(r["video_path"])
    if not src.is_file():
        miss += 1; continue
    dst = vid / f"{r['video_id']}{src.suffix}"
    if dst.exists() or dst.is_symlink():
        skip += 1; continue
    os.symlink(src.resolve(), dst); ok += 1
print(f"symlink +{ok} already={skip} miss_src={miss} meta_rows={len(rows)}")
from collections import Counter
print("splits", dict(Counter(r["split"] for r in rows)))
print("classes", len({r["gloss"] for r in rows}))
PY

echo "=== 3) coverage on OFFICIAL splits ==="
$PY - <<'PY'
import csv, os, os.path as P
root=os.environ["INCLUDE"]; W=os.environ["WORK"]
rows=list(csv.DictReader(open(P.join(root,"metadata.csv"))))
def ok(p,m=100):
    try: return P.getsize(p)>m
    except: return False
todo_pose=[]; todo_dino=[]
for split in ("train","val","test"):
    ids=[r for r in rows if r["split"]==split]
    n=len(ids)
    pose=sum(1 for r in ids if ok(P.join(root,"pose_features",r["video_id"]+".pt")))
    face=sum(1 for r in ids if ok(P.join(W,"face_feats",r["video_id"]+"_face.npy")))
    h1=sum(1 for r in ids if ok(P.join(W,"hand1_feats",r["video_id"]+"_hand1.npy")))
    h2=sum(1 for r in ids if ok(P.join(W,"hand2_feats",r["video_id"]+"_hand2.npy")))
    body=sum(1 for r in ids if ok(P.join(W,"body_feats",r["video_id"]+"_pose.npy")))
    full=sum(1 for r in ids if ok(P.join(root,"pose_features",r["video_id"]+".pt"))
             and ok(P.join(W,"face_feats",r["video_id"]+"_face.npy"))
             and ok(P.join(W,"hand1_feats",r["video_id"]+"_hand1.npy"))
             and ok(P.join(W,"hand2_feats",r["video_id"]+"_hand2.npy"))
             and ok(P.join(W,"body_feats",r["video_id"]+"_pose.npy")))
    print(f"{split}: n={n} pose={pose} face={face} h1={h1} h2={h2} body={body} ensemble_ready={full}")
    for r in ids:
        v=r["video_id"]
        if not ok(P.join(root,"pose_features",v+".pt")):
            todo_pose.append(r)
        sh=ok(P.join(W,"face_feats",v+"_face.npy")) and ok(P.join(W,"hand1_feats",v+"_hand1.npy")) \
           and ok(P.join(W,"hand2_feats",v+"_hand2.npy")) and ok(P.join(W,"body_feats",v+"_pose.npy"))
        if not sh:
            todo_dino.append(v)
open(P.join(os.environ["ISLR"],"logs","include_official_todo_pose.txt"),"w").write(str(len(todo_pose))+"\n")
open(P.join(os.environ["ISLR"],"logs","include_official_todo_sh.txt"),"w").write(str(len(todo_dino))+"\n")
print("todo_pose", len(todo_pose), "todo_shubert_streams", len(todo_dino))
PY

NPOSE=$(cat "$ISLR/logs/include_official_todo_pose.txt")
NSH=$(cat "$ISLR/logs/include_official_todo_sh.txt")
echo "missing pose=$NPOSE  missing shubert streams=$NSH"

# Pose top-up for official-split missing
if [ "$NPOSE" -gt 0 ]; then
  $PY - <<'PY'
import csv, os, os.path as P
root=os.environ["INCLUDE"]; vid=os.environ["VID"]
out=P.join(root,"_pose_todo_official"); os.makedirs(out, exist_ok=True)
for f in list(os.listdir(out)):
    p=P.join(out,f)
    if P.lexists(p): os.remove(p)
n=0
for r in csv.DictReader(open(P.join(root,"metadata.csv"))):
    v=r["video_id"]; pt=P.join(root,"pose_features",v+".pt")
    if P.exists(pt) and P.getsize(pt)>64: continue
    hits=[P.join(vid,f) for f in os.listdir(vid) if f.startswith(v+".")]
    if not hits: continue
    dst=P.join(out, v+P.splitext(hits[0])[1])
    if P.lexists(dst): os.remove(dst)
    os.symlink(hits[0], dst); n+=1
print("official pose todo symlinks", n)
PY
  cat > "$ISLR/include_pose_official.sbatch" <<'SBEOF'
#!/bin/bash
#SBATCH --job-name=incl_pose_o
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64GB
#SBATCH --partition=nlp --time=12:00:00 --account=YOUR_ACCOUNT --requeue
#SBATCH --output=${PROJECT_ROOT}/islr_pipeline/logs/include_pose_official_%j.out
set -euo pipefail
source ~/semlex_paths.sh; source "$ISLR/include_env.sh"
module purge 2>/dev/null || true; unset LD_LIBRARY_PATH || true
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate ${PROJECT_ROOT}/envs/cslr
cd "$ISLR"
for ext in .MOV .mov .mp4; do
  n=$(ls "$INCLUDE/_pose_todo_official"/*"$ext" 2>/dev/null | wc -l || true)
  [ "${n:-0}" -eq 0 ] && continue
  python extract_pose_islr.py --video_dir "$INCLUDE/_pose_todo_official" \
    --out_dir "$INCLUDE/pose_features" --ext "$ext" --workers 4 --model_complexity 1
done
echo "pose_features: $(ls $INCLUDE/pose_features/*.pt 2>/dev/null | wc -l)"
SBEOF
  JP=$(sbatch --parsable --requeue include_pose_official.sbatch)
  echo "pose_official = $JP"
else
  JP=""; echo "pose_official = skip"
fi

# If shubert streams missing, rebuild remaining + submit crop/dino/body (reuse include pipeline pieces)
DEP_TRAIN=""
if [ "$NSH" -gt 50 ]; then
  echo "=== topping up SHuBERT streams for official ids ==="
  $PY "$ISLR/include_rebuild_lists.py" remaining
  $PY "$ISLR/include_rebuild_lists.py" dino
  # ensure videos.list exists
  $PY - <<'PY'
import os, os.path as P, gzip, pickle
VID=os.environ["VID"]; W=os.environ["WORK"]
vids=sorted(P.abspath(P.join(VID,f)) for f in os.listdir(VID) if f.lower().endswith((".mov",".mp4")))
with gzip.GzipFile(P.join(W,"videos.list"),"wb") as f: f.write(pickle.dumps(vids,protocol=0))
print("videos.list", len(vids))
PY
  JK=$(sbatch --parsable --requeue incl_kpe.sbatch); echo "kpe = $JK"
  JL1=$(sbatch --parsable -A YOUR_ACCOUNT -p nlp -t 00:30:00 --mem=8G -c 2 \
    -o $ISLR/logs/glue_%j.out -J incl_off_l1 --dependency=afterany:$JK \
    --wrap "source ~/semlex_paths.sh; source $ISLR/include_env.sh; $PY $ISLR/include_rebuild_lists.py remaining")
  JCF=$(sbatch --parsable --requeue --dependency=afterok:$JL1 incl_crop_face.sbatch)
  JCH=$(sbatch --parsable --requeue --dependency=afterok:$JL1 incl_crop_hands.sbatch)
  JB=$(sbatch --parsable --requeue --dependency=afterok:$JL1 incl_body.sbatch)
  JL2=$(sbatch --parsable -A YOUR_ACCOUNT -p nlp -t 00:30:00 --mem=8G -c 2 \
    -o $ISLR/logs/glue_%j.out -J incl_off_l2 --dependency=afterany:$JCF:$JCH \
    --wrap "source ~/semlex_paths.sh; source $ISLR/include_env.sh; $PY $ISLR/include_rebuild_lists.py dino")
  JD=$(sbatch --parsable --requeue --dependency=afterok:$JL2 incl_dino.sbatch)
  echo "crop/dino/body = $JCF $JCH $JB $JD"
  DEP_TRAIN="afterany:$JD:$JB"
  [ -n "$JP" ] && DEP_TRAIN="${DEP_TRAIN}:$JP"
else
  echo "SHuBERT streams mostly present — train on what we have"
  if [ -n "$JP" ]; then DEP_TRAIN="afterany:$JP"; fi
fi

# Official-split train (writes to ckpt_include_pose_hf / ckpt_include_shft_hf)
cat > "$ISLR/incl_train_official.sbatch" <<'SBEOF'
#!/bin/bash
#SBATCH --job-name=incl_train_hf
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64GB --gres=gpu:a100:1
#SBATCH --partition=nlp --time=24:00:00 --account=YOUR_ACCOUNT --requeue
#SBATCH --output=${PROJECT_ROOT}/islr_pipeline/logs/incl_train_hf_%j.out
set -euo pipefail
source ~/semlex_paths.sh
source "$ISLR/include_env.sh"
module purge 2>/dev/null; unset LD_LIBRARY_PATH
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate ${PROJECT_ROOT}/envs/cslr
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
  --data_root ${PROJECT_ROOT} \
  --save_dir ckpt_include_pose_hf --epochs 80 --num_workers 4

echo "=== OFFICIAL INCLUDE SHuBERT-FT ==="
python train_shubert_ft.py --epochs 40 --batch_size 32 --seed 42 \
  --data_root ${PROJECT_ROOT} \
  --save_dir ckpt_include_shft_hf \
  --ckpt "$W/shubert.pt" --num_workers 4

echo "=== OFFICIAL INCLUDE ensemble ==="
python multi_ensemble.py --data_root ${PROJECT_ROOT} \
  --pose_ckpt $ISLR/ckpt_include_pose_hf/fusion_pose_base_seed42/best.pt \
  --shft_ckpt $ISLR/ckpt_include_shft_hf/best.pt \
  --shubert_base $W/shubert.pt

echo "DONE OFFICIAL INCLUDE $(date)"
SBEOF

if [ -n "$DEP_TRAIN" ]; then
  JT=$(sbatch --parsable --requeue --dependency=$DEP_TRAIN incl_train_official.sbatch)
else
  JT=$(sbatch --parsable --requeue incl_train_official.sbatch)
fi
echo "train_official = $JT -> logs/incl_train_hf_${JT}.out"
echo
echo "This overwrites $INCLUDE/metadata.csv with HF official splits."
echo "Previous carved-split results remain in logs/incl_train_5244876.out"
