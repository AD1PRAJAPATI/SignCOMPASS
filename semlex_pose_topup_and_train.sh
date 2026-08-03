#!/bin/bash
# Resume Sem-Lex pose top-up (skips done .pt) then train.
# Previous 5235711 hit TIME LIMIT @53% + OOM → train 5235712 DependencyNeverSatisfied.
#
#   scancel 5235712   # kill the stuck train
#   source ~/semlex_paths.sh && bash $ISLR/semlex_pose_topup_and_train.sh
set -euo pipefail
source ~/semlex_paths.sh
cd "$ISLR"
PY=/project2/jessetho_1732/aditeya/envs/cslr/bin/python
mkdir -p "$ISLR/logs" "$WORK/trainval_pose_mp4"

echo "=== rebuild trainval_pose_mp4 = only STILL-MISSING pose .pt ==="
$PY - <<'PY'
import csv, os, os.path as P
S=os.environ["SEMLEX"]; W=os.environ["WORK"]
out=P.join(W,"trainval_pose_mp4"); os.makedirs(out,exist_ok=True)
# clear old symlinks (incl. broken)
for f in list(os.listdir(out)):
    p=P.join(out,f)
    if P.lexists(p):
        os.remove(p)
rows=list(csv.DictReader(open(P.join(S,"metadata.csv"))))
need=ok=miss_src=corrupt=0
seen=set()
for r in rows:
    if str(r["split"]).strip().lower() not in ("train","val"): continue
    v=r["video_id"]
    if v in seen: continue
    seen.add(v)
    src=P.join(S,"videos_mp4",v+".mp4")
    dst_pt=P.join(S,"pose_features",v+".pt")
    if not P.exists(src):
        miss_src+=1; continue
    good=False
    if P.exists(dst_pt):
        try:
            if P.getsize(dst_pt)>=64:
                import torch
                torch.load(dst_pt, map_location="cpu", weights_only=True)
                good=True
        except Exception:
            try: os.remove(dst_pt)
            except OSError: pass
            corrupt+=1
    if good:
        ok+=1; continue
    dst=P.join(out,v+".mp4")
    if P.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst); need+=1
print(f"already ok={ok} need_extract={need} corrupt_removed={corrupt} miss_src={miss_src}")
print(f"todo dir count={len(os.listdir(out))}")
open(P.join(os.environ["ISLR"],"logs","semlex_pose_todo.txt"),"w").write(f"{need}\n")
PY

N=$(cat "$ISLR/logs/semlex_pose_todo.txt")
echo "pose todo N=$N"

cat > "$ISLR/semlex_pose_topup.sbatch" <<'SBEOF'
#!/bin/bash
#SBATCH --job-name=sl_pose_tv
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=128GB
#SBATCH --partition=nlp --time=2-00:00:00 --account=jessetho_1732 --requeue
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/sl_pose_topup_%j.out
set -euo pipefail
source ~/semlex_paths.sh
module purge 2>/dev/null || true; unset LD_LIBRARY_PATH || true
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
cd "$ISLR"
N=$(ls "$WORK/trainval_pose_mp4" 2>/dev/null | wc -l)
echo "extracting pose for N=$N videos (workers=4, 48h wall)"
if [ "$N" -eq 0 ]; then echo "nothing to do"; exit 0; fi
# fewer workers → less RAM (previous job OOMed with 8)
python extract_pose_islr.py --video_dir "$WORK/trainval_pose_mp4" \
  --out_dir "$SEMLEX/pose_features" --ext .mp4 --workers 4 --model_complexity 1
echo "pose_features now: $(ls $SEMLEX/pose_features/*.pt 2>/dev/null | wc -l)"
SBEOF

if [ "$N" -eq 0 ]; then
  echo "pose complete — submitting train only"
  JT=$(sbatch --parsable --requeue sl_train_full.sbatch)
else
  JP=$(sbatch --parsable --requeue semlex_pose_topup.sbatch)
  echo "pose_topup = $JP"
  # afterany: if wall-clock cuts again, still attempt train on whatever we have
  # (train scrubs corrupt pts). Prefer finishing pose; re-run this script if N still high.
  JT=$(sbatch --parsable --requeue --dependency=afterany:$JP sl_train_full.sbatch)
fi
echo "train_full = $JT -> logs/sltrain_full_${JT}.out"
echo
echo "Also: scancel 5235712  # remove DependencyNeverSatisfied zombie"
