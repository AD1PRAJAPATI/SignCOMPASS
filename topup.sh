source ~/semlex_paths.sh
PY=/project2/jessetho_1732/aditeya/envs/cslr/bin/python
$PY - <<'PYEOF'
import csv,os,os.path as P,gzip,pickle
S=os.environ["SEMLEX"]; W=os.environ["WORK"]
rows=list(csv.DictReader(open(P.join(S,"metadata.csv"))))
test=[r["video_id"] for r in rows if str(r["split"]).strip().lower()=="test"]
ex=lambda *a:P.exists(P.join(*a))
ids=[v for v in test if ex(W,"face_crops",v+"_face.mp4") and ex(W,"hand_crops",v+"_hand1.mp4") and ex(W,"hand_crops",v+"_hand2.mp4")]
def dump(o,p):
    with gzip.GzipFile(o,"wb") as f: f.write(pickle.dumps(p,protocol=0)); 
    print(o,len(p))
dump(P.join(W,"face.list"), [P.join(W,"face_crops",v+"_face.mp4") for v in ids])
dump(P.join(W,"hand1.list"),[P.join(W,"hand_crops",v+"_hand1.mp4") for v in ids])
dump(P.join(W,"hand2.list"),[P.join(W,"hand_crops",v+"_hand2.mp4") for v in ids])
d=P.join(W,"test_pose_mp4"); os.makedirs(d,exist_ok=True); n=0
for v in test:
    src=P.join(S,"videos_mp4",v+".mp4"); dst=P.join(d,v+".mp4")
    if ex(S,"videos_mp4",v+".mp4") and not ex(S,"pose_features",v+".pt") and not P.islink(dst):
        os.symlink(src,dst); n+=1
print("dino ids:",len(ids)," pose symlinks(missing):",n)
PYEOF

cat > "$ISLR/semlex_pose_test.sbatch" <<'SBEOF'
#!/bin/bash
#SBATCH --job-name=sl_pose_test
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96GB
#SBATCH --partition=nlp --time=12:00:00 --account=jessetho_1732 --requeue
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/sl_pose_test_%j.out
set -uo pipefail
source ~/semlex_paths.sh
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
export PATH=/project2/jessetho_1732/aditeya/envs/cslr/bin:$PATH
module purge 2>/dev/null || true; unset LD_LIBRARY_PATH || true
cd "$ISLR"
python extract_pose_islr.py --video_dir "$WORK/test_pose_mp4" --out_dir "$SEMLEX/pose_features" \
  --ext .mp4 --workers 8 --model_complexity 1
echo "pose_features now: $(ls $SEMLEX/pose_features | wc -l)"
SBEOF

JP=$(sbatch --parsable --requeue semlex_pose_test.sbatch);            echo "pose_test = $JP"
JD=$(sbatch --parsable --requeue sl_sh_dino.sbatch);                  echo "dino      = $JD"
JE=$(sbatch --parsable --dependency=afterok:$JP:$JD run_ensemble.sbatch); echo "ENSEMBLE  = $JE -> logs/ensemble_${JE}.out"
