#!/bin/bash
# Prepare INCLUDE (ISL) in parallel with Sem-Lex — CPU/scratch only, no GPU.
# Safe while Sem-Lex crops/dino run: does NOT touch sem_lex or GPUs.
#
#   source ~/semlex_paths.sh
#   bash $ISLR/include_prepare.sh
set -euo pipefail
source ~/semlex_paths.sh
PY=/project2/jessetho_1732/aditeya/envs/cslr/bin/python
ROOT=${INCLUDE_ROOT:-/project2/jessetho_1732/aditeya/data/include}
# resolve symlink → usually /scratch1/aditeyak/include
ROOT=$(readlink -f "$ROOT")
echo "INCLUDE root: $ROOT"

if [ ! -d "$ROOT" ]; then
  echo "Missing $ROOT — run include_download + ln -sfn first"; exit 1
fi

echo "=== 1) rebuild clean metadata ==="
$PY "$ISLR/include_make_metadata.py" --root "$ROOT" --no-hf
# keep a copy under project data path name for ISLR_DATASET=include
# (ROOT already is the symlink target)

echo "=== 2) flat video dir (symlinks named by video_id) ==="
$PY - <<PY
import csv, os
from pathlib import Path
root = Path("$ROOT")
vid_dir = root / "videos_mp4"
vid_dir.mkdir(exist_ok=True)
rows = list(csv.DictReader(open(root / "metadata.csv")))
ok = skip = miss = 0
for r in rows:
    src = Path(r["video_path"])
    if not src.is_file():
        miss += 1
        continue
    ext = src.suffix  # .MOV
    dst = vid_dir / f"{r['video_id']}{ext}"
    if dst.is_symlink() or dst.exists():
        skip += 1
        continue
    os.symlink(src.resolve(), dst)
    ok += 1
print(f"symlinks created={ok} already={skip} missing_src={miss}")
print(f"videos_mp4 count:", sum(1 for _ in vid_dir.iterdir()))
# sanity: pose out dir on scratch (same tree)
(root / "pose_features").mkdir(exist_ok=True)
(root / "shubert").mkdir(exist_ok=True)
print("pose_features ->", root / "pose_features")
PY

echo "=== 3) submit MediaPipe pose (CPU, scratch) ==="
cat > "$ISLR/include_pose.sbatch" <<'SBEOF'
#!/bin/bash
#SBATCH --job-name=incl_pose
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64GB
#SBATCH --partition=nlp --time=12:00:00 --account=jessetho_1732 --requeue
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/include_pose_%j.out
set -uo pipefail
source ~/semlex_paths.sh
module purge 2>/dev/null || true; unset LD_LIBRARY_PATH || true
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
export PATH=/project2/jessetho_1732/aditeya/envs/cslr/bin:$PATH
ROOT=$(readlink -f /project2/jessetho_1732/aditeya/data/include)
cd "$ISLR"
# OpenCV reads .MOV; --ext match is case-insensitive
python extract_pose_islr.py \
  --video_dir "$ROOT/videos_mp4" \
  --out_dir "$ROOT/pose_features" \
  --ext .mov \
  --workers 8 --model_complexity 1
echo "pose_features count: $(ls "$ROOT/pose_features"/*.pt 2>/dev/null | wc -l)"
SBEOF

JP=$(sbatch --parsable --requeue "$ISLR/include_pose.sbatch")
echo "include_pose = $JP  -> $ISLR/logs/include_pose_${JP}.out"
echo
echo "Done preparing INCLUDE kickoff."
echo "  metadata + flat videos: ready"
echo "  pose job: $JP (CPU; won't steal Sem-Lex GPUs)"
echo "  DO NOT start INCLUDE dino/train until Sem-Lex ensemble finishes (quota + GPUs)."
echo
echo "Later (after Sem-Lex results): ISLR_DATASET=include ..."
