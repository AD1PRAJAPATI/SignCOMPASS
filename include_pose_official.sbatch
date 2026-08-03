#!/bin/bash
#SBATCH --job-name=incl_pose_o
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64GB
#SBATCH --partition=nlp --time=12:00:00 --account=jessetho_1732 --requeue
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/include_pose_official_%j.out
set -euo pipefail
source ~/semlex_paths.sh
source "$ISLR/include_env.sh"
module purge 2>/dev/null || true
unset LD_LIBRARY_PATH || true
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
cd "$ISLR"

# unbuffered python so logs update live
export PYTHONUNBUFFERED=1

TODO="$INCLUDE/_pose_todo_official"
OUT="$INCLUDE/pose_features"
mkdir -p "$TODO" "$OUT"

echo "INCLUDE=$INCLUDE $(date)"
echo "TODO_pre=$(ls -1 "$TODO" 2>/dev/null | wc -l) entries"

python -u - <<'PY'
import csv, os, os.path as P, sys
root = os.environ["INCLUDE"]
vid = os.environ["VID"]
todo = P.join(root, "_pose_todo_official")
pose = P.join(root, "pose_features")
os.makedirs(todo, exist_ok=True)

# Index videos_mp4 ONCE (avoid O(missing * dirsize) hang)
print("indexing videos_mp4 ...", flush=True)
idx = {}
if P.isdir(vid):
    for f in os.listdir(vid):
        stem, ext = P.splitext(f)
        if ext.lower() in {".mov", ".mp4", ".avi", ".mkv", ".webm"}:
            idx.setdefault(stem, P.join(vid, f))
print(f"videos_mp4 index={len(idx)}", flush=True)

# Clear + rebuild todo only for missing pose
print("clearing old todo ...", flush=True)
for f in os.listdir(todo):
    p = P.join(todo, f)
    if P.lexists(p):
        os.remove(p)

n = miss_src = already = 0
for r in csv.DictReader(open(P.join(root, "metadata.csv"))):
    v = r["video_id"]
    pt = P.join(pose, v + ".pt")
    if P.exists(pt) and P.getsize(pt) > 64:
        already += 1
        continue
    src = idx.get(v)
    if src is None:
        # fall back to metadata path
        src = r.get("video_path") or ""
        if not (src and P.isfile(src)):
            miss_src += 1
            continue
    ext = P.splitext(src)[1] or ".MP4"
    dst = P.join(todo, v + ext)
    os.symlink(P.abspath(src), dst)
    n += 1
print(f"rebuilt todo symlinks={n} already_pose={already} miss_src={miss_src}", flush=True)
PY

echo "TODO_after=$(ls -1 "$TODO" 2>/dev/null | wc -l) $(date)"

python -u - <<'PY'
import os, os.path as P, subprocess, sys
todo = P.join(os.environ["INCLUDE"], "_pose_todo_official")
out = P.join(os.environ["INCLUDE"], "pose_features")
files = [f for f in os.listdir(todo) if P.islink(P.join(todo, f)) or P.isfile(P.join(todo, f))]
exact = {}
for f in files:
    exact.setdefault(P.splitext(f)[1], []).append(f)
print("todo by ext:", {k: len(v) for k, v in exact.items()}, flush=True)
if not files:
    print("NOTHING TO EXTRACT", flush=True)
    sys.exit(0)
for ext, flist in exact.items():
    print(f"=== extracting ext={ext!r} n={len(flist)} ===", flush=True)
    cmd = [
        sys.executable, "-u", "extract_pose_islr.py",
        "--video_dir", todo,
        "--out_dir", out,
        "--ext", ext,
        "--workers", "4",
        "--model_complexity", "1",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd)
n_pt = sum(1 for f in os.listdir(out) if f.endswith(".pt"))
print(f"pose_features: {n_pt}", flush=True)
PY

echo "DONE pose official $(date)"
