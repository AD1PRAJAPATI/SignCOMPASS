#!/bin/bash
#SBATCH --job-name=sl_test_conv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --partition=nlp
#SBATCH --time=12:00:00
#SBATCH --account=jessetho_1732
set -uo pipefail
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr 2>/dev/null || true
module load ffmpeg 2>/dev/null || module load FFmpeg 2>/dev/null || true
FFMPEG=$(command -v ffmpeg || true)
echo "ffmpeg at: ${FFMPEG:-NOT FOUND}"
if [ -z "$FFMPEG" ]; then echo "ERROR: no ffmpeg on PATH"; exit 1; fi
SRC=/project2/jessetho_1732/aditeya/data/sem_lex/download/test
VID=/project2/jessetho_1732/aditeya/data/sem_lex/videos_mp4
mkdir -p "$VID"; export VID FFMPEG
convert_one() {
  f="$1"; id=$(basename "$f" .webm); out="$VID/$id.mp4"
  [ -f "$out" ] && return 0
  "$FFMPEG" -nostdin -y -loglevel error -i "$f" -c:v libx264 -pix_fmt yuv420p -an "$out" \
    || { echo "FAIL $id"; rm -f "$out"; }
}
export -f convert_one
find "$SRC" -name '*.webm' | xargs -P 16 -I {} bash -c 'convert_one "$@"' _ {}
echo "done: $(ls "$VID"/*.mp4 | wc -l) total mp4s in VID"
