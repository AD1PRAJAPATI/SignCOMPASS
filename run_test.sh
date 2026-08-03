set -e
source ~/semlex_paths.sh
cd "$ISLR"; mkdir -p "$ISLR/logs"
PY=/project2/jessetho_1732/aditeya/envs/cslr/bin/python
GLUE="--parsable -A jessetho_1732 -p nlp -t 00:20:00 --mem=8G -c 2 -o $ISLR/logs/glue_%j.out"

cat > "$WORK/mk_list.py" <<'PYEOF'
import glob,gzip,os,pickle,sys
out,vid,src,suf = sys.argv[1:5]
ids=sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(vid+"/*.mp4"))
paths=[os.path.join(src,i+suf) for i in ids if os.path.exists(os.path.join(src,i+suf))]
with gzip.GzipFile(out,"wb") as f: f.write(pickle.dumps(paths,protocol=0))
print(out, len(paths), "of", len(ids))
PYEOF

cat > "$ISLR/semlex_pose_fix.sbatch" <<'SBEOF'
#!/bin/bash
#SBATCH --job-name=sl_pose_islr
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96GB
#SBATCH --partition=nlp --time=24:00:00 --account=jessetho_1732
set -euo pipefail
source ~/semlex_paths.sh
TS=$(date +"%Y%m%d_%H%M%S")
exec > "$ISLR/logs/sl_pose_${TS}.out" 2> "$ISLR/logs/sl_pose_${TS}.err"
echo "Host: $(hostname) | $(date)"
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
export PATH=/project2/jessetho_1732/aditeya/envs/cslr/bin:$PATH
module purge 2>/dev/null || true; unset LD_LIBRARY_PATH || true
cd "$ISLR"
python extract_pose_islr.py --video_dir "$VID" --out_dir "$SEMLEX/pose_features" \
  --ext .mp4 --workers 8 --model_complexity 1
echo "Done $(date) | count: $(ls $SEMLEX/pose_features/*.pt 2>/dev/null | wc -l)"
SBEOF

cat > "$ISLR/run_ensemble.sbatch" <<'SBEOF'
#!/bin/bash
#SBATCH --job-name=sl_ens
#SBATCH --account=jessetho_1732 --partition=nlp --gres=gpu:a100:1
#SBATCH --cpus-per-task=4 --mem=32GB --time=04:00:00
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/ensemble_%j.out
set -uo pipefail
source ~/semlex_paths.sh
module purge 2>/dev/null; unset LD_LIBRARY_PATH
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
export PYTHONPATH="$REPO:$REPO/fairseq:${PYTHONPATH:-}"
export ISLR_DATASET=sem_lex
cd "$ISLR"
python multi_ensemble.py --data_root /project2/jessetho_1732/aditeya \
  --pose_ckpt $ISLR/ckpt_sl_pose/fusion_pose_base_seed42/best.pt \
  --shft_ckpt $ISLR/ckpt_sl_shft/best.pt \
  --shubert_base $W/shubert.pt
SBEOF

J1=$(sbatch --parsable sl_test_convert.sbatch);                            echo "convert     = $J1"
JPOSE=$(sbatch --parsable --dependency=afterok:$J1 semlex_pose_fix.sbatch); echo "pose_stream = $JPOSE"
JVID=$(sbatch $GLUE -J g_vids --dependency=afterok:$J1 --wrap "$PY $ISLR/shubert_make_lists.py list $VID $WORK/videos.list --ext .mp4")
JKPE=$(sbatch --parsable --dependency=afterok:$JVID sl_sh_pose.sbatch);     echo "vids/kpe    = $JVID / $JKPE"
JL1=$(sbatch $GLUE -J g_lists1 --dependency=afterany:$JKPE --wrap "$PY $WORK/mk_list.py $WORK/pose.list $VID $WORK/pose _pose.json ; $PY $ISLR/shubert_make_lists.py remaining $VID $WORK/face_remaining.list --ext .mp4 --need $WORK/face_crops:_face.mp4 ; $PY $ISLR/shubert_make_lists.py remaining $VID $WORK/hands_remaining.list --ext .mp4 --need $WORK/hand_crops:_hand1.mp4 --need $WORK/hand_crops:_hand2.mp4")
JBODY=$(sbatch --parsable --dependency=afterok:$JL1 sl_sh_body.sbatch)
JCF=$(sbatch --parsable --dependency=afterok:$JL1 sl_sh_crop_face.sbatch)
JCH=$(sbatch --parsable --dependency=afterok:$JL1 sl_sh_crop_hands.sbatch)
echo "lists1/body/cf/ch = $JL1 / $JBODY / $JCF / $JCH"
JL2=$(sbatch $GLUE -J g_lists2 --dependency=afterany:$JCF:$JCH --wrap "$PY $WORK/mk_list.py $WORK/face.list $VID $WORK/face_crops _face.mp4 ; $PY $WORK/mk_list.py $WORK/hand1.list $VID $WORK/hand_crops _hand1.mp4 ; $PY $WORK/mk_list.py $WORK/hand2.list $VID $WORK/hand_crops _hand2.mp4")
JDINO=$(sbatch --parsable --dependency=afterok:$JL2 sl_sh_dino.sbatch);     echo "lists2/dino = $JL2 / $JDINO"
JENS=$(sbatch --parsable --dependency=afterany:$JPOSE:$JBODY:$JDINO run_ensemble.sbatch)
echo "ENSEMBLE    = $JENS  -> $ISLR/logs/ensemble_${JENS}.out"
