#!/bin/bash
set -euo pipefail
export ISLR=/project2/jessetho_1732/aditeya/islr_pipeline
source ~/semlex_paths.sh
source "$ISLR/autsl_env.sh"
cd "$ISLR"; mkdir -p logs
PY=/project2/jessetho_1732/aditeya/envs/cslr/bin/python
GLUE="--parsable -A jessetho_1732 -p nlp -t 00:45:00 --mem=8G -c 2 -o $ISLR/logs/glue_%j.out"
chmod +x "$ISLR"/autsl_*.sbatch 2>/dev/null || true
echo "AUTSL=$AUTSL VID=$VID symlinks=$(ls $VID|wc -l)"

cat > $ISLR/autsl_pose_topup.sbatch <<'SB'
#!/bin/bash
#SBATCH --job-name=autsl_pose
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=nlp
#SBATCH --time=24:00:00
#SBATCH --account=jessetho_1732
#SBATCH --requeue
#SBATCH --output=/project2/jessetho_1732/aditeya/islr_pipeline/logs/autsl_pose_%j.out
set -euo pipefail
export ISLR=/project2/jessetho_1732/aditeya/islr_pipeline
source ~/semlex_paths.sh; source "$ISLR/autsl_env.sh"
module purge 2>/dev/null||true; unset LD_LIBRARY_PATH||true
source /home1/aditeyak/miniconda3/etc/profile.d/conda.sh
conda activate /project2/jessetho_1732/aditeya/envs/cslr
cd "$ISLR"
python extract_pose_islr.py --video_dir "$VID" --out_dir "$AUTSL/pose_features" --ext .mp4 --workers 8 --model_complexity 1
echo "pose_features: $(ls $AUTSL/pose_features/*.pt 2>/dev/null|wc -l)"
SB
JP=$(sbatch --parsable --requeue autsl_pose_topup.sbatch); echo "pose=$JP"

$PY - <<'PY'
import os,os.path as P,gzip,pickle
VID=os.environ["VID"]; W=os.environ["WORK"]
vids=sorted(P.abspath(P.join(VID,f)) for f in os.listdir(VID) if f.lower().endswith(".mp4"))
gzip.GzipFile(P.join(W,"videos.list"),"wb").write(pickle.dumps(vids,protocol=0))
pd=P.join(W,"pose"); nj=len([f for f in os.listdir(pd) if f.endswith("_pose.json")]) if P.isdir(pd) else 0
if nj<100:
    for n in ("face_remaining.list","hands_remaining.list"):
        gzip.GzipFile(P.join(W,n),"wb").write(pickle.dumps(vids,protocol=0))
    gzip.GzipFile(P.join(W,"pose.list"),"wb").write(pickle.dumps([],protocol=0))
print("videos.list",len(vids),"| pose.json",nj)
PY

JK=$(sbatch --parsable --requeue autsl_kpe.sbatch); echo "kpe=$JK"
JL1=$(sbatch $GLUE -J autsl_l1 --dependency=afterany:$JK --wrap "source ~/semlex_paths.sh; source $ISLR/autsl_env.sh; $PY $ISLR/include_rebuild_lists.py remaining"); echo "l1=$JL1"
JCF=$(sbatch --parsable --requeue --dependency=afterok:$JL1 autsl_crop_face.sbatch); echo "cropf=$JCF"
JCH=$(sbatch --parsable --requeue --dependency=afterok:$JL1 autsl_crop_hands.sbatch); echo "croph=$JCH"
JB=$(sbatch --parsable --requeue --dependency=afterok:$JL1 autsl_body.sbatch); echo "body=$JB"
JL2=$(sbatch $GLUE -J autsl_l2 --dependency=afterany:$JCF:$JCH --wrap "source ~/semlex_paths.sh; source $ISLR/autsl_env.sh; $PY $ISLR/include_rebuild_lists.py dino"); echo "l2=$JL2"
JD=$(sbatch --parsable --requeue --dependency=afterok:$JL2 autsl_dino.sbatch); echo "dino=$JD"
JT=$(sbatch --parsable --requeue --dependency=afterany:$JD:$JB:$JP autsl_train.sbatch); echo "train=$JT"
echo "SUBMITTED pose=$JP kpe=$JK crops=$JCF/$JCH body=$JB dino=$JD train=$JT"
