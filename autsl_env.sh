export AUTSL=$(readlink -f /project2/jessetho_1732/aditeya/data/autsl)
export INCLUDE="$AUTSL"
export VID="$AUTSL/videos_mp4"
export WORK="$AUTSL/shubert"
export ISLR_DATASET=autsl
mkdir -p "$WORK"/{pose,face_crops,hand_crops,face_feats,hand1_feats,hand2_feats,body_feats,mp_models,stats} "$VID"
SL_MP=/project2/jessetho_1732/aditeya/data/sem_lex/shubert/mp_models
[ -d "$SL_MP" ] && [ ! -e "$WORK/mp_models/face_landmarker.task" ] && ln -sfn "$SL_MP"/* "$WORK/mp_models/" 2>/dev/null || true
