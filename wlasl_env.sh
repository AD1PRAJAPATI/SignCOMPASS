export WLASL=$(readlink -f /project2/jessetho_1732/aditeya/data/wlasl)
export INCLUDE="$WLASL"
export VID="$WLASL/videos"
export WORK="$WLASL/shubert"
export ISLR_DATASET=wlasl
mkdir -p "$WORK"/{pose,face_crops,hand_crops,face_feats,hand1_feats,hand2_feats,body_feats,mp_models,stats}
SL_MP=/project2/jessetho_1732/aditeya/data/sem_lex/shubert/mp_models
[ -d "$SL_MP" ] && [ ! -e "$WORK/mp_models/face_landmarker.task" ] && ln -sfn "$SL_MP"/* "$WORK/mp_models/" 2>/dev/null || true
