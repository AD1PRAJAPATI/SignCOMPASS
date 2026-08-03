# Source after ~/semlex_paths.sh — redirects WORK/VID onto INCLUDE (scratch).
# Keeps REPO, W (dino/shubert weights), ISLR from semlex_paths.
export INCLUDE=$(readlink -f /project2/jessetho_1732/aditeya/data/include)
export VID="$INCLUDE/videos_mp4"
export WORK="$INCLUDE/shubert"
export ISLR_DATASET=include
mkdir -p "$WORK"/{pose,face_crops,hand_crops,face_feats,hand1_feats,hand2_feats,body_feats,mp_models,stats}
# reuse MediaPipe tasks from Sem-Lex if present
SL_MP=/project2/jessetho_1732/aditeya/data/sem_lex/shubert/mp_models
if [ -d "$SL_MP" ] && [ ! -e "$WORK/mp_models/face_landmarker.task" ]; then
  ln -sfn "$SL_MP"/* "$WORK/mp_models/" 2>/dev/null || true
fi
