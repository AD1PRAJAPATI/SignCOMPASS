# SHuBERT feature extraction on ASL-Citizen → fusion training (the best-ceiling path)

Goal: produce in-distribution SHuBERT `(T,768)` features for every ASL-Citizen video,
then train the pose+SHuBERT fusion model. Uses SHuBERT's OWN scripts (not the
home-rolled all-in-one) so inputs match what SHuBERT was trained on.

## Paths (edit if yours differ)
```
REPO=/project2/jessetho_1732/aditeya/SHuBERT
W=/project2/jessetho_1732/aditeya/data/shubert_weights
ASL=/project2/jessetho_1732/aditeya/data/asl_citizen
VID=$ASL/ASL_Citizen/videos
WORK=$ASL/shubert            # intermediate artifacts
ISLR=/project2/jessetho_1732/aditeya/islr_pipeline
mkdir -p $WORK
```

## Prerequisites (one-time, you must do these)
1. Conda envs (from the yml files in $REPO):
   ```
   conda env create -f $REPO/environment_feature_extraction.yml   # -> feature_extraction
   conda env create -f $REPO/environment_dino.yml                 # -> dino
   # SHuBERT inference env: your cslr env already has the fairseq fork (setup_shubert.sh)
   ```
2. MediaPipe Tasks models (for stage 1):
   - face_landmarker.task, hand_landmarker.task  (download from MediaPipe model cards)
   - put them somewhere, note the paths.

## Parallelism
Every stage script takes `--index I --batch_size B`: it processes slice I of the file
list in chunks of B. To do ALL videos in one job, set `--index 0 --batch_size 100000000`.
To parallelize, submit a SLURM array `--array=0-31` and pass `--index $SLURM_ARRAY_TASK_ID`
with `--batch_size = ceil(N_files / 32)`. Start with one job to validate, then scale.

## Stage 1 — Pose (env: feature_extraction)
```
python $ISLR/shubert_make_lists.py list $VID $WORK/videos.list --ext .mp4
conda activate feature_extraction
cd $REPO/dataset
python kpe_mediapipe.py --index 0 --batch_size 100000000 \
  --files_list $WORK/videos.list --pose_path $WORK/pose --stats_path $WORK/stats \
  --problem_file_path $WORK/prob_pose.txt --time_limit 999999 \
  --face_model_path /path/face_landmarker.task --hand_model_path /path/hand_landmarker.task
```

## Stage 2 — Face & hand crops (env: feature_extraction)
```
python crop_face.py  --index 0 --batch_size 100000000 --files_list $WORK/videos.list \
  --pose_path $WORK/pose --face_path $WORK/face_crops --problem_file_path $WORK/prob_face.txt --time_limit 999999
python crop_hands.py --index 0 --batch_size 100000000 --files_list $WORK/videos.list \
  --pose_path $WORK/pose --hand_path $WORK/hand_crops --problem_file_path $WORK/prob_hand.txt --time_limit 999999
```
> Check how crop_hands.py names left vs right (e.g. `<id>_left.mp4` / `<id>_right.mp4`).
> Build per-hand crop lists accordingly in stage 3.

## Stage 3 — Stream features
Body (env: feature_extraction) — pose files → body .npy:
```
python $ISLR/shubert_make_lists.py list $WORK/pose $WORK/pose.list --ext <pose_ext>
cd $REPO/features
python body_features.py --index 0 --batch_size 100000000 \
  --files_list $WORK/pose.list --pose_features_path $WORK/body_feats --time_limit 999999
```
Face + hands (env: dino) — crops → DINOv2 .npy:
```
conda activate dino
python $ISLR/shubert_make_lists.py list $WORK/face_crops $WORK/face.list
python dinov2_features.py --index 0 --batch_size 100000000 --files_list $WORK/face.list \
  --output_folder $WORK/face_feats --dino_path $W/dino_face.pt --time_limit 999999
# left + right hands: build a .list for each (glob the left/right crops), run twice:
python dinov2_features.py --index 0 --batch_size 100000000 --files_list $WORK/lh.list \
  --output_folder $WORK/lh_feats --dino_path $W/dino_left_hand.pt --time_limit 999999
python dinov2_features.py --index 0 --batch_size 100000000 --files_list $WORK/rh.list \
  --output_folder $WORK/rh_feats --dino_path $W/dino_right_hand.pt --time_limit 999999
```

## Stage 4 — SHuBERT inference (env: cslr, with fairseq fork)
```
python $ISLR/shubert_make_lists.py csv $WORK/face_feats $WORK/lh_feats $WORK/rh_feats $WORK/body_feats $WORK/infer.csv
conda activate /project2/jessetho_1732/aditeya/envs/cslr
export PYTHONPATH=$PYTHONPATH:$REPO/fairseq
cd $REPO/features
python shubert_inference.py --index 0 --batch_size 100000000 \
  --csv_path $WORK/infer.csv --checkpoint_path $W/shubert.pt --output_dir $WORK/shubert_npy
```

## Stage 5 — Convert [L,T,768] → (T,768) .pt
```
python $ISLR/shubert_convert.py --in_dir $WORK/shubert_npy \
  --out_dir $ASL/shubert_features --layer -1
```

## Stage 6 — Train fusion with SHuBERT as the RGB stream
```
cd $ISLR
python train_fusion.py --streams pose+rgb --rgb_dir $ASL/shubert_features --size base
```
Same trainer as VideoMAE — SHuBERT just replaces the RGB feature dir.

## Sanity checks between stages
- After each stage: `find $WORK/<out> -name '*.npy' | wc -l` should approach 83399.
- Watch the `prob_*.txt` / `failed*.txt` logs for videos that errored.
- Validate ONE video end-to-end before launching the full array.
