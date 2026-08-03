# SignCOMPASS

**COMPASS: Complementary Motion and Appearance Streams for Isolated Sign Recognition**

Two-stream isolated sign language recognition (ISLR). A pose stream (MediaPipe keypoints -> Conformer) captures motion, an appearance stream (face/hand crops -> DINOv2 -> fine-tuned SHuBERT) captures shape, and a simple score fusion combines them.

| Dataset (language)   | Pose | SHuBERT | Fusion |
|----------------------|:----:|:-------:|:------:|
| ASL-Citizen (ASL)    | 64.3 | 64.3    | 71.0   |
| WLASL-2000 (ASL)*    | 22.1 | 24.4    | 27.5   |
| AUTSL (Turkish)      | 92.3 | 91.7    | 93.7   |
| INCLUDE (Indian)     | 97.5 | 97.0    | 96.7   |

*57% class-coverage subset (relative comparison only). INCLUDE is saturated, so fusion gain shrinks as headroom vanishes.

## Quickstart

    git clone https://github.com/AD1PRAJAPATI/SignCOMPASS.git
    cd SignCOMPASS
    conda create -n compass python=3.10 -y && conda activate compass
    pip install -r requirements.txt

## Train and evaluate

Each script takes a --dataset flag (asl_citizen, wlasl, autsl, include).

1. Extract features

    python extract_pose_islr.py        --dataset asl_citizen
    python extract_shubert_features.py --dataset asl_citizen

2. Train the pose stream

    python train_islr.py --dataset asl_citizen

3. Fine-tune the appearance stream

    python train_shubert_ft.py --dataset asl_citizen

4. Fuse and evaluate

    python multi_ensemble.py --dataset asl_citizen --alpha 0.5

Fusion rule: P(y|V) = (1 - alpha) * p_pose + alpha * p_shubert, with alpha tuned on validation (best alpha = 0.5 on ASL-Citizen).

## Method

Each stream ends in an ArcFace head (s * cos(theta_y + m), m=0.3, s=64) over 2,731 signs. The pose stream is a 4-layer Conformer on 261-d MediaPipe Holistic keypoints. The appearance stream feeds face/hand crops through a frozen DINOv2 ViT-S/14 and a fine-tuned 12-layer SHuBERT transformer. Both attention-pool before the head. Scores are averaged with one tuned weight, so fusion adds no parameters.

## Requirements

Python 3.10 and a CUDA GPU for training. See requirements.txt.

## Citation

    @inproceedings{prajapati2026compass,
      title  = {COMPASS: Complementary Motion and Appearance Streams for Isolated Sign Recognition},
      author = {Prajapati, Aditeya and Thomason, Jesse},
      year   = {2026}
    }

## Acknowledgements

Built at the USC GLAMOR Lab. Uses SHuBERT, DINOv2, and MediaPipe. Datasets belong to their respective authors; no video data is included in this repo.
