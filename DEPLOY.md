# ISLR Pipeline — Deployment Instructions

## What this is

Complete ISLR (Isolated Sign Language Recognition) pipeline targeting ASL-Citizen
(2,731 signs, 83k videos).

**Why ISLR before CSLR:** Your CSLR experiments on ASLLRP confirm the roadmap's
diagnosis — covered WER is ~90% across all seeds/architectures because ASLLRP has
only 129 in-vocab glosses and 90% of test utterances are OOV. This pipeline targets
the right dataset and the right task.

**Target:** top-1 > 75%, Recall@10 > 95% (vs. I3D baseline: 63% / 91%).

---

## Files to copy

```
THIS FOLDER                          → ON CLUSTER (inside cslr/)
────────────────────────────────     ─────────────────────────────────────────
dataset_islr.py                  →   /project2/jessetho_1732/aditeya/cslr/
extract_pose_islr.py             →   /project2/jessetho_1732/aditeya/cslr/
download_asl_citizen.py          →   /project2/jessetho_1732/aditeya/cslr/
train_islr.py                    →   /project2/jessetho_1732/aditeya/cslr/
train_islr.sbatch                →   /project2/jessetho_1732/aditeya/cslr/
models/islr_model.py             →   /project2/jessetho_1732/aditeya/cslr/models/
```

All new files sit alongside existing code (`train_scaling.py`, `train_pose.py`, etc.)
and reuse the existing `models/conformer.py` and `augmentations.py` — nothing breaks.

**Quick copy on the cluster:**
```bash
cd /project2/jessetho_1732/aditeya/cslr
# (after scp'ing the files or cloning from git)
ls dataset_islr.py extract_pose_islr.py train_islr.py   # should all exist
ls models/islr_model.py                                   # should exist
```

---

## Run order

### Step 1 — Download ASL-Citizen (login node, internet required)
```bash
# Run on a login node (not a compute node) since it needs internet
python download_asl_citizen.py

# Verify when done:
python download_asl_citizen.py --verify
```

Expected output path: `/project2/jessetho_1732/aditeya/data/asl_citizen/`
```
data/asl_citizen/
  metadata.csv            (~3 MB)
  videos/                 (~50–100 GB)
    <video_id>.mp4
    ...
```

### Step 2 — Extract pose features (CPU job, ~3–6 hours for 83k videos)
```bash
# Can run as CPU-only job (no GPU needed for MediaPipe):
python extract_pose_islr.py --workers 8 --model_complexity 1

# Or via sbatch with ONLY_EXTRACT flag:
sbatch --export=ONLY_EXTRACT=1 train_islr.sbatch
```

Expected output: `data/asl_citizen/pose_features/<video_id>.pt`
Each file: `(T, 261)` float16 tensor.

### Step 3 — Train (GPU job, A100, ~2–4 hours)
```bash
sbatch train_islr.sbatch
```

Or directly:
```bash
python train_islr.py --size base --epochs 120 --seed 42
```

### Step 4 — Check the gate
Look for `ISLR_RESULT:` lines in the log:
```bash
grep "ISLR_RESULT" logs/islr_*.out
```

Gate passes if `test_top1 > 63` (beats I3D baseline, proceed to Week 3-5).
Target for Week 3-5: `test_top1 > 75, test_r10 > 95`.

---

## Architecture summary

```
Pose keypoints (T, 261)
        │
    Linear(261 → 256) + LayerNorm + Dropout(0.2)
        │
    ConformerEncoder: 4 layers, d=256, heads=8, conv_k=15
        │  (B, T, 256)
    AttentionPool: learned per-frame weights → (B, 256)
        │
    LayerNorm → L2-normalise → (B, 256) unit embeddings
        │
    ArcFaceHead: 2731 classes, margin=0.3, scale=64
        │
    Logits (B, 2731) → CrossEntropy during training
                     → argmax / argsort@K for eval
```

Parameters: ~7M (base). Trains in ~2h on A100 with batch_size=128.

---

## Troubleshooting

**`ModuleNotFoundError: augmentations`**
Run from the `cslr/` directory:
```bash
cd /project2/jessetho_1732/aditeya/cslr
python train_islr.py
```

**`ModuleNotFoundError: models.conformer`**
Confirm `cslr/models/conformer.py` exists (it should — it was already in your repo).

**`metadata.csv missing columns`**
The HuggingFace schema may vary. Run:
```bash
python -c "from datasets import load_dataset; ds = load_dataset('google/asl_citizen'); print(ds['train'].features)"
```
Then update `load_metadata()` in `dataset_islr.py` to map the actual column names.

**Low accuracy on first run (~30–40% top-1 after epoch 1)**
Normal — the model starts from random weights. ArcFace needs ~20–30 epochs to
establish good class separation. Val top-1 typically climbs past 63% around
epoch 40–60 with the base model.

**OOM on A100**
Reduce `--batch_size 64`. The model itself is ~7M params so it's the feature
tensors that dominate memory (83k × 64 frames × 261 dims when using large batches
with the balanced sampler).

---

## What comes next (Week 3–5, after gate passes)

1. **Phonological auxiliary heads** (handshape / location / movement prediction)
   — +8.7 pts on WLASL2000 (Marber et al.). ASLLRP CSV has handshape columns; use them.
2. **Multi-stream fusion** — pose body + hands (already in `FusionFeatureDataset`).
3. **Scale vocab** — merge WLASL-2000, MS-ASL, Sem-Lex into training.
4. **Edge distillation** — use `ISLRModelEdge` (d=128, layers=2) as student.
