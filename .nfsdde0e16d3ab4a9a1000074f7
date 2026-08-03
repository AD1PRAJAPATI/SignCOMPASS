"""
shubert_convert.py  —  convert SHuBERT inference output to fusion-ready features.

shubert_inference.py saves per-video [L, T, 768] .npy (all transformer layers).
The fusion model wants (T, 768). We take the LAST layer (most semantic) and save
as float16 .pt into shubert_features/, the dir train_fusion.py --rgb_dir points at.

Usage:
    python shubert_convert.py --in_dir  .../shubert_npy \
                              --out_dir .../data/asl_citizen/shubert_features \
                              --layer -1
"""
import argparse, glob, os
import numpy as np
import torch
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--in_dir", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--layer", type=int, default=-1, help="which transformer layer (-1=last)")
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

files = glob.glob(os.path.join(a.in_dir, "*.npy"))
for p in tqdm(files, desc="convert"):
    arr = np.load(p)                       # [L, T, 768]  (or [T,768] if single layer saved)
    seq = arr[a.layer] if arr.ndim == 3 else arr
    out = os.path.join(a.out_dir, os.path.splitext(os.path.basename(p))[0] + ".pt")
    torch.save(torch.from_numpy(seq).half(), out)
print(f"Converted {len(files)} files -> {a.out_dir}  (shape (T,768) float16)")
