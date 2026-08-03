"""
extract_videomae.py  —  RGB video stream for the fusion model.

Produces per-video (T_w, 768) embeddings using VideoMAE (HuggingFace), saved to
    data/asl_citizen/rgb_features/<video_id>.pt   (float16)

This is the RGB/appearance stream that fuses with the pose stream. It captures
handshape/appearance detail that MediaPipe keypoints lose. VideoMAE-base hidden
size is 768, matching FusionISLRModel's rgb_dim default.

Method: slide a 16-frame window (stride `--stride`) over the clip; for each window
run VideoMAE and mean-pool its patch tokens → one 768-d vector per window. The
resulting (num_windows, 768) sequence is pooled by the model's AttentionPool, so
it does NOT need to align frame-for-frame with the pose stream.

NOTE: For the *best possible* model, swap VideoMAE for SHuBERT features later —
just write SHuBERT (T,768) tensors into a parallel dir and point train_fusion at it.
VideoMAE is the immediately-runnable strong baseline RGB stream.

Install:
    pip install transformers decord av --break-system-packages

Run:
    python extract_videomae.py --workers 1           # GPU; 1 proc, internal batching
    python extract_videomae.py --model MCG-NJU/videomae-base
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

WINDOW = 16   # VideoMAE default temporal size


def read_video_frames(path, max_frames=256):
    """Decode an mp4 to a list of HxWx3 uint8 RGB frames (downsampled if very long)."""
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]
    return frames


def make_windows(n_frames, window=WINDOW, stride=8):
    """Return list of frame-index windows (each length=window, padded by repeat)."""
    if n_frames == 0:
        return []
    starts = list(range(0, max(1, n_frames - window + 1), stride))
    if not starts:
        starts = [0]
    wins = []
    for s in starts:
        idx = list(range(s, min(s + window, n_frames)))
        if len(idx) < window:                      # pad by repeating last frame
            idx += [idx[-1]] * (window - len(idx))
        wins.append(idx)
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--video_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--model", default="MCG-NJU/videomae-base")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--max_frames", type=int, default=256)
    ap.add_argument("--clip_batch", type=int, default=16, help="windows per forward pass")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from transformers import VideoMAEModel, VideoMAEImageProcessor

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = args.data_root or os.path.dirname(script_dir)
    asl_dir = os.path.join(data_root, "data", "asl_citizen")
    video_dir = args.video_dir or os.path.join(asl_dir, "ASL_Citizen", "videos")
    out_dir = args.out_dir or os.path.join(asl_dir, "rgb_features")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(video_dir):
        print(f"[ERROR] video dir not found: {video_dir}")
        sys.exit(1)

    vids = sorted(f for f in os.listdir(video_dir) if f.lower().endswith(".mp4"))
    if not args.force:
        vids = [f for f in vids
                if not os.path.exists(os.path.join(out_dir, os.path.splitext(f)[0] + ".pt"))]
    print(f"To process: {len(vids)} videos")
    if not vids:
        print("All RGB features already extracted. Use --force to redo.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = VideoMAEImageProcessor.from_pretrained(args.model)
    model = VideoMAEModel.from_pretrained(args.model).to(device).eval()

    for f in tqdm(vids, desc="VideoMAE"):
        vid_id = os.path.splitext(f)[0]
        out_path = os.path.join(out_dir, vid_id + ".pt")
        try:
            frames = read_video_frames(os.path.join(video_dir, f), args.max_frames)
            wins = make_windows(len(frames), WINDOW, args.stride)
            if not wins:
                continue
            embs = []
            for i in range(0, len(wins), args.clip_batch):
                batch_wins = wins[i:i + args.clip_batch]
                clips = [[frames[j] for j in w] for w in batch_wins]  # list of 16-frame clips
                inputs = proc(clips, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model(**inputs).last_hidden_state      # (b, num_patches, 768)
                embs.append(out.mean(dim=1).float().cpu())        # (b, 768) per window
            seq = torch.cat(embs, dim=0)                          # (num_windows, 768)
            torch.save(seq.half(), out_path)
        except Exception as e:
            print(f"  WARN {vid_id}: {type(e).__name__}: {e}")

    print(f"Done. RGB features (T_w, 768) float16 -> {out_dir}")


if __name__ == "__main__":
    main()
