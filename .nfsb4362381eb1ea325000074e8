"""
extract_pose_islr.py  —  Pre-extract MediaPipe Holistic keypoints from ASL-Citizen MP4 videos.

WHY: Same reasoning as extract_pose.py for ASLLRP — frozen CLIP discards the fine
motor detail (handshape, finger configuration) that IS the sign. MediaPipe Holistic
gives compact, motion-rich, scale-invariant features.

Output: per-video (T, D_pose) float16 tensors saved to
        data/asl_citizen/pose_features/<video_id>.pt

Feature layout per frame (261-dim, same as ASLLRP pose_features):
    pose body   : 33 landmarks × 4 (x, y, z, visibility)  = 132
    left hand   : 21 landmarks × 3 (x, y, z)              =  63
    right hand  : 21 landmarks × 3 (x, y, z)              =  63
    presence    : 3 flags (pose_present, lh_present, rh_present) =  3
    ───────────────────────────────────────────────────────────────
    D_pose = 261

Normalization (translation + scale invariant):
    - All x,y re-centred on the mid-shoulder point (shoulder midpoint = origin).
    - Divided by inter-shoulder distance (scale = 1 shoulder width).
    - Missing detections → zeros; presence flag = 0.0.

Run ONCE on the cluster (CPU is sufficient; MediaPipe does not need GPU):
    python extract_pose_islr.py                     # extract all missing
    python extract_pose_islr.py --force             # re-extract everything
    python extract_pose_islr.py --workers 8         # parallel workers

Requires:
    pip install mediapipe opencv-python --break-system-packages
"""

import argparse
import os
import sys
import traceback
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import torch
from tqdm import tqdm


# ─── Pose feature layout ──────────────────────────────────────────────────────

D_POSE = 261   # 33×4 + 21×3 + 21×3 + 3 — must match extract_pose.py for compatibility


def _xyzv(landmark_list, n: int, dims: int):
    """
    Extract (n, dims) float32 array from a MediaPipe landmark list.
    Returns zeros + presence=0 if landmark_list is None.
    """
    if landmark_list is None:
        return np.zeros((n, dims), dtype=np.float32), 0.0
    out = np.zeros((n, dims), dtype=np.float32)
    for i, lm in enumerate(landmark_list.landmark[:n]):
        out[i, 0] = lm.x
        out[i, 1] = lm.y
        if dims >= 3:
            out[i, 2] = lm.z
        if dims >= 4:
            out[i, 3] = getattr(lm, "visibility", 0.0)
    return out, 1.0


def _normalize_xy(arr, center, scale):
    """
    Re-centre and scale the x,y coordinates of an (n, ≥2) array in-place-safe.
    """
    if scale < 1e-6:
        scale = 1.0
    out = arr.copy()
    out[..., 0] = (out[..., 0] - center[0]) / scale
    out[..., 1] = (out[..., 1] - center[1]) / scale
    return out


def frame_to_pose_vector(holistic, frame_rgb: np.ndarray) -> np.ndarray:
    """
    Run MediaPipe Holistic on one HxWx3 uint8 RGB frame.
    Returns D_POSE-dimensional float32 feature vector.
    """
    results = holistic.process(frame_rgb)

    pose, pose_p = _xyzv(results.pose_landmarks,       33, 4)
    lh,   lh_p   = _xyzv(results.left_hand_landmarks,  21, 3)
    rh,   rh_p   = _xyzv(results.right_hand_landmarks, 21, 3)

    # Reference frame: mid-shoulder centre + shoulder-width scale
    if pose_p:
        ls, rs = pose[11, :2], pose[12, :2]   # left/right shoulder landmarks
        center = (ls + rs) / 2.0
        scale  = float(np.linalg.norm(ls - rs))
    else:
        center = np.array([0.5, 0.5], np.float32)
        scale  = 1.0

    pose[:, :2] = _normalize_xy(pose[:, :2], center, scale)
    lh[:, :2]   = _normalize_xy(lh[:, :2],   center, scale)
    rh[:, :2]   = _normalize_xy(rh[:, :2],   center, scale)

    presence = np.array([pose_p, lh_p, rh_p], dtype=np.float32)
    feat = np.concatenate([
        pose.reshape(-1),   # 132
        lh.reshape(-1),     #  63
        rh.reshape(-1),     #  63
        presence,           #   3
    ])
    assert feat.shape[0] == D_POSE, f"Expected {D_POSE} dims, got {feat.shape[0]}"
    return feat


def _resolve_holistic():
    """
    Robustly obtain the MediaPipe Holistic solution across install quirks.
    Raises ImportError with actionable fix message if unavailable.
    """
    try:
        import mediapipe as mp
        return mp.solutions.holistic
    except Exception:
        pass
    try:
        from mediapipe.python.solutions import holistic as h
        return h
    except Exception:
        pass
    try:
        import mediapipe.solutions.holistic as h
        return h
    except Exception as e:
        import mediapipe as mp
        ver = getattr(mp, "__version__", "unknown")
        raise ImportError(
            f"Could not load MediaPipe Holistic (mediapipe=={ver}).\n"
            "Fix: pip install 'mediapipe==0.10.14' --break-system-packages"
        ) from e


# ─── Per-video extraction ─────────────────────────────────────────────────────

def extract_one(args):
    """
    Worker function: extract pose features from one MP4 file.
    Args: (video_path, out_path, model_complexity)
    Returns: (video_id, ok, message)
    """
    video_path, out_path, model_complexity = args
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    try:
        mp_holistic = _resolve_holistic()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return video_id, False, f"Cannot open: {video_path}"

        frames_rgb = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if not frames_rgb:
            return video_id, False, f"No frames decoded: {video_path}"

        with mp_holistic.Holistic(
            static_image_mode=False,           # use tracking between frames
            model_complexity=model_complexity,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            feats = np.stack([
                frame_to_pose_vector(holistic, f) for f in frames_rgb
            ])   # (T, D_POSE)

        torch.save(
            torch.from_numpy(feats).half(),    # float16 saves ~50% disk space
            out_path
        )
        return video_id, True, ""

    except Exception as e:
        return video_id, False, f"{type(e).__name__}: {e}"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Extract MediaPipe Holistic pose features from ASL-Citizen MP4 files."
    )
    ap.add_argument("--data_root", default=None,
                    help="Project root. Defaults to parent of this script's directory.")
    ap.add_argument("--video_dir", default=None,
                    help="Override path to video files (default: data/asl_citizen/videos/).")
    ap.add_argument("--out_dir", default=None,
                    help="Override output directory (default: data/asl_citizen/pose_features/).")
    ap.add_argument("--force", action="store_true",
                    help="Re-extract even if output already exists.")
    ap.add_argument("--workers", type=int, default=min(8, cpu_count()),
                    help="Number of parallel worker processes (default: min(8, nCPU)).")
    ap.add_argument("--model_complexity", type=int, default=1, choices=[0, 1, 2],
                    help="MediaPipe model complexity (0=fast, 1=balanced, 2=accurate).")
    ap.add_argument("--ext", default=".mp4",
                    help="Video file extension to scan for (default: .mp4).")
    args = ap.parse_args()

    # ── Resolve paths ──────────────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root  = args.data_root or os.path.dirname(script_dir)
    asl_dir    = os.path.join(data_root, "data", "asl_citizen")

    video_dir = args.video_dir or os.path.join(asl_dir, "videos")
    out_dir   = args.out_dir   or os.path.join(asl_dir, "pose_features")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(video_dir):
        print(f"[ERROR] Video directory not found: {video_dir}")
        print("Run: python download_asl_citizen.py first.")
        sys.exit(1)

    # ── Scan video files ───────────────────────────────────────────────────────
    ext = args.ext.lower() if args.ext.startswith(".") else f".{args.ext.lower()}"
    all_videos = sorted([
        f for f in os.listdir(video_dir)
        if f.lower().endswith(ext)
    ])
    if not all_videos:
        print(f"[ERROR] No *{ext} files found in {video_dir}")
        sys.exit(1)

    if args.force:
        remaining = all_videos
    else:
        remaining = [
            f for f in all_videos
            if not os.path.exists(
                os.path.join(out_dir, os.path.splitext(f)[0] + ".pt")
            )
        ]

    done = len(all_videos) - len(remaining)
    print(f"Total: {len(all_videos)} | Done: {done} | Remaining: {len(remaining)}")
    if not remaining:
        print("All pose features extracted. Use --force to redo.")
        return

    # ── Build task list ────────────────────────────────────────────────────────
    tasks = [
        (
            os.path.join(video_dir, f),
            os.path.join(out_dir, os.path.splitext(f)[0] + ".pt"),
            args.model_complexity,
        )
        for f in remaining
    ]

    # ── Run extraction ─────────────────────────────────────────────────────────
    failed = []
    n_workers = min(args.workers, len(tasks))

    if n_workers <= 1:
        # Single-process (easier to debug)
        for task in tqdm(tasks, desc="Pose extraction"):
            vid_id, ok, msg = extract_one(task)
            if not ok:
                print(f"  WARN: {vid_id} — {msg}")
                failed.append(vid_id)
    else:
        with Pool(n_workers) as pool:
            for vid_id, ok, msg in tqdm(
                pool.imap_unordered(extract_one, tasks),
                total=len(tasks), desc=f"Pose extraction ({n_workers} workers)"
            ):
                if not ok:
                    print(f"  WARN: {vid_id} — {msg}")
                    failed.append(vid_id)

    # ── Summary ────────────────────────────────────────────────────────────────
    n_done   = len(remaining) - len(failed)
    n_total  = done + n_done
    print(f"\nExtraction complete.")
    print(f"  Saved: {n_done} / {len(remaining)} attempted")
    print(f"  Total done: {n_total} / {len(all_videos)}")
    print(f"  Feature shape: (T, {D_POSE}) float16 → {out_dir}")
    if failed:
        print(f"  Failed ({len(failed)}): {failed[:10]}{'...' if len(failed)>10 else ''}")
        fail_log = os.path.join(out_dir, "failed_extractions.txt")
        with open(fail_log, "w") as f:
            f.write("\n".join(failed))
        print(f"  Full list written to: {fail_log}")


if __name__ == "__main__":
    main()
