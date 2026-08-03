"""
extract_shubert_features.py  —  Extract SHuBERT frame embeddings from ASL-Citizen MP4s.

SHuBERT (ACL 2025): self-supervised 4-stream encoder pretrained on ~1,000h ASL.
  Paper:   https://arxiv.org/abs/2411.16765
  Weights: https://drive.google.com/drive/folders/1aOZEkENp2B-5sRq5F67dYsirnHwsFjKV
  Code:    https://github.com/ShesterG/SHuBERT

PREREQUISITE: Run setup_shubert.sh first.

4-STREAM INPUT PER FRAME:
    face        : (T, 768)  — DINOv2 features from face crop
    left_hand   : (T, 768)  — DINOv2 features from left-hand crop
    right_hand  : (T, 768)  — DINOv2 features from right-hand crop
    body_posture: (T, 258)  — MediaPipe body keypoints (same as pose stream)

OUTPUT: per-video (T, 768) float16 tensor — last layer of SHuBERT transformer.
        Saved to: data/asl_citizen/shubert_features/<video_id>.pt

PIPELINE PER VIDEO (automated here):
    1. Decode frames with cv2.VideoCapture
    2. MediaPipe Holistic → body_posture features + landmark positions
    3. Crop face and hand regions using MediaPipe landmarks
    4. Resize crops → DINOv2 feature extraction (finetuned models)
    5. Feed all 4 streams to SHuBERT → last layer [T, 768]
    6. Save as float16

NOTE: ASL-Citizen videos are already signer-cropped (webcam recordings), so we
      skip the YOLOv8 signer-crop step from QUICKSTART.md. We go straight to
      MediaPipe → DINOv2 → SHuBERT.

USAGE:
    python extract_shubert_features.py
    python extract_shubert_features.py --force     # re-extract all
    python extract_shubert_features.py --workers 1 # single GPU, debug
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm

# ─── Default paths ────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BASE_DIR     = "/project2/jessetho_1732/aditeya"
ASL_DIR      = os.path.join(PROJECT_ROOT, "data", "asl_citizen")

SHUBERT_REPO = os.path.join(BASE_DIR, "SHuBERT")
WEIGHTS_DIR  = os.path.join(BASE_DIR, "data", "shubert_weights")

VIDEO_DIR    = os.path.join(ASL_DIR, "ASL_Citizen", "videos")
OUT_DIR      = os.path.join(ASL_DIR, "shubert_features")

# DINOv2 input resolution
DINO_IMG_SIZE = 224

# Face / hand crop sizes (pixels, relative to cropped signer frame)
FACE_CROP_SIZE = 224
HAND_CROP_SIZE = 224

# SHuBERT output dimension (hidden size of the transformer)
SHUBERT_DIM = 768


# ─── MediaPipe setup ──────────────────────────────────────────────────────────

def get_mediapipe():
    """Load MediaPipe Holistic and Tasks landmarkers."""
    try:
        import mediapipe as mp
        return mp.solutions.holistic
    except Exception:
        raise ImportError(
            "mediapipe not installed.\n"
            "Fix: pip install mediapipe==0.10.14 --break-system-packages"
        )


def landmark_to_px(lm, w, h):
    """Convert normalised MediaPipe landmark to pixel coords."""
    return int(lm.x * w), int(lm.y * h)


def crop_region(frame, cx, cy, size, w, h):
    """
    Crop a square region of `size` pixels centred at (cx, cy).
    Pads with zeros if out of bounds.
    """
    half = size // 2
    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half

    pad_l = max(0, -x1)
    pad_t = max(0, -y1)
    pad_r = max(0, x2 - w)
    pad_b = max(0, y2 - h)

    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)

    crop = frame[y1:y2, x1:x2]
    if pad_l or pad_t or pad_r or pad_b:
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(crop, (size, size))


def extract_crops_and_pose(holistic, frame_rgb: np.ndarray):
    """
    Run MediaPipe Holistic on one frame.
    Returns:
        face_crop   : (224, 224, 3) uint8 RGB  or None
        lh_crop     : (224, 224, 3) uint8 RGB  or None
        rh_crop     : (224, 224, 3) uint8 RGB  or None
        pose_vec    : (258,) float32            or zeros
    """
    h, w = frame_rgb.shape[:2]
    results = holistic.process(frame_rgb)

    # ── Body pose vector (re-use extract_pose logic) ──────────────────────────
    from extract_pose_islr import frame_to_pose_vector
    pose_vec = frame_to_pose_vector(holistic, frame_rgb)   # uses existing holistic result
    # NOTE: frame_to_pose_vector runs holistic internally; here we pass the frame again.
    # For efficiency in production, refactor to share the holistic results.

    # ── Face crop (use nose + eye landmarks for centre/scale) ────────────────
    face_crop = np.zeros((FACE_CROP_SIZE, FACE_CROP_SIZE, 3), dtype=np.uint8)
    if results.face_landmarks:
        # Use nose tip (1) and left/right eye outer corners (33, 263) for crop
        nose = results.face_landmarks.landmark[1]
        cx, cy = landmark_to_px(nose, w, h)
        # Estimate face size from eye distance
        lye = results.face_landmarks.landmark[33]
        rye = results.face_landmarks.landmark[263]
        eye_dist = abs(landmark_to_px(lye, w, h)[0] - landmark_to_px(rye, w, h)[0])
        face_size = max(FACE_CROP_SIZE, eye_dist * 4)
        face_crop = crop_region(frame_rgb, cx, cy, int(face_size), w, h)
        face_crop = cv2.resize(face_crop, (FACE_CROP_SIZE, FACE_CROP_SIZE))

    # ── Hand crops (centre on wrist landmark) ─────────────────────────────────
    def hand_crop(hand_landmarks):
        if hand_landmarks is None:
            return np.zeros((HAND_CROP_SIZE, HAND_CROP_SIZE, 3), dtype=np.uint8)
        wrist = hand_landmarks.landmark[0]
        mid   = hand_landmarks.landmark[9]   # middle finger MCP — gives hand centre
        cx = (landmark_to_px(wrist, w, h)[0] + landmark_to_px(mid, w, h)[0]) // 2
        cy = (landmark_to_px(wrist, w, h)[1] + landmark_to_px(mid, w, h)[1]) // 2
        # Estimate hand size from wrist-to-middle distance
        dist = max(50, int(((landmark_to_px(mid, w, h)[0] - landmark_to_px(wrist, w, h)[0])**2 +
                            (landmark_to_px(mid, w, h)[1] - landmark_to_px(wrist, w, h)[1])**2)**0.5 * 3))
        crop = crop_region(frame_rgb, cx, cy, dist, w, h)
        return cv2.resize(crop, (HAND_CROP_SIZE, HAND_CROP_SIZE))

    lh_crop = hand_crop(results.left_hand_landmarks)
    rh_crop = hand_crop(results.right_hand_landmarks)

    return face_crop, lh_crop, rh_crop, pose_vec


# ─── DINOv2 feature extraction ────────────────────────────────────────────────

def _find_weight(weights_dir: str, candidates: list) -> str:
    """Return the first existing file from a list of candidate names."""
    for name in candidates:
        p = os.path.join(weights_dir, name)
        if os.path.exists(p):
            return p
    return None


def load_dino_models(weights_dir: str, device: torch.device):
    """
    Load fine-tuned DINOv2 models for face and hands.
    Returns dict: {'face': model, 'left_hand': model, 'right_hand': model}

    Actual Drive filenames:
        face_dinov2_checkpoint.pth  (or symlink dino_face.pt)
        hands_dinov2_checkpoint.pth (same model for left AND right hand)
    """
    # SHuBERT uses DINOv2-base (ViT-B/14) as backbone
    try:
        import timm
        base_model = timm.create_model("vit_base_patch14_dinov2.lvd142m",
                                       pretrained=True, num_classes=0)
    except Exception:
        base_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

    # Map each stream to candidate filenames (actual Drive name first, symlink second)
    ckpt_map = {
        "face":       ["face_dinov2_checkpoint.pth", "dino_face.pt"],
        "left_hand":  ["hands_dinov2_checkpoint.pth", "dino_left_hand.pt"],
        "right_hand": ["hands_dinov2_checkpoint.pth", "dino_right_hand.pt"],
    }
    # Cache loaded models so we don't load hands_dinov2 twice
    loaded_ckpts = {}
    models = {}

    for stream, candidates in ckpt_map.items():
        ckpt_path = _find_weight(weights_dir, candidates)
        if ckpt_path is None:
            print(f"  WARNING: No checkpoint found for {stream} — using base DINOv2")
            import copy
            m = copy.deepcopy(base_model)
            models[stream] = m.eval().to(device)
            continue

        import copy
        m = copy.deepcopy(base_model)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        m.load_state_dict(state, strict=False)
        models[stream] = m.eval().to(device)
        print(f"  Loaded {stream} DINOv2 from {os.path.basename(ckpt_path)}")

    return models


@torch.no_grad()
def dino_features(model, crop_rgb: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Extract (768,) DINOv2 CLS token from a (H, W, 3) uint8 RGB crop.
    Returns float32 numpy array.
    """
    import torchvision.transforms.functional as TF
    from PIL import Image

    img = Image.fromarray(crop_rgb)
    img = TF.resize(img, (DINO_IMG_SIZE, DINO_IMG_SIZE))
    x   = TF.to_tensor(img).unsqueeze(0).to(device)
    x   = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    out = model(x)   # (1, 768) CLS token
    return out.squeeze(0).cpu().float().numpy()


# ─── SHuBERT model loading ────────────────────────────────────────────────────

def load_shubert(shubert_repo: str, weights_dir: str, device: torch.device):
    """
    Load SHuBERT transformer from weights.
    Adds the SHuBERT fairseq code to sys.path and imports SHubertModel.
    """
    fairseq_path = os.path.join(shubert_repo, "fairseq")
    if fairseq_path not in sys.path:
        sys.path.insert(0, fairseq_path)
    if shubert_repo not in sys.path:
        sys.path.insert(0, shubert_repo)

    try:
        from examples.shubert.models.shubert import SHubertModel, SHubertConfig
    except ImportError as e:
        raise ImportError(
            f"Cannot import SHuBERT model ({e}).\n"
            f"Make sure {fairseq_path} is installed: cd {fairseq_path} && pip install -e ."
        )

    # Find checkpoint — actual filename is checkpoint_836_400000.pt
    ckpt_path = _find_weight(weights_dir, [
        "shubert.pt",               # symlink created by setup_shubert.sh
        "checkpoint_836_400000.pt", # actual Drive filename
    ])
    if ckpt_path is None:
        # Fallback: any .pt file that looks like a shubert checkpoint
        alts = [f for f in os.listdir(weights_dir)
                if f.endswith(".pt") and "checkpoint" in f.lower()]
        if alts:
            ckpt_path = os.path.join(weights_dir, sorted(alts)[-1])
            print(f"  Using checkpoint: {os.path.basename(ckpt_path)}")
        else:
            raise FileNotFoundError(
                f"SHuBERT checkpoint not found in {weights_dir}.\n"
                "Expected: checkpoint_836_400000.pt\n"
                "Run setup_shubert.sh to download weights."
            )

    cfg   = SHubertConfig()
    model = SHubertModel(cfg)
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    print(f"Loaded SHuBERT from {os.path.basename(ckpt_path)}")
    return model


@torch.no_grad()
def shubert_extract(model, face_feats, lh_feats, rh_feats, pose_feats, device):
    """
    Run SHuBERT on one video's pre-extracted stream features.

    Args:
        face_feats  : (T, 768) float32
        lh_feats    : (T, 768) float32
        rh_feats    : (T, 768) float32
        pose_feats  : (T, 258) float32

    Returns: (T, 768) float32 — last transformer layer
    """
    T = face_feats.shape[0]
    source = [{
        "face":         torch.from_numpy(face_feats).float().to(device),
        "left_hand":    torch.from_numpy(lh_feats).float().to(device),
        "right_hand":   torch.from_numpy(rh_feats).float().to(device),
        "body_posture": torch.from_numpy(pose_feats).float().to(device),
        # Dummy labels (not used during inference)
        "label_face":        torch.zeros(T, 1).to(device),
        "label_left_hand":   torch.zeros(T, 1).to(device),
        "label_right_hand":  torch.zeros(T, 1).to(device),
        "label_body_posture": torch.zeros(T, 1).to(device),
    }]
    result = model.extract_features(source, padding_mask=None,
                                     kmeans_labels=None, mask=False)
    # Last layer output: shape [T, B, D] → squeeze B → [T, D]
    last_layer = result["layer_results"][-1][-1]   # last layer's hidden states
    return last_layer.squeeze(1).cpu().float().numpy()  # (T, 768)


# ─── Per-video extraction ─────────────────────────────────────────────────────

def extract_video(video_path, shubert, dino_models, device, max_frames=None):
    """
    Full pipeline for one video: decode → MediaPipe → DINOv2 → SHuBERT → (T, 768).
    """
    mp_holistic = get_mediapipe()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames_rgb = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames_rgb:
        raise RuntimeError(f"No frames decoded: {video_path}")

    if max_frames and len(frames_rgb) > max_frames:
        # Centre subsample
        step = len(frames_rgb) / max_frames
        frames_rgb = [frames_rgb[int(i * step)] for i in range(max_frames)]

    face_list, lh_list, rh_list, pose_list = [], [], [], []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=True,   # needed for precise face crop
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for frame in frames_rgb:
            face_crop, lh_crop, rh_crop, pose_vec = extract_crops_and_pose(holistic, frame)

            face_list.append(dino_features(dino_models["face"],       face_crop, device))
            lh_list.append(  dino_features(dino_models["left_hand"],  lh_crop,   device))
            rh_list.append(  dino_features(dino_models["right_hand"], rh_crop,   device))
            pose_list.append(pose_vec)

    face_feats = np.stack(face_list)   # (T, 768)
    lh_feats   = np.stack(lh_list)    # (T, 768)
    rh_feats   = np.stack(rh_list)    # (T, 768)
    pose_feats = np.stack(pose_list)  # (T, 258)

    # Run SHuBERT
    features = shubert_extract(shubert, face_feats, lh_feats, rh_feats, pose_feats, device)
    return features  # (T, 768)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Extract SHuBERT frame embeddings from ASL-Citizen MP4 videos."
    )
    ap.add_argument("--video_dir",   default=VIDEO_DIR)
    ap.add_argument("--out_dir",     default=OUT_DIR)
    ap.add_argument("--shubert_dir", default=SHUBERT_REPO,
                    help="Path to cloned ShesterG/SHuBERT repo.")
    ap.add_argument("--weights_dir", default=WEIGHTS_DIR,
                    help="Path to downloaded SHuBERT + DINOv2 weights.")
    ap.add_argument("--force",       action="store_true",
                    help="Re-extract even if output exists.")
    ap.add_argument("--max_frames",  type=int, default=None,
                    help="Max frames per video (default: all frames).")
    ap.add_argument("--ext",         default=".mp4")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load models ────────────────────────────────────────────────────────────
    print("\nLoading DINOv2 models...")
    dino_models = load_dino_models(args.weights_dir, device)

    print("\nLoading SHuBERT...")
    shubert = load_shubert(args.shubert_dir, args.weights_dir, device)

    # ── Scan videos ────────────────────────────────────────────────────────────
    all_videos = sorted([f for f in os.listdir(args.video_dir)
                         if f.lower().endswith(args.ext)])
    if not all_videos:
        print(f"No {args.ext} files in {args.video_dir}")
        sys.exit(1)

    remaining = [f for f in all_videos if args.force or not os.path.exists(
        os.path.join(args.out_dir, os.path.splitext(f)[0] + ".pt"))]

    print(f"\nVideos: {len(all_videos)} total | {len(all_videos)-len(remaining)} done "
          f"| {len(remaining)} remaining")

    # ── Extract ────────────────────────────────────────────────────────────────
    failed = []
    for fname in tqdm(remaining, desc="SHuBERT extraction"):
        video_id  = os.path.splitext(fname)[0]
        out_path  = os.path.join(args.out_dir, video_id + ".pt")
        video_path = os.path.join(args.video_dir, fname)
        try:
            feats = extract_video(video_path, shubert, dino_models, device,
                                  max_frames=args.max_frames)
            torch.save(torch.from_numpy(feats).half(), out_path)
        except Exception as e:
            print(f"\n  WARN: {video_id} — {e}")
            failed.append(video_id)

    n_done = len(remaining) - len(failed)
    print(f"\nDone: {n_done}/{len(remaining)} extracted → {args.out_dir}")
    print(f"Feature shape: (T, {SHUBERT_DIM}) float16")
    if failed:
        print(f"Failed: {len(failed)}")
        with open(os.path.join(args.out_dir, "failed.txt"), "w") as f:
            f.write("\n".join(failed))


if __name__ == "__main__":
    main()
