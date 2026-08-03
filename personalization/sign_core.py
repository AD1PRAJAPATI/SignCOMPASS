"""
sign_core.py — shared helpers for the live demo AND personalization.
Load the pose model, turn MediaPipe keypoints into a 256-d embedding (a sign
"fingerprint"), build the prototype gallery and do cosine-similarity top-k.
If needed:  export ISLR=/project2/jessetho_1732/aditeya/islr_pipeline
"""
import os, sys, json
import numpy as np
import torch

_ISLR = os.environ.get("ISLR",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ISLR not in sys.path:
    sys.path.insert(0, _ISLR)

D_POSE = 261  # 33*4 + 21*3 + 21*3 + 3 — must match training


def load_model(ckpt_path, vocab_path, device="cpu"):
    from models.fusion_model import build_fusion_model
    vocab = json.load(open(vocab_path))
    model = build_fusion_model(num_classes=len(vocab), pose_dim=D_POSE,
                               rgb_dim=768, size="base",
                               use_pose=True, use_rgb=False)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state.get("model_state", state))
    model.eval().to(device)
    return model, vocab


def _find_head(model, n_classes):
    if os.environ.get("SIGN_DEBUG"):
        for n, m in model.named_modules():
            print("MODULE:", n, type(m).__name__)
    for name, mod in model.named_modules():
        w = getattr(mod, "weight", None)
        if isinstance(w, torch.nn.Parameter) and w.dim() == 2 and n_classes in tuple(w.shape):
            return name, mod
    raise RuntimeError("ArcFace head not found. Re-run with SIGN_DEBUG=1 and share the module list.")


class Embedder:
    def __init__(self, model, vocab, device="cpu"):
        self.model, self.vocab, self.device = model, vocab, device
        self._cap = {}
        _, self.head = _find_head(model, len(vocab))
        self.head.register_forward_pre_hook(
            lambda m, inp: self._cap.__setitem__("e", inp[0].detach()))
        w = self.head.weight.detach().float()
        protos = w if w.shape[0] == len(vocab) else w.t()
        self.embed_dim = protos.shape[1]
        self._base = torch.nn.functional.normalize(protos, dim=1)

    @torch.no_grad()
    def embed(self, pose_np):
        pose_np = np.asarray(pose_np, dtype=np.float32)
        x = torch.from_numpy(pose_np).unsqueeze(0).to(self.device)
        ln = torch.tensor([x.shape[1]], device=self.device)
        self.model(pose_feats=x, pose_lengths=ln)
        e = self._cap["e"].reshape(1, -1).float()
        return torch.nn.functional.normalize(e, dim=1)[0].cpu()

    def base_gallery(self):
        return self._base.clone(), list(self.vocab)


def save_gallery(path, prototypes, labels, meta=None):
    torch.save({"prototypes": prototypes.cpu(), "labels": list(labels),
                "meta": meta or {}}, path)

def load_gallery(path):
    g = torch.load(path, map_location="cpu")
    return g["prototypes"].float(), list(g["labels"])

def merge_galleries(base_protos, base_labels, extra_path=None):
    protos, labels = base_protos.clone(), list(base_labels)
    if extra_path and os.path.exists(extra_path):
        ep, el = load_gallery(extra_path)
        protos = torch.cat([protos, torch.nn.functional.normalize(ep, dim=1)], 0)
        labels = labels + el
    return protos, labels

def topk(query_vec, prototypes, labels, k=5):
    sims = (prototypes @ query_vec).tolist()
    order = sorted(range(len(labels)), key=lambda i: sims[i], reverse=True)[:k]
    return [(labels[i], float(sims[i])) for i in order]


def _xyzv(landmarks, n, dims):
    out = np.zeros((n, dims), np.float32)
    if landmarks is None:
        return out, 0.0
    for i, lm in enumerate(landmarks.landmark[:n]):
        out[i, 0] = lm.x; out[i, 1] = lm.y
        if dims >= 3: out[i, 2] = lm.z
        if dims >= 4: out[i, 3] = getattr(lm, "visibility", 0.0)
    return out, 1.0

def _norm(a, c, s):
    s = s if s > 1e-6 else 1.0
    a = a.copy(); a[..., 0] = (a[..., 0] - c[0]) / s; a[..., 1] = (a[..., 1] - c[1]) / s
    return a

def result_to_pose_vec(res):
    pose, pp = _xyzv(res.pose_landmarks, 33, 4)
    lh, lp = _xyzv(res.left_hand_landmarks, 21, 3)
    rh, rp = _xyzv(res.right_hand_landmarks, 21, 3)
    if pp:
        ls, rs = pose[11, :2], pose[12, :2]
        c = (ls + rs) / 2.0; s = float(np.linalg.norm(ls - rs))
    else:
        c, s = np.array([0.5, 0.5], np.float32), 1.0
    pose[:, :2] = _norm(pose[:, :2], c, s)
    lh[:, :2] = _norm(lh[:, :2], c, s)
    rh[:, :2] = _norm(rh[:, :2], c, s)
    return np.concatenate([pose.reshape(-1), lh.reshape(-1), rh.reshape(-1),
                           np.array([pp, lp, rp], np.float32)]).astype(np.float32)

def make_holistic():
    import mediapipe as mp
    return mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

def video_to_pose(path, holistic=None, max_frames=64):
    import cv2
    own = holistic is None
    holistic = holistic or make_holistic()
    cap = cv2.VideoCapture(path); frames = []
    while True:
        ok, frame = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(result_to_pose_vec(holistic.process(rgb)))
    cap.release()
    if own: holistic.close()
    if not frames:
        raise RuntimeError("No frames decoded from %s" % path)
    return np.stack(frames)[:max_frames]
