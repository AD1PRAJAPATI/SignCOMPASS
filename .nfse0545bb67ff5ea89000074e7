import argparse, torch
from torch.utils.data import DataLoader
from dataset_fusion import build_fusion_datasets, collate_fusion
from models.fusion_model import build_fusion_model

ap = argparse.ArgumentParser()
ap.add_argument("--pose_ckpt", required=True)
ap.add_argument("--rgb_ckpt", required=True)
ap.add_argument("--rgb_dir", required=True)
ap.add_argument("--data_root", default="/project2/jessetho_1732/aditeya")
a = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset with BOTH streams so test set is consistent
tr, va, te, vocab = build_fusion_datasets(a.data_root, use_pose=True, use_rgb=True,
                                          rgb_dir=a.rgb_dir, max_frames=64)
NC = len(vocab)

def load(ckpt, up, ur):
    m = build_fusion_model(num_classes=NC, pose_dim=261, rgb_dim=768, size="base",
                           use_pose=up, use_rgb=ur)
    m.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state"])
    return m.eval().to(dev)

pose_m = load(a.pose_ckpt, True, False)
rgb_m  = load(a.rgb_ckpt, False, True)

@torch.no_grad()
def get_probs(ds):
    dl = DataLoader(ds, batch_size=256, collate_fn=collate_fusion, num_workers=0)
    Pp, Pr, Y = [], [], []
    for b in dl:
        pf, pl = b["pose_feats"].to(dev), b["pose_lengths"].to(dev)
        rf, rl = b["rgb_feats"].to(dev), b["rgb_lengths"].to(dev)
        Pp.append(torch.softmax(pose_m(pose_feats=pf, pose_lengths=pl)["logits"], -1).cpu())
        Pr.append(torch.softmax(rgb_m(rgb_feats=rf, rgb_lengths=rl)["logits"], -1).cpu())
        Y.append(b["labels"])
    return torch.cat(Pp), torch.cat(Pr), torch.cat(Y)

def topk(P, Y, ks=(1,5,10)):
    o = P.topk(max(ks), 1).indices
    return {k: 100.0*(o[:, :k] == Y[:, None]).any(1).float().mean().item() for k in ks}

print("scoring val..."); vPp, vPr, vY = get_probs(va)
print("scoring test..."); tPp, tPr, tY = get_probs(te)

best_a, best = 0.0, -1
for i in range(0, 21):
    al = i/20
    m = topk((1-al)*vPp + al*vPr, vY)[1]
    if m > best: best, best_a = m, al
print(f"best alpha (rgb weight) on val = {best_a:.2f}  val_top1={best:.2f}")

res = topk((1-best_a)*tPp + best_a*tPr, tY)
print(f"ENSEMBLE TEST: top1={res[1]:.2f}  top5={res[5]:.2f}  r10={res[10]:.2f}")
print(f"  (pose-only test top1 = {topk(tPp,tY)[1]:.2f} | rgb-only = {topk(tPr,tY)[1]:.2f})")
