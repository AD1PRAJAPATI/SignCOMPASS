"""Print Sem-Lex feature coverage by split. Run on cluster with SEMLEX/WORK set."""
import csv, os, os.path as P, sys

MINB = 1024
S = os.environ["SEMLEX"]
W = os.environ["WORK"]
rows = list(csv.DictReader(open(P.join(S, "metadata.csv"))))


def ok(p, minb=MINB):
    try:
        return P.getsize(p) > minb
    except OSError:
        return False


def ex(p):
    return P.exists(p)


for split in ("train", "val", "test"):
    ids = [r["video_id"] for r in rows if str(r["split"]).strip().lower() == split]
    n = len(ids)

    def c(fn):
        return sum(1 for v in ids if fn(v))

    print(f"\n{split} n={n}")
    print(f"  mp4              {c(lambda v: ex(P.join(S,'videos_mp4',v+'.mp4')))}")
    print(f"  pose.json        {c(lambda v: ex(P.join(W,'pose',v+'_pose.json')))}")
    print(f"  face_crop>1KB    {c(lambda v: ok(P.join(W,'face_crops',v+'_face.mp4')))}")
    print(f"  h1_crop>1KB      {c(lambda v: ok(P.join(W,'hand_crops',v+'_hand1.mp4')))}")
    print(f"  h2_crop>1KB      {c(lambda v: ok(P.join(W,'hand_crops',v+'_hand2.mp4')))}")
    print(f"  face_feat        {c(lambda v: ok(P.join(W,'face_feats',v+'_face.npy'),0))}")
    print(f"  h1_feat          {c(lambda v: ok(P.join(W,'hand1_feats',v+'_hand1.npy'),0))}")
    print(f"  h2_feat          {c(lambda v: ok(P.join(W,'hand2_feats',v+'_hand2.npy'),0))}")
    print(f"  body_feat        {c(lambda v: ok(P.join(W,'body_feats',v+'_pose.npy'),0))}")
    print(f"  pose_features.pt {c(lambda v: ok(P.join(S,'pose_features',v+'.pt'),0))}")
    sh = c(
        lambda v: ok(P.join(W, "face_feats", v + "_face.npy"), 0)
        and ok(P.join(W, "hand1_feats", v + "_hand1.npy"), 0)
        and ok(P.join(W, "hand2_feats", v + "_hand2.npy"), 0)
        and ok(P.join(W, "body_feats", v + "_pose.npy"), 0)
    )
    full = c(
        lambda v: ok(P.join(S, "pose_features", v + ".pt"), 0)
        and ok(P.join(W, "face_feats", v + "_face.npy"), 0)
        and ok(P.join(W, "hand1_feats", v + "_hand1.npy"), 0)
        and ok(P.join(W, "hand2_feats", v + "_hand2.npy"), 0)
        and ok(P.join(W, "body_feats", v + "_pose.npy"), 0)
    )
    print(f"  SHuBERT-ready    {sh}")
    print(f"  ensemble-ready   {full}")

# write train usable for gates
train_ids = [r["video_id"] for r in rows if str(r["split"]).strip().lower() == "train"]
train_sh = sum(
    1
    for v in train_ids
    if ok(P.join(W, "face_feats", v + "_face.npy"), 0)
    and ok(P.join(W, "hand1_feats", v + "_hand1.npy"), 0)
    and ok(P.join(W, "hand2_feats", v + "_hand2.npy"), 0)
    and ok(P.join(W, "body_feats", v + "_pose.npy"), 0)
)
out = P.join(os.environ.get("ISLR", "."), "logs", "semlex_train_usable.txt")
os.makedirs(P.dirname(out), exist_ok=True)
open(out, "w").write(f"{train_sh}\n")
print(f"\nwrote {out}: train SHuBERT-ready={train_sh}")
sys.stdout.flush()
