"""Build face/hand*.list for ids with real crops (>min_bytes) and missing DINO feats.

Usage:
  python build_dino_lists_good.py [min_bytes] [splits]
  splits = comma list: train,val,test  OR  all  (default: all)
"""
import csv, gzip, os, os.path as P, pickle, sys

MINB = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
splits_arg = sys.argv[2] if len(sys.argv) > 2 else "all"
S = os.environ["SEMLEX"]
W = os.environ["WORK"]
rows = list(csv.DictReader(open(P.join(S, "metadata.csv"))))
if splits_arg.strip().lower() == "all":
    want = {"train", "val", "test"}
else:
    want = {s.strip().lower() for s in splits_arg.split(",") if s.strip()}
ids = [r["video_id"] for r in rows if str(r["split"]).strip().lower() in want]
print(f"splits={sorted(want)} candidates={len(ids)} min_bytes={MINB}")


def ok_crop(p):
    try:
        return P.getsize(p) > MINB
    except OSError:
        return False


def ok_npy(p):
    try:
        return P.getsize(p) > 100
    except OSError:
        return False


need = []
for v in ids:
    fc = P.join(W, "face_crops", v + "_face.mp4")
    h1 = P.join(W, "hand_crops", v + "_hand1.mp4")
    h2 = P.join(W, "hand_crops", v + "_hand2.mp4")
    if not (ok_crop(fc) and ok_crop(h1) and ok_crop(h2)):
        continue
    if (
        ok_npy(P.join(W, "face_feats", v + "_face.npy"))
        and ok_npy(P.join(W, "hand1_feats", v + "_hand1.npy"))
        and ok_npy(P.join(W, "hand2_feats", v + "_hand2.npy"))
    ):
        continue
    need.append(v)


def dump(o, paths):
    with gzip.GzipFile(o, "wb") as f:
        f.write(pickle.dumps(paths, protocol=0))
    print(o, len(paths))


dump(P.join(W, "face.list"), [P.join(W, "face_crops", v + "_face.mp4") for v in need])
dump(P.join(W, "hand1.list"), [P.join(W, "hand_crops", v + "_hand1.mp4") for v in need])
dump(P.join(W, "hand2.list"), [P.join(W, "hand_crops", v + "_hand2.mp4") for v in need])
good = sum(
    1
    for v in ids
    if ok_crop(P.join(W, "face_crops", v + "_face.mp4"))
    and ok_crop(P.join(W, "hand_crops", v + "_hand1.mp4"))
    and ok_crop(P.join(W, "hand_crops", v + "_hand2.mp4"))
)
print(f"good crops with all 3 (selected splits): {good} / {len(ids)}")
print(f"need dino: {len(need)}")
