"""Rebuild INCLUDE remaining crop/body/dino lists. Args: remaining | dino"""
import gzip, os, os.path as P, pickle, sys

cmd = sys.argv[1] if len(sys.argv) > 1 else "remaining"
VID = os.environ["VID"]
W = os.environ["WORK"]
MINB = 1024


def dump(o, paths):
    with gzip.GzipFile(o, "wb") as f:
        f.write(pickle.dumps(paths, protocol=0))
    print(o, len(paths))


def bad(path):
    try:
        return P.getsize(path) <= MINB
    except OSError:
        return True


def ok(path, m=MINB):
    try:
        return P.getsize(path) > m
    except OSError:
        return False


vids = sorted(
    P.abspath(P.join(VID, f))
    for f in os.listdir(VID)
    if f.lower().endswith((".mov", ".mp4", ".avi"))
)

if cmd == "remaining":
    face_rem, hand_rem, body_rem = [], [], []
    for vp in vids:
        stem = P.splitext(P.basename(vp))[0]
        pj = P.join(W, "pose", stem + "_pose.json")
        if not P.exists(pj):
            continue
        if bad(P.join(W, "face_crops", stem + "_face.mp4")):
            face_rem.append(vp)
        if bad(P.join(W, "hand_crops", stem + "_hand1.mp4")) or bad(
            P.join(W, "hand_crops", stem + "_hand2.mp4")
        ):
            hand_rem.append(vp)
        if not P.exists(P.join(W, "body_feats", stem + "_pose.npy")):
            body_rem.append(pj)
    dump(P.join(W, "face_remaining.list"), face_rem)
    dump(P.join(W, "hands_remaining.list"), hand_rem)
    dump(P.join(W, "pose.list"), body_rem)

elif cmd == "dino":
    need = []
    for fc in sorted(os.listdir(P.join(W, "face_crops"))):
        if not fc.endswith("_face.mp4"):
            continue
        stem = fc[: -len("_face.mp4")]
        if not (
            ok(P.join(W, "face_crops", fc))
            and ok(P.join(W, "hand_crops", stem + "_hand1.mp4"))
            and ok(P.join(W, "hand_crops", stem + "_hand2.mp4"))
        ):
            continue
        if (
            ok(P.join(W, "face_feats", stem + "_face.npy"), 100)
            and ok(P.join(W, "hand1_feats", stem + "_hand1.npy"), 100)
            and ok(P.join(W, "hand2_feats", stem + "_hand2.npy"), 100)
        ):
            continue
        need.append(stem)
    dump(P.join(W, "face.list"), [P.join(W, "face_crops", v + "_face.mp4") for v in need])
    dump(P.join(W, "hand1.list"), [P.join(W, "hand_crops", v + "_hand1.mp4") for v in need])
    dump(P.join(W, "hand2.list"), [P.join(W, "hand_crops", v + "_hand2.mp4") for v in need])
    print("need dino", len(need))
else:
    raise SystemExit(f"unknown cmd {cmd}")
