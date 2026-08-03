"""
shubert_make_lists.py  —  glue for running SHuBERT's official pipeline on ASL-Citizen.

SHuBERT's stage scripts read gzipped-pickle .list files (see write_list.py) and the
final inference reads a CSV of tab-separated stream-feature paths. This builds them.

Subcommands:
  list  <input_dir> <out.list> [--ext .mp4]
        gzip-pickle a list of absolute file paths in input_dir (for a stage's --files_list)

  csv   <face_dir> <lh_dir> <rh_dir> <body_dir> <out.csv>
        build the shubert_inference CSV. Matches the 4 stream .npy files by VIDEO ID
        (filename with extension + trailing _suffix stripped), one row per video:
            face.npy \\t left_hand.npy \\t right_hand.npy \\t body.npy
        Only videos present in ALL four dirs are written.
"""
import argparse, csv, glob, gzip, os, pickle, re


def _vid_id(path):
    """Strip dir, extension, and a trailing _face/_left/_right/_body style suffix."""
    base = os.path.basename(path)
    base = base.rsplit(".", 1)[0]
    return re.sub(r"_(face|hand1|hand2|pose|left|right|left_hand|right_hand|body|body_posture|hand)$", "", base)


def cmd_list(a):
    pat = a.glob if a.glob else ("*" + a.ext)
    files = sorted(os.path.abspath(p) for p in glob.glob(os.path.join(a.input_dir, pat)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with gzip.GzipFile(a.out, "wb") as f:
        f.write(pickle.dumps(files, protocol=0))
    print(f"Wrote {len(files)} paths -> {a.out}")


def _index(d):
    m = {}
    for p in glob.glob(os.path.join(d, "*.npy")):
        m[_vid_id(p)] = os.path.abspath(p)
    return m


def cmd_csv(a):
    face, lh, rh, body = _index(a.face_dir), _index(a.lh_dir), _index(a.rh_dir), _index(a.body_dir)
    ids = sorted(set(face) & set(lh) & set(rh) & set(body))
    miss = (set(face) | set(lh) | set(rh) | set(body)) - set(ids)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        for vid in ids:
            w.writerow([f"{face[vid]}\t{lh[vid]}\t{rh[vid]}\t{body[vid]}"])
    print(f"Wrote {len(ids)} rows -> {a.out}  ({len(miss)} videos missing a stream, skipped)")



def _good(path, min_bytes=1024):
    """True if path exists and is larger than empty/stub crop size."""
    try:
        return os.path.getsize(path) > min_bytes
    except OSError:
        return False


def cmd_remaining(a):
    import glob as _g
    vids = sorted(_g.glob(os.path.join(a.video_dir, "*" + a.ext)))
    needs = [n.split(":", 1) for n in a.need]          # ["dir:suffix", ...]
    out = []
    for v in vids:
        stem = os.path.splitext(os.path.basename(v))[0]
        # treat tiny 257B stub mp4s as missing (crop_face skips if path exists)
        done = all(_good(os.path.join(d, stem + suf)) for d, suf in needs)
        if not done:
            out.append(os.path.abspath(v))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with gzip.GzipFile(a.out, "wb") as f:
        f.write(pickle.dumps(out, protocol=0))
    print(f"remaining: {len(out)} / {len(vids)} videos still need work -> {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("list"); pl.add_argument("input_dir"); pl.add_argument("out")
    pl.add_argument("--ext", default=".mp4")
    pl.add_argument("--glob", default=None, help="glob pattern, e.g. '*_hand1.mp4' (overrides --ext)")
    pl.set_defaults(func=cmd_list)
    pc = sub.add_parser("csv")
    for x in ("face_dir", "lh_dir", "rh_dir", "body_dir", "out"):
        pc.add_argument(x)
    pc.set_defaults(func=cmd_csv)
    pr = sub.add_parser("remaining")
    pr.add_argument("video_dir"); pr.add_argument("out")
    pr.add_argument("--ext", default=".mp4")
    pr.add_argument("--need", action="append", required=True,
                    help="dir:suffix that must exist for a video to count as done; repeatable")
    pr.set_defaults(func=cmd_remaining)
    args = ap.parse_args(); args.func(args)
