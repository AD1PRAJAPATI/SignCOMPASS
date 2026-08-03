"""
Build ISLR metadata.csv for INCLUDE (Indian Sign Language).

Videos are Canon .MOV, e.g. Colours/50. Yellow/MVI_5194.MOV

Preferred: HuggingFace ai4bharat/INCLUDE splits (via parquet HTTP, no `datasets` pkg).
Fallback: walk disk and carve train/val/test per gloss.

Usage:
  python include_make_metadata.py --root /project2/.../data/include
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import urllib.request
from collections import Counter
from pathlib import Path

VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv", ".webm"}

HF_PARQUET = {
    "train": "https://huggingface.co/datasets/ai4bharat/INCLUDE/resolve/main/data/train-00000-of-00001.parquet",
    "val": "https://huggingface.co/datasets/ai4bharat/INCLUDE/resolve/main/data/val-00000-of-00001.parquet",
    "test": "https://huggingface.co/datasets/ai4bharat/INCLUDE/resolve/main/data/test-00000-of-00001.parquet",
}


def _stem_id(rel: Path) -> str:
    parts = list(rel.parts[-3:]) if len(rel.parts) >= 3 else list(rel.parts)
    s = "__".join(parts)
    return re.sub(r"[^\w.\-]+", "_", re.sub(r"\.(mov|mp4|avi|mkv|webm)$", "", s, flags=re.I))


def _gloss_from_folder(name: str) -> str:
    m = re.match(r"^\d+\.\s*(.+)$", name.strip())
    return (m.group(1) if m else name).strip()


def _resolve(root: Path, rel: str) -> Path | None:
    rel = rel.replace("\\", "/").lstrip("./")
    for c in (root / rel, root / Path(rel).name):
        if c.is_file():
            return c.resolve()
    parent = (root / rel).parent
    if parent.is_dir():
        want = Path(rel).name.lower()
        for p in parent.iterdir():
            if p.is_file() and p.name.lower() == want:
                return p.resolve()
    return None


def _walk_videos(root: Path) -> list[Path]:
    out = []
    seen = set()
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        if "raw" in p.parts:
            continue
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p.resolve())
    return sorted(out)


def _hf_row(root: Path, split: str, rel: str, label: str | None) -> dict | None:
    vp = _resolve(root, rel)
    if vp is None:
        return None
    try:
        vid = _stem_id(vp.relative_to(root.resolve()))
    except ValueError:
        # video may live under a nested/extracted zip path; fall back to last 3 parts
        vid = _stem_id(Path(rel))
    gloss = _gloss_from_folder(str(label or vp.parent.name))
    return {
        "video_id": vid,
        "gloss": gloss,
        "participant_id": "unk",
        "split": split,
        "video_path": str(vp),
    }


def _from_hf_api(root: Path):
    """Official splits via HuggingFace datasets-server (no pandas/pyarrow)."""
    import json

    rows, miss = [], 0
    for split in ("train", "val", "test"):
        offset = 0
        print(f"[hf-api] fetching {split} ...")
        while True:
            url = (
                "https://datasets-server.huggingface.co/rows"
                f"?dataset=ai4bharat%2FINCLUDE&config=default&split={split}"
                f"&offset={offset}&length=100"
            )
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    payload = json.loads(r.read().decode())
            except Exception as e:
                print(f"[info] HF API {split}@{offset} failed: {e}")
                return None
            batch = payload.get("rows") or []
            if not batch:
                break
            for item in batch:
                row = item.get("row") or {}
                rel = str(row.get("video_path") or "")
                if not rel:
                    miss += 1
                    continue
                out = _hf_row(root, split, rel, row.get("label"))
                if out is None:
                    miss += 1
                else:
                    rows.append(out)
            offset += len(batch)
            if len(batch) < 100:
                break
            print(f"  {split}: {offset} rows ...")
    return _dedupe_hf(rows, miss)


def _dedupe_hf(rows: list, miss: int):
    uniq, seen = [], set()
    for r in rows:
        if r["video_path"] in seen:
            continue
        seen.add(r["video_path"])
        uniq.append(r)
    print(f"[hf] kept {len(uniq)} on-disk videos; missing on disk: {miss}")
    return uniq or None


def _from_hf_parquet(root: Path):
    """Read official splits from HF parquet (needs pandas+pyarrow OR fastparquet)."""
    try:
        import pandas as pd
    except ImportError:
        print("[info] pandas not available — try HF API")
        return _from_hf_api(root)
    rows = []
    miss = 0
    for split, url in HF_PARQUET.items():
        try:
            print(f"[hf] fetching {split} ...")
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            df = pd.read_parquet(io.BytesIO(data))
        except Exception as e:
            print(f"[info] HF parquet {split} failed: {e} — try HF API")
            return _from_hf_api(root)
        for _, r in df.iterrows():
            out = _hf_row(root, split, str(r["video_path"]), r.get("label"))
            if out is None:
                miss += 1
            else:
                rows.append(out)
    return _dedupe_hf(rows, miss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-hf", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    disk = _walk_videos(root)
    print(f"[scan] unique videos on disk: {len(disk)} under {root}")

    out_rows = None if args.no_hf else _from_hf_parquet(root)

    if not out_rows:
        print("[warn] carving local train/val/test from disk walk")
        out_rows = []
        for vp in disk:
            rel = vp.relative_to(root)
            out_rows.append(
                {
                    "video_id": _stem_id(rel),
                    "gloss": _gloss_from_folder(vp.parent.name),
                    "participant_id": "unk",
                    "split": "train",
                    "video_path": str(vp),
                }
            )
        by_gloss: dict[str, list] = {}
        for r in out_rows:
            by_gloss.setdefault(r["gloss"], []).append(r)
        for items in by_gloss.values():
            for i, r in enumerate(sorted(items, key=lambda x: x["video_id"])):
                if i % 10 == 8:
                    r["split"] = "val"
                elif i % 10 == 9:
                    r["split"] = "test"

    # final safety: only existing files, unique paths
    cleaned, seen = [], set()
    for r in out_rows:
        p = Path(r["video_path"])
        if not p.is_file():
            continue
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        r["video_path"] = key
        cleaned.append(r)
    out_rows = cleaned

    out = Path(args.out) if args.out else root / "metadata.csv"
    fields = ["video_id", "gloss", "participant_id", "split", "video_path"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print("wrote", out, "rows", len(out_rows), "classes", len({r["gloss"] for r in out_rows}))
    print("splits", dict(Counter(r["split"] for r in out_rows)))
    if len(out_rows) == 0:
        raise SystemExit("No videos found.")
    if len(out_rows) != len(disk) and not args.no_hf:
        print(f"[note] HF-matched {len(out_rows)} vs disk {len(disk)} (ok if some zips incomplete)")


if __name__ == "__main__":
    main()
