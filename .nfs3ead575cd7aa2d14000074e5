"""
download_asl_citizen.py  —  Download and prepare ASL-Citizen from Microsoft Research.

ASL-Citizen is a Microsoft Research dataset (NOT on HuggingFace).
  Paper:    https://arxiv.org/abs/2304.05934
  Project:  https://www.microsoft.com/en-us/research/project/asl-citizen/
  Download: https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip

ZIP contents (~50 GB unzipped):
    ASL_Citizen/
        videos/              — 83,399 MP4 files
        splits/
            train.csv        — 40,154 rows  (35 signers)
            val.csv          —  10,304 rows  (6 signers)
            test.csv         — 32,941 rows  (11 signers)
        README.txt

Each CSV row: [participant_id, video_filename, gloss]
(no header row)

This script:
  1. Downloads ASL_Citizen.zip via wget.
  2. Extracts it.
  3. Merges train/val/test CSVs into a single metadata.csv that train_islr.py expects:
       video_id, gloss, participant_id, split

USAGE:
    # Full pipeline (download + extract + build metadata.csv)
    python download_asl_citizen.py

    # Skip download if zip already exists locally
    python download_asl_citizen.py --skip_download

    # Verify download completeness
    python download_asl_citizen.py --verify
"""

import argparse
import csv
import os
import subprocess
import sys
import zipfile

# ─── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)          # parent of cslr/
ASL_DIR      = os.path.join(PROJECT_ROOT, "data", "asl_citizen")
ZIP_PATH     = os.path.join(ASL_DIR, "ASL_Citizen.zip")
EXTRACT_DIR  = ASL_DIR                               # extracts to ASL_DIR/ASL_Citizen/
META_OUT     = os.path.join(ASL_DIR, "metadata.csv")

DOWNLOAD_URL = (
    "https://download.microsoft.com/download/"
    "b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def run(cmd, check=True):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, check=check)
    return result.returncode


def verify(args):
    """Check metadata.csv and count present video files."""
    if not os.path.exists(META_OUT):
        print(f"[MISSING] metadata.csv not found at {META_OUT}")
        return

    import csv as _csv
    rows = []
    with open(META_OUT) as f:
        reader = _csv.DictReader(f)
        rows = list(reader)

    n = len(rows)
    glosses   = {r["gloss"]          for r in rows}
    signers   = {r["participant_id"] for r in rows}
    splits    = {}
    for r in rows:
        splits[r["split"]] = splits.get(r["split"], 0) + 1

    print(f"metadata.csv: {n} rows | {len(glosses)} glosses | {len(signers)} signers")
    for sp, cnt in sorted(splits.items()):
        print(f"  {sp}: {cnt}")

    # Count video files
    video_dir = os.path.join(ASL_DIR, "ASL_Citizen", "videos")
    if os.path.isdir(video_dir):
        n_vids = len([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
        print(f"Videos present: {n_vids} / {n}  ({100*n_vids/max(n,1):.1f}%)")
    else:
        print(f"Video dir not found: {video_dir}")

    # Count pose features
    feat_dir = os.path.join(ASL_DIR, "pose_features")
    if os.path.isdir(feat_dir):
        n_feats = len([f for f in os.listdir(feat_dir) if f.endswith(".pt")])
        print(f"Pose features:  {n_feats} / {n}  ({100*n_feats/max(n,1):.1f}%)")
    else:
        print("Pose features:  not extracted yet  (run extract_pose_islr.py)")

    print()
    if n_vids == n:
        print("✅ Download complete!")
    else:
        print(f"⚠️  {n - n_vids} videos missing — re-run without --skip_download to finish.")


# ─── Build unified metadata.csv ───────────────────────────────────────────────

def build_metadata_csv(asl_citizen_dir: str) -> str:
    """
    Merge train/val/test CSVs from ASL_Citizen/splits/ into a single metadata.csv.

    Input CSVs have NO header row; columns are:
        0: participant_id  (e.g. "P01")
        1: video_filename  (e.g. "P01_0001.mp4")
        2: gloss           (e.g. "HELLO")

    Output (metadata.csv):
        video_id        — filename stem without .mp4 (e.g. "P01_0001")
        gloss           — uppercased gloss string
        participant_id  — signer ID
        split           — train / val / test
    """
    splits_dir = os.path.join(asl_citizen_dir, "splits")
    if not os.path.isdir(splits_dir):
        raise RuntimeError(
            f"splits/ directory not found at {splits_dir}\n"
            "Make sure the ZIP extracted correctly."
        )

    rows_out = []
    for split_name in ("train", "val", "test"):
        csv_path = os.path.join(splits_dir, f"{split_name}.csv")
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found, skipping.")
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 3:
                    continue
                participant_id = row[0].strip()
                video_filename = row[1].strip()
                gloss          = row[2].strip().upper()
                # Strip .mp4 extension for video_id
                video_id = os.path.splitext(video_filename)[0]
                rows_out.append({
                    "video_id":       video_id,
                    "gloss":          gloss,
                    "participant_id": participant_id,
                    "split":          split_name,
                })

    if not rows_out:
        raise RuntimeError("No rows read from splits CSVs. Check the ZIP contents.")

    with open(META_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "gloss", "participant_id", "split"])
        writer.writeheader()
        writer.writerows(rows_out)

    by_split = {}
    for r in rows_out:
        by_split[r["split"]] = by_split.get(r["split"], 0) + 1

    print(f"Saved metadata.csv → {META_OUT}")
    print(f"  Total rows: {len(rows_out)}")
    print(f"  Glosses:    {len({r['gloss'] for r in rows_out})}")
    print(f"  Signers:    {len({r['participant_id'] for r in rows_out})}")
    for sp in ("train", "val", "test"):
        print(f"  {sp}: {by_split.get(sp, 0)} rows")

    return META_OUT


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Download ASL-Citizen from Microsoft Research and prepare metadata."
    )
    ap.add_argument("--skip_download", action="store_true",
                    help="Skip wget download (use if ZIP already exists at data/asl_citizen/ASL_Citizen.zip).")
    ap.add_argument("--skip_extract", action="store_true",
                    help="Skip unzip (use if already extracted).")
    ap.add_argument("--verify", action="store_true",
                    help="Check download/extraction completeness and exit.")
    args = ap.parse_args()

    if args.verify:
        verify(args)
        return

    os.makedirs(ASL_DIR, exist_ok=True)

    # ── Step 1: Download ──────────────────────────────────────────────────────
    if not args.skip_download:
        if os.path.exists(ZIP_PATH):
            print(f"ZIP already exists: {ZIP_PATH}")
            print("Use --skip_download to skip, or delete it to re-download.")
        else:
            print(f"Downloading ASL-Citizen (~50 GB) → {ZIP_PATH}")
            print("This will take a while on a slow connection.")
            print()
            # Try wget first (available on most Linux HPC), fall back to curl
            if run(f"which wget", check=False) == 0:
                run(f"wget -c -O '{ZIP_PATH}' '{DOWNLOAD_URL}'")
            elif run(f"which curl", check=False) == 0:
                run(f"curl -L -C - -o '{ZIP_PATH}' '{DOWNLOAD_URL}'")
            else:
                print("[ERROR] Neither wget nor curl found.")
                print(f"Manually download from:\n  {DOWNLOAD_URL}")
                print(f"Save to: {ZIP_PATH}")
                sys.exit(1)
    else:
        print(f"--skip_download: assuming {ZIP_PATH} exists.")

    # ── Step 2: Extract ───────────────────────────────────────────────────────
    asl_citizen_dir = os.path.join(EXTRACT_DIR, "ASL_Citizen")

    if not args.skip_extract:
        if os.path.isdir(asl_citizen_dir):
            print(f"Already extracted: {asl_citizen_dir}")
        else:
            if not os.path.exists(ZIP_PATH):
                print(f"[ERROR] ZIP not found at {ZIP_PATH}")
                sys.exit(1)
            print(f"Extracting {ZIP_PATH} → {EXTRACT_DIR}")
            print("(~50 GB — will take several minutes)")
            with zipfile.ZipFile(ZIP_PATH, "r") as zf:
                zf.extractall(EXTRACT_DIR)
            print("Extraction complete.")
    else:
        print(f"--skip_extract: assuming {asl_citizen_dir} exists.")

    # ── Step 3: Build metadata.csv ────────────────────────────────────────────
    if os.path.exists(META_OUT) and not args.skip_download:
        print(f"metadata.csv already exists: {META_OUT}")
    else:
        print("\nBuilding unified metadata.csv from splits/train|val|test.csv...")
        build_metadata_csv(asl_citizen_dir)

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n✅ Dataset ready.")
    print(f"  Videos:       {asl_citizen_dir}/videos/")
    print(f"  Metadata:     {META_OUT}")
    print()
    print("Next steps:")
    print("  1. python extract_pose_islr.py --video_dir "
          f"'{asl_citizen_dir}/videos' --workers 8")
    print("  2. python train_islr.py")


if __name__ == "__main__":
    main()
