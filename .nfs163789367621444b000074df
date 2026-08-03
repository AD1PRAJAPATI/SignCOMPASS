"""
asl_lex_labels.py  —  Build phonological auxiliary supervision labels from ASL-LEX 2.0.

WHY: Auxiliary prediction of phonological features (handshape, location, movement)
gave +8.7 top-1 pts on WLASL-2000. ASL-Citizen's 2,731 signs map directly to
ASL-LEX sign codes, so this supervision is essentially free.

WHAT THIS SCRIPT DOES:
  1. Downloads ASL-LEX 2.0 CSV from the public OSF repository.
  2. Parses handshape, major location, and movement columns.
  3. Builds integer label encodings for each phonological feature.
  4. Saves a phonological_labels.json that train_fusion.py loads at runtime.
     Format: { "SIGN_CODE": {"handshape": int, "location": int, "movement": int}, ... }
  5. Also saves phonological_vocab.json with the class counts per feature.

USAGE:
    python asl_lex_labels.py                    # downloads + builds
    python asl_lex_labels.py --asl_lex_csv /path/to/asl_lex.csv  # use local file
    python asl_lex_labels.py --verify           # print coverage stats

ASL-LEX 2.0 columns used:
    SignCode (or Code)  — matches ASL-Citizen gloss identifier
    Handshape           — dominant hand configuration (e.g. "B", "5", "A")
    MajorLocation       — coarse signing location (e.g. "Head", "Chest", "Neutral")
    Movement            — primary movement type (e.g. "Straight", "Arc", "Circular")

Coverage: ASL-Citizen uses ASL-LEX sign codes as glosses, so coverage is ~95%+.
The remaining ~5% (variants / fingerspelling) map to an "UNKNOWN" class.
"""

import argparse
import json
import os
import sys
import urllib.request

# ─── ASL-LEX 2.0 download URL ─────────────────────────────────────────────────
# Public OSF repository: https://osf.io/2epcb/
ASL_LEX_URL = (
    "https://osf.io/download/t26yr/"
    # Direct download for ASL-LEX 2.0 CSV (SignData.csv)
    # If this link breaks, download manually from https://asl-lex.org/
)
# Fallback: raw GitHub mirror sometimes available
ASL_LEX_FALLBACK = (
    "https://raw.githubusercontent.com/ASL-LEX/asl-lex/main/data/SignData.csv"
)

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ASL_DIR      = os.path.join(PROJECT_ROOT, "data", "asl_citizen")
ASL_LEX_CSV  = os.path.join(ASL_DIR, "asl_lex_signdata.csv")
LABELS_OUT   = os.path.join(ASL_DIR, "phonological_labels.json")
VOCAB_OUT    = os.path.join(ASL_DIR, "phonological_vocab.json")

# Column aliases: ASL-LEX 2.0 uses slightly different names depending on version
COLMAP = {
    "code":     ["SignCode", "Code", "sign_code", "code", "LexicalEntry"],
    "hs":       ["Handshape", "DomHandshape", "dom_handshape", "handshape"],
    "loc":      ["MajorLocation", "major_location", "Location", "SignLocation"],
    "movement": ["Movement", "MovementType", "movement", "PrimaryMovement"],
}

UNKNOWN = "UNKNOWN"


# ─── Download ─────────────────────────────────────────────────────────────────

def download_asl_lex(out_path: str) -> bool:
    """Try to download ASL-LEX CSV. Returns True if successful."""
    for url in [ASL_LEX_URL, ASL_LEX_FALLBACK]:
        try:
            print(f"Downloading ASL-LEX 2.0 from:\n  {url}")
            urllib.request.urlretrieve(url, out_path)
            size = os.path.getsize(out_path)
            if size > 1000:
                print(f"Downloaded: {out_path} ({size//1024} KB)")
                return True
            else:
                os.remove(out_path)
        except Exception as e:
            print(f"  Failed ({e}), trying fallback...")
    return False


# ─── Parse ASL-LEX CSV ────────────────────────────────────────────────────────

def _find_col(header: list, aliases: list) -> str:
    """Return the first alias that appears in header, or None."""
    h_lower = [c.lower().strip() for c in header]
    for alias in aliases:
        if alias.lower() in h_lower:
            return header[h_lower.index(alias.lower())]
    return None


def parse_asl_lex(csv_path: str) -> dict:
    """
    Parse ASL-LEX CSV and return a dict:
        { sign_code (str): {"handshape": str, "location": str, "movement": str} }
    """
    import csv
    rows = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        if not header:
            raise ValueError("Empty or headerless CSV")

        col_code = _find_col(header, COLMAP["code"])
        col_hs   = _find_col(header, COLMAP["hs"])
        col_loc  = _find_col(header, COLMAP["loc"])
        col_mov  = _find_col(header, COLMAP["movement"])

        missing = [n for n, c in [("code", col_code), ("handshape", col_hs),
                                   ("location", col_loc), ("movement", col_mov)] if c is None]
        if col_code is None:
            raise ValueError(
                f"Cannot find sign code column. Available: {header}\n"
                "Edit COLMAP in asl_lex_labels.py to match your CSV."
            )
        if missing:
            print(f"WARNING: columns not found: {missing}. Those features will be UNKNOWN.")

        for row in reader:
            code = row[col_code].strip().upper()
            if not code:
                continue
            rows[code] = {
                "handshape": row[col_hs].strip()   if col_hs  and row.get(col_hs)  else UNKNOWN,
                "location":  row[col_loc].strip()  if col_loc and row.get(col_loc) else UNKNOWN,
                "movement":  row[col_mov].strip()  if col_mov and row.get(col_mov) else UNKNOWN,
            }

    print(f"Parsed {len(rows)} entries from ASL-LEX CSV.")
    return rows


# ─── Build integer encodings ──────────────────────────────────────────────────

def build_encodings(asl_lex: dict, asl_citizen_glosses: list) -> tuple:
    """
    Build integer label maps for each phonological feature.

    Returns:
        labels_out  : { gloss: {"handshape": int, "location": int, "movement": int} }
        vocab_out   : { feature_name: {"classes": [...], "n_classes": int} }
    """
    # Collect all values that appear in ASL-Citizen glosses
    hs_vals, loc_vals, mov_vals = {UNKNOWN}, {UNKNOWN}, {UNKNOWN}
    for gloss in asl_citizen_glosses:
        entry = asl_lex.get(gloss, {})
        hs_vals.add(entry.get("handshape", UNKNOWN))
        loc_vals.add(entry.get("location",  UNKNOWN))
        mov_vals.add(entry.get("movement",  UNKNOWN))

    # Sorted lists → integer IDs (UNKNOWN = 0 by convention)
    def make_vocab(vals):
        v = sorted(vals - {UNKNOWN})
        return [UNKNOWN] + v

    hs_vocab  = make_vocab(hs_vals)
    loc_vocab = make_vocab(loc_vals)
    mov_vocab = make_vocab(mov_vals)

    hs_id  = {v: i for i, v in enumerate(hs_vocab)}
    loc_id = {v: i for i, v in enumerate(loc_vocab)}
    mov_id = {v: i for i, v in enumerate(mov_vocab)}

    labels_out = {}
    covered = 0
    for gloss in asl_citizen_glosses:
        entry = asl_lex.get(gloss, {})
        hs  = entry.get("handshape", UNKNOWN)
        loc = entry.get("location",  UNKNOWN)
        mov = entry.get("movement",  UNKNOWN)
        if gloss in asl_lex:
            covered += 1
        labels_out[gloss] = {
            "handshape": hs_id.get(hs,  0),
            "location":  loc_id.get(loc, 0),
            "movement":  mov_id.get(mov, 0),
        }

    vocab_out = {
        "handshape": {"classes": hs_vocab,  "n_classes": len(hs_vocab)},
        "location":  {"classes": loc_vocab, "n_classes": len(loc_vocab)},
        "movement":  {"classes": mov_vocab, "n_classes": len(mov_vocab)},
    }

    print(f"Coverage: {covered}/{len(asl_citizen_glosses)} ASL-Citizen glosses "
          f"({100*covered/max(len(asl_citizen_glosses),1):.1f}%) found in ASL-LEX.")
    print(f"Feature vocabs: handshape={len(hs_vocab)} | "
          f"location={len(loc_vocab)} | movement={len(mov_vocab)}")

    return labels_out, vocab_out


# ─── Verify ───────────────────────────────────────────────────────────────────

def verify():
    if not os.path.exists(LABELS_OUT):
        print(f"Labels not built yet: {LABELS_OUT}")
        return
    with open(LABELS_OUT) as f:
        labels = json.load(f)
    with open(VOCAB_OUT) as f:
        vocab = json.load(f)
    print(f"phonological_labels.json: {len(labels)} entries")
    for feat, info in vocab.items():
        n_unknown = sum(1 for v in labels.values() if v[feat] == 0)
        print(f"  {feat}: {info['n_classes']} classes | "
              f"UNKNOWN: {n_unknown}/{len(labels)} ({100*n_unknown/len(labels):.1f}%)")
    # Sample
    sample = list(labels.items())[:3]
    for gloss, feats in sample:
        print(f"  {gloss}: {feats}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Build ASL-LEX 2.0 phonological labels for ASL-Citizen training."
    )
    ap.add_argument("--asl_lex_csv", default=None,
                    help="Path to local ASL-LEX CSV (skip download if provided).")
    ap.add_argument("--metadata_csv", default=None,
                    help="Path to ASL-Citizen metadata.csv (to get gloss list).")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory (default: data/asl_citizen/).")
    ap.add_argument("--verify", action="store_true",
                    help="Print coverage stats for already-built labels.")
    args = ap.parse_args()

    if args.verify:
        verify()
        return

    out_dir = args.out_dir or ASL_DIR
    os.makedirs(out_dir, exist_ok=True)
    labels_path = os.path.join(out_dir, "phonological_labels.json")
    vocab_path  = os.path.join(out_dir, "phonological_vocab.json")

    # ── Step 1: Get ASL-LEX CSV ────────────────────────────────────────────────
    csv_path = args.asl_lex_csv or ASL_LEX_CSV
    if not os.path.exists(csv_path):
        print("ASL-LEX CSV not found locally. Attempting download...")
        ok = download_asl_lex(csv_path)
        if not ok:
            print(
                "\n[ERROR] Auto-download failed. Please download manually:\n"
                "  1. Go to: https://asl-lex.org/  →  Download  →  ASL-LEX 2.0\n"
                "  2. Save SignData.csv to:\n"
                f"       {csv_path}\n"
                "  3. Re-run: python asl_lex_labels.py\n"
            )
            sys.exit(1)

    # ── Step 2: Get ASL-Citizen gloss list ─────────────────────────────────────
    meta_csv = args.metadata_csv or os.path.join(out_dir, "metadata.csv")
    if os.path.exists(meta_csv):
        import csv as _csv
        with open(meta_csv) as f:
            reader = _csv.DictReader(f)
            all_glosses = sorted({row["gloss"].strip().upper() for row in reader})
        print(f"Loaded {len(all_glosses)} unique glosses from metadata.csv")
    else:
        print(f"WARNING: metadata.csv not found at {meta_csv}.")
        print("Building labels for all ASL-LEX entries instead.")
        all_glosses = None   # Will use all entries from ASL-LEX

    # ── Step 3: Parse ASL-LEX ─────────────────────────────────────────────────
    asl_lex = parse_asl_lex(csv_path)

    if all_glosses is None:
        all_glosses = sorted(asl_lex.keys())

    # ── Step 4: Build encodings ────────────────────────────────────────────────
    labels, vocab = build_encodings(asl_lex, all_glosses)

    # ── Step 5: Save ──────────────────────────────────────────────────────────
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)

    print(f"\nSaved:")
    print(f"  {labels_path}  ({os.path.getsize(labels_path)//1024} KB)")
    print(f"  {vocab_path}")
    print()
    print("Next: pass --phon_labels_path to train_fusion.py to enable aux supervision.")


# ─── Runtime loader (used by train_fusion.py) ─────────────────────────────────

def load_phon_labels(labels_path: str, vocab_path: str):
    """
    Load pre-built phonological labels and return:
        labels : dict  { gloss_str: {"handshape": int, "location": int, "movement": int} }
        vocab  : dict  { feature: {"classes": [...], "n_classes": int} }
    """
    with open(labels_path) as f:
        labels = json.load(f)
    with open(vocab_path) as f:
        vocab = json.load(f)
    return labels, vocab


if __name__ == "__main__":
    main()
