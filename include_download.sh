#!/bin/bash
# Download INCLUDE (Indian Sign Language) from Zenodo — no registration.
# Run on the cluster login node.
set -euo pipefail
DEST=${1:-${SCRATCH}/include}
mkdir -p "$DEST/raw"
cd "$DEST/raw"

echo "Fetching Zenodo file list..."
# Zenodo file URLs look like: .../api/records/4010759/files/<name>/content
python - <<'PY'
import json, urllib.request
r = json.load(urllib.request.urlopen("https://zenodo.org/api/records/4010759"))
lines = []
for f in r["files"]:
    url = f["links"]["self"]
    # ensure /content suffix
    if not url.endswith("/content"):
        url = url.rstrip("/") + "/content"
    lines.append(f"{url}\t{f['key']}")
open("files.tsv", "w").write("\n".join(lines) + "\n")
print("wrote", len(lines), "entries")
PY

echo "Files to download:"
cut -f2 files.tsv
echo

while IFS=$'\t' read -r url name; do
  [ -z "$name" ] && continue
  if [ -f "$name" ]; then
    echo "skip existing: $name"
    continue
  fi
  echo "Downloading $name ..."
  # -L follow redirects; -C - resume; --retry for flaky links
  wget -c -L --retry-connrefused --tries=5 -O "$name" "$url" \
    || curl -L --retry 5 -C - -o "$name" "$url"
done < files.tsv

echo "Unzipping..."
for z in *.zip; do
  [ -f "$z" ] || continue
  echo "  $z"
  unzip -n -q "$z" -d "$DEST" || unzip -n "$z" -d "$DEST"
done

echo "Done. Tree:"
du -sh "$DEST"
find "$DEST" -maxdepth 3 -type d | head -40
echo
echo "INCLUDE videos are .MOV (not .mp4). Verify:"
echo "  find $DEST -iname '*.mov' | wc -l   # expect ~4287"
echo
echo "Next: ln -sfn $DEST ${PROJECT_ROOT}/data/include"
echo "Then: python include_make_metadata.py --root ${PROJECT_ROOT}/data/include"
