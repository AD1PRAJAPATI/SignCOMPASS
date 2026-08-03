"""
download_wlasl.py — fetch WLASL-2000 videos via yt-dlp and build metadata.csv
in the same format dataset_islr.py expects (video_id, gloss, participant_id, split).

Prereqs:
    pip install yt-dlp --break-system-packages   (and ffmpeg available)
    Get WLASL_v0.3.json from https://github.com/dxli94/WLASL (start_kit), place its path below.

Usage:
    python download_wlasl.py --json /path/WLASL_v0.3.json --out_root /project2/jessetho_1732/aditeya/data/wlasl
Heavy link rot is expected; dead URLs are logged to failures.txt and just skipped.
"""
import argparse, json, os, subprocess, csv

def have(cmd):
    from shutil import which; return which(cmd) is not None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N glosses")
    a=ap.parse_args()
    vid_dir=os.path.join(a.out_root,"videos"); os.makedirs(vid_dir,exist_ok=True)
    data=json.load(open(a.json))
    if a.limit: data=data[:a.limit]
    rows=[]; failures=[]
    for entry in data:
        gloss=entry["gloss"]
        for inst in entry["instances"]:
            vid=inst["video_id"]; url=inst.get("url",""); split=inst.get("split","train")
            signer=str(inst.get("signer_id", inst.get("signer","unk")))
            out=os.path.join(vid_dir, f"{vid}.mp4")
            if not os.path.exists(out) and url:
                try:
                    subprocess.run(["yt-dlp","-q","--no-warnings","-f","mp4",
                                    "-o",out,url], check=True, timeout=120)
                except Exception as e:
                    failures.append(f"{vid}\t{url}\t{e}"); continue
            if os.path.exists(out):
                # optional: trim to frame range if present
                fs,fe=inst.get("frame_start",1),inst.get("frame_end",-1)
                rows.append({"video_id":vid,"gloss":gloss.upper(),
                             "participant_id":signer,"split":split})
    meta=os.path.join(a.out_root,"metadata.csv")
    with open(meta,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["video_id","gloss","participant_id","split"]); w.writeheader()
        for r in rows: w.writerow(r)
    if failures:
        open(os.path.join(a.out_root,"failures.txt"),"w").write("\n".join(failures))
    print(f"Downloaded {len(rows)} clips | {len(failures)} failed -> {a.out_root}")
    print(f"metadata.csv written. Glosses: {len(set(r['gloss'] for r in rows))}")

if __name__=="__main__": main()
