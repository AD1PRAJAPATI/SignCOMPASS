"""enroll.py — teach a NEW sign from a few clips (no retraining)."""
import argparse, os, glob
import numpy as np, torch
import sign_core as sc

def collect_videos(path):
    if os.path.isdir(path):
        fs = []
        for e in ("*.mp4","*.mov","*.webm","*.avi","*.MP4","*.MOV"):
            fs += glob.glob(os.path.join(path, e))
        return sorted(fs)
    return [path]

def record_webcam(shots, max_frames=64):
    import cv2
    hol = sc.make_holistic(); cap = cv2.VideoCapture(0)
    clips, buf, rec = [], [], False
    print(f"Record {shots} clips. SPACE=start/stop each, Q=done.")
    while len(clips) < shots:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        res = hol.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if rec: buf.append(sc.result_to_pose_vec(res))
        msg = f"REC {len(buf)}" if rec else f"clip {len(clips)+1}/{shots}  SPACE=start"
        cv2.putText(frame, msg, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0,0,255) if rec else (0,200,0), 2)
        cv2.imshow("enroll", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        if k == ord(' '):
            rec = not rec
            if not rec and len(buf) >= 4:
                clips.append(np.stack(buf)[:max_frames]); print(f"  saved clip {len(clips)}")
            buf = []
    cap.release(); cv2.destroyAllWindows(); hol.close()
    return clips

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--vocab", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--videos"); ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--gallery", default="my_signs.pt")
    a = ap.parse_args()
    model, vocab = sc.load_model(a.ckpt, a.vocab, "cpu")
    emb = sc.Embedder(model, vocab, "cpu")
    if a.webcam:
        poses = record_webcam(a.shots)
    else:
        assert a.videos, "give --videos or --webcam"
        hol = sc.make_holistic()
        poses = [sc.video_to_pose(v, hol) for v in collect_videos(a.videos)]
        hol.close()
    if not poses:
        print("no clips captured"); return
    vecs = torch.stack([emb.embed(p) for p in poses])
    proto = torch.nn.functional.normalize(vecs.mean(0), dim=0, eps=1e-8)
    if os.path.exists(a.gallery):
        P, L = sc.load_gallery(a.gallery)
        P = torch.cat([P, proto[None]], 0); L = L + [a.label]
    else:
        P, L = proto[None], [a.label]
    sc.save_gallery(a.gallery, P, L, {"custom": True})
    consistency = float((vecs @ proto).mean())
    print(f"enrolled '{a.label}' from {len(poses)} clips -> {a.gallery} "
          f"({len(L)} custom signs). clip-consistency={consistency:.2f}")

if __name__ == "__main__":
    main()
