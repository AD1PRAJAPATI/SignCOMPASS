"""recognize.py — recognize a sign against built-in + custom signs."""
import argparse
import numpy as np
import sign_core as sc

def _fmt(preds, tau):
    if not preds or preds[0][1] < tau:
        top = preds[0] if preds else ("<none>", 0.0)
        return f"unknown (best guess {top[0]} @ {top[1]:.2f} < tau={tau})"
    return "   ".join(f"{i+1}.{g} ({s:.2f})" for i,(g,s) in enumerate(preds))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--vocab", required=True)
    ap.add_argument("--gallery", default=None)
    ap.add_argument("--video"); ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.0)
    a = ap.parse_args()
    model, vocab = sc.load_model(a.ckpt, a.vocab, "cpu")
    emb = sc.Embedder(model, vocab, "cpu")
    bp, bl = emb.base_gallery()
    protos, labels = sc.merge_galleries(bp, bl, a.gallery)
    print(f"gallery: {len(labels)} signs ({len(bl)} built-in + {len(labels)-len(bl)} custom)")
    if a.webcam:
        import cv2
        hol = sc.make_holistic(); cap = cv2.VideoCapture(0); buf=[]; rec=False
        print("SPACE=start/stop, Q=quit")
        while True:
            ok, fr = cap.read()
            if not ok: break
            fr = cv2.flip(fr,1); res = hol.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            if rec: buf.append(sc.result_to_pose_vec(res))
            cv2.putText(fr, "REC" if rec else "SPACE=sign Q=quit",(20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255) if rec else (0,200,0),2)
            cv2.imshow("recognize", fr); k = cv2.waitKey(1)&0xFF
            if k==ord('q'): break
            if k==ord(' '):
                rec = not rec
                if not rec and len(buf)>=4:
                    print(_fmt(sc.topk(emb.embed(np.stack(buf)[:64]), protos, labels, a.k), a.tau))
                buf=[]
        cap.release(); cv2.destroyAllWindows(); hol.close()
    else:
        assert a.video, "give --video or --webcam"
        pose = sc.video_to_pose(a.video)
        print(_fmt(sc.topk(emb.embed(pose), protos, labels, a.k), a.tau))

if __name__ == "__main__":
    main()
