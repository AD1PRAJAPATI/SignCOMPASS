"""webcam_client.py — laptop client: live sentence mode + custom-sign enrollment."""
import argparse, json, urllib.request
import numpy as np, cv2
import sign_core as sc

HAND0, HAND1 = 132, 258
LP_I, RP_I = 259, 260

def _post(url, obj, timeout=60):
    body = json.dumps(obj).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def predict_server(url, pose, k=5):
    return _post(url.rstrip("/") + "/predict", {"pose": pose.tolist(), "k": k})["preds"]

def enroll_flow(server, label, shots, hol, cap, min_frames=6, max_frames=64):
    clips, buf, rec = [], [], False
    print(f"Enroll '{label}': record {shots} clips. SPACE=start/stop each, Q=abort.")
    while len(clips) < shots:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        res = hol.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if rec: buf.append(sc.result_to_pose_vec(res))
        msg = f"REC {len(buf)}" if rec else f"clip {len(clips)+1}/{shots}  SPACE=start"
        cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0,0,255) if rec else (0,200,0), 2)
        cv2.putText(frame, f"enrolling: {label}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.imshow("enroll", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        if k == ord(' '):
            rec = not rec
            if not rec and len(buf) >= min_frames:
                clips.append(np.stack(buf)[:max_frames].tolist()); print(f"  clip {len(clips)} saved")
            buf = []
    if not clips:
        print("no clips recorded"); return
    print("ENROLLED:", _post(server.rstrip("/") + "/enroll", {"label": label, "clips": clips}))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--ckpt"); ap.add_argument("--vocab")
    ap.add_argument("--gallery", default=None)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=64)
    ap.add_argument("--manual", action="store_true")
    ap.add_argument("--enroll", default=None, help="enroll a new sign under this label")
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--start_motion", type=float, default=0.12)
    ap.add_argument("--stop_motion", type=float, default=0.04)
    ap.add_argument("--pause_frames", type=int, default=8)
    ap.add_argument("--min_frames", type=int, default=6)
    args = ap.parse_args()

    emb = protos = labels = None
    if args.local:
        model, vocab = sc.load_model(args.ckpt, args.vocab, "cpu")
        emb = sc.Embedder(model, vocab, "cpu")
        bp, bl = emb.base_gallery()
        protos, labels = sc.merge_galleries(bp, bl, args.gallery)
        print(f"[local] {len(labels)} signs")

    hol = sc.make_holistic(); cap = cv2.VideoCapture(args.cam)

    if args.enroll:
        enroll_flow(args.server, args.enroll, args.shots, hol, cap)
        cap.release(); cv2.destroyAllWindows(); hol.close(); return

    def classify(buf):
        pose = np.stack(buf)[:args.max_frames]
        if args.local: return sc.topk(emb.embed(pose), protos, labels, 5)
        return predict_server(args.server, pose, 5)

    auto = not args.manual
    buffer, prev_hand, prev_present = [], None, False
    still, recording, sentence, last = 0, False, [], []

    def finalize():
        nonlocal buffer, recording, still, sentence, last
        if len(buffer) >= args.min_frames:
            try:
                last = classify(buffer); sentence.append(last[0][0])
                print("word:", last[0], "| sentence:", " ".join(sentence))
            except Exception as e:
                print("predict error:", e)
        buffer, recording, still = [], False, 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        res = hol.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        vec = sc.result_to_pose_vec(res)
        present = vec[LP_I] > 0 or vec[RP_I] > 0
        hand = vec[HAND0:HAND1]
        motion = float(np.linalg.norm(hand - prev_hand)) if (present and prev_present and prev_hand is not None) else 0.0
        prev_hand, prev_present = hand, present

        if auto:
            if not recording and motion > args.start_motion:
                recording, buffer, still = True, [vec], 0
            elif recording:
                buffer.append(vec)
                still = still + 1 if motion < args.stop_motion else 0
                if still >= args.pause_frames or len(buffer) >= args.max_frames:
                    finalize()
        else:
            if recording: buffer.append(vec)

        state = f"REC {len(buffer)}" if recording else ("AUTO" if auto else "MANUAL")
        col = (0,0,255) if recording else (0,200,0)
        cv2.putText(frame, f"{state}  motion={motion:.2f}", (20,35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        cv2.putText(frame, "C clear  Z undo  M mode  SPACE cut(manual)  Q quit", (20,63),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        y = 105
        for j,(g,p) in enumerate(last):
            c = (0,255,0) if j==0 else (200,200,200)
            cv2.putText(frame, f"{j+1}. {g}  {p:.2f}", (20,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2); y += 30
        strip = " ".join(sentence) if sentence else "(sentence...)"
        cv2.rectangle(frame, (0, frame.shape[0]-46), (frame.shape[1], frame.shape[0]), (0,0,0), -1)
        cv2.putText(frame, strip, (15, frame.shape[0]-16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("Sign recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('c'): sentence, last = [], []
        elif key == ord('z'): sentence = sentence[:-1]
        elif key == ord('m'): auto = not auto; buffer, recording, still = [], False, 0
        elif key == ord(' ') and not auto:
            recording = not recording
            if not recording: finalize()
            else: buffer = []
    cap.release(); cv2.destroyAllWindows(); hol.close()

if __name__ == "__main__":
    main()
