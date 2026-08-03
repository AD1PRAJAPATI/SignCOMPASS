"""infer_server.py — serve sign predictions + live enrollment (run on SLURM)."""
import argparse, json, os, socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np, torch
import sign_core as sc


class State:
    def __init__(self, emb, protos, labels, n_base, save_path):
        self.emb = emb; self.labels = labels
        self.n_base = n_base; self.save_path = save_path
        self.protos = protos                       # [N, D] on device

    def predict(self, pose, k):
        q = self.emb.embed(pose).to(self.protos.device)
        return sc.topk(q, self.protos, self.labels, k)

    def enroll(self, label, clips):
        vecs = []
        for c in clips:
            p = np.asarray(c, np.float32)
            if p.ndim == 2 and p.shape[1] == sc.D_POSE:
                vecs.append(self.emb.embed(p))
        if not vecs:
            return None, 0.0
        stack = torch.stack(vecs)
        proto = torch.nn.functional.normalize(stack.mean(0), dim=0)
        consistency = float((stack @ proto).mean())
        proto = proto.to(self.protos.device)
        self.protos = torch.cat([self.protos, proto[None]], 0)
        self.labels = self.labels + [label]
        self.save()
        return len(self.labels) - self.n_base, consistency

    def save(self):
        if self.save_path and len(self.labels) > self.n_base:
            sc.save_gallery(self.save_path, self.protos[self.n_base:].cpu(),
                            self.labels[self.n_base:], {"custom": True})


def build(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab = sc.load_model(args.ckpt, args.vocab, dev)
    emb = sc.Embedder(model, vocab, dev)
    bp, bl = emb.base_gallery()
    here = os.path.dirname(os.path.abspath(__file__))
    default_gallery = os.path.join(here, "custom_gallery.pt")
    load_from = args.gallery or (default_gallery if os.path.exists(default_gallery) else None)
    protos, labels = sc.merge_galleries(bp, bl, load_from)
    save_path = args.gallery or default_gallery
    st = State(emb, protos.to(dev), labels, len(bl), save_path)
    print(f"[server] {len(labels)} signs ({len(bl)} built-in + "
          f"{len(labels)-len(bl)} custom) | dim={emb.embed_dim} | device={dev}")
    print(f"[server] custom signs persist to {save_path}")
    return st


def make_handler(st):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        def log_message(self, *a): pass
        def _read(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")
        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "n": len(st.labels),
                                 "custom": len(st.labels) - st.n_base})
            else:
                self._send(404, {"error": "not found"})
        def do_POST(self):
            if self.path == "/predict":
                req = self._read()
                pose = np.asarray(req.get("pose", []), np.float32)
                if pose.ndim != 2 or pose.shape[1] != sc.D_POSE:
                    return self._send(400, {"error": f"pose must be (T,{sc.D_POSE})"})
                self._send(200, {"preds": st.predict(pose, int(req.get("k", 5)))})
            elif self.path == "/enroll":
                req = self._read()
                label = req.get("label"); clips = req.get("clips", [])
                if not label or not clips:
                    return self._send(400, {"error": "need label + clips"})
                cnt, cons = st.enroll(label, clips)
                if cnt is None:
                    return self._send(400, {"error": "no valid clips"})
                self._send(200, {"ok": True, "label": label,
                                 "custom_signs": cnt, "consistency": round(cons, 3)})
            else:
                self._send(404, {"error": "not found"})
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--gallery", default=None)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    st = build(args)
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(st))
    host = socket.gethostname()
    print(f"[server] listening on {host}:{args.port}")
    print(f"[server] tunnel: ssh -N -L {args.port}:{host}:{args.port} "
          f"{os.environ.get('USER','you')}@<login-node>")
    srv.serve_forever()


if __name__ == "__main__":
    main()
