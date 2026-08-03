"""gallery.py — build/inspect prototype galleries."""
import argparse
import sign_core as sc

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--ckpt", required=True); b.add_argument("--vocab", required=True)
    b.add_argument("--out", default="base_gallery.pt")
    s = sub.add_parser("show"); s.add_argument("path")
    a = ap.parse_args()
    if a.cmd == "build":
        model, vocab = sc.load_model(a.ckpt, a.vocab, "cpu")
        emb = sc.Embedder(model, vocab, "cpu")
        p, l = emb.base_gallery()
        sc.save_gallery(a.out, p, l, {"source": "arcface_head"})
        print(f"wrote {a.out}: {len(l)} built-in signs, dim={p.shape[1]}")
    else:
        p, l = sc.load_gallery(a.path)
        print(f"{a.path}: {len(l)} signs, dim={p.shape[1]}"); print("first few:", l[:10])

if __name__ == "__main__":
    main()
