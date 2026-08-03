source ~/semlex_paths.sh
cd "$ISLR"
PY=/project2/jessetho_1732/aditeya/envs/cslr/bin/python

# 1) test coverage + rebuild remaining-only dino lists
$PY - <<'PY'
import csv,os,os.path as P,gzip,pickle
S=os.environ["SEMLEX"]; W=os.environ["WORK"]
rows=list(csv.DictReader(open(P.join(S,"metadata.csv"))))
test=[r["video_id"] for r in rows if str(r["split"]).strip().lower()=="test"]
ex=lambda *a: P.exists(P.join(*a))
tot={k:0 for k in "mp4 posejson facecrop h1crop h2crop facefeat h1feat h2feat bodyfeat posefeat".split()}
need=[]
for v in test:
    ok={
      "mp4":ex(S,"videos_mp4",v+".mp4"),
      "posejson":ex(W,"pose",v+"_pose.json"),
      "facecrop":ex(W,"face_crops",v+"_face.mp4"),
      "h1crop":ex(W,"hand_crops",v+"_hand1.mp4"),
      "h2crop":ex(W,"hand_crops",v+"_hand2.mp4"),
      "facefeat":ex(W,"face_feats",v+"_face.npy"),
      "h1feat":ex(W,"hand1_feats",v+"_hand1.npy"),
      "h2feat":ex(W,"hand2_feats",v+"_hand2.npy"),
      "bodyfeat":ex(W,"body_feats",v+"_pose.npy"),
      "posefeat":ex(S,"pose_features",v+".pt"),
    }
    for k,b in ok.items(): tot[k]+=int(b)
    # usable for ensemble needs pose + 4 streams
    if ok["facecrop"] and ok["h1crop"] and ok["h2crop"] and not (ok["facefeat"] and ok["h1feat"] and ok["h2feat"]):
        need.append(v)
print("TEST coverage /",len(test))
for k,n in tot.items(): print(f"  {k:10s} {n}")
print("need dino (have crops, missing some feats):", len(need))
# sample why hand2 might be missing
miss_h2=[v for v in test if ex(W,"hand_crops",v+"_hand1.mp4") and not ex(W,"hand_crops",v+"_hand2.mp4")]
print("have hand1 crop but NOT hand2:", len(miss_h2), "eg", miss_h2[:3])
def dump(o,paths):
    with gzip.GzipFile(o,"wb") as f: f.write(pickle.dumps(paths,protocol=0))
    print(o, len(paths))
dump(P.join(W,"face.list"),  [P.join(W,"face_crops",v+"_face.mp4") for v in need])
dump(P.join(W,"hand1.list"), [P.join(W,"hand_crops",v+"_hand1.mp4") for v in need])
dump(P.join(W,"hand2.list"), [P.join(W,"hand_crops",v+"_hand2.mp4") for v in need])
# how many would be usable RIGHT NOW if we ran ensemble
usable=sum(1 for v in test if ex(S,"pose_features",v+".pt") and ex(W,"face_feats",v+"_face.npy")
           and ex(W,"hand1_feats",v+"_hand1.npy") and ex(W,"hand2_feats",v+"_hand2.npy")
           and ex(W,"body_feats",v+"_pose.npy"))
print("usable RIGHT NOW for ensemble test:", usable, "/", len(test))
PY

# 2) resubmit dino only (pose already done); use afterany so 1-2 preempts don't kill ens
JD=$(sbatch --parsable --requeue sl_sh_dino.sbatch)
echo "dino = $JD"
# wait for dino array to finish (any outcome), then run ensemble
JE=$(sbatch --parsable --dependency=afterany:$JD run_ensemble.sbatch)
echo "ENSEMBLE = $JE -> $ISLR/logs/ensemble_${JE}.out"
echo "NOTE: if usable RIGHT NOW is already >1000, you can instead just: sbatch run_ensemble.sbatch"
