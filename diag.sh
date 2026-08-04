source ~/semlex_paths.sh
PY=${PROJECT_ROOT}/envs/cslr/bin/python
echo "===== stage states (non-COMPLETED shown) ====="
sacct -X -j 5165737,5165738,5165739,5165740,5165741,5165742,5165743,5165744,5165745,5165746,5165747 \
  --format=JobID%14,JobName%12,State%16,ExitCode,Elapsed,Timelimit | grep -Ev 'COMPLETED'
echo "===== TEST-split coverage at each stage ====="
$PY - <<'PYEOF'
import csv,os,os.path as P
S=os.environ["SEMLEX"]; W=os.environ["WORK"]
rows=list(csv.DictReader(open(P.join(S,"metadata.csv"))))
cols=list(rows[0].keys()); print("cols:",cols)
sc=next((c for c in cols if 'split' in c.lower()),None)
test=[r for r in rows if sc and str(r[sc]).strip().lower()=='test']
print("test rows:",len(test))
ex=lambda *a: P.exists(P.join(*a))
tot={}
for r in test:
    v=r["video_id"]
    for k,b in {"mp4":ex(S,"videos_mp4",v+".mp4"),"posejson":ex(W,"pose",v+"_pose.json"),
      "facecrop":ex(W,"face_crops",v+"_face.mp4"),"handcrop":ex(W,"hand_crops",v+"_hand1.mp4"),
      "facefeat":ex(W,"face_feats",v+"_face.npy"),"hand1feat":ex(W,"hand1_feats",v+"_hand1.npy"),
      "hand2feat":ex(W,"hand2_feats",v+"_hand2.npy"),"bodyfeat":ex(W,"body_feats",v+"_pose.npy"),
      "posefeat":ex(S,"pose_features",v+".pt")}.items():
        tot[k]=tot.get(k,0)+(1 if b else 0)
print("TEST coverage out of",len(test),":")
for k in ["mp4","posejson","facecrop","handcrop","facefeat","hand1feat","hand2feat","bodyfeat","posefeat"]:
    print("  %-10s %d"%(k,tot.get(k,0)))
print("sample test ids:", [r["video_id"] for r in test[:3]])
PYEOF
