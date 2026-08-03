import json, os, pandas as pd
ROOT="/project2/jessetho_1732/aditeya/data/wlasl"
VID=os.path.join(ROOT,"videos")
data=json.load(open(os.path.join(ROOT,"WLASL_v0.3.json")))
rows=[]; total={}; present={}
for entry in data:
    g=entry["gloss"]
    for inst in entry["instances"]:
        vid=str(inst["video_id"]); sp=inst.get("split","train"); sig=str(inst.get("signer_id","unk"))
        total[sp]=total.get(sp,0)+1
        if os.path.exists(os.path.join(VID, vid+".mp4")):
            present[sp]=present.get(sp,0)+1
            rows.append({"video_id":vid,"gloss":g,"participant_id":sig,"split":sp})
df=pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT,"metadata.csv"), index=False)
print("classes:", df["gloss"].nunique(), "| videos present:", len(df))
for sp in ["train","val","test"]:
    t=total.get(sp,0); p=present.get(sp,0)
    print(f"  {sp}: {p}/{t} present ({100*p/max(t,1):.1f}% coverage)")
