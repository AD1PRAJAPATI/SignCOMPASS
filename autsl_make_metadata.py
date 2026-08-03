import os, csv, collections
ROOT="/project2/jessetho_1732/aditeya/data/autsl/AUTSL"
OUT="/project2/jessetho_1732/aditeya/data/autsl/metadata.csv"
rows=[]
for split in ["train","val","test"]:
    cp=os.path.join(ROOT,split+".csv"); vd=os.path.join(ROOT,split)
    if not os.path.isfile(cp): continue
    for line in open(cp):
        line=line.strip()
        if not line: continue
        parts=line.split(","); fn=parts[0]; cls=parts[1] if len(parts)>1 else ""
        p=os.path.join(vd,fn)
        if not os.path.isfile(p): continue
        rows.append([os.path.splitext(fn)[0], str(cls), fn.split("_")[0], split, p])
with open(OUT,"w",newline="") as o:
    w=csv.writer(o); w.writerow(["video_id","gloss","participant_id","split","video_path"]); w.writerows(rows)
print("videos:",len(rows),"| classes:",len({r[1] for r in rows}),"| splits:",dict(collections.Counter(r[3] for r in rows)))
