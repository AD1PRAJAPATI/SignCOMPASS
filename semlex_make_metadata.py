import pandas as pd, os
ROOT="/project2/jessetho_1732/aditeya/data/sem_lex"
df = pd.read_csv(os.path.join(ROOT,"semlex_metadata.csv"))
df = df[df["label_type"]=="asllex"].copy()          # the 3,149-sign benchmark vocab (has phonology)
out = pd.DataFrame({
    "video_id":       df["video_id"].astype(str).str.strip(),
    "gloss":          df["label"].astype(str).str.strip(),
    "participant_id": df["signer_id"].astype(str).str.strip(),
    "split":          df["split"].astype(str).str.strip().str.lower(),
})
# keep phonological columns for the aux heads (used later)
for c in ["Handshape","Major Location","Minor Location","Path Movement","Repeated Movement","Sign Type"]:
    if c in df.columns: out[c]=df[c]
out.to_csv(os.path.join(ROOT,"metadata.csv"), index=False)
print(out["split"].value_counts().to_dict())
print("classes:", out["gloss"].nunique(), "| rows:", len(out))
